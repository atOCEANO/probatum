//! Perp funding. A row charges the events it owns at its closes, after the bar's
//! liquidation check and on the positions still held, and the events stamped
//! between it and the row before at its start, at the last marks. An event charges
//! the instrument it names, or every perp held where it names none. A positive rate
//! is paid by a long and received by a short, and what was paid is kept for the
//! result (ADR 0020).

use probatum_core::{Market, Price};
use serde::{Deserialize, Serialize};

use super::Engine;
use crate::candles::printed;
use crate::reporter::Reporter;

/// A total reported over many bars, summed with Neumaier's compensation so that what
/// rounding drops from each addition is carried rather than lost (Specification,
/// Numbers that are the same everywhere).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct Total {
    sum: f64,
    carry: f64,
}

impl Total {
    /// Add `x` to the total.
    pub(super) fn add(&mut self, x: f64) {
        let sum = self.sum + x;
        self.carry += if self.sum.abs() >= x.abs() {
            (self.sum - sum) + x
        } else {
            (x - sum) + self.sum
        };
        self.sum = sum;
    }

    pub(super) fn get(self) -> f64 {
        self.sum + self.carry
    }
}

/// Keep a funding payment on instrument `i` for the result and charge it to its bar
/// and to the round trip it was paid on (ADRs 0020, 0022, 0023). It takes the two
/// fields it writes, so a schedule borrowed from the candles can stay borrowed while
/// it runs.
fn pay(funding_paid: &mut Total, reporter: &mut Option<Reporter>, i: usize, payment: f64) {
    funding_paid.add(payment);
    if let Some(reporter) = reporter.as_mut() {
        reporter.costs_mut().funding += payment;
        reporter.flow(i, -payment);
        if let Some(trip) = reporter.trip_mut(i) {
            trip.funding += payment;
        }
    }
}

impl Engine {
    /// True when the current row owns emsl's event, at the close of bars `n`, `2n`
    /// and so on, counted on the row's index in the series rather than the bars since
    /// a reset, so every env of a batch is charged on the same bars however its
    /// episode was offset (ADR 0020).
    fn owns_every_bars(&self) -> bool {
        self.config.funding_interval > 0 && self.tick % self.config.funding_interval == 0
    }

    /// True when an event naming `instrument`, or none, charges instrument `i`: a
    /// perp holding a position.
    fn charges(&self, instrument: Option<usize>, i: usize) -> bool {
        instrument.is_none_or(|named| named == i)
            && self.account.markets[i] == Market::Perp
            && self.account.positions[i].qty.get() != 0.0
    }

    /// Instrument `i`'s close on the current row, or its last mark where it printed
    /// no bar.
    pub(super) fn close_of(&self, i: usize) -> f64 {
        let bar = self.candles.bar(self.tick, i);
        if printed(&bar) {
            bar.close
        } else {
            self.marks[i]
        }
    }

    /// Charge the events stamped between the row before and this one, across a
    /// closure or a hole in the data, at the last marks, on the positions carried
    /// through them. This comes before the bar's path is solved, so a debit that
    /// takes the account to its trigger liquidates it at the start of the gap.
    pub(super) fn charge_between(&mut self) {
        if let Some(schedule) = self.candles.funding() {
            for &(rate, instrument) in schedule.between(self.tick) {
                self.cleared = false;
                for i in 0..self.marks.len() {
                    if self.charges(instrument, i) {
                        let paid = self.account.apply_funding(i, rate, Price(self.marks[i]));
                        pay(&mut self.funding_paid, &mut self.reporter, i, paid);
                    }
                }
            }
        }
    }

    /// Whether the bar counts a funding `charge` of the `event`th event the row owns,
    /// stamped inside it, toward its path. A bar cannot show whether such an event
    /// came before its worst price: the pessimistic way counts a debit and not a
    /// credit, the optimistic way a credit and not a debit, and a random setting
    /// draws the way of each event.
    fn counts(&self, event: usize, charge: f64) -> bool {
        let worse = self
            .unseen
            .funding
            .get(event)
            .copied()
            .unwrap_or(self.unseen.worse);
        if worse {
            charge > 0.0
        } else {
            charge < 0.0
        }
    }

    /// What the charges stamped inside this row the bar counts toward its path come
    /// to, on the positions as they now stand, at their closes: those carried in for
    /// the path solved before the fills, those the fills left for the check. Asked on
    /// every bar a perp is held, most often of a row that owns no funding, so a call
    /// would cost more than the answer.
    #[inline(always)]
    pub(super) fn counted_inside(&self) -> f64 {
        let Some(schedule) = self.candles.funding() else {
            return 0.0;
        };
        let mut counted = 0.0;
        for (event, &(rate, inside, instrument)) in schedule.owned(self.tick).iter().enumerate() {
            for i in 0..self.marks.len() {
                if !inside || !self.charges(instrument, i) {
                    continue;
                }
                let charge = self.account.positions[i].qty.get() * self.close_of(i) * rate;
                if self.counts(event, charge) {
                    counted += charge;
                }
            }
        }
        counted
    }

    /// Pay the charges stamped inside this row, at the closes, which the check
    /// counted against an account it then liquidated: counted is paid, and what it
    /// did not count is not.
    pub(super) fn pay_counted_inside(&mut self) {
        if let Some(schedule) = self.candles.funding() {
            for (event, &(rate, inside, instrument)) in schedule.owned(self.tick).iter().enumerate()
            {
                for i in 0..self.marks.len() {
                    if !inside || !self.charges(instrument, i) {
                        continue;
                    }
                    let close = self.close_of(i);
                    let charge = self.account.positions[i].qty.get() * close * rate;
                    if self.counts(event, charge) {
                        let paid = self.account.apply_funding(i, rate, Price(close));
                        pay(&mut self.funding_paid, &mut self.reporter, i, paid);
                    }
                }
            }
        }
    }

    /// Charge the events this row owns at the closes, on the positions still held,
    /// so an account the check liquidated, having paid what it counted, pays nothing
    /// more.
    pub(super) fn charge_funding(&mut self) {
        if self.owns_every_bars() {
            self.clear_through = false;
            for i in 0..self.marks.len() {
                if self.charges(None, i) {
                    let rate = self.config.funding_rate;
                    let paid = self.account.apply_funding(i, rate, Price(self.close_of(i)));
                    pay(&mut self.funding_paid, &mut self.reporter, i, paid);
                }
            }
        }
        if let Some(schedule) = self.candles.funding() {
            for &(rate, _, instrument) in schedule.owned(self.tick) {
                self.clear_through = false;
                for i in 0..self.marks.len() {
                    if self.charges(instrument, i) {
                        let paid = self.account.apply_funding(i, rate, Price(self.close_of(i)));
                        pay(&mut self.funding_paid, &mut self.reporter, i, paid);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::{Candles, Instrument};
    use crate::clock::Clock;
    use crate::engine::fixtures::{cfg, flat_series, levered, ohlc, perp_cfg, series};
    use crate::engine::{Engine, EngineConfig, CLOSE_EPS};
    use crate::reporter::FillReason;
    use crate::schedule::every;
    use probatum_core::{Candle, Market, OrderType, Side, TimeInForce};

    #[test]
    fn a_reported_total_carries_what_each_addition_rounds_away() {
        // 1e16 + 1 rounds back to 1e16, so a plain running sum of these is 0
        let mut total = super::Total::default();
        for payment in [1e16, 1.0, -1e16] {
            total.add(payment);
        }
        assert_eq!(total.get(), 1.0);
    }

    #[test]
    fn perp_funding_charges_a_held_long_each_interval() {
        // funding every bar at 0.001 of notional; a long pays, a short receives
        let mut config = perp_cfg();
        config.quote = 10_000.0;
        config.funding_rate = 0.001;
        config.funding_interval = 1;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        let s1 = e.step(); // long 1 @ 200; funding at bar 1 close 250: pay 0.25
        let q1 = s1.quote;
        assert!((10_000.0 - q1 - 0.25).abs() < 1e-9); // 1 * 250 * 0.001
        let s2 = e.step(); // funding at bar 2 close 350: pay another 0.35
        assert!((q1 - s2.quote - 0.35).abs() < 1e-9);
    }

    #[test]
    fn funding_is_off_by_default_and_on_spot() {
        // default config funds nothing (interval 0); spot never funds either
        let mut e = Engine::new(series(), cfg()); // spot, no funding
        e.reset();
        e.market_buy(1.0);
        let s = e.step();
        assert_eq!(s.quote, 10_000.0 - 200.0); // only the buy, no funding debit
    }

    #[test]
    fn funding_can_bust_the_account_on_the_same_bar() {
        // a punitive funding rate on a leveraged long takes equity below zero at the
        // bar's close, after its check has passed, and the account is closed there
        // with exactly nothing left (ADR 0020)
        let mut config = perp_cfg(); // quote 100
        config.funding_rate = 1.0; // 100% of notional per event, for the test
        config.funding_interval = 1;
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(2.0); // long 2 @ 100, notional 200 on 100 equity
        let s = e.step(); // funding 2*100*1.0 = 200 debited at the close
        assert!(e.is_bust());
        assert_eq!(s.position, 0.0);
        assert_eq!(s.equity, 0.0);
        assert_eq!(s.funding_paid, 200.0);
    }

    #[test]
    fn funding_is_charged_only_on_interval_bars() {
        // every other funding test uses interval 1, where the modulo is trivially
        // true on every bar, so the schedule itself is never exercised (ADR 0020)
        let mut config = perp_cfg();
        config.quote = 10_000.0;
        config.funding_rate = 0.001;
        config.funding_interval = 3;
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 10_000.0);
        let mut e = Engine::new(Candles::new(vec![flat; 8]), config);
        e.reset();
        e.market_buy(1.0);
        let mut charged = Vec::new();
        let mut prev = 10_000.0;
        for _ in 0..7 {
            let s = e.step();
            charged.push((prev - s.quote).abs() > 1e-12);
            prev = s.quote;
        }
        // ticks 1..7; funding fires where tick % 3 == 0, so on ticks 3 and 6 only
        assert_eq!(
            charged,
            vec![false, false, true, false, false, true, false],
            "charged on {charged:?}"
        );
    }

    #[test]
    fn funding_is_kept_where_the_result_can_read_it() {
        // the account returned every payment and nothing kept it, so the one cost
        // unique to a perp was unmeasurable from a finished run, and a carry could
        // not be told apart from a direction (ADR 0020)
        let config = EngineConfig {
            market: Market::Perp,
            funding_rate: 0.001,
            funding_interval: 1,
            report: true,
            ..cfg()
        };
        let mut long = Engine::new(series(), config);
        long.reset();
        long.market_buy(1.0);
        let s1 = long.step(); // fills at the open of 200, funded at the close of 250
        assert!((s1.funding_paid - 0.25).abs() < CLOSE_EPS);
        let s2 = long.step(); // funded again at the close of 350
        assert!((s2.funding_paid - 0.60).abs() < CLOSE_EPS);
        let stats = long.stats(365.0, 0.0).expect("reporting is on");
        assert!((stats.funding_paid - 0.60).abs() < CLOSE_EPS);

        // a positive rate is paid BY the long and TO the short, so the sign flips
        let mut short = Engine::new(series(), config);
        short.reset();
        short.market_sell(1.0);
        let s = short.step();
        assert!((s.funding_paid + 0.25).abs() < CLOSE_EPS);
    }

    #[test]
    fn an_account_closed_out_on_a_bar_pays_nothing_for_that_bars_funding() {
        // the stamp at a bar's end comes after every price the bar printed, so the
        // check runs first, and an account it liquidates holds nothing at the stamp.
        // emsl charged the bar's funding before the check, at the clipped close,
        // which credited this short 11 for a rise it had been closed out of and moved
        // its exit with the credit (ADR 0020)
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 130.0, 100.0, 128.0, 1e6),
        ]);
        let config = EngineConfig {
            funding_rate: 0.01,
            funding_interval: 2, // tick 1 is not a boundary, tick 2 is
            ..levered()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_sell(10.0); // short 10 at the open of 100, on 100 of margin
        let state = e.step();
        assert_eq!(state.position, -10.0);
        assert_eq!(state.funding_paid, 0.0, "tick 1 is not a funding boundary");

        // the margin runs out at 110, where the short is closed, and the stamp
        // finds it flat
        let state = e.step();
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        assert_eq!(state.funding_paid, 0.0);
        assert_eq!(state.equity, 0.0);
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit, 110.0);
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
    }

    #[test]
    fn a_bar_the_account_survives_is_funded_at_its_close_not_where_it_was_clipped() {
        // the fence of the position carried in clips the bar at 101.8, and a buy at
        // the open shrinks the short enough to live through the rest of it. The
        // stamp still comes at the bar's close of 104.9, which emsl read as the
        // clipped close (ADR 0020)
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(91.0, 108.6, 90.0, 104.9, 1e6),
        ]);
        let config = EngineConfig {
            quote: 900.0,
            max_leverage: 0.0,
            funding_rate: 0.0001,
            funding_interval: 2,
            ..levered()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_sell(500.0); // short 500 at 100 on 900: the margin runs out at 101.8
        e.step();
        e.market_buy(495.0); // fills at the open of 91, leaving a short of 5
        let state = e.step();
        assert!(!e.is_bust());
        assert_eq!(state.position, -5.0);
        assert!((state.funding_paid - (-5.0 * 104.9 * 0.0001)).abs() < 1e-12);
    }

    #[test]
    fn funding_fires_on_the_absolute_bar_index_not_on_bars_since_the_reset() {
        // ADR 0020: the funding schedule belongs to the series, so every
        // env in a vectorized run charges the same bars however its episode was
        // offset. Only an offset start separates the two counters, and nothing did
        // that: starting at bar 2 with an interval of 3 funds ticks 3 and 6, while
        // a per-episode counter funds the third bar of the episode, tick 5, and
        // nothing else in this window
        let config = EngineConfig {
            quote: 10_000.0,
            funding_rate: 0.0625, // one sixteenth, so each charge is exactly 6.25
            funding_interval: 3,
            ..perp_cfg()
        };
        let mut e = Engine::new(flat_series(), config); // eight flat bars at 100
        e.reset_at(2);
        e.market_buy(1.0); // fills at the open of bar 3, ahead of that bar's funding
        let mut charged = Vec::new();
        for _ in 0..5 {
            let state = e.step();
            charged.push((state.tick_index, state.funding_paid));
        }
        // 1 unit at 100 pays 6.25 a time, on tick 3 and on tick 6 and on no other
        assert_eq!(
            charged,
            vec![(3, 6.25), (4, 6.25), (5, 6.25), (6, 12.5), (7, 12.5)]
        );
        assert_eq!(e.current_state().quote, 9_987.5);
    }

    const HOUR: i64 = 3_600 * 1_000_000_000;
    const DAY: i64 = 24 * HOUR;

    /// Bars of these open, high, low and close prices on a clock of `times`, with
    /// funding `events` placed on it.
    fn on_clock(
        prices: &[(f64, f64, f64, f64)],
        times: Vec<i64>,
        events: &[(i64, f64)],
    ) -> Candles {
        let bars: Vec<Candle> = prices
            .iter()
            .map(|&(open, high, low, close)| ohlc(open, high, low, close, 1e6))
            .collect();
        Candles::on_clock(bars, Clock::new(times)).with_funding(events)
    }

    #[test]
    fn a_stamp_a_hair_after_a_bars_end_is_charged_at_that_bars_close() {
        // the Specification's example: hourly bars and 8-hour funding, an event
        // stamped 08:00:00.003 taken as 08:00 and so charged on the bar that opened
        // at 07:00; a long of 0.5 at a close of 60,000 pays 0.0001 of it, 3.00
        let flat = (60_000.0, 60_000.0, 60_000.0, 60_000.0);
        let times = (0..10).map(|row| row * HOUR).collect();
        let bars = on_clock(&[flat; 10], times, &[(8 * HOUR + 3_000_000, 0.0001)]);
        let config = EngineConfig {
            quote: 100_000.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(0.5);
        for _ in 0..6 {
            e.step();
        }
        assert_eq!(
            e.current_state().funding_paid,
            0.0,
            "nothing is due by 07:00"
        );
        let seven = e.step();
        assert_eq!(seven.tick_index, 7);
        assert!((seven.funding_paid - 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_long_stopped_out_during_a_day_pays_nothing_for_that_days_funding() {
        // the Specification's example of the sources: on daily bars an event at
        // 08:00 is charged at the day's close on the position after its fills, so a
        // long stopped out during the day pays nothing for it and a long held
        // through the day pays all three of the day's events at its close
        let times = vec![0, DAY, 2 * DAY];
        let prices = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 94.0, 96.0),
            (96.0, 96.0, 96.0, 96.0),
        ];
        let events = every(&Clock::new(times.clone()), 8.0, 0.001);
        for stopped in [true, false] {
            let bars = on_clock(&prices, times.clone(), &events);
            let config = EngineConfig {
                quote: 10_000.0,
                ..perp_cfg()
            };
            let mut e = Engine::new(bars, config);
            e.reset();
            e.market_buy(1.0);
            if stopped {
                e.stop(Side::Sell, 1.0, 95.0, true);
            }
            let state = e.step();
            let paid = if stopped { 0.0 } else { 3.0 * 96.0 * 0.001 };
            assert!(
                (state.funding_paid - paid).abs() < 1e-12,
                "stopped={stopped} paid {}",
                state.funding_paid
            );
        }
    }

    #[test]
    fn a_debit_stamped_inside_a_bar_counts_toward_its_check_and_a_credit_does_not() {
        // daily bars and an event at 08:00 on the second day. The long of 10 on 100
        // of margin survives that day's low of 90.5 by 5, and a debit of 9.5 inside
        // the bar takes it under: the bar cannot show whether the payment came
        // before the low, so the debit counts and is paid, and the account is closed
        // at 90.95, where the 90.5 left after paying it runs out. A credit to the
        // long does not count, and the account lives to be paid it at the close
        for (rate, survives) in [(0.01, false), (-0.01, true)] {
            let bars = on_clock(
                &[
                    (100.0, 100.0, 100.0, 100.0),
                    (100.0, 101.0, 90.5, 95.0),
                    (95.0, 95.0, 95.0, 95.0),
                ],
                vec![0, DAY, 2 * DAY],
                &[(DAY + 8 * HOUR, rate)],
            );
            let mut e = Engine::new(bars, levered());
            e.reset();
            e.market_buy(10.0);
            let state = e.step();
            assert_eq!(e.is_bust(), !survives, "rate {rate}");
            assert!(
                (state.funding_paid - 10.0 * 95.0 * rate).abs() < 1e-9,
                "rate {rate}"
            );
            if survives {
                assert_eq!(state.position, 10.0);
            } else {
                assert_eq!(state.position, 0.0);
                assert_eq!(state.equity, 0.0);
                let trades = e.reporter().expect("reporting is on").trades();
                assert!((trades[0].exit - 90.95).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn an_event_across_a_closure_is_charged_at_the_last_mark_before_the_bar_opens() {
        // daily bars stop at Friday's close of 100 and Monday opens at 90; a Saturday
        // event is charged at the start of Monday's bar on the long carried through
        // the weekend, at Friday's close: 0.01 of 1 at 100, not at Monday's 90
        let week = vec![0, DAY, 2 * DAY, 3 * DAY, 4 * DAY, 7 * DAY];
        let mut prices = vec![(100.0, 100.0, 100.0, 100.0); 5];
        prices.push((90.0, 90.0, 90.0, 90.0));
        let bars = on_clock(&prices, week, &[(5 * DAY + 8 * HOUR, 0.01)]);
        let config = EngineConfig {
            quote: 1_000.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.0);
        for _ in 0..4 {
            e.step();
        }
        assert_eq!(e.current_state().funding_paid, 0.0);
        let monday = e.step();
        assert!((monday.funding_paid - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_debit_across_a_closure_that_empties_the_margin_closes_before_the_bar_opens() {
        // a long of 10 on 100 of margin ends Friday at 91 with 10 left, and the
        // Saturday debit of 18.2 leaves it owing at Friday's close. It is closed
        // there, at the start of the gap, before Monday's open of 92 could lift it
        // back above water
        let week = vec![0, DAY, 2 * DAY, 3 * DAY, 4 * DAY, 7 * DAY];
        let mut prices = vec![(100.0, 100.0, 100.0, 100.0); 4];
        prices.push((100.0, 100.0, 91.0, 91.0));
        prices.push((92.0, 92.0, 92.0, 92.0));
        let bars = on_clock(&prices, week, &[(5 * DAY + 8 * HOUR, 0.02)]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0);
        for _ in 0..4 {
            e.step();
        }
        assert!(!e.is_bust(), "Friday's low leaves 10 of margin");
        let monday = e.step();
        assert!(e.is_bust());
        assert_eq!(monday.position, 0.0);
        assert_eq!(monday.equity, 0.0);
        assert!((monday.funding_paid - 18.2).abs() < 1e-9);
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades[0].exit, 91.0);
    }

    #[test]
    fn a_round_trip_nets_the_funding_charged_while_it_was_open() {
        // long 1 from 200 pays 0.25 at the close of 250, and the close at 300
        // realizes 100, so the round trip nets 99.75, what the equity gained
        let config = EngineConfig {
            quote: 10_000.0,
            funding_rate: 0.001,
            funding_interval: 1,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.close();
        let last = e.step();
        let [t] = e.reporter().expect("reporting is on").trades() else {
            panic!("one round trip");
        };
        assert!((t.funding - 0.25).abs() < 1e-12);
        assert!((t.pnl - 99.75).abs() < 1e-9, "pnl {}", t.pnl);
        assert!((t.pnl - (last.equity - 10_000.0)).abs() < 1e-9);
    }

    #[test]
    fn a_debit_inside_a_bar_that_closes_a_short_is_paid_and_one_at_its_end_is_not() {
        // short 10 from 100 on 100 of margin holds 5 at the high of 109.5, and a
        // debit of 10.5 inside the bar takes it under. It is paid, and the close
        // comes where the 89.5 left runs out, at 108.95; the event at the bar's end
        // finds it flat
        let bars = on_clock(
            &[
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 109.5, 99.0, 105.0),
                (105.0, 105.0, 105.0, 105.0),
            ],
            vec![0, DAY, 2 * DAY],
            &[(DAY + 8 * HOUR, -0.01), (2 * DAY, -0.01)],
        );
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_sell(10.0);
        let state = e.step();
        assert!(e.is_bust());
        assert_eq!((state.position, state.equity), (0.0, 0.0));
        assert!((state.funding_paid - 10.5).abs() < 1e-9);
        let trades = e.reporter().expect("reporting is on").trades();
        assert!(
            (trades[0].exit - 108.95).abs() < 1e-9,
            "exit {}",
            trades[0].exit
        );
    }

    #[test]
    fn an_event_charges_the_instrument_it_names_or_every_perp_held() {
        // a long of 1 at 100 and of 10 at 10: an event at 1% naming the second
        // charges it 1.00, and one naming none charges both, 2.00
        let hour = 3_600 * 1_000_000_000;
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows: Vec<Candle> = (0..3).flat_map(|_| [flat(100.0), flat(10.0)]).collect();
        let perp = Instrument {
            market: Some(Market::Perp),
            ..Instrument::default()
        };
        let bars = Candles::universe(rows, 2, Some(Clock::new(vec![0, hour, 2 * hour])))
            .with_instruments(vec![perp.clone(), perp])
            .with_tagged_funding(&[(2 * hour, Some(1), 0.01), (3 * hour, None, 0.01)]);
        let config = EngineConfig {
            quote: 10_000.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.0);
        e.order_on(
            1,
            Side::Buy,
            10.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            TimeInForce::Ioc,
        );
        assert_eq!(e.step().margin_cash, 9_999.0);
        let last = e.step();
        assert_eq!((last.margin_cash, last.funding_paid), (9_997.0, 3.0));
    }
}
