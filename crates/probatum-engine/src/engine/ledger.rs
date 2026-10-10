//! Booking a fill, forced closes included: the account moves, the fill is counted,
//! and when reporting is on it is logged with what it paid and carried into the
//! round trips, each of which holds every fee of its own fills (ADRs 0022, 0023).

use probatum_core::{Candle, CostModel, Fill, Market, OrderId, Side};

use super::path::FillNote;
use super::{Engine, CLOSE_EPS};
use crate::candles::printed;
use crate::execution::FillCosts;
use crate::reporter::{EventKind, EventReason, FillReason, FillRecord, RoundTrip};

/// Where in its bar a fill sits, which decides how much of the bar a round trip it
/// ends lived through (ADR 0023).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Moment {
    /// Before the bar opened: a close-out on the gap, or at the last mark.
    BeforeOpen,
    /// Inside the bar: at its open or along its range.
    InBar,
    /// At its close, once the bar had printed every price.
    AtClose,
}

impl Engine {
    /// Instrument `i`'s position fields a fill has to be compared against once it
    /// has landed.
    pub(super) fn before_fill(&self, i: usize) -> (f64, f64, f64) {
        let position = &self.account.positions[i];
        (
            position.qty.get(),
            position.avg_entry.get(),
            position.realized,
        )
    }

    /// Clamp `fill` on instrument `i`, bound it by the margin inside `candle`, the bar it
    /// filled on, book it and log it as `order`'s, with `costs` the fill model measured
    /// and `requested` what the order still asked for. Returns the size that landed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_fill_clamped(
        &mut self,
        i: usize,
        order: OrderId,
        reason: FillReason,
        reduce_only: bool,
        requested: f64,
        fill: Fill,
        costs: FillCosts,
        candle: &Candle,
    ) -> f64 {
        let mut fill = self.clamp_fill(i, reduce_only, &fill);
        if fill.size.get() <= 0.0 {
            return 0.0;
        }
        let priced = fill.price.get();
        let credit = self.bound_by_margin(i, &mut fill, candle);
        let applied = fill.size.get();
        let fee = self.cost.fee(&fill);
        let before = self.before_fill(i);
        // what the bar could not hold of the bound's lift is paid into the margin
        // wallet, and the close then realizes against it
        self.account.margin_cash += credit;
        self.account.apply_fill(i, &fill, fee);
        // a fill the bound moved that leaves no perp held leaves the wallet at zero,
        // which floating point misses by a hair either side, and a hair above zero is
        // equity the cap levers, so the wallet is set to zero as a takeover sets it
        // and what it held is the liquidation charge (ADRs 0015, 0018)
        let moved = fill.price.get() != priced || credit > 0.0;
        if moved && self.account.perps_held() == 0 {
            if let Some(reporter) = self.reporter.as_mut() {
                reporter.costs_mut().liquidation += self.account.margin_cash;
            }
            self.account.margin_cash = 0.0;
        }
        if self.account.markets[i] == Market::Perp {
            let (held, entry, _) = before;
            let signed = fill.side.sign() * applied;
            let closed = if held * signed < 0.0 {
                applied.min(held.abs())
            } else {
                0.0
            };
            let after = self.account.positions[i].qty.get();
            self.note(FillNote {
                i,
                before: held,
                entry,
                after,
                price: fill.price.get(),
                opened: applied - closed,
            });
            if after == 0.0 {
                self.starts[i] = f64::NAN;
            } else if held == 0.0 || held * after < 0.0 {
                self.starts[i] = costs.reference;
            }
        }
        self.book_fill(
            i,
            applied,
            fill.price.get(),
            fee,
            before,
            reason,
            Moment::InBar,
        );
        if let Some(reporter) = self.reporter.as_mut() {
            // what the fill paid beyond its reference at the price its costs and the
            // bar set, each part in proportion when the bar held that price back, and
            // what the margin bound then lifted it by and paid in for what the bar
            // could not hold, a negative liquidation charge
            let paid = applied * (priced - costs.reference).abs();
            let asked = costs.spread + costs.slippage + costs.impact;
            let bar = reporter.costs_mut();
            bar.fees += fee;
            if asked > 0.0 {
                bar.spread += paid * costs.spread / asked;
                bar.slippage += paid * costs.slippage / asked;
                bar.impact += paid * costs.impact / asked;
            }
            bar.liquidation -= applied * (fill.price.get() - priced).abs() + credit;
            reporter.record_fill(FillRecord {
                tick: self.tick,
                instrument: i,
                order: Some(order),
                side: fill.side,
                size: applied,
                price: fill.price.get(),
                costs: FillCosts {
                    adjustment: costs.adjustment + fill.price.get() - priced,
                    ..costs
                },
                fee,
                taker: fill.is_taker,
                reason,
            });
        }
        if applied + CLOSE_EPS < requested {
            self.log_event(
                EventKind::Clamped,
                Some(order),
                Some(i),
                Some(applied),
                None,
                EventReason::Fits { asked: requested },
            );
        }
        let why = match reason {
            FillReason::Market => EventReason::Market,
            FillReason::Limit => EventReason::Limit,
            FillReason::Stop => EventReason::Stop,
            FillReason::Rebalance => EventReason::Rebalance,
            FillReason::Liquidation => EventReason::Liquidation,
        };
        self.log_event(
            EventKind::Filled,
            Some(order),
            Some(i),
            Some(applied),
            Some(fill.price.get()),
            why,
        );
        applied
    }

    /// Book what a fill on instrument `i` just did to the account: count it, and with reporting on,
    /// carry it into the round trips. Whatever it closed shrinks or ends the round
    /// trip standing, and whatever it opened grows it or begins the next, each part
    /// with its share of the fee, so a flip's fee is split by size between the round
    /// trip it ends and the one it begins (ADRs 0014, 0023).
    ///
    /// The single place this happens, which is the point. A liquidation used to
    /// close the position on the account directly and reach none of it, so the
    /// forced close appeared in no trade, counted toward no fill, and left the dead
    /// position's entry fee standing to be charged against the NEXT trade.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn book_fill(
        &mut self,
        i: usize,
        applied: f64,
        price: f64,
        fee: f64,
        before: (f64, f64, f64),
        reason: FillReason,
        moment: Moment,
    ) {
        let (pos_before, _, realized_before) = before;
        self.fills += 1;
        let pos_after = self.account.positions[i].qty.get();
        let realized = self.account.positions[i].realized - realized_before;
        let tick = self.tick;
        let Some(reporter) = self.reporter.as_mut() else {
            return;
        };
        let bar = self.candles.bar(tick, i);

        // The base amount the fill moved the position toward zero: a reduce keeps
        // the same-side remainder, a close or flip keeps none of the old side.
        let kept = if (pos_before > 0.0) == (pos_after > 0.0) {
            pos_after.abs()
        } else {
            0.0
        };
        // Clamped at zero: a same-side add keeps MORE than it started with, so the
        // raw difference goes negative and `opened` would then exceed the fill,
        // charging the entry fee twice over. Nothing is closed by an add.
        let closed = (pos_before.abs() - kept).max(0.0);
        let opened = applied - closed;

        if closed > CLOSE_EPS {
            if let Some(trip) = reporter.trip_mut(i) {
                // the part of this bar a round trip carried into it lived through: up
                // to the fill from the open, the whole bar when it closed at the
                // close, and none of it when it closed before the bar opened or on a
                // row its instrument printed no bar on
                if trip.open_tick != tick && printed(&bar) {
                    match moment {
                        Moment::BeforeOpen => {}
                        Moment::InBar => trip.see(bar.open),
                        Moment::AtClose => {
                            trip.see(bar.low);
                            trip.see(bar.high);
                        }
                    }
                }
                trip.shrink(closed, price, fee * closed / applied, realized);
            }
            if pos_after.abs() <= CLOSE_EPS || (pos_before > 0.0) != (pos_after > 0.0) {
                reporter.close_trip(i, tick, reason);
            }
        }
        if opened > CLOSE_EPS && pos_after.abs() > CLOSE_EPS {
            let share = fee * opened / applied;
            match reporter.trip_mut(i) {
                Some(trip) => trip.grow(opened, price, share),
                None => {
                    let side = if pos_after > 0.0 {
                        Side::Buy
                    } else {
                        Side::Sell
                    };
                    reporter.open_trip(RoundTrip::open(i, side, tick, opened, price, share));
                }
            }
        }
        if let Some(trip) = reporter.trip_mut(i) {
            trip.size = trip.size.max(pos_after.abs());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{
        cfg, crash_series, flat_series, levered, ohlc, perp_cfg, report_cfg, series,
    };
    use crate::engine::CLOSE_EPS;
    use crate::engine::{Engine, EngineConfig};
    use crate::execution::FillCosts;
    use crate::reporter::{BarCosts, EventKind, FillReason, FillRecord, RoundTrip};
    use probatum_core::{Candle, Fill, Market, OrderId, Price, Qty, Side, Wallet};

    fn fills(e: &Engine) -> &[FillRecord] {
        e.reporter().expect("reporting is on").fills()
    }

    fn trips(e: &Engine) -> &[RoundTrip] {
        e.reporter().expect("reporting is on").trades()
    }

    #[test]
    fn a_close_ends_the_round_trip_with_what_it_realized() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2 @ 200 (bar 1 open)
        e.close();
        e.step(); // close 2 @ 300 (bar 2 open)
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert_eq!(
            (t.side, t.size, t.entry, t.exit),
            (Side::Buy, 2.0, 200.0, 300.0)
        );
        assert_eq!((t.open_tick, t.close_tick), (1, Some(2)));
        assert_eq!((t.realized, t.pnl, t.ret), (200.0, 200.0, 0.5));
        assert_eq!(t.reason, Some(FillReason::Market));
    }

    #[test]
    fn a_partial_close_leaves_the_round_trip_open() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(5.0);
        e.step(); // long 5 @ 200
        e.market_sell(2.0);
        e.step(); // sell 2 @ 300, still long 3, and the bar closes at 350
        assert!(trips(&e).is_empty());
        let all = e.round_trips();
        let [open] = all.as_slice() else {
            panic!("the one still open");
        };
        assert_eq!((open.size, open.exit, open.realized), (5.0, 300.0, 200.0));
        assert_eq!(open.close_tick, None);
        // marked at the close: the 200 realized and the 3 * (350 - 200) still held
        assert_eq!(open.pnl, 650.0);
    }

    #[test]
    fn a_flip_ends_one_round_trip_and_begins_the_next() {
        // the Specification's example: a long of 1.0 at an average of 100 sells 1.5
        // at 110, so the long closes, realizing 10.00 before fees, and a short of
        // 0.5 opens at 110
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(110.0, 110.0, 110.0, 110.0, 1e6),
        ]);
        let config = EngineConfig {
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.market_sell(1.5);
        let s = e.step();
        assert_eq!((s.position, s.avg_entry), (-0.5, 110.0));
        let all = e.round_trips();
        let [closed, opened] = all.as_slice() else {
            panic!("two round trips");
        };
        assert_eq!((closed.side, closed.realized), (Side::Buy, 10.0));
        assert_eq!(closed.close_tick, Some(2));
        assert_eq!(
            (opened.side, opened.entry, opened.size),
            (Side::Sell, 110.0, 0.5)
        );
        assert_eq!((opened.open_tick, opened.close_tick), (2, None));
    }

    #[test]
    fn adding_to_a_position_grows_its_one_round_trip() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
        e.market_buy(1.0);
        e.step(); // add 1 @ 300, no close
        assert!(trips(&e).is_empty());
        let open = e.round_trips()[0];
        assert_eq!((open.size, open.entry), (2.0, 250.0));
    }

    #[test]
    fn the_round_trips_realize_what_the_account_realized() {
        // the Validation Guide's reconciliation invariant: with no liquidation, the
        // round trips realize between them the account's cumulative realized PnL
        let mut config = perp_cfg(); // perp, quote 100, no cap
        config.report = true;
        let candles = Candles::new(vec![
            ohlc(100.0, 110.0, 90.0, 100.0, 1_000.0),
            ohlc(100.0, 110.0, 90.0, 105.0, 1_000.0),
            ohlc(105.0, 115.0, 95.0, 110.0, 1_000.0),
            ohlc(110.0, 120.0, 100.0, 108.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1
        e.market_sell(2.0);
        e.step(); // flip to short 1, which ends the long
        e.market_buy(1.0);
        let last = e.step(); // close the short

        assert_eq!(trips(&e).len(), 2);
        let realized: f64 = trips(&e).iter().map(|t| t.realized).sum();
        assert!((realized - last.realized_pnl).abs() < 1e-9);
    }

    #[test]
    fn the_round_trips_net_pnl_is_the_change_in_equity_on_a_run_that_ends_flat() {
        // the invariant the fee attribution buys: with the position closed and no
        // liquidation, the net PnL of the round trips IS the change in equity.
        // Charging only the exit fee left about half the cost attributed to nothing
        // (ADR 0023).
        let mut config = cfg(); // spot
        config.report = true;
        config.fee_taker = 0.001;
        config.fee_maker = 0.001;
        let bars: Vec<Candle> = (0..30)
            .map(|i| {
                let c = 100.0 + (i as f64 * 0.7).sin() * 5.0;
                ohlc(c, c + 1.0, c - 1.0, c, 10_000.0)
            })
            .collect();
        let mut e = Engine::new(Candles::new(bars), config);
        let mut state = e.reset();
        let mut hold = 0;
        while !e.done() {
            // an order decided on the last bar has no bar to fill against, so the
            // flattening close has to be decided on the one before it
            if e.tick() + 2 >= e.num_bars() {
                e.close();
            } else if state.position == 0.0 {
                e.market_buy(2.0);
                hold = 0;
            } else {
                hold += 1;
                if hold >= 2 {
                    e.close();
                }
            }
            state = e.step();
        }
        assert_eq!(
            state.position, 0.0,
            "the run must end flat for the identity"
        );

        let trades = trips(&e);
        assert!(trades.len() > 3, "only {} round trips", trades.len());
        let net: f64 = trades.iter().map(|t| t.pnl).sum();
        let equity_change = state.equity - 10_000.0;
        assert!(
            (net - equity_change).abs() < 1e-9,
            "sum(pnl) {net} != equity change {equity_change}"
        );
        // and the gross figure alone does not reconcile, which is the whole point
        let gross: f64 = trades.iter().map(|t| t.realized).sum();
        assert!((gross - equity_change).abs() > 1e-6);
        for t in trades {
            assert!((t.pnl - (t.realized - t.fees - t.funding)).abs() < 1e-12);
        }
    }

    #[test]
    fn a_round_trip_carries_both_sides_of_its_fee() {
        let mut config = cfg(); // spot, bar 1 open 200, bar 2 open 300
        config.report = true;
        config.fee_taker = 0.01;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // buy 1 at 200, fee 2.0
        e.close();
        e.step(); // sell 1 at 300, fee 3.0
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert!((t.realized - 100.0).abs() < 1e-9); // gross price PnL
        assert!((t.fees - 5.0).abs() < 1e-9); // 2.0 entry plus 3.0 exit
        assert!((t.pnl - 95.0).abs() < 1e-9);
    }

    #[test]
    fn adding_to_a_position_does_not_double_charge_its_entry_fee() {
        // buy 1 at 200 (fee 2), buy 1 at 400 (fee 4), then close both at 300 (fee
        // 6). The round trip carries every fee paid: 12, not 16. A same-side add
        // keeps more than it started with, so the closed amount goes negative unless
        // it is clamped, and the opening share of the fee is counted twice.
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000_000.0),
            ohlc(200.0, 200.0, 200.0, 200.0, 1_000_000.0),
            ohlc(400.0, 400.0, 400.0, 400.0, 1_000_000.0),
            ohlc(300.0, 300.0, 300.0, 300.0, 1_000_000.0),
            ohlc(300.0, 300.0, 300.0, 300.0, 1_000_000.0),
        ]);
        let mut config = perp_cfg();
        config.quote = 1_000_000.0;
        config.report = true;
        config.fee_taker = 0.01;
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // 1 @ 200, fee 2
        e.market_buy(1.0);
        e.step(); // 1 @ 400, fee 4; position 2, average entry 300
        e.market_sell(2.0);
        let last = e.step(); // 2 @ 300, fee 6

        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert!(
            (t.fees - 12.0).abs() < 1e-9,
            "fees {} should be 2 + 4 + 6",
            t.fees
        );
        assert!(t.realized.abs() < 1e-9);
        assert!((t.pnl + 12.0).abs() < 1e-9);
        // the identity must survive a scale-in as well as a single entry
        assert!((t.pnl - (last.equity - 1_000_000.0)).abs() < 1e-9);
    }

    #[test]
    fn scaling_in_and_out_reads_as_one_long_round_trip() {
        // ADR 0023's cost: a position that never returns to zero until the end is
        // one round trip, however often it was resized, and its net PnL is still
        // the change in equity
        let mut config = cfg(); // spot
        config.report = true;
        config.fee_taker = 0.002;
        config.fee_maker = 0.002;
        let bars: Vec<Candle> = (0..40)
            .map(|i| {
                let c = 100.0 + (i as f64 * 0.9).sin() * 8.0;
                ohlc(c, c + 1.0, c - 1.0, c, 1_000_000.0)
            })
            .collect();
        let mut e = Engine::new(Candles::new(bars), config);
        let mut state = e.reset();
        while !e.done() {
            let tick = e.tick();
            if tick + 2 >= e.num_bars() {
                e.close();
            } else if state.position < 3.0 {
                e.market_buy(1.0); // scale in, one unit at a time
            } else if tick % 5 == 0 {
                e.market_sell(1.0); // and scale out again
            }
            state = e.step();
        }
        assert_eq!(state.position, 0.0);
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert!(
            (t.pnl - (state.equity - 10_000.0)).abs() < 1e-9,
            "pnl {} != equity change {}",
            t.pnl,
            state.equity - 10_000.0
        );
    }

    #[test]
    fn num_fills_separates_a_dead_feed_from_a_quiet_strategy() {
        // a zero-volume series fills nothing, and without a counter the result is
        // identical to a strategy that never placed an order
        let dead = Candles::new(vec![ohlc(100.0, 100.0, 100.0, 100.0, 0.0); 6]);
        let mut config = cfg();
        config.report = true;
        let mut e = Engine::new(dead, config);
        e.reset();
        while !e.done() {
            e.market_buy(1.0);
            e.step();
        }
        assert_eq!(e.num_fills(), 0);
        assert_eq!(e.stats(365.0, 0.0).unwrap().num_fills, 0);

        let mut live = Engine::new(series(), config);
        live.reset();
        live.market_buy(1.0);
        live.step();
        assert_eq!(live.num_fills(), 1);
    }

    #[test]
    fn a_sub_dust_fill_moves_nothing_and_counts_nothing() {
        // the position refuses a size at or below its dust epsilon, so one in
        // (0, 1e-9] used to pass the engine's own gate, move no cash and no
        // position, and still count a fill and bank its fee. The leverage cap
        // produces exactly such a residual when a position sits a sub-ulp under it
        let mut e = Engine::new(series(), cfg());
        e.reset();
        let dust = Fill {
            side: Side::Buy,
            size: Qty(5e-13),
            price: Price(100.0),
            is_taker: true,
        };
        assert_eq!(e.clamp_fill(0, false, &dust).size.get(), 0.0);
        let applied = e.apply_fill_clamped(
            0,
            OrderId(1),
            FillReason::Market,
            false,
            5e-13,
            dust,
            FillCosts::default(),
            &ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        );
        assert_eq!(applied, 0.0);
        assert_eq!(e.num_fills(), 0);
        assert_eq!(e.state().position, 0.0);
    }

    #[test]
    fn a_round_trip_spans_the_bars_from_its_side_opening_to_its_end() {
        // adding to a position does not restart it, a partial close does not end it,
        // and a flip ends one and begins the next on the same bar
        let mut e = Engine::new(
            flat_series(),
            EngineConfig {
                report: true,
                ..perp_cfg()
            },
        );
        e.reset();
        e.market_buy(1.0);
        e.step(); // tick 1: long 1 opens here
        e.market_buy(1.0);
        e.step(); // tick 2: long 2
        e.market_sell(1.0);
        e.step(); // tick 3: closes 1 of it
        e.market_sell(3.0);
        e.step(); // tick 4: closes the last 1 and opens a short of 2 here
        e.market_buy(2.0);
        e.step(); // tick 5: closes the short

        let spans: Vec<(usize, Option<usize>)> = trips(&e)
            .iter()
            .map(|t| (t.open_tick, t.close_tick))
            .collect();
        assert_eq!(spans, vec![(1, Some(4)), (4, Some(5))]);
    }

    #[test]
    fn partial_closes_carry_the_entry_fee_exactly_once() {
        // a round trip carries its entry fee and every exit fee once, however many
        // fills close it
        let config = EngineConfig {
            fee_taker: 0.01,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(flat_series(), config);
        e.reset();
        e.market_buy(2.0);
        e.step(); // entry fee is 2 * 100 * 1% = 2.0
        e.market_sell(1.0);
        e.step(); // exit fee 1.0
        e.market_sell(1.0);
        e.step(); // exit fee 1.0
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert!((t.fees - 4.0).abs() < 1e-9, "{}", t.fees);
    }

    #[test]
    fn six_fills_make_two_closed_round_trips_and_an_open_one() {
        // the Specification's example: fills on bars 10, 12, 15, 20, 25 and 30 make
        // a long from 10 to 20 that reached 2, a short from 25 to 30, and a long
        // still open at the end, and `num_trades` is 2
        let bars = Candles::new(vec![ohlc(100.0, 100.0, 100.0, 100.0, 1e6); 32]);
        let config = EngineConfig {
            market: Market::Perp,
            report: true,
            ..EngineConfig::default()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        let decisions = [
            (9, Side::Buy, 1.0),
            (11, Side::Buy, 1.0),
            (14, Side::Sell, 0.5),
            (19, Side::Sell, 1.5),
            (24, Side::Sell, 1.0),
            (29, Side::Buy, 2.0),
        ];
        while !e.done() {
            for &(tick, side, size) in &decisions {
                if tick == e.tick() {
                    if side == Side::Buy {
                        e.market_buy(size);
                    } else {
                        e.market_sell(size);
                    }
                }
            }
            e.step();
        }
        let summary: Vec<(Side, usize, Option<usize>, f64)> = e
            .round_trips()
            .iter()
            .map(|t| (t.side, t.open_tick, t.close_tick, t.size))
            .collect();
        assert_eq!(
            summary,
            vec![
                (Side::Buy, 10, Some(20), 2.0),
                (Side::Sell, 25, Some(30), 1.0),
                (Side::Buy, 30, None, 1.0),
            ]
        );
        let stats = e.stats(365.0, 0.0).expect("reporting is on");
        assert_eq!(stats.num_trades, 2);
    }

    #[test]
    fn an_exit_at_the_open_counts_none_of_what_its_bar_printed_after() {
        // the long fills at the open of bar 1 and leaves at the open of bar 3, which
        // goes on to print 180 and 20 after the account had gone, so its excursions
        // are the one percent bar 2 showed (ADR 0023)
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 180.0, 20.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(bars, report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.step();
        e.close();
        e.step();
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert_eq!(t.close_tick, Some(3));
        assert!((t.best - 0.01).abs() < 1e-12, "best {}", t.best);
        assert!((t.worst + 0.01).abs() < 1e-12, "worst {}", t.worst);
    }

    #[test]
    fn a_limit_entry_is_not_credited_with_its_bar_before_it_filled() {
        // the bid rests below the market, so its bar opens at 130 and has to come
        // down to it, and nothing from the fill on printed above 101
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(130.0, 130.0, 100.0, 101.0, 1e6),
            ohlc(101.0, 101.0, 101.0, 101.0, 1e6),
            ohlc(101.0, 101.0, 101.0, 101.0, 1e6),
        ]);
        let mut e = Engine::new(bars, report_cfg());
        e.reset();
        e.limit_buy(1.0, 100.0);
        e.step();
        e.close();
        e.step();
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert_eq!((t.open_tick, t.entry), (1, 100.0));
        assert!((t.best - 0.01).abs() < 1e-12, "best {}", t.best);
        assert_eq!(t.worst, 0.0);
    }

    #[test]
    fn a_close_out_before_the_open_counts_none_of_the_bar_that_gapped() {
        // the long of 10 on 100 of margin dies at 90, on the way down to an open of
        // 85, so the 85 and the 80 after it are a market it had already left
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(85.0, 86.0, 80.0, 85.0, 1e6),
        ]);
        let config = EngineConfig {
            max_leverage: 10.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.step();
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert_eq!((t.exit, t.reason), (90.0, Some(FillReason::Liquidation)));
        assert!((t.worst + 0.1).abs() < 1e-12, "worst {}", t.worst);
        assert_eq!(t.best, 0.0);
    }

    #[test]
    fn a_short_is_helped_by_the_low_and_hurt_by_the_high() {
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 130.0, 70.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let config = EngineConfig {
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_sell(1.0);
        e.step();
        e.step();
        e.close();
        e.step();
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert!((t.best - 0.3).abs() < 1e-12, "best {}", t.best);
        assert!((t.worst + 0.3).abs() < 1e-12, "worst {}", t.worst);
    }

    #[test]
    fn a_fill_is_logged_with_its_order_its_reference_and_what_it_paid() {
        let config = EngineConfig {
            slippage_bps: 10.0,
            fee_taker: 0.0005,
            ..report_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        let id = e.market_buy(1.0).expect("placed");
        e.step(); // at the open of 200, 10 bps worse
        let [fill] = fills(&e) else {
            panic!("one fill");
        };
        assert_eq!((fill.tick, fill.order), (1, Some(id)));
        assert_eq!((fill.reason, fill.taker), (FillReason::Market, true));
        assert_eq!(fill.costs.reference, 200.0);
        assert!((fill.costs.slippage - 0.2).abs() < 1e-9);
        assert!((fill.price - 200.2).abs() < 1e-9);
        assert!((fill.fee - 200.2 * 0.0005).abs() < 1e-12);
    }

    #[test]
    fn a_stop_and_a_limit_are_logged_from_their_own_references() {
        // the next bar opens at 300 and trades from 290 to 360: the stop is reached
        // at its trigger and the limit as a maker at its price, the stop first
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(2.0);
        e.step();
        e.limit_sell(1.0, 320.0);
        e.stop(Side::Sell, 1.0, 295.0, true);
        e.step();
        let log = fills(&e);
        let reasons: Vec<FillReason> = log.iter().map(|f| f.reason).collect();
        assert_eq!(
            reasons,
            vec![FillReason::Market, FillReason::Stop, FillReason::Limit]
        );
        assert_eq!((log[1].costs.reference, log[1].price), (295.0, 295.0));
        let measured = FillCosts {
            reference: 320.0,
            ..FillCosts::default()
        };
        assert_eq!((log[2].costs, log[2].taker), (measured, false));
    }

    #[test]
    fn a_forced_close_is_logged_as_the_venues_with_no_order() {
        let mut e = Engine::new(crash_series(), levered());
        e.reset();
        e.market_buy(10.0);
        e.step(); // closed out at 90 on the way to the low of 80
        let close = fills(&e).last().copied().expect("the forced close");
        assert_eq!(close.order, None);
        assert_eq!(
            (close.reason, close.taker),
            (FillReason::Liquidation, false)
        );
        assert_eq!((close.price, close.costs.reference), (90.0, 90.0));
    }

    #[test]
    fn a_fill_the_margin_bound_lifts_logs_the_lift_as_its_adjustment() {
        // a stop armed with its entry triggers at 90, past the 901 / 9.99 where the
        // margin left after the entry's fee runs out, and is booked there
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            ..report_cfg()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        e.market_buy(10.0);
        e.stop(Side::Sell, 10.0, 90.0, true);
        e.step();
        let stop = fills(&e)
            .iter()
            .find(|f| f.reason == FillReason::Stop)
            .copied()
            .expect("the stop filled");
        assert_eq!(stop.costs.reference, 90.0);
        assert!((stop.price - 901.0 / 9.99).abs() < 1e-9);
        assert!((stop.costs.adjustment - (stop.price - 90.0)).abs() < 1e-12);
    }

    #[test]
    fn a_fill_pays_its_size_times_its_price_times_the_fee() {
        // the Specification's example: 0.141 BTC at 21,014.75 is 2,963.08 of
        // notional, and the taker fee of 0.05% on it is 1.48
        let bars = Candles::new(vec![
            ohlc(21_000.0, 21_000.0, 21_000.0, 21_000.0, 1e6),
            ohlc(21_014.75, 21_014.75, 21_014.75, 21_014.75, 1e6),
        ]);
        let config = EngineConfig {
            fee_taker: 0.0005,
            ..report_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(0.141);
        e.step();
        let [fill] = fills(&e) else {
            panic!("one fill");
        };
        assert!((fill.size * fill.price - 2_963.08).abs() < 0.005);
        assert!((fill.fee - 1.48).abs() < 0.005);
    }

    #[test]
    fn the_ledger_holds_a_row_per_bar_from_the_opening_balance() {
        let config = EngineConfig {
            fee_taker: 0.001,
            ..report_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(2.0);
        e.step(); // 2 at the open of 200, a fee of 0.4
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let ledger = reporter.ledger();
        assert_eq!(ledger.len(), 3, "one row per bar, the opening one included");
        assert_eq!((ledger[0].tick, ledger[0].equity), (0, 10_000.0));
        assert_eq!(ledger[0].costs, BarCosts::default());
        let (row, leg) = (ledger[1], reporter.legs()[1]);
        assert!((row.costs.fees - 0.4).abs() < 1e-12);
        assert_eq!((leg.position, leg.entry, leg.mark), (2.0, 200.0, 250.0));
        assert_eq!((leg.notional, leg.unrealized), (500.0, 100.0));
        assert!((leg.weight - 500.0 / row.equity).abs() < 1e-12);
        assert_eq!(ledger[2].costs, BarCosts::default());
    }

    #[test]
    fn the_ledger_identity_holds_on_every_bar() {
        // the change in equity is the price PnL, measured from each fill's reference
        // and then each bar's mark, less every cost, each of which is computed on its
        // own (ADR 0022). Slippage, impact a bar holds back, fees, funding, the margin
        // bound and a forced close all pass through it here
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 103.0, 98.0, 102.0, 1_000.0),
            ohlc(102.0, 102.5, 80.0, 85.0, 1_000.0),
            ohlc(85.0, 90.0, 84.0, 88.0, 1_000.0),
            ohlc(88.0, 89.0, 87.0, 88.0, 1_000.0),
        ]);
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            idle_cash: 50.0,
            fee_taker: 0.001,
            fee_maker: 0.0005,
            slippage_bps: 20.0,
            impact: 2.0,
            max_leverage: 10.0,
            funding_rate: 0.002,
            funding_interval: 1,
            report: true,
            ..cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(5.0);
        e.step();
        e.market_buy(4.0);
        e.stop(Side::Sell, 3.0, 90.0, true);
        e.step();
        e.step(); // the stop, then the forced close on the way to 80
        e.transfer(50.0, Wallet::Margin);
        e.market_sell(0.5);
        e.step(); // a short from the funded wallet
        e.step();

        let reporter = e.reporter().expect("reporting is on");
        let ledger = reporter.ledger();
        assert_eq!(ledger.len(), 6);
        assert!(ledger.iter().any(|row| row.costs.liquidation != 0.0));
        for t in 1..ledger.len() {
            let (was, row) = (ledger[t - 1], ledger[t]);
            let (held, leg) = (reporter.legs()[t - 1], reporter.legs()[t]);
            let mut pnl = held.position * (leg.mark - held.mark);
            for fill in reporter.fills().iter().filter(|fill| fill.tick == t) {
                pnl += fill.side.sign() * fill.size * (leg.mark - fill.costs.reference);
            }
            let c = row.costs;
            let costs = c.spread + c.slippage + c.impact + c.fees + c.funding;
            let costs = costs + c.liquidation + c.dust;
            let change = row.equity - was.equity;
            assert!(
                (change - (pnl - costs)).abs() < 1e-9 * row.equity.abs().max(1.0),
                "bar {t}: equity moved {change}, price PnL {pnl} less costs {costs}"
            );
        }
    }

    #[test]
    fn a_fill_at_its_reference_pays_nothing_beyond_it_and_costs_its_bar_its_fee() {
        // 2 at the open of 200 with no slippage: 0.4 of fee and nothing else
        let config = EngineConfig {
            fee_taker: 0.001,
            ..report_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(2.0);
        e.step();
        let [fill] = fills(&e) else {
            panic!("one fill");
        };
        let measured = FillCosts {
            reference: 200.0,
            ..FillCosts::default()
        };
        assert_eq!(fill.costs, measured);
        let costs = e.reporter().expect("reporting is on").ledger()[1].costs;
        assert!((costs.fees - 0.4).abs() < 1e-12);
        let charged = BarCosts {
            fees: costs.fees,
            ..BarCosts::default()
        };
        assert_eq!(costs, charged);
    }

    #[test]
    fn a_lift_by_the_margin_bound_is_a_negative_liquidation_charge_on_its_bar() {
        // the stop of the test above is booked 901 / 9.99 - 90 a unit over its
        // trigger, and the bar books that lift as a negative charge
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            ..report_cfg()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        e.market_buy(10.0);
        e.stop(Side::Sell, 10.0, 90.0, true);
        e.step();
        let row = e.reporter().expect("reporting is on").ledger()[1];
        let lift = 10.0 * (901.0 / 9.99 - 90.0);
        assert!(
            (row.costs.liquidation + lift).abs() < 1e-9,
            "{}",
            row.costs.liquidation
        );
    }

    #[test]
    fn a_fill_short_of_its_order_by_the_dust_epsilon_is_not_a_clamp() {
        // a bar trading 0.5 caps a buy of 0.5 and a dust's worth at 0.5, which is
        // the whole order to within the dust the engine ignores
        let bars = Candles::new(vec![ohlc(100.0, 100.0, 100.0, 100.0, 0.5); 2]);
        let mut e = Engine::new(bars, report_cfg());
        e.reset();
        e.market_buy(0.5 + CLOSE_EPS);
        e.step();
        let kinds: Vec<EventKind> = e
            .reporter()
            .expect("reporting is on")
            .events()
            .iter()
            .map(|event| event.kind)
            .collect();
        assert_eq!(kinds, vec![EventKind::Placed, EventKind::Filled]);
    }

    #[test]
    fn an_exit_inside_its_bar_counts_the_bar_from_its_open_to_the_fill() {
        // long 1 from 100; the next bar opens at 110 and falls through the stop at
        // 95, so the round trip saw 110 at that open and 95 at its exit
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(110.0, 110.0, 90.0, 92.0, 1e6),
        ]);
        let mut e = Engine::new(bars, report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.stop(Side::Sell, 1.0, 95.0, true);
        e.step();
        let [t] = trips(&e) else {
            panic!("one round trip");
        };
        assert_eq!((t.exit, t.reason), (95.0, Some(FillReason::Stop)));
        assert!((t.best - 0.1).abs() < 1e-12, "best {}", t.best);
        assert!((t.worst + 0.05).abs() < 1e-12, "worst {}", t.worst);
    }

    #[test]
    fn a_ledger_row_levers_notional_over_equity_and_a_dead_one_reads_zero() {
        // 2 bought at 200 and marked at 250 is 500 of notional on 10,100 of equity,
        // and an account liquidated to nothing has no leverage and no weight
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(2.0);
        e.step();
        let row = e.reporter().expect("reporting is on").ledger()[1];
        assert_eq!(row.leverage, 500.0 / 10_100.0);

        let mut dead = Engine::new(crash_series(), levered());
        dead.reset();
        dead.market_buy(10.0);
        dead.step();
        let reporter = dead.reporter().expect("reporting is on");
        let (row, leg) = (reporter.ledger()[1], reporter.legs()[1]);
        assert_eq!((row.equity, leg.position), (0.0, 0.0));
        assert_eq!((row.leverage, leg.weight), (0.0, 0.0));
    }
}
