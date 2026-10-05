//! Booking a fill, forced closes included: the account moves, the fill is
//! counted, whatever it closed is logged as a trade carrying both sides of its
//! fee when reporting is on, and the entry fee of what is left waits for the
//! close that consumes it (ADR 0023).

use probatum_core::{CostModel, Fill, Side};

use super::{Engine, CLOSE_EPS};
use crate::reporter::Trade;

impl Engine {
    /// The position fields a fill has to be compared against once it has landed.
    pub(super) fn before_fill(&self) -> (f64, f64, f64) {
        (
            self.account.position.qty.get(),
            self.account.position.avg_entry.get(),
            self.account.realized(),
        )
    }

    pub(super) fn apply_fill_clamped(&mut self, reduce_only: bool, fill: Fill) -> f64 {
        let mut fill = self.clamp_fill(reduce_only, &fill);
        if fill.size.get() <= 0.0 {
            return 0.0;
        }
        self.bound_by_margin(&mut fill);
        let applied = fill.size.get();
        let fee = self.cost.fee(&fill);
        let before = self.before_fill();
        self.account.apply_fill(&fill, fee);
        self.book_fill(applied, fill.price.get(), fee, before, false);
        applied
    }

    /// Book what a fill just did to the account: count it, log whatever it closed
    /// as a trade, carry the entry-side fee of whatever is left, and stamp a new
    /// position's open tick.
    ///
    /// The single place this happens, which is the point. A liquidation used to
    /// close the position on the account directly and reach none of it, so the
    /// forced close appeared in no trade row, counted toward no fill, and left the
    /// dead position's entry fee standing to be charged against the NEXT trade
    /// (ADR 0023).
    ///
    /// Twelve boundary mutants survive here and ALL TWELVE are equivalent or sit
    /// on an unreachable branch. Measured, not argued: probes asserting the three
    /// invariants below ran, at 1.0.0, against 179 Rust tests, 547 Python tests
    /// and 3,600 randomised differential cases without firing once.
    ///
    ///   `applied` is never zero or less. `apply_fill_clamped` returns before
    ///   calling this on a non-positive size and `liquidate` never books a flat
    ///   position, so the two `applied > 0.0` guards and the `else` that carries
    ///   `open_fee` without a fee share are all dead. Four mutants.
    ///
    ///   `closed` is never between zero and CLOSE_EPS. It comes from the
    ///   position's own dust rule (ADR 0014), so it is either exactly zero or
    ///   comfortably above the epsilon, and `>` cannot be told from `>=`. That
    ///   also makes the `&&` in the entry-fee guard equivalent to `||`: at
    ///   `closed == 0` the share is `open_fee * 0 / pos` either way. Four mutants.
    ///
    ///   `kept` is only ever read through `(pos_before.abs() - kept).max(0.0)`,
    ///   and every sign comparison in it collapses at exactly zero: opening from
    ///   flat closes nothing under both readings, and a full close closes the
    ///   whole old side under both. Two mutants, and two more on the new-side
    ///   stamp, which is guarded by `pos_after != 0.0` before either sign is read.
    ///
    /// So the count here overstates the gap. What this function actually does is
    /// pinned by value: `a_trade_carries_both_sides_of_its_fee`,
    /// `adding_to_a_position_does_not_double_charge_its_entry_fee` and the two
    /// partial-close tests.
    pub(super) fn book_fill(
        &mut self,
        applied: f64,
        exit_price: f64,
        fee: f64,
        before: (f64, f64, f64),
        liquidated: bool,
    ) {
        let (pos_before, entry_before, realized_before) = before;
        self.fills += 1;
        let pos_after = self.account.position.qty.get();

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

        // Split this fill's fee between the part that closed (the exit side of a
        // completed trade) and the part that opened (the entry side of the position
        // now standing, to be charged when it closes later).
        let exit_fee = if applied > 0.0 {
            fee * closed / applied
        } else {
            0.0
        };
        // The share of the position's own entry fee that this close consumes.
        let entry_fee = if closed > CLOSE_EPS && pos_before.abs() > CLOSE_EPS {
            self.open_fee * closed / pos_before.abs()
        } else {
            0.0
        };

        if closed > CLOSE_EPS && self.reporter.is_some() {
            let pnl = self.account.realized() - realized_before;
            // The whole round trip: the fee paid to open this size and the fee paid
            // to close it. Charging only the exit side made a strategy whose edge is
            // smaller than its costs look profitable per trade (ADR 0023).
            let fees = entry_fee + exit_fee;
            let trade = Trade {
                entry_tick: self.position_entry_tick,
                exit_tick: self.tick,
                side: if pos_before > 0.0 {
                    Side::Buy
                } else {
                    Side::Sell
                },
                size: closed,
                entry_price: entry_before,
                exit_price,
                fees,
                pnl,
                net_pnl: pnl - fees,
                bars_held: self.tick.saturating_sub(self.position_entry_tick),
                liquidated,
            };
            if let Some(reporter) = self.reporter.as_mut() {
                reporter.record_trade(trade);
            }
        }

        // Carry the entry-side fee of whatever position is left: what this close
        // consumed is gone, and the part of this fill that opened size pays in.
        self.open_fee = if pos_after.abs() <= CLOSE_EPS {
            0.0
        } else if applied > 0.0 {
            (self.open_fee - entry_fee) + fee * opened / applied
        } else {
            self.open_fee - entry_fee
        };

        // Stamp the open tick whenever a new side is established, from flat or
        // through a flip, so the next close measures its holding time from here.
        let opened_new_side =
            pos_after != 0.0 && (pos_before == 0.0 || (pos_before > 0.0) != (pos_after > 0.0));
        if opened_new_side {
            self.position_entry_tick = self.tick;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{cfg, flat_series, ohlc, perp_cfg, report_cfg, series};
    use crate::engine::{Engine, EngineConfig};
    use probatum_core::{Candle, Fill, Price, Qty, Side};

    #[test]
    fn a_close_records_a_trade_with_realized_pnl() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2 @ 200 (bar 1 open)
        e.close();
        e.step(); // close 2 @ 300 (bar 2 open)
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        let t = trades[0];
        assert_eq!(t.side, Side::Buy);
        assert_eq!(t.size, 2.0);
        assert_eq!(t.entry_price, 200.0);
        assert_eq!(t.exit_price, 300.0);
        assert_eq!(t.pnl, 200.0); // 2 * (300 - 200)
        assert_eq!(t.entry_tick, 1);
        assert_eq!(t.exit_tick, 2);
        assert_eq!(t.bars_held, 1);
    }

    #[test]
    fn a_partial_close_records_only_the_reduced_size() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(5.0);
        e.step(); // long 5 @ 200
        e.market_sell(2.0);
        e.step(); // sell 2 @ 300, still long 3
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].size, 2.0);
        assert_eq!(trades[0].pnl, 200.0); // 2 * (300 - 200)
    }

    #[test]
    fn a_flip_records_only_the_closed_portion() {
        // a flip through zero reopens on the opposite side; spot cannot go short
        // (ADR 0008), so the flip is exercised on a perp
        let mut e = Engine::new(
            series(),
            EngineConfig {
                report: true,
                ..perp_cfg()
            },
        );
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2 @ 200
        e.market_sell(5.0);
        let s = e.step(); // sell 5 @ 300: close 2, open short 3
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].size, 2.0); // only the closed 2, not the whole 5
        assert_eq!(trades[0].pnl, 200.0);
        assert_eq!(s.position, -3.0); // flipped to short 3
    }

    #[test]
    fn adding_to_a_position_records_no_trade() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
        e.market_buy(1.0);
        e.step(); // add 1 @ 300, no close
        assert!(e.reporter().unwrap().trades().is_empty());
    }

    #[test]
    fn logged_trade_pnl_reconciles_with_realized() {
        // the Validation Guide's reconciliation invariant: with no liquidation, the
        // sum of logged trade PnL equals the account's cumulative realized PnL
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
        e.step(); // flip to short 1, books the long's PnL
        e.market_buy(1.0);
        let last = e.step(); // close the short, books its PnL

        let trade_sum: f64 = e.reporter().unwrap().trades().iter().map(|t| t.pnl).sum();
        assert!(!e.reporter().unwrap().trades().is_empty()); // trades were logged
        assert!((trade_sum - last.realized_pnl).abs() < 1e-9);
    }

    #[test]
    fn net_trade_pnl_accounts_for_the_whole_run_when_it_ends_flat() {
        // the invariant the entry-fee attribution buys: with the position closed and
        // no liquidation, the net PnL of the logged trades IS the change in equity.
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

        let trades = e.reporter().unwrap().trades();
        assert!(trades.len() > 3, "only {} trades", trades.len());
        let net: f64 = trades.iter().map(|t| t.net_pnl).sum();
        let equity_change = state.equity - 10_000.0;
        assert!(
            (net - equity_change).abs() < 1e-9,
            "sum(net_pnl) {net} != equity change {equity_change}"
        );
        // and the gross figure alone does not reconcile, which is the whole point
        let gross: f64 = trades.iter().map(|t| t.pnl).sum();
        assert!((gross - equity_change).abs() > 1e-6);
        for t in trades {
            assert!((t.net_pnl - (t.pnl - t.fees)).abs() < 1e-12);
        }
    }

    #[test]
    fn a_trade_carries_both_sides_of_its_fee() {
        let mut config = cfg(); // spot, bar 1 open 200, bar 2 open 300
        config.report = true;
        config.fee_taker = 0.01;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // buy 1 at 200, fee 2.0
        e.close();
        e.step(); // sell 1 at 300, fee 3.0
        let t = e.reporter().unwrap().trades()[0];
        assert!((t.pnl - 100.0).abs() < 1e-9); // gross price PnL
        assert!((t.fees - 5.0).abs() < 1e-9); // 2.0 entry plus 3.0 exit
        assert!((t.net_pnl - 95.0).abs() < 1e-9);
    }

    #[test]
    fn adding_to_a_position_does_not_double_charge_its_entry_fee() {
        // buy 1 at 200 (fee 2), buy 1 at 400 (fee 4), then close both at 300 (fee 6).
        // The whole position closes, so the row must carry every fee paid: 12, not 16.
        // A same-side add keeps more than it started with, so the closed amount goes
        // negative unless it is clamped, and the opening share of the fee is counted
        // twice.
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

        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert!(
            (trades[0].fees - 12.0).abs() < 1e-9,
            "fees {} should be 2 + 4 + 6",
            trades[0].fees
        );
        assert!((trades[0].pnl - 0.0).abs() < 1e-9);
        assert!((trades[0].net_pnl + 12.0).abs() < 1e-9);
        // the identity must survive a scale-in as well as a single entry
        assert!((trades[0].net_pnl - (last.equity - 1_000_000.0)).abs() < 1e-9);
    }

    #[test]
    fn net_pnl_reconciles_through_scale_ins_and_partial_exits() {
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
        let trades = e.reporter().unwrap().trades();
        assert!(trades.len() > 3, "only {} trades", trades.len());
        let net: f64 = trades.iter().map(|t| t.net_pnl).sum();
        assert!(
            (net - (state.equity - 10_000.0)).abs() < 1e-9,
            "sum(net_pnl) {net} != equity change {}",
            state.equity - 10_000.0
        );
    }

    #[test]
    fn a_single_partial_close_splits_the_entry_fee_by_size() {
        let mut config = cfg();
        config.report = true;
        config.fee_taker = 0.01;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(4.0);
        e.step(); // buy 4 at 200, entry fee 8.0 for the whole position
        e.market_sell(1.0);
        e.step(); // sell 1 at 300, exit fee 3.0; entry share is 8.0 * 1/4 = 2.0
        let t = e.reporter().unwrap().trades()[0];
        assert!((t.size - 1.0).abs() < 1e-9);
        assert!((t.fees - 5.0).abs() < 1e-9);
        assert!((t.net_pnl - 95.0).abs() < 1e-9);
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
        assert_eq!(e.clamp_fill(false, &dust).size.get(), 0.0);
        assert_eq!(e.apply_fill_clamped(false, dust), 0.0);
        assert_eq!(e.num_fills(), 0);
        assert_eq!(e.state().position, 0.0);
    }

    #[test]
    fn a_holding_time_is_measured_from_when_the_side_was_opened() {
        // bars_held rests on one predicate deciding whether a fill opened a new
        // side, and every branch of it was reachable only through a whole run, so
        // six single-line mutations of it survived. Adding to a position must not
        // restart the clock, a partial close must not either, and a flip must
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
        e.step(); // tick 2: long 2, still opened at 1
        e.market_sell(1.0);
        e.step(); // tick 3: closes 1 of it, held 3 - 1
        e.market_sell(3.0);
        e.step(); // tick 4: closes the last 1, held 4 - 1, and opens short 2 here
        e.market_buy(2.0);
        e.step(); // tick 5: closes the short, held 5 - 4

        let held: Vec<usize> = e
            .reporter()
            .expect("reporting is on")
            .trades()
            .iter()
            .map(|t| t.bars_held)
            .collect();
        assert_eq!(held, vec![2, 3, 1]);
    }

    #[test]
    fn two_partial_closes_consume_the_entry_fee_exactly_once() {
        // ADR 0023's whole point: a trade carries its round trip, so the entry fee
        // is split across the closes by size and consumed exactly once. The
        // arithmetic that does it had no test of its own
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
        e.step(); // exit fee 1.0, plus half the entry fee
        e.market_sell(1.0);
        e.step(); // exit fee 1.0, plus the other half

        let fees: Vec<f64> = e
            .reporter()
            .expect("reporting is on")
            .trades()
            .iter()
            .map(|t| t.fees)
            .collect();
        assert_eq!(fees, vec![2.0, 2.0]);
        // and the whole round trip is exactly what the account paid
        let paid: f64 = fees.iter().sum();
        assert!((paid - 4.0).abs() < 1e-9, "{paid}");
    }
}
