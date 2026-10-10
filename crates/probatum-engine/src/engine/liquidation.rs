//! Where the margin behind a perp position runs out. The fence is worked out
//! before a bar's orders are resolved and clips the bar there (ADR 0017); the
//! forced close is priced at the bankruptcy price rather than at the bar's
//! extreme, and no fill that reduces the position books past that price, so an
//! account can end at zero but never owing (ADR 0018). The fence is the one perp's
//! while the account holds one, and the bound counts every other perp held.

use probatum_core::{Candle, CostModel, Fill, Market, OrderId, Price, Qty, Side};

use super::ledger::Moment;
use super::Engine;
use crate::execution::FillCosts;
use crate::reporter::{EventKind, EventReason, FillReason, FillRecord};

impl Engine {
    /// True when perp `p`, the only perp the account holds, is dead at `price` once
    /// `debits` of funding are counted against the margin wallet: the check at a
    /// bar's adverse extreme, with the debits stamped inside the bar (ADRs 0008,
    /// 0017, 0020).
    pub(super) fn dead_at(&self, p: usize, price: f64, debits: f64) -> bool {
        let unrealized = self.account.positions[p].unrealized(Price(price));
        price.is_finite() && self.account.margin_cash + unrealized - debits <= 0.0
    }

    /// The price `bar` would take perp `p`'s margin to zero at, or `None` when
    /// nothing on the bar can kill the account.
    ///
    /// The trigger is the bar's adverse extreme and the answer is the bankruptcy
    /// price, which are two different numbers on purpose (ADR 0018). Read before
    /// the bar's orders are resolved, because it is the point past which this
    /// account was no longer in the market.
    pub(super) fn liquidation_fence(&self, p: usize, bar: &Candle) -> Option<f64> {
        let adverse = if self.account.positions[p].qty.get() >= 0.0 {
            bar.low
        } else {
            bar.high
        };
        if !self.dead_at(p, adverse, 0.0) {
            return None;
        }
        self.account
            .bankruptcy_price(p, self.cost.fee_taker, self.account.margin_cash)
            .map(|price| price.get())
    }

    /// The bar as far as this account got: a long sees nothing below `fence` and a
    /// short nothing above it. Clamping is monotone, so the OHLC ordering survives
    /// it, and a bar lying entirely past the fence collapses to the fence itself,
    /// which is the honest picture of a market the account was not in.
    pub(super) fn fenced(bar: &Candle, fence: f64, long: bool) -> Candle {
        let clip = |price: f64| {
            if long {
                price.max(fence)
            } else {
                price.min(fence)
            }
        };
        Candle {
            open: clip(bar.open),
            high: clip(bar.high),
            low: clip(bar.low),
            close: clip(bar.close),
            volume: bar.volume,
        }
    }

    /// Force-close perp `p`, the one the account holds, at the price that leaves
    /// nothing.
    ///
    /// Not at the bar's adverse extreme, which is what the trigger was tested
    /// against. Closing there charged the account for however far the bar happened
    /// to run past the point its margin was gone, so a long on 100 of margin
    /// against a bar that wicked to 80 booked a loss of 200 and left the account
    /// owing 100. A position cannot lose more than the margin behind it, so the
    /// exit is priced where the margin runs out, and the liquidation fee is inside
    /// that price rather than charged past it (ADR 0018).
    pub(super) fn liquidate(&mut self, p: usize, moment: Moment) {
        let fee = self.cost.fee_taker;
        if let Some(price) = self
            .account
            .bankruptcy_price(p, fee, self.account.margin_cash)
        {
            self.close_out(p, price.get(), moment);
        }
    }

    /// Force-close every perp the account holds, each at its entry in `prices`,
    /// paying the liquidation fee, and leave the margin wallet at exactly zero:
    /// whatever it then holds, positive or negative, is the liquidation charge. A
    /// funding debit that takes the account to its trigger at a bar's close is
    /// closed here at the closes (ADR 0020).
    pub(super) fn liquidate_at(&mut self, prices: &[f64], moment: Moment) {
        for (i, &price) in prices.iter().enumerate() {
            let qty = self.account.positions[i].qty.get();
            if self.account.markets[i] != Market::Perp || qty == 0.0 {
                continue;
            }
            let fill = Fill {
                side: if qty > 0.0 { Side::Sell } else { Side::Buy },
                size: Qty(qty.abs()),
                price: Price(price),
                is_taker: true,
            };
            let fee = self.cost.fee(&fill);
            let before = self.before_fill(i);
            self.account.apply_fill(i, &fill, fee);
            self.book_fill(
                i,
                qty.abs(),
                price,
                fee,
                before,
                FillReason::Liquidation,
                moment,
            );
            if let Some(reporter) = self.reporter.as_mut() {
                reporter.costs_mut().fees += fee;
                reporter.record_fill(FillRecord {
                    tick: self.tick,
                    instrument: i,
                    order: None,
                    side: fill.side,
                    size: qty.abs(),
                    price,
                    costs: FillCosts {
                        reference: price,
                        ..FillCosts::default()
                    },
                    fee,
                    taker: false,
                    reason: FillReason::Liquidation,
                });
            }
        }
        // whatever the wallet still holds once the positions are closed, taken by
        // the venue or absorbed by it, is the liquidation charge (ADR 0018)
        if let Some(reporter) = self.reporter.as_mut() {
            reporter.costs_mut().liquidation += self.account.margin_cash;
        }
        // The bankruptcy price leaves nothing in exact arithmetic and a hair either
        // side of it in floating point, and a hair above zero is equity the cap
        // levers: with the cap off, an account closed out there opened a position
        // again on the next bar. So the wallet ends at exactly zero (ADRs 0015, 0018).
        self.account.margin_cash = 0.0;
        // Every perp order the closed-out account left goes with it. The empty
        // wallet refused them anyway until a transfer could fund it again, after
        // which they filled for an account that had already left the market
        // (ADR 0018). The spot wallet is untouched, and its orders stand.
        let perp = |markets: &[Market], i: usize| markets[i] == Market::Perp;
        if self.reporter.is_some() {
            let left: Vec<(OrderId, usize)> = self
                .pending
                .iter()
                .chain(self.book.iter())
                .filter(|order| perp(&self.account.markets, order.instrument))
                .map(|order| (order.id, order.instrument))
                .collect();
            for (id, instrument) in left {
                self.log_event(
                    EventKind::Cancelled,
                    Some(id),
                    Some(instrument),
                    None,
                    None,
                    EventReason::Liquidation,
                );
            }
        }
        let markets = &self.account.markets;
        self.pending
            .retain(|order| !perp(markets, order.instrument));
        self.book
            .cancel_where(|order| perp(markets, order.instrument));
    }

    /// No fill closes a perp past the point the margin behind it ran out.
    ///
    /// `fenced` clips the whole candle, and it is read once from the position
    /// carried INTO the bar, which is the approximation ADR 0017 accepted on the
    /// grounds that its error runs pessimistic. It does not, in three shapes. A
    /// position GROWN during the bar and exited on the same bar is bounded by a
    /// fence computed for a smaller position, or by no fence at all when the
    /// account entered the bar flat, so a market entry and a protective stop
    /// armed in one decision booked straight through the margin and left the
    /// account owing money. A resting limit prices at its own limit, which
    /// never passes through the taker clamp, so it walks out of the fenced candle
    /// from underneath. And a partial exit that drives quote negative leaves a
    /// residual for the forced close to price off a balance the closed form at
    /// `bankruptcy_price` assumes is positive, which solves to a price on the far
    /// side of the entry: a liquidated WINNER in the trade log, on a dead account.
    /// That third one is the dangerous shape, because the first two are loud and
    /// pessimistic while it flatters. Bounding the fill instead of the bar covers
    /// all three, because the bound is read at the moment the fill lands rather
    /// than a step earlier (ADR 0018).
    ///
    /// Closing at the bankruptcy price leaves equity exactly zero AT that price,
    /// so a partial close is bounded by the same number as a full one and no
    /// ordering of fills within a bar can book past it. With other perps held, the
    /// price is solved against the wallet with what they hold beyond their own
    /// closing fees, each at the fraction of its own bar the fill sits at on its
    /// path, so no reducing fill on one leg overdraws the account its other legs
    /// still stand in.
    ///
    /// The lift stops at the position's average entry, past which the close owes
    /// nothing, and inside `candle`, the bar the fill was priced on: a price the bar
    /// never printed is no trade. What the bar cannot hold, the part of the lift past
    /// it on the size closed, net of the fee it would have moved, is returned to be
    /// paid into the margin wallet (Specification, The margin bound).
    pub(super) fn bound_by_margin(&self, i: usize, fill: &mut Fill, candle: &Candle) -> f64 {
        if self.account.markets[i] != Market::Perp {
            return 0.0;
        }
        let position = self.account.positions[i].qty.get();
        let long = position > 0.0;
        let reduces =
            (long && fill.side == Side::Sell) || (position < 0.0 && fill.side == Side::Buy);
        if !reduces {
            return 0.0;
        }
        let fee = self.cost.fee_taker;
        let price = fill.price.get();
        let wallet = self.wallet_beside(i, fee, price);
        let Some(star) = self.account.bankruptcy_price(i, fee, wallet) else {
            return 0.0;
        };
        let entry = self.account.positions[i].avg_entry.get();
        let (goal, lifted) = if long {
            let goal = star.get().min(entry);
            (goal, price.max(goal.min(candle.high)))
        } else {
            let goal = star.get().max(entry);
            (goal, price.min(goal.max(candle.low)))
        };
        fill.price = Price(lifted);
        let closed = fill.size.get().min(position.abs());
        let rate = if fill.is_taker {
            self.cost.fee_taker
        } else {
            self.cost.fee_maker
        };
        // what closing that part realizes at a price, net of the fill's own fee
        let kept = |p: f64| position.signum() * closed * (p - entry) - closed * p * rate;
        (kept(goal) - kept(lifted)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{cfg, crash_series, levered, ohlc, perp_cfg, universe};
    use crate::engine::{Engine, EngineConfig, CLOSE_EPS};
    use crate::reporter::{EventKind, EventReason, FillReason};
    use probatum_core::{Market, OrderType, Side, TimeInForce, Wallet};

    #[test]
    fn leveraged_long_is_liquidated_on_a_crashing_bar() {
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        e.market_buy(10.0); // 10x notional on 100 margin, fills at bar 1 open 100
        let s1 = e.step(); // bar 1 low 80: equity 100 + 10*(80-100) = -100 -> liquidated
        assert_eq!(s1.position, 0.0);
    }

    #[test]
    fn is_bust_is_set_by_liquidation_and_cleared_by_reset() {
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        assert!(!e.is_bust());
        e.market_buy(10.0);
        e.step(); // liquidated on the crash bar
        assert!(e.is_bust());
        e.reset();
        assert!(!e.is_bust());
    }

    #[test]
    fn a_liquidation_cancels_the_orders_the_account_left_behind() {
        // the spot wallet funds the margin wallet again, and a sell limit the
        // closed-out account left resting would have opened a short with it
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 80.0, 85.0, 1e6),
            ohlc(85.0, 120.0, 85.0, 110.0, 1e6),
        ]);
        let config = EngineConfig {
            idle_cash: 500.0,
            ..levered()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(10.0);
        e.limit_sell(1.0, 115.0);
        let s = e.step(); // the long dies at 90 on the way to 80
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty());
        e.transfer(500.0, Wallet::Margin);
        let s = e.step(); // the bar trades through 115
        assert_eq!(s.position, 0.0);
        assert_eq!(s.margin_cash, 500.0);
    }

    /// Perp at 10x with a fee, reporting on, over `crash_series`.
    fn perp_10x() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            report: true,
            ..cfg()
        }
    }

    #[test]
    fn a_stop_armed_with_the_entry_cannot_book_past_the_margin() {
        // the fence is read from the position carried INTO the bar, and an account
        // that entered flat carries none, so the stop reduced a position the
        // margin no longer stood behind and booked the whole distance. Placing a
        // stop was four times worse than placing nothing (ADR 0018)
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(10.0);
        e.stop(Side::Sell, 10.0, 90.0, true);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);

        // and it is no worse than the same bar with nothing resting on it
        let mut bare = Engine::new(crash_series(), perp_10x());
        bare.reset();
        bare.market_buy(10.0);
        let alone = bare.step();
        assert!((s.equity - alone.equity).abs() < CLOSE_EPS);
    }

    #[test]
    fn a_partial_exit_that_empties_the_margin_books_no_winning_liquidation() {
        // the residual was liquidated off a NEGATIVE quote, which solves to a price
        // on the far side of the entry: a profit booked on a bar that never traded
        // there, lifting win_rate and profit_factor while the return read -100
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        // the trigger is BELOW the bankruptcy price of 901/9.99, about 90.19, so
        // the partial exit alone loses more than the whole margin: six units at
        // 82 is 108 against 99 of it, which leaves the quote negative and four
        // units still open for the forced close to price off
        e.market_buy(10.0);
        e.stop(Side::Sell, 6.0, 82.0, true);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);
        for trade in e.reporter().expect("reporting is on").trades() {
            assert!(
                !(trade.reason == Some(FillReason::Liquidation) && trade.pnl > 0.0),
                "liquidated at {} for a profit of {}",
                trade.exit,
                trade.pnl
            );
        }
    }

    #[test]
    fn a_resting_limit_cannot_price_an_exit_past_the_margin() {
        // fill_limit returns the limit itself and never passes through the taker
        // clamp, so it got out from under the fenced candle the trigger respected
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(10.0);
        e.limit_sell(10.0, 20.0);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);
    }

    #[test]
    fn a_healthy_account_prices_its_exit_where_it_asked() {
        // the bound must bite only where the margin has actually run out, or it
        // would quietly improve every ordinary exit
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(1.0); // a tenth of the cap, nowhere near the fence
        e.limit_sell(1.0, 99.0);
        e.step();
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert!(
            (trades[0].exit - 99.0).abs() < 1e-9,
            "exited at {}",
            trades[0].exit
        );
    }

    #[test]
    fn a_liquidation_is_booked_like_any_other_close() {
        // the forced close ran on the account directly and reached none of the
        // engine's bookkeeping: it appeared in no trade row, counted toward no
        // fill, and left the dead position's entry fee standing to be charged
        // against the NEXT position (ADR 0023)
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            report: true,
            ..cfg()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        e.market_buy(10.0); // 1000 of notional on 100 of quote, exactly at the cap
        let s = e.step(); // fills at the open of 100, then the low of 80 busts it
        assert!(e.is_bust());
        assert_eq!(s.position, 0.0);
        assert_eq!(e.num_fills(), 2); // the buy, and the forced close
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
        // closed where the margin ran out, not at the 80 the bar printed. The entry
        // fee left 99 of margin, so (10*100 - 99) / (10 - 10*0.001) = 90.1902
        assert!(
            (trades[0].exit - 901.0 / 9.99).abs() < 1e-9,
            "closed at {}",
            trades[0].exit
        );
        // and the account is left with exactly nothing, never owing (ADR 0018)
        assert!(s.equity.abs() < CLOSE_EPS, "left {}", s.equity);
        assert!(trades[0].fees > 1.0); // the entry fee, plus a real fee on the exit
                                       // and nothing is left over to charge the next position
        assert!(e
            .reporter()
            .expect("reporting is on")
            .open_trades()
            .next()
            .is_none());
    }

    #[test]
    fn a_liquidation_can_never_leave_the_account_owing() {
        // whatever the bar prints past the point the margin is gone, the loss is
        // bounded by the margin: closing at the extreme booked 200 on a 100 account
        for low in [90.0, 80.0, 40.0, 1.0] {
            let config = EngineConfig {
                market: Market::Perp,
                quote: 100.0,
                fee_taker: 0.0006,
                max_leverage: 10.0,
                ..cfg()
            };
            let candles = Candles::new(vec![
                ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                ohlc(100.0, 101.0, low, low, 1000.0),
                ohlc(low, low + 1.0, low - 1.0, low, 1000.0),
            ]);
            let mut e = Engine::new(candles, config);
            e.reset();
            e.market_buy(10.0);
            let s = e.step();
            assert!(s.equity >= -CLOSE_EPS, "a low of {low} left {}", s.equity);
            assert!(
                s.quote >= -CLOSE_EPS,
                "a low of {low} left quote {}",
                s.quote
            );
        }
    }

    #[test]
    fn a_forced_close_leaves_the_wallet_at_exactly_zero() {
        // the bankruptcy price leaves nothing in exact arithmetic and a hair either
        // side of it in floating point, and a hair above zero is equity the cap
        // levers: with the cap off, an account the engine had just closed out
        // opened a position again, which the next bar liquidated (ADRs 0015, 0018)
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 150.0, 50.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        for fee in [0.0006, 0.002] {
            for size in [23.289_100_547_331_138, 38.434_488_167_935_7, 61.7] {
                for long in [true, false] {
                    let config = EngineConfig {
                        market: Market::Perp,
                        quote: 1_000.0,
                        fee_taker: fee,
                        max_leverage: 0.0,
                        ..cfg()
                    };
                    let mut e = Engine::new(bars.clone(), config);
                    e.reset();
                    if long {
                        e.market_buy(size);
                    } else {
                        e.market_sell(size);
                    }
                    let s = e.step();
                    assert!(e.is_bust(), "fee {fee} size {size} long {long}");
                    assert_eq!(s.quote, 0.0, "fee {fee} size {size} long {long}");
                }
            }
        }
    }

    // A first bar at 100, then a bar that opens at 50, far past where a 10x long
    // opened at 100 runs out of margin.
    fn gap_series() -> Candles {
        Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(50.0, 50.0, 20.0, 30.0, 1_000.0),
            ohlc(30.0, 31.0, 29.0, 30.0, 1_000.0),
        ])
    }

    #[test]
    fn closing_on_a_bar_that_gapped_past_the_liquidation_leaves_nothing_owed() {
        // ADR 0018 says bad debt is structurally unreachable rather than clamped,
        // and that rested on the position still being open when the check ran at
        // the end of the bar. An exit order flattened it first, so nothing was left
        // to liquidate and the account booked the whole gap: a 10x long on 100 of
        // margin ended at -400. The run with the exit must land where the run
        // without one does, because the account died before either could matter
        for exit in [false, true] {
            let mut e = Engine::new(gap_series(), levered());
            e.reset();
            e.market_buy(10.0);
            e.step();
            if exit {
                e.close();
            }
            let state = e.step();
            assert_eq!(state.equity, 0.0, "exit={exit} left {}", state.equity);
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades.len(), 1);
            assert_eq!(trades[0].exit, 90.0);
            assert_eq!(
                trades[0].reason,
                Some(FillReason::Liquidation),
                "exit={exit} booked no liquidation"
            );
        }
    }

    #[test]
    fn a_stop_below_the_bankruptcy_price_never_fills_and_one_above_it_does() {
        // the fence is a price, not a moment: coming down from the open the market
        // reaches 95 before 90, so a stop at 95 saves the account and one at 60 is
        // beyond a point it had already been closed at
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 50.0, 60.0, 1_000.0),
            ohlc(60.0, 61.0, 59.0, 60.0, 1_000.0),
        ]);
        for (trigger, want_equity, want_exit, reason) in [
            (95.0, 50.0, 95.0, FillReason::Stop),
            (60.0, 0.0, 90.0, FillReason::Liquidation),
        ] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_buy(10.0);
            e.step();
            e.stop(Side::Sell, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades[0].exit, want_exit, "stop at {trigger}");
            assert_eq!(trades[0].reason, Some(reason), "stop at {trigger}");
        }
    }

    #[test]
    fn a_liquidation_never_books_a_price_outside_the_bar_that_triggered_it() {
        // a partial close on the gap bar drove quote negative, and the bankruptcy
        // solve is derived assuming quote is positive, so it landed on the far side
        // of the entry: the forced close booked a PROFIT at 120 on a bar whose high
        // was 72, and the log showed a liquidated winner
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(70.0, 72.0, 60.0, 65.0, 1e6),
            ohlc(65.0, 66.0, 64.0, 65.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.market_sell(6.0);
        let state = e.step();
        assert_eq!(state.equity, 0.0);
        for trade in e.reporter().expect("reporting is on").trades() {
            assert!(trade.pnl <= 0.0, "a liquidation booked {}", trade.pnl);
            assert!(trade.exit <= 101.0, "exited at {}", trade.exit);
        }
    }

    #[test]
    fn a_wick_that_busts_a_long_liquidates_even_when_the_bar_closes_back_above_it() {
        // ADR 0017 puts the trigger at the bar's ADVERSE EXTREME, the low for a
        // long, because a bar that traded there is a bar the account was closed out
        // on whatever it printed afterwards. Every other liquidation fixture ends
        // at or below its own bust level, so marking the check at bar.close passes
        // all of them; this bar wicks to 85 and closes back at 100, where a
        // close-marked check reads an equity of 100 and lets a dead account trade on.
        //
        // The favourable wick reaches 130, which is TWICE as far from the entry as
        // the adverse one: a check reading whichever extreme is furthest from the
        // entry, rather than the one that is adverse to this side, marks the long
        // at 130 and finds it healthy. On a bar wicking to 85 and 101 the two
        // readings agree, which is why the high is 130 and not 101
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 130.0, 85.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0); // 1000 of notional on 100 of margin, filled at the open of 100
        let state = e.step();
        // margin at the low is 100 + 10 * (85 - 100) = -50, so the account died on
        // the way down and is closed where the margin ran out, at
        // (10 * 100 - 100) / 10 = 90, leaving exactly nothing (ADR 0018)
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        assert_eq!(state.equity, 0.0);
        assert_eq!(state.bar_close, 100.0); // the bar recovered, the account did not
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit, 90.0);
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
    }

    #[test]
    fn a_wick_that_busts_a_short_liquidates_even_when_the_bar_closes_back_below_it() {
        // the mirror of the rule above, and no engine test has ever liquidated a
        // short at all: a short's adverse extreme is the HIGH, so a wick up that
        // takes the margin kills it however calmly the bar ends (ADR 0017). A check
        // reading bar.close, or one that kept the long's low for both sides, reads
        // 100 here and carries a dead short into the next bar. The favourable wick
        // reaches 70, twice as far from the entry as the adverse one, so a check
        // reading the furthest extreme rather than the adverse one marks the short
        // at 70 and finds it richer than ever
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 115.0, 70.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_sell(10.0); // short 10 at the open of 100, on 100 of margin
        let state = e.step();
        // margin at the high is 100 - 10 * (115 - 100) = -50, and the margin runs
        // out at (-10 * 100 - 100) / -10 = 110, which is where it is closed
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        assert_eq!(state.equity, 0.0);
        assert_eq!(state.bar_close, 100.0); // the bar came back, the account did not
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit, 110.0);
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
    }

    #[test]
    fn a_bar_that_gaps_up_past_a_shorts_bankruptcy_price_closes_it_at_the_fence() {
        // ADR 0017 works the fence out before the bar's orders are resolved, and it
        // faces the position's OWN direction: a short dies going up, so a bar that
        // OPENS above its bankruptcy price is a bar the account was already closed
        // out on. Every fence fixture is a 10x long, so the short side of that
        // comparison was never taken; with the long's test a close() here books the
        // raw open of 150 and leaves the account owing 400 (ADR 0018)
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(150.0, 180.0, 140.0, 160.0, 1e6),
            ohlc(160.0, 161.0, 159.0, 160.0, 1e6),
        ]);
        for exit in [false, true] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_sell(10.0); // short 10 at 100, on 100 of margin
            e.step();
            if exit {
                e.close();
            }
            let state = e.step();
            // the margin runs out at (-10 * 100 - 100) / -10 = 110 and the bar
            // opened past it, so the exit is priced there and leaves nothing
            assert_eq!(state.position, 0.0, "exit={exit}");
            assert_eq!(state.equity, 0.0, "exit={exit} left {}", state.equity);
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades.len(), 1, "exit={exit}");
            assert_eq!(trades[0].exit, 110.0, "exit={exit}");
            assert_eq!(
                trades[0].reason,
                Some(FillReason::Liquidation),
                "exit={exit} booked no liquidation"
            );
        }
    }

    #[test]
    fn a_shorts_stop_above_the_bankruptcy_price_never_fills_and_one_below_it_does() {
        // the mirror of the long fence: a short's bar is clipped at the fence from
        // ABOVE, so coming up from the open the market reaches 105 before 110 and a
        // buy stop there saves the account, while one at 140 sits beyond a price
        // the account had already been closed at. Clipping a short's bar with the
        // long's max() leaves that 140 reachable and books an exit 30 past the
        // point the margin was gone (ADR 0017)
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 150.0, 99.0, 140.0, 1e6),
            ohlc(140.0, 141.0, 139.0, 140.0, 1e6),
        ]);
        for (trigger, want_equity, want_exit, reason) in [
            (105.0, 50.0, 105.0, FillReason::Stop),
            (140.0, 0.0, 110.0, FillReason::Liquidation),
        ] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_sell(10.0);
            e.step(); // short 10 at 100
            e.stop(Side::Buy, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades[0].exit, want_exit, "stop at {trigger}");
            assert_eq!(trades[0].reason, Some(reason), "stop at {trigger}");
        }
    }

    #[test]
    fn a_long_gapped_past_its_fence_is_closed_before_its_bar_can_fill_a_target() {
        // long 10 from 100 on 100 of margin runs out at 90, and the next bar opens at
        // 85: the account is closed at 90 before that bar opens, so the take-profit
        // at 94 its high of 95 reaches never fills
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 92.0, 93.0, 1e6),
            ohlc(85.0, 95.0, 80.0, 90.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.limit_sell(10.0, 94.0);
        let state = e.step();
        assert!(e.is_bust());
        assert_eq!(state.equity, 0.0);
        let [t] = e.reporter().expect("reporting is on").trades() else {
            panic!("one round trip");
        };
        assert_eq!((t.exit, t.reason), (90.0, Some(FillReason::Liquidation)));
    }

    #[test]
    fn a_close_out_at_the_close_books_the_wallets_deficit_as_a_negative_charge() {
        // a debit of 200 of funding leaves 100 of margin at -100 at the close, the
        // close at 100 realizes nothing, and the deficit the venue absorbs is a
        // charge of -100, so the bar costs 200 - 100 and equity falls by 100
        let config = EngineConfig {
            funding_rate: 1.0,
            funding_interval: 1,
            report: true,
            ..perp_cfg()
        };
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0);
        let mut e = Engine::new(Candles::new(vec![flat; 2]), config);
        e.reset();
        e.market_buy(2.0);
        e.step();
        let ledger = e.reporter().expect("reporting is on").ledger();
        let costs = ledger[1].costs;
        assert_eq!((costs.funding, costs.liquidation), (200.0, -100.0));
        assert_eq!(ledger[1].equity - ledger[0].equity, -100.0);
    }

    #[test]
    fn a_fill_that_grows_the_position_keeps_its_own_price_under_the_margin_bound() {
        // long 10 from 100 with a fee of 1% leaves 90 of margin; at an open of 91.5
        // the account still holds 5, and a buy of 1 there adds at 91.5, so the
        // entry averages to 1,091.5 / 11 and equity is 89.085 - 85
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.01,
            ..cfg()
        };
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(91.5, 91.5, 91.5, 91.5, 1e6),
        ]);
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.market_buy(1.0);
        let state = e.step();
        assert!((state.avg_entry - 1_091.5 / 11.0).abs() < 1e-9);
        assert!(
            (state.equity - 4.085).abs() < 1e-9,
            "equity {}",
            state.equity
        );
    }

    #[test]
    fn two_perps_are_closed_out_together_where_their_margin_runs_out() {
        // 100 of margin behind a long of 1 from 100 and of 10 from 10: with both at
        // their lows of 40 and 4 the wallet holds -20, so both close there, the -20
        // is the liquidation charge, and the perp order goes while the spot one stays
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(10.0), flat(50.0)],
            vec![flat(100.0), flat(10.0), flat(50.0)],
            vec![
                ohlc(100.0, 100.0, 40.0, 50.0, 1e6),
                ohlc(10.0, 10.0, 4.0, 5.0, 1e6),
                flat(50.0),
            ],
        ];
        let config = EngineConfig {
            idle_cash: 1_000.0,
            report: true,
            ..perp_cfg()
        };
        let markets = [Market::Perp, Market::Perp, Market::Spot];
        let mut e = Engine::new(universe(&markets, rows), config);
        let place = |e: &mut Engine, i, side, size, kind, price| {
            e.order_on(
                i,
                side,
                size,
                kind,
                price,
                None,
                false,
                false,
                TimeInForce::Gtc,
            )
        };
        e.reset();
        place(&mut e, 0, Side::Buy, 1.0, OrderType::Market, None);
        place(&mut e, 1, Side::Buy, 10.0, OrderType::Market, None);
        e.step();
        let perp_order = place(&mut e, 0, Side::Sell, 1.0, OrderType::Limit, Some(150.0));
        place(&mut e, 2, Side::Buy, 1.0, OrderType::Limit, Some(40.0));
        let state = e.step();
        assert_eq!((state.margin_cash, state.spot_cash), (0.0, 1_000.0));
        assert_eq!(state.open_orders.len(), 1);
        assert_eq!(state.open_orders[0].instrument, 2);
        let reporter = e.reporter().expect("reporting is on");
        let closes: Vec<(usize, f64)> = reporter
            .fills()
            .iter()
            .filter(|fill| fill.reason == FillReason::Liquidation)
            .map(|fill| (fill.instrument, fill.price))
            .collect();
        assert_eq!(closes, vec![(0, 40.0), (1, 4.0)]);
        assert_eq!(reporter.ledger()[2].costs.liquidation, -20.0);
        let cancelled = reporter
            .events()
            .iter()
            .find(|event| event.kind == EventKind::Cancelled)
            .expect("a cancel");
        assert_eq!(
            (cancelled.order, cancelled.reason),
            (perp_order, EventReason::Liquidation)
        );
    }

    #[test]
    fn a_reducing_fill_beside_another_perp_is_bounded_where_that_perp_sits() {
        // 10 of margin behind 0.1 of A from 50 and 1.3 of B from 100. B's stop at 85
        // sits at the end of its adverse move, where A sits at its low of 80, and the
        // wallet with A there holds 13, so the stop is lifted to 90; A is then closed
        // out at 80, where that wallet runs out, a price its bar printed
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(50.0), flat(100.0)],
            vec![flat(50.0), flat(100.0)],
            vec![flat(100.0), flat(100.0)],
            vec![
                ohlc(100.0, 100.0, 80.0, 90.0, 1e6),
                ohlc(100.0, 100.0, 85.0, 86.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: 10.0,
            max_leverage: 10.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        let place = |e: &mut Engine, i, side, size, kind, trigger, reduce_only| {
            let gtc = TimeInForce::Gtc;
            e.order_on(i, side, size, kind, None, trigger, reduce_only, false, gtc)
        };
        e.reset();
        place(&mut e, 0, Side::Buy, 0.1, OrderType::Market, None, false);
        e.step();
        e.step();
        place(&mut e, 1, Side::Buy, 1.3, OrderType::Market, None, false);
        place(
            &mut e,
            1,
            Side::Sell,
            1.3,
            OrderType::Stop,
            Some(85.0),
            true,
        );
        let last = e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let priced = |reason| -> Vec<(usize, f64)> {
            fills
                .iter()
                .filter(|fill| fill.reason == reason)
                .map(|fill| (fill.instrument, fill.price))
                .collect()
        };
        let (stops, closes) = (priced(FillReason::Stop), priced(FillReason::Liquidation));
        let ([(1, stop)], [(0, close)]) = (stops.as_slice(), closes.as_slice()) else {
            panic!("one stop on B and one close of A");
        };
        assert!((stop - 90.0).abs() < 1e-9, "the stop filled at {stop}");
        assert!((close - 80.0).abs() < 1e-9, "A closed at {close}");
        assert_eq!(last.margin_cash, 0.0);
    }

    /// A long of 10 on A from `entry` with a reduce-only sell stop at 98, and a buy
    /// of `b` B at B's open of 50, on a bar where A trades from 100 down to 96 and B
    /// falls to 36, on `margin` with no fees and no cap. Returns the stop's price,
    /// that bar's liquidation charge and the price B is closed out at.
    fn stop_beside(entry: f64, margin: f64, b: f64) -> (f64, f64, f64) {
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(entry), flat(50.0)],
            vec![flat(entry), flat(50.0)],
            vec![
                ohlc(100.0, 105.0, 96.0, 100.0, 1e6),
                ohlc(50.0, 50.0, 36.0, 40.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: margin,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.stop(Side::Sell, 10.0, 98.0, true);
        let ioc = TimeInForce::Ioc;
        e.order_on(
            1,
            Side::Buy,
            b,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let price = |reason| {
            let fill = reporter.fills().iter().find(|fill| fill.reason == reason);
            fill.expect("the fill").price
        };
        let charge = reporter.ledger()[2].costs.liquidation;
        (
            price(FillReason::Stop),
            charge,
            price(FillReason::Liquidation),
        )
    }

    #[test]
    fn the_bound_lifts_a_fill_no_further_than_its_entry_or_its_bar() {
        // B sits at 43 where A's stop at 98 sits on its path, so the stop's close
        // point is 104, and the lift stops at A's entry of 100. From an entry of 110
        // beside 30 B the close point is 111 and the entry 110, but the bar printed
        // no more than 105: the stop fills there, and the 50 the bar could not hold is
        // paid into the wallet, a negative liquidation charge with the lift's 70
        let (stop, _, close) = stop_beside(100.0, 100.0, 20.0);
        assert!((stop - 100.0).abs() < 1e-9, "the stop filled at {stop}");
        assert!((close - 45.0).abs() < 1e-9, "B closed at {close}");
        let (stop, charge, close) = stop_beside(110.0, 200.0, 30.0);
        assert!((stop - 105.0).abs() < 1e-9, "the stop filled at {stop}");
        assert!((charge + 120.0).abs() < 1e-9, "the charge was {charge}");
        assert!((close - 1_300.0 / 30.0).abs() < 1e-9, "B closed at {close}");
    }

    #[test]
    fn a_reducing_fill_on_the_gap_reads_the_other_perps_at_their_opens() {
        // 270 of margin and fees of 0.1% behind a long of 10 A from 100, marked 90,
        // and 10 B from 50. A's resting sell at 92 fills on the gap up to an open of
        // 100, so it traded before the bar opened, with B at 50: nothing is lifted,
        // the wallet keeps 189.08, and B is closed where that reaches its close
        // level, 310.92 / 9.99. B at its low of 30 would have lifted the sale to
        // 93.12 and left B at its close level there
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(50.0)],
            vec![flat(100.0), flat(50.0)],
            vec![flat(90.0), flat(50.0)],
            vec![
                ohlc(100.0, 101.0, 95.0, 100.0, 1e6),
                ohlc(50.0, 50.0, 30.0, 40.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: 271.5,
            fee_taker: 0.001,
            fee_maker: 0.001,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        let (ioc, gtc) = (TimeInForce::Ioc, TimeInForce::Gtc);
        e.reset();
        e.market_buy(10.0);
        e.order_on(
            1,
            Side::Buy,
            10.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        e.step();
        e.order_on(
            0,
            Side::Sell,
            10.0,
            OrderType::Limit,
            Some(92.0),
            None,
            true,
            false,
            gtc,
        );
        e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let priced = |reason| -> Vec<(usize, f64)> {
            fills
                .iter()
                .filter(|fill| fill.reason == reason)
                .map(|fill| (fill.instrument, fill.price))
                .collect()
        };
        let (sales, closes) = (priced(FillReason::Limit), priced(FillReason::Liquidation));
        let ([(0, sale)], [(1, close)]) = (sales.as_slice(), closes.as_slice()) else {
            panic!("one sale of A and one close of B");
        };
        assert_eq!(*sale, 92.0);
        assert!((close - 310.92 / 9.99).abs() < 1e-9, "B closed at {close}");
    }
}
