//! Where the margin behind a perp position runs out. The fence is worked out
//! before a bar's orders are resolved and clips the bar there (ADR 0017); the
//! forced close is priced at the bankruptcy price rather than at the bar's
//! extreme, and no fill that reduces the position books past that price, so an
//! account can end at zero but never owing (ADR 0018).

use probatum_core::{Candle, CostModel, Fill, Market, Price, Qty, Side, Wallet};

use super::Engine;

impl Engine {
    /// True when a perp position is dead at `mark` once `debits` of funding are
    /// counted against the margin wallet: the check at a bar's adverse extreme, with
    /// the debits stamped inside the bar (ADRs 0008, 0017, 0020).
    pub(super) fn dead_at(&self, mark: f64, debits: f64) -> bool {
        self.config.market == Market::Perp
            && !self.account.position.is_flat()
            && mark.is_finite()
            && self.account.wallet_equity(Wallet::Margin, Price(mark)) - debits <= 0.0
    }

    /// The price this bar would take the position's margin to zero at, or `None`
    /// when nothing on the bar can kill the account.
    ///
    /// The trigger is the bar's adverse extreme and the answer is the bankruptcy
    /// price, which are two different numbers on purpose (ADR 0018). Read before
    /// the bar's orders are resolved, because it is the point past which this
    /// account was no longer in the market.
    pub(super) fn liquidation_fence(&self, bar: &Candle) -> Option<f64> {
        let adverse = if self.account.position.qty.get() >= 0.0 {
            bar.low
        } else {
            bar.high
        };
        if !self.account.is_bust_at(Price(adverse)) {
            return None;
        }
        self.account
            .bankruptcy_price(self.cost.fee_taker)
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

    /// Force-close the whole position at the price that leaves nothing.
    ///
    /// Not at the bar's adverse extreme, which is what the trigger was tested
    /// against. Closing there charged the account for however far the bar happened
    /// to run past the point its margin was gone, so a long on 100 of margin
    /// against a bar that wicked to 80 booked a loss of 200 and left the account
    /// owing 100. A position cannot lose more than the margin behind it, so the
    /// exit is priced where the margin runs out, and the liquidation fee is inside
    /// that price rather than charged past it (ADR 0018).
    pub(super) fn liquidate(&mut self) {
        if let Some(price) = self.account.bankruptcy_price(self.cost.fee_taker) {
            self.liquidate_at(price);
        }
    }

    /// Force-close the whole position at `price`, paying the liquidation fee, and
    /// leave the margin wallet at exactly zero: whatever it then holds, positive or
    /// negative, is the liquidation charge. A funding debit that takes the account to
    /// its trigger at a bar's close is closed here at that close (ADR 0020).
    pub(super) fn liquidate_at(&mut self, price: Price) {
        let size = self.account.position.qty.get().abs();
        let fill = Fill {
            side: if self.account.position.qty.get() > 0.0 {
                Side::Sell
            } else {
                Side::Buy
            },
            size: Qty(size),
            price,
            is_taker: true,
        };
        let fee = self.cost.fee(&fill);
        let before = self.before_fill();
        self.account.apply_fill(&fill, fee);
        self.book_fill(size, price.get(), fee, before, true);
        // The bankruptcy price leaves nothing in exact arithmetic and a hair either
        // side of it in floating point, and a hair above zero is equity the cap
        // levers: with the cap off, an account closed out there opened a position
        // again on the next bar. So the wallet ends at exactly zero (ADRs 0015, 0018).
        self.account.margin_cash = 0.0;
        // Every order the closed-out account left goes with it. The empty wallet
        // refused them anyway until a transfer could fund it again, after which
        // they filled for an account that had already left the market (ADR 0018).
        self.book.cancel_all();
        self.pending.clear();
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
    /// ordering of fills within a bar can book past it.
    pub(super) fn bound_by_margin(&self, fill: &mut Fill) {
        if self.config.market != Market::Perp {
            return;
        }
        let position = self.account.position.qty.get();
        let long = position > 0.0;
        let reduces =
            (long && fill.side == Side::Sell) || (position < 0.0 && fill.side == Side::Buy);
        if !reduces {
            return;
        }
        if let Some(bound) = self.account.bankruptcy_price(self.cost.fee_taker) {
            let price = fill.price.get();
            fill.price = Price(if long {
                price.max(bound.get())
            } else {
                price.min(bound.get())
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{cfg, crash_series, levered, ohlc, perp_cfg};
    use crate::engine::{Engine, EngineConfig, CLOSE_EPS};
    use probatum_core::{Market, Side, Wallet};

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
                !(trade.liquidated && trade.net_pnl > 0.0),
                "liquidated at {} for a profit of {}",
                trade.exit_price,
                trade.net_pnl
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
            (trades[0].exit_price - 99.0).abs() < 1e-9,
            "exited at {}",
            trades[0].exit_price
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
        assert!(trades[0].liquidated);
        // closed where the margin ran out, not at the 80 the bar printed. The entry
        // fee left 99 of margin, so (10*100 - 99) / (10 - 10*0.001) = 90.1902
        assert!(
            (trades[0].exit_price - 901.0 / 9.99).abs() < 1e-9,
            "closed at {}",
            trades[0].exit_price
        );
        // and the account is left with exactly nothing, never owing (ADR 0018)
        assert!(s.equity.abs() < CLOSE_EPS, "left {}", s.equity);
        assert!(trades[0].fees > 1.0); // the entry fee, plus a real fee on the exit
        assert_eq!(e.open_fee, 0.0); // nothing left over to charge the next position
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
            assert_eq!(trades[0].exit_price, 90.0);
            assert!(trades[0].liquidated, "exit={exit} booked no liquidation");
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
        for (trigger, want_equity, want_exit) in [(95.0, 50.0, 95.0), (60.0, 0.0, 90.0)] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_buy(10.0);
            e.step();
            e.stop(Side::Sell, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades[0].exit_price, want_exit, "stop at {trigger}");
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
            assert!(trade.exit_price <= 101.0, "exited at {}", trade.exit_price);
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
        assert_eq!(trades[0].exit_price, 90.0);
        assert!(trades[0].liquidated);
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
        assert_eq!(trades[0].exit_price, 110.0);
        assert!(trades[0].liquidated);
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
            assert_eq!(trades[0].exit_price, 110.0, "exit={exit}");
            assert!(trades[0].liquidated, "exit={exit} booked no liquidation");
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
        for (trigger, want_equity, want_exit) in [(105.0, 50.0, 105.0), (140.0, 0.0, 110.0)] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_sell(10.0);
            e.step(); // short 10 at 100
            e.stop(Side::Buy, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades[0].exit_price, want_exit, "stop at {trigger}");
        }
    }
}
