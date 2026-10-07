//! What a fill may do to the position, given what stands behind it. On a perp the
//! position may not grow past `max_leverage` times equity (ADR 0015), on spot a
//! sell may not open a short and a buy may not spend more cash than there is, and
//! a reduce-only order may only shrink the position. Every clamp works on a copy,
//! so an all-or-nothing order is judged against what would really be booked.

use probatum_core::{Fill, Market, Price, Qty, Side, Wallet};

use super::{Engine, CLOSE_EPS};

impl Engine {
    /// Clamp a fill so it does not grow the position past the perp margin cap
    /// (`max_leverage` times the margin wallet's equity, in notional). Reductions
    /// are never capped, and a position already over the cap (from an equity drop)
    /// is not force-reduced, only blocked from growing. With no cap set only an
    /// insolvent account is held, to reductions. No-op on spot.
    fn cap_leverage(&self, fill: &mut Fill) {
        if self.config.market != Market::Perp {
            return;
        }
        let mark = fill.price.get();
        if !mark.is_finite() || mark <= 0.0 {
            return;
        }
        let equity = self.account.wallet_equity(Wallet::Margin, Price(mark));
        // No equity backs new notional, so an insolvent account may only shrink. It
        // used to be waved through here on the grounds that liquidation handles it,
        // but liquidation cannot force-close an account that is flat, so a position
        // opened from negative equity was closed at the bar's adverse extreme and
        // deepened the bad debt instead of being refused. That holds with the cap
        // off as well: the cap bounds how far equity is levered, and an account
        // with none has nothing to lever (ADR 0015).
        let max_size = if equity <= 0.0 {
            0.0
        } else if self.config.max_leverage > 0.0 {
            self.config.max_leverage * equity / mark
        } else {
            f64::INFINITY
        };
        let pos = self.account.position.qty.get();
        let new_pos = pos + fill.side.sign() * fill.size.get();
        // A same-side grow may keep an existing over-cap position (from an equity
        // drop) but not extend it, so the allowance is the larger of the cap and
        // the current size. A flip closes the old side and opens a fresh one, which
        // must fit the cap outright, so it gets no such allowance (ADR 0015).
        let flipping = pos != 0.0 && new_pos != 0.0 && (new_pos > 0.0) != (pos > 0.0);
        let max_abs = if flipping {
            max_size
        } else {
            max_size.max(pos.abs())
        };
        if new_pos.abs() > max_abs {
            let capped_new = max_abs * new_pos.signum();
            fill.size = Qty((capped_new - pos).abs());
        }
    }

    /// The margin the cap asks of the position at `mark`: its notional over
    /// `max_leverage`, and nothing on spot or with the cap off. A transfer out of
    /// the margin wallet has to leave it covered (ADR 0008).
    pub(super) fn requirement(&self, mark: f64) -> f64 {
        if self.config.market != Market::Perp || self.config.max_leverage <= 0.0 {
            return 0.0;
        }
        self.account.position.qty.get().abs() * mark / self.config.max_leverage
    }

    /// On spot, keep the position from going below zero: a sell may not exceed the
    /// current long, so it can flatten but never open or extend a short (ADR 0008).
    /// A sell from flat is clamped to nothing. No-op on perp, where a short is a
    /// normal signed position, and on a buy.
    fn cap_spot_short(&self, fill: &mut Fill) {
        if self.config.market != Market::Spot || fill.side != Side::Sell {
            return;
        }
        let pos = self.account.position.qty.get();
        let sellable = pos.max(0.0);
        if fill.size.get() > sellable {
            fill.size = Qty(sellable);
        }
    }

    /// On spot, keep a buy within the cash on hand: its cost, notional plus fee,
    /// may not exceed the spot wallet's cash, so the wallet never goes negative. This
    /// is the buy-side mirror of `cap_spot_short`; spot has no borrow and no margin,
    /// so credit is refused rather than pretended (ADR 0008). No-op on perp, where
    /// the margin cap governs, and on a sell.
    fn cap_spot_buy(&self, fill: &mut Fill) {
        if self.config.market != Market::Spot || fill.side != Side::Buy {
            return;
        }
        let price = fill.price.get();
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        if self.account.spot_cash <= 0.0 {
            fill.size = Qty(0.0);
            return;
        }
        let fee_rate = if fill.is_taker {
            self.cost.fee_taker
        } else {
            self.cost.fee_maker
        };
        let affordable = self.account.spot_cash / (price * (1.0 + fee_rate));
        if fill.size.get() > affordable {
            fill.size = Qty(affordable);
        }
    }

    /// Every risk clamp applied to a copy of `fill` without booking anything: the
    /// perp margin cap, the spot short and cash clamps, then reduce_only. Returns
    /// the fill that would actually be booked, so an all-or-nothing order can be
    /// judged against the real fillable size rather than the raw liquidity the
    /// fill model offered (ADR 0010). A refused fill comes back sized zero.
    pub(super) fn clamp_fill(&self, reduce_only: bool, fill: &Fill) -> Fill {
        let mut fill = *fill;
        // size is a positive magnitude; a non-positive or non-finite fill is not a
        // real trade, so it moves nothing and pays no fee. Neither is a price at or
        // below zero, and that one is the input both risk clamps have to give up
        // on: each divides by the mark, so each returns without clamping anything,
        // and the fill that follows buys an unbounded position for nothing or for a
        // credit. MAX_SLIP exists so a slipped fill cannot reach zero (ADR 0013);
        // this is the same rule applied to the bar's own price, which the candle
        // validator lets through because it checks only for finiteness (ADR 0007)
        let size = fill.size.get();
        let price = fill.price.get();
        if !size.is_finite() || size <= 0.0 || !price.is_finite() || price <= 0.0 {
            fill.size = Qty(0.0);
            return fill;
        }
        self.cap_leverage(&mut fill);
        self.cap_spot_short(&mut fill);
        self.cap_spot_buy(&mut fill);
        if reduce_only {
            let pos = self.account.position.qty.get();
            let reduces =
                (pos > 0.0 && fill.side == Side::Sell) || (pos < 0.0 && fill.side == Side::Buy);
            if !reduces {
                fill.size = Qty(0.0);
                return fill;
            }
            fill.size = Qty(fill.size.get().min(pos.abs()));
        }
        // a clamp can leave NaN or a negative remainder; normalize so the caller only
        // has to test for a positive size. The gate is the dust epsilon, not zero,
        // because the position refuses anything at or below it: a size in between
        // moved no cash and no position, but still counted a fill and banked a fee,
        // which shows up as a phantom fill in the dead-feed canary and as drift in
        // the round-trip fee identity (ADR 0023)
        let clamped = fill.size.get();
        if clamped.is_nan() || clamped <= CLOSE_EPS {
            fill.size = Qty(0.0);
        }
        fill
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{
        cfg, crash_series, flat_series, ohlc, perp_cfg, report_cfg, series,
    };
    use crate::engine::{Engine, EngineConfig, CLOSE_EPS};
    use crate::reporter::{EventKind, FillReason};
    use probatum_core::{Candle, OrderType, Side, TimeInForce};

    #[test]
    fn default_config_caps_perp_leverage_at_ten() {
        // the shipped default is a finite 10x cap, not uncapped notional (ADR 0015)
        assert_eq!(EngineConfig::default().max_leverage, 10.0);
    }

    #[test]
    fn spot_sell_cannot_open_a_short() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_sell(1.0); // sell from flat on spot
        let s = e.step();
        assert_eq!(s.position, 0.0); // clamped to flat, no short opened
        assert_eq!(s.quote, 10_000.0); // nothing sold, quote untouched
    }

    #[test]
    fn spot_sell_clamps_to_the_current_long() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
        e.market_sell(3.0); // wants 3, holds only 1
        let s = e.step(); // fills at bar 2 open 300
        assert_eq!(s.position, 0.0); // sold exactly the 1 held, not -2
        assert_eq!(s.quote, 10_100.0); // 10000 - 200 (buy) + 300 (sell 1)
    }

    #[test]
    fn spot_buy_clamps_to_available_cash() {
        let mut e = Engine::new(series(), cfg()); // spot, 10k, no fee
        e.reset();
        e.market_buy(1000.0); // wants 1000 base at bar 1 open 200, can afford 50
        let s = e.step();
        assert_eq!(s.position, 50.0); // 10000 / 200
        assert_eq!(s.quote, 0.0); // spent exactly the cash, never negative
    }

    #[test]
    fn spot_buy_clamp_accounts_for_the_fee() {
        let mut c = cfg();
        c.fee_taker = 0.01; // 1% taker
        let mut e = Engine::new(series(), c);
        e.reset();
        e.market_buy(1000.0); // affordable = 10000 / (200 * 1.01)
        let s = e.step();
        assert!((s.position - 10_000.0 / 202.0).abs() < 1e-9);
        assert!(s.quote >= -1e-9 && s.quote < 1e-6); // lands at ~0, never negative
    }

    #[test]
    fn the_spot_wallet_pays_for_what_its_cash_covers_fee_included() {
        // the Specification's example: 1,000 buys 20 at an open of 100 with a taker
        // fee of 0.1%, and its cash pays for 1,000 / (100 * 1.001) = 9.99001
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 1_000.0,
            fee_taker: 0.001,
            ..report_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(20.0);
        let s = e.step();
        assert!((s.position - 9.99001).abs() < 5e-6, "filled {}", s.position);
        assert!(s.spot_cash.abs() < 1e-9, "left {}", s.spot_cash);
        // and the order log records the clamp
        let kinds: Vec<EventKind> = e
            .reporter()
            .expect("reporting is on")
            .events()
            .iter()
            .map(|event| event.kind)
            .collect();
        assert_eq!(
            kinds,
            vec![
                EventKind::Placed,
                EventKind::Clamped,
                EventKind::Filled,
                EventKind::Expired,
            ]
        );
    }

    #[test]
    fn perp_sell_still_opens_a_short() {
        let mut e = Engine::new(series(), perp_cfg());
        e.reset();
        e.market_sell(0.1); // perp: a short is a normal signed position
        let s = e.step();
        assert_eq!(s.position, -0.1);
    }

    #[test]
    fn a_reduce_only_order_only_shrinks_the_position() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200

        // a reduce_only buy cannot grow the long, so it applies nothing
        e.order(
            Side::Buy,
            1.0,
            OrderType::Market,
            None,
            None,
            true,
            false,
            TimeInForce::Ioc,
        );
        let s = e.step();
        assert_eq!(s.position, 1.0);
    }

    #[test]
    fn leverage_caps_the_perp_position() {
        let mut config = perp_cfg(); // perp, quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        // fill at bar 1 open 200; equity 100, so max size = 2 * 100 / 200 = 1.0
        e.market_buy(10.0);
        let s = e.step();
        assert_eq!(s.position, 1.0); // capped at the margin limit
    }

    #[test]
    fn leverage_is_ignored_on_spot() {
        let mut config = cfg(); // spot
        config.max_leverage = 1.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(10.0);
        let s = e.step();
        assert_eq!(s.position, 10.0); // spot is unleveraged; no cap
    }

    #[test]
    fn leverage_caps_the_opened_side_of_a_flip() {
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 3.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200

        // at bar 2 (fill 300) equity is 100 + 1*(300-200) = 200, so the cap is
        // 3*200/300 = 2
        e.market_sell(10.0);
        let s = e.step(); // flips: closes 1, opens short capped at 2
        assert_eq!(s.position, -2.0);
    }

    #[test]
    fn leverage_caps_the_new_side_of_a_flip_from_over_cap() {
        // bar 1 fills the open long at 100; by bar 2 the price has fallen to 80, so
        // equity drops and the 2-unit long is over the 2x cap. A big sell flips: the
        // fresh short must fit the cap (1.5), not inherit the old side's size (2).
        let candles = Candles::new(vec![
            Candle {
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1_000.0,
            },
            Candle {
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1_000.0,
            },
            Candle {
                open: 80.0,
                high: 80.0,
                low: 80.0,
                close: 80.0,
                volume: 1_000.0,
            },
        ]);
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(2.0);
        let s1 = e.step(); // long 2 @ 100 (2x on 100 equity)
        assert_eq!(s1.position, 2.0);
        // bar 2 fill 80: equity = 100 + 2*(80-100) = 60; max size = 2*60/80 = 1.5
        e.market_sell(20.0);
        let s2 = e.step(); // flips: closes 2, opens short capped at 1.5, not 2
        assert_eq!(s2.position, -1.5);
    }

    #[test]
    fn reduce_only_market_cannot_flip_the_position() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2 @ 200
        e.place_market(Side::Sell, 5.0, true, TimeInForce::Ioc); // reduce_only sell of 5
        let s = e.step(); // clamped to 2, so it closes rather than flipping to short 3
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn reduce_only_same_side_is_a_noop() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2
        e.place_market(Side::Buy, 1.0, true, TimeInForce::Ioc); // reduce_only buy cannot increase
        let s = e.step();
        assert_eq!(s.position, 2.0);
    }

    #[test]
    fn an_insolvent_perp_cannot_open_a_position() {
        // after a gap liquidation the account is flat with negative equity. Nothing
        // backs new notional, and liquidation cannot force-close a flat account, so
        // the fill is refused rather than opened and closed at the low (ADR 0015)
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000_000.0),
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000_000.0),
            ohlc(40.0, 40.0, 40.0, 40.0, 1_000_000.0),
            ohlc(40.0, 41.0, 25.0, 30.0, 1_000_000.0),
        ]);
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 10.0;
        config.report = true; // the only surface that can see the refusal
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        let busted = e.step();
        // dead, and holding exactly nothing rather than owing anything: the bar
        // crashed to 40 but the position was closed where its margin ran out at 90
        assert!(busted.equity.abs() < CLOSE_EPS, "left {}", busted.equity);
        assert_eq!(busted.position, 0.0);
        e.market_buy(5_000.0);
        let after = e.step();
        assert_eq!(after.position, 0.0);
        // the refused fill costs nothing; equity only marks the same dead account
        assert!(
            (after.equity - busted.equity).abs() < 1e-9,
            "equity moved {} -> {}",
            busted.equity,
            after.equity
        );
        // and the two assertions above cannot see the refusal on their own, which
        // is why these are here. Waving an insolvent account through opens 5000 at
        // the open of 40 and has it liquidated again at the bankruptcy price,
        // which for a position bought with nothing behind it IS the entry, so the
        // realized pnl is zero, the account ends flat, and equity lands on exactly
        // the same dead zero. Only the fill count and the trade log differ, and
        // the fixture had reporting off
        assert_eq!(e.num_fills(), 2, "the refused fill was applied anyway");
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1, "a phantom position was opened and closed");
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
    }

    #[test]
    fn an_insolvent_perp_cannot_open_a_position_with_the_cap_off() {
        // the refusal lived inside the cap, so switching the cap off switched it off
        // too, and a dead account opened positions that the same bar liquidated,
        // once for a profit. The cap bounds how far equity is levered, and an
        // account with none has nothing to lever (ADR 0015)
        let config = EngineConfig {
            max_leverage: 0.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        e.market_buy(10.0);
        let busted = e.step(); // 10 on 100 of margin, liquidated on the low of 80
        assert!(e.is_bust());
        assert!(busted.equity.abs() < CLOSE_EPS, "left {}", busted.equity);
        e.market_buy(1.0);
        let after = e.step();
        assert_eq!(after.position, 0.0);
        assert_eq!(e.num_fills(), 2, "the dead account opened a position");
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1, "a phantom position was opened and closed");
    }

    #[test]
    fn a_reduce_only_buy_closes_a_short() {
        let mut e = Engine::new(series(), perp_cfg());
        e.reset();
        e.market_sell(2.0);
        e.step(); // short 2
        e.place_market(Side::Buy, 5.0, true, TimeInForce::Ioc);
        let s = e.step(); // clamped to 2, so it closes rather than flipping long
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_fill_priced_at_or_below_zero_is_refused_rather_than_booked() {
        // both risk clamps divide by the mark, so each returns without clamping when
        // it is zero or negative, and the fill behind them bought an unbounded
        // position for nothing. The candle validator lets those prices through
        // because it checks only finiteness (ADRs 0007, 0013)
        for price in [0.0, -100.0] {
            let bars = Candles::new(vec![
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(price, price, price, price, 1e6),
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ]);
            let mut e = Engine::new(bars, cfg());
            e.reset();
            e.market_buy(1_000.0);
            e.step();
            let state = e.step();
            assert_eq!(state.position, 0.0, "price {price} opened a position");
            assert_eq!(state.equity, 10_000.0, "price {price} minted equity");
        }
    }

    #[test]
    fn a_flip_must_fit_the_cap_outright_where_a_grow_may_keep_what_it_has() {
        // ADR 0015's allowance: a position already over the cap because equity
        // fell is not force-reduced, so a same-side add may keep it, but a flip
        // closes that side and opens a fresh one which gets no such credit. The
        // two branches only differ when the position IS over the cap, and nothing
        // reached that state, so the predicate deciding it was unpinned
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 1_000.0,
            max_leverage: 3.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(30.0);
        let state = e.step(); // long 30 at 100, exactly the cap
        assert_eq!(state.position, 30.0);

        // the mark falls to 90, so equity is 700 and the cap is now 23.33 while
        // the position is still 30: over it, and a flip gets no allowance for that
        e.market_sell(60.0);
        let state = e.step();
        assert!(
            (state.position + 3.0 * 700.0 / 90.0).abs() < 1e-9,
            "flipped to {} rather than the cap",
            state.position
        );
    }

    #[test]
    fn a_reduce_only_order_on_the_wrong_side_fills_nothing_either_way() {
        // the predicate has a branch per side and only one of them was ever taken
        for (opening, wrong) in [(Side::Buy, Side::Buy), (Side::Sell, Side::Sell)] {
            let mut e = Engine::new(flat_series(), perp_cfg());
            e.reset();
            e.order(
                opening,
                1.0,
                OrderType::Market,
                None,
                None,
                false,
                false,
                TimeInForce::Ioc,
            );
            let state = e.step();
            let held = state.position;
            assert!(held.abs() > 0.0);

            e.order(
                wrong,
                5.0,
                OrderType::Market,
                None,
                None,
                true,
                false,
                TimeInForce::Ioc,
            );
            let state = e.step();
            assert_eq!(
                state.position, held,
                "a reduce-only {wrong:?} grew the position"
            );

            // and the right side reduces, clamped to what is actually held
            let closing = if held > 0.0 { Side::Sell } else { Side::Buy };
            e.order(
                closing,
                5.0,
                OrderType::Market,
                None,
                None,
                true,
                false,
                TimeInForce::Ioc,
            );
            let state = e.step();
            assert_eq!(state.position, 0.0);
        }
    }

    #[test]
    fn an_over_cap_position_is_kept_whole_refuses_an_add_and_reduces_only_what_was_asked() {
        // ADR 0015's allowance runs one way only: a position pushed over its cap by
        // a fall in equity is not force-reduced, and it may not be extended either.
        // Dropping the max(pos.abs()) from the cap turns this refusal into a fill,
        // and the fill is on the SAME side, so the order that was meant to be
        // blocked is the one that grows the long from 30 to 36.67.
        //
        // The refusal alone does not pin "kept whole", because an add and a cap
        // that clamps to max_size both answer with a zero fill here. The reduce at
        // the end is what separates them: it asks for 5 of the 30 and must get
        // exactly 5, where a cap applied to every fill sells 6.67 to land the
        // position on the cap itself, which is a liquidation the account did not owe
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 1_000.0,
            max_leverage: 3.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(30.0);
        let state = e.step(); // long 30 at 100: 3000 of notional on 1000, exactly the cap
        assert_eq!(state.position, 30.0);
        assert_eq!(e.num_fills(), 1);

        // the mark falls to 90, so equity is 1000 + 30 * (90 - 100) = 700 and the
        // cap is 3 * 700 / 90 = 23.33 against a position of 30: over it, and kept
        e.market_buy(5.0);
        let state = e.step();
        assert_eq!(
            state.position, 30.0,
            "the add moved a position it may not move"
        );
        assert_eq!(state.equity, 700.0);
        assert_eq!(state.quote, 1_000.0);
        assert_eq!(e.num_fills(), 1, "the refused add booked a fill");

        // and a reduce is the size that was asked for, not the size that would put
        // the position back inside the cap: 30 less 5 is 25, still over the 23.33
        // the cap allows, and it stays there
        e.market_sell(5.0);
        let state = e.step();
        assert_eq!(
            state.position, 25.0,
            "the reduce was widened to land the position on the cap"
        );
        assert_eq!(state.equity, 700.0);
        assert_eq!(e.num_fills(), 2);
        assert!(!e.is_bust());
    }
}
