//! What a fill may do to a position, given what stands behind it. On a perp the
//! positions may not grow past what margin equity carries, each at the leverage of
//! its tier or of the venue (ADR 0015), on spot a sell may not open a short and a
//! buy may not spend more cash than there is, and a reduce-only order may only
//! shrink the position. Every clamp works on a copy, so an all-or-nothing order is
//! judged against what would really be booked.

use probatum_core::{Fill, Market, Price, Qty, Side, TierBasis};

use super::{Engine, CLOSE_EPS};
use crate::candles::printed;

impl Engine {
    /// Clamp a fill on perp `i` so it does not grow the positions past the margin
    /// cap: summed over the perps, each notional over its leverage may not exceed
    /// the margin wallet's equity, the fill's own instrument at the fill's price and
    /// every other at the same fraction of its own bar's adverse move as the fill
    /// sits at on the path of the position it reduces, flips or adds to, or from
    /// flat of the side it opens. Reductions are never capped, and a position
    /// already over the cap (from an equity drop) is not force-reduced, only blocked
    /// from growing. With no cap set only an insolvent account is held, to
    /// reductions. No-op on spot.
    fn cap_leverage(&self, i: usize, fill: &mut Fill) {
        if self.account.markets[i] != Market::Perp {
            return;
        }
        let mark = fill.price.get();
        if !mark.is_finite() || mark <= 0.0 {
            return;
        }
        let pos = self.account.positions[i].qty.get();
        let own = usize::from(!self.account.positions[i].is_flat());
        let beside = self.account.perps_held() > own;
        // a flip sits where the old side leaves, as the path places it
        let long = if pos == 0.0 {
            fill.side == Side::Buy
        } else {
            pos > 0.0
        };
        let s = if beside {
            self.fraction(i, long, mark)
        } else {
            0.0
        };
        let equity = self.margin_equity_with(i, mark, s);
        // No equity backs new notional, so an insolvent account may only shrink. It
        // used to be waved through here on the grounds that liquidation handles it,
        // but liquidation cannot force-close an account that is flat, so a position
        // opened from negative equity was closed at the bar's adverse extreme and
        // deepened the bad debt instead of being refused. That holds with the cap
        // off as well: the cap bounds how far equity is levered, and an account
        // with none has nothing to lever (ADR 0015).
        let max_size = if equity <= 0.0 {
            0.0
        } else {
            let others: f64 = if beside {
                (0..self.account.positions.len())
                    .filter(|&j| j != i && !self.account.positions[j].is_flat())
                    .map(|j| self.requirement(j, self.at_fraction(j, s)))
                    .sum()
            } else {
                0.0
            };
            self.max_size(i, mark, (equity - others).max(0.0))
        };
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
            let mut size = (capped_new - pos).abs();
            // the position books pos plus the size, and that sum can round an ulp
            // past a tier's bound, into a tier that asks more than the cap allowed
            while (pos + fill.side.sign() * size).abs() > max_abs {
                size = size.next_down();
            }
            fill.size = Qty(size);
        }
    }

    /// Where `price` lies on instrument `i`'s adverse segment of the current bar,
    /// from its open to its low on the `long` side and to its high on the short:
    /// a price on the favourable side of the open, or a bar with no adverse move, is
    /// at 0, and so is a price past the adverse extreme, which traded on the gap
    /// before the bar opened (Specification, The path).
    pub(super) fn fraction(&self, i: usize, long: bool, price: f64) -> f64 {
        let bar = self.candles.bar(self.tick, i);
        let moved = if long { bar.low } else { bar.high } - bar.open;
        if !printed(&bar) || moved == 0.0 {
            return 0.0;
        }
        let s = (price - bar.open) / moved;
        if s > 1.0 {
            0.0
        } else {
            s.max(0.0)
        }
    }

    /// Perp `j` at fraction `s` of its own bar's adverse move, or at its last mark
    /// where it printed no bar.
    fn at_fraction(&self, j: usize, s: f64) -> f64 {
        let bar = self.candles.bar(self.tick, j);
        if !printed(&bar) {
            return self.marks[j];
        }
        let long = self.account.positions[j].qty.get() > 0.0;
        let adverse = if long { bar.low } else { bar.high };
        bar.open + s * (adverse - bar.open)
    }

    /// The margin wallet's equity with perp `i` at `price` and every other perp
    /// at fraction `s` of its own bar's adverse move.
    fn margin_equity_with(&self, i: usize, price: f64, s: f64) -> f64 {
        let held = self.account.perps_held();
        let own = &self.account.positions[i];
        let alone = self.account.markets[i] == Market::Perp && !own.is_flat();
        if held == 0 || (held == 1 && alone) {
            return if alone {
                self.account.margin_cash + own.unrealized(Price(price))
            } else {
                self.account.margin_cash
            };
        }
        let mut equity = self.account.margin_cash;
        for (j, position) in self.account.positions.iter().enumerate() {
            if self.account.markets[j] != Market::Perp || position.qty.get() == 0.0 {
                continue;
            }
            let at = if j == i {
                price
            } else {
                self.at_fraction(j, s)
            };
            equity += position.unrealized(Price(at));
        }
        equity
    }

    /// The wallet a close of perp `i` at `price` is solved against: the margin
    /// wallet's cash, and every other perp's unrealized PnL less the `fee` its own
    /// close would pay, each at the fraction of its own bar's adverse move that
    /// `price` sits at on the path of `i`'s position, so the account stays at or
    /// above its close level.
    pub(super) fn wallet_beside(&self, i: usize, fee: f64, price: f64) -> f64 {
        let own = usize::from(!self.account.positions[i].is_flat());
        if self.account.perps_held() <= own {
            return self.account.margin_cash;
        }
        let s = self.fraction(i, self.account.positions[i].qty.get() > 0.0, price);
        let mut wallet = self.account.margin_cash;
        for (j, position) in self.account.positions.iter().enumerate() {
            let qty = position.qty.get();
            if j == i || self.account.markets[j] != Market::Perp || qty == 0.0 {
                continue;
            }
            let at = self.at_fraction(j, s);
            wallet += position.unrealized(Price(at)) - qty.abs() * at * fee;
        }
        wallet
    }

    /// The largest position on perp `i` that `room` of margin equity carries at
    /// `price`. Through a tier table the requirement is the notional over its tier's
    /// leverage, which rises with the size and jumps up at each bound, so the answer
    /// lies in the first tier that cannot carry its own bound, and never past the
    /// table's last bound, where a fill is held (ADR 0015).
    fn max_size(&self, i: usize, price: f64, room: f64) -> f64 {
        let tiers = self.candles.tiers(i);
        if tiers.rows().is_empty() {
            return if self.config.max_leverage > 0.0 {
                self.config.max_leverage * room / price
            } else {
                f64::INFINITY
            };
        }
        let mut low = 0.0;
        for tier in tiers.rows() {
            let high = match tiers.basis() {
                TierBasis::Notional => {
                    // the quotient can carry a notional an ulp past the bound, where
                    // the tier after it would be read
                    let mut high = tier.up_to / price;
                    while high * price > tier.up_to {
                        high = high.next_down();
                    }
                    high
                }
                TierBasis::Quantity => tier.up_to,
            };
            if tier.leverage > 0.0 {
                let carried = tier.leverage * room / price;
                if carried < high {
                    return carried.max(low);
                }
            }
            low = high;
        }
        low
    }

    /// The maintenance instrument `i`'s position keeps at `mark`: its notional
    /// times its tier's rate, less the tier's amount on a notional table, and
    /// nothing without a table or on spot. The trigger stays at zero equity until
    /// the liquidation path reads this.
    pub(super) fn maintenance(&self, i: usize, mark: f64) -> f64 {
        if self.account.markets[i] != Market::Perp {
            return 0.0;
        }
        let size = self.account.positions[i].qty.get().abs();
        let notional = size * mark;
        let tiers = self.candles.tiers(i);
        match tiers.tier(size, notional) {
            Some(tier) if tier.rate > 0.0 && notional > 0.0 => match tiers.basis() {
                TierBasis::Notional => notional * tier.rate - tier.amount,
                TierBasis::Quantity => notional * tier.rate,
            },
            _ => 0.0,
        }
    }

    /// The margin the cap asks of instrument `i`'s position at `mark`: its notional
    /// over the leverage of its tier, or of the venue without a table, and nothing
    /// on spot, flat or with the cap off. A transfer out of the margin wallet has to
    /// leave it covered (ADR 0008).
    pub(super) fn requirement(&self, i: usize, mark: f64) -> f64 {
        let size = self.account.positions[i].qty.get().abs();
        if self.account.markets[i] != Market::Perp || size == 0.0 {
            return 0.0;
        }
        let notional = size * mark;
        let leverage = match self.candles.tiers(i).tier(size, notional) {
            Some(tier) => tier.leverage,
            None => self.config.max_leverage,
        };
        if leverage > 0.0 {
            notional / leverage
        } else {
            0.0
        }
    }

    /// On spot, keep the position from going below zero: a sell may not exceed the
    /// current long, so it can flatten but never open or extend a short (ADR 0008).
    /// A sell from flat is clamped to nothing. No-op on perp, where a short is a
    /// normal signed position, and on a buy.
    fn cap_spot_short(&self, i: usize, fill: &mut Fill) {
        if self.account.markets[i] != Market::Spot || fill.side != Side::Sell {
            return;
        }
        let pos = self.account.positions[i].qty.get();
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
    fn cap_spot_buy(&self, i: usize, fill: &mut Fill) {
        if self.account.markets[i] != Market::Spot || fill.side != Side::Buy {
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

    /// Every risk clamp applied to a copy of `fill` on instrument `i` without
    /// booking anything: the perp margin cap, the spot short and cash clamps, then
    /// reduce_only. Returns the fill that would actually be booked, so an
    /// all-or-nothing order can be judged against the real fillable size rather than
    /// the raw liquidity the fill model offered (ADR 0010). A refused fill comes
    /// back sized zero.
    pub(super) fn clamp_fill(&self, i: usize, reduce_only: bool, fill: &Fill) -> Fill {
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
        self.cap_leverage(i, &mut fill);
        self.cap_spot_short(i, &mut fill);
        self.cap_spot_buy(i, &mut fill);
        if reduce_only {
            let pos = self.account.positions[i].qty.get();
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
    use crate::candles::{Candles, Instrument};
    use crate::engine::fixtures::{
        cfg, crash_series, flat_series, ohlc, perp_cfg, report_cfg, series, universe,
    };
    use crate::engine::{Engine, EngineConfig, CLOSE_EPS};
    use crate::reporter::{EventKind, FillReason};
    use probatum_core::{
        Candle, Fill, Market, OrderType, Price, Qty, Side, Tier, TierBasis, Tiers, TimeInForce,
    };

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
        e.place_market(0, Side::Sell, 5.0, true, TimeInForce::Ioc); // reduce_only sell of 5
        let s = e.step(); // clamped to 2, so it closes rather than flipping to short 3
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn reduce_only_same_side_is_a_noop() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2
        e.place_market(0, Side::Buy, 1.0, true, TimeInForce::Ioc); // reduce_only buy cannot increase
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
        e.place_market(0, Side::Buy, 5.0, true, TimeInForce::Ioc);
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

    #[test]
    fn a_fill_with_no_real_size_or_price_is_refused_before_any_clamp() {
        // an infinite size or a price that is not a number is no trade, so it moves
        // nothing rather than reaching the clamps that divide by it
        let e = Engine::new(series(), cfg());
        for (size, price) in [(f64::INFINITY, 100.0), (1.0, f64::NAN)] {
            let fill = Fill {
                side: Side::Buy,
                size: Qty(size),
                price: Price(price),
                is_taker: true,
            };
            assert_eq!(
                e.clamp_fill(0, false, &fill).size.get(),
                0.0,
                "{size} at {price}"
            );
        }
    }

    fn tier(up_to: f64, leverage: f64) -> Tier {
        Tier {
            up_to,
            leverage,
            rate: 0.0,
            amount: 0.0,
        }
    }

    /// The position a market buy of 5 reaches at 60,000 on `margin` under `tiers`.
    fn bought_under(tiers: Tiers, margin: f64) -> f64 {
        let flat = ohlc(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e6);
        let bars = Candles::new(vec![flat; 2]).with_tiers(tiers);
        let config = EngineConfig {
            quote: margin,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(5.0);
        e.step().position
    }

    #[test]
    fn a_tier_table_holds_the_position_where_its_margin_stops_carrying_it() {
        // 20x up to 100,000 of notional and 10x to a million: 10,000 of margin
        // carries the 20x tier to its bound and only 100,000 in the 10x tier, so a
        // buy of 5 is held at the bound; 15,000 carries 150,000 in the 10x tier
        let tiers = Tiers::new(
            TierBasis::Notional,
            vec![tier(100_000.0, 20.0), tier(1_000_000.0, 10.0)],
        )
        .expect("a table");
        assert_eq!(bought_under(tiers.clone(), 10_000.0), 100_000.0 / 60_000.0);
        assert_eq!(bought_under(tiers, 15_000.0), 2.5);
    }

    #[test]
    fn a_fill_is_held_at_the_last_bound_of_a_notional_or_a_quantity_table() {
        let notional =
            Tiers::new(TierBasis::Notional, vec![tier(100_000.0, 20.0)]).expect("a table");
        assert_eq!(bought_under(notional, 10_000.0), 100_000.0 / 60_000.0);
        // 20x up to 1 BTC and 5x to 10: the 5x tier carries 0.83 BTC, less than
        // the 1 the 20x tier already holds, so the position stops at 1
        let quantity = Tiers::new(TierBasis::Quantity, vec![tier(1.0, 20.0), tier(10.0, 5.0)])
            .expect("a table");
        assert_eq!(bought_under(quantity, 10_000.0), 1.0);
    }

    #[test]
    fn a_position_held_at_a_bound_stays_in_the_tier_that_carried_it() {
        // a bound reached through a quotient, 100,000 / 11,000, or through a sum, 0.3
        // held and 0.6 more to a bound of 0.9, can land an ulp past it, where the
        // tier after it asks more than the 6,000 of margin holds
        let cases = [
            (TierBasis::Notional, 100_000.0, 10.0, 11_000.0, vec![20.0]),
            (TierBasis::Quantity, 0.9, 5.0, 60_000.0, vec![0.3, 5.0]),
        ];
        for (basis, bound, after, price, buys) in cases {
            let rows = vec![tier(bound, 20.0), tier(10_000_000.0, after)];
            let tiers = Tiers::new(basis, rows).expect("a table");
            let flat = ohlc(price, price, price, price, 1e6);
            let bars = Candles::new(vec![flat; buys.len() + 1]).with_tiers(tiers);
            let config = EngineConfig {
                quote: 6_000.0,
                report: true,
                ..perp_cfg()
            };
            let mut e = Engine::new(bars, config);
            e.reset();
            let mut held = 0.0;
            for size in buys {
                e.market_buy(size);
                held = e.step().position;
            }
            let ledger = e.reporter().expect("reporting is on").ledger();
            let used = ledger.last().expect("a row").margin_used;
            assert!(
                (used - held * price / 20.0).abs() < 1e-9,
                "{basis:?}: {used} on {held}"
            );
        }
    }

    #[test]
    fn the_maintenance_is_continuous_across_binances_first_btc_bound() {
        // the Specification's example: 0.4% up to 300,000 with no amount, then 0.5%
        // less 300, so 5 BTC at 60,000 keep 1,200 in either tier and 6 keep 1,500
        let tiers = Tiers::new(
            TierBasis::Notional,
            vec![
                Tier {
                    up_to: 300_000.0,
                    leverage: 125.0,
                    rate: 0.004,
                    amount: 0.0,
                },
                Tier {
                    up_to: 800_000.0,
                    leverage: 100.0,
                    rate: 0.005,
                    amount: 300.0,
                },
            ],
        )
        .expect("continuous");
        for (size, kept) in [(5.0, 1_200.0), (6.0, 1_500.0)] {
            let flat = ohlc(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e6);
            let bars = Candles::new(vec![flat; 2]).with_tiers(tiers.clone());
            let config = EngineConfig {
                quote: 100_000.0,
                report: true,
                ..perp_cfg()
            };
            let mut e = Engine::new(bars, config);
            e.reset();
            e.market_buy(size);
            e.step();
            let row = e.reporter().expect("reporting is on").ledger()[1];
            assert!(
                (row.maintenance - kept).abs() < 1e-9,
                "{size}: {}",
                row.maintenance
            );
        }
    }

    /// BTC at 20x and ETH at 10x, by tables of one tier, on 10,000 of margin: a long
    /// of 1 BTC bought at the second row's open, and an ETH order decided at its
    /// close, placed by `sell`, filling on the third row. Returns the ETH position.
    fn eth_beside_btc(third: [Candle; 2], sell: impl Fn(&mut Engine)) -> f64 {
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = [[flat(60_000.0), flat(3_000.0)]; 2].concat();
        let at = |leverage| Instrument {
            market: Some(Market::Perp),
            tiers: Tiers::new(TierBasis::Notional, vec![tier(1e12, leverage)]).expect("a table"),
        };
        let bars = Candles::universe([rows, third.to_vec()].concat(), 2, None)
            .with_instruments(vec![at(20.0), at(10.0)]);
        let config = EngineConfig {
            quote: 10_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.0);
        e.step();
        sell(&mut e);
        e.step();
        e.reporter().expect("reporting is on").legs()[5].position
    }

    fn sell_eth(e: &mut Engine, kind: OrderType, price: Option<f64>, tif: TimeInForce) {
        e.order_on(1, Side::Sell, 30.0, kind, price, None, false, false, tif);
    }

    #[test]
    fn the_cap_sums_the_requirement_of_every_perp() {
        // the Specification's example: 10,000 of margin, BTC at 20x and ETH at 10x. A
        // long of 1 BTC at 60,000 uses 3,000, so a market sell of 30 ETH at 3,000
        // has room for (10,000 - 3,000) * 10 = 70,000 of notional, 23.333 ETH
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let held = eth_beside_btc([flat(60_000.0), flat(3_000.0)], |e| {
            sell_eth(e, OrderType::Market, None, TimeInForce::Ioc)
        });
        assert!((held + 70_000.0 / 3_000.0).abs() < 1e-9, "sold {held}");
    }

    #[test]
    fn the_cap_reads_the_other_perps_where_the_fill_sits_on_its_path() {
        // the Specification's example: a sell limit on ETH that fills at its high of
        // 3,300 sits at the end of its own adverse move, on a bar where BTC falls to
        // 54,000, so BTC is marked there too. Equity is 4,000, BTC uses 2,700, and
        // the room is 13,000 of notional, 3.94 ETH, where BTC at its last mark would
        // have left room for 21.2
        let bars = [
            ohlc(60_000.0, 60_000.0, 54_000.0, 55_000.0, 1e6),
            ohlc(3_000.0, 3_300.0, 3_000.0, 3_100.0, 1e6),
        ];
        let held = eth_beside_btc(bars, |e| {
            sell_eth(e, OrderType::Limit, Some(3_300.0), TimeInForce::Gtc)
        });
        assert!((held + 13_000.0 / 3_300.0).abs() < 1e-9, "sold {held}");
    }

    #[test]
    fn the_cap_reads_a_flip_where_its_old_side_leaves() {
        // 1,000 of margin at 10x behind a long of 10 A from 100 and 100 B from 50. A
        // sell stop for 30 at 94 flips A at 0.6 of the long's fall to 90, where B
        // stands at 47: equity is 640 and B uses 470, so the short has room for 1,700
        // of notional, 18.09 A. On the short's own path the stop sits at 0, B at 50,
        // and all 20 would have fit
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(50.0)],
            vec![flat(100.0), flat(50.0)],
            vec![
                ohlc(100.0, 105.0, 90.0, 100.0, 1e6),
                ohlc(50.0, 50.0, 45.0, 50.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: 1_000.0,
            max_leverage: 10.0,
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
            100.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        e.order_on(
            0,
            Side::Sell,
            30.0,
            OrderType::Stop,
            None,
            Some(94.0),
            false,
            false,
            gtc,
        );
        e.step();
        let held = e.reporter().expect("reporting is on").legs()[4].position;
        assert!((held + 1_700.0 / 94.0).abs() < 1e-9, "A holds {held}");
    }
}
