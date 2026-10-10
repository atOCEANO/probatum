//! Transfers between the two wallets. One decided at a close lands at the start of
//! the next bar, before funding and before any fill, so cash moved into the margin
//! wallet backs the orders decided with it. Out of the margin wallet a transfer is
//! held to the cash there, so unrealized profit stays, and to what leaves the cap's
//! requirement covered; out of the spot wallet, to its cash. With the cap off nothing
//! is required, so a transfer can take a margin wallet holding a loss to its trigger,
//! and the position is closed at the last close before the bar opens. A clamped
//! transfer is an order-log event (ADRs 0008, 0022).

use probatum_core::{Market, Price, Wallet};

use super::Engine;
use crate::reporter::{EventKind, EventReason};

impl Engine {
    /// Move `amount` into the `to` wallet from the other one, at the start of the
    /// next bar. Transfers decided on one bar land in the order they were made, and
    /// an amount that is not a positive number moves nothing.
    pub fn transfer(&mut self, amount: f64, to: Wallet) {
        if amount > 0.0 {
            self.transfers.push((amount, to));
        }
    }

    /// Apply the transfers decided at the last closes, each held to what the wallet
    /// it leaves may give.
    pub(super) fn apply_transfers(&mut self) {
        for k in 0..self.transfers.len() {
            let (amount, to) = self.transfers[k];
            // the cash the margin wallet keeps: what still covers the requirement once
            // its unrealized PnL is counted, floored at zero so a profit cannot move
            let keep = match to {
                Wallet::Spot => {
                    let unrealized: f64 = (0..self.marks.len())
                        .filter(|&i| self.account.markets[i] == Market::Perp)
                        .filter(|&i| self.account.positions[i].qty.get() != 0.0)
                        .map(|i| self.account.unrealized(i, Price(self.marks[i])))
                        .sum();
                    let required: f64 = (0..self.marks.len())
                        .map(|i| self.requirement(i, self.marks[i]))
                        .sum();
                    (required - unrealized).max(0.0)
                }
                Wallet::Margin => 0.0,
            };
            let allowed = (self.account.cash(to.other()) - keep).max(0.0);
            let moved = amount.min(allowed);
            if moved < amount {
                self.log_event(
                    EventKind::Clamped,
                    None,
                    None,
                    Some(moved),
                    None,
                    EventReason::Transfer { asked: amount },
                );
            }
            self.account.transfer(moved, to);
            if to == Wallet::Spot && moved > 0.0 && moved == allowed {
                // all it may give leaves exactly what it keeps, so with the cap off a
                // wallet holding a loss lands on zero equity, its trigger, rather than
                // on a rounding to either side of it
                self.account.margin_cash = keep;
            }
        }
        self.transfers.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{cfg, crash_series, levered, ohlc, perp_cfg, series};
    use crate::engine::{Engine, EngineConfig};
    use crate::reporter::{EventKind, EventReason, FillReason};
    use probatum_core::Wallet;

    #[test]
    fn a_transfer_lands_before_the_orders_decided_with_it() {
        // the margin wallet is empty when the buy is decided, and the buy fills
        // because the cash moved across lands before it
        let config = EngineConfig {
            quote: 0.0,
            idle_cash: 1_000.0,
            max_leverage: 10.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.transfer(1_000.0, Wallet::Margin);
        e.market_buy(1.0);
        let s = e.step(); // the buy fills at the open of 200
        assert_eq!(s.position, 1.0);
        assert_eq!((s.spot_cash, s.margin_cash), (0.0, 1_000.0));
    }

    #[test]
    fn the_margin_wallet_keeps_its_unrealized_profit_and_the_caps_requirement() {
        // the Specification's example: 600 of cash and 400 of unrealized profit under
        // a requirement of 200 moves at most 600, though equity less the requirement
        // is 800
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 200.0, 100.0, 200.0, 1e6),
            ohlc(200.0, 200.0, 200.0, 200.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 600.0,
            max_leverage: 4.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(4.0);
        let s = e.step(); // long 4 from 100, marked at 200: 800 of notional over 4x
        assert_eq!(s.unrealized_pnl, 400.0);
        e.transfer(1_000.0, Wallet::Spot);
        let s = e.step();
        assert_eq!((s.spot_cash, s.margin_cash), (600.0, 0.0));
        assert_eq!(s.equity, 1_000.0);
        // and the clamp is an order-log event, which no order owns
        let clamp = e
            .reporter()
            .expect("reporting is on")
            .events()
            .last()
            .copied()
            .expect("an event");
        assert_eq!((clamp.kind, clamp.order), (EventKind::Clamped, None));
        assert_eq!(clamp.size, Some(600.0));
        assert_eq!(clamp.reason, EventReason::Transfer { asked: 1_000.0 });
    }

    #[test]
    fn a_loss_holds_the_margin_wallet_to_what_covers_the_cap() {
        // long 10 from 100 on 1,000 at 10x, marked at 95: equity 950 less a
        // requirement of 95 frees 855, and the 145 left is worth exactly 95
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 95.0, 95.0, 1e6),
            ohlc(95.0, 95.0, 95.0, 95.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 1_000.0,
            max_leverage: 10.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.transfer(1_000.0, Wallet::Spot);
        let s = e.step();
        assert_eq!((s.spot_cash, s.margin_cash), (855.0, 145.0));
        assert_eq!(s.margin_cash + s.unrealized_pnl, 95.0);
    }

    #[test]
    fn a_transfer_the_wallet_covers_moves_whole_and_logs_nothing() {
        let config = EngineConfig {
            idle_cash: 300.0,
            report: true,
            ..cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.transfer(100.0, Wallet::Spot);
        let s = e.step();
        assert_eq!((s.spot_cash, s.margin_cash), (10_100.0, 200.0));
        assert!(e.reporter().expect("reporting is on").events().is_empty());
    }

    #[test]
    fn a_margin_wallet_already_short_of_the_caps_requirement_gives_nothing() {
        // long 10 from 100 on 100 at 10x, marked at 95: it must keep the cap's 95
        // and the loss's 50, 145 of cash, and holds 100, so nothing moves and
        // nothing is made up
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 95.0, 95.0, 1e6),
            ohlc(95.0, 95.0, 95.0, 95.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.transfer(1_000.0, Wallet::Spot);
        let s = e.step();
        assert_eq!((s.spot_cash, s.margin_cash), (0.0, 100.0));
        assert_eq!(s.equity, 50.0);
    }

    #[test]
    fn with_the_cap_off_a_transfer_can_take_the_margin_wallet_to_its_trigger() {
        // long 1.5 from 100 on 999.7, marked at 95.1: only the loss stays behind, and
        // the long is closed at that close before the next bar opens. At these
        // prices the cash less the loss, with the loss added back, is not zero
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 95.1, 95.1, 1e6),
            ohlc(95.1, 95.1, 95.1, 95.1, 1e6),
        ]);
        let config = EngineConfig {
            quote: 999.7,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.5);
        e.step();
        e.transfer(2_000.0, Wallet::Spot);
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert_eq!(
            (s.spot_cash, s.margin_cash),
            (999.7 - 1.5 * (100.0 - 95.1), 0.0)
        );
        let fill = e
            .reporter()
            .expect("reporting is on")
            .fills()
            .last()
            .copied()
            .expect("a fill");
        assert_eq!((fill.reason, fill.price), (FillReason::Liquidation, 95.1));
    }

    #[test]
    fn a_transfer_out_of_the_spot_wallet_leaves_the_holding() {
        let config = EngineConfig {
            quote: 1_000.0,
            ..cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(2.5); // 500 at the open of 200
        e.step();
        e.transfer(2_000.0, Wallet::Margin);
        let s = e.step();
        assert_eq!((s.spot_cash, s.margin_cash), (0.0, 500.0));
        assert_eq!(s.position, 2.5);
        assert_eq!(s.equity, 2.5 * 350.0 + 500.0);
    }

    #[test]
    fn a_liquidation_takes_the_margin_wallet_and_leaves_the_spot_wallet() {
        // the spot wallet backs no perp, so the long dies where its 100 of margin
        // runs out, and the account it leaves is alive and can fund the next one
        let config = EngineConfig {
            idle_cash: 500.0,
            ..levered()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        e.market_buy(10.0);
        let s = e.step(); // closed out at 90 on the way to the low of 80
        assert_eq!(s.position, 0.0);
        assert_eq!((s.spot_cash, s.margin_cash), (500.0, 0.0));
        assert!(!e.is_bust());
        e.transfer(500.0, Wallet::Margin);
        e.market_buy(1.0);
        let s = e.step();
        assert_eq!(s.position, 1.0);
        assert_eq!((s.spot_cash, s.margin_cash), (0.0, 500.0));
    }

    #[test]
    fn an_amount_that_is_not_positive_moves_nothing_and_a_reset_drops_the_rest() {
        let config = EngineConfig {
            idle_cash: 300.0,
            ..cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.transfer(-5.0, Wallet::Margin);
        e.transfer(f64::NAN, Wallet::Spot);
        let s = e.step();
        assert_eq!((s.spot_cash, s.margin_cash), (10_000.0, 300.0));
        e.transfer(100.0, Wallet::Spot);
        e.reset();
        let s = e.step();
        assert_eq!((s.spot_cash, s.margin_cash), (10_000.0, 300.0));
    }
}
