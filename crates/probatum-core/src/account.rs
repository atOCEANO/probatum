//! The account: two wallets and a netted position per instrument, each settling in
//! its market's wallet. On spot the full notional moves through the spot wallet on
//! every fill (a buy spends cash, a sell receives it), so realized PnL is implicit
//! in the cash. On a perp only realized PnL moves the margin wallet on a close, and
//! the open position is marked separately. A wallet no instrument settles in holds
//! cash that backs nothing until a transfer moves it across (ADR 0008).

use crate::enums::{Market, Wallet};
use crate::fill::Fill;
use crate::position::{moves_position, Position, POS_EPS};
use crate::units::{Price, Qty};

/// True when `price` can divide a notional: finite and strictly positive. A bar
/// whose close is zero is accepted by the input contract (only non-finite values
/// are rejected), so the sizing helpers check it themselves.
#[inline]
fn tradeable(price: Price) -> bool {
    price.get().is_finite() && price.get() > 0.0
}

/// The two wallets' cash and the position each instrument holds against them.
#[derive(Clone, Debug, PartialEq)]
pub struct Account {
    /// The spot wallet's cash, in the quote asset.
    pub spot_cash: f64,
    /// The margin wallet's cash, in the quote asset: the perps' collateral, which
    /// their realized PnL, fees and funding move.
    pub margin_cash: f64,
    /// Each instrument's position, in the universe's order.
    pub positions: Vec<Position>,
    /// The market each instrument trades in, which decides the wallet it settles in.
    pub markets: Vec<Market>,
    // the perps holding a position, the one that does when it is one, and the
    // realized PnL of every position, kept as the fills land so a step reads them
    // without a pass over the instruments; a position moves only through
    // `apply_fill`
    perps: usize,
    sole: usize,
    realized: f64,
}

impl Account {
    /// A fresh account on one instrument, with `quote` cash in the wallet `market`
    /// settles in, the other wallet empty, and no position. Nothing is validated
    /// here; the Python boundary is where a negative or non-finite quote is refused
    /// (ADR 0007).
    pub fn new(market: Market, quote: f64) -> Account {
        let mut account = Account::universe(vec![market], 0.0, 0.0);
        *account.cash_mut(market.wallet()) = quote;
        account
    }

    /// A fresh account on the instruments of `markets`, with each wallet's cash and
    /// no position.
    pub fn universe(markets: Vec<Market>, spot_cash: f64, margin_cash: f64) -> Account {
        Account {
            spot_cash,
            margin_cash,
            positions: vec![Position::flat(); markets.len()],
            markets,
            perps: 0,
            sole: 0,
            realized: 0.0,
        }
    }

    /// The same account with `cash` in the wallet its first instrument does not
    /// settle in.
    pub fn with_idle(mut self, cash: f64) -> Account {
        *self.cash_mut(self.markets[0].wallet().other()) = cash;
        self
    }

    /// Start over with each wallet's cash and every position flat, keeping the
    /// buffers.
    pub fn reset(&mut self, spot_cash: f64, margin_cash: f64) {
        self.spot_cash = spot_cash;
        self.margin_cash = margin_cash;
        self.positions.fill(Position::flat());
        self.perps = 0;
        self.realized = 0.0;
    }

    /// The cash `wallet` holds.
    #[inline]
    pub fn cash(&self, wallet: Wallet) -> f64 {
        match wallet {
            Wallet::Spot => self.spot_cash,
            Wallet::Margin => self.margin_cash,
        }
    }

    #[inline]
    fn cash_mut(&mut self, wallet: Wallet) -> &mut f64 {
        match wallet {
            Wallet::Spot => &mut self.spot_cash,
            Wallet::Margin => &mut self.margin_cash,
        }
    }

    /// Apply a fill on instrument `i` and its fee to the wallet it settles in,
    /// moving cash by its market's convention. A fill the position refuses
    /// (non-finite, or sized at or below the dust epsilon) moves no cash and pays no
    /// fee, so a wallet can never change without a position changing with it
    /// (ADR 0014).
    pub fn apply_fill(&mut self, i: usize, fill: &Fill, fee: f64) {
        if !moves_position(fill) {
            return;
        }
        let market = self.markets[i];
        let held = !self.positions[i].is_flat();
        let realized = self.positions[i].apply(fill);
        self.realized += realized;
        if market == Market::Perp {
            match (held, self.positions[i].is_flat()) {
                (false, false) => {
                    self.perps += 1;
                    self.sole = i;
                }
                (true, true) => {
                    self.perps -= 1;
                    if self.perps == 1 {
                        self.sole = (0..self.positions.len())
                            .find(|&j| {
                                self.markets[j] == Market::Perp && !self.positions[j].is_flat()
                            })
                            .unwrap_or(0);
                    }
                }
                _ => {}
            }
        }
        let cash = self.cash_mut(market.wallet());
        *cash -= fee;
        match market {
            Market::Spot => {
                *cash -= fill.side.sign() * fill.size.get() * fill.price.get();
            }
            Market::Perp => {
                *cash += realized;
            }
        }
    }

    /// What `wallet` is worth with each instrument marked at its entry in `marks`,
    /// in quote: its cash, plus the spot holdings when it is the spot wallet, or the
    /// perps' unrealized PnL when it is the margin wallet. The margin wallet's is the
    /// equity the cap and the liquidation read, and the spot wallet's cash backs no
    /// perp. A flat instrument adds nothing, so its mark may be NaN before it has
    /// printed a bar.
    pub fn wallet_equity(&self, wallet: Wallet, marks: &[f64]) -> f64 {
        let mut equity = self.cash(wallet);
        for (i, position) in self.positions.iter().enumerate() {
            if position.qty.get() == 0.0 || self.markets[i].wallet() != wallet {
                continue;
            }
            equity += match self.markets[i] {
                Market::Spot => position.qty.get() * marks[i],
                Market::Perp => position.unrealized(Price(marks[i])),
            };
        }
        equity
    }

    /// Account value marked at `marks`, in the quote asset, the sum of both wallets.
    pub fn equity(&self, marks: &[f64]) -> f64 {
        let (mut spot, mut margin) = (self.spot_cash, self.margin_cash);
        for (i, position) in self.positions.iter().enumerate() {
            let qty = position.qty.get();
            if qty == 0.0 {
                continue;
            }
            match self.markets[i] {
                Market::Spot => spot += qty * marks[i],
                Market::Perp => margin += position.unrealized(Price(marks[i])),
            }
        }
        spot + margin
    }

    /// Unrealized PnL of instrument `i`'s position at `mark`, in quote: the price
    /// move alone, with no fee and no funding in it.
    #[inline]
    pub fn unrealized(&self, i: usize, mark: Price) -> f64 {
        self.positions[i].unrealized(mark)
    }

    /// Cumulative realized PnL since the account was built, in quote, gross of
    /// fees: a fee comes straight off the wallet and never enters this.
    #[inline]
    pub fn realized(&self) -> f64 {
        self.realized
    }

    /// The perps holding a position.
    #[inline]
    pub fn perps_held(&self) -> usize {
        self.perps
    }

    /// The perp holding a position when exactly one does.
    #[inline]
    pub fn sole_perp(&self) -> Option<usize> {
        (self.perps == 1).then_some(self.sole)
    }

    /// Base size worth `fraction` of the equity at `marks`, at `price`. `fraction`
    /// can exceed 1 for leverage on a perp; the margin cap is enforced elsewhere. A
    /// price that is not finite and positive has no size to give, so it returns
    /// zero rather than an infinity the caller would have to notice (ADR 0014).
    pub fn qty_from_weight(&self, fraction: f64, price: Price, marks: &[f64]) -> Qty {
        if !tradeable(price) {
            return Qty(0.0);
        }
        Qty(fraction * self.equity(marks) / price.get())
    }

    /// Base size worth `cash` quote at `price`. Zero on a non-positive or
    /// non-finite price, for the same reason as `qty_from_weight`.
    pub fn qty_from_quote(&self, cash: f64, price: Price) -> Qty {
        if !tradeable(price) {
            return Qty(0.0);
        }
        Qty(cash / price.get())
    }

    /// Apply one funding charge at `rate` on instrument `i`'s position marked at
    /// `mark`, to the margin wallet. A long pays when the rate is positive and a
    /// short receives; spot has no funding. Returns the amount paid (positive) or
    /// received (negative), in quote.
    pub fn apply_funding(&mut self, i: usize, rate: f64, mark: Price) -> f64 {
        if self.markets[i] != Market::Perp || !rate.is_finite() || !mark.get().is_finite() {
            return 0.0;
        }
        let payment = self.positions[i].qty.get() * mark.get() * rate;
        self.margin_cash -= payment;
        payment
    }

    /// Move `amount` into the `to` wallet from the other one. The engine clamps it
    /// first, since what may leave the margin wallet depends on the cap.
    pub fn transfer(&mut self, amount: f64, to: Wallet) {
        *self.cash_mut(to.other()) -= amount;
        *self.cash_mut(to) += amount;
    }

    /// True when the account is dead at `marks`: it holds a perp and its margin
    /// wallet has fallen to zero or below. Spot is never liquidated, and the spot
    /// wallet's cash does not keep a perp alive (ADRs 0008, 0017).
    ///
    /// The test is the trigger only. The price a position is then closed at is
    /// `bankruptcy_price`, which is not this mark: `marks` hold the worst the bar
    /// printed and closing there is what used to leave the account owing money.
    pub fn is_bust_at(&self, marks: &[f64]) -> bool {
        self.perps_held() > 0 && self.wallet_equity(Wallet::Margin, marks) <= 0.0
    }

    /// The price at which closing the whole of perp `i`'s position, and paying
    /// `fee_rate` on the closing notional, leaves `wallet` with exactly nothing:
    /// the margin wallet's cash when `i` is the only perp held, or that cash with
    /// what the other perps hold beyond their own closing fees.
    ///
    /// Solving `W + qty * (price - entry) - |qty| * price * fee_rate == 0` for
    /// price, with `W` that wallet. This is what makes a liquidation
    /// unable to leave a debt: the loss is bounded by the margin because the exit is
    /// priced at the point the margin runs out, rather than at whatever the bar
    /// happened to print afterwards. It also answers where a fee comes from when
    /// nothing is left to pay it with: the close lands far enough before ruin that
    /// the fee fits inside what remains (ADR 0018). `None` when the position is
    /// dust or the arithmetic is degenerate.
    pub fn bankruptcy_price(&self, i: usize, fee_rate: f64, wallet: f64) -> Option<Price> {
        let qty = self.positions[i].qty.get();
        let size = qty.abs();
        if size <= POS_EPS || !fee_rate.is_finite() {
            return None;
        }
        let denominator = qty - size * fee_rate;
        if denominator.abs() <= f64::EPSILON {
            return None;
        }
        let price = (qty * self.positions[i].avg_entry.get() - wallet) / denominator;
        if price.is_finite() && price > 0.0 {
            Some(Price(price))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Account;
    use crate::enums::{Market, Side, Wallet};
    use crate::fill::Fill;
    use crate::units::{Price, Qty};
    use proptest::prelude::*;

    fn fill(side: Side, size: f64, price: f64) -> Fill {
        Fill {
            side,
            size: Qty(size),
            price: Price(price),
            is_taker: true,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn spot_buy_spends_quote_and_holds_base() {
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0);
        assert!(close(a.spot_cash, 800.0));
        assert!(close(a.equity(&[100.0]), 1000.0)); // no PnL at entry
        assert!(close(a.equity(&[110.0]), 1020.0)); // +20 unrealized
    }

    #[test]
    fn spot_round_trip_realizes_in_cash() {
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0);
        a.apply_fill(0, &fill(Side::Sell, 2.0, 110.0), 0.0);
        assert!(close(a.spot_cash, 1020.0));
        assert!(a.positions[0].is_flat());
        // Flat: equity is just the cash, independent of mark, and not double-counted.
        assert!(close(a.equity(&[999.0]), 1020.0));
    }

    #[test]
    fn perp_holds_margin_and_marks_unrealized() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0);
        assert!(close(a.margin_cash, 1000.0)); // opening a perp moves no cash
        assert!(close(a.equity(&[110.0]), 1020.0));
    }

    #[test]
    fn perp_close_books_pnl_to_quote() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0);
        a.apply_fill(0, &fill(Side::Sell, 2.0, 110.0), 0.0);
        assert!(close(a.margin_cash, 1020.0));
        assert!(a.positions[0].is_flat());
    }

    #[test]
    fn fee_reduces_quote() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.12);
        assert!(close(a.margin_cash, 999.88));
    }

    #[test]
    fn qty_from_weight_is_fraction_of_equity() {
        let a = Account::new(Market::Perp, 1000.0);
        // flat: equity is 1000; weight 0.5 at price 100 is 500 notional, 5 base
        assert!(close(
            a.qty_from_weight(0.5, Price(100.0), &[100.0]).get(),
            5.0
        ));
    }

    #[test]
    fn qty_from_quote_is_cash_over_price() {
        let a = Account::new(Market::Perp, 1000.0);
        assert!(close(a.qty_from_quote(250.0, Price(100.0)).get(), 2.5));
    }

    #[test]
    fn perp_long_pays_positive_funding() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0);
        let paid = a.apply_funding(0, 0.0001, Price(100.0));
        assert!(close(paid, 0.02)); // 2 * 100 * 0.0001
        assert!(close(a.margin_cash, 999.98));
    }

    #[test]
    fn perp_short_receives_positive_funding() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(0, &fill(Side::Sell, 2.0, 100.0), 0.0);
        let paid = a.apply_funding(0, 0.0001, Price(100.0));
        assert!(close(paid, -0.02)); // received
        assert!(close(a.margin_cash, 1000.02));
    }

    #[test]
    fn spot_has_no_funding() {
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0); // quote 800
        assert_eq!(a.apply_funding(0, 0.01, Price(100.0)), 0.0);
        assert!(close(a.spot_cash, 800.0));
    }

    #[test]
    fn flat_position_pays_no_funding() {
        let mut a = Account::new(Market::Perp, 1000.0);
        assert_eq!(a.apply_funding(0, 0.01, Price(100.0)), 0.0);
        assert!(close(a.margin_cash, 1000.0));
    }

    #[test]
    fn perp_long_is_bust_where_its_margin_runs_out() {
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(0, &fill(Side::Buy, 10.0, 100.0), 0.0); // 10x notional on 100 margin

        // equity at 90 = 100 + 10*(90-100) = 0
        assert!(a.is_bust_at(&[90.0]));
        assert!(!a.is_bust_at(&[90.01]));
        assert!(close(
            a.bankruptcy_price(0, 0.0, a.margin_cash).unwrap().get(),
            90.0
        ));
    }

    #[test]
    fn perp_short_is_bust_when_the_mark_rises_far_enough() {
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(0, &fill(Side::Sell, 10.0, 100.0), 0.0);
        // equity at 110 = 100 + (-10)*(110-100) = 0
        assert!(a.is_bust_at(&[110.0]));
        assert!(!a.is_bust_at(&[109.99]));
        assert!(close(
            a.bankruptcy_price(0, 0.0, a.margin_cash).unwrap().get(),
            110.0
        ));
    }

    #[test]
    fn closing_at_the_bankruptcy_price_leaves_exactly_nothing() {
        // whatever the bar printed, the position cannot lose more than the margin
        // behind it, so the exit is priced where that margin runs out (ADR 0018)
        for side in [Side::Buy, Side::Sell] {
            let mut a = Account::new(Market::Perp, 100.0);
            a.apply_fill(0, &fill(side, 10.0, 100.0), 0.0);
            let price = a.bankruptcy_price(0, 0.0, a.margin_cash).unwrap();
            let closing = if side == Side::Buy {
                Side::Sell
            } else {
                Side::Buy
            };
            a.apply_fill(0, &fill(closing, 10.0, price.get()), 0.0);
            assert!(a.positions[0].is_flat());
            assert!(close(a.margin_cash, 0.0), "{side:?} left {}", a.margin_cash);
        }
    }

    #[test]
    fn the_liquidation_fee_fits_inside_what_is_left() {
        // there is nothing to pay a fee from at the moment of ruin, so the exit is
        // priced far enough before it that the fee still lands the account at zero
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(0, &fill(Side::Buy, 10.0, 100.0), 0.0);
        let price = a.bankruptcy_price(0, 0.0006, a.margin_cash).unwrap();
        assert!(price.get() > 90.0, "priced at {}", price.get()); // before ruin, not after
        let fee = 10.0 * price.get() * 0.0006;
        a.apply_fill(0, &fill(Side::Sell, 10.0, price.get()), fee);
        assert!(close(a.margin_cash, 0.0), "left {}", a.margin_cash);
    }

    #[test]
    fn solvent_position_is_not_bust() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 1.0, 100.0), 0.0);
        assert!(!a.is_bust_at(&[90.0])); // equity 990, fine
        assert_eq!(a.positions[0].qty, Qty(1.0));
    }

    #[test]
    fn spot_and_flat_are_never_bust() {
        let mut spot = Account::new(Market::Spot, 1000.0);
        spot.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0);
        assert!(!spot.is_bust_at(&[1.0]));

        let flat = Account::new(Market::Perp, 1000.0);
        assert!(!flat.is_bust_at(&[50.0]));
        assert!(flat.bankruptcy_price(0, 0.0, flat.margin_cash).is_none());
    }

    #[test]
    fn the_account_counts_its_perps_and_realized_pnl_as_fills_land() {
        // a spot holding is no perp, a flip keeps its perp held, and the realized
        // total is the 20 the short closed at a profit and the 10 the long did
        let markets = vec![Market::Perp, Market::Spot, Market::Perp];
        let mut a = Account::universe(markets, 1_000.0, 1_000.0);
        a.apply_fill(0, &fill(Side::Buy, 1.0, 100.0), 0.0);
        a.apply_fill(1, &fill(Side::Buy, 1.0, 100.0), 0.0);
        a.apply_fill(2, &fill(Side::Sell, 2.0, 50.0), 0.0);
        assert_eq!(a.perps_held(), 2);
        a.apply_fill(2, &fill(Side::Buy, 3.0, 40.0), 0.0);
        assert_eq!(a.perps_held(), 2);
        a.apply_fill(0, &fill(Side::Sell, 1.0, 110.0), 0.0);
        assert_eq!((a.perps_held(), a.sole_perp()), (1, Some(2)));
        assert!(close(a.realized(), 30.0));
        a.reset(1_000.0, 1_000.0);
        assert_eq!((a.perps_held(), a.realized()), (0, 0.0));
    }

    #[test]
    fn each_instrument_settles_in_its_own_markets_wallet() {
        // a spot buy spends the spot wallet's cash and a perp long moves none, and a
        // third instrument that has printed nothing adds nothing at its NaN mark
        let markets = vec![Market::Spot, Market::Perp, Market::Perp];
        let mut a = Account::universe(markets, 1_000.0, 500.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0);
        a.apply_fill(1, &fill(Side::Buy, 3.0, 50.0), 0.0);
        assert_eq!((a.spot_cash, a.margin_cash), (800.0, 500.0));
        let marks = [110.0, 40.0, f64::NAN];
        assert!(close(a.wallet_equity(Wallet::Spot, &marks), 1_020.0));
        assert!(close(a.wallet_equity(Wallet::Margin, &marks), 470.0));
        assert!(close(a.equity(&marks), 1_490.0));
        assert_eq!(a.perps_held(), 1);
    }

    #[test]
    fn a_number_funds_the_wallet_the_market_settles_in() {
        let spot = Account::new(Market::Spot, 1000.0).with_idle(300.0);
        assert_eq!((spot.spot_cash, spot.margin_cash), (1000.0, 300.0));
        let perp = Account::new(Market::Perp, 1000.0).with_idle(300.0);
        assert_eq!((perp.spot_cash, perp.margin_cash), (300.0, 1000.0));
        assert_eq!(perp.cash(Wallet::Margin), 1000.0);
    }

    #[test]
    fn the_spot_wallet_counts_toward_equity_but_backs_no_perp() {
        let mut a = Account::new(Market::Perp, 100.0).with_idle(1000.0);
        a.apply_fill(0, &fill(Side::Buy, 10.0, 100.0), 0.0);
        // at 90 the margin wallet is worth nothing, whatever the spot wallet holds
        assert!(close(a.wallet_equity(Wallet::Margin, &[90.0]), 0.0));
        assert!(close(a.equity(&[90.0]), 1000.0));
        assert!(a.is_bust_at(&[90.0]));
        assert!(close(
            a.bankruptcy_price(0, 0.0, a.margin_cash).unwrap().get(),
            90.0
        ));
    }

    #[test]
    fn a_transfer_moves_cash_from_the_other_wallet() {
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0); // 800 left, long 2
        a.transfer(300.0, Wallet::Margin);
        assert_eq!((a.spot_cash, a.margin_cash), (500.0, 300.0));
        assert!(close(a.equity(&[100.0]), 1000.0));
        a.transfer(100.0, Wallet::Spot);
        assert_eq!((a.spot_cash, a.margin_cash), (600.0, 200.0));
    }

    #[test]
    fn a_sub_dust_fill_moves_neither_position_nor_cash() {
        // the position ignores anything at or below POS_EPS, so the cash move and
        // the fee must be gated on the same test or quote drifts on its own (ADR 0014)
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(0, &fill(Side::Buy, 2.0, 100.0), 0.0); // quote 800, long 2
        let before = a.clone();
        a.apply_fill(0, &fill(Side::Sell, 1e-10, 300.0), 0.5);
        assert_eq!(a, before);
    }

    #[test]
    fn a_non_finite_or_negative_fill_moves_no_cash() {
        let mut a = Account::new(Market::Spot, 1000.0);
        let before = a.clone();
        a.apply_fill(0, &fill(Side::Buy, f64::NAN, 100.0), 1.0);
        a.apply_fill(0, &fill(Side::Buy, 1.0, f64::NAN), 1.0);
        a.apply_fill(0, &fill(Side::Buy, -100.0, 100.0), -20_000.0);
        assert_eq!(a, before);
        assert!(a.spot_cash.is_finite());
    }

    #[test]
    fn sizing_helpers_return_zero_on_an_untradeable_price() {
        // a zero close passes the candle contract (only non-finite is rejected), and
        // dividing by it would hand the caller an infinity instead of a size
        let a = Account::new(Market::Spot, 1000.0);
        assert_eq!(a.qty_from_weight(0.5, Price(0.0), &[0.0]), Qty(0.0));
        assert_eq!(a.qty_from_quote(250.0, Price(0.0)), Qty(0.0));
        assert_eq!(a.qty_from_weight(0.5, Price(-1.0), &[-1.0]), Qty(0.0));
        assert_eq!(a.qty_from_quote(250.0, Price(f64::NAN)), Qty(0.0));
    }

    fn fill_seq(len: std::ops::Range<usize>) -> impl Strategy<Value = Vec<Fill>> {
        prop::collection::vec((any::<bool>(), 0.1f64..10.0, 1.0f64..500.0), len).prop_map(|rows| {
            rows.into_iter()
                .map(|(buy, size, price)| Fill {
                    side: if buy { Side::Buy } else { Side::Sell },
                    size: Qty(size),
                    price: Price(price),
                    is_taker: true,
                })
                .collect()
        })
    }

    proptest! {
        #[test]
        fn perp_quote_moves_only_by_realized(seq in fill_seq(1..30)) {
            // on a perp with no fees, cash changes only by booked PnL: opening and
            // marking never move quote, so quote stays at start plus realized
            let mut a = Account::new(Market::Perp, 10_000.0);
            for f in &seq {
                a.apply_fill(0, f, 0.0);
            }
            prop_assert!(a.margin_cash.is_finite());
            prop_assert!((a.margin_cash - (10_000.0 + a.positions[0].realized)).abs() < 1e-6);
        }

        #[test]
        fn spot_round_trip_returns_quote_plus_pnl(
            size in 0.1f64..10.0,
            p in 1.0f64..500.0,
            q in 1.0f64..500.0,
        ) {
            // buy then sell the same size on spot: the account ends flat and quote
            // moves by exactly the price difference times size
            let mut a = Account::new(Market::Spot, 1_000_000.0);
            let buy = Fill { side: Side::Buy, size: Qty(size), price: Price(p), is_taker: true };
            let sell = Fill { side: Side::Sell, size: Qty(size), price: Price(q), is_taker: true };
            a.apply_fill(0, &buy, 0.0);
            a.apply_fill(0, &sell, 0.0);
            prop_assert!(a.positions[0].is_flat());
            prop_assert!((a.spot_cash - (1_000_000.0 + size * (q - p))).abs() < 1e-6);
        }
    }
}
