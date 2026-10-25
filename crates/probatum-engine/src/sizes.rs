//! What a venue holds an instrument's orders to where its sizes are on, and the grid
//! its prices sit on: the lot every size is a whole number of, the price rule, the
//! smallest notional an order may have and the largest size (Specification, Sizes
//! and dust). Each is in the engine's terms, an inverse contract's lot in the
//! dollars its contracts are worth and its tick on the reciprocal of the price.

use probatum_core::Side;
use serde::{Deserialize, Serialize};

use crate::inverse::quoted;

/// The powers of ten a double holds exactly.
const TENS: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `10^k`, exact for `k >= 0` and `1 / 10^-k` below zero, for `k` within 22 of zero.
fn power(k: i32) -> f64 {
    if k >= 0 {
        TENS[k as usize]
    } else {
        1.0 / TENS[(-k) as usize]
    }
}

/// The whole lots in `size`, a size floating point left a hair under a whole number
/// of lots kept at that number.
pub fn lots(size: f64, lot: f64) -> f64 {
    (size / lot + 1e-9).floor()
}

/// The grid an instrument's prices sit on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum PriceRule {
    /// None: a price stands as given, and `through` is `touch`.
    #[default]
    Free,
    /// A fixed tick.
    Tick(f64),
    /// Hyperliquid's: `figures` significant figures within `cap` decimals, a whole
    /// number always allowed.
    Figures { figures: u32, cap: u32 },
    /// A fixed tick on the price's reciprocal, which is an inverse contract's dollar
    /// price in the engine's terms.
    Reciprocal(f64),
}

impl PriceRule {
    /// The decimals `figures` significant figures within `cap` keep at `price`, a cap
    /// past the 22 decimals whose powers of ten a double holds exactly read as 22.
    fn decimals(figures: u32, cap: u32, price: f64) -> i32 {
        let size = price.abs();
        let mut e = 0;
        while e < 22 && size >= power(e + 1) {
            e += 1;
        }
        while e > -22 && size < power(e) {
            e -= 1;
        }
        (figures as i32 - 1 - e).max(0).min(cap.min(22) as i32)
    }

    /// `price` on the grid, ties to even.
    pub fn round(self, price: f64) -> f64 {
        match self {
            PriceRule::Free => price,
            PriceRule::Tick(tick) => (price / tick).round_ties_even() * tick,
            PriceRule::Figures { figures, cap } => {
                let scale = power(Self::decimals(figures, cap, price));
                (price * scale).round_ties_even() / scale
            }
            // rounded on the dollar price quoted, read back off its reciprocal
            PriceRule::Reciprocal(tick) => 1.0 / PriceRule::Tick(tick).round(quoted(price)),
        }
    }

    /// The price a bar has to trade at or past for a limit at `price` on `side` to
    /// fill under `through`: half the grid's step past it, measured on the venue's
    /// own prices.
    pub fn through(self, side: Side, price: f64) -> f64 {
        let half = |step: f64| side.sign() * step / 2.0;
        match self {
            PriceRule::Free => price,
            PriceRule::Tick(tick) => price - half(tick),
            PriceRule::Figures { figures, cap } => {
                price - half(power(-Self::decimals(figures, cap, price)))
            }
            // on the dollar price quoted, and never reached where a dollar buy sits
            // within half a tick of zero
            PriceRule::Reciprocal(tick) => {
                let dollars = quoted(price) + half(tick);
                if dollars > 0.0 {
                    1.0 / dollars
                } else {
                    f64::INFINITY
                }
            }
        }
    }
}

/// The rules a venue's sizes hold one instrument's orders to.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sizes {
    /// The lot every size is a whole number of, none at zero.
    pub lot: f64,
    /// The grid every price sits on.
    pub prices: PriceRule,
    /// The smallest notional an order may have at the last close.
    pub min_notional: f64,
    /// The largest size an order may have.
    pub max_qty: f64,
    /// Whether a reduce-only order is spared the minimum notional.
    pub reduce_only_exempt: bool,
}

impl Default for Sizes {
    fn default() -> Sizes {
        Sizes {
            lot: 0.0,
            prices: PriceRule::Free,
            min_notional: 0.0,
            max_qty: f64::INFINITY,
            reduce_only_exempt: false,
        }
    }
}

impl Sizes {
    /// `size` rounded down to the lot, as it stands with no lot.
    pub fn down(self, size: f64) -> f64 {
        if self.lot > 0.0 {
            lots(size, self.lot) * self.lot
        } else {
            size
        }
    }

    /// Whether an order with `left` to fill can no longer fill: what remains rounds
    /// down to no lot, or with no lot is at most `dust`.
    pub fn spent(self, left: f64, dust: f64) -> bool {
        if self.lot > 0.0 {
            lots(left, self.lot) <= 0.0
        } else {
            left <= dust
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{lots, PriceRule, Sizes};
    use probatum_core::Side;

    #[test]
    fn a_size_a_hair_under_a_whole_number_of_lots_keeps_it() {
        // 0.7 - 0.4 is a hair under 0.3, three lots of 0.1
        assert_eq!(lots(0.7 - 0.4, 0.1), 3.0);
        assert_eq!(lots(0.0754, 0.001), 75.0);
        let sizes = Sizes {
            lot: 0.001,
            ..Sizes::default()
        };
        assert!((sizes.down(0.0754) - 0.075).abs() < 1e-15);
        assert_eq!(sizes.down(0.0004), 0.0);
        assert!(sizes.spent(0.0009, 1e-9));
        assert!(!sizes.spent(0.001, 1e-9));
    }

    #[test]
    fn five_figures_with_a_cap_of_one_decimal_rest_1_234_56_at_1_234_6() {
        let rule = PriceRule::Figures { figures: 5, cap: 1 };
        assert_eq!(rule.round(1_234.56), 1_234.6);
        // a whole number is always allowed
        assert_eq!(rule.round(123_456.7), 123_457.0);
        // the cap holds a small price to one decimal, and one past 22 is read as 22
        assert_eq!(rule.round(99.987), 100.0);
        let wide = PriceRule::Figures {
            figures: 22,
            cap: 30,
        };
        assert_eq!(wide.round(0.05), 0.05);
        // the step at 1,234.6 is 0.1, so a buy there under through needs a low half
        // of it below, which on prices on the grid is 1,234.5
        assert!((rule.through(Side::Buy, 1_234.6) - 1_234.55).abs() < 1e-9);
        assert!((rule.through(Side::Sell, 1_234.6) - 1_234.65).abs() < 1e-9);
    }

    #[test]
    fn a_tick_rounds_ties_to_even_and_takes_half_a_tick_through() {
        let rule = PriceRule::Tick(0.5);
        assert_eq!(rule.round(100.25), 100.0);
        assert_eq!(rule.round(100.75), 101.0);
        assert_eq!(rule.through(Side::Buy, 99.5), 99.25);
        assert_eq!(rule.through(Side::Sell, 99.5), 99.75);
        assert_eq!(PriceRule::Free.through(Side::Buy, 99.5), 99.5);
    }

    #[test]
    fn an_inverse_tick_is_a_dollar_tick_on_the_reciprocal() {
        // a dollar buy at 60,000 is an engine sell at its reciprocal, which a bar
        // reaches under through once the dollar low is half a tick below
        let rule = PriceRule::Reciprocal(0.1);
        let x = 1.0 / 60_000.0;
        let near = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b;
        assert!(near(rule.through(Side::Sell, x), 1.0 / (60_000.0 - 0.05)));
        assert!(near(rule.through(Side::Buy, x), 1.0 / (60_000.0 + 0.05)));
        assert!(near(rule.round(1.0 / 60_000.04), 1.0 / 60_000.0));
        // a dollar sell at 49 under a tick of 1 is reached by a dollar high of 49.5
        // exactly, read off the price quoted rather than the reciprocal taken twice
        let one = PriceRule::Reciprocal(1.0);
        assert_eq!(one.through(Side::Buy, 1.0 / 49.0), 1.0 / 49.5);
        // and a dollar buy within half a tick of zero is never reached
        let half = PriceRule::Reciprocal(0.5);
        assert_eq!(half.through(Side::Sell, 1.0 / 0.2), f64::INFINITY);
        // a price at half a tick rounds as the dollar price quoted does, where its
        // reciprocal taken twice could land either side of the half
        for quoted in [60_000.05, 60_000.15, 60_000.25, 99.95] {
            let dollar = PriceRule::Tick(0.1).round(quoted);
            assert_eq!(rule.round(1.0 / quoted), 1.0 / dollar, "{quoted}");
        }
    }
}
