//! The next-bar fill model. An order decided at the close of one bar fills on the
//! NEXT bar, so a fill is never priced off information the decision could not have
//! seen. Market fills are takers, a limit is a taker only on a first bar that opens
//! at or through it, and a stop triggers and then fills as a taker. What a fill may
//! take and what it pays read the window before its bar (Specification, Fills; ADRs
//! 0011, 0013).

use probatum_core::{Bps, Candle, Fill, Order, Price, Qty, Side};

use crate::window::Trailing;

/// The most a taker fill may be moved against itself, as a fraction of the
/// reference price. A slip of 1.0 would price a sell at zero and anything beyond
/// it below zero, which is not a trade, so the total is held just under 1 (ADR 0013).
const MAX_SLIP: f64 = 0.99;

/// How a taker's impact grows with its share of the trailing volume.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Impact {
    /// `k * sigma * q / V`.
    #[default]
    Linear,
    /// `k * sigma * sqrt(q / V)`.
    Sqrt,
}

/// The spread a taker crosses half of.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Spread {
    #[default]
    None,
    /// A fixed spread, in basis points of the price.
    Fixed(f64),
    /// Ardia, Guidotti and Kroencke's estimate over the window.
    Estimated,
}

/// The cost side of execution: what a taker pays beyond its reference. What a fill
/// may take of a bar is the engine's budget, which every order on the bar shares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillModel {
    /// Adverse slippage on a taker fill, in basis points off the reference price.
    pub slippage_bps: Bps,
    /// The impact coefficient `k`, scaled by the window's volatility and the fill's
    /// share of its trailing volume. Zero disables it.
    pub impact: f64,
    pub impact_model: Impact,
    pub spread: Spread,
}

/// What a fill is measured against and what it paid beyond that, each per unit of
/// size and in price, as the fill log records it (ADR 0022).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FillCosts {
    /// The price the fill is measured from: the open for a market order, the worse
    /// of the open and the trigger for a stop, and a limit's own price, or the
    /// price the market reached for it since the last close.
    pub reference: f64,
    /// Half the spread.
    pub spread: f64,
    pub slippage: f64,
    pub impact: f64,
    /// How far the price was moved after its costs set it: held inside the bar, or
    /// lifted by the margin bound.
    pub adjustment: f64,
}

/// Whether `x` is a positive finite number, as a size, a price or a volume has to be:
/// its bits run from the smallest subnormal's, 1, to the largest float's, so one
/// unsigned comparison of them decides it. Written as float comparisons, the test
/// compiles to a classification of those bits some twenty instructions long, on
/// every fill of the step loop.
pub(crate) fn positive(x: f64) -> bool {
    x.to_bits().wrapping_sub(1) < f64::MAX.to_bits()
}

impl FillModel {
    /// The fillable size: the order's remaining, held to `available`, what the bar's
    /// budget still allows its instrument. A bar that printed no volume fills
    /// nothing. A non-finite remaining would launder a NaN size into a fill of the
    /// whole budget (ADR 0014), so it fills nothing too; guarded so, the smaller is
    /// taken with one comparison, where `f64::min` would pay to skip a NaN neither
    /// side can be.
    fn cap(&self, order: &Order, bar: &Candle, available: f64) -> f64 {
        let remaining = order.remaining().get();
        // a NaN budget, which no bar can hold, is no budget either
        let left = available > 0.0;
        if !remaining.is_finite() || !positive(bar.volume) || !left {
            return 0.0;
        }
        if available < remaining {
            available
        } else {
            remaining
        }
    }

    /// The adverse move on a taker fill of `size`, as its total and its three parts:
    /// half the spread, the flat slippage and the impact on the window `trailing`
    /// gives, which is read only where the impact or an estimated spread asks for it.
    /// The slippage and the impact are floored at zero, since a cost is adverse by
    /// definition (ADR 0013) and a negative one would fill better than the market,
    /// and the total is capped below 1 so the reference price can never be dropped to
    /// zero or through it, the parts scaled down with it (ADR 0013).
    fn taker_slip(&self, size: f64, trailing: impl Fn() -> Trailing) -> (f64, f64, f64, f64) {
        let flat = self.slippage_bps.as_fraction().max(0.0);
        let coefficient = self.impact.max(0.0);
        // with no spread and no impact the slip is the flat rate under its cap, as
        // the sums below would find it
        if coefficient == 0.0 && matches!(self.spread, Spread::None) && flat <= MAX_SLIP {
            return (flat, 0.0, flat, 0.0);
        }
        let window = if coefficient > 0.0 || matches!(self.spread, Spread::Estimated) {
            trailing()
        } else {
            Trailing::default()
        };
        let half = match self.spread {
            Spread::None => 0.0,
            Spread::Fixed(bps) => bps / 10_000.0 / 2.0,
            Spread::Estimated => window.spread / 2.0,
        };
        let impact = if coefficient > 0.0 && positive(window.volume) {
            let share = size / window.volume;
            coefficient
                * window.sigma
                * match self.impact_model {
                    Impact::Linear => share,
                    Impact::Sqrt => share.sqrt(),
                }
        } else {
            0.0
        };
        let slip = half + flat + impact;
        if !slip.is_finite() {
            // an impact past every float, from a window that traded almost nothing
            return (MAX_SLIP, 0.0, 0.0, MAX_SLIP);
        }
        let total = slip.min(MAX_SLIP);
        let scale = if slip > 0.0 { total / slip } else { 1.0 };
        (total, half * scale, flat * scale, impact * scale)
    }

    /// A taker price held inside the bar that produced it.
    ///
    /// Slippage and impact are adverse by construction, and impact is bounded above
    /// by nothing the bar shows: it grows with the share of the trailing volume
    /// taken (ADR 0013), and the total is capped only at `MAX_SLIP`, which is a rule
    /// about not reaching zero rather than about the market. So an order taking a
    /// whole bar was priced far from anything the bar traded near. A venue cannot
    /// fill outside what traded, and a loss booked against a price the bar does not
    /// show is bounded by nothing at all, which defeats the liquidation's own
    /// guarantee from outside the liquidation (ADRs 0017, 0018). The bound is the
    /// bar's own range (ADR 0013).
    fn within(side: Side, price: f64, bar: &Candle) -> f64 {
        match side {
            Side::Buy => price.min(bar.high),
            Side::Sell => price.max(bar.low),
        }
    }

    /// A taker's fill of `size` off `reference`: lifted on a buy and dropped on a
    /// sell by its costs on the window `trailing`, then held inside the bar, with
    /// what it paid.
    pub(crate) fn taker_fill(
        &self,
        side: Side,
        size: f64,
        reference: f64,
        bar: &Candle,
        trailing: impl Fn() -> Trailing,
    ) -> (Fill, FillCosts) {
        let (total, spread, slippage, impact) = self.taker_slip(size, trailing);
        let raw = match side {
            Side::Buy => reference * (1.0 + total),
            Side::Sell => reference * (1.0 - total),
        };
        let price = Self::within(side, raw, bar);
        let costs = FillCosts {
            reference,
            spread: reference * spread,
            slippage: reference * slippage,
            impact: reference * impact,
            adjustment: price - raw,
        };
        (taker(side, size, price), costs)
    }

    /// Fill a market order against `bar` (the bar after the decision). Priced off
    /// the bar open, lifted on a buy and dropped on a sell by its costs, sized at
    /// the remaining held to `available`. `None` if the bar gives no volume or the
    /// budget is spent.
    pub fn fill_market(
        &self,
        order: &Order,
        bar: &Candle,
        available: f64,
        trailing: impl Fn() -> Trailing,
    ) -> Option<(Fill, FillCosts)> {
        let size = self.cap(order, bar, available);
        if size <= 0.0 {
            return None;
        }
        Some(self.taker_fill(order.side, size, bar.open, bar, trailing))
    }

    /// Fill a resting limit against `bar`. A buy fills if the bar trades at or
    /// below its price, a sell if at or above, sized at the remaining held to
    /// `available`, and pays no slippage. It fills at its own price, or, where no
    /// trade from `close`, the last close, through the bar reached that price, at
    /// the highest of those trades for a buy and the lowest for a sell: a buy placed
    /// through the market fills where the market was, not where it was asked to.
    /// `None` if untouched, the bar gives no volume or the budget is spent.
    ///
    /// It pays the taker rate only on its first bar, when that bar opens at or
    /// through it, since it is then lifted on arrival; one that has rested through
    /// a bar is on the book and pays the maker rate whatever later bars do. The
    /// straddle a maker rebate would pay, a buy and a sell at one price both filled
    /// on a wide bar, is refused at placement instead (ADR 0011).
    pub fn fill_limit(
        &self,
        order: &Order,
        bar: &Candle,
        available: f64,
        close: f64,
    ) -> Option<(Fill, FillCosts)> {
        let limit = order.price?.get();
        let touched = match order.side {
            Side::Buy => bar.low <= limit,
            Side::Sell => bar.high >= limit,
        };
        if !touched {
            return None;
        }
        let size = self.cap(order, bar, available);
        if size <= 0.0 {
            return None;
        }
        let price = match order.side {
            Side::Buy => limit.min(close.max(bar.high)),
            Side::Sell => limit.max(close.min(bar.low)),
        };
        let fill = Fill {
            side: order.side,
            size: Qty(size),
            price: Price(price),
            is_taker: !order.on_book
                && self.limit_crosses(order.side, Price(limit), Price(bar.open)),
        };
        let costs = FillCosts {
            reference: price,
            ..FillCosts::default()
        };
        Some((fill, costs))
    }

    /// Trigger a stop against `bar`. A buy stop triggers when the bar reaches its
    /// trigger from below, a sell stop from above. Once triggered it fills as a
    /// taker at the worse of the trigger and the bar open (so a gap through the
    /// stop fills at the gap), then pays its costs on the window `trailing`, sized
    /// at the remaining held to `available`. `None` if not triggered, the bar gives
    /// no volume or the budget is spent.
    pub fn fill_stop(
        &self,
        order: &Order,
        bar: &Candle,
        available: f64,
        trailing: impl Fn() -> Trailing,
    ) -> Option<(Fill, FillCosts)> {
        let trigger = order.trigger?.get();
        let triggered = match order.side {
            Side::Buy => bar.high >= trigger,
            Side::Sell => bar.low <= trigger,
        };
        if !triggered {
            return None;
        }
        let size = self.cap(order, bar, available);
        if size <= 0.0 {
            return None;
        }
        let reference = match order.side {
            Side::Buy => bar.open.max(trigger),
            Side::Sell => bar.open.min(trigger),
        };
        Some(self.taker_fill(order.side, size, reference, bar, trailing))
    }

    /// Would a limit at `price` on `side` cross immediately against `reference`,
    /// making it a taker? A post_only order that would cross is rejected at
    /// placement rather than turned taker.
    pub fn limit_crosses(&self, side: Side, price: Price, reference: Price) -> bool {
        match side {
            Side::Buy => price.get() >= reference.get(),
            Side::Sell => price.get() <= reference.get(),
        }
    }
}

/// Build a taker fill on `side` of `size` at `price`. The caller has already
/// slipped the price and held it inside the bar; this only stamps `is_taker`.
fn taker(side: Side, size: f64, price: f64) -> Fill {
    Fill {
        side,
        size: Qty(size),
        price: Price(price),
        is_taker: true,
    }
}

#[cfg(test)]
mod tests {
    use super::{positive, FillCosts, FillModel, Impact, Spread};
    use crate::window::Trailing;
    use probatum_core::{Bps, Candle, Order, OrderId, Price, Qty, Side, TimeInForce};

    /// No budget spent yet: an order takes all it asks.
    const ALL: f64 = f64::INFINITY;

    #[test]
    fn positive_holds_for_a_positive_finite_number_alone() {
        for x in [5e-324, f64::MIN_POSITIVE, 1.0, f64::MAX] {
            assert!(positive(x), "{x}");
        }
        let refused = [
            0.0,
            -0.0,
            -5e-324,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
        ];
        for x in refused {
            assert!(!positive(x), "{x}");
        }
    }

    fn ohlc(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
        Candle {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// A bar whose extremes sit far from the open, for the market no-lookahead test.
    fn bar(open: f64, volume: f64) -> Candle {
        ohlc(open, open + 50.0, open - 50.0, open + 30.0, volume)
    }

    fn model(slip_bps: f64) -> FillModel {
        FillModel {
            slippage_bps: Bps(slip_bps),
            impact: 0.0,
            impact_model: Impact::Linear,
            spread: Spread::None,
        }
    }

    /// A model with impact `k` under `impact_model` and nothing else.
    fn impacted(k: f64, impact_model: Impact) -> FillModel {
        FillModel {
            impact: k,
            impact_model,
            ..model(0.0)
        }
    }

    /// A window that traded `volume` a bar with log returns of deviation `sigma`.
    fn window(volume: f64, sigma: f64) -> Trailing {
        Trailing {
            volume,
            sigma,
            spread: 0.0,
        }
    }

    fn quiet() -> Trailing {
        window(1000.0, 0.0)
    }

    fn market(side: Side, size: f64) -> Order {
        Order::market(OrderId(0), side, Qty(size), false)
    }

    fn limit(side: Side, size: f64, price: f64) -> Order {
        Order::limit(
            OrderId(0),
            side,
            Qty(size),
            Price(price),
            TimeInForce::Gtc,
            false,
            false,
        )
    }

    fn stop(side: Side, size: f64, trigger: f64) -> Order {
        Order::stop(OrderId(0), side, Qty(size), Price(trigger), false)
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // market

    #[test]
    fn buy_market_fills_at_open_plus_slippage() {
        let f = model(10.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, 1000.0), ALL, quiet)
            .unwrap()
            .0;
        assert!(approx(f.price.get(), 100.1));
        assert!(f.is_taker);
    }

    #[test]
    fn sell_market_fills_at_open_minus_slippage() {
        let f = model(10.0)
            .fill_market(&market(Side::Sell, 1.0), &bar(100.0, 1000.0), ALL, quiet)
            .unwrap()
            .0;
        assert!(approx(f.price.get(), 99.9));
    }

    #[test]
    fn market_uses_open_not_intrabar_extremes() {
        let b = bar(100.0, 1000.0); // high 150, low 50, close 130
        let buy = model(0.0)
            .fill_market(&market(Side::Buy, 1.0), &b, ALL, quiet)
            .unwrap()
            .0;
        let sell = model(0.0)
            .fill_market(&market(Side::Sell, 1.0), &b, ALL, quiet)
            .unwrap()
            .0;
        assert_eq!(buy.price.get(), 100.0);
        assert_eq!(sell.price.get(), 100.0);
    }

    #[test]
    fn the_budget_left_limits_a_market_order() {
        let f = model(0.0)
            .fill_market(&market(Side::Buy, 250.0), &bar(100.0, 1000.0), 100.0, quiet)
            .unwrap()
            .0;
        assert_eq!(f.size, Qty(100.0));
    }

    #[test]
    fn a_spent_budget_fills_nothing() {
        for available in [0.0, -1.0, f64::NAN] {
            let f = model(0.0).fill_market(
                &market(Side::Buy, 1.0),
                &bar(100.0, 1000.0),
                available,
                quiet,
            );
            assert!(f.is_none(), "{available} left filled");
        }
    }

    #[test]
    fn impact_grows_with_the_share_of_the_trailing_volume_and_its_volatility() {
        // a buy of 100 against a trailing volume of 1000 takes a share of 0.1, which
        // at k of 0.5 and a deviation of 0.02 moves it 0.1% linearly and 0.316% under
        // the square root
        let trailing = window(1000.0, 0.02);
        for (impact_model, moved) in [
            (Impact::Linear, 0.001),
            (Impact::Sqrt, 0.01 * 0.1f64.sqrt()),
        ] {
            let fill = impacted(0.5, impact_model)
                .fill_market(&market(Side::Buy, 100.0), &bar(100.0, 1000.0), ALL, || {
                    trailing
                })
                .unwrap()
                .0;
            assert!(
                approx(fill.price.get(), 100.0 * (1.0 + moved)),
                "{impact_model:?}"
            );
        }
    }

    #[test]
    fn a_taker_pays_half_the_spread_the_slippage_and_the_impact() {
        // the Specification's example: a buy of 0.141 BTC on a bar that opens at
        // 21,012.00, an estimated spread of 0.6 bps, a slippage of 1 bp and impact
        // under sqrt(0.1) with a deviation of 0.004 and a trailing volume of 24,800,
        // which add 0.63, 2.10 and 0.02 and fill it at 21,014.75
        let m = FillModel {
            slippage_bps: Bps(1.0),
            impact: 0.1,
            impact_model: Impact::Sqrt,
            spread: Spread::Estimated,
        };
        let trailing = Trailing {
            volume: 24_800.0,
            sigma: 0.004,
            spread: 0.6e-4,
        };
        let bar = ohlc(21_012.0, 21_030.0, 21_000.0, 21_020.0, 30_000.0);
        let (fill, costs) = m
            .fill_market(&market(Side::Buy, 0.141), &bar, ALL, || trailing)
            .unwrap();
        assert!((costs.spread - 0.63).abs() < 0.005, "{}", costs.spread);
        assert!((costs.slippage - 2.10).abs() < 0.005, "{}", costs.slippage);
        assert!((costs.impact - 0.02).abs() < 0.005, "{}", costs.impact);
        assert!(
            (fill.price.get() - 21_014.75).abs() < 0.005,
            "{}",
            fill.price.get()
        );
        // and exactly as the formula gives it, which the bar's own volume of 30,000
        // in place of the trailing one would miss by 0.0018
        let moved = 0.6e-4 / 2.0 + 1e-4 + 0.1 * 0.004 * (0.141f64 / 24_800.0).sqrt();
        assert!((fill.price.get() - 21_012.0 * (1.0 + moved)).abs() < 1e-9);
    }

    #[test]
    fn zero_volume_bar_fills_nothing() {
        assert!(model(0.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, 0.0), ALL, quiet)
            .is_none());
    }

    #[test]
    fn a_negative_volume_bar_fills_nothing() {
        assert!(model(0.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, -5.0), ALL, quiet)
            .is_none());
    }

    #[test]
    fn a_negative_slippage_cannot_improve_a_fill() {
        // slippage is adverse by definition (ADR 0013); a negative rate would fill
        // a buy below the open and mint equity out of the cost model
        let f = model(-20_000.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, 1000.0), ALL, quiet)
            .unwrap()
            .0;
        assert_eq!(f.price.get(), 100.0);
        let s = model(-20_000.0)
            .fill_market(&market(Side::Sell, 1.0), &bar(100.0, 1000.0), ALL, quiet)
            .unwrap()
            .0;
        assert_eq!(s.price.get(), 100.0);
    }

    #[test]
    fn a_sell_is_never_priced_at_or_below_zero() {
        // a huge flat slip or a huge impact would otherwise reach 1.0 and price the
        // fill at zero, or beyond it and price it negative, on a bar that traded down
        // to nothing (ADR 0013)
        let to_nothing = ohlc(100.0, 150.0, 0.0, 130.0, 1000.0);
        let flat = model(50_000.0)
            .fill_market(&market(Side::Sell, 1.0), &to_nothing, ALL, quiet)
            .unwrap()
            .0;
        assert!(flat.price.get() > 0.0);
        let f = impacted(5.0, Impact::Linear)
            .fill_market(&market(Side::Sell, 1000.0), &to_nothing, ALL, || {
                window(1000.0, 1.0)
            })
            .unwrap()
            .0;
        assert!(f.price.get() > 0.0, "priced at {}", f.price.get());
    }

    // limit

    #[test]
    fn buy_limit_fills_when_bar_dips_to_it() {
        let f = model(50.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(101.0, 102.0, 98.0, 101.0, 1000.0),
                ALL,
                101.0,
            )
            .unwrap()
            .0;
        assert_eq!(f.price.get(), 100.0); // at the limit
        assert!(!f.is_taker); // maker, no slippage despite slippage_bps
    }

    #[test]
    fn buy_limit_untouched_does_not_fill() {
        assert!(model(0.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(102.0, 103.0, 101.0, 102.0, 1000.0),
                ALL,
                102.0,
            )
            .is_none());
    }

    #[test]
    fn sell_limit_fills_when_bar_rises_to_it() {
        let f = model(0.0)
            .fill_limit(
                &limit(Side::Sell, 1.0, 100.0),
                &ohlc(99.0, 101.0, 98.0, 100.0, 1000.0),
                ALL,
                99.0,
            )
            .unwrap()
            .0;
        assert_eq!(f.price.get(), 100.0);
    }

    #[test]
    fn buy_limit_gap_through_still_fills_at_limit_never_better() {
        // limit 100 placed at a close of 101; the bar gaps down, opens 95 (low 90).
        // The close traded through the limit, so it fills at 100, not at 95
        let f = model(0.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(95.0, 96.0, 90.0, 95.0, 1000.0),
                ALL,
                101.0,
            )
            .unwrap()
            .0;
        assert_eq!(f.price.get(), 100.0);
        // and it is marketable at that open, so it takes rather than makes
        assert!(f.is_taker);
    }

    #[test]
    fn a_limit_placed_through_the_market_fills_where_the_market_traded() {
        // the Specification's example: a buy limit at 110 placed at a close of 100
        // meets a bar that opens at 96 with a high of 98, and since no trade from
        // that close through the bar reached 110 it fills at 100, the highest of
        // them; a sell at 90 on a bar from 104 to a low of 102 fills at 100 too
        let (bought, costs) = model(0.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 110.0),
                &ohlc(96.0, 98.0, 95.0, 97.0, 1000.0),
                ALL,
                100.0,
            )
            .unwrap();
        assert_eq!((bought.price.get(), costs.reference), (100.0, 100.0));
        assert!(bought.is_taker);
        let sold = model(0.0)
            .fill_limit(
                &limit(Side::Sell, 1.0, 90.0),
                &ohlc(104.0, 106.0, 102.0, 103.0, 1000.0),
                ALL,
                100.0,
            )
            .unwrap()
            .0;
        assert_eq!(sold.price.get(), 100.0);
    }

    #[test]
    fn a_limit_is_a_taker_only_on_a_first_bar_that_opens_through_it() {
        let m = model(0.0);
        // the bar opens away from the limit and has to come to it: a maker
        let made = m
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(101.0, 102.0, 98.0, 101.0, 1000.0),
                ALL,
                101.0,
            )
            .unwrap()
            .0;
        assert!(!made.is_taker);
        // its first bar opens at the limit, so it is lifted on arrival: a taker
        let bar = ohlc(100.0, 101.0, 99.0, 100.0, 1000.0);
        let bought = m
            .fill_limit(&limit(Side::Buy, 1.0, 100.0), &bar, ALL, 100.0)
            .unwrap()
            .0;
        let sold = m
            .fill_limit(&limit(Side::Sell, 1.0, 100.0), &bar, ALL, 100.0)
            .unwrap()
            .0;
        assert!(bought.is_taker);
        assert!(sold.is_taker);
        // the same bar after the limit has rested through one finds it on the book
        let rested = Order {
            on_book: true,
            ..limit(Side::Buy, 1.0, 100.0)
        };
        assert!(!m.fill_limit(&rested, &bar, ALL, 100.0).unwrap().0.is_taker);
    }

    #[test]
    fn the_budget_left_limits_a_limit() {
        let f = model(0.0)
            .fill_limit(
                &limit(Side::Buy, 250.0, 100.0),
                &ohlc(101.0, 102.0, 98.0, 101.0, 1000.0),
                100.0,
                101.0,
            )
            .unwrap()
            .0;
        assert_eq!(f.size, Qty(100.0));
    }

    // stop

    #[test]
    fn sell_stop_triggers_and_fills_at_trigger_with_slippage() {
        // sell stop 90; bar low 88 triggers, no gap (open 95). Fill at 90 - slip.
        let f = model(10.0)
            .fill_stop(
                &stop(Side::Sell, 1.0, 90.0),
                &ohlc(95.0, 96.0, 88.0, 92.0, 1000.0),
                ALL,
                quiet,
            )
            .unwrap()
            .0;
        assert!(approx(f.price.get(), 90.0 * 0.999));
        assert!(f.is_taker);
    }

    #[test]
    fn sell_stop_gap_fills_at_the_worse_open() {
        // sell stop 90; bar gaps down, opens 85 (worse). Fill at 85 - slip.
        let f = model(10.0)
            .fill_stop(
                &stop(Side::Sell, 1.0, 90.0),
                &ohlc(85.0, 86.0, 84.0, 85.0, 1000.0),
                ALL,
                quiet,
            )
            .unwrap()
            .0;
        assert!(approx(f.price.get(), 85.0 * 0.999));
    }

    #[test]
    fn buy_stop_triggers_at_trigger_with_slippage() {
        // buy stop 110; bar high 112 triggers, no gap (open 105). Fill at 110 + slip.
        let f = model(10.0)
            .fill_stop(
                &stop(Side::Buy, 1.0, 110.0),
                &ohlc(105.0, 112.0, 104.0, 111.0, 1000.0),
                ALL,
                quiet,
            )
            .unwrap()
            .0;
        assert!(approx(f.price.get(), 110.0 * 1.001));
    }

    #[test]
    fn buy_stop_gap_fills_at_the_worse_open() {
        // buy stop 110; bar gaps up, opens 115. Fill at 115 + slip.
        let f = model(10.0)
            .fill_stop(
                &stop(Side::Buy, 1.0, 110.0),
                &ohlc(115.0, 116.0, 114.0, 115.0, 1000.0),
                ALL,
                quiet,
            )
            .unwrap()
            .0;
        assert!(approx(f.price.get(), 115.0 * 1.001));
    }

    #[test]
    fn untriggered_stop_does_not_fill() {
        assert!(model(0.0)
            .fill_stop(
                &stop(Side::Sell, 1.0, 90.0),
                &ohlc(95.0, 96.0, 92.0, 94.0, 1000.0),
                ALL,
                quiet,
            )
            .is_none());
    }

    // post_only cross check

    #[test]
    fn limit_crosses_detects_marketable_orders() {
        let m = model(0.0);
        assert!(m.limit_crosses(Side::Buy, Price(105.0), Price(100.0))); // buy above market
        assert!(!m.limit_crosses(Side::Buy, Price(95.0), Price(100.0))); // buy below: maker
        assert!(m.limit_crosses(Side::Sell, Price(95.0), Price(100.0))); // sell below market
        assert!(!m.limit_crosses(Side::Sell, Price(105.0), Price(100.0))); // sell above: maker
    }

    // a taker price never leaves the bar that produced it (ADR 0013)

    #[test]
    fn impact_cannot_price_a_market_fill_outside_the_bar() {
        // impact grows with the share of the trailing volume taken, and the total is
        // bounded only by MAX_SLIP, which is a rule about not reaching zero. Taking
        // the whole window at 0.5 asked a buy 50% above anything that traded, and
        // the loss booked against it was bounded by nothing
        let m = impacted(0.5, Impact::Linear);
        let bar = ohlc(100.0, 101.0, 99.0, 100.0, 10.0);
        let trailing = window(10.0, 1.0);
        let bought = m
            .fill_market(&market(Side::Buy, 10.0), &bar, ALL, || trailing)
            .expect("fills")
            .0;
        assert_eq!(bought.price.get(), 101.0);
        let sold = m
            .fill_market(&market(Side::Sell, 10.0), &bar, ALL, || trailing)
            .expect("fills")
            .0;
        assert_eq!(sold.price.get(), 99.0);
    }

    #[test]
    fn a_slipped_stop_stays_inside_its_bar() {
        let m = model(5_000.0); // 50%, far past the bar
        let bar = ohlc(100.0, 101.0, 90.0, 95.0, 1000.0);
        let fill = m
            .fill_stop(&stop(Side::Sell, 1.0, 97.0), &bar, ALL, quiet)
            .expect("triggered")
            .0;
        assert_eq!(fill.price.get(), 90.0);
    }

    #[test]
    fn an_ordinary_slip_is_untouched_by_the_bound() {
        // the bound is a bound, not a behaviour change: a normal slip off the open
        // lands well inside the bar and comes through exactly as before
        let bar = ohlc(100.0, 105.0, 95.0, 102.0, 1000.0);
        let fill = model(10.0)
            .fill_market(&market(Side::Buy, 1.0), &bar, ALL, quiet)
            .expect("fills")
            .0;
        assert!(
            (fill.price.get() - 100.1).abs() < 1e-12,
            "{}",
            fill.price.get()
        );
    }

    // what a fill paid beyond its reference (ADR 0022)

    #[test]
    fn a_market_fill_is_measured_from_the_open_and_pays_its_slippage_beyond_it() {
        let (fill, costs) = model(10.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, 1000.0), ALL, quiet)
            .unwrap();
        assert_eq!(costs.reference, 100.0);
        assert!(approx(costs.slippage, 0.1));
        assert_eq!(costs.adjustment, 0.0);
        assert!(approx(fill.price.get(), costs.reference + costs.slippage));
    }

    #[test]
    fn a_stop_is_measured_from_the_worse_of_the_open_and_its_trigger() {
        let m = model(10.0);
        let stopped = stop(Side::Sell, 1.0, 90.0);
        let (_, reached) = m
            .fill_stop(&stopped, &ohlc(95.0, 96.0, 88.0, 92.0, 1000.0), ALL, quiet)
            .unwrap();
        assert_eq!(reached.reference, 90.0);
        let (_, gapped) = m
            .fill_stop(&stopped, &ohlc(85.0, 86.0, 84.0, 85.0, 1000.0), ALL, quiet)
            .unwrap();
        assert_eq!(gapped.reference, 85.0);
    }

    #[test]
    fn a_limit_is_measured_from_its_own_price_and_pays_nothing_beyond_it() {
        let (_, costs) = model(50.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(101.0, 102.0, 98.0, 101.0, 1000.0),
                ALL,
                101.0,
            )
            .unwrap();
        let measured = FillCosts {
            reference: 100.0,
            ..FillCosts::default()
        };
        assert_eq!(costs, measured);
    }

    #[test]
    fn a_price_the_bar_held_back_records_how_far() {
        // impact asks 50% over the open and the bar's high of 101 holds the buy, so
        // the costs say what it was asked and the adjustment what the bar took back
        let (fill, costs) = impacted(0.5, Impact::Linear)
            .fill_market(
                &market(Side::Buy, 10.0),
                &ohlc(100.0, 101.0, 99.0, 100.0, 10.0),
                ALL,
                || window(10.0, 1.0),
            )
            .unwrap();
        assert_eq!(fill.price.get(), 101.0);
        assert!(approx(costs.impact, 50.0));
        assert!(approx(costs.adjustment, -49.0));
    }

    #[test]
    fn a_slip_held_below_one_keeps_each_part_its_share() {
        // a flat 60% and an impact of 60% come to 120%, held to 99%
        let m = FillModel {
            slippage_bps: Bps(6_000.0),
            ..impacted(0.6, Impact::Linear)
        };
        let (_, costs) = m
            .fill_market(
                &market(Side::Sell, 1000.0),
                &bar(100.0, 1000.0),
                ALL,
                || window(1000.0, 1.0),
            )
            .unwrap();
        assert!(approx(costs.slippage + costs.impact, 99.0));
        assert!(approx(costs.slippage, costs.impact));
    }
}
