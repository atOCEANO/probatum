//! Inverse contracts. An inverse perpetual is quoted in dollars and settled in its
//! coin, and everything its account counts is linear in the reciprocal of the price
//! (ADR 0016), so the engine runs it as a linear perpetual on that reciprocal: its
//! bars are `1 / p`, the high and the low trading places, and its position is the
//! dollars its contracts are worth, held the other way round, since a long gains as
//! the reciprocal falls, as each funding rate is charged the other way round. Its
//! equity, fees, funding, a taker's costs, its requirement, path and margin bound
//! then come out in the coin with no formula of their own. `Units` carries a price,
//! a size, a side or a record between the venue's terms and the engine's.

use probatum_core::{Candle, Order, Price, Qty, Side, State, Tier, TierBasis, Tiers};

use crate::execution::FillCosts;
use crate::reporter::{EventReason, FillRecord, LedgerRow, Leg, OrderEvent, Refusal, RoundTrip};
use crate::script::Action;

/// How an account's prices and sizes meet the engine's: as they are for spot and
/// linear contracts, and for an inverse contract each price's reciprocal, each size
/// in the dollars its contracts are worth and each side the other way round.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Units {
    #[default]
    Linear,
    /// An inverse contract worth this many dollars.
    Inverse(f64),
}

impl Units {
    /// A price in the other's terms, either way: its reciprocal on an inverse
    /// contract, with a zero, which a record holds for no price, kept zero.
    pub fn price(self, price: f64) -> f64 {
        match self {
            Units::Inverse(_) if price != 0.0 => 1.0 / price,
            _ => price,
        }
    }

    /// A size of contracts in the engine's terms.
    pub fn size(self, contracts: f64) -> f64 {
        match self {
            Units::Inverse(value) => contracts * value,
            Units::Linear => contracts,
        }
    }

    /// A size in the engine's terms, in contracts.
    pub fn contracts(self, size: f64) -> f64 {
        match self {
            Units::Inverse(value) => size / value,
            Units::Linear => size,
        }
    }

    /// A side in the other's terms, either way.
    pub fn side(self, side: Side) -> Side {
        match self {
            Units::Inverse(_) => side.opposite(),
            Units::Linear => side,
        }
    }

    /// Anything a side signs, a position, a weight or a funding rate, in the other's
    /// terms, either way.
    pub fn signed(self, value: f64) -> f64 {
        match self {
            Units::Inverse(_) => -value,
            Units::Linear => value,
        }
    }

    /// A bar as the venue printed it, in the engine's terms: each price's
    /// reciprocal, the low's for the high and the high's for the low, and its volume
    /// in the dollars its contracts are worth. A row the instrument printed no bar on
    /// stays five NaNs.
    pub fn bar(self, bar: Candle) -> Candle {
        match self {
            Units::Inverse(value) => Candle {
                open: 1.0 / bar.open,
                high: 1.0 / bar.low,
                low: 1.0 / bar.high,
                close: 1.0 / bar.close,
                volume: bar.volume * value,
            },
            Units::Linear => bar,
        }
    }

    /// A bar in the engine's terms, as the venue printed it.
    pub fn shown(self, bar: Candle) -> Candle {
        match self {
            Units::Inverse(value) => Candle {
                open: 1.0 / bar.open,
                high: 1.0 / bar.low,
                low: 1.0 / bar.high,
                close: 1.0 / bar.close,
                volume: bar.volume / value,
            },
            Units::Linear => bar,
        }
    }

    /// A tier table in the engine's terms: a quantity table's bounds in the dollars
    /// its contracts are worth.
    pub fn tiers(self, tiers: Tiers) -> Tiers {
        if self == Units::Linear || tiers.basis() != TierBasis::Quantity {
            return tiers;
        }
        let rows = tiers
            .rows()
            .iter()
            .map(|tier| Tier {
                up_to: self.size(tier.up_to),
                ..*tier
            })
            .collect();
        Tiers::new(TierBasis::Quantity, rows).expect("bounds in dollars keep their order")
    }

    /// A decision of a script in the engine's terms.
    pub fn action(self, action: Action) -> Action {
        let price = |price: Option<f64>| price.map(|price| self.price(price));
        match action {
            Action::Order {
                instrument,
                side,
                size,
                kind,
                price: limit,
                trigger,
                reduce_only,
                post_only,
                tif,
            } => Action::Order {
                instrument,
                side: self.side(side),
                size: self.size(size),
                kind,
                price: price(limit),
                trigger: price(trigger),
                reduce_only,
                post_only,
                tif,
            },
            Action::Bracket {
                instrument,
                stop,
                take_profit,
            } => Action::Bracket {
                instrument,
                stop: price(stop),
                take_profit: price(take_profit),
            },
            Action::Replace {
                order,
                size,
                price: limit,
                trigger,
            } => Action::Replace {
                order,
                size: size.map(|size| self.size(size)),
                price: price(limit),
                trigger: price(trigger),
            },
            Action::Target {
                weights,
                tolerance,
                transfer,
            } => Action::Target {
                weights: weights
                    .into_iter()
                    .map(|(i, weight)| (i, self.signed(weight)))
                    .collect(),
                tolerance,
                transfer,
            },
            other => other,
        }
    }

    /// Equity in the coin, `equity`, valued at the engine's mark `mark`, in
    /// dollars: as it is, unless the account is inverse.
    pub fn in_quote(self, equity: f64, mark: f64) -> f64 {
        match self {
            Units::Inverse(_) => equity / mark,
            Units::Linear => equity,
        }
    }

    /// A state in the venue's terms.
    pub fn state(self, state: State) -> State {
        if self == Units::Linear {
            return state;
        }
        let bar = self.shown(Candle {
            open: state.bar_open,
            high: state.bar_high,
            low: state.bar_low,
            close: state.bar_close,
            volume: state.bar_volume,
        });
        State {
            position: self.contracts(self.signed(state.position)),
            avg_entry: self.price(state.avg_entry),
            mark_price: self.price(state.mark_price),
            bar_open: bar.open,
            bar_high: bar.high,
            bar_low: bar.low,
            bar_close: bar.close,
            bar_volume: bar.volume,
            open_orders: state
                .open_orders
                .into_iter()
                .map(|order| self.order(order))
                .collect(),
            ..state
        }
    }

    /// An order in the venue's terms.
    pub fn order(self, order: Order) -> Order {
        if self == Units::Linear {
            return order;
        }
        Order {
            side: self.side(order.side),
            price: order.price.map(|price| Price(self.price(price.get()))),
            trigger: order
                .trigger
                .map(|trigger| Price(self.price(trigger.get()))),
            size: Qty(self.contracts(order.size.get())),
            filled: Qty(self.contracts(order.filled.get())),
            ..order
        }
    }

    /// A fill in the venue's terms. Its spread, slippage and impact are each its
    /// share of what the price moved beyond its reference before the bar held it, in
    /// dollars, as they are on a linear contract.
    pub fn fill(self, fill: FillRecord) -> FillRecord {
        if self == Units::Linear {
            return fill;
        }
        let costs = fill.costs;
        let (price, reference) = (self.price(fill.price), self.price(costs.reference));
        let raw = self.price(fill.price - costs.adjustment);
        let total = costs.spread + costs.slippage + costs.impact;
        let share = |cost: f64| {
            if total > 0.0 {
                (raw - reference).abs() * cost / total
            } else {
                0.0
            }
        };
        FillRecord {
            side: self.side(fill.side),
            size: self.contracts(fill.size),
            price,
            costs: FillCosts {
                reference,
                spread: share(costs.spread),
                slippage: share(costs.slippage),
                impact: share(costs.impact),
                adjustment: price - raw,
            },
            ..fill
        }
    }

    /// An order's event in the venue's terms.
    pub fn event(self, event: OrderEvent) -> OrderEvent {
        if self == Units::Linear {
            return event;
        }
        let refusal = |why: Refusal| match why {
            Refusal::Size(size) => Refusal::Size(self.contracts(size)),
            Refusal::Price(price) => Refusal::Price(self.price(price)),
            Refusal::Trigger(trigger) => Refusal::Trigger(self.price(trigger)),
            other => other,
        };
        OrderEvent {
            size: event.size.map(|size| self.contracts(size)),
            price: event.price.map(|price| self.price(price)),
            reason: match event.reason {
                EventReason::Fits { asked } => EventReason::Fits {
                    asked: self.contracts(asked),
                },
                EventReason::Refused(why) => EventReason::Refused(refusal(why)),
                EventReason::Replacing(why) => EventReason::Replacing(refusal(why)),
                other => other,
            },
            ..event
        }
    }

    /// A round trip in the venue's terms, its excursions the dollar price's moves
    /// from its entry. Its entry and its exit, averaged over its fills in the
    /// reciprocal, are the harmonic means of their dollar prices weighted by
    /// contracts, which keeps its profit the sum of its fills' own.
    pub fn trip(self, trip: RoundTrip) -> RoundTrip {
        if self == Units::Linear {
            return trip;
        }
        let mut out = RoundTrip {
            side: self.side(trip.side),
            size: self.contracts(trip.size),
            entry: self.price(trip.entry),
            exit: self.price(trip.exit),
            opened: self.contracts(trip.opened),
            closed: self.contracts(trip.closed),
            high: self.price(trip.low),
            low: self.price(trip.high),
            ..trip
        };
        out.excursions();
        out
    }

    /// A ledger row in the venue's terms.
    pub fn row(self, row: LedgerRow) -> LedgerRow {
        LedgerRow {
            net_exposure: self.signed(row.net_exposure),
            ..row
        }
    }

    /// An instrument's line of a ledger row in the venue's terms.
    pub fn leg(self, leg: Leg) -> Leg {
        if self == Units::Linear {
            return leg;
        }
        Leg {
            position: self.contracts(self.signed(leg.position)),
            entry: self.price(leg.entry),
            mark: self.price(leg.mark),
            weight: self.signed(leg.weight),
            drift: self.signed(leg.drift),
            ..leg
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Units;
    use crate::reporter::RoundTrip;
    use probatum_core::{Candle, Side};

    const INVERSE: Units = Units::Inverse(100.0);

    #[test]
    fn a_price_a_side_and_a_sign_come_back_as_they_went() {
        for price in [60_000.0, 0.5, 0.0] {
            assert_eq!(INVERSE.price(INVERSE.price(price)), price);
        }
        assert_eq!(INVERSE.side(Side::Buy), Side::Sell);
        assert_eq!(
            INVERSE.contracts(INVERSE.signed(INVERSE.signed(INVERSE.size(600.0)))),
            600.0
        );
        assert_eq!(Units::Linear.price(60_000.0), 60_000.0);
    }

    #[test]
    fn a_bar_in_the_reciprocal_trades_its_high_and_its_low() {
        let bar = Candle {
            open: 50.0,
            high: 80.0,
            low: 40.0,
            close: 64.0,
            volume: 3.0,
        };
        let engine = INVERSE.bar(bar);
        assert_eq!((engine.high, engine.low), (1.0 / 40.0, 1.0 / 80.0));
        assert_eq!(engine.volume, 300.0);
        assert_eq!(INVERSE.shown(engine), bar);
        let missing = Candle {
            open: f64::NAN,
            high: f64::NAN,
            low: f64::NAN,
            close: f64::NAN,
            volume: f64::NAN,
        };
        assert!(INVERSE.bar(missing).close.is_nan());
    }

    #[test]
    fn a_round_trip_averaged_in_the_reciprocal_enters_at_the_harmonic_mean() {
        // the Specification's example: 300 contracts bought at 60,000 and 300 at
        // 40,000 enter at 48,000, a sale in the reciprocal of 30,000 dollars at each
        let mut trip = RoundTrip::open(0, Side::Sell, 0, 30_000.0, 1.0 / 60_000.0, 0.0);
        trip.grow(30_000.0, 1.0 / 40_000.0, 0.0);
        trip.see(1.0 / 50_000.0);
        let out = INVERSE.trip(trip);
        assert_eq!(out.side, Side::Buy);
        assert_eq!(out.opened, 600.0);
        assert!(
            (out.entry - 48_000.0).abs() < 1e-9,
            "entered at {}",
            out.entry
        );
        assert!((out.high - 60_000.0).abs() < 1e-9 && (out.low - 40_000.0).abs() < 1e-9);
        assert!((out.best - 0.25).abs() < 1e-12 && (out.worst + 1.0 / 6.0).abs() < 1e-12);
    }
}
