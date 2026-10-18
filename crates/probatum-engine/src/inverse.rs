//! Inverse contracts. An inverse perpetual is quoted in dollars and settled in its
//! coin, and everything its account counts is linear in the reciprocal of the price
//! (ADR 0016), so the engine runs it as a linear perpetual on that reciprocal: its
//! bars are `1 / p`, the high and the low trading places, and its position is the
//! dollars its contracts are worth, held the other way round, since a long gains as
//! the reciprocal falls, as each funding rate is charged the other way round. Its
//! equity, fees, funding, a taker's costs, its requirement, path and margin bound
//! then come out in the coin with no formula of their own. `Units` carries a price,
//! a size, a side or a record between the venue's terms and the engine's.

use probatum_core::{
    Candle, Order, OrderType, Price, Qty, Side, State, Tier, TierBasis, TierError, Tiers,
};

use crate::execution::{positive, FillCosts};
use crate::reporter::{EventReason, FillRecord, LedgerRow, Leg, OrderEvent, Refusal, RoundTrip};
use crate::script::Action;
use crate::sizes::{PriceRule, Sizes};

/// Of the floats whose reciprocal is `price`, the one written in the fewest digits,
/// which is the price quoted wherever one was, since a reciprocal taken twice can
/// land up to two floats away from it.
pub(crate) fn quoted(price: f64) -> f64 {
    let quoted = 1.0 / price;
    let (below, above) = (quoted.next_down(), quoted.next_up());
    [quoted, below, above, below.next_down(), above.next_up()]
        .into_iter()
        .filter(|&candidate| 1.0 / candidate == price)
        .min_by_key(|candidate| candidate.to_string().len())
        .unwrap_or(quoted)
}

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
    /// Whether the account can trade at `price`: any price on a linear contract,
    /// which the engine judges itself, and on an inverse contract one whose
    /// reciprocal is a price, which none at or below zero has, nor one so small its
    /// reciprocal is past every float.
    pub fn takes(self, price: f64) -> bool {
        match self {
            Units::Inverse(_) => {
                let reciprocal = 1.0 / price;
                reciprocal > 0.0 && reciprocal.is_finite()
            }
            Units::Linear => true,
        }
    }

    /// A price the venue quoted, in the engine's terms: its reciprocal on an inverse
    /// contract.
    pub fn price(self, price: f64) -> f64 {
        match self {
            Units::Inverse(_) => 1.0 / price,
            Units::Linear => price,
        }
    }

    /// A price in the engine's terms as the venue quotes it: on an inverse contract
    /// the price `quoted` reads off its reciprocal. A zero, which a record holds for
    /// no price, stays zero.
    pub fn dollars(self, price: f64) -> f64 {
        if self == Units::Linear || price == 0.0 {
            return price;
        }
        quoted(price)
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
    /// terms, either way; a zero stays a positive zero, so a flat position reads as
    /// one.
    pub fn signed(self, value: f64) -> f64 {
        match self {
            Units::Inverse(_) => 0.0 - value,
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

    /// The venue's size and price rules in the engine's terms: on an inverse
    /// contract its lot and largest size in the dollars its contracts are worth and
    /// its tick on the reciprocal of the price.
    pub fn sizes(self, sizes: Sizes) -> Sizes {
        match self {
            Units::Inverse(_) => Sizes {
                lot: self.size(sizes.lot),
                max_qty: self.size(sizes.max_qty),
                prices: match sizes.prices {
                    PriceRule::Tick(tick) => PriceRule::Reciprocal(tick),
                    other => other,
                },
                ..sizes
            },
            Units::Linear => sizes,
        }
    }

    /// A tier table in the engine's terms: a quantity table's bounds in the dollars
    /// its contracts are worth, refused where a bound is past every float there or
    /// no longer above the one before it. Only a bound given open stays open.
    pub fn tiers(self, tiers: Tiers) -> Result<Tiers, TierError> {
        if self == Units::Linear || tiers.basis() != TierBasis::Quantity {
            return Ok(tiers);
        }
        let rows: Vec<Tier> = tiers
            .rows()
            .iter()
            .map(|tier| Tier {
                up_to: self.size(tier.up_to),
                ..*tier
            })
            .collect();
        let past = tiers
            .rows()
            .iter()
            .zip(&rows)
            .position(|(given, held)| given.up_to.is_finite() && !held.up_to.is_finite());
        if let Some(row) = past {
            return Err(TierError::NotFinite {
                row,
                field: "up_to",
                value: rows[row].up_to,
            });
        }
        Tiers::new(TierBasis::Quantity, rows)
    }

    /// A decision of a script in the engine's terms, its prices ones the account
    /// `takes`.
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
            // an inverse contract's amount, in dollars, buys the contracts it is worth,
            // a size the order has as it is placed, and one that is no amount is
            // refused as one, its side in the engine's terms as every order's is
            Action::Value {
                instrument,
                side,
                amount,
                reduce_only,
                tif,
            } => match self {
                Units::Inverse(value) if positive(amount) => Action::Order {
                    instrument,
                    side: self.side(side),
                    size: self.size(amount / value),
                    kind: OrderType::Market,
                    price: None,
                    trigger: None,
                    reduce_only,
                    post_only: false,
                    tif,
                },
                _ => Action::Value {
                    instrument,
                    side: self.side(side),
                    amount,
                    reduce_only,
                    tif,
                },
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

    /// Equity in the coin, `equity`, valued in dollars at `close`, the close the
    /// venue printed: as it is, unless the account is inverse.
    pub fn in_quote(self, equity: f64, close: f64) -> f64 {
        match self {
            Units::Inverse(_) => equity * close,
            Units::Linear => equity,
        }
    }

    /// A state in the venue's terms, its bar and its mark read off `printed`, the
    /// bars as the venue printed them, of the one instrument an inverse account
    /// holds.
    pub fn state(self, state: State, printed: &[Candle]) -> State {
        if self == Units::Linear {
            return state;
        }
        let bar = printed[state.tick_index];
        State {
            position: self.contracts(self.signed(state.position)),
            avg_entry: self.dollars(state.avg_entry),
            mark_price: bar.close,
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
            price: order.price.map(|price| Price(self.dollars(price.get()))),
            trigger: order
                .trigger
                .map(|trigger| Price(self.dollars(trigger.get()))),
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
        let (price, reference) = (self.dollars(fill.price), self.dollars(costs.reference));
        let raw = self.dollars(fill.price - costs.adjustment);
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
            Refusal::Price(price) => Refusal::Price(self.dollars(price)),
            Refusal::Trigger(trigger) => Refusal::Trigger(self.dollars(trigger)),
            Refusal::Lot { size, lot } => Refusal::Lot {
                size: self.contracts(size),
                lot: self.contracts(lot),
            },
            Refusal::MinNotional { size, least } => Refusal::MinNotional {
                size: self.contracts(size),
                least,
            },
            other => other,
        };
        OrderEvent {
            side: event.side.map(|side| self.side(side)),
            size: event.size.map(|size| self.contracts(size)),
            price: event.price.map(|price| self.dollars(price)),
            reason: match event.reason {
                EventReason::Fits { asked, by } => EventReason::Fits {
                    asked: self.contracts(asked),
                    by,
                },
                EventReason::Fok { by, size, asked } => EventReason::Fok {
                    by,
                    size: self.contracts(size),
                    asked: self.contracts(asked),
                },
                EventReason::HeldTo { asked } => EventReason::HeldTo {
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
            entry: self.dollars(trip.entry),
            exit: self.dollars(trip.exit),
            opened: self.contracts(trip.opened),
            closed: self.contracts(trip.closed),
            high: self.dollars(trip.low),
            low: self.dollars(trip.high),
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

    /// An instrument's line of a ledger row in the venue's terms, marked at `close`,
    /// the close the venue printed on the row, where an inverse account holds the
    /// one instrument that always prints.
    pub fn leg(self, leg: Leg, close: f64) -> Leg {
        if self == Units::Linear {
            return leg;
        }
        Leg {
            position: self.contracts(self.signed(leg.position)),
            entry: self.dollars(leg.entry),
            mark: close,
            weight: self.signed(leg.weight),
            drift: self.signed(leg.drift),
            ..leg
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Units;
    use crate::reporter::{Cause, EventKind, EventReason, OrderEvent, RoundTrip};
    use probatum_core::{Candle, OrderId, Side, Tier, TierBasis, Tiers};

    const INVERSE: Units = Units::Inverse(100.0);

    #[test]
    fn a_quoted_price_a_side_and_a_sign_come_back_as_they_went() {
        // one price in six comes back a float away from a reciprocal taken twice,
        // 49 among them, and the fewest digits find the one quoted again
        for price in [60_000.0, 0.5, 49.0, 56.31, 40_201.01] {
            assert_eq!(INVERSE.dollars(INVERSE.price(price)), price);
        }
        assert_eq!(INVERSE.dollars(0.0), 0.0);
        assert_eq!(INVERSE.side(Side::Buy), Side::Sell);
        assert_eq!(
            INVERSE.contracts(INVERSE.signed(INVERSE.signed(INVERSE.size(600.0)))),
            600.0
        );
        assert!(INVERSE.signed(0.0).is_sign_positive());
        assert_eq!(Units::Linear.dollars(60_000.0), 60_000.0);
    }

    #[test]
    fn an_inverse_contract_takes_no_price_without_a_reciprocal() {
        for price in [0.0, -1.0, 1e-310] {
            assert!(!INVERSE.takes(price), "{price}");
        }
        assert!(INVERSE.takes(1e-300) && Units::Linear.takes(0.0));
    }

    #[test]
    fn a_quantity_table_counts_its_bounds_in_dollars_and_keeps_an_open_one_open() {
        let row = |up_to, leverage, rate| Tier {
            up_to,
            leverage,
            rate,
            amount: 0.0,
        };
        let given = Tiers::new(
            TierBasis::Quantity,
            vec![row(10.0, 50.0, 0.01), row(f64::INFINITY, 20.0, 0.02)],
        )
        .unwrap();
        let held = INVERSE.tiers(given).unwrap();
        let bounds: Vec<f64> = held.rows().iter().map(|tier| tier.up_to).collect();
        assert_eq!(bounds, [1_000.0, f64::INFINITY]);
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
        assert_eq!((out.high, out.low), (60_000.0, 40_000.0));
        assert!((out.best - 0.25).abs() < 1e-12 && (out.worst + 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn an_event_in_the_venues_terms_takes_its_side_and_counts_its_sizes_in_contracts() {
        // the engine trades an inverse contract in the reciprocal, its side the other
        // way round and its sizes in dollars, so a fok buy the cap holds to 120,000 of
        // the 200,000 asked is at the venue a sale held to 1,200 of 2,000 contracts
        let event = OrderEvent {
            tick: 1,
            kind: EventKind::NotFilled,
            order: Some(OrderId(1)),
            instrument: Some(0),
            side: Some(Side::Buy),
            size: None,
            price: None,
            reason: EventReason::Fok {
                by: Cause::Cap,
                size: 120_000.0,
                asked: 200_000.0,
            },
        };
        let shown = INVERSE.event(event);
        assert_eq!(shown.side, Some(Side::Sell));
        let fok = EventReason::Fok {
            by: Cause::Cap,
            size: 1_200.0,
            asked: 2_000.0,
        };
        assert_eq!(shown.reason, fok);
    }
}
