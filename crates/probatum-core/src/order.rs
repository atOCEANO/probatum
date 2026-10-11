//! Order identity and the order record. Price and trigger are optional because a
//! market order needs neither and a stop needs only a trigger.

use crate::enums::{OrderType, Side, TimeInForce};
use crate::units::{Price, Qty};

/// An opaque handle to a resting order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrderId(pub u64);

/// A single order: its identity, terms, and fill progress.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Order {
    pub id: OrderId,
    /// The instrument it trades, by its place in the universe.
    pub instrument: usize,
    pub side: Side,
    pub kind: OrderType,
    /// Limit price; set for a limit order, `None` otherwise.
    pub price: Option<Price>,
    /// Stop trigger price; set for a stop order, `None` otherwise.
    pub trigger: Option<Price>,
    /// Original size, a positive base quantity, or infinite on a bracket's leg
    /// while it closes the position it finds.
    pub size: Qty,
    /// Filled so far, from zero up to `size`.
    pub filled: Qty,
    /// Can only shrink the position (take-profit or stop-loss).
    pub reduce_only: bool,
    /// A limit that would cross is refused at placement, and cancelled when its
    /// first bar opens at or through it, rather than turned taker (ADR 0010).
    pub post_only: bool,
    pub tif: TimeInForce,
    /// Whether it has rested through a bar its instrument printed, after which a
    /// limit is on the book and fills as a maker (ADR 0011).
    pub on_book: bool,
    /// The bracket leg it is, or `None`.
    pub leg: Option<Leg>,
}

/// A bracket's leg: a reduce-only exit placed beside another as one OCO pair
/// (ADR 0010).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leg {
    /// The other leg of its pair, which its first fill cancels.
    pub pair: Option<OrderId>,
    /// Whether it has taken its side from a position. A leg placed while flat waits
    /// for the entry to fill, and keeps the side it arms on.
    pub armed: bool,
    /// Whether it closes the position it finds, as it does until a replace gives it
    /// a size of its own.
    pub sizeless: bool,
}

impl Order {
    /// A market order: fills at the next bar, no resting price.
    pub fn market(id: OrderId, side: Side, size: Qty, reduce_only: bool) -> Order {
        Order {
            id,
            instrument: 0,
            side,
            kind: OrderType::Market,
            price: None,
            trigger: None,
            size,
            filled: Qty(0.0),
            reduce_only,
            post_only: false,
            tif: TimeInForce::Ioc,
            on_book: false,
            leg: None,
        }
    }

    /// A limit order resting at `price`.
    pub fn limit(
        id: OrderId,
        side: Side,
        size: Qty,
        price: Price,
        tif: TimeInForce,
        post_only: bool,
        reduce_only: bool,
    ) -> Order {
        Order {
            id,
            instrument: 0,
            side,
            kind: OrderType::Limit,
            price: Some(price),
            trigger: None,
            size,
            filled: Qty(0.0),
            reduce_only,
            post_only,
            tif,
            on_book: false,
            leg: None,
        }
    }

    /// A stop order that becomes a market order once `trigger` is crossed.
    pub fn stop(id: OrderId, side: Side, size: Qty, trigger: Price, reduce_only: bool) -> Order {
        Order {
            id,
            instrument: 0,
            side,
            kind: OrderType::Stop,
            price: None,
            trigger: Some(trigger),
            size,
            filled: Qty(0.0),
            reduce_only,
            post_only: false,
            tif: TimeInForce::Gtc,
            on_book: false,
            leg: None,
        }
    }

    /// Size not yet filled, in base units. A partial fill leaves the order resting
    /// on this remainder rather than retiring it (ADR 0010).
    #[inline]
    pub fn remaining(&self) -> Qty {
        self.size - self.filled
    }

    /// The same order on `instrument`, which every constructor leaves at 0.
    pub fn on(mut self, instrument: usize) -> Order {
        self.instrument = instrument;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Order, OrderId};
    use crate::enums::{OrderType, Side, TimeInForce};
    use crate::units::{Price, Qty};

    #[test]
    fn market_has_no_price_or_trigger() {
        let o = Order::market(OrderId(1), Side::Buy, Qty(2.0), false);
        assert_eq!(o.kind, OrderType::Market);
        assert!(o.price.is_none());
        assert!(o.trigger.is_none());
        assert_eq!(o.remaining(), Qty(2.0));
    }

    #[test]
    fn limit_rests_at_price() {
        let o = Order::limit(
            OrderId(2),
            Side::Sell,
            Qty(1.0),
            Price(100.0),
            TimeInForce::Gtc,
            true,
            false,
        );
        assert_eq!(o.kind, OrderType::Limit);
        assert_eq!(o.price, Some(Price(100.0)));
        assert!(o.post_only);
    }

    #[test]
    fn stop_carries_a_trigger() {
        let o = Order::stop(OrderId(3), Side::Sell, Qty(1.0), Price(90.0), true);
        assert_eq!(o.kind, OrderType::Stop);
        assert_eq!(o.trigger, Some(Price(90.0)));
        assert!(o.reduce_only);
    }

    #[test]
    fn remaining_after_partial_fill() {
        let mut o = Order::market(OrderId(4), Side::Buy, Qty(5.0), false);
        o.filled = Qty(2.0);
        assert_eq!(o.remaining(), Qty(3.0));
    }
}
