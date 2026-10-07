//! Optional reporting buffers: an equity curve sampled each step, a log of closed
//! trades, a log of fills and a log of every order's events. Off by default (the
//! engine holds it behind an `Option`) so the RL hot path allocates nothing; the
//! backtester turns it on. Stats are derived from these later.

use probatum_core::{OrderId, Side, TimeInForce};

use crate::execution::FillCosts;
use crate::stats::Stats;

/// What made a fill, as the fill log names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillReason {
    Market,
    Limit,
    Stop,
    Liquidation,
}

impl FillReason {
    /// The name the fill log gives it.
    pub const fn name(self) -> &'static str {
        match self {
            FillReason::Market => "market",
            FillReason::Limit => "limit",
            FillReason::Stop => "stop",
            FillReason::Liquidation => "liquidation",
        }
    }
}

/// One fill, with the reference it is measured from and what it paid beyond it
/// (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillRecord {
    /// The bar the fill landed on.
    pub tick: usize,
    /// The order that filled, or `None` for a forced close.
    pub order: Option<OrderId>,
    pub side: Side,
    pub size: f64,
    pub price: f64,
    /// The reference and the spread, slippage, impact and adjustment, per unit.
    pub costs: FillCosts,
    /// The fee, in quote.
    pub fee: f64,
    /// True when the fill took liquidity. A forced close is the venue's and takes
    /// none of the account's.
    pub taker: bool,
    pub reason: FillReason,
}

/// What happened to an order, as the order log names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Placed,
    Refused,
    Clamped,
    Filled,
    NotFilled,
    Expired,
    Cancelled,
}

impl EventKind {
    /// The name the order log gives it.
    pub const fn name(self) -> &'static str {
        match self {
            EventKind::Placed => "placed",
            EventKind::Refused => "refused",
            EventKind::Clamped => "clamped",
            EventKind::Filled => "filled",
            EventKind::NotFilled => "not_filled",
            EventKind::Expired => "expired",
            EventKind::Cancelled => "cancelled",
        }
    }
}

/// Why a placement was refused, with what it quotes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refusal {
    /// The pending queue already holds `max_open_orders`.
    Waiting(usize),
    /// The book already rests `max_open_orders`.
    Resting(usize),
    /// A size that is not a positive finite number can never fill.
    Size(f64),
    /// A limit price that is not a price.
    Price(f64),
    /// A stop trigger that is not a price.
    Trigger(f64),
    /// A post_only limit that would take liquidity at the last close.
    PostOnly,
}

/// Why an event happened, with the numbers its text quotes (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventReason {
    /// Placed or filled as its own type.
    Market,
    Limit,
    Stop,
    /// Placed again by a replace, keeping its id.
    Replace,
    Refused(Refusal),
    /// A replace refused, which leaves the order as it was.
    Replacing(Refusal),
    /// A fill clamped to the event's size, of `asked`.
    Fits {
        asked: f64,
    },
    /// A transfer clamped to the event's size, of `asked`.
    Transfer {
        asked: f64,
    },
    /// A market order that nothing on its bar could fill.
    NothingFilled,
    /// A fok market order that could not fill whole.
    Fok,
    /// A market order's remainder, gone after its one bar.
    MarketBar,
    /// An ioc or fok limit, gone after its one bar.
    OneBar(TimeInForce),
    CancelledByAccount,
    Liquidation,
}

/// One event in an order's life (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderEvent {
    /// The bar it happened on: the decision's for a placement, the fill's for a fill.
    pub tick: usize,
    pub kind: EventKind,
    /// The order, or `None` for a transfer's clamp.
    pub order: Option<OrderId>,
    /// The size a fill or a clamp came to.
    pub size: Option<f64>,
    /// The price a fill was booked at.
    pub price: Option<f64>,
    pub reason: EventReason,
}

/// One round trip: a maximal run of one sign, from the fill that takes the position
/// off zero to the fill that returns it to zero or flips it. Resizing inside it does
/// not end it, so a strategy that scales in and out reads as one long trade, and a
/// count of them counts decisions rather than fills (ADR 0023).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundTrip {
    /// `Buy` for a long, `Sell` for a short.
    pub side: Side,
    /// The largest position it reached.
    pub size: f64,
    /// The average price of its opening fills.
    pub entry: f64,
    /// The average price of its closing fills, NaN until one has closed any of it.
    pub exit: f64,
    pub open_tick: usize,
    /// The bar it closed on, or `None` while it is open.
    pub close_tick: Option<usize>,
    /// What its closing fills realized from the position's average entry, before
    /// fees and funding.
    pub realized: f64,
    /// Every fee of its fills, a flip's split by size with the round trip it ends,
    /// so each is carried exactly once (ADR 0014).
    pub fees: f64,
    /// The funding charged on it while it was open.
    pub funding: f64,
    /// Its net PnL: what it realized less its fees and its funding, and for one
    /// still open, what it holds unrealized at the last close as well.
    pub pnl: f64,
    /// `pnl` over `size * entry`.
    pub ret: f64,
    /// What closed it, or `None` while it is open.
    pub reason: Option<FillReason>,
    /// The sizes its opening and its closing fills came to.
    pub opened: f64,
    pub closed: f64,
    /// The highest and the lowest price over the part of each bar it was in: its
    /// entry bar from the fill to the close, its exit bar from the open to the fill,
    /// and every bar between whole (ADR 0023).
    pub high: f64,
    pub low: f64,
    /// How far its side's price went for it and against it over those parts, as
    /// a fraction of `entry`. Bounds that never overstate, since nothing inside a
    /// bar says when its high and its low came.
    pub best: f64,
    pub worst: f64,
}

impl RoundTrip {
    /// A round trip opened on bar `tick` by `size` at `price`, paying `fee`.
    pub fn open(side: Side, tick: usize, size: f64, price: f64, fee: f64) -> RoundTrip {
        let mut trip = RoundTrip {
            side,
            size,
            entry: price,
            exit: f64::NAN,
            open_tick: tick,
            close_tick: None,
            realized: 0.0,
            fees: fee,
            funding: 0.0,
            pnl: 0.0,
            ret: 0.0,
            reason: None,
            opened: size,
            closed: 0.0,
            high: price,
            low: price,
            best: 0.0,
            worst: 0.0,
        };
        trip.settle(0.0);
        trip
    }

    /// Count `price` among the prices it lived through.
    pub fn see(&mut self, price: f64) {
        self.high = self.high.max(price);
        self.low = self.low.min(price);
    }

    /// A fill of `size` at `price` that grows it, paying `fee`.
    pub fn grow(&mut self, size: f64, price: f64, fee: f64) {
        self.entry = (self.entry * self.opened + price * size) / (self.opened + size);
        self.opened += size;
        self.fees += fee;
        self.see(price);
    }

    /// A fill of `size` at `price` that shrinks it, paying `fee` and realizing
    /// `realized`.
    pub fn shrink(&mut self, size: f64, price: f64, fee: f64, realized: f64) {
        self.exit = if self.closed > 0.0 {
            (self.exit * self.closed + price * size) / (self.closed + size)
        } else {
            price
        };
        self.closed += size;
        self.fees += fee;
        self.realized += realized;
        self.see(price);
    }

    /// End it on bar `tick`, for `reason`.
    pub fn finish(&mut self, tick: usize, reason: FillReason) {
        self.close_tick = Some(tick);
        self.reason = Some(reason);
        self.settle(0.0);
    }

    /// The round trip still open, marked: `unrealized` is what its position holds
    /// at the last close.
    pub fn marked(&self, unrealized: f64) -> RoundTrip {
        let mut trip = *self;
        trip.settle(unrealized);
        trip
    }

    fn settle(&mut self, unrealized: f64) {
        self.pnl = self.realized + unrealized - self.fees - self.funding;
        let notional = self.size * self.entry;
        self.ret = if notional > 0.0 {
            self.pnl / notional
        } else {
            0.0
        };
        if self.entry > 0.0 {
            let (up, down) = (self.high / self.entry - 1.0, self.low / self.entry - 1.0);
            (self.best, self.worst) = if self.side == Side::Buy {
                (up, down)
            } else {
                (-down, -up)
            };
        }
    }
}

/// Side buffers for the equity curve, the round trips and the logs. Never part of
/// the hot-path state or the RL observation.
#[derive(Clone, Debug, Default)]
pub struct Reporter {
    equity_curve: Vec<f64>,
    trades: Vec<RoundTrip>,
    open: Option<RoundTrip>,
    fills: Vec<FillRecord>,
    events: Vec<OrderEvent>,
    /// Steps recorded while holding a nonzero position, for the exposure stat.
    in_position_steps: usize,
}

impl Reporter {
    /// An empty reporter.
    pub fn new() -> Reporter {
        Reporter::default()
    }

    /// Append the equity value for the current step, counting the step as exposure
    /// when a position is held.
    pub fn record_equity(&mut self, equity: f64, in_position: bool) {
        self.equity_curve.push(equity);
        if in_position {
            self.in_position_steps += 1;
        }
    }

    /// The round trip still open, to grow, shrink or charge.
    pub fn trip_mut(&mut self) -> Option<&mut RoundTrip> {
        self.open.as_mut()
    }

    /// Begin `trip`, the position having left zero.
    pub fn open_trip(&mut self, trip: RoundTrip) {
        self.open = Some(trip);
    }

    /// End the open round trip on bar `tick`, for `reason`.
    pub fn close_trip(&mut self, tick: usize, reason: FillReason) {
        if let Some(mut trip) = self.open.take() {
            trip.finish(tick, reason);
            self.trades.push(trip);
        }
    }

    /// Append a fill to the fill log.
    pub fn record_fill(&mut self, fill: FillRecord) {
        self.fills.push(fill);
    }

    /// Append an event to the order log.
    pub fn record_event(&mut self, event: OrderEvent) {
        self.events.push(event);
    }

    /// The equity sampled at each recorded step: one point per bar the engine
    /// actually advanced, so a driver that keeps calling `step` past the last bar
    /// adds none.
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// The closed round trips, in the order they closed. One a forced close ended
    /// carries the reason `liquidation` (ADRs 0017, 0023).
    pub fn trades(&self) -> &[RoundTrip] {
        &self.trades
    }

    /// The round trip still open, unmarked.
    pub fn open_trade(&self) -> Option<&RoundTrip> {
        self.open.as_ref()
    }

    /// Every fill in the order it landed, forced closes included (ADR 0022).
    pub fn fills(&self) -> &[FillRecord] {
        &self.fills
    }

    /// Every order's events in the order they happened, refusals included
    /// (ADR 0022).
    pub fn events(&self) -> &[OrderEvent] {
        &self.events
    }

    /// Number of recorded equity points, and nothing else. It does not pair with
    /// `is_empty`, which wants every other buffer empty too, so a reporter holding
    /// a round trip and no equity point answers zero here and `false` there.
    pub fn len(&self) -> usize {
        self.equity_curve.len()
    }

    /// True when no buffer holds anything: no equity point, trade, fill or event.
    pub fn is_empty(&self) -> bool {
        self.equity_curve.is_empty()
            && self.trades.is_empty()
            && self.open.is_none()
            && self.fills.is_empty()
            && self.events.is_empty()
    }

    /// Performance statistics over the recorded curve and trades. `initial` is the
    /// starting equity and `num_fills` the engine's fill counter; see ADR 0030 for
    /// the conventions.
    pub fn stats(
        &self,
        initial: f64,
        periods_per_year: f64,
        risk_free: f64,
        num_fills: usize,
        funding_paid: f64,
    ) -> Stats {
        Stats::compute(
            initial,
            &self.equity_curve,
            &self.trades,
            periods_per_year,
            risk_free,
            self.in_position_steps,
            num_fills,
            funding_paid,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{FillReason, Reporter, RoundTrip};
    use probatum_core::Side;

    #[test]
    fn records_the_equity_curve() {
        let mut r = Reporter::new();
        r.record_equity(1000.0, false);
        r.record_equity(1010.0, true);
        assert_eq!(r.equity_curve(), &[1000.0, 1010.0]);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn a_round_trip_is_a_trade_only_once_it_closes() {
        let mut r = Reporter::new();
        r.open_trip(RoundTrip::open(Side::Buy, 1, 2.0, 100.0, 0.1));
        assert!(r.trades().is_empty());
        assert_eq!(r.open_trade().map(|t| t.size), Some(2.0));
        if let Some(trip) = r.trip_mut() {
            trip.shrink(2.0, 110.0, 0.1, 20.0);
        }
        r.close_trip(5, FillReason::Market);
        let [trip] = r.trades() else {
            panic!("one round trip");
        };
        assert!((trip.pnl - 19.8).abs() < 1e-12);
        assert!((trip.ret - 19.8 / 200.0).abs() < 1e-12);
        assert_eq!(trip.close_tick, Some(5));
        assert!(r.open_trade().is_none());
    }

    #[test]
    fn its_entry_and_exit_are_the_averages_of_its_fills() {
        let mut trip = RoundTrip::open(Side::Sell, 0, 1.0, 100.0, 0.0);
        trip.grow(3.0, 120.0, 0.0);
        trip.shrink(2.0, 90.0, 0.0, 0.0);
        trip.shrink(2.0, 80.0, 0.0, 0.0);
        assert_eq!((trip.entry, trip.exit), (115.0, 85.0));
        assert_eq!((trip.opened, trip.closed), (4.0, 4.0));
    }

    #[test]
    fn default_is_empty() {
        assert!(Reporter::new().is_empty());
    }
}
