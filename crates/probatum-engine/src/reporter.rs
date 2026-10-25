//! Optional reporting buffers: an equity curve sampled each step, the ledger, the
//! round trips, a log of fills and a log of every order's events. Off by default
//! (the engine holds it behind an `Option`) so the RL hot path allocates nothing;
//! the backtester turns it on. Stats are derived from these later.

use probatum_core::{OrderId, Side, TimeInForce, Wallet};
use serde::{Deserialize, Serialize};

use crate::execution::FillCosts;
use crate::stats::Stats;

/// What made a fill, as the fill log names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillReason {
    Market,
    Limit,
    Stop,
    /// A bracket's take-profit leg.
    TakeProfit,
    /// A leg a `target` placed.
    Rebalance,
    /// A position a validation's seam closed.
    Seam,
    Liquidation,
    /// A position below half a lot, zeroed at the close, which ends its round trip.
    Dust,
    /// A position closed at its last close as its instrument was delisted.
    Delist,
}

impl FillReason {
    /// The name the fill log gives it.
    pub const fn name(self) -> &'static str {
        match self {
            FillReason::Market => "market",
            FillReason::Limit => "limit",
            FillReason::Stop => "stop",
            FillReason::TakeProfit => "take_profit",
            FillReason::Rebalance => "rebalance",
            FillReason::Seam => "seam",
            FillReason::Liquidation => "liquidation",
            FillReason::Dust => "dust",
            FillReason::Delist => "delist",
        }
    }
}

/// One fill, with the reference it is measured from and what it paid beyond it
/// (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FillRecord {
    /// The bar the fill landed on.
    pub tick: usize,
    /// The instrument it traded.
    pub instrument: usize,
    /// The order that filled, or `None` for a forced close or a delisting's close.
    pub order: Option<OrderId>,
    pub side: Side,
    pub size: f64,
    pub price: f64,
    /// The reference and the spread, slippage, impact and adjustment, per unit.
    pub costs: FillCosts,
    /// The part of the adjustment the margin bound applied, per unit, so that the rest
    /// is the bar's hold; zero for a forced close.
    pub lifted: f64,
    /// What the margin bound paid into the margin wallet for the part of its lift the
    /// bar could not hold, never below zero; zero for a forced close.
    pub paid_in: f64,
    /// The fee, in quote.
    pub fee: f64,
    /// True when the fill took liquidity. A forced close is the venue's and takes
    /// none of the account's.
    pub taker: bool,
    pub reason: FillReason,
}

/// What happened to an order, as the order log names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    Placed,
    Refused,
    Clamped,
    Filled,
    NotFilled,
    Expired,
    Cancelled,
    /// A bracket leg that took its side from a position.
    Armed,
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
            EventKind::Armed => "armed",
        }
    }
}

/// Why a placement was refused, with what it quotes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
    /// A limit that reaches the account's own resting limit of the other side.
    SelfTrade(OrderId),
    /// An order on an instrument that has not printed its first bar.
    Unlisted,
    /// An order on an instrument the venue has delisted.
    Delisted,
    /// A size that rounds down to no lot.
    Lot { size: f64, lot: f64 },
    /// A size whose notional at the last close falls short of the venue's minimum.
    MinNotional { size: f64, least: f64 },
    /// An amount in the quote currency that is not a positive finite number.
    Amount(f64),
    /// An amount in the quote currency below the venue's minimum notional.
    MinAmount { amount: f64, least: f64 },
}

/// The rule that held a fill, or that a market order's bar filled nothing for: the
/// last clamp that reduced the size before the lot rounded it, a bar that printed no
/// volume before every clamp, and a size below what a fill can book after them
/// (Specification, The fill log and the order log).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cause {
    /// The share of the trailing volume the bar's fills may take.
    Budget,
    ReduceOnly,
    /// A spot sale, held to the holding.
    SpotHolding,
    /// A spot buy, held to what the cash pays for, fee included.
    SpotCash,
    /// A perpetual account at or below zero margin equity, which may only shrink.
    Equity,
    /// The last tier's bound, the largest position a table allows.
    LastTier,
    /// The summed requirement of the margin cap.
    Cap,
    NoVolume,
    Lot,
}

/// Why an event happened, with the numbers its text quotes (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum EventReason {
    /// Placed or filled as its own type.
    Market,
    Limit,
    Stop,
    /// Placed, armed or filled as a bracket's take-profit leg.
    TakeProfit,
    /// Placed or filled as a leg of a `target`.
    Rebalance,
    /// Placed again by a replace, keeping its id.
    Replace,
    /// A market order placed for `amount` of the quote currency, on spot or a linear
    /// contract.
    Value {
        amount: f64,
    },
    /// A placement refused, or a bracket's take-profit cancelled as it arms against
    /// the account's own order.
    Refused(Refusal),
    /// A replace refused, which leaves the order as it was.
    Replacing(Refusal),
    /// A fill `by` clamped to the event's size, of `asked`.
    Fits {
        asked: f64,
        by: Cause,
    },
    /// A transfer into `to` clamped to the event's size, of `asked`, by what the
    /// wallet it leaves may give.
    Transfer {
        asked: f64,
        to: Wallet,
    },
    /// A placement held to the maximum order quantity, the event's size, of `asked`.
    HeldTo {
        asked: f64,
    },
    /// A market order that nothing on its bar could fill, for `by`.
    NothingFilled {
        by: Cause,
    },
    /// A fok market order `by` held to `size` of the `asked` it fills whole or not at
    /// all.
    Fok {
        by: Cause,
        size: f64,
        asked: f64,
    },
    /// A market order's remainder, gone after its one bar.
    MarketBar,
    /// An ioc or fok limit, gone after its one bar.
    OneBar(TimeInForce),
    CancelledByAccount,
    /// A post_only limit its first bar opened at or through, at the open, or
    /// `inside` the bar, where the part of it after a fill opened so.
    OpenedThrough {
        inside: bool,
    },
    /// A bracket leg whose position closed or flipped.
    PositionClosed,
    /// A bracket leg whose partner filled.
    PartnerFilled,
    /// A target's leg cancelled by a later target in the same decision.
    Retargeted,
    /// Cancelled, placed or filled at a validation's seam.
    Seam,
    Liquidation,
    /// Cancelled as its instrument was delisted.
    Delist,
}

/// One event in an order's life (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderEvent {
    /// The bar it happened on: the decision's for a placement, the fill's for a fill.
    pub tick: usize,
    pub kind: EventKind,
    /// The order, or `None` for a transfer's clamp.
    pub order: Option<OrderId>,
    /// Its order's instrument, or `None` for a transfer's clamp.
    pub instrument: Option<usize>,
    /// The side its order asked, on a placed, refused or armed event, and `None`
    /// elsewhere and for a bracket leg placed while flat until it arms.
    pub side: Option<Side>,
    /// The size a fill or a clamp came to, or what a placed or armed event's order
    /// asks in all, or a refused one's call gave.
    pub size: Option<f64>,
    /// The price a fill was booked at, or a placed or armed limit's price or stop's
    /// trigger.
    pub price: Option<f64>,
    pub reason: EventReason,
}

/// One round trip: a maximal run of one sign, from the fill that takes the position
/// off zero to the fill that returns it to zero or flips it. Resizing inside it does
/// not end it, so a strategy that scales in and out reads as one long trade, and a
/// count of them counts decisions rather than fills (ADR 0023).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundTrip {
    /// The instrument it traded.
    pub instrument: usize,
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
    /// A round trip on `instrument` opened on bar `tick` by `size` at `price`,
    /// paying `fee`.
    pub fn open(
        instrument: usize,
        side: Side,
        tick: usize,
        size: f64,
        price: f64,
        fee: f64,
    ) -> RoundTrip {
        let mut trip = RoundTrip {
            instrument,
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
        self.excursions();
    }

    /// Its best and worst from its entry, its high and its low.
    pub(crate) fn excursions(&mut self) {
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

/// What a bar cost, in quote, each term computed on its own so that the ledger
/// identity is a check rather than a residual (ADR 0022).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BarCosts {
    pub fees: f64,
    /// Funding paid, positive when paid away.
    pub funding: f64,
    /// What fills paid beyond their reference prices, each part in proportion
    /// when the bar's range held a price back.
    pub spread: f64,
    pub slippage: f64,
    pub impact: f64,
    /// What a forced close left in the margin wallet, taken or absorbed, less
    /// what the margin bound lifted fills by.
    pub liquidation: f64,
    pub dust: f64,
}

/// One row of the ledger: the account marked at a bar's close, and what that bar
/// cost (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerRow {
    pub tick: usize,
    pub equity: f64,
    pub spot_cash: f64,
    pub margin_cash: f64,
    /// The positions' notional at the close, unsigned and signed.
    pub gross_exposure: f64,
    pub net_exposure: f64,
    /// Gross exposure over equity, zero once equity is gone.
    pub leverage: f64,
    /// What the margin cap asks of the positions at the close.
    pub margin_used: f64,
    /// The maintenance requirement their tier tables set, zero without one.
    pub maintenance: f64,
    pub costs: BarCosts,
}

/// One instrument's line of a ledger row, marked at its close, or at its last mark
/// on a row it printed no bar on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Leg {
    pub position: f64,
    pub entry: f64,
    pub mark: f64,
    pub notional: f64,
    pub unrealized: f64,
    /// The signed notional over equity, zero once equity is gone.
    pub weight: f64,
    /// The weight the latest target asked of the instrument less `weight`, NaN where
    /// no target has named it.
    pub drift: f64,
    /// What the instrument has made since the run's first row: the cash its fills,
    /// its funding and the charges given to it moved in the wallet it settles in, and
    /// its position's value at the close, nothing while it is flat.
    pub pnl: f64,
}

/// What a reporter books, a recording level (ADR 0022).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Record {
    /// The ledger and its legs, the round trips, the fill log and the order log.
    #[default]
    Full,
    /// The ledger without its legs, and the round trips. A run's statistics read only
    /// the equity curve, the closed round trips and the engine's own counts, so they
    /// are the full level's.
    Account,
    /// The equity curve and the round trips alone, for a run read only for its
    /// statistics.
    Stats,
}

/// Side buffers for the equity curve, the ledger, the round trips and the logs.
/// Never part of the hot-path state or the RL observation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reporter {
    /// What it books, which a reset keeps.
    level: Record,
    equity_curve: Vec<f64>,
    trades: Vec<RoundTrip>,
    /// Each instrument's round trip still open, grown to the highest instrument
    /// that opened one.
    open: Vec<Option<RoundTrip>>,
    fills: Vec<FillRecord>,
    events: Vec<OrderEvent>,
    ledger: Vec<LedgerRow>,
    /// Each row's legs, one per instrument, row by row.
    legs: Vec<Leg>,
    /// What the bar being stepped has cost so far.
    costs: BarCosts,
    /// Steps recorded while holding a nonzero position, for the exposure stat.
    in_position_steps: usize,
    /// What each instrument's fills, funding and the charges given to it have moved in
    /// the wallet it settles in, grown to the highest instrument that moved any.
    flows: Vec<f64>,
    /// The instrument whose fill or dust sweep last left no perpetual held, which a
    /// margin wallet's deficit set to zero is given to.
    closer: Option<usize>,
}

impl Reporter {
    /// An empty reporter that books everything.
    pub fn new() -> Reporter {
        Reporter::default()
    }

    /// An empty reporter that books what `level` keeps.
    pub fn at(level: Record) -> Reporter {
        Reporter {
            level,
            ..Reporter::default()
        }
    }

    /// Steps recorded while holding a position, which the exposure statistic counts.
    pub fn in_position_steps(&self) -> usize {
        self.in_position_steps
    }

    /// What it books.
    pub fn level(&self) -> Record {
        self.level
    }

    /// Append the equity value for the current step, counting the step as exposure
    /// when a position is held.
    pub fn record_equity(&mut self, equity: f64, in_position: bool) {
        self.equity_curve.push(equity);
        if in_position {
            self.in_position_steps += 1;
        }
    }

    /// What the bar being stepped has cost so far, to add to.
    pub fn costs_mut(&mut self) -> &mut BarCosts {
        &mut self.costs
    }

    /// Append the ledger row of a bar, carrying what the bar cost, and start the
    /// next bar's costs from nothing.
    pub fn record_row(&mut self, mut row: LedgerRow) {
        row.costs = std::mem::take(&mut self.costs);
        self.ledger.push(row);
    }

    /// Append one instrument's leg of the row just recorded.
    pub fn record_leg(&mut self, leg: Leg) {
        self.legs.push(leg);
    }

    /// Add `cash` to what instrument `i` has moved in the wallet it settles in.
    pub fn flow(&mut self, i: usize, cash: f64) {
        if self.flows.len() <= i {
            self.flows.resize(i + 1, 0.0);
        }
        self.flows[i] += cash;
    }

    /// What instrument `i` has moved in the wallet it settles in since the run's first
    /// row.
    pub fn flow_of(&self, i: usize) -> f64 {
        self.flows.get(i).copied().unwrap_or(0.0)
    }

    /// Record instrument `i` as the one whose fill or dust sweep last left no
    /// perpetual held.
    pub fn set_closer(&mut self, i: usize) {
        self.closer = Some(i);
    }

    /// The instrument whose fill or dust sweep last left no perpetual held.
    pub fn closer(&self) -> Option<usize> {
        self.closer
    }

    /// Instrument `i`'s round trip still open, to grow, shrink or charge.
    pub fn trip_mut(&mut self, i: usize) -> Option<&mut RoundTrip> {
        self.open.get_mut(i).and_then(Option::as_mut)
    }

    /// Begin `trip`, its instrument's position having left zero.
    pub fn open_trip(&mut self, trip: RoundTrip) {
        let i = trip.instrument;
        if self.open.len() <= i {
            self.open.resize(i + 1, None);
        }
        self.open[i] = Some(trip);
    }

    /// End instrument `i`'s open round trip on bar `tick`, for `reason`.
    pub fn close_trip(&mut self, i: usize, tick: usize, reason: FillReason) {
        if let Some(mut trip) = self.open.get_mut(i).and_then(Option::take) {
            trip.finish(tick, reason);
            self.trades.push(trip);
        }
    }

    /// Append a fill to the fill log, which only the full level keeps.
    pub fn record_fill(&mut self, fill: FillRecord) {
        if self.level == Record::Full {
            self.fills.push(fill);
        }
    }

    /// Append an event to the order log, which only the full level keeps.
    pub fn record_event(&mut self, event: OrderEvent) {
        if self.level == Record::Full {
            self.events.push(event);
        }
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

    /// The round trips still open, unmarked, in instrument order.
    pub fn open_trades(&self) -> impl Iterator<Item = &RoundTrip> {
        self.open.iter().flatten()
    }

    /// One row per bar from the opening balance on, so no curve is one shorter than
    /// the data (ADR 0022).
    pub fn ledger(&self) -> &[LedgerRow] {
        &self.ledger
    }

    /// Every row's legs, one per instrument, row by row.
    pub fn legs(&self) -> &[Leg] {
        &self.legs
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
            && self.open.iter().all(Option::is_none)
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
            self.trades.iter().map(|trip| trip.pnl),
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
    use super::{EventKind, FillReason, Reporter, RoundTrip};
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
        r.open_trip(RoundTrip::open(0, Side::Buy, 1, 2.0, 100.0, 0.1));
        assert!(r.trades().is_empty());
        assert_eq!(r.open_trades().next().map(|t| t.size), Some(2.0));
        if let Some(trip) = r.trip_mut(0) {
            trip.shrink(2.0, 110.0, 0.1, 20.0);
        }
        r.close_trip(0, 5, FillReason::Market);
        let [trip] = r.trades() else {
            panic!("one round trip");
        };
        assert!((trip.pnl - 19.8).abs() < 1e-12);
        assert!((trip.ret - 19.8 / 200.0).abs() < 1e-12);
        assert_eq!(trip.close_tick, Some(5));
        assert!(r.open_trades().next().is_none());
    }

    #[test]
    fn its_entry_and_exit_are_the_averages_of_its_fills() {
        let mut trip = RoundTrip::open(0, Side::Sell, 0, 1.0, 100.0, 0.0);
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

    #[test]
    fn the_logs_name_reasons_and_kinds_as_the_specification_does() {
        let reasons = [
            FillReason::Market,
            FillReason::Limit,
            FillReason::Stop,
            FillReason::Liquidation,
        ];
        assert_eq!(
            reasons.map(|reason| reason.name()),
            ["market", "limit", "stop", "liquidation"]
        );
        let kinds = [
            EventKind::Placed,
            EventKind::Refused,
            EventKind::Clamped,
            EventKind::Filled,
            EventKind::NotFilled,
            EventKind::Expired,
            EventKind::Cancelled,
        ];
        assert_eq!(
            kinds.map(|kind| kind.name()),
            [
                "placed",
                "refused",
                "clamped",
                "filled",
                "not_filled",
                "expired",
                "cancelled"
            ]
        );
    }

    #[test]
    fn its_entry_and_exit_weigh_each_fill_by_its_size() {
        // 2 at 100 and 3 at 200 enter at 160; 1 at 150 and 4 at 200 leave at 190
        let mut trip = RoundTrip::open(0, Side::Buy, 0, 2.0, 100.0, 0.0);
        trip.grow(3.0, 200.0, 0.0);
        trip.shrink(1.0, 150.0, 0.0, 0.0);
        trip.shrink(4.0, 200.0, 0.0, 0.0);
        assert_eq!((trip.entry, trip.exit), (160.0, 190.0));
    }

    #[test]
    fn exposure_counts_the_steps_recorded_in_a_position() {
        let mut r = Reporter::new();
        r.record_equity(1000.0, false);
        r.record_equity(1010.0, true);
        r.record_equity(1020.0, true);
        r.record_equity(1030.0, false);
        assert_eq!(r.stats(1000.0, 365.0, 0.0, 0, 0.0).exposure_pct, 50.0);
    }

    #[test]
    fn a_reporter_holding_anything_is_not_empty() {
        let mut marked = Reporter::new();
        marked.record_equity(1000.0, false);
        assert!(!marked.is_empty());
        let mut tripped = Reporter::new();
        tripped.open_trip(RoundTrip::open(0, Side::Buy, 1, 1.0, 100.0, 0.0));
        assert_eq!(tripped.len(), 0);
        assert!(!tripped.is_empty());
    }
}
