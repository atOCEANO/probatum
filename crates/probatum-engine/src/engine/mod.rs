//! The single-env step state machine. `reset()` positions at the first bar;
//! `step()` advances exactly one bar, resolves the orders placed since the last
//! step against the NEW bar, marks the account, and returns the new `State`. An
//! order decided on one bar fills on the next, never the bar the decision saw.
//!
//! Within a step the order of events follows the bar. Transfers decided at the last
//! close land first, then funding stamped between the last row and this one, at
//! the last marks, and then the path: every perp held moves along one synchronous
//! adverse path, from its last mark to its open and from the open to its adverse
//! extreme, and the point its margin runs out at there is found before any order
//! resolves. A clip on the gap closed the account before the bar opened, so the
//! forced close happens first and nothing on a perp fills; any other clip cuts
//! every held perp's bar there, so an order that would have filled beyond it never
//! triggers. Then, against the clipped bars: 1. pending market orders fill at the
//! open, 2. resting limit and stop orders fill against the range, a bracket placed
//! while flat arming on the fill that opens its position, and a take-profit held back
//! where the bar liquidates the account without it, 3. the path is solved again for
//! the account the fills left, a leg opened during the bar joining it at its fill
//! price, counting any debit stamped inside the bar, 4. the funding the row owns is
//! charged at the closes on the positions still held, and an account a debit takes
//! to its trigger is closed there. Equity is then marked at the close. A fill that
//! reduces a perp is bounded once more as it lands, by what the other legs leave at
//! the same point of their paths (ADRs 0008, 0017, 0018, 0020).
//!
//! Several instruments share the clock. Each order fills on its own instrument's
//! bar, and an instrument that printed no bar on a row keeps its last mark, its
//! orders wait, and it stands still on the path. The cap and the bound read every
//! other perp at the fraction of its own bar the fill sits at on its path.

mod brackets;
#[cfg(test)]
mod fixtures;
mod funding;
mod ledger;
mod liquidation;
mod margin;
mod orders;
mod path;
mod resolution;
mod target;
mod transfers;

use path::{CheckLeg, FillNote, HeldLeg, Path};
pub use target::TargetError;
use target::Targets;

use probatum_core::{
    Account, Bps, Candle, FlatCostModel, Market, Order, Price, RestingOrderBook, State, Wallet,
};
use rand_chacha::rand_core::RngCore;
use rand_chacha::ChaCha20Rng;

use crate::candles::{printed, Candles};
use crate::clock::Clock;
use crate::execution::{FillModel, Impact, Limit, Spread};
use crate::reporter::{BarCosts, LedgerRow, Leg, Reporter, RoundTrip};
use crate::stats::Stats;
use crate::streams;

/// A position moved toward zero by less than this counts as no close, mirroring
/// the position's own dust epsilon so a non-representable fractional close does
/// not log a phantom trade.
const CLOSE_EPS: f64 = 1e-9;

/// How a venue decides every event a bar's four prices cannot order (ADR 0012).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrabar {
    /// The worse way for the account, every time: the default everywhere.
    Pessimistic,
    /// The better way, every time.
    Optimistic,
    /// Each event drawn one way or the other from the stream named `intrabar` in a
    /// run seeded with this, for training, where a fixed way could be learnt.
    Random(u128),
}

/// How a venue closes out an account that reaches its trigger (ADR 0019).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Liquidation {
    /// Every leg closed at its price at the clip point, and what the margin wallet
    /// still holds booked as the liquidation charge: the default everywhere.
    Takeover,
    /// Every leg sold as a taker from its trigger price, the trader keeping what
    /// remains, unless the path also reaches the backstop level, equity at this share
    /// of the requirement, or the sale's own costs would take equity there below it,
    /// where takeover applies.
    BookThenBackstop(f64),
}

/// How the bar on hand decides each event its prices cannot order, true where it
/// takes the worse way (Specification, What a bar cannot show).
#[derive(Clone, Debug)]
struct Unseen {
    /// The way of a fixed setting, which every event takes.
    worse: bool,
    /// Several perps reach their worst prices at the same moment.
    together: bool,
    /// Exits on a liquidating bar are held back.
    hold: bool,
    /// Each instrument's exits fill worst first.
    exits: Vec<bool>,
    /// The extreme on the open's side of a fill inside the bar that opens a position
    /// on each instrument came before the fill, where it favours that position.
    opened: Vec<bool>,
    /// Each funding event the row owns counts its debits and not its credits, drawn
    /// by a random setting and empty under a fixed one.
    funding: Vec<bool>,
    /// The backstop takes over a sale on the book whose path reaches its level.
    backstop: bool,
}

impl Unseen {
    /// Draw this bar's way for each event from `stream`, in a fixed order: the
    /// perps together, the exits held back, each instrument's exits, each
    /// instrument's fill inside the bar, each of the `events` funding events the row
    /// owns, then the backstop.
    fn draw(&mut self, stream: &mut ChaCha20Rng, events: usize) {
        let mut worse = || stream.next_u32() >> 31 == 1;
        self.together = worse();
        self.hold = worse();
        for exits in &mut self.exits {
            *exits = worse();
        }
        for opened in &mut self.opened {
            *opened = worse();
        }
        self.funding.clear();
        self.funding.extend((0..events).map(|_| worse()));
        self.backstop = worse();
    }
}

/// Construction parameters for an engine.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub market: Market,
    /// Opening cash of the wallet the market settles in: the spot wallet on spot,
    /// the margin wallet on a perp.
    pub quote: f64,
    /// Opening cash of the other wallet, which backs nothing until a transfer
    /// moves it across (ADR 0008).
    pub idle_cash: f64,
    pub fee_taker: f64,
    pub fee_maker: f64,
    pub slippage_bps: f64,
    /// The share of each instrument's trailing volume its fills may take on a bar,
    /// shared by every order in resolution order (Specification, The liquidity
    /// budget).
    pub budget: f64,
    /// The bars before a fill its window reads (Specification, The window).
    pub window: usize,
    pub max_open_orders: usize,
    pub report: bool,
    /// Perp margin cap: a fill may not grow the position past `max_leverage`
    /// times equity in notional. Zero means no cap; the shipped default is a
    /// finite 10x (ADR 0015). Ignored on spot.
    pub max_leverage: f64,
    /// Market-impact coefficient for taker fills, scaled by the window's volatility
    /// and the fill's share of its trailing volume (ADR 0013). Zero disables it.
    pub impact: f64,
    pub impact_model: Impact,
    /// The spread a taker crosses half of.
    pub spread: Spread,
    /// When a resting limit fills: on touch, or half a step of its instrument's price
    /// rule past it (Specification, Limits).
    pub limit: Limit,
    /// Whether each instrument's sizes hold its orders: the lot, the price rule, the
    /// minimum notional and the maximum order quantity (Specification, Sizes and
    /// dust).
    pub sizes: bool,
    /// Perp funding rate charged per event on the position notional; a long pays a
    /// positive rate, a short receives (ADR 0020). Ignored on spot.
    pub funding_rate: f64,
    /// Bars between funding events. Zero disables funding (ADR 0020).
    pub funding_interval: usize,
    /// How every event a bar's prices cannot order is decided (ADR 0012).
    pub intrabar: Intrabar,
    /// How an account that reaches its trigger is closed out (ADR 0019).
    pub liquidation: Liquidation,
}

impl Default for EngineConfig {
    fn default() -> EngineConfig {
        EngineConfig {
            market: Market::Spot,
            quote: 10_000.0,
            idle_cash: 0.0,
            fee_taker: 0.0006,
            fee_maker: 0.0002,
            slippage_bps: 0.0,
            budget: 1.0,
            window: 24,
            max_open_orders: 8,
            report: false,
            max_leverage: 10.0,
            impact: 0.0,
            impact_model: Impact::Linear,
            spread: Spread::None,
            limit: Limit::Touch,
            sizes: false,
            funding_rate: 0.0,
            funding_interval: 0,
            intrabar: Intrabar::Pessimistic,
            liquidation: Liquidation::Takeover,
        }
    }
}

/// The single-env bar engine: candles, an account, a resting book, the fill and
/// cost models, and an optional reporter.
pub struct Engine {
    // Set once in `new` and never reassigned or mutated. probatum-py hands out
    // zero-copy, read-only numpy views into this buffer (ADR 0040), so its
    // address and contents must stay stable for the engine's whole life; do not
    // add a method that reallocates it or vends a mutable candle view.
    candles: Candles,
    config: EngineConfig,
    fill_model: FillModel,
    cost: FlatCostModel,
    account: Account,
    book: RestingOrderBook,
    reporter: Option<Reporter>,
    pending: Vec<Order>,
    /// The queue's other buffer: the orders a bar resolves are read from one while
    /// those that wait go into the other, so a step allocates no queue.
    spare: Vec<Order>,
    /// Transfers decided since the last step, in order, each an amount and the
    /// wallet it goes into.
    transfers: Vec<(f64, Wallet)>,
    targets: Targets,
    tick: usize,
    id_counter: u64,
    /// Fills applied since the last reset. A run whose orders never filled, on a
    /// series with no volume say, is otherwise indistinguishable from one that never
    /// placed an order.
    fills: usize,
    /// Funding paid since the last reset, in quote, positive when paid away. The
    /// account returned each payment and this is what keeps it (ADR 0020).
    funding_paid: f64,
    /// Set when the account is dead: its equity, both wallets together, marked at
    /// or below zero, where a takeover leaves it unless the spot wallet holds
    /// cash (ADR 0008). A terminal event for the RL env. Cleared on reset.
    bust: bool,
    /// Each instrument's last close, the mark it keeps on a row it printed no bar
    /// on, NaN until its first bar.
    marks: Vec<f64>,
    /// What each perp's position did during the bar, for the check: the part
    /// carried into it and the parts opened or flipped during it (The path), built
    /// from the notes of the bar's perp fills on a bar whose check needs them.
    legs: Vec<Vec<HeldLeg>>,
    notes: Vec<FillNote>,
    /// Each instrument's bar clipped where the account left the market on it, or
    /// none, for the orders a bar resolves, and whether any is.
    clips: Vec<Option<Candle>>,
    clipped: bool,
    /// Whether a perp flipped on this bar, whose old side then left the path.
    flipped: bool,
    /// Whether the last closes left the account clear of its trigger, which holds
    /// at the next bar's last marks until a transfer or a funding charge moves it.
    cleared: bool,
    /// Whether this bar's check found the account clear with every leg at its
    /// adverse extreme, which leaves it clear at the closes unless the row's funding
    /// is charged first.
    clear_through: bool,
    /// Whether each instrument's tier table keeps any maintenance, which it does
    /// for the engine's whole life.
    maintains: Vec<bool>,
    /// Whether a tier table or the venue's leverage caps each instrument's perp
    /// positions, which holds for the engine's whole life.
    capped: Vec<bool>,
    /// Where each perp's position started on this bar's path, which decides the
    /// exits a liquidating bar holds back: the open for one carried into the bar or
    /// opened at it, the reference of the fill that opened or flipped it, or of a
    /// later fill inside the bar that grew it, otherwise, NaN while it is flat.
    starts: Vec<f64>,
    /// Whether a fill inside the bar opened, flipped or last grew each position, at
    /// a price the open had not reached, which an order that can only reduce it then
    /// meets the bar from (ADR 0012).
    inside: Vec<bool>,
    /// The reference of the fill that opened or flipped each position on this bar,
    /// and whether that fill came inside the bar, which the round trip it began
    /// counts its entry bar from (ADR 0023).
    entered: Vec<(f64, bool)>,
    /// The money the bar's perp fills have put at stake so far, for each the margin
    /// wallet before it, what the bound paid in and its notional, which the rounding a
    /// close can leave in a margin wallet it empties is measured against.
    reach: f64,
    /// Each instrument's budget: the bar it was filled for and what its fills may still
    /// take there, filled from its trailing volume at the bar's first fill on it
    /// (Specification, The liquidity budget).
    budget: Vec<(usize, f64)>,
    unseen: Unseen,
    /// The stream a random setting draws each bar's way from, which runs on across
    /// resets, so each episode of one engine draws afresh.
    stream: Option<ChaCha20Rng>,
    // the path's buffers, refilled on every bar so a step allocates nothing
    path: Path,
    checked: Vec<CheckLeg>,
    cuts: Vec<f64>,
    spans: Vec<(f64, f64)>,
    closes: Vec<(usize, f64, f64)>,
}

impl Engine {
    /// Build an engine over `candles`. Panics if `candles` is empty, since a
    /// zero-bar series has no bar to reset or step to. Call `reset()` to get the
    /// first state.
    pub fn new(candles: Candles, config: EngineConfig) -> Engine {
        assert!(!candles.is_empty(), "candles must have at least one row");
        let candles = Engine::windowed(candles, &[config]);
        let width = candles.width();
        let markets = (0..width)
            .map(|i| candles.market(i).unwrap_or(config.market))
            .collect();
        let (spot, margin) = Engine::opening(&config);
        let maintains = (0..width)
            .map(|i| candles.tiers(i).rows().iter().any(|tier| tier.rate > 0.0))
            .collect();
        let capped = (0..width)
            .map(|i| config.max_leverage > 0.0 || !candles.tiers(i).rows().is_empty())
            .collect();
        let mut engine = Engine {
            fill_model: FillModel {
                slippage_bps: Bps(config.slippage_bps),
                impact: config.impact,
                impact_model: config.impact_model,
                spread: config.spread,
            },
            cost: FlatCostModel {
                fee_taker: config.fee_taker,
                fee_maker: config.fee_maker,
            },
            account: Account::universe(markets, spot, margin),
            book: RestingOrderBook::new(width * config.max_open_orders),
            reporter: if config.report {
                Some(Reporter::new())
            } else {
                None
            },
            pending: Vec::new(),
            spare: Vec::new(),
            transfers: Vec::new(),
            targets: Targets::new(width),
            tick: 0,
            id_counter: 1,
            fills: 0,
            funding_paid: 0.0,
            bust: false,
            marks: vec![f64::NAN; width],
            legs: vec![Vec::new(); width],
            clips: vec![None; width],
            clipped: false,
            flipped: false,
            cleared: false,
            clear_through: false,
            notes: Vec::new(),
            maintains,
            capped,
            starts: vec![f64::NAN; width],
            inside: vec![false; width],
            entered: vec![(f64::NAN, false); width],
            reach: 0.0,
            budget: vec![(usize::MAX, 0.0); width],
            unseen: {
                let worse = config.intrabar != Intrabar::Optimistic;
                Unseen {
                    worse,
                    together: worse,
                    hold: worse,
                    exits: vec![worse; width],
                    opened: vec![worse; width],
                    funding: Vec::new(),
                    backstop: worse,
                }
            },
            stream: match config.intrabar {
                Intrabar::Random(seed) => Some(streams::stream(seed, &streams::named("intrabar"))),
                _ => None,
            },
            path: Path::default(),
            checked: Vec::new(),
            cuts: Vec::new(),
            spans: Vec::new(),
            closes: Vec::new(),
            candles,
            config,
        };
        engine.mark_from(0);
        if engine.reporter.is_some() {
            engine.record_ledger();
        }
        engine
    }

    /// The engine of env `env` of a batch, its random stream named apart from every
    /// other env's, `intrabar` and the env's index, so the envs of one seed draw
    /// apart.
    pub fn for_env(mut self, env: usize) -> Engine {
        if let Intrabar::Random(seed) = self.config.intrabar {
            let mut key = streams::named("intrabar");
            key.push(env as u32);
            self.stream = Some(streams::stream(seed, &key));
        }
        self
    }

    /// `candles` with each row's window read as `configs` ask between them, the
    /// deviation where an impact reads it and the spread where one is estimated, kept
    /// where they hold it already, as a batch's shared series does.
    pub(crate) fn windowed(candles: Candles, configs: &[EngineConfig]) -> Candles {
        let Some(config) = configs.first() else {
            return candles;
        };
        let sigma = configs.iter().any(|config| config.impact > 0.0);
        let estimated = configs
            .iter()
            .any(|config| config.spread == Spread::Estimated);
        if candles.windowed(config.window, sigma, estimated) {
            candles
        } else {
            candles.with_window(config.window, sigma, estimated)
        }
    }

    /// What instrument `i`'s fills may still take on this bar: a share of its trailing
    /// volume, which every order on its bar draws on in resolution order, filled at
    /// the bar's first fill on it.
    pub(super) fn available(&mut self, i: usize) -> f64 {
        let (at, left) = self.budget[i];
        if at == self.tick {
            return left;
        }
        let left = self.config.budget * self.candles.trailing_volume(self.tick, i);
        self.budget[i] = (self.tick, left);
        left
    }

    /// Take `applied` from instrument `i`'s budget for this bar, which what is left
    /// can round a hair below zero, and a bar never owes volume.
    pub(super) fn spend_budget(&mut self, i: usize, applied: f64) {
        let left = self.budget[i].1 - applied;
        self.budget[i].1 = if left > 0.0 { left } else { 0.0 };
    }

    /// Each wallet's opening cash: `quote` in the one the config's market settles
    /// in and the idle cash in the other.
    fn opening(config: &EngineConfig) -> (f64, f64) {
        match config.market {
            Market::Spot => (config.quote, config.idle_cash),
            Market::Perp => (config.idle_cash, config.quote),
        }
    }

    /// Mark each instrument at its last close on or before row `start`.
    fn mark_from(&mut self, start: usize) {
        for i in 0..self.marks.len() {
            self.marks[i] = (0..=start)
                .rev()
                .map(|t| self.candles.bar(t, i))
                .find(printed)
                .map_or(f64::NAN, |bar| bar.close);
        }
    }

    /// Append the ledger row of the current bar and each instrument's leg of it,
    /// reporting being on.
    fn record_ledger(&mut self) {
        let row = self.ledger_row();
        let equity = row.equity;
        if let Some(reporter) = self.reporter.as_mut() {
            reporter.record_row(row);
            for (i, position) in self.account.positions.iter().enumerate() {
                let qty = position.qty.get();
                let mark = self.marks[i];
                let signed = if qty == 0.0 { 0.0 } else { qty * mark };
                let weight = if equity > 0.0 { signed / equity } else { 0.0 };
                reporter.record_leg(Leg {
                    position: qty,
                    entry: position.avg_entry.get(),
                    mark,
                    notional: signed.abs(),
                    unrealized: if qty == 0.0 {
                        0.0
                    } else {
                        position.unrealized(Price(mark))
                    },
                    weight,
                    drift: self.targets.asked[i] - weight,
                });
            }
        }
    }

    /// Reset to the first bar and a fresh account, returning the initial state.
    pub fn reset(&mut self) -> State {
        self.reset_at(0)
    }

    /// Reset the account and cursor, starting at `start_tick` instead of the first
    /// bar. The vectorized env uses this to give each env its own random start
    /// offset. `start_tick` is clamped to the last bar.
    pub fn reset_at(&mut self, start_tick: usize) -> State {
        let (spot, margin) = Engine::opening(&self.config);
        self.account.reset(spot, margin);
        self.book.cancel_all();
        self.pending.clear();
        self.transfers.clear();
        self.targets.reset();
        self.tick = start_tick.min(self.candles.len().saturating_sub(1));
        self.mark_from(self.tick);
        // The id counter deliberately survives a reset. Restarting it made an id
        // handed out in one episode name a live order in the next, so a handle
        // carried across a reset cancelled somebody else's order (ADR 0010).
        self.fills = 0;
        self.funding_paid = 0.0;
        self.bust = false;
        // what an earlier episode took of a bar's budget is not taken in this one
        self.budget.fill((usize::MAX, 0.0));
        if self.reporter.is_some() {
            self.reporter = Some(Reporter::new());
            self.record_ledger();
        }
        self.state()
    }

    /// True when the account died since the last reset: its equity, both wallets
    /// together, reached zero, on either market. A takeover does that unless the
    /// spot wallet holds cash. The RL env's terminal signal (ADR 0008).
    /// The engine does not stop on it; a driver that wants to halt checks it.
    pub fn is_bust(&self) -> bool {
        self.bust
    }

    /// The current state without advancing, for reading between steps.
    pub fn current_state(&self) -> State {
        self.state()
    }

    /// The current bar index. Zero after `reset()`, and it stops at the last bar
    /// rather than running past it, so a driver that keeps stepping sees it stand
    /// still.
    pub fn tick(&self) -> usize {
        self.tick
    }

    /// True when there is no next bar to step into.
    pub fn done(&self) -> bool {
        self.tick + 1 >= self.candles.len()
    }

    /// The number of candles the engine holds, fixed for its life: the series is
    /// shared immutably and never appended to.
    pub fn num_bars(&self) -> usize {
        self.candles.len()
    }

    /// The number of instruments on each row.
    pub fn width(&self) -> usize {
        self.candles.width()
    }

    /// The market instrument `i` trades on.
    pub fn market(&self, i: usize) -> Market {
        self.account.markets[i]
    }

    /// Instrument `i`'s signed position, in base units or contracts.
    pub fn position(&self, i: usize) -> f64 {
        self.account.positions[i].qty.get()
    }

    /// Instrument `i`'s last close, the mark its position is valued at.
    pub fn mark(&self, i: usize) -> f64 {
        self.marks[i]
    }

    /// Bars a year by the series' own clock, or `None` when its bars came without
    /// times (ADR 0004).
    pub fn periods_per_year(&self) -> Option<f64> {
        self.candles.clock().and_then(Clock::periods_per_year)
    }

    /// The open time of bar `tick` in UTC nanoseconds, or `None` when the bars came
    /// without times.
    pub fn open_time(&self, tick: usize) -> Option<i64> {
        self.candles.clock().and_then(|clock| clock.time(tick))
    }

    /// The whole candle series as a slice, row by row, for a zero-copy full-series
    /// view of an engine on one instrument. Like `candle_window`, it borrows the
    /// immutable `Arc` buffer with no copy.
    pub fn candles_all(&self) -> &[Candle] {
        self.candles.as_slice()
    }

    /// The `lookback` bars ending at (and including) the current tick, as a
    /// zero-copy borrow into the shared series, for an engine on one instrument.
    /// Clamped to what exists, so early in the series it returns the available
    /// bars. The Python layer turns this into a read-only numpy view (ADR 0040).
    pub fn candle_window(&self, lookback: usize) -> &[Candle] {
        self.candles.window(self.tick + 1, lookback)
    }

    /// The first instrument's last close, the price the one-instrument surface
    /// sizes and checks its orders against.
    fn current_close(&self) -> f64 {
        self.marks[0]
    }

    /// Instrument `i`'s bar on the current row, clipped where the account left the
    /// market on it.
    fn bar_for(&self, i: usize) -> Candle {
        self.clips[i].unwrap_or_else(|| self.candles.bar(self.tick, i))
    }

    /// Advance one bar and resolve the bar, then mark and return the state. At the
    /// end of the data the tick does not advance and the state is returned as is.
    pub fn step(&mut self) -> State {
        let advanced = self.tick + 1 < self.candles.len();
        if advanced {
            self.tick += 1;

            // a random setting draws the bar's way with each event before any of
            // them arises
            if let Some(stream) = self.stream.as_mut() {
                let events = self
                    .candles
                    .funding()
                    .map_or(0, |s| s.owned(self.tick).len());
                self.unseen.draw(stream, events);
            }

            // transfers decided at the last close land first, before funding and
            // before any fill, so cash moved into the margin wallet backs the orders
            // decided with it (ADR 0008)
            self.apply_transfers();

            // funding stamped between the last row and this one, across a closure or
            // a hole in the data, is charged next, at the last marks the path starts
            // from (ADR 0020)
            self.charge_between();

            // 0. the path of the perps carried into the bar is solved before any
            //    order resolves, because the point the margin runs out at is a point
            //    on this bar's own path and everything past it belongs to a market the
            //    account had already left. A clip on the gap closed the account before
            //    the bar opened, so nothing on a perp fills; any other clip cuts every
            //    held perp's bar there, so an order that would have filled beyond it
            //    never triggers. The funding stamped inside the bar counts on the bar
            //    from the open, as the check counts it (ADR 0017)
            self.solve_carried();
            self.notes.clear();
            self.flipped = false;
            self.clear_through = false;
            // read by the resting orders, which no step adds to
            if !self.book.is_empty() {
                for (i, start) in self.starts.iter_mut().enumerate() {
                    let bar = self.candles.bar(self.tick, i);
                    let held = self.account.positions[i].qty.get() != 0.0;
                    *start = if held && printed(&bar) {
                        bar.open
                    } else {
                        f64::NAN
                    };
                }
                self.inside.fill(false);
            }
            self.reach = 0.0;

            // 1. pending market orders fill at the open, and a perp they opened, grew
            //    or flipped has its path solved before the resting orders meet the
            //    bar, so none of them fills past where the account leaves it
            self.resolve_pending();
            let joined = |note: &FillNote| note.opened > path::DUST && note.after != 0.0;
            if !self.book.is_empty() && self.notes.iter().any(joined) {
                self.clip_joined();
            }

            // 2. resting limit and stop orders fill against the bar range, and a
            //    take-profit is held back where the check would find the trigger
            //    without it, since the bar cannot show it came first (ADR 0012)
            self.resolve_resting();

            // 3. the check: the path solved again for the account the fills left,
            //    with the funding stamped inside the bar counted as the setting decides;
            //    an order it leaves standing that takes one bar has had it
            self.check();
            if !self.book.is_empty() {
                self.end_resting_bar();
            }

            // 4. the funding the row owns, at its close and after the check, because
            //    a stamp at the bar's end comes after every price the bar printed:
            //    an account the check closed out, having paid what it counted, holds
            //    nothing at the stamp and pays nothing more. An account a debit takes
            //    to its trigger here is closed at the closes (ADR 0020)
            self.charge_funding();
            self.check_closes();
            // a margin wallet that fees left below zero with no perp held, where no
            // bound stood behind the fill that left it there, is set to zero and the
            // deficit booked as a negative liquidation charge
            if self.account.margin_cash < 0.0 && self.account.perps_held() == 0 {
                if let Some(reporter) = self.reporter.as_mut() {
                    reporter.costs_mut().liquidation += self.account.margin_cash;
                }
                self.account.margin_cash = 0.0;
            }

            for i in 0..self.marks.len() {
                let bar = self.candles.bar(self.tick, i);
                if !printed(&bar) {
                    continue;
                }
                // the part of this bar a round trip still open surely lived through:
                // from its fill to the close when it opened on this bar, and the whole
                // bar otherwise, so an excursion never counts a price that may have
                // printed before the fill (ADR 0023)
                if let Some(trip) = self.reporter.as_mut().and_then(|r| r.trip_mut(i)) {
                    let (low, high) = if trip.open_tick == self.tick {
                        let (start, inside) = self.entered[i];
                        ledger::entry_part(&bar, start, inside)
                    } else {
                        (bar.low, bar.high)
                    };
                    trip.see(low);
                    trip.see(high);
                }
                self.marks[i] = bar.close;
            }
            if self.config.sizes {
                self.sweep_dust();
            }
            if self.reporter.is_some() {
                self.record_ledger();
            }
        }

        let state = self.state();
        // A dead account is terminal, marked at the close, for both markets: its
        // equity, both wallets together, at or below zero. A takeover leaves it
        // there unless the spot wallet holds cash, and a spot account or a perp
        // drained by fees or funding can reach it without a forced close; either way
        // it is a true terminal, not a truncation (ADR 0008).
        if state.equity <= 0.0 {
            self.bust = true;
        }
        // Record only on a real advance, so a driver that keeps calling step past
        // the last bar does not re-append the same equity point and distort stats.
        if advanced {
            if let Some(reporter) = &mut self.reporter {
                let held = self.account.positions.iter().any(|p| p.qty.get() != 0.0);
                reporter.record_equity(state.equity, held);
            }
        }
        state
    }

    /// The reporter, or `None` when `report` was false at construction. It cannot
    /// be turned on afterwards; `reset()` only empties the one already there.
    pub fn reporter(&self) -> Option<&Reporter> {
        self.reporter.as_ref()
    }

    /// Performance statistics for the run, or `None` when reporting is off. The
    /// starting equity is both wallets' opening cash.
    pub fn stats(&self, periods_per_year: f64, risk_free: f64) -> Option<Stats> {
        self.reporter.as_ref().map(|reporter| {
            reporter.stats(
                self.config.quote + self.config.idle_cash,
                periods_per_year,
                risk_free,
                self.fills,
                self.funding_paid,
            )
        })
    }

    /// The round trips: every one closed so far, in the order they closed, then each
    /// one still open, in instrument order, marked at its instrument's last close and
    /// known by having no close. Empty when reporting is off (ADR 0023).
    pub fn round_trips(&self) -> Vec<RoundTrip> {
        let Some(reporter) = &self.reporter else {
            return Vec::new();
        };
        let mut trips = reporter.trades().to_vec();
        for open in reporter.open_trades() {
            let i = open.instrument;
            trips.push(open.marked(self.account.unrealized(i, Price(self.marks[i]))));
        }
        trips
    }

    /// Fills applied since the last reset, whatever the reporter is set to. Zero
    /// beside orders you placed means none of them ever filled.
    pub fn num_fills(&self) -> usize {
        self.fills
    }

    /// The account equity marked at the current bar's close, without building a
    /// full `State`. The batched RL reward reads this per env, so it skips the
    /// open-orders vector a snapshot would allocate. Matches `state().equity`.
    pub fn equity(&self) -> f64 {
        self.account.equity(&self.marks)
    }

    /// The ledger row of the current bar, the account marked at the closes.
    /// `record_row` adds what the bar cost.
    fn ledger_row(&self) -> LedgerRow {
        let equity = self.account.equity(&self.marks);
        let (mut gross, mut net, mut used, mut kept) = (0.0, 0.0, 0.0, 0.0);
        for (i, position) in self.account.positions.iter().enumerate() {
            let qty = position.qty.get();
            if qty == 0.0 {
                continue;
            }
            let mark = self.marks[i];
            gross += (qty * mark).abs();
            net += qty * mark;
            used += self.requirement(i, mark);
            kept += self.maintenance(i, mark);
        }
        let alive = equity > 0.0;
        LedgerRow {
            tick: self.tick,
            equity,
            spot_cash: self.account.spot_cash,
            margin_cash: self.account.margin_cash,
            gross_exposure: gross,
            net_exposure: net,
            leverage: if alive { gross / equity } else { 0.0 },
            margin_used: used,
            maintenance: kept,
            costs: BarCosts::default(),
        }
    }

    /// The state the one-instrument surface reads: the account, and the first
    /// instrument's position and bar.
    #[inline(always)]
    fn state(&self) -> State {
        let bar = self.candles.bar(self.tick, 0);
        let mark = Price(self.marks[0]);
        let position = self.account.positions[0].qty.get();
        State {
            tick_index: self.tick,
            base: if self.account.markets[0] == Market::Spot {
                position
            } else {
                0.0
            },
            quote: self.account.spot_cash + self.account.margin_cash,
            spot_cash: self.account.spot_cash,
            margin_cash: self.account.margin_cash,
            position,
            avg_entry: self.account.positions[0].avg_entry.get(),
            equity: self.account.equity(&self.marks),
            mark_price: mark.get(),
            bar_open: bar.open,
            bar_high: bar.high,
            bar_low: bar.low,
            bar_close: bar.close,
            bar_volume: bar.volume,
            realized_pnl: self.account.realized(),
            unrealized_pnl: if position == 0.0 {
                0.0
            } else {
                self.account.unrealized(0, mark)
            },
            funding_paid: self.funding_paid,
            open_orders: if self.book.is_empty() {
                Vec::new()
            } else {
                self.book.iter().copied().collect()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{cfg, ohlc, perp_cfg, report_cfg, series, universe, ABSENT};
    use super::{Engine, EngineConfig, Intrabar, Liquidation};
    use crate::candles::{printed, Candles, Instrument};
    use crate::clock::Clock;
    use crate::execution::{Impact, Limit, Spread};
    use probatum_core::TimeInForce::Ioc;
    use probatum_core::{Candle, Market, OrderType, Side, State};
    use proptest::prelude::*;

    #[test]
    #[should_panic]
    fn new_rejects_an_empty_series() {
        // a zero-bar series has no bar to reset or step to
        Engine::new(Candles::new(Vec::<Candle>::new()), cfg());
    }

    #[test]
    fn the_engine_reads_back_its_series_and_each_bars_open_time() {
        let day = 86_400 * 1_000_000_000;
        let bars = series().as_slice().to_vec();
        let timed = Candles::on_clock(bars.clone(), Clock::new(vec![0, day, 4 * day]));
        let e = Engine::new(timed, cfg());
        assert_eq!(e.candles_all(), bars.as_slice());
        assert_eq!((e.open_time(2), e.open_time(3)), (Some(4 * day), None));
        assert_eq!(Engine::new(series(), cfg()).open_time(0), None);
    }

    #[test]
    fn stepping_past_the_last_bar_stands_still_and_records_nothing() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step();
        let last = e.step();
        assert_eq!(e.step(), last);
        assert_eq!(e.tick(), 2);
        let reporter = e.reporter().expect("reporting is on");
        assert_eq!(
            (reporter.equity_curve().len(), reporter.ledger().len()),
            (2, 3)
        );
    }

    #[test]
    fn the_stats_start_from_both_wallets_and_count_the_bars_held() {
        // 10,000 in the spot wallet and 5,000 in the margin wallet: one unit bought
        // at the open of 200 is marked at 350, for 15,150 against 15,000, and it is
        // held on both steps
        let config = EngineConfig {
            idle_cash: 5_000.0,
            ..report_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        e.step();
        assert_eq!(e.step().equity, 15_150.0);
        let s = e.stats(1.0, 0.0).expect("reporting is on");
        assert!((s.total_return_pct - 1.0).abs() < 1e-9);
        assert_eq!(s.exposure_pct, 100.0);
    }

    #[test]
    fn the_base_balance_is_the_spot_holding_and_nothing_on_a_perp() {
        for (config, base) in [(cfg(), 2.0), (perp_cfg(), 0.0)] {
            let mut e = Engine::new(series(), config);
            e.reset();
            e.market_buy(2.0);
            let s = e.step();
            assert_eq!((s.position, s.base), (2.0, base));
        }
    }

    #[test]
    fn the_rows_run_straight_across_a_weekend() {
        // the Specification's example: Thursday, Friday and Monday are three rows
        // of the clock, so a market order decided at Friday's close fills at
        // Monday's open, and a stop that open gaps through fills there (ADR 0004)
        let day = 86_400 * 1_000_000_000;
        let bars = Candles::on_clock(
            vec![
                ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
                ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
                ohlc(90.0, 92.0, 89.0, 91.0, 1e6),
            ],
            Clock::new(vec![0, day, 4 * day]),
        );
        let mut decided_friday = Engine::new(bars.clone(), cfg());
        decided_friday.reset();
        decided_friday.step();
        decided_friday.market_buy(1.0);
        let monday = decided_friday.step();
        assert_eq!(monday.avg_entry, 90.0);

        let mut stopped = Engine::new(bars, cfg());
        stopped.reset();
        stopped.market_buy(1.0);
        stopped.step(); // long 1 from Friday's open of 100
        stopped.stop(Side::Sell, 1.0, 95.0, true);
        let monday = stopped.step();
        assert_eq!(monday.position, 0.0);
        assert_eq!(monday.quote, 10_000.0 - 100.0 + 90.0);
        assert_eq!(stopped.periods_per_year(), Some(2.0 * 365.0 / 4.0));
    }

    #[test]
    fn a_series_without_times_has_no_annualisation_to_report() {
        assert_eq!(Engine::new(series(), cfg()).periods_per_year(), None);
    }

    #[test]
    fn reset_starts_flat_at_the_first_bar() {
        let mut e = Engine::new(series(), cfg());
        let s = e.reset();
        assert_eq!(s.tick_index, 0);
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0);
        assert_eq!(s.bar_open, 100.0);
    }

    #[test]
    fn reset_at_starts_the_cursor_at_the_offset() {
        let mut e = Engine::new(series(), cfg());
        let s = e.reset_at(1);
        assert_eq!(s.tick_index, 1);
        assert_eq!(s.bar_open, 200.0); // bar 1
        assert_eq!(s.position, 0.0);
        assert!(!e.done());
    }

    #[test]
    fn reset_at_clamps_past_the_end() {
        let mut e = Engine::new(series(), cfg());
        let s = e.reset_at(99);
        assert_eq!(s.tick_index, 2); // clamped to the last bar
        assert!(e.done());
    }

    #[test]
    fn market_order_fills_on_the_next_bar_open_not_the_decision_bar() {
        let mut e = Engine::new(series(), cfg());
        let s0 = e.reset();
        assert_eq!(s0.tick_index, 0); // looking at bar 0
        e.market_buy(1.0); // decided here
        let s1 = e.step(); // advance to bar 1
        assert_eq!(s1.tick_index, 1);
        // Filled at bar 1 OPEN (200), not bar 0 close (150) or bar 1 close (250).
        assert_eq!(s1.avg_entry, 200.0);
        assert_eq!(s1.position, 1.0);
    }

    #[test]
    fn done_only_at_the_last_bar() {
        let mut e = Engine::new(series(), cfg()); // 3 bars
        e.reset();
        assert!(!e.done());
        e.step();
        assert!(!e.done());
        e.step();
        assert!(e.done());
    }

    #[test]
    fn candle_window_ends_at_the_current_tick_and_clamps_early() {
        let mut e = Engine::new(series(), cfg());
        e.reset(); // tick 0
        let w = e.candle_window(2);
        assert_eq!(w.len(), 1); // only bar 0 exists so far
        assert_eq!(w[0].open, 100.0);
        e.step(); // tick 1
        let w = e.candle_window(2);
        assert_eq!(w.len(), 2); // bars 0 and 1
        assert_eq!(w[0].open, 100.0);
        assert_eq!(w[1].open, 200.0);
    }

    #[test]
    fn step_without_orders_just_advances_and_marks() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        let s = e.step();
        assert_eq!(s.tick_index, 1);
        assert_eq!(s.position, 0.0);
        assert_eq!(s.equity, 10_000.0);
    }

    #[test]
    fn reporter_records_equity_each_step_when_on() {
        let mut config = cfg();
        config.report = true;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.step();
        e.step();
        assert_eq!(e.reporter().unwrap().equity_curve().len(), 2);
    }

    #[test]
    fn stats_count_the_closed_trades() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
        e.close();
        e.step(); // close @ 300, +100: a winning trade
        let stats = e.stats(365.0, 0.0).unwrap();
        assert_eq!(stats.num_trades, 1);
        assert_eq!(stats.win_rate, 1.0);
    }

    #[test]
    fn an_instrument_nobody_trades_leaves_the_run_of_the_one_traded_as_it_was() {
        // a second instrument printing every row and never traded changes nothing
        // the first records, on either market
        let other = ohlc(10.0, 11.0, 9.0, 10.0, 1e6);
        let rows: Vec<_> = series()
            .as_slice()
            .iter()
            .map(|&bar| vec![bar, other])
            .collect();
        for market in [Market::Spot, Market::Perp] {
            let config = EngineConfig {
                market,
                report: true,
                ..cfg()
            };
            let pair = universe(&[market, market], rows.clone());
            let mut runs = [Engine::new(series(), config), Engine::new(pair, config)];
            for e in &mut runs {
                e.reset();
                e.market_buy(2.0);
                e.limit_sell(1.0, 255.0);
                e.step();
                e.close();
                e.step();
            }
            let [alone, pair] = &runs;
            let (a, b) = (alone.reporter().unwrap(), pair.reporter().unwrap());
            assert_eq!(a.fills(), b.fills(), "{market:?}");
            assert_eq!(a.events(), b.events(), "{market:?}");
            assert_eq!(alone.round_trips(), pair.round_trips(), "{market:?}");
            assert_eq!(a.ledger(), b.ledger(), "{market:?}");
            let firsts: Vec<_> = b.legs().iter().step_by(2).copied().collect();
            // compared as printed, where the drift no target asked, NaN, matches
            assert_eq!(
                format!("{:?}", a.legs()),
                format!("{firsts:?}"),
                "{market:?}"
            );
        }
    }

    #[test]
    fn an_instrument_keeps_its_last_mark_through_a_row_it_printed_no_bar_on() {
        // the second instrument prints no bar on the third row: the long it bought at
        // 11 keeps that mark through the row, and a buy decided before it waits for
        // the fourth row's open of 12
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(10.0)],
            vec![flat(100.0), flat(11.0)],
            vec![flat(100.0), ABSENT],
            vec![flat(100.0), flat(12.0)],
        ];
        let mut e = Engine::new(universe(&[Market::Spot, Market::Spot], rows), report_cfg());
        let buy = |e: &mut Engine, size| {
            e.order_on(
                1,
                Side::Buy,
                size,
                OrderType::Market,
                None,
                None,
                false,
                false,
                Ioc,
            )
        };
        e.reset();
        buy(&mut e, 10.0);
        e.step();
        buy(&mut e, 5.0);
        assert_eq!(e.step().equity, 10_000.0);
        let legs = e.reporter().expect("reporting is on").legs();
        assert_eq!((legs[5].position, legs[5].mark), (10.0, 11.0));
        let last = e.step();
        assert_eq!((last.spot_cash, last.equity), (9_830.0, 10_010.0));
    }

    #[test]
    fn a_perp_marks_equity_at_the_close_on_an_open_position() {
        // the only numeric equity assertions were on spot, so a perp marked with the
        // spot formula would have gone unnoticed
        let mut e = Engine::new(series(), perp_cfg()); // quote 100
        e.reset();
        e.market_buy(1.0);
        let s1 = e.step(); // long 1 @ bar 1 open 200, bar 1 close 250
        assert_eq!(s1.equity, 100.0 + 1.0 * (250.0 - 200.0));
        let s2 = e.step(); // bar 2 close 350
        assert_eq!(s2.equity, 100.0 + 1.0 * (350.0 - 200.0));
    }

    /// A random walk built into consistent OHLC bars, violent enough to liquidate a
    /// 10x perp in a single step, because that is the path the guarantees have to
    /// survive rather than the one they are comfortable on.
    fn walk(len: usize) -> impl Strategy<Value = Vec<Candle>> {
        prop::collection::vec(
            (-0.3f64..0.3, 0.0f64..0.25, 0.0f64..0.25, 0.0f64..2_000.0),
            len,
        )
        .prop_map(|steps| {
            let mut price = 100.0;
            steps
                .into_iter()
                .map(|(ret, up, down, volume)| {
                    let open = price;
                    let close = (open * (1.0 + ret)).max(0.01);
                    let bar = ohlc(
                        open,
                        open.max(close) * (1.0 + up),
                        (open.min(close) * (1.0 - down)).max(0.001),
                        close,
                        volume,
                    );
                    price = close;
                    bar
                })
                .collect()
        })
    }

    fn walk_and_actions(
        bars: usize,
        acts: usize,
    ) -> impl Strategy<Value = (Vec<Candle>, Vec<f64>)> {
        (walk(bars), prop::collection::vec(-3.0f64..3.0, acts))
    }

    /// A universe of up to fifty instruments, each spot or a perp on its own walk and
    /// printing no bar on about one row in seven, and actions that each name an
    /// instrument and a signed weight.
    fn universe_and_actions(
        rows: usize,
        acts: usize,
    ) -> impl Strategy<Value = (Vec<Market>, Vec<Candle>, Vec<(usize, f64)>)> {
        (1usize..=50)
            .prop_flat_map(move |n| {
                (
                    prop::collection::vec(any::<bool>(), n),
                    prop::collection::vec(walk(rows), n),
                    prop::collection::vec(prop::collection::vec(0.0f64..1.0, rows), n),
                    prop::collection::vec((0..n, -3.0f64..3.0), acts),
                )
            })
            .prop_map(move |(perps, walks, holes, actions)| {
                let markets = perps
                    .iter()
                    .map(|&perp| if perp { Market::Perp } else { Market::Spot })
                    .collect();
                let mut bars = Vec::with_capacity(rows * walks.len());
                for t in 0..rows {
                    for (walk, holes) in walks.iter().zip(&holes) {
                        bars.push(if holes[t] < 0.15 { ABSENT } else { walk[t] });
                    }
                }
                (markets, bars, actions)
            })
    }

    /// A perp with every cost switched on and reporting, so nothing is left out of
    /// the paths these properties have to hold across.
    ///
    /// `impact` is deliberately extreme. At 0.5 against a bar the order takes all
    /// of, the raw slip is 50%, which is what found ADR 0013: the price left the
    /// bar entirely and the loss it booked was bounded by nothing, so the bad-debt
    /// guarantee fell over from outside the liquidation. Holding a taker price
    /// inside its bar is what makes the property below hold at this coefficient.
    fn every_cost() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 10_000.0,
            idle_cash: 0.0,
            fee_taker: 0.0006,
            fee_maker: 0.0002,
            slippage_bps: 5.0,
            budget: 1.0,
            window: 24,
            max_open_orders: 8,
            report: true,
            max_leverage: 10.0,
            impact: 0.5,
            funding_rate: 0.0001,
            funding_interval: 3,
            intrabar: Intrabar::Pessimistic,
            liquidation: Liquidation::Takeover,
            impact_model: Impact::Linear,
            spread: Spread::None,
            limit: Limit::Touch,
            sizes: false,
        }
    }

    /// One action per bar, and the magnitude picks the ORDER KIND as well as the
    /// size, so the same generator reaches a resting book.
    ///
    /// This drove market orders alone for as long as it existed, which is what
    /// let three separate bad-debt paths live under a property test written to
    /// say there was no fourth one: every one of them needed an order still
    /// resting when the bar resolved, and no run this ever produced had one.
    /// A stop armed away from the close and a limit armed through it are the two
    /// shapes that reach the book, and the protective pair is what a reader of
    /// the guide actually writes (ADR 0018).
    fn drive(engine: &mut Engine, actions: &[f64]) -> Vec<State> {
        let mut seen = Vec::with_capacity(actions.len());
        for action in actions {
            let weight = action.abs();
            let size = engine.qty_from_weight(weight);
            let close = engine.current_close();
            let buying = *action > 0.0;
            if buying {
                engine.market_buy(size);
            } else if *action < 0.0 {
                engine.market_sell(size);
            }
            // a protective stop armed in the SAME decision as the entry, which is
            // what the guide teaches and what nothing here used to place. The
            // trigger is further out than the liquidation distance at this
            // leverage on purpose: that is the case where the account dies before
            // the stop does, so the stop is the fill that reduces a position the
            // margin no longer stands behind (ADR 0018)
            if weight > 1.0 {
                let (side, trigger) = if buying {
                    (Side::Sell, close * 0.90)
                } else {
                    (Side::Buy, close * 1.10)
                };
                engine.stop(side, size, trigger, true);
            }
            // and a marketable resting exit, which prices at its own limit rather
            // than at anything the bar clamped
            if weight > 2.0 {
                if buying {
                    engine.limit_sell(size, close * 0.90);
                } else {
                    engine.limit_buy(size, close * 1.10);
                }
            }
            seen.push(engine.step());
            if engine.done() {
                break;
            }
        }
        seen
    }

    proptest! {
        #[test]
        fn a_perp_never_ends_a_bar_owing_money((bars, actions) in walk_and_actions(50, 40)) {
            // ADRs 0017 and 0018: bad debt is unreachable rather than caught, so no
            // path through a violent series at full leverage may take equity below
            // zero on any bar. The point tests pin the three shapes that broke it;
            // this is the one that says there is no fourth shape
            let mut e = Engine::new(Candles::new(bars), every_cost());
            e.reset();
            for state in drive(&mut e, &actions) {
                prop_assert!(
                    state.equity >= -1e-6,
                    "equity {} at tick {}",
                    state.equity,
                    state.tick_index
                );
            }
        }

        #[test]
        fn a_state_reconciles_its_own_cash_and_position(
            (bars, actions) in walk_and_actions(40, 30),
        ) {
            // the fields the caller reads have to agree with each other: on a perp
            // equity is cash plus the open position's unrealized PnL, and on spot it
            // is cash plus the base holding marked at the close
            for market in [Market::Spot, Market::Perp] {
                let config = EngineConfig { market, ..every_cost() };
                let mut e = Engine::new(Candles::new(bars.clone()), config);
                e.reset();
                for state in drive(&mut e, &actions) {
                    let expected = if market == Market::Spot {
                        state.quote + state.position * state.mark_price
                    } else {
                        state.quote + state.unrealized_pnl
                    };
                    prop_assert!(
                        (state.equity - expected).abs() <= 1e-6 * state.equity.abs().max(1.0),
                        "{:?} equity {} against {}",
                        market,
                        state.equity,
                        expected
                    );
                }
            }
        }

        #[test]
        fn no_state_depends_on_a_bar_after_it(
            (bars, actions) in walk_and_actions(50, 40),
            cut in 5usize..35,
        ) {
            // ADR 0001 promises an order decided on bar i fills against bar i + 1 and
            // never a later one, so rewriting every bar past the cut has to leave
            // every state up to it identical. Lookahead has exactly this shape
            let mut altered = bars.clone();
            for bar in altered.iter_mut().skip(cut + 1) {
                *bar = ohlc(7.0, 900.0, 0.5, 3.0, 5.0);
            }
            let mut first = Engine::new(Candles::new(bars), every_cost());
            first.reset();
            let mut second = Engine::new(Candles::new(altered), every_cost());
            second.reset();
            let left = drive(&mut first, &actions);
            let right = drive(&mut second, &actions);
            for i in 0..cut.min(left.len()).min(right.len()) {
                prop_assert_eq!(&left[i], &right[i], "diverged at index {}", i);
            }
        }

        #[test]
        fn every_statistic_stays_finite_or_deliberately_infinite(
            (bars, actions) in walk_and_actions(50, 40),
            ppy in prop_oneof![
                Just(0.0), Just(-1.0), Just(f64::NAN), Just(365.0), Just(525_600.0)
            ],
        ) {
            // ADR 0030: nothing is ever NaN, and the only
            // infinity is the deliberate one a reward earned against no measured risk
            // goes to. A high annualization is ordinary input, since ADR 0004 reads
            // it off the candles and minute bars are a normal feed
            let mut e = Engine::new(Candles::new(bars), every_cost());
            e.reset();
            drive(&mut e, &actions);
            let s = e.stats(ppy, 0.02).expect("reporting is on");
            for (name, value) in [
                ("total_return_pct", s.total_return_pct),
                ("cagr_pct", s.cagr_pct),
                ("max_drawdown_pct", s.max_drawdown_pct),
                ("volatility_pct", s.volatility_pct),
                ("exposure_pct", s.exposure_pct),
                ("win_rate", s.win_rate),
                ("avg_trade_pct", s.avg_trade_pct),
                ("funding_paid", s.funding_paid),
            ] {
                prop_assert!(value.is_finite(), "{} was {}", name, value);
            }
            for (name, value) in [
                ("sharpe", s.sharpe),
                ("sortino", s.sortino),
                ("calmar", s.calmar),
                ("profit_factor", s.profit_factor),
            ] {
                prop_assert!(!value.is_nan(), "{} was NaN", name);
                prop_assert!(
                    value.is_finite() || value == f64::INFINITY,
                    "{} was {}",
                    name,
                    value
                );
            }
            prop_assert!(s.max_drawdown_pct <= 100.0 + 1e-9);
        }

        #[test]
        fn no_wallet_owes_money_and_every_row_balances_across_a_universe(
            (markets, bars, actions) in universe_and_actions(20, 20),
        ) {
            // the account across many instruments, every cost on and holes in the
            // bars: no wallet ends a bar owing money, and each bar's change in equity
            // is its price PnL less what it cost, from the fills' references and
            // each instrument's marks
            let width = markets.len();
            let specs = markets
                .iter()
                .map(|&market| Instrument { market: Some(market), ..Instrument::default() })
                .collect();
            let config = EngineConfig {
                quote: 5_000.0,
                idle_cash: 5_000.0,
                ..every_cost()
            };
            let universe = Candles::universe(bars.clone(), width, None).with_instruments(specs);
            let mut e = Engine::new(universe, config);
            e.reset();
            let mut last = vec![f64::NAN; width];
            for &(i, action) in &actions {
                for (j, bar) in bars[e.tick() * width..][..width].iter().enumerate() {
                    if printed(bar) {
                        last[j] = bar.close;
                    }
                }
                if last[i].is_finite() {
                    let size = action.abs() * 1_000.0 / last[i];
                    let side = if action > 0.0 { Side::Buy } else { Side::Sell };
                    e.order_on(i, side, size, OrderType::Market, None, None, false, false, Ioc);
                    if action.abs() > 1.0 {
                        let trigger = last[i] * if action > 0.0 { 0.9 } else { 1.1 };
                        let exit = side.opposite();
                        let stop = OrderType::Stop;
                        e.order_on(i, exit, size, stop, None, Some(trigger), true, false, Ioc);
                    }
                }
                e.step();
                if e.done() {
                    break;
                }
            }
            let reporter = e.reporter().expect("reporting is on");
            let (rows, legs) = (reporter.ledger(), reporter.legs());
            for t in 0..rows.len() {
                let (row, leg) = (rows[t], &legs[t * width..][..width]);
                let held = |j: usize| leg[j].position != 0.0;
                let margin = row.margin_cash
                    + (0..width)
                        .filter(|&j| markets[j] == Market::Perp && held(j))
                        .map(|j| leg[j].unrealized)
                        .sum::<f64>();
                let scale = row.equity.abs().max(row.gross_exposure).max(1.0);
                prop_assert!(margin >= -1e-9 * scale, "row {t}: margin equity {margin}");
                prop_assert!(row.spot_cash >= -1e-9 * scale, "row {t}: spot {}", row.spot_cash);
                if t == 0 {
                    continue;
                }
                let (was, before) = (rows[t - 1], &legs[(t - 1) * width..][..width]);
                let mut pnl = 0.0;
                for j in (0..width).filter(|&j| before[j].position != 0.0) {
                    pnl += before[j].position * (leg[j].mark - before[j].mark);
                }
                for fill in reporter.fills().iter().filter(|fill| fill.tick == t) {
                    let mark = leg[fill.instrument].mark;
                    pnl += fill.side.sign() * fill.size * (mark - fill.costs.reference);
                }
                let c = row.costs;
                let costs = c.fees + c.funding + c.spread + c.slippage + c.impact;
                let costs = costs + c.liquidation + c.dust;
                let change = row.equity - was.equity;
                let scale = scale.max(was.equity.abs()).max(was.gross_exposure);
                prop_assert!(
                    (change - (pnl - costs)).abs() <= 1e-9 * scale,
                    "row {}: equity moved {}, price PnL {} less costs {}", t, change, pnl, costs
                );
            }
        }
    }
}
