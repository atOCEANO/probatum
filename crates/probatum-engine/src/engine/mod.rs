//! The single-env step state machine. `reset()` positions at the first bar;
//! `step()` advances exactly one bar, resolves the orders placed since the last
//! step against the NEW bar, marks the account, and returns the new `State`. An
//! order decided on one bar fills on the next, never the bar the decision saw.
//!
//! Within a step the order of events follows the bar. Transfers decided at the last
//! close land first, then funding stamped between the last row and this one, at
//! the last mark, and then the fence. The price the position's margin runs out at
//! is worked out before any order is resolved: a bar that OPENED past it killed
//! the account before it opened, so the forced close happens first and nothing on
//! that bar fills, and any other bar is clipped there, so an order that would have
//! filled beyond the fence never triggers. Then, against that clipped candle: 1.
//! pending market orders fill at the open, 2. resting limit and stop orders fill
//! against the range, 3. liquidation is checked at the bar's own adverse extreme,
//! counting any debit stamped inside the bar, 4. the funding the row owns is
//! charged at its close on the position still held, and an account a debit takes
//! to zero is closed there. Equity is then marked at the close. A fill that reduces
//! a perp is bounded once more as it lands, because the fence was read from the
//! position carried into the bar and one grown or opened during the bar has none
//! of its own (ADRs 0008, 0017, 0018, 0020).

#[cfg(test)]
mod fixtures;
mod funding;
mod ledger;
mod liquidation;
mod margin;
mod orders;
mod resolution;
mod transfers;

use ledger::Moment;

use probatum_core::{
    Account, Bps, Candle, FlatCostModel, Market, Order, Price, RestingOrderBook, State, Wallet,
};

use crate::candles::Candles;
use crate::clock::Clock;
use crate::execution::FillModel;
use crate::reporter::{Reporter, RoundTrip};
use crate::stats::Stats;

/// A position moved toward zero by less than this counts as no close, mirroring
/// the position's own dust epsilon so a non-representable fractional close does
/// not log a phantom trade.
const CLOSE_EPS: f64 = 1e-9;

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
    pub max_fill_fraction: f64,
    pub max_open_orders: usize,
    pub report: bool,
    /// Perp margin cap: a fill may not grow the position past `max_leverage`
    /// times equity in notional. Zero means no cap; the shipped default is a
    /// finite 10x (ADR 0015). Ignored on spot.
    pub max_leverage: f64,
    /// Market-impact coefficient for taker fills (ADR 0013). Zero disables it.
    pub impact: f64,
    /// Perp funding rate charged per event on the position notional; a long pays a
    /// positive rate, a short receives (ADR 0020). Ignored on spot.
    pub funding_rate: f64,
    /// Bars between funding events. Zero disables funding (ADR 0020).
    pub funding_interval: usize,
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
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: false,
            max_leverage: 10.0,
            impact: 0.0,
            funding_rate: 0.0,
            funding_interval: 0,
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
    /// Transfers decided since the last step, in order, each an amount and the
    /// wallet it goes into.
    transfers: Vec<(f64, Wallet)>,
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
    /// or below zero, where a perp liquidation leaves it unless the spot wallet
    /// holds cash (ADR 0008). A terminal event for the RL env. Cleared on reset.
    bust: bool,
}

impl Engine {
    /// Build an engine over `candles`. Panics if `candles` is empty, since a
    /// zero-bar series has no bar to reset or step to. Call `reset()` to get the
    /// first state.
    pub fn new(candles: Candles, config: EngineConfig) -> Engine {
        assert!(!candles.is_empty(), "candles must have at least one row");
        Engine {
            fill_model: FillModel {
                slippage_bps: Bps(config.slippage_bps),
                max_fill_fraction: config.max_fill_fraction,
                impact: config.impact,
            },
            cost: FlatCostModel {
                fee_taker: config.fee_taker,
                fee_maker: config.fee_maker,
            },
            account: Account::new(config.market, config.quote).with_idle(config.idle_cash),
            book: RestingOrderBook::new(config.max_open_orders),
            reporter: if config.report {
                Some(Reporter::new())
            } else {
                None
            },
            pending: Vec::new(),
            transfers: Vec::new(),
            tick: 0,
            id_counter: 1,
            fills: 0,
            funding_paid: 0.0,
            bust: false,
            candles,
            config,
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
        self.account =
            Account::new(self.config.market, self.config.quote).with_idle(self.config.idle_cash);
        self.book.cancel_all();
        self.pending.clear();
        self.transfers.clear();
        self.tick = start_tick.min(self.candles.len().saturating_sub(1));
        // The id counter deliberately survives a reset. Restarting it made an id
        // handed out in one episode name a live order in the next, so a handle
        // carried across a reset cancelled somebody else's order (ADR 0010).
        self.fills = 0;
        self.funding_paid = 0.0;
        self.bust = false;
        if self.reporter.is_some() {
            self.reporter = Some(Reporter::new());
        }
        self.state()
    }

    /// True when the account died since the last reset: its equity, both wallets
    /// together, reached zero, on either market. A perp liquidation does that
    /// unless the spot wallet holds cash. The RL env's terminal signal (ADR 0008).
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

    /// The whole candle series as a slice, for a zero-copy full-series view. Like
    /// `candle_window`, it borrows the immutable `Arc` buffer with no copy.
    pub fn candles_all(&self) -> &[Candle] {
        self.candles.as_slice()
    }

    /// The `lookback` bars ending at (and including) the current tick, as a
    /// zero-copy borrow into the shared series. Clamped to what exists, so early
    /// in the series it returns the available bars. The Python layer turns this
    /// into a read-only numpy view (ADR 0040).
    pub fn candle_window(&self, lookback: usize) -> &[Candle] {
        self.candles.window(self.tick + 1, lookback)
    }

    fn current_close(&self) -> f64 {
        self.candles.get(self.tick).map_or(0.0, |c| c.close)
    }

    /// Advance one bar and resolve the bar, then mark and return the state. At the
    /// end of the data the tick does not advance and the state is returned as is.
    pub fn step(&mut self) -> State {
        let advanced = self.tick + 1 < self.candles.len();
        if advanced {
            let last = self.current_close();
            self.tick += 1;
            let bar = self.candles.get(self.tick).expect("tick in range");

            // transfers decided at the last close land first, before funding and
            // before any fill, so cash moved into the margin wallet backs the orders
            // decided with it (ADR 0008)
            self.apply_transfers(last);

            // funding stamped between the last row and this one, across a closure or
            // a hole in the data, is charged next, at the last mark and before the
            // path, so a debit that takes the account to its trigger liquidates it
            // at the start of the gap, there if it is dead already and on the gap
            // below if the open lies past the fence (ADR 0020)
            self.charge_between(last);
            if self.account.is_bust_at(Price(last)) {
                self.liquidate_at(Price(last), Moment::BeforeOpen);
            }

            // 0. the price a position's margin runs out at is a point on this bar's
            //    own path, and everything past it belongs to a market the account
            //    had already been closed out of. So the fence is worked out before
            //    any order is resolved. A bar that OPENED past it killed the account
            //    before it opened, so it is closed first and nothing on the bar
            //    fills; on any other bar the orders are resolved against a candle
            //    clipped at the fence, so an exit that would have filled beyond it
            //    simply never triggers. Checking only afterwards, as this used to,
            //    left ADR 0018's guarantee resting on the position still being open
            //    at the end of the bar: a `close()` on a bar that gapped past the
            //    liquidation booked the whole gap and left the account owing money,
            //    and a partial close left a "liquidation" priced outside the bar
            //    that booked a profit (ADR 0017).
            let long = self.account.position.qty.get() >= 0.0;
            let fence = self.liquidation_fence(&bar);
            let gapped = fence.is_some_and(|f| if long { bar.open <= f } else { bar.open >= f });
            if gapped {
                self.liquidate(Moment::BeforeOpen);
            }
            let resolving = match fence {
                Some(f) if !gapped => Engine::fenced(&bar, f, long),
                _ => bar,
            };

            // 1. pending market orders fill at the open
            self.resolve_pending(&resolving);

            // 2. resting limit and stop orders fill against the bar range
            self.resolve_resting(&resolving);

            // 3. liquidation at the bar's adverse extreme (long at the low, short
            //    at the high), with the debits stamped inside the bar counted toward
            //    it; an account it liquidates pays what it counted and nothing more
            let adverse = if self.account.position.qty.get() >= 0.0 {
                bar.low
            } else {
                bar.high
            };
            let debits = self.debits_inside(bar.close);
            if self.dead_at(adverse, debits) {
                self.pay_debits_inside(bar.close);
                self.liquidate(Moment::InBar);
            }

            // 4. the funding the row owns, at its close and after the check, because
            //    a stamp at the bar's end comes after every price the bar printed:
            //    an account the check closed out holds nothing at the stamp and pays
            //    nothing. emsl charged it before the check, at the clipped close,
            //    which liquidated accounts on payments not yet made. An account a
            //    debit takes to its trigger here is closed at the close (ADR 0020)
            self.charge_funding(bar.close);
            if self.account.is_bust_at(Price(bar.close)) {
                self.liquidate_at(Price(bar.close), Moment::AtClose);
            }

            // the part of this bar a round trip still open lived through: from its
            // fill to the close when it opened on this bar, the whole bar otherwise
            if let Some(trip) = self.reporter.as_mut().and_then(Reporter::trip_mut) {
                if trip.open_tick == self.tick {
                    trip.see(bar.close);
                } else {
                    trip.see(bar.low);
                    trip.see(bar.high);
                }
            }
        }

        let state = self.state();
        // A dead account is terminal, marked at the close, for both markets: its
        // equity, both wallets together, at or below zero. A perp liquidation leaves
        // it there unless the spot wallet holds cash, and a spot account or a perp
        // drained by fees or funding can reach it without a forced close; either way
        // it is a true terminal, not a truncation (ADR 0008).
        if state.equity <= 0.0 {
            self.bust = true;
        }
        // Record only on a real advance, so a driver that keeps calling step past
        // the last bar does not re-append the same equity point and distort stats.
        if advanced {
            if let Some(reporter) = &mut self.reporter {
                reporter.record_equity(state.equity, state.position != 0.0);
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

    /// The round trips: every one closed so far, in the order they closed, then the
    /// one still open, marked at the current close and known by having no close.
    /// Empty when reporting is off (ADR 0023).
    pub fn round_trips(&self) -> Vec<RoundTrip> {
        let Some(reporter) = &self.reporter else {
            return Vec::new();
        };
        let mut trips = reporter.trades().to_vec();
        if let Some(open) = reporter.open_trade() {
            let mark = Price(self.current_close());
            trips.push(open.marked(self.account.unrealized(mark)));
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
        let mark = self.candles.get(self.tick).expect("tick in range").close;
        self.account.equity(Price(mark))
    }

    fn state(&self) -> State {
        let bar = self.candles.get(self.tick).expect("tick in range");
        let mark = Price(bar.close);
        let position = self.account.position.qty.get();
        State {
            tick_index: self.tick,
            base: if self.config.market == Market::Spot {
                position
            } else {
                0.0
            },
            quote: self.account.spot_cash + self.account.margin_cash,
            spot_cash: self.account.spot_cash,
            margin_cash: self.account.margin_cash,
            position,
            avg_entry: self.account.position.avg_entry.get(),
            equity: self.account.equity(mark),
            mark_price: bar.close,
            bar_open: bar.open,
            bar_high: bar.high,
            bar_low: bar.low,
            bar_close: bar.close,
            bar_volume: bar.volume,
            realized_pnl: self.account.realized(),
            unrealized_pnl: self.account.unrealized(mark),
            funding_paid: self.funding_paid,
            open_orders: self.book.iter().copied().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{cfg, ohlc, perp_cfg, report_cfg, series};
    use super::{Engine, EngineConfig};
    use crate::candles::Candles;
    use crate::clock::Clock;
    use probatum_core::{Candle, Market, Side, State};
    use proptest::prelude::*;

    #[test]
    #[should_panic]
    fn new_rejects_an_empty_series() {
        // a zero-bar series has no bar to reset or step to
        Engine::new(Candles::new(Vec::<Candle>::new()), cfg());
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
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: true,
            max_leverage: 10.0,
            impact: 0.5,
            funding_rate: 0.0001,
            funding_interval: 3,
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
    }
}
