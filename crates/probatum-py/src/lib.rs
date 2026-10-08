//! probatum-py: the PyO3 binding layer, compiled to the `probatum._probatum` extension. This
//! is the only Python-linked crate, and it is a thin shell over `probatum-engine`: it
//! parses arguments, copies the candles once from a numpy array into the
//! Rust-owned series, and hands `State` back as a typed record. No simulation logic
//! lives here.
//!
//! The crate-level allow silences a clippy false positive: pyo3's `#[pymethods]`
//! macro emits an identity `PyErr` conversion in the trampolines it generates for
//! `PyResult`-returning methods, which clippy reports against our return types.
//! The conversion is the macro's, not ours.
#![allow(clippy::useless_conversion)]

use numpy::ndarray::{Array3, ArrayView2};
use numpy::npyffi::flags::NPY_ARRAY_WRITEABLE;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use probatum_core::candle::floats;
use probatum_core::{Candle, Market, Order, OrderId, OrderType, Side, State, TimeInForce, Wallet};
use probatum_engine::reporter::{EventReason, FillRecord, LedgerRow, OrderEvent, Refusal};
use probatum_engine::schedule::every;
use probatum_engine::{
    Action, Candles, Clock, Engine as BarEngine, EngineConfig, EnvBatch, OrderRef, RoundTrip,
    Script, Stats,
};

/// Parse a market string into the core enum, case-insensitively.
fn parse_market(market: &str) -> PyResult<Market> {
    match market.to_ascii_lowercase().as_str() {
        "spot" => Ok(Market::Spot),
        "perp" => Ok(Market::Perp),
        other => Err(PyValueError::new_err(format!(
            "market must be 'spot' or 'perp', got '{other}'"
        ))),
    }
}

/// Parse a side string into the core enum, case-insensitively.
fn parse_side(side: &str) -> PyResult<Side> {
    match side.to_ascii_lowercase().as_str() {
        "buy" => Ok(Side::Buy),
        "sell" => Ok(Side::Sell),
        other => Err(PyValueError::new_err(format!(
            "side must be 'buy' or 'sell', got '{other}'"
        ))),
    }
}

/// Parse an order-type string into the core enum, case-insensitively.
fn parse_order_type(kind: &str) -> PyResult<OrderType> {
    match kind.to_ascii_lowercase().as_str() {
        "market" => Ok(OrderType::Market),
        "limit" => Ok(OrderType::Limit),
        "stop" => Ok(OrderType::Stop),
        other => Err(PyValueError::new_err(format!(
            "type must be 'market', 'limit', or 'stop', got '{other}'"
        ))),
    }
}

/// Parse a time-in-force string into the core enum, case-insensitively.
fn parse_tif(tif: &str) -> PyResult<TimeInForce> {
    match tif.to_ascii_uppercase().as_str() {
        "GTC" => Ok(TimeInForce::Gtc),
        "IOC" => Ok(TimeInForce::Ioc),
        "FOK" => Ok(TimeInForce::Fok),
        other => Err(PyValueError::new_err(format!(
            "tif must be 'GTC', 'IOC', or 'FOK', got '{other}'"
        ))),
    }
}

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "buy",
        Side::Sell => "sell",
    }
}

fn kind_str(kind: OrderType) -> &'static str {
    match kind {
        OrderType::Market => "market",
        OrderType::Limit => "limit",
        OrderType::Stop => "stop",
    }
}

/// Copy a `(T, 5)` OHLCV array into an owned candle series, on the clock `times`
/// when given. Columns are open, high, low, close, volume. This is the one input
/// copy; observation windows handed back later are the zero-copy path.
fn candles_from_array(
    array: PyReadonlyArray2<'_, f64>,
    times: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<Candles> {
    let view = array.as_array();
    let cols = view.ncols();
    if cols != 5 {
        return Err(PyValueError::new_err(format!(
            "candles must have 5 columns (open, high, low, close, volume), got {cols}"
        )));
    }
    let bars: Vec<Candle> = view
        .outer_iter()
        .map(|row| Candle {
            open: row[0],
            high: row[1],
            low: row[2],
            close: row[3],
            volume: row[4],
        })
        .collect();
    if bars.is_empty() {
        return Err(PyValueError::new_err("candles must have at least one row"));
    }
    // Reject NaN and infinity: the spec's input contract says NaN rows are rejected,
    // and a non-finite candle would silently poison every mark and equity downstream.
    for (i, bar) in bars.iter().enumerate() {
        if !(bar.open.is_finite()
            && bar.high.is_finite()
            && bar.low.is_finite()
            && bar.close.is_finite()
            && bar.volume.is_finite())
        {
            return Err(PyValueError::new_err(format!(
                "candles contain a non-finite value at row {i}; clean NaN and inf rows first"
            )));
        }
        // and that the four prices are a bar. The taker clamp bounds a fill to
        // what the bar traded by taking price.min(high) on a buy and
        // price.max(low) on a sell, so on a row whose prices are out of order
        // that min and max move the fill TOWARD the account instead of away, and
        // a dead flat market pays out. The reachable cause is not a typo: a feed
        // that ships low/high/open/close, sliced positionally, satisfies
        // high >= low on every row and still mints money, so the whole ordering
        // is what has to be checked rather than the one pair (ADR 0007)
        if !(bar.low <= bar.open
            && bar.open <= bar.high
            && bar.low <= bar.close
            && bar.close <= bar.high)
        {
            return Err(PyValueError::new_err(format!(
                "candles are not a bar at row {i}: open {}, high {}, low {}, close {}; \
                 need low <= open <= high and low <= close <= high, so check the \
                 column order, which is open, high, low, close, volume",
                bar.open, bar.high, bar.low, bar.close
            )));
        }
    }
    match times {
        None => Ok(Candles::new(bars)),
        Some(times) => {
            let clock = clock_from(times, Some(bars.len()))?;
            Ok(Candles::on_clock(bars, clock))
        }
    }
}

/// `series` with a time-stamped funding source placed on its clock: every
/// `funding_hours` from 00:00 UTC at `funding_rate`, or the events `funding_times`
/// and `funding_rates`. Refused unless at most one source is given, counting
/// `funding_interval`, and a stamped one comes with the clock it needs (ADR 0020).
fn place_funding(
    series: Candles,
    funding_rate: f64,
    funding_interval: usize,
    funding_hours: f64,
    funding_times: Option<PyReadonlyArray1<'_, i64>>,
    funding_rates: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Candles> {
    if !(funding_hours.is_finite() && funding_hours >= 0.0) {
        return Err(PyValueError::new_err(format!(
            "funding_hours: {funding_hours}; pass the hours between events, or 0.0 for none"
        )));
    }
    let stamped = match (funding_times, funding_rates) {
        (None, None) => None,
        (Some(times), Some(rates)) => {
            let (times, rates) = (times.as_array(), rates.as_array());
            if times.len() != rates.len() {
                return Err(PyValueError::new_err(format!(
                    "funding_rates: {} for {} funding_times; pass one rate per time",
                    rates.len(),
                    times.len()
                )));
            }
            if let Some(row) = rates.iter().position(|rate| !rate.is_finite()) {
                return Err(PyValueError::new_err(format!(
                    "funding_rates: row {row} is {}; pass finite rates",
                    rates[row]
                )));
            }
            Some(
                times
                    .iter()
                    .copied()
                    .zip(rates.iter().copied())
                    .collect::<Vec<_>>(),
            )
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(PyValueError::new_err(
                "funding_times: given without funding_rates, or the other way round; pass \
                 both or neither",
            ))
        }
    };
    let sources = [funding_interval > 0, funding_hours > 0.0, stamped.is_some()];
    if sources.iter().filter(|&&given| given).count() > 1 {
        return Err(PyValueError::new_err(
            "funding: more than one source; pass funding_interval, funding_hours or \
             funding_times, one of them",
        ));
    }
    if funding_hours == 0.0 && stamped.is_none() {
        return Ok(series);
    }
    let events = match series.clock() {
        None => {
            return Err(PyValueError::new_err(
                "times: none, and funding_hours and funding_times place events on the \
                 clock; pass times=",
            ))
        }
        Some(clock) => stamped.unwrap_or_else(|| every(clock, funding_hours, funding_rate)),
    };
    Ok(series.with_funding(&events))
}

/// The clock `times` describes, each bar's open in UTC nanoseconds. Refused unless
/// every time comes after the one before it and, when `rows` is given, there is one
/// time per row (ADR 0004).
fn clock_from(times: PyReadonlyArray1<'_, i64>, rows: Option<usize>) -> PyResult<Clock> {
    let times = times.as_array();
    if let Some(rows) = rows {
        if times.len() != rows {
            return Err(PyValueError::new_err(format!(
                "times: {} for {rows} bars; pass one open time per bar",
                times.len()
            )));
        }
    }
    if let Some(row) = (1..times.len()).find(|&row| times[row] <= times[row - 1]) {
        return Err(PyValueError::new_err(format!(
            "times: row {row} opens at {}, not after row {} at {}; pass the bars in \
             time order, one per time",
            times[row],
            row - 1,
            times[row - 1]
        )));
    }
    Ok(Clock::new(times.to_vec()))
}

/// The largest resting-order book the caller may ask for. Slots are allocated up
/// front, once per env, so an unbounded request aborts the process on allocation
/// failure instead of raising; far above any real strategy (ADR 0007).
const MAX_OPEN_ORDERS_LIMIT: usize = 4096;

/// Reject a zero observation window. The batched gathers chunk their output buffer
/// by the window, and rayon asserts a chunk size is nonzero, so a zero window used
/// to reach Python as a PanicException, which is not an `Exception` and so escapes
/// an ordinary `except` (ADR 0007).
fn check_window(window: usize) -> PyResult<()> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be at least 1, got 0"));
    }
    Ok(())
}

/// Reject a configuration scalar that is not finite. The candle array is already
/// checked for exactly this, and a NaN knob is no less poisonous: a NaN quote makes
/// every state field NaN, and a NaN `max_fill_fraction` deletes the volume cap
/// because `f64::min` keeps the other operand (ADR 0007).
fn finite(name: &str, value: f64) -> PyResult<f64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite number, got {value}"
        )));
    }
    Ok(value)
}

/// Reject a configuration scalar that is not finite, or that is below `low`.
fn at_least(name: &str, value: f64, low: f64) -> PyResult<f64> {
    finite(name, value)?;
    if value < low {
        return Err(PyValueError::new_err(format!(
            "{name} must be at least {low}, got {value}"
        )));
    }
    Ok(value)
}

/// The opening cash as `quote` takes it: a number for the wallet the market settles
/// in, or a dict naming each wallet's, `{"spot": s, "margin": m}` (ADR 0008).
enum Opening {
    Cash(f64),
    Wallets { spot: f64, margin: f64 },
}

impl<'a, 'py> FromPyObject<'a, 'py> for Opening {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Opening> {
        let Ok(wallets) = obj.cast::<PyDict>() else {
            return Ok(Opening::Cash(obj.extract()?));
        };
        let (mut spot, mut margin) = (0.0, 0.0);
        for (name, cash) in wallets.iter() {
            let name: String = name.extract()?;
            match name.as_str() {
                "spot" => spot = cash.extract()?,
                "margin" => margin = cash.extract()?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "quote: a wallet named {name:?}; name spot and margin"
                    )))
                }
            }
        }
        Ok(Opening::Wallets { spot, margin })
    }
}

/// The wallet a transfer of `amount` goes `to`, once the amount is known to be cash
/// that can move. `at` leads each error, to say where in a script it came from.
fn transfer_to(at: &str, amount: f64, to: &str) -> PyResult<Wallet> {
    if !amount.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{at}amount: {amount} is not a finite number; pass the cash to move"
        )));
    }
    if amount < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{at}amount: {amount} is negative; pass the cash to move and name the wallet \
             it goes into with to"
        )));
    }
    match to.to_ascii_lowercase().as_str() {
        "spot" => Ok(Wallet::Spot),
        "margin" => Ok(Wallet::Margin),
        _ => Err(PyValueError::new_err(format!(
            "{at}to: {to:?} is not a wallet; use spot or margin"
        ))),
    }
}

/// Reject a fee rate at or below -1. A rebate larger than the notional turns the
/// spot cash clamp's `1 + rate` divisor non-positive and mints equity out of a
/// fill that never happened. Its own function because the per-env arrays need the
/// identical rule, and a rule written twice is a rule two paths will disagree
/// about, which is exactly what happened.
fn a_rebate(name: &str, rate: f64) -> PyResult<f64> {
    if finite(name, rate)? <= -1.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be greater than -1.0 (a rebate cannot exceed the notional), got {rate}"
        )));
    }
    Ok(rate)
}

/// Validate every engine knob and build the config. One place, so `Engine` and
/// `Batch` cannot drift apart on what they accept (ADR 0007).
#[allow(clippy::too_many_arguments)]
fn build_config(
    market: &str,
    quote: Opening,
    fee_taker: f64,
    fee_maker: f64,
    slippage_bps: f64,
    max_fill_fraction: f64,
    max_open_orders: usize,
    report: bool,
    leverage: f64,
    impact: f64,
    funding_rate: f64,
    funding_interval: usize,
) -> PyResult<EngineConfig> {
    if max_open_orders == 0 || max_open_orders > MAX_OPEN_ORDERS_LIMIT {
        return Err(PyValueError::new_err(format!(
            "max_open_orders must be between 1 and {MAX_OPEN_ORDERS_LIMIT}, got {max_open_orders}"
        )));
    }
    for (name, rate) in [("fee_taker", fee_taker), ("fee_maker", fee_maker)] {
        a_rebate(name, rate)?;
    }
    let market = parse_market(market)?;
    let (quote, idle_cash) = match quote {
        Opening::Cash(cash) => (at_least("quote", cash, 0.0)?, 0.0),
        Opening::Wallets { spot, margin } => {
            let spot = at_least("quote[\"spot\"]", spot, 0.0)?;
            let margin = at_least("quote[\"margin\"]", margin, 0.0)?;
            match market {
                Market::Spot => (spot, margin),
                Market::Perp => (margin, spot),
            }
        }
    };
    Ok(EngineConfig {
        market,
        quote,
        idle_cash,
        fee_taker,
        fee_maker,
        // slippage and impact are adverse by definition (ADR 0013); a
        // negative rate would fill better than the market and mint equity
        slippage_bps: at_least("slippage_bps", slippage_bps, 0.0)?,
        max_fill_fraction: {
            let v = at_least("max_fill_fraction", max_fill_fraction, 0.0)?;
            if v <= 0.0 {
                return Err(PyValueError::new_err(
                    "max_fill_fraction must be greater than 0, got 0",
                ));
            }
            v
        },
        max_open_orders,
        report,
        max_leverage: at_least("leverage", leverage, 0.0)?,
        impact: at_least("impact", impact, 0.0)?,
        funding_rate: finite("funding_rate", funding_rate)?,
        funding_interval,
    })
}

/// Read an optional per-env cost-override array, checked to length `num_envs`.
/// `None` leaves the batch on its shared scalar for that field (ADR 0038).
fn per_env_vec(
    name: &str,
    num_envs: usize,
    array: &Option<PyReadonlyArray1<'_, f64>>,
    check: impl Fn(&str, f64) -> PyResult<f64>,
) -> PyResult<Option<Vec<f64>>> {
    match array {
        Some(a) => {
            let slice = a.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{name} must be a contiguous 1-D float64 array"))
            })?;
            if slice.len() != num_envs {
                return Err(PyValueError::new_err(format!(
                    "{name} length {} must equal num_envs {num_envs}",
                    slice.len()
                )));
            }
            // the same rule the scalar gets, one entry at a time. This
            // checked contiguity and length and nothing else, so a per-env array
            // was the one way past every range guard in `build_config`: a fee of
            // -2.0 reached an env and minted equity the scalar path refuses
            for (env, value) in slice.iter().enumerate() {
                check(&format!("{name}[{env}]"), *value)?;
            }
            Ok(Some(slice.to_vec()))
        }
        None => Ok(None),
    }
}

fn order_to_dict<'py>(py: Python<'py>, order: &Order) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", order.id.0)?;
    d.set_item("side", side_str(order.side))?;
    d.set_item("kind", kind_str(order.kind))?;
    d.set_item("price", order.price.map(|p| p.get()))?;
    d.set_item("trigger", order.trigger.map(|t| t.get()))?;
    d.set_item("size", order.size.get())?;
    d.set_item("filled", order.filled.get())?;
    d.set_item("remaining", order.remaining().get())?;
    // only resting orders are ever handed out, so this is a constant; the key stays
    // because the API documents it
    d.set_item("status", "resting")?;
    d.set_item("reduce_only", order.reduce_only)?;
    d.set_item("post_only", order.post_only)?;
    Ok(d)
}

/// The account and its bar as a step left them, read by attribute or by key:
/// `state.position` and `state["position"]` are the same number. The resting
/// orders become dicts only when they are read. Two states are equal when every
/// field is, as the dicts they replace were, and the class is a mapping so that
/// no sequence protocol is read off its `__getitem__`.
#[pyclass(frozen, eq, mapping, name = "State")]
#[derive(PartialEq)]
struct PyState {
    #[pyo3(get)]
    tick_index: usize,
    #[pyo3(get)]
    base: f64,
    #[pyo3(get)]
    quote: f64,
    #[pyo3(get)]
    spot_cash: f64,
    #[pyo3(get)]
    margin_cash: f64,
    #[pyo3(get)]
    position: f64,
    #[pyo3(get)]
    avg_entry: f64,
    #[pyo3(get)]
    equity: f64,
    #[pyo3(get)]
    mark_price: f64,
    #[pyo3(get)]
    bar_open: f64,
    #[pyo3(get)]
    bar_high: f64,
    #[pyo3(get)]
    bar_low: f64,
    #[pyo3(get)]
    bar_close: f64,
    #[pyo3(get)]
    bar_volume: f64,
    #[pyo3(get)]
    realized_pnl: f64,
    #[pyo3(get)]
    unrealized_pnl: f64,
    #[pyo3(get)]
    funding_paid: f64,
    orders: Vec<Order>,
}

impl From<State> for PyState {
    fn from(state: State) -> PyState {
        PyState {
            tick_index: state.tick_index,
            base: state.base,
            quote: state.quote,
            spot_cash: state.spot_cash,
            margin_cash: state.margin_cash,
            position: state.position,
            avg_entry: state.avg_entry,
            equity: state.equity,
            mark_price: state.mark_price,
            bar_open: state.bar_open,
            bar_high: state.bar_high,
            bar_low: state.bar_low,
            bar_close: state.bar_close,
            bar_volume: state.bar_volume,
            realized_pnl: state.realized_pnl,
            unrealized_pnl: state.unrealized_pnl,
            funding_paid: state.funding_paid,
            orders: state.open_orders,
        }
    }
}

#[pymethods]
impl PyState {
    /// The resting limit and stop orders, each a dict.
    #[getter]
    fn open_orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for order in &self.orders {
            list.append(order_to_dict(py, order)?)?;
        }
        Ok(list)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        let value = match key {
            "tick_index" => return Ok(self.tick_index.into_pyobject(py)?.into_any()),
            "open_orders" => return Ok(self.open_orders(py)?.into_any()),
            "base" => self.base,
            "quote" => self.quote,
            "spot_cash" => self.spot_cash,
            "margin_cash" => self.margin_cash,
            "position" => self.position,
            "avg_entry" => self.avg_entry,
            "equity" => self.equity,
            "mark_price" => self.mark_price,
            "bar_open" => self.bar_open,
            "bar_high" => self.bar_high,
            "bar_low" => self.bar_low,
            "bar_close" => self.bar_close,
            "bar_volume" => self.bar_volume,
            "realized_pnl" => self.realized_pnl,
            "unrealized_pnl" => self.unrealized_pnl,
            "funding_paid" => self.funding_paid,
            _ => return Err(PyKeyError::new_err(key.to_string())),
        };
        Ok(value.into_pyobject(py)?.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "State(tick_index={}, position={}, equity={})",
            self.tick_index, self.position, self.equity
        )
    }
}

fn states_to_list<'py>(py: Python<'py>, states: Vec<State>) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for state in states {
        list.append(Bound::new(py, PyState::from(state))?)?;
    }
    Ok(list)
}

fn stats_to_dict<'py>(py: Python<'py>, stats: &Stats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("total_return_pct", stats.total_return_pct)?;
    d.set_item("net_profit_pct", stats.net_profit_pct)?;
    d.set_item("cagr_pct", stats.cagr_pct)?;
    d.set_item("sharpe", stats.sharpe)?;
    d.set_item("sortino", stats.sortino)?;
    d.set_item("calmar", stats.calmar)?;
    d.set_item("max_drawdown_pct", stats.max_drawdown_pct)?;
    d.set_item("volatility_pct", stats.volatility_pct)?;
    d.set_item("exposure_pct", stats.exposure_pct)?;
    d.set_item("win_rate", stats.win_rate)?;
    d.set_item("profit_factor", stats.profit_factor)?;
    d.set_item("num_trades", stats.num_trades)?;
    d.set_item("avg_trade_pct", stats.avg_trade_pct)?;
    d.set_item("num_fills", stats.num_fills)?;
    d.set_item("funding_paid", stats.funding_paid)?;
    Ok(d)
}

fn trade_to_dict<'py>(
    py: Python<'py>,
    trade: &RoundTrip,
    engine: &BarEngine,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("instrument", 0)?;
    d.set_item(
        "side",
        if trade.side == Side::Buy {
            "long"
        } else {
            "short"
        },
    )?;
    d.set_item("size", trade.size)?;
    d.set_item("entry", trade.entry)?;
    d.set_item("exit", trade.exit)?;
    d.set_item("open_bar", trade.open_tick)?;
    d.set_item("close_bar", trade.close_tick)?;
    d.set_item("open_time", engine.open_time(trade.open_tick))?;
    d.set_item(
        "close_time",
        trade.close_tick.and_then(|tick| engine.open_time(tick)),
    )?;
    d.set_item("realized", trade.realized)?;
    d.set_item("fees", trade.fees)?;
    d.set_item("funding", trade.funding)?;
    d.set_item("pnl", trade.pnl)?;
    d.set_item("ret", trade.ret)?;
    d.set_item("best", trade.best)?;
    d.set_item("worst", trade.worst)?;
    d.set_item("reason", trade.reason.map(|reason| reason.name()))?;
    d.set_item("open", trade.close_tick.is_none())?;
    Ok(d)
}

fn fill_to_dict<'py>(
    py: Python<'py>,
    fill: &FillRecord,
    time: Option<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bar", fill.tick)?;
    d.set_item("time", time)?;
    d.set_item("instrument", 0)?;
    d.set_item("order", fill.order.map(|id| id.0))?;
    d.set_item("side", side_str(fill.side))?;
    d.set_item("size", fill.size)?;
    d.set_item("price", fill.price)?;
    d.set_item("reference", fill.costs.reference)?;
    d.set_item("spread", fill.costs.spread)?;
    d.set_item("slippage", fill.costs.slippage)?;
    d.set_item("impact", fill.costs.impact)?;
    d.set_item("fee", fill.fee)?;
    d.set_item("adjustment", fill.costs.adjustment)?;
    d.set_item("taker", fill.taker)?;
    d.set_item("reason", fill.reason.name())?;
    Ok(d)
}

/// How one ledger column is read off a row.
type Column = fn(&LedgerRow) -> f64;

fn refusal_text(why: Refusal) -> String {
    match why {
        Refusal::Waiting(n) => format!("max_open_orders: {n} already waiting"),
        Refusal::Resting(n) => format!("max_open_orders: {n} already resting"),
        Refusal::Size(size) => format!("size: {size} cannot fill"),
        Refusal::Price(price) => format!("limit: {price} is not a price"),
        Refusal::Trigger(trigger) => format!("stop: {trigger} is not a price"),
        Refusal::PostOnly => "post_only: it would take liquidity at the last close".to_string(),
    }
}

/// The reason an event happened, as the order log writes it.
fn reason_text(event: &OrderEvent) -> String {
    let size = event.size.unwrap_or(0.0);
    match event.reason {
        EventReason::Market => "market".to_string(),
        EventReason::Limit => "limit".to_string(),
        EventReason::Stop => "stop".to_string(),
        EventReason::Replace => "replace".to_string(),
        EventReason::Refused(why) => refusal_text(why),
        EventReason::Replacing(why) => format!("replace: {}", refusal_text(why)),
        EventReason::Fits { asked } => format!("{size} of the {asked} asked fits"),
        EventReason::Transfer { asked } => format!("transfer: {size} of the {asked} asked"),
        EventReason::NothingFilled => "nothing could fill on its bar".to_string(),
        EventReason::Fok => "fok: the whole size could not".to_string(),
        EventReason::MarketBar => "a market order takes one bar".to_string(),
        EventReason::OneBar(TimeInForce::Fok) => "fok: one bar".to_string(),
        EventReason::OneBar(_) => "ioc: one bar".to_string(),
        EventReason::CancelledByAccount => "cancelled by the account".to_string(),
        EventReason::Liquidation => "liquidation".to_string(),
    }
}

fn event_to_dict<'py>(
    py: Python<'py>,
    event: &OrderEvent,
    time: Option<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bar", event.tick)?;
    d.set_item("time", time)?;
    d.set_item("kind", event.kind.name())?;
    d.set_item("order", event.order.map(|id| id.0))?;
    d.set_item("instrument", event.order.map(|_| 0))?;
    d.set_item("size", event.size)?;
    d.set_item("price", event.price)?;
    d.set_item("reason", reason_text(event))?;
    Ok(d)
}

fn round_trips_of<'py>(
    py: Python<'py>,
    engine: &BarEngine,
) -> PyResult<Option<Bound<'py, PyList>>> {
    if engine.reporter().is_none() {
        return Ok(None);
    }
    let list = PyList::empty(py);
    for trade in engine.round_trips() {
        list.append(trade_to_dict(py, &trade, engine)?)?;
    }
    Ok(Some(list))
}

fn fills_of<'py>(py: Python<'py>, engine: &BarEngine) -> PyResult<Option<Bound<'py, PyList>>> {
    let Some(reporter) = engine.reporter() else {
        return Ok(None);
    };
    let list = PyList::empty(py);
    for fill in reporter.fills() {
        list.append(fill_to_dict(py, fill, engine.open_time(fill.tick))?)?;
    }
    Ok(Some(list))
}

fn ledger_of<'py>(py: Python<'py>, engine: &BarEngine) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(reporter) = engine.reporter() else {
        return Ok(None);
    };
    let rows = reporter.ledger();
    let column = |get: Column| -> Vec<f64> { rows.iter().map(get).collect() };
    let d = PyDict::new(py);
    let bars: Vec<usize> = rows.iter().map(|row| row.tick).collect();
    d.set_item("bar", PyArray1::from_vec(py, bars))?;
    let times: Option<Vec<i64>> = rows.iter().map(|row| engine.open_time(row.tick)).collect();
    d.set_item("time", times.map(|times| PyArray1::from_vec(py, times)))?;
    let columns: [(&str, Column); 21] = [
        ("equity", |row| row.equity),
        ("spot_cash", |row| row.spot_cash),
        ("margin_cash", |row| row.margin_cash),
        ("gross_exposure", |row| row.gross_exposure),
        ("net_exposure", |row| row.net_exposure),
        ("leverage", |row| row.leverage),
        ("margin_used", |row| row.margin_used),
        ("maintenance", |row| row.maintenance),
        ("fees", |row| row.costs.fees),
        ("funding", |row| row.costs.funding),
        ("spread", |row| row.costs.spread),
        ("slippage", |row| row.costs.slippage),
        ("impact", |row| row.costs.impact),
        ("liquidation", |row| row.costs.liquidation),
        ("dust", |row| row.costs.dust),
        ("position", |row| row.position),
        ("entry", |row| row.entry),
        ("mark", |row| row.mark),
        ("notional", |row| row.notional),
        ("unrealized", |row| row.unrealized),
        ("weight", |row| row.weight),
    ];
    for (name, get) in columns {
        d.set_item(name, PyArray1::from_vec(py, column(get)))?;
    }
    Ok(Some(d))
}

fn order_log_of<'py>(py: Python<'py>, engine: &BarEngine) -> PyResult<Option<Bound<'py, PyList>>> {
    let Some(reporter) = engine.reporter() else {
        return Ok(None);
    };
    let list = PyList::empty(py);
    for event in reporter.events() {
        list.append(event_to_dict(py, event, engine.open_time(event.tick))?)?;
    }
    Ok(Some(list))
}

/// Build a read-only, zero-copy `(rows, 5)` numpy view over `window`, a slice of an
/// engine's immutable candle buffer. `container` (the engine pyobject, which owns
/// the `Arc`) becomes the array's base so the data outlives the view, and the array
/// is made read-only because the buffer is shared across envs. Both `observation`
/// and `data` go through it, and the floats come from probatum-core's `floats`, the
/// library's one reinterpretation of candle memory (ADR 0040).
fn candle_view<'py>(window: &[Candle], container: Bound<'py, PyAny>) -> Bound<'py, PyArray2<f64>> {
    let rows = window.len();
    let view = ArrayView2::from_shape((rows, 5), floats(window))
        .expect("candle slice length is exactly rows * 5");

    // SAFETY: `view` points into the immutable candle buffer, which outlives the
    // array because `container` (the engine, holding the Arc) is its base. The buffer
    // is never mutated and the array is made read-only just below, so this alias into
    // shared data can never be written through.
    let array = unsafe { PyArray2::borrow_from_array(&view, container) };
    // SAFETY: the array was just created and is not yet shared with Python, so
    // clearing its WRITEABLE flag races with nothing; a read-only view is required
    // because the candle buffer is shared across every env in a batch.
    unsafe {
        (*array.as_array_ptr()).flags &= !(NPY_ARRAY_WRITEABLE as std::os::raw::c_int);
    }
    array
}

/// One field of every candle, as a zero-copy read-only view. `column` is the
/// index into open/high/low/close/volume.
fn column_view<'py>(
    window: &[Candle],
    column: usize,
    container: Bound<'py, PyAny>,
) -> Bound<'py, PyArray1<f64>> {
    // one field is every fifth float starting at `column`, which is exactly a strided
    // column of the (rows, 5) view
    let rows = window.len();
    let view = ArrayView2::from_shape((rows, 5), floats(window))
        .expect("candle slice length is exactly rows * 5");

    // SAFETY: as in `candle_view`, the view points into the immutable candle buffer,
    // which outlives the array because `container` (the engine, holding the Arc) is
    // its base, and the array is made read-only immediately below.
    let array = unsafe { PyArray1::borrow_from_array(&view.column(column), container) };
    // SAFETY: the array was just created and is not yet shared with Python, so
    // clearing its WRITEABLE flag races with nothing.
    unsafe {
        (*array.as_array_ptr()).flags &= !(NPY_ARRAY_WRITEABLE as std::os::raw::c_int);
    }
    array
}

/// The market simulation engine, exposed to Python as `probatum.Engine`. It wraps one
/// single-env bar engine; orders decided on a bar fill on the next.
#[pyclass]
struct Engine {
    inner: BarEngine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        candles,
        market = "spot",
        quote = Opening::Cash(10_000.0),
        fee_taker = 0.0006,
        fee_maker = 0.0002,
        slippage_bps = 0.0,
        max_fill_fraction = 1.0,
        max_open_orders = 8,
        report = false,
        leverage = 10.0,
        impact = 0.0,
        funding_rate = 0.0,
        funding_interval = 0,
        times = None,
        funding_hours = 0.0,
        funding_times = None,
        funding_rates = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        candles: PyReadonlyArray2<'_, f64>,
        market: &str,
        quote: Opening,
        fee_taker: f64,
        fee_maker: f64,
        slippage_bps: f64,
        max_fill_fraction: f64,
        max_open_orders: usize,
        report: bool,
        leverage: f64,
        impact: f64,
        funding_rate: f64,
        funding_interval: usize,
        times: Option<PyReadonlyArray1<'_, i64>>,
        funding_hours: f64,
        funding_times: Option<PyReadonlyArray1<'_, i64>>,
        funding_rates: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Engine> {
        let config = build_config(
            market,
            quote,
            fee_taker,
            fee_maker,
            slippage_bps,
            max_fill_fraction,
            max_open_orders,
            report,
            leverage,
            impact,
            funding_rate,
            funding_interval,
        )?;
        let series = place_funding(
            candles_from_array(candles, times)?,
            funding_rate,
            funding_interval,
            funding_hours,
            funding_times,
            funding_rates,
        )?;
        Ok(Engine {
            inner: BarEngine::new(series, config),
        })
    }

    /// Reset to the first bar and return the initial state.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyState>> {
        Bound::new(py, PyState::from(self.inner.reset()))
    }

    /// Advance one bar, resolve the orders placed since the last step, mark the
    /// account, and return the new state.
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyState>> {
        Bound::new(py, PyState::from(self.inner.step()))
    }

    /// True when there is no next bar to step into.
    fn done(&self) -> bool {
        self.inner.done()
    }

    /// Bars a year by the candles' own clock, or `None` when they came without
    /// `times` (ADR 0004).
    fn periods_per_year(&self) -> Option<f64> {
        self.inner.periods_per_year()
    }

    /// The shape of the loaded candles as `(T, 5)`, the honesty check on how many
    /// bars the engine actually holds.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.num_bars(), 5)
    }

    /// A zero-copy, read-only numpy view of the `lookback` bars ending at the
    /// current tick: shape `(rows, 5)`, columns open, high, low, close, volume.
    /// `rows` is `lookback`, or fewer early in the series. The array borrows the
    /// engine's immutable candle buffer directly with no copy; it is read-only
    /// because that buffer is shared across envs, and it holds the engine alive
    /// for as long as the view exists (ADR 0040).
    fn observation<'py>(
        slf: Bound<'py, Self>,
        lookback: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // The engine owns the Arc-backed candle buffer, so keeping it as the
        // array's base keeps the data alive for the whole life of the view.
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        Ok(candle_view(engine.inner.candle_window(lookback), container))
    }

    /// A zero-copy, read-only numpy view of the whole candle series: shape
    /// `(T, 5)`, columns open, high, low, close, volume by position (close is
    /// column 3). Like `observation` it borrows the engine's immutable buffer with
    /// no copy and holds the engine alive for the view's life; it is the
    /// full-series companion, so `eng.data[:, 3]` reads every close (ADR 0040).
    #[getter]
    fn data<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        candle_view(engine.inner.candles_all(), container)
    }

    /// Every open, as a zero-copy read-only view. The named companions to `data`,
    /// so a strategy reads `engine.closes` rather than remembering that the close
    /// is column 3 of a `(T, 5)` array.
    #[getter]
    fn opens<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        column_view(engine.inner.candles_all(), 0, container)
    }

    /// Every high, as a zero-copy read-only view.
    #[getter]
    fn highs<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        column_view(engine.inner.candles_all(), 1, container)
    }

    /// Every low, as a zero-copy read-only view.
    #[getter]
    fn lows<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        column_view(engine.inner.candles_all(), 2, container)
    }

    /// Every close, as a zero-copy read-only view.
    #[getter]
    fn closes<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        column_view(engine.inner.candles_all(), 3, container)
    }

    /// Every volume, as a zero-copy read-only view. Base units (ADR 0013).
    #[getter]
    fn volumes<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        column_view(engine.inner.candles_all(), 4, container)
    }

    /// Queue a market buy; it fills on the next bar's open. Returns the order id,
    /// or None if the queue already holds max_open_orders for this bar, or size is
    /// not a positive finite number.
    fn market_buy(&mut self, size: f64) -> Option<u64> {
        self.inner.market_buy(size).map(|id| id.0)
    }

    /// Queue a market sell; it fills on the next bar's open. Returns the order id,
    /// or None if the queue already holds max_open_orders for this bar, or size is
    /// not a positive finite number.
    fn market_sell(&mut self, size: f64) -> Option<u64> {
        self.inner.market_sell(size).map(|id| id.0)
    }

    /// Rest a buy limit. Returns the order id, or None if the book is full or size
    /// is not a positive finite number; a non-finite price raises instead.
    fn limit_buy(&mut self, size: f64, price: f64) -> PyResult<Option<u64>> {
        finite("price", price)?;
        Ok(self.inner.limit_buy(size, price).map(|id| id.0))
    }

    /// Rest a sell limit. Returns the order id, or None if the book is full or size
    /// is not a positive finite number; a non-finite price raises instead.
    fn limit_sell(&mut self, size: f64, price: f64) -> PyResult<Option<u64>> {
        finite("price", price)?;
        Ok(self.inner.limit_sell(size, price).map(|id| id.0))
    }

    /// Rest a stop on `side` that triggers at `trigger`. None if the book is full
    /// or the size or trigger is not finite. `reduce_only` makes it a protective
    /// stop that can only shrink the position: without it a stop that outlives the
    /// one that closed the position opens a fresh one on the other side, which is
    /// what a stop-loss almost never wants (ADR 0010).
    #[pyo3(signature = (side, size, trigger, reduce_only = false))]
    fn stop(
        &mut self,
        side: &str,
        size: f64,
        trigger: f64,
        reduce_only: bool,
    ) -> PyResult<Option<u64>> {
        let side = parse_side(side)?;
        finite("trigger", trigger)?;
        Ok(self
            .inner
            .stop(side, size, trigger, reduce_only)
            .map(|id| id.0))
    }

    /// Flatten the position with a reduce-only market order. None when flat.
    fn close(&mut self) -> Option<u64> {
        self.inner.close().map(|id| id.0)
    }

    /// Cancel a resting order by id. True if it was found and removed.
    fn cancel(&mut self, order_id: u64) -> bool {
        self.inner.cancel(OrderId(order_id))
    }

    /// Move a resting order under the same id, with the same side, type and flags,
    /// overriding whichever of `size`, `price` and `trigger` are given, into the
    /// first free slot. Returns the id.
    ///
    /// None when `order_id` is no longer resting or the moved order cannot rest, and
    /// nothing changes in that case. Use it for a trailing stop: re-placing with
    /// `stop()` each bar rests a new order every time, and the one that fills leaves
    /// its siblings live to open a position on the other side (ADR 0010).
    #[pyo3(signature = (order_id, size = None, price = None, trigger = None))]
    fn replace(
        &mut self,
        order_id: u64,
        size: Option<f64>,
        price: Option<f64>,
        trigger: Option<f64>,
    ) -> PyResult<Option<u64>> {
        if let Some(p) = price {
            finite("price", p)?;
        }
        if let Some(t) = trigger {
            finite("trigger", t)?;
        }
        Ok(self
            .inner
            .replace(OrderId(order_id), size, price, trigger)
            .map(|id| id.0))
    }

    /// Cancel every resting order, returning how many were dropped. Pending market
    /// orders are not resting and still fill on the next step.
    fn cancel_all(&mut self) -> usize {
        self.inner.cancel_all()
    }

    /// Move `amount` of cash into the `to` wallet, "spot" or "margin", from the
    /// other one. It lands at the start of the next bar, before any fill, held to
    /// the cash the wallet it leaves holds and, out of the margin wallet, to what
    /// leaves the cap's requirement covered (ADR 0008).
    fn transfer(&mut self, amount: f64, to: &str) -> PyResult<()> {
        let to = transfer_to("", amount, to)?;
        self.inner.transfer(amount, to);
        Ok(())
    }

    /// True when the account died since the last reset: its equity, both wallets
    /// together, reached zero, on either market. A perp liquidation does that unless
    /// the spot wallet holds cash (ADR 0008). The engine does not stop on it, so a
    /// backtest that wants to halt has to read it.
    fn is_bust(&self) -> bool {
        self.inner.is_bust()
    }

    /// Fills applied since the last reset. Zero beside orders you placed means none
    /// of them ever filled, which a series with no volume does silently.
    fn num_fills(&self) -> usize {
        self.inner.num_fills()
    }

    /// The one order primitive the shortcuts wrap. `side` is "buy" or "sell",
    /// `type` is "market", "limit", or "stop", `tif` is "GTC", "IOC", or "FOK". A
    /// limit needs `price`, a stop needs `trigger`. Returns the order id, or None if
    /// the book is full, a post_only limit would cross, or `size` is not a positive
    /// finite number. `tif` applies to market and limit orders; a stop rests until
    /// it triggers (ADR 0010).
    #[pyo3(signature = (
        side,
        size,
        r#type = "market",
        price = None,
        trigger = None,
        reduce_only = false,
        post_only = false,
        tif = "GTC",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn order(
        &mut self,
        side: &str,
        size: f64,
        r#type: &str,
        price: Option<f64>,
        trigger: Option<f64>,
        reduce_only: bool,
        post_only: bool,
        tif: &str,
    ) -> PyResult<Option<u64>> {
        let side = parse_side(side)?;
        let kind = parse_order_type(r#type)?;
        let tif = parse_tif(tif)?;
        if kind == OrderType::Limit && price.is_none() {
            return Err(PyValueError::new_err(
                "price is required when type is 'limit'",
            ));
        }
        if kind == OrderType::Stop && trigger.is_none() {
            return Err(PyValueError::new_err(
                "trigger is required when type is 'stop'",
            ));
        }
        if let Some(p) = price {
            finite("price", p)?;
        }
        if let Some(t) = trigger {
            finite("trigger", t)?;
        }
        Ok(self
            .inner
            .order(
                side,
                size,
                kind,
                price,
                trigger,
                reduce_only,
                post_only,
                tif,
            )
            .map(|id| id.0))
    }

    /// Base size for `fraction` of current equity, marked at the current close.
    fn qty_from_weight(&self, fraction: f64) -> f64 {
        self.inner.qty_from_weight(fraction)
    }

    /// Base size for a `cash` amount in quote, at the current close.
    fn qty_from_quote(&self, cash: f64) -> f64 {
        self.inner.qty_from_quote(cash)
    }

    /// Performance stats when reporting is on, else None. `periods_per_year`
    /// annualizes to the bar interval; `risk_free` is an annual rate.
    #[pyo3(signature = (periods_per_year = 365.0, risk_free = 0.0))]
    fn stats<'py>(
        &self,
        py: Python<'py>,
        periods_per_year: f64,
        risk_free: f64,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // A non-positive annualization has no square root and no per-period rate,
        // and the stats set must never come back NaN (ADRs 0007, 0030).
        if !(periods_per_year.is_finite() && periods_per_year > 0.0) {
            return Err(PyValueError::new_err(format!(
                "periods_per_year must be a finite positive number, got {periods_per_year}"
            )));
        }
        finite("risk_free", risk_free)?;
        match self.inner.stats(periods_per_year, risk_free) {
            Some(stats) => Ok(Some(stats_to_dict(py, &stats)?)),
            None => Ok(None),
        }
    }

    /// The recorded equity curve as a numpy array, one point per step, or None
    /// when reporting is off.
    fn equity_curve<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .reporter()
            .map(|reporter| PyArray1::from_slice(py, reporter.equity_curve()))
    }

    /// The round trips as a list of dicts, every one closed in the order it closed
    /// and then the one still open, marked at the current close with `open` true,
    /// or None when reporting is off (ADR 0023).
    fn trades<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        round_trips_of(py, &self.inner)
    }

    /// The fill log as a list of dicts, one per fill in the order they landed, or
    /// None when reporting is off. Each carries its bar and open time, the order
    /// that made it (None for a forced close), its side, size and price, the
    /// reference it is measured from, the spread, slippage, impact and adjustment
    /// beyond that per unit, its fee, whether it took liquidity, and its reason
    /// (ADR 0022).
    fn fills<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        fills_of(py, &self.inner)
    }

    /// The ledger as a dict of `(T,)` numpy columns, one row per bar from the opening
    /// balance on, or None when reporting is off: the bar and its open time, equity,
    /// each wallet's cash, gross and net exposure, leverage, margin used and the
    /// maintenance requirement, what the bar cost in fees, funding, spread, slippage,
    /// impact, the liquidation charge and dust, and the position, its entry, mark,
    /// notional, unrealized PnL and weight (ADR 0022).
    fn ledger<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        ledger_of(py, &self.inner)
    }

    /// The order log as a list of dicts, one per event in the order they happened,
    /// or None when reporting is off. Each carries its bar and open time, its kind
    /// (placed, refused, clamped, filled, not_filled, expired or cancelled), the
    /// order (None for a transfer's clamp), the size and price it concerns, and its
    /// reason. A refused placement is here under the id it would have had
    /// (ADRs 0010, 0022).
    fn order_log<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        order_log_of(py, &self.inner)
    }

    /// Drive `strategy` over the whole series and return the final state. Resets,
    /// calls `strategy.init(engine)` if it exists, then for each bar calls
    /// `strategy.next(state, engine)` and steps. The strategy places its orders
    /// through the engine's own methods (market, limit, stop, close), so it can
    /// express any action; it should not call `step` itself, since the driver
    /// does. This is the single-env callback path: `next` runs under the GIL, so
    /// it is the slow path next to `Batch.step_all`.
    fn run<'py>(
        slf: Bound<'py, Self>,
        strategy: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyState>> {
        let py = slf.py();

        // Each engine borrow below is scoped and dropped before Python runs, so
        // the callback is free to re-enter the engine (place orders) without a
        // double-borrow panic.
        let mut state = {
            let mut engine = slf.borrow_mut();
            Bound::new(py, PyState::from(engine.inner.reset()))?
        };

        if strategy.hasattr("init")? {
            strategy.call_method1("init", (slf.clone(),))?;
        }

        // A declared warm-up is read once, after init, so a strategy can compute it
        // from the data it just looked at. The bars before it still advance and
        // still fill whatever is resting; only the decision is skipped, which is
        // what makes the guard everybody writes by hand unnecessary (ADR 0025).
        let warmup: usize = match strategy.getattr("warmup") {
            Ok(value) if !value.is_none() => value.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "warmup must be a non-negative whole number of bars, got {}",
                    value.repr().map(|r| r.to_string()).unwrap_or_default()
                ))
            })?,
            _ => 0,
        };

        let mut tick = 0usize;
        while !slf.borrow().inner.done() {
            if tick >= warmup {
                strategy.call_method1("next", (state.clone(), slf.clone()))?;
            }
            state = {
                let mut engine = slf.borrow_mut();
                Bound::new(py, PyState::from(engine.inner.step()))?
            };
            tick += 1;
        }

        Ok(state)
    }
}

/// A batch of independent envs over one shared candle series, stepped in
/// parallel. Every env sees the same candles (shared by `Arc`, no duplication)
/// but carries its own account and orders, so the batched step is bit-identical
/// to stepping each env in a loop. `step_all` releases the GIL around the
/// parallel fills, so many envs advance while Python threads run.
#[pyclass]
struct Batch {
    inner: EnvBatch,
    // the feature matrix, cached once by set_features (ADR 0038): observe_features
    // gathers from this buffer each step rather than recopying the whole (T, F)
    // matrix from numpy, matching how the candles are copied once and shared
    features: Vec<f64>,
    n_features: usize,
}

#[pymethods]
impl Batch {
    #[new]
    #[pyo3(signature = (
        candles,
        num_envs,
        market = "spot",
        quote = Opening::Cash(10_000.0),
        fee_taker = 0.0006,
        fee_maker = 0.0002,
        slippage_bps = 0.0,
        max_fill_fraction = 1.0,
        max_open_orders = 8,
        report = false,
        leverage = 10.0,
        impact = 0.0,
        funding_rate = 0.0,
        funding_interval = 0,
        fee_taker_per_env = None,
        fee_maker_per_env = None,
        slippage_bps_per_env = None,
        impact_per_env = None,
        times = None,
        funding_hours = 0.0,
        funding_times = None,
        funding_rates = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        candles: PyReadonlyArray2<'_, f64>,
        num_envs: usize,
        market: &str,
        quote: Opening,
        fee_taker: f64,
        fee_maker: f64,
        slippage_bps: f64,
        max_fill_fraction: f64,
        max_open_orders: usize,
        report: bool,
        leverage: f64,
        impact: f64,
        funding_rate: f64,
        funding_interval: usize,
        fee_taker_per_env: Option<PyReadonlyArray1<'_, f64>>,
        fee_maker_per_env: Option<PyReadonlyArray1<'_, f64>>,
        slippage_bps_per_env: Option<PyReadonlyArray1<'_, f64>>,
        impact_per_env: Option<PyReadonlyArray1<'_, f64>>,
        times: Option<PyReadonlyArray1<'_, i64>>,
        funding_hours: f64,
        funding_times: Option<PyReadonlyArray1<'_, i64>>,
        funding_rates: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Batch> {
        let config = build_config(
            market,
            quote,
            fee_taker,
            fee_maker,
            slippage_bps,
            max_fill_fraction,
            max_open_orders,
            report,
            leverage,
            impact,
            funding_rate,
            funding_interval,
        )?;
        let series = place_funding(
            candles_from_array(candles, times)?,
            funding_rate,
            funding_interval,
            funding_hours,
            funding_times,
            funding_rates,
        )?;

        // Per-env cost randomization (ADR 0038): each override, if given, replaces
        // its scalar field for the matching env; absent overrides keep the scalar.
        let ft = per_env_vec("fee_taker_per_env", num_envs, &fee_taker_per_env, a_rebate)?;
        let fm = per_env_vec("fee_maker_per_env", num_envs, &fee_maker_per_env, a_rebate)?;
        let adverse = |name: &str, value: f64| at_least(name, value, 0.0);
        let sb = per_env_vec(
            "slippage_bps_per_env",
            num_envs,
            &slippage_bps_per_env,
            adverse,
        )?;
        let im = per_env_vec("impact_per_env", num_envs, &impact_per_env, adverse)?;

        let inner = if ft.is_some() || fm.is_some() || sb.is_some() || im.is_some() {
            let configs: Vec<EngineConfig> = (0..num_envs)
                .map(|i| {
                    let mut c = config;
                    if let Some(v) = &ft {
                        c.fee_taker = v[i];
                    }
                    if let Some(v) = &fm {
                        c.fee_maker = v[i];
                    }
                    if let Some(v) = &sb {
                        c.slippage_bps = v[i];
                    }
                    if let Some(v) = &im {
                        c.impact = v[i];
                    }
                    c
                })
                .collect();
            EnvBatch::with_configs(series, &configs)
        } else {
            EnvBatch::new(num_envs, series, config)
        };

        Ok(Batch {
            inner,
            features: Vec::new(),
            n_features: 0,
        })
    }

    /// Number of envs, so `len(batch)` and `batch.num_envs` are the same number.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of envs, fixed at construction: every per-env array passed in has to
    /// be exactly this long.
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.len()
    }

    /// True when no env has a next bar to step into. After per-env random-start
    /// resets that takes until the last env reaches the end; an empty batch is
    /// done from the start.
    fn done(&self) -> bool {
        self.inner.done()
    }

    /// Bars a year by the shared candles' own clock, or `None` when they came
    /// without `times` (ADR 0004).
    fn periods_per_year(&self) -> Option<f64> {
        self.inner.periods_per_year()
    }

    /// Reset every env to the first bar, returning one initial state per env in
    /// env order.
    fn reset_all<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let states = py.detach(|| self.inner.reset_all());
        states_to_list(py, states)
    }

    /// Apply the per-env `actions` as market orders (positive buys, negative
    /// sells, zero holds), then step every env one bar in parallel with the GIL
    /// released, returning the per-env states. `actions` is a 1-D float64 array of
    /// length num_envs, or None to step placing no new orders.
    #[pyo3(signature = (actions = None))]
    fn step_all<'py>(
        &mut self,
        py: Python<'py>,
        actions: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyList>> {
        // Read the actions into an owned Vec under the GIL; the numpy array must
        // not be touched once the GIL is released.
        let action_vec = self.action_vec(&actions)?;

        // GIL released: apply the actions and run the parallel step. The closure
        // touches only Rust data, never a Python object.
        let states = py.detach(|| {
            if let Some(sizes) = &action_vec {
                for (engine, &size) in self.inner.engines_mut().iter_mut().zip(sizes) {
                    if size > 0.0 {
                        engine.market_buy(size);
                    } else if size < 0.0 {
                        engine.market_sell(-size);
                    }
                }
            }
            self.inner.step_all()
        });

        states_to_list(py, states)
    }

    // --- Batched RL surface (used by probatum.rl.VectorEnv) ---

    /// Reset every env to its own start offset in `offsets` (length num_envs).
    fn reset_at_each(
        &mut self,
        py: Python<'_>,
        offsets: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let offs = self.offsets_vec(offsets)?;
        py.detach(|| {
            self.inner.reset_at_each(&offs);
        });
        Ok(())
    }

    /// Reset only the envs where `mask` is true, each to its entry in `offsets`;
    /// the rest keep running. The autoreset primitive.
    fn reset_masked(
        &mut self,
        py: Python<'_>,
        mask: PyReadonlyArray1<'_, bool>,
        offsets: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let mask_slice = mask
            .as_slice()
            .map_err(|_| PyValueError::new_err("mask must be a contiguous 1-D bool array"))?;
        if mask_slice.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "mask length {} must equal num_envs {}",
                mask_slice.len(),
                self.inner.len()
            )));
        }
        let mask_vec = mask_slice.to_vec();
        let offs = self.offsets_vec(offsets)?;
        py.detach(|| {
            self.inner.reset_masked(&mask_vec, &offs);
        });
        Ok(())
    }

    /// Apply the per-env `actions` (positive buys, negative sells, zero holds) and
    /// step every env in parallel with the GIL released, discarding the states.
    /// The lean step for the RL loop, which reads observation and equity after.
    #[pyo3(signature = (actions = None))]
    fn advance(
        &mut self,
        py: Python<'_>,
        actions: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let action_vec = self.action_vec(&actions)?;
        py.detach(|| {
            if let Some(sizes) = &action_vec {
                for (engine, &size) in self.inner.engines_mut().iter_mut().zip(sizes) {
                    if size > 0.0 {
                        engine.market_buy(size);
                    } else if size < 0.0 {
                        engine.market_sell(-size);
                    }
                }
            }
            self.inner.step_all();
        });
        Ok(())
    }

    /// Batched observation: the `window` most recent candles per env as an
    /// `(num_envs, window, 5)` float64 array, front-padded with zeros at warmup.
    fn observe<'py>(&self, py: Python<'py>, window: usize) -> PyResult<Bound<'py, PyArray3<f64>>> {
        check_window(window)?;
        let n = self.inner.len();
        let flat = py.detach(|| self.inner.observe_all(window));
        let array = Array3::from_shape_vec((n, window, 5), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray3::from_owned_array(py, array))
    }

    /// Per-env equity as a `(num_envs,)` array, for building the reward.
    fn equities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.equities())
    }

    /// Per-env truncation flags (reached the last bar) as a `(num_envs,)` array.
    fn dones<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.dones())
    }

    /// Per-env termination flags as a `(num_envs,)` array: true where the account
    /// died, its equity, both wallets together, reaching zero (ADR 0008).
    fn busts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.busts())
    }

    /// Cache the shared `(T, F)` feature matrix once, so `observe_features` gathers
    /// from it each step instead of recopying the whole matrix from numpy. Call it
    /// once before observing; the matrix must have one row per candle (ADR 0038).
    fn set_features(&mut self, features: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let n_features = features.as_array().ncols();
        let n_rows = features.as_array().nrows();
        let num_bars = self.inner.num_bars();
        if n_rows != num_bars {
            return Err(PyValueError::new_err(format!(
                "features must have one row per candle: got {n_rows} rows for {num_bars} bars"
            )));
        }
        self.features = features
            .as_slice()
            .map_err(|_| PyValueError::new_err("features must be a contiguous 2-D float64 array"))?
            .to_vec();
        self.n_features = n_features;
        Ok(())
    }

    /// Batched feature observation: the `window` most recent rows of the feature
    /// matrix set by `set_features` per env, as `(num_envs, window, F)` float64. The
    /// gather runs on the Rayon pool over the cached buffer with the GIL released.
    fn observe_features<'py>(
        &self,
        py: Python<'py>,
        window: usize,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        check_window(window)?;
        if self.n_features == 0 {
            return Err(PyValueError::new_err(
                "no features set; call set_features before observe_features",
            ));
        }
        let n = self.inner.len();
        let n_features = self.n_features;
        let out = py.detach(|| {
            self.inner
                .observe_features_all(window, &self.features, n_features)
        });
        let array = Array3::from_shape_vec((n, window, n_features), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray3::from_owned_array(py, array))
    }

    /// A batched per-env account snapshot as a dict of `(num_envs,)` arrays, for
    /// building a reward: equity, position, unrealized_pnl, realized_pnl,
    /// mark_price, funding_paid.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snap = self.inner.snapshot();
        let d = PyDict::new(py);
        d.set_item("equity", PyArray1::from_vec(py, snap.equity))?;
        d.set_item("position", PyArray1::from_vec(py, snap.position))?;
        d.set_item(
            "unrealized_pnl",
            PyArray1::from_vec(py, snap.unrealized_pnl),
        )?;
        d.set_item("realized_pnl", PyArray1::from_vec(py, snap.realized_pnl))?;
        d.set_item("mark_price", PyArray1::from_vec(py, snap.mark_price))?;
        d.set_item("funding_paid", PyArray1::from_vec(py, snap.funding_paid))?;
        Ok(d)
    }
}

impl Batch {
    /// Validate and convert a per-env `i64` offset array into clamped `usize`.
    fn offsets_vec(&self, offsets: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
        let slice = offsets
            .as_slice()
            .map_err(|_| PyValueError::new_err("offsets must be a contiguous 1-D int64 array"))?;
        if slice.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "offsets length {} must equal num_envs {}",
                slice.len(),
                self.inner.len()
            )));
        }
        Ok(slice.iter().map(|&x| x.max(0) as usize).collect())
    }

    /// Validate and copy an optional per-env signed-size action array.
    fn action_vec(
        &self,
        actions: &Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Option<Vec<f64>>> {
        match actions {
            Some(array) => {
                let slice = array.as_slice().map_err(|_| {
                    PyValueError::new_err("actions must be a contiguous 1-D float64 array")
                })?;
                if slice.len() != self.inner.len() {
                    return Err(PyValueError::new_err(format!(
                        "actions length {} must equal num_envs {}",
                        slice.len(),
                        self.inner.len()
                    )));
                }
                Ok(Some(slice.to_vec()))
            }
            None => Ok(None),
        }
    }
}

/// The verbs a script takes, as an error lists them.
const VERBS: &str = "buy, sell, close, cancel, cancel_all, replace or transfer";

/// A value as it arrived, for an error to name: its repr, so "1" and 1 read apart.
fn shown(value: &Bound<'_, PyAny>) -> String {
    value
        .repr()
        .map(|repr| repr.to_string())
        .unwrap_or_else(|_| "a value".to_string())
}

/// One script action's arguments, read by name, with `at` naming the action in an
/// error. A value of None is read as an argument left out.
struct Arguments<'a, 'py> {
    at: &'a str,
    verb: &'a str,
    given: &'a Bound<'py, PyDict>,
}

impl<'py> Arguments<'_, 'py> {
    fn value(&self, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.given.get_item(name)?.filter(|value| !value.is_none()))
    }

    fn number(&self, name: &str) -> PyResult<Option<f64>> {
        let Some(value) = self.value(name)? else {
            return Ok(None);
        };
        value.extract::<f64>().map(Some).map_err(|_| {
            PyValueError::new_err(format!(
                "{}: {name} {} is not a number; give a float",
                self.at,
                shown(&value)
            ))
        })
    }

    /// A limit, a stop or a trigger, which a bar can only reach if it is finite.
    fn price(&self, name: &str) -> PyResult<Option<f64>> {
        let price = self.number(name)?;
        match price {
            Some(p) if !p.is_finite() => Err(PyValueError::new_err(format!(
                "{}: {name} {p} is not a price; give a finite number",
                self.at
            ))),
            _ => Ok(price),
        }
    }

    fn flag(&self, name: &str) -> PyResult<bool> {
        let Some(value) = self.value(name)? else {
            return Ok(false);
        };
        value.extract::<bool>().map_err(|_| {
            PyValueError::new_err(format!(
                "{}: {name} {} is not True or False; give a bool",
                self.at,
                shown(&value)
            ))
        })
    }

    fn text(&self, name: &str, default: &str, choices: &str) -> PyResult<String> {
        let Some(value) = self.value(name)? else {
            return Ok(default.to_string());
        };
        value.extract::<String>().map_err(|_| {
            PyValueError::new_err(format!(
                "{}: {name} {} is not a string; use {choices}",
                self.at,
                shown(&value)
            ))
        })
    }

    /// The order a cancel or a replace names, given the `placements` the script
    /// made before it.
    fn order(&self, placements: usize) -> PyResult<OrderRef> {
        let Some(value) = self.value("order")? else {
            return Err(PyValueError::new_err(format!(
                "{}: {} has no order; name an earlier placement, counted from 0, or \
                 \"oldest\"",
                self.at, self.verb
            )));
        };
        if value.extract::<String>().is_ok_and(|text| text == "oldest") {
            return Ok(OrderRef::Oldest);
        }
        match value.extract::<usize>() {
            Ok(n) if n < placements => Ok(OrderRef::Placed(n)),
            _ => Err(PyValueError::new_err(format!(
                "{}: order {} is not one of the {placements} placements before it; \
                 name one counted from 0, or \"oldest\"",
                self.at,
                shown(&value)
            ))),
        }
    }

    /// At one instrument there is only instrument 0 to name.
    fn instrument(&self) -> PyResult<()> {
        match self.value("instrument")? {
            Some(value) if value.extract::<usize>().ok() != Some(0) => {
                Err(PyValueError::new_err(format!(
                    "{}: instrument {} is not one the engine holds; name 0, the one it \
                     holds, or leave it out",
                    self.at,
                    shown(&value)
                )))
            }
            _ => Ok(()),
        }
    }
}

/// Read one script action, `placements` being how many the script made before it.
fn read_action(
    at: &str,
    verb: &str,
    given: &Bound<'_, PyDict>,
    placements: usize,
) -> PyResult<Action> {
    let takes: &[&str] = match verb {
        "buy" | "sell" => &[
            "size",
            "limit",
            "stop",
            "tif",
            "post_only",
            "reduce_only",
            "instrument",
        ],
        "close" | "cancel_all" => &["instrument"],
        "cancel" => &["order"],
        "replace" => &["order", "price", "trigger", "size"],
        "transfer" => &["amount", "to"],
        _ => {
            return Err(PyValueError::new_err(format!(
                "{at}: {verb:?} is not a verb; use {VERBS}"
            )))
        }
    };
    for key in given.keys() {
        let Ok(name) = key.extract::<String>() else {
            return Err(PyValueError::new_err(format!(
                "{at}: {} is not an argument's name; name each argument with a string",
                shown(&key)
            )));
        };
        if !takes.contains(&name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "{at}: {verb} takes no {name}; it takes {}",
                takes.join(", ")
            )));
        }
    }
    let args = Arguments { at, verb, given };
    let action = match verb {
        "buy" | "sell" => {
            args.instrument()?;
            let Some(size) = args.number("size")? else {
                return Err(PyValueError::new_err(format!(
                    "{at}: {verb} has no size; give it one, in base units"
                )));
            };
            let (price, trigger) = (args.price("limit")?, args.price("stop")?);
            let kind = match (price, trigger) {
                (None, None) => OrderType::Market,
                (Some(_), None) => OrderType::Limit,
                (None, Some(_)) => OrderType::Stop,
                (Some(limit), Some(stop)) => {
                    return Err(PyValueError::new_err(format!(
                        "{at}: {verb} has a limit {limit} and a stop {stop}; give one or \
                         the other"
                    )))
                }
            };
            let tif = args.text("tif", "gtc", "gtc, ioc or fok")?;
            Action::Order {
                side: if verb == "buy" { Side::Buy } else { Side::Sell },
                size,
                kind,
                price,
                trigger,
                reduce_only: args.flag("reduce_only")?,
                post_only: args.flag("post_only")?,
                tif: parse_tif(&tif).map_err(|_| {
                    PyValueError::new_err(format!(
                        "{at}: tif {tif:?} is not a time in force; use gtc, ioc or fok"
                    ))
                })?,
            }
        }
        "close" => {
            args.instrument()?;
            Action::Close
        }
        "cancel_all" => {
            args.instrument()?;
            Action::CancelAll
        }
        "cancel" => Action::Cancel(args.order(placements)?),
        "replace" => Action::Replace {
            order: args.order(placements)?,
            size: args.number("size")?,
            price: args.price("price")?,
            trigger: args.price("trigger")?,
        },
        _ => {
            let Some(amount) = args.number("amount")? else {
                return Err(PyValueError::new_err(format!(
                    "{at}: transfer has no amount; give it the cash to move"
                )));
            };
            let to = args.text("to", "margin", "spot or margin")?;
            Action::Transfer {
                amount,
                to: transfer_to(&format!("{at}: "), amount, &to)?,
            }
        }
    };
    Ok(action)
}

/// Read a script: a dict from a bar to the actions taken at its close, each a verb
/// and a dict of its arguments, into the actions of each of the `bars`. A decision
/// on the last bar or after it could never fill, so it is refused.
fn read_script<'py>(script: &Bound<'py, PyDict>, bars: usize) -> PyResult<Vec<Vec<Action>>> {
    let mut entries = Vec::with_capacity(script.len());
    for (bar, actions) in script.iter() {
        let Ok(t) = bar.extract::<usize>() else {
            return Err(PyValueError::new_err(format!(
                "script: {} is not a bar; key each bar's actions by its index from 0",
                shown(&bar)
            )));
        };
        if t >= bars.saturating_sub(1) {
            let fix = match bars {
                0 | 1 => "a script needs two bars or more".to_string(),
                _ => format!("key decisions by the bars from 0 to {}", bars - 2),
            };
            return Err(PyValueError::new_err(format!(
                "script: bar {t} has no bar after it to fill on; {fix}"
            )));
        }
        entries.push((t, actions));
    }
    // placements are counted in the order the run takes them
    entries.sort_by_key(|(t, _)| *t);
    let mut out = vec![Vec::new(); bars];
    let mut placements = 0;
    for (t, actions) in entries {
        let Ok(actions_iter) = actions.try_iter() else {
            return Err(PyValueError::new_err(format!(
                "script: bar {t} holds {}; give it a list of actions",
                shown(&actions)
            )));
        };
        for (k, action) in actions_iter.enumerate() {
            let action = action?;
            let at = format!("script[{t}][{k}]");
            let Ok((verb, given)) = action.extract::<(String, Bound<'py, PyDict>)>() else {
                return Err(PyValueError::new_err(format!(
                    "{at}: {} is not a verb and its arguments; write \
                     (\"buy\", {{\"size\": 1.0}})",
                    shown(&action)
                )));
            };
            let action = read_action(&at, &verb, &given, placements)?;
            if matches!(action, Action::Order { .. } | Action::Close) {
                placements += 1;
            }
            out[t].push(action);
        }
    }
    Ok(out)
}

/// Play `script` over `candles` at `venue` in one call and return what the run
/// recorded: its ledger, fill log, order log and round trips, as `Engine` hands
/// them out. `venue` takes `Engine`'s settings by name. `script` maps a bar to the
/// actions taken at its close, each a verb and a dict of its arguments: `buy` and
/// `sell` with a `size`, and a `limit` or a `stop`, `tif`, `post_only` and
/// `reduce_only`; `close` and `cancel_all`; `cancel` and `replace` naming an
/// `order`, the second with the `price`, `trigger` or `size` it moves to; and
/// `transfer` with an `amount` and the wallet it goes `to`, the margin wallet
/// unless named. `buy`, `sell`, `close` and `cancel_all` take `instrument` too,
/// which at one instrument can only be 0. An order is named by an earlier placement
/// in the script, counting its buys, sells and closes from 0, which names nothing
/// once the engine refused that placement or a close found nothing to close, or as
/// `"oldest"`, the oldest order still resting. The comparison with the reference
/// simulator and the tests drive the engine through this.
#[pyfunction]
#[pyo3(name = "run", signature = (candles, venue, script, times = None))]
fn run_script<'py>(
    py: Python<'py>,
    candles: Bound<'py, PyAny>,
    venue: Option<Bound<'py, PyDict>>,
    script: Bound<'py, PyDict>,
    times: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let settings = match &venue {
        Some(venue) => venue.copy()?,
        None => PyDict::new(py),
    };
    if let Some(times) = times {
        settings.set_item("times", times)?;
    }
    settings.set_item("report", true)?;
    let engine = py
        .get_type::<Engine>()
        .call((candles,), Some(&settings))?
        .cast_into::<Engine>()?;
    let mut engine = engine.borrow_mut();
    let mut script = Script::new(read_script(&script, engine.inner.num_bars())?);
    let inner = &mut engine.inner;
    py.detach(|| {
        probatum_engine::run(inner, &mut script);
    });
    let out = PyDict::new(py);
    out.set_item("ledger", ledger_of(py, &engine.inner)?)?;
    out.set_item("fills", fills_of(py, &engine.inner)?)?;
    out.set_item("order_log", order_log_of(py, &engine.inner)?)?;
    out.set_item("trades", round_trips_of(py, &engine.inner)?)?;
    Ok(out)
}

/// Bars a year on the clock `times`, each bar's open in UTC nanoseconds: the rows
/// less one per 365 days of their span, or `None` with fewer than two rows
/// (ADR 0004).
#[pyfunction]
fn periods_per_year(times: PyReadonlyArray1<'_, i64>) -> PyResult<Option<f64>> {
    Ok(clock_from(times, None)?.periods_per_year())
}

/// The `_probatum` extension module, imported by Python as `probatum._probatum`.
#[pymodule]
fn _probatum(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<Batch>()?;
    m.add_class::<PyState>()?;
    m.add_function(wrap_pyfunction!(periods_per_year, m)?)?;
    m.add_function(wrap_pyfunction!(run_script, m)?)?;
    Ok(())
}
