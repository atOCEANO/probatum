<h1>OCEΛNO <small><code>embeddable-market-simulation-library</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/embeddable-market-simulation-library/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/embeddable-market-simulation-library?label=release&color=2ea043" alt="Latest release" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <a href="../README.md">Introduction</a> &nbsp;•&nbsp;
  <b>Python API</b> &nbsp;•&nbsp;
  <a href="Specification.md">Specification</a> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Plotting.md">Plotting</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Python API

Everything the package exports, and where each piece is documented in full:

| Import | What it is |
| :--- | :--- |
| `probatum.Data` | N instruments over T bars on one clock, read from any frame or array, every field and slice a read-only view of the bars it owns. [Below](#data). |
| `probatum.Instrument` | One instrument of a Data: its symbol, market and venue, its quote and base assets and the one it settles in, an inverse contract's size, and the venue's limits on an order. [Below](#data). |
| `probatum.Venue` | A venue's rules for a run, every one in one immutable object. [Below](#venue). |
| `probatum.venues` | The presets, each a Venue: `binance_perp`, `bybit_perp`, `okx_perp`, `hyperliquid_perp`, the spot, inverse and combined ones, and `frictionless`. [Below](#venue). |
| `probatum.Engine` | An account at a venue on a Data for a loop the caller owns: `reset`, `step`, and the account's state and verbs, recording the Result `backtest` records. [Below](#engine). |
| `probatum.Batch` | Many independent envs over one shared series, stepped in parallel with the GIL released. [Below](#batch). |
| `probatum.backtest` | Run a policy on a Data at a venue: an array of weights or quantities, or a table of them, stepped bar by bar in Rust with its barriers, or a Strategy deciding at each close. [Below](#backtesting). |
| `probatum.Result` | What a run recorded: its equity and returns, its ledger, holdings, fill log, order log and round trips as tables any Arrow reader takes, the totals of its costs, each modelling choice it made, graded, three baselines beside it, each instrument's record, its statistics and its provenance, saved as files any tool reads. [Below](#the-result). |
| `probatum.combine` | Add Results from accounts at different venues into one book. [Below](#combining-books). |
| `probatum.search` | Search a policy's parameters on the bars before a holdout, each trial one run there, every trial kept. [Below](#search). |
| `probatum.Study` | What a search or a validation found: every trial, or every out-of-sample path with the fits and windows that made it, as rows beside their returns, and the winner's Result, run once on the held-out bars when asked. [Below](#the-study). |
| `probatum.Int` | A whole number a search draws, on a lattice. [Below](#the-space). |
| `probatum.Float` | A number a search draws, from an interval or a lattice. [Below](#the-space). |
| `probatum.Categorical` | One of the choices a search draws among. [Below](#the-space). |
| `probatum.cv` | The splitters, `Holdout`, `WalkForward`, `PurgedKFold` and `CombinatorialPurged`, which scikit-learn's tools take, each leaving out of training every row whose label reaches a test span. [Below](#splitters). |
| `probatum.validate` | Run a search on the training bars of each split a splitter gives and trade each split's winner on its test span, every out-of-sample path one run across explicit seams. [Below](#validation). |
| `probatum.stats` | How far a run's statistics can be trusted, read from a Result or any series of returns: the Sharpe's interval, its probability of beating a benchmark and the track record that would show it. [Below](#inference). |
| `probatum.Backtester` | emsl's strategy runner: drive a `Strategy` over one series, get stats, equity, and trades. [Below](#the-strategy-runner). |
| `probatum.BacktestResult` | What a `Backtester` run returns. [Below](#the-strategy-runner). |
| `probatum.Strategy` | A policy written as what it decides at each close, run by `backtest` with the bar and the account, and by the `Backtester` as emsl ran it. [Below](#a-strategy). |
| `probatum.tune` | Search a strategy's parameters, each trial a full backtest, across worker processes. [Below](#tuning). |
| `probatum.Market` | The venue and its costs as one object, which hands out every surface configured identically. [Below](#the-market). |
| `probatum.metrics` | Evaluate a finished run: where the money went, and what the number is worth believing. [Below](#metrics). |
| `probatum.ta` | 30 indicators, one value per bar, with the conventions written down. [Below](#indicators). |
| `probatum.rl` | `VectorEnv`, the Gymnasium vector env. [Below](#reinforcement-learning), in full in the [RL Guide](RL_Guide.md). |
| `probatum.sb3` | `ProbatumVecEnv`, the Stable-Baselines3 adapter. [RL Guide](RL_Guide.md#training). |
| `probatum.chart` | Draw a frame, your own arrays and a run as one self-contained HTML document. [Below](#plotting), in full in the [Plotting](Plotting.md) guide. |
| `probatum.chart_defaults` | Set the theme, height and palette every later chart uses. [Plotting](Plotting.md#output). |
| `probatum.plot` | `Line`, `Histogram`, `Band`, `Level`, `Marker`, `Markers`, `Background`, `Panel`, `Recorder`, `ramp`, `at_bar` and `at_next`: the marks a chart carries and the helpers that record and align them. [Plotting](Plotting.md#the-marks). |
| `probatum.to_ohlcv` | Turn a DataFrame, a parquet path, or an array into the `(T, 5)` the engine takes. [Below](#data-input). |
| `probatum.ProbatumError` | What every error probatum raises derives from, beside the built-in it means: a class for each built-in, `ProbatumValueError` and its siblings, and the kinds `DataError`, `VenueError`, `SpecError`, `LookaheadError`, `PolicyError` and `MissingDependency`. [Below](#errors-and-warnings). |
| `probatum.ProbatumWarning` | The class of every warning probatum gives, a `UserWarning`. [Below](#errors-and-warnings). |

All of it sits on the one Rust engine, so a backtest, an RL rollout, and a parameter search see the same fill model and the same costs.

<br>

## Quickstart

Build a Data, run a policy on it at a venue, read the Result. Everything else on this page is detail layered on top of this call.

```python
import numpy
import probatum

close = 100.0 + numpy.random.default_rng(0).normal(0.0, 1.0, 1_000).cumsum()
bars = numpy.column_stack([close, close + 0.5, close - 0.5, close, numpy.full(1_000, 1e3)])
hours = numpy.arange(1_000, dtype=numpy.int64) * 3_600_000_000_000
data = probatum.Data(bars, symbol="BTCUSDT", market="linear", time=hours)


class Hundred(probatum.Strategy):
    def next(self, bar, account):
        if account.position == 0.0 and bar.i % 100 == 0:
            account.buy(0.1)                    # decided now, fills on the next bar's open


result = probatum.backtest(Hundred(), data, "frictionless")
print(result)                                   # what ran, its equity, its optimistic assumptions
```

<br>

## Data

A `probatum.Data` is a universe: N instruments over T bars on one clock. It copies its bars once into a buffer it owns, bar-major, and every field and every slice is a read-only view of that buffer, or a read-only copy where a selection puts the instruments in another order, so no later edit to the frame it came from can change the bars it holds ([ADR 0005](Decisions.md)).

```python
import pandas
import probatum

btc = probatum.Data(pandas.read_parquet("btc.parquet"), symbol="BTCUSDT",
                    market="linear")
pair = probatum.Data({"BTCUSDT": btc_frame, "ETHUSDT": eth_frame})
bars = probatum.Data(ohlcv, time=times, symbol="BTCUSDT")
universe = probatum.Data.from_router(router, "binance", "linear", "1h",
                                     symbols=["BTCUSDT", "ETHUSDT"],
                                     start="2023-01-01", funding=True)
```

**What it reads.** A frame is anything that exports the Arrow C stream or array interface, as polars, pyarrow, DuckDB and pandas with pyarrow do, read in the binding so that no frame library is imported to read it, or a pandas frame read by its own methods where pyarrow is missing or cannot convert one of its columns. It needs `open`, `high`, `low`, `close` and `volume` columns, and the bars' open times in its one datetime index or column, or in a `timestamp` column of milliseconds since 1970; a frame with times in more than one is refused, naming them, unless `time=` gives the open times. An array is `(T, 5)` for one instrument or `(T, N, 5)` for N, NaN in all five values where an instrument printed no bar, with `time=` the open time of each row, as datetimes or int64 nanoseconds since 1970. A dict of frames keys each by its symbol. A frame from exchange-router names itself: its venue, market, symbol and interval come from its `attrs`, and a `symbol=` or `market=` that says otherwise is refused.

**From the router.** `Data.from_router(router, exchange, market, interval, symbols, start, end=None, funding=False)` fetches the bars through an exchange-router `Router`, from the first at or after `start` to the last at or before `end`, a text end taking in the whole period it names, and never the bar still open at the time of the fetch, which a venue serves as though it were done. `start` and `end` are times or the text of one. One router call walks back from its newest bar for at most a hundred requests to the venue, so the calls are chained back to `start`. A call that brings nothing earlier is asked once more, and where it brings nothing again the venue's history is taken to begin there, which `provenance["history"]` records and a warning names, as another names bars that end before `end`. The router's own warnings are not printed but kept in `provenance["warnings"]`, and `provenance["fetched"]` holds the time of the fetch. A symbol the router fails on raises the router's own error rather than being left out, as a pull of that symbol alone does, an `AsyncRouter` is refused, and a gap in the bars is warned of. Each instrument takes the router's own name for its symbol, however it was typed, and its base asset and the venue's limits on an order from the router's symbol information, and two symbols the router names as one are refused. A symbol the router does not list now, as one delisted or paused since its bars, keeps its bars without them where the router still serves them, and a warning names it with what the router said. With `funding=True` each instrument's funding events are fetched the same way, and a series that misses part of its instrument's span is refused, naming the span: one that begins more than a cycle after the instrument's first bar or ends more than a cycle before its last bar ends, one with a stretch between two events more than half as long again as the spacing on either side, one with an event that has no rate, and one with no event over bars that span more than a day. So a run never charges nothing in silence.

**Naming.** `symbol` names an instrument, or with an array of N a list of N names, and `market` is `"spot"`, `"linear"` or `"inverse"`, left for the venue to give when the Data runs where it is left out. An instrument is keyed by its venue, market and symbol, and `data.names` gives each a short name by which the Data is indexed: its symbol, with its market before it where another market holds the same symbol (`spot:BTCUSDT`, `linear:BTCUSDT`), and its venue before that where another venue does, a market or venue the bars do not say left out; an instrument with no symbol is named by its position, `"0"`, `"1"` and so on. `data.instruments` holds each one's `probatum.Instrument`: its `symbol`, `market`, `venue`, `quote`, the currency its price is quoted in, and `contract`, the dollars an inverse contract is worth, each `None` where the bars do not say; `base`, its base asset, `min_notional`, the smallest notional the venue takes in an order, in the quote currency, and `max_qty`, the largest size it takes in a limit order, in base units or in contracts on an inverse contract, which the router's symbol information gives as of the fetch, the minimum left out on an inverse contract, where the router gives it in dollars and the engine counts it in the coin; and `settlement`, the asset its account settles in, the coin of an inverse contract and the quote otherwise, `None` where the market is the venue's to give.

**The clock** is the open times at which any instrument printed a bar ([Specification](Specification.md#the-clock)). Its `interval` is the one a router frame declares, which an `interval=` that says otherwise cannot change, or the one `interval=` gives, or else the one the median gap between bars gives, the shortest among the instruments, since one that prints seldom still prints on the clock's steps, and measured between dates for days, which across a change of clocks are an hour short or long; it is one of `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d` and `1w`. Every row is a whole number of intervals from the first, so instruments whose bars open at different minutes are refused, and a daily or longer bar is placed at 00:00 UTC on the date it carries in its own time zone.

**Every bar is checked as it arrives**, as the engine checks it: a row holds all five values or none, each finite, the low at or below the open and the close and the high at or above both, the low above zero and the volume not below it, and an instrument's times are all there and rise, each once ([ADR 0007](Decisions.md)). A refusal names the instrument and the time of the bar.

**The fields.** `data.time` is `(T,)` int64 nanoseconds since 1970, UTC; `opens`, `highs`, `lows`, `closes` and `volumes` are `(T, N)`, NaN where an instrument printed no bar, and `bars` holds all five as `(T, N, 5)`; `available` is `(T, N)` bool; `funding` holds the funding events the bars own, at their real times, as a structured array of `time`, `instrument` and `rate`, and `funding_rate` is `(T, N)`, the last rate known at each bar's end; `interval` names the clock; `fingerprint` is 128 bits of blake2b over everything a run on the bars can read, written as `"b2:"` and 32 hex characters, the same however the same bars and instruments arrived and changed by any change to the clock, the interval, the bars, which of them printed, the funding events or the instruments, their order limits as of the fetch among them, a slice's covering the bars before it too, which its features warm up on, and where it starts among them ([ADR 0005](Decisions.md)); and `provenance` says where the bars came from: the source, the venue, the instruments, the interval, the first and the last bar, the warnings that came with them and the fingerprint. With one instrument the singular names read `(T,)`, as `data.close`; with several they are refused, naming the plural and the instrument's own view, `data.closes` and `data["ETHUSDT"].close`.

**Slicing.** A Data slices on time and on instruments, and every slice is a read-only view that keeps the bars before it, so features and an agent's window can warm up on them.

```python
recent = btc["2025-06":"2025-12"]              # June to December, both included
first = btc[:1_000]                            # by position
train, test = universe.split("2026-01-01")     # before it, and from it on
train, test = universe.split(8_000)            # the first 8,000 bars, and the rest
late = universe.after("2026-01-01")
eth = universe["ETHUSDT"]                      # one instrument
pair = universe.select(["ETHUSDT", "BTCUSDT"])
both = spot.join(perp)                         # two markets on one clock
print(universe.gaps())                         # where the bars jump past the interval
```

A text bound names a year, a month, a day, an hour, a minute or a second, and a slice runs from the start of the first to the end of the second; a time names itself and a whole number is a position. `split` and `after` take the same, a period starting the later part where it starts. `join` puts two Datas on one clock, such as a venue's spot and perpetual markets, and `gaps()` lists every place an instrument's bars jump by more than the interval, each as the instrument, the bar before the jump, the bar after it and how many bars are missing between them: on a market that never closes, an outage or a hole in the data. `describe()` gives the rows of a run's assumptions that the bars decide ([Transparency](#transparency)).

**Coarser bars and outside series** reach a Data only from the moment each was known ([ADR 0006](Decisions.md), [Specification](Specification.md#coarser-bars-and-outside-series)).

```python
daily = btc.resample("1d")
trend = btc.asof(probatum.ta.sma(daily.close, 50), daily)   # known at each day's end
oi = btc.asof(open_interest)                                 # by its own timestamps
```

`resample(interval)` gives the same instruments at a whole multiple of the interval: each coarse bar takes the first open, the highest high, the lowest low and the last close of its bars and the sum of their volumes, is available where any of them was, and the funding events keep their stamps. Coarse bars align to 00:00 UTC and weeks to Monday, moved onto the Data's own steps, so hourly bars that open at half past the hour give coarse bars that do too; a coarse bar the bars do not cover to its end is dropped, since its close would be a price still trading, and the first holds the bars from the Data's first, wherever in it they begin. The instruments keep their names, and a slice's coarse bars keep the ones before it, as the slice keeps its bars, and read nothing after its end. `asof(values, when)` puts on row t the last value known by the end of bar t, and NaN before the first. `when` is a Data, or anything a Data is built from, whose values are known at the ends of its bars, or the time each row of `values` became known; where `values` holds a column for each of its instruments, an instrument that printed no bar on a row gave no new value there, so its last one stands. A frame passes alone: a frame of bars is read as a Data is, each value known at its bar's end, a column of a router candle frame at its candle's end, and any other frame or series at its datetime index or its one time column, or a `timestamp` column of milliseconds, its numeric columns being the values. A column of bars from elsewhere carries nothing that says so, so it comes with `when=`, its bars as a Data. A daily bar on this clock ends at the next midnight UTC, after the close of any session that ends by then, a stock's among them; the bars of a day that ends later, such as an FX day from 17:00 New York, belong in UTC or at a finer interval, where their ends are their own.

**Saving.** A Data saves to a directory the user names and loads back under the same fingerprint ([ADR 0005](Decisions.md)).

```python
universe.save("data/thirty")                    # three Arrow files
universe = probatum.Data.load("data/thirty")    # the same bars, next session
```

`save(path)` makes the directory where it is missing and writes three Arrow IPC files that any Arrow reader opens: `bars.arrow`, one row for each bar and instrument with its time, the instrument as its row of `instruments.arrow`, and the five values, null where the instrument printed no bar, with the interval, the provenance and the fingerprint in its schema's metadata; `instruments.arrow`, one row for each instrument with its name and every field of its `Instrument` but `settlement`, which follows from them; and `funding.arrow`, one row for each funding event. A slice saves the bars before it as well, which its features warm up on, and loads back as the same slice. Each file is written under another name and then put in place, the bars last. `probatum.Data.load(path)` reads them back, checks the bars as a Data checks its bars, and refuses them where they do not give the fingerprint they were saved under, so bars fetched once can be studied many times, and a result's fingerprint can be checked against the files it came from. probatum writes and reads the files itself, so saving needs no pyarrow.

<br>

## Venue

A `probatum.Venue` holds a venue's rules for a run in one object, so a setting has one place to be given ([ADR 0002](Decisions.md)). The presets are Venues, each reached as `probatum.venues.NAME` and read when first asked for ([ADR 0028](Decisions.md)), and a venue built by hand starts from the plain option in each row: a setting it leaves out, or gives as `None`, takes the plain option, and `None` inside a table leaves that key out.

```python
import probatum

perp = probatum.venues.binance_perp
cheaper = perp.replace(fees={"taker": 0.0004}, max_open_orders=16)
both = probatum.venues.binance.replace(balance={"spot": 5_000.0, "margin": 5_000.0})
mine = probatum.Venue(fees={"maker": 0.0, "taker": 0.0005},
                      funding={"every_bars": 8, "rate": 0.0001})
print(cheaper.to_dict()["fees"])            # {'maker': 0.0002, 'taker': 0.0004}
```

| Setting | What it is | Plain |
| :--- | :--- | :--- |
| `balance` | the opening cash: a number for an account of one wallet, `{"spot": s, "margin": m}` for one of both | `10_000.0` |
| `fees` | `{"maker": r, "taker": r}`, each a share of the notional | `{"maker": 0.0002, "taker": 0.0006}` |
| `leverage` | the most a perpetual is levered where no tier table caps it; `0.0` removes the cap | `10.0` |
| `liquidation` | `"takeover"`, or `"book_then_backstop"` with `backstop`, the share of the requirement at which the backstop takes over | `"takeover"` |
| `liquidation_fee` | the rate a close at the close point pays where no maintenance is kept | the taker rate |
| `max_open_orders` | the orders each instrument may rest | `8` |
| `fills` | the fill model: `limit`, `"touch"` or `"through"`; `window`, the bars before a fill it reads; `liquidity`, `{"budget": share}`; `spread`, `"none"`, `"estimated"` or `{"fixed": bps}`; `slippage`, `"none"` or `{"fixed": bps}`; `impact`, `"none"`, `{"linear": k}` or `{"sqrt": k}`; and `sizes`, `"on"` or `"off"` | `"touch"`, `24`, `{"budget": 1.0}`, `"none"`, `"none"`, `"none"`, `"off"` |
| `intrabar` | how what a bar cannot show is decided: `"pessimistic"`, `"optimistic"`, or `"random"`, which draws from the run's seed | `"pessimistic"` |
| `funding` | `"none"`, `"data"`, read from the Data, `{"every": hours, "rate": r}` or `{"every_bars": n, "rate": r}` | `"none"` |

**Every instrument takes** the venue's `tiers`, `tiers_by`, `tick`, `lot`, `sigfigs`, `decimals`, `min_notional`, `max_qty` and `reduce_only_exempt`, where it gives them, unless its kind's table in `kinds` or its own in `instruments` gives another, and its fees side by side over the venue's. `instruments` keys a table by its symbol where `kind` names the kind of every instrument the venue lists, and as `kind:SYMBOL` where `kind` is None, as in the presets that combine a venue's spot and perpetual markets, which keep their rates in each kind's table. An instrument's own table alone gives its `delist`, the time the venue delisted it, a time or the text of one held as its moment in UTC, `"2024-01-02 06:00"`: on the first row that opens at or after it, its waiting orders are cancelled and its position is closed at its last close with no fee, and every order on it is refused from then on ([Specification](Specification.md#the-clock)). Each table is written as a preset file writes it, a tier table as a list of `{"up_to": n, "leverage": x, "rate": m, "amount": c}`, an amount left out being 0, and every word in lowercase ([Contributor Guide](Contributor_Guide.md#adding-a-preset)).

**A Venue does not change.** It is hashable, and equal to any venue with the same settings, however their numbers were written. `to_dict()` gives every setting as the constructor takes it, so `probatum.Venue(**venue.to_dict())` is the same venue, and its repr is the code that builds it, `probatum.venues.binance_perp.replace(max_open_orders=16)` or `probatum.Venue(fees=...)`. `replace(**changes)` gives a variant: `fees` and `fills` merge into the venue's own, a table in `kinds` or `instruments` merges into the one it names key by key and its fees side by side, a table given `None` is dropped, which the venue must hold, `liquidation="takeover"` drops a backstop the venue held, and every other setting is replaced whole; a change every kind's table overrides, such as `fees` on a combined preset, reaches no instrument and is refused. Every setting passes the checks a preset's pass as the venue is built, the engine's own among them, so a variant is checked as a new venue is, and a refusal opens with the setting as it was passed, such as `fees.maker` or `instruments.BTCUSDT.tick`, and the value ([ADR 0007](Decisions.md)). `name`, `taken` and `kind` give the preset's name, the day its tables were taken and the kind of the instruments it lists, each None where it is not given, and `describe()` grades what a run on the venue assumes, read from the venue alone ([Transparency](#transparency)).

**A Data runs on a venue as one account.** Each instrument takes the venue's tables by its symbol, or as `kind:SYMBOL` on a venue of several kinds, a market the Data leaves out being the venue's, and the minimum notional and largest size the router gave it, each where no table of the venue gives one. The run reads the bars before a slice, which its window warms up on, leaves out the rows none of its instruments printed, as a selection from a wider universe holds, and where the venue's funding is `"data"`, reads the Data's funding events. Refused before it runs are:
- an empty Data, and instruments at two venues;
- an unnamed instrument where the venue keys tables by symbol, since none can name it;
- an account whose cash would be two assets: perpetuals that settle in two, or spot pairs quoted in another asset than the account's;
- an instrument with no market on a venue of several kinds, or of a kind the venue does not list;
- an inverse contract whose bars do not give its size, or beside other instruments;
- one balance for an account whose spot and perpetual instruments use both wallets, and two for an inverse contract's, which has one;
- funding from the data where the Data holds none for a perpetual, which `Data.from_router(..., funding=True)` fetches;
- significant figures with no decimals, and where the venue's sizes are on, an instrument it gives no lot or price rule, as a preset gives none for one it does not list;
- where the venue books then backstops, a perpetual it gives no maintenance;
- and a delisting time at or before the open of its instrument's last bar, which would trade it after the venue delisted it.

<br>

## Data input

`probatum.to_ohlcv(data)` returns the `(T, 5)` float64 array the engine takes, from any of three inputs:

| Input | Handling |
| :--- | :--- |
| numpy array | Must be `(T, 5)` OHLCV; returned contiguous and float64. |
| pandas DataFrame | Needs `open`, `high`, `low`, `close`, `volume` columns; any extra column is ignored. |
| parquet path | Read with pandas, then treated as the DataFrame case. |

**Volume is in the units an order's size is in**: base units, or contracts on an inverse perpetual. The liquidity budget is a share of it and the market-impact term divides a fill's size by it, so a feed shipping quote volume instead inflates the budget by roughly the price: `budget` stops binding, impact goes to zero, and nothing says so. An inverse feed in coin rather than contracts errs the other way. The router's frames carry base `volume` alongside a separate `volume_usd`, and only the first is read.

A frame carrying a real (datetime) index must be sorted ascending and unique; a plain `RangeIndex` holds no timestamps, so it is exempt. A missing column, a wrong shape, or an unsorted index raises `ValueError`, and an unsupported type raises `TypeError`.

**Every row has to be a bar**: `low <= open <= high` and `low <= close <= high`, checked per row alongside finiteness ([ADR 0007](Decisions.md)). The fill model bounds a taker to what the bar traded, and on an unordered row that bound moves the fill toward the account instead, so a dead flat market pays out. The realistic cause is a positional slice of a feed that ships its columns in another order, which is why the message names the order it wants: a venue returning `low, high, open, close` satisfies `high >= low` on every row and still mints money. A bar that never moved, every price equal, is a bar. pandas and pyarrow are imported only when a frame or a path is passed, so the numpy path needs neither installed.

The wrappers call it for you, so a DataFrame can go straight into `Backtester` or `VectorEnv`. Handing the `Backtester` a datetime index is also what stamps each round trip with `open_time` and `close_time` ([Trades](#trades)).

<br>

## The market

A backtest, a search and an RL rollout each take the same eleven of the engine's knobs, and leave its `window`, `spread`, `impact_model`, `limit` and `sizes` at their defaults: a window of 24 bars, no spread, linear impact, limits filled on touch and sizes off. Written out at three call sites they are three chances to disagree, and the claim that one fill model sits behind every surface was being held up by you retyping them identically. `probatum.Market` holds them once and hands out the surfaces itself.

```python
import probatum

binance = probatum.Market(
    kind="perp",              # "spot" or "perp"
    fee_taker=0.0004,
    fee_maker=0.0002,
    slippage_bps=2.0,
    leverage=5.0,
    funding_rate=0.0001,
    funding_interval=8,
)

result = binance.backtest(candles).run(MyStrategy())
study = binance.tune(MyStrategy, space, candles, n_trials=200, oos=0.3)
env = binance.env(candles, num_envs=4096, window=60)
engine = binance.engine(candles)          # takes a frame like its siblings
```

Each method takes only the arguments that are **not** the venue, so a knob cannot be passed twice and no rule about which copy wins is needed: there is only ever one copy. Passing one anyway is refused rather than merged ([ADR 0002](Decisions.md)).

| Call | Gives |
| :--- | :--- |
| `engine(candles, report=False)` | [the engine](#the-engine-it-drives) the strategy runner drives. |
| `backtest(candles, periods_per_year=None, risk_free=0.0)` | a `Backtester`. |
| `tune(strategy, space, data, **search)` | a `TuneResult`; `search` is the search controls only. |
| `env(data, **rest)` | a `VectorEnv`; `rest` is the RL arguments only. |
| `replace(**changes)` | a copy with some knobs changed. |
| `as_dict()` | the knobs as the keyword arguments the engine takes. |

`replace` is what makes a cost comparison honest: `binance.replace(fee_taker=0.001)` differs in exactly one number and is visibly the same venue otherwise. A market prints only the knobs that differ from the defaults, so `repr` reads as the handful of choices you actually made. Two markets with the same knobs compare equal.

The keyword form is untouched and remains the right thing for a one-off: `Backtester(candles, market="perp", fee_taker=0.0004)`. `Market` is for when the venue is fixed and the surfaces are many.

<br>

## Engine

`probatum.Engine(data, venue, seed=None)` is an account at `venue`, a Venue or a preset's name, on `data`, a Data or anything one is built from, for a loop the caller owns. It is the fourth depth beside an array, a Strategy and an agent, and records the same Result ([ADR 0002](Decisions.md)).

```python
engine = probatum.Engine(data, "binance_perp")
bar = engine.reset()
while not engine.done():
    decision = service.decide(bar, engine)     # anything that returns weights or None
    if decision is not None:
        engine.target(decision)
    bar = engine.step()
result = engine.result()
```

| Call | Returns | Does |
| :--- | :--- | :--- |
| `reset()` | a `Bar` | Open the account again at the Data's first bar, and return that bar. |
| `step()` | a `Bar` | Fill on the next bar what was decided at the last close, mark the account at that bar's close and return the bar; at the last bar the account stays where it is. |
| `done()` | bool | True at the last bar, which no bar follows to fill on. |
| `result(risk_free=0.0)` | a `Result` | What the account has recorded from its first bar to the current one, as `backtest` returns it, with no `policy`. |

Between steps the engine is the account a Strategy's `next` reads, with the same state and the same verbs ([A strategy](#a-strategy)), and its `events` hold what happened to its orders since the last step. `seed` names the stream a venue whose `intrabar` is `"random"` draws from, which `backtest` refuses so that a backtest repeats without one ([Specification](Specification.md#what-a-bar-cannot-show)). emsl's engine, built over candles with keyword settings, is `probatum._probatum.Engine`, described with [the strategy runner](#the-engine-it-drives) that drives it.

<br>

## Batch

Many independent envs over one shared candle series, stepped in parallel with the GIL released. Every env sees the same candles (shared by `Arc`, no duplication) but carries its own account and orders.

```python
import numpy as np
from probatum import Batch

b = Batch(candles, num_envs=512, market="spot", quote=10_000.0)  # same engine knobs
b.reset_all()                                # list of per-env states
actions = np.zeros(b.num_envs)               # one signed size per env
states = b.step_all(actions)                 # positive buys, negative sells, zero holds
```

| Call | Returns | Does |
| :--- | :--- | :--- |
| `reset_all()` | list of states | Reset every env, in parallel. |
| `step_all(actions=None)` | list of states | Apply the per-env signed-size actions (market orders) and step every env in parallel with the GIL released. |
| `target(weights, tolerance=0.0, transfer=False)` | `None` | Bring every env to its row of an `(num_envs, N)` array of weights of equity, refusing a non-finite weight, or a negative one on spot, before any env places anything; each leg fills at its instrument's next open, or waits for the next bar it prints. |
| `records(env)` | dict | What env `env` recorded, as `run` returns it: its ledger, fill log, order log and round trips, each `None` without `report=True`. |
| `done()` | bool | True when the envs have reached the last bar (they step in lockstep here). |
| `num_envs` | int | Number of envs (also `len(batch)`). |

Because the envs are independent, batched stepping is bit-identical to stepping each in a loop, but under `intrabar="random"`, where env k draws from a stream named with its index as well, which the engine built on its own does not ([Specification](Specification.md#what-a-bar-cannot-show)). The `Batch` also carries a lower-level RL surface (random-start reset, masked autoreset, batched feature observation, and per-env equity/done/bust arrays) used by [`probatum.rl.VectorEnv`](RL_Guide.md); most users drive RL through that env rather than these directly.

Each cost knob also takes an optional per-env override: pass `fee_taker_per_env`, `fee_maker_per_env`, `slippage_bps_per_env`, or `impact_per_env` as a length-`num_envs` float64 array and each env carries its own cost while sharing the one candle buffer; a fee override replaces the venue's rate and leaves alone a rate an instrument charges of its own. This is the primitive [`VectorEnv`](RL_Guide.md) uses for cost domain randomization ([ADR 0038](Decisions.md)); an override whose length is not `num_envs` is a `ValueError`.

**A batch over a universe** takes `(T, N, 5)` candles and the same `instruments` and `funding_instruments` as [the engine](#the-engine-it-drives), and steps each env through `target`; the signed sizes `step_all` and `advance` take trade the first instrument. Its `observe(window)` is `(num_envs, window, N, 5)`, zero where an instrument printed no bar, and `snapshot()` adds every instrument's `positions` and `marks` as `(num_envs, N)` arrays beside its `(num_envs,)` fields, whose position, unrealized PnL and mark read the first instrument.

Under `intrabar="random"` each env draws from its own stream, named from `seed` and the env's index, so each env decides the bars' hidden events apart from the others.

<br>

## Reinforcement learning

`probatum.rl.VectorEnv` is a Gymnasium-compatible vectorized env over the `Batch`: `num_envs` independent envs share one copy of the series, each starting at a random offset, all stepped in parallel with the GIL released. You supply what the agent sees (`features`), how it is rewarded (`reward_fn`), and what its actions mean (`action_fn`, `action_space`).

```python
from probatum.rl import VectorEnv

env = VectorEnv(candles, features=indicators, num_envs=4096, window=60, market="perp")
obs, info = env.reset(seed=0)                  # (4096, 60, F) float32
obs, rewards, terminations, truncations, infos = env.step(actions)
```

Stable-Baselines3 will not take a Gymnasium vector env directly, so `probatum.sb3.ProbatumVecEnv` presents it as an SB3 `VecEnv`, which keeps the batch intact under PPO, A2C, DQN, and SAC:

```python
from stable_baselines3 import PPO
from probatum.sb3 import ProbatumVecEnv

PPO("MlpPolicy", ProbatumVecEnv(env)).learn(total_timesteps=100_000)
```

The full contract, every constructor argument, the observation and reward shapes, the action decoder, cost randomization, and the autoreset rules, is in the [RL Guide](RL_Guide.md).

<br>

## Backtesting

`probatum.backtest(policy, data, venue, sizing="weight", stop=None, take_profit=None, tolerance=0.0, record="full", risk_free=0.0)` runs a policy on a Data at a venue and returns its `Result`. `data` is a `probatum.Data` or anything one is built from, and `venue` a `probatum.Venue` or a preset's name. The policy is read in one fixed order, so no input is read two ways ([ADR 0024](Decisions.md)):

| Order | Policy | How it runs |
| :--- | :--- | :--- |
| 1 | a table: anything exporting the Arrow C stream interface, or a pandas frame or series | its rows matched to the Data's clock by their times, in a datetime index or column or a `timestamp` column of milliseconds, and a daily Data's by their dates, and its columns to the instruments by name, then run as an array; a time or a name that does not match is refused, and a row may be left out only where no instrument printed a bar |
| 2 | an array: anything with `__dlpack__`, the buffer protocol, `__array__` or `__arrow_c_array__` | converted once to float64 on the host and run whole in Rust |
| 3 | a class | refused: pass an instance |
| 4 | a `Strategy` | a deep copy of it, `init(data)` once and `next(bar, account)` at each close ([below](#a-strategy)) |
| 5 | an object with `fit` that is not callable | refused: pass its predictions as an array |
| 6 | an agent: a Stable-Baselines3 model, a torch module or any other callable | not run yet |
| 7 | anything else | refused |

**An array** is `(T, N)` over the Data's rows and instruments, or `(T,)` for one instrument. Row t is the decision at the close of bar t and a NaN holds, and the engine still steps every bar, so an array run is no vectorized backtest. With `sizing="weight"` each value is a weight of equity, reached as `target` reaches it, a leg below `tolerance` of equity skipped; with `sizing="quantity"` it is the position wanted, in base units, or in contracts on an inverse contract. The last row is never read, since no bar follows it to fill on. An array of booleans is refused, since False would flatten where it means to hold: 1.0 enters, 0.0 leaves and NaN holds. A negative value on a spot instrument is refused before the run, and so is a venue that decides at random what a bar cannot show, so a backtest repeats without a seed ([Specification](Specification.md#an-array)).

```python
fast = probatum.ta.ema(btc.close, 20)
slow = probatum.ta.ema(btc.close, 80)
weights = numpy.full(len(btc), numpy.nan)                     # NaN holds the position
weights[probatum.ta.crossover(fast, slow)] = 1.0              # enter long
weights[probatum.ta.crossunder(fast, slow)] = 0.0             # flatten
venue = probatum.venues.binance_perp.replace(funding="none")  # btc holds no funding
result = probatum.backtest(weights, btc, venue, stop=0.04, take_profit=0.08)
```

**Barriers.** `stop` and `take_profit` give each round trip of an array run a stop and a take-profit at those returns from its average entry, each a number, or a `(T, N)` array or a table of them read on the row whose decision opened the round trip, NaN for none. They rest from the close of the bar the entry filled on and move when an add moves the average entry, and a barrier that fills, even in part, holds its instrument flat until a row asks for zero or for the other side, so the rows still asking for the position do not buy it straight back ([Specification](Specification.md#barriers-on-an-array)).

**The Result** holds what the run recorded: its equity and returns, its tables, its statistics, the policy it ran and its provenance ([The Result](#the-result)).

**The recording level.** `record="account"` keeps the ledger, the round trips, the costs and the statistics of the run at the full level, to the bit, its assumptions priced where its ledger and round trips measure them, and none of its holdings, fill log, order log, baselines or policy, for a run read for its numbers alone ([Specification](Specification.md#recording-levels)). A Strategy still runs at the full level, since its account reads the order log, and keeps the account level's records when it ends. `"transitions"` is not recorded yet, and `"none"`, which keeps nothing a Result holds, is refused.

```python
light = probatum.backtest(weights, btc, venue, stop=0.04, take_profit=0.08,
                          record="account")
print(light)                                    # names the level it was recorded at
print(light.stats == result.stats, light.trades == result.trades)
```

### A strategy

A `probatum.Strategy` decides at each close. `init(data)` runs once over the whole Data before the first bar, the place to compute indicators over every bar at once, and `next(bar, account)` runs at the close of every bar from `warmup` on, counted in the Data's rows, except the last, which no bar follows to fill on, and a row none of the instruments printed. What `next` places fills on a later bar, never on that one. `warmup` is a class attribute or is set in `init`, and should cover the longest history a decision reads, since an index below zero reads numpy's array from its end. A strategy's parameters are its constructor's arguments, kept as attributes of the same names. At a validation's seam `on_inherit(bar, account)` is called before the incoming window's first decision, once every order the window before it left is cancelled, so a strategy whose exits rest on the book can place them again for the position it inherits ([Validation](#validation)). `backtest` runs a deep copy of it, so two runs of one instance start alike, and the copy, holding what `init` computed, is the Result's `policy`. `stop`, `take_profit`, `sizing` and `tolerance` belong to an array run and are refused beside a Strategy, which places its own exits and gives `target` its own tolerance ([Specification](Specification.md#a-strategy), [ADR 0025](Decisions.md)).

```python
class Breakout(probatum.Strategy):
    warmup = 21

    def init(self, data):
        self.high = probatum.ta.highest(data.high, 20)
        self.atr = probatum.ta.atr(data.high, data.low, data.close, 14)

    def next(self, bar, account):
        atr = self.atr[bar.i]
        if account.position == 0.0 and bar.close > self.high[bar.i - 1]:
            account.buy(account.equity * 0.01 / (2.0 * atr))
            self.exits = account.bracket(stop=bar.close - 2.0 * atr,
                                         take_profit=bar.close + 4.0 * atr)
        elif account.position > 0.0:
            trail = bar.close - 2.0 * atr
            if trail > self.exits.stop.trigger:
                account.replace(self.exits.stop, trigger=trail)


result = probatum.backtest(Breakout(), btc, venue)
```

**The bar** is the bar just closed: `i`, its row of the Data, by which an array computed in `init` is read; `time`, its open as a `datetime` in UTC, and `ns`, the same in int64 nanoseconds since 1970; `n`, the number of instruments; with one instrument `open`, `high`, `low`, `close`, `volume` and `funding_rate`, the last rate known at the bar's end, NaN before the first, and with any number their plural forms, each `(N,)` and NaN where an instrument printed no bar; `available`, which instruments printed it; and `bar["ETHUSDT"]`, one instrument's bar. With several instruments the singular forms are refused with the fix, as a Data's are. `bar.history(n)` is the last `n` bars up to and including this one as a read-only Data, the bars before a slice's first among them, and `bar.history()` every bar so far, so logic computed in `next` cannot read a later bar.

**The account** is the one the bar's close left, in the venue's terms: `equity` and `leverage`; `position` and `entry` with one instrument, and `positions`, `(N,)`, with any number; `orders`, the orders still open, resting on the book or pending until the next open; `events`, what happened to its orders since the last close, oldest first, the orders placed at that close among them; `trades`, the round trips closed so far, each [column](#trades) a read-only array; `equity_curve`, the equity at the opening and at every bar since, one a row its instruments printed, as the Result's `equity` counts them, both starting again at each seam of a validation's path; and the two wallets, `spot` and `margin`, each with its `cash` and `equity`, the margin wallet also with its maintenance `requirement` and `available`, its equity less the margin the cap asks, floored at zero. `account["ETHUSDT"]` is one instrument's view, with its `position`, `entry`, `upnl`, `weight` and `orders` and the verbs that place an order on it. With several instruments those verbs are taken through an instrument's view and refused on the whole account with the fix; `target`, `transfer` and `cancel_all` act on the whole account. Every bar the account names is the Data's row, as `bar.i` is.

| Verb | Places |
| :--- | :--- |
| `buy(size)`, `sell(size)` | a market order of `size` base units, or contracts on an inverse contract, for the next open |
| `buy(size, limit=price)`, `sell(size, limit=price)` | a limit order at `price` |
| `buy(size, stop=trigger)`, `sell(size, stop=trigger)` | a stop that triggers at `trigger` and fills as a taker |
| `buy_value(amount)`, `sell_value(amount)` | a market order for `amount` of the quote currency, its size the amount over its fill price ([Specification](Specification.md#units)) |
| `target(weights, tolerance=0.0, transfer=False)` | the legs that bring each instrument to its weight of equity, `weights` one number with one instrument, one for each in order, or a dict from name to weight ([ADR 0009](Decisions.md)) |
| `bracket(stop=None, take_profit=None)` | an OCO pair of reduce-only exits with no size of their own, at those prices |
| `close()` | a reduce-only market order that flattens the position, or None where it is flat |
| `cancel(order)`, `cancel_all()` | the cancellation of an order still open, resting or waiting for its open, or of every one |
| `replace(order, price=None, trigger=None, size=None)` | the same order moved under its id, a limit by its `price` and a stop by its `trigger`, or None where it no longer rests or cannot move |
| `transfer(amount, to="margin")` | cash moved between the wallets at the start of the next bar ([ADR 0008](Decisions.md)) |

`buy` and `sell` also take `tif`, `"gtc"`, `"ioc"` or `"fok"`, on a market or a limit order, `post_only` on a limit and `reduce_only` on any order; their value forms take `tif`, `"ioc"` or `"fok"`, and `reduce_only`.

**Every placing verb returns an `Order`**, a live handle that shows the order as it stands on any later bar: its `id`, `instrument`, `side`, None for a bracket's leg until it arms, `kind`, `size`, None for a bracket's leg that closes the position it finds and for a value order, whose `filled` is what its amount bought, `filled`, `price`, `trigger`, `placed`, the Data's row it was placed at, `status` and `reason`, what the order log said last of it. Its status is `pending` while a market order waits for the next open and `resting` while a limit or a stop is on the book, then `filled`, `cancelled`, `expired`, as an order is once its time in force ends what is left of it, or `refused`, since a refused order is returned too, with its reason ([ADR 0010](Decisions.md)). `bracket` returns both legs as `.stop` and `.take_profit`, and `target` the list of its legs, refused ones included. Each of the account's events is an `Event` with its `kind`, `placed`, `armed`, `filled`, `clamped`, `refused`, `not_filled`, `expired` or `cancelled`, its `order`, None for a transfer's clamp, its `instrument`, its `bar`, its `side`, `size` and `price` as the order log gives them, and its `reason`.

### The strategy runner

`probatum.Backtester` drives a `Strategy` over one series, as emsl did. Reporting is always on, so the result carries the stats, equity curve, and trades.

```python
from probatum import Backtester, Strategy

class SmaCross(Strategy):
    warmup = 30                               # next() waits for 30 bars of history

    def init(self, engine):
        self.close = engine.closes            # optional; runs once after reset

    def next(self, state, engine):
        i = state["tick_index"]
        fast = self.close[i - 10:i].mean()
        slow = self.close[i - 30:i].mean()
        if state["position"] == 0 and fast > slow:
            engine.market_buy(1.0)
        elif state["position"] > 0 and fast < slow:
            engine.close()

result = Backtester(candles, market="spot", fee_taker=0.0006).run(SmaCross())
print(result.stats["sharpe"], result.stats["max_drawdown_pct"])
print(result.equity_curve[-1], len(result.trades))
```

`Backtester(candles, market, quote, fee_taker, fee_maker, slippage_bps, budget, max_open_orders, leverage, impact, funding_rate, funding_interval, periods_per_year, risk_free)` shares the engine knobs and adds the two stats parameters. `run(strategy)` returns a `BacktestResult` with `.stats` (dict), `.equity_curve` (numpy array), `.trades` (the [round trips](#trades), a list of dicts), `.fills` (the [fill log](#fills)), `.initial` (the balance the run opened with, which the curve does not contain because it records a point per advance), `.periods_per_year` (the annualization the stats were computed at), `.bust` (the account reached zero, which the round trips cannot always say: a forced close ends one with the reason `liquidation`, but an account drained by fees or funding ends none at all), and the identity fields `.config`, `.data_hash`, `.strategy` and `.version` that let one run be told from another ([Telling two runs apart](#telling-two-runs-apart)).

**`periods_per_year` is read from the candles.** Hand it a DataFrame or a parquet path with a datetime index and it counts the bars the index holds, the rows less one per 365 days of their span, so hourly candles annualize at 8760 without being told, a feed missing a day annualizes at the bars it actually has, and a stock's daily bars at the 252 or so a year it trades. A numpy array carries no timestamps, so it warns and falls back to 365 rather than assuming. Passing a number overrides all of it ([ADR 0004](Decisions.md)). Subclass `Strategy` and override `next(state, engine)`; `init(engine)` is optional. The same `Strategy` is the unit [tuning](#tuning) searches: declare the tunable parameters as constructor arguments, and `tune` builds a fresh strategy per trial.

A strategy uses any order type, the sizing helpers, and the state's `open_orders` to manage risk. This one sizes each entry to half of equity and rests a reduce-only stop-loss under the position:

```python
class SmaWithStop(Strategy):
    def __init__(self, fast=10, slow=30, stop_pct=0.05):
        self.fast, self.slow, self.stop_pct = fast, slow, stop_pct

    def init(self, engine):
        self.close = engine.closes                     # zero-copy view of every close
        self.warmup = self.slow                        # read after init, so compute it here

    def next(self, state, engine):
        i = state["tick_index"]
        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()
        pos = state["position"]
        if pos == 0 and fast > slow:
            engine.market_buy(engine.qty_from_weight(0.5))          # enter with half of equity
        elif pos > 0:
            if fast < slow:
                engine.close()                                      # exit on the crossover
            elif not state["open_orders"]:                          # rest the stop once, while held
                trigger = state["bar_close"] * (1.0 - self.stop_pct)
                engine.stop("sell", pos, trigger, reduce_only=True)
```

The stop rests only once a position is held (orders fill on the next bar, so there is nothing to protect until then), and `reduce_only` keeps it from flipping the position short instead of closing it. This same class is ready for [tuning](#tuning): its `fast`, `slow`, and `stop_pct` are already constructor arguments.

### The engine it drives

The Backtester and `tune` drive emsl's engine, `probatum._probatum.Engine`, which `probatum.Market` hands out as well: built over a candle array with the engine's settings as keywords, it is reset, given orders, stepped and read a bar at a time, and the `Batch` and the RL environment take the same settings.

```python
import probatum

eng = probatum._probatum.Engine(
    candles,                 # (T, 5) float64 open, high, low, close, volume, or (T, N, 5)
    market="spot",           # "spot", "perp", or "inverse" with contract=
    quote=10_000.0,          # opening cash, or {"spot": s, "margin": m}
    fee_taker=0.0006,        # taker fee, fraction of notional
    fee_maker=0.0002,        # maker fee, fraction of notional
    slippage_bps=0.0,        # slippage on market fills, basis points
    budget=1.0,              # the share of the trailing volume a bar's fills may take
    window=24,               # the bars before a fill its window reads
    max_open_orders=8,       # resting-order slots
    report=False,            # True keeps the equity curve, round trips and logs
    record="full",           # or "account", which keeps no legs, fills or order log
    leverage=10.0,           # perp margin cap on notional; 10x default, 0.0 = none
    impact=0.0,              # impact coefficient on the window; 0.0 means none
    impact_model="linear",   # or "sqrt", in the fill's share of the trailing volume
    spread=None,             # a fixed spread in basis points, or "estimated"
    limit="touch",           # when a resting limit fills, or "through", half a step past it
    sizes=False,             # True holds orders to the lot, the price rule and the bounds
    funding_rate=0.0,        # perp funding per event on notional; long pays, short receives
    funding_interval=0,      # bars between funding events; 0 disables funding
    times=None,              # (T,) int64 open times in UTC nanoseconds: the clock
    funding_hours=0.0,       # or funding every this many hours from 00:00 UTC; needs times
    funding_times=None,      # or the venue's own (K,) int64 stamps; needs times
    funding_rates=None,      # and the (K,) rate of each stamp
    funding_instruments=None,  # the instrument each stamp charges, or None for every perp held
    tiers=None,              # or the venue's tiers: rows of (up_to, leverage, rate, amount)
    tiers_by="notional",     # what the tiers' bounds measure: "notional" or "quantity"
    lot=0.0,                 # the lot every size is a whole number of; 0.0 = none
    tick=0.0,                # the tick prices sit on; 0.0 = none
    sigfigs=None,            # or Hyperliquid's significant figures, with decimals=
    decimals=None,           # the most decimals such a price keeps
    min_notional=0.0,        # the smallest notional an order may have
    max_qty=None,            # the largest size an order may have
    reduce_only_exempt=False,  # True spares a reduce-only order the minimum notional
    instruments=None,        # a universe's market, tiers, size rules, fees and delist, a dict each
    intrabar="pessimistic",  # how what a bar cannot show is decided, "optimistic" or "random"
    seed=None,               # the run's seed, which "random" draws from
    contract=None,           # the dollars one inverse contract is worth
    liquidation="takeover",  # or "book_then_backstop", which sells on the book first
    backstop=None,           # the share of the requirement book then backstop takes over at
    liquidation_fee=None,    # a takeover's rate with no maintenance; None = the taker rate
)
```

**A tier table caps a perp by the tier its position falls in**, in place of the one `leverage`: up to each bound of notional, or of size with `tiers_by="quantity"`, the position may be levered no further than its tier allows, and a fill is held at the table's last bound, unless that bound is `float("inf")`, left open as by a venue that sets no largest position. The table is refused where the Specification's loader refuses it: bounds that do not rise, a leverage that rises or a maintenance rate that falls from one tier to the next, a maintenance rate not below one or reaching its tier's initial rate, one over the leverage, a notional table whose first amount is not zero or whose amounts break continuity, and any value that is not finite, but a last bound left open, or is below zero ([Specification](Specification.md#maintenance)).

**A venue's sizes hold every order to its instrument's grid.** With `sizes=True` a size is rounded down to the `lot` when it is placed, and so is every fill after its clamps, and an order that rounds to no lot is refused; a limit or a trigger is rounded to the `tick`, or to `sigfigs` significant figures within `decimals` decimals, Hyperliquid's rule, where a whole-number price is always allowed and an asset with `n` size decimals takes a lot of `10**-n` and `decimals` of `6 - n` on a perpetual and `8 - n` on spot; an order below `min_notional` at the last close is refused, a reduce-only one spared where `reduce_only_exempt` says the venue spares it; one above `max_qty` is held to it, rounded down to the lot; and a position floating point leaves below half a lot is zeroed at the close as dust. On an inverse contract the lot and `max_qty` are in contracts, the tick in dollars and `min_notional` in the coin. `limit="through"` fills a limit only once its bar trades half a step of the price rule past it, and with no price rule it fills on touch ([Specification](Specification.md#sizes-and-dust)).

**A universe of N instruments shares one account.** Pass `(T, N, 5)` candles, NaN in all five values where an instrument printed no bar, with `instruments`, a dict per instrument giving its `market`, `tiers`, `tiers_by`, size rules and, where it charges apart from the venue, its own `fee_maker`, `fee_taker` or both, a side it leaves out charging the venue's rate, and where the venue delisted it, its `delist` time in nanoseconds since 1970, which needs `times=` and comes after the open of its last bar, or `market`, `tiers` and the size rules for every one. Each instrument fills on its own bar and waits where it printed none, `target` brings each to its weight of equity, and the views carry an instrument axis: `shape` is `(T, N, 5)`, `data` and `observation` are `(T, N, 5)` and `(rows, N, 5)`, and `closes` and its companions `(T, N)`. The order shortcuts trade the first instrument, and the state `step` returns reads its position and bar beside the account.

**An inverse contract is counted in its coin.** `market="inverse"` with `contract=`, the dollars one contract is worth, takes one instrument's `(T, 5)` candles, quoted in dollars. Sizes are contracts and prices are dollars, as the venue quotes them, while `quote`, equity, fees, funding and every other sum are in the coin, a taker's costs among them, each its rate of the coin notional ([Specification](Specification.md#inverse-contracts)). The account holds that one contract in one wallet, so there is no `transfer`, `qty_from_quote(cash)` is `cash / contract` contracts at any price, and the ledger's `equity_in_quote` values equity in dollars at each close ([ADR 0016](Decisions.md)).

**A venue that books then backstops sells a closed-out account on the book.** `liquidation="book_then_backstop"` with `backstop=`, the share of the requirement at which the venue's backstop takes over, two thirds on Hyperliquid, sells every leg as a taker from its trigger price, paying the taker fee and a taker's costs, and the trader keeps what the margin wallet then holds. Where the sale would leave the wallet below the backstop level, or the path also reaches that level and the bar cannot show that the sale finished first, which `intrabar` decides, the account is taken over instead, as under the default `"takeover"`, which closes every leg where the margin ran out and books what the wallet still holds as the liquidation charge. It needs a perp whose tier table keeps maintenance, since with none the backstop level is the trigger itself ([Specification](Specification.md#the-forced-close), [ADR 0019](Decisions.md)).

**A venue's liquidation fee places the close point of an account with no maintenance.** The close point is where equity would just pay the fee on closing every position, at `liquidation_fee` where it is given and each instrument's taker rate where it is not. An account with no maintenance whose path reaches its trigger leaves the market from the point where its equity stays at or below that level all the way to the trigger, and its takeover charges the fee on each close. The margin bound charges it on what a reducing fill leaves of the position and on the other perps, wherever the account the fill leaves keeps no maintenance. With maintenance the rate is not used: the account leaves at its trigger, its takeover pays no fee of its own, and the bound charges the taker rate. It is refused below 0 and from 1 ([Specification](Specification.md#the-trigger-close-and-clip-points)).

`eng.shape` is `(T, 5)`, the number of candles the engine loaded; ask for 1000 and see `(873, 5)` and you know the source returned fewer, with no guessing.

**The account keeps two wallets**, as a venue does. The spot wallet holds cash and spot holdings, never borrows and is never liquidated; the margin wallet holds the collateral behind a perp, and the cap, the liquidation and funding read it alone. A number for `quote` opens the wallet the market settles in, the spot wallet on spot and the margin wallet on a perp, and `{"spot": s, "margin": m}` opens each by name. Cash crosses only by `transfer`, which out of the margin wallet leaves unrealized profit and the cap's requirement behind. Equity is both wallets together, so a perp closed out with cash in the spot wallet leaves an account that is still alive ([ADR 0008](Decisions.md)).

### Stepping

| Call | Returns | Does |
| :--- | :--- | :--- |
| `reset()` | state | Reset the account and cursor to the first bar. |
| `step()` | state | Advance one bar, resolve the orders placed since the last step, mark, return the new state. |
| `done()` | bool | True when there is no next bar to step into. |
| `run(strategy)` | final state | Drive a strategy over the whole series (see [below](#running-a-strategy)). |

### Orders

| Call | Returns | Does |
| :--- | :--- | :--- |
| `market_buy(size)` / `market_sell(size)` | id or `None` | Taker order; fills at the next bar's open, pays the taker fee, half the spread, the slippage and the impact. `None` when the queue already holds `max_open_orders` for this bar, the size is not a positive finite number, or the venue's sizes refuse it. |
| `limit_buy(size, price)` / `limit_sell(size, price)` | id or `None` | Limit order; rests until a later bar reaches the price, and pays the maker fee unless its first bar opens at or through it. `None` if the book is full, the size is not a positive finite number, it would trade against the account's own resting limit, or the venue's sizes refuse it; a non-finite price raises. |
| `stop(side, size, trigger, reduce_only=False)` | id or `None` | Becomes a market order once a bar crosses `trigger`. `side` is `"buy"` or `"sell"`. `None` if the book is full, the size is not a positive finite number, the trigger is not finite, or the venue's sizes refuse it. Pass `reduce_only=True` for a stop-loss. |
| `bracket(stop=None, take_profit=None)` | two ids or `None`s | A stop and a take-profit as one OCO pair, each a reduce-only leg with no size of its own that closes the position it finds. A leg takes its side from the position it arms on, at once when one is held and otherwise at the first fill that opens one, so a leg in a later slot can close it on that bar; the first fill of either cancels the other, and the bracket goes when that position closes or flips. `None` for a leg not asked for or refused; a non-finite price raises ([ADR 0010](Decisions.md)). |
| `order(side, size, type, price, trigger, reduce_only, post_only, tif)` | id or `None` | The primitive the shortcuts wrap; the only call that sets `post_only` and `tif`. |
| `close()` | id or `None` | Queue a reduce-only market order sized to the whole position, held to `max_qty` where the venue's sizes are on. `None` when flat or where those sizes refuse it. |
| `cancel(order_id)` | bool | Drop an order not yet filled, resting or waiting for its open. True if it was found. |
| `cancel_all()` | int | Drop every order not yet filled, those waiting for their open first; returns how many were dropped. |
| `transfer(amount, to)` | `None` | Move cash into the `"spot"` or `"margin"` wallet from the other. It lands at the start of the next bar, before any fill, held to what the wallet it leaves can give ([ADR 0008](Decisions.md)). |
| `target(weights, tolerance=0.0, transfer=False)` | list of ids | Queue the market orders that bring each instrument to a weight of equity at the last close, every reducing leg before every growing one. `weights` is a number with one instrument, one weight per instrument, or a dict from instrument to weight; a leg below `tolerance` of equity is skipped, and `transfer=True` moves the spot legs' shortfall across from the margin wallet. A second target in one decision replaces the first, and a later one replaces the legs earlier ones left waiting on what it names ([ADR 0009](Decisions.md)). |
| `replace(order_id, size=, price=, trigger=)` | id or `None` | Move a resting order under the same id, with the same side, type and flags, into the first free slot. `None`, and nothing changed, if it is no longer resting or the moved order cannot rest ([ADR 0010](Decisions.md)). |
| `is_bust()` | bool | True when the account died: its equity, both wallets together, reached zero, which a takeover does unless the spot wallet holds cash ([ADR 0008](Decisions.md)). The engine does not stop on it. |
| `num_fills()` | int | Fills applied since the reset. Zero beside orders you placed means none of them filled. |
| `qty_from_weight(fraction)` | float | Base size for `fraction` of current equity, at the current close. |
| `qty_from_quote(cash)` | float | Base size for a `cash` amount in quote, at the current close. |

`order(...)` is the full primitive: `type` is `"market"`, `"limit"`, or `"stop"`, `tif` is `"GTC"`, `"IOC"`, or `"FOK"`, and it carries the three flags the shortcuts leave at their defaults. `reduce_only` marks a take-profit or stop-loss that can only shrink the position; `post_only` refuses a limit that would cross the last close, and cancels one its first bar, or the part of it after a fill, opens at or through, rather than turning it taker; `tif` decides what happens to the part of an order the next bar does not fill: `GTC` rests, `IOC` takes one bar then cancels the remainder, `FOK` fills the whole size against the bar or nothing. A market order is `IOC` unless you ask for `FOK`, since it never rests and so cannot tell `GTC` from `IOC`; a stop rests until it triggers ([ADR 0010](Decisions.md)). `FOK` is all or nothing against every clamp, not only the bar's liquidity, so a `FOK` the spot cash balance or the margin cap could only fill in part books nothing ([ADR 0010](Decisions.md)). A limit needs a `price` and a stop a `trigger`, else it is a `ValueError`, and a non-finite price or trigger is a `ValueError` too ([ADR 0007](Decisions.md)).

**A bar that reaches both a stop and a take-profit books the stop.** The engine cannot see the path inside a bar, so it cannot know which the price touched first, and it used to answer with book-slot order: place the stop before the take-profit and every ambiguous bar booked a loss, swap the two lines and every one booked a win. Exits from one position are now applied worst first, so the result is a rule rather than a property of how the strategy was typed ([ADR 0012](Decisions.md)). Pessimism is the right direction because an optimistic guess about an unobservable ordering flatters a backtest. A bar that reaches only one leg is unaffected. A reduce-only exit, or a sale on spot, of a position a resting order opened or flipped inside the bar meets the bar from that fill: under the default, a take-profit the bar may have reached only before the entry rests unless the close reaches it, and under either setting a stop fills where the price crossed it after the entry. With `intrabar="optimistic"` the better exit fills first instead, and the setting decides every other event a bar cannot order the same way, so the gap between a run under each shows how much a result leans on what the bars cannot show, though the two are not a strict bound ([Specification](Specification.md#the-error-and-its-direction)). `intrabar="random"` draws each event one way or the other from a stream named from `seed`, which it needs, so a seed draws the same way on every machine; it is for training, where an agent would otherwise learn the fixed tie-break and dodge it.

Three things about resting orders that are easy to get wrong. Nothing links them into an OCO group, so **re-placing a trailing stop with `stop()` each bar rests a new order every time**; the one that fills leaves its siblings live, and on a perp those open a position on the other side. Use `replace`, which cannot leave two alive and quietly does nothing once the order has filled ([ADR 0010](Decisions.md)). Pass `reduce_only=True` as well, so even a leaked stop can only shrink the position ([ADR 0010](Decisions.md)). And `close()` queues an ordinary market order, so it is `IOC`: on a thin bar the budget can fill only part of it and the remainder is cancelled rather than carried, which leaves a residual position. Check `state["position"]` rather than assuming.

```python
#  a trailing stop that stays one order and can only ever shrink the position
if state["position"] > 0.0:
    wanted = state["bar_close"] * 0.94
    if self.stop_id is None:
        self.stop_id = engine.stop("sell", state["position"], wanted, reduce_only=True)
    elif wanted > self.trigger:
        #  None here means the stop already filled, so nothing is armed again
        self.stop_id = engine.replace(self.stop_id, trigger=wanted)
    self.trigger = wanted
```

You decide on a bar and the order fills on the next one; there is no same-bar lookahead.

<br>

<div align="center">
  <img src="imgs/205316.png" alt="No same-bar lookahead" width="70%" />
  <p style="margin: 0;"><i>An order decided at the close of bar t is resolved against bar t+1, never the bar the decision saw.</i></p>
</div>

<br>

A market order crosses half the spread and fills at the next open, moved by its slippage and impact; a limit rests at your price and fills only when a candle reaches it, and a price that gaps through your limit still fills at the limit, never better, unless no trade since the last close reached it, as when it was placed through the market, where a buy fills at the highest price the market traded and a sell at the lowest ([Specification](Specification.md#limits)). A limit its first bar opens at or through is marketable, so it pays the taker fee even though it fills at your price; one that the bar has to come to, or that has rested through a bar, is on the book and pays the maker fee ([ADR 0011](Decisions.md)). A `post_only` limit that would cross the last close is refused, and one its first bar opens at or through is cancelled, rather than turned into a taker. A limit that reaches the account's own resting limit of the other side on its instrument, a buy at or above its own sell or a sell at or below its own buy, is refused, as a venue refuses a trade against the same account ([ADR 0011](Decisions.md)). Each bar holds `budget` of its instrument's trailing volume for every order on it, taken in resolution order, so splitting an order takes no more of a bar than one order would ([ADR 0013](Decisions.md)). Resting orders and the market orders waiting for the next bar each use `max_open_orders` slots, and beyond either cap a new order is rejected; the market queue empties every bar.

The position is netted: net long, net short, or flat, never both. Adding on the same side averages the entries; reducing or closing books the realized PnL on the part closed; flipping through zero closes the old side and reopens the rest at the new fill. On spot the position cannot go below zero: a sell is clamped to the current long, since shorting base needs a borrow this tier does not model, so a short and a flip through zero are perp behaviors ([ADR 0008](Decisions.md)).

### The state

Every `reset` and `step` returns the state, a typed record read by attribute or by key: `state.position` and `state["position"]` are the same number, two states are equal when every field is, and nothing can be written to one.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `tick_index` | int | Current bar index. |
| `base` | float | Base-asset balance (spot; zero on a perp). |
| `quote` | float | Cash across both wallets. |
| `spot_cash` | float | The spot wallet's cash. |
| `margin_cash` | float | The margin wallet's cash. |
| `position` | float | Signed size in base units, negative is short. |
| `avg_entry` | float | Volume-weighted entry price, zero when flat. |
| `equity` | float | Account value, marked in quote. |
| `mark_price` | float | Close on spot, mark on a perp. |
| `bar_open`, `bar_high`, `bar_low`, `bar_close` | float | The current candle. |
| `bar_volume` | float | The current candle's volume. |
| `realized_pnl` | float | Cumulative booked PnL. |
| `unrealized_pnl` | float | Open-position PnL at the mark. |
| `funding_paid` | float | Funding paid since the reset: positive is paid away, negative is received. Zero on spot. |
| `open_orders` | list | Resting limit and stop orders, each a dict. |

Each order in `open_orders` carries `id`, `side`, `kind` (`market`, `limit`, `stop`), `price`, `trigger`, `size`, `filled`, `remaining`, `status` (always `resting` here, since these are the resting orders), `reduce_only`, and `post_only`. A bracket's leg has `side` `None` until it arms, and `size` and `remaining` `None` while it closes the position it finds.

### Observations

`observation(lookback)` returns a read-only numpy `(rows, 5)` view of the `lookback` bars ending at the current bar, straight onto the shared candle buffer with no copy. It is `rows = lookback`, or fewer early in the series. The array is read-only, and because the candle buffer is never mutated a view held across a `step()` stays a snapshot of that tick's window rather than going stale; it stays valid for the whole life of the engine (numpy keeps the engine alive as the array's base), so you never need to copy it ([ADR 0040](Decisions.md)).

`eng.data` is the same kind of read-only, zero-copy view over the whole `(T, 5)` series rather than a window (columns are OHLCV by position). It is the full-series companion to `observation`, cursor-independent.

`eng.opens`, `eng.highs`, `eng.lows`, `eng.closes` and `eng.volumes` are the same view one field at a time, so a strategy reads `engine.closes` rather than remembering that the close is column 3. They are strided views into the same buffer, not copies, so taking all five costs nothing.

### Running a strategy

`run(strategy)` resets, calls `strategy.init(engine)` if it exists, then for each bar calls `strategy.next(state, engine)` and steps, returning the final state. The strategy places orders through the engine, so it can use any order type. It should not call `step` itself; the driver does.

If the strategy carries a `warmup`, read once after `init` so it can be computed there, `next` is not called until that many bars exist. The bars before it still advance and still fill whatever is resting; only the decision is skipped. It replaces the `if i < self.slow: return` a strategy would otherwise open with, and it closes what that guard is really for: on an early bar `close[i - 50]` is a **negative** index, numpy resolves it from the end of the array, and the rule reads prices from the future, reports a wonderful result, and raises nothing ([ADR 0025](Decisions.md)).

```python
class BuyAndHold:
    def next(self, state, engine):
        if state["position"] == 0:
            engine.market_buy(1.0)

eng = probatum._probatum.Engine(candles, report=True)
final = eng.run(BuyAndHold())
```

### Results

With `report=True`, seven accessors read the recorded run (each returns `None` when reporting is off):

- `stats(periods_per_year=365.0, risk_free=0.0)` returns the [stats dict](#statistics).
- `equity_curve()` returns a numpy array, one equity value per step.
- `trades()` returns a list of [round trips](#trades), the one still open at the end marked and flagged.
- `ledger()` returns [the ledger](#the-ledger), a dict of numpy columns with one row per bar from the opening balance.
- `fills()` returns a list of [fill dicts](#fills), one per fill in the order they landed.
- `order_log()` returns a list of [order events](#the-order-log), every order's life in the order it happened.
- `tables(start, names)` returns the run's ledger, holdings, fill log, order log and round trips as a dict of the binding's tables under those names, which a [Result](#the-tables) wraps, each read through the Arrow C stream or by `column(name)` and `row(i)`, every bar counted from `start` and each instrument by its name in `names`; it needs an engine built with `times=`.

An engine built with `record="account"` keeps less: `fills()` and `order_log()` return `None` with reporting on, `ledger()` has none of the columns for each instrument, and `tables()` gives the ledger and the round trips alone ([Specification](Specification.md#recording-levels)).

`periods_per_year` annualizes to your bar interval: 525600 for 1m, 8760 for 1h, 365 for 1d. The engine's `stats` is the raw call and takes it as a number, so state it; an engine built with `times=` gives its own clock's count from `periods_per_year()`. `Backtester` and `tune` read it from the candles' own timestamps instead, and only fall back when there are none ([ADR 0004](Decisions.md)).

### Statistics

`stats` is a dict with these keys:

| Key | Unit | Meaning |
| :--- | :--- | :--- |
| `total_return_pct`, `net_profit_pct`, `cagr_pct` | percent | Total return (net of fees; both names give the same figure) and annualized return. |
| `sharpe`, `sortino`, `calmar` | ratio | Risk-adjusted return; annualized. Each is `inf` when it earned something against no measured risk at all, so the set stays orderable ([ADR 0030](Decisions.md)). |
| `max_drawdown_pct`, `volatility_pct`, `exposure_pct` | percent | Largest peak-to-trough decline; annualized volatility; fraction of steps holding a position. |
| `win_rate` | fraction | Fraction of closed trades whose PnL net of the recorded fee is positive. |
| `profit_factor` | ratio | Net profit over net loss, both after fees (`inf` with no losses). |
| `num_trades` | count | Closed round trips. A position still open at the end is in the return and the exposure but in none of these four. |
| `avg_trade_pct` | percent | Mean net trade PnL as a percent of **starting** equity, not of the equity the trade opened with. |
| `num_fills` | count | Fills applied over the run. Zero beside orders you placed means the feed never filled them. |
| `funding_paid` | quote | Funding over the run: positive is paid away, negative is received. Zero on spot. A perp result cannot be decomposed without it, since a carry and a direction otherwise look the same ([ADR 0020](Decisions.md)). |

Two things about ranking, because `tune` takes an argmax over these. `calmar` scales a **negative** return by its drawdown rather than dividing by it, so the losing runs order by how much they lost; dividing made the deepest bust score highest, since the deeper the loss the larger the divisor ([ADR 0030](Decisions.md)). And `cagr_pct` saturates rather than escaping to infinity: annualizing raises the growth ratio to one-over-the-years, so on minute candles the exponent leaves the float range and every profitable run ties at the top. An annualized rate off a few hours is an extrapolation and not a measurement, so rank a sub-year run on `total_return_pct` or `sharpe`.

The four trade metrics count completed round trips only, so `exposure_pct > 0` beside `num_trades == 0` means "still holding", not "never traded"; buy and hold reports a real return with zero trades ([ADR 0022](Decisions.md)). They read each closed round trip's `pnl`, after its fees and its funding, because on gross PnL a strategy whose edge is smaller than its costs reports a perfect win rate and an infinite profit factor beside a negative return ([ADRs 0023, 0030](Decisions.md)). So they reproduce from the log:

```python
closed = [t for t in result.trades if not t["open"]]
win_rate = sum(t["pnl"] > 0 for t in closed) / len(closed)
```

`num_fills` is the canary for a dead feed: a series with no volume fills nothing, silently and correctly, and without it that run is indistinguishable from a strategy that never triggered. `periods_per_year` must be finite and positive.

The conventions (risk-free rate, sample deviation, annualization) are fixed in [ADR 0030](Decisions.md).

### Trades

Each entry in `trades()` is one round trip: a maximal run of one sign, from the fill that takes the position off zero to the fill that returns it to zero or flips it, so resizing inside it does not end it and `num_trades` counts decisions rather than fills ([ADR 0023](Decisions.md)). The closed ones come in the order they closed, then the one still open at the end, marked at the last close:

| Field | Meaning |
| :--- | :--- |
| `instrument` | The instrument, 0 with one. |
| `side` | `long` or `short`. |
| `size` | The largest position it reached. |
| `entry`, `exit` | The average prices of its opening fills and of its closing fills; `exit` is NaN until a fill has closed some of it. |
| `open_bar`, `close_bar` | The bars it opened and closed on; `close_bar` is `None` while it is open. |
| `open_time`, `close_time` | Those bars' open times when the engine was built with `times=`, else `None`. |
| `realized` | The price PnL its closing fills realized, before fees and funding. |
| `fees` | Every fee of its fills, a flip's split by size between the round trip it ends and the one it begins, so each is carried exactly once ([ADR 0014](Decisions.md)). |
| `funding` | The funding charged on it while it was open. |
| `pnl` | Its net PnL: `realized` less `fees` and `funding`, and for one still open what it holds at the last close as well. The trade statistics are built from it, so they reproduce from the log. |
| `ret` | `pnl` over `size * entry`. |
| `best`, `worst` | How far its side's price went for it and against it, as a fraction of `entry`, over the part of each bar it was in: its entry bar from the fill to the close, its exit bar from the open to the fill, and every bar between whole. They are bounds that never overstate, since nothing inside a bar says when its high and its low came ([ADR 0023](Decisions.md)). |
| `reason` | What closed it, `market`, `limit`, `stop`, `take_profit`, `rebalance`, `seam`, `liquidation`, `dust` or `delist`, and `None` while it is open. A forced close is priced where the margin ran out, a sale on the book less its costs, not at the extreme the bar printed, so the loss is bounded by the margin ([ADRs 0017, 0018](Decisions.md)). |
| `open` | True for the round trip still open at the end. |

When the `Backtester` is given a pandas DataFrame or parquet input with a datetime index, it sets `open_time` and `close_time` from the index values at those bars. A raw numpy input has no index, so they stay `None`.

### The ledger

Each column of `ledger()` holds one value per bar, the first for the opening balance, so no column is one shorter than the data, and a universe's columns from `position` to `pnl` hold a row of `N` per bar ([ADR 0022](Decisions.md)):

| Column | Meaning |
| :--- | :--- |
| `bar`, `time` | The bar, and its open time when the engine was built with `times=`; `time` is `None` without. |
| `equity`, `spot_cash`, `margin_cash` | The account at the close: both wallets together, and each wallet's cash. |
| `equity_in_quote` | Equity valued in the quote currency at the close: dollars for an inverse contract, and equity itself for any other. |
| `gross_exposure`, `net_exposure` | The position's notional at the close, unsigned and signed. |
| `leverage` | Gross exposure over equity, zero once equity is gone. |
| `margin_used` | What the margin cap asks of the position at the close. |
| `maintenance` | The maintenance requirement the tier table sets, zero without one. |
| `fees`, `funding`, `spread`, `slippage`, `impact` | What the bar cost, in quote: a fill's spread, slippage and impact are what it paid beyond its reference, each in proportion when the bar's range held its price back. |
| `liquidation` | What a takeover left in the margin wallet, taken by the venue or absorbed by it, less what the margin bound lifted fills by ([ADR 0018](Decisions.md)). |
| `dust` | The value of positions below half a lot, zeroed at the close where sizes are on, less the hair below zero a spot fill's rounding can leave in the spot wallet, which is set back to zero. |
| `position`, `entry`, `mark`, `notional`, `unrealized`, `weight` | The position, its average entry, the close it is marked at, its notional, its unrealized PnL, and its signed notional over equity. |
| `drift` | The weight the latest `target` asked of the instrument less `weight`, from the bar after that target was decided; NaN where no target has named it. |
| `pnl` | What the instrument has made since the first row: the cash its fills, its funding and the charges given to it moved in the wallet it settles in, plus its position's value at the close, nothing while it is flat; the instruments' sum to equity less the opening equity ([Specification](Specification.md#the-ledger)). |

Each cost is computed on its own, so on every bar the change in equity is the price PnL, measured from each fill's reference and then from the bar's close, less every cost.

### Fills

Each entry in `fills()` is one fill, forced closes included ([ADR 0022](Decisions.md)):

| Field | Meaning |
| :--- | :--- |
| `bar`, `time` | The bar it landed on, and that bar's open time when the engine was built with `times=`, else `None`. |
| `instrument` | The instrument, 0 with one. |
| `order` | The id of the order that filled, or `None` for a forced close or a delisting's close. |
| `side`, `size`, `price` | What filled, and at what price. |
| `reference` | The price the fill is measured from: the open for a market order, the worse of the open and the trigger for a stop, a limit's own price or the price the market reached it at since the last close, for a forced close its price where the margin ran out, and for a delisting's close its instrument's last close. |
| `spread`, `slippage`, `impact` | What it paid beyond the reference, in price, each its share of how far the price moved before the bar held it. |
| `fee` | The fee, in quote. |
| `adjustment` | How far the price then moved: held back inside the bar, or lifted by the margin bound. |
| `lifted` | The part of the adjustment the margin bound applied, per unit, so `adjustment - lifted` is the bar's hold; in dollars on an inverse account, and zero for a forced close. |
| `paid_in` | What the margin bound paid into the margin wallet for the part of its lift the bar could not hold, in the account's asset, never below zero. |
| `taker` | True when the fill took liquidity. A takeover is the venue's and is not, nor is a delisting's close; a sale on the book is. |
| `reason` | `market`, `limit`, `stop`, `take_profit` for a bracket's take-profit, `rebalance` for a leg of a `target`, `seam` for a close at a validation's seam, `liquidation`, or `delist` for the close of a position its instrument's delisting ended. |

### The order log

Each entry in `order_log()` is one event in an order's life ([ADR 0022](Decisions.md)):

| Field | Meaning |
| :--- | :--- |
| `bar`, `time` | The bar it happened on, the decision's for a placement and the fill's for a fill, and that bar's open time when the engine was built with `times=`, else `None`. |
| `kind` | `placed`, `refused`, `clamped`, `filled`, `not_filled`, `expired`, `cancelled`, or `armed` for a bracket's leg taking its side. |
| `order`, `instrument` | The order and its instrument, both `None` for a clamped transfer. |
| `side` | `buy` or `sell`, what a placed, refused or armed event's order asked, else `None`, as for a bracket's leg until it arms. |
| `size` | What a fill or a clamp came to, what a placed or armed order asks in all, or what a refused call gave, else `None`, as for a value order on spot or a linear contract and a bracket's leg that closes the position it finds. |
| `price` | What a fill was booked at, or a placed or armed limit's price or stop's trigger on the price rule, else `None`. |
| `reason` | Why, in words: `max_open_orders: 8 already resting`, `the spot wallet's cash allows 5 of the 20 asked`, `nothing could fill: the liquidity budget allows none`, `ioc: one bar`, `liquidation`, `delist`, `the instrument is delisted`. |

A refused placement still returns `None`, and the log holds it under the id it would have had, with its reason. A clamp, a market order's miss and a `fok` market order that could not fill whole each name the rule that held them ([Specification](Specification.md#the-fill-log-and-the-order-log)): `the liquidity budget`, `reduce_only`, `the spot wallet's holding`, `the spot wallet's cash`, `the margin wallet's equity`, `the largest position`, which is the last tier's bound, or `the margin cap`, as in `fok: the margin cap allows 3.3333333333333335 of the 5 asked`; a market order that nothing could fill reads `nothing could fill: the bar printed no volume` or `nothing could fill: the size is below what a fill can book` where no rule held it. An `ioc` limit a rule empties, or a `fok` limit one holds short of its size, logs nothing on its bar and expires as `ioc: one bar` or `fok: one bar`. A hold to the largest order reads `the maximum order quantity allows 1 of the 5 asked`, and a clamped transfer names the wallet it leaves, as in `transfer: the spot wallet's cash allows 600 of the 1000 asked` or `transfer: the margin wallet's free cash allows 600 of the 1000 asked`. The numbers are printed as the engine holds them, with no unit.

<br>

## The Result

`backtest`, a Strategy and `probatum.Engine.result()` return the same `Result`, whatever decided the run ([ADR 0026](Decisions.md)). It holds what the run recorded on the rows its account saw from its first: `time`, each row's open in int64 nanoseconds since 1970, and `equity` and `returns`, each `(T,)` and read-only, a row's return its equity over the row before's, less one, and nothing once the equity is gone; five [tables](#the-tables); `costs`, the totals of what trading cost, `assumptions`, a table of every modelling choice it made, graded, `baselines`, the run beside holding cash, buying and holding, and itself without costs, `result[name]`, one instrument's record, and `explain(item)`, a fill's price or an order's life in words ([Transparency](#transparency)); `stats`, the [statistics](#statistics) at `risk_free` and at the annualization of the whole Data the run's instruments are drawn from, counted on the rows they printed, so every slice of it annualizes alike, and None where the clock is too short; `policy`, an array's values on the Data's rows as the run read them, or the copy of a Strategy it ran; `bust`, whether the account's equity reached zero; `liquidations`, how many bars it was liquidated on, each account of a book counted apart; and `provenance`, what the run was given and where it ran. Its repr names the policy, the venue, the bars and the equity, and every assumption graded optimistic. `to_pandas()` and `to_polars()` give the equity and the returns on the time, `to_dict()` the whole Result as plain data, its costs and its assumptions included and its baselines once they are read, a statistic infinite where it earned something against no measured risk, and a Result copies whole and pickles with its tables, its assumptions, its baselines once they are read and an array's policy, a Strategy's copy left out. A run recorded at the account level keeps its ledger and its round trips, its costs, its assumptions and its statistics, and no policy; it refuses its holdings, fills, orders and baselines, an instrument's record, `position`, `weight`, `pnl` and `explain`, each by name, its repr names the level, `to_dict()` gives None for what it does not keep, and `combine` refuses it.

```python
print(result.to_pandas().tail())                # equity and returns on a UTC index
print(result.stats["total_return_pct"], result.bust, result.liquidations)
```

### The tables

`ledger`, `holdings`, `fills`, `orders` and `trades` are tables the engine builds, and `assumptions` and `baselines` ones the Result builds ([Transparency](#transparency)), each handed to any reader through the Arrow C stream interface, so `polars.DataFrame(table)`, `pyarrow.table(table)` and a DuckDB query that names the table read them with nothing imported by probatum, and `table.to_pandas()` builds a pandas DataFrame with no pyarrow at all. Every bar counts from the Result's first row, so `result.time[bar]` is its time, every instrument is its name, and a value the run did not record is a null, never a number that stands for none ([Specification](Specification.md#tables)).

| Table | A row per | Columns |
| :--- | :--- | :--- |
| `ledger` | bar, from the opening balance | `bar`, `time` and the account's columns of [the ledger](#the-ledger) |
| `holdings` | bar and instrument, a bar's in the Data's order | `bar`, `time`, `instrument`, `position`, `entry`, `mark`, `notional`, `unrealized`, `weight`, `drift`, `pnl` |
| `fills` | fill, in the order they landed | the [fill log](#fills)'s fields |
| `orders` | event in an order's life | the [order log](#the-order-log)'s fields |
| `trades` | round trip, the closed in the order they closed, then those still open | the [round trips](#trades)' fields |

A table reads like the lists of records it replaced: `table[i]` is row `i` as a dict, None where a value is missing, iterating gives the rows and `len(table)` counts them. `table["pnl"]`, or `table.pnl`, is a column as a read-only numpy array: floats with NaN where a value is missing; integers and times as int64, masked where a value can be missing, so `trades.close_bar` masks the round trips still open; booleans; and text as objects. A slice, a boolean mask or an array of positions gives a table of those rows, a masked position refused, since it names no row, `orders.events(kind)` the order log's events of one kind, and `table.columns` names the columns. Two tables are equal where their columns and their values are.

```python
import duckdb
import polars

trades = result.trades
closed = trades[~trades.open]
print(result.time[closed.close_bar])            # when each closed round trip closed
print(result.orders.events("filled"))           # the order log's fills alone
fills = result.fills
print(polars.DataFrame(fills).select("bar", "price", "fee"))
print(duckdb.sql("select instrument, count(*), sum(fee) from fills group by 1"))
```

### Saving and loading

`result.save(path)` writes a directory of files any tool reads: `result.json`, the manifest, holding the format it is written in, the version of probatum that wrote it, the provenance, whether the policy was an array and whether the account went bust; `time.npy`, `equity.npy` and `returns.npy`, and `policy.npy` for an array run; each table as an Arrow IPC file, the assumptions as `assumptions.arrow` and the baselines as `baselines.arrow`, read first where they were not, a run below the full level writing the tables it keeps alone; and `stats.arrow`, the statistics as one row. A Result pickled before its baselines were read cannot give them, so save refuses it, or a book of one, before writing anything. `probatum.Result.load(path)` reads the format, never the class that wrote it, so a Result one version saves opens in the next, and it refuses a format it does not read and files that save did not write or that disagree with one another. Nothing is pickled, so a loaded Strategy's `policy` is None, and its provenance still describes it. Every file is written under another name before the first is put in place, the manifest last, so a save that an error or a killed process stops leaves the Result that was there or a directory that load refuses.

```python
import tempfile

folder = tempfile.mkdtemp()
result.save(folder)
again = probatum.Result.load(folder)
print(again.trades == result.trades, again.stats == result.stats)
```

### The provenance

`provenance` records what the run was given and where it ran: `data`, the Data's provenance with its fingerprint; `venue`, the code that builds the venue, and `venue_name`; `markets`, each instrument's market as it ran, `spot`, `linear` or `inverse`; `settlement`, the asset the account kept its cash in, None where its instruments do not say; `contract`, the dollars an inverse contract is worth, None on any other market; `policy`, a description of it, beside an array's `sizing`, `tolerance`, `stop` and `take_profit`; `record`, the level `backtest` recorded the run at; `seed`, and `streams`, the random streams a venue that decides at random draws from; `risk_free` and `periods_per_year`; and `version`, `packages`, the version of every installed distribution behind a module the run had loaded, read from their metadata so nothing is imported to ask, `python`, `platform` and `wheel`, the tags of the wheel probatum was installed from.

### Combining books

`probatum.combine(results, settlement=None)` adds Results from accounts at different venues into one book, a list of them each named by its venue's name, or a mapping of a name to each ([Specification](Specification.md#combining-books)). The book's clock is every time any of them has a row, each account counts its opening balance before its first row and its last equity after its end, and the book's equity is their sum in the order given, its returns and statistics those of the sum. Its ledger and its baselines are its own, a row per row of its clock, the baselines its accounts' summed as its equity is, and its other tables are theirs joined, each row with its `venue` first and every bar counted on the book's clock. Every account is margined on its own and no capital moves between them. Accounts kept in different dollar assets, such as USDT and USDC, add only where `settlement` names the dollar asset the book is kept in, counted at par; an account kept in any other asset adds only to accounts kept in it, and an inverse contract's only to others of its coin; and accounts on different intervals, off one interval's steps or at different risk-free rates not at all. A column a later version added to an account's table is left out of the book's.

```python
binance = probatum.backtest(Hundred(), data,
                            probatum.venues.binance_perp.replace(funding="none"))
book = probatum.combine([result, binance])      # named frictionless and binance_perp
print(book)
print(book.fills.to_pandas()[["venue", "bar", "instrument", "price"]])
```

### Transparency

`costs` is a dict of the run's costs, the totals of the ledger's cost columns in its order, `fees`, `funding`, `spread`, `slippage`, `impact`, `liquidation` and `dust`, each a compensated sum, positive where the account paid it, in the asset it settles in, and measured from each fill's reference, so the move from a decision's close to the next open is price PnL ([Specification](Specification.md#costs)). Each read is a dict of its own.

`result[name]` is one instrument's record: its `name`; its `position`, `weight` and `pnl`, each `(T,)` and read-only, the weight over its account's own equity and the pnl what it has made since the first row; and its own `fills` and `trades`. With one instrument the Result answers `position`, `weight` and `pnl` itself, and with several it names the one to ask. In a book an instrument is `book[venue, name]`, or `book[name]` where one venue alone holds it, its series carried from each of its account's rows to the next inside the account's span, its position and weight nothing outside it, and its pnl nothing before its first row and held after its last.

`assumptions` is a table of every modelling choice the run made, a row each ([Specification](Specification.md#assumptions), [ADR 0027](Decisions.md)): its `name`; its `value`, as it reads after the name, "limit fills on touch"; its `source`, the preset and the day its tables were taken, or its starting calibration for a preset's slippage, impact and liquidity apart from the plain venue's, `plain`, `set by the user`, the router, the Data or the engine, each other input that decided part of it named after the first; its `grade`, `venue` where the run does what a venue does as far as bars show it, `simplified` where it approximates that, with the `direction` it leans, `pessimistic`, `optimistic` or `either`, and `optimistic` where it leaves out what every venue charges or limits, or decides what a bar cannot show in the trader's favour; and its `cost`, where the run's records measure it, in the asset the account settles in. Its rows are the venue's settings, each read for the instruments the run held and naming those that depart from the rest, the engine's own rules, each in its off form where it cannot act on the run, and the Data's, and at the full level those with a cost add up to the costs. Its repr shows every row, `table[table.grade == "optimistic"]` keeps the optimistic ones, and a book's are its accounts', in the order they were given, each naming its venue. `venue.describe()` gives the venue's and the engine's rows from the venue alone, with no cost, for every instrument it lists and one of each kind it takes but lists none of, and `data.describe()` gives the Data's.

`explain(item)` tells one fill's price or one order's life as text. A fill, a row of `fills`, reads as its reference, what its price was measured from; the half spread and the slippage it paid, each with its rate, and the impact with its model, none for a maker, in a takeover or at a delisting; how far the bar's range held its price, or that a delisting's close came before the bar opened; what the margin bound lifted and paid in, where it acted; and its fee, as a rate on its notional. Its price lines add up to its price, each to the decimals its reference sets, two from 1 up and enough for five significant figures below, its parts and its hold signed, a sale's costs negative. Money prints to the cent, or to five significant figures in a coin, rates in basis points to one decimal and to three significant figures below one, fees as percentages to three significant figures, and sizes to twelve significant figures. An order, a row of `orders` or an `Order` the run's account handed out, reads as a line for each of its events, its time, its kind and what the order log says, under a head from the event that placed it, or that refused it where none did. A Result tells the handles of its own run alone, so one loaded, unpickled or combined asks for a row of its order log, and a book's text names the venue, as its rows do.

`baselines` sets the run beside three baselines on its own data and venue ([Specification](Specification.md#baselines)), a table of a row per row of the Result: `bar`, `time`, `cash`, the opening equity on every row; `buy_and_hold`, an equal share of the opening balance in each of the Data's instruments, bought as the quantity it buys at the instrument's first close and asked for at every close from then on, on the run's venue and paying its costs; and `frictionless`, the same policy on the run's venue with the costs of trading removed and every rule kept, which still pays what a forced close takes. An array plays again with its rows, sizing, tolerance and barriers, and a Strategy on a second copy `backtest` takes at the call, played once; a loop the caller stepped has no policy to play, and its `frictionless` is missing, as it is with a warning where the play raises. They are computed when first read, and `save` reads them first, while pickling plays no policy and carries them only once they are read. A book's are its accounts' summed as its equity is, each at its opening balance before its first row and at its last value after its end.

```python
print(binance.costs)                            # what trading on binance_perp cost
print(binance["BTCUSDT"].pnl[-1], binance.pnl[-1])
print(book["binance_perp", "BTCUSDT"].weight[-5:])
print(binance.assumptions)                      # every choice the run made, graded
print(probatum.venues.binance_perp.describe())
print(data.describe())
print(binance.explain(binance.fills[0]))        # its reference, each cost, its fee
print(binance.explain(binance.orders[0]))       # that order's events
print(book.baselines.to_pandas().tail())        # cash, buying and holding, no costs
```

<br>

## Search

`probatum.search(policy, space, data, venue, objective="sharpe", direction="maximize", trials=100, sampler="tpe", min_trades=0, holdout=None, keep="all", n_jobs=1, seed=None, storage=None, risk_free=0.0)` searches a policy's parameters on the bars before a holdout and returns a `probatum.Study` of every trial ([ADR 0032](Decisions.md), [Specification](Specification.md#search)).

- **The policy** is a Strategy subclass, each trial built as `Cross(**params)`, or a function `policy(data, **params)` of the bars a trial runs on that returns weights, an array or a table of them, a dict of `backtest`'s arguments, `{"policy": weights, "stop": 0.02}`, so a barrier, the `tolerance` or the `sizing` is searched as any other parameter, or a Strategy, copied before it runs. Every trial returns one kind, weights or a Strategy, and the space's names are checked against the policy's signature before any trial runs.
- **The holdout** is the tail: `holdout=0.3` holds out that share of the rows, `holdout=300` that many and a time the rows from it on. `holdout=0` searches every bar, and so does leaving it out, with a warning, since every score is then in-sample. The trials run on a Data of the rows before it alone, which holds no later bar.
- **A trial** is one run over those bars, with no Result, scored by `objective`, a statistic's name, which `direction` maximizes or minimizes. It is `complete`; `floored`, where it closed fewer round trips than `min_trades`, which keeps its statistics and its returns and has no score; or `failed`, where the policy or its run raised, its error kept as text. The winner is the complete trial with the best score, the lowest-numbered of equal ones, and a search with none raises.
- **The samplers.** `"random"` and `"grid"` are probatum's own, drawn from streams named from `seed`, and `"tpe"` is Optuna's TPE, which needs optuna 5.0 or later and is told a floored or failed trial as failed. `seed=None` draws a seed from the operating system and records it. A seeded random or grid search repeats on every machine, and a seeded TPE search on the machine it ran on.
- **The causality check.** Before it trusts a trial, the search runs the first trial that decided something twice on the in-sample bars and once on the bars up to each of its cuts, at most ten, placed where each instrument's decisions change and on the row before, where they held, and refuses a policy whose decisions do not repeat, as a `ProbatumValueError`, or whose decisions on the bars up to a cut change when the bars after it are cut away, as a `probatum.LookaheadError` naming the cut and the first decision that differs; it checks the winner too where its parameters differ. A Strategy's run on a cut decides at the cut's last close as well, where a value read from a later bar shows. A refusal of what a cut decides, by `backtest`'s reader or by a verb, is a `LookaheadError` chained from it, and an error the policy's own code raises there is its own, with a note naming the cut, and the last close where a Strategy raised it deciding there. Each run builds the policy afresh, so a stream it seeds itself repeats, and the check reads the in-sample bars alone, so it is no look ([Specification](Specification.md#the-causality-check)).
- **Weights run together.** Once a trial is checked, a random or grid search whose policy returns weights runs the rest in chunks of one call into Rust, each trial on an engine of its own over the shared bars that keeps only what its statistics read, in parallel with the GIL released, so a sweep of thousands of parameter sets runs no Python between bars. TPE, which chooses each trial from the last, and a Strategy run one trial at a time.
- **Workers.** With `n_jobs` above 1, or -1 for one a core, the trials that do not run together, a Strategy's and TPE's, run in that many workers `spawn` starts once a trial is checked, TPE asking a round of as many at a time and told them in trial order, so a seeded random or grid search repeats at any number of workers and a seeded TPE search at the same number. The search writes what they run from into a temporary folder and passes them its path, and each rebuilds the engine. A worker's warnings come back once a place, from the module they came from, under their own categories, or as a `ProbatumWarning` naming one this process cannot import or build from its text, and one your filters make an error fails its trial. A script that calls `search` outside `if __name__ == "__main__":` has every worker run it again as it starts, which is refused naming the guard; a pool that breaks before any trial comes back raises, and one that breaks later fails the trials in flight and warns, raising where no trial is complete. Workers need cloudpickle, which the `search` extra installs with optuna.
- `keep="all"` keeps every trial's returns, and `keep="none"` none. `storage` is not built yet.

```python
def crossing(data, fast, slow, stop):
    up = probatum.ta.ema(data.close, fast)
    down = probatum.ta.ema(data.close, slow)
    weights = numpy.full(len(data), numpy.nan)
    weights[probatum.ta.crossover(up, down)] = 1.0
    weights[probatum.ta.crossunder(up, down)] = 0.0
    return {"policy": weights, "stop": stop}


space = {"fast": probatum.Int(5, 40), "slow": probatum.Int(50, 200, step=10),
         "stop": probatum.Float(0.01, 0.05, step=0.01)}
study = probatum.search(crossing, space, data, "frictionless", sampler="random",
                        trials=40, holdout=0.3, seed=7)
print(study)                                    # the best score, what is held out
print(study.trials.to_pandas()[["number", "state", "score", "fast", "slow"]].head())
held = study.evaluate_holdout()                 # the one look at the tail, counted
```

### The Study

`trials` is a table of a row a trial: `number`, `state`, `score`, each parameter, an int, a float, a bool or text by the values its range holds, the fifteen [statistics](#statistics) and `error`, a value a trial lacks missing. `returns` is the `(R, T)` float64 matrix of every trial's returns on the in-sample bars, read-only, counted as a Result's returns are, a failed trial's row NaN, or None under `keep="none"`. `best_params` and `best_score` are the winner's, and `best` is its Result on the in-sample bars, run again at the full level when first read, whose statistics are its trial's to the bit; a winner that runs otherwise now, the policy's code or the globals it reads having changed since the search, is refused as a `ProbatumRuntimeError`. `evaluate_holdout()` runs the winner once on the held-out bars, its warm-up inside them, and keeps the Result as `holdout`, which a second call returns. It reads `best` first, and counts a look in `looks` once the winner is handed the held-out bars, so a run that raises there has spent one. `provenance` records what the search was given: the sampler with its arguments and Optuna's version, the seed, the trials asked and run, the parameters its causality check passed, the holdout and the in-sample bars' annualisation and fingerprint, beside the venue, the Data's provenance and the packages. A search logs its start and its end at INFO and each trial at DEBUG to the `probatum` logger. Its `kind` is `"trials"`, and a validation's Study, of kind `"paths"`, is [below](#the-study-of-paths).

### The space

- `Int(low, high, step=1, log=False)`: a whole number from `low` to `high`, both included, on the lattice `low + k * step`, the step a whole number from 1 that divides the span; `log=True` draws it evenly in its logarithm, with a `low` from 1 and a step of 1.
- `Float(low, high, step=None, log=False)`: a number from `low` to `high`, with no step the interval `[low, high)`, and with one the lattice from `low` whose last point is `high` exactly, the step dividing the span as their decimals do, so `Float(0.01, 0.1, step=0.01)` is ten points; `log=True` draws it evenly in its logarithm, with a `low` above 0 and no step.
- `Categorical(choices)`: one of a list of None, bools, numbers and strings, no two of them equal, so `1`, `1.0` and `True` are one choice.

A trial receives each value as a plain Python int, float, bool, str or None. A range compares and hashes by what it holds and pickles as the call that builds it.

```python
space = {"fast": probatum.Int(5, 40), "slow": probatum.Int(40, 200, step=10),
         "stop": probatum.Float(0.01, 0.05, step=0.01),
         "kind": probatum.Categorical(["ema", "sma"])}
print(space["slow"], space["stop"])      # Int(40, 200, step=10) Float(0.01, 0.05, step=0.01)
```

<br>

## Splitters

`probatum.cv` holds the ways of dividing time, each a splitter scikit-learn's own tools take, built without importing scikit-learn: `split(X, y=None, groups=None)` gives each split's training and test rows, and `get_n_splits` counts them ([ADR 0035](Decisions.md), [Specification](Specification.md#the-splitters)).

| Splitter | What it gives |
| :--- | :--- |
| `probatum.cv.Holdout(tail, end=None)` | one split: the last `tail` bars tested, a share of them or a number, and the bars before trained on |
| `probatum.cv.WalkForward(train, test, anchored=False, end=None)` | windows from the first bar, each testing the next `test` bars and trained on the `train` bars before it, or with `anchored` on every bar before it, the last window taking the remainder |
| `probatum.cv.PurgedKFold(k, embargo=None, end=None)` | `k` groups of bars in order, as scikit-learn's KFold sizes folds, each tested in turn |
| `probatum.cv.CombinatorialPurged(groups, test_groups, embargo=None, end=None)` | every combination of `test_groups` of `groups` such groups tested, recombined into `C(groups - 1, test_groups - 1)` out-of-sample paths |

- **Bars and rows.** A splitter divides bars. Its rows are the long, time-major rows an estimator reads, row `r` bar `r // N` of instrument `r % N`, so X holds every bar and instrument of the grid in that order, and an X of another length is refused. `end` is each row's label end, an int64 array `(T,)` or `(T, N)`: the bar whose price closed its trade, from its own bar plus one on, at `T` or later for a label that runs past the bars, and the least int64 for a row with no label, which is in no set. Without `end` each row is a bar holding a label of no length.
- **The purge.** A training row is left out where its label reaches a test span from before it, or where it starts after the span up to the latest end of the span's labels, and the embargo leaves out the bars after that; a label that ends past the last bar, at `T` or later, never trains. `embargo=None` is the longest label held. Holdout and WalkForward take no embargo, since no training row follows their test spans.
- **The contract.** Every split is checked before the first is given, and one the purge leaves with no row to train or test on is refused rather than skipped, since scikit-learn's searches count on `get_n_splits`. `y` is not read, and `groups` is ignored with a warning. A splitter has no `get_params`, `shuffle` or `random_state`, never changes, compares and hashes by what it holds and pickles as its call. scikit-learn's `cross_val_predict` takes PurgedKFold and CombinatorialPurged with one test group built without `end`, whose test sets cover each row once; built with it, the last bar's rows, whose labels end past it, are in no test set, so `cross_val_predict` refuses them.

```python
end = numpy.arange(1_000) + 25                   # each label ends 24 bars after its entry
splitter = probatum.cv.PurgedKFold(5, embargo=24, end=end)
train, test = list(splitter.split(numpy.zeros(1_000)))[2]
print(test.min(), test.max(), train.max())       # 400 599 974: purged, embargoed, censored
print(probatum.cv.CombinatorialPurged(8, 2).get_n_splits())   # 28, recombined into 7 paths
```

<br>

## Validation

`probatum.validate(policy, data, venue, cv, space=None, search=None, features=None, labels=None, to_weights=None, holdout=None, fill=None, keep="all", seam="cancel", seed=None, risk_free=0.0)` runs a search on the training bars of each split a [splitter](#splitters) gives and trades each split's winner on its test span, every out-of-sample path one run of those winners in turn, and returns a `probatum.Study` of the paths ([ADRs 0034 and 0035](Decisions.md), [Specification](Specification.md#validation)).

- **The policy** is a Strategy subclass and `space` its parameters' ranges, as `search` takes them, given always, since a fixed parameter set fits nothing. A model with `fit`, a function of the data, and `features`, `labels`, `to_weights` and `fill`, which a model takes, are not run yet. `search` is a dict of `probatum.search`'s `objective`, `direction`, `trials`, `sampler`, `min_trades` and `n_jobs`, which every search of the validation runs by, and `holdout`, `keep`, `seed` and `risk_free` are validate's own. `cv` is built without `end`, since a Strategy has no labels, and with `PurgedKFold` and `CombinatorialPurged`, whose fits join pieces end to end, `max_drawdown_pct` and `calmar`, which the pieces' order changes, are refused as the objective.
- **The first search.** The holdout is the tail, held out as `search` holds it. Before any split a plain `search` runs over the bars the splitter gives the holdout as its test span, every bar before it, or with a rolling `WalkForward` the last `train` of them. Its winner is the Study's, which `evaluate_holdout()` runs once on the tail, and its causality check refuses a policy that reads ahead before any split runs.
- **Fits.** Each split's search runs on its training bars, each run of them a piece of its own: one from the first bar runs as a search's trial does, and one after a test span on a Data that starts after the bar the span's last decision is scored on, its account opening past the embargo, its warm-up and the venue's fill window of each instrument that printed before that Data, or the instrument's last bar there where it prints fewer, so no fit reads a test bar. A trial on several pieces is scored on their returns joined end to end. A fit with no complete trial fails, each winner the fits chose that the first search did not check is checked for causality on the bars before the holdout, and so is each path that ran, at its seams, where `on_inherit` decides.
- **Paths and seams.** A path is one run over the bars before the holdout, each of its windows deciding at its test span's closes with its fit's winner and the account carried across. At a seam every order not yet filled is cancelled with the reason `seam`, the targets are forgotten and the incoming window's `on_inherit(bar, account)` is called before its first decision, its account's `events`, `trades` and `equity_curve` starting from that close; `seam="flatten"` closes every position at the outgoing window's last close instead of deciding. A window owns the bars from the one after its first decision to the one after its last, and a path whose run raises, or that a failed fit trades, fails alone.
- **Seeds and workers.** `seed` seeds the first search from the stream named `validate final` and split `i`'s from `validate fit i`, so a seeded validation repeats, and is drawn from the operating system and recorded where it is None. `n_jobs` above 1 runs every search's trials in workers, one pool serving every fit.

```python
class Channel(probatum.Strategy):
    def __init__(self, lookback=20, stop=0.02):
        self.lookback, self.stop = lookback, stop

    def init(self, data):
        self.warmup = self.lookback + 1
        self.high = probatum.ta.highest(data.high, self.lookback)

    def next(self, bar, account):
        if account.position == 0.0 and bar.close > self.high[bar.i - 1]:
            account.buy(1.0)
            account.bracket(stop=bar.close * (1.0 - self.stop))

    def on_inherit(self, bar, account):
        if account.position > 0.0:              # the seam cancelled the stop it held
            account.bracket(stop=bar.close * (1.0 - self.stop))


space = {"lookback": probatum.Int(10, 40), "stop": probatum.Float(0.01, 0.05, step=0.01)}
study = probatum.validate(Channel, data, "frictionless",
                          cv=probatum.cv.CombinatorialPurged(5, 2), space=space,
                          search={"trials": 10, "sampler": "random"}, holdout=0.3, seed=7)
print(study)                                    # 4 paths from 10 fits, the median score
print(study.windows.to_pandas()[["path", "fit", "first", "last", "score"]].head())
held = study.evaluate_holdout()                 # the first search's winner, once, on the tail
```

### The Study of paths

A validation's Study is of `kind` `"paths"`. `paths` is a table of a row a path: `number`, `state`, `score`, the fifteen [statistics](#statistics) and the seven [costs](#the-ledger) of the account rows it owns, and `error`. `fits` is a row a split: `number`; `test`, the bars it tests; `train`, `purged` and `embargoed`, its training bars and those the purge and the embargo took; `pieces`; each parameter of its winner, missing where the fit failed; `fitted`, the winner's score; `trials` and `complete`; `seed`; `state` and `error`. `windows` is a row a stretch of a path one fit traded: `path`, `fit`, `first` and `last`, the account rows it owns, and their times `first_time` and `last_time`, `decided`, the closes it decided at, `score`, the statistics and the costs, so `returns[path, first:last + 1]` is what it was scored on. `returns` is `(paths, T)` over the in-sample account rows, read-only, NaN where a path owns no row and on a failed path, and None under `keep="none"`. `best`, `best_params`, `best_score`, `evaluate_holdout()`, `holdout` and `looks` are the first search's. `trials` is refused, as `paths`, `fits` and `windows` are on a search's Study. `provenance` records the splitter and its layout, the seam, the seeds of the first search and of every fit and the parameters checked for causality, beside what a search's records. A validation logs its start and its end, each fit and each path at INFO to the `probatum` logger.

<br>

## Inference

`probatum.stats` says how far a run's statistics can be trusted ([ADR 0031](Decisions.md), [Specification](Specification.md#inference)). Each function takes a Result, whose returns after its first row it reads at the run's own annualisation and risk-free rate, or a plain series of returns, a 1-D array, a list or a pandas Series read by position, with `periods_per_year`, which has no default, and `risk_free`, an annual rate, 0 unless given.

| Call | Gives |
| :--- | :--- |
| `probatum.stats.sharpe(returns, periods_per_year=None, risk_free=None, long_run=False)` | the Sharpe, a Result's its record's; with `long_run`, over the long-run deviation of the returns, which their autocorrelation moves |
| `probatum.stats.sharpe_interval(returns, periods_per_year=None, risk_free=None, level=0.95)` | an `Interval`, the Sharpe's two-sided interval by the delta method |
| `probatum.stats.probabilistic_sharpe(returns, benchmark=0.0, periods_per_year=None, risk_free=None)` | a `Probability`, the one-sided probability that the Sharpe is above `benchmark`, an annual Sharpe |
| `probatum.stats.min_track_record_length(returns, benchmark=0.0, confidence=0.95, periods_per_year=None, risk_free=None)` | a `TrackRecord`, the returns and the years at which a Sharpe this high would give that confidence |
| `probatum.stats.bootstrap(returns, statistic, periods_per_year=None, risk_free=None, level=0.95, resamples=2000, seed=None)` | an `Interval` of `statistic` by a bootstrap: of the returns for `total_return_pct`, `net_profit_pct`, `cagr_pct`, `volatility_pct`, `sortino`, `value_at_risk` and `conditional_value_at_risk`, with `max_drawdown_pct` and `calmar` as the middle of the resampled histories, and of a Result's closed round trips for `win_rate`, `profit_factor` and `avg_trade_pct` |
| `probatum.stats.autocorrelation(returns, lag=1)`, `probatum.stats.effective_sample(returns)` | emsl's autocorrelation and its count of independent returns, which no interval reads |

- **One variance.** The interval, the probability and the track record read one variance: the long-run variance of the Sharpe's influence series, with Newey and West's weights at the lag their 1994 rule picks on that series, floored at its value at lag zero, so autocorrelation widens an interval and never narrows it. Each names that lag and whether the floor bound.
- **Two bootstraps.** The returns are resampled in blocks whose mean length is the longest Politis and White's rule, as Patton's code computes it, finds on the returns, their squared deviations or their tail, so the stretches a strategy's exposure switches on and off set it; each interval records it as `block`. Total return, net profit, CAGR, volatility, Sortino and conditional value at risk are studentized, each resample's move over its own standard error, and value at risk is percentile. The round trips are drawn evenly, and the win rate's interval is percentile, the profit factor's bias-corrected and accelerated and the average trade's studentized. A drawdown's and a Calmar's `kind` is `"range"`, the middle of the resampled histories, which says what histories like the run's do and is no confidence interval. Each resampling draws from a stream the values it resamples seed, so an interval is the same on every read and in any process with no seed, and `seed=` draws others.
- **What comes back.** An `Interval` holds `statistic`, `estimate`, `low`, `high`, `level`, `method`, `count`, `lag`, `floored`, `block` and `kind`, a drawdown and a value at risk as positive losses that its text prints as negative returns; a `Probability` holds `value`, `benchmark`, `lag`, `floored` and `count`; a `TrackRecord` holds `observations` and `years`, None where the Sharpe is not above the benchmark, `series_years`, `sharpe`, `confidence`, `benchmark`, `lag`, `floored` and `count`. Each never changes, compares by what it holds, and prints as a sentence that says it.
- **What is refused.** A plain series without `periods_per_year` and a Result given one or a `risk_free`, which it names itself (ProbatumTypeError); a NaN or an infinity, naming its position, and an interval the sample cannot support: fewer than 4 returns or returns that never vary, returns whose Sharpe's influence is zero on every bar, fewer than 500 returns or 100 of the bootstrap's blocks for a bootstrap of them, a return of -100% or worse for growth, a tail whose `k` lowest returns are all equal for conditional value at risk, a studentized statistic with no standard error, an interval whose resamples put both its bounds on one value, fewer than 60 closed round trips or trips that cannot vary in what the statistic counts, fewer than 2 resamples or too few for the level, resamples all on one side of the profit factor's estimate, and an estimate ranked at infinity (ProbatumValueError); a statistic that is not a string (ProbatumTypeError).

```python
monthly = [0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01]
print(probatum.stats.sharpe_interval(monthly, periods_per_year=12))
print(probatum.stats.probabilistic_sharpe(monthly, periods_per_year=12))
print(probatum.stats.min_track_record_length(monthly, periods_per_year=12))
print(round(probatum.stats.sharpe(monthly, periods_per_year=12, long_run=True), 2))
print(probatum.stats.sharpe_interval(result).count)   # the Quickstart's 999 returns
print(probatum.stats.bootstrap(result, "cagr_pct"))
print(probatum.stats.bootstrap(result, "max_drawdown_pct"))
```

```text
sharpe 1.73, two-sided 95% interval -0.92 to 4.39, delta method, Newey-West lag 2 floored at lag zero, 8 returns
probabilistic sharpe 0.899 against zero, one-sided
minimum track record 1.0 years for 95% one-sided confidence against zero; this series has 0.67
2.95
999
cagr -0.42%, two-sided 95% interval -0.95% to 0.09%, studentized stationary block bootstrap, block 2.6, 999 returns
max drawdown -0.06%; middle 95% of resampled histories -0.12% to -0.02%, stationary block bootstrap, block 2.6, 999 returns
```

The two-sided interval holds zero while the one-sided probability is 0.899; both are right, and each says which it is. The returns reverse from month to month, so the long-run Sharpe, 2.95, is above the Sharpe the square root of time scales, and the interval's floor holds it to the width independent returns would give. Eight returns are far too few to resample, so the bootstrap reads the Quickstart's run, its 999 returns in blocks of 2.6.

<br>

## Tuning

`probatum.tune` searches a strategy's parameters for the configuration that maximizes (or minimizes) an objective. Each trial is a full backtest through the same `Backtester` above, so the `Strategy` you backtest is the `Strategy` you tune: declare the tunable parameters as constructor arguments and store them as fields, and `tune` builds a fresh strategy per trial from the sampled values. The search runs across worker processes, rebuilding the engine in each worker rather than shipping a live one across the boundary, so it scales over cores ([ADR 0032](Decisions.md)). It needs optuna and cloudpickle, which the `[tune]` extra pulls in; probatum is not on PyPI, so install it from a release as the [README](../README.md#install) shows.

```python
from probatum import tune
from probatum import Strategy


class SmaCross(Strategy):
    def __init__(self, fast, slow):        # the tunables, as constructor arguments
        self.fast = fast
        self.slow = slow

    def init(self, engine):
        self.close = engine.closes
        self.warmup = self.slow

    def next(self, state, engine):
        i = state["tick_index"]
        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()
        if state["position"] == 0 and fast > slow:
            engine.market_buy(1.0)
        elif state["position"] > 0 and fast < slow:
            engine.close()


result = tune(
    SmaCross,                              # the strategy, or any callable that builds one
    {"fast": (5, 40), "slow": (40, 200)},  # the search space
    candles,                               # numpy, DataFrame, or parquet path
    objective="sharpe",
    n_trials=200,
    oos=0.3,                               # search the first 70%, keep the rest back
    n_jobs=-1,                             # every core; 1 runs in this process
)
print(result.best_params)
print(result.best_value)                  # in-sample: the best of 200 tries
print(result.oos_stats["sharpe"])         # the same strategy on bars no trial saw
best = result.best_strategy()             # a fresh SmaCross with the best parameters
```

`tune(strategy, space, data, ...)` takes the strategy, the search space, the data, and then the same engine knobs as the `Backtester` (`market`, `quote`, the costs, `leverage`, `periods_per_year`, `risk_free`), plus the search controls below. It returns a `TuneResult`.

### In-sample and out-of-sample

`TuneResult.best_stats` is in-sample. It is the maximum of a noisy score over every trial, on the bars the search was allowed to see, so it is biased upward by the act of searching and by more the harder you searched. Quoting it as a result is the most common way a backtest misleads, and no statistic computed from the same bars can undo it.

`oos=0.3` fits every trial on the first 70% of the series and re-runs the winner on the last 30%, which no trial ever saw, into `TuneResult.oos_stats` and `TuneResult.oos_result`. The split is always the **end** of the series, never a random slice, because a strategy is a claim about what comes next and shuffling would let a trial fit around the very bars meant to test it ([ADR 0032](Decisions.md)).

Two costs worth knowing. The search sees less data, which matters on a short series. And the winner warms up **inside** the held-out bars rather than being handed history across the boundary, so a strategy with a 200-bar warm-up gives up its first 200 bars there; that understates the out-of-sample result rather than flattering it, which is the direction to err in.

Leaving `oos` out warns, because the default cannot be silent without being a trap. `oos=0` says you meant to search the whole series and passes quietly.

### The search space

`space` maps each parameter name (matching a constructor argument) to a range:

| Form | Meaning |
| :--- | :--- |
| `(low, high)` of two ints | Integer axis over `[low, high]`. |
| `(low, high)` with a float end | Continuous axis over `[low, high]`. |
| `(low, high, "log")` | Same, sampled on a log scale. |
| `[a, b, c]` | Categorical choice from the list. |
| `tune.Int(low, high, step=, log=)` | Integer axis with an explicit step or log scale. |
| `tune.Float(low, high, step=, log=)` | Continuous axis with an explicit step or log scale. |
| `tune.Categorical([...])` | Categorical choice. |

### The objective

`objective` is either a stats key (`"sharpe"`, `"calmar"`, `"total_return_pct"`, any of the [stats](#statistics)) or a callable that receives the `BacktestResult` and returns a number. `direction` (`"maximize"` by default, or `"minimize"`) sets which way is better. A trial whose strategy or objective raises, or whose objective is `NaN`, is marked failed and the search moves on; an infinite value (`profit_factor` with no losing trades, say) is kept and ranks at the extreme. If every trial fails, `tune` raises, with the last error attached. Before the search, `tune` checks the space names against the strategy's constructor signature and a string objective against the known stat keys, so a misspelled name or key raises at once; the check is by inspection, not a probe run, so no one corner of the space can abort an otherwise valid search.

### Controls and the result

| Argument | Default | Does |
| :--- | :--- | :--- |
| `n_trials` | 100 | How many parameter sets to try. |
| `n_jobs` | 1 | Worker processes; `-1` uses every core. `1` runs in this process, with no pickling. |
| `seed` | `None` | Seeds the sampler. |
| `direction` | `"maximize"` | `"maximize"` or `"minimize"`. |
| `verbose` | `False` | `True` restores optuna's per-trial logging. |
| `min_trades` | 0 | Fails a trial that closed fewer round trips, where `search` floors it. |

The `TuneResult` carries `.best_params` (a dict), `.best_value` (the objective at the best trial), `.best_stats` (the winning run's full [stats](#statistics) dict, so its return, drawdown, and trade count are there without a re-run), `.trials` (a list of `{number, params, value, state, stats}`), and `.study` (the underlying optuna study for deeper inspection). `.best_strategy()` builds a fresh strategy from `.best_params`.

It also carries `.best_result`, the winner re-run on the bars it was chosen on, so it can be charted or measured without rebuilding the search's configuration by hand, plus `.oos_result` and `.oos_stats` when a [holdout](#in-sample-and-out-of-sample) was kept. The last five, `.objective`, `.sampler`, `.direction`, `.min_trades` and `.data_hash`, are what a search knows about itself, and they exist so that [`deflated_sharpe`](#what-the-number-is-worth) can check a null belongs to this search rather than trusting you: the right statistic, an honest sampler, the direction it was pointed, no activity floor narrowing its spread, and the same bars. A deflation asks how high the best of many looks would reach on luck alone, so a search that minimized its objective is refused rather than measured against the wrong end of its own spread.

`sampler` is `"tpe"` by default, which learns where the good parameters are, or `"random"`, which does not. Random search is worse at finding a winner and is exactly what you want for a null.

With `n_jobs=1` a seeded search is reproducible, since the sampler is told results in a fixed order. With `n_jobs>1` the sampler sees results in completion order, so the exact sequence of trials can vary run to run even with a seed; each trial's score is still deterministic. On Windows, calling `tune` with `n_jobs>1` from a script means putting the call under `if __name__ == "__main__":`, the standard requirement for the spawn start method; a notebook needs no guard.

<br>

## Metrics

The fifteen [statistics](#statistics) come back from the engine. `probatum.metrics` is what you read afterwards: where the money actually went, and whether the number is worth believing. It simulates nothing and every function takes the `BacktestResult` itself, so it reads the opening balance and the annualization the run recorded rather than being told them again.

```python
from probatum import metrics

result = Backtester(candles, market="perp", funding_rate=0.0001, funding_interval=8).run(MyStrategy())

metrics.summary(result, candles)          # the headline, printed
metrics.decompose(result)                 # gross, fees, funding, still open, net
metrics.long_short_split(result)          # the same trade stats, by side
metrics.drawdown_table(result, top=5)     # the five worst falls, with durations
metrics.probabilistic_sharpe(result)      # the chance the true sharpe beats zero
```

`returns(result)` is the per-period return series everything else here is built on, and it is worth knowing the rule because two of this library's bugs came from a second reading of it. It is seeded from the opening balance, so a strategy that loses on its first bar shows that loss: the engine records an equity point only on a real advance, so the balance a run opened with is not in `equity_curve` and anything reading that array alone cannot see the first bar's move. A bar following a non-positive equity contributes zero rather than a ratio through zero. That is the identical rule the engine uses, which is what lets `metrics.sharpe` reproduce `result.stats["sharpe"]` exactly, and a test pins the two together.

### Where the money went

`decompose(result)` splits the change in equity into `gross_pnl`, `fees`, `funding`, and `unrealized`: gross less fees less funding plus unrealized is `net`, by construction. The two costs are reported as **positive amounts paid away**, the convention `funding_paid` already uses, so they subtract. On a perp this is the first thing to look at: a large share of what looks like alpha in crypto is a funding carry wearing a costume, and a larger share of dead strategies died on the fee line rather than on the idea. `unrealized` is what a position still open at the end holds at the last close; it is zero on a run that ends flat.

`decompose` also carries `fee_share`, the costs as a fraction of the gross edge, and `turnover`, the notional every fill moved, both sides of every round trip, over the opening balance. Read together they say whether you have a cost problem or an idea problem, which neither says alone.

`long_short_split(result)` gives trades, net PnL, fees, win rate and average holding time for each side separately, because almost every naive crypto rule earns long and bleeds short, and one win rate averages the two into something describing neither.

`trade_stats(result)` is the shape of the trade distribution, which a win rate hides completely: `avg_win`, `avg_loss`, `payoff` (the two divided), `expectancy` per trade, `largest_win`, `largest_loss` and `max_consecutive_losses`. A 70% win rate at a payoff of 0.3 is a losing rule, and a win rate and a profit factor are the same pair for a strategy that grinds out small wins and for one that is short gamma waiting for the bar that ends it. `summary` prints the win rate and the payoff on adjacent lines for this reason and never one without the other.

### How much friction it survives

The shipped defaults are a frictionless venue: no spread, `slippage_bps=0.0`, `impact=0.0`, `budget=1.0`, and on a perp `funding_rate=0.0`. An edge found there is an edge nobody can trade, so the useful question is not whether a strategy makes money but how much cost it lives through.

```python
metrics.cost_curve(SmaCross(20, 60), candles, costs=(0.0, 2.0, 5.0, 10.0, 20.0))
metrics.breakeven_bps(SmaCross(20, 60), candles)  # the round-trip cost that kills it
```

Both re-run the backtest, so they take the strategy and the data rather than a finished result. Costs are **round trip** in basis points, split evenly across the two sides, so `10` is five in and five out. A `Strategy` subclass is rebuilt for each run; an instance has its attributes put back as they were before each one, so your configuration survives and nothing a previous run left does ([ADR 0024](Decisions.md)). Either way every run starts from the same strategy, which is the whole premise of a sweep that moves only the fee. Anything else you pass goes to the `Backtester` unchanged, except `fee_taker` and `fee_maker`, which the sweep sets itself and refuses as a contradiction. `breakeven_bps` returns `None` when the strategy already loses for free, and the ceiling when it survives all the way there.

### The shape of the ride

`drawdown(result)` is the fall from the running peak at every bar, seeded from the opening balance so a loss on the first bar shows on the first bar. `drawdown_table(result, top=5)` gives the worst falls with where each began, bottomed and recovered, as **bar indices**, the same numbering as a round trip's `open_bar`, so one goes straight to the candle or onto a chart; `recovered_bar` is `None` for a fall the run never came back from. `time_under_water(result)` reports the longest and average stretch below a previous high, and the share of the run spent there: five shallow dips and one long one are the same `max_drawdown_pct` and not remotely the same thing to hold. `ulcer_index(result)` charges for depth and duration together, so a year spent down 15% scores worse than a week at 30%, which is the right way round for anybody who has to hold it.

`skew(result)` and `kurtosis(result)` describe the return distribution's third and fourth moments; negative skew means the losses are the fat tail. `kurtosis` is **not** excess by default, so a normal distribution scores 3.0, because that is the convention the probabilistic Sharpe below wants and handing it an excess figure is the standard way to get a confidently wrong answer. `value_at_risk(result, alpha=0.95)` is the per-period loss only one bar in twenty is worse than, and `conditional_value_at_risk` the mean across those bars, which is the one to read: the quantile says where the bad days start and this says how bad they are. `alpha` is a **confidence**, so it is at least 0.5; passing 0.05 asks about the top of the gains and is refused rather than answered. Both are historical, with no normal assumption, and `value_at_risk` comes back negative when even the worst bar in twenty made money.

`omega(result)` is the total gain above a threshold over the total loss below it, which reads the whole distribution rather than its first two moments, so a fat left tail cannot hide behind a tidy standard deviation. `tail_ratio(result)` is the size of the right tail over the left: below one, the run makes its money in small pieces and gives it back in large ones.

### Over time rather than in total

`segment(result, start, stop)` restricts a finished run to a range of bars and computes the engine's own statistics over it, seeded from the balance the account carried into that range rather than a fresh one, so consecutive segments compound to the run. `exposure_pct` and `num_fills` are absent from it, because a finished result does not carry what they are computed from.

```python
metrics.period_returns(result, candles, by="month")   # or "quarter", "year"
metrics.rolling_sharpe(result, window=720)            # length T, aligned, NaN warm-up
```

`period_returns` is the first stability question anybody asks of a crypto backtest, which is whether the whole edge sits in one quarter, and a single number cannot answer it. `rolling_sharpe` is the same question as a shape: aligned and padded on the same rule `probatum.ta` uses, so it drops straight into [`probatum.chart`](#plotting) beside the equity curve.

### The three that need the candles

`buy_and_hold(result, frame)` answers the first question anyone asks, with the excess return, the beta against holding, and the information ratio.

`excursions(result, frame=None)` gives the worst and best each closed round trip went while it was open, in percent of its entry: the round trip's own `worst` and `best`, which the engine measures as it runs. A trade log says what a rule earned; this says what it lived through, and it is the direct answer to where a stop belongs. Both are a **bound rather than an exact figure, and neither overstates**: a bar prints its open first and its close last and nothing else in it is ordered, so the bar a trade entered on counts only from the fill forward to its close, the bar it left on only from its open up to the fill, a bar it was closed out of before the open not at all, and the bars strictly between count whole ([ADR 0023](Decisions.md)). A market entry at the open is counted the same way, from its fill to the close. `session_buckets(result, frame, by="hour")` groups trade PnL by hour of day or day of week, booked at the exit bar; it needs the DataFrame rather than the array, because only a real clock can be bucketed by hour. Crypto trades around the clock and an hourly strategy routinely has its whole edge inside the few hours a day funding is stamped, which one bucket holding the entire result will show faster than any statistic. Hours are UTC, and **weekdays run Monday 0 to Sunday 6**, matching `datetime.weekday` and pandas' `dayofweek` so a key reads against any other clock you are holding.

**Each of the three checks that the frame is the one the run saw, and they do not check the same thing** ([ADR 0030](Decisions.md)). `session_buckets` reads the run's own bar indices straight into these bars, and `excursions` hands them back, so both need the *identical* series and compare its fingerprint against `result.data_hash`; another asset of the same length made a losing trade report both of its excursions as gains. `buy_and_hold` needs only the same *number* of bars, because a different asset there is the whole point: `beta` and `information_ratio` are benchmark statistics and are only interesting against something other than what you traded, so benchmarking an ETH strategy against holding BTC is a supported question. All three used to absorb a mismatch silently, which is how a 2,000-bar slice of a 100,000-bar frame returned 35 excursion rows out of 1,516 and said nothing.

### What the number is worth

`probabilistic_sharpe(result, benchmark=0.0)` is the probability the true Sharpe is above the benchmark, given the sample length and how far the returns are from normal. `min_track_record_length(result)` is how many bars it would take to distinguish the two at 95%.

`sharpe_interval(result)` is the same estimator read as an interval instead of a tail, which is usually the sentence you want: "sharpe 1.4, 95% interval 0.2 to 2.6". A `low` at or below zero says the run does not distinguish itself from nothing, however good the point estimate looks.

Three things to know before quoting any of them. The benchmark is stated **annualized**, like everything else here, and de-annualized internally, because handing the estimator an annualized figure is the standard way to get a confidently wrong answer.

The published estimators count bars and assume each is an independent bet, which a strategy holding a position for two days on hourly candles badly violates. So the sample size counted here is `effective_sample(result)`, the standard `n (1 - r) / (1 + r)` at the first autocorrelation, and not the bar count ([ADR 0031](Decisions.md)). The library used to report `autocorrelation(result)` beside these, say in its own docstring that every confidence figure computed from the bar count was therefore overstated, and then compute them from the bar count anyway. Pass `independent=True` for the unadjusted formula. Read `num_trades` too: forty round trips over three thousand bars is forty bets and it is the bets that carry the information.

And a sample that cannot support the estimate raises rather than returning `NaN` ([ADR 0030](Decisions.md)).

None of them corrects for having searched. `deflated_sharpe(study, null)` does:

```python
study = venue.tune(SmaCross, space, candles, n_trials=200, oos=0.3)
null  = venue.tune(SmaCross, space, candles, n_trials=200, oos=0.3, sampler="random")

metrics.deflated_sharpe(study, null)      # is the winner real, given 200 looks?
```

Both take the same `oos`. The fingerprint is checked, and a holdout changes which bars a search actually ran on, so a null fitted on the whole series is not a null for a search fitted on 70% of it.

A search returns the best of many noisy scores, so its winner is high partly because it is good and partly because you looked a lot. This computes the Sharpe the best of that many looks would reach **by luck alone** and asks for the probability the winner beats it.

**The null is required, and that is the design rather than an inconvenience.** The threshold depends on how many independent looks were taken and how far the scores scatter, and a TPE search supplies neither honestly: it concentrates its trials in the winning basin, so they are not independent draws and the spread between them *shrinks as it converges*. Computed from the search's own trials this number would grow more permissive the harder you overfit, which is exactly backwards. A random search over the same space is an honest "best of N looks here" and cannot do that ([ADR 0032](Decisions.md)).

**The two arguments answer different halves of the threshold.** The null says how far a sharpe scatters over this space and these bars, and nothing else. How many times you looked is a property of your search, so it is read off the study: every trial of it, including the failed ones, because a configuration you evaluated and rejected is one you evaluated. Taking the count off the null instead answered about the wrong search, and made a winner more convincing the smaller the null ([ADR 0032](Decisions.md)).

Five things are refused rather than guessed: a null that is not a random search, a null over different bars (checked by fingerprint), a search that selected on something other than `sharpe`, a null too small to set a threshold, and a null carrying a `min_trades` floor, since an activity floor keeps the trials that traded most and those scatter less than the space does. A null thinner than the search it judges warns instead of raising. `deflation_threshold(spread, looks)` is the bar itself, if you want to see it.

A tuned parameter set is still in-sample no matter what probability is attached to it, so this complements the [holdout](#in-sample-and-out-of-sample) rather than replacing it.

`report(result, frame=None)` returns everything above as one flat dict, for storing or comparing runs; `summary(result, frame=None)` prints the headline and returns the same dict.

### Telling two runs apart

A `BacktestResult` carries `config` (the engine settings it ran under), `data_hash` (eight characters over the candles it saw), `strategy` (the class name plus its `repr`), `version`, `bust`, `periods_per_year` and `risk_free`. `to_dict()` returns all of it plus the stats as plain data, without the equity curve or the trade log, since neither distinguishes anything and both are large.

```python
metrics.compare({"cheap": cheap_run, "dear": expensive_run})
```

prints one row per run, leading with the data fingerprint, and returns the rows. `keys=` picks the columns, and a key no result reports is refused rather than printed as the word `None` in every row, which reads as a run that scored nothing instead of a column that does not exist. A notebook accumulates a great many backtests and nothing on a bare result says which bars or which fees produced it, so "was Tuesday's 2.1 sharpe on the same costs as today's 1.9" had no answer at all ([ADR 0026](Decisions.md)). Defining `__repr__` on a tunable strategy is worth it here: two rows both reading `SmaCross` tell you nothing.

<br>

## Indicators

30 functions, chosen to cover what a bar-level strategy usually reaches for and stopped there. This is not an attempt to be ta-lib: an indicator library is mostly somebody's opinion about warm-up and smoothing, and a hundred of those opinions is a hundred chances to disagree with the chart you have read for years.

```python
class Cross(probatum.Strategy):
    def init(self, data):
        self.fast = probatum.ta.ema(data.close, 20)
        self.slow = probatum.ta.ema(data.close, 60)
        self.warmup = 60

    def next(self, bar, account):
        if account.position == 0.0 and self.fast[bar.i] > self.slow[bar.i]:
            account.buy(1.0)
        elif account.position > 0.0 and self.fast[bar.i] < self.slow[bar.i]:
            account.close()


result = probatum.backtest(Cross(), data, "frictionless")
```

| Group | Functions |
| :--- | :--- |
| Trend | `sma(values, length)`, `ema(values, length)`, `wma(values, length)`, `hma(values, length)`, `vwma(values, volume, length)`, `vwap(high, low, close, volume, length)` |
| Momentum | `rsi(values, length=14)`, `macd(values, fast=12, slow=26, signal=9)`, `stoch(high, low, close, length=14, smooth=3)`, `willr(high, low, close, length=14)`, `cci(high, low, close, length=20)`, `roc(values, length)` |
| Volatility | `atr(high, low, close, length=14)`, `natr(high, low, close, length=14)`, `true_range(high, low, close)`, `stdev(values, length)`, `bbands(values, length=20, deviations=2.0)`, `keltner(high, low, close, length=20, atr_length=10, multiplier=2.0)` |
| Regime | `adx(high, low, close, length=14)`, `supertrend(high, low, close, length=10, multiplier=3.0)` |
| Structure | `donchian(high, low, length=20)`, `highest(values, length)`, `lowest(values, length)`, `zscore(values, length)` |
| Participation | `obv(close, volume)`, `mfi(high, low, close, volume, length=14)` |
| Comparison | `shift(values, bars=1)`, `change(values, bars=1)`, `crossover(values, other)`, `crossunder(values, other)` |

Every function also takes `axis`, the axis time runs along, 0 by default.

**The comparison group is the one that is about safety rather than convenience.** `shift(values, n)` is how you look back: written by hand, `values[i - n]` on an early bar is a negative index, which numpy resolves from the END of the array, so the rule reads a price from the far future and raises nothing. `Strategy.warmup` ([ADR 0025](Decisions.md)) closes that only when the declared warm-up is at least as deep as the lookback, and declaring the indicator's warm-up and then reaching further back inside `next` is the natural way to get it wrong. `shift` pads rather than wrapping, so there is no index left to get wrong. On a universe an index also counts rows, so the row before an instrument's bar can be one it printed nothing on, where `shift` lags each instrument over its own bars; and `warmup` counts rows too, so `next` can run while an instrument that lists late is still in its own warm-up.

`crossover` and `crossunder` return **booleans**, and they are the only functions here that do not pad with `NaN`. `bool(nan)` is `True` in Python, so a float warm-up would fire a rule on precisely the bars where nothing is known; they pad with `False` instead, and a cross across a bar either of the two lacks is reported on the first bar after it that both have ([ADR 0037](Decisions.md)). `other` may be a second series of the same shape or a plain number, since crossing a level is the commoner case:

```python
class Cross(probatum.Strategy):
    def init(self, data):
        self.up = probatum.ta.crossover(probatum.ta.ema(data.close, 20),
                                        probatum.ta.ema(data.close, 60))
        self.strong = probatum.ta.adx(data.high, data.low, data.close).adx
        self.warmup = 80

    def next(self, bar, account):
        if self.up[bar.i] and self.strong[bar.i] > 25.0:      # a cross, in a trend
            account.target(1.0)


result = probatum.backtest(Cross(), data, "frictionless")
```

**Four rules hold for every one of them**, and they are why this is in the library rather than in your notebook.

*One value a bar, each instrument over its own.* Every function takes a series `(T,)`, or a universe `(T, N)` with a column for each instrument, time along `axis`, and returns a float64 array of the same shape aligned so entry `i` is bar `i`, with the warm-up as `NaN`, never a shorter array. Each column runs over its own values: a value that is `NaN` or infinite is a row the column does not have and a gap in the output, and a window, a smoother, a lookback and a crossing carry across it rather than restarting. A Data's fields are `NaN` exactly where an instrument printed no bar, so a closure, a halt or a late listing costs an indicator only those rows, and a series too short for its warm-up is all warm-up rather than an error, only one of no bars being refused, so a cut of a series that has a bar gives the first rows of the whole. `funding_rate` and what `asof` puts on the clock hold their last value through a closure, so an indicator runs over those rows as well; where an instrument's own bars are meant, mask a `(T, N)` one with `numpy.where(data.available, x, numpy.nan)`, and one instrument's `(T,)` line with `data.available[:, 0]`. A plain two-dimensional array is a column each, so a `(T, 5)` frame comes back as five series, and an `(N, T)` array takes `axis=1`. So the same object goes into your rule and into [`probatum.chart`](#plotting), a column at a time, with no padding decision in between ([ADR 0037](Decisions.md)).

```python
gapped = bars.copy()
gapped[500:510] = numpy.nan                     # ETHUSDT prints nothing for ten hours
pair = probatum.Data(numpy.stack([bars, gapped], axis=1), symbol=["BTCUSDT", "ETHUSDT"],
                     market="linear", time=hours)
trend = probatum.ta.ema(pair.closes, 50)        # (1000, 2), each over its own bars
print(trend[505], trend[510])                   # NaN in ETHUSDT's hole, then carried on
```

*The convention is written down.* `ema` is smoothed at `2 / (length + 1)` and seeded from the simple average of the first `length` values, not from the first value, which is TradingView's convention and stops one opening print steering the curve for hundreds of bars. `rsi`, `atr` and `adx` use Wilder's `1 / length`, which is what those indicators were defined with. `stdev` is population, dividing by `length`, because that is what `bbands` is defined against. `cci` divides by the mean **absolute** deviation and not the standard deviation, which is the definition and is what makes it behave differently from `zscore` on a fat-tailed series. `hma` floors both of its inner lengths to whole bars. Each function repeats its own choice in its docstring.

*Where there is nothing to say, it says nothing rather than a number.* A window with no range has no position inside it, so `stoch` and `willr` are a gap there rather than a fifty, and `stoch`'s `d` averages over the bars `k` has a value on; a window that traded nothing has no weighted price, so `vwap` and `vwma` are a gap rather than an unweighted average; a market with neither gains nor losses has no ratio of strength to weakness, so `rsi` and `mfi` are a gap rather than the 100 some implementations report, which would fire an overbought rule on a market that has not moved. A `NaN` in an array of your own is a gap like any other, so a signal whose `NaN` means something else is lagged by rows with numpy rather than with `shift`.

*Nothing knows about the engine.* These are functions of arrays: no state, no engine, no pandas needed. Computing once in `init` and handing the same array to the chart is what keeps the line on screen from drifting from the line that made the decision ([ADR 0042](Decisions.md)).

Functions with more than one output return a named object rather than a tuple to unpack, so a caller cannot one day unpack three anonymous arrays in the wrong order. There are five of them: `Bands` (`.upper`, `.middle`, `.lower`) from `bbands`, `keltner` and `donchian`; `Macd` (`.line`, `.signal`, `.histogram`); `Stochastic` (`.k`, `.d`); `Trend` (`.adx`, `.plus`, `.minus`); and `Channel` from `supertrend`, which adds `.direction` to the three band lines. Read `.direction`, which is `1.0` while long and `-1.0` while short, rather than comparing the close against the line: the line is the edge in force and that comparison is what set it.

Three behaviours worth knowing. `vwap` is rolling over `length` bars rather than anchored to a session, because the engine has bars and no notion of a trading day. `donchian`, `highest` and `lowest` all include the current bar, so a breakout of `upper` cannot happen on the bar that set it; compare against `shift(highest(...), 1)` if that is what you mean. And `willr` is `stoch`'s `k` read from the top of the range rather than the bottom, so it runs -100 to 0 where `k` runs 0 to 100; both are here because both sign conventions are in wide use and quietly picking one is how a rule ends up backwards.

<br>

## Plotting

`probatum.chart` draws a frame, your own arrays and a run. `show()` puts the chart in the notebook cell, and it is stored in the `.ipynb`, so reopening that notebook shows the same charts with no kernel running. `save(path)` writes one HTML file that opens in a browser instead.

It computes nothing and knows no indicator names: you bring arrays, it turns them into marks ([ADR 0042](Decisions.md)). Everything is precomputed, which is what lets a chart outlive the process that made it ([ADR 0042](Decisions.md)). No dependency beyond numpy and the standard library: pandas is duck-typed, and IPython is imported only inside `show`.

```python
import probatum
from probatum.plot import Line, Band, Level, Panel

result = probatum.Backtester(frame, market="perp", fee_taker=0.0005).run(s)

probatum.chart(frame, [                                  # matched by type, in any order
    Line(s.fast, "SMA 20"),                          # the strategy's own array, not a copy
    Line(s.slow, "SMA 50", style="dashed"),
    Line(rsi, "RSI 14", panel="momentum"),           # a name not yet used makes a panel
    Level(70, panel="momentum", style="dotted"),
    Band(rsi, 70, only="above", panel="momentum",    # shaded only where it is past 70
         fill=("#ff547000", "#ff547059")),
], result,                                           # candles, arrows, equity, drawdown, log
    panels=[Panel("momentum", weight=1.6, range=(0, 100))],
    title="SMA cross 20/50",
).show()
```

`chart(frame, *args, marks=, run=, panels=, focus=, candle_color=, theme=, palette=, height=, title=, future=0, stats=, trades=True, drawdown=, notes=)` returns a `Chart`, with `show()` for a notebook cell, `save(path)` for a file, and `spec()` for the underlying document. `frame` is a DataFrame with a DatetimeIndex or a parquet path, narrower than the `Backtester` deliberately: a chart cannot omit its x axis, and fabricating one is how a file ends up reading 1970.

Arrays are read by position. Length `T` maps entry `i` to bar `i` and length `T - 1` maps entry `i` to bar `i + 1`, which is what `equity_curve` and a diff are; any other length raises and names both numbers ([ADR 0042](Decisions.md)). A `NaN` is drawn as a gap rather than a dropped row ([ADR 0042](Decisions.md)).

The full contract, the mark reference, the padding helpers for values recorded inside `next`, and what cannot be charted are in the [Plotting](Plotting.md) guide.

<br>

## Errors and warnings

Every error probatum raises is one of its own classes and the built-in it means, so `except ValueError` still catches a bad value and `except probatum.ProbatumError` catches probatum's errors and nothing else. Each names the argument as you typed it and the value that arrived, and puts the fix after a semicolon.

| Class | Built-in | What it covers |
| :--- | :--- | :--- |
| `probatum.ProbatumError` | `Exception` | Every class below. |
| `probatum.ProbatumValueError` | `ValueError` | A wrong value, the four kinds below among them. |
| `probatum.DataError` | `ValueError` | A wrong value in bars, frames, the clock, features, funding or a series an indicator reads. |
| `probatum.VenueError` | `ValueError` | A wrong value in a venue's settings, and any wrong setting in a preset's file. |
| `probatum.SpecError` | `ValueError` | An agent whose spec does not match the run it is given; nothing raises it yet. |
| `probatum.LookaheadError` | `ValueError` | A policy the causality check of `search` or `validate` finds deciding on bars that had not closed. |
| `probatum.ProbatumTypeError` | `TypeError` | An argument of the wrong type, `PolicyError` among them. |
| `probatum.PolicyError` | `TypeError` | A policy `backtest`, `search` or `validate` cannot run as given: a class for an instance, and for a search or a validation an instance for its class, a class that is no Strategy or a function that takes no data, a model that fits, an object of no kind it reads, an array it cannot read as decisions, a trial that decides in another kind than the trials before it, and a policy the workers cannot carry. |
| `probatum.ProbatumKeyError` | `KeyError` | A name none of a set's holds, such as an instrument a Data does not have. |
| `probatum.ProbatumIndexError` | `IndexError` | A position past the end of a table. |
| `probatum.ProbatumAttributeError` | `AttributeError` | An attribute an object does not have, such as one instrument's `close` on a universe, which `hasattr` reads as missing. |
| `probatum.ProbatumRuntimeError` | `RuntimeError` | A search or a validation that cannot go on: its workers lost, every trial of a search or every path of a validation failed, or its winner run again not matching its trial, and `ProbatumNotImplementedError`, as `NotImplementedError` is a `RuntimeError`. |
| `probatum.ProbatumNotImplementedError` | `NotImplementedError` | A Strategy that writes no `next`, an agent, which `backtest` does not run yet, the transitions level, which it does not record yet, a search's `storage`, which it does not save to yet, and a model or a function, which `validate` does not run yet. |
| `probatum.MissingDependency` | `ImportError` | A package a call needs that is not installed, named with the line that installs it. |

A refusal is a kind where both its topic and its built-in are the kind's, and otherwise the class for its built-in, which follows what arrived: a value of a type the argument does not take raises a `TypeError`, and one of its type that the call cannot take a `ValueError`, so `tolerance="0.01"` is a `ProbatumTypeError` and `tolerance=-0.01` a `ProbatumValueError`. So a Data's wrong value is a `DataError` and its wrong type a `ProbatumTypeError`, which `except probatum.DataError` does not catch, and a Venue built in Python splits the same way between `VenueError` and `ProbatumTypeError`, while every wrong setting in a preset's file is a `VenueError`. A policy's wrong value is a `ProbatumValueError`, not a `PolicyError`. A symbol the router fails on in `Data.from_router` raises the router's own error, as a pull of that symbol alone does, so `except ValueError` does not catch a failed batch.

What another library raises stays its own: the router's errors, the file system's from `save` and `load`, numpy's for values an indicator, `asof` or `features` cannot read as numbers, Python's `OverflowError` for an integer no float holds, and whatever your own code raises inside a policy. The binding's raw objects, `probatum.Batch` and those of `probatum._probatum`, refuse a wrong value with the same classes, but what pyo3 converts itself, an argument of a type their signature cannot take or a part of one it cannot unpack, it refuses as a `TypeError`, an `OverflowError` or a `ValueError`, with a note naming an argument the signature declares, and a call from a second thread into an object another call holds as a `RuntimeError`. emsl's modules the package still carries, `Backtester`, `Market`, `tune`, `metrics`, `chart`, `plot`, `rl`, `sb3` and `to_ohlcv`, raise the built-ins from their own checks, while what reaches them from the engine or from a `Strategy` is probatum's, and warn with `UserWarning` or `RuntimeWarning`; `metrics.summary` and `metrics.compare` print their tables.

Every warning probatum gives is a `probatum.ProbatumWarning`, a `UserWarning`, named at the line of your code that caused it, so a test can make probatum's warnings errors without touching anyone else's. Outside those modules probatum prints nothing: long work is logged to the `probatum` logger, `Data.from_router` logging each instrument it built at INFO and each call to the router at DEBUG, which Python's `logging` turns on.

```python
import logging
import warnings

try:
    probatum.backtest(Hundred, data, "frictionless")      # the class, not an instance
except probatum.PolicyError as error:
    print(error)            # policy: got the class Hundred, not an instance; pass Hundred()

try:
    data["SOLUSDT"]
except KeyError as error:                                # a ProbatumKeyError is one
    print(type(error).__name__, error)

logging.basicConfig(level=logging.INFO)                  # what long work reports
warnings.simplefilter("error", probatum.ProbatumWarning)  # probatum's warnings as errors
```
