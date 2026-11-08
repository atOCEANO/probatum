<h1>OCEΛNO <small><code>probatum</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/probatum/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/probatum?label=release&color=2ea043" alt="Latest release" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-1.2+-0081a5.svg" alt="Gymnasium 1.2+" /></a>
  <a href="https://github.com/atOCEANO/probatum/releases"><img src="https://img.shields.io/badge/wheels-Linux%20%7C%20macOS%20%7C%20Windows-informational.svg" alt="Wheels for Linux, macOS and Windows" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <a href="../README.md">Introduction</a> &nbsp;•&nbsp;
  <a href="Research_Guide.md">Research Guide</a> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Charts.md">Charts</a> &nbsp;•&nbsp;
  <b>Python API</b> &nbsp;•&nbsp;
  <a href="Specification.md">Specification</a> &nbsp;•&nbsp;
  <a href="Venue_Notes.md">Venue Notes</a> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href="Migration.md">Migration</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Python API

Everything the package exports, and where each piece is documented in full:

| Import | What it is |
| :--- | :--- |
| `probatum.Data` | N instruments over T bars on one clock, read from any frame or array. [Below](#data). |
| `probatum.Instrument` | One instrument of a Data: its symbol, market and venue, its assets, an inverse contract's size and the venue's limits on an order. [Below](#data). |
| `probatum.Venue` | A venue's rules for a run, in one immutable object. [Below](#venue). |
| `probatum.venues` | The presets, each a Venue: `bybit_perp`, `okx_perp`, `hyperliquid_perp`, the spot, inverse and combined ones, and `frictionless`. [Below](#venue). |
| `probatum.Engine` | An account at a venue on a Data, for a loop the caller owns. [Below](#engine). |
| `probatum.Batch` | Many independent accounts over one shared series, stepped in parallel with the GIL released. [Below](#batch). |
| `probatum.backtest` | Run a policy on a Data at a venue: an array or a table of weights or quantities, a Strategy, or a trained agent. [Below](#backtesting). |
| `probatum.Result` | What a run recorded: its equity and returns, its tables, the totals of its costs, its graded assumptions, its baselines, each instrument's record, its statistics and its provenance. [Below](#the-result). |
| `probatum.combine` | Results from accounts at different venues added into one book. [Below](#combining-books). |
| `probatum.search` | Search a policy's parameters on the bars before a holdout, keeping every trial. [Below](#search). |
| `probatum.Study` | What a search or a validation found: every trial or every out-of-sample path, their returns, and the winner's Result. [Below](#the-study). |
| `probatum.Int` | A whole number a search draws. [Below](#the-space). |
| `probatum.Float` | A number a search draws, from an interval or a lattice. [Below](#the-space). |
| `probatum.Categorical` | One of the choices a search draws among. [Below](#the-space). |
| `probatum.labels` | What one trade decided at each row's close makes through the fill model, for a model to fit on. [Below](#labels). |
| `probatum.Labels` | What `labels` found: each label's return, exit, bars, end and uniqueness. [Below](#labels). |
| `probatum.cv` | The splitters `Holdout`, `WalkForward`, `PurgedKFold` and `CombinatorialPurged`, which scikit-learn's tools take and which purge by label end. [Below](#splitters). |
| `probatum.validate` | Search or fit on each split's training bars and trade the winner on its test span, each out-of-sample path one run. [Below](#validation). |
| `probatum.stats` | How far a run's statistics and a search's winner can be trusted, and the statistics that describe a run. [Below](#inference). |
| `probatum.compare` | Runs set side by side, a row each. [Below](#statistics-of-a-run). |
| `probatum.capacity` | A policy run again at several balances. [Below](#capacity-and-costs). |
| `probatum.cost_curve` | A policy run again with every cost its venue charges for trading scaled together. [Below](#capacity-and-costs). |
| `probatum.breakeven_bps` | The round-trip cost, in basis points, at which a policy stops making money. [Below](#capacity-and-costs). |
| `probatum.Capacity` | What `capacity` found, a row per balance. [Below](#capacity-and-costs). |
| `probatum.CostCurve` | What `cost_curve` found, a row per scale. [Below](#capacity-and-costs). |
| `probatum.Strategy` | A policy written as what it decides at each close. [Below](#a-strategy). |
| `probatum.ta` | 30 indicators, one value per bar, with the conventions written down. [Below](#indicators). |
| `probatum.env` | Accounts at a venue on a Data, stepped together in Rust as a Gymnasium vector environment. [Below](#reinforcement-learning), in full in the [RL Guide](RL_Guide.md). |
| `probatum.rl` | `VectorEnv`, the environment `probatum.env` returns, `Env`, one environment of it, both registered as `probatum/Trading-v0`, and `Batch`. [Below](#reinforcement-learning). |
| `probatum.sb3` | `wrap`, which presents an environment to Stable-Baselines3 as a `ProbatumVecEnv` carrying its agent spec, and `SpecCallback`, which keeps `VecNormalize`'s statistics in the model's spec. [RL Guide](RL_Guide.md#training). |
| `probatum.save` | Write a trained agent beside the spec it trained with. [Below](#an-agent). |
| `probatum.torch` | `TensorEnv`, which hands an environment's observation, rewards and flags out as tensors on a device. [RL Guide](RL_Guide.md#tensors-and-the-gpu). |
| `probatum.chart` | The interactive chart: `Chart`, `defaults` for the look of every later chart, and `marks`, holding `Line`, `Histogram`, `Band`, `Level`, `Marker`, `Markers`, `Background`, `Panel`, `Recorder`, `ramp`, `at_bar` and `at_next`. [Below](#charts), in full in the [Charts](Charts.md) page. |
| `probatum.ProbatumError` | The root of every error probatum raises: a class for each built-in, such as `ProbatumValueError`, and the kinds `DataError`, `VenueError`, `SpecError`, `LookaheadError`, `PolicyError` and `MissingDependency`. [Below](#errors-and-warnings). |
| `probatum.ProbatumWarning` | The class of every warning probatum gives, a `UserWarning`. [Below](#errors-and-warnings). |

Everything here runs on the one Rust engine, so a backtest, an RL rollout and a parameter search see the same fill model and the same costs.

<br>

## Quickstart

Build a Data, run a policy on it at a venue, read the Result. The sections from The Result on build on these names.

```python
import numpy
import probatum

close = 100.0 + numpy.random.default_rng(seed=0).normal(loc=0.0, scale=1.0, size=1_000).cumsum()
bars = numpy.column_stack([close, close + 0.5, close - 0.5, close,
                           numpy.full(shape=1_000, fill_value=1e3)])
hours = numpy.arange(stop=1_000, dtype=numpy.int64) * 3_600_000_000_000
data = probatum.Data(bars, symbol="BTCUSDT", market="linear", time=hours)


class Hundred(probatum.Strategy):
    def next(self, bar, account):
        if account.position == 0.0 and bar.i % 100 == 0:
            account.buy(size=0.1)               # decided now, fills on the next bar's open


result = probatum.backtest(Hundred(), data=data, venue="frictionless")
print(result)                                   # what ran, its equity, its optimistic assumptions
```

<br>

## Data

A `probatum.Data` holds N instruments over T bars on one clock. It copies its bars once into a buffer it owns, and every field and every slice is a read-only view of that buffer, or a read-only copy where a selection reorders the instruments. No later edit to the frame it came from can change the bars it holds ([0005](Decisions.md)).

```python
import pandas
import probatum

btc = probatum.Data(pandas.read_parquet(path="btc.parquet"), symbol="BTCUSDT",
                    market="linear")

pair = probatum.Data({"BTCUSDT": btc_frame, "ETHUSDT": eth_frame})
bars = probatum.Data(ohlcv, time=times, symbol="BTCUSDT")

universe = probatum.Data.from_router(router, exchange="bybit", market="linear",
                                     interval="1h", symbols=["BTCUSDT", "ETHUSDT"],
                                     start="2023-01-01", funding=True)
```

### What it reads

* **A frame** that exports the Arrow C stream or array interface, as polars, pyarrow, DuckDB and pandas with pyarrow do. The binding reads it directly, so no frame library is imported. A pandas frame is read by its own methods where pyarrow is missing or cannot convert one of its columns.
* The frame needs `open`, `high`, `low`, `close` and `volume` columns, and the bars' open times in its one datetime index or column, or in a `timestamp` column of milliseconds since 1970. A frame with times in more than one place is refused, naming them, unless `time=` gives the open times.
* **An array** is `(T, 5)` for one instrument or `(T, N, 5)` for N, with NaN in all five values where an instrument printed no bar. `time=` gives each row's open time, as datetimes or int64 nanoseconds since 1970.
* **A dict of frames** keys each frame by its symbol.
* **A frame from exchange-router** names itself: its venue, market, symbol and interval come from its `attrs`, and a `symbol=` or `market=` that says otherwise is refused.

### From the router

`Data.from_router(router, exchange, market, interval, symbols, start, end=None, funding=False)` fetches the bars through an exchange-router `Router`, from the first bar at or after `start` to the last at or before `end`. `start` and `end` are times or the text of one, and a text end takes in the whole period it names. The bar still open at the time of the fetch is never taken, since a venue serves it as though it were done.

* The router's calls are chained back to `start`. A call that brings nothing earlier is asked once more, and where it brings nothing again, the venue's history is taken to begin there, which `provenance["history"]` records and a warning names. Another warning names bars that end before `end`, and a gap in the bars is warned of too.
* The router's own warnings are kept in `provenance["warnings"]` rather than printed, and `provenance["fetched"]` holds the time of the fetch.
* A symbol the router fails on raises the router's own error, as a pull of that symbol alone does, rather than being left out. An `AsyncRouter` is refused.
* Each instrument takes the router's own name for its symbol, however it was typed, and its base asset and the venue's limits on an order from the router's symbol information. Two symbols the router names as one are refused. A symbol the router no longer lists, such as one delisted or paused since its bars, keeps its bars where the router still serves them, and a warning names it.
* With `funding=True` each instrument's funding events are fetched the same way. A series that misses part of its instrument's span is refused, naming the span: one that begins more than a cycle after the instrument's first bar or ends more than a cycle before its last bar ends, one with a stretch between two events more than half as long again as the spacing on either side, one with an event that has no rate, and one with no event over bars that span more than a day.

### Naming

`symbol` names an instrument, or with an array of N a list of N names. `market` is `"spot"`, `"linear"` or `"inverse"`; left out, it is the venue's to give when the Data runs. An instrument is keyed by its venue, market and symbol, and `data.names` gives each the short name the Data is indexed by: its symbol, with its market in front where another market holds the same symbol (`spot:BTCUSDT`, `linear:BTCUSDT`), and its venue in front of that where another venue does. An instrument with no symbol is named by its position, `"0"`, `"1"` and so on.

`data.instruments` holds each one's `probatum.Instrument`:

| Field | What it is |
| :--- | :--- |
| `symbol`, `market`, `venue` | where it trades, each `None` where the bars do not say |
| `quote` | the currency its price is quoted in, `None` where the bars do not say |
| `contract` | the dollars an inverse contract is worth, `None` where the bars do not say |
| `base` | its base asset |
| `min_notional` | the smallest notional the venue takes in an order, in the quote currency; left out on an inverse contract, where the router gives it in dollars and the engine counts in the coin |
| `max_qty` | the largest size the venue takes in a limit order, in base units, or in contracts on an inverse contract |
| `settlement` | the asset its account settles in: the coin of an inverse contract and the quote otherwise, `None` where the market is the venue's to give |

`base`, `min_notional` and `max_qty` come from the router's symbol information as of the fetch.

### The clock

The clock is the set of open times at which any instrument printed a bar ([Specification](Specification.md#the-clock)). Its `interval` is the one a router frame declares, which an `interval=` that says otherwise cannot change, or else the one `interval=` gives, or else the one the median gap between bars gives. That median is the shortest among the instruments, since one that prints seldom still prints on the clock's steps, and it is measured between dates for days, which across a change of clocks are an hour short or long. The interval is one of `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d` and `1w`.

Every row is a whole number of intervals from the first, so instruments whose bars open at different minutes are refused, and a daily or longer bar is placed at 00:00 UTC on the date it carries in its own time zone.

**Every bar is checked as it arrives**, as the engine checks it ([0007](Decisions.md)). A row holds all five values or none, each finite. The low is at or below the open and the close, and the high at or above both. The low is above zero and the volume not below it. An instrument's times are all present and rise, each once. A refusal names the instrument and the time of the bar.

### The fields

| Field | What it holds |
| :--- | :--- |
| `time` | `(T,)` int64 nanoseconds since 1970, UTC |
| `opens`, `highs`, `lows`, `closes`, `volumes` | `(T, N)`, NaN where an instrument printed no bar |
| `bars` | all five as `(T, N, 5)` |
| `available` | `(T, N)` bool |
| `funding` | the funding events the bars own, at their real times, as a structured array of `time`, `instrument` and `rate` |
| `funding_rate` | `(T, N)`, the last rate known at each bar's end |
| `interval` | the clock's interval |
| `fingerprint` | 128 bits of blake2b over everything a run on the bars can read, written as `"b2:"` and 32 hex characters |
| `provenance` | where the bars came from: the source, the venue, the instruments, the interval, the first and the last bar, the warnings that came with them and the fingerprint |

The fingerprint is the same however the same bars and instruments arrived, and changes with any change to the clock, the interval, the bars, which of them printed, the funding events or the instruments, their order limits included ([0005](Decisions.md)). A slice's fingerprint also covers the bars before it, which its features warm up on, and where it starts among them.

With one instrument the singular names read `(T,)`, as `data.close`. With several they are refused, naming the plural and the instrument's own view, `data.closes` and `data["ETHUSDT"].close`.

### Slicing

A Data slices on time and on instruments, and every slice is a read-only view that keeps the bars before it, so features and an agent's window can warm up on them.

```python
recent = btc["2025-06":"2025-12"]                # June to December, both included
first = btc[:1_000]                              # by position

train, test = universe.split(when="2026-01-01")  # before it, and from it on
train, test = universe.split(when=8_000)         # the first 8,000 bars, and the rest
late = universe.after(when="2026-01-01")

eth = universe["ETHUSDT"]                        # one instrument
pair = universe.select(names=["ETHUSDT", "BTCUSDT"])
both = spot.join(perp)                           # two markets on one clock

print(universe.gaps())                           # where the bars jump past the interval
```

A text bound names a year, a month, a day, an hour, a minute or a second, and a slice runs from the start of the first bound to the end of the second. A time names itself and a whole number is a position. `split` and `after` take the same bounds, and a period starts the later part where the period starts. `join` puts two Datas on one clock, such as a venue's spot and perpetual markets.

`gaps()` lists every place an instrument's bars jump by more than the interval, each as the instrument, the bar before the jump, the bar after it and how many bars are missing between them. On a market that never closes, a gap is an outage or a hole in the data. `describe()` gives the rows of a run's assumptions that the bars decide ([Costs, assumptions and baselines](#costs-assumptions-and-baselines)), and `chart()` draws the bars ([Charts](#charts)).

### Coarser bars and outside series

Coarser bars and outside series reach a Data only from the moment each was known ([0006](Decisions.md), [Specification](Specification.md#coarser-bars-and-outside-series)).

```python
daily = btc.resample(interval="1d")
trend = btc.asof(probatum.ta.sma(daily.close, length=50), when=daily)  # known at each day's end
oi = btc.asof(open_interest)                                 # by its own timestamps
```

`resample(interval)` gives the same instruments at a whole multiple of the interval. Each coarse bar takes the first open, the highest high, the lowest low and the last close of its bars and the sum of their volumes, and is available where any of them was; the funding events keep their stamps. Coarse bars align to 00:00 UTC, and weeks to Monday, moved onto the Data's own steps, so hourly bars that open at half past the hour give coarse bars that do too. A coarse bar the bars do not cover to its end is dropped, since its close would be a price still trading, and the first coarse bar holds the bars from the Data's first, wherever in it they begin. The instruments keep their names, and a slice's coarse bars keep the ones before it and read nothing after its end.

`asof(values, when)` puts on row t the last value known by the end of bar t, and NaN before the first. `when` is a Data, or anything a Data is built from, whose values are known at the ends of its bars, or the time each row of `values` became known. Where `values` holds a column for each of its instruments, an instrument that printed no bar on a row gave no new value there, so its last one stands.

A frame can be passed alone. A frame of bars is read as a Data is, each value known at its bar's end, and a column of a router candle frame at its candle's end. Any other frame or series is read at its datetime index, its one time column or a `timestamp` column of milliseconds, its numeric columns being the values. A column of bars from elsewhere carries nothing that says so, so it comes with `when=`, its bars as a Data.

A daily bar on this clock ends at the next midnight UTC, after the close of any session that ends by then, a stock's among them. The bars of a day that ends later, such as an FX day from 17:00 New York, belong in UTC or at a finer interval, where their ends are their own.

### Saving

A Data saves to a directory the user names and loads back under the same fingerprint ([0005](Decisions.md)).

```python
universe.save(path="data/thirty")                  # three Arrow files
universe = probatum.Data.load(path="data/thirty")  # the same bars, next session
```

`save(path)` makes the directory where it is missing and writes three Arrow IPC files that any Arrow reader opens:

| File | What it holds |
| :--- | :--- |
| `bars.arrow` | a row for each bar and instrument: its time, the instrument as its row of `instruments.arrow`, and the five values, null where the instrument printed no bar; the interval, the provenance and the fingerprint sit in its schema's metadata |
| `instruments.arrow` | a row for each instrument, with its name and every field of its `Instrument` but `settlement`, which follows from them |
| `funding.arrow` | a row for each funding event |

A slice saves the bars before it as well, which its features warm up on, and loads back as the same slice. Each file is written under another name and then put in place, the bars last. `probatum.Data.load(path)` reads them back, checks the bars as a Data checks its bars, and refuses them where they do not give the fingerprint they were saved under. Bars fetched once can then be studied many times, and a result's fingerprint can be checked against the files it came from. probatum writes and reads the files itself, so saving needs no pyarrow.

<br>

## Venue

A `probatum.Venue` holds a venue's rules for a run in one object, so a setting has one place to be given ([0002](Decisions.md)). The presets are Venues, each reached as `probatum.venues.NAME` and read when first asked for ([0028](Decisions.md)). A venue built by hand starts from the plain option in each row: a setting left out or given as `None` takes the plain option, and `None` inside a table leaves that key out.

```python
import probatum

perp = probatum.venues.bybit_perp
cheaper = perp.replace(fees={"taker": 0.0004}, max_open_orders=16)
print(cheaper.to_dict()["fees"])            # {'maker': 0.0002, 'taker': 0.0004}

both = probatum.venues.bybit.replace(balance={"spot": 5_000.0, "margin": 5_000.0})

mine = probatum.Venue(fees={"maker": 0.0, "taker": 0.0005},
                      funding={"every_bars": 8, "rate": 0.0001})
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

### Instruments' tables

Every instrument takes the venue's `tiers`, `tiers_by`, `tick`, `lot`, `sigfigs`, `decimals`, `min_notional`, `max_qty` and `reduce_only_exempt` where the venue gives them, unless its kind's table in `kinds` or its own table in `instruments` gives others, and its fees side by side over the venue's. `instruments` keys a table by its symbol where `kind` names the kind of every instrument the venue lists, and as `kind:SYMBOL` where `kind` is None. The presets that combine a venue's spot and perpetual markets are of that second kind, and keep their rates in each kind's table.

Only an instrument's own table gives its `delist`, the time the venue delisted it, as a time or the text of one in UTC, such as `"2024-01-02 06:00"`. On the first row that opens at or after it, its waiting orders are cancelled, its position is closed at its last close with no fee, and every later order on it is refused ([Specification](Specification.md#the-clock)). Each table is written as a preset file writes it: a tier table is a list of `{"up_to": n, "leverage": x, "rate": m, "amount": c}`, an amount left out being 0, and every word is in lowercase ([Contributor Guide](Contributor_Guide.md#adding-a-preset)).

### A Venue does not change

A Venue is hashable, and equal to any venue with the same settings, however their numbers were written. `to_dict()` gives every setting as the constructor takes it, so `probatum.Venue(**venue.to_dict())` is the same venue, and its repr is the code that builds it, such as `probatum.venues.bybit_perp.replace(max_open_orders=16)` or `probatum.Venue(fees=...)`.

`replace(**changes)` gives a variant. `fees` and `fills` merge into the venue's own. A table in `kinds` or `instruments` merges into the one it names key by key, its fees side by side, and a table given `None` is dropped, which the venue must hold. `liquidation="takeover"` drops a backstop the venue held, and every other setting is replaced whole. A change that every kind's table overrides, such as `fees` on a combined preset, would reach no instrument and is refused.

Every setting passes the checks a preset's pass as the venue is built, the engine's own among them, so a variant is checked as a new venue is. A refusal opens with the setting as it was passed, such as `fees.maker` or `instruments.BTCUSDT.tick`, and the value ([0007](Decisions.md)). `name`, `taken` and `kind` give the preset's name, the day its tables were taken and the kind of the instruments it lists, each None where it is not given, and `describe()` grades what a run on the venue assumes, read from the venue alone ([Costs, assumptions and baselines](#costs-assumptions-and-baselines)).

### A Data runs on a venue as one account

Each instrument takes the venue's tables by its symbol, or as `kind:SYMBOL` on a venue of several kinds, a market the Data leaves out being the venue's. It also takes the minimum notional and largest size the router gave it, wherever no table of the venue gives one. The run reads the bars before a slice, which its window warms up on, leaves out the rows none of its instruments printed, as a selection from a wider universe holds, and reads the Data's funding events where the venue's funding is `"data"`. Refused before the run starts are:
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

## Engine

`probatum.Engine(data, venue, seed=None)` is an account at `venue`, a Venue or a preset's name, on `data`, a Data or anything one is built from, for a loop the caller owns. It is the fourth depth beside an array, a Strategy and an agent, and records the same Result ([0002](Decisions.md)).

```python
engine = probatum.Engine(data, venue="bybit_perp")

bar = engine.reset()
while not engine.done():
    decision = service.decide(bar, engine=engine)  # anything that returns weights or None
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

Between steps the engine is the account a Strategy's `next` reads, with the same state and the same verbs ([A strategy](#a-strategy)), and its `events` hold what happened to its orders since the last step. `seed` names the stream that a venue whose `intrabar` is `"random"` draws from; `backtest` refuses such a venue, so that a backtest repeats without a seed ([Specification](Specification.md#what-a-bar-cannot-show)).

`engine.clone()` copies the account as it stands, with handles of its own for the orders placed so far, and steps on as the original would. An engine also pickles, to move into another process of the same version ([Specification](Specification.md#clones-and-pickles)). The compiled engine every run steps, built over candles with keyword settings, is `probatum._probatum.Engine` ([below](#the-compiled-engine)).

<br>

## Batch

Many independent envs over one shared candle series, stepped in parallel with the GIL released. Every env sees the same candles, shared rather than copied, but carries its own account and orders.

```python
import numpy
from probatum import Batch

b = Batch(candles, num_envs=512, market="spot", quote=10_000.0)  # the compiled engine's settings

b.reset_all()                                # list of per-env states

actions = numpy.zeros(shape=b.num_envs)      # one signed size per env
states = b.step_all(actions)                 # positive buys, negative sells, zero holds
```

| Call | Returns | Does |
| :--- | :--- | :--- |
| `reset_all()` | list of states | Reset every env, in parallel. |
| `step_all(actions=None)` | list of states | Apply the per-env signed-size actions (market orders) and step every env in parallel with the GIL released. |
| `target(weights, tolerance=0.0, transfer=False)` | `None` | Bring every env to its row of an `(num_envs, N)` array of weights of equity, refusing a non-finite weight, or a negative one on spot, before any env places anything; each leg fills at its instrument's next open, or waits for the next bar it prints. |
| `records(env)` | dict | What env `env` recorded, as the binding's `run` returns it ([Specification](Specification.md#the-low-level-binding)): its ledger, fill log, order log and round trips, each `None` without `report=True`. |
| `done()` | bool | True when the envs have reached the last bar (they step in lockstep here). |
| `clone(env=None, n=None)` | a `Batch` | A copy of every env, or with `env` a batch of `n` copies of that env, each stepping on as it would; a Batch also pickles, to move into another process of the same version. |
| `num_envs` | int | Number of envs (also `len(batch)`). |

Because the envs are independent, stepping them as a batch is bit-identical to stepping each in a loop, except under `intrabar="random"`, where env k also draws from a stream named with its index, which an engine built on its own does not ([Specification](Specification.md#what-a-bar-cannot-show)). The `Batch` also carries a lower-level RL surface for a loop of your own: random-start reset, masked autoreset, batched feature observation, and per-env equity, done and bust arrays. [`probatum.env`](#reinforcement-learning) builds its environments over the same engines and steps them itself. `probatum.rl.Batch` is the same class.

Each cost setting also takes an optional per-env override. Pass `fee_taker_per_env`, `fee_maker_per_env`, `slippage_bps_per_env` or `impact_per_env` as a float64 array of length `num_envs`, and each env carries its own cost while sharing the one candle buffer. A fee override replaces the venue's rate and leaves alone a rate an instrument charges of its own. An environment's `costs` draws into these overrides, every instrument's own rate set aside for a fee it draws ([0038](Decisions.md)), and an override whose length is not `num_envs` is a `ValueError`.

**A batch over a universe** takes `(T, N, 5)` candles and the same `instruments` and `funding_instruments` as [the engine](#the-compiled-engine), and steps each env through `target`; the signed sizes `step_all` and `advance` take trade the first instrument. Its `observe(window)` is `(num_envs, window, N, 5)`, zero where an instrument printed no bar. `snapshot()` adds every instrument's `positions` and `marks` as `(num_envs, N)` arrays beside its `(num_envs,)` fields, whose position, unrealized PnL and mark read the first instrument.

<br>

## Reinforcement learning

`probatum.env(data, venue, window=64, features=None, layout="flat", max_leverage=1.0, actions="box", max_offset=20, decode=None, reward="log", episode_length=None, sample=None, costs=None, normalise=False, fill=None, tolerance=0.0, num_envs=1, threads=None, autoreset="next_step", copy=True, record="none", seed=None)` steps `num_envs` accounts at `venue` on `data` together in Rust and returns a Gymnasium vector environment, a `probatum.rl.VectorEnv`. Each account is a backtest's, with the same fills, costs, funding, caps and liquidation. Each environment draws its episodes from a stream of its own, so an episode runs again as a backtest from the start its info gives ([Specification](Specification.md#the-environment), [0038](Decisions.md)). The [RL Guide](RL_Guide.md) gives every setting, the observation, the actions, the rewards, the episodes and the autoreset modes, with examples.

| Name | What it is |
| :--- | :--- |
| `probatum.rl.VectorEnv` | the environment `probatum.env` returns, with `describe()`, `provenance`, `agent_spec`, `clone()`, `transitions` and `export()` |
| `probatum.rl.Env` | one environment, stepped as Gymnasium's single API steps it; `gymnasium.make` and `gymnasium.make_vec` build both as `probatum/Trading-v0` once `probatum.rl` or `probatum.env` has loaded |
| `probatum.sb3.wrap(env)` | the environment as a Stable-Baselines3 `VecEnv`, a `ProbatumVecEnv` carrying its agent spec on a copy of its action space ([RL Guide](RL_Guide.md#training)) |
| `probatum.sb3.SpecCallback` | copies `VecNormalize`'s statistics into the model's spec after each rollout |
| `probatum.torch.TensorEnv(env, device="cpu", resident=False)` | the observation, rewards and flags handed out as torch tensors on a device, each the numpy path's to the bit ([RL Guide](RL_Guide.md#tensors-and-the-gpu)) |
| `probatum.save(agent, path, spec=None)` | a trained agent written beside the spec it trained with ([An agent](#an-agent)) |

A plain `import probatum` never loads Gymnasium, which the `rl` extra installs.

<br>

## Backtesting

`probatum.backtest(policy, data, venue, sizing="weight", stop=None, take_profit=None, tolerance=0.0, record="full", risk_free=0.0, spec=None, features=None, decode=None, instruments=None)` runs a policy on a Data at a venue and returns its `Result`. `data` is a `probatum.Data` or anything one is built from, and `venue` a `probatum.Venue` or a preset's name. The policy is read in one fixed order, so no input is read two ways ([0024](Decisions.md)):

| Order | Policy | How it runs |
| :--- | :--- | :--- |
| 1 | a table: anything exporting the Arrow C stream interface, or a pandas frame or series | its rows matched to the Data's clock by their times, in a datetime index or column or a `timestamp` column of milliseconds, and a daily Data's by their dates, and its columns to the instruments by name, then run as an array; a time or a name that does not match is refused, and a row may be left out only where no instrument printed a bar |
| 2 | an array: anything with `__dlpack__`, the buffer protocol, `__array__` or `__arrow_c_array__` | converted once to float64 on the host and run whole in Rust |
| 3 | a class | refused: pass an instance |
| 4 | a `Strategy` | a deep copy of it, `init(data)` once and `next(bar, account)` at each close ([below](#a-strategy)) |
| 5 | an object with `fit` that is not callable | refused: pass its predictions as an array |
| 6 | an agent: a Stable-Baselines3 model or policy, a torch module, each recognised only where its framework is already imported, or any other callable | one environment of its spec, asked at each close for its action ([below](#an-agent)) |
| 7 | anything else | refused |

**An array** is `(T, N)` over the Data's rows and instruments, or `(T,)` for one instrument. Row t is the decision at the close of bar t, and a NaN holds. The engine still steps every bar, so an array run is no vectorized backtest. With `sizing="weight"` each value is a weight of equity, reached as `target` reaches it, and a leg below `tolerance` of equity is skipped. With `sizing="quantity"` each value is the position wanted, in base units, or in contracts on an inverse contract. The last row is never read, since no bar follows it to fill on.

An array of booleans is refused, since False would flatten where it means to hold: 1.0 enters, 0.0 leaves and NaN holds. A negative value on a spot instrument is refused before the run, and so is a venue that decides at random what a bar cannot show, so that a backtest repeats without a seed ([Specification](Specification.md#an-array)).

```python
fast = probatum.ta.ema(btc.close, length=20)
slow = probatum.ta.ema(btc.close, length=80)

weights = numpy.full(shape=len(btc), fill_value=numpy.nan)    # NaN holds the position
weights[probatum.ta.crossover(fast, other=slow)] = 1.0        # enter long
weights[probatum.ta.crossunder(fast, other=slow)] = 0.0       # flatten

venue = probatum.venues.bybit_perp.replace(funding="none")    # btc holds no funding
result = probatum.backtest(weights, data=btc, venue=venue, stop=0.04, take_profit=0.08)
```

**Barriers.** `stop` and `take_profit` give each round trip of an array run a stop and a take-profit at those returns from its average entry. Each is a number, or a `(T, N)` array or a table of them read on the row whose decision opened the round trip, NaN for none. They rest from the close of the bar the entry filled on and move when an add moves the average entry. A barrier that fills, even in part, holds its instrument flat until a row asks for zero or for the other side, so the rows still asking for the position do not buy it straight back ([Specification](Specification.md#barriers-on-an-array)).

**The Result** holds what the run recorded: its equity and returns, its tables, its statistics, the policy it ran and its provenance ([The Result](#the-result)).

**The recording level.** `record="account"` keeps the ledger, the round trips, the costs and the statistics of the run at the full level, to the bit, with its assumptions priced where its ledger and round trips measure them. It keeps none of the holdings, fill log, order log, baselines or policy, for a run read for its numbers alone ([Specification](Specification.md#recording-levels)). A Strategy still runs at the full level, since its account reads the order log, and keeps the account level's records when it ends. `"transitions"`, for an agent alone, keeps the account level's records and `result.transitions`, a row for each close the agent was asked at until its episode ended, which `result.export("arrow", path)` writes ([Specification](Specification.md#transitions)). `"none"`, which would keep nothing a Result holds, is refused.

```python
light = probatum.backtest(weights, data=btc, venue=venue, stop=0.04, take_profit=0.08,
                          record="account")

print(light)                                    # names the level it was recorded at
print(light.stats == result.stats, light.trades == result.trades)
```

### A strategy

A `probatum.Strategy` decides at each close ([Specification](Specification.md#a-strategy), [0025](Decisions.md)):
* `init(data)` runs once over the whole Data before the first bar, the place to compute indicators over every bar at once.
* `next(bar, account)` runs at the close of every bar from `warmup` on, counted in the Data's rows, except the last, which no bar follows to fill on, and except a row none of the instruments printed. What `next` places fills on a later bar, never on that one.
* `warmup` is a class attribute or is set in `init`. It should cover the longest history a decision reads, since an index below zero reads numpy's array from its end.
* A strategy's parameters are its constructor's arguments, kept as attributes of the same names.
* At a validation's seam, `on_inherit(bar, account)` is called before the incoming window's first decision, once every order the window before it left is cancelled, so a strategy whose exits rest on the book can place them again for the position it inherits ([Validation](#validation)).
* `backtest` runs a deep copy of it, so two runs of one instance start alike, and the copy, holding what `init` computed, is the Result's `policy`.
* `stop`, `take_profit`, `sizing` and `tolerance` belong to an array run and are refused beside a Strategy, which places its own exits and gives `target` its own tolerance.

```python
class Breakout(probatum.Strategy):
    warmup = 21

    def init(self, data):
        self.high = probatum.ta.highest(data.high, length=20)
        self.atr = probatum.ta.atr(data.high, low=data.low, close=data.close, length=14)

    def next(self, bar, account):
        atr = self.atr[bar.i]
        if account.position == 0.0 and bar.close > self.high[bar.i - 1]:
            account.buy(size=account.equity * 0.01 / (2.0 * atr))
            self.exits = account.bracket(stop=bar.close - 2.0 * atr,
                                         take_profit=bar.close + 4.0 * atr)
        elif account.position > 0.0:
            trail = bar.close - 2.0 * atr
            if trail > self.exits.stop.trigger:
                account.replace(self.exits.stop, trigger=trail)


result = probatum.backtest(Breakout(), data=btc, venue=venue)
```

**The bar** is the bar just closed:

| Field | What it is |
| :--- | :--- |
| `i` | its row of the Data, by which an array computed in `init` is read |
| `time`, `ns` | its open, as a `datetime` in UTC and as int64 nanoseconds since 1970 |
| `n` | the number of instruments |
| `open`, `high`, `low`, `close`, `volume` | with one instrument, its values; with any number, the plural forms, each `(N,)` and NaN where an instrument printed no bar |
| `funding_rate` | the last rate known at the bar's end, NaN before the first, plural with several instruments |
| `available` | which instruments printed it |
| `bar["ETHUSDT"]` | one instrument's bar |
| `history(n)` | the last `n` bars up to and including this one, as a read-only Data that includes the bars before a slice's first; `history()` gives every bar so far |

With several instruments the singular forms are refused with the fix, as a Data's are. `bar.history` is the only window onto earlier bars, so logic computed in `next` cannot read a later bar.

**The account** is the one the bar's close left, in the venue's terms:

| Field | What it is |
| :--- | :--- |
| `equity`, `leverage` | the account's equity and leverage |
| `position`, `entry` | with one instrument, its position and average entry; `positions`, `(N,)`, with any number |
| `orders` | the orders still open, resting on the book or pending until the next open |
| `events` | what happened to its orders since the last close, oldest first, the orders placed at that close among them |
| `trades` | the round trips closed so far, each [column](#trades) a read-only array |
| `equity_curve` | the equity at the opening and at every bar since, one a row its instruments printed, as the Result's `equity` counts them |
| `spot`, `margin` | the two wallets, each with its `cash` and `equity`; the margin wallet also has its maintenance `requirement` and `available`, its equity less the margin the cap asks, floored at zero |
| `account["ETHUSDT"]` | one instrument's view, with its `position`, `entry`, `upnl`, `weight` and `orders` and the verbs that place an order on it |

`trades` and `equity_curve` both start again at each seam of a validation's path. With several instruments the order verbs are taken through an instrument's view and refused on the whole account with the fix; `target`, `transfer` and `cancel_all` act on the whole account. Every bar the account names is the Data's row, as `bar.i` is.

| Verb | Places |
| :--- | :--- |
| `buy(size)`, `sell(size)` | a market order of `size` base units, or contracts on an inverse contract, for the next open |
| `buy(size, limit=price)`, `sell(size, limit=price)` | a limit order at `price` |
| `buy(size, stop=trigger)`, `sell(size, stop=trigger)` | a stop that triggers at `trigger` and fills as a taker |
| `buy_value(amount)`, `sell_value(amount)` | a market order for `amount` of the quote currency, its size the amount over its fill price ([Specification](Specification.md#units)) |
| `target(weights, tolerance=0.0, transfer=False)` | the legs that bring each instrument to its weight of equity, `weights` one number with one instrument, one for each in order, or a dict from name to weight ([0009](Decisions.md)) |
| `bracket(stop=None, take_profit=None)` | an OCO pair of reduce-only exits with no size of their own, at those prices |
| `close()` | a reduce-only market order that flattens the position, or None where it is flat |
| `cancel(order)`, `cancel_all()` | the cancellation of an order still open, resting or waiting for its open, or of every one |
| `replace(order, price=None, trigger=None, size=None)` | the same order moved under its id, a limit by its `price` and a stop by its `trigger`, or None where it no longer rests or cannot move |
| `transfer(amount, to="margin")` | cash moved between the wallets at the start of the next bar ([0008](Decisions.md)) |

`buy` and `sell` also take `tif`, `"gtc"`, `"ioc"` or `"fok"`, on a market or a limit order, `post_only` on a limit and `reduce_only` on any order. Their value forms take `tif`, `"ioc"` or `"fok"`, and `reduce_only`.

**Every placing verb returns an `Order`**, a live handle that shows the order as it stands on any later bar, refused orders included, with their reason ([0010](Decisions.md)):

| Field | What it is |
| :--- | :--- |
| `id`, `instrument` | the order and its instrument |
| `side` | `buy` or `sell`; None for a bracket's leg until it arms |
| `kind` | `market`, `limit` or `stop` |
| `size` | what it asks; None for a bracket's leg that closes the position it finds, and for a value order, whose `filled` is what its amount bought |
| `filled`, `price`, `trigger` | what has filled, and its prices |
| `placed` | the Data's row it was placed at |
| `status` | `pending` while a market order waits for the next open, `resting` while a limit or a stop is on the book, then `filled`, `cancelled`, `expired` once its time in force ends what is left of it, or `refused` |
| `reason` | what the order log said last of it |

`bracket` returns both legs as `.stop` and `.take_profit`, and `target` the list of its legs, refused ones included. Each of the account's events is an `Event` with its `kind` (`placed`, `armed`, `filled`, `clamped`, `refused`, `not_filled`, `expired` or `cancelled`), its `order`, None for a transfer's clamp, its `instrument` and `bar`, its `side`, `size` and `price` as the order log gives them, and its `reason`.

### An agent

A trained agent runs as one environment of the spec it trained with, on the Data's bars at the venue from the Data's first bar to its last, and records as a run does ([Specification](Specification.md#an-agent-in-a-backtest), [0039](Decisions.md)). An SB3 model or policy is asked for its deterministic prediction, carrying a recurrent policy's state. A torch module is called on the observation as a tensor on its device, in eval mode with no gradient. Any other function is called on the observation, and the first element of a tuple it returns is its action.

The window reads the bars before a slice, so with the candles or a function of features an agent decides at the slice's first close; with an array of features over the slice alone it decides once a window of them is full. Once the account holds nothing, the agent is no longer asked.

```python
result = probatum.backtest(agent, data=test, venue="bybit_perp", features=features)
again = probatum.backtest(result.policy, data=test, venue="bybit_perp")  # the same equity
```

`spec` is the agent spec: a path to its JSON, a directory holding `spec.json`, or the environment the agent trained in, and an SB3 agent's own where it is left out. A feature function or a decode the agent trained with is passed again as `features` or `decode` and held to its probe, so a changed function is refused as a `probatum.SpecError`. So is a Data of other instruments, unless `instruments` names the slots in the agent's order, and a slot on another market than the agent trained on. An array of features is over the Data given.

`result.policy` holds what the agent decided: each close's weights, or each slot's size and offset for orders as placed, NaN where nothing was decided. Its frictionless baseline is the agent run again without costs at once, since an agent cannot be copied to be played later. A run on bars that overlap the training span, at another venue or on another interval warns, and the provenance records the spec, the overlap and whether the venue is the one it trained at. `stop`, `take_profit`, `sizing` and `tolerance` are refused beside an agent, whose spec says how its actions trade, and an agent's settings are refused beside any other policy.

`probatum.save(agent, path, spec=None)` writes an agent into the directory `path` beside its spec as `spec.json`: a torch module as its state dict, `weights.pt`, and an SB3 model as its own zip, `agent.zip`, so no module is pickled. `spec` is read as `backtest` reads it.

```python
probatum.save(policy, path="agents/ppo-1", spec=env)           # weights.pt and spec.json
policy.load_state_dict(torch.load(f="agents/ppo-1/weights.pt"))
result = probatum.backtest(policy, data=test, venue="bybit_perp", spec="agents/ppo-1")
```

### The compiled engine

Every run steps the compiled engine, `probatum._probatum.Engine`, which is private and promised nothing ([Architecture](Architecture.md#versioning-policy)). It is built over a candle array with the engine's settings as keywords, then reset, given orders, stepped and read a bar at a time. The `Batch` and the RL environment take the same settings.

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

**A tier table caps a perp by the tier its position falls in**, in place of the one `leverage`. Up to each bound of notional, or of size with `tiers_by="quantity"`, the position may be levered no further than its tier allows, and a fill is held at the table's last bound, unless that bound is `float("inf")`, left open as by a venue that sets no largest position. The table is refused where the Specification's loader refuses it ([Specification](Specification.md#maintenance)): bounds that do not rise; a leverage that rises or a maintenance rate that falls from one tier to the next; a maintenance rate not below one, or reaching its tier's initial rate of one over the leverage; a notional table whose first amount is not zero or whose amounts break continuity; and any value that is not finite, a last bound left open aside, or that is below zero.

**A venue's sizes hold every order to its instrument's grid** ([Specification](Specification.md#sizes-and-dust)). With `sizes=True`:
* A size is rounded down to the `lot` when it is placed, and so is every fill after its clamps, and an order that rounds to no lot is refused.
* A limit or a trigger is rounded to the `tick`, or to `sigfigs` significant figures within `decimals` decimals, Hyperliquid's rule, where a whole-number price is always allowed. On Hyperliquid an asset with `n` size decimals takes a lot of `10**-n`, and `decimals` of `6 - n` on a perpetual and `8 - n` on spot.
* An order below `min_notional` at the last close is refused, a reduce-only one spared where `reduce_only_exempt` says the venue spares it, and one above `max_qty` is held to it, rounded down to the lot.
* A position that floating point leaves below half a lot is zeroed at the close as dust.

On an inverse contract the lot and `max_qty` are in contracts, the tick in dollars and `min_notional` in the coin. `limit="through"` fills a limit only once its bar trades half a step of the price rule past it, and with no price rule it fills on touch.

**A universe of N instruments shares one account.** Pass `(T, N, 5)` candles, NaN in all five values where an instrument printed no bar, with `instruments`, a dict per instrument. Each gives its `market`, `tiers`, `tiers_by` and size rules, its own `fee_maker`, `fee_taker` or both where it charges apart from the venue, a side it leaves out charging the venue's rate, and its `delist` time in nanoseconds since 1970 where the venue delisted it, which needs `times=` and comes after the open of its last bar. Each instrument fills on its own bar and waits where it printed none, and `target` brings each to its weight of equity. The views carry an instrument axis: `shape` is `(T, N, 5)`, `data` and `observation` are `(T, N, 5)` and `(rows, N, 5)`, and `closes` and its companions `(T, N)`. The order shortcuts trade the first instrument, and the state `step` returns reads its position and bar beside the account.

**An inverse contract is counted in its coin.** `market="inverse"` with `contract=`, the dollars one contract is worth, takes one instrument's `(T, 5)` candles, quoted in dollars. Sizes are contracts and prices are dollars, as the venue quotes them, while `quote`, equity, fees, funding and every other sum are in the coin, a taker's costs among them, each its rate of the coin notional ([Specification](Specification.md#inverse-contracts)). The account holds that one contract in one wallet, so there is no `transfer`, `qty_from_quote(cash)` is `cash / contract` contracts at any price, and the ledger's `equity_in_quote` values equity in dollars at each close ([0016](Decisions.md)).

**A venue that books then backstops sells a closed-out account on the book** ([Specification](Specification.md#the-forced-close), [0019](Decisions.md)). `liquidation="book_then_backstop"` takes `backstop=`, the share of the requirement at which the venue's backstop takes over, two thirds on Hyperliquid. It sells every leg as a taker from its trigger price, paying the taker fee and a taker's costs, and the trader keeps what the margin wallet then holds. The account is taken over instead where the sale would leave the wallet below the backstop level, or where the path also reaches that level and the bar cannot show that the sale finished first, which `intrabar` decides. The default `"takeover"` closes every leg where the margin ran out and books what the wallet still holds as the liquidation charge. Book then backstop needs a perp whose tier table keeps maintenance, since with none the backstop level is the trigger itself.

**A venue's liquidation fee places the close point of an account with no maintenance** ([Specification](Specification.md#the-trigger-close-and-clip-points)). The close point is where equity would just pay the fee on closing every position, at `liquidation_fee` where it is given and each instrument's taker rate where it is not. An account with no maintenance whose path reaches its trigger leaves the market from the point where its equity stays at or below that level all the way to the trigger, and its takeover charges the fee on each close. The margin bound charges it on what a reducing fill leaves of the position and on the other perps, wherever the account the fill leaves keeps no maintenance. With maintenance the rate is not used: the account leaves at its trigger, its takeover pays no fee of its own, and the bound charges the taker rate. The fee is refused below 0 and from 1.

`eng.shape` is `(T, 5)` for one instrument.

**The account keeps two wallets**, as a venue does ([0008](Decisions.md)). The spot wallet holds cash and spot holdings, never borrows and is never liquidated. The margin wallet holds the collateral behind a perp, and the cap, the liquidation and funding read it alone. A number for `quote` opens the wallet the market settles in, the spot wallet on spot and the margin wallet on a perp, and `{"spot": s, "margin": m}` opens each by name. Cash crosses only by `transfer`, which out of the margin wallet leaves unrealized profit and the cap's requirement behind. Equity is both wallets together, so a perp closed out with cash in the spot wallet leaves an account that is still alive.

### Stepping

| Call | Returns | Does |
| :--- | :--- | :--- |
| `reset()` | state | Reset the account and cursor to the first bar. |
| `step()` | state | Advance one bar, resolve the orders placed since the last step, mark, return the new state. |
| `done()` | bool | True when there is no next bar to step into. |

### Orders

| Call | Returns | Does |
| :--- | :--- | :--- |
| `market_buy(size)` / `market_sell(size)` | id or `None` | Taker order; fills at the next bar's open, pays the taker fee, half the spread, the slippage and the impact. `None` when the queue already holds `max_open_orders` for this bar, the size is not a positive finite number, or the venue's sizes refuse it. |
| `limit_buy(size, price)` / `limit_sell(size, price)` | id or `None` | Limit order; rests until a later bar reaches the price, and pays the maker fee unless its first bar opens at or through it. `None` if the book is full, the size is not a positive finite number, it would trade against the account's own resting limit, or the venue's sizes refuse it; a non-finite price raises. |
| `stop(side, size, trigger, reduce_only=False)` | id or `None` | Becomes a market order once a bar crosses `trigger`. `side` is `"buy"` or `"sell"`. `None` if the book is full, the size is not a positive finite number, the trigger is not finite, or the venue's sizes refuse it. Pass `reduce_only=True` for a stop-loss. |
| `bracket(stop=None, take_profit=None)` | two ids or `None`s | A stop and a take-profit as one OCO pair, each a reduce-only leg with no size of its own that closes the position it finds. A leg takes its side from the position it arms on, at once when one is held and otherwise at the first fill that opens one, so a leg in a later slot can close it on that bar; the first fill of either cancels the other, and the bracket goes when that position closes or flips. `None` for a leg not asked for or refused; a non-finite price raises ([0010](Decisions.md)). |
| `order(side, size, type, price, trigger, reduce_only, post_only, tif)` | id or `None` | The primitive the shortcuts wrap; the only call that sets `post_only` and `tif`. |
| `close()` | id or `None` | Queue a reduce-only market order sized to the whole position, held to `max_qty` where the venue's sizes are on. `None` when flat or where those sizes refuse it. |
| `cancel(order_id)` | bool | Drop an order not yet filled, resting or waiting for its open. True if it was found. |
| `cancel_all()` | int | Drop every order not yet filled, those waiting for their open first; returns how many were dropped. |
| `transfer(amount, to)` | `None` | Move cash into the `"spot"` or `"margin"` wallet from the other. It lands at the start of the next bar, before any fill, held to what the wallet it leaves can give ([0008](Decisions.md)). |
| `target(weights, tolerance=0.0, transfer=False)` | list of ids | Queue the market orders that bring each instrument to a weight of equity at the last close, every reducing leg before every growing one. `weights` is a number with one instrument, one weight per instrument, or a dict from instrument to weight; a leg below `tolerance` of equity is skipped, and `transfer=True` moves the spot legs' shortfall across from the margin wallet. A second target in one decision replaces the first, and a later one replaces the legs earlier ones left waiting on what it names ([0009](Decisions.md)). |
| `replace(order_id, size=, price=, trigger=)` | id or `None` | Move a resting order under the same id, with the same side, type and flags, into the first free slot. `None`, and nothing changed, if it is no longer resting or the moved order cannot rest ([0010](Decisions.md)). |
| `is_bust()` | bool | True when the account died: its equity, both wallets together, reached zero, which a takeover does unless the spot wallet holds cash ([0008](Decisions.md)). The engine does not stop on it. |
| `num_fills()` | int | Fills applied since the reset. Zero beside orders you placed means none of them filled. |
| `qty_from_weight(fraction)` | float | Base size for `fraction` of current equity, at the current close. |
| `qty_from_quote(cash)` | float | Base size for a `cash` amount in quote, at the current close. |

`order(...)` is the full primitive ([0010](Decisions.md)). `type` is `"market"`, `"limit"` or `"stop"`, and `tif` is `"GTC"`, `"IOC"` or `"FOK"`. It carries the three flags the shortcuts leave at their defaults:
* `reduce_only` marks a take-profit or a stop-loss that can only shrink the position.
* `post_only` refuses a limit that would cross the last close, and cancels one that its first bar, or the part of the bar after a fill, opens at or through, rather than turning it taker.
* `tif` decides what happens to the part of an order the next bar does not fill: `GTC` rests, `IOC` takes one bar and cancels the remainder, and `FOK` fills the whole size against the bar or nothing.

A market order is `IOC` unless you ask for `FOK`, since it never rests and so cannot tell `GTC` from `IOC`, and a stop rests until it triggers. `FOK` is all or nothing against every clamp, not only the bar's liquidity, so a `FOK` that the spot cash balance or the margin cap could fill only in part books nothing. A limit needs a `price` and a stop a `trigger`, else it is a `ValueError`, and a non-finite price or trigger is a `ValueError` too ([0007](Decisions.md)).

**A bar that reaches both a stop and a take-profit books the stop** ([0012](Decisions.md)). The engine cannot see the path inside a bar, so it cannot know which the price touched first. Exits from one position are applied worst first, so the result does not depend on the order the strategy placed them in, and a bar that reaches only one leg is unaffected. A reduce-only exit, or a sale on spot, of a position that a resting order opened or flipped inside the bar meets the bar from that fill: under the default, a take-profit the bar may have reached only before the entry rests unless the close reaches it, and under either setting a stop fills where the price crossed it after the entry.

With `intrabar="optimistic"` the better exit fills first instead, and the setting decides every other event a bar cannot order the same way. The gap between a run under each shows how much a result leans on what the bars cannot show, though the two are not a strict bound ([Specification](Specification.md#the-error-and-its-direction)). `intrabar="random"` draws each event one way or the other from a stream named from `seed`, which it needs, so a seed draws the same way on every machine. It is for training, where an agent would otherwise learn the fixed tie-break and dodge it.

**Re-placing a trailing stop with `stop()` each bar rests a new order every time**, since nothing but `bracket` links resting orders into an OCO group. The one that fills leaves its siblings live, and on a perp those open a position on the other side. Use `replace`, which cannot leave two alive and does nothing once the order has filled, and pass `reduce_only=True` as well, so even a leaked stop can only shrink the position ([0010](Decisions.md)). `close()` queues an ordinary market order, so it is `IOC`: on a thin bar the budget can fill only part of it, and the remainder is cancelled rather than carried, which leaves a residual position. Check `state["position"]` rather than assuming.

```python
#  a trailing stop that stays one order and can only ever shrink the position, inside
#  a loop that set stop_id and trigger to None before its first step
if state["position"] > 0.0:
    wanted = state["bar_close"] * 0.94
    if stop_id is None:
        stop_id = engine.stop(side="sell", size=state["position"], trigger=wanted,
                              reduce_only=True)
    elif wanted > trigger:
        #  None here means the stop already filled, so nothing is armed again
        stop_id = engine.replace(stop_id, trigger=wanted)
    trigger = wanted
```

<br>

<div align="center">
  <img src="imgs/205316.png" alt="No same-bar lookahead" width="70%" />
  <p style="margin: 0;"><i>An order decided at the close of bar t is resolved against bar t+1, never the bar the decision saw.</i></p>
</div>

<br>

**How each order fills:**
* A market order crosses half the spread and fills at the next open, moved by its slippage and impact.
* A limit rests at your price and fills only when a candle reaches it. A price that gaps through your limit still fills at the limit, never better, unless no trade since the last close reached it, as when it was placed through the market; then a buy fills at the highest price the market traded and a sell at the lowest ([Specification](Specification.md#limits)).
* A limit its first bar opens at or through is marketable, so it pays the taker fee even though it fills at your price. One that the bar has to come to, or that has rested through a bar, is on the book and pays the maker fee ([0011](Decisions.md)).
* A `post_only` limit that would cross the last close is refused, and one its first bar opens at or through is cancelled rather than turned into a taker.
* A limit that reaches the account's own resting limit of the other side on its instrument, a buy at or above its own sell or a sell at or below its own buy, is refused, as a venue refuses a trade against the same account ([0011](Decisions.md)).
* Each bar holds `budget` of its instrument's trailing volume for every order on it, taken in resolution order, so splitting an order takes no more of a bar than one order would ([0013](Decisions.md)).
* Resting orders and the market orders waiting for the next bar each use `max_open_orders` slots, and beyond either cap a new order is rejected. The market queue empties every bar.

The position is netted: net long, net short or flat, never both. Adding on the same side averages the entries, reducing or closing books the realized PnL on the part closed, and flipping through zero closes the old side and reopens the rest at the new fill. On spot the position cannot go below zero: a sell is clamped to the current long, since shorting base needs a borrow the engine does not model, so a short and a flip through zero happen only on a perp ([0008](Decisions.md)).

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

Each order in `open_orders` carries `id`, `side`, `kind` (`market`, `limit`, `stop`), `price`, `trigger`, `size`, `filled`, `remaining`, `status` (always `resting` here, since these are the resting orders), `reduce_only` and `post_only`. A bracket's leg has `side` `None` until it arms, and `size` and `remaining` `None` while it closes the position it finds.

### Observations

`observation(lookback)` returns a read-only numpy `(rows, 5)` view of the `lookback` bars ending at the current bar, straight onto the shared candle buffer with no copy; `rows` is `lookback`, or fewer early in the series ([0040](Decisions.md)). Because the candle buffer never changes, a view held across a `step()` stays a snapshot of that bar's window rather than going stale. It stays valid for the whole life of the engine, since numpy keeps the engine alive as the array's base, so it never needs copying.

`eng.data` is the same kind of read-only, zero-copy view over the whole `(T, 5)` series, its columns OHLCV by position. `eng.opens`, `eng.highs`, `eng.lows`, `eng.closes` and `eng.volumes` are the same view one field at a time, so a loop reads `eng.closes` rather than remembering that the close is column 3. They are strided views into the same buffer, so taking all five costs nothing.

### Results

With `report=True`, seven accessors read the recorded run, each returning `None` when reporting is off:

- `stats(periods_per_year=365.0, risk_free=0.0)` returns the [stats dict](#statistics).
- `equity_curve()` returns a numpy array, one equity value per step.
- `trades()` returns a list of [round trips](#trades), the one still open at the end marked and flagged.
- `ledger()` returns [the ledger](#the-ledger), a dict of numpy columns with one row per bar from the opening balance.
- `fills()` returns a list of [fill dicts](#fills), one per fill in the order they landed.
- `order_log()` returns a list of [order events](#the-order-log), every order's life in the order it happened.
- `tables(start, names)` returns the run's ledger, holdings, fill log, order log and round trips as a dict of the binding's tables, which a [Result](#the-tables) wraps, each read through the Arrow C stream or by `column(name)` and `row(i)`. Every bar is counted from `start` and each instrument named by its entry in `names`, and it needs an engine built with `times=`.

An engine built with `record="account"` keeps less: `fills()` and `order_log()` return `None` with reporting on, `ledger()` has none of the columns for each instrument, and `tables()` gives the ledger and the round trips alone ([Specification](Specification.md#recording-levels)).

`periods_per_year` annualizes to your bar interval: 525600 for 1m, 8760 for 1h, 365 for 1d. The engine's `stats` is the raw call and takes it as a number, so state it. An engine built with `times=` gives its own clock's count from `periods_per_year()`, which is what a `Result` reads off its Data's clock ([0004](Decisions.md)).

### Statistics

A Result's `stats` is a `probatum.stats.Statistics`, and the compiled engine's `stats()` a plain dict of the same keys. A `Statistics` is a read-only Mapping of these keys to their values, each key an attribute too, so `stats["sharpe"]`, `stats.sharpe`, `dict(stats)` and `==` with a dict all work. `stats.interval(name, level=0.95)` is a statistic's [interval](#inference); it is None for exposure, the counts and funding paid, and where the sample cannot support one.

| Key | Unit | Meaning |
| :--- | :--- | :--- |
| `total_return_pct`, `net_profit_pct`, `cagr_pct` | percent | Total return (net of fees; both names give the same figure) and annualized return. |
| `sharpe`, `sortino`, `calmar` | ratio | Risk-adjusted return; annualized. Each is `inf` when it earned something against no measured risk at all, so the set stays orderable ([0030](Decisions.md)). |
| `max_drawdown_pct`, `volatility_pct`, `exposure_pct` | percent | Largest peak-to-trough decline; annualized volatility; fraction of steps holding a position. |
| `win_rate` | fraction | Fraction of closed trades whose PnL net of the recorded fee is positive. |
| `profit_factor` | ratio | Net profit over net loss, both after fees (`inf` with no losses). |
| `num_trades` | count | Closed round trips. A position still open at the end is in the return and the exposure but in none of these four. |
| `avg_trade_pct` | percent | Mean net trade PnL as a percent of **starting** equity, not of the equity the trade opened with. |
| `num_fills` | count | Fills applied over the run. Zero beside orders you placed means the feed never filled them. |
| `funding_paid` | quote | Funding over the run: positive is paid away, negative is received. Zero on spot. A perp result cannot be decomposed without it, since a carry and a direction otherwise look the same ([0020](Decisions.md)). |

These are the values a search ranks on. `calmar` scales a **negative** return by its drawdown rather than dividing by it, so the losing runs order by how much they lost, and `cagr_pct` saturates rather than escaping to infinity on a run of minutes ([0030](Decisions.md)). A Result's record holds none where a number would describe nothing, a CAGR off less than a year among them, each with its reason.

The four trade metrics count completed round trips only, so `exposure_pct > 0` beside `num_trades == 0` means "still holding", not "never traded", and buy and hold reports a real return with zero trades ([0023](Decisions.md)). They read each closed round trip's `pnl`, after its fees and its funding, because on gross PnL a strategy whose edge is smaller than its costs reports a perfect win rate and an infinite profit factor beside a negative return ([0023](Decisions.md), [0030](Decisions.md)). So they reproduce from the log:

```python
closed = [t for t in result.trades if not t["open"]]
win_rate = sum(t["pnl"] > 0 for t in closed) / len(closed)
```

`periods_per_year` must be finite and positive. The conventions for the risk-free rate, the sample deviation and annualization are fixed in [0030](Decisions.md).

### Trades

Each entry in `trades()` is one round trip: a maximal run of one sign, from the fill that takes the position off zero to the fill that returns it to zero or flips it, so resizing inside it does not end it and `num_trades` counts decisions rather than fills ([0023](Decisions.md)). The closed ones come in the order they closed, then the one still open at the end, marked at the last close:

| Field | Meaning |
| :--- | :--- |
| `instrument` | The instrument, 0 with one. |
| `side` | `long` or `short`. |
| `size` | The largest position it reached. |
| `entry`, `exit` | The average prices of its opening fills and of its closing fills; `exit` is NaN until a fill has closed some of it. |
| `open_bar`, `close_bar` | The bars it opened and closed on; `close_bar` is `None` while it is open. |
| `open_time`, `close_time` | Those bars' open times when the engine was built with `times=`, else `None`. |
| `realized` | The price PnL its closing fills realized, before fees and funding. |
| `fees` | Every fee of its fills, a flip's split by size between the round trip it ends and the one it begins, so each is carried exactly once ([0014](Decisions.md)). |
| `funding` | The funding charged on it while it was open. |
| `pnl` | Its net PnL: `realized` less `fees` and `funding`, and for one still open what it holds at the last close as well. The trade statistics are built from it, so they reproduce from the log. |
| `ret` | `pnl` over `size * entry`. |
| `best`, `worst` | How far its side's price went for it and against it, as a fraction of `entry`, over the part of each bar it was in: its entry bar from the fill to the close, its exit bar from the open to the fill, and every bar between whole. They are bounds that never overstate, since nothing inside a bar says when its high and its low came ([0023](Decisions.md)). |
| `reason` | What closed it, `market`, `limit`, `stop`, `take_profit`, `rebalance`, `seam`, `liquidation`, `dust` or `delist`, and `None` while it is open. A forced close is priced where the margin ran out, a sale on the book less its costs, not at the extreme the bar printed, so the loss is bounded by the margin ([0017](Decisions.md), [0018](Decisions.md)). |
| `open` | True for the round trip still open at the end. |

A Result's round trips are the same rows as a [table](#the-tables), every bar counted from its first row and every time its bar's.

### The ledger

Each column of `ledger()` holds one value per bar, the first for the opening balance, so no column is one shorter than the data, and a universe's columns from `position` to `pnl` hold a row of `N` per bar ([0022](Decisions.md)):

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
| `liquidation` | What a takeover left in the margin wallet, taken by the venue or absorbed by it, less what the margin bound lifted fills by ([0018](Decisions.md)). |
| `dust` | The value of positions below half a lot, zeroed at the close where sizes are on, less the hair below zero a spot fill's rounding can leave in the spot wallet, which is set back to zero. |
| `position`, `entry`, `mark`, `notional`, `unrealized`, `weight` | The position, its average entry, the close it is marked at, its notional, its unrealized PnL, and its signed notional over equity. |
| `drift` | The weight the latest `target` asked of the instrument less `weight`, from the bar after that target was decided; NaN where no target has named it. |
| `pnl` | What the instrument has made since the first row: the cash its fills, its funding and the charges given to it moved in the wallet it settles in, plus its position's value at the close, nothing while it is flat; the instruments' sum to equity less the opening equity ([Specification](Specification.md#the-ledger)). |

Each cost is computed on its own, so on every bar the change in equity is the price PnL, measured from each fill's reference and then from the bar's close, less every cost.

### Fills

Each entry in `fills()` is one fill, forced closes included ([0022](Decisions.md)):

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

Each entry in `order_log()` is one event in an order's life ([0022](Decisions.md)):

| Field | Meaning |
| :--- | :--- |
| `bar`, `time` | The bar it happened on, the decision's for a placement and the fill's for a fill, and that bar's open time when the engine was built with `times=`, else `None`. |
| `kind` | `placed`, `refused`, `clamped`, `filled`, `not_filled`, `expired`, `cancelled`, or `armed` for a bracket's leg taking its side. |
| `order`, `instrument` | The order and its instrument, both `None` for a clamped transfer. |
| `side` | `buy` or `sell`, what a placed, refused or armed event's order asked, else `None`, as for a bracket's leg until it arms. |
| `size` | What a fill or a clamp came to, what a placed or armed order asks in all, or what a refused call gave, else `None`, as for a value order on spot or a linear contract and a bracket's leg that closes the position it finds. |
| `price` | What a fill was booked at, or a placed or armed limit's price or stop's trigger on the price rule, else `None`. |
| `reason` | Why, in words: `max_open_orders: 8 already resting`, `the spot wallet's cash allows 5 of the 20 asked`, `nothing could fill: the liquidity budget allows none`, `ioc: one bar`, `liquidation`, `delist`, `the instrument is delisted`. |

A refused placement still returns `None`, and the log holds it under the id it would have had, with its reason. A clamp, a market order's miss and a `fok` market order that could not fill whole each name the rule that held them ([Specification](Specification.md#the-fill-log-and-the-order-log)):
* `the liquidity budget`, `reduce_only`, `the spot wallet's holding`, `the spot wallet's cash`, `the margin wallet's equity`, `the largest position`, which is the last tier's bound, or `the margin cap`, as in `fok: the margin cap allows 3.3333333333333335 of the 5 asked`.
* A market order that nothing could fill, where no rule held it, reads `nothing could fill: the bar printed no volume` or `nothing could fill: the size is below what a fill can book`.
* An `ioc` limit a rule empties, or a `fok` limit one holds short of its size, logs nothing on its bar and expires as `ioc: one bar` or `fok: one bar`.
* A hold to the largest order reads `the maximum order quantity allows 1 of the 5 asked`.
* A clamped transfer names the wallet it leaves, as in `transfer: the spot wallet's cash allows 600 of the 1000 asked` or `transfer: the margin wallet's free cash allows 600 of the 1000 asked`.

The numbers are printed as the engine holds them, with no unit.

<br>

## The Result

`backtest`, a Strategy and `probatum.Engine.result()` return the same `Result`, whatever decided the run ([0026](Decisions.md)). It holds what the run recorded on the rows its account saw, from its first:

| Name | What it is |
| :--- | :--- |
| `time` | each row's open in int64 nanoseconds since 1970 |
| `equity`, `returns` | each `(T,)` and read-only; a row's return is its equity over the row before's, less one, and nothing once the equity is gone |
| `ledger`, `holdings`, `fills`, `orders`, `trades` | the five [tables](#the-tables) |
| `costs` | the totals of what trading cost |
| `assumptions` | a table of every modelling choice the run made, graded |
| `baselines` | the run beside holding cash, buying and holding, and itself without costs |
| `result[name]` | one instrument's record |
| `explain(item)` | a fill's price or an order's life, in words ([Costs, assumptions and baselines](#costs-assumptions-and-baselines)) |
| `summary()` | a frame of its headline statistics beside buying and holding's, each with its interval or why it has none, and beside it its `context`: what ran, on which bars, and its assumptions, a book's accounts named as margined apart ([Specification](Specification.md#summaries)) |
| `stats` | the [statistics](#statistics), at `risk_free` and at the annualization of the whole Data the run's instruments are drawn from, counted on the rows they printed so every slice of it annualizes alike; None where the clock is too short |
| `policy` | an array's values on the Data's rows as the run read them, or the copy of a Strategy it ran |
| `bust` | whether the account's equity reached zero |
| `liquidations` | how many bars it was liquidated on, each account of a book counted apart |
| `provenance` | what the run was given and where it ran |

Its repr names the policy, the venue, the bars and the equity, and every assumption graded optimistic. `to_pandas()` and `to_polars()` give the equity and the returns on the time. `to_dict()` gives the whole Result as plain data, its costs and its assumptions included and its baselines once they are read, a statistic infinite where it earned something against no measured risk. A Result copies whole and pickles with its tables, its assumptions, its baselines once they are read and an array's policy, leaving a Strategy's copy out.

A run recorded at the account level keeps its ledger and its round trips, its costs, its assumptions and its statistics, and no policy. It refuses its holdings, fills, orders and baselines, an instrument's record, `position`, `weight`, `pnl` and `explain`, each by name; its repr names the level, `to_dict()` gives None for what it does not keep, and `combine` refuses it. A run recorded at the transitions level keeps the same and `transitions`, its agent's steps, which `export("arrow", path)` writes and which save, load and a pickle carry.

`chart()` draws the run over its Data's candles ([Charts](#charts)). A notebook shows a Result as its summary, and every table and record as the pandas frame its `to_pandas()` gives: what holds for every row of a record in a frame of one column beside it, and `stats` with each number as the summary shows its row and its reasons in a frame under it ([0041](Decisions.md)).

```python
print(result.to_pandas().tail())                # equity and returns on a UTC index
print(result.stats["total_return_pct"], result.bust, result.liquidations)
print(result.summary())                         # the run beside buying and holding
```

### The tables

`ledger`, `holdings`, `fills`, `orders` and `trades` are tables the engine builds, and `assumptions` and `baselines` tables the Result builds ([Costs, assumptions and baselines](#costs-assumptions-and-baselines)). Each is handed to any reader through the Arrow C stream interface, so `polars.DataFrame(table)`, `pyarrow.table(table)` and a DuckDB query that names the table read them with nothing imported by probatum, and `table.to_pandas()` builds a pandas DataFrame with no pyarrow at all. Every bar counts from the Result's first row, so `result.time[bar]` is its time, every instrument is its name, and a value the run did not record is a null, never a number that stands for none ([Specification](Specification.md#tables)).

| Table | A row per | Columns |
| :--- | :--- | :--- |
| `ledger` | bar, from the opening balance | `bar`, `time` and the account's columns of [the ledger](#the-ledger) |
| `holdings` | bar and instrument, a bar's in the Data's order | `bar`, `time`, `instrument`, `position`, `entry`, `mark`, `notional`, `unrealized`, `weight`, `drift`, `pnl` |
| `fills` | fill, in the order they landed | the [fill log](#fills)'s fields |
| `orders` | event in an order's life | the [order log](#the-order-log)'s fields |
| `trades` | round trip, the closed in the order they closed, then those still open | the [round trips](#trades)' fields |

`table[i]` is row `i` as a dict, None where a value is missing; iterating gives the rows, and `len(table)` counts them. `table["pnl"]`, or `table.pnl`, is a column as a read-only numpy array: floats with NaN where a value is missing; integers and times as int64, masked where a value can be missing, so `trades.close_bar` masks the round trips still open; booleans; and text as objects. A slice, a boolean mask or an array of positions gives a table of those rows, a masked position being refused since it names no row. `orders.events(kind)` gives the order log's events of one kind, `table.columns` names the columns, and two tables are equal where their columns and their values are.

```python
import duckdb
import polars

trades = result.trades
closed = trades[~trades.open]
print(result.time[closed.close_bar])            # when each closed round trip closed

print(result.orders.events(kind="filled"))      # the order log's fills alone

fills = result.fills
print(polars.DataFrame(fills).select("bar", "price", "fee"))
print(duckdb.sql(query="select instrument, count(*), sum(fee) from fills group by 1"))
```

### Saving and loading

`result.save(path)` writes a directory of files any tool reads:

| File | What it holds |
| :--- | :--- |
| `result.json` | the manifest: the format it is written in, the version of probatum that wrote it, the provenance, whether the policy was an array and whether the account went bust |
| `time.npy`, `equity.npy`, `returns.npy` | the run's series, and `policy.npy` for an array run |
| `ledger.arrow`, `holdings.arrow`, `fills.arrow`, `orders.arrow`, `trades.arrow` | the five tables as Arrow IPC files; a run below the full level writes only the tables it keeps |
| `assumptions.arrow`, `baselines.arrow` | the assumptions and the baselines, read first where they were not |
| `stats.arrow` | the statistics, as one row, none where the clock was too short to annualise |
| `transitions.arrow` | an agent's steps, as `export` writes them, where its run kept them |

A Result pickled before its baselines were read cannot give them, so save refuses it, or a book of one, before writing anything. `probatum.Result.load(path)` reads the format, never the class that wrote it, so a Result one version saves opens in the next. It refuses a format it does not read, and files that save did not write or that disagree with one another. Nothing is pickled, so a loaded Strategy's `policy` is None, and its provenance still describes it. Every file is written under another name before the first is put in place, the manifest last, so a save that an error or a killed process stops leaves the Result that was there, or a directory that load refuses.

```python
import tempfile

folder = tempfile.mkdtemp()

result.save(folder)

again = probatum.Result.load(folder)
print(again.trades == result.trades, again.stats == result.stats)
```

### The provenance

`provenance` records what the run was given and where it ran:

| Key | What it holds |
| :--- | :--- |
| `data` | the Data's provenance, with its fingerprint |
| `venue`, `venue_name` | the code that builds the venue, and its name |
| `markets` | each instrument's market as it ran: `spot`, `linear` or `inverse` |
| `settlement` | the asset the account kept its cash in, None where its instruments do not say |
| `contract` | the dollars an inverse contract is worth, None on any other market |
| `policy` | a description of it, beside an array's `sizing`, `tolerance`, `stop` and `take_profit` |
| `record` | the level `backtest` recorded the run at |
| `seed`, `streams` | the seed, and the random streams a venue that decides at random draws from |
| `risk_free`, `periods_per_year` | the run's risk-free rate and annualization |
| `version`, `packages` | probatum's version, and the version of every installed distribution behind a module the run had loaded, read from their metadata so nothing is imported to ask |
| `python`, `platform`, `wheel` | the interpreter, the machine, and the tags of the wheel probatum was installed from |

### Combining books

`probatum.combine(results, settlement=None)` adds Results from accounts at different venues into one book, given as a list of them, each named by its venue's name, or as a mapping of a name to each ([Specification](Specification.md#combining-books)). The book's clock is every time any of them has a row. Each account counts its opening balance before its first row and its last equity after its end, and the book's equity is their sum in the order given, its returns and statistics those of the sum. Every account is margined on its own, and no capital moves between them.

The book's ledger and its baselines are its own, a row per row of its clock, the baselines its accounts' summed as its equity is. Its other tables are theirs joined, each row with its `venue` first and every bar counted on the book's clock. A column a later version added to an account's table is left out of the book's.

Accounts kept in different dollar assets, such as USDT and USDC, add only where `settlement` names the dollar asset the book is kept in, counted at par. An account kept in any other asset adds only to accounts kept in it, and an inverse contract's only to others of its coin. Accounts on different intervals, off one interval's steps or at different risk-free rates do not add at all.

```python
bybit = probatum.backtest(Hundred(), data=data,
                          venue=probatum.venues.bybit_perp.replace(funding="none"))

book = probatum.combine([result, bybit])        # named frictionless and bybit_perp

print(book)
print(book.fills.to_pandas()[["venue", "bar", "instrument", "price"]])
```

### Costs, assumptions and baselines

`costs` is a dict of the run's costs, the totals of the ledger's cost columns in its order: `fees`, `funding`, `spread`, `slippage`, `impact`, `liquidation` and `dust`. Each is a compensated sum, positive where the account paid it, in the asset the account settles in, and measured from each fill's reference, so the move from a decision's close to the next open is price PnL ([Specification](Specification.md#costs)). Each read is a dict of its own.

`result[name]` is one instrument's record: its `name`; its `position`, `weight` and `pnl`, each `(T,)` and read-only, the weight over its account's own equity and the pnl what it has made since the first row; and its own `fills` and `trades`. With one instrument the Result answers `position`, `weight` and `pnl` itself, and with several it names the one to ask. In a book an instrument is `book[venue, name]`, or `book[name]` where one venue alone holds it. Its series are carried from each of its account's rows to the next inside the account's span; its position and weight are nothing outside it, and its pnl nothing before its first row and held after its last.

`assumptions` is a table of every modelling choice the run made, a row each ([Specification](Specification.md#assumptions), [0027](Decisions.md)):

| Column | What it holds |
| :--- | :--- |
| `name` | the choice |
| `value` | how it reads after the name, as in "limit fills on touch" |
| `source` | the preset and the day its tables were taken, or its starting calibration for a preset's slippage, impact and liquidity apart from the plain venue's; `plain`, `set by the user`, the router, the Data or the engine; each other input that decided part of it named after the first |
| `grade` | `venue` where the run does what a venue does as far as bars show it; `simplified` where it approximates that; `optimistic` where it leaves out what every venue charges or limits, or decides what a bar cannot show in the trader's favour |
| `direction` | for a simplified choice, the way it leans: `pessimistic`, `optimistic` or `either` |
| `cost` | what it cost, where the run's records measure it, in the asset the account settles in |

Its rows are the venue's settings, each read for the instruments the run held and naming those that depart from the rest; the engine's own rules, each in its off form where it cannot act on the run; and the Data's. At the full level the rows with a cost add up to the costs. Its repr shows every row, `table[table.grade == "optimistic"]` keeps the optimistic ones, and a book's are its accounts', in the order they were given, each naming its venue. `venue.describe()` gives the venue's and the engine's rows from the venue alone, with no cost, for every instrument it lists and one of each kind it takes but lists none of, and `data.describe()` gives the Data's.

`explain(item)` tells one fill's price or one order's life as text. A fill, a row of `fills`, reads as:
* its reference, the price it was measured from;
* the half spread and the slippage it paid, each with its rate, and the impact with its model, none for a maker, in a takeover or at a delisting;
* how far the bar's range held its price, or that a delisting's close came before the bar opened;
* what the margin bound lifted and paid in, where it acted;
* and its fee, as a rate on its notional.

Its price lines add up to its price, each to the decimals its reference sets, two from 1 up and enough for five significant figures below, its parts and its hold signed and a sale's costs negative. Money prints to the cent, or to five significant figures in a coin; rates print in basis points to one decimal, and to three significant figures below one; fees print as percentages to three significant figures, and sizes to twelve significant figures. An order, a row of `orders` or an `Order` the run's account handed out, reads as a line for each of its events, its time, its kind and what the order log says, under a head from the event that placed it, or that refused it where none did. A Result tells the handles of its own run alone, so one loaded, unpickled or combined asks for a row of its order log, and a book's text names the venue, as its rows do.

`baselines` sets the run beside three baselines on its own data and venue ([Specification](Specification.md#baselines)), as a table of a row per row of the Result:

| Column | What it holds |
| :--- | :--- |
| `bar`, `time` | the row |
| `cash` | the opening equity on every row |
| `buy_and_hold` | an equal share of the opening balance in each of the Data's instruments, bought as the quantity it buys at the instrument's first close and asked for at every close from then on, on the run's venue and paying its costs |
| `frictionless` | the same policy on the run's venue with the costs of trading removed and every rule kept, which still pays what a forced close takes |

An array plays again with its rows, sizing, tolerance and barriers, and a Strategy on a second copy `backtest` takes at the call, played once. A loop the caller stepped has no policy to play, so its `frictionless` is missing, as it is, with a warning, where the play raises. The baselines are computed when first read, an agent's as its backtest runs, since an agent cannot be copied to be played later. `save` reads them first, while pickling plays no policy and carries them only once they are computed. A book's are its accounts' summed as its equity is, each at its opening balance before its first row and at its last value after its end.

```python
print(bybit.costs)                              # what trading on bybit_perp cost
print(bybit["BTCUSDT"].pnl[-1], bybit.pnl[-1])
print(book["bybit_perp", "BTCUSDT"].weight[-5:])

print(bybit.assumptions)                        # every choice the run made, graded
print(probatum.venues.bybit_perp.describe())
print(data.describe())

print(bybit.explain(bybit.fills[0]))            # its reference, each cost, its fee
print(bybit.explain(bybit.orders[0]))           # that order's events

print(book.baselines.to_pandas().tail())        # cash, buying and holding, no costs
```

<br>

## Search

`probatum.search(policy, space, data, venue, objective="sharpe", direction="maximize", trials=100, sampler="tpe", min_trades=0, holdout=None, keep="all", n_jobs=1, seed=None, storage=None, risk_free=0.0)` searches a policy's parameters on the bars before a holdout and returns a `probatum.Study` of every trial ([0032](Decisions.md), [Specification](Specification.md#search)).

- **The policy** is a Strategy subclass, each trial built as `Cross(**params)`, or a function `policy(data, **params)` of the bars a trial runs on. The function returns weights, as an array or a table, a Strategy, which is copied before it runs, or a dict of `backtest`'s arguments, `{"policy": weights, "stop": 0.02}`, so a barrier, the `tolerance` or the `sizing` is searched as any other parameter. Every trial returns one kind, weights or a Strategy, and the space's names are checked against the policy's signature before any trial runs.
- **The holdout** is the tail: `holdout=0.3` holds out that share of the rows, `holdout=300` that many, and a time the rows from it on. `holdout=0` searches every bar, and so does leaving it out, with a warning, since every score is then in-sample. The trials run on a Data of the rows before the holdout alone, which holds no later bar.
- **A trial** is one run over those bars, with no Result, scored by `objective`, a statistic's name, which `direction` maximizes or minimizes. It is `complete`; `floored`, where it closed fewer round trips than `min_trades`, which keeps its statistics and its returns and has no score; or `failed`, where the policy or its run raised, its error kept as text. The winner is the complete trial with the best score, the lowest-numbered of equal ones, and a search with none raises.
- **The samplers.** `"random"` and `"grid"` are probatum's own, drawn from streams named from `seed`. `"tpe"` is Optuna's TPE, which needs optuna 5.0 or later and is told a floored or failed trial as failed. `seed=None` draws a seed from the operating system and records it. A seeded random or grid search repeats on every machine, and a seeded TPE search on the machine it ran on.
- `keep="all"` keeps every trial's returns, and `keep="none"` none.

```python
def crossing(data, fast, slow, stop):
    up = probatum.ta.ema(data.close, length=fast)
    down = probatum.ta.ema(data.close, length=slow)
    weights = numpy.full(shape=len(data), fill_value=numpy.nan)
    weights[probatum.ta.crossover(up, other=down)] = 1.0
    weights[probatum.ta.crossunder(up, other=down)] = 0.0
    return {"policy": weights, "stop": stop}


space = {"fast": probatum.Int(low=5, high=40),
         "slow": probatum.Int(low=50, high=200, step=10),
         "stop": probatum.Float(low=0.01, high=0.05, step=0.01)}

study = probatum.search(crossing, space=space, data=data, venue="frictionless",
                        sampler="random", trials=40, holdout=0.3, seed=7)

print(study)                                    # the best score, what is held out
print(study.trials.to_pandas()[["number", "state", "score", "fast", "slow"]].head())

held = study.evaluate_holdout()                 # the one look at the tail, counted
```

### The causality check

Before it trusts a trial, the search runs the first trial that decided something twice on the in-sample bars, and once on the bars up to each of its cuts ([Specification](Specification.md#the-causality-check)). There are at most ten cuts, placed where each instrument's decisions change and on the row before, where they held. A policy whose decisions do not repeat is refused as a `ProbatumValueError`. One whose decisions on the bars up to a cut change when the bars after it are cut away is refused as a `probatum.LookaheadError`, naming the cut and the first decision that differs. The winner is checked too, where its parameters differ.

A Strategy's run on a cut also decides at the cut's last close, where a value read from a later bar shows. A refusal of what a cut decides, by `backtest`'s reader or by a verb, is a `LookaheadError` chained from it, and an error the policy's own code raises there is its own, with a note naming the cut, and for a Strategy the last close it was deciding at. Each run builds the policy afresh, so a stream it seeds itself repeats, and the check reads the in-sample bars alone, so it is no look at the holdout.

### Speed and workers

Once a trial is checked, a random or grid search whose policy returns weights runs the rest in chunks of one call into Rust. Each trial runs on an engine of its own over the shared bars, keeping only what its statistics read, in parallel with the GIL released, so a sweep of thousands of parameter sets runs no Python between bars. TPE, which chooses each trial from the last, and a Strategy run one trial at a time.

With `n_jobs` above 1, or -1 for one a core, the trials that do not run together, a Strategy's and TPE's, run in that many workers that `spawn` starts once a trial is checked. TPE asks for a round of as many at a time and is told them in trial order, so a seeded random or grid search repeats at any number of workers, and a seeded TPE search at the same number. The search writes what the workers run from into a temporary folder and passes them its path, and each rebuilds the engine. Workers need cloudpickle, which the `search` extra installs with optuna.

* A worker's warnings come back once a place, from the module they came from and under their own categories, or as a `ProbatumWarning` naming one this process cannot import or build from its text. One your filters make an error fails its trial.
* A script that calls `search` outside `if __name__ == "__main__":` has every worker run it again as it starts, which is refused, naming the guard.
* A pool that breaks before any trial comes back raises. One that breaks later fails the trials in flight and warns, and raises where no trial is complete.

### Storage

`storage="studies/cross"` keeps the search's Study in that directory as it runs: its settings first, each batch of trials as it arrives, and the Study, as `save` writes it, when the search ends ([0046](Decisions.md), [Specification](Specification.md#saving-and-resuming)). The same call made again resumes it there, a seed of None taking the one it ran with. It runs the first trial kept there again and refuses a policy that now runs it otherwise, takes each kept trial in its turn once the sampler gives its parameters, and runs only the rest. The Study is then the one a search that never stopped returns, and a larger `trials` extends a random or TPE search that ended.

It refuses a directory that holds a search another version of probatum ran, or another search, naming the first setting that differs or the data where its bars differ. It also refuses a TPE search at another number of workers, which repeats only at its own, and a directory that holds no search. A trial that failed because the workers broke is not kept, nor is the Study of a search whose workers broke, so the next call runs those trials. A Study whose winner runs otherwise than its trial did is returned, with a warning, and not kept. A look is counted there before the winner is handed the held-out bars, and a resume keeps the looks, and the holdout where the winner and the held-out bars are the ones it ran on.

### The Study

| Name | What it is |
| :--- | :--- |
| `trials` | a table of a row a trial: `number`, `state`, `score`, each parameter, an int, a float, a bool or text by the values its range holds, the fifteen [statistics](#statistics) and `error`, a value a trial lacks missing |
| `returns` | the `(R, T)` float64 matrix of every trial's returns on the in-sample bars, read-only, counted as a Result's returns are, a failed trial's row NaN, or None under `keep="none"` |
| `best_params`, `best_score` | the winner's parameters and score |
| `best` | the winner's Result on the in-sample bars, run again at the full level when first read, its statistics its trial's to the bit |
| `evaluate_holdout()` | runs the winner once on the held-out bars, its warm-up inside them, and keeps the Result as `holdout`, which a second call returns |
| `looks` | the looks spent at the holdout |
| `provenance` | what the search was given: the sampler with its arguments and Optuna's version, the seed, the trials asked and run, the parameters its causality check passed, the holdout, and the in-sample bars' annualisation and fingerprint, beside the venue, the Data's provenance and the packages |
| `kind` | `"trials"`; a validation's Study, of kind `"paths"`, is [below](#the-study-of-paths) |

A winner that runs otherwise now, because the policy's code or the globals it reads changed since the search, is refused as a `ProbatumRuntimeError`. `evaluate_holdout()` reads `best` first, and counts a look in `looks` once the winner is handed the held-out bars, so a run that raises there has spent one. A search logs its start and its end at INFO, and each trial at DEBUG, to the `probatum` logger.

`summary(null=None, plateau=None)` is the Study's summary, a frame of a row a number. It holds:
* the winner's score;
* with `null`, its deflated Sharpe against every trial and its cluster's, and the null's mean score;
* the overfitting probability and its probability of loss, or why it has none;
* with `plateau`, a score as the trials table holds it, the share of the configurations within 0.1 of the winner that scored past it;
* and the holdout as it stands: its interval at the level its looks leave, or why it has no score (not yet run, run and raised, or read before with no result kept for this winner), beside buying and holding over the held-out bars.

Its `context`, beside it, is what was searched and on what: the in-sample span, what is held out, the looks spent and the winner's parameters. `null` is a random search of the same policy over the same space, on the same venue and in-sample bars and scored alike, whose spread deflates the winner's Sharpe; any other is refused, naming what differs.

`deflated_sharpe(null)`, `effective_trials()`, `overfitting()`, `reality_check(benchmark=None, resamples=1000, seed=None)`, `min_backtest_length()`, `plateau(level)` and `importance()` judge a search across its trials, as [`probatum.stats`](#inference) does ([0033](Decisions.md), [0047](Decisions.md)). `consistency(stretches=None)` scores the winner's in-sample run on `stretches` stretches of its rows, 4 unless given, as a validation scores a window, and returns a `probatum.stats.Consistency` that counts those past the objective's break-even. `decay()` returns a `probatum.stats.Decay`, the winner's score less its holdout's, signed so a positive decay is worse out of sample. It needs a look whose run kept its result, and refuses an objective that grows with the span it is scored on ([Specification](Specification.md#consistency-and-decay)).

`save(path)` writes the Study into a directory of open files, as a Result is written ([0046](Decisions.md), [Specification](Specification.md#saving-and-resuming)):

| File | What it holds |
| :--- | :--- |
| `study.json` | the manifest, with the provenance, the looks and the winner |
| `trials.arrow`, `choices.arrow` | a search's trials, and each trial's choice of a `Categorical` as its index, since the table prints some choices alike |
| `paths.arrow`, `fits.arrow`, `windows.arrow` | a validation's tables, in place of the trials |
| `returns.npy` | the returns |
| `best`, `holdout` | the winner's Results as Result directories, `best` run first where it was not read |

Each file is written under another name and the manifest last. The one there before, or the new one where there is none, is kept as `study.json.old` until then, so a save that stops leaves the Study that was there, or a directory `load` refuses until the Study is saved there again. A directory that holds files that are no Study's, a manifest save did not write, or another search's storage is refused. Each look the Study takes from then on is counted there too, before its winner is handed the held-out bars, as it is in a search's storage. `probatum.Study.load(path)` reads it back with every table, statistic and summary it had. It holds no policy or data, so `evaluate_holdout()` gives the saved holdout, and refuses, before any look, one that never ran.

```python
null = probatum.search(crossing, space=space, data=data, venue="frictionless",
                       sampler="random", trials=40, holdout=0.3, seed=8)
print(study.summary(null=null))                 # the winner, deflated, the null's mean

print(study.reality_check())                    # does any trial beat cash at all
print(study.consistency())                      # the winner's run in four stretches
print(study.decay())                            # its score less its holdout's
```

### The space

- `Int(low, high, step=1, log=False)`: a whole number from `low` to `high`, both included, on the lattice `low + k * step`, the step a whole number from 1 that divides the span. `log=True` draws it evenly in its logarithm, with a `low` from 1 and a step of 1.
- `Float(low, high, step=None, log=False)`: a number from `low` to `high`. With no step it is the interval `[low, high)`, and with one the lattice from `low` whose last point is `high` exactly, the step dividing the span as their decimals do, so `Float(0.01, 0.1, step=0.01)` is ten points. `log=True` draws it evenly in its logarithm, with a `low` above 0 and no step.
- `Categorical(choices)`: one of a list of None, bools, numbers and strings, no two of them equal, so `1`, `1.0` and `True` are one choice.

A trial receives each value as a plain Python int, float, bool, str or None. A range compares and hashes by what it holds, and pickles as the call that builds it.

```python
space = {"fast": probatum.Int(low=5, high=40),
         "slow": probatum.Int(low=40, high=200, step=10),
         "stop": probatum.Float(low=0.01, high=0.05, step=0.01),
         "kind": probatum.Categorical(choices=["ema", "sma"])}

print(space["slow"], space["stop"])      # Int(40, 200, step=10) Float(0.01, 0.05, step=0.01)
```

<br>

## Labels

`probatum.labels(data, venue, horizon, notional, stop=None, take_profit=None, side="long")` gives, for each row of a Data and each of its instruments, what one trade of `notional` decided at that row's close makes through the fill model a backtest runs, as a `probatum.Labels` ([0036](Decisions.md), [Specification](Specification.md#labels)).

**A trade.** Each label is a fresh account on the Data's clock trading its instrument alone, its balance the notional in the currency the instrument settles in, an inverse contract's coin. It buys the notional at the next open, or sells it with `side="short"`, which a spot instrument refuses. It places a bracket at `stop` and `take_profit`, returns from its average entry, at the close of the bar its entry filled on, as an array run places one; each is None, a number, or a `(T, N)` array of them read on the label's row. It closes what is left at the close of the row `horizon` bars on, or of its instrument's last printed row before it, that close filling at the next open and placed again at each close after while some is left. Its return is net of fees, funding, spread, slippage and impact, and a short label is a trade of its own rather than the long one's negative. A venue whose `intrabar` is `"random"` is refused, as `backtest` refuses one.

**The record.** A Labels holds, each `(T, N)` and read-only:

| Name | What it holds |
| :--- | :--- |
| `ret` | the trade's return, NaN where the label is censored or has none |
| `exit` | what left the account flat: `"stop"`, `"take_profit"`, `"horizon"`, `"delist"` or `"liquidation"`; `"censored"` where the horizon runs past the last bar, even where a barrier would have filled first, or where the trade is still open at the last bar; and `""` where the row has no label, its instrument not printing there, the row being its last printed one, or the entry filling nothing by the horizon |
| `end` | the bar whose price closed the trade, at `T` or later where it is censored and the least int64 where it has none, as the [splitters](#splitters) read it |
| `bars` | the bars from the entry's fill to `end` |
| `weight` | its average uniqueness among its instrument's labels that have a return (Lopez de Prado 2018, chapter 4), a fit's `sample_weight` |

Censoring past the last bar never depends on the outcome. A Labels keeps `horizon`, `notional`, `side`, `stop`, `take_profit`, `venue`, the Data's `names` and its `fingerprint`, never changes, and pickles. The labels run in Rust in one call, in parallel with the GIL released, each on an engine of its own over the Data's bars.

```python
labels = probatum.labels(data, venue="frictionless", horizon=24, notional=10_000.0,
                         stop=0.03, take_profit=0.03)

print(labels)                   # what was labelled, on which side, notional and horizon
print((labels.exit == "stop").mean(), numpy.nanmean(labels.ret))

splitter = probatum.cv.PurgedKFold(k=5, end=labels.end)  # purged where each trade closed
```

<br>

## Splitters

`probatum.cv` holds the ways of dividing time, each a splitter scikit-learn's own tools take, built without importing scikit-learn: `split(X, y=None, groups=None)` gives each split's training and test rows, and `get_n_splits` counts them ([0035](Decisions.md), [Specification](Specification.md#the-splitters)).

| Splitter | What it gives |
| :--- | :--- |
| `probatum.cv.Holdout(tail, end=None)` | one split: the last `tail` bars tested, a share of them or a number, and the bars before trained on |
| `probatum.cv.WalkForward(train, test, anchored=False, end=None)` | windows from the first bar, each testing the next `test` bars and trained on the `train` bars before it, or with `anchored` on every bar before it, the last window taking the remainder |
| `probatum.cv.PurgedKFold(k, embargo=None, end=None)` | `k` groups of bars in order, as scikit-learn's KFold sizes folds, each tested in turn |
| `probatum.cv.CombinatorialPurged(groups, test_groups, embargo=None, end=None)` | every combination of `test_groups` of `groups` such groups tested, recombined into `C(groups - 1, test_groups - 1)` out-of-sample paths |

- **Bars and rows.** A splitter divides bars. Its rows are the long, time-major rows an estimator reads, row `r` being bar `r // N` of instrument `r % N`, so X holds every bar and instrument of the grid in that order, and an X of another length is refused. `end` is each row's label end, an int64 array `(T,)` or `(T, N)`: the bar whose price closed its trade, from its own bar plus one on, at `T` or later for a label that runs past the bars, and the least int64 for a row with no label, which is in no set. Without `end` each row is a bar holding a label of no length.
- **The purge.** A training row is left out where its label reaches a test span from before it, or where it starts after the span up to the latest end of the labels that test there, and the embargo leaves out the bars after that. A label that ends past the last bar, at `T` or later, never trains. `embargo=None` is the longest label held. Holdout and WalkForward take no embargo, since no training row follows their test spans.
- **The contract.** Every split is checked before the first is given, and one the purge leaves with no row to train or test on is refused rather than skipped, since scikit-learn's searches count on `get_n_splits`. `y` is not read, and `groups` is ignored with a warning. A splitter has no `get_params`, `shuffle` or `random_state`, never changes, compares and hashes by what it holds, and pickles as its call.
- **With `cross_val_predict`.** scikit-learn's `cross_val_predict` takes PurgedKFold, and CombinatorialPurged with one test group, built without `end`, since their test sets then cover each row once. Built with `end`, the last bar's rows, whose labels end past it, are in no test set, so `cross_val_predict` refuses them.

```python
end = numpy.arange(stop=1_000) + 25              # each label ends 24 bars after its entry
splitter = probatum.cv.PurgedKFold(k=5, embargo=24, end=end)
train, test = list(splitter.split(numpy.zeros(shape=1_000)))[2]
print(test.min(), test.max(), train.max())       # 400 599 974: purged, embargoed, censored

combinatorial = probatum.cv.CombinatorialPurged(groups=8, test_groups=2)
print(combinatorial.get_n_splits())              # 28, recombined into 7 paths
```

<br>

## Validation

`probatum.validate(policy, data, venue, cv, space=None, search=None, features=None, labels=None, to_weights=None, holdout=None, fill=None, keep="all", seam="cancel", seed=None, risk_free=0.0)` runs a search, or fits a model, on the training bars of each split a [splitter](#splitters) gives, and trades each split's winner on its test span. Every out-of-sample path is one run of those winners in turn, and the result is a `probatum.Study` of the paths ([0034](Decisions.md), [0035](Decisions.md), [Specification](Specification.md#validation)).

- **The policy** is a Strategy subclass or a function `policy(data, **params)` that returns weights or a Strategy, as `search` takes them, and `space` holds its parameters' ranges, given always, since a fixed parameter set fits nothing. A model is fitted rather than searched ([below](#a-model)). `search` is a dict of `probatum.search`'s `objective`, `direction`, `trials`, `sampler`, `min_trades` and `n_jobs`, which every search of the validation runs by; `holdout`, `keep`, `seed` and `risk_free` are validate's own. `cv` is built without `end`, since neither a Strategy nor a function has labels. With `PurgedKFold` and `CombinatorialPurged`, whose fits join pieces end to end, `max_drawdown_pct` and `calmar`, which the pieces' order changes, are refused as the objective.
- **The first search.** The holdout is the tail, held out as `search` holds it. Before any split, a plain `search` runs over the bars the splitter gives the holdout as its test span: every bar before it, or with a rolling `WalkForward` the last `train` of them. Its winner is the Study's, which `evaluate_holdout()` runs once on the tail, and its causality check refuses a policy that reads ahead before any split runs.
- **Fits.** Each split's search runs on its training bars, each run of them a piece of its own. A function is called with each piece's Data, and its weights run as an array that opens where the piece's account opens. A piece from the first bar runs as a search's trial does. A piece after a test span runs on a Data that starts after the bar the span's last decision is scored on, with its account opening past the embargo, its warm-up and the venue's fill window of each instrument that printed before that Data, or the instrument's last bar there where it prints fewer, so no fit reads a test bar.
- A trial on several pieces is scored on their returns joined end to end, and a function's trial decides in the kind the first search's winner did, weights or a Strategy, or fails. A fit with no complete trial fails. Each winner the fits chose that the first search did not check is checked for causality on the bars before the holdout, and so is each path that ran, at its seams, where `on_inherit` decides.
- **Paths and seams.** A path is one run over the bars before the holdout, each of its windows deciding at its test span's closes with its fit's winner, the account carried across. At a seam every order not yet filled is cancelled with the reason `seam`, the targets are forgotten, and the incoming window's `on_inherit(bar, account)` is called before its first decision, its account's `events`, `trades` and `equity_curve` starting from that close. `seam="flatten"` closes every position at the outgoing window's last close instead of deciding.
- A path of a function's weights is one array run of each window's rows of its winner's weights, the function called with the bars before the holdout, with the stop and take-profit the winner returned. At a seam each instrument holding a round trip places again, at its average entry, the legs of its bracket still resting, or after a flattening close those resting until it, and one a barrier holds flat stays held. Every winner of such a path returns one sizing and one tolerance, or the path fails, naming its windows. A window owns the bars from the one after its first decision to the one after its last, and a path whose run raises, or that a failed fit trades, fails alone.
- **Seeds and workers.** `seed` seeds the first search from the stream named `validate final` and split `i`'s from `validate fit i`, so a seeded validation repeats; it is drawn from the operating system and recorded where it is None. `n_jobs` above 1 runs every search's trials in workers, one pool serving every fit.

```python
class Channel(probatum.Strategy):
    def __init__(self, lookback=20, stop=0.02):
        self.lookback, self.stop = lookback, stop

    def init(self, data):
        self.warmup = self.lookback + 1
        self.high = probatum.ta.highest(data.high, length=self.lookback)

    def next(self, bar, account):
        if account.position == 0.0 and bar.close > self.high[bar.i - 1]:
            account.buy(size=1.0)
            account.bracket(stop=bar.close * (1.0 - self.stop))

    def on_inherit(self, bar, account):
        if account.position > 0.0:              # the seam cancelled the stop it held
            account.bracket(stop=bar.close * (1.0 - self.stop))


space = {"lookback": probatum.Int(low=10, high=40),
         "stop": probatum.Float(low=0.01, high=0.05, step=0.01)}

study = probatum.validate(Channel, data=data, venue="frictionless",
                          cv=probatum.cv.CombinatorialPurged(groups=5, test_groups=2),
                          space=space, search={"trials": 10, "sampler": "random"},
                          holdout=0.3, seed=7)

print(study)                                    # 4 paths from 10 fits, the median score
print(study.windows.to_pandas()[["path", "fit", "first", "last", "score"]].head())

held = study.evaluate_holdout()                 # the first search's winner, once, on the tail
```

### A model

`validate(model, data, venue, cv, features=..., labels=..., to_weights=..., fill=None)` fits `model`, which has `fit` and `predict`, to `labels`, a `probatum.Labels` of the same Data, on `features`, a `(T, N, F)` array over the Data's rows or a function of a Data that returns one ([Specification](Specification.md#a-model)).

- **A fresh estimator for every fit.** Each fit starts from scikit-learn's `clone` of the model, so no fit sees another's rows, and a model fitted beforehand predicts as an unfitted one does. A model with no `get_params` is refused. A function of no arguments, or a class, given in its place is called for each fit instead, as a model whose `fit` resumes from its weights is given. `space`, `search` and `seed` are refused beside a model.
- **Features.** A function of the Data is computed once on the whole Data, a second time to see that it repeats, and again on it cut after each of up to 10 rows spread over the bars before the holdout. Where a value up to a cut changes by more than the search's check lets a weight change, it is refused as a `probatum.LookaheadError` naming the row, the instrument, the feature and the cut, so a feature normalised over the whole series is refused before anything is fitted. The final fit's predictions, and `to_weights` of them, are checked twice and on the same cuts. A NaN past a feature's warm-up is filled where `fill` is a number or `"previous"`, and otherwise leaves its row out of every fit and window.
- **Fits.** The labels' `end` on the in-sample bars is passed to `cv`, which is built without `end`, and its embargo, left as None, is the longest label held. A fit reads, in the long, time-major order, its split's training rows that are available, have finite features and a finite `ret`, and are left by the purge and the embargo, with their `weight` as `sample_weight` where `fit` takes that keyword. A final fit reads every in-sample row whose label ends before the holdout, or with a rolling `WalkForward` those of its last `train` bars, and its predictions traded on the bars it read from are the Study's `best`.
- **Windows and paths.** A window predicts on its rows available with finite features, one finite number a row. The predictions go back to their rows of a `(T, N)` grid, NaN elsewhere, and `to_weights` of the window's rows of that grid gives its weights, NaN holding. A path is one array run of its windows' weights with the labels' stop and take-profit, each on the side of the position it guards, whose seams place each inherited bracket again, as a function's path of weights does. `to_weights` decides the side, and a short on a spot instrument is refused, naming it.
- **Skill apart from the costs.** Each path's forecast is Spearman's correlation of its predictions with its labels' returns, over the rows it predicted whose label has a return and ends before the holdout. Its standard error is Newey-West over the bars, at a lag of the longest label among them, and its R squared is against a forecast of zero. The Study's `forecast` pools every complete path's rows.

```python
import sklearn.linear_model


def momentum(data):                                              # (T, N, 2)
    return numpy.stack([probatum.ta.roc(data.closes, length=24),
                        probatum.ta.rsi(data.closes, length=14)], axis=-1)


labels = probatum.labels(data, venue="frictionless", horizon=24, notional=10_000.0,
                         stop=0.03, take_profit=0.03)

study = probatum.validate(
    sklearn.linear_model.Ridge(alpha=1.0), data=data, venue="frictionless",
    cv=probatum.cv.CombinatorialPurged(groups=8, test_groups=2, embargo=24),
    features=momentum, labels=labels,
    to_weights=lambda predicted: numpy.clip(predicted * 10.0, a_min=0.0, a_max=0.2),
    holdout=0.2,
)

print(study.summary())                          # the paths, their median, the forecast
print(study.forecast)                           # rank IC, its Newey-West t, R squared

held = study.evaluate_holdout()                 # the final fit, once, on the tail
```

### The Study of paths

A validation's Study is of `kind` `"paths"`, and holds three tables beside its returns:

| Table | A row per | Columns |
| :--- | :--- | :--- |
| `paths` | path | `number`, `state`, `score`, the fifteen [statistics](#statistics), the seven [costs](#the-ledger) of the account rows it owns, and `error` |
| `fits` | split | `number`; `test`, the bars it tests; `train`, `purged` and `embargoed`, its training bars and those the purge and the embargo took; `pieces`; each parameter of its winner, missing where the fit failed; `fitted`, the winner's score; `trials` and `complete`; `seed`; `state` and `error` |
| `windows` | stretch of a path one fit traded | `path`, `fit`, `first` and `last`, the account rows it owns, their times `first_time` and `last_time`, `decided`, the closes it decided at, `score`, the statistics and the costs |

`returns[path, first:last + 1]` is what a window was scored on. `returns` is `(paths, T)` over the in-sample account rows, read-only, NaN where a path owns no row and on a failed path, and None under `keep="none"`. `best`, `best_params`, `best_score`, `evaluate_holdout()`, `holdout` and `looks` are the first search's. `trials` is refused, as `paths`, `fits` and `windows` are on a search's Study.

A model's validation differs:
* Its paths and windows are scored on the Sharpe, and its `paths` add each path's `rank_ic`, `rank_ic_t`, `r_squared` and `predicted`, the rows its forecast read.
* Its `fits` count rows rather than bars, `train`, `purged` and `embargoed` among them, and hold `rows`, the rows the fit read, and `dropped`, those a NaN feature left out, in place of the pieces, the parameters, the score, the trials and the seed.
* `forecast` is a `probatum.stats.Forecast`, None for a searched policy.
* `best` is the final fit traded on the in-sample bars, `best_params` is empty and `best_score` is that run's Sharpe.
* `decay()` is refused, since the fits score no bars of their own.

`provenance` records the splitter and its layout, the seam, the seeds of the first search and of every fit, and the parameters checked for causality, beside what a search's records. A validation logs its start and its end, each fit and each path, at INFO to the `probatum` logger. Its `summary()` is a frame of its paths: their median score beside the median path's interval, or its range for drawdown and Calmar; its windows and those past the break-even; its decay and that decay's standard error; a model's forecast; the paths' costs and price pnl; and the holdout, with how the bars were cut as its context. It takes no null. `consistency()` reads its windows and refuses `stretches`, and `decay()` sets each window's score against its fit's in-sample score.

<br>

## Inference

`probatum.stats` says how far a run's statistics can be trusted ([0031](Decisions.md), [Specification](Specification.md#inference)). Each function takes a Result, whose returns after its first row it reads at the run's own annualisation and risk-free rate, or a plain series of returns: a 1-D array, a list or a pandas Series read by position, with `periods_per_year`, which has no default, and `risk_free`, an annual rate, 0 unless given.

The statistics of a search's trials read its Study: the returns its trials earned on the in-sample bars, at the search's annualisation and risk-free rate. `effective_trials`, `overfitting` and `reality_check` also take a matrix of a row a trial's returns on the same bars, with `periods_per_year` and `risk_free` as a plain series takes them ([0033](Decisions.md), [Specification](Specification.md#across-a-search)).

| Call | Gives |
| :--- | :--- |
| `probatum.stats.sharpe(returns, periods_per_year=None, risk_free=None, long_run=False)` | the Sharpe, a Result's its record's; with `long_run`, over the long-run deviation of the returns, which their autocorrelation moves |
| `probatum.stats.sharpe_interval(returns, periods_per_year=None, risk_free=None, level=0.95)` | an `Interval`, the Sharpe's two-sided interval by the delta method |
| `probatum.stats.probabilistic_sharpe(returns, benchmark=0.0, periods_per_year=None, risk_free=None)` | a `Probability`, the one-sided probability that the Sharpe is above `benchmark`, an annual Sharpe |
| `probatum.stats.min_track_record_length(returns, benchmark=0.0, confidence=0.95, periods_per_year=None, risk_free=None)` | a `TrackRecord`, the returns and the years at which a Sharpe this high would give that confidence |
| `probatum.stats.bootstrap(returns, statistic, periods_per_year=None, risk_free=None, level=0.95, resamples=2000, seed=None)` | an `Interval` of `statistic` by a bootstrap: of the returns for `total_return_pct`, `net_profit_pct`, `cagr_pct`, `volatility_pct`, `sortino`, `value_at_risk` and `conditional_value_at_risk`, with `max_drawdown_pct` and `calmar` as the middle of the resampled histories, and of a Result's closed round trips for `win_rate`, `profit_factor` and `avg_trade_pct` |
| `probatum.stats.sharpe_difference(a, b=None, periods_per_year=None, risk_free=None, level=0.95, resamples=2000, seed=None)` | a `Difference`, two Sharpes on the same bars tested by the delta method and by Ledoit and Wolf's studentized circular block bootstrap, `b` a Result's own buy-and-hold baseline unless given |
| `probatum.stats.alpha_beta(result, benchmark=None, lag=None, level=0.95)` | an `AlphaBeta`, a run's alpha a year and its beta against buying and holding, or a given benchmark, with Newey and West's standard errors, the tracking error and the information ratio |
| `probatum.stats.trade_order(result, n=2000, seed=None)` | a `Permutations`, the drawdown at trade closes and the longest losing run of a run's round trips in the order they closed, beside `n` other orders of them |
| `probatum.stats.deflated_sharpe(study, null)` | a `Deflated`, a search's winner's probabilistic Sharpe against what the best of every trial it asked reaches by luck, their Sharpes spread as a random null's, and its cluster's against the effective number of trials; a null that asked fewer trials than the search warns |
| `probatum.stats.deflation_threshold(spread, trials)` | the annual Sharpe the best of `trials` independent trials reaches by luck when their annual Sharpes spread `spread` about zero |
| `probatum.stats.effective_trials(trials, periods_per_year=None, risk_free=None, seed=None)` | an `EffectiveTrials`, the clusters Lopez de Prado and Lewis's clustering finds among a search's trials or a matrix's rows, and the spread of their Sharpes, in a time and a memory that grow with the square of the trials, which a Study keeps once found |
| `probatum.stats.min_backtest_length(trials, sharpe=None)` | a `BacktestLength`, the years over which the expected best of `trials` trials with no skill falls to the annual `sharpe`, or a search's at its winner's Sharpe |
| `probatum.stats.plateau(study, level)` | a `Plateau`, the share of the configurations a search ran near its winner in normalised parameters, each on its range's scale and a choice 0 or 1 from the winner's, and each counted once, that completed past `level`, a score as the trials table holds it, a drawdown a positive loss, at each radius from 0.05 to 0.5, so a winner alone on a spike reads as one |
| `probatum.stats.overfitting(trials, objective=None, periods_per_year=None, risk_free=None)` | an `Overfitting`, the probability of backtest overfitting by combinatorially symmetric cross-validation: the share of the 12,870 halvings of 16 blocks whose in-sample best ranks below the median out of sample, and the share where it loses there |
| `probatum.stats.reality_check(trials, benchmark=None, periods_per_year=None, risk_free=None, resamples=1000, seed=None)` | a `RealityCheck`, White's Reality Check and Hansen's SPA of whether any trial beats cash, buying and holding or a given series, on the trials' log growth over it under one shared stationary bootstrap |
| `probatum.stats.autocorrelation(returns, lag=1)`, `probatum.stats.effective_sample(returns)` | the returns' autocorrelation at a lag and their count of independent returns, which no interval reads |

- **A run's record.** A Result's `stats` is a `Statistics` that holds what its intervals read, so `result.stats.interval("win_rate")` is the round trips' interval and `result.stats.interval("sharpe", level=0.9)` the Sharpe's at 90%. Each is computed when first read and kept, is the same after a save and a load, and is never computed inside a search's trials, whose rows hold point estimates. `Result.summary()` reads them.
- **One variance.** The interval, the probability and the track record read one variance: the long-run variance of the Sharpe's influence series, with Newey and West's weights at the lag their 1994 rule picks on that series, floored at its value at lag zero, so autocorrelation widens an interval and never narrows it. Each names that lag and whether the floor bound.
- **Two bootstraps.** The returns are resampled in blocks whose mean length is the longest that Politis and White's rule, as Patton's code computes it, finds on the returns, their squared deviations or their tail, so the stretches a strategy's exposure switches on and off set it. Each interval records it as `block`. Total return, net profit, CAGR, volatility, Sortino and conditional value at risk are studentized, each resample's move over its own standard error, and value at risk is percentile.
- The round trips are drawn evenly: the win rate's interval is percentile, the profit factor's bias-corrected and accelerated, and the average trade's studentized. A drawdown's and a Calmar's `kind` is `"range"`, the middle of the resampled histories, which says what histories like the run's do and is no confidence interval. Each resampling draws from a stream the values it resamples seed, so an interval is the same on every read and in any process with no seed, and `seed=` draws others.
- **Against something else.** `sharpe_difference` reports both its tests, each with its p-value of no difference: the delta method on the two influence series' difference, at the 1994 lag on it and floored at lag zero, and Ledoit and Wolf's studentized bootstrap, in blocks of Patton's circular length on that difference. `alpha_beta` regresses the run's excess returns on the benchmark's, its covariance Newey and West's at the 1994 lag on the slope's score, and its information ratio is the arithmetic one, the mean active return over the tracking error. `trade_order` compounds each round trip's return on the equity it opened on, the model of a policy sized on its equity, so the order is all that moves; its drawdown is at trade closes and leaves out what the account marked inside a trade.

**What comes back.** Each result never changes, compares by what it holds, and shows as the frame its `to_pandas()` gives:

| Result | Holds |
| :--- | :--- |
| `Interval` | `statistic`, `estimate`, `low`, `high`, `level`, `method`, `count`, `lag`, `floored`, `block`, `kind` and `p`, a drawdown and a value at risk as positive losses |
| `Probability` | `value`, `benchmark`, `lag`, `floored` and `count` |
| `TrackRecord` | `observations` and `years`, None where the Sharpe is not above the benchmark, `series_years`, `sharpe`, `confidence`, `benchmark`, `lag`, `floored` and `count` |
| `Difference` | `estimate` and its tests `delta` and `bootstrap`, each an `Interval` |
| `AlphaBeta` | `alpha`, `alpha_error`, `alpha_low`, `alpha_high` and `alpha_t`, `beta`, `beta_error`, `beta_low` and `beta_high`, `tracking_error`, `information_ratio`, `lag`, `level` and `count`, alpha and the tracking error as fractions |
| `Permutations` | `drawdown`, `drawdown_low`, `drawdown_high`, `deeper`, `losing_run`, `losing_run_low`, `losing_run_high`, `count` and `n` |
| `Deflated` | `value`, `threshold`, `trials`, `spread`, `null_trials`, `sharpe`, `effective`, `effective_value` and `effective_threshold`, None where the search kept no returns, `lag`, `floored`, `count` and `descriptive` |
| `EffectiveTrials` | `count`, `trials`, `numbers` and `clusters`, the trials it clustered and each one's cluster, `quality`, `spread` and `dropped` |
| `Overfitting` | `probability`, `loss`, None for volatility, `trials`, `blocks`, `dropped`, the oldest returns left out, `objective`, `direction` and `descriptive` |
| `RealityCheck` | `p`, White's p-value, `spa`, Hansen's consistent one studentized in each resample, `statistic` and `spa_statistic`, `best`, the trial of the highest log growth over the benchmark, `block`, `resamples`, `trials`, `excluded`, those that lost all of a bar, `benchmark` and `descriptive` |
| `BacktestLength` | `years`, `trials`, `sharpe` and `series_years` |
| `Plateau` | `radii`, `trials` and `passed`, the configurations within each radius and those past the level, `share`, NaN where none lies within, `level`, `objective` and `direction` |
| `Forecast` | a model's validation's `rank_ic`, `error`, `t`, `lag`, `r_squared`, `rows` and `paths` |

**What is refused.**
* As a `ProbatumTypeError`: a plain series without `periods_per_year`; a Result given one, or a `risk_free`, which it names itself; a Result beside a plain series; and a statistic or a record's name that is not a string.
* As a `DataError`: two Results on other rows, or at another annualisation or risk-free rate, and a benchmark Series not indexed by the run's row times.
* As a `ProbatumValueError`: a NaN or an infinity, naming its position; a run at the account level with no benchmark given; and an interval the sample cannot support. That is fewer than 4 returns or returns that never vary; returns whose Sharpe's influence is zero on every bar; two series whose Sharpes move alike on every bar; fewer than 500 returns or 100 of the bootstrap's blocks for a bootstrap of them; a return of -100% or worse for growth; a tail whose `k` lowest returns are all equal for conditional value at risk; a studentized statistic with no standard error; an interval, though not a range, whose resamples put both its bounds on one value; fewer than 60 closed round trips, or trips that cannot vary in what the statistic counts; fewer than 2 to permute; fewer than 2 resamples or too few for the level; resamples all on one side of the profit factor's estimate; and an estimate ranked at infinity.
* Of a search's trials, as a `ProbatumTypeError`: a Study given `periods_per_year`, `risk_free` or an `objective`; anything but a Study or an array of numbers; a matrix without `periods_per_year`; and a benchmark that is no series, `"buy_and_hold"` or None.
* Of a search's trials, as a `ProbatumValueError`: an array that is not 2-D, or holds a NaN or an infinity, naming its trial and position; a validation's Study; a Study kept with `keep="none"` where its returns are read; a search that did not maximise the Sharpe, or that asked one trial, for the deflated Sharpe; a null as a summary refuses it, or with fewer than 2 trials that ran; a count of trials below 2; a spread below 0; a Sharpe not above 0 for the backtest length; for the overfitting probability, an objective the halves cannot read whatever the order of their returns, fewer than 2 complete trials or fewer than 16 returns; and for the Reality Check, fewer than 3 returns, every trial having lost all of a bar, a benchmark text other than `"buy_and_hold"`, a benchmark return at or below -100%, naming its position, buying and holding beside a matrix, and a series of another length.

```python
monthly = [0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01]

interval = probatum.stats.sharpe_interval(monthly, periods_per_year=12)
print(f"{interval.estimate:.2f} from {interval.low:.2f} to {interval.high:.2f}, lag "
      f"{interval.lag}, floored {interval.floored}")

print(round(probatum.stats.probabilistic_sharpe(monthly, periods_per_year=12).value, 3))
print(round(probatum.stats.min_track_record_length(monthly, periods_per_year=12).years, 1))
print(round(probatum.stats.sharpe(monthly, periods_per_year=12, long_run=True), 2))

print(probatum.stats.sharpe_interval(result).count)   # the Quickstart's 999 returns
print(probatum.stats.bootstrap(result, statistic="cagr_pct").to_pandas()[
    ["estimate", "low", "high", "block"]])
print(probatum.stats.sharpe_difference(result).to_pandas()[["low", "high", "p"]])
print(probatum.stats.alpha_beta(result).to_pandas()[["alpha", "beta", "information_ratio"]])
```

```text
1.73 from -0.92 to 4.39, lag 2, floored True
0.899
1.0
2.95
999
          estimate       low      high     block
cagr_pct -0.420308 -0.951354  0.093712  2.616038
                low      high         p
delta     -0.617124  1.211760  0.523959
bootstrap -0.605106  1.199742  0.525237
              alpha      beta  information_ratio
AlphaBeta  0.000215  0.000856           4.899349
```

<br>

## Statistics of a run

`probatum.stats` also describes a run: how far and how long it fell, how its Sharpe and its volatility moved, what each calendar period returned, the shape of its returns, its round trips, another series held beside it, and where its money went ([Specification](Specification.md#statistics-of-a-run)). The drawdown, rolling and shape statistics take a Result or a plain series of returns, as [Inference](#inference) does; a plain series needs `periods_per_year` only where the statistic is annualised, and its rows are its path's from 1, its return at position `i` on row `i + 1`. The rest read what only a Result holds. None of them has an interval.

| Call | Gives |
| :--- | :--- |
| `probatum.stats.drawdown(returns)` | the fall from the running peak in percent, a Result's at each row, a plain series' at each return; its largest is the record's `max_drawdown_pct` |
| `probatum.stats.drawdown_table(returns, top=5)` | a table of the `top` deepest drawdowns, deepest first: `peak_bar`, `trough_bar`, `recovered_bar`, missing while still below, a Result's `peak_time`, `trough_time` and `recovered_time`, `depth_pct`, `bars_to_trough` and `bars_under`, from the peak to the recovery or the last row |
| `probatum.stats.time_under_water(returns)` | an `UnderWater`: the longest and the mean drawdown in rows from peak to recovery, their count, the share of rows below their peak, and a Result's longest and mean in days |
| `probatum.stats.ulcer_index(returns)` | the root mean square of the drawdown, in percent |
| `probatum.stats.rolling_sharpe(returns, window, periods_per_year=None, risk_free=None)` | the record's Sharpe of the `window` returns ending at each row, NaN before the first whole window, a Result's a value a row and a plain series' a value a return |
| `probatum.stats.rolling_volatility(returns, window, periods_per_year=None)` | the volatility of the same windows, in percent |
| `probatum.stats.period_returns(result, by="month")` | a table of a row an `"hour"`, a `"day"`, a `"week"` from a Monday, or a calendar `"month"`, `"quarter"` or `"year"` in UTC: `period`, `first_bar`, `rows`, `start`, `end` and `return_pct`, which compound to the run |
| `probatum.stats.segment(result, start=0, stop=None)` | the record's `Statistics` over rows `start` to `stop - 1`, from the equity carried into them, with intervals of their own; consecutive segments compound to the run |
| `probatum.stats.skew(returns)`, `probatum.stats.kurtosis(returns, excess=False)` | the returns' third and fourth standardised moments, a normal series' kurtosis 3, or 0 with `excess` |
| `probatum.stats.omega(returns, threshold=0.0, periods_per_year=None)` | the gains past an annual threshold over the losses short of it |
| `probatum.stats.value_at_risk(returns, alpha=0.95)`, `probatum.stats.conditional_value_at_risk(returns, alpha=0.95)` | the loss of the `k`-th lowest return and the mean loss of the `k` lowest, in percent, `k = ceil(n * (1 - alpha))`, at 0.95 the ones their intervals read |
| `probatum.stats.tail_ratio(returns, alpha=0.95)` | the `k`-th highest return over the loss of the `k`-th lowest |
| `probatum.stats.trade_stats(result)` | a `TradeStats` of the closed round trips in the order they closed |
| `probatum.stats.long_short_split(result)` | a `Sides`, the `TradeStats` of the long round trips and of the short as `long` and `short` |
| `probatum.stats.session_buckets(result, by="hour")` | a table of the closed round trips booked at the UTC hour, or with `by="weekday"` the day from Monday 0, of the row each closed on: `bucket`, `trades`, `pnl` and `win_rate` |
| `probatum.stats.benchmark(result, prices)` | a `Benchmark`, the run beside holding another series of one price a row |
| `probatum.stats.decompose(result)` | a `Decomposition`, the change in equity split into the price pnl and each cost, with the costs' share of the price pnl and the turnover |
| `probatum.compare(results, keys=None, reference=None)` | a table of a row a run, a list's named by their venues, a mapping's by their keys: `name`; with `reference`, the name of one of them, each run's Sharpe tested against the reference's on the same rows, `sharpe_difference`, `difference_low`, `difference_high`, `p_delta` and `p_bootstrap`; the statistics, every one or those `keys` names; then `venue`, `policy`, `data`, the fingerprint, and `rows` |
| `study.importance()`, `probatum.stats.importance(study)` | an `Importance`, each parameter's share of the variation in a search's scores its own values explain, eta squared, beside the share chance gives ([0047](Decisions.md)) |

**What comes back.** Each never changes, compares by what it holds, and shows as the frame its `to_pandas()` gives, and each table is one of a Result's, which any Arrow reader takes:

| Result | Holds |
| :--- | :--- |
| `UnderWater` | `longest`, `mean`, `episodes`, `share`, `longest_days` and `mean_days`, the last two None for a plain series |
| `TradeStats` | `trades`, `wins`, `losses`, `avg_win`, `avg_loss`, `payoff`, `expectancy`, `expectancy_pct`, `largest_win`, `largest_loss`, `max_consecutive_losses` and `avg_bars_held` |
| `Sides` | `long` and `short`, a `TradeStats` each |
| `Benchmark` | `hold_return_pct`, `hold_sharpe`, `excess_return_pct`, `beta` and `information_ratio` |
| `Decomposition` | `price`, the seven costs, `costs`, their sum, `net`, `net_pct`, `cost_share` and `turnover`, NaN for a run kept at the account level |
| `Importance` | `names`, `importance`, `chance`, `groups` and `trials` |

**What is refused.**
* As a `ProbatumTypeError`: a plain series where only a Result is read; a `window`, a `top`, a `start` or a `stop` that is no whole number; and a `by` that is no string.
* As a `ProbatumValueError`: a `window` below 2; a `top` below 1; a stretch that holds no return; a `by` other than those named; a tail of no returns; prices of another length, or not finite and above zero; and a Study with fewer than 2 complete trials whose score is finite.
* As a `DataError`: a price Series not indexed by the run's row times.

```python
print(probatum.stats.drawdown_table(result, top=3).to_pandas()[
    ["peak_bar", "trough_bar", "recovered_bar", "depth_pct", "bars_under"]])
print(probatum.stats.time_under_water(result))

print(probatum.stats.benchmark(result, prices=close).to_pandas()[
    ["hold_return_pct", "excess_return_pct", "beta"]])

monthly = [0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01]
print(probatum.stats.rolling_sharpe(monthly, window=4, periods_per_year=12).round(decimals=2))
```

```text
   peak_bar  trough_bar  recovered_bar  depth_pct  bars_under
0       144         994           <NA>   0.061405         855
1         7          17             47   0.007933          40
2       122         130            144   0.006136          22
            longest       mean  episodes    share  longest_days  mean_days
UnderWater      855  89.909091        11  0.97998        35.625   3.746212
           hold_return_pct  excess_return_pct      beta
Benchmark       -48.093539          48.045517  0.000855
[ nan  nan  nan 1.9  1.52 0.83 1.04 1.41]
```

<br>

## Capacity and costs

`probatum.capacity(policy, data, venue, balances, sizing="weight", stop=None, take_profit=None, tolerance=0.0, risk_free=0.0, spec=None, features=None, decode=None, instruments=None)`, `probatum.cost_curve(policy, data, venue, scales=(0.0, 0.5, 1.0, 2.0, 4.0), ...)` and `probatum.breakeven_bps(policy, data, venue, ceiling=100.0, precision=0.05, ...)` run a policy again where only its balance or its costs differ ([0045](Decisions.md), [Specification](Specification.md#capacity-and-costs)). Each run is the `backtest` of the policy with the settings given, a Strategy copied afresh for each, and each function takes an agent's settings as `backtest` does ([An agent](#an-agent)).

- **Capacity.** Each of `balances` opens the account as `Venue.replace(balance=...)` would, a number or `{"spot": s, "margin": m}`, and one the account cannot open with, a single wallet's where the data trades spot and perpetuals or two for an inverse contract, is refused before any run. The `Capacity` it gives holds, a run a balance, `balances`, `total_return_pct`, `sharpe`, `impact` paid, `held`, the share of its orders the liquidity budget held, `cost_bps`, the round-trip cost its fills paid in basis points, and `turnover`, a year of what its fills traded over its mean equity, and shows as its frame, a row a balance. A weight grows with the balance and a quantity does not, so `sizing="quantity"` is refused. Once the budget holds most of its orders, a book trades another, slower path, and its return can move either way.
- **Cost sweeps.** Each of `scales` multiplies together every cost the venue charges for trading: each fee rate above 0, maker and taker, the spread, fixed or estimated, the slippage and the impact coefficient. A rebate, a fee rate below 0, the liquidity budget, funding and a liquidation fee the venue sets stay; one it does not set is the taker rate, and moves as it does. The `CostCurve` holds, a run a scale, `scales`, `cost_bps`, `total_return_pct`, `sharpe`, `num_trades`, `fees`, `spread`, `slippage`, `impact` and `turnover`, and shows as its frame, a row a scale. A venue that charges none of the four on the data's instruments is refused.
- **The break-even.** `breakeven_bps` bisects the scale from 0 to `ceiling` until the two runs that bracket the return's zero pay within `precision` basis points a round trip of each other, and gives their mean, to set beside what the venue charges at scale 1. It warns where no scale between them brings them that close, as where a cost jumps at the crossing, and gives the lower one's cost where the upper filled nothing. It is None where the policy makes nothing with no costs at all, and the cost at `ceiling`, with a warning, where the policy still makes money there. The costs change which fills happen, so it finds a crossing rather than the crossing.

```python
fast, slow = probatum.ta.ema(data.close, length=24), probatum.ta.ema(data.close, length=96)
trend = numpy.where(numpy.isnan(slow), numpy.nan, numpy.where(fast > slow, 1.0, -1.0))

venue = probatum.Venue(fees={"maker": 0.0002, "taker": 0.0005},
                       fills={"spread": {"fixed": 2.0}, "slippage": {"fixed": 1.0},
                              "impact": {"sqrt": 0.1}, "liquidity": {"budget": 0.1}})

print(probatum.capacity(trend, data=data, venue=venue, balances=[10_000.0, 100_000.0]))

print(probatum.cost_curve(trend, data=data, venue=venue))     # 18.7 bps a round trip at scale 1
print(probatum.breakeven_bps(trend, data=data, venue=venue))  # where it stops making money
```

<br>

## Indicators

30 functions, one value a bar, each with its conventions written down.

```python
class Cross(probatum.Strategy):
    def init(self, data):
        self.fast = probatum.ta.ema(data.close, length=20)
        self.slow = probatum.ta.ema(data.close, length=60)
        self.warmup = 60

    def next(self, bar, account):
        if account.position == 0.0 and self.fast[bar.i] > self.slow[bar.i]:
            account.buy(size=1.0)
        elif account.position > 0.0 and self.fast[bar.i] < self.slow[bar.i]:
            account.close()


result = probatum.backtest(Cross(), data=data, venue="frictionless")
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

**`shift`, `crossover` and `crossunder` look back without an index.** `shift(values, n)` is how you look back. Written by hand, `values[i - n]` on an early bar is a negative index, which numpy resolves from the end of the array, so the rule reads a price from the far future and nothing raises. `Strategy.warmup` ([0025](Decisions.md)) closes that only when the declared warm-up is at least as deep as the lookback; `shift` pads rather than wraps, so there is no index to get wrong. On a universe an index also counts rows, so the row before an instrument's bar can be one it printed nothing on, while `shift` lags each instrument over its own bars. `warmup` counts rows too, so `next` can run while an instrument that lists late is still in its own warm-up.

`crossover` and `crossunder` return **booleans**, and they are the only functions here that do not pad with `NaN`. `bool(nan)` is `True` in Python, so a float warm-up would fire a rule on the very bars where nothing is known; they pad with `False` instead, and a cross across a bar either of the two lacks is reported on the first bar after it that both have ([0037](Decisions.md)). `other` may be a second series of the same shape or a plain number, since crossing a level is the commoner case:

```python
class Cross(probatum.Strategy):
    def init(self, data):
        self.up = probatum.ta.crossover(probatum.ta.ema(data.close, length=20),
                                        other=probatum.ta.ema(data.close, length=60))
        self.strong = probatum.ta.adx(data.high, low=data.low, close=data.close).adx
        self.warmup = 80

    def next(self, bar, account):
        if self.up[bar.i] and self.strong[bar.i] > 25.0:      # a cross, in a trend
            account.target(weights=1.0)


result = probatum.backtest(Cross(), data=data, venue="frictionless")
```

**Four rules hold for every one of them.**

*One value a bar, each instrument over its own.* Every function takes a series `(T,)`, or a universe `(T, N)` with a column for each instrument, time along `axis`, and returns a float64 array of the same shape, aligned so entry `i` is bar `i`, with the warm-up as `NaN` and never a shorter array. Each column runs over its own values. A value that is `NaN` or infinite is a row the column does not have and a gap in the output, and a window, a smoother, a lookback and a crossing carry across it rather than restarting. A Data's fields are `NaN` exactly where an instrument printed no bar, so a closure, a halt or a late listing costs an indicator only those rows.

A series too short for its warm-up is all warm-up rather than an error, and only a series of no bars is refused, so a cut of a series that has a bar gives the first rows of the whole. `funding_rate` and what `asof` puts on the clock hold their last value through a closure, so an indicator runs over those rows as well. Where an instrument's own bars are meant, mask a `(T, N)` series with `numpy.where(data.available, x, numpy.nan)`, and one instrument's `(T,)` line with `data.available[:, 0]`. A plain two-dimensional array is a column each, so a `(T, 5)` frame comes back as five series, and an `(N, T)` array takes `axis=1`. The same object goes into your rule and into [`probatum.chart`](#charts), a column at a time, with no padding decision in between ([0037](Decisions.md)).

```python
gapped = bars.copy()
gapped[500:510] = numpy.nan                      # ETHUSDT prints nothing for ten hours
pair = probatum.Data(numpy.stack([bars, gapped], axis=1), symbol=["BTCUSDT", "ETHUSDT"],
                     market="linear", time=hours)

trend = probatum.ta.ema(pair.closes, length=50)  # (1000, 2), each over its own bars
print(trend[505], trend[510])                    # NaN in ETHUSDT's hole, then carried on
```

*The convention is written down.* `ema` is smoothed at `2 / (length + 1)` and seeded from the simple average of the first `length` values rather than the first value, which is TradingView's convention and stops one opening print steering the curve for hundreds of bars. `rsi`, `atr` and `adx` use Wilder's `1 / length`, as those indicators were defined. `stdev` is the population deviation, dividing by `length`, because that is what `bbands` is defined against. `cci` divides by the mean **absolute** deviation, not the standard deviation, which is its definition and makes it behave differently from `zscore` on a fat-tailed series. `hma` floors both of its inner lengths to whole bars. Each function repeats its own choice in its docstring.

*A gap where there is no value.* A window with no range has no position inside it, so `stoch` and `willr` are a gap there rather than fifty, and `stoch`'s `d` averages over the bars `k` has a value on. A window that traded nothing has no weighted price, so `vwap` and `vwma` are a gap rather than an unweighted average. A market with neither gains nor losses has no ratio of strength to weakness, so `rsi` and `mfi` are a gap rather than the 100 some implementations report, which would fire an overbought rule on a market that has not moved. A `NaN` in an array of your own is a gap like any other, so a signal whose `NaN` means something else is lagged by rows with numpy rather than with `shift`.

*Nothing knows about the engine.* These are functions of arrays, with no state, no engine and no pandas needed. Computing once in `init` and handing the same array to the chart keeps the line on screen from drifting from the line that made the decision ([0042](Decisions.md)).

Functions with more than one output return a named object rather than a tuple, so a caller cannot unpack three anonymous arrays in the wrong order. There are five: `Bands` (`.upper`, `.middle`, `.lower`) from `bbands`, `keltner` and `donchian`; `Macd` (`.line`, `.signal`, `.histogram`); `Stochastic` (`.k`, `.d`); `Trend` (`.adx`, `.plus`, `.minus`); and `Channel` from `supertrend`, which adds `.direction` to the three band lines. Read `.direction`, `1.0` while long and `-1.0` while short, rather than comparing the close against the line: the line is the edge in force, and that comparison is what set it.

`vwap` is rolling over `length` bars rather than anchored to a session, because the engine has bars and no notion of a trading day. `donchian`, `highest` and `lowest` all include the current bar, so a breakout of `upper` cannot happen on the bar that set it; compare against `shift(highest(...), 1)` for that. `willr` is `stoch`'s `k` read from the top of the range rather than the bottom, so it runs from -100 to 0 where `k` runs from 0 to 100; both are here because both sign conventions are in wide use.

<br>

## Charts

`result.chart()` draws a run and `data.chart()` draws bars, each with arrays of your own beside them, as a `probatum.chart.Chart`. `show()` puts it in the notebook cell, where it is stored in the `.ipynb`, so the notebook reopened shows it with no kernel running; `save(path)` writes one HTML file that opens in a browser offline; and `spec()` gives the underlying document.

It computes no indicator and knows no indicator's name: every number on it is the Data's, the Result's or what `probatum.stats` reads of the Result, and you bring the arrays ([0042](Decisions.md)). Everything is computed before it is drawn, which lets a chart outlive the process that made it. It needs numpy and the standard library alone: pandas is read by duck typing, and IPython is imported only inside `show`.

```python
import probatum
from probatum.chart.marks import Band, Level, Line, Panel

fast, slow = probatum.ta.ema(data.close, length=20), probatum.ta.ema(data.close, length=60)
rsi = probatum.ta.rsi(data.close, length=14)
result = probatum.backtest(numpy.where(fast > slow, 1.0, 0.0),
                           data=data, venue="frictionless")

result.chart(
    Line(fast, name="EMA 20"),                           # the arrays the run decided on
    Line(slow, name="EMA 60", style="dashed"),
    Line(rsi, name="RSI 14", panel="momentum"),          # a name not yet used makes a panel
    Level(value=70, panel="momentum", style="dotted"),
    Band(rsi, lower=70, only="above", panel="momentum",  # shaded only where it is past 70
         fill=("#ff547000", "#ff547059")),
    panels=[Panel(name="momentum", weight=1.6, range=(0, 100))],
    title="EMA cross 20/60",
).show()
```

`result.chart(*marks, instrument=None, data=None, panels=None, focus=None, candle_color=None, theme=None, palette=None, height=None, title=None, future=0, stats=None, trades=True, drawdown=None, notes=None, price=None)` draws the candles of `instrument`, or of the run's one instrument where it holds one. Beside them it draws the round trips the run made on that instrument, a trip still open drawn to the last bar, the account's equity with its drawdown shaded under it, four statistics above the plot and the trade log under it. A Result loaded, unpickled or recorded below the full level keeps no Data, so `data=` gives it, held to the fingerprint the run recorded, and a book is refused. A bar an instrument printed none on keeps its slot with no candle.

`data.chart(*marks, instrument=None, panels=None, focus=None, candle_color=None, theme=None, palette=None, height=None, title=None, future=0, notes=None, price=None)` takes the keywords that read no run. `probatum.chart.defaults(theme=None, height=None, palette=None, drawdown=None, price=None)` sets the look of every later chart and returns the settings before.

A mark is one of `probatum.chart.marks`, or an array drawn as a `Line`, and a list of them is flattened. Arrays are read by position, one value a bar of the Data: length `T` maps entry `i` to bar `i`, and length `T - 1` maps entry `i` to bar `i + 1`, as a diff does; any other length raises and names both numbers ([0042](Decisions.md)). On a run that did not record every row of the Data, an array as long as the rows it did record is refused, and so is a diff one shorter, since neither can be placed by its length ([Charts](Charts.md#the-length-contract)). A `NaN` is drawn as a gap rather than a dropped row ([0042](Decisions.md)). `Recorder(data)` collects values inside `next` with their alignment declared, read back off `result.policy`.

The full contract, the mark reference, the padding helpers for values recorded inside `next`, and what cannot be charted are in the [Charts](Charts.md) page.

<br>

## Errors and warnings

Every error probatum raises is one of its own classes and the built-in it means, so `except ValueError` still catches a bad value and `except probatum.ProbatumError` catches probatum's errors and nothing else.

| Class | Built-in | What it covers |
| :--- | :--- | :--- |
| `probatum.ProbatumError` | `Exception` | Every class below. |
| `probatum.ProbatumValueError` | `ValueError` | A wrong value, the four kinds below among them. |
| `probatum.DataError` | `ValueError` | A wrong value in bars, frames, the clock, features, funding or a series an indicator reads. |
| `probatum.VenueError` | `ValueError` | A wrong value in a venue's settings, and any wrong setting in a preset's file. |
| `probatum.SpecError` | `ValueError` | An agent whose spec does not match the run it is given: one that carries no spec, a spec of another schema, a Data of other instruments, and a feature function or a decode that is not the one it trained with. |
| `probatum.LookaheadError` | `ValueError` | A policy the causality check of `search` or `validate` finds deciding on bars that had not closed, or a model's features, predictions or `to_weights` of them that `validate` finds reading them. |
| `probatum.ProbatumTypeError` | `TypeError` | An argument of the wrong type, `PolicyError` among them. |
| `probatum.PolicyError` | `TypeError` | A policy `backtest`, `search` or `validate` cannot run as given: a class for an instance, and for a search or a validation an instance for its class, a class that is no Strategy or a function that takes no data, a model that fits, an object of no kind it reads, an array it cannot read as decisions, a trial that decides in another kind than the trials before it, and a policy the workers cannot carry. |
| `probatum.ProbatumKeyError` | `KeyError` | A name none of a set's holds, such as an instrument a Data does not have. |
| `probatum.ProbatumIndexError` | `IndexError` | A position past the end of a table. |
| `probatum.ProbatumAttributeError` | `AttributeError` | An attribute an object does not have, such as one instrument's `close` on a universe, which `hasattr` reads as missing. |
| `probatum.ProbatumRuntimeError` | `RuntimeError` | A search or a validation that cannot go on: its workers lost, every trial of a search or every path of a validation failed, or its winner run again not matching its trial; an environment stepped before any reset, or given a `reset_mask` before every environment has had an episode; a chart's `show()` where there is no notebook to draw it in; and `ProbatumNotImplementedError`, as `NotImplementedError` is a `RuntimeError`. |
| `probatum.ProbatumNotImplementedError` | `NotImplementedError` | A Strategy that writes no `next`. |
| `probatum.MissingDependency` | `ImportError` | A package a call needs that is not installed, named with the line that installs it. |

A refusal of a known kind raises that kind, where both its topic and its built-in are the kind's, and any other refusal the class for its built-in. The built-in follows what arrived: a value of a type the argument does not take raises a `TypeError`, and one of its type that the call cannot take a `ValueError`, so `tolerance="0.01"` is a `ProbatumTypeError` and `tolerance=-0.01` a `ProbatumValueError`. A Data's wrong value is then a `DataError` and its wrong type a `ProbatumTypeError`, which `except probatum.DataError` does not catch, and a Venue built in Python splits the same way between `VenueError` and `ProbatumTypeError`, while every wrong setting in a preset's file is a `VenueError`. A policy's wrong value is a `ProbatumValueError`, not a `PolicyError`. A symbol the router fails on in `Data.from_router` raises the router's own error, as a pull of that symbol alone does, so `except ValueError` does not catch a failed batch.

What another library raises stays its own: the router's errors, the file system's from `save` and `load`, numpy's for values an indicator, `asof` or `features` cannot read as numbers, Python's `OverflowError` for an integer no float holds, and whatever your own code raises inside a policy. The binding's raw objects, `probatum.Batch` and those of `probatum._probatum`, refuse a wrong value with the same classes. What pyo3 converts itself, an argument of a type their signature cannot take or a part of one it cannot unpack, it refuses as a `TypeError`, an `OverflowError` or a `ValueError`, with a note naming an argument the signature declares, and a call from a second thread into an object another call holds as a `RuntimeError`. A package a module needs and does not find, as `probatum.sb3` needs stable-baselines3, is a `MissingDependency` when the module is first imported.

Every warning probatum gives is a `probatum.ProbatumWarning`, a `UserWarning`, named at the line of your code that caused it, so a test can make probatum's warnings errors without touching anyone else's. probatum prints nothing. Long work is logged to the `probatum` logger, `Data.from_router` logging each instrument it built at INFO and each call to the router at DEBUG, which Python's `logging` turns on.

```python
import logging
import warnings

try:
    probatum.backtest(Hundred, data=data, venue="frictionless")  # the class, not an instance
except probatum.PolicyError as error:
    print(error)            # policy: got the class Hundred, not an instance; pass Hundred()

try:
    data["SOLUSDT"]
except KeyError as error:                                 # a ProbatumKeyError is one
    print(type(error).__name__, error)

logging.basicConfig(level=logging.INFO)                   # what long work reports
warnings.simplefilter(action="error",
                      category=probatum.ProbatumWarning)  # probatum's warnings as errors
```
