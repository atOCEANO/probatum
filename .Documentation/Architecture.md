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
  <a href="Python_API.md">Python API</a> &nbsp;•&nbsp;
  <a href="Specification.md">Specification</a> &nbsp;•&nbsp;
  <a href="Venue_Notes.md">Venue Notes</a> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <b>Architecture</b> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Architecture

A cargo workspace of three crates, exposed to Python through one extension module, `probatum._probatum`, under the Python package `probatum`. Each layer builds only on the one beneath it, and only the crate that links Python knows about numpy, Arrow's capsules or the GIL.

| Layer | Holds | Knows nothing about |
| :--- | :--- | :--- |
| `probatum-core` | units, orders, fills, the netted position, the account and its two wallets, the fee schedule, tier tables, the fixed-slot order book, and a table of named columns with its Arrow export | candle series, time, Python |
| `probatum-engine` | the candle series and its clock, the step, order resolution, the fill model and its window, the liquidation path, funding, the ledger, the array run, the batch, the environment and labels | Python |
| `probatum-py` | argument checks at the boundary, the buffers and views handed to numpy, a frame's columns read through the Arrow C interfaces, a run's tables built in Rust and lent as Arrow capsules, and the low-level binding | strategies, statistics |
| `probatum` (Python) | everything a user calls: `Data`, `Venue`, `backtest`, `Strategy`, `Engine`, `Result`, `search`, `validate`, `labels`, `cv`, `stats`, `ta`, `env`, `chart` | how a fill is priced |

<br>

<div align="center">
  <img src="imgs/205312.png" alt="Crate layering" width="100%" />
  <p style="margin: 0;"><i>The dependency order is bottom-up: <code>probatum-core</code>, then <code>probatum-engine</code>, then the one Python-linked crate <code>probatum-py</code>, then the Python package a user calls.</i></p>
</div>

<br>
<br>

## Crates

`probatum-core` holds no candle series and no bar-level rule: the position, the account, the costs and the book are the same however a fill is resolved. The account settles each instrument in its spot wallet or its margin wallet and nets a position per instrument. It marks equity as both wallets' cash, with every spot holding at its close and every perpetual's unrealized PnL beside it. Its Arrow table is lent through the C stream interface and released by whoever reads it, so it lives where Miri runs ([0026](Decisions.md)).

`probatum-engine` is the bar-level engine. `engine/` holds the step and its parts, one module each, and the modules beside it hold the bars and price the fills:

| Module | What it does |
| :--- | :--- |
| `engine/orders.rs` | places, moves and cancels orders |
| `engine/resolution.rs` | resolves a bar's orders |
| `engine/margin.rs` | holds a fill to the cap, to a spot wallet's cash and holding, and to a reduce-only order's position |
| `engine/path.rs`, `engine/liquidation.rs` | solve where the margin runs out, close the account there, and hold every reducing fill to the margin bound |
| `engine/funding.rs`, `engine/transfers.rs`, `engine/brackets.rs`, `engine/target.rs` | funding, transfers, brackets and targets |
| `engine/ledger.rs` | books every fill |
| `candles.rs`, `clock.rs` | hold the bars of one instrument or a universe on one clock |
| `execution.rs`, `window.rs` | price a fill |
| `schedule.rs` | places funding events |
| `sizes.rs` | holds an order to its lot and price rule |
| `inverse.rs` | turns an inverse contract into a linear one in the reciprocal of the price ([0016](Decisions.md)) |
| `reporter.rs`, `stats.rs` | the records a run keeps when it records, and the statistics derived from them |
| `array.rs` | runs an array whole |
| `script.rs`, `strategy.rs` | a script of decisions, and the loop it and an array run are played through |
| `sweep.rs` | runs a compiled strategy over many parameter rows in parallel |
| `batch.rs`, `env.rs` | step many accounts at once |
| `labels.rs` | runs one trade a row |
| `streams.rs` | names every random stream |

`probatum-py` is a thin shell over the engine and holds no simulation logic. It checks every argument as it enters and copies candles once from numpy. It reads a frame's columns through the Arrow C stream or array interface, so no frame library is imported to read one, builds a run's tables from the engine's records with no Python object a row, and releases the GIL around every batch step. Its low-level binding, `probatum._probatum.run`, takes candles, a venue record and a script of decisions, and returns the ledger, the fill log, the order log and the round trips. The reference simulator and the tests drive the engine through it ([Specification](Specification.md#the-low-level-binding)).

The Python package simulates nothing:

| Module | What it holds |
| :--- | :--- |
| `_data.py` | `Data` |
| `_venue.py`, `venues/` | the `Venue`, and the presets with their loader |
| `_backtest.py` | reads a policy into one of its depths |
| `_strategy.py`, `_engine.py` | drive the account |
| `_search.py`, `_samplers.py`, `_space.py`, `_workers.py`, `_causality.py` | the search |
| `cv.py`, `_validate.py`, `_labels.py` | validation |
| `_costs.py` | capacity and the cost sweeps |
| `_result.py`, `_ipc.py`, `_storage.py` | what a run and a search keep |
| `_assumptions.py`, `_baselines.py`, `_explain.py` | what a run assumed, what it sets itself beside, and how a fill is told |
| `_frames.py` | how a run's tables and records show |
| `stats.py` | the statistics |
| `ta.py` | the indicators |
| `_env.py`, `rl.py`, `_agent.py`, `_transitions.py`, `sb3.py`, `torch.py` | the environments |
| `_streams.py` | the named random streams |
| `_errors.py` | the errors |
| `chart/`, with `_static/` | the chart |

`stats.py` and `chart/` read a finished run, so a defect there is a wrong number about a correct run.

<br>
<br>

## The step

`step()` advances exactly one bar, through the nine events the [Specification](Specification.md#the-order-of-events) numbers, drawn below.

<br>

<div align="center">
  <img src="imgs/205314.png" alt="The order of events within one step" width="25%" />
  <p style="margin: 0;"><i>The events resolved within one <code>step()</code>, in the order the Specification gives them.</i></p>
</div>

<br>

An order decided at bar `t`'s close fills on bar `t + 1` at the earliest, never on the bar the decision saw. The liquidation is a price on the bar's path rather than a check at its end, so the fills after it read bars clipped where the account left the market ([0017](Decisions.md), [0018](Decisions.md)).

<br>
<br>

## Batching and the environment

The candle series lives once behind an `Arc`, so thousands of accounts read one buffer and each carries only its own account, orders and cursor. `Batch.step_all` steps every account in parallel with the GIL released. No account reads another, so the batch is bit-identical to stepping each alone, except under `intrabar="random"`, where each account also draws from a stream named with its index ([0021](Decisions.md)).

`probatum.env` builds its environments on the same engines. A step decodes every environment's action, steps every account, works out its reward and its flags, and writes every observation into one float32 buffer, on threads sized to the work so a small batch does not pay for a large pool ([0040](Decisions.md)). `copy=False` hands out that buffer rather than a copy of it, and `probatum.torch.TensorEnv` hands it to a device, or writes the market block on the device itself.

An engine, a batch or an environment clones. An engine or a batch also pickles into another process of the same version, through serde and postcard, and another version refuses the bytes. An environment holds the functions it was given and does not pickle.

<br>
<br>

## What a run records

The compiled engine keeps only its running state unless it records, so its memory does not grow with the run. `backtest` records at the full level unless `record=` asks otherwise, `probatum.Engine` always records at it, and an environment records nothing unless asked. The full level keeps the ledger, a row per bar from the opening balance, the holdings of each instrument, the round trips, a log of every fill and a log of every order's events. The account level keeps the ledger, the round trips and the costs alone, and a search's array trials keep only what their statistics read ([0022](Decisions.md), [0023](Decisions.md)).

The engine derives the statistics a search ranks on from the equity and the round trips. Their intervals are `probatum.stats`', computed in Python when first read and never inside a trial ([0031](Decisions.md)). Every transcendental function goes through `libm`, sums run in a fixed order and every random draw comes from a named stream, so a run gives the same bits on every supported machine ([0021](Decisions.md)).

<br>
<br>

## Speed

Each depth spends its time in a different place, which decides what to reach for:

| Path | What bounds it |
| :--- | :--- |
| an array through `backtest` | the engine's step alone, whole in Rust, and building the run's tables |
| a `Strategy` through `backtest` | the call into `next` on every bar and the views it reads |
| `probatum.Engine` | the call into `step` on every bar |
| a random or grid search of an array | the engines stepped together in Rust, many trials to a call |
| a search of a `Strategy` | one trial at a time in each worker process |
| `probatum.env` | writing the observation, with the GIL released |

`benchmarks/surfaces.py` measures each of them on the machine it runs on, in the `bench-surfaces` stage the [Validation Guide](Validation_Guide.md#the-stages) gives. CI holds the step loop's instruction count, which no machine moves, against the commit before.

<br>
<br>

## Versioning Policy

A backtest is a number somebody acts on, so the simulator's behaviour is part of the API ([0001](Decisions.md)). From 2.0.0 a public name, a keyword's meaning, a return type or a rule the [Specification](Specification.md) states changes only in a major release, with a record in [Decisions](Decisions.md) saying what was wrong. A minor release adds, and a keyword it adds has a default that keeps the old behaviour.

Not promised: anything whose name starts with an underscore, the compiled binding `probatum._probatum` among them; the wording of an error or a warning; the chart's document; an indicator's last bit; and the Rust crates, which are workspace members rather than a published API.
