<h1>OCEΛNO <small><code>embeddable-market-simulation-library</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/embeddable-market-simulation-library/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/embeddable-market-simulation-library?label=release&color=2ea043" alt="Latest release" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <a href="../README.md">Introduction</a> &nbsp;•&nbsp;
  <a href="Python_API.md">Python API</a> &nbsp;•&nbsp;
  <a href="Specification.md">Specification</a> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Plotting.md">Plotting</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <b>Contributor Guide</b> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Contributor Guide

The library is a cargo workspace of three crates plus a thin Python package. The one rule that shapes every change is that **logic flows up, never sideways**: each layer builds only on the one below it, and Python-specific concerns live only in the crate that links Python. Get the layer right and the change is small; put it in the wrong layer and it fights the design.

<br>

### Where a change belongs

| If it is... | It goes in... |
| :--- | :--- |
| A fidelity-neutral primitive (a unit, an order, position or account math, a cost model) | `crates/probatum-core` |
| Memory a C interface lends out and takes back (the Arrow export and its release) | `crates/probatum-core`, where Miri runs it |
| A bar-level behavior (fills, the step machine, the reporter, batched stepping) | `crates/probatum-engine` |
| A Python binding (a pyclass method, an argument type, a numpy bridge) | `crates/probatum-py` |
| A convenience over the bindings (a new runner, a wrapper) | `python/probatum` |
| A statistic or an interval read off a finished run, or a test across a search's trials | `python/probatum/stats.py`, and the same rule in plain loops in `tests/reference/statistics.py` |
| How a table, a record or a summary shows | `python/probatum/_frames.py` for how a frame is laid out with what holds for every row under it, each record's own frame in `stats.py` and the summaries' rows in `_result.py` and `_search.py` ([ADR 0041](Decisions.md)) |
| A label's trade, or how a function or a model is validated | `crates/probatum-engine/src/labels.rs` for the trade, `python/probatum/_labels.py` and `_validate.py` for the rest, and the reference's labels in `tests/reference/simulator.py` |
| Capacity, a cost sweep or its break-even | `python/probatum/_costs.py`, the costs a sweep multiplies in `_venue.py`, and a scaled spread estimate in `crates/probatum-engine/src/execution.rs` |
| How a Study is saved and loaded, or how a search keeps its storage and resumes from it | `python/probatum/_storage.py`, and the loop that takes a stored trial in its turn in `_search.py` |
| The environment an agent trains on | `crates/probatum-engine/src/env.rs` for what each step works out, `python/probatum/_env.py` for the episodes drawn, the spaces and Gymnasium's API, and the reference's environment in `tests/reference/environment.py` |
| What an agent's spec records, its probes and its normalisation statistics | `python/probatum/_agent.py`, the scaling each observation is written with in `crates/probatum-engine/src/env.rs`, and the same rules in plain loops in `tests/reference/environment.py` |
| How an environment's tensors reach a device, or the resident mode's market block | `python/probatum/torch.py`, and the account block alone and the market's cells it reads in `crates/probatum-engine/src/env.rs` |
| How an engine, a batch or an environment clones or pickles | the bytes and the forks in `crates/probatum-engine/src/engine/mod.rs`, `batch.rs` and `env.rs`, the pickles in `crates/probatum-py/src/lib.rs`, and the Python state they carry in `_engine.py`, `_strategy.py` and `_env.py` |
| How transitions are recorded or exported | `python/probatum/_transitions.py` for the rows and the export, the recording in `_env.py`, the Result's in `_result.py`, and the Arrow lists in `_ipc.py` |
| How a backtest runs an agent, or how the spec travels with it | `python/probatum/_agent.py` for the spec found, the slots, the warnings and `save`, the episode it plays in `_env.py`, and the SB3 adapter and its callback in `sb3.py` |
| An indicator, which is a function of arrays and never sees the engine | `python/probatum/ta.py` |
| Anything a chart draws (a mark, a guard, a panel rule) | `python/probatum/chart/__init__.py`, and the marks in `chart/marks.py` |
| How a chart is painted (a primitive, the legend, the controls) | `python/probatum/_static` |

A good test of placement: could a future tick or L2 engine reuse it? If yes, it is a core primitive and belongs in `probatum-core`, which holds no candle series and no bar-level logic. If it only makes sense per candle, it belongs in `probatum-engine`. The Python crate and package hold no simulation logic at all; they parse, convert, and orchestrate.

The chart layer splits on the same principle one layer further out: Python decides what is drawn and where, and the JavaScript decides only how it is painted ([ADR 0042](Decisions.md)). That is the rule the seam is held to, not a description of what the assets already contain, and four exceptions ship. The axis grain thresholds decide when a tick set is ragged, the number locale carries its own grouping, the timestamp format is written there in full, and two precision maps keep a trade caption and its table row one number rather than two opinions about it. Each of those is a question about the surface being painted rather than about the run being drawn, and Python cannot answer it without knowing the viewport; a colour, an alpha, or a threshold measured in bars is the other kind and belongs on the Python side ([ADR 0042](Decisions.md)). Two tests hold that half: one greps the shipped assets for a colour and allows only full transparency, and one refuses an alpha composed onto a theme colour, which is the shape that walked past the first for the life of the layer. Every asset is a plain script that declares and executes nothing at load, which is what lets the same files run as separate script tags in the development harness and as one bundle in the wheel; there is no npm and no build step anywhere in the project.

<br>

### The build loop

Pure-Rust work needs no Python, so the loop is short. `rust-toolchain.toml` pins the compiler to 1.88.0; add the two components once, since the toolchain file deliberately does not (declaring them there makes rustup try to install them into an already-present toolchain, which conflicts on some CI runners):

```bash
rustup component add rustfmt clippy     # once per checkout

cargo test -p probatum-core -p probatum-engine
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Python-facing work (anything in `probatum-py` or `python/probatum`) is validated on the Docker gate that builds the one abi3 wheel and runs the suite across Python versions; the full split is in the [Validation Guide](Validation_Guide.md). Four test stages sit outside that gate because each is much heavier; CI runs them, and a change to what they cover should run them locally before it is pushed: `--target test-browser` executes the chart's JavaScript in a real browser, `--target test-sb3` installs torch and exercises the Stable-Baselines3 adapter and TensorEnv on the CPU, `--target test-sdist` installs the source archive and runs the suite against it, and `--target miri` checks under Miri the candle memory behind every zero-copy view and the release of what a table lends through the Arrow C interfaces, down to the stream a capsule frees. Anything under `_static/` needs the first one, since nothing else in the project runs that code at all. A change to `python/probatum/torch.py` needs the gpu stage too, which CI cannot run, as its runners have no card; the [Validation Guide](Validation_Guide.md#the-stages-outside-the-gate) gives its two commands. The `probatum-py` crate is an `extension-module` cdylib, so it cannot be unit-tested with `cargo test` (there is no interpreter to link); it is checked with `cargo build` and `clippy`, then exercised through the wheel.

<br>

### The dev directory

`dev/` holds the tools that check the library, or write the data it ships, rather than any part of it. Nothing there is packaged into the wheel and nothing under `python/probatum` imports it. The line between it and `tests/` is that a test asserts and a tool produces: the gate runs a test and fails a build, while a tool here is run by hand and writes a file you then read and commit. There are four, and each is explained in this documentation set rather than beside itself, which is the same rule every other folder follows. The second simulator the engine is compared with asserts rather than produces, so it is a test, in `tests/reference/`, and the [Validation Guide](Validation_Guide.md) is where it is explained.

`dev/charts/` produces every chart image the documentation shows. `build.py` runs the documented examples for real against a frozen parquet and saves each one as a chart, and `shoot.py` points a headless browser at what it saved and screenshots it at the size it was drawn. Nothing is cropped or retouched, so what a reader sees is what the library drew, and the snippet printed beside a picture is the snippet that produced it. Both run in one opt-in stage, which needs the published [sample dataset](https://github.com/atOCEANO/sample-market-data) mounted because a picture built from live candles cannot be regenerated, only redrawn differently:

```bash
docker build --target charts -t probatum-charts .
docker run --rm -v "<sample-market-data>/data:/data:ro" \
  -v "${PWD}/.Documentation:/out" probatum-charts
```

Regenerating should change nothing unless the drawing changed, and that is the check worth making: thirteen of the fourteen that existed when the stage was introduced survived the move byte for byte, and fifteen ship today. If a rerun rewrites images you did not touch, something moved that you did not mean to move, and the fonts are the first place to look.

`dev/golden.py` rewrites `tests/chart_schema.json`, the key-path golden for the chart spec. The renderer is JavaScript and the gate has no browser, so the contract between the two halves is the one thing a Python test cannot reach directly: a key renamed in `python/probatum/chart/__init__.py` passes every test in the suite and draws a blank chart. Recording every key path the spec carries turns that rename into a one-line diff instead. Run it whenever the spec gains, loses or renames a key, in the same commit. It imports the test module rather than rebuilding the fixture itself, so the golden cannot be recorded against a different chart from the one the test checks.

`dev/diagrams/` holds the mermaid source for every hand-made diagram in `.Documentation/imgs/`, one `.mmd` per image, and its own stage renders all of them:

```bash
docker build --target diagrams -t probatum-diagrams .
docker run --rm --shm-size=1g \
  -v "${PWD}/dev/diagrams:/diagrams" \
  -v "${PWD}/.Documentation/imgs:/out" \
  probatum-diagrams
```

The sources are mounted rather than built into the image, so editing one and rerunning draws what is on disk. The router renders its set with the same two mounts and the same flags against the same digest, so the recipes differ only in that this one names a stage and that one names the image.

The images are numbered rather than named and the descriptive name survives only in their alt text: 205310 is the README hero, 205312 the crate layering, 205314 the step lifecycle, 205316 no-lookahead, 205318 the RL loop. Four of the five were reconstructed by reading the rendered PNG, because the originals were never kept, so a rerun redraws them rather than reproducing them byte for byte; 205314 is the one whose source survived and it still renders identical, which is what says the pipeline is faithful rather than merely working.

The renderer is pinned to a digest in the Dockerfile rather than tracked as `latest`, which is what makes a rerun evidence rather than a coin flip: `latest` and the version tag are different images, and the tag names the mermaid release rather than the `mmdc` build inside it. At this digest all five come back byte identical, not merely at the same dimensions. That holds because of the shapes as much as the digest: a stadium node's rounded outline anti-aliases a handful of edge pixels differently between runs, so a set containing one is visually identical and never byte identical, and every rerun shows it as a modified file carrying no change. Nothing here uses that shape and nothing should. The router keeps the same rule now, so both sets are byte reproducible and a diff from a rerun is a real one in either repo.

Three things the setup depends on, each of which cost an afternoon. The image's bundled headless-shell is broken with an ENOENT, so `puppeteer.json` points `executablePath` at the chromium the image also ships, and its entrypoint is `mmdc` itself, so the stage clears it and calls the binary by full path. The background has to be `transparent` rather than white, or the images invert badly against GitHub's dark mode. And no `-w`: setting a width made every wide diagram 2304 across where the rest are 2352.

The palette in `config.json` is the project's own: node fill `#16232e`, a teal `#2ee6a6` border for an engine stage, a blue `#4d9feb` one for anything the caller invokes, `#e6edf3` text, `#8b949e` arrows, trebuchet sans. It also names the cluster and edge-label colours, which look like padding and are not: mermaid derives whatever the palette leaves out, and left to itself it drew the no-lookahead frames as opaque brown boxes.

`dev/venues/build.py` writes the venue presets in `python/probatum/venues/`: each venue's instruments, with each one's tick, lot and tier table, from its public endpoints, and the fee rates, liquidation models and Hyperliquid's price rule the venues document, as the script gives them. It asks the venues themselves, so it runs by hand and never in the gate, and each preset records the day it ran as the day its tables were taken:

```bash
docker run --rm -v "${PWD}:/src" -w /src python:3.13-slim python dev/venues/build.py
```

A rerun rewrites every preset but frictionless, so its diff is what the venues changed since the last one.

<br>

### Code standards

- **Clippy is a gate, not advice.** `clippy --all-targets -- -D warnings` must pass. Do not silence a lint without a comment saying why (the one crate-level allow, for a pyo3-macro false positive, is documented at its site).
- **Unsafe carries a `SAFETY` comment.** Every `unsafe` block states the invariant that makes it sound. The zero-copy observation is the worked example.
- **An error message names the offending argument as the caller typed it**, gives the value it got, and puts the fix after a semicolon. Every error is a `ProbatumError` and the built-in for what arrived, a `TypeError` for a value of the wrong type and a `ValueError` for one of its type the call cannot take: a kind where both its topic and its built-in are the kind's, `DataError` or `VenueError` for a wrong value in bars or a venue, `SpecError` for an agent's spec, `LookaheadError` for what the causality check finds, `PolicyError` for a policy `backtest` cannot run and `MissingDependency` for a package that is not installed, with the line that installs it; otherwise the class for its built-in, `ProbatumValueError` for a wrong value, `ProbatumTypeError` a wrong type, `ProbatumKeyError` an unknown name in a known set, `ProbatumIndexError` a position past the end, `ProbatumAttributeError` a missing attribute, `ProbatumRuntimeError` a search, an environment or a chart's display that cannot go on and `ProbatumNotImplementedError` what is not built yet. Every warning is a `ProbatumWarning`, long work logs to the `probatum` logger rather than printing, and every re-raise chains with `from`.
- **Refusing beats lying.** A function handed something it cannot answer about raises and says what would have been wrong, rather than answering about the nearest thing it can. Most of the defects this library has shipped were a plausible number where an exception belonged.

<br>

### Recording a decision

Every non-obvious decision, a PnL-booking rule, a cost convention, an autoreset semantics, is recorded in [Decisions](Decisions.md), numbered in order and settled before the code that depends on it. The entry states what was decided and why, so the next contributor reads the reasoning rather than reverse-engineering it. When you add behavior that had a real choice behind it, record its decision in the same change.

<br>

### Submitting a change

Run the build loop above green before you submit, and the Docker gate too for anything Python-facing. A non-obvious decision lands its ADR in the same change, never a follow-up.

<br>

### Cutting a release

Releases are built by CI, never by hand. Pushing a `v*` tag runs the release workflow, which builds the `abi3` wheels for Linux (x86_64 and aarch64), macOS (one universal2 file covering Intel and Apple silicon), and Windows, plus an sdist, and attaches them all to a GitHub Release:

```bash
git tag -a vX.Y.Z -m "probatum X.Y.Z: one line on what changed"
git push origin vX.Y.Z
```

Two things before you tag. The version in `Cargo.toml` is what names the wheels, so bump it first and let the tag match it, or the release says one number while its artifacts say another. And run the gate on the exact commit you intend to tag: a tag builds, it does not test, so nothing else will catch a regression at that point.

To rehearse without releasing, run the workflow by hand from the Actions tab. A manual dispatch builds every wheel and stops there; only a tag creates the Release. That distinction is worth using, because the macOS and Windows legs exist nowhere but the runners, so a tag is the first time they are exercised.

Those wheels are the install path for anyone without a Rust toolchain, so a release is what makes a version usable at all. probatum is not published to PyPI.

<br>

### Adding a wrapper

A new way to drive the engine (say a different backtest style, or a distributed runner) is a Python module under `python/probatum/`, built on the compiled `Engine` or `Batch`. It holds no simulation logic: it configures the engine, drives the loop, and shapes the results. Re-export it from `python/probatum/__init__.py` if it belongs on the top level, and add a test file under `tests/` so the Docker gate covers it. Keep documentation in this `.Documentation/` set and the top-level README; there are no per-folder READMEs.

<br>

### Adding a preset

A venue preset is a TOML file in `python/probatum/venues/`, named for the preset, and a new venue or a corrected table is a change to such a file and nothing else ([ADR 0028](Decisions.md)). It holds four tables:

- `[venue]`: its `name`; `taken`, the day its tables were taken, as `YYYY-MM-DD`; and `kind`, `spot`, `linear` or `inverse`, where every instrument it lists is of one kind, or `combines`, the presets of one kind each it puts in one account. A combining preset holds no other table: each part's instruments come under their kind, as `kind:SYMBOL`, each part's own settings and rates become its kind's, and the account's settings are the same in every part that gives one, but for `funding`, where a part's `"none"` yields to another's.
- `[defaults]`: the venue's rates, `fees = { maker = r, taker = r }`, its `liquidation` and `backstop`, its `liquidation_fee`, which places the close point where no maintenance is kept, its `leverage`, the cap on a perpetual with no tier table, its `max_open_orders`, its `funding`, which is `"none"`, `"data"`, `{ every = hours, rate = r }` or `{ every_bars = n, rate = r }`, and any setting an instrument carries but its delisting time, for every instrument; and `[defaults.fills]`, its fill model: `limit`, `window`, `liquidity = { budget = f }`, `spread` (`"none"`, `"estimated"` or `{ fixed = bps }`), `slippage` (`"none"` or `{ fixed = bps }`), `impact` (`"none"`, `{ linear = k }` or `{ sqrt = k }`) and `sizes` (`"on"` or `"off"`).
- `[kinds.spot]`, `[kinds.linear]` and `[kinds.inverse]`, in a preset that mixes kinds: the settings every instrument of that kind carries, such as a market's own `fees`, but a delisting time.
- `[instruments.KEY]`, one per instrument, keyed by the router's symbol, or by `kind:SYMBOL` in a preset that mixes kinds: its `tiers`, a list of `{ up_to, leverage, rate, amount }`, its `tiers_by`, any size rule or `fees` of its own, and `delist`, the time the venue delisted it, a TOML date or datetime read in UTC, which only an instrument's own table gives.

An instrument takes the defaults, then its kind's settings, then its own. The loader checks a file's shape and the type of every setting, and every value passes the checks the compiled engine runs, on one row of placeholder bars, but a delisting time, which a run checks against the bars it falls among, so a preset is held to exactly what a venue built by hand is: a `probatum.Venue` passes the same checks, and each preset is reached as one, `probatum.venues.NAME`. The tests load every file in the folder, so a new preset is covered as soon as it is there.

<br>

### Adding a fidelity tier

The larger extension the workspace is built for is a second engine at a different fidelity, a tick or L2 sibling. It would be a new crate beside `probatum-engine`, reusing `probatum-core` for the position, account, costs, and book, and replacing only the resolution layer (how a fill is decided). Nothing in `probatum-core` should assume bars, so that reuse stays clean; if a change to core starts to assume a candle, it belongs in `probatum-engine` instead.
