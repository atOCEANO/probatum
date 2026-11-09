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
  <a href="Migration.md">Migration</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <b>Contributor Guide</b>
</sub>

<br>
<br>
<br>
<br>

## Contributor Guide

The library is a cargo workspace of three crates under a Python package, laid out in [Architecture](Architecture.md). Each layer builds only on the one below it, and only `probatum-py` knows about Python, so a change belongs in the lowest layer that can hold it.

<br>
<br>

## Local Development

The two pure-Rust crates need no Python to test. Clippy over the workspace also builds `probatum-py`, which needs a Python 3.13 interpreter on the path, as the `rust-checks` stage has. `rust-toolchain.toml` pins the compiler to 1.88.0 and leaves its two components to add once per checkout, since declaring them there conflicts on some CI runners:

```bash
rustup component add rustfmt clippy     # once per checkout

cargo test -p probatum-core -p probatum-engine
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

`probatum-py` is an `extension-module` cdylib, which `cargo test` cannot link without an interpreter, so it is checked by `cargo build` and clippy and exercised through the wheel. For work on the Python side, maturin builds the extension into the active environment:

```bash
pip install maturin
maturin develop --release        # the abi3 extension, into the active venv
```

Every local check that touches Python runs in Docker, on the wheel the release ships, as the [Validation Guide](Validation_Guide.md#the-stages) gives.

<br>

### Where a change belongs

| If it is... | It goes in... |
| :--- | :--- |
| a primitive that holds no bars: a unit, an order, the position, the account, a cost | `crates/probatum-core` |
| memory a C interface lends out and takes back, the Arrow export and its release | `crates/probatum-core`, where Miri runs it |
| a bar-level rule: fills, the step, liquidation, funding, the ledger, the batch | `crates/probatum-engine`, its step's parts in `src/engine/` |
| a Python binding: a pyclass method, an argument's type, a numpy or Arrow bridge | `crates/probatum-py` |
| how bars are read, held, sliced and saved | `python/probatum/_data.py` and `_ipc.py` |
| a venue's settings, or the presets' loader | `python/probatum/_venue.py` and `venues/__init__.py` |
| how `backtest` reads a policy, and the Strategy's account | `python/probatum/_backtest.py` and `_strategy.py` |
| a search: its samplers, its space, its workers, its causality check, its storage | `python/probatum/_search.py`, `_samplers.py`, `_space.py`, `_workers.py`, `_causality.py` and `_storage.py` |
| a splitter, a validation or a label | `python/probatum/cv.py`, `_validate.py` and `_labels.py`, the label's trade in `crates/probatum-engine/src/labels.rs` |
| capacity, a cost sweep or its break-even | `python/probatum/_costs.py`, the costs a sweep multiplies in `_venue.py` |
| a statistic, an interval, or a test across a search's trials | `python/probatum/stats.py`, and the same rule in plain loops in `tests/reference/statistics.py` |
| what a Result holds, its assumptions, its baselines, a fill or an order told in words | `python/probatum/_result.py`, `_assumptions.py`, `_baselines.py` and `_explain.py` |
| how a table, a record or a summary shows | `python/probatum/_frames.py`, each record's frame in `stats.py` and the summaries' rows in `_result.py` and `_search.py` ([0041](Decisions.md)) |
| the environment an agent trains on | `crates/probatum-engine/src/env.rs` for each step, `python/probatum/_env.py` for the episodes, the spaces and Gymnasium's API, and the reference's environment in `tests/reference/environment.py` |
| an agent's spec, its backtest, its transitions | `python/probatum/_agent.py`, `_env.py` and `_transitions.py`, and the SB3 adapter in `sb3.py` |
| how an environment's tensors reach a device | `python/probatum/torch.py`, and the account block it writes in `crates/probatum-engine/src/env.rs` |
| an indicator, a function of arrays that never sees the engine | `python/probatum/ta.py` |
| an error's class | `python/probatum/_errors.py` |
| what a chart draws, a mark, a panel rule | `python/probatum/chart/__init__.py` and `chart/marks.py` |
| how a chart is painted, the legend, the controls | `python/probatum/_static` |

The chart splits on the same line one layer out: Python decides what is drawn and where, and the JavaScript only how it is painted ([0042](Decisions.md)). The JavaScript decides what depends on the plot's size: which of an axis's ticks to drop, the type scale and which exit captions fit. It also owns the number grouping, the timestamp format and the precision a trade's caption and its table row share. A colour, an alpha or a threshold in bars belongs to Python, and two tests hold it there: one finds no colour in the shipped assets but full transparency, and one refuses an alpha composed onto a theme colour. Every asset is a plain script, joined into each chart's document as it is built, with no npm and no build step.

<br>

### The dev directory

`dev/` holds the tools that check the library or write the data it ships. None of it is packaged, and nothing in `python/probatum` imports it. A test asserts and fails a build; a tool here is run by hand and writes a file you then read and commit. Each is explained here rather than beside itself.

`dev/charts/` draws every chart image the documentation shows. `build.py` runs the documented examples against the frozen [sample data](https://github.com/atOCEANO/sample-market-data) and saves each as a chart. `shoot.py` points a headless browser at what it saved and screenshots it at the size it was drawn, in the dark and the light scheme, which a page shows by its reader's theme. Nothing is cropped or retouched, and the snippet printed beside a picture is the call that drew it. Both run in one opt-in stage, with the dataset mounted:

```bash
docker build --target charts -t probatum-charts .
docker run --rm -v "<sample-market-data>/data:/data:ro" \
  -v "${PWD}/.Documentation:/out" probatum-charts
```

A rerun changes no image whose drawing did not change, so one that rewrites an image you did not touch points at something that moved. The fonts are the first place to look: the stage installs JetBrains Mono, which the chart asks for.

`dev/golden.py` rewrites `tests/chart_schema.json`, every key path the chart's document carries. The renderer is JavaScript and the gate has no browser, so a key renamed in `python/probatum/chart/__init__.py` would pass every Python test and draw a blank chart; the golden turns that rename into a one-line diff. Run it in the same commit whenever the document gains, loses or renames a key. It imports the test module, so the golden is recorded against the chart the test checks, and it runs on the wheel the `builder` stage makes:

```bash
docker build --target builder -t probatum-builder .
docker run --rm -v "${PWD}:/repo" probatum-builder sh -c \
  "pip install --no-cache-dir /wheels/*.whl numpy pandas pyarrow pytest \
   && python /repo/dev/golden.py"
```

`dev/diagrams/` holds the mermaid source of every diagram in `.Documentation/imgs/`, one `.mmd` per image, and its own stage renders them:

```bash
docker build --target diagrams -t probatum-diagrams .
docker run --rm --shm-size=1g \
  -v "${PWD}/dev/diagrams:/diagrams" \
  -v "${PWD}/.Documentation/imgs:/out" \
  probatum-diagrams
```

The sources are mounted, so a rerun draws what is on disk. The images are numbered, their names in their alt text: 205312 is the crate layering, 205314 the order of events within a step, 205316 no-lookahead and 205318 the RL loop. The renderer is pinned to a digest, and no diagram uses mermaid's stadium node, whose rounded outline renders a few pixels differently on every run; a rerun of an unchanged source then gives the same bytes. `puppeteer.json` points the renderer at the chromium the image ships, since its bundled headless shell does not start. The background is transparent, so an image reads on a light page and a dark one, and no width is set, so every wide diagram renders at the same size. `config.json` holds the palette, the cluster and edge-label colours among it, since mermaid derives whatever the palette leaves out. A source's `stage` class keeps the teal border, and its `entry` class, where it has one, gives what the caller invokes a blue one.

`dev/venues/build.py` writes every venue preset in `python/probatum/venues/` but `frictionless`, from what exchange-router lists and the rates each venue documents, as [Venue Notes](Venue_Notes.md) describes. It asks the venues through the router, so it runs by hand and never in the gate, and each preset records the day it ran as `taken`:

```bash
docker run --rm -v "${PWD}:/src" -w /src python:3.13-slim sh -c \
  "pip install -q 'exchange-router @ https://github.com/atOCEANO/exchange-router/releases/download/v4.0.0/exchange_router-4.0.0-py3-none-any.whl' && \
   python dev/venues/build.py"
```

A rerun rewrites every preset but frictionless, so its diff is what the venues changed since the last one. The router prints a warning where it served a list or a table short: a builder-deployed dex Hyperliquid did not answer for, or a perpetual with no tiers, which the builder leaves out. A run that printed one is run again before its diff is read, and a warning the second run repeats is the venue's own state.

`dev/licences/` holds the settings and the template that cargo-about writes `THIRD-PARTY-LICENSES` from: the licence of every crate in `Cargo.lock` the extension is built from, in each crate's own text. The wheel and the source archive ship it beside probatum's `LICENSE` and the licences of the chart's renderer and icons. Run it in the same commit whenever `Cargo.lock` changes:

```bash
docker run --rm -v "${PWD}:/src" -w /src rust:1.88.0-slim sh -c \
  "cargo install --locked cargo-about --version 0.9.3 --features cli && \
   cargo about generate --locked --fail -c dev/licences/about.toml -o THIRD-PARTY-LICENSES dev/licences/about.hbs"
```

A rerun on an unchanged lock writes the same bytes. The settings name two crates' licence files by their checksums: libm's, the one file that carries the copyrights of the musl code it derives from, and pyo3's, which cargo-about otherwise lists twice. A new version of either can change its file. cargo-about then only warns that the checksum no longer matches and writes that crate as it would unnamed, so read its warnings and update the checksum.

`dev/constraints.txt` pins every package the gate's stages and CI install, so both test the same versions.

<br>
<br>

## Code Standards

- **Clippy passes** with `--all-targets -- -D warnings`.
- **Every error is one of the classes** [Errors and warnings](Python_API.md#errors-and-warnings) **lists** ([0029](Decisions.md)), every warning a `ProbatumWarning`, long work logs to the `probatum` logger rather than printing, and every re-raise chains with `from`.
- **A function refuses what it cannot answer**: it raises and says what would have been wrong, rather than answering about the nearest thing it can.

<br>
<br>

## Testing

The [Validation Guide](Validation_Guide.md) gives the gate's stages, what each group of tests proves, and the markers that keep the slow, coverage and gpu tests out of the gate.

The pages are held to the code too. `tests/test_docs.py` parses every Python block on every page, and checks each keyword passed to probatum's functions, a venue's `replace` and the chart's marks against their signatures. It also fails where:
* the Python API page leaves an export out;
* no page names a stage of the Dockerfile or a tool under `dev/`;
* a page cites a record that does not exist, or links a page that does not exist;
* or the install lines name another version than the package's.

`tests/test_examples.py` runs the Charts page's examples, and `test_the_api_page_s_*.py` run the Python API's Quickstart and, after it, ten of its sections.

A change is done when the gate passes on it, and so does the stage outside the gate that covers what it touched:

| A change to | Also passes |
| :--- | :--- |
| an engine rule | `test-slow` |
| `_static/` | `test-browser` |
| `sb3.py`, `torch.py` | `test-sb3`, `test-gpu` |
| the candle views, the Arrow tables | `miri` |
| what the source archive carries | `test-sdist` |
| a statistic | `coverage` |
| the step | `bench-instructions`, since CI fails a rise of more than 10% in the step loop's instructions against the commit before |

<br>
<br>

## Releasing

`version` in `Cargo.toml`'s `[workspace.package]` sets the version. maturin names the wheels and the source archive from it, `probatum.__version__` reads it back from the installed package, and the install lines in the pages repeat it, which a test holds to it. **The tag must equal it**, and the release's first job stops a tag that does not. To cut a release:

1. Bump the version in its own commit, with the install lines and with `Cargo.lock`, which `cargo update -w` brings along, since every `--locked` build refuses a lock that disagrees.
2. Push that commit to main.
3. Tag it `vX.Y.Z` and push the tag:

```bash
git tag -a vX.Y.Z -m "probatum X.Y.Z"
git push origin vX.Y.Z
```

A pre-release takes Cargo's form, `vX.Y.Z-rc.N`, and is published as one. `.github/workflows/release.yml` checks the tag against `Cargo.toml` and runs CI on the tagged commit. It then builds each wheel on the machine it is for, Linux on x86_64 and aarch64, macOS on Apple silicon and Windows, and tests each there on Python 3.13 and 3.14. The Linux wheels pass the gate's test stages before the `wheels` stage hands them out, and the source archive is the one the `sdist` stage hands out once `test-sdist` has built and tested it. Last, it checks every file's name against the version and publishes them on a GitHub Release with their SHA-256 checksums, its body written above GitHub's generated notes.

A manual dispatch from the Actions tab runs all of it but the version check and the publishing, on whichever branch or tag it names. probatum is not published to PyPI; the release's wheels are its install path.

<br>
<br>

## Adding an engine rule

A rule reaches the engine last:
1. It is written in the [Specification](Specification.md), with its equation and a worked example. A record goes into [Decisions](Decisions.md) only where the rule is a choice someone could reasonably make the other way.
2. It goes into the reference simulator in `tests/reference/simulator.py`, its worked example into `tests/test_every_worked_example_holds_to_the_cent.py`, and the comparisons learn to draw the cases that reach it.
3. It goes into the engine, with unit tests in the module that applies it, until the comparisons agree on every case and reach the rule at length.

<br>
<br>

## Adding a preset

A venue preset is a TOML file in `python/probatum/venues/`, named for the preset, and a new venue or a corrected table is a change to such a file and nothing else ([0028](Decisions.md)), made by hand or, for a preset `dev/venues/build.py` writes, by the builder's next run. It holds four tables:

- `[venue]`: its `name`; `taken`, the day its tables were taken, as `YYYY-MM-DD`; and `kind`, `spot`, `linear` or `inverse`, where every instrument it lists is of one kind, or `combines`, the presets of one kind each it puts in one account. A combining preset holds no other table. Each part's instruments come under their kind, as `kind:SYMBOL`, each part's own settings and rates become its kind's, and the account's settings are the same in every part that gives one, except `funding`, where a part's `"none"` yields to another's.
- `[defaults]`: the venue's rates, `fees = { maker = r, taker = r }`; its `liquidation` and `backstop`; its `liquidation_fee`, which places the close point where no maintenance is kept; its `leverage`, the cap on a perpetual with no tier table; its `max_open_orders`; its `funding`, which is `"none"`, `"data"`, `{ every = hours, rate = r }` or `{ every_bars = n, rate = r }`; and any setting an instrument carries but its delisting time, for every instrument. `[defaults.fills]` holds its fill model: `limit`, `window`, `liquidity = { budget = f }`, `spread` (`"none"`, `"estimated"` or `{ fixed = bps }`), `slippage` (`"none"` or `{ fixed = bps }`), `impact` (`"none"`, `{ linear = k }` or `{ sqrt = k }`) and `sizes` (`"on"` or `"off"`).
- `[kinds.spot]`, `[kinds.linear]` and `[kinds.inverse]`, in a preset that mixes kinds: the settings every instrument of that kind carries, such as a market's own `fees`, but a delisting time.
- `[instruments.KEY]`, one per instrument, keyed by the router's symbol, or by `kind:SYMBOL` in a preset that mixes kinds: its `tiers`, a list of `{ up_to, leverage, rate, amount }`, its `tiers_by`, any size rule or `fees` of its own, and `delist`, the time the venue delisted it, a TOML date or datetime read in UTC, which only an instrument's own table gives.

An instrument takes the defaults, then its kind's settings, then its own. The loader checks a file's shape and the type of every setting. Every value but a delisting time passes the checks the compiled engine runs, on one row of placeholder bars, so a preset is held to exactly what a venue built by hand is; a delisting time is checked by a run against the bars it falls among. The tests load every file in the folder, so a new preset is covered as soon as it is there.

<br>
<br>

## Adding a wrapper

A new way to drive the engine is a Python module under `python/probatum/` built on the compiled `Engine` or `Batch`. It holds no simulation logic: it configures the engine, drives the loop and shapes the results. Re-export it from `python/probatum/__init__.py` if it belongs on the top level, document it on the [Python API](Python_API.md) page, and add a test file under `tests/` whose name states what it proves.
