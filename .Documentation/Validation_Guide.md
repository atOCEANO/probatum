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
  <a href="Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <b>Validation Guide</b>
</sub>

<br>
<br>
<br>
<br>

## Validation Guide

Validation is split by where the risk lives. Every line of simulation is pure Rust, so it is tested locally with `cargo`, with no interpreter in the loop. The Python side simulates nothing, but it is no longer small: the indicators, the post-hoc statistics and the chart are roughly as much code as the engine, and they carry their own risk, which is a wrong number rather than a wrong fill. The Python boundary, that code, and the promise that one wheel works across Python versions are all checked in Docker.

<br>

### Pure Rust, locally

`probatum-core` and `probatum-engine` hold the simulation, so they carry the bulk of the tests and every core-correctness invariant. They validate with cargo:

```bash
cargo test -p probatum-core -p probatum-engine
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

This needs no interpreter, so it is the inner loop for anything below the Python boundary. The same three commands are also a stage of the gate, `docker build --target rust-checks .`, which runs them against the pinned toolchain with no Rust on the host. That stage exists because the wheel gate alone will not catch a formatting or lint failure: it builds the wheel and runs pytest, so a Rust change can pass it and still fail CI. Rust changes go through both.

<br>

### The Python wheel, in Docker

The `probatum-py` crate is an `extension-module` cdylib: it cannot be unit-tested with `cargo test`, because the test binary has no interpreter to link. So the binding is checked two ways. The Rust side compiles and lints with `cargo build -p probatum-py` and `clippy`. The behavior is checked through the built wheel on a Docker gate that builds one `abi3` wheel and installs it on Python 3.13 and 3.14, running the `tests/` suite on each:

```bash
docker build --target test313 .
docker build --target test314 .
```

The `builder` stage makes the wheel with maturin in PyPA's manylinux_2_28 image, which holds it to what glibc 2.28 provides ([ADR 0044](Decisions.md)). Because the wheel is `abi3-py313`, the same artifact is what ships, so the gate tests the real deliverable, not a per-version rebuild. This is the "works on Python 3.13 and up" check, run before a Python-facing change is done, not on every edit. Every gate stage installs its packages with `-c dev/constraints.txt`, as CI does, so both test the versions pinned there.

The builder makes the wheel for the machine it runs on, so the same stages test the ARM wheel on an ARM machine, as CI does on GitHub's ARM runners, or on any other under the emulation Docker Desktop carries, where both the build and the suite run many times slower:

```bash
docker build --platform linux/arm64 --target test313 .
```

<br>

### The differential gate

The sharpest check in the project, and the only one that can find a defect nobody thought to look for:

```bash
docker build --target test-slow .
```

`tests/reference/simulator.py` is a second, deliberately slow implementation of the same simulator, written from the [Specification](Specification.md) and never from the Rust. That constraint is the whole point of it. A test written against the implementation asks whether the code does what the code does; this asks whether two independent readings of the same rules produce the same number, over randomised paths through leverage, liquidation by takeover and on the book, funding, transfers between the wallets, partial fills, the order book and inverse contracts. The reference decides each case's orders on its own account as it goes, and the engine plays the same decisions in one call through the binding's `run`, which names an order by the placement that made it. The engine still carries emsl's rules where the Specification has moved on, so a case is compared only where the reference applied none of the rules the engine has yet to take up: the reference counts every rule it applies, `NOT_IN_THE_ENGINE` in `tests/reference/differential.py` names those rules, and a stage that brings one into the engine takes it off the list. Each run reports how many cases it set aside, and for which rule. `test_the_engine_agrees_with_the_reference.py` runs the comparison over a few hundred cases in every gate stage, and over six seeds of five hundred in this one, at a relative tolerance of 1e-6; any disagreement fails the build, and at length so does a rule the two share that no compared case reached. A test marked `slow` is skipped in the ordinary run with that reason, and runs only where `--slow` is passed, as this stage passes it.

`tests/reference/batch_differential.py` is the same idea one tier up, over `Batch`, run by `test_the_batch_agrees_with_the_reference.py`, briefly in every stage and at length in this one. Over universes it holds each env to everything it recorded, against a reference opened at that env's own row of the same series.

`tests/reference/environment_differential.py` does the same for the environment an agent trains on, run by `test_the_environment_agrees_with_the_reference.py` in the same two ways. Each environment of a batch over random universes and inverse perpetuals is held, step by step, to a reference episode opened at the same row on the same slots, on its observation, its reward, its state and its flags, across every kind of action, both rewards, and a decode and a reward of its own; it reports how many episodes were liquidated, cut, hid a slot or rested an order, and fails if any of those is zero. `test_an_episode_runs_again_as_a_backtest_from_its_start.py` replays episodes through `backtest` from the start their info gives, with the weights they traded, and holds the two equities equal to the bit. The comparison draws a scale for some of its environments, so a normalised observation is held cell by cell too, and `test_an_agent_carries_the_spec_it_trained_with.py` holds the agent spec's probe bars, probe actions and hashes to the reference's plain loops to the bit, and its normalisation statistics to the reference's over random universes with bars left unprinted, a late listing, a delisting and windows that traded nothing. `test_a_trained_agent_is_judged_by_the_same_backtest_as_a_rule.py` plays what an agent decided in its backtest again as weights, over every kind of action and a decode of its own, and holds the two equities equal to the bit; it holds the observation an agent sees at a slice's first close to the one at the same bar of the whole Data, and an observation seen through SB3's statistics cell by cell to the reference's. `test_sb3.py`, in the SB3 stage, holds an agent saved and loaded through SB3's zip, and a torch module through `probatum.save`, to the run of the same agent in memory, to the bit.

`tests/reference/statistics.py` is the same idea for `probatum.stats`: each rule of the Specification's Inference, Statistics of a run and Across a search sections, a model's forecast and what a sweep reads of a run, in plain loops over lists, written from the page and never from the module. It draws nothing, so `test_the_statistics_agree_with_the_reference.py` hands it the module's own resamples and compares what the two compute from the same rows, and `test_every_estimator_reproduces_its_paper.py` holds it to the numbers the papers it follows print.

The reference labels a Data as the Specification's Labels section says, each label one trade of a fresh account through its own fill model, and `test_labels_agree_with_the_reference.py` holds `probatum.labels` to the same exit, end and bars and to its return and uniqueness on every row of random universes, gaps, delistings, inverse contracts and both sides among them.

A sweep's level is a backtest at another balance or with its venue's costs scaled, so `test_a_sweep_runs_a_policy_again_at_each_balance_or_scale_of_its_costs.py` holds each level to the backtest of a venue replaced by hand to match it, and its round-trip cost, turnover and share of orders held to the reference's reading of that run. A scaled spread estimate is the engine's own, and the comparisons draw one, as a cost sweep scales it, for half their cases whose spread is estimated.

A Study that is saved must read back as it was, so `test_a_saved_study_loads_back_with_every_table_and_statistic.py` holds every table, statistic, summary and repr of a loaded Study, a search's and a validation's, to the one that was saved. A resumed search must give the Study one that never stopped gives, so `test_a_search_into_storage_resumes_where_it_stopped.py` stops a search of each sampler part way, as Ctrl-C stops one, resumes it, and compares the two trial by trial, and does the same for searches extended in workers and in Rust.

Beside them, `test_every_worked_example_holds_to_the_cent.py` holds the reference to every worked example of the engine's rules on the Specification page, the search tests holding those of its Search section and the splitter and validate tests those of its Validation section, and `test_the_reference_reaches_every_rule.py` runs it over random accounts and counts each rule it applies. The batch's own verbs, `reset_at_each`, `advance` and `snapshot`, had a Rust test comparing the batch against the engine it is built from behind them, which is worth having and shares every mistake the engine makes. Each env is compared against a reference run from that env's own offset, because an episode beginning part way into the series must still fund on the bars the series funds on ([ADR 0020](Decisions.md)), and no test that starts every env at bar zero can see that rule at all.

It has already earned its place twice. It found the slot leak of [ADR 0010](Decisions.md), which had survived every test ever written against the engine because none of them thought to ask whether an order filled down to float dust still holds its slot; it surfaced as five separate disagreements that all reduced to one line. The batched half then found [ADR 0020](Decisions.md) on its first run, funding marked at a price past the liquidation fence.

**A differential is only as good as the fields it compares**, and [ADR 0020](Decisions.md) is the lesson. The single-engine harness read position and equity, which is what the engine's own promises are about. A takeover zeroes the equity on both sides, so a disagreement about what the account paid on the way there was invisible: three thousand cases over six seeds could not see it, and two hundred batched cases could, only because the batched comparison happened to include `funding_paid`. Both harnesses now compare it. The single-engine harness compares each wallet's cash, and the two fill logs, order logs, round trips and ledgers entry by entry, as well. When a harness agrees too easily, ask what it is not looking at.

`batch_differential.py` also reports what its cases reached: how many traded, funded, busted and ran out of bars, and over universes how many envs were liquidated and how many opened on a row an instrument printed no bar on or before one listed, and it fails if any of those is zero. "0 disagreed" over a hundred flat accounts that never funded and never died is a pass that means nothing, and after the fact there is no way to tell one from the other.

It also reaches things a unit test structurally cannot. `replace` carries the side of the order it moves ([ADR 0010](Decisions.md)), and that claim was asserted nowhere, in Rust or in Python: flipping Sell to Buy inside `Engine::replace` leaves every Rust test in the workspace green. The harness places, moves and cancels orders through the same sequence on both simulators, so a flipped side parts company with the reference.

Read a disagreement as a question rather than a verdict: the reference is a second reading and can be the wrong one, and the Specification decides which. `tests/reference/trace.py` shrinks a failing case to its smallest form and prints both simulators bar by bar, which is what makes the question answerable.

<br>

### The stages outside the gate

The gate does not run the stages below. The first four are tests that CI runs on every push and that are opt-in locally, because each pulls a far heavier image, compiles the whole engine again or installs a nightly compiler; each covers something the gate structurally cannot:

```bash
docker build --target test-browser .    # the chart's javascript, in chromium
docker build --target test-sb3 .        # the Stable-Baselines3 adapter and TensorEnv, with torch
docker build --target test-sdist .      # the source archive, built and tested
docker build --target miri .            # the zero-copy views' and the tables' memory, under Miri
```

The gpu stage runs apart from CI, whose runners have no card, after any change to the device code. A build cannot reach a GPU, so its image is built and then run with the card handed in:

```bash
docker build --target test-gpu -t probatum-test-gpu .
docker run --rm --gpus all probatum-test-gpu
```

It prints the card, its driver, its load and torch's version, then runs `tests/test_tensors_on_the_device_match_the_numpy_path.py` on the card: every tensor `TensorEnv` hands out, host path and resident, held to the numpy path's to the bit over a run of 400 steps whose policy keeps the card busy and the host waiting at random, so a buffer written again before its copy has run would show, and a loop of 4,096 environments whose queued work on the card is still running after every step, so the host never waited for it. The same file's tests on the CPU run in the SB3 stage. A test marked `gpu` needs a CUDA device and is skipped where none is found.

The coverage cells run apart from all of them, at each stage that changes a statistic:

```bash
docker build --progress=plain --target coverage .
```

`tests/test_intervals_cover_as_often_as_they_claim.py` simulates each interval `probatum.stats` gives on the shapes a strategy's returns and round trips take and holds it to cover what it estimates between 93% and 97% of the time at 95% ([ADR 0031](Decisions.md)): the Sharpe's on independent, heavy-tailed, autocorrelated and switched returns at 1,000 daily bars and a year of hourly ones, the bootstrap's of the returns on the first three at 500 and 1,000 daily bars and a year of hourly ones, with the switched returns held to the floor of blocks that refuses them, the round trips' at 60 and 160 trips that win evenly, rarely and large or often and small, the Sharpe difference's at two equal Sharpes, and alpha's and beta's. Each cell's replications draw from streams of their own, so its count is the same on any number of cores, and each prints its coverage. A cell outside the band is a defect of its statistic, fixed before the statistic ships. A test marked `coverage` runs only where `--coverage` is passed, as this stage passes it.

`tests/test_a_search_s_tests_reject_as_rarely_as_they_claim.py` runs in the same stage on searches with no skill, coin flips and moving-average crossovers on a random walk, each beside a random null of as many trials. It holds the deflated Sharpe, against every trial and for the winner's cluster, to pass no more than one search in twenty from ten trials on, and against every trial fewer at a hundred than at two, and the Reality Check and SPA to reject trials that gain nothing over cash between 3% and 7% of the time at 5% ([ADR 0033](Decisions.md)).

`test-sdist` builds the source archive a release ships, installs it the way pip does on a platform with no wheel, and runs the whole suite against it, so a file left out of the archive fails there rather than for the first user on an unusual platform.

`miri` runs the candle tests and the Arrow table's under Miri, an interpreter that checks every memory access a test makes and reports what is never freed. Every zero-copy view reads the candles through `floats` in probatum-core, the library's one reinterpretation of memory ([ADR 0040](Decisions.md)), and a mistake there can read past the buffer and still return the right values, which an ordinary test cannot see. A table leaves through the Arrow C stream interface, whose reader may move the stream out of the capsule that holds it and a column out of its batch, and release what it holds in any order ([ADR 0026](Decisions.md)), and a mistake there frees what a reader still holds or never frees it at all; the release of all of it, the capsule's free among it, is probatum-core's, which Miri runs. The stage needs a nightly compiler, pinned by date.

`test-browser` is the other half of the chart layer. Everything in `tests/test_chart.py` reads `Chart.spec()`, because the gate has no browser and a spec assertion is the sharper test of what Python decided ([ADR 0042](Decisions.md)). What that cannot reach is whether the shipped JavaScript parses, runs and draws, and roughly 1,250 lines of it had never been executed by anything in the project. The first run of it found that the vendored renderer joins a line straight across whitespace, so ADR 0042's central claim was false in the artifact while true in the document. `tests/test_render.py` skips wherever playwright is absent, so the correctness gate stays green without one.

The rest assert nothing when run locally, which is why they are not gates:

```bash
docker build --target bench .                                 # engine throughput
docker build --target bench-surfaces -t probatum-bench-surfaces . # then docker run it
docker build --target bench-instructions .                    # step instruction counts
docker build --target mutants -t probatum-mutants .               # then docker run it, alone
docker build --target charts -t probatum-charts .                 # the documentation images
docker build --target diagrams -t probatum-diagrams .             # the hand-made diagrams
```

`bench` prints the engine's throughput and `bench-surfaces` prints it per surface (the compiled engine, an array and a Strategy through `backtest` at one instrument and at fifty, `search` over weights and over a Strategy, `probatum.env`, `Batch`). Timings depend on the machine, so neither can fail. `bench-instructions` counts the step loop's instructions under Valgrind instead, a number that does not depend on the machine, and CI compares it with the commit before and fails on a rise of more than 10%. It covers the Rust step loop only: a slower indicator or a slower Python surface still passes. `mutants` runs cargo-mutants over the two pure-Rust crates and lists every mutant the tests miss; CI runs it on the lines each push changes and fails on a miss. `charts` rebuilds every chart image the documentation shows, from frozen data so that a rerun redraws rather than redecides; the [Contributor Guide](Contributor_Guide.md) has the recipe. `diagrams` renders the mermaid sources behind the hand-made images, and its recipe and the three settings it depends on are in the [Contributor Guide](Contributor_Guide.md) too.

<br>

### Cross-cutting invariants

The properties that would be easy to break silently are their own tests:

- **No same-bar lookahead.** An order decided on bar `t` fills on bar `t+1`; a market fill lands at the next open, never the decision bar's price.
- **Batched equals looped.** `Batch.step_all` over N envs is bit-identical, by exact state equality, to stepping N single engines in a loop. The envs are independent with no cross-env reduction, so parallelism introduces no drift.
- **A clone is its original.** A cloned engine, batch or environment, and an engine or batch read back from its bytes, steps on to the same states, records and observations as the one it came from, to the bit, on a venue whose hidden events a stream decides, so a stream drawn afresh rather than copied would part from it; a fork's copies each repeat the environment they came from, and an Engine and a Batch pickled into a spawned process step on there as here ([Specification](Specification.md#clones-and-pickles)).
- **A transition is the step it was.** Every row an environment records is worked again from what each step returned, under all three autoreset modes, both layouts and box, discrete and order actions, a step that resets an environment taking none; an agent's backtest records the observation at each close it was asked at until its episode ended, and the rewards its own equity gives; and the exported file reads back, through pyarrow, polars and the package's own reader, to the same bytes ([Specification](Specification.md#transitions)).
- **View lifetime and read-only.** A zero-copy observation stays valid after the engine is dropped (its base keeps the buffer alive), and writing through it is rejected, so a view onto the shared candle buffer can never corrupt it.
- **Round trip reconciliation.** The round trips' realized PnL sums to the account's cumulative realized PnL, and on a run that ends flat their net PnL equals the change in equity ([ADR 0023](Decisions.md)). A forced close ends a round trip like any other close, with the reason `liquidation`, because it is booked through the one path every other fill takes; it used to reach none of that and appear in no trade at all ([ADR 0018](Decisions.md)).
- **Bad debt is unreachable, not caught.** A perp cannot end a bar owing money, on any path, because the liquidation is a price on the bar rather than a check at the end of it ([ADRs 0017, 0018](Decisions.md)). The guarantee is worth more than the check because it is what the exit price is derived from; when it was only a check, an ordinary `close()` on a bar that gapped past the liquidation stepped around it and booked the whole gap.
- **A mutation has to fail the suite.** A pass that breaks one line at a time and reruns the tests found 13 of 27 mutations surviving the Python tier, including one that held out the *start* of a series where [ADR 0032](Decisions.md) promises the end. The same pass over the Rust crates is the `mutants` stage, which CI runs on the lines each push changes; a statistic or a fill rule whose defect no test can see is one nobody should quote, so a new one arrives with a test that fails when its arithmetic is wrong rather than one that only checks its shape. Run the Python pass against `python/probatum` after adding a statistic, and pass `-p no:cacheprovider`, or a stale `.pytest_cache` will hide a mutation.

<br>

### Adversarial verification of the riskiest code

Some code is dense enough that tests alone are not enough confidence: the netted position's PnL booking, the step state machine, the unsafe zero-copy view, and the trade accounting. Each was checked before it was committed by an adversarial pass, two independent reviewers working from the decision record, not the implementation:

- An **oracle** derives the expected result (the trade log, the fill, the PnL) from the ADR conventions by hand, without reading the code.
- A **code auditor** traces the implementation on the same scenario and hunts for the specific failure modes that code has (a use-after-free, an off-by-one in a close, a stale entry price).

When the two agree, and agree with the hand-written tests, the code ships. When they disagree, the gap is a bug or a missing decision. This caught real defects during the build (a float-dust close, an unenforced reduce-only, an aliasing hazard) before they reached a commit.

<br>

### Decisions gate the code

Every non-obvious behavior is settled as a numbered [decision](Decisions.md) before the code that depends on it, so a reviewer checks the implementation against a written intent rather than a guess. The decisions are the specification the tests and the adversarial passes verify against.

<br>

### Nothing ships unverified

A change below the Python boundary is not done until cargo test, fmt, and clippy pass; a Python-facing change is not done until the Docker gate is green across the version matrix. The riskiest changes add an adversarial pass on top. Continuous integration runs the same two loops on every push to main and on every pull request, the crates tested, formatted and linted, and the one abi3 wheel built on each of its four platforms and imported and tested there on 3.13 and 3.14, the Linux ones in the gate's own stages, and it builds the `test-slow`, `test-browser`, `test-sb3`, `test-sdist` and `miri` stages exactly as they are built locally; against the commit before, it runs cargo-mutants on the changed Rust lines and compares the step loop's instruction count (see [`.github/workflows/`](../.github/workflows/)). A release runs that CI again on the tagged commit before it builds anything, where its two comparisons with the commit before find none and compare nothing, then builds every wheel and the source archive again and tests each on its own machine before it publishes.
