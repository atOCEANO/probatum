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
  <b>Validation Guide</b> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Validation Guide

The maths is checked inside the repository, against the [Specification](Specification.md) ([0043](Decisions.md)). Every rule is written there with an equation and a worked example before it is coded, every worked example is a test, and a second simulator written from that page is compared with the engine over random cases. The two pure-Rust crates are tested with cargo. Everything that crosses into Python is tested on the wheel a release ships, the Linux ones in Docker and the macOS and Windows ones on CI's runners.

<br>
<br>

## What the suite proves

| What it proves | Where | When it runs |
| :--- | :--- | :--- |
| every worked example on the Specification page, to the cent | `test_every_worked_example_holds_to_the_cent.py` | every gate stage |
| the engine agrees with the reference over random accounts, universes, arrays, batches and environments | `test_the_engine_agrees_with_the_reference.py`, `test_universes_agree_with_the_reference.py`, `test_an_array_run_agrees_with_the_reference.py`, `test_the_batch_agrees_with_the_reference.py`, `test_the_environment_agrees_with_the_reference.py` | briefly in every gate stage, at length in `test-slow` |
| the reference applies every rule it states, and keeps its invariants doing so | `test_the_reference_reaches_every_rule.py` | every gate stage, at length in `test-slow` |
| the engine's invariants hold at up to fifty instruments | `test_the_engine_keeps_every_invariant_at_fifty_instruments.py` | every gate stage |
| every depth records one ledger, and a universe weighted on one instrument is that instrument's run | `test_every_depth_gives_one_ledger.py`, `test_a_universe_weighted_on_one_instrument_is_its_own_run.py` | every gate stage |
| the optimistic setting liquidates only where every path through the bars does | `test_the_optimistic_setting_liquidates_only_when_every_path_does.py` | every gate stage |
| nothing reads a bar that has not closed: coarse bars, outside series, indicators, labels, splitters, a search's policy and a validation's pieces | `test_nothing_reads_a_bar_that_has_not_closed.py`, `test_a_label_is_one_trade_that_reads_nothing_past_its_end.py`, `test_a_splitter_trains_on_no_row_whose_label_reaches_a_test_span.py`, `test_a_search_refuses_a_policy_that_decides_on_bars_not_yet_closed.py`, `test_a_training_piece_reads_no_bar_of_the_span_before_it.py` | every gate stage |
| each statistic agrees with the reference's plain loops, and each estimator gives its paper's numbers | `test_the_statistics_agree_with_the_reference.py`, `test_every_estimator_reproduces_its_paper.py` | every gate stage |
| each interval covers what it estimates as often as it claims, and a search's tests reject as rarely as they claim | `test_intervals_cover_as_often_as_they_claim.py`, `test_a_search_s_tests_reject_as_rarely_as_they_claim.py` | `coverage` |
| an episode runs again as a backtest, and a trained agent is judged by the same backtest as a rule | `test_an_episode_runs_again_as_a_backtest_from_its_start.py`, `test_a_trained_agent_is_judged_by_the_same_backtest_as_a_rule.py` | every gate stage |
| an SB3 agent saved and loaded backtests as it did in memory | `test_sb3.py` | `test-sb3` |
| every tensor `TensorEnv` hands out equals the numpy path's, to the bit | `test_tensors_on_the_device_match_the_numpy_path.py` | `test-sb3` on the CPU, `test-gpu` on a card |
| the chart's JavaScript parses, runs and draws | `test_render.py` | `test-browser` |
| every page's examples parse and pass real keywords to probatum's functions, the Charts page's run, and the Python API's Quickstart runs with ten of its sections | `test_docs.py`, `test_examples.py`, `test_the_api_page_s_*.py` | every gate stage |

<br>
<br>

## The reference simulator

`tests/reference/simulator.py` is a second implementation of the simulator, written from the Specification and never from the engine, so a case the two disagree on is a question about a rule rather than one mistake made twice. It is slow on purpose: plain loops over plain lists, with the liquidation path's crossings found by bisection along the path, where the engine solves them in closed form.

The comparisons drive both simulators through the same decisions:
* `differential.py` decides each case's orders on the reference's account as it goes, and plays them through the engine in one call, through the binding's `run`.
* `universe_differential.py` does the same over universes of two to five spot and perpetual instruments sharing one clock with holes in it.
* `batch_differential.py` holds each account of a `Batch` to a reference run from its own row.
* `environment_differential.py` holds each environment of `probatum.env` to a reference episode, step by step, on its observation, reward, state and flags.
* `statistics.py` restates `probatum.stats` in plain loops, and the labels, the array run and the cost sweeps are compared the same way.

What the comparisons hold to:
* every count, side and decision exactly;
* every reason by its words, a clamp's or a miss's numbers held as amounts are, and any other's to twelve figures;
* every amount to within a millionth of the larger of the two values, or of one;
* and the account's state at each bar to a millionth of the case's money as well.

Each comparison counts the rules its cases reach, and fails at length when a rule both simulators apply goes unreached, or when its cases never trade, fund or liquidate. `tests/reference/trace.py` shrinks a failing case to its smallest form and prints both simulators bar by bar, and the Specification decides which of them is wrong.

<br>
<br>

## Invariants

`test_the_engine_keeps_every_invariant_at_fifty_instruments.py` runs universes of up to fifty instruments and holds every run to these:

- the ledger identity holds on every row: the change in equity is the price PnL less every cost, within its bound;
- no margin wallet ends a bar owing money;
- no fill is priced better than every price its market traded from the last close through its bar, and none worse but a limit at its own price and a forced close;
- no fill the margin bound lifts is booked past its position's average entry;
- no perpetual grows while the margin wallet is empty, and no perpetual order outlives a liquidation;
- every round trip carries its fills' fees once, and on a run that ends flat with no liquidation charge or dust the round trips sum to the change in equity.

Stepping a batch equals stepping each account alone, except under `intrabar="random"`, and a clone or a pickle steps on as its original would, to the bit ([Architecture](Architecture.md#batching-and-the-environment)).

<br>
<br>

## The stages

The gate builds the one abi3 wheel and runs the whole suite on it on each supported Python, beside the Rust checks:

```bash
docker build --target rust-checks .     # cargo fmt, clippy and the Rust tests
docker build --target test313 .         # the wheel and the suite on Python 3.13
docker build --target test314 .         # and on 3.14
```

The `builder` stage makes the wheel with maturin in PyPA's manylinux_2_28 image for the machine it runs on, so the same stages test the ARM wheel on an ARM machine or under emulation ([0044](Decisions.md)). The gate's stages install from `dev/constraints.txt`, as CI does.

Five stages sit outside the gate because each is heavier, and CI runs all five:

```bash
docker build --target test-slow .       # the comparisons at length
docker build --target test-browser .    # the chart's JavaScript, in chromium
docker build --target test-sb3 .        # the Stable-Baselines3 adapter and TensorEnv, with torch
docker build --target test-sdist .      # the source archive, built and tested
docker build --target miri .            # the candle views and the Arrow tables' memory, under Miri
```

A test marked `slow` runs only with `--slow`, which `test-slow` passes. `miri` runs the candle tests and the Arrow table's under Miri, which checks every memory access and reports what is never freed, on a nightly compiler pinned by date. `test-sdist` installs the source archive as pip does on a machine with no wheel, and runs the suite against it.

Two stages run by hand. The gpu stage needs a card, which CI's runners do not have, so after any change to the device code its image is built and then run with the card handed in. A test marked `gpu` is skipped where no card is found:

```bash
docker build --target test-gpu -t probatum-test-gpu .
docker run --rm --gpus all probatum-test-gpu
```

The coverage stage runs the cells marked `coverage`, which run only with `--coverage`, as the stage passes it. Run it after any change to a statistic: each cell prints its rate, and a cell outside its band is a defect of its statistic ([0031](Decisions.md), [0033](Decisions.md)):

```bash
docker build --progress=plain --target coverage .
```

The rest measure or produce rather than assert:

```bash
docker build --target bench .                                       # engine throughput
docker build --target bench-surfaces -t probatum-bench-surfaces .   # then docker run it
docker build --target bench-instructions .                          # step instruction counts
docker build --target mutants -t probatum-mutants .                 # then docker run it
docker build --target charts -t probatum-charts .                   # the documentation's charts
docker build --target diagrams -t probatum-diagrams .               # its diagrams
```

`bench` and `bench-surfaces` print throughput, which depends on the machine. `bench-instructions` counts the step loop's instructions under Valgrind, which does not depend on the machine, and `mutants` runs cargo-mutants over the two pure-Rust crates and lists every mutant the tests miss. The [Contributor Guide](Contributor_Guide.md#the-dev-directory) gives the recipes of `charts` and `diagrams`.

<br>
<br>

## Continuous integration

`.github/workflows/ci.yml` runs on every push to main and every pull request:
* the Rust tests, formatting and lints;
* the wheel built and tested through `test313` and `test314` on Linux x86_64 and ARM runners;
* the wheel built on macOS and Windows and tested there on Python 3.13 and 3.14;
* the five stages outside the gate;
* cargo-mutants over the Rust lines the push changed, failing on a mutant the tests miss;
* and the step loop's instruction count against the commit before, failing on a rise of more than 10%.

A release runs the same CI on the tagged commit before it builds anything ([Contributor Guide](Contributor_Guide.md#releasing)).
