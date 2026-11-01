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
  <b>Migration</b> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Migration

One hop: emsl 1.3.2 to probatum 2.0.0. probatum is emsl's repository renamed and released as 2.0.0, and it does not stay compatible with emsl ([0001](Decisions.md)): a compatibility mode that kept emsl's numbers would be a second engine to test, and a way to reach a flattering number by accident. Names move, the inputs change shape, and where emsl simulated something no venue does, a result moves. This page says what changed and why, then gives every public emsl name and every emsl record number its place in probatum.

<br>

## emsl 1.3.2 to probatum 2.0.0

**The package is probatum, installed from its release.** emsl was installed with `pip install --find-links <release> emsl` and imported as `emsl`. probatum is installed from its wheel's URL, as the [Quick Start](../README.md#quick-start) shows, and imported as `probatum`. Its version continues emsl's line, so 2.0.0 follows 1.3.2, and the repository, renamed from `embeddable-market-simulation-library`, still answers at the old address. The base install depends on numpy alone: Gymnasium comes with the `rl` extra, and Optuna and cloudpickle, which emsl's `tune` extra brought, come with `search`.

**Bars are a Data, and the venue is an argument.** emsl's verbs took a pandas frame and the market's settings as keywords, or came from a `Market` that held the settings once and handed the verbs out. probatum's take a `probatum.Data`, built from any frame, from arrays or from exchange-router, and a venue, a preset's name or a `probatum.Venue`. No verb takes a single venue setting as a keyword, so a setting has one place to be given and cannot arrive twice with two values ([0002](Decisions.md)). A venue built by hand starts from each setting's plain option, so a setting it leaves out takes that option:

```python
# emsl 1.3.2
result = emsl.backtest.Backtester(frame, market="perp", fee_taker=0.0005,
                                  fee_maker=0.0002, funding_rate=0.0001,
                                  funding_interval=8).run(Cross())

# probatum 2.0
venue = probatum.Venue(fees={"maker": 0.0002, "taker": 0.0005},
                       funding={"every_bars": 8, "rate": 0.0001})
data = probatum.Data(frame, symbol="BTCUSDT", market="linear")

result = probatum.backtest(Cross(), data=data, venue=venue)
```

**A strategy's `next` gets the bar and its account.** emsl passed a state dict and the engine itself; probatum passes the bar and the account, both typed, and each order method moves to the account ([0025](Decisions.md)). `next` is not called before a declared warm-up, while the bars before it still advance and fill resting orders, and a strategy's parameters are its constructor's arguments. Every method that places an order returns its `Order`, a refused one included with its reason, where emsl returned an id or `None` ([0010](Decisions.md)):

| emsl 1.3.2, on the engine | probatum 2.0, on the account |
| :--- | :--- |
| `market_buy(size)`, `market_sell(size)` | `buy(size)`, `sell(size)` |
| `limit_buy(size, price)`, `limit_sell(size, price)` | `buy(size, limit=price)`, `sell(size, limit=price)` |
| `stop(...)` | `buy(size, stop=trigger)` or `sell(size, stop=trigger)` |
| `order(...)` with a time in force, `post_only` or `reduce_only` | `buy` or `sell` with `tif`, `post_only` or `reduce_only` |
| `close()`, `cancel(order_id)`, `cancel_all()` | `close()`, `cancel(order)`, `cancel_all()` |
| `replace(...)` | `replace(order, price=, trigger=, size=)` |
| `qty_from_weight(fraction)`, `qty_from_quote(cash)`, then an order of that size | `target(weights=fraction)`, `buy_value(amount=cash)`, `sell_value(amount=cash)` |
| an order id, or `None` when refused | an `Order` handle, refused ones included with their reason |
| `state["position"]`, `state["equity"]` and the rest of the dict | `account.position`, `account.equity`, typed |

**Every order emsl had stays, and three do more.** Time in force, all-or-nothing judged after every clamp, a stop that cannot fill not counting as triggered, and `replace` keeping an order's id carry over, now on a named instrument. `reduce_only` works on every order type, where emsl had it on stops alone; a `post_only` limit the first bar opens through is cancelled, where emsl filled it as a taker; and `bracket` places two exits as one OCO pair ([0010](Decisions.md)).

**A search and a validation return a Study.** `tune` becomes `probatum.search`, and `walk_forward` becomes `probatum.validate` with a `cv=probatum.cv.WalkForward(...)` splitter; both return a `probatum.Study`. It holds a table of the trials beside every trial's returns, which the statistics across a search need and emsl discarded, and its winner as a Result, run again when first read ([0026](Decisions.md)). A validation's seams are explicit: at a window's first close every order the outgoing window left is cancelled with the reason `seam` and `on_inherit` hands over the position, where emsl left resting orders behind with nothing owning them ([0034](Decisions.md)). Each fit's search is seeded from a stream of its own, where emsl gave every window the same seed.

**Results move where emsl was wrong.** probatum does not promise emsl's numbers, and each move has its reason:

- **Funding** lands at the venue's funding times, on the positions held then, and is charged after the bar's liquidation check. emsl charged it every n bars from the start of the array, and before the check, so a payment not yet made could liquidate an account at the bar's low ([0020](Decisions.md)).
- **Liquidation** triggers at the maintenance requirement where a venue sets one, earlier than emsl's trigger at zero equity, at the first point on the bar's path that reaches it ([0017](Decisions.md)). A fill the margin bound lifts stays inside its bar, where emsl lifted it past the bar's high ([0018](Decisions.md)).
- **A limit that has rested** through a bar fills as a maker whatever later bars do, where emsl charged the taker rate to any limit a bar opened through, rested or not, and an order against the account's own resting order is refused ([0011](Decisions.md)).
- **Fills are sized against the trailing volume**, by a liquidity budget each bar that every order on it shares, where emsl capped each order at a share of the volume of the bar it filled on, which a split order multiplied ([0013](Decisions.md)).
- **A take-profit on a bar that also reaches the liquidation** is held back, because the bar cannot show which came first. One setting, `intrabar`, decides every such event, pessimistic by default as emsl already was for two exits on one bar, and `intrabar="optimistic"` decides each the other way ([0012](Decisions.md)).
- **A trade is a round trip**, its net after its fees and the funding it paid while open, so `num_trades` and the `min_trades` floor count decisions rather than closing fills, and a round trip still open at the end is reported, where emsl left it out of every trade metric ([0023](Decisions.md)).
- **The equity curve has one point per bar**, starting at the opening balance, and the fill log says what each clamp took, where emsl shrank a clamped fill without a word ([0022](Decisions.md)).
- **A Batch reads its features by row** whatever order numpy holds them in, where emsl's Batch, called directly rather than through `VectorEnv`, copied them as stored, so an array held by column, as a transpose or a frame's values often are, gave every bar another bar's features.
- **Indicators run over each instrument's own values**, a `NaN` or infinite value being a row the series does not have, so a window, a smoother and a crossing carry across it where emsl restarted them; `shift`, `change` and `roc` lag over values rather than rows, a cross across a gap is reported on the first bar after it, `true_range` after a gap reads the close before it, and `stoch`'s `d` averages over the bars `k` has ([0037](Decisions.md)).
- **Annualisation counts the bars a year of the clock holds**, so a stock's daily bars annualise at about 252 where emsl's median gap gave 365, and a crypto hourly series at 8,760 as before ([0004](Decisions.md)).
- **`cost_curve` moves every cost** a venue charges for trading, its fees, its spread, its slippage and its impact together, where emsl's moved the fee alone ([0045](Decisions.md)).

**Errors keep their built-ins, and warnings get one class.** Every error probatum raises is one of its own classes under `probatum.ProbatumError` and also the built-in it means, so `except ValueError` catches what it caught under emsl. Every warning is a `probatum.ProbatumWarning`, a `UserWarning`, and long work is logged to the `probatum` logger rather than printed, where emsl raised and warned with the built-ins themselves ([0029](Decisions.md)).

**The environment follows Gymnasium.** `probatum.env` is a Gymnasium vector environment over the Rust batch, with all three of Gymnasium's autoreset modes and Gymnasium's own default, `NEXT_STEP`, as its default, where emsl had `SAME_STEP` alone; the Stable-Baselines3 adapter switches to `SAME_STEP`, which SB3 expects. Order actions are decoded in Rust, which ends emsl's market-order-only batch, the reward is floored so a liquidation is a large finite penalty, and costs can be drawn per environment ([0038](Decisions.md)). A single environment fills the same output buffer, which replaces emsl's read-only view onto the candle buffer, since an observation now carries the account beside the bars ([0040](Decisions.md)).

**The record starts again at 0001.** emsl's record, in the repository's history at its 1.3.2 tag, grew to 113 entries numbered to 0115 by recording behaviours one at a time. probatum keeps exchange-router's measure, one decision a record, in 47 records, and each emsl entry is absorbed into one of them, carried as it stood, amended, or replaced where emsl was wrong. A comment or a commit that cites an emsl number is followed through [Record numbers](#record-numbers).

<br>

### Names

| emsl 1.3.2 | probatum 2.0 |
| :--- | :--- |
| `Engine` | `probatum.Engine` |
| `Batch` | `probatum.Batch` |
| `Market` | `probatum.Venue` |
| `Market.engine`, `.backtest`, `.tune`, `.walk_forward`, `.env` | gone: the venue is an argument of `probatum.Engine`, `backtest`, `search`, `validate` and `env` |
| `Market.replace`, `Market.as_dict` | `Venue.replace`, `Venue.to_dict()` |
| `Strategy`, `backtest.Strategy` | `probatum.Strategy` |
| `backtest.Backtester(...).run(s)` | `probatum.backtest(s, data=data, venue=venue)` |
| `backtest.BacktestResult` | `probatum.Result` |
| `to_ohlcv` | `probatum.Data` |
| `tune`, `tune.TuneResult` | `probatum.search`, `probatum.Study` |
| `tune.Int`, `tune.Float`, `tune.Categorical` | `probatum.Int`, `Float`, `Categorical`, checked as they are built; a tuple, a list or a `tune` range in a space is refused |
| `tune(n_trials=)`, `tune(oos=)` | `search(trials=)`, `search(holdout=)`, a share, a number of rows or a time |
| `tune(market=, quote=, fee_taker=, ...)` | `search(venue=)`, a Venue or a preset's name |
| `tune(verbose=)` | gone: a search logs each trial at `DEBUG` and its outcome at `INFO` to the `probatum` logger |
| `TuneResult.best_value` | `Study.best_score` |
| `TuneResult.best_result`, `.best_stats` | `Study.best`, the winner run again at the full level, and `Study.best.stats` |
| `TuneResult.oos_result`, `.oos_stats` | `Study.evaluate_holdout()`, which counts its look, then `Study.holdout` and `Study.holdout.stats` |
| `TuneResult.trials`, a list of dicts | `Study.trials`, a table of a row a trial, and `Study.returns`, every trial's returns |
| `TuneResult.objective`, `.sampler`, `.direction`, `.min_trades`, `.data_hash` | `Study.provenance`, with the sampler's arguments, and the in-sample bars' fingerprint in place of the hash |
| `TuneResult.study`, `.best_strategy()` | gone: a Study keeps its trials rather than Optuna's study, and `Study.best_params` are the arguments the winner is built from |
| `walk_forward`, the `WalkForward` it returns | `probatum.validate(..., cv=probatum.cv.WalkForward(...))`, `probatum.Study` |
| `walk_forward(windows=W, train=s)` on T bars | `cv=probatum.cv.WalkForward(train=round(s * T), test=(T - round(s * T)) // W)`, in bars, the windows from the first bar and the last taking the remainder, so there are more than W where the remainder holds a whole window; `holdout=0` where nothing is to be held out, which emsl never did |
| `walk_forward(n_trials=)`, `walk_forward(seed=)` | `validate(search={"trials": ...})`, and `validate(seed=)`, each fit's search seeded from a stream of its own |
| `WalkForward.result`, `.stats` | each path's statistics over the bars it owns in `Study.paths`, and its returns in `Study.returns`; emsl's statistics counted the flat stretch before the first window |
| `WalkForward.windows`, `.span` | `Study.fits`, a row a split with its winner and fitted score, and `Study.windows`, a row a stretch one fit traded with its rows, times and traded statistics |
| `WalkForward.direction` | `Study.provenance`, with the search's settings |
| `WalkForward.decay`, `.consistency` | `Study.decay()`, `Study.consistency()` |
| `rl.VectorEnv` | `probatum.env`, `probatum.rl.VectorEnv` |
| `rl.VectorEnv(action_fn=)` | `probatum.env(decode=)` |
| `rl.VectorEnv(reward_fn=)` | `probatum.env(reward=)`, over typed state |
| `sb3.EmslVecEnv` | `probatum.sb3.wrap` |
| `ta.*`, the 30 functions, and `ta.Bands`, `Channel`, `Macd`, `Stochastic`, `Trend` | `probatum.ta.*`, the same names, each over a `(T,)` series or a `(T, N)` universe along `axis`, where emsl refused two dimensions and flattened a `(T, 1)` column; a length longer than the series, or rsi's at or above it, gives all warm-up rather than a refusal; a parameter that is not an int or a float, text and a Decimal among them, is a `TypeError`, where emsl raised a `ValueError` for a length and read a width or a level written as text as its number, and a bool is refused as a number |
| `chart(frame, run, ...)` | `result.chart(...)`, or `data.chart(...)` for bars alone |
| `chart_defaults` | `probatum.chart.defaults`, which takes `price=` beside emsl's `theme`, `height`, `palette` and `drawdown` |
| `plot.Panel`, `Line`, `Histogram`, `Band`, `Level`, `Marker`, `Markers`, `Background`, `Recorder`, `ramp`, `at_bar`, `at_next` | `probatum.chart.marks.*` |
| `metrics.returns` | `Result.returns` |
| `metrics.sharpe`, `rolling_sharpe`, `sharpe_interval`, `skew`, `kurtosis`, `omega`, `tail_ratio` | `probatum.stats.*` |
| `metrics.drawdown`, `drawdown_table`, `time_under_water`, `ulcer_index`, `value_at_risk`, `conditional_value_at_risk` | `probatum.stats.*` |
| `metrics.probabilistic_sharpe`, `deflated_sharpe`, `deflation_threshold`, `min_track_record_length` | `probatum.stats.*`, reading the Newey-West variance |
| `metrics.autocorrelation`, `effective_sample` | `probatum.stats.*`, the effective sample no longer the default |
| `metrics.period_returns`, `session_buckets` | `probatum.stats.*`, with no `frame` argument, because a Result knows its time |
| `metrics.excursions` | the `best` and `worst` columns of `Result.trades` |
| `metrics.segment`, `long_short_split`, `trade_stats` | `probatum.stats.*` |
| `metrics.buy_and_hold` | `probatum.stats.benchmark`, which still takes another asset's series |
| `metrics.decompose` | `Result.costs` and `probatum.stats.decompose` |
| `metrics.cost_curve`, `breakeven_bps` | `probatum.cost_curve`, `probatum.breakeven_bps`, moving every cost |
| `metrics.compare` | `probatum.compare` |
| `metrics.report`, `summary` | `Result.stats`, `Result.summary()` |
| `ValueError`, `TypeError`, `KeyError` and the other built-ins emsl raised | the same built-ins, each also one of probatum's classes under `probatum.ProbatumError` |
| `UserWarning` and `RuntimeWarning` | `probatum.ProbatumWarning`, a `UserWarning` |

<br>

### Record numbers

Every number emsl 1.3.2's record carries, the entry it titles, and the probatum record that holds its decision now. An entry with two numbers, one amending or extending the other, appears under each.

| emsl | emsl's record | probatum |
| :--- | :--- | :--- |
| 0001 | Flip-through-zero PnL | [0014](Decisions.md) A flip realizes through zero, and each fee is paid once |
| 0002 | Funding | [0020](Decisions.md) Funding is charged when the venue charges it |
| 0003 | Liquidation | [0017](Decisions.md) A liquidation is a price on the bar's path |
| 0004 | Slippage | [0013](Decisions.md) A fill is what a venue would give |
| 0005 | Volume cap | [0013](Decisions.md) A fill is what a venue would give |
| 0006 | Bar-fidelity limits | [0011](Decisions.md) A resting limit is on the book |
| 0007 | Statistics | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0008 | Zero-copy observation | [0040](Decisions.md) One output buffer, and a path to the GPU |
| 0009 | Trade recording | [0022](Decisions.md) A run records its ledger, its fills and its orders |
| 0010 | RL autoreset | [0038](Decisions.md) The environment shows the agent its account |
| 0011 | Compiled sweep, and why it is not in the Python API | [0003](Decisions.md) Every run steps every bar |
| 0012 | Leverage cap | [0015](Decisions.md) Margin is shared, and an account over its cap may only shrink |
| 0013 | Market impact | [0013](Decisions.md) A fill is what a venue would give |
| 0014 | Per-env cost randomization | [0038](Decisions.md) The environment shows the agent its account |
| 0015 | Spot cannot short | [0008](Decisions.md) One account, two wallets |
| 0016 | Time in force | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0017 | Funding | [0020](Decisions.md) Funding is charged when the venue charges it |
| 0018 | Spot buys clamp to cash | [0008](Decisions.md) One account, two wallets |
| 0019 | A non-positive equity is terminal | [0008](Decisions.md) One account, two wallets |
| 0020 | The batched tier is market-order-only | [0038](Decisions.md) The environment shows the agent its account |
| 0021 | Tuning over the Strategy spine | [0032](Decisions.md) A search never sees its holdout |
| 0022 | Stable-Baselines3 adapter | [0038](Decisions.md) The environment shows the agent its account |
| 0023 | Cash moves only with the position | [0014](Decisions.md) A flip realizes through zero, and each fee is paid once |
| 0024 | The cap and the slip are bounded | [0013](Decisions.md) A fill is what a venue would give |
| 0025 | All-or-nothing is decided after the clamps | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0026 | An insolvent perp may only shrink | [0015](Decisions.md) Margin is shared, and an account over its cap may only shrink |
| 0027 | The boundary validates its arguments | [0007](Decisions.md) The boundary refuses what it cannot simulate |
| 0028 | Protective stops and stable order ids | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0029 | Statistics stay finite, monotone, and net | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0030 | A trade carries its whole round trip | [0023](Decisions.md) A trade is a round trip |
| 0031 | A run that never filled is visible | [0022](Decisions.md) A run records its ledger, its fills and its orders |
| 0032 | Moving a resting order | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0033 | Order status is not modelled | [0022](Decisions.md) A run records its ledger, its fills and its orders |
| 0034 | A search needs a floor on activity | [0032](Decisions.md) A search never sees its holdout |
| 0035 | A stop that cannot fill has not triggered | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0036 | A parallel search is not reproducible from the seed | [0021](Decisions.md) The same numbers on every machine |
| 0037 | A series is aligned by position, and a short array draws late | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0038 | A NaN is a gap, never a dropped row | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0039 | A bare array overlays the candles unless it would flatten them | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0040 | The renderer is vendored into the wheel | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0041 | A chart is precomputed and self-contained, never served | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0042 | The chart computes nothing and knows no indicator names | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0043 | The spec is the seam, and the bundle is an explicit list | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0044 | Plotting is not declared inside a strategy | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0045 | A crossing limit pays the taker rate | [0011](Decisions.md) A resting limit is on the book |
| 0046 | A zero denominator ranks at infinity, not at zero | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0047 | The queue of pending market orders is bounded | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0048 | The annualization is read from the candles | [0004](Decisions.md) Time is the bars that exist |
| 0049 | A tuned result is in-sample until it has been held out | [0032](Decisions.md) A search never sees its holdout |
| 0050 | A warm-up is declared, not guarded by hand | [0025](Decisions.md) A Strategy sees only the past and its own account |
| 0051 | A result says what produced it | [0026](Decisions.md) One Result and one Study |
| 0052 | Liquidation | [0018](Decisions.md) No account ends a bar owing money |
| 0053 | The venue is one object, and it hands out the surfaces | [0002](Decisions.md) Four depths of one step |
| 0054 | The deflated Sharpe needs a null it cannot supply itself | [0032](Decisions.md) A search never sees its holdout |
| 0055 | Fourteen indicators, and the conventions written down | [0037](Decisions.md) Indicators run over each instrument's own bars |
| 0056 | A wide bar books the worse exit | [0012](Decisions.md) What a bar cannot show is decided by one setting |
| 0057 | A walk-forward is one run, not a stitching | [0034](Decisions.md) A validation path is one run |
| 0058 | The deflation is about the search you ran | [0032](Decisions.md) A search never sees its holdout |
| 0059 | A metric that takes a frame says which frame | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0060 | A walk-forward window is scored on the run that happened | [0034](Decisions.md) A validation path is one run |
| 0061 | Confidence is counted in bets, not in bars | [0031](Decisions.md) Every interval names its method |
| 0062 | The trade log gets a distribution, and the run gets a calendar | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0063 | Fourteen indicators, and the conventions written down | [0037](Decisions.md) Indicators run over each instrument's own bars |
| 0064 | The recursive indicators are vectorized between their gaps | [0037](Decisions.md) Indicators run over each instrument's own bars |
| 0065 | What 1.0 promises, and what it does not | [0001](Decisions.md) probatum replaces emsl rather than staying compatible with it |
| 0066 | A documented example that composes two calls gets an executable test | [0043](Decisions.md) The maths is verified inside the repository |
| 0067 | A liquidation cannot leave you owing money | [0017](Decisions.md) A liquidation is a price on the bar's path |
| 0068 | A stop that cannot fill has not triggered | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0069 | Moving a resting order | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0070 | The boundary validates its arguments | [0007](Decisions.md) The boundary refuses what it cannot simulate |
| 0071 | A wide bar books the worse exit | [0012](Decisions.md) What a bar cannot show is decided by one setting |
| 0072 | Statistics stay finite, monotone, and net | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0073 | A NaN is a gap, never a dropped row | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0074 | Market impact | [0013](Decisions.md) A fill is what a venue would give |
| 0075 | A panel with nothing to scale by draws nothing, which is the worst way to be wrong | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0076 | The legend cannot report a colour the chart did not paint | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0077 | A control nobody clicks is a control nobody knows is broken | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0078 | A documented example that composes two calls gets an executable test | [0043](Decisions.md) The maths is verified inside the repository |
| 0079 | A stop is consumed by filling, and only by what it filled | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0081 | The trade log gets a distribution, and the run gets a calendar | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0082 | Funding is marked on the bar the account was actually in | [0020](Decisions.md) Funding is charged when the venue charges it |
| 0083 | A refused replacement keeps its place in the queue | [0010](Decisions.md) Orders keep emsl's semantics, on a named instrument |
| 0084 | The tools that check the library are not the library | [0043](Decisions.md) The maths is verified inside the repository |
| 0085 | A documentation picture is built from data that cannot move | [0043](Decisions.md) The maths is verified inside the repository |
| 0086 | A per-env cost is an argument like any other | [0007](Decisions.md) The boundary refuses what it cannot simulate |
| 0087 | Every picture in the documentation is produced by something committed here | [0043](Decisions.md) The maths is verified inside the repository |
| 0088 | A page may only point at something that ships | [0043](Decisions.md) The maths is verified inside the repository |
| 0089 | A golden records what the code does, not what it was meant to do | [0043](Decisions.md) The maths is verified inside the repository |
| 0090 | A number computed about a search knows which direction it searched | [0033](Decisions.md) A search is judged across all its trials |
| 0091 | A run that ended at zero says so | [0018](Decisions.md) No account ends a bar owing money |
| 0092 | A sweep names the strategy it cannot build | [0045](Decisions.md) A strategy is judged at the size and the costs it would trade |
| 0093 | A number that saturates has to saturate on both sides of the boundary | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0094 | The margin bounds the fill, not the bar | [0018](Decisions.md) No account ends a bar owing money |
| 0095 | An excursion is a bound, and it counts only the part of a bar the trade was in | [0023](Decisions.md) A trade is a round trip |
| 0096 | The four prices have to be a bar | [0007](Decisions.md) The boundary refuses what it cannot simulate |
| 0097 | A reference draws after the thing it is a reference to | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0098 | The equity panel starts at the balance the run opened with | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0099 | Room asked for is room framed | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0100 | A tail is taken by rank, not by comparison | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0101 | A span covers the bars it names | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0102 | A position reported in bars is a bar index | [0030](Decisions.md) The statistics stay finite and rank monotonically |
| 0103 | A cost sweep moves the cost and nothing else | [0024](Decisions.md) `backtest` accepts anything that decides, read one way |
| 0104 | A backtester owns the bars it fingerprinted | [0005](Decisions.md) A Data owns its bars |
| 0105 | A venue's five surfaces take the same input | [0002](Decisions.md) Four depths of one step |
| 0106 | A stated identity has to be the one the code satisfies | [0022](Decisions.md) A run records its ledger, its fills and its orders |
| 0107 | A coverage guard that over-reports is worse than none | [0043](Decisions.md) The maths is verified inside the repository |
| 0108 | A fixture degenerate in the dimension its claim is about tests nothing | [0043](Decisions.md) The maths is verified inside the repository |
| 0109 | A chart's appearance is settled where the information is | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0110 | A shading that means something carries its key | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0111 | A dense axis is aggregated, and a grain finer than the span is noise | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0112 | A shading that means something carries its key | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0113 | A saved chart carries the evidence for its own claim | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
| 0114 | A number written out in prose is a claim, and the gate reads it | [0043](Decisions.md) The maths is verified inside the repository |
| 0115 | The chart computes nothing and knows no indicator names | [0042](Decisions.md) The interactive chart is the one picture probatum draws |
