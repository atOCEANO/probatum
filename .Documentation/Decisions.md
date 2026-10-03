<h1>OCEΛNO <small><code>embeddable-market-simulation-library</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/embeddable-market-simulation-library/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/embeddable-market-simulation-library?label=release&color=2ea043" alt="Latest release" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <a href="../README.md">Introduction</a> &nbsp;•&nbsp;
  <a href="Python_API.md">Python API</a> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Plotting.md">Plotting</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <b>Decisions</b> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Decisions

This page records the choices that could reasonably have gone another way, and why each went the way it did. The record restarts at 0001 with probatum: the decisions emsl took that still hold are carried over, numbered in topic order, and emsl's own numbers stay in its history. The code, the tests and the other guides cite them by number (`ADR 0030`).

Each entry is one paragraph saying what was decided and why.

<br>

**0001 Flip-through-zero PnL.** A fill that crosses zero closes the old side, booking realized PnL on that part at the old average entry, and opens the remainder at the fill price. A same-side add volume-weights the entry, and a non-finite or non-positive size fills nothing.

**0002 Slippage.** A market fill slips by `slippage_bps` off the next bar's open, against the side. It is taken off the next open rather than the decision bar's close, because the close is the price the strategy read when it decided, and a slip against it would make the fill price known at the moment of the decision.

**0003 Volume cap.** One order takes at most `max_fill_fraction` of a bar's volume, so a large limit fills over several bars.

**0004 Market impact.** An extra adverse slip proportional to the share of the bar's volume the fill takes, so a bigger order pays more. A per-fill cost function is refused; this coefficient stands in for it. ADR 0005 extends it.

**0005 A taker price never leaves the bar that produced it.** A taker fill is held inside the bar's range, at worst the high for a buy and the low for a sell, so impact cannot price a fill at a level the bar never traded. On a bar that printed a single price, slippage is therefore zero.

**0006 The cap and the slip are bounded.** The volume cap guards against `NaN` on both sides, so neither a non-finite remaining size nor a non-finite fraction can fill a whole cap or remove it. Slippage and impact are adverse by definition, so both are floored at zero, and their total is held just under 1, since a slip of 1.0 would price a sell at zero. ADR 0005 extends it.

**0007 Time in force.** `GTC` rests, `IOC` takes one bar and cancels the rest, and `FOK` fills the whole size against one bar or nothing. A market order is `IOC` unless `FOK` is asked for, so on a bar with no volume it fills nothing and is gone, while a `GTC` limit waits for the next bar.

**0008 All-or-nothing is decided after the clamps.** `FOK` is judged against the size that can actually fill after the margin cap, the spot cash and short clamps and `reduce_only`, so a `FOK` that could only fill in part books nothing. A resting `FOK` limit is judged against the account as it stands when its turn comes.

**0009 A crossing limit pays the taker rate.** A limit already through the market at the bar's open is marketable and pays the taker rate; a limit the market comes to pays the maker rate. The fill still prices at the limit and never better. With negative maker rates allowed and two crossing limits able to fill on one wide bar (ADR 0018), booking every limit as a maker let a straddle of limits gain on every bar.

**0010 The queue of pending market orders is bounded.** `max_open_orders` bounds the pending market queue as well as the resting book, so splitting an order into many slices cannot take many times the per-order volume cap against one bar. A liquidity budget shared across a bar is the full answer and is left for a later release. `market_buy` and `market_sell` return an optional id.

**0011 A stop that cannot fill has not triggered.** A stop is consumed only by filling, so on a bar that trades no volume it stays armed for the next bar that can trade. The cost is that a stop can fire later than a live venue would fill it, which is the safer direction for the error on illiquid data. ADR 0012 extends it.

**0012 A stop is consumed by filling, and only by what it filled.** A stop accumulates its fills and rests until nothing remains, as a limit does, so a clamp that reduces a fill to nothing, or a bar with a sliver of volume, does not delete the protection. A triggered stop is therefore not `IOC`. ADR 0013 extends it.

**0013 An order retires at the epsilon it can no longer fill at.** A resting order is retired when its remaining size falls to the dust epsilon below which fills are refused (ADR 0026), not only at exactly zero, so float residue cannot hold a book slot forever.

**0014 Protective stops and stable order ids.** `stop()` takes `reduce_only`, so a stop left over after its position closed cannot open a position on the other side. Order ids are unique for the engine's life rather than per episode, so a handle kept across a reset cannot address an order from the next episode. ADR 0015 supersedes it in part.

**0015 Moving a resting order.** `replace(id, ...)` cancels a resting order and rests a replacement with the same side, kind and flags. It returns `None` and places nothing when the id is no longer resting, so a trailing stop moved with `replace` can never leave two orders alive. There are no OCO groups. ADR 0016 extends it.

**0016 A refused replacement puts the original back.** When `replace` cannot place the replacement (a non-finite trigger, a non-positive size, a `post_only` limit that would cross), the cancelled original is restored with its id, so `None` still means nothing happened.

**0017 A refused replacement keeps its place in the queue.** A restored original (ADR 0016) goes back into the slot it held, not the first free slot, because slot order decides which fill gets the room when a clamp binds. A replacement that succeeds is a new order and takes the first free slot.

**0018 Bar-fidelity limits.** At bar granularity the fill model cannot see the path inside a bar, so it can be more generous than a live book, and two contradictory resting orders can both fill on one wide bar. Resting orders resolve in book-slot order and a cancelled slot is reused by the next placement, so when a cash or margin clamp binds, which order gets the room depends on slot order rather than placement time. Two exits from one position are settled by ADR 0019 instead.

**0019 A wide bar books the worse exit.** When a bar's range reaches both a stop and a take-profit on the same position, the exits are applied worst first, by the PnL each would realize. Slot order would make the sign of a bracket's result depend on which line the strategy placed first. Only fills that reduce the position are reordered. ADR 0020 corrects it.

**0020 Only the exits reorder.** The worst-first sort of ADR 0019 reorders exits among the slots exits already hold; every other fill keeps its slot, so an entry cannot be funded by an exit that may have happened after it.

**0021 Spot buys clamp to cash.** A spot buy fills only what the quote balance affords, so a buy never drives the account negative.

**0022 Spot cannot short.** On spot a sell is clamped to the current holding; shorting needs a borrow this tier does not model, so shorts and flips through zero are perp behaviours.

**0023 The boundary validates its arguments.** Every argument crossing into the engine is checked, not only the candles: a non-finite or out-of-range quote, fee, slippage, impact, fill fraction, order bound or observation window raises, naming the argument and what was expected. A non-finite limit price or stop trigger is refused a book slot, since every comparison against it is false and it would hold the slot until the book filled. ADR 0024 extends it.

**0024 The boundary refuses what cannot trade, on both sides of the book.** A candle price at or below zero is refused, because the risk clamps divide by the mark and do nothing when it is not positive. A market order whose size can never fill does not take a slot in the pending queue.

**0025 The four prices have to be a bar.** The boundary requires `low <= open <= high` and `low <= close <= high` on every row, inclusive, because the taker clamp (ADR 0005) only bounds a fill while the high is above the low, and a feed with two columns transposed can pass a high-against-low check alone. The error names the column order.

**0026 Cash moves only with the position.** A fill that is non-finite or at or below the dust epsilon moves neither the position nor cash nor fees, so quote cannot change unless the position changes with it. The sizing helpers return zero rather than an infinity when the mark is not finite and positive.

**0027 Leverage cap.** `leverage` caps a perp's notional at that multiple of equity; the default is 10x and `0.0` means uncapped. Spot ignores it. A position over the cap because equity fell is not force-reduced, only blocked from growing.

**0028 An insolvent perp may only shrink.** With equity at or below zero the margin allowance is zero, so only fills that reduce the position pass. Switching the cap off at zero equity, on the grounds that liquidation would handle it, let a flat account open a position from negative equity.

**0029 A non-positive equity is terminal.** Equity at or below zero ends the account on both markets; the RL env reads it as a termination, not a truncation.

**0030 Funding.** A perp charges funding on the held notional: a long pays a positive rate and a short receives it. The cadence is counted in bars from the absolute bar index (`funding_rate`, `funding_interval`), and funding is charged before the liquidation check, so a funding debit can bust the account on the same bar.

**0031 Funding is marked on the bar the account was actually in.** Funding is marked at the resolving bar's close: the raw close when there is no liquidation fence, and the fence when the bar ran past it, so a position is never charged or credited for prices after it was closed out.

**0032 Liquidation.** A perp is force-closed when its margin is exhausted. The trigger is tested at the bar's adverse extreme, the low for a long and the high for a short, and the close is priced where the margin runs out (ADR 0033). ADR 0033 amends it.

**0033 A liquidation cannot leave you owing money.** The forced close is priced where the margin runs out, not at the bar's adverse extreme, so the account ends at exactly zero and bad debt is unreachable rather than clamped afterwards. The liquidation fee is priced inside that exit. There is no maintenance margin, so 10x means a 10% adverse move rather than the slightly smaller one a real venue uses. ADR 0034 extends it.

**0034 A liquidation is a price on the bar, not a check at the end of it.** The liquidation level is computed from the position carried into the bar before any order resolves, and the bar is clipped at it, so no order fills beyond it; a bar that opened past it closes the position first and fills nothing else. Checked after the fills instead, an exit that flattened the account first booked the whole loss past the margin. The level is read once per bar, so a partial exit is not credited within the bar it happens on (see ADR 0035).

**0035 The margin bounds the fill, not the bar.** Every fill that reduces a perp position is bounded at the bankruptcy price when it lands, so no ordering of fills within a bar can book past the point where the margin runs out, whether the position grew during the bar, the account entered the bar flat, or the fill is a resting limit. A take-profit on a healthy account still fills at its limit. The cost is a floating-point residue of about 1e-12 on an account bounded exactly at that point, which then stays live with a dust balance; ADR 0029's terminal test is not loosened for it.

**0036 Trade recording.** One trade row per closed portion of a position, with `pnl` gross of fees, `fees` the round trip on that size (ADR 0037) and `net_pnl` the difference. A position still open when the data ends is not closed: it counts in `total_return_pct` and `exposure_pct` and in none of the trade metrics.

**0037 A trade carries its whole round trip.** A trade row's `fees` is the entry and exit fees on the size it closes, and `net_pnl` is `pnl - fees`. On a run that ends flat with no liquidation, the trades' net PnL equals the change in equity.

**0038 An excursion is a bound, and it counts only the part of a bar the trade was in.** A trade's best and worst excursion take the entry bar from the fill to its close, the exit bar from its open to the fill, and the bars between whole, so both numbers are bounds that never overstate. A market entry that lived through its whole entry bar is not credited for it, since a trade row does not record which order opened it.

**0039 A result says what produced it.** A `BacktestResult` carries the engine settings it ran under, a short fingerprint of its candles, the strategy's name and repr, and the library version, so two runs can be compared later. The fingerprint is computed once per backtester. `to_dict` leaves out the equity curve and the trade log.

**0040 Statistics.** The statistics (return, CAGR, Sharpe, Sortino, Calmar, drawdown, volatility and the trade metrics) use fixed conventions: the sample deviation for volatility and the Sharpe denominator, the population deviation for Sortino's downside, and square-root annualization. A degenerate or busted run returns finite numbers rather than `NaN`, so one bad run cannot poison a sweep's argmax. `avg_trade_pct` is measured against starting equity.

**0041 Statistics stay finite, monotone and net.** A non-positive or non-finite `periods_per_year` falls back to no annualization rather than producing a `NaN`. Drawdown is capped at 100%. The trade metrics are net of fees, because on gross PnL a strategy whose edge is smaller than its costs reports a perfect win rate beside a negative return. ADR 0043 extends it.

**0042 A zero denominator ranks at infinity, not at zero.** Sharpe, Sortino and Calmar, like the profit factor, return infinity for a positive reward earned against zero measured risk, and zero otherwise. Returning zero ranked a run with no drawdown below the same run with a small loss. ADR 0043 extends it.

**0043 A ratio that ranks losers must not reward the loss.** For a losing run, Calmar scales the return by the drawdown rather than dividing by it, so a deeper loss ranks lower and the two branches meet at a total loss. CAGR saturates rather than overflowing on very short spans, where `total_return_pct` is the number to rank on. Sharpe's usual treatment of a loser is kept.

**0044 The annualization is read from the candles.** `Backtester` and `tune` derive `periods_per_year` from the median gap between timestamps, snapping to a standard interval when within one percent of it and reporting an irregular spacing rather than rounding it. A numpy array carries no timestamps, so it warns and falls back. The result records the annualization it used.

**0045 Confidence is counted in bets, not in bars.** `probabilistic_sharpe` and `min_track_record_length` count the effective sample, `n (1 - r) / (1 + r)` at the first autocorrelation, rather than bars; `independent=True` restores the published formula. Negative autocorrelation is floored at zero rather than credited.

**0046 A tail is taken by rank, not by comparison.** Expected shortfall is the mean of the worst `ceil((1 - alpha) * n)` returns by rank, so a mass of zero returns at the cutoff cannot dilute it; with no ties it equals the textbook definition. The count is nudged below the ceiling by far less than one return, to absorb float error in `1 - alpha`.

**0047 A metric that takes a frame checks the frame.** `excursions` and `session_buckets` read the run's bar indices into the frame, so they require the identical series and check the result's candle fingerprint. `buy_and_hold` needs only the same number of bars, since a benchmark is usually another asset. `session_buckets(by="weekday")` numbers days from Monday.

**0048 The venue is one object, and it hands out the surfaces.** `Market` holds the venue's settings once and hands out `backtest`, `tune`, `walk_forward`, `env` and `engine`, each taking only the arguments that are not the venue. Passing a `Market` in a keyword slot beside the knobs could not tell a knob left at its default from one passed at its default. The keyword form stays for one-off calls.

**0049 A warm-up is declared, not guarded by hand.** `Strategy.warmup` is read once after `init`, and `next` is not called before it, because `closes[i - n]` on an early bar is a negative index that reads from the end of the array. Bars before it still advance and fill resting orders. A price view that refuses negative indices was rejected because it would stop behaving like a numpy array; `closes`, `opens`, `highs`, `lows` and `volumes` ship as read-only views.

**0050 What 1.0 promises, and what it does not.** From 1.0, a minor release does not remove a public name, function or keyword, change what a keyword means, or change a return type. The simulator's behaviour is part of the API: an order decided on bar `i` fills on bar `i + 1`; a `T`-bar series calls `next` `T - 1` times; a bar reaching both a stop and a target books the worse exit (ADR 0019); a liquidation is priced where the margin runs out (ADR 0033); the statistics are never `NaN` and rank monotonically (ADRs 0040, 0041 and 0042); `emsl.ta` returns one value per bar (ADR 0059); a `T-1` array on a chart maps entry `i` to bar `i + 1` (ADR 0068); and `tune` holds out the end of a series (ADR 0053). Changing any of these is a major release, with an ADR. Not promised: an indicator's output to the last bit (ADR 0061), anything private, the wording of errors, the chart's JSON spec, and the Rust crates. The fidelity limits (ADR 0018) may become more faithful in a minor release.

**0051 Tuning over the Strategy spine.** `emsl.tune` searches a strategy's parameters by running the same backtest in worker processes. It never serializes a live engine: each worker rebuilds one from the candle array, the strategy and objective travel with cloudpickle, and optuna drives the search.

**0052 A search needs a floor on activity.** `min_trades` fails a trial that closed fewer trades, as a `NaN` objective is failed, because an unconstrained search drifts to the configuration with the fewest trades and the widest interval. It defaults to zero.

**0053 A tuned result is in-sample until it has been held out.** `oos=0.3` fits every trial on the first 70% of the series and re-runs the winner on the last 30%, which no trial saw. The held-out part is always the end of the series, never a random slice, and the winner warms up inside it, which errs toward understating the result. Leaving `oos` out warns; `oos=0` searches everything without a warning.

**0054 The deflated Sharpe needs a null it cannot supply itself.** The deflated Sharpe takes a separate random-search null over the same space and bars, because a TPE search concentrates its trials and its own spread shrinks as it overfits. A null that is not a random search, ran on different bars, selected on another statistic, or is too small to have a spread is refused. Failed trials count as looks.

**0055 The deflation is about the search you ran.** The number of looks comes from the study, failed trials included, and only the spread comes from the null; a null smaller than the study still works and warns. A null that used an activity floor is refused, because the floor drops the thin configurations where the extreme Sharpes are and narrows the spread. `TuneResult` carries `min_trades` for that check.

**0056 A walk-forward is one run, not a stitching.** A walk-forward is a single engine pass with a composite strategy that hands each bar to the window that owns it, so the account carries across seams, every warm-up sees the bars before its window, and no position is closed at a boundary. Stitching separate backtests would reset the account, restart warm-ups and invent trades at each seam. Combinatorially symmetric cross-validation was rejected because it recombines non-contiguous blocks, which is unsound for a path-dependent engine.

**0057 A walk-forward window is scored on the run that happened.** Each window's out-of-sample score is read off the composite run with `metrics.segment`, not from a fresh backtest of the window, which would reset the account and the warm-ups (ADR 0056). Each window reports `bars_traded`, so a window that could not trade is visible. A callable objective returns `None` per window.

**0058 A cost sweep moves the cost and nothing else.** `cost_curve` and `breakeven_bps` restore an instance's attributes before each run, so state left by one cost level cannot leak into the next, and the object the caller passed is the one restored. A class is rebuilt for each run.

**0059 Indicator conventions are written down.** `emsl.ta` is a small, closed set, and each function states its convention where implementations differ: `ema` seeds from the simple average of the first window, `rsi` and `atr` use Wilder's smoothing, and `stdev` is the population deviation. Every function returns one value per bar with the warm-up as a gap, never a shorter array, and a function with several outputs returns a named object rather than a tuple. ADR 0060 extends it.

**0060 Crossings return booleans, and a shift pads.** `crossover` and `crossunder` return booleans and are `False` during the warm-up, because `bool(nan)` is `True` and a float warm-up would fire a rule on bars where nothing is known. `shift` pads the start rather than wrapping the end.

**0061 The recursive indicators are vectorized between their gaps.** The recursive smoothers run as one vectorized pass per stretch between gaps, in blocks sized from `alpha` so the weights stay within the float range. This is about 33 times faster than a per-bar loop and differs from it by at most about 2e-15 relative, which is accepted; staying bit-exact would mean moving the indicators into Rust.

**0062 Zero-copy observation.** A single env's observation is a read-only numpy view onto the shared candle buffer, made sound by the candle's `repr(C)` layout.

**0063 Compiled sweep, and why it is not in the Python API.** `bar-engine` has an in-core sweep over built-in Rust strategies with the GIL released. It is not exposed to Python because it can only run strategies compiled into the engine; parameter search from Python goes through `tune`, and the sweep stays in the crate as the benchmark of the GIL-free path.

**0064 RL autoreset.** The vector env uses same-step autoreset: a finished env's `step` returns the next episode's first observation, with the final observation and equity in `infos`. The observation is a `(window, F)` window over a feature matrix you supply. On Gymnasium versions that tag the autoreset mode, it declares `SAME_STEP`.

**0065 Stable-Baselines3 adapter.** SB3 takes its own `VecEnv` rather than a Gymnasium vector env, so `emsl.sb3.EmslVecEnv` presents the batch as one; emsl's same-step autoreset already matches SB3's contract.

**0066 Per-env cost randomization.** Each cost knob accepts a `(low, high)` pair sampled once per env, so a batch trains across a spread of cost regimes.

**0067 The batched tier is market-order-only.** The batched RL path takes only market orders, because a per-order Python decoder cannot run inside the GIL-free step.

**0068 A series is aligned by position, and a short array draws late.** Chart arrays carry no index. A `T`-long array maps entry `i` to bar `i`, and a `T-1`-long array maps entry `i` to bar `i+1`, which is how the engine's equity curve and `numpy.diff` line up. Any other length raises and names both lengths, because a silent trim or pad shifts the chart by an amount nobody can see.

**0069 A NaN is a gap, never a dropped row.** A missing value is drawn as a gap and the line stops, where dropping the row would join its neighbours straight across the hole. Infinities are gaps too. A leading run of non-finite values, an indicator's warm-up, is folded into the series' start offset.

**0070 A bare array overlays the candles unless it would flatten them.** An array passed without a panel is drawn on the price panel when the merged axis still leaves the candles at least half the panel, and gets its own panel otherwise. `panel=` overrides it.

**0071 The renderer is vendored into the wheel.** The chart is drawn by TradingView's lightweight-charts, one standalone JavaScript file under Apache 2.0, shipped unmodified in `python/emsl/_static/` with its licence. It is not loaded from a CDN, so a saved chart works offline and keeps working; the cost is about 200 KB in the wheel, and an upgrade is a deliberate file replacement.

**0072 A chart is precomputed and self-contained, never served.** A chart is one HTML document carrying its renderer, styles and data, all computed in Python before anything is drawn. There is no server and no network call, so a chart survives being saved, sent and reopened without a kernel. It is drawn in an iframe with `srcdoc`, because JupyterLab strips scripts from raw HTML output. The cost is size, about 260 KB per cell plus the data.

**0073 The chart computes nothing and knows no indicator names.** Arrays arrive already computed and the chart only draws them, so the line on screen is the array the strategy traded on, not a recomputation that can differ by a warm-up convention. The one derived number is the drawdown of the engine's equity curve, seeded from the run's opening balance and capped at a total loss. ADR 0076 corrects it.

**0074 The spec is the seam, and the bundle is an explicit list.** Python and JavaScript meet at one JSON document: Python decides what is drawn and where, and JavaScript only how it is painted, so anything a reader could disagree with arrives as data and can be asserted from Python. The JavaScript is a fixed, ordered list of plain scripts concatenated at render time, so a stray file in `_static/` cannot join the bundle, and there is no build step.

**0075 Plotting is not declared inside a strategy.** There is no `Strategy.plot` hook. What you look at changes more often than what you trade, `init` runs on every trial of a search, and a strategy's arrays are already plain attributes that `Line(s.fast)` can draw. A `marks()` method returning a list serves the same purpose and is never called by a sweep.

**0076 The chart's JavaScript may compute geometry, not policy.** Anything a reader could disagree with, such as a colour, a threshold, a rounding rule or an alpha, is decided in Python and arrives as data; the JavaScript computes only what the renderer alone knows, such as bar spacing in pixels or the viewer's locale. The equity legend's return on hover, the hovered value over the first equity point, is the one derived number it shows. This is ADR 0074's seam as it is actually kept.
