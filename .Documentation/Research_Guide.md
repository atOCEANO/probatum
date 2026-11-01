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
  <b>Research Guide</b> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Charts.md">Charts</a> &nbsp;•&nbsp;
  <a href="Python_API.md">Python API</a> &nbsp;•&nbsp;
  <a href="Specification.md">Specification</a> &nbsp;•&nbsp;
  <a href="Venue_Notes.md">Venue Notes</a> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Research Guide

This guide goes from bars to a result that can be trusted. A policy is searched on the bars before a holdout, validated out of sample, judged against every trial the search took, and run again at the size and the costs it would trade. Each step runs the engine a backtest runs, so the costs that choose the winner are the ones its holdout pays. The rules behind each step are in the [Specification](Specification.md), every call is in the [Python API](Python_API.md), and the papers each estimator follows are [listed at the end](#the-estimators-and-their-papers).

The examples read four years of hourly bars of one pair:

```python
import numpy
import probatum
from exchange_router import Router
from probatum import stats

with Router.local(exchanges=["binance"]) as router:
    data = probatum.Data.from_router(router, exchange="binance", market="spot",
                                     interval="1h", symbols=["BTCUSDT"],
                                     start="2022-01-01", end="2025-12-31")

venue = probatum.venues.binance_spot
```

<br>

## A policy

A policy a search can tune is a function of the bars and its parameters that returns weights, or a `Strategy` class built from them. A function's weights run whole in Rust, and a random or grid search runs its trials many to a call.

```python
def crossing(data, fast, slow):
    up = probatum.ta.ema(data.close, length=fast)
    down = probatum.ta.ema(data.close, length=slow)
    weights = numpy.full(shape=len(data), fill_value=numpy.nan)  # NaN holds the position
    weights[probatum.ta.crossover(up, other=down)]  = 1.0
    weights[probatum.ta.crossunder(up, other=down)] = 0.0
    return weights


result = probatum.backtest(crossing(data, fast=20, slow=100), data=data, venue=venue)
```

[Charts](Charts.md#before-there-is-a-strategy) draws a feature beside the price and against what came after it, before any search.

<br>

## Searching

```python
space = {"fast": probatum.Int(low=5, high=40),
         "slow": probatum.Int(low=40, high=200, step=10)}

study = probatum.search(crossing, space=space, data=data, venue=venue, sampler="random",
                        trials=200, holdout=0.3, seed=7)

print(study.best_params, study.best_score)        # in-sample: the best of 200 tries

held = study.evaluate_holdout()                   # the one run on bars no trial saw
print(held.stats["sharpe"], study.looks)
```

`best_score` is the largest of 200 noisy scores on the bars the search could see, so it is biased upward, and by more the harder the search looks. `holdout=0.3` runs every trial on the first 70% of the rows and keeps the last 30% for one run of the winner, which `evaluate_holdout()` counts as a look. Leave `holdout` out and every score is in-sample, which the search warns of ([ADR 0032](Decisions.md)).

Before it trusts a trial, the search runs the policy again on the bars cut short at the points its decisions change, and refuses one whose decisions there move once the later bars are gone, which is a policy reading bars that had not closed ([Specification](Specification.md#the-causality-check)).

`"random"` and `"grid"` are probatum's own samplers, and repeat on any machine from their seed. The default, `"tpe"`, is Optuna's and needs the `search` extra. Trials that do not run together in Rust, a `Strategy`'s and TPE's, run one at a time, or across `n_jobs` worker processes where `n_jobs` is above 1. `storage="studies/cross"` keeps the Study on disk as it runs, and the same call made again resumes it.

<br>

## Validating

A holdout is one look at one stretch. A validation trades several stretches out of sample, each with the parameters a search chose on bars that hold none of it, and joins them into paths:

```python
paths = probatum.validate(crossing, data=data, venue=venue,
                          cv=probatum.cv.WalkForward(train=4_000, test=1_000),
                          space=space, search={"trials": 50, "sampler": "random"},
                          holdout=0.2, seed=7)

print(paths.summary())                            # the paths, their median, the decay
print(paths.windows.to_pandas()[["path", "fit", "first", "last", "score"]])
```

Each path is one run of the engine through its windows. At a seam the orders not yet filled are cancelled and the next window decides from the account it inherits, so a position carried across a seam pays what it would have paid ([ADR 0034](Decisions.md)). `WalkForward` trains each window on the bars before it. `PurgedKFold` and `CombinatorialPurged` train on both sides of a test span, leaving out every training row whose label reaches it and embargoing the bars after it ([ADR 0035](Decisions.md)). `CombinatorialPurged` also recombines its splits into several paths, so a median and a spread replace a single number.

<br>

## Labels and a model

`probatum.labels` gives each row what one trade decided at its close makes through the same fill model, its exit by a barrier or a horizon, the bar that ended it, and its uniqueness among the trades that overlap it ([ADR 0036](Decisions.md)). The splitters purge by that end:

```python
labels   = probatum.labels(data, venue=venue, horizon=24, notional=10_000.0,
                           stop=0.03, take_profit=0.03)
splitter = probatum.cv.PurgedKFold(k=5, end=labels.end)
```

`validate` fits a scikit-learn model on features and those labels, starting every fit from a fresh estimator. It checks the features for lookahead the way a search checks a policy, and trades each path's windows of predictions as one run. The [Python API](Python_API.md#a-model) has it end to end.

<br>

## Judging a run

```python
print(result.summary())                           # beside buying and holding, each interval
print(stats.decompose(result))                    # the price pnl, each cost, the turnover
print(stats.trade_stats(result))                  # payoff, expectancy, longest losing run
print(stats.period_returns(result, by="month"))   # or the hour, day, week, quarter or year
```

`decompose` splits the change in equity into the price pnl and each cost, which make up the whole of it, so a return carried by funding, or lost to fees, shows here before any ratio is read. `trade_stats` holds the payoff beside the wins and the losses, and `period_returns` each period's return, which compound to the run's.

`probatum.stats` gives each statistic its interval, named for its method ([ADR 0031](Decisions.md)). The Sharpe's comes from the delta method over a long-run variance that autocorrelation widens and never narrows, beside the probability that it beats a benchmark and the track record that would show it. The other intervals resample, either the returns in stationary blocks or the round trips. A bootstrap of the returns needs at least 500 returns in 100 blocks, and one of the round trips at least 60 of them; below those an interval is refused rather than given.

```python
print(stats.sharpe_interval(result))
print(stats.probabilistic_sharpe(result, benchmark=1.0))      # P(annual Sharpe > 1)
print(stats.bootstrap(result, statistic="cagr_pct"))
print(stats.sharpe_difference(result))            # against its own buying and holding
print(stats.alpha_beta(result))
```

`probatum.compare` sets runs side by side, and with `reference` tests each one's Sharpe against that run's on the same rows:

```python
slower = probatum.backtest(crossing(data, fast=40, slow=200), data=data, venue=venue)
print(probatum.compare({"20/100": result, "40/200": slower}, reference="20/100"))
```

<br>

## Judging a search

The holdout's run is one look. A search's winner is also judged against everything the search tried, and its deflated Sharpe against a null, a random search of the same policy and space at the same venue on the same bars:

```python
null = probatum.search(crossing, space=space, data=data, venue=venue, sampler="random",
                       trials=200, holdout=0.3, seed=8)
print(study.summary(null=null))                   # deflated, overfit, the holdout's look

print(study.reality_check())                      # does any trial beat cash
print(study.min_backtest_length())                # the years over which luck falls to this Sharpe
print(study.plateau(level=0.5))                   # the winner's neighbours past 0.5
print(study.importance())                         # each parameter's share of the scores
```

- **The deflated Sharpe** asks whether the winner's Sharpe beats what the best of that many trials reaches by luck, their Sharpes spread as the null's are, and again against the effective number of trials, the clusters the trials' returns fall into.
- **The probability of backtest overfitting** halves the in-sample bars every way across 16 blocks and counts how often the winner of one half ranks below the median in the other.
- **The Reality Check and SPA** ask whether any trial's growth beats cash, buying and holding or a series of yours, on one shared bootstrap of the bars.
- **The minimum backtest length** is the span over which the best of the trials with no skill falls to the winner's Sharpe.
- **The plateau** is the share of the winner's neighbours in the space that also scored past a level, so a winner alone on a spike reads as one.
- **Parameter importance** is the share of the scores' variation each parameter explains on its own, beside what chance gives.

A search that adapts, as TPE does, draws its trials from what it has seen, so these read it as descriptive and say so ([ADR 0033](Decisions.md)).

<br>

## Costs and capacity

A result holds at the costs it was run at. These run the same policy again where only its costs or its size differ ([ADR 0045](Decisions.md)):

```python
best = crossing(data, **study.best_params)

print(probatum.cost_curve(best, data=data, venue=venue))     # costs at 0, 0.5, 1, 2 and 4 times
print(probatum.breakeven_bps(best, data=data, venue=venue))  # where it stops making money

print(probatum.capacity(best, data=data, venue=venue,
                        balances=[10_000.0, 1_000_000.0, 10_000_000.0]))
```

`cost_curve` multiplies every cost the venue charges for trading together, each fee, the spread, the slippage and the impact, and leaves funding as it is. `breakeven_bps` finds the round-trip cost at which the return reaches zero, to set beside what the venue charges at scale 1. `capacity` opens the account at each balance, so the impact, which grows with the order, and the liquidity budget, which fills a smaller share of a larger order, show where size starts to cost the policy.

<br>

## The estimators and their papers

| Estimator | In probatum | Paper |
| :--- | :--- | :--- |
| the Sharpe's long-run variance, at an automatic lag | `stats.sharpe_interval`, `stats.sharpe(long_run=True)` | Newey and West (1994), Automatic lag selection in covariance matrix estimation, Review of Economic Studies 61(4) |
| the probabilistic Sharpe and the minimum track record | `stats.probabilistic_sharpe`, `stats.min_track_record_length` | Bailey and Lopez de Prado (2012), The Sharpe ratio efficient frontier, Journal of Risk 15(2) |
| the stationary bootstrap | `stats.bootstrap`, `stats.reality_check` | Politis and Romano (1994), The stationary bootstrap, Journal of the American Statistical Association 89(428) |
| the block length | `stats.bootstrap`, `stats.sharpe_difference` | Politis and White (2004), Automatic block-length selection for the dependent bootstrap, Econometric Reviews 23(1), and Patton, Politis and White (2009), its correction, Econometric Reviews 28(4) |
| studentized intervals | `stats.bootstrap` | Hall (1988), Theoretical comparison of bootstrap confidence intervals, Annals of Statistics 16(3) |
| the bias-corrected and accelerated interval | `stats.bootstrap(result, "profit_factor")` | Efron (1987), Better bootstrap confidence intervals, Journal of the American Statistical Association 82(397) |
| two Sharpes on the same bars | `stats.sharpe_difference` | Ledoit and Wolf (2008), Robust performance hypothesis testing with the Sharpe ratio, Journal of Empirical Finance 15(5); Memmel (2003), Performance hypothesis testing with the Sharpe ratio, Finance Letters 1; Jobson and Korkie (1981), Performance hypothesis testing with the Sharpe and Treynor measures, Journal of Finance 36(4) |
| alpha and beta with autocorrelated errors | `stats.alpha_beta` | Newey and West (1987), A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix, Econometrica 55(3) |
| the deflated Sharpe | `stats.deflated_sharpe`, `study.summary(null=)` | Bailey and Lopez de Prado (2014), The deflated Sharpe ratio, Journal of Portfolio Management 40(5) |
| the effective number of trials | `stats.effective_trials` | Lopez de Prado and Lewis (2019), Detection of false investment strategies using unsupervised learning methods, Quantitative Finance 19(9) |
| the probability of backtest overfitting | `stats.overfitting` | Bailey, Borwein, Lopez de Prado and Zhu (2017), The probability of backtest overfitting, Journal of Computational Finance 20(4) |
| the minimum backtest length | `stats.min_backtest_length` | Bailey, Borwein, Lopez de Prado and Zhu (2014), Pseudo-mathematics and financial charlatanism, Notices of the AMS 61(5) |
| the Reality Check and SPA | `stats.reality_check` | White (2000), A reality check for data snooping, Econometrica 68(5); Sullivan, Timmermann and White (1999), Data-snooping, technical trading rule performance, and the bootstrap, Journal of Finance 54(5); Hansen (2005), A test for superior predictive ability, Journal of Business and Economic Statistics 23(4) |
| labels' uniqueness, the purge and the combinatorial paths | `labels`, `cv.PurgedKFold`, `cv.CombinatorialPurged` | Lopez de Prado (2018), Advances in Financial Machine Learning, Wiley, chapters 4, 7 and 12 |
| the spread estimate | `fills={"spread": "estimated"}` | Ardia, Guidotti and Kroencke (2024), Efficient estimation of bid-ask spreads from open, high, low, and close prices, Journal of Financial Economics 161 |
| the ulcer index | `stats.ulcer_index` | Martin and McCann (1989), The Investor's Guide to Fidelity Funds, Wiley |
| omega | `stats.omega` | Keating and Shadwick (2002), A universal performance measure, Journal of Performance Measurement 6(3) |

Where probatum departs from a paper, the [Specification](Specification.md) says where and how.
