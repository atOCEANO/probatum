"""A statistic reads a Result, at the annualisation and the risk-free rate it ran at, or
any series of returns by position, at the bars a year it is told, and refuses what it
cannot read or cannot support by name.

What it computes is held to the reference by the agreement test, and the numbers of
the Specification's worked example by the page's own test; these hold what it reads,
what it refuses and the shape of what it gives back.
"""

import math
import pickle

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import (ProbatumAttributeError, ProbatumTypeError, ProbatumValueError,
                      stats)

MONTHLY = [0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01]


@pytest.fixture(scope="module")
def result():
    close = 100.0 + np.random.default_rng(5).normal(0.0, 1.0, 400).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(400, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(400, dtype=np.int64) * 3_600_000_000_000,
    )

    weights = np.where(np.arange(400) % 40 < 20, 1.0, 0.0)
    return probatum.backtest(weights, data, "frictionless")


def test_a_plain_series_without_its_bars_a_year_is_refused_in_the_plan_s_words():
    for call in (stats.sharpe, stats.sharpe_interval, stats.probabilistic_sharpe,
                 stats.min_track_record_length):
        with pytest.raises(ProbatumTypeError, match="periods_per_year: missing for a plain "
                           "return series; pass 365 for daily crypto returns"):
            call(MONTHLY)


def test_a_result_refuses_an_annualisation_or_a_rate_it_names_itself(result):
    with pytest.raises(ProbatumTypeError, match="periods_per_year: given beside a "
                       "Result, which runs at 8760.0; leave it out"):
        stats.sharpe(result, periods_per_year=365)
    with pytest.raises(ProbatumTypeError, match="risk_free: given beside a Result, which "
                       "ran at 0.0; leave it out"):
        stats.sharpe_interval(result, risk_free=0.04)


def test_a_result_s_sharpe_is_its_record_s(result):
    assert stats.sharpe(result) == pytest.approx(result.stats["sharpe"], rel=1e-12)


def test_a_pandas_series_is_read_by_position_and_never_by_its_index():
    shuffled = pd.Series(MONTHLY, index=[7, 3, 5, 1, 0, 2, 6, 4])

    assert stats.sharpe_interval(shuffled, periods_per_year=12) == stats.sharpe_interval(
        np.array(MONTHLY), periods_per_year=12)


def test_a_nan_or_an_infinity_is_refused_at_its_first_position():
    with pytest.raises(ProbatumValueError, match="returns: position 2 holds nan; a "
                       "return series holds finite numbers"):
        stats.sharpe([0.01, 0.02, math.nan, math.inf], periods_per_year=12)


@pytest.mark.parametrize("given, said", [
    ({"key": 0.01}, "returns: dict; pass a Result, or a 1-D array"),
    ("0.01", "returns: str; pass a Result"),
    (np.zeros((3, 2)), r"returns: an array of shape \(3, 2\); pass one series, 1-D")])
def test_what_is_no_series_of_returns_is_refused_by_what_it_is(given, said):
    with pytest.raises((ProbatumTypeError, ProbatumValueError), match=said):
        stats.sharpe(given, periods_per_year=12)


@pytest.mark.parametrize("name", ["periods_per_year", "risk_free", "level", "benchmark",
                                  "confidence"])
def test_a_setting_that_is_no_finite_number_is_refused_by_name(name):
    call = (stats.min_track_record_length if name in ("benchmark", "confidence")
            else stats.sharpe_interval)
    given = {"periods_per_year": 12, name: math.nan}

    with pytest.raises(ProbatumValueError, match=f"{name}: nan; give"):
        call(MONTHLY, **given)

    given[name] = "0.5"

    with pytest.raises(ProbatumTypeError, match=f"{name}: '0.5'; give"):
        call(MONTHLY, **given)


def test_a_level_or_a_confidence_outside_0_and_1_is_refused():
    with pytest.raises(ProbatumValueError, match="level: 95.0; give a fraction between 0 "
                       "and 1, such as 0.95"):
        stats.sharpe_interval(MONTHLY, periods_per_year=12, level=95)
    with pytest.raises(ProbatumValueError, match="confidence: 1.0; give a fraction"):
        stats.min_track_record_length(MONTHLY, periods_per_year=12, confidence=1.0)
    with pytest.raises(ProbatumValueError, match="periods_per_year: 0; give the bars a "
                       "year, such as 365"):
        stats.sharpe(MONTHLY, periods_per_year=0)


def test_three_returns_or_returns_that_never_vary_support_no_interval_but_a_sharpe():
    with pytest.raises(ProbatumValueError, match="returns: 3 of them; an interval needs "
                       "at least 4"):
        stats.sharpe_interval(MONTHLY[:3], periods_per_year=12)
    for call in (stats.sharpe_interval, stats.probabilistic_sharpe,
                 stats.min_track_record_length):
        with pytest.raises(ProbatumValueError, match="returns: all 5 are the same; an "
                           "interval needs returns that vary"):
            call([0.0] * 5, periods_per_year=12)
        # two values in the one proportion whose Sharpe's influence is zero on each bar,
        # which rounding leaves a hair above zero, so only the threshold refuses them
        with pytest.raises(ProbatumValueError, match="returns: two values whose Sharpe's "
                           "influence is zero on every bar, so it has no standard error; "
                           "an interval needs returns that take a third"):
            call([0.3, 0.075, 0.075, 0.075, 0.075], periods_per_year=12)

    assert stats.sharpe(MONTHLY[:3], periods_per_year=12) == pytest.approx(
        np.mean(MONTHLY[:3]) / np.std(MONTHLY[:3], ddof=1) * math.sqrt(12.0))
    assert stats.sharpe([0.0] * 5, periods_per_year=12) == 0.0
    assert stats.sharpe([0.01], periods_per_year=12)    == 0.0


def test_a_track_record_has_no_years_where_the_sharpe_is_not_above_its_benchmark():
    record = stats.min_track_record_length(MONTHLY, benchmark=3.0, periods_per_year=12)

    assert (record.observations, record.years) == (None, None)
    assert str(record)         == "minimum track record none: the sharpe 1.73 is not above 3.00"
    assert record.series_years == pytest.approx(8 / 12)


def test_each_result_says_its_numbers_never_changes_and_pickles_as_it_was():
    interval    = stats.sharpe_interval(MONTHLY, periods_per_year=12)
    probability = stats.probabilistic_sharpe(MONTHLY, benchmark=1.0, periods_per_year=12)
    record      = stats.min_track_record_length(MONTHLY, periods_per_year=12)

    assert (interval.statistic, interval.method, interval.count, interval.lag,
            interval.floored, interval.level) == ("sharpe", "delta method", 8, 2, True, 0.95)
    assert str(probability) == "probabilistic sharpe 0.705 against 1.00, one-sided"
    assert repr(record).startswith("TrackRecord(minimum track record 1.0 years for 95% ")

    for made in (interval, probability, record):
        assert pickle.loads(pickle.dumps(made))       == made
        assert hash(pickle.loads(pickle.dumps(made))) == hash(made)
        with pytest.raises(ProbatumAttributeError, match="a result does not change; "
                           "compute another"):
            made.count = 9


@pytest.fixture(scope="module")
def traded():
    """A run of 251 round trips, a long held on the bars a coin flip picks, so its
    exposure leaves its returns no memory for the bootstrap's blocks to span."""
    close = 100.0 + np.random.default_rng(9).normal(0.0, 1.0, 1_000).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(1_000, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(1_000, dtype=np.int64) * 3_600_000_000_000,
    )

    weights = np.where(np.random.default_rng(9).random(1_000) < 0.5, 1.0, 0.0)
    return probatum.backtest(weights, data, "frictionless")


def test_a_bootstrap_repeats_on_every_read_for_a_result_and_its_series(traded):
    rate = traded.provenance["periods_per_year"]
    once = stats.bootstrap(traded, "cagr_pct")

    assert once == stats.bootstrap(traded, "cagr_pct")
    assert once == stats.bootstrap(traded.returns[1:], "cagr_pct", periods_per_year=rate)
    assert once != stats.bootstrap(traded, "cagr_pct", seed=3)
    assert once.estimate == pytest.approx(traded.stats["cagr_pct"], rel=1e-9)
    assert (once.method, once.kind, once.count) == (
        "studentized stationary block bootstrap", "interval", 999)
    assert once.block >= 1.0 and once.lag is None and once.floored is None


def test_a_drawdown_is_a_range_of_histories_and_a_loss_prints_as_a_negative_return(traded):
    fall = stats.bootstrap(traded, "max_drawdown_pct")

    assert fall.kind == "range" and 0.0 < fall.low <= fall.estimate <= fall.high
    assert str(fall).startswith(f"max drawdown -{fall.estimate:.2f}%; middle 95% of "
                                f"resampled histories -{fall.high:.2f}% to "
                                f"-{fall.low:.2f}%, stationary block bootstrap, block ")

    worst = stats.bootstrap(traded, "value_at_risk")
    assert worst.kind == "interval" and str(worst).startswith(
        f"value at risk -{worst.estimate:.2f}%, two-sided 95% interval")
    assert worst.method == "percentile stationary block bootstrap"


def test_a_result_s_trade_statistics_take_a_bootstrap_of_its_round_trips(traded):
    assert traded.stats["num_trades"] == 251

    for name, method in (("win_rate", "percentile bootstrap of round trips"),
                         ("profit_factor", "bias-corrected and accelerated bootstrap of "
                                           "round trips"),
                         ("avg_trade_pct", "studentized bootstrap of round trips")):
        interval = stats.bootstrap(traded, name)
        assert interval.estimate == pytest.approx(traded.stats[name], rel=1e-12)
        assert interval.low <= interval.estimate <= interval.high
        assert (interval.method, interval.count, interval.block) == (method, 251, None)
        assert str(interval).endswith(f"{method}, 251 round trips")


def test_a_bootstrap_refuses_what_its_sample_cannot_support_by_name(result, traded):
    with pytest.raises(ProbatumValueError, match="statistic: 'max_drawdown'; give one of "
                       "total_return_pct, net_profit_pct, cagr_pct"):
        stats.bootstrap(result, "max_drawdown")
    with pytest.raises(ProbatumValueError, match="statistic: win_rate reads round trips, "
                       "which a plain series does not hold; pass the Result"):
        stats.bootstrap(MONTHLY, "win_rate", periods_per_year=12)
    with pytest.raises(ProbatumValueError, match=f"round trips: "
                       f"{result.stats['num_trades']} closed; an interval of win_rate "
                       f"needs at least 60"):
        stats.bootstrap(result, "win_rate")
    with pytest.raises(ProbatumValueError, match="sortino: inf, a reward over no measured "
                       "risk; an estimate ranked at infinity has no interval"):
        stats.bootstrap([0.01, 0.02, 0.0, 0.03], "sortino", periods_per_year=12)
    with pytest.raises(ProbatumValueError, match="returns: 399 of them; a bootstrap of "
                       "returns needs at least 500"):
        stats.bootstrap(result, "cagr_pct")

    rng       = np.random.default_rng(6)
    stretched = np.where(np.repeat(rng.random(20) < 0.5, 30), rng.normal(0.0, 0.01, 600),
                         0.0)
    with pytest.raises(ProbatumValueError, match=r"returns: 600 in blocks of 29\.7, 20\.2 "
                       r"of them; a bootstrap of returns needs at least 100 blocks"):
        stats.bootstrap(stretched, "volatility_pct", periods_per_year=365)

    gone = np.random.default_rng(4).normal(0.0, 0.01, 1_000)
    gone[300] = -1.0
    with pytest.raises(ProbatumValueError, match="total_return_pct: a return of -100.00%, "
                       "which leaves no equity to grow; growth has no interval past a "
                       "loss of all of it"):
        stats.bootstrap(gone, "total_return_pct", periods_per_year=365)
    assert stats.bootstrap(gone, "volatility_pct", periods_per_year=365).high > 0.0

    tied = np.random.default_rng(2).normal(0.001, 0.01, 1_000)
    tied[np.argsort(tied)[:60]] = -0.02
    with pytest.raises(ProbatumValueError, match="conditional_value_at_risk: the 50 "
                       "lowest returns are all -2.00%; an interval needs a tail that "
                       "varies"):
        stats.bootstrap(tied, "conditional_value_at_risk", periods_per_year=365)
    assert stats.bootstrap(tied, "value_at_risk", periods_per_year=365).low < 2.0

    # ten losses beside 490 flat bars: every resample's 50th lowest return is flat
    flat = np.random.default_rng(8).permutation(np.concatenate(
        [np.full(10, -0.01), np.zeros(490), np.random.default_rng(7).uniform(0.0, 0.01,
                                                                          500)]))
    with pytest.raises(ProbatumValueError, match="value_at_risk: its resamples put both "
                       "bounds on 0.00%; an interval needs resamples that differ "
                       "between them"):
        stats.bootstrap(flat, "value_at_risk", periods_per_year=365)

    deep = tied.copy()
    deep[np.argsort(deep)[:300]] = -0.02
    with pytest.raises(ProbatumValueError, match="value_at_risk: its resamples put both "
                       "bounds on -2.00%"):
        stats.bootstrap(deep, "value_at_risk", periods_per_year=365)
    with pytest.raises(ProbatumValueError, match="win_rate: its resamples put both bounds "
                       "on 0.47; an interval needs resamples that differ between them"):
        stats.bootstrap(traded, "win_rate", level=0.01)
    with pytest.raises(ProbatumValueError, match="profit_factor: its resamples put both "
                       "bounds on 1.07"):
        stats.bootstrap(traded, "profit_factor", level=0.2, resamples=4, seed=0)
    with pytest.raises(ProbatumValueError, match="resamples: 3 at a level of 0.2, which "
                       "puts both bounds on one resample; give more"):
        stats.bootstrap(tied, "volatility_pct", periods_per_year=365, level=0.2,
                        resamples=3)
    with pytest.raises(ProbatumValueError, match="resamples: 1; give a whole number "
                       "from 2"):
        stats.bootstrap(traded, "win_rate", resamples=1)
    with pytest.raises(ProbatumValueError, match="resamples: 2, every one of them on one "
                       "side of the profit factor's estimate; give more"):
        stats.bootstrap(traded, "profit_factor", resamples=2, seed=0)
    with pytest.raises(ProbatumTypeError, match=r"statistic: \['sortino'\]; give one of"):
        stats.bootstrap(traded, ["sortino"])
    with pytest.raises(ProbatumTypeError, match="seed: 1.5; give a whole number from 0"):
        stats.bootstrap(traded, "win_rate", seed=1.5)


def test_the_autocorrelation_is_1_at_lag_0_and_0_where_nothing_varies():
    assert stats.autocorrelation(MONTHLY, 0)     == 1.0
    assert stats.autocorrelation([0.01] * 6, 1)  == 0.0
    assert stats.autocorrelation(MONTHLY[:2], 1) == 0.0
    assert stats.effective_sample(MONTHLY[:3])   == 3.0
    with pytest.raises(ProbatumValueError, match="lag: -1; give a whole number of bars, "
                       "from 0"):
        stats.autocorrelation(MONTHLY, -1)
