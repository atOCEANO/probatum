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


def test_the_autocorrelation_is_1_at_lag_0_and_0_where_nothing_varies():
    assert stats.autocorrelation(MONTHLY, 0)     == 1.0
    assert stats.autocorrelation([0.01] * 6, 1)  == 0.0
    assert stats.autocorrelation(MONTHLY[:2], 1) == 0.0
    assert stats.effective_sample(MONTHLY[:3])   == 3.0
    with pytest.raises(ProbatumValueError, match="lag: -1; give a whole number of bars, "
                       "from 0"):
        stats.autocorrelation(MONTHLY, -1)
