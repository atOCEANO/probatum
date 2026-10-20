"""probatum.stats agrees with the reference statistics on random series.

The reference is written from the Specification's Inference section with plain loops,
so a misreading of the page made in the module shows here as a disagreement. The
series are drawn to reach every branch the rules have: independent, autocorrelated
either way, heavy-tailed, and flat most of the time with exposure switching on and off,
at counts from the fewest an interval takes to thousands, so the 1994 lag ranges from 0
up and the floor both binds and does not.
"""

import math

import numpy as np
import pytest

import probatum
from probatum import stats

from reference import statistics


def series(seed):
    """A series of returns, its bars a year and its annual risk-free rate, drawn from
    one of five kinds by ``seed``."""
    rng    = np.random.Generator(np.random.PCG64(seed))
    count  = int(rng.choice([4, 5, 6, 9, 17, 40, 120, 365, 1_000, 2_500]))
    kind   = seed % 5
    shocks = rng.normal(0.0005, 0.01, count)

    if kind == 1 or kind == 2:
        phi    = 0.6 if kind == 1 else -0.5
        values = np.empty(count)
        values[0] = shocks[0]
        for t in range(1, count):
            values[t] = 0.0005 + phi * (values[t - 1] - 0.0005) + shocks[t]
    elif kind == 3:
        values = shocks * np.sqrt(3.0 / rng.chisquare(3.0, count))
    elif kind == 4:
        held   = np.repeat(rng.random(count // 5 + 1) < 0.4, 5)[:count]
        values = np.where(held, shocks, 0.0)
        values[0] = shocks[0]
    else:
        values = shocks

    return values, float(rng.choice([12.0, 52.0, 252.0, 365.0, 8_760.0])), float(
        rng.choice([0.0, 0.0, 0.04]))


CASES = range(240)


@pytest.mark.parametrize("seed", CASES)
def test_the_sharpe_its_interval_its_probability_and_its_track_record_agree(seed):
    values, rate, free = series(seed)
    x     = statistics.excess(values.tolist(), rate, free)
    given = dict(periods_per_year=rate, risk_free=free)
    near  = dict(rel=1e-9, abs=1e-12)

    assert stats.sharpe(values, **given) == pytest.approx(statistics.sharpe(x, rate), **near)
    assert stats.sharpe(values, long_run=True, **given) == pytest.approx(
        statistics.long_run_sharpe(x, rate), **near)

    interval = stats.sharpe_interval(values, level=0.9, **given)
    low, high, lag, floored = statistics.sharpe_interval(x, rate, level=0.9)

    assert (interval.lag, interval.floored, interval.count) == (lag, floored, len(x))
    assert (interval.low, interval.high) == (pytest.approx(low, **near),
                                             pytest.approx(high, **near))
    assert interval.estimate == pytest.approx(statistics.sharpe(x, rate), **near)

    for benchmark in (0.0, 1.0):
        probability = stats.probabilistic_sharpe(values, benchmark=benchmark, **given)

        assert probability.value == pytest.approx(
            statistics.probabilistic_sharpe(x, rate, benchmark), **near)

        record = stats.min_track_record_length(values, benchmark=benchmark,
                                               confidence=0.99, **given)
        needed = statistics.min_track_record(x, rate, benchmark, 0.99)

        if needed is None:
            assert (record.observations, record.years) == (None, None)
        else:
            assert record.observations == pytest.approx(needed, **near)
            assert record.years        == pytest.approx(needed / rate, **near)


@pytest.mark.parametrize("seed", CASES[:60])
def test_the_autocorrelation_and_the_effective_sample_agree(seed):
    values, rate, free = series(seed)

    for lag in range(4):
        assert stats.autocorrelation(values, lag) == pytest.approx(
            statistics.autocorrelation(values.tolist(), lag), rel=1e-9, abs=1e-12)

    assert stats.effective_sample(values) == pytest.approx(
        statistics.effective_sample(values.tolist()), rel=1e-9)


def test_the_cases_reach_every_lag_from_0_and_the_floor_both_ways():
    lags, floors = set(), set()
    for seed in CASES:
        values, rate, free = series(seed)
        interval = stats.sharpe_interval(values, periods_per_year=rate, risk_free=free)
        lags.add(interval.lag)
        floors.add(interval.floored)

    assert floors == {True, False}
    assert {0, 1, 2, 3}.issubset(lags) and max(lags) >= 8


def test_a_result_is_read_at_its_own_annualisation_and_risk_free_rate():
    close = 100.0 + np.random.default_rng(3).normal(0.0, 1.0, 600).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(600, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(600, dtype=np.int64) * 3_600_000_000_000,
    )

    weights = np.where(np.arange(600) % 50 < 25, 1.0, 0.0)
    result  = probatum.backtest(weights, data, "frictionless", risk_free=0.04)
    rate    = result.provenance["periods_per_year"]
    x       = statistics.excess(result.returns[1:].tolist(), rate, 0.04)

    assert stats.sharpe(result) == pytest.approx(result.stats["sharpe"], rel=1e-12)
    assert stats.sharpe(result) == pytest.approx(statistics.sharpe(x, rate), rel=1e-12)
    assert stats.sharpe_interval(result) == stats.sharpe_interval(
        result.returns[1:], periods_per_year=rate, risk_free=0.04)
    assert stats.sharpe_interval(result).count == 599
