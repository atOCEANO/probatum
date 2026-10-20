"""probatum.stats agrees with the reference statistics on random series.

The reference is written from the Specification's Inference section with plain loops,
so a misreading of the page made in the module shows here as a disagreement. The
series are drawn to reach every branch the rules have: independent, autocorrelated
either way, heavy-tailed, and flat most of the time with exposure switching on and off,
at counts from the fewest an interval takes to thousands, so the 1994 lag ranges from 0
up and the floor both binds and does not.
"""

import collections
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


@pytest.mark.parametrize("seed", CASES[:80])
def test_the_block_lengths_agree_on_the_returns_their_squared_deviations_and_tail(seed):
    values, rate, free = series(seed)
    k      = (values.size + 19) // 20
    cutoff = np.sort(values)[k - 1]

    for y in (values, (values - values.mean()) ** 2, (values <= cutoff) * 1.0):
        stationary, circular = stats._block_lengths(y)
        theirs = statistics.block_lengths(y.tolist())
        assert stationary == pytest.approx(theirs[0], rel=1e-9)
        assert circular   == theirs[1]

    assert stats._return_block(values) == pytest.approx(
        statistics.return_block(values.tolist()), rel=1e-9)


def test_an_order_statistic_takes_the_rank_its_level_names():
    # (1 - 0.95) / 2 is a hair above 0.025, so 2,000 times it is a hair above 50, which
    # takes the 51st without the nudge
    values = np.arange(2_000.0)

    for level in (0.5, 0.9, 0.95, 0.99):
        for q in ((1.0 - level) / 2.0, (1.0 + level) / 2.0):
            assert stats._ordered(values, q) == statistics.ordered(values.tolist(), q)

    assert stats._ordered(values, (1.0 - 0.95) / 2.0) == 49.0


def statistic(name, r, x, rate):
    """The reference's ``name`` on one resample's returns ``r`` and excess returns
    ``x``."""
    if name in ("total_return_pct", "net_profit_pct"):
        return statistics.total_return(r)
    if name == "cagr_pct":
        return statistics.cagr(r, rate)
    if name == "volatility_pct":
        return statistics.volatility(r, rate)
    if name == "sortino":
        return statistics.sortino(x, rate)
    if name == "max_drawdown_pct":
        return statistics.max_drawdown(r)
    if name == "calmar":
        return statistics.calmar(r, rate)
    if name == "value_at_risk":
        return statistics.value_at_risk(r)
    return statistics.conditional_value_at_risk(r)


@pytest.mark.parametrize("seed", CASES[:80])
def test_each_return_statistic_s_bounds_agree_on_the_module_s_own_resamples(seed):
    values, rate, free = series(seed)
    values = values[:600]
    x      = statistics.excess(values.tolist(), rate, free)
    block  = stats._return_block(values)
    drawn  = np.concatenate(list(stats._stationary(
        values.size, block, 60, stats._drawn(values, "stats returns", None)))).tolist()

    for name, (_, _, _, how) in stats.RETURNS.items():
        whole = statistic(name, values.tolist(), x, rate)
        if math.isinf(whole):
            with pytest.raises(probatum.ProbatumValueError, match="ranked at infinity"):
                stats.bootstrap(values, name, periods_per_year=rate, risk_free=free)
            continue

        if not statistics.supports_bootstrap(values.tolist(), x, name):
            with pytest.raises(probatum.ProbatumValueError, match="needs at least|a tail "
                               "that varies|returns that do"):
                stats.bootstrap(values, name, periods_per_year=rate, risk_free=free)
            continue

        if how == "studentized":
            low, high = statistics.studentized_returns(name, values.tolist(), x, drawn,
                                                       rate, level=0.9)
        else:
            low, high = statistics.percentile(
                [statistic(name, [values[i] for i in row], [x[i] for i in row], rate)
                 for row in drawn], 0.9)

        if how != "range" and low == high:
            with pytest.raises(probatum.ProbatumValueError, match="put both bounds on"):
                stats.bootstrap(values, name, periods_per_year=rate, risk_free=free,
                                level=0.9, resamples=60)
            continue

        interval = stats.bootstrap(values, name, periods_per_year=rate, risk_free=free,
                                   level=0.9, resamples=60)
        near = dict(rel=1e-9, abs=1e-9)

        assert interval.estimate == pytest.approx(whole, **near)
        assert (interval.low, interval.high) == (pytest.approx(low, **near),
                                                 pytest.approx(high, **near))
        assert (interval.block, interval.count) == (block, values.size)


def test_the_cases_reach_a_bootstrap_and_both_its_floors():
    held = collections.Counter()
    for seed in CASES[:80]:
        values, rate, free = series(seed)
        values = values[:600].tolist()
        x      = statistics.excess(values, rate, free)
        held[(len(values) >= statistics.SPAN,
              statistics.supports_bootstrap(values, x, "volatility_pct"))] += 1

    assert set(held) == {(False, False), (True, False), (True, True)}


def test_a_resample_with_no_error_to_measure_moves_without_bound():
    # a lone gain in 500 flat returns: the resamples that miss it have no spread, so
    # the sign rule sets every bound they reach
    values = np.zeros(statistics.SPAN)
    values[100] = 0.01

    block = stats._return_block(values)
    drawn = np.concatenate(list(stats._stationary(
        values.size, block, 60, stats._drawn(values, "stats returns", None)))).tolist()

    for name in ("total_return_pct", "volatility_pct"):
        low, high = statistics.studentized_returns(name, values.tolist(), values.tolist(),
                                                   drawn, 365.0, level=0.9)
        interval = stats.bootstrap(values, name, periods_per_year=365, level=0.9,
                                   resamples=60)
        assert (interval.low, interval.high) == (pytest.approx(low, rel=1e-9),
                                                 pytest.approx(high, rel=1e-9))
        assert math.isinf(interval.high) and math.isfinite(interval.low)


def test_an_interval_whose_resamples_put_both_bounds_on_one_value_is_refused():
    # ten losses beside 490 flat bars: the 50th lowest of every resample is flat
    flat = np.random.default_rng(8).permutation(np.concatenate(
        [np.full(10, -0.01), np.zeros(490), np.random.default_rng(7).uniform(0.0, 0.01,
                                                                          500)]))

    block = stats._return_block(flat)
    drawn = np.concatenate(list(stats._stationary(
        flat.size, block, 60, stats._drawn(flat, "stats returns", None)))).tolist()
    low, high = statistics.percentile([statistic("value_at_risk", [flat[i] for i in row],
                                                 [flat[i] for i in row], 365.0)
                                       for row in drawn], 0.9)

    assert low == high

    with pytest.raises(probatum.ProbatumValueError, match="value_at_risk: its resamples "
                       "put both bounds on 0.00%"):
        stats.bootstrap(flat, "value_at_risk", periods_per_year=365, level=0.9,
                        resamples=60)


def test_a_statistic_with_no_standard_error_is_refused_by_both():
    # gains and losses of one size, as many of each: every return lies as far from the
    # mean, so volatility's influence is zero on every bar
    half  = statistics.SPAN // 2
    signs = np.where(np.random.default_rng(5).permutation(statistics.SPAN) < half, 0.5,
                     -0.5)

    assert not statistics.supports_bootstrap(signs.tolist(), signs.tolist(),
                                             "volatility_pct")

    with pytest.raises(probatum.ProbatumValueError, match="volatility_pct: no standard "
                       "error, as the returns never vary in what it reads"):
        stats.bootstrap(signs, "volatility_pct", periods_per_year=365)


def trips(seed):
    """Closed round trips' pnl of one of three kinds: even, many small losses and a few
    large gains, or many small gains and a few large losses."""
    rng   = np.random.Generator(np.random.PCG64(1_000 + seed))
    count = int(rng.choice([40, 41, 75, 160, 400]))
    kind  = seed % 3
    if kind == 0:
        return rng.normal(20.0, 100.0, count)

    won  = rng.random(count) < (0.4 if kind == 1 else 0.65)
    paid = rng.exponential(250.0 if kind == 1 else 80.0, count)
    lost = rng.exponential(100.0 if kind == 1 else 120.0, count)
    return np.where(won, paid, -lost)


def closing(count):
    """A run that closes ``count`` round trips, a long held for 2 bars of every 4."""
    rows  = 4 * count + 1
    close = 100.0 + np.random.default_rng(count).normal(0.0, 1.0, rows).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(rows, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * 3_600_000_000_000,
    )

    return probatum.backtest(np.where(np.arange(rows) % 4 < 2, 1.0, 0.0), data,
                             "frictionless")


@pytest.mark.parametrize("side", (-1, 0))
def test_the_floors_refuse_what_the_reference_refuses_on_either_side_of_them(side):
    run    = closing(statistics.FLOOR + side)
    pnl    = np.asarray(run.trades["pnl"])[~np.asarray(run.trades["open"])].tolist()
    values = np.random.default_rng(2).normal(0.0005, 0.01, statistics.SPAN + side)
    cases  = [(run, name, {}, statistics.supports_trips(pnl, name), len(pnl))
              for name in stats.TRADES]
    cases.append((values, "cagr_pct", dict(periods_per_year=365),
                  statistics.supports_bootstrap(values.tolist(), values.tolist(),
                                                "cagr_pct"), values.size))

    for given, name, read, held, count in cases:
        if held:
            assert stats.bootstrap(given, name, resamples=20, **read).count == count
            continue
        with pytest.raises(probatum.ProbatumValueError, match="needs at least"):
            stats.bootstrap(given, name, resamples=20, **read)


@pytest.mark.parametrize("seed", range(15))
def test_each_trade_statistic_s_bounds_agree_on_the_module_s_own_resamples(seed):
    pnl     = trips(seed)
    listed  = pnl.tolist()
    drawn   = stats._drawn(pnl, "stats trades", None).integers(0, pnl.size, (300, pnl.size))
    samples = [[listed[i] for i in row] for row in drawn]
    near    = dict(rel=1e-9, abs=1e-12)
    theirs  = {
        "win_rate":      (statistics.win_rate(listed),
                          *statistics.percentile([statistics.win_rate(s) for s in samples], 0.95)),
        "profit_factor": (statistics.profit_factor(listed),
                          *statistics.profit_factor_interval(listed, samples)),
        "avg_trade_pct": (statistics.average_trade(listed, 10_000.0),
                          *statistics.average_trade_interval(listed, samples, 10_000.0)),
    }

    for name, expected in theirs.items():
        rng   = stats._drawn(pnl, "stats trades", None)
        found = stats._trade_bounds(pnl, 10_000.0, name, 0.95, 300, rng)
        assert found == tuple(pytest.approx(value, **near) for value in expected)


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
