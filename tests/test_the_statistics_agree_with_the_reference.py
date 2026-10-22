"""probatum.stats agrees with the reference statistics on random series.

The reference is written from the Specification's Inference section with plain loops,
so a misreading of the page made in the module shows here as a disagreement. The
series are drawn to reach every branch the rules have: independent, autocorrelated
either way, heavy-tailed, and flat most of the time with exposure switching on and off,
at counts from the fewest an interval takes to thousands, so the 1994 lag ranges from 0
up and the floor both binds and does not.
"""

import collections
import itertools
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


def pair(seed):
    """Two series on the same bars, the second half the first and its own noise, with
    their bars a year and their annual risk-free rate."""
    first, rate, free = series(seed)
    noise = np.random.Generator(np.random.PCG64(5_000 + seed)).normal(0.0003, 0.01,
                                                                      first.size)
    return first, 0.5 * first + noise, rate, free


@pytest.mark.parametrize("seed", CASES[:40])
def test_the_delta_method_s_difference_of_two_sharpes_agrees(seed):
    a, b, rate, free = pair(seed)
    xa, xb = (statistics.excess(v.tolist(), rate, free) for v in (a, b))
    d, error, p, (low, high), lag, floored = statistics.difference(xa, xb, rate, level=0.9)
    found = stats.sharpe_difference(a, b, periods_per_year=rate, risk_free=free, level=0.9,
                                    resamples=20).delta
    near = dict(rel=1e-9, abs=1e-12)

    assert (found.lag, found.floored, found.count) == (lag, floored, a.size)
    assert (found.estimate, found.low, found.high, found.p) == (
        pytest.approx(d, **near), pytest.approx(low, **near), pytest.approx(high, **near),
        pytest.approx(p, **near))


def test_two_series_with_one_influence_have_no_difference_to_test():
    # three times a series is that series to rounding alone
    x       = statistics.excess([0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01], 12.0, 0.0)
    tripled = [3.0 * v for v in x]

    assert statistics.difference(x, tripled, 12.0) is None
    assert statistics.studentized_difference(x, tripled, 12.0, [[0]] * 4) is None

    with pytest.raises(probatum.ProbatumValueError, match="b: returns whose Sharpe moves "
                       "as a's does"):
        stats.sharpe_difference(x, tripled, periods_per_year=12)

    two = [0.3125, 0.078125, 0.078125, 0.078125, 0.078125]

    assert statistics.sharpe_interval(two, 12.0) is None
    assert statistics.probabilistic_sharpe(two, 12.0) is None


@pytest.mark.parametrize("seed", CASES[:12])
def test_the_studentized_bootstrap_agrees_on_the_module_s_own_blocks(seed):
    a, b, rate, free = pair(seed)
    a, b = a[:300], b[:300]
    xa, xb = (statistics.excess(v.tolist(), rate, free) for v in (a, b))
    z      = stats._influence(np.array(xa)) - stats._influence(np.array(xb))
    length = stats._block_lengths(z)[1]
    rng    = stats._drawn(np.concatenate([a, b]), "stats sharpe difference", None)
    starts = np.concatenate(list(stats._block_starts(a.size, length, 200, rng)))
    p, (low, high), theirs = statistics.studentized_difference(
        xa, xb, rate, starts.tolist(), level=0.9)
    found = stats.sharpe_difference(a, b, periods_per_year=rate, risk_free=free, level=0.9,
                                    resamples=200).bootstrap
    near = dict(rel=1e-9, abs=1e-12)

    assert (found.block, found.p) == (theirs, pytest.approx(p, rel=1e-12))
    assert (found.low, found.high) == (pytest.approx(low, **near),
                                       pytest.approx(high, **near))


@pytest.mark.parametrize("seed", CASES[:30])
def test_alpha_and_beta_agree_at_the_1994_lag_and_at_a_given_one(seed):
    x, y, rate, free = pair(seed)

    for given in (None, 3):
        found  = stats._regression(x, y, rate, given, 0.95)
        theirs = statistics.alpha_beta(x.tolist(), y.tolist(), rate, given=given)
        near   = dict(rel=1e-9, abs=1e-12)
        assert found.lag == theirs["lag"]
        for name in ("alpha", "alpha_error", "beta", "beta_error", "tracking_error",
                     "information_ratio"):
            assert getattr(found, name) == pytest.approx(theirs[name], **near)


@pytest.mark.parametrize("seed", range(10))
def test_each_order_of_trades_agrees_on_the_module_s_own_permutations(seed):
    pnl       = trips(seed)
    opened_on = 10_000.0 + np.cumsum(pnl) - pnl
    returned  = pnl / opened_on
    rng       = stats._drawn(returned, "stats trade order", None)
    orders    = rng.permuted(np.broadcast_to(np.arange(pnl.size), (200, pnl.size)), axis=1)

    falls, runs = stats._paths(returned, pnl, orders)

    for order, fall, run in zip(orders, falls, runs):
        theirs = statistics.trade_path(pnl.tolist(), opened_on.tolist(), order.tolist())
        assert (fall, run) == (pytest.approx(theirs[0], rel=1e-9, abs=1e-12), theirs[1])


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


def test_the_order_of_trades_reads_each_trip_on_the_equity_before_it_opened():
    # a fee moves the equity of the row a trip opens on, so only the row before it is
    # the equity the rule names
    rows  = 4 * statistics.FLOOR + 1
    close = 100.0 + np.random.default_rng(3).normal(0.0, 1.0, rows).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(rows, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * 3_600_000_000_000,
    )

    run = probatum.backtest(np.where(np.arange(rows) % 4 < 2, 1.0, 0.0), data,
                            probatum.venues.binance_perp.replace(funding="none"))

    closed    = ~np.asarray(run.trades["open"])
    order     = np.argsort(np.asarray(run.trades["close_bar"])[closed], kind="stable")
    pnl       = np.asarray(run.trades["pnl"])[closed][order].tolist()
    opened    = np.asarray(run.trades["open_bar"])[closed][order].tolist()
    opened_on = [float(run.equity[max(bar - 1, 0)]) for bar in opened]
    fall, longest = statistics.trade_path(pnl, opened_on, list(range(len(pnl))))
    found = stats.trade_order(run, n=20)

    assert (found.drawdown, found.losing_run) == (pytest.approx(fall, rel=1e-12), longest)


def test_an_order_deeper_than_the_run_s_only_by_rounding_is_level():
    # the Specification's example: trips of 20%, -10% and -10% in each of their orders,
    # where the two that put the losses first fall a few ulps further than the run's
    returned = np.array([0.2, -0.1, -0.1])
    orders   = np.array(list(itertools.permutations(range(3))))

    falls, _ = stats._paths(returned, returned, orders)

    assert np.count_nonzero(falls > falls[0]) == 2
    assert stats._deeper(falls, falls[0])     == statistics.deeper(falls.tolist(), falls[0])
    assert stats._deeper(falls, falls[0])     == 0.0


class Recording:
    """A generator that keeps every draw it hands out, in order, and how it was asked."""


    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)
        self.drawn     = []
        self.asked     = []


    def integers(self, *given, **named):
        self.asked.append(("integers", given, named))
        self.drawn.append(self.generator.integers(*given, **named))
        return self.drawn[-1]


    def random(self, *given, **named):
        self.asked.append(("random", given, named))
        self.drawn.append(self.generator.random(*given, **named))
        return self.drawn[-1]


@pytest.mark.parametrize("points", [np.random.default_rng(5).normal(size=(40, 6)),
                                    np.zeros((6, 2))])
def test_a_start_is_the_greedy_sequence_its_draws_give(points):
    # the module's draws handed to the reference, which reads them by the page's rule;
    # trials all in one place leave every distance 0, so each is drawn evenly
    squared = ((points[:, None] - points[None]) ** 2).sum(axis=2)
    stream  = Recording(1)
    found   = stats._sequence(squared, 5, stream)
    draws   = iter(stream.drawn)

    assert found == statistics.sequence(squared.tolist(), 5, draws)
    assert next(draws, None) is None

    # the first among every trial, and each step's 2 + floor(ln 5) = 3 candidates
    count = len(points)

    assert stream.asked[0] == ("integers", (count,), {})
    assert all(asked in (("random", (3,), {}), ("integers", (count,), {"size": 3}))
               for asked in stream.asked[1:])


@pytest.mark.parametrize("seed", range(6))
def test_the_effective_number_agrees_on_the_module_s_own_starts(seed, monkeypatch):
    # blocks of trials that share a factor, of sizes and noise that vary by seed, the
    # reference handed each start's sequence as the module drew it
    rng    = np.random.default_rng(seed)
    sizes  = rng.integers(1, 6, int(rng.integers(2, 6)))
    noise  = float(rng.choice([0.3, 0.8, 1.5]))
    factor = rng.normal(size=(sizes.size, 120))
    rows   = np.array([factor[block] + rng.normal(scale=noise, size=120)
                       for block, size in enumerate(sizes) for _ in range(size)]) * 0.01
    drawn  = []
    module = stats._sequence

    def recording(squared, most, generator):
        drawn.append(module(squared, most, generator))
        return drawn[-1]

    monkeypatch.setattr(stats, "_sequence", recording)

    found = stats.effective_trials(rows, periods_per_year=252, risk_free=0.02)
    queue = list(drawn)

    def draw(far, most):
        return [queue.pop(0) for _ in range(stats.STARTS)]

    x = statistics.excess(rows.ravel().tolist(), 252, 0.02)
    x = [x[i * rows.shape[1]:(i + 1) * rows.shape[1]] for i in range(rows.shape[0])]
    count, spread = statistics.effective_trials(x, 252, draw)

    assert (found.count, queue) == (count, [])
    assert found.spread         == (None if spread is None else pytest.approx(spread, rel=1e-9))

    if found.quality is not None:
        far = statistics.distances(x)[1]
        assert found.quality == pytest.approx(
            statistics.quality(statistics.silhouettes(far, list(found.clusters))),
            rel=1e-9)


@pytest.mark.parametrize("seed, kept", [(191, False), (34, True)])
def test_the_refinement_agrees_where_it_refuses_and_where_it_keeps_new_clusters(
        seed, kept, monkeypatch):
    # six or more blocks of mixed noise, so more than two clusters fall below the mean
    # quality and are clustered again: new clusters refused for seed 191 and kept
    # for seed 34
    rng    = np.random.default_rng(seed)
    sizes  = rng.integers(2, 7, int(rng.integers(6, 11)))
    noise  = rng.choice([0.2, 0.6, 1.2, 2.0], sizes.size)
    factor = rng.normal(size=(sizes.size, 120))
    rows   = np.array([factor[block] + rng.normal(scale=noise[block], size=120)
                       for block, size in enumerate(sizes) for _ in range(size)]) * 0.01
    drawn, bases = [], []
    module, base = stats._sequence, stats._base

    def recording(squared, most, generator):
        drawn.append(module(squared, most, generator))
        return drawn[-1]

    def based(points, far, generator):
        bases.append(base(points, far, generator))
        return bases[-1]

    monkeypatch.setattr(stats, "_sequence", recording)
    monkeypatch.setattr(stats, "_base", based)

    labels, _ = stats._onc(rows, np.random.default_rng(seed))
    queue = list(drawn)

    def draw(far, most):
        return [queue.pop(0) for _ in range(stats.STARTS)]

    assert labels.tolist() == statistics.onc(rows.tolist(), draw)
    assert (len(bases) > 1, not np.array_equal(labels, bases[0])) == (True, kept)


@pytest.mark.parametrize("trials", [2, 3, 10, 46, 100, 1_000])
def test_the_deflation_threshold_and_the_backtest_length_agree(trials):
    assert stats.deflation_threshold(0.7, trials) == pytest.approx(
        statistics.deflation_threshold(0.7, trials), rel=1e-12)
    assert stats.min_backtest_length(trials, 1.3).years == pytest.approx(
        statistics.min_backtest_length(trials, 1.3), rel=1e-12)
