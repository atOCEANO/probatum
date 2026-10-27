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


@pytest.mark.parametrize("objective", ["sharpe", "sortino", "total_return_pct",
                                       "cagr_pct", "volatility_pct"])
def test_the_overfitting_probability_agrees_on_every_halving(objective):
    # trials over 37 returns, the oldest 5 left out, one a copy of another so ranks tie,
    # one that never falls so its Sortino ranks at infinity, one that lost a whole bar,
    # and one that never trades, whose Sharpe at a positive rate the record ranks at zero
    rng  = np.random.default_rng(len(objective))
    rows = np.vstack([rng.normal(0.001, 0.01, (5, 37)), np.zeros((1, 37))])
    rows[3]     = rows[1]
    rows[4]     = np.abs(rows[4])
    rows[2, 20] = -1.0

    found = stats.overfitting(rows, objective, periods_per_year=252, risk_free=0.03)
    x     = [statistics.excess(row, 252, 0.03) for row in rows.tolist()]
    read  = {
        "sharpe":           (rows.tolist(), lambda r: statistics.recorded(r, 252, 0.03)),
        "sortino":          (x, lambda r: statistics.sortino(r, 252)),
        "total_return_pct": (rows.tolist(), statistics.total_return),
        "cagr_pct":         (rows.tolist(), lambda r: statistics.cagr(r, 252)),
        "volatility_pct":   (rows.tolist(), lambda r: statistics.volatility(r, 252)),
    }
    given, statistic = read[objective]
    probability, loss = statistics.overfitting(given, statistic,
                                               maximize=objective != "volatility_pct")

    assert found.probability == pytest.approx(probability, rel=1e-12)
    assert found.loss == (None if objective == "volatility_pct" else
                          pytest.approx(loss, rel=1e-12))
    assert found.dropped == 5


def test_a_trial_that_never_trades_ranks_at_zero_on_every_halving_at_a_positive_rate():
    # its excess returns are a constant below zero whose mean over a block of three
    # rounds at this rate, so their merged deviation is not 0 where the record reads the
    # returns' own, 0; beside trials that drift down, the rank it takes moves the
    # probability
    rows  = np.vstack([np.zeros(48),
                       np.random.default_rng(0).normal(-0.001, 0.01, (3, 48))])
    found = stats.overfitting(rows, periods_per_year=252, risk_free=0.05)
    probability, loss = statistics.overfitting(
        rows.tolist(), lambda r: statistics.recorded(r, 252, 0.05))

    assert found.probability == pytest.approx(probability, rel=1e-12)
    assert found.loss        == pytest.approx(loss, rel=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_the_reality_check_and_spa_agree_on_the_module_s_own_resamples(seed):
    # five trials over 40 bars against a benchmark, or against cash for seed 4, one
    # autocorrelated so the blocks run long, one that trails the benchmark far enough to
    # keep its mean, and one that never trades, constant against cash; the resamples
    # drawn from the stream the page names, seeded by the trials' returns and the
    # benchmark's
    rng  = np.random.default_rng(seed)
    rows = rng.normal(0.0005, 0.01, (5, 40))
    for t in range(1, 40):
        rows[0, t] += 0.6 * rows[0, t - 1]

    rows[1] -= 0.02
    rows[4]  = 0.0

    given = rng.normal(0.0002, 0.008, 40) if seed < 4 else np.zeros(40)
    found = stats.reality_check(rows, None if seed == 4 else given, periods_per_year=252,
                                resamples=60)
    drawn = [row for chunk in stats._stationary(
        40, found.block, 60, stats._drawn(np.vstack([rows, given]), "stats reality check",
                                          None)) for row in chunk.tolist()]

    f     = statistics.relative(rows.tolist(), given.tolist())
    block = statistics.mean_block(f)
    white, spa = statistics.reality_check(f, drawn, 1.0 / block)

    assert found.block == pytest.approx(block, rel=1e-12)
    assert (found.p, found.spa) == (pytest.approx(white), pytest.approx(spa))


@pytest.mark.parametrize("trials", [2, 3, 10, 46, 100, 1_000])
def test_the_deflation_threshold_and_the_backtest_length_agree(trials):
    assert stats.deflation_threshold(0.7, trials) == pytest.approx(
        statistics.deflation_threshold(0.7, trials), rel=1e-12)
    assert stats.min_backtest_length(trials, 1.3).years == pytest.approx(
        statistics.min_backtest_length(trials, 1.3), rel=1e-12)


VENUE = probatum.venues.binance_perp.replace(funding="none")
NEAR  = dict(rel=1e-9, abs=1e-9)


def run(seed, risk_free=0.0, record="full"):
    """A run long, short or flat in stretches of 20 rows over random bars four hours
    apart, which cross months, quarters and a year, at fees, slippage and impact, and
    the bars' closes."""
    rng   = np.random.Generator(np.random.PCG64(2_000 + seed))
    rows  = int(rng.choice([400, 900, 2_400]))
    close = 100.0 * np.exp(rng.normal(0.0, 0.01, rows).cumsum())
    bars  = np.column_stack([close, close * 1.003, close * 0.997, close, np.full(rows, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = 1_700_000_000 * 10**9 + np.arange(rows, dtype=np.int64) * 4 * 3_600 * 10**9,
    )

    held = np.repeat(rng.choice([-1.0, 0.0, 1.0], rows // 20 + 1), 20)[:rows]
    return probatum.backtest(held, data, VENUE, tolerance=0.05, risk_free=risk_free,
                             record=record), close


def closed(result):
    """A run's closed round trips in the order they closed: their pnl, the rows each was
    held, their sides and the rows they closed on."""
    trades = result.trades
    shut   = ~np.asarray(trades["open"])
    order  = np.argsort(np.asarray(np.ma.filled(trades["close_bar"], -1))[shut],
                        kind="stable")
    rows   = np.asarray(np.ma.filled(trades["close_bar"], -1))[shut][order]
    return (np.asarray(trades["pnl"])[shut][order].tolist(),
            (rows - np.asarray(trades["open_bar"])[shut][order]).tolist(),
            np.asarray(trades["side"])[shut][order].tolist(), rows.tolist())


@pytest.mark.parametrize("seed", CASES[:60])
def test_a_series_drawdowns_time_under_water_and_ulcer_index_agree(seed):
    values, rate, free = series(seed)
    path   = statistics.path(values.tolist())
    table  = stats.drawdown_table(values, top=1_000)
    theirs = statistics.drawdown_table(path, 1_000)
    held   = stats.time_under_water(values)
    found  = statistics.time_under_water(path)

    assert stats.drawdown(values) == pytest.approx(statistics.drawdown(path)[1:], **NEAR)
    assert [(e["peak"], e["trough"], -1 if e["recovered"] is None else e["recovered"],
             e["to_trough"], e["under"]) for e in theirs] == (
        list(zip(table["peak_bar"].tolist(), table["trough_bar"].tolist(),
                 np.ma.filled(table["recovered_bar"], -1).tolist(),
                 table["bars_to_trough"].tolist(), table["bars_under"].tolist())))
    assert table["depth_pct"] == pytest.approx([e["depth"] for e in theirs], **NEAR)
    assert np.ma.getmaskarray(table["peak_time"]).all()

    assert (held.longest, held.episodes, held.longest_days) == (
        found["longest"], found["episodes"], None)
    assert (held.mean, held.share) == (pytest.approx(found["mean"], **NEAR),
                                       pytest.approx(found["share"], **NEAR))
    assert stats.ulcer_index(values) == pytest.approx(statistics.ulcer_index(path), **NEAR)


@pytest.mark.parametrize("seed", range(6))
def test_a_run_s_drawdowns_are_read_on_its_equity_and_its_rows_times(seed):
    result, _ = run(seed)
    equity = result.equity.tolist()
    table  = stats.drawdown_table(result, top=3)
    theirs = statistics.drawdown_table(equity, 3)
    held   = stats.time_under_water(result)
    found  = statistics.time_under_water(equity, result.time.tolist())

    assert stats.drawdown(result) == pytest.approx(statistics.drawdown(equity), **NEAR)
    assert stats.drawdown(result).max() == pytest.approx(result.stats["max_drawdown_pct"],
                                                         rel=1e-12)
    assert table["peak_bar"].tolist() == [e["peak"] for e in theirs]
    assert table["peak_time"].tolist() == result.time[[e["peak"] for e in theirs]].tolist()
    assert table["trough_time"].tolist() == result.time[
        [e["trough"] for e in theirs]].tolist()

    assert (held.longest, held.episodes) == (found["longest"], found["episodes"])
    assert (held.longest_days, held.mean_days) == (
        pytest.approx(found["longest_days"], rel=1e-12),
        pytest.approx(found["mean_days"], rel=1e-12))
    assert f"{held.longest_days:,.0f} days under water" in result.summary()


@pytest.mark.parametrize("seed", CASES[:60])
def test_the_rolling_sharpe_and_volatility_agree_window_by_window(seed):
    values, rate, free = series(seed)
    listed = values.tolist()

    for window in (2, 3, 7, 30):
        sharpes = stats.rolling_sharpe(values, window, periods_per_year=rate, risk_free=free)
        spreads = stats.rolling_volatility(values, window, periods_per_year=rate)
        theirs  = zip(statistics.rolling_sharpe(listed, window, rate, free),
                      statistics.rolling_volatility(listed, window, rate))

        for sharpe, spread, (want, wide) in zip(sharpes, spreads, theirs):
            if want is None:
                assert math.isnan(sharpe) and math.isnan(spread)
                continue
            assert sharpe == pytest.approx(want, **NEAR)
            assert spread == pytest.approx(wide, **NEAR)


def test_a_window_of_returns_that_never_vary_ranks_as_the_record_ranks_it():
    # zeros at a rate rank at zero, and a gain on no risk at infinity, whatever the
    # running sums' rounding
    values = np.concatenate([np.random.default_rng(4).normal(0.0, 0.01, 50),
                             np.zeros(12), np.full(12, 0.0013)])
    found  = stats.rolling_sharpe(values, 10, periods_per_year=365, risk_free=0.04)
    wide   = stats.rolling_volatility(values, 10, periods_per_year=365)

    assert np.all(found[60:62] == 0.0) and np.all(np.isinf(found[-3:]))
    assert np.all(wide[-3:] == 0.0)
    assert found.tolist() == pytest.approx(
        [math.nan if v is None else v for v in statistics.rolling_sharpe(
            values.tolist(), 10, 365, 0.04)], nan_ok=True, **NEAR)
    assert wide.tolist() == pytest.approx(
        [math.nan if v is None else v for v in statistics.rolling_volatility(
            values.tolist(), 10, 365)], nan_ok=True, **NEAR)


@pytest.mark.parametrize("seed", range(6))
def test_a_run_s_rolling_series_are_its_returns_a_row_later(seed):
    result, _ = run(seed, risk_free=0.04)
    rate      = result.provenance["periods_per_year"]
    found     = stats.rolling_sharpe(result, 30)
    plain     = stats.rolling_sharpe(result.returns[1:], 30, periods_per_year=rate,
                                     risk_free=0.04)

    assert found.size == len(result.time) and np.isnan(found[:30]).all()
    assert found[1:] == pytest.approx(plain, nan_ok=True, rel=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_a_run_s_calendar_periods_agree_and_compound_to_it(seed):
    result, _ = run(seed)
    equity, times = result.equity.tolist(), result.time.tolist()

    for by in ("month", "quarter", "year"):
        table  = stats.period_returns(result, by=by)
        theirs = statistics.period_returns(times, equity, by)
        last   = [first + rows - 1 for _, first, rows, _ in theirs]

        assert table["period"].tolist() == [name for name, *_ in theirs]
        assert table["first_bar"].tolist() == [first for _, first, _, _ in theirs]
        assert table["rows"].tolist() == [rows for _, _, rows, _ in theirs]
        assert table["end"].tolist() == [times[at] for at in last]
        assert table["return_pct"] == pytest.approx([gain for *_, gain in theirs], **NEAR)
        assert math.prod(1.0 + gain / 100.0 for gain in table["return_pct"]) == (
            pytest.approx(1.0 + result.stats["total_return_pct"] / 100.0, rel=1e-9))


@pytest.mark.parametrize("seed", range(6))
def test_a_run_s_segments_agree_and_the_whole_is_its_record(seed):
    result, _ = run(seed, risk_free=0.04)
    rows      = len(result.time)
    rate      = result.provenance["periods_per_year"]
    pnl, _, _, shut = closed(result)
    holding   = np.abs(result.holdings["position"].reshape(rows, -1)).max(axis=1) > 0.0
    bars      = np.asarray(result.fills["bar"])
    # a round trip closing on a segment's first row is its own and not the one before's
    cut       = shut[len(shut) // 2]

    for start, stop in ((0, rows), (0, rows // 3), (rows // 3, 2 * rows // 3),
                        (2 * rows // 3, rows), (rows - 3, rows), (0, cut), (cut, rows)):
        found  = stats.segment(result, start, stop)
        theirs = statistics.segment(result.equity.tolist(), list(zip(shut, pnl)),
                                    holding.tolist(), start, stop, rate, 0.04)

        for name, value in theirs.items():
            assert found[name] == pytest.approx(value, **NEAR), name
        assert found["num_fills"] == np.count_nonzero((bars >= start) & (bars < stop))

    whole = stats.segment(result)
    assert {name: whole[name] for name in result.stats} == pytest.approx(
        dict(result.stats), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("seed", CASES[:80])
def test_the_returns_shape_and_tail_agree(seed):
    values, rate, free = series(seed)
    listed = values.tolist()

    assert stats.skew(values) == pytest.approx(statistics.skew(listed), **NEAR)
    assert stats.kurtosis(values) == pytest.approx(statistics.kurtosis(listed), **NEAR)
    assert stats.kurtosis(values, excess=True) == pytest.approx(
        statistics.kurtosis(listed) - 3.0, **NEAR)
    assert stats.omega(values) == pytest.approx(statistics.omega(listed), **NEAR)
    assert stats.omega(values, threshold=0.1, periods_per_year=rate) == pytest.approx(
        statistics.omega(listed, 0.1, rate), **NEAR)

    for alpha in (0.5, 0.9, 0.95, 0.99):
        assert stats.value_at_risk(values, alpha) == pytest.approx(
            statistics.value_at_risk(listed, alpha), **NEAR)
        assert stats.conditional_value_at_risk(values, alpha) == pytest.approx(
            statistics.conditional_value_at_risk(listed, alpha), **NEAR)
        assert stats.tail_ratio(values, alpha) == pytest.approx(
            statistics.tail_ratio(listed, alpha), **NEAR)


def test_the_tail_at_0_95_is_the_one_its_interval_reads():
    # at 24 million returns n * (1 - 0.95) is a part in a billion past n / 20 in
    # floating point, which a nudge of 1e-9 no longer takes back
    assert all(stats._counted(n, 0.95) == statistics.counted(n, 0.95)
               == statistics.tail(range(n)) for n in range(1, 10_001))
    assert stats._counted(24_000_000, 0.95) == statistics.counted(24_000_000, 0.95) == (
        1_200_000)
    assert stats.value_at_risk(np.arange(24_000_000.0)) == -1_199_999 * 100.0

    values = np.random.default_rng(6).normal(0.0005, 0.01, 1_000)
    for name in ("value_at_risk", "conditional_value_at_risk"):
        assert getattr(stats, name)(values) == pytest.approx(
            stats.bootstrap(values, name, periods_per_year=365, resamples=20).estimate,
            rel=1e-12)


@pytest.mark.parametrize("seed", range(15))
def test_the_round_trips_statistics_agree_on_any_pnl_and_on_trips_that_never_lost(seed):
    pnl  = trips(seed)
    held = np.random.default_rng(seed).integers(1, 30, pnl.size)

    # to the nearest hundred, so trips that made nothing part runs of losses
    for given in (pnl, np.abs(pnl), np.round(pnl, -2), pnl[:0]):
        found  = stats._traded(given, held[:given.size], 10_000.0)
        theirs = statistics.trade_stats(given.tolist(), held[:given.size].tolist(),
                                        10_000.0)
        for name, value in theirs.items():
            assert getattr(found, name) == pytest.approx(value, **NEAR), name


@pytest.mark.parametrize("seed", range(6))
def test_a_run_s_round_trips_agree_by_side_and_by_session(seed):
    result, _ = run(seed)
    pnl, held, side, shut = closed(result)
    opening = float(result.equity[0])
    times   = result.time.tolist()

    pairs = [(stats.trade_stats(result), statistics.trade_stats(pnl, held, opening))]
    for name, found in stats.long_short_split(result).items():
        mine = [k for k, at in enumerate(side) if at == name]
        pairs.append((found, statistics.trade_stats([pnl[k] for k in mine],
                                                    [held[k] for k in mine], opening)))

    for found, theirs in pairs:
        for name, value in theirs.items():
            assert getattr(found, name) == pytest.approx(value, **NEAR), name

    for by in ("hour", "weekday"):
        table  = stats.session_buckets(result, by=by)
        theirs = statistics.session_buckets(list(zip(shut, pnl)), times, by)

        assert table["bucket"].tolist() == list(theirs)
        assert table["trades"].tolist() == [count for count, _, _ in theirs.values()]
        assert table["pnl"] == pytest.approx([total for _, total, _ in theirs.values()],
                                             **NEAR)
        assert table["win_rate"] == pytest.approx([rate for *_, rate in theirs.values()],
                                                  **NEAR)


@pytest.mark.parametrize("seed", range(6))
def test_a_run_beside_another_series_and_where_its_money_went_agree(seed):
    result, close = run(seed, risk_free=0.04 * (seed % 2))
    given  = result.provenance
    rate   = given["periods_per_year"]
    found  = stats.benchmark(result, close)
    theirs = statistics.benchmark(result.returns[1:].tolist(), close.tolist(), rate,
                                  given["risk_free"])

    for name, value in theirs.items():
        assert getattr(found, name) == pytest.approx(value, **NEAR), name

    fills  = result.fills
    traded = math.fsum((np.abs(np.asarray(fills["size"]))
                        * np.asarray(fills["price"])).tolist())
    found  = stats.decompose(result)
    theirs = statistics.decompose(result.equity.tolist(), list(result.costs.values()),
                                  traded, rate)

    for name, value in theirs.items():
        assert getattr(found, name) == pytest.approx(value, **NEAR), name
    assert {name: getattr(found, name) for name in result.costs} == result.costs


def crossed(data, fast, slow, kind, stop):
    """A policy of two averages' crossing for a search of each kind of parameter."""
    up      = np.asarray(probatum.ta.ema(data.close, fast)).reshape(-1)
    down    = np.asarray(probatum.ta.ema(data.close, int(slow))).reshape(-1)
    weights = np.where(up > down, 1.0, 0.0 if kind == "flat" else -1.0)
    weights[np.isnan(up) | np.isnan(down)] = np.nan
    return {"policy": weights, "stop": stop}


@pytest.mark.parametrize("seed", range(3))
def test_a_search_s_parameter_importance_agrees_on_every_kind_of_parameter(seed):
    close = 100.0 * np.exp(np.random.default_rng(seed).normal(0.0, 0.01, 600).cumsum())
    bars  = np.column_stack([close, close * 1.003, close * 0.997, close, np.full(600, 1e3)])
    data  = probatum.Data(bars, symbol="BTCUSDT", market="linear",
                          time=np.arange(600, dtype=np.int64) * 3_600 * 10**9)
    space = {"fast": probatum.Int(2, 60), "slow": probatum.Float(61.0, 200.0),
             "kind": probatum.Categorical(["flat", "short"]),
             "stop": probatum.Categorical([None, 0.02, 0.05])}
    study = probatum.search(crossed, space, data, VENUE, sampler="random", trials=40,
                            holdout=0, seed=seed)

    table  = study.trials
    trials = [((row["fast"], row["slow"], row["kind"], row["stop"]), row["state"],
               row["score"]) for row in (table[k] for k in range(len(table)))]
    theirs = statistics.importance(trials, ["number", "number", "categorical",
                                            "categorical"])
    found  = study.importance()

    assert found.names == tuple(space)
    assert found.importance == pytest.approx([value for value, _ in theirs], **NEAR)
    assert found.chance == pytest.approx([level for _, level in theirs], rel=1e-12)
    assert found == stats.importance(study)


def lengths(data, length, kind):
    """A policy of a close against its average, for a search of a length's ten values."""
    average = np.asarray(probatum.ta.sma(data.close, 10 * length)).reshape(-1)
    weights = np.where(np.asarray(data.close).reshape(-1) > average, 1.0,
                       0.0 if kind == "flat" else -1.0)
    weights[np.isnan(average)] = np.nan
    return weights


def test_a_number_that_takes_ten_values_is_grouped_by_them_and_eleven_by_rank():
    # trials drawn unevenly over the values, so ten groups by rank would join some
    close = 100.0 * np.exp(np.random.default_rng(8).normal(0.0, 0.01, 600).cumsum())
    bars  = np.column_stack([close, close * 1.003, close * 0.997, close, np.full(600, 1e3)])
    data  = probatum.Data(bars, symbol="BTCUSDT", market="linear",
                          time=np.arange(600, dtype=np.int64) * 3_600 * 10**9)

    for top in (10, 11):
        space  = {"length": probatum.Int(1, top),
                  "kind":   probatum.Categorical(["flat", "short"])}
        study  = probatum.search(lengths, space, data, VENUE, sampler="random", trials=60,
                                 holdout=0, seed=5)
        table  = study.trials
        trials = [((row["length"], row["kind"]), row["state"], row["score"])
                  for row in (table[k] for k in range(len(table)))]
        theirs = statistics.importance(trials, ["number", "categorical"])
        found  = study.importance()

        assert len(set(table["length"].tolist())) == top
        assert found.importance == pytest.approx([value for value, _ in theirs], **NEAR)
        assert found.chance == pytest.approx([level for _, level in theirs], rel=1e-12)


def test_runs_set_side_by_side_hold_each_one_s_record():
    one, two = run(1)[0], run(2)[0]
    table    = probatum.compare({"one": one, "two": two})

    assert table["name"].tolist() == ["one", "two"]
    assert table["venue"].tolist() == ["binance_perp"] * 2
    assert table["data"].tolist() == [one.provenance["data"]["fingerprint"],
                                      two.provenance["data"]["fingerprint"]]
    assert table["rows"].tolist() == [len(one.time), len(two.time)]
    for key in one.stats:
        assert table[key].tolist() == [one.stats[key], two.stats[key]]

    picked = probatum.compare([one, two], keys=["sharpe"])
    assert picked.columns == ("name", "venue", "policy", "data", "rows", "sharpe")
    assert picked["name"].tolist() == ["binance_perp"] * 2
