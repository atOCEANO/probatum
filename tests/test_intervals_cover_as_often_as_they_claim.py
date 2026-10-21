"""Each interval covers what it estimates between 93% and 97% of the time at 95%. The
Sharpe's and the bootstrap's of the returns are drawn on the shapes a strategy's
returns take, independent, heavy-tailed, autocorrelated, and in the market for
stretches with nothing between them, at 1,000 daily bars and a year of hourly ones, the
bootstrap's at 500 daily bars too, the fewest it takes; the round trips' on trips that
win evenly, rarely and large, or often and small; and the difference of two Sharpes,
alpha and beta on independent returns at 1,000 daily bars. Returns in stretches too
long for their length get no interval of a bootstrap, and their cells hold the floor
that refuses them. Each cell draws its replications from a seed of its own, and a cell
outside the band is a defect of its statistic, fixed before the statistic ships (ADR
0031; Specification, Inference).

What a cell's intervals cover is each process's population value, integrated from its
marginal law. The cells run at length, in the opt-in coverage stage, and only with
--coverage.
"""

import collections
import concurrent.futures
import functools
import math
import os

import numpy as np
import pytest
import scipy.integrate
import scipy.signal
import scipy.stats

from probatum import stats

pytestmark = pytest.mark.coverage

BAND = (0.93, 0.97)
# 500 daily bars, the fewest a bootstrap of the returns takes, 1,000 of them and a year
# of hourly bars: the count, the bars a year, and each return's mean and deviation
SIZES = {
    "short":  (500, 365.0, 0.0008, 0.02),
    "daily":  (1_000, 365.0, 0.0008, 0.02),
    "hourly": (8_760, 8_760.0, 0.00003, 0.004),
}
# the replications a size's return cells draw: at the floor, the heavy-tailed returns'
# volatility covers near 93.8%, which 8,000 tell from 93% at three standard errors
REPLICATIONS = {"short": 8_000, "daily": 4_000, "hourly": 2_000}
KINDS        = ("normal", "student", "autocorrelated", "switched")
FREEDOM      = 5    # the Student t's degrees of freedom
PHI          = 0.2  # the autocorrelated process's first autocorrelation
STRETCH      = 50   # the mean bars the switched process stays in or out of the market
RETURNS      = ("total_return_pct", "cagr_pct", "volatility_pct", "sortino", "value_at_risk",
                "conditional_value_at_risk")


def seeded(*place):
    """The stream of one replication of one cell, by their places, so a cell's counts
    are the same whichever core ran which replication."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([31, *place])))


def replicated(job, cell, replications):
    """The counts ``job`` gives, summed over ``replications``, each drawn from a stream
    of its own, on up to 12 cores: an hourly replication holds a few hundred megabytes
    of resamples, and more workers would outgrow a machine of 16 gigabytes."""
    totals  = collections.Counter()
    workers = min(12, os.cpu_count() or 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for counts in pool.map(job, [(*cell, at) for at in range(replications)],
                               chunksize=max(1, replications // 256)):
            totals.update(counts)

    return totals


def returns(kind, count, mu, sigma, rng):
    """``count`` returns of ``kind``, each of mean ``mu`` and deviation ``sigma``; the
    switched process holds twice the mean on the half of its bars in the market, and
    within rounding the same deviation."""
    if kind == "normal":
        return rng.normal(mu, sigma, count)

    if kind == "student":
        return mu + sigma * rng.standard_t(FREEDOM, count) / math.sqrt(
            FREEDOM / (FREEDOM - 2.0))

    if kind == "autocorrelated":
        shocks = rng.normal(0.0, sigma * math.sqrt(1.0 - PHI * PHI), count)
        shocks[0] = rng.normal(0.0, sigma)
        return mu + scipy.signal.lfilter([1.0], [1.0, -PHI], shocks)

    flips = rng.random(count) < 1.0 / STRETCH
    held  = (np.cumsum(flips) + (rng.random() < 0.5)) % 2 == 1
    return np.where(held, rng.normal(2.0 * mu, sigma * math.sqrt(2.0), count), 0.0)


def law(kind, mu, sigma):
    """A return's marginal law: its continuous part and the weight of an atom at zero,
    the switched process's bars outside the market."""
    if kind in ("normal", "autocorrelated"):
        return scipy.stats.norm(mu, sigma), 0.0

    if kind == "student":
        return scipy.stats.t(FREEDOM, mu, sigma / math.sqrt(FREEDOM / (FREEDOM - 2.0))), 0.0

    return scipy.stats.norm(2.0 * mu, sigma * math.sqrt(2.0)), 0.5


def expected(kind, mu, sigma, f, upper=math.inf):
    """The mean of ``f`` of a return below ``upper``, a return of -99% or worse left out
    as the Student t's ruin, whose chance over a year is below one in ten thousand."""
    part, atom = law(kind, mu, sigma)
    low   = max(part.ppf(1e-13), -0.99)
    high  = min(part.ppf(1.0 - 1e-13), upper)
    value = (1.0 - atom) * scipy.integrate.quad(lambda r: f(r) * part.pdf(r), low, high,
                                                limit=500, epsabs=0.0, epsrel=1e-11)[0]
    return value + (atom * f(0.0) if upper > 0.0 else 0.0)


def population(kind, size):
    """The value each statistic estimates, for a return series of ``kind`` and
    ``size``."""
    count, rate, mu, sigma = SIZES[size]
    part, atom = law(kind, mu, sigma)

    mean     = expected(kind, mu, sigma, lambda r: r)
    spread   = math.sqrt(expected(kind, mu, sigma, lambda r: r * r) - mean * mean)
    growth   = expected(kind, mu, sigma, math.log1p)
    cutoff   = part.ppf(0.05 / (1.0 - atom))
    below    = expected(kind, mu, sigma, lambda r: r, upper=cutoff)
    downside = math.sqrt(expected(kind, mu, sigma, lambda r: r * r, upper=0.0))

    return {
        "sharpe":                    mean / spread * math.sqrt(rate),
        "total_return_pct":          math.expm1(count * growth) * 100.0,
        "cagr_pct":                  math.expm1(rate * growth) * 100.0,
        "volatility_pct":            spread * math.sqrt(rate) * 100.0,
        "sortino":                   mean / downside * math.sqrt(rate),
        "value_at_risk":             -cutoff * 100.0,
        "conditional_value_at_risk": -below / 0.05 * 100.0,
    }


def assert_covered(totals, names, cell):
    """Hold the share of a cell's replications whose interval covered each of ``names``
    to the band."""
    covered = {name: totals[name] / totals[f"{name} read"] for name in names}

    # printed so the stage's log holds every cell's coverage, not a pass alone
    print(f"{cell}: " + ", ".join(f"{name} {value:.4f}" for name, value in covered.items()))

    outside = {name: round(value, 4) for name, value in covered.items()
               if not BAND[0] <= value <= BAND[1]}
    assert not outside, f"{cell}: {outside} of {covered}"


def sharpe_once(kind, size, truth, place):
    count, rate, mu, sigma = SIZES[size]
    held = stats.sharpe_interval(returns(kind, count, mu, sigma, seeded(*place)),
                                 periods_per_year=rate)
    return {"sharpe": int(held.low <= truth <= held.high), "sharpe read": 1}


@pytest.mark.parametrize("size", ("daily", "hourly"))
@pytest.mark.parametrize("kind", KINDS)
def test_the_sharpe_s_interval_covers_its_sharpe(kind, size):
    job    = functools.partial(sharpe_once, kind, size, population(kind, size)["sharpe"])
    totals = replicated(job, (1, KINDS.index(kind), ("daily", "hourly").index(size)),
                        4_000)

    assert_covered(totals, ["sharpe"], (kind, size))


def returns_once(kind, size, truths, place):
    # one draw of the shipped 2,000 resamples serves every statistic, each drawing the
    # same rows from the stream its returns seed; a sample a floor refuses is no read
    count, rate, mu, sigma = SIZES[size]
    r      = returns(kind, count, mu, sigma, seeded(*place))
    counts = {}
    for name, (held, _) in stats._return_core(r, r, rate, tuple(truths), 0.95, 2_000,
                                               None).items():
        if held is not None:
            counts[name] = int(held.low <= truths[name] <= held.high)
            counts[f"{name} read"] = 1

    return counts


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kind", KINDS[:3])
def test_each_return_statistic_s_interval_covers_its_population_value(kind, size):
    truths = {name: value for name, value in population(kind, size).items()
              if name in RETURNS}
    job    = functools.partial(returns_once, kind, size, truths)
    totals = replicated(job, (2, KINDS.index(kind), list(SIZES).index(size)),
                        REPLICATIONS[size])

    assert_covered(totals, truths, (kind, size))


def blocks_once(size, place):
    count, rate, mu, sigma = SIZES[size]
    r = returns("switched", count, mu, sigma, seeded(*place))
    return {"refused": int(r.size < stats.BLOCKS * stats._return_block(r)), "drawn": 1}


@pytest.mark.parametrize("size", SIZES)
def test_returns_in_long_stretches_get_no_bootstrap_until_they_hold_its_blocks(size):
    # a mean stretch of 50 bars leaves these returns some 20 to 65 of the bootstrap's
    # blocks, where their intervals cover from 86% to 93%; at 500 bars a run can spend
    # nearly all of them in one stretch, whose returns no rule can tell from independent
    totals = replicated(functools.partial(blocks_once, size), (6, list(SIZES).index(size)),
                        2_000)

    least = 0.9 if size == "short" else 0.99
    assert totals["refused"] / totals["drawn"] >= least, (size, totals)


TRIPS = ("even", "trend", "reversion")


def trips(kind, count, rng):
    """``count`` round trips' pnl: even and normal, a trend follower's that win 40% of
    the time and pay 2.5 times what they lose, or a reversion's that win 65% of the time
    and lose 1.5 times what they win."""
    if kind == "even":
        return rng.normal(0.2, 1.0, count)

    won  = rng.random(count) < (0.4 if kind == "trend" else 0.65)
    paid = rng.exponential(2.5 if kind == "trend" else 0.8, count)
    lost = rng.exponential(1.0 if kind == "trend" else 1.2, count)
    return np.where(won, paid, -lost)


def trip_truths(kind):
    """Each trade statistic's population value: the chance of a win, the expected gain
    over the expected loss, and the mean in percent of an opening balance of 1."""
    if kind == "even":
        gain = 0.2 * scipy.stats.norm.cdf(0.2) + scipy.stats.norm.pdf(0.2)
        loss = -0.2 * scipy.stats.norm.cdf(-0.2) + scipy.stats.norm.pdf(0.2)
        return {
            "win_rate":      scipy.stats.norm.sf(-0.2),
            "profit_factor": gain / loss,
            "avg_trade_pct": 20.0,
        }

    if kind == "trend":
        return {
            "win_rate":      0.4,
            "profit_factor": 0.4 * 2.5 / (0.6 * 1.0),
            "avg_trade_pct": (0.4 * 2.5 - 0.6 * 1.0) * 100.0,
        }

    return {
        "win_rate":      0.65,
        "profit_factor": 0.65 * 0.8 / (0.35 * 1.2),
        "avg_trade_pct": (0.65 * 0.8 - 0.35 * 1.2) * 100.0,
    }


def trips_once(kind, count, truths, place):
    pnl    = trips(kind, count, seeded(*place))
    counts = {}
    for name, truth in truths.items():
        if stats._thin_trips(pnl, name):
            continue

        _, low, high = stats._trade_bounds(pnl, 1.0, name, 0.95, 2_000,
                                           stats._drawn(pnl, "stats trades", None))
        counts[name] = int(low <= truth <= high)
        counts[f"{name} read"] = 1

    return counts


@pytest.mark.parametrize("count", (60, 160))
@pytest.mark.parametrize("kind", TRIPS)
def test_each_trade_statistic_s_interval_covers_its_population_value(kind, count):
    # at the floor, 60 round trips, a true coverage near 93.7% needs the 8,000
    # replications to tell it from 93% at two and a half standard errors
    truths = trip_truths(kind)
    totals = replicated(functools.partial(trips_once, kind, count, truths),
                        (3, TRIPS.index(kind), count), 8_000)

    assert_covered(totals, truths, (kind, count))


def difference_once(place):
    count, rate, mu, sigma = SIZES["daily"]
    first, other = seeded(*place).normal(0.0, 1.0, (2, count))
    a     = mu + sigma * first
    b     = mu + sigma * (0.5 * first + math.sqrt(0.75) * other)
    found = stats.sharpe_difference(a, b, periods_per_year=rate)

    return {
        "delta":          int(found.delta.low <= 0.0 <= found.delta.high),
        "delta read":     1,
        "bootstrap":      int(found.bootstrap.low <= 0.0 <= found.bootstrap.high),
        "bootstrap read": 1,
    }


def test_two_equal_sharpes_differ_at_both_methods_size_of_5_percent():
    totals = replicated(difference_once, (4,), 2_000)

    assert_covered(totals, ["delta", "bootstrap"], "sharpe difference")


def regression_once(place):
    count, rate, alpha, beta = 1_000, 365.0, 0.0003, 0.6
    rng   = seeded(*place)
    y     = rng.normal(0.0004, 0.02, count)
    x     = alpha + beta * y + rng.normal(0.0, 0.01, count)
    found = stats._regression(x, y, rate, None, 0.95)

    return {
        "alpha":      int(found.alpha_low <= alpha * rate <= found.alpha_high),
        "alpha read": 1,
        "beta":       int(found.beta_low <= beta <= found.beta_high),
        "beta read":  1,
    }


def test_alpha_s_and_beta_s_intervals_cover_them():
    totals = replicated(regression_once, (5,), 4_000)

    assert_covered(totals, ["alpha", "beta"], "alpha and beta")
