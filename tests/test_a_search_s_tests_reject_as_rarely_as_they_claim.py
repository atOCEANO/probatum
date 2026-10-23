"""A search's tests reject as rarely as they claim. The deflated Sharpe passes searches
with no skill, against every trial and for the winner's cluster, no more than one time
in twenty from ten trials on, and against every trial fewer at a hundred trials than at
two: searches of independent trials, coin flips, and of trials that move together,
moving-average crossovers, each on a random walk beside a random null of as many
trials. The Reality Check and SPA reject trials that gain nothing over cash between 3%
and 7% of the time at 5%, on independent returns. Each cell draws its replications from
a seed of its own, and a cell past its bound is a defect of its test, fixed before the
test ships (ADR 0033; Specification, Across a search).

The cells run at length, in the opt-in coverage stage, and only with --coverage.
"""

import functools

import numpy as np
import pytest

import probatum
from probatum import stats

from test_intervals_cover_as_often_as_they_claim import replicated, seeded

pytestmark = pytest.mark.coverage

DAY   = 86_400_000_000_000
LEVEL = 0.05
BAND  = (0.03, 0.07)
# the searches a count of trials draws: the winner's cluster among ten crossovers
# passes near 3.9% of them, which 4,000 tell from 5% at three and a half standard errors
SEARCHES = {2: 3_000, 10: 4_000, 100: 1_000}


def coin(data, k):
    """Long or short at every close, the side drawn from the trial's own stream, so no
    trial can know a bar's return."""
    return np.where(np.random.default_rng(k).standard_normal(len(data)) > 0.0, 1.0, -1.0)


def crossing(data, fast, slow):
    """Long while the fast average is above the slow one, and short while it is below."""
    up   = probatum.ta.sma(data.close, fast)
    down = probatum.ta.sma(data.close, slow)

    return np.where(np.isnan(down), np.nan, np.where(up > down, 1.0, -1.0))


POLICIES = {
    "coin":     (coin, {"k": probatum.Int(0, 2**40)}),
    "crossing": (crossing, {"fast": probatum.Int(2, 30), "slow": probatum.Int(31, 200)}),
}


def deflated_once(name, trials, place):
    # a thousand daily bars of a random walk, which no policy trades at a profit,
    # searched twice: the search judged and its random null
    rng   = seeded(*place)
    close = 100.0 * np.exp(rng.normal(0.0, 0.02, 1_000).cumsum())
    data  = probatum.Data(
        np.column_stack([close, close * 1.01, close * 0.99, close, np.full(1_000, 1e9)]),
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(1_000, dtype=np.int64) * DAY,
    )

    policy, space = POLICIES[name]
    first, second = (int(seed) for seed in rng.integers(0, 2**63, 2))

    study = probatum.search(policy, space, data, "frictionless", sampler="random",
                            trials=trials, holdout=0, seed=first)
    null  = probatum.search(policy, space, data, "frictionless", sampler="random",
                            trials=trials, holdout=0, seed=second)

    found = stats.deflated_sharpe(study, null)
    return {
        "every":   int(found.value >= 0.95),
        "cluster": int(found.effective_value >= 0.95),
        "drawn":   1,
    }


@pytest.mark.parametrize("name", POLICIES)
def test_a_search_with_no_skill_passes_the_deflation_at_most_one_time_in_twenty(name):
    passed = {}
    for trials, searches in SEARCHES.items():
        totals = replicated(functools.partial(deflated_once, name, trials),
                            (7, list(POLICIES).index(name), trials), searches)
        passed[trials] = {kind: totals[kind] / totals["drawn"]
                          for kind in ("every", "cluster")}
        # printed so the stage's log holds every cell's rate, not a pass alone
        print(f"{name}, {trials} trials: " + ", ".join(
            f"{kind} {value:.4f}" for kind, value in passed[trials].items()))

    # two trials' threshold falls short of the best of two by luck, so the bound holds
    # from ten
    over = {trials: rates for trials, rates in passed.items()
            if trials >= 10 and max(rates.values()) > LEVEL}

    assert not over, (name, passed)
    assert passed[100]["every"] < passed[2]["every"], (name, passed)


def checked_once(place):
    found = stats.reality_check(np.expm1(seeded(*place).normal(0.0, 0.01, (10, 1_000))),
                                periods_per_year=365.0)
    return {"white": int(found.p <= LEVEL), "spa": int(found.spa <= LEVEL), "drawn": 1}


def test_trials_that_gain_nothing_over_cash_are_rejected_at_their_level():
    # independent returns are rejected near 4.9% of the time, which 3,000 hold inside
    # the band at four standard errors
    totals   = replicated(checked_once, (8,), 3_000)
    rejected = {name: totals[name] / totals["drawn"] for name in ("white", "spa")}

    print("reality check: " + ", ".join(f"{name} {value:.4f}"
                                        for name, value in rejected.items()))

    outside = {name: value for name, value in rejected.items()
               if not BAND[0] <= value <= BAND[1]}

    assert not outside, rejected
