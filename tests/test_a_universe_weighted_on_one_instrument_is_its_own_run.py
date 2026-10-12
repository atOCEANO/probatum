"""A universe in which every weight but one is zero reproduces that instrument's own
run: the shared account, the cross margin, the path and the funding add nothing for
instruments the account never holds.
"""

import math
import random

import numpy as np
import pytest

import probatum
from reference import differential, universe_differential
from reference.simulator import Bar

ACCOUNT = ("equity", "spot_cash", "margin_cash", "gross_exposure", "net_exposure",
           "leverage", "margin_used", "maintenance", "fees", "funding", "spread",
           "slippage", "impact", "liquidation", "dust")
LEGS = ("position", "entry", "mark", "notional", "unrealized", "weight", "drift")


def case(rng):
    """Two to five instruments of one kind, each printing every row on its own walk
    at its own scale, and a venue the universe comparison draws for them."""
    n, rows = rng.randint(2, 5), rng.randint(8, 40)
    times   = differential.clock(rng, rows)
    kind    = rng.choice(["spot", "linear"])
    columns = []

    for _ in range(n):
        scale = rng.choice([0.05, 1.0, 30.0])
        columns.append([Bar(*(price * scale for price in bar[:4]), bar.volume)
                        for bar in differential.walk(rng, rows)])

    cfg = universe_differential.config(rng, times, [kind] * n, columns)
    cfg["intrabar"] = rng.choice(["pessimistic", "optimistic"])

    return np.array(times, dtype=np.int64), kind, columns, cfg


def alone(cfg, k):
    """The venue for instrument `k` on its own: its market and tier table, and the
    funding events that name it or every perpetual held."""
    venue = universe_differential.venue(cfg)
    spec  = venue.pop("instruments")[k]
    venue.update(market=spec["market"], tiers=spec["tiers"], tiers_by=spec["tiers_by"])

    named = venue.pop("funding_instruments", None)
    if named is not None:
        kept = [j for j, i in enumerate(named) if i in (None, k)]
        venue["funding_times"] = venue["funding_times"][kept]
        venue["funding_rates"] = venue["funding_rates"][kept]

    return venue


def without_instrument(records):
    return [{key: value for key, value in record.items() if key != "instrument"}
            for record in records]


def same(mine, theirs):
    """Whether two lists of records are equal, NaN where both hold NaN."""
    def equal(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a == b or (math.isnan(a) and math.isnan(b))
        return a == b

    return len(mine) == len(theirs) and all(
        a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        for a, b in zip(mine, theirs))


@pytest.mark.parametrize("seed", range(24))
def test_a_universe_weighted_on_one_instrument_runs_as_that_instrument_alone(seed):
    rng = random.Random(seed)
    times, kind, columns, cfg = case(rng)
    n, rows = len(columns), len(times) - 1
    k = rng.randrange(n)

    # up to eight times equity either way on a perpetual, so the cap and the tiers
    # bind and a violent bar liquidates
    low, high = (0.0, 3.0) if kind == "spot" else (-8.0, 8.0)
    weights = [rng.uniform(low, high) for _ in range(rows)]
    candles = np.array([[list(column[t]) for column in columns]
                        for t in range(len(times))], dtype=np.float64)

    together = probatum._probatum.run(
        candles, universe_differential.venue(cfg),
        {t: [("target", {"weights": [weights[t] if i == k else 0.0
                                     for i in range(n)]})] for t in range(rows)},
        times=times)
    on_its_own = probatum._probatum.run(
        candles[:, k, :], alone(cfg, k),
        {t: [("target", {"weights": weights[t]})] for t in range(rows)}, times=times)

    assert together["fills"]
    for name in ACCOUNT:
        assert np.array_equal(together["ledger"][name], on_its_own["ledger"][name],
                              equal_nan=True), name
    for name in LEGS:
        assert np.array_equal(together["ledger"][name][:, k],
                              on_its_own["ledger"][name], equal_nan=True), name
    for log in ("fills", "order_log", "trades"):
        assert same(without_instrument(together[log]),
                    without_instrument(on_its_own[log])), log
