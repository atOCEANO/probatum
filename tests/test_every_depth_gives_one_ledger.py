"""Every depth runs the same step: a script of targets played in one call, the Engine
stepped from Python with the same targets, and the batch with each env given its own,
over random universes, record one ledger, one fill log, one order log and one set of
round trips.
"""

import math
import random

import numpy as np
import pytest

import probatum
from reference import universe_differential


def same(mine, theirs):
    """Whether two records are equal, NaN where both hold NaN."""
    if isinstance(mine, dict):
        return mine.keys() == theirs.keys() and all(
            same(mine[key], theirs[key]) for key in mine)

    if isinstance(mine, list):
        return len(mine) == len(theirs) and all(map(same, mine, theirs))

    if isinstance(mine, np.ndarray):
        return np.array_equal(mine, theirs, equal_nan=mine.dtype.kind == "f")

    if isinstance(mine, float) and isinstance(theirs, float):
        return mine == theirs or (math.isnan(mine) and math.isnan(theirs))

    return mine == theirs


def case(rng):
    """A universe of the universe comparison's, its venue under one fixed setting,
    and its candles with NaN where an instrument printed no bar."""
    times, kinds, columns = universe_differential.universe(rng)
    cfg = universe_differential.config(rng, times, kinds, columns)
    cfg["intrabar"] = rng.choice(["pessimistic", "optimistic"])

    nan     = [math.nan] * 5
    candles = np.array([[nan if column[t] is None else list(column[t])
                         for column in columns] for t in range(len(times))],
                       dtype=np.float64)

    return np.array(times, dtype=np.int64), kinds, candles, cfg


def targets(rng, kinds, rows):
    """A target at every close but the last: a spot weight from 0 to 1 and a
    perpetual's from -1 to 1."""
    return [[rng.uniform(0.0, 1.0) if kind == "spot" else rng.uniform(-1.0, 1.0)
             for kind in kinds] for _ in range(rows)]


def records(engine):
    return {
        "ledger":    engine.ledger(),
        "fills":     engine.fills(),
        "order_log": engine.order_log(),
        "trades":    engine.trades(),
    }


@pytest.mark.parametrize("seed", range(8))
def test_a_script_the_engine_stepped_from_python_and_the_batch_record_one_run(seed):
    rng = random.Random(seed)
    times, kinds, candles, cfg = case(rng)

    venue  = universe_differential.venue(cfg)
    rows   = len(times) - 1
    envs   = [targets(rng, kinds, rows) for _ in range(3)]
    played = [probatum._probatum.run(
        candles, venue, {t: [("target", {"weights": weights[t]})] for t in range(rows)},
        times=times) for weights in envs]

    assert any(run["fills"] for run in played)

    for weights, run in zip(envs, played):
        engine = probatum.Engine(candles, times=times, report=True, **venue)
        engine.reset()
        for t in range(rows):
            engine.target(weights[t])
            engine.step()

        assert same(records(engine), run)

    batch = probatum.Batch(candles, len(envs), times=times, report=True, **venue)
    batch.reset_all()
    for t in range(rows):
        batch.target(np.array([weights[t] for weights in envs]))
        batch.advance()

    for env, run in enumerate(played):
        assert same(batch.records(env), run)
