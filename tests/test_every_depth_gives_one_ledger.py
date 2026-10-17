"""Every depth runs the same step: a script of targets played in one call, the same
targets as an array run in one call, the Engine stepped from Python with them, a
Strategy deciding them through its account, and the batch with each env given its own,
over random universes, record one ledger, one fill log, one order log and one set of
round trips; and an array run's barriers are the brackets the Engine stepped from Python
places by the same rules, and those a Strategy places through its account's handles and
events, at one instrument.
"""

import collections
import math
import random

import numpy as np
import pytest

import probatum
from probatum._strategy import _Run, _play
from reference import differential, universe_differential


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


class Targets(probatum.Strategy):
    """A target at each close, of the weights its rows give the Data's row."""


    def __init__(self, rows):
        self.rows = rows


    def next(self, bar, account):
        account.target(self.rows[bar.i])


def through_account(candles, times, venue, markets, strategy):
    """The Engine after ``strategy`` decided at each close of a Data of the candles, every
    row of which an instrument printed, through its account."""
    engine = probatum._probatum.Engine(candles, times=times, report=True, **venue)
    _play(strategy, _Run(engine, probatum.Data(candles, time=times), 0, markets))

    return engine


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
        engine = probatum._probatum.Engine(candles, times=times, report=True, **venue)
        engine.reset()
        for t in range(rows):
            engine.target(weights[t])
            engine.step()

        assert same(records(engine), run)

        # the last row is never read, so it holds
        array = probatum._probatum.Engine(candles, times=times, report=True, **venue)
        probatum._probatum.run_array(array,
                                     np.array(weights + [[math.nan] * len(kinds)]))

        assert same(records(array), run)

        strategy = through_account(candles, times, venue, kinds, Targets(weights))

        assert same(records(strategy), run)

    batch = probatum.Batch(candles, len(envs), times=times, report=True, **venue)
    batch.reset_all()
    for t in range(rows):
        batch.target(np.array([weights[t] for weights in envs]))
        batch.advance()

    for env, run in enumerate(played):
        assert same(batch.records(env), run)


def barrier(rng, rows):
    """A barrier as an array run takes it: none, a number, or a return for each row,
    a NaN or a return of exactly 1 or more among them."""
    u       = rng.random()
    returns = [0.005, 0.01, 0.03, 0.08]
    if u < 0.2:
        return None

    if u < 0.6:
        return rng.choice(returns)

    return [[math.nan if rng.random() < 0.15 else
             rng.choice([1.0, 1.5]) if rng.random() < 0.05 else rng.choice(returns)]
            for _ in range(rows)]


def levels(entry, q, k, stop, take_profit):
    """A round trip's stop and take-profit off its average `entry`, on the side of `q`,
    at the returns row `k` gives, None for a barrier it does not give or a price at or
    below zero."""
    def at(barrier):
        value = barrier if barrier is None or isinstance(barrier, float) else (
            barrier[k][0])
        return None if value is None or math.isnan(value) else value

    side = 1.0 if q > 0.0 else -1.0
    stop_at, take_at = at(stop), at(take_profit)
    out = [None if stop_at is None else entry * (1.0 - side * stop_at),
           None if take_at is None else entry * (1.0 + side * take_at)]
    return [None if price is None or price <= 0.0 else price for price in out]


def placed(engine, rows, stop, take_profit):
    """The Engine stepped from Python, placing at each close what an array run places
    by the rules of Barriers on an array: each round trip's bracket at the close of the
    bar it opened on, priced off its average entry at the returns of the row whose
    decision opened it, its legs still resting moved when a fill moves the entry, and a
    barrier's fill holding the instrument flat until a row asks for zero or for the
    other side. Returns the engine and how often a barrier filled, a bracket moved and
    a row was held flat."""
    state = engine.reset()
    held, decided, opened, priced, latched, seen = 0.0, None, None, None, 0.0, 0
    legs, counts = (None, None), collections.Counter()

    for k, value in enumerate(rows[:-1]):
        fills = engine.fills()
        if any(fill["order"] in legs for fill in fills[seen:] if fill["order"]):
            latched = math.copysign(1.0, held)
            counts["filled"] += 1
        seen = len(fills)

        q, entry = state.position, state.avg_entry
        if q == 0.0:
            legs = (None, None)
        elif held == 0.0 or held * q < 0.0:
            opened, priced = decided, entry
            stop_at, take_at = levels(entry, q, opened, stop, take_profit)
            legs = (None, None)
            if stop_at is not None or take_at is not None:
                legs = engine.bracket(stop_at, take_at)
        elif legs != (None, None) and entry != priced:
            priced = entry
            stop_at, take_at = levels(entry, q, opened, stop, take_profit)
            counts["moved"] += 1
            if legs[0] is not None:
                engine.replace(legs[0], trigger=stop_at)
            if legs[1] is not None:
                engine.replace(legs[1], price=take_at)

        held = q
        if not math.isnan(value):
            if latched and value != 0.0 and math.copysign(1.0, value) == latched:
                counts["held"] += 1
            else:
                latched = 0.0
                engine.target(value)
                decided = k

        state = engine.step()

    return engine, counts


class Bracketed(probatum.Strategy):
    """What `placed` places, decided through the account: a leg's fill read off its
    events, a bracket's legs as handles and moved through them."""


    def __init__(self, rows, stop, take_profit):
        self.rows        = rows
        self.stop        = stop
        self.take_profit = take_profit


    def init(self, data):
        self.held, self.decided, self.opened, self.priced = 0.0, None, None, None
        self.latched, self.legs = 0.0, (None, None)


    def next(self, bar, account):
        if any(event.kind == "filled" and event.order is not None
               and event.order in self.legs for event in account.events):
            self.latched = math.copysign(1.0, self.held)

        q, entry = account.position, account.entry
        if q == 0.0:
            self.legs = (None, None)
        elif self.held == 0.0 or self.held * q < 0.0:
            self.opened, self.priced = self.decided, entry
            stop_at, take_at = levels(entry, q, self.opened, self.stop, self.take_profit)
            self.legs = (None, None)
            if stop_at is not None or take_at is not None:
                self.legs = tuple(None if leg is None or leg.status == "refused" else leg
                                  for leg in account.bracket(stop_at, take_at))
        elif self.legs != (None, None) and entry != self.priced:
            self.priced = entry
            stop_at, take_at = levels(entry, q, self.opened, self.stop, self.take_profit)
            if self.legs[0] is not None:
                account.replace(self.legs[0], trigger=stop_at)
            if self.legs[1] is not None:
                account.replace(self.legs[1], price=take_at)

        self.held = q
        value     = self.rows[bar.i]
        if math.isnan(value):
            return

        if self.latched and value != 0.0 and math.copysign(1.0, value) == self.latched:
            return

        self.latched = 0.0
        account.target(value)
        self.decided = bar.i


def barred(seed):
    """One instrument of the first comparison's, its fill model and sizes drawn as
    there, with weights and barriers through the array run, the Engine stepped from
    Python and a Strategy, and what the Engine placed."""
    rng   = random.Random(seed)
    bars  = differential.walk(rng, rng.randint(12, 40))
    times = np.array(differential.clock(rng, len(bars)), dtype=np.int64)
    cfg   = differential.config(rng, times, inverse=seed % 4 == 3)

    cfg["intrabar"] = rng.choice(["pessimistic", "optimistic"])
    cfg.update(differential.fills(seed, seed))
    cfg.update(differential.sizes(seed, seed, cfg["quote"], [bars[0].open],
                                  cfg.get("contract")))

    spot = cfg["market"] == "spot"
    rows = [math.nan if rng.random() < 0.3 else 0.0 if rng.random() < 0.15 else
            rng.uniform(0.0 if spot else -2.0, 1.0 if spot else 2.0) for _ in bars]
    stop, take_profit = barrier(rng, len(bars)), barrier(rng, len(bars))

    candles  = np.array([list(bar) for bar in bars], dtype=np.float64)
    settings = differential.venue(cfg)
    array    = probatum._probatum.Engine(candles, times=times, report=True, **settings)
    shaped   = [b if b is None or isinstance(b, float) else np.array(b)
                for b in (stop, take_profit)]
    probatum._probatum.run_array(array, np.array(rows)[:, None], "weight", 0.0, *shaped)

    engine = probatum._probatum.Engine(candles, times=times, report=True, **settings)
    engine, counts = placed(engine, rows, stop, take_profit)

    strategy = through_account(candles, times, settings, [cfg["market"]],
                               Bracketed(rows, stop, take_profit))

    return array, engine, strategy, counts


@pytest.fixture(scope="module")
def cases():
    return [barred(seed) for seed in range(16)]


@pytest.mark.parametrize("seed", range(16))
def test_an_array_s_barriers_are_the_brackets_the_engine_stepped_from_python_places(
        seed, cases):
    array, engine, _, _ = cases[seed]

    assert same(records(array), records(engine))


@pytest.mark.parametrize("seed", range(16))
def test_an_array_s_barriers_are_the_brackets_a_strategy_places_through_its_account(
        seed, cases):
    array, _, strategy, _ = cases[seed]

    assert same(records(array), records(strategy))


def test_the_barrier_cases_fill_move_and_hold_their_brackets(cases):
    counts = sum((counts for *_, counts in cases), collections.Counter())

    assert counts["filled"] and counts["moved"] and counts["held"], counts
