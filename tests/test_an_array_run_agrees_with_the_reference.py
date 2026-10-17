"""An array run through the engine in one call against the reference's, on every record.

Two in three cases are a universe the universe comparison draws and the rest one
instrument the first comparison draws, every other one an inverse contract. Each runs
rows of weights or of quantities, a NaN holding and a zero now and then, with barriers
that are a number or rows of their own, a NaN or a return of exactly 1 or more among
them, and a third of the accounts open part way in. Besides every record the two
write, the orders resting at a few closes are compared by id, so a barrier priced
wrong is caught where no bar reaches it. Every gate stage runs a short comparison and
the slow stage one at length, which has to reach each rule of an array run in every
kind of case.
"""

import collections
import math
import random

import numpy as np
import pytest

import probatum
from reference import differential, universe_differential
from reference.simulator import play_array

# the rules of an array run, and those its barriers and sizing lean on, which a
# comparison at length must reach
ARRAY = [
    "a bracket cancelled by a liquidation",
    "a bracket leg filled",
    "a bracket leg filled in part",
    "a flip split by a target",
    "a position's exits reordered",
    "a replace",
    "a target in quantities",
    "an array's barrier at or below zero left out",
    "an array's bracket moved with its entry",
    "an array's bracket placed at its entry",
    "an array's instrument held flat after its barrier",
    "an array's instrument named again after its barrier",
]
# what every kind of case has to reach on its own
EACH = ["a bracket leg filled", "an array's bracket moved with its entry",
        "an array's bracket placed at its entry"]
KINDS = ("universe", "one", "inverse")

Outcome = collections.namedtuple("Outcome", "compared agreed disagreed rules")


def array(rng, kinds, opens, balance, contract, rows):
    """An array's rows, its sizing, its tolerance and its two barriers: a spot weight
    from 0 and a perpetual's either side of it, or as many base units or contracts as
    that weight of `balance` buys at the instrument's first open."""
    sizing = rng.choice(["weight", "weight", "quantity"])

    def value(k, kind):
        u = rng.random()
        if u < 0.35:
            return math.nan
        if u < 0.5:
            return 0.0

        weight = rng.uniform(0.0, 1.0) if kind == "spot" else rng.uniform(-2.0, 2.0)
        if sizing == "weight":
            return weight
        if kind == "inverse":
            return weight * balance * opens[k] / contract
        return weight * balance / opens[k]

    def barrier():
        u       = rng.random()
        returns = [0.005, 0.01, 0.03, 0.08]
        if u < 0.25:
            return None
        if u < 0.6:
            return rng.choice(returns)
        return [[math.nan if rng.random() < 0.15 else
                 rng.choice([1.0, 1.5]) if rng.random() < 0.05 else rng.choice(returns)
                 for _ in kinds] for _ in range(rows)]

    values = [[value(k, kind) for k, kind in enumerate(kinds)] for _ in range(rows)]
    return values, sizing, rng.choice([0.0, 0.0, 0.1]), barrier(), barrier()


def from_row(drawn, start):
    """The array's rows, and any barrier given as rows, from the account's first."""
    values, sizing, tolerance, stop, take_profit = drawn
    cut = [barrier[start:] if isinstance(barrier, list) else barrier
           for barrier in (stop, take_profit)]
    return values[start:], sizing, tolerance, *cut


def engine(candles, times, settings, drawn, start):
    """The engine after the array run in one call, its account opened at row `start`."""
    run = probatum._probatum.Engine(
        candles,
        times  = np.array(times, dtype=np.int64),
        report = True,
        **settings,
    )

    values, sizing, tolerance, stop, take_profit = from_row(drawn, start)

    def shaped(barrier):
        if barrier is None or isinstance(barrier, float):
            return barrier
        return np.array(barrier, dtype=np.float64)

    probatum._probatum.run_array(run, np.array(values, dtype=np.float64), sizing,
                                 tolerance, shaped(stop), shaped(take_profit), start=start)

    return run


def records(run):
    return dict(ledger=run.ledger(), fills=run.fills(), order_log=run.order_log(),
                trades=run.trades())


def resting(run):
    """The orders resting once the run has ended, by id: each one's kind, side, price
    and trigger."""
    return {order["id"]: (order["kind"], order["side"], order["price"], order["trigger"])
            for order in run.step().open_orders}


def rested(sim):
    """The reference's resting orders as `resting` gives the engine's."""
    return {order["id"]: (order["kind"], order["side"], order["price"], order["trigger"])
            for order in sim.resting()}


def apart(mine, theirs):
    """Where two sets of resting orders differ, or None."""
    if mine.keys() != theirs.keys():
        return f"orders {sorted(mine)} rest against {sorted(theirs)}"

    for order, (kind, side, *prices) in mine.items():
        their_kind, their_side, *their_prices = theirs[order]
        if (kind, side) != (their_kind, their_side) or not all(
                (a is None) == (b is None) and (a is None or differential.close_enough(a, b))
                for a, b in zip(prices, their_prices)):
            return f"order {order} rests as {mine[order]} against {theirs[order]}"
    return None


def funded_to(cfg, time):
    """`cfg` with the funding events its data holds cut at `time`."""
    source = cfg["funding"]
    if source[0] != "data":
        return cfg

    return dict(cfg, funding=("data", [event for event in source[1]
                                       if event[0] <= time]))


def universe(rng, seed, number):
    """A universe of the universe comparison's through both, where they first part
    company before a bar, and where the orders resting at a few closes differ."""
    times, kinds, columns = universe_differential.universe(rng)
    cfg   = universe_differential.config(rng, times, kinds, columns)
    opens = universe_differential.opens(columns)

    cfg.update(differential.fills(seed, number))
    cfg.update(differential.sizes(seed, number, cfg["quote"], opens))
    cfg.update(differential.fees(seed, number, kinds))

    tables = differential.opened(seed, number, [rows for rows, _ in cfg["tiers"]])
    cfg["tiers"]           = [(rows, by) for rows, (_, by) in zip(tables, cfg["tiers"])]
    cfg["intrabar"]        = differential.setting(seed, number)
    cfg["liquidation"]     = differential.liquidation(seed, number, tables)
    cfg["liquidation_fee"] = differential.liquidation_fee(seed, number)

    drawn = array(random.Random(f"array {seed} {number}"), kinds, opens, cfg["quote"],
                  None, len(times))
    start = opening(seed, number, len(times))

    def through(rows):
        nan     = [math.nan] * 5
        candles = np.array([[nan if column[t] is None else list(column[t])
                             for column in columns] for t in range(rows)],
                           dtype=np.float64)

        part = funded_to(cfg, times[rows - 1])
        run  = engine(candles, times[:rows], universe_differential.venue(part),
                      cut(drawn, rows), start)
        sim = universe_differential.make_reference(
            times[:rows], kinds, [column[:rows] for column in columns], part, start=start)
        return run, play_array(sim, *from_row(cut(drawn, rows), start))

    run, sim = through(len(times))
    found = universe_differential.disagreement(records(run), sim, end=None)
    return sim, found or closes(through, start, len(times), seed, number)


def one(rng, seed, number, inverse):
    """One instrument of the first comparison's, an inverse contract with `inverse`,
    through both, where they first part company before a bar, and where the orders
    resting at a few closes differ."""
    bars  = differential.walk(rng, rng.randint(8, 40))
    times = differential.clock(rng, len(bars))
    cfg   = differential.config(rng, times, inverse=inverse)

    cfg["intrabar"]        = differential.setting(seed, number)
    cfg["liquidation"]     = differential.liquidation(seed, number, [cfg["tiers"]])
    cfg["liquidation_fee"] = differential.liquidation_fee(seed, number)
    [cfg["tiers"]] = differential.opened(seed, number, [cfg["tiers"]])

    cfg.update(differential.fills(seed, number))
    cfg.update(differential.sizes(seed, number, cfg["quote"], [bars[0].open],
                                  cfg.get("contract")))

    kind  = {"perp": "linear"}.get(cfg["market"], cfg["market"])
    drawn = array(random.Random(f"array one {seed} {number}"), [kind], [bars[0].open],
                  cfg["quote"], cfg.get("contract"), len(bars))
    start = opening(seed, number, len(bars))

    def through(rows):
        part    = funded_to(cfg, times[rows - 1])
        candles = np.array([list(bar) for bar in bars[:rows]], dtype=np.float64)
        run     = engine(candles, times[:rows], differential.venue(part), cut(drawn, rows),
                         start)
        sim = differential.make_reference(bars[:rows], part, times[:rows], start=start)
        return run, play_array(sim, *from_row(cut(drawn, rows), start))

    run, sim = through(len(bars))
    found = differential.disagreement([], records(run), sim, cfg, end=None)
    return sim, found or closes(through, start, len(bars), seed, number)


def opening(seed, number, rows):
    """The row a case's account opens at: the first, or one up to halfway in for a
    third of the cases, drawn apart from the stream the cases come from."""
    draw = random.Random(f"opening {seed} {number}")
    return draw.randint(1, rows // 2) if draw.random() < 1.0 / 3.0 and rows > 4 else 0


def cut(drawn, rows):
    """The array's rows, and any barrier of rows, up to row `rows`."""
    values, sizing, tolerance, stop, take_profit = drawn
    return (values[:rows], sizing, tolerance,
            *[barrier[:rows] if isinstance(barrier, list) else barrier
              for barrier in (stop, take_profit)])


def closes(through, start, rows, seed, number):
    """Where the orders resting at the end of a few runs cut short of the whole first
    differ, each run's end the state its bar left before the close's decisions."""
    draw = random.Random(f"closes {seed} {number}")
    for end in sorted(draw.sample(range(start + 2, rows), min(3, rows - start - 2))):
        run, sim = through(end + 1)
        found = apart(resting(run), rested(sim))
        if found is not None:
            return end, f"after bar {end}: {found}"
    return None


def compare(cases, seed):
    """`cases` array runs from `seed` through both: how many were compared whole and
    agreed, the disagreements, and the rules each kind of case reached."""
    rng   = random.Random(f"arrays {seed}")
    rules = {kind: collections.Counter() for kind in KINDS}
    whole, agreed, disagreed = 0, 0, []

    for number in range(cases):
        kind = KINDS[0] if number % 3 < 2 else KINDS[1 + (number % 6 == 5)]
        try:
            if kind == "universe":
                sim, found = universe(rng, seed, number)
            else:
                sim, found = one(rng, seed, number, inverse=kind == "inverse")
        except Exception as exc:
            disagreed.append((number, f"raised: {type(exc).__name__}: {exc}"))
            continue

        if differential.cut(sim) is not None:
            continue

        rules[kind].update(sim.first)
        whole += 1
        if found is not None:
            disagreed.append((number, found[1]))
        else:
            agreed += 1

    return Outcome(whole, agreed, disagreed, rules)


def check(cases, seed):
    out    = compare(cases, seed)
    report = (f"{out.compared} of {cases} compared whole, {out.agreed} agreed; "
              + "; ".join(f"case {number}: {why}" for number, why in out.disagreed[:5]))

    assert not out.disagreed, report
    # no array run cancels an order the engine has yet to, so every case is whole
    assert out.compared == cases, report

    return out


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_random_array_runs_agree_on_every_record_and_every_resting_order(seed):
    out     = check(45, seed)
    reached = sum(out.rules.values(), collections.Counter())

    assert not [rule for rule in ["a bracket leg filled",
                                  "an array's bracket placed at its entry",
                                  "an array's instrument held flat after its barrier"]
                if not reached[rule]]


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42, 1337, 20261005, 99991])
def test_random_array_runs_agree_at_length_and_reach_every_rule_of_an_array(seed):
    out     = check(300, seed)
    reached = sum(out.rules.values(), collections.Counter())

    assert not [rule for rule in ARRAY if not reached[rule]]
    assert not [(kind, rule) for kind in KINDS for rule in EACH
                if not out.rules[kind][rule]]
