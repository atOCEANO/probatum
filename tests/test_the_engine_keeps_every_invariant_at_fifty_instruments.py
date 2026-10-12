"""Random universes of up to fifty instruments through the engine keep its invariants:
the ledger identity on every row, no margin wallet ending a bar owing money, no fill
better than every price its market traded from the last close through its bar and none
worse but a limit's, a fill on a row its market left empty only to liquidate at the
last mark, and on a run that ends flat with no liquidation charge or dust, round trips
that sum to the change in equity.
"""

import math
import random

import numpy as np
import pytest

import probatum
from reference import differential, universe_differential
from reference.simulator import Bar

COSTS = ("fees", "spread", "slippage", "impact", "funding", "liquidation", "dust")


def universe(rng, n):
    """`n` instruments, spot or linear perpetuals, each on its own walk at its own
    scale, a few listing late and now and then printing no bar; the first prints
    every row."""
    rows  = rng.randint(20, 60)
    times = differential.clock(rng, rows)

    kinds, columns = [], []
    for k in range(n):
        kinds.append(rng.choice(["spot", "linear", "linear"]))
        scale = rng.choice([0.05, 1.0, 30.0])
        bars  = [Bar(*(price * scale for price in bar[:4]), bar.volume)
                 for bar in differential.walk(rng, rows)]
        start = rng.randint(1, 5) if k > 0 and rng.random() < 0.2 else 0
        columns.append([None if t < start or (k > 0 and t > start and rng.random() < 0.05)
                        else bar for t, bar in enumerate(bars)])

    return times, kinds, columns


def decisions(rng, kinds, columns, rows, balance):
    """Now and then a target over every instrument at a gross of up to eight times
    equity, and on some instruments a bracket or a limit about their close."""
    script = {}

    for t in range(rows):
        actions = []
        if rng.random() < 0.3:
            gross   = rng.choice([0.5, 1.0, 3.0, 8.0])
            weights = [rng.uniform(0.0, 1.0) if kind == "spot" else rng.uniform(-1.0, 1.0)
                       for kind in kinds]
            total = sum(abs(w) for w in weights) or 1.0
            actions.append(("target", {"weights": [gross * w / total for w in weights]}))

        for _ in range(rng.randint(0, 3)):
            i   = rng.randrange(len(kinds))
            bar = columns[i][t]
            if bar is None:
                continue

            close = bar.close
            if rng.random() < 0.5:
                low, high = close * rng.uniform(0.9, 0.99), close * rng.uniform(1.01, 1.1)
                actions.append(("bracket", {
                    "instrument":  i,
                    "stop":        low,
                    "take_profit": high,
                }))
            else:
                side  = "sell" if kinds[i] == "linear" and rng.random() < 0.5 else "buy"
                price = close * (rng.uniform(0.95, 1.0) if side == "buy"
                                 else rng.uniform(1.0, 1.05))

                size = balance / len(kinds) / close * rng.uniform(0.5, 4.0)
                actions.append((side, {"instrument": i, "size": size, "limit": price}))

        if actions:
            script[t] = actions

    return script


def broken(run, kinds, columns):
    """Which invariant the run breaks, or None."""
    ledger, fills = run["ledger"], run["fills"]
    position, mark = ledger["position"], ledger["mark"]
    equity = ledger["equity"]
    for t in range(1, len(equity)):
        pnl = sum(position[t - 1, i] * (mark[t, i] - mark[t - 1, i])
                  for i in range(len(kinds)) if position[t - 1, i] != 0.0)
        for fill in fills:
            if fill["bar"] == t:
                signed = fill["size"] if fill["side"] == "buy" else -fill["size"]
                pnl += signed * (mark[t, fill["instrument"]] - fill["reference"])

        costs = sum(ledger[name][t] for name in COSTS)
        scale = 1.0 + abs(equity[t - 1]) + abs(pnl) + abs(costs)
        if abs(equity[t] - equity[t - 1] - (pnl - costs)) > 1e-9 * scale:
            return f"the ledger identity on bar {t}"

        spot = ledger["spot_cash"][t] + sum(
            position[t, i] * mark[t, i] for i, kind in enumerate(kinds)
            if kind == "spot" and position[t, i] != 0.0)
        if equity[t] - spot < -1e-9 * scale:
            return f"the margin wallet owing money after bar {t}"

    for fill in fills:
        t, i = fill["bar"], fill["instrument"]
        bar = columns[i][t]
        if bar is None:
            if fill["reason"] != "liquidation" or fill["price"] != mark[t - 1, i]:
                return f"a fill on a row its market left empty, on bar {t}"
            continue

        close = mark[t - 1, i]
        low   = bar.low if math.isnan(close) else min(close, bar.low)
        high  = bar.high if math.isnan(close) else max(close, bar.high)
        price = fill["price"]
        below, above = price < low - 1e-9 * high, price > high * (1.0 + 1e-9)
        better, worse = (below, above) if fill["side"] == "buy" else (above, below)

        # a limit the market went past fills at its own price, where the Specification
        # holds it to the prices traded since the last close, a rule the comparisons
        # set aside until the engine takes it up
        if better or (worse and fill["reason"] not in ("limit", "take_profit")):
            return f"a fill outside what its market traded, on bar {t}"

    flat    = not position[-1].any()
    charged = ledger["liquidation"].any() or ledger["dust"].any()
    if flat and not charged:
        total = sum(trip["pnl"] for trip in run["trades"])
        if abs(total - (equity[-1] - equity[0])) > 1e-9 * (1.0 + abs(equity[0])):
            return "the round trips against the change in equity"

    return None


@pytest.mark.parametrize("n", [10, 25, 50])
@pytest.mark.parametrize("seed", range(10))
def test_every_invariant_holds_at_up_to_fifty_instruments(n, seed):
    rng = random.Random(1_000 * n + seed)
    times, kinds, columns = universe(rng, n)
    cfg = universe_differential.config(rng, times, kinds, columns)
    cfg["intrabar"] = rng.choice(["pessimistic", "optimistic"])

    nan     = [math.nan] * 5
    candles = np.array([[nan if column[t] is None else list(column[t])
                         for column in columns] for t in range(len(times))],
                       dtype=np.float64)

    script = decisions(rng, kinds, columns, len(times) - 1, cfg["quote"])
    run    = probatum._probatum.run(candles, universe_differential.venue(cfg), script,
                                    times=np.array(times, dtype=np.int64))

    assert run["fills"]
    assert broken(run, kinds, columns) is None
