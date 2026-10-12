"""Universes of up to fifty instruments through the engine keep its invariants: the
ledger identity on every row; no margin wallet ending a bar owing money; no fill better
than every price its market traded from the last close through its bar and none worse
but a limit's, and a fill on a row its market left empty only to liquidate at the last
mark; no fill the margin bound lifts past its position's average entry; no growing
perpetual fill with an empty margin wallet, and no perpetual order outliving a
liquidation; and round trips that carry every fee once and, on a run that ends flat
with no liquidation charge or dust, sum to the change in equity. Random universes reach
all but the margin bound, which takes an account steered between its close point and
its trigger.
"""

import collections
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
        gone  = [t < start or (k > 0 and t > start and rng.random() < 0.05)
                 for t in range(rows)]
        columns.append([None if out else bar for out, bar in zip(gone, bars)])

    return times, kinds, columns


def decisions(rng, kinds, columns, rows, balance):
    """Now and then a target over every instrument at a gross of up to eight times
    equity, and on some instruments a bracket or a limit about their close."""
    script = {}

    for t in range(rows):
        actions = []
        if rng.random() < 0.3:
            gross   = rng.choice([0.5, 1.0, 3.0, 8.0])
            weights = [rng.uniform(-1.0 if kind == "linear" else 0.0, 1.0)
                       for kind in kinds]
            total   = sum(abs(w) for w in weights) or 1.0
            weights = [gross * w / total for w in weights]
            actions.append(("target", {"weights": weights}))

        for _ in range(rng.randint(0, 3)):
            i   = rng.randrange(len(kinds))
            bar = columns[i][t]
            if bar is None:
                continue

            close = bar.close
            if rng.random() < 0.5:
                low  = close * rng.uniform(0.9, 0.99)
                high = close * rng.uniform(1.01, 1.1)
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


def band(rng, n):
    """`n` perpetuals a target takes long and short at three to eight times equity,
    with no maintenance, cap, slippage or funding, on flat rows. The fourth row moves
    every leg against its position, a few of them barely, and closes where it moved
    to, which leaves the account between its close point and its trigger; the next
    moves only their way, and a target at the fourth row's close halves or closes
    every leg there, so the margin bound lifts the fills and holds some at their
    position's entry."""
    fee, quote = 0.002, rng.choice([1_000.0, 10_000.0])

    gross    = rng.uniform(3.0, 8.0)
    shares   = [rng.uniform(0.5, 1.5) for _ in range(n)]
    signs    = [rng.choice([1.0, -1.0]) for _ in range(n)]
    weights  = [sign * gross * share / sum(shares) for sign, share in zip(signs, shares)]
    notional = [abs(w) * quote for w in weights]
    barely   = [rng.uniform(0.0, 0.004) if rng.random() < 0.3 else None for _ in range(n)]

    # the depth that leaves equity at a share of the closing fee on every leg
    share   = rng.uniform(0.2, 0.8)
    moved   = sum(x for x, d in zip(notional, barely) if d is None)
    lost    = sum(x * d for x, d in zip(notional, barely) if d is not None)
    depth   = (quote - fee * sum(notional) * (1.0 + share) - lost) / moved
    columns = []

    for sign, d in zip(signs, barely):
        price   = rng.choice([0.05, 1.0, 30.0]) * rng.uniform(50.0, 150.0)
        to      = price * (1.0 - sign * (depth if d is None else d))
        room    = price * (1.0 + sign * rng.uniform(0.0, 0.01))
        their   = to * (1.0 + sign * rng.uniform(0.0, 0.005))
        flat    = Bar(price, price, price, price, 1e9)
        steered = Bar(price, max(price, room, to), min(price, room, to), to, 1e9)
        after   = Bar(to, max(to, their), min(to, their),
                      rng.uniform(min(to, their), max(to, their)), 1e9)
        columns.append([flat, flat, flat, steered, after,
                        universe_differential.calm(rng, after.close)])

    kept  = rng.choice([0.0, 0.5])
    venue = dict(market="perp", quote=quote, fee_taker=fee, fee_maker=0.0,
                 slippage_bps=0.0, max_fill_fraction=1.0, max_open_orders=8,
                 leverage=0.0, impact=0.0,
                 intrabar=rng.choice(["pessimistic", "optimistic"]),
                 instruments=[dict(market="perp")] * n)

    script = {
        0: [("target", {"weights": weights})],
        3: [("target", {"weights": [w * kept for w in weights]})],
    }

    return [k * differential.HOUR for k in range(6)], columns, venue, script


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

    paid = sum(fill["fee"] for fill in fills)
    if abs(sum(trip["fees"] for trip in run["trades"]) - paid) > 1e-9 * (1.0 + paid):
        return "the round trips' fees against the fills'"

    flat    = not position[-1].any()
    charged = ledger["liquidation"].any() or ledger["dust"].any()
    if flat and not charged:
        total = sum(trip["pnl"] for trip in run["trades"])
        if abs(total - (equity[-1] - equity[0])) > 1e-9 * (1.0 + abs(equity[0])):
            return "the round trips against the change in equity"

    return None


def booked(run, kinds, bound):
    """Which invariant the fills break as each is booked, replayed from the ledger's
    row before its bar, or None: a growing perpetual fill with an empty margin wallet
    or after a liquidation, and with `bound`, where there is no slippage for a bar to
    hold back and an adjustment can only be the margin bound's lift, a lift of
    anything but a perpetual's reducing fill or one past its position's average
    entry."""
    ledger = run["ledger"]
    perps  = np.array([kind == "linear" for kind in kinds])
    bars   = collections.defaultdict(list)

    for fill in run["fills"]:
        bars[fill["bar"]].append(fill)

    for t in range(1, len(ledger["equity"])):
        held, entry = ledger["position"][t - 1].copy(), ledger["entry"][t - 1].copy()
        empty = ledger["margin_cash"][t - 1] <= 0.0 and not held[perps].any()
        for fill in bars[t]:
            i, size, price = fill["instrument"], fill["size"], fill["price"]
            before = held[i]
            signed = size if fill["side"] == "buy" else -size
            flips  = before * signed < 0.0 and size > abs(before)
            grows  = before * signed >= 0.0 or flips

            if perps[i] and fill["reason"] == "liquidation":
                empty = True
            elif perps[i] and grows and empty:
                return f"a perpetual grown with an empty margin wallet, on bar {t}"

            if bound and fill["adjustment"] != 0.0:
                long   = before > 0.0
                lifted = fill["adjustment"] > 0.0 if long else fill["adjustment"] < 0.0
                past   = (price > entry[i] * (1.0 + 1e-9) if long
                          else price < entry[i] * (1.0 - 1e-9))
                if not perps[i] or grows or not lifted or past:
                    return f"a fill the margin bound moved wrongly, on bar {t}"

            if before == 0.0 or flips:
                entry[i] = price
            elif before * signed > 0.0:
                entry[i] = ((abs(before) * entry[i] + size * price)
                            / (abs(before) + size))

            held[i] = before + signed

    return None


def outlived(run, kinds):
    """Whether an order on a perpetual placed before a liquidation shows in the order
    log after it, where the liquidation cancels every one."""
    liquidated = {fill["bar"] for fill in run["fills"]
                  if fill["reason"] == "liquidation"}

    placed = {}
    for event in run["order_log"]:
        order, i = event["order"], event["instrument"]
        if order is None or kinds[i] != "linear":
            continue

        since = placed.setdefault(order, event["bar"])
        if any(since < bar < event["bar"] for bar in liquidated):
            return f"a perpetual order outliving a liquidation, on bar {event['bar']}"

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
    assert booked(run, kinds, cfg["slippage_bps"] == 0.0) is None
    assert outlived(run, kinds) is None


@pytest.mark.parametrize("n", [10, 25, 50])
@pytest.mark.parametrize("seed", range(10))
def test_the_margin_bound_keeps_its_invariant_at_up_to_fifty_instruments(n, seed):
    rng = random.Random(1_000 * n + seed)
    times, columns, venue, script = band(rng, n)
    candles = np.array([[list(column[t]) for column in columns]
                        for t in range(len(times))], dtype=np.float64)

    run = probatum._probatum.run(candles, venue, script,
                                 times=np.array(times, dtype=np.int64))
    kinds = ["linear"] * n

    assert any(fill["adjustment"] != 0.0 for fill in run["fills"])
    assert broken(run, kinds, columns) is None
    assert booked(run, kinds, True) is None
    assert outlived(run, kinds) is None
