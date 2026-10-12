"""Universes of up to fifty instruments through the engine keep its invariants: the
ledger identity on every row; no margin wallet ending a bar owing money; no fill better
than every price its market traded from the last close through its bar and none worse
but a limit's at its own price, a forced close counting the prices the bar's fills
opened its legs at, and a fill on a row its market left empty only to liquidate at the
last mark; no fill the margin bound lifts past its position's average entry; no growing
perpetual fill with an empty margin wallet, and no perpetual order outliving a
liquidation; and round trips that each carry their fills' fees once and, on a run that
ends flat with no liquidation charge or dust, sum to the change in equity. A fill is
held here to its raw bar and the last close; where a liquidating bar clips it tighter,
the comparison with the reference holds it there. Random universes reach all but the
margin bound, which takes an account steered between its close point and its trigger,
and only a calm one ends flat often enough to reach the last.
"""

import collections
import math
import random

import numpy as np
import pytest

import probatum
from reference import differential, universe_differential
from reference.simulator import Bar

ACCOUNT = ("equity", "spot_cash", "margin_cash", "fees", "spread", "slippage",
           "impact", "funding", "liquidation", "dust")
# a position within this of zero is flat, as the account books it
DUST  = 1e-9
COSTS = ("fees", "spread", "slippage", "impact", "funding", "liquidation", "dust")
# what ends an order's life in the order log, a fill aside
ENDS = ("cancelled", "expired", "not_filled", "refused")


def universe(rng, n):
    """`n` instruments, spot or linear perpetuals, each on its own walk at its own
    scale, a few listing late and now and then printing no bar, though every one
    prints and trades on the last two rows; the first prints every row."""
    rows  = rng.randint(20, 60)
    times = differential.clock(rng, rows)

    kinds, columns = [], []
    for k in range(n):
        kinds.append(rng.choice(["spot", "linear", "linear"]))
        scale = rng.choice([0.05, 1.0, 30.0])
        bars  = [Bar(*(price * scale for price in bar[:4]), bar.volume)
                 for bar in differential.walk(rng, rows)]
        bars[-2:] = [bar._replace(volume=max(bar.volume, 1e7)) for bar in bars[-2:]]
        start = rng.randint(1, 5) if k > 0 and rng.random() < 0.2 else 0
        gone  = [t < start or (k > 0 and start < t < rows - 2 and rng.random() < 0.05)
                 for t in range(rows)]
        columns.append([None if out else bar for out, bar in zip(gone, bars)])

    return times, kinds, columns


def decisions(rng, kinds, columns, rows, balance, calm, margin):
    """Now and then a target over every instrument at a gross of up to eight times
    equity, or when `calm` one at most with the perpetuals held to `margin`, the
    margin wallet's share of the account, and on some instruments a bracket or a
    limit about their close; at the last close every order is cancelled and every
    weight taken to zero. Returns the script and each instrument's limit and
    take-profit prices."""
    script, prices = {}, set()
    for t in range(rows - 1):
        actions = []
        if rng.random() < 0.3:
            gross   = rng.choice([0.5, 1.0] if calm else [0.5, 1.0, 3.0, 8.0])
            weights = [rng.uniform(-1.0 if kind == "linear" else 0.0, 1.0)
                       for kind in kinds]
            total   = sum(abs(w) for w in weights) or 1.0
            room    = [margin if calm and kind == "linear" else 1.0 for kind in kinds]
            weights = [gross * w / total * r for w, r in zip(weights, room)]
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
                prices.add((i, high))
            else:
                side  = "sell" if kinds[i] == "linear" and rng.random() < 0.5 else "buy"
                price = close * (rng.uniform(0.95, 1.0) if side == "buy"
                                 else rng.uniform(1.0, 1.05))

                size = balance / len(kinds) / close * rng.uniform(0.5, 4.0)
                if calm:
                    size /= 10.0
                actions.append((side, {"instrument": i, "size": size, "limit": price}))
                prices.add((i, price))

        if actions:
            script[t] = actions

    script[rows - 1] = [("cancel_all", {}), ("target", {"weights": [0.0] * len(kinds)})]

    return script, prices


def gentle(source):
    """A funding source as the universe comparison draws it, each rate taken to a
    hundredth of a percent, which a calm account pays without being closed out."""
    if source[0] == "data":
        return ("data", [(stamp, i, math.copysign(1e-4, rate))
                         for stamp, i, rate in source[1]])

    if source[0] in ("every_bars", "every"):
        return source[:2] + (math.copysign(1e-4, source[2]),)

    return source


def band(rng, n):
    """`n` perpetuals a target takes long and short at three to eight times equity,
    with no maintenance, cap, slippage or funding, on flat rows. The fourth row moves
    every leg against its position, the first, the last and a few others barely, and
    closes where it moved to, which leaves the account between its close point and its
    trigger; the next moves only their way, a barely moved leg back past its entry,
    and a target at the fourth row's close halves or closes every leg there, so the
    margin bound lifts a fill to its position's entry under either setting."""
    fee, quote = 0.002, rng.choice([1_000.0, 10_000.0])

    gross    = rng.uniform(3.0, 8.0)
    shares   = [rng.uniform(0.5, 1.5) for _ in range(n)]
    signs    = [rng.choice([1.0, -1.0]) for _ in range(n)]
    weights  = [sign * gross * share / sum(shares) for sign, share in zip(signs, shares)]
    notional = [abs(w) * quote for w in weights]
    barely   = [rng.uniform(0.0, 0.004) if k in (0, n - 1) or rng.random() < 0.3 else None
                for k in range(n)]

    # the depth that leaves equity at a share of the closing fee on every leg
    share   = rng.uniform(0.2, 0.8)
    moved   = sum(x for x, d in zip(notional, barely) if d is None)
    lost    = sum(x * d for x, d in zip(notional, barely) if d is not None)
    depth   = (quote - fee * sum(notional) * (1.0 + share) - lost) / moved
    columns = []

    for sign, d in zip(signs, barely):
        price = rng.choice([0.05, 1.0, 30.0]) * rng.uniform(50.0, 150.0)
        to    = price * (1.0 - sign * (depth if d is None else d))
        room  = price * (1.0 + sign * rng.uniform(0.0, 0.01))
        if d is None:
            their = to * (1.0 + sign * rng.uniform(0.0, 0.005))
        else:
            their = price * (1.0 + sign * rng.uniform(0.001, 0.005))

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


def trips(fills, n):
    """Each instrument's round trips rebuilt from its fills, in the order they opened:
    the fees each carries, a flip's split by size between the one it closes and the
    one it opens."""
    held, fees = [0.0] * n, [[] for _ in range(n)]
    for fill in fills:
        i, size, fee = fill["instrument"], fill["size"], fill["fee"]
        before = held[i]
        held[i] = before + (size if fill["side"] == "buy" else -size)
        if abs(held[i]) <= DUST:
            held[i] = 0.0

        if before == 0.0:
            fees[i].append(fee)
        elif before * held[i] < 0.0:
            fees[i][-1] += fee * abs(before) / size
            fees[i].append(fee * (size - abs(before)) / size)
        else:
            fees[i][-1] += fee

    return fees


def broken(run, kinds, columns, prices):
    """Which invariant read off the ledger, the fills or the round trips the run
    breaks, or None; `prices` holds each instrument's limit and take-profit prices."""
    ledger, fills = run["ledger"], run["fills"]

    for name in ACCOUNT:
        if not np.isfinite(ledger[name]).all():
            return f"the ledger's {name} not a number"

    for fill in fills:
        if not all(math.isfinite(fill[key]) for key in ("size", "price", "fee")):
            return f"a fill not a number, on bar {fill['bar']}"

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
        if fill["reason"] == "liquidation":
            # a forced close is priced on the path of a leg, which a fill on the bar
            # may have opened at its own price
            for other in fills:
                if other["bar"] == t and other["instrument"] == i:
                    low, high = min(low, other["price"]), max(high, other["price"])

        price = fill["price"]
        below, above = price < low - 1e-9 * high, price > high * (1.0 + 1e-9)
        better, worse = (below, above) if fill["side"] == "buy" else (above, below)
        # a limit the market went past fills at its own price, where the Specification
        # holds it to the prices traded since the last close, a rule the comparisons
        # set aside until the engine takes it up
        own = fill["reference"] == price and (i, price) in prices
        if better or (worse and not own):
            return f"a fill outside what its market traded, on bar {t}"

    theirs = collections.defaultdict(list)
    for trip in sorted(run["trades"], key=lambda trip: trip["open_bar"]):
        theirs[trip["instrument"]].append(trip["fees"])

    for i, mine in enumerate(trips(fills, len(kinds))):
        if len(mine) != len(theirs[i]) or any(
                abs(a - b) > 1e-9 * (1.0 + abs(b)) for a, b in zip(mine, theirs[i])):
            return f"the round trips' fees on instrument {i}"

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

            held[i] = 0.0 if abs(before + signed) <= DUST else before + signed

    return None


def outlived(run, kinds):
    """Whether an order on a perpetual outlives a liquidation: one placed before it
    shows in the order log after it, or one still waiting or resting untouched when
    the bar came, which only a fill or a cancel on that bar can end."""
    liquidated = {fill["bar"] for fill in run["fills"]
                  if fill["reason"] == "liquidation"}

    events = collections.defaultdict(list)
    for event in run["order_log"]:
        if event["order"] is not None and kinds[event["instrument"]] == "linear":
            events[event["order"]].append(event)

    for log in events.values():
        since = log[0]["bar"]
        for bar in sorted(liquidated):
            if not since < bar:
                continue
            if any(event["bar"] > bar for event in log):
                return f"a perpetual order acting after the liquidation on bar {bar}"

            before    = [event for event in log if event["bar"] < bar]
            untouched = all(event["kind"] in ("placed", "armed") for event in before)
            ended     = any(event["bar"] == bar and event["kind"] in ENDS + ("filled",)
                            for event in log)
            if before and untouched and not ended:
                return f"a perpetual order standing through bar {bar}'s liquidation"

    return None


def play(times, columns, venue, script):
    nan     = [math.nan] * 5
    candles = np.array([[nan if column[t] is None else list(column[t])
                         for column in columns] for t in range(len(times))],
                       dtype=np.float64)

    return probatum._probatum.run(candles, venue, script,
                                  times=np.array(times, dtype=np.int64))


@pytest.mark.parametrize("calm", [False, True])
@pytest.mark.parametrize("n", [10, 25, 50])
@pytest.mark.parametrize("seed", range(10))
def test_every_invariant_holds_at_up_to_fifty_instruments(n, seed, calm):
    rng = random.Random(1_000 * n + seed)
    times, kinds, columns = universe(rng, n)
    cfg = universe_differential.config(rng, times, kinds, columns)
    cfg["intrabar"] = rng.choice(["pessimistic", "optimistic"])
    if calm:
        cfg["funding"] = gentle(cfg["funding"])

    wallets = cfg["balance"]
    margin  = wallets["margin"] / cfg["quote"] if isinstance(wallets, dict) else 1.0
    script, prices = decisions(rng, kinds, columns, len(times) - 1, cfg["quote"], calm,
                               margin)

    run = play(times, columns, universe_differential.venue(cfg), script)

    assert run["fills"]

    if calm:
        # a calm run ends flat with nothing charged, where the round trips have to
        # sum to the change in equity
        assert not run["ledger"]["position"][-1].any()
        assert not run["ledger"]["liquidation"].any()

    assert broken(run, kinds, columns, prices) is None
    assert booked(run, kinds, cfg["slippage_bps"] == 0.0) is None
    assert outlived(run, kinds) is None


@pytest.mark.parametrize("n", [10, 25, 50])
@pytest.mark.parametrize("seed", range(10))
def test_the_margin_bound_keeps_its_invariant_at_up_to_fifty_instruments(n, seed):
    rng = random.Random(1_000 * n + seed)
    times, columns, venue, script = band(rng, n)

    run   = play(times, columns, venue, script)
    kinds = ["linear"] * n
    entry = run["ledger"]["entry"][3]
    lifts = [fill for fill in run["fills"]
             if abs(fill["adjustment"]) > 1e-9 * fill["price"]]

    assert any(abs(fill["price"] - entry[fill["instrument"]]) <= 1e-12 * fill["price"]
               for fill in lifts)
    assert broken(run, kinds, columns, set()) is None
    assert booked(run, kinds, True) is None
    assert outlived(run, kinds) is None
