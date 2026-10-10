"""Under intrabar="optimistic" an account of several perps is liquidated on a bar only
if every path through the bars would have reached its trigger, which the engine
decides at each perp's corner and at the closes (Specification, The path). These
tests hold its decisions to paths of their own. On a bar where it liquidated, paths
drawn at random through the bars must every one reach the trigger; on a bar where it
did not, the path the Specification builds, from the opens out to every favourable
extreme, to each adverse extreme and back in turn, then to the closes, must never
reach it. Equity less the requirement is concave along a straight move of the prices,
so a path that moves straight between its corners is judged at its corners alone."""

import random

import numpy as np
import pytest

import probatum


def walk(rng, rows, width):
    """`rows` wide bars of `width` instruments, each opening near its last close."""
    candles = np.empty((rows, width, 5))
    for i in range(width):
        close = rng.uniform(50.0, 5_000.0)
        for t in range(rows):
            open_ = close * rng.uniform(0.99, 1.01)
            close = open_ * rng.uniform(0.92, 1.08)
            high  = max(open_, close) * rng.uniform(1.0, 1.06)
            low   = min(open_, close) * rng.uniform(0.94, 1.0)
            candles[t, i] = open_, high, low, close, 1e12

    return candles


def table(rng, quote):
    """No table, or three notional tiers of rising maintenance and no cap, each
    amount continuous across the bound below it."""
    if rng.random() < 0.3:
        return None

    bounds  = [quote * b for b in sorted(rng.sample([1.0, 2.0, 4.0, 8.0], 3))]
    rates   = sorted(rng.sample([0.005, 0.01, 0.02, 0.04], 3))
    amounts = [0.0]
    for k in (1, 2):
        amounts.append(amounts[-1] + (rates[k] - rates[k - 1]) * bounds[k - 1])

    return [[up_to, 0.0, rate, amount]
            for up_to, rate, amount in zip(bounds, rates, amounts)]


def requirement(tiers, notional):
    """The maintenance a position of `notional` keeps, in the tier its notional falls
    in, the last tier's past every bound."""
    if tiers is None:
        return 0.0

    tier = next((row for row in tiers if notional <= row[0]), tiers[-1])
    return notional * tier[2] - tier[3]


def above(account, prices):
    """How far the account's margin equity sits above its maintenance at `prices`."""
    wallet, legs = account
    value = wallet
    for (q, entry, tiers), price in zip(legs, prices):
        value += q * (price - entry) - requirement(tiers, abs(q) * price)

    return value


def drawn(rng, bars):
    """A path drawn at random through `bars`, as the prices at each of its corners:
    every instrument from its open to its high and its low in a random order, at
    random moments, then to its close, moving straight between them."""
    turns = []
    for open_, high, low, close in bars:
        first, second = sorted([rng.random(), rng.random()])
        far = [high, low] if rng.random() < 0.5 else [low, high]
        turns.append([(0.0, open_), (first, far[0]), (second, far[1]), (1.0, close)])

    moments = sorted({moment for points in turns for moment, _ in points})
    return [[np.interp(moment, *zip(*points)) for points in turns]
            for moment in moments]


def built(bars, legs):
    """The Specification's path through `bars`, as the prices at its corners: the
    opens, every leg at its favourable extreme, each leg in turn at its adverse
    extreme with the others still at their favourable, and the closes."""
    favourable = [high if q > 0.0 else low
                  for (q, _, _), (_, high, low, _) in zip(legs, bars)]
    adverse = [low if q > 0.0 else high
               for (q, _, _), (_, high, low, _) in zip(legs, bars)]
    corners = [favourable[:k] + [adverse[k]] + favourable[k + 1:]
               for k in range(len(legs))]

    return [[bar[0] for bar in bars], favourable, *corners, [bar[3] for bar in bars]]


def universe(rng):
    """A universe of two to four perps, an account opening a position in each on its
    first bar at a few times its margin, and the engine's run of it."""
    width   = rng.randint(2, 4)
    quote   = 10_000.0
    candles = walk(rng, 24, width)
    tiers   = [table(rng, quote) for _ in range(width)]
    script  = {0: [(rng.choice(["buy", "sell"]),
                    {"size": quote * rng.uniform(1.0, 5.0) / candles[0, i, 3],
                     "instrument": i})
                   for i in range(width)]}

    venue = dict(market="perp", quote=quote, fee_taker=0.0005, fee_maker=0.0002,
                 leverage=0.0, intrabar="optimistic",
                 instruments=[{"market": "perp", "tiers": rows} if rows else
                              {"market": "perp"} for rows in tiers])

    return candles, tiers, probatum._probatum.run(candles, venue, script)


def judged(seed, universes):
    """Every bar the account carried perps into and traded nothing on, judged by the
    paths: how many it was liquidated on, and how many it outlived although every
    perp at its adverse extreme at once would have reached the trigger."""
    rng = random.Random(f"every path {seed}")

    liquidated = outlived = 0

    for _ in range(universes):
        candles, tiers, out = universe(rng)
        ledger = out["ledger"]
        closed = {fill["bar"] for fill in out["fills"] if fill["reason"] == "liquidation"}
        traded = {fill["bar"] for fill in out["fills"]} - closed

        for t in range(1, len(candles)):
            held = [i for i in range(candles.shape[1]) if ledger["position"][t - 1][i]]
            if t in traded or not held:
                continue

            legs = [(ledger["position"][t - 1][i], ledger["entry"][t - 1][i], tiers[i])
                    for i in held]
            account = ledger["margin_cash"][t - 1], legs
            bars    = [candles[t, i, :4] for i in held]
            marks   = [ledger["mark"][t - 1][i] for i in held]
            scale   = abs(account[0]) + sum(abs(q) * mark for (q, _, _), mark
                                            in zip(legs, marks))
            tolerance = 1e-9 * scale

            if t in closed:
                liquidated += 1
                for _ in range(32):
                    path = [marks, *drawn(rng, bars)]
                    assert min(above(account, prices) for prices in path) <= tolerance, (
                        f"seed {seed}: bar {t} liquidated where a path never reaches "
                        f"the trigger: {path}")
            else:
                path   = [marks, *built(bars, legs)]
                lowest = min(above(account, prices) for prices in path)
                assert lowest > -tolerance, (
                    f"seed {seed}: bar {t} outlived where the built path reaches the "
                    f"trigger, {lowest} below it: {path}")

                adverse = [low if q > 0.0 else high
                           for (q, _, _), (_, high, low, _) in zip(legs, bars)]
                outlived += above(account, adverse) <= 0.0

    return liquidated, outlived


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_a_bar_liquidates_only_where_every_path_reaches_the_trigger(seed):
    liquidated, outlived = judged(seed, 150)

    assert liquidated and outlived, (liquidated, outlived)
