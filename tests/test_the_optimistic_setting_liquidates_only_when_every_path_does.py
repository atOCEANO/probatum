"""Under intrabar="optimistic" an account of several perps is liquidated on a bar only
if every path through the bars would have reached its trigger, which the engine
decides at each perp's corner and at the closes (Specification, The path). These
tests hold its decisions to paths of their own. On a bar where it liquidated, the
path the Specification builds, from the opens out to every favourable extreme, to
each adverse extreme and back in turn, then to the closes, must reach the trigger,
and so must paths drawn at random through the bars; on a bar where it did not, the
built path must never reach it. Equity less the requirement is concave along a
straight move of the prices, so a path that moves straight between its corners is
judged at its corners alone. Funding stamped inside a bar is taken the way most
favourable to the account, as the optimistic setting takes it: a credit from the
open and a debit at the close, where every path has paid it."""

import random

import numpy as np
import pytest

import probatum

HOUR = 3_600 * 1_000_000_000


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


def above(wallet, legs, prices):
    """How far margin equity on `wallet` sits above the maintenance at `prices`."""
    value = wallet
    for (q, entry, tiers), price in zip(legs, prices):
        value += q * (price - entry) - requirement(tiers, abs(q) * price)

    return value


def lowest(wallets, legs, path):
    """The least the account stands above its trigger at the corners of `path`: the
    last marks and the opens on the wallet the bar opened with, the corners inside
    the bar with the funding it counts, and the closes once all of it is charged."""
    opened, inside, closed = wallets
    held = [opened, opened] + [inside] * (len(path) - 3) + [closed]
    return min(above(wallet, legs, prices) for wallet, prices in zip(held, path))


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


def corners(bars, legs):
    """The corners of the Specification's path through `bars`: every leg at its
    favourable extreme, then each leg in turn at its adverse extreme with the others
    still at their favourable."""
    favourable = [high if q > 0.0 else low
                  for (q, _, _), (_, high, low, _) in zip(legs, bars)]

    adverse = [low if q > 0.0 else high
               for (q, _, _), (_, high, low, _) in zip(legs, bars)]

    return [favourable] + [favourable[:k] + [adverse[k]] + favourable[k + 1:]
                           for k in range(len(legs))]


def universe(rng):
    """A universe of two to four perps on hourly bars, an account opening a position
    in each on its first bar at a few times its margin, funding stamped at the half
    hour inside some of the bars, and the engine's run of it."""
    width   = rng.randint(2, 4)
    quote   = 10_000.0
    candles = walk(rng, 24, width)
    tiers   = [table(rng, quote) for _ in range(width)]
    stamped = sorted(rng.sample(range(1, 24), 8))
    funding = {t: rng.uniform(-0.002, 0.002) for t in stamped}
    script  = {0: [(rng.choice(["buy", "sell"]),
                    {"size": quote * rng.uniform(1.0, 5.0) / candles[0, i, 3],
                     "instrument": i})
                   for i in range(width)]}

    venue = dict(market="perp", quote=quote, fee_taker=0.0005, fee_maker=0.0002,
                 leverage=0.0, intrabar="optimistic",
                 funding_times=np.array([t * HOUR + HOUR // 2 for t in stamped]),
                 funding_rates=np.array([funding[t] for t in stamped]),
                 instruments=[{"market": "perp", "tiers": rows} if rows else
                              {"market": "perp"} for rows in tiers])

    times = np.arange(len(candles), dtype=np.int64) * HOUR
    return candles, tiers, funding, probatum._probatum.run(candles, venue, script,
                                                           times)


def judged(seed, universes):
    """Every bar the account carried perps into and traded nothing on, judged by the
    paths: how many it was liquidated on at a corner, opens and closes clear, and
    how many it outlived although every perp at its adverse extreme at once would
    have reached the trigger."""
    rng = random.Random(f"every path {seed}")

    cornered = outlived = 0
    for _ in range(universes):
        candles, tiers, funding, out = universe(rng)
        ledger = out["ledger"]
        closed = {fill["bar"] for fill in out["fills"] if fill["reason"] == "liquidation"}
        traded = {fill["bar"] for fill in out["fills"] if fill["reason"] != "liquidation"}

        for t in range(1, len(candles)):
            held = [i for i in range(candles.shape[1]) if ledger["position"][t - 1][i]]
            if t in traded or not held:
                continue

            legs = [(ledger["position"][t - 1][i], ledger["entry"][t - 1][i], tiers[i])
                    for i in held]
            bars    = [candles[t, i, :4] for i in held]
            marks   = [ledger["mark"][t - 1][i] for i in held]
            opens   = [bar[0] for bar in bars]
            closes  = [bar[3] for bar in bars]
            charges = [q * close * funding.get(t, 0.0)
                       for (q, _, _), close in zip(legs, closes)]
            wallet  = ledger["margin_cash"][t - 1]
            wallets = (wallet, wallet - sum(c for c in charges if c < 0.0),
                       wallet - sum(charges))

            built = [marks, opens, *corners(bars, legs), closes]
            scale = abs(wallet) + sum(abs(q) * mark for (q, _, _), mark
                                      in zip(legs, marks))
            tolerance = 1e-9 * scale

            if t in closed:
                assert lowest(wallets, legs, built) <= tolerance, (
                    f"seed {seed}: bar {t} liquidated where the built path never "
                    f"reaches the trigger: {built}")
                for _ in range(32):
                    path = [marks, *drawn(rng, bars)]
                    assert lowest(wallets, legs, path) <= tolerance, (
                        f"seed {seed}: bar {t} liquidated where a path never reaches "
                        f"the trigger: {path}")

                cornered += (above(wallet, legs, opens) > 0.0
                             and above(wallets[2], legs, closes) > 0.0)
            else:
                least = lowest(wallets, legs, built)
                assert least > -tolerance, (
                    f"seed {seed}: bar {t} outlived where the built path reaches the "
                    f"trigger, {least} below it: {built}")

                adverse = [low if q > 0.0 else high
                           for (q, _, _), (_, high, low, _) in zip(legs, bars)]
                outlived += above(wallets[1], legs, adverse) <= 0.0

    return cornered, outlived


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_a_bar_liquidates_only_where_every_path_reaches_the_trigger(seed):
    cornered, outlived = judged(seed, 150)

    assert cornered and outlived, (cornered, outlived)
