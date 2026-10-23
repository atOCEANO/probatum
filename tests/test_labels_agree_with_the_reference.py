"""Labels against the reference's on random cases, row by row: each label one trade of a
fresh account through the fill model, its exit, end and bars the same and its return
and weight within rounding.

A case is up to three instruments on an hourly clock with closures, spot, linear or
both, or an inverse contract alone, each printing from a row of its own to one of its
own with gaps, one of a universe now and then delisted after its last bar, at a venue
whose fees, fill model, intrabar setting and funding are drawn, its sizes held to a lot
and a tick a third of the time off an inverse contract, labelled on a side, at a
horizon and with barriers drawn, each barrier a number, rows of its own or none.
Every gate stage runs a short comparison, which has to see each exit but a liquidation
and a row with no label.
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

import probatum
from reference import differential
from reference.simulator import Bar, Instrument, Venue, labels as reference

START = 1_704_067_200 * 10**9   # 2024-01-01 00:00 UTC
# what a short comparison has to see, the row with no label among them
SEEN = ("stop", "take_profit", "horizon", "delist", "censored", "")


def fills(rng):
    """A fill model as the probatum Venue's ``fills`` table and the reference Venue's
    keywords."""
    budget = rng.choice([1.0, 1.0, 1e-4])
    window = rng.choice([24, 3])
    limit  = rng.choice(["touch", "through"])
    spread = rng.choice([None, None, ("fixed", 10.0), "estimated"])
    slip   = rng.choice([0.0, 5.0])
    impact = rng.choice([None, None, ("linear", 50.0), ("sqrt", 1.0)])
    table  = {
        "liquidity": {"budget": budget},
        "window":    window,
        "limit":     limit,
        "spread":    ("none" if spread is None else spread if spread == "estimated"
                      else {"fixed": spread[1]}),
        "slippage":  {"fixed": slip} if slip else "none",
        "impact":    "none" if impact is None else {impact[0]: impact[1]},
    }

    return table, dict(budget=budget, window=window, limit=limit, spread=spread,
                       slippage=slip, impact=impact)


def spans(rng, count, width):
    """Which rows each instrument prints: from a row of its own to one of its own, a gap
    now and then, the first instrument on every row the others leave bare."""
    printed = np.zeros((count, width), dtype=bool)
    for k in range(width):
        first = 0 if k == 0 else rng.randint(0, count // 3)
        last  = count - 1 if rng.random() < 0.5 else rng.randint(2 * count // 3, count - 1)
        printed[first:last + 1, k] = True
        for _ in range(rng.randint(0, 2)):
            at = rng.randint(first, last)
            printed[at:at + rng.randint(1, 6), k] = False

    printed[~printed.any(axis=1), 0] = True
    return printed


def case(rng):
    """A random case: the Data, the probatum Venue, the reference's instruments and
    Venue, and the labels' horizon, notional, side and barriers."""
    shape = rng.choice(["linear", "linear", "spot", "mixed", "inverse"])
    width = 1 if shape == "inverse" else rng.randint(1, 3)
    kinds = (["spot"] + ["linear"] * (width - 1) if shape == "mixed" and width > 1
             else [{"mixed": "linear"}.get(shape, shape)] * width)

    count   = rng.randint(12, 40)
    times   = np.array(differential.clock(rng, count), dtype=np.int64) + START
    printed = spans(rng, count, width) if shape != "inverse" else np.ones((count, 1), bool)
    bars    = np.full((count, width, 5), np.nan)
    for k in range(width):
        walked = differential.walk(rng, count)
        bars[printed[:, k], k] = np.array(walked, dtype=np.float64)[printed[:, k]]

    names = [f"C{k}USDT" for k in range(width)] if shape != "inverse" else ["BTCUSD"]
    if shape == "inverse":
        contract = rng.choice([10.0, 100.0])
        frame    = pd.DataFrame(bars[:, 0], index=pd.to_datetime(times, utc=True),
                                columns=["open", "high", "low", "close", "volume"])
        frame.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                           interval="1h", quote="USD", warnings=[],
                           volume_unit="contract", contract_size=contract)
        data = probatum.Data(frame)
    else:
        contract = None
        parts    = [probatum.Data(
            bars[:, [k]],
            symbol   = [names[k]],
            market   = kinds[k],
            time     = times,
            interval = "1h",
        ) for k in range(width)]
        data = parts[0]
        for part in parts[1:]:
            data = data.join(part)

    fees    = dict(maker=rng.choice([0.0, 0.0002]), taker=rng.choice([0.0, 0.0006, 0.002]))
    rate    = rng.choice([0.0001, 0.02, -0.01])
    funding = rng.choice([None, None, ("every", rng.choice([1.0, 3.0, 8.0]), rate)])
    table, keywords = fills(rng)

    # an instrument of a universe delisted a row or two after its last bar
    held   = data.available
    ending = [k for k in range(width) if not held[-1, k] and width > 1]
    delist = {}
    if ending and rng.random() < 0.5:
        k    = rng.choice(ending)
        last = int(np.flatnonzero(held[:, k])[-1])
        at   = int(data.time[min(last + rng.randint(1, 2), len(data) - 1)])
        delist[k] = at

    kind     = None if len(set(kinds)) > 1 else kinds[0]
    keyed    = [names[k] if kind else f"{kinds[k]}:{names[k]}" for k in range(width)]
    intrabar = rng.choice(["pessimistic", "optimistic"])
    sized    = shape != "inverse" and rng.random() < 1.0 / 3.0
    rules    = dict(lot=0.01, tick=0.01) if sized else {}
    venue    = probatum.Venue(
        kind        = kind,
        fees        = fees,
        fills       = {**table, "sizes": "on" if sized else "off"},
        **rules,
        intrabar    = intrabar,
        funding     = ("none" if funding is None else
                       {"every": funding[1], "rate": funding[2]}),
        instruments = {keyed[k]: {"delist": str(np.datetime64(at, "ns"))[:16]}
                       for k, at in delist.items()} or None,
    )

    theirs = Venue(
        maker    = fees["maker"],
        taker    = fees["taker"],
        intrabar = intrabar,
        funding  = "none" if funding is None else funding,
        sizes    = sized,
        **keywords,
    )

    instruments = [Instrument(
        names[k],
        kind     = kinds[k],
        contract = contract,
        delist   = delist.get(k),
        **rules,
    ) for k in range(width)]

    horizon  = rng.choice([1, 2, 3, 5, 8])
    notional = 100.0 if shape == "inverse" else rng.choice([1_000.0, 10_000.0])
    side     = "long" if "spot" in kinds or rng.random() < 0.5 else "short"

    def barrier():
        u = rng.random()
        if u < 0.3:
            return None
        if u < 0.65:
            return rng.choice([0.01, 0.03, 0.08])
        return np.array([[math.nan if rng.random() < 0.2 else rng.choice([0.01, 0.03])
                          for _ in range(width)] for _ in range(len(data))])

    return data, venue, instruments, theirs, horizon, notional, side, barrier(), barrier()


def differ(mine, theirs, data):
    """Where two sets of labels first differ, or None."""
    for t in range(len(data)):
        for k in range(len(data.names)):
            ours = (mine.exit[t, k], int(mine.end[t, k]), int(mine.bars[t, k]))
            got  = (theirs["exit"][t][k], theirs["end"][t][k], theirs["bars"][t][k])
            if ours != got:
                return f"row {t}, instrument {k}: {ours} against {got}"
            for field in ("ret", "weight"):
                a, b = float(getattr(mine, field)[t, k]), theirs[field][t][k]
                if not differential.close_enough(a, b):
                    return f"row {t}, instrument {k}: {field} {a!r} against {b!r}"

    return None


def compare(cases, seed):
    """`cases` random cases through both: the disagreements, and every exit seen."""
    rng = random.Random(f"labels {seed}")
    disagreed, seen = [], set()
    for number in range(cases):
        data, venue, instruments, theirs, horizon, notional, side, stop, take_profit = (
            case(rng))

        try:
            mine = probatum.labels(data, venue, horizon, notional, stop=stop,
                                   take_profit=take_profit, side=side)
        except Exception as exc:
            disagreed.append((number, f"raised: {type(exc).__name__}: {exc}"))
            continue

        bars = [[None if np.isnan(row[k, 3]) else Bar(*map(float, row[k]))
                 for k in range(len(data.names))] for row in data.bars]
        read = [barrier if barrier is None or isinstance(barrier, float) else
                barrier.tolist() for barrier in (stop, take_profit)]
        got = reference(data.time.tolist(), bars, instruments, theirs, horizon, notional,
                        *read, side=side, interval=differential.HOUR)

        found = differ(mine, got, data)
        if found is not None:
            disagreed.append((number, found))

        seen.update(mine.exit.ravel().tolist())

    return disagreed, seen


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_random_labels_agree_with_the_reference_on_every_row(seed):
    disagreed, seen = compare(40, seed)

    assert not disagreed, "; ".join(f"case {number}: {why}"
                                    for number, why in disagreed[:5])
    assert not [exit for exit in SEEN if exit not in seen]
