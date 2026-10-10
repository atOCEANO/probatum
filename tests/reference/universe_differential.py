"""Run the engine and the reference over the same random universes and diff them.

The comparison `differential` makes at one instrument, over universes of two to five
instruments, spot and linear perpetuals, on one clock. An instrument other than the
first lists a row or a few late now and then and prints no bar on some rows, so its
orders wait and its position keeps its mark, and the perpetuals share one margin
wallet, so the cap, the margin bound and funding run across them. Every fourth
universe is drawn around a shape random ones rarely reach, a stop beside another
perp that moves against both or a limit filled on its gap, and every eighth around a
take-profit, on a bar that may liquidate the account or with the account between its
close point and its trigger. A case is compared up to the bar on which the reference
first applied a rule the engine does not have yet, as `differential` cuts its own.

  python -m reference.universe_differential [cases] [seed]     (from tests/)
"""

import collections
import math
import random
import sys

import numpy as np

import probatum
from .differential import (KINDS, Script, before, clock, close_enough,
                           closed_before, cut, limit_price, live, log_difference,
                           resting, setting, settings, stamps, stop_price, tally,
                           tier_table, walk, weight, worded)
from .simulator import Bar, Instrument, Simulator, Venue

# the kinds a shaped universe places, which a narrower KINDS leaves out
SHAPED = {"market", "stop", "limit"}


def universe(rng):
    """The clock and each instrument's bars on it, spot or a linear perpetual, each
    on its own walk at its own price, None on a row it printed no bar. The first
    prints every row, so every row of the clock has a bar."""
    n     = rng.randint(2, 5)
    rows  = rng.randint(8, 40)
    times = clock(rng, rows)

    kinds, columns = [], []
    for k in range(n):
        kinds.append(rng.choice(["spot", "linear", "linear"]))
        scale = rng.choice([0.05, 1.0, 30.0])
        bars  = [Bar(*(price * scale for price in bar[:4]), bar.volume)
                 for bar in walk(rng, rows)]
        start   = rng.randint(1, 3) if k > 0 and rng.random() < 0.3 else 0
        missing = 0.1 if k > 0 else 0.0
        columns.append([None if t < start or (t > start and rng.random() < missing)
                        else bar for t, bar in enumerate(bars)])

    return times, kinds, columns


def calm(rng, price):
    """A quiet bar opening at `price`."""
    close = price * (1.0 + rng.uniform(-0.01, 0.01))
    return Bar(price, max(price, close) * (1.0 + rng.uniform(0.0, 0.005)),
               min(price, close) * (1.0 - rng.uniform(0.0, 0.005)), close,
               rng.uniform(1e6, 1e7))


def against(rng, price, long, depth, room=0.01):
    """A bar opening at `price` that moves `depth` against a long or a short and up
    to `room` the other way."""
    if long:
        low, high = price * (1.0 - depth), price * (1.0 + rng.uniform(0.0, room))
    else:
        low, high = price * (1.0 - rng.uniform(0.0, room)), price * (1.0 + depth)

    return Bar(price, high, low, rng.uniform(low, high), rng.uniform(1e6, 1e7))


def shaped(rng):
    """A universe drawn around a shape random ones rarely reach. Perpetual A is
    taken at the second row's open, and the fourth row's bar either moves against A
    through a reduce-only stop resting on it while perpetual B, taken at that bar's
    open, moves against B too, which is where the margin bound lifts a fill beside
    another perp, or gaps through a limit decided at the close before it, which
    fills on the gap and reduces or flips A, with B carried in and moving against
    it. The rows before are quiet, so the stop rests, and so are the rows after."""
    rows  = rng.randint(5, 9)
    times = clock(rng, rows)
    kinds = ["linear", "linear"]

    long_a, long_b, stop = (rng.random() < 0.5 for _ in range(3))
    gap     = rng.uniform(0.025, 0.15) if stop else rng.uniform(0.005, 0.1)
    columns = [[calm(rng, rng.choice([5.0, 100.0, 3_000.0]))] for _ in kinds]
    for column in columns:
        for _ in range(2):
            column.append(calm(rng, column[-1].close))

    a, b = columns[0][2].close, columns[1][2].close
    if stop:
        # placed off the second row's close, beyond what the third row trades; the
        # fourth opens on a gap against A now and then, past its entry, where a lift
        # toward the entry can go further than the bar
        trigger = columns[0][1].close * (1.0 - gap if long_a else 1.0 + gap)
        open_   = a * (1.0 + rng.uniform(0.0, gap / 2.0) * (-1.0 if long_a else 1.0))
        past    = abs(trigger / open_ - 1.0) + rng.uniform(0.01, 0.3)
        columns[0].append(against(rng, open_, long_a, past))
        u = (0.2 - gap if long_a else 0.2 + gap) / 0.4
    else:
        limit = a * (1.0 + gap if long_a else 1.0 - gap)
        open_ = limit * (1.0 + rng.uniform(0.005, 0.1) * (1.0 if long_a else -1.0))
        columns[0].append(against(rng, open_, not long_a, rng.uniform(0.0, 0.05),
                                  room=0.004))
        u = (0.01 + gap if long_a else 0.2 - gap) / 0.21

    columns[1].append(against(rng, b, long_b, rng.uniform(0.05, 0.6)))
    for column in columns:
        while len(column) < rows:
            column.append(calm(rng, column[-1].close))

    cfg   = config(rng, times, kinds, columns)
    worth = [cfg["quote"] / column[0].close for column in columns]
    a_side, b_side = ("buy" if long_a else "sell"), ("buy" if long_b else "sell")
    size = worth[0] * rng.uniform(0.3, 6.0)

    actions = [None] * (rows - 1)
    actions[0] = (0, "market", a_side, size, 0.0)
    close_side = "sell" if long_a else "buy"
    b_order    = (1, "market", b_side, worth[1] * rng.uniform(0.3, 12.0), 0.0)

    if stop:
        actions[1] = (0, "stop", close_side, size, u)
        actions[2] = b_order
    else:
        actions[1] = b_order
        actions[2] = (0, "limit", close_side, size * rng.choice([0.5, 1.0, 2.0, 4.0]),
                      u)

    return times, kinds, columns, cfg, actions


def take_profit(rng, band):
    """A universe drawn around a take-profit on perpetual A, which is taken at the
    second row's open. The take-profit is decided at the third row's close, and the
    fourth row's bar reaches it on the favourable side of its open. Without `band`
    that bar moves against A, to its trigger or short of it, so the take-profit is
    held back or fills, beside a protective stop the bar reaches first, with B
    taken at the third row's open and moving against it, with B printing no bar
    while the take-profit flips A past its cap, or with A opened on that bar by a
    limit it reaches first. With `band` the third row leaves the account between its
    close point and its trigger, with no slippage, cap, funding or maintenance to
    move either, and the fourth row stays there, where a maker's exit is bounded at
    its own fee."""
    rows  = rng.randint(5, 9)
    times = clock(rng, rows)
    kinds = ["linear", "linear"]

    long_a, long_b = (rng.random() < 0.5 for _ in range(2))
    sign = 1.0 if long_a else -1.0
    side, exit_side = ("buy", "sell") if long_a else ("sell", "buy")

    columns = [[calm(rng, rng.choice([5.0, 100.0, 3_000.0]))] for _ in kinds]
    for column in columns:
        column.append(calm(rng, column[-1].close))

    cfg = config(rng, times, kinds, columns)
    cfg["max_open_orders"] = 8

    a, b = columns
    worth   = cfg["quote"] / a[0].close
    actions = [None] * (rows - 1)

    def bar(open_, adverse, favourable, close):
        prices = (open_, adverse, favourable, close)
        return Bar(open_, max(prices), min(prices), close, rng.uniform(1e6, 1e7))

    beside = None
    if band:
        cfg.update(slippage_bps=0.0, fee_taker=0.002, leverage=0.0, funding=("none",))
        cfg["tiers"][0] = (None, "notional")

        size    = worth * rng.uniform(2.0, 6.0)
        wallet  = cfg["quote"] - size * a[1].open * cfg["fee_taker"]
        trigger = a[1].open - sign * wallet / size
        # the close point less the trigger, signed toward the favourable side
        width = trigger / (1.0 - sign * cfg["fee_taker"]) - trigger

        def at(x):
            return trigger + x * width

        close = at(rng.uniform(0.2, 0.4))
        a.append(bar(a[1].close, at(rng.uniform(0.05, 0.15)),
                     a[1].close * (1.0 + sign * rng.uniform(0.0, 0.005)), close))
        price = at(rng.uniform(0.45, 0.8))
        a.append(bar(close, at(rng.uniform(0.05, 0.15)), at(rng.uniform(0.85, 0.97)),
                     at(rng.uniform(0.15, 0.85))))
        b.append(calm(rng, b[-1].close))
        b.append(calm(rng, b[-1].close))
        share = rng.choice([0.5, 1.0])
    else:
        beside = rng.choice(["stop", "against", "absent", "opened"])
        # the take-profit a liquidating bar holds back is the pessimistic setting's
        cfg["intrabar"] = "pessimistic"
        size = worth * rng.uniform(3.0 if beside == "opened" else 1.0,
                                   2.0 if beside == "absent" else 6.0)

        a.append(calm(rng, a[-1].close))
        b.append(calm(rng, b[-1].close))
        close = a[2].close
        price = close * (1.0 + sign * rng.uniform(0.002, 0.03))
        entry = rng.uniform(0.02, 0.1)
        if beside == "absent":
            depth = rng.uniform(0.01, 0.1)
        else:
            # deep enough, for the position A opens on the bar, to reach its trigger
            depth = rng.uniform(entry + (0.25 if beside == "opened" else 0.02), 0.6)

        a.append(bar(close, close * (1.0 - sign * depth),
                     price * (1.0 + sign * rng.uniform(0.0, 0.01)),
                     close * (1.0 + sign * rng.uniform(-depth, 0.0))))
        share = rng.choice([0.5, 1.0, 2.0])

        if beside == "stop":
            # a reduce-only stop placed off the second row's close, past what the
            # third row trades
            y = rng.uniform(0.02, 0.3)
            actions[1] = (0, "stop", exit_side, size * rng.choice([0.2, 0.5, 1.0]),
                          (0.2 - sign * y) / 0.4)
            b.append(calm(rng, b[-1].close))
        elif beside == "opened":
            # a limit placed off the second row's close, past what the third row
            # trades, that opens A on the fourth row's bar before the bar reaches the
            # take-profit, which then closes what that bar opened
            actions[1] = (0, "limit", side, size,
                          (0.2 - entry) / 0.21 if long_a else (0.01 + entry) / 0.21)
            b.append(calm(rng, b[-1].close))
            cfg["leverage"] = 0.0
            cfg["tiers"][0] = (None, "notional")
        else:
            b_worth = cfg["quote"] / b[1].close
            most    = 1.0 if beside == "absent" else 6.0
            actions[1] = (1, "market", "buy" if long_b else "sell",
                          b_worth * rng.uniform(0.3, most), 0.0)
            if beside == "against":
                b.append(against(rng, b[-1].close, long_b, rng.uniform(0.05, 0.6)))
            else:
                # B prints no bar on the fourth row, and the take-profit flips A
                # past a cap of 3x, which reads B at its last mark
                b.append(None)
                cfg["leverage"] = 3.0
                cfg["tiers"]    = [(None, "notional"), (None, "notional")]
                share = 4.0

    for column in columns:
        while len(column) < rows:
            last = [row for row in column if row is not None][-1]
            column.append(calm(rng, last.close))

    if beside != "opened":
        actions[0] = (0, "market", side, size, 0.0)

    base = 0.99 if exit_side == "sell" else 0.8
    # reduce-only, as a take-profit is, unless it is to flip A
    kind = "limit" if share > 1.0 else "reduce_limit"
    actions[2] = (0, kind, exit_side, size * share, (price / close - base) / 0.21)

    return times, kinds, columns, cfg, actions


def config(rng, times, kinds, columns):
    """The venue: a funding source charging every perpetual held, or each event its
    own instrument, the wallets the kinds use, and a tier table for some perpetuals,
    their bounds a few times the balance."""
    perps   = [k for k, kind in enumerate(kinds) if kind == "linear"]
    funding = ("none",)
    if perps:
        rate    = rng.choice([0.0001, 0.02, -0.01, 0.1])
        funding = rng.choice([
            ("none",),
            ("every_bars", 3, rate),
            ("every", rng.choice([0.5, 1.0, 3.0, 8.0]), rate),
            ("data", [(stamp, rng.choice(perps + [None]), rate)
                      for stamp, rate in stamps(rng, times)]),
        ])

    quote = rng.choice([1_000.0, 10_000.0])
    if "spot" in kinds and perps:
        share   = rng.choice([0.2, 0.5, 0.8])
        balance = {"spot": quote * share, "margin": quote * (1.0 - share)}
    else:
        balance = quote

    tiers = []
    for kind, column in zip(kinds, columns):
        first = next(bar for bar in column if bar is not None)
        tiers.append(tier_table(rng, quote, kind == "linear", first.open))

    return dict(
        kinds=kinds, quote=quote, balance=balance,
        fee_taker=rng.choice([0.0, 0.0006, 0.002]),
        fee_maker=rng.choice([0.0, 0.0002]),
        slippage_bps=rng.choice([0.0, 5.0, 50.0]),
        max_open_orders=rng.choice([1, 2, 8]),
        leverage=rng.choice([0.0, 3.0, 10.0]),
        funding=funding, tiers=tiers,
    )


def plan(rng, columns, cfg):
    """One action per row but the last, on an instrument that has printed a bar: an
    instrument, a kind, a side, a size and a draw `u` that sets the action's price
    off that instrument's last close, sized from a twentieth of the balance's worth
    to twelve times it. Now and then a market order goes to an instrument that has
    not printed its first bar, which is refused."""
    out, last = [], [None] * len(columns)
    for t in range(len(columns[0]) - 1):
        for k, column in enumerate(columns):
            if column[t] is not None:
                last[k] = column[t].close

        listed   = [k for k in range(len(columns)) if last[k] is not None]
        unlisted = [k for k in range(len(columns)) if last[k] is None]
        if rng.random() < 0.45:
            out.append(None)
            continue

        if unlisted and rng.random() < 0.1:
            out.append((rng.choice(unlisted), "market", "buy", 1.0, 0.0))
            continue

        k     = rng.choice(listed)
        worth = cfg["quote"] / last[k]
        size  = worth * math.exp(rng.uniform(math.log(0.05), math.log(12.0)))
        out.append((k, rng.choice(KINDS), rng.choice(["buy", "sell"]), size,
                    rng.random()))

    return out


def place(sim, action, t, theirs, script, reached):
    """One action, decided by the reference on its own account and taken there, and
    written into `script` for the engine, as `differential.place` does, on the
    instrument the action names. An order first acts on the next row its instrument
    prints a bar on, and is counted there."""
    if action is None:
        return

    i, kind, side, size, u = action
    close = sim.mark[i]
    on    = next((r for r in range(t + 1, len(sim.bars)) if sim.bars[r][i] is not None),
                 None)

    if kind in ("close", "reduce", "exits") and sim.position[i] != 0.0:
        if on is not None:
            reached[on][kind] += 1
    elif kind in ("cancel", "replace") and theirs and resting(sim, theirs[0][0]):
        reached[t][kind] += 1
    elif kind == "cancel_all" and sim.resting(i):
        reached[t][kind] += 1

    got = None
    if kind in ("market", "fok", "reduce"):
        tif         = "fok" if kind == "fok" else "ioc"
        reduce_only = kind == "reduce"
        script.take(t, side, size=size, tif=tif, reduce_only=reduce_only,
                    instrument=i)
        sim.order(i, side, size, tif=tif, reduce_only=reduce_only)
    elif kind in ("limit", "post", "ioc_limit", "fok_limit", "reduce_limit"):
        price = limit_price(side, close, u)
        tif   = {"ioc_limit": "ioc", "fok_limit": "fok"}.get(kind, "gtc")
        post_only, reduce_only = kind == "post", kind == "reduce_limit"
        placed = script.take(t, side, size=size, limit=price, tif=tif,
                             post_only=post_only, reduce_only=reduce_only,
                             instrument=i)
        got = (sim.order(i, side, size, limit=price, tif=tif, post_only=post_only,
                         reduce_only=reduce_only), placed)
    elif kind in ("stop", "stop_entry"):
        trigger, reduce_only = stop_price(close, u), kind == "stop"
        placed = script.take(t, side, size=size, stop=trigger, reduce_only=reduce_only,
                             instrument=i)
        got = (sim.order(i, side, size, stop=trigger, reduce_only=reduce_only),
               placed)
    elif kind == "close":
        script.take(t, "close", instrument=i)
        sim.close(i)
    elif kind == "exits":
        held = sim.position[i]
        if held == 0.0:
            return

        side = "sell" if held > 0.0 else "buy"
        low, high = close * (0.8 + 0.19 * u), close * (1.2 - 0.19 * u)
        trigger, price = (low, high) if held > 0.0 else (high, low)
        for leg in (dict(limit=price), dict(stop=trigger)):
            placed = script.take(t, side, size=size, reduce_only=True, instrument=i,
                                 **leg)
            order = sim.order(i, side, size, reduce_only=True, **leg)
            if resting(sim, order):
                theirs.append((order, placed))
    elif kind == "transfer":
        to = "margin" if side == "buy" else "spot"
        script.take(t, "transfer", amount=size * close, to=to)
        sim.transfer(size * close, to=to)
    elif kind == "weights":
        # a built case's own target, its weights given whole
        script.take(t, "target", weights=size, **settings(side, u))
        sim.target(size, **settings(side, u))
    elif kind in ("target", "retarget"):
        # some of the instruments that have printed a bar, and now and then one that
        # has not, whose leg is refused; a retarget's first target names every one
        # and moves any spot shortfall across, and the second replaces it in the
        # same decision
        candidates = [k for k in range(len(sim.instruments))
                      if sim.mark[k] is not None or u < 0.2]
        named   = [k for k in candidates if (u + 0.53 * k) % 1.0 < 0.75]
        weights = {k: weight(sim.instruments[k].kind, u, k, len(named))
                   for k in named}
        if kind == "retarget":
            first = {k: weight(sim.instruments[k].kind, u, k, len(candidates)) / 2.0
                     for k in candidates}
            moving = dict(settings(side, u), transfer=True)
            script.take(t, "target", weights=first, **moving)
            sim.target(first, **moving)

        script.take(t, "target", weights=weights, **settings(side, u))
        sim.target(weights, **settings(side, u))
    elif kind == "cancel":
        if theirs:
            order, placed = theirs.pop(0)
            script.take(t, "cancel", order=placed)
            sim.cancel(order)
    elif kind == "cancel_all":
        # every instrument's orders, or one instrument's
        if u < 0.5:
            script.take(t, "cancel_all")
            sim.cancel_all()
            theirs.clear()
        else:
            script.take(t, "cancel_all", instrument=i)
            sim.cancel_all(i)
            theirs[:] = [got for got in theirs if sim.orders[got[0]]["instrument"] != i]
    elif kind == "replace" and theirs:
        order, placed = theirs[0]
        size = size / 2.0
        mark = sim.mark[sim.orders[order]["instrument"]]
        if sim.orders[order]["kind"] == "stop":
            trigger = stop_price(mark, u)
            script.take(t, "replace", order=placed, size=size, trigger=trigger)
            sim.replace(order, trigger=trigger, size=size)
        else:
            price = None if u < 0.25 else limit_price(sim.orders[order]["side"], mark,
                                                      u)
            script.take(t, "replace", order=placed, size=size, price=price)
            sim.replace(order, price=price, size=size)

    if got is not None and resting(sim, got[0]):
        theirs.append(got)
        if on is not None and live(kind, sim.position[i], side):
            reached[on][kind] += 1


def make_reference(times, kinds, columns, cfg):
    source = cfg["funding"]
    events = None
    if source[0] == "data":
        funding, events = "data", source[1]
    elif source[0] in ("every_bars", "every"):
        funding = source
    else:
        funding = "none"

    venue = Venue(
        balance         = cfg["balance"],
        maker           = cfg["fee_maker"],
        taker           = cfg["fee_taker"],
        leverage        = cfg["leverage"],
        max_open_orders = cfg["max_open_orders"],
        slippage        = cfg["slippage_bps"],
        funding         = funding,
        intrabar        = cfg["intrabar"],
    )

    instruments = [Instrument(f"I{k}", kind=kind, tiers=cfg["tiers"][k][0],
                              tiers_by=cfg["tiers"][k][1])
                   for k, kind in enumerate(kinds)]
    rows = [[column[t] for column in columns] for t in range(len(times))]

    return Simulator(times, rows, instruments, venue, funding=events)


def venue(cfg):
    """The engine's settings for a universe, as the binding takes them."""
    out = dict(
        market="perp", quote=cfg["balance"], fee_taker=cfg["fee_taker"],
        fee_maker=cfg["fee_maker"], slippage_bps=cfg["slippage_bps"],
        max_fill_fraction=1.0, max_open_orders=cfg["max_open_orders"],
        leverage=cfg["leverage"], impact=0.0,
        intrabar=cfg["intrabar"],
        instruments=[dict(market="spot" if kind == "spot" else "perp", tiers=rows,
                          tiers_by=by)
                     for kind, (rows, by) in zip(cfg["kinds"], cfg["tiers"])],
    )

    source = cfg["funding"]
    if source[0] == "every_bars":
        out.update(funding_interval=source[1], funding_rate=source[2])
    elif source[0] == "every":
        out.update(funding_hours=source[1], funding_rate=source[2])
    elif source[0] == "data":
        out.update(
            funding_times=np.array([stamp for stamp, _, _ in source[1]],
                                   dtype=np.int64),
            funding_rates=np.array([rate for _, _, rate in source[1]],
                                   dtype=np.float64),
            funding_instruments=[instrument for _, instrument, _ in source[1]],
        )

    return out


def play(times, columns, cfg, script):
    """The engine through a whole script in one call over the universe, an
    instrument that printed no bar on a row holding NaN there."""
    nan     = [math.nan] * 5
    candles = np.array([[nan if column[t] is None else list(column[t])
                         for column in columns] for t in range(len(times))],
                       dtype=np.float64)

    return probatum._probatum.run(candles, venue(cfg), script.bars,
                                  times=np.array(times, dtype=np.int64))


def run(case, reached):
    """The reference through the case, deciding as it goes, then the engine through
    the same decisions in one call."""
    times, kinds, columns, cfg, actions = case

    sim = make_reference(times, kinds, columns, cfg)
    script, theirs = Script(), []
    for t, action in enumerate(actions):
        place(sim, action, t, theirs, script, reached)
        sim.step()

    return play(times, columns, cfg, script), sim


FILL_FIELDS = ("bar", "instrument", "order", "side", "size", "price", "reference",
               "spread", "slippage", "impact", "fee", "adjustment", "taker", "reason")
EVENT_FIELDS = ("bar", "kind", "order", "instrument", "size", "price", "reason")
TRIP_FIELDS  = ("instrument", "side", "size", "entry", "exit", "open_bar",
                "close_bar", "fees", "funding", "pnl", "ret", "best", "worst",
                "reason", "open")
ACCOUNT_FIELDS = ("equity", "spot_cash", "margin_cash", "fees", "funding", "spread",
                  "slippage", "impact", "liquidation", "dust", "margin_used",
                  "maintenance")
LEG_FIELDS = (("position", "position"), ("entry", "entry"), ("mark", "mark"),
              ("notional", "notional"), ("unrealized", "upnl"), ("weight", "weight"),
              ("drift", "drift"))


def ledger_difference(mine, theirs, end=None):
    """The first row the two ledgers part company on before bar `end`, or on any row
    with no `end`, the account's columns and then each instrument's, as its bar and
    what about, or None. An instrument that has not printed is marked NaN by the
    engine and None by the reference."""
    if end is None and len(mine["equity"]) != len(theirs):
        return 0, f"{len(mine['equity'])} ledger rows against {len(theirs)}"

    for t, row in enumerate(theirs[:end]):
        for name in ACCOUNT_FIELDS:
            x = float(mine[name][t])
            if not close_enough(x, row[name]):
                return t, f"ledger row {t}: {name} {x!r} against {row[name]!r}"

        for k, leg in enumerate(row["instruments"]):
            for name, key in LEG_FIELDS:
                x = float(mine[name][t, k])
                y = math.nan if leg[key] is None else leg[key]
                if not close_enough(x, y):
                    return t, (f"ledger row {t}, instrument {k}: {name} {x!r} "
                               f"against {y!r}")

    return None


def disagreement(records, sim, end=None):
    """Where the two first part company on any record written before bar `end`, or
    on any record with no `end`, as the bar and what about, or None: the fill log,
    the order log, the round trips and the ledger."""
    return (log_difference("fill", FILL_FIELDS, before(records["fills"], end),
                           before(sim.fills, end))
            or log_difference("event", EVENT_FIELDS,
                              worded(before(records["order_log"], end)),
                              worded(before(sim.events, end)))
            or log_difference("round trip", TRIP_FIELDS,
                              closed_before(records["trades"], end),
                              closed_before(sim.round_trips(), end))
            or ledger_difference(records["ledger"], sim.ledger, end))


Outcome = collections.namedtuple(
    "Outcome", "compared agreed disagreed set_aside rules reached")


def compare(cases, seed):
    """`cases` random universes from `seed` through both, as `differential.compare`
    counts them."""
    rng     = random.Random(seed)
    reached = collections.Counter()
    rules   = collections.Counter()
    aside   = collections.Counter()
    whole, agreed, disagreed = 0, 0, []

    for number in range(cases):
        if number % 4 == 3 and SHAPED <= set(KINDS):
            case = shaped(rng)
        elif number % 8 == 1 and SHAPED <= set(KINDS):
            case = take_profit(rng, rng.random() < 0.5)
        else:
            times, kinds, columns = universe(rng)
            cfg  = config(rng, times, kinds, columns)
            case = (times, kinds, columns, cfg, plan(rng, columns, cfg))

        case[3].setdefault("intrabar", setting(seed, number))

        counted = collections.defaultdict(collections.Counter)
        try:
            records, sim = run(case, counted)
        except Exception as exc:
            disagreed.append((number, case, f"raised: {type(exc).__name__}: {exc}"))
            continue

        end   = None
        found = cut(sim)
        if found is not None:
            rule, end = found
            aside[rule] += 1

        tally(reached, rules, counted, sim, end)
        whole += end is None
        found = disagreement(records, sim, end)
        if found is not None:
            disagreed.append((number, case, found[1]))
        elif end is None:
            agreed += 1

    return Outcome(whole, agreed, disagreed, aside, rules, reached)


def render(out, cases):
    lines = [f"\n{out.compared} of {cases} compared whole, {out.agreed} agreed, "
             f"{len(out.disagreed)} disagreed",
             "compared up to a rule the engine does not have yet: "
             + (", ".join(f"{why} {n}" for why, n in
                          sorted(out.set_aside.items())) or "none"),
             "actions placed with something to act on: "
             + ", ".join(f"{k} {v}" for k, v in sorted(out.reached.items())) + "\n"]

    for number, (times, kinds, columns, cfg, actions), why in out.disagreed[:8]:
        lines.append(f"--- case {number} ({', '.join(kinds)}) {why}")
        lines.append(f"    cfg  {cfg}")
        lines.append(f"    acts {actions[:8]}\n")

    if len(out.disagreed) > 8:
        lines.append(f"... and {len(out.disagreed) - 8} more")

    return "\n".join(lines)


def main(argv=None):
    argv  = sys.argv[1:] if argv is None else argv
    cases = int(argv[0]) if len(argv) > 0 else 200
    seed  = int(argv[1]) if len(argv) > 1 else 20261005
    out   = compare(cases, seed)

    print(render(out, cases))
    return 1 if out.disagreed else 0


if __name__ == "__main__":
    sys.exit(main())
