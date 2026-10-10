"""Run the engine and the reference over the same random universes and diff them.

The comparison `differential` makes at one instrument, over universes of two to five
instruments, spot and linear perpetuals, on one clock. An instrument other than the
first lists a row or a few late now and then and prints no bar on some rows, so its
orders wait and its position keeps its mark, and the perpetuals share one margin
wallet, so the cap, the margin bound and funding run across them. A case is set
aside where the reference applied a rule the engine does not have yet, the rules
`differential` sets aside and a liquidation or a clip that takes several
perpetuals at once.

  python -m reference.universe_differential [cases] [seed]     (from tests/)
"""

import collections
import math
import random
import sys

import numpy as np

import probatum
from .differential import (KINDS, NOT_IN_THE_ENGINE, Script, clock,
                           close_enough, limit_price, log_difference, resting,
                           stamps, stop_price, tier_table, walk)
from .simulator import Bar, Instrument, Simulator, Venue

SET_ASIDE = NOT_IN_THE_ENGINE | {
    "a liquidation of several perpetuals",
    "a bar clipped across several perpetuals",
}


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
    instrument the action names."""
    if action is None:
        return

    i, kind, side, size, u = action
    close = sim.mark[i]

    if kind in ("close", "reduce", "exits") and sim.position[i] != 0.0:
        reached[kind] += 1
    elif kind in ("cancel", "replace") and theirs and resting(sim, theirs[0][0]):
        reached[kind] += 1
    elif kind == "cancel_all" and sim.resting(i):
        reached[kind] += 1

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
        if kind in ("ioc_limit", "fok_limit", "reduce_limit", "stop_entry"):
            reached[kind] += 1


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
EVENT_FIELDS = ("bar", "kind", "order", "instrument", "size", "price")
TRIP_FIELDS  = ("instrument", "side", "size", "entry", "exit", "open_bar",
                "close_bar", "fees", "funding", "pnl", "ret", "best", "worst",
                "reason", "open")
ACCOUNT_FIELDS = ("equity", "spot_cash", "margin_cash", "fees", "funding", "spread",
                  "slippage", "impact", "liquidation", "dust", "margin_used",
                  "maintenance")
LEG_FIELDS = (("position", "position"), ("entry", "entry"), ("mark", "mark"),
              ("notional", "notional"), ("unrealized", "upnl"), ("weight", "weight"))


def ledger_difference(mine, theirs):
    """The first row the two ledgers part company on, the account's columns and
    then each instrument's, as its bar and what about, or None. An instrument that
    has not printed is marked NaN by the engine and None by the reference."""
    if len(mine["equity"]) != len(theirs):
        return 0, f"{len(mine['equity'])} ledger rows against {len(theirs)}"

    for t, row in enumerate(theirs):
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


def disagreement(records, sim):
    """Where the two first part company, as the bar and what about, or None: the
    fill log, the order log, the round trips and the ledger."""
    return (log_difference("fill", FILL_FIELDS, records["fills"], sim.fills)
            or log_difference("event", EVENT_FIELDS, records["order_log"], sim.events)
            or log_difference("round trip", TRIP_FIELDS, records["trades"],
                              sim.round_trips())
            or ledger_difference(records["ledger"], sim.ledger))


def set_aside(sim):
    """The first rule the case reached that the engine does not have yet, or None."""
    for rule in sorted(SET_ASIDE):
        if sim.reached[rule]:
            return rule

    return None


Outcome = collections.namedtuple(
    "Outcome", "compared agreed disagreed set_aside rules reached")


def compare(cases, seed):
    rng     = random.Random(seed)
    reached = collections.Counter()
    rules   = collections.Counter()
    aside   = collections.Counter()
    agreed, disagreed = 0, []

    for number in range(cases):
        times, kinds, columns = universe(rng)
        cfg     = config(rng, times, kinds, columns)
        case    = (times, kinds, columns, cfg, plan(rng, columns, cfg))
        counted = collections.Counter()

        try:
            records, sim = run(case, counted)
        except Exception as exc:
            disagreed.append((number, case, f"raised: {type(exc).__name__}: {exc}"))
            continue

        why = set_aside(sim)
        if why is not None:
            aside[why] += 1
            continue

        reached.update(counted)
        rules.update(sim.reached)

        found = disagreement(records, sim)
        if found is None:
            agreed += 1
        else:
            disagreed.append((number, case, found[1]))

    return Outcome(agreed + len(disagreed), agreed, disagreed, aside, rules, reached)


def render(out, cases):
    lines = [f"\n{out.compared} of {cases} compared, {out.agreed} agreed, "
             f"{len(out.disagreed)} disagreed",
             "set aside: " + (", ".join(f"{why} {n}" for why, n in
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
