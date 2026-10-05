"""Run the engine and the reference over the same random cases and diff them.

The reference is written from the Specification and never from the engine, so a case
the two part company on is a question about a rule, and somebody has to decide which
reading is wrong. The engine still carries emsl's rules where the Specification has
moved on, so a case is compared only where the reference applied none of the rules
the engine has yet to take up. The reference counts every rule it applies, which is
what decides it, and a stage that brings a rule into the engine takes it off
`NOT_IN_THE_ENGINE`, so the set of cases compared grows with the engine.

  python -m reference.differential [cases] [seed]     (from tests/)
"""

import collections
import math
import os
import random
import sys

import numpy as np

import probatum
from .simulator import Bar, Instrument, Simulator, Venue

HOUR = 3_600 * 10**9
TOL  = 1e-6

# the rules the reference applies that the engine does not have yet
NOT_IN_THE_ENGINE = {
    "the liquidity budget",
    "a limit held to the prices traded since the last close",
    "a limit on the book its bar opened through",
    "a post_only limit its first bar opened through",
    "an order trading against the account's own refused",
    "a favourable exit held back on a liquidating bar",
    "a liquidation at the closes",
    "a wallet that fees left below zero set to zero",
}


def walk(rng, n):
    """Consistent bars from a random walk, a third of them violent enough to liquidate
    at 10x and the rest calm enough for a levered account to live until a gap kills
    it, many opening with one. A bar trades nothing or far more than any order here
    asks: the engine caps a fill at a share of its own bar and the Specification at
    a share of the trailing volume, and the two agree only where neither binds."""
    bars, price = [], 100.0
    for t in range(n):
        if t > 0 and rng.random() < 0.4:
            price = max(price * (1.0 + rng.uniform(-0.2, 0.2)), 0.01)

        move, wick = rng.choice([(0.02, 0.01), (0.02, 0.01), (0.25, 0.2)])
        close  = max(price * (1.0 + rng.uniform(-move, move)), 0.01)
        high   = max(price, close) * (1.0 + rng.uniform(0.0, wick))
        low    = max(min(price, close) * (1.0 - rng.uniform(0.0, wick)), 0.001)
        dead   = t > 0 and bars[-1].volume > 0.0 and rng.random() < 0.1
        volume = 0.0 if dead else rng.uniform(1e6, 1e7)
        bars.append(Bar(price, high, low, close, volume))
        price = close

    return bars


def config(rng):
    market = rng.choice(["spot", "perp"])
    perp   = market == "perp"
    return dict(
        market=market,
        quote=rng.choice([1_000.0, 10_000.0]),
        fee_taker=rng.choice([0.0, 0.0006, 0.002]),
        fee_maker=rng.choice([0.0, 0.0002]),
        slippage_bps=rng.choice([0.0, 5.0, 50.0]),
        # at 1 every second placement is refused, which is the only way the bound on
        # the book and on the pending queue gets reached
        max_open_orders=rng.choice([1, 2, 8]),
        leverage=rng.choice([0.0, 3.0, 10.0]) if perp else 10.0,
        funding_rate=rng.choice([0.0, 0.0001, 0.01]) if perp else 0.0,
        funding_interval=rng.choice([0, 3]) if perp else 0,
    )


# every order surface both simulators have, so the gate exercises all of them.
# Narrow it with KINDS=limit when a disagreement needs isolating
KINDS = os.environ.get(
    "KINDS",
    "market,limit,stop,fok,post,cancel,replace,reduce,close,ioc_limit,fok_limit,"
    "cancel_all,reduce_limit,stop_entry",
).split(",")

# the kinds that only mean something with a live position or order to act on; a run
# in which one never met anything agreed about nothing
TRACKED = [kind for kind in ("close", "reduce", "cancel", "replace", "cancel_all",
                             "ioc_limit", "fok_limit", "reduce_limit", "stop_entry")
           if kind in KINDS]


def plan(rng, bars, cfg):
    """One action per bar but the last: a kind, a side, a size and a draw `u` that
    sets the action's price off the close it is decided at. A size runs from a
    twentieth of the balance's worth at that close to twelve times it, so the cap
    binds, the spot clamps bind and levered accounts die."""
    out = []
    for t in range(len(bars) - 1):
        if rng.random() < 0.45:
            out.append(None)
            continue

        worth = cfg["quote"] / bars[t].close
        size  = worth * math.exp(rng.uniform(math.log(0.05), math.log(12.0)))
        out.append((rng.choice(KINDS), rng.choice(["buy", "sell"]), size,
                    rng.random()))

    return out


def limit_price(side, close, u):
    """Mostly passive, and within a hundredth of the close on the other side, where
    the next bar usually trades through it and fills it as a taker."""
    if side == "buy":
        return close * (0.8 + 0.21 * u)
    return close * (0.99 + 0.21 * u)


def stop_price(close, u):
    return close * (0.8 + 0.4 * u)


def make_engine(bars, cfg):
    array = np.array([list(bar) for bar in bars], dtype=np.float64)
    return probatum.Engine(
        array,
        market            = cfg["market"],
        quote             = cfg["quote"],
        fee_taker         = cfg["fee_taker"],
        fee_maker         = cfg["fee_maker"],
        slippage_bps      = cfg["slippage_bps"],
        max_fill_fraction = 1.0,
        max_open_orders   = cfg["max_open_orders"],
        leverage          = cfg["leverage"],
        impact            = 0.0,
        funding_rate      = cfg["funding_rate"],
        funding_interval  = cfg["funding_interval"],
        report            = True,
    )


def funding_of(cfg):
    if cfg["funding_interval"] > 0 and cfg["funding_rate"] != 0.0:
        return ("every_bars", cfg["funding_interval"], cfg["funding_rate"])
    return "none"


def make_reference(bars, cfg, funding=None, times=None):
    """The reference over one instrument on an hourly clock. `funding`, when given,
    is the venue's own events, in place of the schedule every n bars."""
    times = [k * HOUR for k in range(len(bars))] if times is None else times
    kind  = "spot" if cfg["market"] == "spot" else "linear"
    venue = Venue(
        balance         = cfg["quote"],
        maker           = cfg["fee_maker"],
        taker           = cfg["fee_taker"],
        leverage        = cfg["leverage"],
        max_open_orders = cfg["max_open_orders"],
        slippage        = cfg["slippage_bps"],
        funding         = funding_of(cfg) if funding is None else "data",
    )

    return Simulator(times, [[bar] for bar in bars], [Instrument("X", kind=kind)],
                     venue, funding=funding)


def resting(sim, order_id):
    order = sim.orders.get(order_id)
    return order is not None and order["status"] == "resting"


def place(engine, sim, action, close, mine, theirs, reached=None):
    """One action on both sides. `mine` and `theirs` hold the ids each believes it
    placed, in order, so a cancel or a replace addresses the same order in each."""
    if action is None:
        return

    kind, side, size, u = action

    if reached is not None:
        # what the action had to act on when it was placed: one that never meets a
        # live position or order agrees with anything, and after the fact there is
        # no telling that from a real agreement
        if kind in ("close", "reduce") and sim.position[0] != 0.0:
            reached[kind] += 1
        elif kind in ("cancel", "replace") and theirs and resting(sim, theirs[0]):
            reached[kind] += 1
        elif kind == "cancel_all" and sim.resting():
            reached[kind] += 1

    mine_got = theirs_got = None
    if kind in ("market", "fok", "reduce"):
        tif         = "fok" if kind == "fok" else "ioc"
        reduce_only = kind == "reduce"
        mine_got    = engine.order(side, size, "market", None, None, reduce_only, False,
                                   tif)
        theirs_got = sim.order(0, side, size, tif=tif, reduce_only=reduce_only)
    elif kind in ("limit", "post", "ioc_limit", "fok_limit", "reduce_limit"):
        price = limit_price(side, close, u)
        tif   = {"ioc_limit": "ioc", "fok_limit": "fok"}.get(kind, "gtc")
        post_only, reduce_only = kind == "post", kind == "reduce_limit"
        mine_got = engine.order(side, size, "limit", price, None, reduce_only,
                                post_only, tif)
        theirs_got = sim.order(0, side, size, limit=price, tif=tif, post_only=post_only,
                               reduce_only=reduce_only)
    elif kind in ("stop", "stop_entry"):
        # a protective stop only shrinks the position; a breakout stop opens one
        trigger, reduce_only = stop_price(close, u), kind == "stop"
        mine_got   = engine.stop(side, size, trigger, reduce_only)
        theirs_got = sim.order(0, side, size, stop=trigger, reduce_only=reduce_only)
    elif kind == "close":
        mine_got   = engine.close()
        theirs_got = sim.close(0)
    elif kind == "cancel":
        if mine and theirs:
            engine.cancel(mine.pop(0))
            sim.cancel(theirs.pop(0))
    elif kind == "cancel_all":
        engine.cancel_all()
        sim.cancel_all()
        mine.clear()
        theirs.clear()
    elif kind == "replace" and mine and theirs:
        # the side and kind stay the order's own: a replacement moves an order, it
        # never turns one into another
        order = sim.orders[theirs[0]]
        size  = size / 2.0
        if order["kind"] == "stop":
            trigger = stop_price(close, u)
            moved   = engine.replace(mine[0], size, None, trigger)
            sim.replace(theirs[0], trigger=trigger, size=size)
        else:
            price = limit_price(order["side"], close, u)
            moved = engine.replace(mine[0], size, price, None)
            sim.replace(theirs[0], price=price, size=size)

        # a refused replacement leaves the original under its own id on both sides
        if moved is not None:
            mine[0] = moved

    if kind in ("limit", "post", "ioc_limit", "fok_limit", "reduce_limit", "stop",
                "stop_entry"):
        if mine_got is not None:
            mine.append(mine_got)
        if resting(sim, theirs_got):
            theirs.append(theirs_got)
        if reached is not None and resting(sim, theirs_got) and kind in (
                "ioc_limit", "fok_limit", "reduce_limit", "stop_entry"):
            reached[kind] += 1


def run(bars, cfg, actions, reached=None):
    """Both simulators through the case, a row of each per bar."""
    engine, sim = make_engine(bars, cfg), make_reference(bars, cfg)
    engine.reset()

    mine, theirs, rows, ran_out = [], [], [], set()
    funded = 0.0

    for t, action in enumerate(actions):
        place(engine, sim, action, bars[t].close, mine, theirs, reached)
        bust, liquidations = engine.is_bust(), sim.liquidations
        state = engine.step()
        sim.step()
        funded += sim.ledger[-1]["funding"]

        if engine.is_bust() != bust or sim.liquidations != liquidations or sim.clipped:
            ran_out.add(t + 1)

        cash = sim.spot_cash if cfg["market"] == "spot" else sim.margin_cash
        rows.append(((state["position"], state["equity"], state["quote"],
                      state["funding_paid"]),
                     (sim.position[0], sim.equity(), cash, funded)))
        if engine.done():
            break

    return rows, engine, sim, ran_out


def set_aside(sim, cfg, ran_out):
    """Why a case is outside what both simulators model, or None. `ran_out` holds
    the bars the margin ran out on, clipped or liquidated on either side."""
    for rule in sorted(NOT_IN_THE_ENGINE):
        if sim.reached[rule]:
            return rule

    # the engine charges a bar's funding before its liquidation check and at the
    # clipped close, the Specification after the check and at the close, so the
    # two agree on a funding bar only where the margin held all through it
    if funding_of(cfg) != "none" and any(
            t % cfg["funding_interval"] == 0 for t in ran_out):
        return "funding on a bar the margin ran out on"

    return None


FIELDS = ("position", "equity", "cash", "funding_paid")


def close_enough(a, b, scale=1.0):
    if not math.isfinite(a) or not math.isfinite(b):
        return math.isfinite(a) == math.isfinite(b)
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b), scale)


def first_difference(rows, cfg):
    """The first bar the two disagree on and what about, or None."""
    scales = (1.0, cfg["quote"], cfg["quote"], cfg["quote"])

    for i, (mine, theirs) in enumerate(rows):
        for name, a, b, scale in zip(FIELDS, mine, theirs, scales):
            if not close_enough(a, b, scale):
                return i, f"bar {i + 1}: {name} {a:.6f} against {b:.6f}"

    return None


Outcome = collections.namedtuple(
    "Outcome", "compared agreed disagreed set_aside rules reached")


def compare(cases, seed):
    rng     = random.Random(seed)
    reached = collections.Counter()
    rules   = collections.Counter()
    aside   = collections.Counter()
    agreed, disagreed = 0, []

    for case in range(cases):
        bars    = walk(rng, rng.randint(8, 40))
        cfg     = config(rng)
        actions = plan(rng, bars, cfg)
        counted = collections.Counter()

        try:
            rows, _, sim, ran_out = run(bars, cfg, actions, counted)
        except Exception as exc:
            disagreed.append((case, cfg, bars, actions,
                              f"raised: {type(exc).__name__}: {exc}"))
            continue

        why = set_aside(sim, cfg, ran_out)
        if why is not None:
            aside[why] += 1
            continue

        reached.update(counted)
        rules.update(sim.reached)

        found = first_difference(rows, cfg)
        if found is None:
            agreed += 1
        else:
            disagreed.append((case, cfg, bars, actions, found[1]))

    return Outcome(agreed + len(disagreed), agreed, disagreed, aside, rules, reached)


def render(out, cases):
    lines = [f"\n{out.compared} of {cases} compared, {out.agreed} agreed, "
             f"{len(out.disagreed)} disagreed",
             "set aside: " + (", ".join(f"{why} {n}" for why, n in
                                        sorted(out.set_aside.items())) or "none"),
             "actions placed with something to act on: "
             + ", ".join(f"{k} {v}" for k, v in sorted(out.reached.items())) + "\n"]

    for case, cfg, bars, actions, why in out.disagreed[:8]:
        lines.append(f"--- case {case} ({cfg['market']}) {why}")
        lines.append(f"    cfg  {cfg}")
        lines.append(f"    bars {[tuple(round(x, 4) for x in b) for b in bars[:6]]}")
        lines.append(f"    acts {actions[:6]}\n")

    if len(out.disagreed) > 8:
        lines.append(f"... and {len(out.disagreed) - 8} more")

    return "\n".join(lines)


def main(argv=None):
    argv  = sys.argv[1:] if argv is None else argv
    cases = int(argv[0]) if len(argv) > 0 else 400
    seed  = int(argv[1]) if len(argv) > 1 else 20261005
    out   = compare(cases, seed)

    print(render(out, cases))
    return 1 if out.disagreed else 0


if __name__ == "__main__":
    sys.exit(main())
