"""Run the BATCHED path against one independent reference per env.

`Batch` is the tier every reinforcement learning number comes out of: `VectorEnv`
drives `reset_at_each`, `advance` and `snapshot` and nothing else. What stood behind
those was a Rust test comparing the batch against the engine it is built from, which
is worth having and shares every mistake the engine makes. Here each env is compared
against a reference run over the bars from that env's own offset, written from the
Specification and knowing nothing about either implementation, over the cases both
model as `differential` decides them.

The offsets are the point. An episode that starts part way into the series must
still fund on the bars the SERIES funds on (ADR 0020), still hold
its own book, and still reach the end of the data when the data ends rather than
when its own step count runs out. All three are invisible when every env starts
at bar zero, which is what the batched tests do.

  python -m reference.batch_differential [cases] [seed]     (from tests/)
"""

import collections
import math
import random
import statistics
import sys

import numpy as np

import probatum
from . import universe_differential
from .differential import (balance, before, clock, close_enough, closed_before, config,
                           cut, drawn, engine_fills, engine_funding, fills, liquidation,
                           log_difference, make_reference, money, set_aside, setting,
                           walk, worded)


def offsets_for(rng, bars, num_envs):
    # at least one bar to step into, and never all the same: two envs on the same
    # offset run the same episode, and then a batch that broadcast env 0 over the
    # rest of them would agree with the reference
    top   = len(bars) - 2
    drawn = [rng.randint(0, top) for _ in range(num_envs)]
    if len(set(drawn)) == 1 and top > 0:
        drawn[0] = (drawn[0] + 1) % (top + 1)
    return drawn


def plan(rng, num_envs, steps, scale=1.0):
    """A signed size per env per step: positive buys, negative sells, zero holds,
    which is exactly what `advance` reads, `scale` times the base units drawn, as
    many contracts of an inverse perpetual as are worth them at a price of 100."""
    return [
        [0.0 if rng.random() < 0.45 else scale * rng.uniform(-40.0, 40.0)
         for _ in range(num_envs)]
        for _ in range(steps)
    ]


def drive_batch(bars, times, cfg, offsets, actions):
    array = np.array([list(bar) for bar in bars], dtype=np.float64)
    batch = probatum.Batch(
        array,
        num_envs        = len(offsets),
        market          = cfg["market"],
        quote           = balance(cfg),
        fee_taker       = cfg["fee_taker"],
        fee_maker       = cfg["fee_maker"],
        slippage_bps    = cfg["slippage_bps"],
        max_open_orders = cfg["max_open_orders"],
        leverage        = cfg["leverage"],
        times           = np.array(times, dtype=np.int64),
        report          = False,
        intrabar        = cfg["intrabar"],
        contract        = cfg.get("contract"),
        **engine_funding(cfg["funding"]),
        **engine_fills(cfg),
    )
    batch.reset_at_each(np.array(offsets, dtype=np.int64))

    seen = []
    for row in actions:
        batch.advance(np.array(row, dtype=np.float64))
        snap = batch.snapshot()
        seen.append({
            "position":     snap["position"].copy(),
            "equity":       snap["equity"].copy(),
            "mark_price":   snap["mark_price"].copy(),
            "funding_paid": snap["funding_paid"].copy(),
            "bust":         batch.busts().copy(),
            "done":         batch.dones().copy(),
        })

    return seen, batch


def drive_references(bars, times, cfg, offsets, actions):
    """One reference per env, opened at its offset on the whole series, so its
    window reads the bars before that offset as the env's does. A schedule of every
    n bars is passed as the events it stamps on the whole series."""
    interval = int(statistics.median(b - a for a, b in zip(times, times[1:])))
    events   = None
    if cfg["funding"][0] == "every_bars":
        _, n, rate = cfg["funding"]
        events = [(times[r] + interval, rate) for r in range(n, len(bars), n)]

    refs = [make_reference(bars, cfg, times, interval, events, start=off)
            for off in offsets]

    seen = []
    for sizes in actions:
        for sim, size in zip(refs, sizes):
            # `advance` skips a zero outright rather than placing an empty order
            if size > 0.0:
                sim.order(0, "buy", size)
            elif size < 0.0:
                sim.order(0, "sell", -size)
            sim.step()
        seen.append({
            "position":     np.array([sim.position[0] for sim in refs]),
            "equity":       np.array([sim.equity() for sim in refs]),
            "mark_price":   np.array([sim.mark[0] for sim in refs]),
            "funding_paid": np.array([sum(line["funding"] for line in sim.ledger)
                                      for sim in refs]),
            "bust":         np.array([any(line["terminal"] for line in sim.ledger)
                                      for sim in refs]),
            "done":         np.array([sim.t + 1 >= len(sim.times) for sim in refs]),
        })

    return seen, refs


def observation_disagreement(batch, bars, offsets, window):
    """The window each env observes, against the bars it should be looking at.

    Front-padded with zeros before the episode has `window` bars behind it, which
    is the one thing a warm-up rule can get wrong in a direction nothing notices:
    padding at the END would hand the agent the newest bar in the oldest row.
    """
    seen = batch.observe(window)

    for env, tick in enumerate(offsets):
        want  = np.zeros((window, 5), dtype=np.float64)
        first = max(0, tick - window + 1)
        rows  = np.array([list(bar) for bar in bars[first:tick + 1]], dtype=np.float64)
        want[window - rows.shape[0]:] = rows
        if not np.allclose(seen[env], want, rtol=0.0, atol=1e-9):
            return f"env {env} at bar {tick}: observation is not the window ending there"

    return None


def compare_runs(mine, theirs, cfg, num_envs):
    for step, (a, b) in enumerate(zip(mine, theirs)):
        for env in range(num_envs):
            for key, scale in (("position", 1.0), ("equity", money(cfg)),
                               ("mark_price", 1.0), ("funding_paid", money(cfg))):
                if not close_enough(float(a[key][env]), float(b[key][env]), scale):
                    return (f"step {step + 1} env {env}: {key} "
                            f"{float(a[key][env]):.6f} against {float(b[key][env]):.6f}")
            for key in ("bust", "done"):
                if bool(a[key][env]) != bool(b[key][env]):
                    return (f"step {step + 1} env {env}: {key} "
                            f"{bool(a[key][env])} against {bool(b[key][env])}")

    return None


Outcome = collections.namedtuple(
    "Outcome", "compared agreed disagreed set_aside reached")


def compare(cases, seed):
    rng = random.Random(seed)
    # the inverse cases come after the rest, from a stream of their own, as in the
    # comparison of one engine
    inverse = random.Random(f"inverse {seed}")

    # what the cases actually reached. "0 disagreed" over a hundred flat accounts
    # that never funded and never died is a pass that means nothing, and there is
    # no way to tell one from the other after the fact
    reached = collections.Counter(traded=0, funded=0, busted=0, finished=0)
    aside   = collections.Counter()
    agreed, disagreed = 0, []

    for case in range(drawn(cases)):
        draw  = rng if case < cases else inverse
        bars  = walk(draw, draw.randint(12, 40))
        times = clock(draw, len(bars))

        # tier tables are compared over universes, which `compare_universes` runs
        cfg = dict(config(draw, times, inverse=case >= cases), tiers=None,
                   intrabar=setting(seed, case), **fills(seed, case))

        num_envs = draw.randint(2, 5)
        offsets  = offsets_for(draw, bars, num_envs)
        window   = draw.randint(1, 6)

        # past the end for the earliest env, so a finished episode is asked to
        # keep going and has to sit still rather than wrap or run on
        steps   = len(bars) - min(offsets)
        scale   = 100.0 / cfg["contract"] if cfg["market"] == "inverse" else 1.0
        actions = plan(draw, num_envs, steps, scale)

        try:
            mine, batch = drive_batch(bars, times, cfg, offsets, actions)
            theirs, refs = drive_references(bars, times, cfg, offsets, actions)

            reasons = [set_aside(sim) for sim in refs]
            why     = next((reason for reason in reasons if reason is not None), None)
            if why is not None:
                aside[why] += 1
                continue

            fresh = probatum.Batch(
                np.array([list(bar) for bar in bars], dtype=np.float64),
                num_envs = num_envs,
                market   = cfg["market"],
                quote    = balance(cfg),
                report   = False,
                contract = cfg.get("contract"),
            )
            fresh.reset_at_each(np.array(offsets, dtype=np.int64))
            where = observation_disagreement(fresh, bars, offsets, window)
            if where is None:
                # and again at the end of the run, where the window is full and
                # the envs sit on different bars: at reset alone a window built
                # from the wrong end of the series still looks right at warmup
                ended = [min(off + len(actions), len(bars) - 1) for off in offsets]
                where = observation_disagreement(batch, bars, ended, window)

            if where is None:
                where = compare_runs(mine, theirs, cfg, num_envs)

            last = mine[-1]
            reached["traded"] += any(
                abs(p) > 0.0 for step in mine for p in step["position"]
            )
            reached["funded"] += any(f != 0.0 for f in last["funding_paid"])
            reached["busted"] += bool(last["bust"].any())
            reached["finished"] += bool(last["done"].any())
        except Exception as exc:
            where = f"raised: {type(exc).__name__}: {exc}"

        if where is None:
            agreed += 1
        else:
            disagreed.append((case, cfg, offsets, bars, where))

    return Outcome(agreed + len(disagreed), agreed, disagreed, aside, reached)


def weights_for(rng, kinds, num_envs):
    """A target for every env, each its own row over every instrument: a perpetual
    long or short and spot held or not, at a gross of half to four times equity."""
    rows = []
    for _ in range(num_envs):
        row   = [rng.uniform(0.0 if kind == "spot" else -1.0, 1.0) for kind in kinds]
        total = sum(abs(w) for w in row) or 1.0
        gross = rng.choice([0.5, 1.0, 2.0, 4.0])
        rows.append([gross * w / total for w in row])

    return rows


def env_disagreement(records, sim, end, start):
    """Where an env and its reference first part company on any record written
    before bar `end`, as the comparison over universes reads them, the env's ledger
    counting its rows from `start`, the row it was reset at."""
    fields = universe_differential
    return (log_difference("fill", fields.FILL_FIELDS, before(records["fills"], end),
                           before(sim.fills, end))
            or log_difference("event", fields.EVENT_FIELDS,
                              worded(before(records["order_log"], end)),
                              worded(before(sim.events, end)))
            or log_difference("round trip", fields.TRIP_FIELDS,
                              closed_before(records["trades"], end),
                              closed_before(sim.round_trips(), end))
            or fields.ledger_difference(records["ledger"], sim.ledger,
                                        None if end is None else end - start))


def compare_universes(cases, seed):
    """`cases` random universes from `seed`, each a batch of two to four envs reset at
    their own rows and driven now and then by a target per env, every env against a
    reference opened at the same row of the same series, compared on everything it
    recorded as far as the first rule the engine has yet to take up. An env opened
    part way in marks each instrument at its last close by then, so one with a hole
    on that row or listing after it is the case to reach."""
    rng     = random.Random(f"universe batch {seed}")
    reached = collections.Counter({
        "traded":                                    0,
        "liquidated":                                0,
        "funded":                                    0,
        "a hole on the row an env opened at":        0,
        "an instrument listing after an env opened": 0,
    })
    aside = collections.Counter()
    agreed, disagreed = 0, []

    for case in range(cases):
        times, kinds, columns = universe_differential.universe(rng)
        cfg = universe_differential.config(rng, times, kinds, columns)
        cfg["intrabar"]    = setting(seed, case)
        cfg["liquidation"] = liquidation(seed, case, [rows for rows, _ in cfg["tiers"]])
        cfg.update(fills(seed, case))
        rows     = len(times)
        num_envs = rng.randint(2, 4)
        offsets  = offsets_for(rng, times, num_envs)
        plan     = [weights_for(rng, kinds, num_envs) if rng.random() < 0.4 else None
                    for _ in range(rows - min(offsets))]

        where = None
        try:
            nan     = [math.nan] * 5
            candles = np.array([[nan if column[t] is None else list(column[t])
                                 for column in columns] for t in range(rows)],
                               dtype=np.float64)
            batch = probatum.Batch(
                candles,
                num_envs = num_envs,
                report   = True,
                times    = np.array(times, dtype=np.int64),
                **universe_differential.venue(cfg),
            )
            batch.reset_at_each(np.array(offsets, dtype=np.int64))

            refs = [universe_differential.make_reference(times, kinds, columns, cfg,
                                                         start=start)
                    for start in offsets]

            for weights in plan:
                if weights is not None:
                    batch.target(np.array(weights, dtype=np.float64))
                    for sim, row in zip(refs, weights):
                        sim.target(row)
                batch.advance()
                for sim in refs:
                    sim.step()

            for env, (sim, start) in enumerate(zip(refs, offsets)):
                records = batch.records(env)
                end     = None
                found   = cut(sim)
                if found is not None:
                    rule, end = found
                    aside[rule] += 1

                found = env_disagreement(records, sim, end, start)
                if found is not None:
                    where = f"env {env} opened at row {start}: {found[1]}"
                    break

                logged = records["fills"]
                reached["traded"] += bool(logged)
                reached["liquidated"] += any(fill["reason"] == "liquidation"
                                             for fill in logged)
                reached["funded"] += bool(np.any(records["ledger"]["funding"] != 0.0))
                reached["a hole on the row an env opened at"] += any(
                    column[start] is None and any(column[:start]) for column in columns)
                reached["an instrument listing after an env opened"] += any(
                    not any(column[:start + 1]) and any(column[start + 1:])
                    for column in columns)
        except Exception as exc:
            where = f"raised: {type(exc).__name__}: {exc}"

        if where is None:
            agreed += 1
        else:
            disagreed.append((case, cfg, offsets, kinds, where))

    return Outcome(agreed + len(disagreed), agreed, disagreed, aside, reached)


def render_universes(out, cases):
    lines = [f"\n{out.compared} of {cases} compared, {out.agreed} agreed, "
             f"{len(out.disagreed)} disagreed",
             "compared up to a rule the engine does not have yet: "
             + (", ".join(f"{why} {n}" for why, n in
                          sorted(out.set_aside.items())) or "none"),
             "envs that reached: " + ", ".join(f"{what} {n}" for what, n in
                                              sorted(out.reached.items())) + "\n"]

    for case, cfg, offsets, kinds, why in out.disagreed[:8]:
        lines.append(f"--- case {case} ({', '.join(kinds)}) {why}")
        lines.append(f"    cfg     {cfg}")
        lines.append(f"    offsets {offsets}\n")

    if len(out.disagreed) > 8:
        lines.append(f"... and {len(out.disagreed) - 8} more")

    return "\n".join(lines)


def render(out, cases):
    lines = [f"\n{out.compared} of {drawn(cases)} compared, {out.agreed} agreed, "
             f"{len(out.disagreed)} disagreed",
             "set aside: " + (", ".join(f"{why} {n}" for why, n in
                                        sorted(out.set_aside.items())) or "none"),
             f"cases that traded {out.reached['traded']}, funded "
             f"{out.reached['funded']}, busted {out.reached['busted']}, ran out of "
             f"bars {out.reached['finished']}\n"]

    for case, cfg, offsets, bars, why in out.disagreed[:8]:
        lines.append(f"--- case {case} ({cfg['market']}) {why}")
        lines.append(f"    cfg     {cfg}")
        lines.append(f"    offsets {offsets}")
        shown = [tuple(round(x, 4) for x in b) for b in bars[:6]]
        lines.append(f"    bars    {shown}\n")

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
