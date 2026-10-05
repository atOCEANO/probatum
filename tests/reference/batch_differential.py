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
import random
import sys

import numpy as np

import probatum
from .differential import (HOUR, close_enough, config, funding_of, make_reference,
                           set_aside, walk)


def offsets_for(rng, bars, num_envs):
    # at least one bar to step into, and never all the same: two envs on the same
    # offset run the same episode, and then a batch that broadcast env 0 over the
    # rest of them would agree with the reference
    top   = len(bars) - 2
    drawn = [rng.randint(0, top) for _ in range(num_envs)]
    if len(set(drawn)) == 1 and top > 0:
        drawn[0] = (drawn[0] + 1) % (top + 1)
    return drawn


def plan(rng, num_envs, steps):
    """A signed size per env per step: positive buys, negative sells, zero holds,
    which is exactly what `advance` reads."""
    return [
        [0.0 if rng.random() < 0.45 else rng.uniform(-40.0, 40.0)
         for _ in range(num_envs)]
        for _ in range(steps)
    ]


def drive_batch(bars, cfg, offsets, actions):
    array = np.array([list(bar) for bar in bars], dtype=np.float64)
    batch = probatum.Batch(
        array,
        num_envs          = len(offsets),
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
        report            = False,
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


def drive_references(bars, cfg, offsets, actions):
    """One reference per env over the bars from its offset. Funding is passed as the
    venue's own events, stamped where the whole series funds, since a schedule of
    every n bars counts from each reference's first row."""
    times  = [k * HOUR for k in range(len(bars))]
    events = None

    if funding_of(cfg) != "none":
        n      = cfg["funding_interval"]
        events = [(times[r] + HOUR, None, cfg["funding_rate"])
                  for r in range(n, len(bars), n)]

    refs = [make_reference(bars[off:], cfg, funding=events, times=times[off:])
            for off in offsets]
    seen, ran_out = [], [set() for _ in offsets]

    for sizes in actions:
        for env, (sim, size) in enumerate(zip(refs, sizes)):
            # `advance` skips a zero outright rather than placing an empty order
            if size > 0.0:
                sim.order(0, "buy", size)
            elif size < 0.0:
                sim.order(0, "sell", -size)

            before = sim.liquidations
            sim.step()
            if sim.liquidations != before or sim.clipped:
                ran_out[env].add(offsets[env] + sim.t)

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

    return seen, refs, ran_out


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
            for key, scale in (("position", 1.0), ("equity", cfg["quote"]),
                               ("mark_price", 1.0), ("funding_paid", cfg["quote"])):
                if not close_enough(float(a[key][env]), float(b[key][env]), scale):
                    return (f"step {step + 1} env {env}: {key} "
                            f"{float(a[key][env]):.6f} against {float(b[key][env]):.6f}")
            for key in ("bust", "done"):
                if bool(a[key][env]) != bool(b[key][env]):
                    return (f"step {step + 1} env {env}: {key} "
                            f"{bool(a[key][env])} against {bool(b[key][env])}")

    return None


def deaths(mine, offsets, last):
    """The bars each env's engine died on, where its bust flag first rose."""
    out    = [set() for _ in offsets]
    before = [False] * len(offsets)
    for step, row in enumerate(mine):
        for env, bust in enumerate(row["bust"]):
            if bust and not before[env]:
                out[env].add(min(offsets[env] + step + 1, last))
            before[env] = bool(bust)

    return out


Outcome = collections.namedtuple(
    "Outcome", "compared agreed disagreed set_aside reached")


def compare(cases, seed):
    rng = random.Random(seed)

    # what the cases actually reached. "0 disagreed" over a hundred flat accounts
    # that never funded and never died is a pass that means nothing, and there is
    # no way to tell one from the other after the fact
    reached = collections.Counter(traded=0, funded=0, busted=0, finished=0)
    aside   = collections.Counter()
    agreed, disagreed = 0, []

    for case in range(cases):
        bars     = walk(rng, rng.randint(12, 40))
        cfg      = config(rng)
        num_envs = rng.randint(2, 5)
        offsets  = offsets_for(rng, bars, num_envs)
        window   = rng.randint(1, 6)

        # past the end for the earliest env, so a finished episode is asked to
        # keep going and has to sit still rather than wrap or run on
        steps   = len(bars) - min(offsets)
        actions = plan(rng, num_envs, steps)

        try:
            mine, batch = drive_batch(bars, cfg, offsets, actions)
            theirs, refs, ran_out = drive_references(bars, cfg, offsets, actions)
            died    = deaths(mine, offsets, len(bars) - 1)
            reasons = [set_aside(sim, cfg, ran_out[env] | died[env])
                       for env, sim in enumerate(refs)]
            why = next((reason for reason in reasons if reason is not None), None)
            if why is not None:
                aside[why] += 1
                continue

            fresh = probatum.Batch(
                np.array([list(bar) for bar in bars], dtype=np.float64),
                num_envs = num_envs,
                market   = cfg["market"],
                quote    = cfg["quote"],
                report   = False,
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


def render(out, cases):
    lines = [f"\n{out.compared} of {cases} compared, {out.agreed} agreed, "
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
