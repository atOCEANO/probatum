"""Shrink each disagreement to the smallest case that still shows it, and trace it.

The differential run says WHERE the two simulators part company. This says as
little as possible around that point, so the question can be argued from the
Specification rather than from a forty bar random walk.

  python -m reference.trace <seed> [cases]     (from tests/)

It writes the shrunk case to /diff, which no stage creates, so that has to be a
mount or the run ends on a missing directory after the search it just did:

  docker build --target test-slow -t probatum-slow .
  docker run --rm -v "${PWD}/diff:/diff" -w /tests probatum-slow \
      python -m reference.trace 42
"""

import json
import random
import sys

from . import differential as D


def diverges(bars, times, cfg, actions):
    """The first bar the two disagree on, on any record, or None. A case outside what
    both model is not a disagreement, so a shrink that wanders out of it is not one
    either."""
    try:
        rows, engine, sim = D.run(bars, times, cfg, actions)
    except Exception:
        return None

    if D.set_aside(sim) is not None:
        return None

    found = D.disagreement(rows, engine, sim, cfg)
    return None if found is None else found[0]


def shrink(bars, times, cfg, actions):
    """Drop bars off the tail and actions from anywhere, while it still diverges."""
    at = diverges(bars, times, cfg, actions)
    if at is None:
        return bars, times, actions, None

    # tail first: nothing after the divergence can be causing it
    keep = at + 1
    while keep < len(bars):
        if diverges(bars[:keep], times[:keep], cfg, actions[:keep - 1]) is not None:
            bars, times, actions = bars[:keep], times[:keep], actions[:keep - 1]
            break
        keep += 1

    # then blank one action at a time, keeping any blanking that preserves it
    changed = True
    while changed:
        changed = False
        for i in range(len(actions)):
            if actions[i] is None:
                continue
            trial = list(actions)
            trial[i] = None
            if diverges(bars, times, cfg, trial) is not None:
                actions = trial
                changed = True

    return bars, times, actions, diverges(bars, times, cfg, actions)


def rounded(value):
    return round(value, 6) if isinstance(value, float) else value


def trace(bars, times, cfg, actions):
    """Bar by bar, both sides, so the exact step that parts them is visible."""
    engine = D.make_engine(bars, cfg, times)
    sim    = D.make_reference(bars, cfg, times)
    engine.reset()

    mine, theirs, rows = [], [], []
    for t, action in enumerate(actions):
        D.place(engine, sim, action, bars[t].close, mine, theirs)
        state = engine.step()
        sim.step()

        ledger = sim.ledger[-1]
        rows.append(dict(
            bar=sim.t,
            ohlcv=[round(x, 4) for x in bars[sim.t]],
            placed=action,
            engine=dict(position=rounded(state["position"]),
                        equity=rounded(state["equity"]),
                        entry=rounded(state["avg_entry"]),
                        spot_cash=rounded(state["spot_cash"]),
                        margin_cash=rounded(state["margin_cash"]),
                        funding=rounded(state["funding_paid"]),
                        bust=engine.is_bust(),
                        live=list(mine)),
            reference=dict(position=rounded(sim.position[0]),
                           equity=rounded(sim.equity()),
                           entry=rounded(sim.entry[0]),
                           spot_cash=rounded(sim.spot_cash),
                           margin_cash=rounded(sim.margin_cash),
                           costs={k: rounded(ledger[k]) for k in (
                               "fees", "slippage", "funding", "liquidation")},
                           liquidations=sim.liquidations,
                           live=list(theirs),
                           resting=[dict(id=o["id"], kind=o["kind"], side=o["side"],
                                         size=rounded(o["size"]),
                                         filled=rounded(o["filled"]),
                                         price=o["price"], trigger=o["trigger"])
                                    for o in sim.resting()]),
        ))
        if engine.done():
            break

    return rows, engine, sim


def main():
    seed  = int(sys.argv[1])
    cases = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    rng   = random.Random(seed)

    for case in range(cases):
        bars    = D.walk(rng, rng.randint(8, 40))
        times   = D.clock(rng, len(bars))
        cfg     = D.config(rng, times)
        actions = D.plan(rng, bars, cfg)
        if diverges(bars, times, cfg, actions) is None:
            continue

        bars, times, actions, at = shrink(bars, times, cfg, actions)
        rows, engine, sim = trace(bars, times, cfg, actions)

        out = dict(
            seed=seed, case=case, config=cfg, diverges_at_row=at,
            bars=[list(b) for b in bars], times=times, actions=actions, rows=rows,
            engine_trades=[{k: rounded(v) for k, v in trade.items()}
                           for trade in (engine.trades() or [])],
            reference_fills=[{k: rounded(v) for k, v in fill.items()}
                             for fill in sim.fills],
            reference_events=sim.events,
        )

        path = f"/diff/case-seed{seed}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)
        print(f"seed {seed} case {case}: shrank to {len(bars)} bars, "
              f"{sum(1 for a in actions if a)} orders, diverges at row {at} -> {path}")
        return 0

    print(f"seed {seed}: no disagreement in {cases} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
