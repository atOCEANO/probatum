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
    """The first bar the two disagree on, on any record written before the first
    rule the engine has yet to take up, as the comparison reads it, or None. A
    shrink that makes the run raise has wandered out of the case, so it is not a
    disagreement either."""
    try:
        rows, records, sim = D.run(bars, times, cfg, actions)
    except Exception:
        return None

    found = D.cut(sim)
    end   = None if found is None else found[1]
    found = D.disagreement(rows, records, sim, cfg, end)
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


COSTS = ("fees", "spread", "slippage", "impact", "funding", "liquidation", "dust")


def on_bar(log, t):
    return [{k: rounded(v) for k, v in entry.items()} for entry in log
            if entry["bar"] == t]


def trace(bars, times, cfg, actions):
    """Bar by bar, both sides, so the exact step that parts them is visible: each
    side's account at the close, and the events and fills it recorded on the bar."""
    _, records, sim = D.run(bars, times, cfg, actions)
    ledger = records["ledger"]
    rows   = []

    for t in range(1, len(bars)):
        theirs = sim.ledger[t]
        held   = theirs["instruments"][0]
        rows.append(dict(
            bar=t,
            ohlcv=[round(x, 4) for x in bars[t]],
            placed=actions[t - 1],
            engine=dict(position=rounded(float(ledger["position"][t])),
                        equity=rounded(float(ledger["equity"][t])),
                        entry=rounded(float(ledger["entry"][t])),
                        spot_cash=rounded(float(ledger["spot_cash"][t])),
                        margin_cash=rounded(float(ledger["margin_cash"][t])),
                        costs={k: rounded(float(ledger[k][t])) for k in COSTS},
                        events=on_bar(records["order_log"], t),
                        fills=on_bar(records["fills"], t)),
            reference=dict(position=rounded(held["position"]),
                           equity=rounded(theirs["equity"]),
                           entry=rounded(held["entry"]),
                           spot_cash=rounded(theirs["spot_cash"]),
                           margin_cash=rounded(theirs["margin_cash"]),
                           costs={k: rounded(theirs[k]) for k in COSTS},
                           events=on_bar(sim.events, t),
                           fills=on_bar(sim.fills, t)),
        ))

    return rows, records, sim


def main():
    seed  = int(sys.argv[1])
    cases = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    rng   = random.Random(seed)

    # the inverse cases come after the rest, from their own stream, as the comparison
    # draws them
    inverse = random.Random(f"inverse {seed}")

    for case in range(D.drawn(cases)):
        draw  = rng if case < cases else inverse
        bars  = D.walk(draw, draw.randint(8, 40))
        times = D.clock(draw, len(bars))
        cfg   = D.config(draw, times, inverse=case >= cases)

        cfg["intrabar"]        = D.setting(seed, case)
        cfg["liquidation"]     = D.liquidation(seed, case, [cfg["tiers"]])
        cfg["liquidation_fee"] = D.liquidation_fee(seed, case)
        cfg.update(D.fills(seed, case))
        opens = [bars[0].open]
        cfg.update(D.sizes(seed, case, cfg["quote"], opens, cfg.get("contract")))
        actions = D.plan(draw, bars, cfg)
        if diverges(bars, times, cfg, actions) is None:
            continue

        bars, times, actions, at = shrink(bars, times, cfg, actions)
        rows, records, sim = trace(bars, times, cfg, actions)
        out = dict(
            seed=seed, case=case, config=cfg, diverges_at_row=at,
            bars=[list(b) for b in bars], times=times, actions=actions, rows=rows,
            engine_trades=[{k: rounded(v) for k, v in trade.items()}
                           for trade in records["trades"]],
            reference_trades=[{k: rounded(v) for k, v in trade.items()}
                              for trade in sim.round_trips()],
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
