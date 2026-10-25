"""Run the environment against the reference environment over the same random universes.

Each environment of a batch is held, step by step, to an `environment.Episode` of the
reference opened at the same row with the same slots: its observation to float32's
precision, its reward and its state to the comparison's tolerance and its flags
exactly. The universes are `universe_differential`'s, with their delistings, fills,
sizes, fees and liquidation drawn as `batch_differential` draws them, and after them
come an inverse perpetual of `differential`'s for every fourth universe; each case
draws the environment's own settings on a stream of its own: its window, candles or
features with a warm-up, each kind of action, a decode of its own now and then, the
log, the Sharpe or an own reward, a leverage, a tolerance, an episode length and a
sample of the instruments. The engine's side is the binding's environment, given the
weights the functions `probatum.env` decodes with turn its actions into, so the
reference decodes each action itself and the two meet only on what they return. A
case is compared up to the bar on which the reference first applied a rule the engine
does not have yet.

  python -m reference.environment_differential [cases] [seed]     (from tests/)
"""

import collections
import math
import random
import sys

import numpy as np

import probatum
from probatum import _env
from . import differential, universe_differential
from .differential import (clock, close_enough, config, cut, fees, fills, liquidation,
                           liquidation_fee, opened, setting, sizes, walk)
from .environment import Episode, onsets

Outcome = collections.namedtuple("Outcome", "compared agreed disagreed set_aside reached")


def drawn_features(draw, columns, width):
    """Features for each instrument's rows, `width` of them, each a value on every bar
    its instrument printed past a warm-up of up to three rows and NaN before it, and
    NaN or a value on a row it printed nothing on, which no observation reads."""
    rows = len(columns[0])
    warm = [[draw.randint(0, 3) for _ in range(width)] for _ in columns]
    out  = []

    for t in range(rows):
        row = []
        for i, column in enumerate(columns):
            row.append([math.nan if t < warm[i][g]
                        or column[t] is None and draw.random() < 0.5
                        else draw.uniform(-2.0, 2.0) for g in range(width)])
        out.append(row)

    return out


def decoding(lows):
    """A decode of its own: each slot's weight read off its action and the state the
    last step left, NaN where the slot already weighs more than half its equity,
    which leaves it unnamed, and never short a spot slot."""
    def decode(actions, state):
        acted   = np.asarray(actions, dtype=np.float64).reshape(len(state.equity), -1)
        k       = len(lows)
        weights = np.clip(acted[:, :k], lows, 1.0) * 0.8 - 0.2 * state.weights
        weights = np.maximum(weights, lows)
        return np.where(np.abs(state.weights) > 0.5, np.nan, weights)

    return decode


def rewarding(state):
    """An own reward that reads every field of the state."""
    return (state.log_return - 0.5 * state.drawdown ** 2 - 0.001 * state.turnover
            - state.fees - state.funding + 0.01 * state.leverage
            - 1.0 * state.liquidated + 0.001 * np.nansum(state.weights, axis=1)
            + 1e-6 * np.abs(state.positions).sum(axis=1)
            + 1e-9 * (state.equity - state.previous_equity))


def acting(draw, kind, k, lows):
    """One environment's action of `kind` over `k` slots, past the space's bounds
    now and then, so both sides clip."""
    if kind == "box":
        return [draw.uniform(-1.3, 1.3) for _ in range(k)]

    if kind == "simplex":
        return [0.0 if draw.random() < 0.2 else draw.uniform(-0.2, 1.2)
                for _ in range(k + 1)]

    if kind == "discrete":
        return [draw.randint(0, 1 if low == 0.0 else 2) for low in lows]

    return [[0.0 if draw.random() < 0.25 else draw.uniform(-1.2, 1.2),
             draw.uniform(-0.1, 1.1)] for _ in range(k)]


def universe_case(rng, seed, case):
    """A universe of `universe_differential`'s, its delistings, fills, sizes, fees and
    liquidation drawn as the batch's comparison draws them: its clock, kinds, columns
    and delisting times, its config and opening balance, the engine's settings and
    the reference opened at a row."""
    times, kinds, columns = universe_differential.universe(rng)
    columns, delist = universe_differential.delistings(seed, case, times, columns)
    cfg = universe_differential.config(rng, times, kinds, columns)

    cfg["delist"]          = delist
    cfg["intrabar"]        = setting(seed, case)
    cfg["liquidation"]     = liquidation(seed, case, [rows for rows, _ in cfg["tiers"]])
    cfg["liquidation_fee"] = liquidation_fee(seed, case)
    cfg.update(fills(seed, case))
    cfg.update(sizes(seed, case, cfg["quote"], universe_differential.opens(columns)))
    cfg.update(fees(seed, case, kinds))

    balance = cfg["balance"]
    return dict(
        times=times, kinds=kinds, columns=columns, delist=delist, cfg=cfg,
        balance=sum(balance.values()) if isinstance(balance, dict) else balance,
        venue=lambda: universe_differential.venue(cfg),
        make=lambda start: universe_differential.make_reference(times, kinds, columns,
                                                               cfg, start=start),
    )


def inverse_case(rng, seed, case):
    """An inverse perpetual of `differential`'s, its fills, sizes and liquidation
    drawn as its comparison draws them, in the same form."""
    bars  = walk(rng, rng.randint(8, 40))
    times = clock(rng, len(bars))
    cfg   = config(rng, times, inverse=True)

    cfg["intrabar"]        = setting(seed, case)
    cfg["liquidation"]     = liquidation(seed, case, [cfg["tiers"]])
    cfg["liquidation_fee"] = liquidation_fee(seed, case)
    [cfg["tiers"]] = opened(seed, case, [cfg["tiers"]])
    cfg.update(fills(seed, case))
    cfg.update(sizes(seed, case, cfg["quote"], [bars[0].open], cfg["contract"]))

    return dict(
        times=times, kinds=["inverse"], columns=[bars], delist=[None], cfg=cfg,
        balance=cfg["quote"], venue=lambda: differential.venue(cfg),
        make=lambda start: differential.make_reference(bars, cfg, times, start=start),
    )


def compare(cases, seed):
    """`cases` random universes from `seed`, and an inverse perpetual for every fourth
    of them, each a batch of two to four environments opened at rows and on slots of
    their own and stepped until every episode ends or the bars run out, every
    environment against a reference episode, compared on what each step returned as
    far as the first rule the engine has yet to take up."""
    rng = random.Random(f"universe environment {seed}")
    # the inverse cases come after the rest, from a stream of their own, so adding
    # them left every other case as it was
    inverse = random.Random(f"inverse environment {seed}")

    reached = collections.Counter({
        "liquidated":               0,
        "cut at its length":        0,
        "cut at the last bar":      0,
        "a hidden slot":            0,
        "an order resting":         0,
        "features":                 0,
        "a sample":                 0,
        "the sharpe reward":        0,
        "an own reward":            0,
        "a decode of its own":      0,
        "an inverse perpetual":     0,
        "a normalised observation": 0,
    })
    aside = collections.Counter()
    agreed, disagreed = 0, []

    for case in range(cases + cases // 4):
        found = (universe_case(rng, seed, case) if case < cases
                 else inverse_case(inverse, seed, case))
        times, kinds, columns = found["times"], found["kinds"], found["columns"]
        delist, cfg = found["delist"], found["cfg"]

        draw   = random.Random(f"environment {seed} {case}")
        width  = len(columns)
        reward = draw.choice(["log", "log", "sharpe", "own"])
        # the Sharpe needs a deviation, which a window of one bar does not hold
        window = draw.randint(2 if reward == "sharpe" else 1, 4)
        kind   = draw.choice(["box", "box", "simplex", "discrete", "orders"])
        if kind == "orders":
            # a limit rests whole ticks from the close, so every instrument gets one
            opens = universe_differential.opens(columns)
            for rule, price in zip(cfg["rules"], opens):
                rule.update(tick=price / draw.choice([200.0, 10_000.0]), sigfigs=None,
                            decimals=None)

        features = (drawn_features(draw, columns, draw.randint(1, 3))
                    if draw.random() < 0.4 else None)

        # the scale comes from a stream of its own, so drawing it left every other
        # case as it was
        scaling = random.Random(f"scale {seed} {case}")
        scale   = ([(scaling.uniform(-1.0, 1.0), scaling.uniform(0.1, 3.0))
                    for _ in range(5 if features is None else len(features[0][0]))]
                   if scaling.random() < 0.3 else None)

        one_kind = len(set(each == "spot" for each in kinds)) == 1
        k        = draw.randint(1, width) if one_kind and draw.random() < 0.4 else width
        sampled  = k < width
        lows     = np.array([0.0 if kinds[0] == "spot" else -1.0] * k if sampled
                            else [0.0 if each == "spot" else -1.0 for each in kinds])
        decode   = (decoding(lows) if kind != "orders" and draw.random() < 0.25
                    else None)
        settings = dict(
            columns        = columns,
            features       = features,
            window         = window,
            actions        = kind,
            max_leverage   = draw.choice([0.5, 1.0, 2.0, 5.0]),
            max_offset     = draw.choice([0, 5, 20]),
            reward         = reward,
            tolerance      = draw.choice([0.0, 0.0, 0.02]),
            episode_length = draw.choice([None, None, 3, 7]),
            balance        = found["balance"],
            scale          = scale,
        )

        e = onsets(columns, features)

        def shown(i, t):
            return (e[i] is not None and t - window + 1 >= e[i]
                    and columns[i][t] is not None
                    and (delist[i] is None or times[t] < delist[i]))

        rows   = len(times)
        need   = k if sampled else 1
        starts = [t for t in range(rows - 1)
                  if sum(shown(i, t) for i in range(width)) >= need]
        if not starts:
            aside["no start"] += 1
            continue

        num_envs = draw.randint(2, 4)
        at       = [draw.choice(starts) for _ in range(num_envs)]
        slots    = [sorted(draw.sample([i for i in range(width) if shown(i, t)], k))
                    if sampled else list(range(width)) for t in at]

        where = None
        try:
            where = run(found, settings, at, slots, lows, decode, draw, reached, aside)
        except Exception as exc:
            where = f"raised: {type(exc).__name__}: {exc}"

        reached["features"] += features is not None
        reached["a sample"] += sampled
        reached["the sharpe reward"] += reward == "sharpe"
        reached["an own reward"] += reward == "own"
        reached["a decode of its own"] += decode is not None
        reached["an inverse perpetual"] += kinds == ["inverse"]
        reached["a normalised observation"] += scale is not None

        if where is None:
            agreed += 1
        else:
            disagreed.append((case, cfg, at, slots, settings, where))

    return Outcome(agreed + len(disagreed), agreed, disagreed, aside, reached)


def run(case, settings, at, slots, lows, decode, draw, reached, aside):
    """`case` through both sides, its environments opened at the rows `at` on
    `slots`, returning where they first part company, or None where they agree
    throughout."""
    columns, window = settings["columns"], settings["window"]
    times, cfg  = case["times"], case["cfg"]
    rows        = len(times)
    num_envs, k = len(at), len(slots[0])

    kind    = settings["actions"]
    nan     = [math.nan] * 5
    candles = np.array([[nan if column[t] is None else list(column[t])
                         for column in columns] for t in range(rows)], dtype=np.float64)
    if case["kinds"] == ["inverse"]:
        candles = candles[:, 0]

    features = settings["features"]
    reward   = settings["reward"]
    batch    = probatum.Batch(
        candles,
        num_envs = num_envs,
        report   = False,
        times    = np.array(times, dtype=np.int64),
        **case["venue"](),
    )

    env = probatum._probatum.Environment(
        batch, window, onsets(columns, features), k,
        [rule.get("tick") or 0.0 for rule in cfg["rules"]],
        reward         = reward,
        eta            = 2.0 / (window + 1),
        features       = None if features is None else np.array(features,
                                                                dtype=np.float64),
        max_leverage   = settings["max_leverage"],
        max_offset     = float(settings["max_offset"]),
        tolerance      = settings["tolerance"],
        orders         = kind == "orders",
        episode_length = settings["episode_length"],
        floor          = 0.01 * settings["balance"],
        scale          = settings["scale"],
    )

    refs = [Episode(case["make"], start, own, settings)
            for start, own in zip(at, slots)]
    env.reset(np.ones(num_envs, dtype=bool), np.array(at, dtype=np.int64),
              np.array(slots, dtype=np.int64))

    out    = np.zeros((num_envs, env.size), dtype=np.float32)
    active = np.ones(num_envs, dtype=bool)
    found  = observed(env.observe(out), refs, active, "the opening observation")
    if found:
        return found

    own = rewarding if reward == "own" else None
    for step in range(rows):
        if not active.any():
            break

        acted = [acting(draw, kind, k, lows) for _ in range(num_envs)]
        state = _env.State(env.state())
        given = np.array(acted, dtype=np.float64)
        if decode is not None:
            decided = decode(given, state)
        elif kind == "box":
            decided = _env._box(given, lows, env.shown(), settings["max_leverage"])
        elif kind == "simplex":
            decided = _env._simplex(given, env.shown())
        elif kind == "discrete":
            decided = _env._discrete(given, lows, settings["max_leverage"])
        else:
            decided = given

        hidden = ~env.shown()
        rewards, terminated, truncated = env.step(decided, active)
        if own is not None:
            rewards = own(_env.State(env.state()))

        values = env.state()
        for i, ref in enumerate(refs):
            if not active[i]:
                continue

            reached["a hidden slot"] += bool(hidden[i].any())
            obs, got, ended, stopped = ref.step(acted[i], decode=decode, reward=own)
            found = cut(ref.sim)
            if found is not None and found[1] <= ref.sim.t:
                aside[found[0]] += 1
                active[i] = False
                continue

            said = f"env {i} opened at row {at[i]}, step {step + 1}"
            if not close_enough(rewards[i], got):
                return f"{said}: reward {rewards[i]:.9f} against {got:.9f}"

            if (bool(terminated[i]), bool(truncated[i])) != (ended, stopped):
                return (f"{said}: flags {bool(terminated[i])}, {bool(truncated[i])} "
                        f"against {ended}, {stopped}")

            for name in _env.STATE:
                mine, theirs = values[name][i], getattr(ref.last, name)[0]
                if np.any(np.isnan(np.asarray(mine, dtype=np.float64))
                          != np.isnan(np.asarray(theirs, dtype=np.float64))) or not all(
                        close_enough(float(a), float(b)) for a, b
                        in zip(np.ravel(mine), np.ravel(theirs))):
                    return f"{said}: state {name} {mine} against {theirs}"

            if ended and ref.sim.liquidations > ref.liquidations:
                reached["liquidated"] += 1
            if stopped:
                reached["cut at the last bar" if ref.sim.t == rows - 1
                        else "cut at its length"] += 1
            if kind == "orders":
                reached["an order resting"] += any(ref.sim.resting(j) for j in ref.slots)

        found = observed(env.observe(out), refs, active, f"step {step + 1}")
        if found:
            return found

        active &= ~(terminated | truncated)

    return None


def observed(rows, refs, active, when):
    """Where an active environment's observation `rows` first parts from its
    reference's, or None."""
    for i, (row, ref) in enumerate(zip(rows, refs)):
        if not active[i]:
            continue

        theirs = np.array(ref.observe(), dtype=np.float64)
        if row.shape != theirs.shape:
            return f"env {i}, {when}: {row.shape[0]} values against {theirs.shape[0]}"

        for at, (a, b) in enumerate(zip(row.astype(np.float64), theirs)):
            if not close_enough(a, float(np.float32(b)), 1.0):
                return f"env {i}, {when}: value {at} {a:.9g} against {b:.9g}"

    return None


def render(out, cases):
    lines = [f"\n{out.compared} of {cases + cases // 4} compared, {out.agreed} agreed, "
             f"{len(out.disagreed)} disagreed",
             "compared up to a rule the engine does not have yet, or set aside: "
             + (", ".join(f"{why} {n}" for why, n in sorted(out.set_aside.items()))
                or "none"),
             "reached: " + ", ".join(f"{what} {n}"
                                     for what, n in sorted(out.reached.items())) + "\n"]

    for case, cfg, at, slots, settings, why in out.disagreed[:8]:
        shown = {name: value for name, value in settings.items()
                 if name not in ("columns", "features")}
        lines.append(f"--- case {case} ({', '.join(cfg['kinds'])}) {why}")
        lines.append(f"    settings {shown}, features "
                     f"{'given' if settings['features'] else 'none'}")
        lines.append(f"    opened at {at}, slots {slots}\n")

    if len(out.disagreed) > 8:
        lines.append(f"... and {len(out.disagreed) - 8} more")

    return "\n".join(lines)


def main(argv=None):
    argv  = sys.argv[1:] if argv is None else argv
    cases = int(argv[0]) if len(argv) > 0 else 200
    seed  = int(argv[1]) if len(argv) > 1 else 20261024
    out   = compare(cases, seed)

    print(render(out, cases))
    return 1 if out.disagreed else 0


if __name__ == "__main__":
    sys.exit(main())
