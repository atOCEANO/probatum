"""A clone or a pickle steps on as its original would.

An Engine, a Batch and an environment clone as they stand, the bars shared and
everything else copied, so a clone given the same actions returns the same steps to the
bit, on a venue whose hidden events a stream decides and across the episodes the
copied streams draw next; a fork of one environment into many repeats it in every
copy; and an Engine or a Batch pickled into a process of the same version steps on there
as it would here (Specification, Clones and pickles; ADR 0040).
"""

import copy
import multiprocessing
import pickle
import re

import gymnasium
import numpy as np
import pytest

import probatum
import probatum.rl
from probatum import ProbatumTypeError, ProbatumValueError
from test_an_episode_runs_again_as_a_backtest_from_its_start import (VENUE, universe,
                                                                     walked)

# a venue whose hidden events each environment's stream decides, so a clone that drew
# afresh would part from its original
RANDOM = VENUE.replace(intrabar="random")
MODES  = ["next_step", "same_step", "disabled"]


def built(**settings):
    """Three environments on `universe`'s bars at `RANDOM`, episodes of 12 steps."""
    given = {**dict(window=8, num_envs=3, seed=4, episode_length=12), **settings}
    return probatum.env(universe(), RANDOM, **given)


def played(env, actions):
    """What each of ``actions`` returns from ``env``: its observations, rewards, flags
    and each environment's episode, step and start, the disabled mode's ended
    environments reset after the step that ended them."""
    seen = []
    for row in actions:
        obs, reward, terminated, truncated, info = env.step(row)
        parts = list(obs.values()) if isinstance(obs, dict) else [obs]
        seen.append(parts + [reward, terminated, truncated, info["episode"],
                             info["step"], info["start"]])
        ended = terminated | truncated
        if env.autoreset_mode.name == "DISABLED" and ended.any():
            env.reset(options={"reset_mask": ended})

    return seen


def same(steps, others):
    """Whether two runs of `played` hold the same dtypes, shapes and bytes."""
    return len(steps) == len(others) and all(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        for step, other in zip(steps, others) for a, b in zip(step, other))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("settings", [
    dict(),
    dict(layout="dict", sample=1),
    dict(actions="orders"),
], ids=["candles", "sampled", "orders"])
def test_a_clone_of_environments_steps_on_as_they_do(settings, mode):
    env = built(autoreset=mode, **settings)
    env.action_space.seed(3)
    env.reset()
    actions = [env.action_space.sample() for _ in range(40)]

    played(env, actions[:10])
    env.np_random.random()
    twin = env.clone()

    # its spaces are its own, their generators with them, and their bounds shared
    for mine, theirs in ((twin.action_space, env.action_space),
                         (twin.single_action_space, env.single_action_space),
                         (twin.observation_space, env.observation_space)):
        assert np.array_equal(gymnasium.spaces.flatten(mine, mine.sample()),
                              gymnasium.spaces.flatten(theirs, theirs.sample()))

    assert twin.action_space.low is env.action_space.low
    assert twin.np_random.random() == env.np_random.random()

    # the clone steps first, so any state it shared would move the original
    ahead = played(twin, actions[10:])

    assert same(played(env, actions[10:]), ahead)
    assert twin.provenance == env.provenance and twin.provenance is not env.provenance


@pytest.mark.parametrize("mode", ["next_step", "same_step"])
def test_a_fork_repeats_its_environment_in_every_copy(mode):
    env = built(autoreset=mode)
    env.action_space.seed(3)
    env.reset()
    actions = [env.action_space.sample() for _ in range(40)]

    played(env, actions[:10])
    fork = env.clone(env=1, n=3)

    assert fork.num_envs == 3 and fork.action_space.shape == (3, 2)
    assert fork.provenance["forked"] == 1 and fork.provenance["settings"]["num_envs"] == 3

    ahead  = played(env, actions[10:])
    copies = played(fork, [row[[1, 1, 1]] for row in actions[10:]])

    for step, each in zip(ahead, copies):
        assert all(np.array_equal(part, np.stack([whole[1]] * 3))
                   for whole, part in zip(step, each))


def test_a_clone_keeps_the_transitions_and_a_fork_starts_with_none():
    env = built(record="transitions")
    env.action_space.seed(3)
    env.reset()

    for _ in range(5):
        env.step(env.action_space.sample())

    twin, fork = env.clone(), env.clone(env=2, n=2)

    assert len(twin.transitions) == len(env.transitions) == 15
    assert len(fork.transitions) == 0

    row, other = env.action_space.sample(), env.action_space.sample()

    # the clone moves first, its own way, so an observation or a count it shared would
    # show
    twin.reset()
    twin.step(other)
    env.step(row)
    fork.step(row[[2, 2]])

    held = env.transitions
    assert np.array_equal(held["observation"][-3:], held["next_observation"][-6:-3])
    assert not held["episode"].any() and (twin.transitions["episode"][15:] == 1).all()
    for name in held.columns:
        assert np.array_equal(twin.transitions[name][:15], held[name][:15])

    # the last row is environment 2's, which each copy of the fork took
    for name in ("observation", "action", "reward", "next_observation"):
        assert np.array_equal(fork.transitions[name], held[name][[-1, -1]])


def test_a_single_environment_s_clone_steps_on_as_it_does():
    single = probatum.rl.Env(universe(), RANDOM, window=8, seed=4, episode_length=12)
    single.action_space.seed(3)
    single.reset(seed=4)
    actions = [single.action_space.sample() for _ in range(20)]

    for action in actions[:5]:
        single.step(action)

    twin = single.clone()

    # the generator Gymnasium keeps from the seeded reset is the copy's own
    assert twin.np_random.random() == single.np_random.random()

    ahead = [twin.step(action) for action in actions[5:]]

    for (obs, *rest), (seen, *more) in zip([single.step(a) for a in actions[5:]], ahead):
        assert obs.tobytes() == seen.tobytes() and rest[:3] == more[:3]


def test_a_single_environment_gymnasium_made_shares_what_it_was_made_with():
    features = np.ones((len(universe()), 2, 3))
    made     = gymnasium.make("probatum/Trading-v0", data=universe(), venue=RANDOM,
                              window=8, features=features)

    # the spec make keeps holds what it was given, which a copy shares, not copies
    assert made.unwrapped.clone().spec.kwargs["features"] is features


def traded(engine, steps):
    """``engine`` stepped ``steps`` times on a target that flips its long each close,
    and what its run recorded."""
    for k in range(steps):
        engine.target({"BTCUSDT": 0.3 * (-1.0) ** k, "ETHUSDT": 0.1})
        engine.step()

    return engine.result()


def test_an_engine_s_clone_steps_on_as_it_does_with_handles_of_its_own():
    engine = probatum.Engine(universe(), VENUE)
    engine.reset()
    resting = engine["BTCUSDT"].buy(0.01, limit=1.0)
    engine["ETHUSDT"].sell(0.1)
    engine.step()

    twin = engine.clone()
    mine = twin.orders[0]

    assert [order.id for order in twin.orders] == [order.id for order in engine.orders]
    assert mine is not resting and mine.price == resting.price

    with pytest.raises(ProbatumValueError, match=re.escape(", of another run; pass an "
                                                           "Order this engine's verbs "
                                                           "returned")):
        twin.cancel(resting)

    with pytest.raises(ProbatumValueError, match=re.escape(", of a run this Result does "
                                                           "not hold")):
        twin.result().explain(resting)

    assert twin.result().explain(mine) == engine.result().explain(resting)
    assert twin.cancel(mine) and engine.cancel(resting)

    # a copy is a clone, and an order it places is its own, which its original never
    # lists
    apart  = copy.copy(engine)
    placed = apart["BTCUSDT"].buy(0.01, limit=1.0)

    assert engine.orders == [] and apart.orders == [placed] and apart.cancel(placed)

    ahead, kept = traded(twin, 20), traded(engine, 20)

    assert np.array_equal(ahead.equity, kept.equity) and ahead.trades == kept.trades
    assert ahead.fills == kept.fills and ahead.orders == kept.orders

    # a clone gone its own way keeps its own equity curve and round trips, read before
    # the original reads its own
    for k in range(20):
        apart.target({"BTCUSDT": 0.5 * (-1.0) ** k, "ETHUSDT": 0.4})
        apart.step()

    curve, trips = apart.equity_curve, apart.trades
    engine.equity_curve, engine.trades
    held = apart.result()

    assert np.array_equal(curve, held.equity) and len(trips["pnl"]) > 0
    assert np.array_equal(trips["pnl"], held.trades["pnl"][~held.trades["open"]])


def stepped_on(engine, batch, steps):
    """``steps`` closes of ``engine``'s and ``batch``'s, wherever the pickles reached
    them, its resting order cancelled through its own handle, and the equities they
    left."""
    assert engine.cancel(engine.orders[0])

    held = traded(engine, steps)
    for _ in range(steps):
        batch.target(np.array([[0.2, -0.1], [-0.3, 0.4]]))
        batch.step_all()

    return held.equity, np.array(engine.equity_curve), batch.equities()


def test_an_engine_and_a_batch_pickled_into_another_process_step_on_there():
    engine = probatum.Engine(universe(), VENUE)
    engine.reset()
    engine["BTCUSDT"].buy(0.01, limit=1.0)
    engine.step()

    rng   = np.random.default_rng(1)
    batch = probatum.Batch(
        np.stack([walked(100.0, 60, rng), walked(50.0, 60, rng)],
                 axis=1),
        2,
        market = "perp",
        report = True,
    )
    batch.reset_all()
    batch.step_all()

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        there = pool.apply(stepped_on, (engine, batch, 12))

    here = stepped_on(engine, batch, 12)

    assert all(np.array_equal(a, b) for a, b in zip(there, here))


def test_an_environment_does_not_pickle_and_a_fork_it_cannot_make_is_refused():
    env    = built()
    single = probatum.rl.Env(universe(), RANDOM, window=8, seed=4)

    for each in (env, single):
        with pytest.raises(ProbatumTypeError, match=re.escape(
                "pickle: an environment, which holds the functions it was given, does "
                "not pickle; build it in the process that steps it, or clone() it in "
                "this one")):
            pickle.dumps(each)

    # a copy is a clone, which a pickle's refusal never reaches
    assert copy.deepcopy(env).num_envs == 3 and copy.copy(single) is not single

    for given, error, says in [
        (dict(env=3), ProbatumValueError, "env: 3 of 3 environments; give one from 0 to 2"),
        (dict(env=2**64), ProbatumValueError,
         "env: 18446744073709551616 of 3 environments; give one from 0 to 2"),
        (dict(env=-1), ProbatumValueError, "env: -1; give a whole number from 0"),
        (dict(env=1.5), ProbatumTypeError, "env: 1.5; give a whole number from 0"),
        (dict(n=2), ProbatumValueError,
         "n: 2 without env; name the environment to fork with env=, or leave n out"),
        (dict(env=0, n=0), ProbatumValueError,
         "n: 0; give a whole number of copies from 1"),
    ]:
        with pytest.raises(error, match=re.escape(says)):
            env.clone(**given)
