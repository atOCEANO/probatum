"""An environment resets as Gymnasium's three autoreset modes say.

Under "next_step", the default, the step after an episode ends resets its environment
and returns the next episode's first observation with a reward of 0.0 and neither
flag; under "same_step" the step that ends it does, with the last observation and info
under final_obs and final_info; and under "disabled" an episode that ends stays as it
ended, its later steps rewarded 0.0, until a reset names it in its mask. Each mode is
declared in the metadata and as autoreset_mode, and a step refused, or whose reward
raises, leaves no episode where none of the modes would (Specification, Autoreset).
"""

import gymnasium
import numpy as np
import pytest

import probatum
from test_an_episode_runs_again_as_a_backtest_from_its_start import VENUE, universe

# a weight of a half on each of the two slots of each of two environments
LONG = np.ones((2, 2), dtype=np.float32)
# the first slot's weight in the flat observation, past 8 bars of 2 slots' candles
WEIGHT = 8 * 2 * 5


def built(autoreset, **given):
    return probatum.env(universe(), VENUE, window=8, num_envs=2, seed=2,
                        episode_length=2, autoreset=autoreset, **given)


def plus_one(state):
    return state.log_return + 1.0


def test_next_step_resets_on_the_step_after_an_episode_ends():
    env = built("next_step")

    assert env.autoreset_mode == gymnasium.vector.AutoresetMode.NEXT_STEP
    assert env.metadata["autoreset_mode"] == env.autoreset_mode

    env.reset()
    env.step(LONG)
    obs, reward, terminated, truncated, info = env.step(LONG)

    assert truncated.all() and not terminated.any()
    assert (obs[:, WEIGHT] > 0.0).all()

    obs, reward, terminated, truncated, info = env.step(LONG)

    assert reward.tolist() == [0.0, 0.0]
    assert not (terminated | truncated).any()
    assert info["episode"].tolist() == [1, 1] and info["step"].tolist() == [0, 0]
    # a fresh account: nothing held, no return yet, and all of it cash
    assert (obs[:, WEIGHT] == 0.0).all()
    assert (obs[:, -4] == 0.0).all() and (obs[:, -1] == 1.0).all()


def test_next_step_pays_nothing_on_a_restart_whatever_the_reward():
    env = built("next_step", reward=plus_one)

    env.reset()
    paid = [env.step(LONG)[1] for _ in range(3)]

    assert (paid[0] > 0.5).all() and (paid[1] > 0.5).all()
    assert paid[2].tolist() == [0.0, 0.0]


def test_same_step_resets_on_the_step_that_ends_an_episode_and_keeps_its_last():
    env = built("same_step")

    assert env.metadata["autoreset_mode"] == gymnasium.vector.AutoresetMode.SAME_STEP

    env.reset()
    env.step(LONG)
    obs, reward, terminated, truncated, info = env.step(LONG)

    assert truncated.all()
    assert info["_final_obs"].all() and info["_final_info"].all()
    assert info["final_info"]["step"].tolist()    == [2, 2]
    assert info["final_info"]["episode"].tolist() == [0, 0]
    assert info["step"].tolist() == [0, 0] and info["episode"].tolist() == [1, 1]
    assert all(final[WEIGHT] > 0.0 for final in info["final_obs"])
    assert (obs[:, WEIGHT] == 0.0).all()


def test_same_step_keeps_the_last_observation_of_its_own_where_copy_is_off():
    # copy=False writes each observation into one array, which the next episode's
    # first observation takes over on the same step
    env = built("same_step", copy=False)

    env.reset()
    env.step(LONG)
    obs, _, _, _, info = env.step(LONG)

    assert all(final[WEIGHT] > 0.0 for final in info["final_obs"])
    assert (obs[:, WEIGHT] == 0.0).all()


def test_a_refused_step_leaves_every_episode_where_it_stood():
    env = built("next_step")

    env.reset()
    env.step(LONG)
    env.step(LONG)

    with pytest.raises(probatum.ProbatumValueError):
        env.step(np.zeros(3))

    _, _, _, _, info = env.step(LONG)

    assert info["episode"].tolist() == [1, 1] and info["step"].tolist() == [0, 0]


@pytest.mark.parametrize("autoreset", ["next_step", "same_step", "disabled"])
def test_a_reward_that_raises_leaves_its_step_s_episodes_ended(autoreset):
    calls = []

    def reward(state):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("the reward of the step the episodes end on")
        return state.log_return

    env = built(autoreset, reward=reward)
    env.reset()
    env.step(LONG)

    with pytest.raises(RuntimeError):
        env.step(LONG)

    _, paid, terminated, truncated, info = env.step(LONG)

    # an ended episode is never stepped again: reset where the mode resets one, and
    # left ended where it does not
    assert paid.tolist() == [0.0, 0.0] and not terminated.any()

    if autoreset == "disabled":
        assert truncated.all() and info["step"].tolist() == [2, 2]
    else:
        assert not truncated.any() and info["step"].tolist() == [0, 0]
        assert info["episode"].tolist() == [1, 1]


def test_a_reset_mask_held_with_a_stride_resets_the_environments_it_names():
    env = built("disabled")

    env.reset()
    _, info = env.reset(options={"reset_mask": np.array([True, True, False, False])[::2]})

    assert info["episode"].tolist() == [1, 0]


def test_disabled_leaves_an_episode_as_it_ended_until_a_reset_names_it():
    env = built("disabled")

    env.reset()
    env.step(LONG)
    last, _, _, truncated, _ = env.step(LONG)

    assert truncated.all()

    again, reward, terminated, truncated, info = env.step(LONG)

    assert np.array_equal(again, last)
    assert reward.tolist() == [0.0, 0.0] and truncated.all() and not terminated.any()

    obs, info = env.reset(options={"reset_mask": np.array([True, False])})

    assert info["episode"].tolist() == [1, 0]
    assert np.array_equal(obs[1], last[1])

    _, _, _, truncated, _ = env.step(LONG)

    assert truncated.tolist() == [False, True]
