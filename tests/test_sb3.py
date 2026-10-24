"""Tests for probatum.sb3.ProbatumVecEnv: it should present the VectorEnv as an SB3 VecEnv,
map the step tuple and same-step autoreset onto SB3's convention, and let an SB3
algorithm train through it. stable-baselines3 backs it, so the module skips without
it; the PPO smoke test also needs torch.
"""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import VecEnv

import probatum
from probatum.sb3 import ProbatumVecEnv

HOUR  = 3_600_000_000_000
VENUE = probatum.Venue(fees={"maker": 0.0, "taker": 0.0})


def series(n=200):
    t     = np.arange(n, dtype=np.float64)
    close = 100.0 + np.sin(t / 7.0) + 0.02 * t
    bars  = np.stack([close, close + 1.0, close - 1.0, close, np.full(n, 1000.0)], axis=1)

    return probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(n, dtype=np.int64) * HOUR,
    )


def halving(n=16):
    # every bar opens where the last closed and halves inside itself, so a levered
    # long dies on the step it opened whatever offset it drew. The volume is far
    # past anything the position asks for, because a budget that bit would leave
    # the late offsets holding too small a position to bankrupt
    open_ = 100.0 * np.power(0.5, np.arange(n, dtype=np.float64))
    close = open_ * 0.5
    bars  = np.stack([open_, open_, close, close, np.full(n, 1e9)], axis=1)

    return probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(n, dtype=np.int64) * HOUR,
    )


def test_adapter_is_a_vecenv_and_steps():
    venv = ProbatumVecEnv(probatum.env(series(), VENUE, window=8, num_envs=4, seed=0))

    assert isinstance(venv, VecEnv)
    assert venv.num_envs == 4

    obs = venv.reset()

    assert obs.shape == (4, 8 * 5 + 4 + 4)

    venv.step_async(np.array([[1.0], [-1.0], [0.0], [0.5]], dtype=np.float32))
    obs, rewards, dones, infos = venv.step_wait()

    assert obs.shape     == (4, 48)
    assert rewards.shape == (4,)
    assert dones.shape == (4,) and dones.dtype == np.bool_
    assert isinstance(infos, list) and len(infos) == 4


def test_terminal_observation_lands_in_info_on_done():
    # episode_length=1 truncates every env each step, so every info carries the SB3
    # terminal observation, probatum's true final for the episode that just ended
    venv = ProbatumVecEnv(probatum.env(series(), VENUE, window=4, num_envs=3,
                                       episode_length=1, seed=0))
    venv.reset()
    venv.step_async(np.zeros((3, 1), dtype=np.float32))
    _, _, dones, infos = venv.step_wait()

    assert dones.all()

    for info in infos:
        assert "terminal_observation" in info
        assert info["terminal_observation"].shape == (4 * 5 + 8,)
        assert info["TimeLimit.truncated"] is True  # a time limit, not a true terminal


def test_a_liquidation_reaches_sb3_as_a_true_terminal():
    # every other test here ends its episode with episode_length, so the only `done`
    # any of them has ever seen is a truncation, and three claims rest on that gap
    # (ADR 0038). All three of these pass every assertion above: `dones =
    # truncations`, which never reports a death at all; an unconditional
    # `TimeLimit.truncated = True`, which tells PPO to bootstrap the value past a
    # bankruptcy; and `terminal_observation = obs[i]`, which hands back the fresh
    # post-reset window as though the episode had ended there
    venv = ProbatumVecEnv(probatum.env(halving(16), VENUE, window=4, num_envs=2,
                                       max_leverage=5.0, seed=0))

    venv.seed(0)
    venv.reset()
    venv.step_async(np.ones((2, 1), dtype=np.float32))
    obs, _rewards, dones, infos = venv.step_wait()

    # no episode_length is set and the series is nowhere near its end, so nothing
    # here can truncate: every done on this step is a death
    assert dones.all()

    for i, info in enumerate(infos):
        assert info["TimeLimit.truncated"] is False
        assert "terminal_observation" in info
        # the window the episode DIED on, never the one it was reborn into
        assert not np.array_equal(info["terminal_observation"], obs[i])


def test_an_observation_handed_to_sb3_is_its_own_whatever_copy_says():
    # SB3 keeps the last observation across the step after it, which one array
    # written over would change
    venv  = ProbatumVecEnv(probatum.env(series(), VENUE, window=4, num_envs=2,
                                        copy=False, seed=0))
    first = venv.reset()
    kept  = first.copy()

    venv.step_async(np.ones((2, 1), dtype=np.float32))
    venv.step_wait()

    assert np.array_equal(first, kept)

    # and the environment records the settings it now runs under
    settings = venv._env.provenance["settings"]
    assert (settings["autoreset"], settings["copy"]) == ("same_step", True)


def test_seed_forwards_from_sb3_to_the_env():
    # SB3 seeds an env through seed(); forwarding it to the next reset makes the
    # batch's episodes reproducible from an SB3 seed, not only the constructor
    def first_obs(s):
        venv = ProbatumVecEnv(probatum.env(series(300), VENUE, window=8, num_envs=4,
                                           seed=0))
        venv.seed(s)
        return venv.reset()

    assert np.array_equal(first_obs(11), first_obs(11))
    assert not np.array_equal(first_obs(11), first_obs(22))


def test_ppo_learns_through_the_adapter():
    pytest.importorskip("torch")
    from stable_baselines3 import PPO

    venv  = ProbatumVecEnv(probatum.env(series(400), VENUE, window=8, num_envs=8,
                                        seed=0))
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=32, n_epochs=1, verbose=0)
    model.learn(total_timesteps=8 * 16)  # a smoke run: it trains end to end

    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)

    assert np.asarray(action).shape == (8, 1)
