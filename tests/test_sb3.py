"""Tests for probatum.sb3: wrap should present the VectorEnv as an SB3 VecEnv, map the
step tuple and same-step autoreset onto SB3's convention, carry the agent spec on the
action space SB3 saves, and let an SB3 algorithm train through it; a model saved and
loaded again, or a torch module written by probatum.save, should backtest as it did in
memory. stable-baselines3 backs it, so the module skips without it, and torch with it.
"""

import json
import warnings

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

import probatum
import probatum.sb3
from probatum import ProbatumTypeError, ProbatumValueError, SpecError
from test_an_episode_runs_again_as_a_backtest_from_its_start import (VENUE as PERP,
                                                                     universe)

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


def trained(env, steps=128):
    """A PPO trained briefly on ``env`` under VecNormalize, the callback copying its
    statistics into the spec."""
    venv  = VecNormalize(probatum.sb3.wrap(env))
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=32, n_epochs=1, seed=0,
                device="cpu")
    model.learn(total_timesteps=steps, callback=probatum.sb3.SpecCallback())

    return model, venv


def test_adapter_is_a_vecenv_and_steps():
    venv = probatum.sb3.wrap(probatum.env(series(), VENUE, window=8, num_envs=4, seed=0))

    assert isinstance(venv, VecEnv) and isinstance(venv, probatum.sb3.ProbatumVecEnv)
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
    venv = probatum.sb3.wrap(probatum.env(series(), VENUE, window=4, num_envs=3,
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
    venv = probatum.sb3.wrap(probatum.env(halving(16), VENUE, window=4, num_envs=2,
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
    venv  = probatum.sb3.wrap(probatum.env(series(), VENUE, window=4, num_envs=2,
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
        venv = probatum.sb3.wrap(probatum.env(series(300), VENUE, window=8, num_envs=4,
                                              seed=0))
        venv.seed(s)
        return venv.reset()

    assert np.array_equal(first_obs(11), first_obs(11))
    assert not np.array_equal(first_obs(11), first_obs(22))


def test_ppo_learns_through_the_adapter():
    venv  = probatum.sb3.wrap(probatum.env(series(400), VENUE, window=8, num_envs=8,
                                           seed=0))
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=32, n_epochs=1, verbose=0)
    model.learn(total_timesteps=8 * 16)  # a smoke run: it trains end to end

    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)

    assert np.asarray(action).shape == (8, 1)


def test_wrap_carries_the_env_s_spec_on_a_copy_of_its_action_space():
    env  = probatum.env(series(), VENUE, window=8, num_envs=2, seed=0)
    venv = probatum.sb3.wrap(env)

    assert venv.action_space.agent_spec == env.agent_spec == venv.agent_spec
    assert venv.action_space is not env.single_action_space
    assert not hasattr(env.single_action_space, "agent_spec")


def test_an_sb3_agent_saved_loaded_and_backtested_gives_the_in_memory_result_to_the_bit(
        tmp_path):
    data = universe()
    env  = probatum.env(data[:150], PERP, window=8, num_envs=4, seed=1)
    model, venv = trained(env)
    stats       = model.action_space.agent_spec["sb3"]

    assert stats == {
        "clip":    10.0,
        "epsilon": 1e-8,
        "mean":    venv.obs_rms.mean.tolist(),
        "var":     venv.obs_rms.var.tolist(),
    }

    memory = probatum.backtest(model, data[150:], PERP)

    # the statistics are the model's own: the environment's spec holds none, and a
    # backtest given it reads the model's
    assert env.agent_spec["sb3"] is None
    assert np.array_equal(probatum.backtest(model, data[150:], PERP, spec=env).equity,
                          memory.equity)

    model.save(tmp_path / "agent")
    loaded = probatum.backtest(PPO.load(tmp_path / "agent", device="cpu"), data[150:],
                               PERP)

    assert np.array_equal(loaded.equity, memory.equity)
    assert np.array_equal(loaded.policy, memory.policy, equal_nan=True)

    probatum.save(model, tmp_path / "saved")

    assert sorted(path.name for path in (tmp_path / "saved").iterdir()) == [
        "agent.zip", "spec.json"]

    spec  = json.loads((tmp_path / "saved" / "spec.json").read_text(encoding="utf-8"))
    again = probatum.backtest(PPO.load(tmp_path / "saved" / "agent.zip", device="cpu"),
                              data[150:], PERP, spec=tmp_path / "saved")

    assert spec == model.action_space.agent_spec
    assert np.array_equal(again.equity, memory.equity)
    assert memory.provenance["policy"] == "the SB3 model PPO"

    # the statistics are what the run reads: a model whose spec holds none sees another
    # observation, and decides otherwise
    model.action_space.agent_spec["sb3"] = None

    assert not np.array_equal(probatum.backtest(model, data[150:], PERP).policy,
                              memory.policy, equal_nan=True)


def test_the_callback_copies_a_dict_observation_s_statistics_key_by_key():
    data  = universe()
    env   = probatum.env(data[:150], PERP, window=8, layout="dict", num_envs=4, seed=1)
    venv  = VecNormalize(probatum.sb3.wrap(env))
    model = PPO("MultiInputPolicy", venv, n_steps=16, batch_size=32, n_epochs=1, seed=0,
                device="cpu")
    model.learn(total_timesteps=128, callback=probatum.sb3.SpecCallback())
    stats = model.action_space.agent_spec["sb3"]

    assert set(stats) == {"market", "account", "clip", "epsilon"}

    for key in ("market", "account"):
        assert stats[key] == {
            "mean": venv.obs_rms[key].mean.tolist(),
            "var":  venv.obs_rms[key].var.tolist(),
        }

    result = probatum.backtest(model, data[150:], PERP)
    again  = probatum.backtest(result.policy, data[150:], PERP)

    assert np.array_equal(again.equity, result.equity)


class Recorder(torch.nn.Module):
    """A module that records the tensor, the gradient mode and its own mode at each
    call, and returns its action beside a value, as an actor-critic does."""


    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, 2)
        self.calls  = []


    def forward(self, observation):
        self.calls.append((observation.device.type, observation.dtype,
                           torch.is_grad_enabled(), self.training))
        return torch.tanh(self.linear(observation)), "value"


def test_a_torch_module_is_asked_with_no_gradient_in_eval_mode_and_saved_as_weights(
        tmp_path):
    data = universe()
    env  = probatum.env(data[:150], PERP, window=8, seed=1)
    size = env.single_observation_space.shape[0]

    torch.manual_seed(0)
    module = Recorder(size)
    # a part frozen in eval mode is left in it
    module.linear.eval()

    result = probatum.backtest(module, data[150:], PERP, spec=env)

    assert set(module.calls) == {("cpu", torch.float32, False, False)}
    assert module.training and not module.linear.training
    # the run without costs asks it again, at once
    assert len(module.calls) == 2 * 149
    assert result.provenance["policy"] == "the torch module Recorder"

    probatum.save(module, tmp_path / "module", spec=env.agent_spec)

    assert sorted(path.name for path in (tmp_path / "module").iterdir()) == [
        "spec.json", "weights.pt"]

    fresh = Recorder(size)
    fresh.load_state_dict(torch.load(tmp_path / "module" / "weights.pt"))
    again = probatum.backtest(fresh, data[150:], PERP,
                              spec=tmp_path / "module" / "spec.json")

    assert np.array_equal(again.equity, result.equity)

    with pytest.raises(SpecError, match="^spec: this agent carries no spec; pass spec= "
                                        "with the env it trained on or its JSON$"):
        probatum.save(module, tmp_path / "bare")

    with pytest.raises(ProbatumTypeError, match="^agent: a function, which save writes "
                                                "no weights of; pass a torch module or an "
                                                "SB3 model, and keep a function in your "
                                                "own code$"):
        probatum.save(lambda observation: observation, tmp_path / "function", spec=env)

    (tmp_path / "file").write_text("", encoding="utf-8")
    with pytest.raises(ProbatumValueError, match="is a file; name a directory$"):
        probatum.save(module, tmp_path / "file", spec=env)


class Remembering(BasePolicy):
    """A policy whose recurrent state is a count of its predictions, as RecurrentPPO's
    is its LSTM's, recording what each prediction is handed."""


    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.calls = []


    def _predict(self, observation, deterministic=False):
        raise AssertionError("a backtest asks predict")


    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.calls.append((state, episode_start.tolist(), deterministic))
        return np.zeros((1, 2), dtype=np.float32), 0 if state is None else state + 1


def test_a_recurrent_policy_s_state_is_carried_from_one_close_to_the_next():
    data   = universe()
    env    = probatum.env(data[:150], PERP, window=8, seed=1)
    policy = Remembering(env.single_observation_space, env.single_action_space)
    result = probatum.backtest(policy, data[150:], PERP, spec=env)

    assert policy.calls[:3] == [(None, [True], True), (0, [False], True),
                                (1, [False], True)]
    assert policy.calls[148][0] == 147
    # the run without costs starts its state again
    assert len(policy.calls) == 2 * 149 and policy.calls[149] == (None, [True], True)
    assert result.provenance["policy"] == "the SB3 policy Remembering"


def test_the_callback_refuses_a_model_whose_action_space_carries_no_spec():
    env               = probatum.env(series(), VENUE, window=8, num_envs=2, seed=0)
    venv              = probatum.sb3.wrap(env)
    venv.action_space = env.single_action_space
    model             = PPO("MlpPolicy", venv, n_steps=16, batch_size=32, n_epochs=1,
                            device="cpu")

    with pytest.raises(SpecError, match="^callback: the model's action space carries "
                                        "no agent spec; train it on "
                                        "probatum.sb3.wrap\\(env\\)$"):
        model.learn(total_timesteps=32, callback=probatum.sb3.SpecCallback())


def test_capacity_and_a_cost_sweep_run_an_sb3_agent_at_their_balances_and_costs():
    data     = universe()
    model, _ = trained(probatum.env(data[:150], PERP, window=8, num_envs=4, seed=1))
    sized    = probatum.capacity(model, data[150:], PERP, [1_000.0, 50_000.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = [probatum.backtest(model, data[150:], PERP.replace(balance=balance))
                 for balance in (1_000.0, 50_000.0)]

    assert list(sized.total_return_pct) == [run.stats["total_return_pct"] for run in alone]

    swept = probatum.cost_curve(model, data[150:], PERP, scales=[0.0, 1.0])
    plain = probatum.backtest(model, data[150:], PERP)

    assert swept.total_return_pct[1] == plain.stats["total_return_pct"]
    assert swept.total_return_pct[0] != swept.total_return_pct[1]
