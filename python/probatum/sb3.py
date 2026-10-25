"""Stable-Baselines3 adapter for the environment ``probatum.env`` returns.

SB3 does not consume a Gymnasium vector env directly; it wants a single ``gym.Env``
it vectorizes itself, or its own ``VecEnv``. ``wrap`` presents the batched, GIL-free
``VectorEnv`` as an SB3 ``VecEnv``, a ``ProbatumVecEnv``, so PPO, A2C, and the rest of
SB3 train on it without giving up the batch. The mapping is a straight remap rather
than a re-vectorization: it switches the environment to same-step autoreset (a
finished env's step returns the next episode's first observation, the true final under
``final_obs``), which is exactly the SB3 ``VecEnv`` contract, so a done env's final
observation becomes SB3's ``terminal_observation`` (ADR 0038), and to an observation
of its own each step, which SB3 keeps across the step after it. The agent spec rides
on a copy of the action space, which SB3 saves into the model's zip, so a model loaded
again is backtested as it trained, and ``SpecCallback`` copies ``VecNormalize``'s
observation statistics into the model's spec after each rollout, as the model's own
(ADR 0039).

    from stable_baselines3 import PPO

    venv = probatum.sb3.wrap(probatum.env(data, venue, window=32, num_envs=8, seed=0))
    model = PPO("MlpPolicy", venv)
    model.learn(total_timesteps=100_000)

Needs stable-baselines3: ``pip install 'probatum[sb3]'``.
"""

from __future__ import annotations

import copy

import gymnasium
import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required for probatum.sb3; pip install 'probatum[sb3]'"
    ) from exc

from ._errors import SpecError
from ._storage import _plain

__all__ = ["ProbatumVecEnv", "SpecCallback", "wrap"]


def wrap(env):
    """``env``, the environment ``probatum.env`` returns, as a Stable-Baselines3
    ``VecEnv`` any SB3 algorithm trains on, a ``ProbatumVecEnv`` carrying its agent
    spec on a copy of its action space."""
    return ProbatumVecEnv(env)


class ProbatumVecEnv(VecEnv):
    """Present a ``probatum.rl.VectorEnv`` as a Stable-Baselines3 ``VecEnv``, as
    ``wrap`` builds it.

    The underlying env keeps ownership of the batch, the observation, the reward, and
    the action decoding; this only translates the step and reset conventions, and
    carries the env's agent spec on a copy of its action space as ``agent_spec``.
    """


    def __init__(self, env):
        # SB3 resets an env on the step its episode ends, and keeps each observation
        # across the step after it, which one array written over would change
        env.autoreset_mode = gymnasium.vector.AutoresetMode.SAME_STEP
        env.metadata       = {**env.metadata, "autoreset_mode": env.autoreset_mode}
        env._copy          = True
        env.provenance["settings"].update(autoreset="same_step", copy=True)
        self._env = env

        # the space SB3 pickles into a saved model's zip, so the spec loads with it
        acted            = copy.deepcopy(env.single_action_space)
        acted.agent_spec = env.agent_spec
        super().__init__(env.num_envs, env.single_observation_space, acted)

        self._actions    = None
        self._reset_seed = None


    @property
    def agent_spec(self):
        """The agent spec the action space carries, with any statistics
        ``SpecCallback`` copied in."""
        return copy.deepcopy(self.action_space.agent_spec)


    def reset(self):
        if self._reset_seed is not None:
            obs, _info = self._env.reset(seed=self._reset_seed)
            self._reset_seed = None
        else:
            obs, _info = self._env.reset()

        return obs


    def seed(self, seed=None):
        # SB3 seeds the env through this; hold the seed for the next reset so the
        # batch's episode offsets are reproducible from an SB3 seed, not only from
        # the VectorEnv constructor seed
        self._reset_seed = seed
        return [seed] * self.num_envs


    def step_async(self, actions):
        self._actions = actions


    def step_wait(self):
        obs, rewards, terminations, truncations, infos = self._env.step(self._actions)

        dones      = np.logical_or(np.asarray(terminations), np.asarray(truncations))
        final_obs  = infos.get("final_obs")
        final_mask = infos.get("_final_obs")
        info_list  = []

        for i in range(self.num_envs):
            info = {}
            if dones[i]:
                # SB3 distinguishes a time-limit truncation from a true terminal so
                # it can bootstrap the value past a truncation; a done env's final
                # observation is probatum's, carried in infos, and becomes SB3's terminal
                info["TimeLimit.truncated"] = bool(truncations[i] and not terminations[i])
                if final_mask is not None and final_mask[i]:
                    info["terminal_observation"] = final_obs[i]
            info_list.append(info)

        return obs, np.asarray(rewards, dtype=np.float32), dones, info_list


    def close(self):
        self._env.close()


    def _count(self, indices):
        # SB3 addresses sub-envs by index; this adapter fronts one batched env, so a
        # per-env query returns the same value once per requested index
        if indices is None:
            return self.num_envs
        if isinstance(indices, int):
            return 1
        return len(list(indices))


    def get_attr(self, attr_name, indices=None):
        return [getattr(self._env, attr_name, None)] * self._count(indices)


    def set_attr(self, attr_name, value, indices=None):
        setattr(self._env, attr_name, value)


    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        result = getattr(self._env, method_name)(*method_args, **method_kwargs)
        return [result] * self._count(indices)


    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._count(indices)


class SpecCallback(BaseCallback):
    """An SB3 callback that keeps the agent spec on the model's action space: as each
    run of training starts, the spec of the environment ``wrap`` presents, and after
    each rollout ``VecNormalize``'s observation statistics copied into it, each
    feature's mean and variance, by key for a dict, its clip and its epsilon, so a
    backtest shows the agent the observation it learned on. The statistics are the
    model's own, which no environment's spec holds. Pass it to ``learn`` with a
    ``VecNormalize`` over ``wrap``'s environment."""


    def _init_callback(self):
        if getattr(self.model.action_space, "agent_spec", None) is None:
            raise SpecError("callback: the model's action space carries no agent spec; "
                            "train it on probatum.sb3.wrap(env)")

        inner = self.training_env.unwrapped
        if isinstance(inner, ProbatumVecEnv):
            # a model trained again on another environment trades as its spec says
            self.model.action_space.agent_spec = inner._env.agent_spec


    def _on_step(self):
        return True


    def _on_rollout_end(self):
        normalising = self.model.get_vec_normalize_env()
        if normalising is None or not normalising.norm_obs:
            return

        rms  = normalising.obs_rms
        held = {"clip": float(normalising.clip_obs), "epsilon": float(normalising.epsilon)}

        if isinstance(rms, dict):
            for key, part in rms.items():
                held[key] = {"mean": part.mean.tolist(), "var": part.var.tolist()}
        else:
            held.update(mean=rms.mean.tolist(), var=rms.var.tolist())

        self.model.action_space.agent_spec["sb3"] = _plain(held)
