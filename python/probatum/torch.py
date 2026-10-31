"""torch adapter for the environment ``probatum.env`` returns: ``TensorEnv`` hands its
observations, rewards and flags out as tensors on a device, through pinned buffers whose
copies never block the host on a CUDA device, and in its resident mode writes the market
block on the device itself, from the market's cells copied there once, so of the
observation only each environment's row, its slots and its account block cross the link
each step (Specification, Tensors on a device; ADR 0040). It needs torch, which probatum never
installs, as the build for a device is the user's to choose.
"""

import numpy as np

from ._backtest import _kind
from ._env import VectorEnv, gymnasium
from ._errors import MissingDependency, ProbatumTypeError, ProbatumValueError
from ._space import _flag

try:
    import torch
except ImportError:
    raise MissingDependency("probatum.torch: needs torch, which is not installed; pip "
                            "install torch, or the build for your device as pytorch.org "
                            "shows") from None

__all__ = ["TensorEnv"]


class TensorEnv(gymnasium.vector.VectorWrapper):
    """The environment ``env``, which `probatum.env` returns, handing out what each
    ``reset`` and ``step`` returns as torch tensors on ``device``: the observation as
    float32, flat or as a dict of its two blocks, the rewards as float64 and the flags
    as booleans, each the numpy path's to the bit, and the info as the environment
    gives it. Actions, and a reset's ``reset_mask``, are a tensor on any device or
    anything numpy reads. On a CUDA device each step's observation, rewards and flags
    are written into pinned host buffers that take turns, copied to the device without
    blocking the host, and written again only once an event says their last copy has
    finished, so every tensor handed out is a fresh one; on any other device the copies
    are synchronous.
    ``resident=True`` copies the market's cells to the device once and writes each
    window's market block there, the host giving each environment's row, its slots and
    its account block alone; it is refused under ``same_step`` autoreset, whose final
    observations are written on the host, and on an environment recording transitions,
    whose observations are kept whole there. A ``device`` given without an index is the
    one current at the wrap. The wrap takes ``env`` over, which from then on writes into
    the wrapper's buffers whatever its ``copy`` and is stepped through the wrapper
    alone; a copy of the wrapper wraps a clone of ``env`` as one of its own."""


    def __init__(self, env, device="cpu", resident=False):
        if not isinstance(env, VectorEnv):
            raise ProbatumTypeError(f"env: {_kind(env)}; wrap the environment "
                                    f"probatum.env returns")

        _flag("resident", resident)
        device = torch.device(device)

        if resident and env.autoreset_mode == gymnasium.vector.AutoresetMode.SAME_STEP:
            raise ProbatumValueError("resident: True under same_step autoreset, whose "
                                     "final observations are written on the host; give "
                                     "autoreset=\"next_step\" or \"disabled\"")

        if resident and env._recorder is not None:
            raise ProbatumValueError("resident: True on an environment recording "
                                     "transitions, whose observations are kept whole on "
                                     "the host; record them with resident=False")

        # a device given without an index is the one current at the wrap, where the
        # market's cells stay and every step's copies go
        device = torch.empty(0, device=device).device
        super().__init__(env)

        n             = env.num_envs
        self.device   = device
        self.resident = bool(resident)
        self._cuda    = device.type == "cuda"
        self._width   = env._inner.account_size if resident else env._inner.size
        self._turn    = 0

        # the host buffers each step's writes take turns in, so a step never writes
        # over one whose copy to the device may still be reading it
        self._observed = [self._host((n, self._width), torch.float32) for _ in range(2)]
        self._rewards  = [self._host((n,), torch.float64) for _ in range(2)]
        self._flags    = [self._host((2, n), torch.bool) for _ in range(2)]
        self._rows     = [self._host((n,), torch.int64) for _ in range(2)]
        self._slots    = [self._host(env._slots.shape, torch.int64) for _ in range(2)]
        self._events   = [None, None]
        self._market   = _Market(env, device) if resident else None
        env._copy      = False
        env._account   = self.resident
        env._out       = self._observed[0].numpy()


    @property
    def agent_spec(self):
        """The agent spec of the environment underneath."""
        return self.env.agent_spec


    @property
    def provenance(self):
        """The provenance of the environment underneath."""
        return self.env.provenance


    def describe(self):
        """What the environment's runs assume, as its `describe` lists them."""
        return self.env.describe()


    def __copy__(self):
        return TensorEnv(self.env.clone(), self.device, self.resident)


    def __deepcopy__(self, memo):
        return self.__copy__()


    def reset(self, *, seed=None, options=None):
        """Start episodes as the environment's ``reset`` does, and return the
        observations as a tensor on the device beside the info."""
        if options is not None and isinstance(options.get("reset_mask"), torch.Tensor):
            options = {**options, "reset_mask": options["reset_mask"].cpu().numpy()}

        self._free()
        self.env._out = self._observed[self._turn].numpy()
        _, info = self.env.reset(seed=seed, options=options)

        obs = self._sent()
        self._mark()

        return obs, info


    def step(self, actions):
        """Take each environment's action as the environment's ``step`` does, and
        return the observations, rewards and flags as tensors on the device beside the
        info."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._free()
        self.env._out = self._observed[self._turn].numpy()
        _, rewards, terminated, truncated, info = self.env.step(actions)

        turn = self._turn
        self._rewards[turn].numpy()[:]  = rewards
        self._flags[turn].numpy()[0][:] = terminated
        self._flags[turn].numpy()[1][:] = truncated

        obs     = self._sent()
        rewards = self._moved(self._rewards[turn])
        flags   = self._moved(self._flags[turn])
        self._mark()

        return obs, rewards, flags[0], flags[1], info


    def _host(self, shape, dtype):
        """A host buffer, pinned where the device is a CUDA one."""
        return torch.empty(shape, dtype=dtype, pin_memory=self._cuda)


    def _moved(self, held):
        """``held``, a host buffer, as a fresh tensor on the device."""
        if self._cuda:
            return held.to(self.device, non_blocking=True)
        return held.to(self.device, copy=True)


    def _free(self):
        """Wait until the turn's buffers are no longer being copied from."""
        event = self._events[self._turn]
        if event is not None:
            event.synchronize()


    def _mark(self):
        """Mark when the turn's copies have finished, and give the next turn its
        buffers."""
        if self._cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._events[self._turn] = event

        self._turn = 1 - self._turn


    def _sent(self):
        """The observation the environment wrote into the turn's buffer, on the
        device: the buffer itself, or with the resident mode the market block written
        there beside the account blocks."""
        turn, env = self._turn, self.env
        held = self._moved(self._observed[turn])

        if self.resident:
            self._rows[turn].numpy()[:]  = env._inner.ticks()
            self._slots[turn].numpy()[:] = env._slots
            market = self._market.written(self._moved(self._rows[turn]),
                                          self._moved(self._slots[turn]), held)
            held = torch.cat([market, held], dim=1)

        if env._layout == "flat":
            return held
        return {
            "market":  held[:, :env._market].reshape(env._grid),
            "account": held[:, env._market:],
        }


class _Market:
    """The market block of every environment, written on the device from the market's
    cells copied there once, as the observation writes them (Specification, Tensors on
    a device)."""


    def __init__(self, env, device):
        held = env._inner.market()
        _, window, k, _ = env._grid

        self._steps  = torch.arange(window, device=device) - (window - 1)
        self._stride = 6 if env._kind == "orders" else 4
        self._k      = k
        scale        = held["scale"]
        self._scale  = (None if scale is None
                        else torch.tensor(scale, dtype=torch.float64, device=device))

        if held["cells"] is not None:
            self._cells   = torch.as_tensor(held["cells"]).to(device)
            self._candles = None
        else:
            self._cells   = None
            self._candles = torch.as_tensor(held["candles"]).to(device)
            self._printed = ~torch.isnan(self._candles[..., 3])
            self._means   = torch.as_tensor(held["means"]).to(device)


    def written(self, rows, slots, account):
        """The market blocks of the environments at ``rows`` trading ``slots``, each
        slot shown where its ``account`` block says so, flat, as float32."""
        r     = (rows[:, None] + self._steps).clamp(min=0)[:, :, None]
        i     = slots[:, None, :]
        shown = account[:, 3:self._stride * self._k:self._stride] == 1.0

        if self._cells is not None:
            cells = self._cells[r, i]
        else:
            bars   = self._candles[r, i]
            last   = self._candles[rows[:, None], slots, 3][:, None, :, None]
            mean   = self._means[rows[:, None], slots][:, None, :]
            ratios = torch.cat([bars[..., :4] / last, (bars[..., 4] / mean)[..., None]],
                               dim=-1)
            if self._scale is not None:
                ratios = (ratios - self._scale[:, 0]) / self._scale[:, 1]
            cells = ratios.to(torch.float32)
            # a volume over a window that traded nothing is 0.0, scaled or not
            cells[..., 4] = torch.where(mean > 0.0, cells[..., 4], 0.0)
            cells = torch.where(self._printed[r, i][..., None], cells, 0.0)

        cells = torch.where(shown[:, None, :, None], cells, 0.0)
        return cells.reshape(len(rows), -1)
