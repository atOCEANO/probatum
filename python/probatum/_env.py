"""``env``: accounts at a venue on a Data's bars stepped together in Rust as a Gymnasium
vector environment, each in an episode drawn from a stream of its own, and ``Env``, one
of them, which Gymnasium's ``make`` builds as ``probatum/Trading-v0`` (Specification,
The environment). This module imports Gymnasium, which nothing a backtest loads does
(ADR 0044), and registers the environment as it loads."""

import collections.abc
import copy
import inspect
import math

import numpy as np

from . import _agent, _assumptions, _probatum
from ._agent import STATE, State
from ._backtest import _annualised, _kind, _markets, _number, _real
from ._data import Data, _chosen, _count, _when, features as _features
from ._errors import (DataError, MissingDependency, ProbatumKeyError,
                      ProbatumNotImplementedError, ProbatumRuntimeError, ProbatumTypeError,
                      ProbatumValueError, VenueError)
from ._result import Assumptions, _Record, _provenance
from ._search import _counted
from ._space import _flag
from ._streams import stream
from ._venue import _run, _venue_of

try:
    import gymnasium
except ImportError:
    raise MissingDependency("env: needs gymnasium 1.2 or later, which is not installed; "
                            "pip install \"gymnasium>=1.2\"") from None
if tuple(int(part) for part in gymnasium.__version__.split(".")[:2]) < (1, 2):
    raise MissingDependency(f"env: needs gymnasium 1.2 or later, and "
                            f"{gymnasium.__version__} is installed; pip install "
                            f"\"gymnasium>=1.2\"")

ACTIONS = ("box", "simplex", "discrete", "orders")
LAYOUTS = ("flat", "dict")
REWARDS = ("log", "sharpe")
COSTS   = ("maker", "taker", "slippage", "impact")
MODES   = {
    "next_step": gymnasium.vector.AutoresetMode.NEXT_STEP,
    "same_step": gymnasium.vector.AutoresetMode.SAME_STEP,
    "disabled":  gymnasium.vector.AutoresetMode.DISABLED,
}
# the Batch's setting that each cost an environment draws replaces
DRAWN   = {
    "maker":    "fee_maker_per_env",
    "taker":    "fee_taker_per_env",
    "slippage": "slippage_bps_per_env",
    "impact":   "impact_per_env",
}


def env(data, venue, window=64, features=None, layout="flat", max_leverage=1.0,
        actions="box", max_offset=20, decode=None, reward="log", episode_length=None,
        sample=None, costs=None, normalise=False, fill=None, tolerance=0.0, num_envs=1,
        threads=None, autoreset="next_step", copy=True, record="none", seed=None):
    """``num_envs`` accounts at ``venue``, a Venue or a preset's name, on ``data``, a
    Data or anything one is built from, stepped together in Rust as a Gymnasium vector
    environment, a `VectorEnv` (Specification, The environment).

    Each environment draws its episodes from a stream of its own, each a start among
    the bars at which an instrument is shown and its slots, the instruments it trades:
    ``sample`` of them drawn among those shown at its start, or every instrument. Its
    observation is float32, laid out ``"flat"`` or as a ``"dict"`` by ``layout``: the
    market block of each slot's ``features`` on the ``window`` bars up to its close, a
    ``(T, N, F)`` array or a function of a Data that returns one, read as
    ``probatum.Data.features`` reads them with ``fill``, or without them its candles
    over its last close; then the account block, each slot's weight, unrealized PnL
    over equity, distance from its entry and whether it is shown, and the account's
    log return, leverage, margin ratio and cash weight.

    ``actions`` are ``"box"``, each slot's weight ``max_leverage`` times its action over
    the shown slots' gross, at least one; ``"simplex"``, weights over a last cash slot;
    ``"discrete"``, each slot short, flat or long at ``max_leverage`` over the slots,
    flat or long on spot; or ``"orders"``, each slot a limit for a share of equity up to
    ``max_leverage`` at up to ``max_offset`` ticks from the close on the passive side,
    replacing the last. ``decode(actions, state)`` returns the weights in their place,
    and a target trades each shown slot to its weight with ``tolerance``. ``reward`` is
    ``"log"``, the step's log return of equity floored at 1% of the opening balance,
    ``"sharpe"``, its differential Sharpe, or ``reward(state)``, a function of the
    state that returns each environment's. An episode ends at a liquidation or at the
    floor, and is cut at the last bar or after ``episode_length`` steps.

    ``normalise=True`` writes each market feature over its mean and deviation across
    every value the observations of ``data`` hold, which the environment's
    ``agent_spec`` keeps beside everything a backtest needs to show an agent what it saw
    here. ``costs`` maps ``"maker"``, ``"taker"``, ``"slippage"`` and ``"impact"`` to the
    ``(low, high)`` range each environment draws its own from once; ``threads`` is the
    number of threads a step runs on, by default one for each 2^17 floats its
    observations hold and at most half the CPUs, no result depending on it;
    ``autoreset`` is Gymnasium's mode, ``"next_step"``, ``"same_step"`` or
    ``"disabled"``; ``copy=False`` hands out one observation array that the next step
    writes over; and ``seed`` names every stream the environments draw from, a fresh one
    where it is None. Not done yet is ``record="transitions"``.
    """
    return VectorEnv(**locals())


SIGNATURE = inspect.signature(env)


class VectorEnv(gymnasium.vector.VectorEnv):
    """The accounts `env` steps, as Gymnasium's vector API steps a batch, built with
    `env`'s arguments, which ``gymnasium.make_vec`` passes it. ``reset`` starts an
    episode in each environment and ``step`` takes each one's action at its close and
    returns the observation, reward and flags at the next, the info holding each
    environment's ``episode``, its number in its stream, its ``step`` and its
    ``start``, the Data's row it opened on, and with ``sample`` its ``instruments``.
    ``describe()`` lists what its runs assume, and ``provenance`` records what they run
    on and the streams each environment draws from."""


    def __init__(self, data, venue, **settings):
        given = SIGNATURE.bind(data, venue, **settings)
        given.apply_defaults()
        given = given.arguments

        _checked(given)
        held = _venue_of(venue)
        data = data if isinstance(data, Data) else Data(data)
        if not len(data) or not data.available.any():
            raise DataError("data: no bars to step; give a Data that has some")

        n, window = int(given["num_envs"]), int(given["window"])
        seed = (np.random.SeedSequence().entropy if given["seed"] is None
                else int(given["seed"]))

        run     = _run(held, data, seed)
        markets = _markets(run)
        specs   = run.settings.get("instruments", [run.settings])
        names   = data.names
        width   = len(names)
        sampled = given["sample"] is not None
        k       = int(given["sample"]) if sampled else width
        spot    = [market == "spot" for market in markets]

        if k > width:
            raise ProbatumValueError(f"sample: {k} of {_count(width, 'instrument')}; give "
                                     f"at most {width}")

        if sampled and len(set(spot)) > 1:
            raise ProbatumValueError(f"sample: {k} of a universe of spot and perpetual "
                                     f"instruments, whose slots would change kind between "
                                     f"episodes; sample instruments of one kind")

        orders = given["actions"] == "orders"
        ticks  = [float(spec.get("tick") or 0.0) for spec in specs]
        if orders:
            for name, tick in zip(names, ticks):
                if tick <= 0.0:
                    raise VenueError(f"actions: \"orders\" rests each limit whole ticks "
                                     f"from the close, and {held.name or 'the venue'} "
                                     f"gives {name} no tick; give {name} its tick in "
                                     f"instruments")

        times   = run.settings["times"]
        rows    = len(times)
        first   = run.start
        printed = np.isfinite(run.candles[..., 3]).reshape(rows, width)

        if given["features"] is None:
            if given["fill"] is not None:
                raise ProbatumValueError(f"fill: {given['fill']!r}, and no features are "
                                         f"given for it to fill; pass features=, or "
                                         f"leave fill out")
            values, f = None, 5
            ready = printed.copy()
        else:
            read = _features(data, given["features"], given["fill"])

            # the observation is float32, which a finite float64 past its range is not;
            # an infinity is a warm-up's or an unprinted bar's, which none observes
            big = np.argwhere(np.isfinite(read) & (np.abs(read) > np.finfo(np.float32).max))
            if len(big):
                row, i, g = big[0]
                raise DataError(f"features: {read[row, i, g]:g} at bar {row:,} "
                                f"({_when(data.time[row])}) for {names[i]}, feature {g}, "
                                f"past what float32 holds; scale it down")

            f      = read.shape[2]
            values = np.full((rows, width, f), np.nan)
            values[first:] = read[data.available.any(axis=1)]
            ready = printed & np.isfinite(values).all(axis=2)

        # the Engine holds the rows before the Data's own, which no episode reads
        ready[:first] = False

        found    = ready.argmax(axis=0)
        onsets   = [int(t) if ready[t, i] else None for i, t in enumerate(found)]
        delisted = np.zeros((rows, width), dtype=bool)
        for i, spec in enumerate(specs):
            if spec.get("delist") is not None:
                delisted[:, i] = times >= spec["delist"]

        onset  = np.array([rows + window if e is None else e for e in onsets])
        shown  = (np.arange(rows)[:, None] + 1 >= onset + window) & printed & ~delisted
        starts = np.flatnonzero(shown[:rows - 1].sum(axis=1) >= k if sampled
                                else shown[:rows - 1].any(axis=1))

        if not len(starts):
            what = f"{k} instruments" if sampled else "an instrument"
            raise DataError(f"window: {window}, and no bar before the last has {what} "
                            f"shown a whole window past the first bar its features are "
                            f"finite on; give a shorter window, or more bars")

        scale = (_agent._moments(values, run.candles, printed, shown, window)
                 if given["normalise"] else None)

        ranges = _ranges(given["costs"])
        drawn  = {}
        if ranges:
            # each environment takes four draws, one for each cost, so its costs are the
            # same whatever num_envs is and whichever costs are drawn
            u = stream(seed, "costs").random((n, len(COSTS)))
            for j, name in enumerate(COSTS):
                if name in ranges:
                    low, high = ranges[name]
                    drawn[DRAWN[name]] = low + (high - low) * u[:, j]
            for spec in run.settings.get("instruments", []):
                for side in ("maker", "taker"):
                    if side in ranges:
                        spec.pop(f"fee_{side}", None)

        quote       = run.settings["quote"]
        balance     = sum(quote.values()) if isinstance(quote, dict) else float(quote)
        reward      = given["reward"]
        length      = given["episode_length"]
        threads     = given["threads"]
        batch       = _probatum.Batch(run.candles, n, report=False, **run.settings, **drawn)
        self._inner = _probatum.Environment(
            batch, window, onsets, k, ticks,
            reward         = "own" if callable(reward) else reward,
            eta            = 2.0 / (window + 1),
            features       = values,
            max_leverage   = float(given["max_leverage"]),
            max_offset     = float(given["max_offset"]),
            tolerance      = float(given["tolerance"]),
            orders         = orders,
            episode_length = None if length is None else int(length),
            floor          = 0.01 * balance,
            threads        = None if threads is None else int(threads),
            scale          = scale,
        )

        lows   = np.where([spot[0]] * k if sampled else spot, 0.0, -1.0)
        market = window * k * f
        size   = self._inner.size

        if given["layout"] == "flat":
            observed = gymnasium.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        else:
            observed = gymnasium.spaces.Dict({
                "market":  gymnasium.spaces.Box(-np.inf, np.inf, (window, k, f),
                                                np.float32),
                "account": gymnasium.spaces.Box(-np.inf, np.inf, (size - market,),
                                                np.float32),
            })

        kind    = given["actions"]
        choices = np.where(lows == 0.0, 2, 3)

        if kind == "box":
            acted = gymnasium.spaces.Box(lows.astype(np.float32), 1.0, (k,), np.float32)
        elif kind == "simplex":
            acted = gymnasium.spaces.Box(0.0, 1.0, (k + 1,), np.float32)
        elif kind == "discrete":
            acted = (gymnasium.spaces.Discrete(int(choices[0])) if k == 1
                     else gymnasium.spaces.MultiDiscrete(choices))
        else:
            acted = gymnasium.spaces.Box(np.tile([-1.0, 0.0], (k, 1)).astype(np.float32),
                                         1.0, (k, 2), np.float32)

        mode                          = given["autoreset"]
        mode                          = MODES[mode] if isinstance(mode, str) else mode
        self.num_envs                 = n
        self.single_observation_space = observed
        self.single_action_space      = acted
        self.observation_space        = gymnasium.vector.utils.batch_space(observed, n)
        self.action_space             = gymnasium.vector.utils.batch_space(acted, n)
        self.autoreset_mode           = mode
        self.metadata                 = {"autoreset_mode": mode}
        self.render_mode              = None

        self._k, self._sampled, self._kind = k, sampled, kind
        self._lows, self._lever, self._names = lows, float(given["max_leverage"]), names
        self._decode, self._reward = given["decode"], reward
        self._copy, self._layout, self._market = given["copy"], given["layout"], market
        self._grid   = (n, window, k, f)
        self._shown  = shown if sampled else None
        self._starts = starts
        self._first  = first

        # each Engine row from the Data's first is the Data's row of this index
        self._rows    = np.flatnonzero(data.available.any(axis=1))
        self._streams = [stream(seed, f"episodes {i}") for i in range(n)]
        self._episode = np.full(n, -1, dtype=np.int64)
        self._start   = np.full(n, starts[0], dtype=np.int64)
        self._slots   = np.tile(np.arange(k, dtype=np.int64), (n, 1))
        self._out     = np.zeros((n, size), dtype=np.float32)
        self._ended   = None
        self._state   = None
        self._ranges  = ranges
        self._record  = _Record(held, data, run, seed=seed)
        streams       = {"episodes {i}": "PCG64"}
        if ranges:
            streams["costs"] = "PCG64"
        if held._intrabar == "random":
            streams["intrabar"] = "ChaCha20, each environment's"

        provenance = _provenance(data, held, markets, "an environment", 0.0,
                                 _annualised(data), seed=seed, settings=_settings(given))
        provenance.pop("risk_free")
        self.provenance = {**provenance, "streams": streams}
        self._spec      = _agent._spec(given, data, held, names, provenance["markets"],
                                       k, sampled, lows,
                                       choices if kind == "discrete" else None, size,
                                       (window, k, f), scale)


    @property
    def agent_spec(self):
        """What a backtest needs to show an agent what it saw here and to read its
        outputs as they were read, as a JSON document: the inputs, the observation, the
        outputs, the venue and the training span (Specification, The agent spec). The
        first read calls a custom decode on its probe actions, once."""
        return copy.deepcopy(_agent._referenced(self._spec, self._decode))


    def reset(self, *, seed=None, options=None):
        """Start an episode in every environment, or in those ``options`` names as a
        ``"reset_mask"``, and return the observations and the info; ``seed`` starts the
        stream of each environment it resets again from that seed, the costs staying
        as they were drawn."""
        n    = self.num_envs
        mask = np.ones(n, dtype=bool)
        if options is not None and "reset_mask" in options:
            mask = _mask(options["reset_mask"], n)
            if self._ended is None and not mask.all():
                raise ProbatumRuntimeError("options: a reset_mask before every "
                                           "environment has had an episode; reset them "
                                           "all first")

        if seed is not None:
            _counted("seed", seed, 0)
            for i in np.flatnonzero(mask):
                self._streams[i] = stream(int(seed), f"episodes {i}")
                self._episode[i] = -1

        self._begin(mask)
        if self._ended is None:
            self._ended = np.zeros(n, dtype=bool)
        self._ended[mask] = False

        return self._observed(), self._info()


    def step(self, actions):
        """Take each environment's action at its close and return the observations,
        rewards, terminated and truncated flags and info at the next, each environment
        reset as ``autoreset`` says (Specification, Autoreset)."""
        if self._ended is None:
            raise ProbatumRuntimeError("step: before any reset; call reset first")

        n, mode = self.num_envs, self.autoreset_mode
        decided = self._decided(actions)

        # an episode that ended is never stepped again; under same_step one is left
        # ended only where its step's reward raised, and starts again here
        active = ~self._ended
        if mode != gymnasium.vector.AutoresetMode.DISABLED and self._ended.any():
            self._begin(self._ended)

        rewards, terminated, truncated = self._inner.step(decided, active)
        self._state = None
        ended       = terminated | truncated
        self._ended = ended

        if callable(self._reward):
            rewards = np.where(active, self._rewarded(active), 0.0)

        obs  = self._observed()
        info = self._info()

        if mode == gymnasium.vector.AutoresetMode.SAME_STEP and ended.any():
            final = np.full(n, None, dtype=object)
            for i in np.flatnonzero(ended):
                final[i] = (obs[i].copy() if self._layout == "flat"
                            else {key: value[i].copy() for key, value in obs.items()})
            told = {key: value for key, value in info.items() if not key.startswith("_")}
            told.update({f"_{key}": ended.copy() for key in list(told)})

            self._begin(ended)
            self._ended = np.zeros(n, dtype=bool)
            obs         = self._observed()
            info        = {**self._info(), "final_obs": final, "_final_obs": ended.copy(),
                           "final_info": told, "_final_info": ended.copy()}

        return obs, rewards, terminated, truncated, info


    def describe(self):
        """What the environments' runs assume (Specification, Assumptions), as the
        `Assumptions` rows a backtest on their Data at their venue lists, with no cost
        and the funding inside a bar counted over every bar they step, a cost each
        environment draws shown as the range it is drawn from."""
        return Assumptions._of(_assumptions.environment(self._record, self._ranges))


    def _begin(self, mask):
        """Draw the next episode of each environment ``mask`` names from its stream,
        its start and then its slots, and open it."""
        for i in np.flatnonzero(mask):
            rng   = self._streams[i]
            start = self._starts[rng.integers(len(self._starts))]
            if self._sampled:
                held = np.flatnonzero(self._shown[start])
                self._slots[i] = np.sort(rng.choice(held, self._k, replace=False))
            self._start[i]    = start
            self._episode[i] += 1

        self._inner.reset(mask, self._start, self._slots)
        self._state = None


    def _observed(self):
        """The observations, in an array of their own unless ``copy`` is off."""
        out = np.empty_like(self._out) if self._copy else self._out
        self._inner.observe(out)
        if self._layout == "flat":
            return out
        return {
            "market":  out[:, :self._market].reshape(self._grid),
            "account": out[:, self._market:],
        }


    def _info(self):
        """Each environment's episode, its step, its start among the Data's rows and,
        sampled, its instruments, every one present."""
        _, steps = self._inner.progress()
        info = {
            "episode": self._episode.copy(),
            "step":    steps,
            "start":   self._rows[self._start - self._first],
        }
        if self._sampled:
            info["instruments"] = self._slots.copy()

        n = self.num_envs
        return {**info, **{f"_{key}": np.ones(n, dtype=bool) for key in info}}


    def _read(self):
        """The `State` the last step or reset left."""
        if self._state is None:
            self._state = State(self._inner.state())

        return self._state


    def _decided(self, actions):
        """The step's decision from ``actions``: each slot's weight, or with orders each
        slot's size and offset (Specification, Actions)."""
        n, k = self.num_envs, self._k
        if self._decode is not None:
            return self._decoded(self._decode(actions, self._read()))

        shape = self.action_space.shape
        try:
            given = np.asarray(actions)
        except (TypeError, ValueError):
            raise ProbatumValueError(f"actions: {_kind(actions)}, which numpy reads as no "
                                     f"one array; give {shape}") from None

        if given.shape != shape:
            raise ProbatumValueError(f"actions: shape {given.shape} for {n} environments "
                                     f"of {_count(k, 'slot')}; give {shape}")

        if not np.issubdtype(given.dtype, np.number) or given.dtype.kind == "b":
            raise ProbatumTypeError(f"actions: an array of {given.dtype}; give numbers")

        given = given.astype(np.float64)
        lost  = np.argwhere(~np.isfinite(given))
        if len(lost):
            at = tuple(lost[0])
            raise ProbatumValueError(f"actions: {given[at]} for environment {at[0]}; give "
                                     f"finite actions")

        if self._kind == "discrete":
            return _discrete(given.reshape(n, k), self._lows, self._lever)

        if self._kind == "orders":
            return given

        if self._kind == "box":
            return _box(given, self._lows, self._inner.shown(), self._lever)

        return _simplex(given, self._inner.shown())


    def _decoded(self, weights):
        """``weights`` as ``decode`` returned them, refused where they are not one
        finite weight or a NaN for each slot, or short a spot slot."""
        n, k = self.num_envs, self._k

        try:
            weights = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError):
            raise ProbatumTypeError(f"decode: returned {_kind(weights)}; return a "
                                    f"({n}, {k}) array of weights") from None

        if weights.shape != (n, k):
            raise ProbatumValueError(f"decode: returned shape {weights.shape} for {n} "
                                     f"environments of {_count(k, 'slot')}; return "
                                     f"({n}, {k})")

        lost = np.argwhere(np.isinf(weights))
        if len(lost):
            env, slot = lost[0]
            raise ProbatumValueError(f"decode: {weights[env, slot]} for environment "
                                     f"{env}'s slot {slot}; return finite weights, or "
                                     f"NaN to leave a slot unnamed")

        short = np.argwhere((weights < 0.0) & (self._lows == 0.0))
        if len(short):
            env, slot = short[0]
            name = self._names[self._slots[env, slot]]
            raise ProbatumValueError(f"decode: {weights[env, slot]} for environment "
                                     f"{env}'s slot {slot}, {name}, which trades spot "
                                     f"and cannot be short; return weights from 0 for it")

        return weights


    def _rewarded(self, active):
        """The rewards ``reward`` returns from the state, refused where they are not
        one finite reward for each environment ``active`` names."""
        n   = self.num_envs
        got = self._reward(self._read())

        try:
            rewards = np.asarray(got, dtype=np.float64)
        except (TypeError, ValueError):
            raise ProbatumTypeError(f"reward: returned {_kind(got)}; return a ({n},) "
                                    f"array of rewards") from None

        if rewards.shape != (n,):
            raise ProbatumValueError(f"reward: returned shape {rewards.shape} for {n} "
                                     f"environments; return ({n},)")

        lost = np.flatnonzero(active & ~np.isfinite(rewards))
        if len(lost):
            raise ProbatumValueError(f"reward: {rewards[lost[0]]} for environment "
                                     f"{lost[0]}; return finite rewards")

        return rewards


class Env(gymnasium.Env):
    """One environment, as Gymnasium's single-environment API steps it, for Tianshou
    and CleanRL-style scripts: a `VectorEnv` of one, built with `env`'s arguments but
    ``num_envs`` and ``autoreset``, which ``gymnasium.make`` passes it. An episode that
    ends waits for the next ``reset``."""

    metadata = {"render_modes": []}


    def __init__(self, data, venue, **settings):
        for name in ("num_envs", "autoreset"):
            if name in settings:
                raise ProbatumTypeError(f"{name}: {settings[name]!r}, and an Env is one "
                                        f"environment, reset when its caller says; build "
                                        f"several with probatum.env")

        self._batch = VectorEnv(data, venue, num_envs=1, autoreset="disabled", **settings)
        self.observation_space = self._batch.single_observation_space
        self.action_space      = self._batch.single_action_space


    @property
    def provenance(self):
        return self._batch.provenance


    @property
    def agent_spec(self):
        return self._batch.agent_spec


    def reset(self, *, seed=None, options=None):
        """Start an episode and return its first observation and its info; ``seed``
        starts the environment's stream again from that seed."""
        super().reset(seed=seed)
        obs, info = self._batch.reset(seed=seed)
        return _first(obs), _first(info)


    def step(self, action):
        """Take ``action`` at the close and return the observation, reward, terminated
        and truncated flags and info at the next."""
        obs, reward, terminated, truncated, info = self._batch.step([action])
        return (_first(obs), float(reward[0]), bool(terminated[0]), bool(truncated[0]),
                _first(info))


    def describe(self):
        """What the environment's runs assume, as `VectorEnv.describe` lists them."""
        return self._batch.describe()


def _checked(given):
    """Refuse each of `env`'s arguments ``given``, by name, that is wrong in itself."""
    _counted("num_envs", given["num_envs"], 1, "environments")
    if given["threads"] is not None:
        _counted("threads", given["threads"], 1, "threads")
    _counted("window", given["window"], 1, "bars")
    _chosen("layout", given["layout"], LAYOUTS, "give \"flat\" or \"dict\"")
    _chosen("actions", given["actions"], ACTIONS,
            "give \"box\", \"simplex\", \"discrete\" or \"orders\"")

    lever = given["max_leverage"]
    if not (_number(lever) and lever > 0.0):
        error = ProbatumValueError if _real(lever) else ProbatumTypeError
        raise error(f"max_leverage: {lever!r}; give a multiple of equity above 0")

    offset = given["max_offset"]
    if not (_number(offset) and offset >= 0.0):
        error = ProbatumValueError if _real(offset) else ProbatumTypeError
        raise error(f"max_offset: {offset!r}; give a number of ticks from 0")

    decode = given["decode"]
    if decode is not None and not callable(decode):
        raise ProbatumTypeError(f"decode: {_kind(decode)}; give a function of the "
                                f"actions and the state that returns each slot's weight")
    if decode is not None and given["actions"] == "orders":
        raise ProbatumValueError("decode: a function, and actions=\"orders\" are limits, "
                                 "not weights for it to return; give actions=\"box\", "
                                 "\"simplex\" or \"discrete\" with it")

    if not callable(given["reward"]):
        _chosen("reward", given["reward"], REWARDS,
                "give \"log\", \"sharpe\" or a function of the state")
        if given["reward"] == "sharpe" and given["window"] == 1:
            raise ProbatumValueError("reward: \"sharpe\" over a window of 1 bar, whose "
                                     "moments keep the last return alone and so no "
                                     "deviation; give a window of 2 bars or more")

    if given["episode_length"] is not None:
        _counted("episode_length", given["episode_length"], 1, "steps")
    if given["sample"] is not None:
        _counted("sample", given["sample"], 1, "instruments")
    _flag("normalise", given["normalise"])

    tolerance = given["tolerance"]
    if not (_number(tolerance) and tolerance >= 0.0):
        error = ProbatumValueError if _real(tolerance) else ProbatumTypeError
        raise error(f"tolerance: {tolerance!r}; give a fraction of equity from 0, such "
                    f"as 0.01")

    mode = given["autoreset"]
    if not isinstance(mode, gymnasium.vector.AutoresetMode):
        _chosen("autoreset", mode, MODES,
                "give \"next_step\", \"same_step\" or \"disabled\"")

    _flag("copy", given["copy"])

    record = given["record"]
    if isinstance(record, str) and record == "transitions":
        raise ProbatumNotImplementedError("record: 'transitions', which an environment "
                                          "does not record yet; leave it \"none\"")
    _chosen("record", record, ("none",), "leave it \"none\"")

    if given["seed"] is not None:
        _counted("seed", given["seed"], 0)


def _box(actions, lows, shown, lever):
    """Box ``actions`` as weights, each clipped to its slot's bounds, ``lows`` to 1, and
    ``lever`` times itself over the gross of the slots ``shown``, at least 1
    (Specification, Actions)."""
    actions = np.clip(actions, lows, 1.0)
    gross   = np.maximum(1.0, (np.abs(actions) * shown).sum(axis=1, keepdims=True))
    return lever * actions / gross


def _simplex(actions, shown):
    """Simplex ``actions`` as weights, each slot's over the sum of the slots ``shown``
    and the last, the cash slot's, all cash where they sum to zero."""
    actions = np.clip(actions, 0.0, 1.0)
    k       = actions.shape[1] - 1
    total   = (actions[:, :k] * shown).sum(axis=1, keepdims=True) + actions[:, k:]
    held    = total > 0.0
    return np.where(held, actions[:, :k] / np.where(held, total, 1.0), 0.0)


def _discrete(choices, lows, lever):
    """Discrete ``choices`` as weights, a slot short, flat or long, or where its bound
    in ``lows`` is 0, spot, flat or long, at ``lever`` over the slots; refused where a
    choice is not one of its slot's."""
    spot = lows == 0.0
    bad  = np.argwhere((choices != np.floor(choices)) | (choices < 0)
                       | (choices >= np.where(spot, 2, 3)))
    if len(bad):
        env, slot = bad[0]
        raise ProbatumValueError(f"actions: {choices[env, slot]:g} for environment "
                                 f"{env}'s slot {slot}; give a whole choice from 0 to "
                                 f"{1 if spot[slot] else 2}")

    return (choices - np.where(spot, 0.0, 1.0)) * lever / choices.shape[1]


def _ranges(costs):
    """The ``(low, high)`` range of each cost ``costs`` draws, by its name, refused
    where it names no cost or gives a range that is no pair of finite bounds in order,
    a fee's above -1 and the others' from 0."""
    if costs is None:
        return {}

    if not isinstance(costs, collections.abc.Mapping):
        raise ProbatumTypeError(f"costs: {costs!r}; give a dict of the ranges drawn, "
                                f"such as {{\"taker\": (0.0004, 0.0008)}}")

    ranges = {}
    for name, given in costs.items():
        if name not in COSTS:
            raise ProbatumKeyError(f"costs: {name!r}, which is no cost an environment "
                                   f"draws; name \"maker\", \"taker\", \"slippage\" or "
                                   f"\"impact\"")

        pair = (tuple(given) if isinstance(given, (tuple, list, np.ndarray))
                else (given,))
        if len(pair) != 2 or not all(_real(bound) for bound in pair):
            raise ProbatumTypeError(f"costs: {name} {given!r}; give a (low, high) pair "
                                    f"of numbers")

        low, high = float(pair[0]), float(pair[1])
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise VenueError(f"costs: {name} ({low}, {high}); give finite bounds, the "
                             f"low one first")

        if name in ("maker", "taker") and low <= -1.0:
            raise VenueError(f"costs: {name} from {low}, a rebate past the notional; "
                             f"give rates above -1")
        if name not in ("maker", "taker") and low < 0.0:
            raise VenueError(f"costs: {name} from {low}, which would pay the account to "
                             f"trade; give a range from 0")

        ranges[name] = (low, high)

    return ranges


def _mask(mask, n):
    """``mask`` as the ``(n,)`` booleans of the environments a reset starts."""
    try:
        held = np.asarray(mask)
    except (TypeError, ValueError):
        raise ProbatumValueError(f"options: a reset_mask numpy reads as no one array; "
                                 f"give ({n},) booleans") from None

    if held.dtype != np.bool_ or held.shape != (n,):
        raise ProbatumValueError(f"options: a reset_mask of {held.dtype} and shape "
                                 f"{held.shape}; give ({n},) booleans")

    if not held.any():
        raise ProbatumValueError("options: a reset_mask naming no environment; name at "
                                 "least one")

    return np.ascontiguousarray(held)


def _settings(given):
    """What a provenance records of `env`'s settings ``given``: each as given, a
    function by its name, an array by its shape and a cost by its range."""
    shown = {}
    for name, value in given.items():
        if name in ("data", "venue", "seed"):
            continue

        if callable(value):
            value = f"the function {getattr(value, '__qualname__', type(value).__name__)}"
        elif name == "features" and value is not None:
            value = f"a {np.shape(value)} array"
        elif name == "costs" and value is not None:
            value = {cost: [float(bound) for bound in pair] for cost, pair in value.items()}
        elif isinstance(value, gymnasium.vector.AutoresetMode):
            value = next(word for word, mode in MODES.items() if mode == value)

        shown[name] = value

    return shown


def _first(batched):
    """The first environment's part of ``batched``: an observation, flat or a dict, or
    an info, each of its keys but the masks."""
    if isinstance(batched, dict):
        return {key: _first(value) for key, value in batched.items()
                if not key.startswith("_")}

    first = batched[0]
    return first.item() if isinstance(first, np.generic) else first


gymnasium.register("probatum/Trading-v0", entry_point="probatum.rl:Env",
                   vector_entry_point="probatum.rl:VectorEnv")
