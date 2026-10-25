"""The agent spec: what `env` records of an environment so a backtest shows an agent what
it saw in training and reads its outputs as they were read, the probes a custom
function is held to, and the normalisation statistics (Specification, The agent spec;
ADR 0039); and the backtest of an agent, which runs it in one environment of its spec,
and ``save``, which writes it beside its spec (Specification, An agent in a
backtest)."""

import contextlib
import copy
import hashlib
import inspect
import itertools
import json
import os
import pathlib
import sys
import warnings

import numpy as np

from . import _baselines, venues
from ._backtest import (_annualised, _kind, _markets, _module, _rate, _repeatable, _sb3,
                        _slots)
from ._data import INTERVALS, Data, _count, _when
from ._errors import (DataError, ProbatumTypeError, ProbatumValueError, ProbatumWarning,
                      SpecError)
from ._result import Result, _provenance, _version
from ._storage import _given as _decoded_json, _plain
from ._venue import Venue, _changes, _market

PACKAGE = (os.path.dirname(__file__),)
STATE   = ("equity", "previous_equity", "log_return", "drawdown", "leverage", "turnover",
           "fees", "funding", "liquidated", "weights", "positions")
SCHEMA = 1
# the probe generator, x_(n+1) = (A * x_n + C) mod 2^64, and the probe's bars
A, C  = np.uint64(6364136223846793005), np.uint64(1442695040888963407)
ROWS            = 512
START           = 1_704_067_200_000_000_000
ROWS_OF_ACTIONS = 8
ACCOUNT         = ("weight", "upnl", "entry_distance", "shown")
ORDERS          = ("resting", "nearest")
TOTALS          = ("log_return", "leverage", "margin_ratio", "cash_weight")


class State:
    """What a reward or a decode reads of every environment, as its last step or its
    reset left it (Specification, Reward and termination): ``(B,)`` arrays of
    ``equity``, ``previous_equity``, ``log_return``, ``drawdown``, ``leverage``,
    ``turnover``, ``fees``, ``funding`` and ``liquidated``, and ``(B, N)`` arrays of
    each slot's ``weights`` and ``positions``, all read-only."""

    __slots__ = STATE


    def __init__(self, arrays):
        for name in STATE:
            array = arrays[name]
            array.flags.writeable = False
            setattr(self, name, array)


    def __repr__(self):
        return f"State({len(self.equity)} environments)"


def _spec(given, data, venue, names, markets, k, sampled, lows, choices, size, shape,
          scale):
    """The agent spec of an environment built with `env`'s arguments ``given`` on
    ``data`` at ``venue``, its ``k`` slots over the instruments ``names`` on
    ``markets``, each slot's lower bound ``lows`` and with discrete actions its
    ``choices``, its observation of ``size`` floats, the market block of ``shape``, and
    the ``scale`` normalisation took, or None. A custom decode's reference is left for
    `_referenced` to take."""
    kind, lever = given["actions"], float(given["max_leverage"])

    if kind == "box":
        actions = {"space": "box", "low": lows.tolist(), "high": 1.0, "max_leverage": lever}
    elif kind == "simplex":
        # a simplex's weights sum to one, so no leverage scales them
        actions = {"space": "simplex", "low": 0.0, "high": 1.0, "size": k + 1}
    elif kind == "discrete":
        actions = {"space": "discrete", "choices": choices.tolist(), "max_leverage": lever}
    else:
        actions = {
            "space":        "orders",
            "low":          [-1.0, 0.0],
            "high":         [1.0, 1.0],
            "max_leverage": lever,
            "max_offset":   float(given["max_offset"]),
        }

    # a decode of the caller's own returns the weights itself, which no leverage scales
    if given["decode"] is not None:
        actions.pop("max_leverage", None)

    reward = given["reward"]
    if callable(reward):
        reward = {"custom": _named(reward), "source_hash": _source(reward)}
    else:
        reward = {"builtin": reward, "floor": 0.01}
        if reward["builtin"] == "sharpe":
            reward["eta"] = 2.0 / (int(given["window"]) + 1)

    decode = given["decode"]
    decode = ({"builtin": kind} if decode is None
              else {"custom": _named(decode), "source_hash": _source(decode)})
    account = ACCOUNT + (ORDERS if kind == "orders" else ())

    return _plain({
        "schema":        SCHEMA,
        "window":        int(given["window"]),
        "instruments":   list(names),
        "markets":       list(markets),
        "slots":         k,
        "sample":        k if sampled else None,
        "features":      _featured(given["features"], names, markets, data.interval,
                                   shape[2]),
        "fill":          given["fill"],
        "normalisation": None if scale is None else {
            "mean":      [mean for mean, _ in scale],
            "deviation": [deviation for _, deviation in scale],
        },
        # what SB3's VecNormalize kept, which probatum.sb3.SpecCallback copies in
        "sb3":           None,
        "observation":   {
            "size":    size,
            "layout":  given["layout"],
            "shape":   list(shape),
            "account": list(account) * k + list(TOTALS),
        },
        "actions":       actions,
        "decode":        decode,
        "reward":        reward,
        "tolerance":     float(given["tolerance"]),
        "venue":         _venued(venue),
        "training":      {
            "fingerprint": data.fingerprint,
            "interval":    data.interval,
            "bars":        len(data),
            "start":       _iso(data.time[0]),
            "end":         _iso(data.time[-1]),
        },
        "version":       _version(),
    })


def _venued(venue):
    """What the spec records of ``venue``: the preset it was made from, the day that
    preset was taken and every setting changed from it, as its repr shows them."""
    source = venue._source
    return {
        "name":    source,
        "taken":   None if source is None else venues._venue(source).taken,
        "changed": _changes(venue),
    }


def _featured(features, names, markets, interval, width):
    """What the spec records of ``features``, ``width`` of them on the training bars:
    the candles, an array's width, or a function's name, width, source hash and
    reference, a function that cannot run on its probe of the instruments ``names`` on
    ``markets`` and ``interval``, or gives another width there, leaving the reference
    None, which a warning says."""
    if features is None:
        return {"builtin": "candles"}

    if not callable(features):
        return {"array": width}

    held = {
        "custom":      _named(features),
        "width":       width,
        "source_hash": _source(features),
        "reference":   None,
    }

    try:
        out = _probed(features, names, markets, interval)
        if out.shape[2] != width:
            raise _Shaped(f"shape {out.shape} on the agent spec's probe, where the "
                          f"training bars gave {width} features")
    except Exception as error:
        warnings.warn(f"{_failed('features', features, error)}, so the agent spec holds "
                      f"no reference to hold it to, and a backtest cannot tell it from a "
                      f"changed function", ProbatumWarning, skip_file_prefixes=PACKAGE)
        return held

    return {**held, "reference": _digest(out)}


def _failed(name, function, error):
    """What a warning or a refusal says of ``function``, given as ``name``, failing on
    its probe with ``error``."""
    if isinstance(error, _Shaped):
        return f"{name}: {_named(function)} gives {error}"

    probe = ("512 bars of each instrument from 2024-01-01" if name == "features"
             else "8 rows over a fresh flat account")
    return (f"{name}: {_named(function)} raised {type(error).__name__} on the agent "
            f"spec's probe{'' if name == 'features' else ' actions'}, {probe} ({error})")


class _Shaped(ProbatumValueError):
    """A function's output on its probe that is no ``(512, N, F)`` array."""


def _referenced(spec, decode):
    """``spec`` with its custom ``decode``'s reference taken, the first time it is
    read, so a decode that keeps state of its own is called on its probe only where
    the spec is read."""
    held = spec["decode"]

    if "custom" in held and "reference" not in held:
        held["reference"] = None
        try:
            held["reference"] = _digest(_decoded(decode, _probe_actions(*_space(spec)),
                                                 spec["slots"]))
        except Exception as error:
            warnings.warn(f"{_failed('decode', decode, error)}, so the agent spec holds "
                          f"no reference to hold it to, and a backtest cannot tell it "
                          f"from a changed decode", ProbatumWarning,
                          skip_file_prefixes=PACKAGE)

    return spec


def _held(spec, features=None, decode=None):
    """Refuse ``features`` and ``decode``, as a backtest is given them, where they are
    not of the kind ``spec`` records, or are left out where it records one, or a
    function's output on its probe differs from the one recorded, once rounded to
    float32; a function with no reference to be held to is let through with a
    warning."""
    trained = spec["features"]

    if "builtin" in trained:
        if features is not None:
            raise SpecError(f"features: {_given(features)}, and the agent trained on the "
                            f"candles; leave features out")
    elif "array" in trained:
        width = trained["array"]
        if features is None or callable(features):
            raise SpecError(f"features: {_given(features)}, and the agent trained on an "
                            f"array of {width} features; pass the bars' own array as "
                            f"features=")

        got = np.shape(features)
        if len(got) != 3 or got[2] != width:
            shown = got[2] if len(got) == 3 else f"shape {got}"
            raise SpecError(f"features: an array of {shown} features, and the agent "
                            f"trained on {width}; pass the bars' own array of {width}")
    elif not callable(features):
        raise SpecError(f"features: {_given(features)}, and the agent trained on the "
                        f"function {trained['custom']}; pass it as features=")
    elif trained["reference"] is None:
        warnings.warn(f"features: the agent spec holds no reference for "
                      f"{trained['custom']}, which did not run on its probe in "
                      f"training, so {_named(features)} is not held to it",
                      ProbatumWarning, skip_file_prefixes=PACKAGE)
    else:
        try:
            got = _digest(_probed(features, spec["instruments"], spec["markets"],
                                  spec["training"]["interval"]))
        except Exception as error:
            raise SpecError(f"{_failed('features', features, error)}, where "
                            f"{trained['custom']}, which the agent trained on, gave "
                            f"{trained['reference']}; pass the function it trained "
                            f"with") from error

        if got != trained["reference"]:
            raise SpecError(f"features: {_named(features)} gives {got} on its probe, "
                            f"and the agent trained on {trained['custom']}, which gave "
                            f"{trained['reference']}; pass the function it trained with")

    held = spec["decode"]
    if "builtin" in held:
        if decode is not None:
            raise SpecError(f"decode: {_given(decode)}, and the agent's actions are "
                            f"decoded as {held['builtin']}; leave decode out")
        return

    if not callable(decode):
        raise SpecError(f"decode: {_given(decode)}, and the agent trained with "
                        f"{held['custom']}; pass it as decode=")

    if held["reference"] is None:
        warnings.warn(f"decode: the agent spec holds no reference for {held['custom']}, "
                      f"which did not run on its probe actions in training, so "
                      f"{_named(decode)} is not held to it", ProbatumWarning,
                      skip_file_prefixes=PACKAGE)
        return

    try:
        got = _digest(_decoded(decode, _probe_actions(*_space(spec)), spec["slots"]))
    except Exception as error:
        raise SpecError(f"{_failed('decode', decode, error)}, where {held['custom']}, "
                        f"which the agent trained with, gave {held['reference']}; pass "
                        f"the decode it trained with") from error

    if got != held["reference"]:
        raise SpecError(f"decode: {_named(decode)} gives {got} on its probe, and the "
                        f"agent trained with {held['custom']}, which gave "
                        f"{held['reference']}; pass the decode it trained with")


def _given(value):
    """What a refusal says of ``value``, a function or features passed for a backtest."""
    if value is None:
        return "none"
    if callable(value):
        return _named(value)
    if hasattr(value, "__array__") or hasattr(value, "__dlpack__"):
        return "an array"
    return _kind(value)


def _trained_venue(spec):
    """The venue ``spec`` records, built again: the variant of the preset it names with
    every setting it changed, or a venue built by hand with them, each float a JSON
    document cannot hold read back."""
    held    = spec["venue"]
    changed = _decoded_json(held["changed"])
    if held["name"] is None:
        return Venue(**changed)
    return venues._venue(held["name"]).replace(**changed)


def _backtested(agent, data, venue, sizing, stop, take_profit, tolerance, record,
                risk_free, balance=None, costs=1.0, spec=None, features=None,
                decode=None, instruments=None, baseline=True):
    """`backtest` of ``agent``, a trained agent: one environment of its spec on the
    slots' bars at ``venue``, recording at ``record``, the agent asked at each close
    for its action, with the run's account opened with ``balance`` and its costs
    multiplied by ``costs`` where a capacity run or a cost sweep gives them, and at the
    full level its run without costs played at once where ``baseline`` is on, since an
    agent cannot be copied as it stands to play it later."""
    for name, given in (("stop", stop), ("take_profit", take_profit)):
        if given is not None:
            raise ProbatumValueError(f"{name}: {given!r} beside an agent, which places "
                                     f"its own trades; leave it out")

    if sizing != "weight":
        raise ProbatumValueError(f"sizing: {sizing!r} beside an agent, whose spec says "
                                 f"what its actions trade; leave it out")

    if tolerance != 0.0:
        raise ProbatumValueError(f"tolerance: {tolerance!r} beside an agent, which trades "
                                 f"with the tolerance its spec holds; leave it out")

    _rate(risk_free)
    spec = _found(agent, spec)
    held = _repeatable(venue)
    rows, n = len(data), len(data.names)
    data, columns = _slotted(data, spec, instruments, held)

    if instruments is not None and features is not None and not callable(features):
        # an array of features is over the Data given, whose slots are taken
        given = np.asarray(features, dtype=np.float64)
        if given.ndim != 3 or given.shape[:2] != (rows, n):
            raise DataError(f"features: shape {given.shape} for a Data of {rows:,} bars "
                            f"of {_count(n, 'instrument')}; give ({rows}, {n}, F) over "
                            f"it, whose slots instruments= takes")
        features = given[:, columns]

    _held(spec, features, decode)
    overlap, trained = _told(spec, data, held)

    # the environment loads Gymnasium, which no other backtest does
    from ._env import VectorEnv
    actions  = spec["actions"]
    settings = dict(
        window       = spec["window"],
        features     = features,
        layout       = spec["observation"]["layout"],
        max_leverage = actions.get("max_leverage", 1.0),
        actions      = actions["space"],
        max_offset   = actions.get("max_offset", 20),
        decode       = decode,
        fill         = spec["fill"],
        tolerance    = spec["tolerance"],
        autoreset    = "disabled",
        seed         = 0,
    )

    normal = spec["normalisation"]
    scale  = None if normal is None else list(zip(normal["mean"], normal["deviation"]))
    shape  = spec["observation"]["shape"]

    def run(change=None):
        env = VectorEnv._of(data, held, record, scale, shape, balance, costs, change,
                            **settings)
        with _evaluated(agent):
            decided = env._played(_acting(agent, spec), len(data))
        return env, decided

    env, decided = run()

    inner      = env._inner
    rate       = _annualised(data)
    stats      = None if rate is None else inner.stats(0, rate, float(risk_free))
    provenance = _provenance(data, held, _markets(env._record), _described(agent),
                             risk_free, rate, record=record, spec=spec, overlap=overlap,
                             trained_venue=trained)

    full = record == "full"
    if full and baseline:
        def again(change):
            played, _ = run(change)
            quote   = played._record.settings["quote"]
            opening = sum(quote.values()) if isinstance(quote, dict) else quote
            return np.concatenate([[opening], played._inner.equity_curve(0)])

        env._record.policy = again
        _baselines.computed(env._record)

    return Result._of(inner.tables(0, env._first, list(data.names)), stats,
                      decided if full else None, inner.is_bust(0), provenance,
                      env._record)


def _found(agent, spec):
    """The agent spec ``spec`` gives: the spec itself, a path to its JSON or to a
    directory holding ``spec.json``, or the environment the agent trained in, and where
    it is None the one an SB3 agent carries on its action space; refused where there is
    none, or its schema is not the one this probatum reads."""
    if spec is None:
        spec = getattr(agent.action_space, "agent_spec", None) if _sb3(agent) else None
        if spec is None:
            raise SpecError("spec: this agent carries no spec; pass spec= with the env it "
                            "trained on or its JSON")
    elif isinstance(spec, (str, os.PathLike)):
        path = pathlib.Path(spec)
        file = path / "spec.json" if path.is_dir() else path
        try:
            spec = json.loads(file.read_text(encoding="utf-8"))
        except OSError as error:
            raise SpecError(f"spec: {str(file)!r} cannot be read ({error.strerror}); pass "
                            f"the spec.json probatum.save wrote, or its "
                            f"directory") from None
        except ValueError as error:
            raise SpecError(f"spec: {str(file)!r} holds no JSON ({error}); pass the "
                            f"spec.json probatum.save wrote") from None
    elif hasattr(spec, "agent_spec"):
        spec = spec.agent_spec
    elif not isinstance(spec, dict):
        raise ProbatumTypeError(f"spec: {_kind(spec)}; pass the agent spec, a path to its "
                                f"JSON or the environment the agent trained in")

    schema = spec.get("schema") if isinstance(spec, dict) else None
    if schema != SCHEMA:
        raise SpecError(f"spec: schema {schema!r}, and this probatum reads schema "
                        f"{SCHEMA}; pass a spec env.agent_spec recorded")

    return copy.deepcopy(spec)


def _slotted(data, spec, instruments, venue):
    """``data`` as the slots the agent trades, the instruments ``instruments`` names, in
    that order, or every one, beside each slot's place among ``data``'s instruments;
    refused where they are not the spec's slots in number, or not an unsampled spec's
    instruments in order, each the same symbol on the same market, or where a slot is
    of another market than the agent trained on at ``venue``."""
    names = data.names
    if instruments is not None:
        data = _slots(data, instruments)

    slots, k = data.names, spec["slots"]
    typed    = "data" if instruments is None else "instruments"
    if len(slots) != k:
        raise SpecError(f"{typed}: {_count(len(slots), 'instrument')}, "
                        f"{', '.join(slots)}, and the agent trades {_count(k, 'slot')}; "
                        f"name {k} of them with instruments=")

    unsampled = spec["sample"] is None
    if unsampled and not all(_same(name, instrument, own) for name, instrument, own
                             in zip(slots, data.instruments, spec["instruments"])):
        raise SpecError(f"{typed}: {', '.join(slots)}, and the agent trained on "
                        f"{', '.join(spec['instruments'])}, in that order; name them "
                        f"with instruments=")

    origin = venue.name or "the venue"
    for i, (name, instrument) in enumerate(zip(slots, data.instruments)):
        market = _market(instrument, name, venue, origin)
        kind   = spec["markets"][i if unsampled else 0]
        if market != kind:
            raise SpecError(f"{typed}: {name} is {market} at {origin}, and the agent "
                            f"trained on it as {kind}; give it the market it trained on")

    return data, [names.index(name) for name in slots]


def _same(name, instrument, trained):
    """Whether the slot ``name``, ``instrument``, is the instrument the spec names
    ``trained``: the same name, or the same symbol where one of the two Data named it
    with its market or its venue before it, as a Data does where it holds the symbol on
    another, two such names being the same only where they are equal."""
    symbol = instrument.symbol
    return name == trained or symbol is not None and (
        trained == symbol or name == symbol and trained.endswith(f":{symbol}"))


def _told(spec, data, venue):
    """Warn where ``data``'s bars overlap the training span ``spec`` records, ``venue``
    is not the venue it records or the bars are of another interval, and return the
    overlap, the first and the last time both hold, or None, beside whether the venue
    is the one the agent trained at."""
    trained    = spec["training"]
    start, end = (int(np.datetime64(trained[key].removesuffix("Z"), "ns").astype(np.int64))
                  for key in ("start", "end"))
    first, last = int(data.time[0]), int(data.time[-1])

    overlap = None
    if first <= end and last >= start:
        overlap = [_iso(max(first, start)), _iso(min(last, end))]
        warnings.warn(f"data: bars from {_when(first)} to {_when(last)}, which overlap the "
                      f"agent's training span from {_when(start)} to {_when(end)}; give "
                      f"bars after {_when(end)} for a run out of sample", ProbatumWarning,
                      skip_file_prefixes=PACKAGE)

    at = _plain(_venued(venue))
    if at != spec["venue"]:
        warnings.warn(f"venue: {_said(at)}, and the agent trained at "
                      f"{_said(spec['venue'])}; the run judges it at a venue it did not "
                      f"learn", ProbatumWarning, skip_file_prefixes=PACKAGE)

    interval = trained["interval"]
    if data.interval != interval:
        warnings.warn(f"data: {data.interval} bars, and the agent trained on {interval} "
                      f"bars, so its window of {spec['window']} spans another length of "
                      f"time; give {interval} bars", ProbatumWarning,
                      skip_file_prefixes=PACKAGE)

    return overlap, at == spec["venue"]


def _said(venue):
    """What a warning says of a venue as a spec records it."""
    said = venue["name"] or "a venue built by hand"
    if venue["taken"] is not None:
        said += f" taken {venue['taken']}"

    if venue["changed"]:
        said += f" with {', '.join(venue['changed'])} changed"

    return said


def _acting(agent, spec):
    """A function of the observation that returns ``agent``'s action for it, the
    observation first normalised by the statistics SB3's VecNormalize kept where
    ``spec`` holds them: an SB3 model's or policy's deterministic prediction, carrying a
    recurrent policy's state from one close to the next; a torch module's output for it
    as a tensor on the module's device, worked with no gradient; or a function's
    output; the first element of a tuple either returns being the action."""
    stats = spec["sb3"]
    parts = {}

    if stats is not None:
        stats = _decoded_json(stats)
        # a flat observation's statistics are its own, a dict's each key's it normalised
        for key, part in ({None: stats} if "mean" in stats else stats).items():
            if isinstance(part, dict):
                parts[key] = (np.asarray(part["mean"]),
                              np.sqrt(np.asarray(part["var"]) + stats["epsilon"]),
                              stats["clip"])

    def seen(obs):
        if isinstance(obs, dict):
            return {key: _clipped(value, *parts[key]) if key in parts else value
                    for key, value in obs.items()}
        return _clipped(obs, *parts[None]) if None in parts else obs

    if _sb3(agent):
        state = [None, np.ones(1, dtype=bool)]

        def act(obs):
            action, state[0] = agent.predict(seen(obs), state=state[0],
                                             episode_start=state[1], deterministic=True)
            state[1] = np.zeros(1, dtype=bool)
            return action

        return act

    if _module(agent):
        torch  = sys.modules["torch"]
        on     = next(itertools.chain(agent.parameters(), agent.buffers()), None)
        device = torch.device("cpu") if on is None else on.device

        def act(obs):
            obs   = seen(obs)
            given = ({key: torch.as_tensor(value, device=device)
                      for key, value in obs.items()} if isinstance(obs, dict)
                     else torch.as_tensor(obs, device=device))
            with torch.no_grad():
                out = agent(given)
            out = out[0] if isinstance(out, tuple) else out
            return out.cpu().numpy() if isinstance(out, torch.Tensor) else out

        return act

    def act(obs):
        out = agent(seen(obs))
        return out[0] if isinstance(out, tuple) else out

    return act


def _clipped(values, mean, deviation, clip):
    """``values`` as SB3's VecNormalize gives them, over their ``mean`` and
    ``deviation``, clipped at ``clip`` and rounded to float32."""
    return np.clip((values - mean) / deviation, -clip, clip).astype(np.float32)


@contextlib.contextmanager
def _evaluated(agent):
    """``agent`` in eval mode for a run where it is a torch module and no SB3 policy,
    which predicts in it of its own accord, and each of its modules in the mode it was
    in after."""
    module = _module(agent) and not _sb3(agent)
    modes  = [(part, part.training) for part in agent.modules()] if module else []
    if module:
        agent.eval()

    try:
        yield
    finally:
        for part, mode in modes:
            part.training = mode


def _described(agent):
    """What a provenance says of ``agent``."""
    if _sb3(agent):
        return (f"the SB3 {'policy' if _module(agent) else 'model'} "
                f"{type(agent).__name__}")
    if _module(agent):
        return f"the torch module {type(agent).__name__}"
    return f"the {'function' if inspect.isroutine(agent) else 'callable'} {_named(agent)}"


def save(agent, path, spec=None):
    """Write ``agent`` into the directory ``path``, made where it is missing, beside the
    spec it trained with as ``spec.json``: a torch module as its state dict,
    ``weights.pt``, and an SB3 model as its own zip, ``agent.zip``, so no module is
    pickled. ``spec`` is the agent spec, a path to its JSON or the environment the
    agent trained in, an SB3 agent's own where it is left out (ADR 0039)."""
    model = _sb3(agent) and not _module(agent)
    if not (model or _module(agent)):
        raise ProbatumTypeError(f"agent: {_kind(agent)}, which save writes no weights of; "
                                f"pass a torch module or an SB3 model, and keep a "
                                f"function in your own code")

    held   = _found(agent, spec)
    folder = pathlib.Path(path)
    if folder.exists() and not folder.is_dir():
        raise ProbatumValueError(f"path: {str(path)!r} is a file; name a directory")

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.json").write_text(json.dumps(_plain(held), indent=2) + "\n",
                                      encoding="utf-8")

    if model:
        agent.save(folder / "agent.zip")
    else:
        sys.modules["torch"].save(agent.state_dict(), folder / "weights.pt")


def _probe(names, markets, interval):
    """A feature function's probe: 512 bars of each instrument of ``names``, under its
    name and on its market of ``markets``, on ``interval``'s steps from 2024-01-01 00:00
    UTC, drawn with plain arithmetic alone, so they are the same on every machine."""
    width  = len(names)
    x      = np.arange(1, width + 1, dtype=np.uint64)
    bars   = np.empty((ROWS, width, 5))
    opened = np.full(width, 100.0)

    for t in range(ROWS):
        draws = []
        for _ in range(4):
            x = x * A + C
            draws.append(_drawn(x))

        r, h, low, v  = draws
        close         = opened * (1.0 + 0.02 * (r - 0.5))
        bars[t, :, 0] = opened
        bars[t, :, 1] = np.maximum(opened, close) * (1.0 + 0.005 * h)
        bars[t, :, 2] = np.minimum(opened, close) * (1.0 - 0.005 * low)
        bars[t, :, 3] = close
        bars[t, :, 4] = 1000.0 * (1.0 + v)
        opened        = close

    times = START + np.arange(ROWS, dtype=np.int64) * INTERVALS[interval]

    # one Data a market, joined, then the instruments in the training order, as a join
    # of each instrument in turn would take a time that grows with their cube
    probe = None
    for market in dict.fromkeys(markets):
        held = [i for i, own in enumerate(markets) if own == market]
        part = Data(
            bars[:, held],
            symbol   = [names[i] for i in held],
            market   = market,
            time     = times,
            interval = interval,
        )
        probe = part if probe is None else probe.join(part)

    return probe.select(list(names))


def _probed(features, names, markets, interval):
    """What the function ``features`` gives on the probe of the instruments ``names``
    on ``markets`` and ``interval``, refused where it is no ``(512, N, F)`` array."""
    out   = np.asarray(features(_probe(names, markets, interval)), dtype=np.float64)
    shape = (ROWS, len(names))
    if out.ndim != 3 or out.shape[:2] != shape:
        raise _Shaped(f"shape {out.shape} on the agent spec's probe of {shape[0]} bars "
                      f"of {shape[1]} instruments")
    return out


def _probe_actions(shape, kind, lows, choices):
    """A decode's probe: 8 rows of actions, each of the single action space's
    ``shape``, of ``kind``, over slots of the lower bounds ``lows`` and with discrete
    actions the ``choices``, each value drawn in turn, its components in order, a
    box's and a simplex's as float64 and a discrete one's as int64."""
    if kind == "box":
        components = [("box", low, 1.0) for low in lows]
    elif kind == "simplex":
        components = [("simplex",)] * (len(lows) + 1)
    else:
        components = [("discrete", n) for n in choices]

    x, rows = np.array([2**32], dtype=np.uint64), []
    for _ in range(ROWS_OF_ACTIONS):
        row = []
        for component in components:
            x = x * A + C
            u = float(_drawn(x)[0])
            if component[0] == "box":
                row.append(component[1] + (component[2] - component[1]) * u)
            elif component[0] == "simplex":
                row.append(u)
            else:
                row.append(int(np.floor(u * component[1])))
        rows.append(row)

    dtype = np.int64 if kind == "discrete" else np.float64
    return np.array(rows, dtype=dtype).reshape((ROWS_OF_ACTIONS,) + tuple(shape))


def _decoded(decode, probe, k):
    """What ``decode`` gives for the ``probe`` actions over a fresh flat account of
    ``k`` slots, every equity 10,000 and every other value zero."""
    n     = len(probe)
    state = {name: np.zeros(n) for name in ("log_return", "drawdown", "leverage",
                                             "turnover", "fees", "funding")}
    state.update(equity=np.full(n, 10_000.0), previous_equity=np.full(n, 10_000.0),
                 liquidated=np.zeros(n, dtype=bool), weights=np.zeros((n, k)),
                 positions=np.zeros((n, k)))

    return np.asarray(decode(probe, State(state)), dtype=np.float64)


def _space(spec):
    """The shape of the single action space of the actions ``spec`` records, their
    kind, each slot's lower bound and with discrete actions its choices."""
    held = spec["actions"]
    kind = held["space"]
    k    = spec["slots"]

    if kind == "box":
        return (k,), kind, np.array(held["low"]), None
    if kind == "simplex":
        return (k + 1,), kind, np.zeros(k), None
    return (() if k == 1 else (k,)), kind, np.zeros(k), np.array(held["choices"])


def _digest(values):
    """A reference: blake2b's 8 bytes over the shape of ``values``, each a little-endian
    64-bit integer, and the values rounded to little-endian float32, every NaN one NaN
    and every zero +0.0."""
    with np.errstate(over="ignore"):
        held = np.asarray(values, dtype=np.float64).astype(np.float32)

    held[np.isnan(held)] = np.nan
    held[held == 0.0]    = 0.0

    out = hashlib.blake2b(digest_size=8)
    out.update(np.asarray(held.shape, dtype="<i8").tobytes())
    out.update(held.astype("<f4").tobytes())
    return "b2:" + out.hexdigest()


def _source(function):
    """A source hash: blake2b's 8 bytes over ``function``'s source text, or None where
    the source cannot be read."""
    try:
        text = inspect.getsource(function)
    except (OSError, TypeError):
        return None

    return "b2:" + hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _named(function):
    return getattr(function, "__qualname__", None) or type(function).__name__


def _drawn(x):
    """Each draw ``x`` gives, its top 53 bits as a float from 0 to 1."""
    return (x >> np.uint64(11)).astype(np.float64) / 2.0**53


def _iso(ns):
    return str(np.datetime64(int(ns), "ns").astype("datetime64[s]")) + "Z"


def _moments(values, candles, printed, shown, window):
    """Each market feature's mean and deviation over every value the observations hold
    at the bars ``shown`` names, each bar of an instrument's window it ``printed``:
    ``values``, the ``(T, N, F)`` features, or None for the ``candles``, whose volume
    counts only where its window's mean volume is above zero. A feature that never
    varies, or holds no value, takes a deviation of 1, and one of no value a mean of
    0 (Specification, The observation)."""
    rows, width = printed.shape

    if values is not None:
        # how many observations hold bar r of instrument i: those at the bars from r
        # to r + W - 1 that show it
        held   = np.concatenate([np.zeros((1, width)), np.cumsum(shown, axis=0)])
        ends   = np.minimum(np.arange(rows) + window, rows)
        counts = (held[ends] - held[:rows]) * printed
        weight = counts[:, :, None]
        filled = np.where(weight > 0.0, values, 0.0)
        total  = np.full(values.shape[2], weight.sum())
        means  = (weight * filled).sum(axis=(0, 1)) / np.where(total > 0.0, total, 1.0)
        spread = (weight * (filled - means) ** 2).sum(axis=(0, 1))
        lows   = np.where(weight > 0.0, values, np.inf).min(axis=(0, 1))
        highs  = np.where(weight > 0.0, values, -np.inf).max(axis=(0, 1))
        return _paired(means, spread, total, lows, highs)

    bars   = candles.reshape(rows, width, 5)
    prices = np.where(printed[:, :, None], bars[:, :, :4], 0.0)
    volume = np.where(printed, bars[:, :, 4], 0.0)

    # each window's printed volume, which a bar's volume is over the mean of
    summed = np.concatenate([np.zeros((1, width)), np.cumsum(volume, axis=0)])
    seen   = np.concatenate([np.zeros((1, width)), np.cumsum(printed, axis=0)])
    starts = np.maximum(np.arange(rows) - window + 1, 0)
    traded = summed[1:] - summed[starts]
    count  = seen[1:] - seen[starts]
    mean   = np.where(count > 0.0, traded / np.where(count > 0.0, count, 1.0), 0.0)
    last   = np.where(shown, bars[:, :, 3], 1.0)
    sums   = [np.zeros(5), np.zeros(5), np.zeros(5)]
    lows, highs = np.full(5, np.inf), np.full(5, -np.inf)

    for times in (1, 2):
        means = None if times == 1 else sums[1] / np.where(sums[0] > 0.0, sums[0], 1.0)
        for d in range(window):
            at    = shown[d:] & printed[:rows - d]
            price = prices[:rows - d] / last[d:, :, None]
            moved = at & (mean[d:] > 0.0)
            ratio = volume[:rows - d] / np.where(moved, mean[d:], 1.0)
            for g in range(5):
                x, on = (price[:, :, g], at) if g < 4 else (ratio, moved)
                if times == 2:
                    sums[2][g] += ((x[on] - means[g]) ** 2).sum()
                elif on.any():
                    sums[0][g] += on.sum()
                    sums[1][g] += x[on].sum()
                    lows[g]  = min(lows[g], x[on].min())
                    highs[g] = max(highs[g], x[on].max())

    means = sums[1] / np.where(sums[0] > 0.0, sums[0], 1.0)
    return _paired(means, sums[2], sums[0], lows, highs)


def _paired(means, spread, count, lows, highs):
    """Each feature's mean and deviation as a pair, from its ``means``, its summed
    squared ``spread`` about the mean, its ``count`` of values and the ``lows`` and
    ``highs`` they span: a feature whose values lie within a billionth of their size of
    one another never varies, and takes its smallest value and a deviation of 1, as
    does a deviation whose squares underflow to zero."""
    out = []
    for mean, squares, n, low, high in zip(means, spread, count, lows, highs):
        if n <= 0.0:
            out.append((0.0, 1.0))
        elif high - low <= 1e-9 * max(abs(low), abs(high)):
            out.append((float(low), 1.0))
        else:
            deviation = float(np.sqrt(squares / n))
            out.append((float(mean), deviation if deviation > 0.0 else 1.0))

    return out
