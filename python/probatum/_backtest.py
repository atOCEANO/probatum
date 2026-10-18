"""``backtest``: one call that runs a policy on a Data at a venue and returns its
Result, reading the policy in one fixed order, so no input is read two ways
(ADR 0024)."""

import copy
import math
import sys
import types

import numpy as np

from ._data import DAY, Data, _columns, _dated, _renamed, _when
from ._probatum import Engine, arrow_columns, periods_per_year, run_array
from ._result import Result, _Record, _provenance
from ._strategy import Strategy, _Run, _called, _play
from ._venue import _run, _venue_of

SIZINGS = ("weight", "quantity")


def backtest(policy, data, venue, sizing="weight", stop=None, take_profit=None,
             tolerance=0.0, risk_free=0.0):
    """Run ``policy`` on ``data``, a Data or anything one is built from, at ``venue``,
    a Venue or a preset's name, and return its `Result`.

    The policy is read in one order. A table, anything exporting the Arrow C stream
    interface or a pandas frame or series, has its rows matched to the Data's clock by
    their times and its columns to the instruments by name, and runs as an array. An
    array, anything with ``__dlpack__``, the buffer protocol, ``__array__`` or
    ``__arrow_c_array__``, is ``(T, N)`` over the Data's rows and instruments, or
    ``(T,)`` for one instrument, converted once to float64 and run whole in Rust: row
    t is the decision at the close of bar t, a weight of equity or, with ``sizing``
    of ``"quantity"``, the position wanted, in base units or contracts, a NaN holding
    and the last row never read. A class is refused for an instance and a model that
    fits for its predictions, and an agent is not run yet. A Strategy runs on a deep
    copy of itself, which is the Result's ``policy``, deciding at each close.

    ``stop`` and ``take_profit`` give each round trip of an array run its barriers,
    each a return from the average entry, a number, or an array or a table of them
    read on the row whose decision opened it, NaN for none; ``tolerance`` skips each
    leg below that fraction of equity; and ``risk_free`` is the annual rate the
    statistics are computed against (Specification, An array; Barriers on an array).
    """
    data = data if isinstance(data, Data) else Data(data)
    if not len(data) or not data.available.any():
        raise ValueError("data: no bars to run; give a Data that has some")

    if _table_like(policy):
        rows = _table("policy", policy, data)
    elif _array_like(policy):
        rows = _shaped("policy", _floats("policy", _host(policy, "policy")), data)
    elif isinstance(policy, type):
        raise TypeError(f"policy: got the class {policy.__name__}, not an instance; "
                        f"pass {policy.__name__}()")
    elif isinstance(policy, Strategy):
        return _decided(policy, data, venue, sizing, stop, take_profit, tolerance,
                        risk_free)
    elif hasattr(policy, "fit") and not callable(policy):
        raise TypeError(f"policy: {_kind(policy)}, a model that fits, which backtest "
                        f"does not fit; pass its predictions as an array")
    elif _agent(policy) or callable(policy):
        raise NotImplementedError(f"policy: {_kind(policy)}, which backtest does not run "
                                  f"as an agent yet; pass its decisions as an array")
    else:
        raise TypeError(f"policy: {_kind(policy)}; pass a (T, N) array of weights or "
                        f"quantities, or a table of them")

    _finite("policy", rows, data)
    if sizing not in SIZINGS:
        raise ValueError(f"sizing: {sizing!r}; use \"weight\" or \"quantity\"")
    if not (_number(tolerance) and tolerance >= 0.0):
        raise ValueError(f"tolerance: {tolerance!r}; give a fraction of equity from 0, "
                         f"such as 0.01")
    if not _number(risk_free):
        raise ValueError(f"risk_free: {risk_free!r}; give the annual rate as a finite "
                         f"number, such as 0.04")

    barriers = {name: _barrier(name, given, data)
                for name, given in (("stop", stop), ("take_profit", take_profit))}

    held  = _repeatable(venue)
    run   = _run(held, data)
    names = data.names

    for k, market in enumerate(_markets(run)):
        if market == "spot" and (rows[:, k] < 0.0).any():
            t = int(np.flatnonzero(rows[:, k] < 0.0)[0])
            raise ValueError(f"policy: {rows[t, k]} at row {t:,} "
                             f"({_when(data.time[t])}) for {names[k]}, which is spot and "
                             f"cannot be short; trade a perpetual instead")

    values, chosen = _seen(rows, data.available.any(axis=1))
    engine = Engine(run.candles, report=True, **run.settings)

    # what the run is played with, which its Result keeps to play again without costs
    played = (values, sizing, float(tolerance),
              *[None if barrier is None else
                barrier if isinstance(barrier, float) else
                np.ascontiguousarray(np.take_along_axis(barrier, chosen, axis=0))
                for barrier in barriers.values()])
    run_array(engine, *played, start=run.start)

    rate       = _annualised(data)
    stats      = None if rate is None else engine.stats(rate, float(risk_free))
    provenance = _provenance(
        data, held, _markets(run), _described(rows, sizing), risk_free, rate,
        sizing=sizing, tolerance=float(tolerance),
        **{name: _shown(barrier) for name, barrier in barriers.items()})

    rows.flags.writeable = False
    return Result._of(engine.tables(run.start, list(names)), stats, rows,
                      engine.is_bust(), provenance,
                      _Record(held, data, run, policy=played))


def _decided(strategy, data, venue, sizing, stop, take_profit, tolerance, risk_free):
    """``strategy`` run on ``data`` at ``venue`` through a deep copy of it, which places
    its own orders, so the settings of an array run are refused beside it."""
    for name, given in (("stop", stop), ("take_profit", take_profit)):
        if given is not None:
            raise ValueError(f"{name}: {given!r} beside a Strategy, which places its own "
                             f"exits; place them with account.bracket")

    if sizing != "weight":
        raise ValueError(f"sizing: {sizing!r} beside a Strategy, whose verbs say their "
                         f"own units; leave it out")

    if tolerance != 0.0:
        raise ValueError(f"tolerance: {tolerance!r} beside a Strategy; pass it to "
                         f"account.target")

    if not _number(risk_free):
        raise ValueError(f"risk_free: {risk_free!r}; give the annual rate as a finite "
                         f"number, such as 0.04")

    held   = _repeatable(venue)
    run    = _run(held, data)
    engine = Engine(run.candles, report=True, **run.settings)

    # the second copy, taken at the call as the first is, is the one the Result plays
    # again without costs (ADR 0024)
    kept   = copy.deepcopy(strategy)
    copied = copy.deepcopy(strategy)

    played = _Run(engine, data, run.start, _markets(run))
    _play(copied, played)

    rate       = _annualised(data)
    stats      = None if rate is None else engine.stats(rate, float(risk_free))
    provenance = _provenance(data, held, _markets(run), _called(strategy), risk_free,
                             rate)

    return Result._of(engine.tables(run.start, list(data.names)), stats, copied,
                      engine.is_bust(), provenance,
                      _Record(held, data, run, played.token, policy=kept))


def _repeatable(venue):
    """``venue`` as a Venue, refused where it decides at random, so that a backtest
    repeats without a seed."""
    held = _venue_of(venue)
    if held._intrabar == "random":
        raise ValueError("venue: intrabar is \"random\", which backtest refuses so that "
                         "a backtest repeats without a seed; replace(intrabar="
                         "\"pessimistic\") or \"optimistic\"")
    return held


def _markets(run):
    """The market of each instrument a `Run` trades."""
    return [spec["market"] for spec in run.settings.get("instruments", [run.settings])]


def _number(value):
    """Whether ``value`` is a finite number and no bool."""
    return (isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
            and math.isfinite(value))


def _kind(value):
    """What ``value`` is, for a refusal to name."""
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        return "a function"
    return f"an object of type {type(value).__name__}"


def _buffer(value):
    """Whether ``value`` exports the buffer protocol."""
    try:
        memoryview(value)
    except TypeError:
        return False
    return True


def _table_like(value):
    """Whether ``value`` is a table: an Arrow C stream, or a pandas frame or series."""
    pandas = type(value).__module__.split(".")[0] == "pandas"
    return hasattr(value, "__arrow_c_stream__") or pandas and hasattr(value, "index")


def _array_like(value):
    """Whether ``value`` is an array by one of the protocols the boundary reads."""
    return (hasattr(value, "__dlpack__") or hasattr(value, "__array__")
            or hasattr(value, "__arrow_c_array__") or _buffer(value))


def _agent(policy):
    """Whether ``policy`` is a Stable-Baselines3 model or policy or a torch module, each
    recognised only where its framework is already imported."""
    sb3 = sys.modules.get("stable_baselines3")
    if sb3 is not None:
        common = sys.modules.get("stable_baselines3.common.policies")
        kinds  = [sb3.common.base_class.BaseAlgorithm]
        if common is not None:
            kinds.append(common.BasePolicy)
        if isinstance(policy, tuple(kinds)):
            return True

    torch = sys.modules.get("torch")
    return torch is not None and isinstance(policy, torch.nn.Module)


def _host(value, name):
    """``value``, an array argument, as a numpy array on the host: through DLPack,
    asking a device array for a host copy, then the buffer protocol, ``__array__``
    and the Arrow array interface. A host array DLPack cannot carry, one byte-swapped
    or of objects, is read the other ways."""
    if hasattr(value, "__dlpack__"):
        try:
            try:
                return np.from_dlpack(value, device="cpu")
            except TypeError:
                return np.from_dlpack(value)
        except (BufferError, RuntimeError, TypeError, ValueError) as error:
            try:
                kind = value.__dlpack_device__()[0]
            except Exception:
                kind = None

            # 1 is DLPack's CPU
            if kind != 1:
                raise ValueError(f"{name}: an array on a device the host cannot read "
                                 f"({error}); pass a host copy") from None

    if (hasattr(value, "__arrow_c_array__") and not hasattr(value, "__array__")
            and not _buffer(value)):
        try:
            columns = arrow_columns(value)
        except (TypeError, ValueError) as error:
            raise _renamed(name, error) from None

        if len(columns) != 1:
            raise TypeError(f"{name}: an Arrow array of {len(columns)} columns; pass "
                            f"it as a table, or one column as an array")

        [(_, _, value, _)] = columns

    return np.asarray(value)


def _floats(name, array):
    """``array`` as float64, a copy the caller cannot reach, refusing booleans, which
    would read False as a zero."""
    if array.dtype.kind == "b":
        fix = ("give weights, 1.0 to enter, 0.0 to leave and NaN to hold"
               if name == "policy" else "give returns from the entry above 0")
        raise TypeError(f"{name}: an array of booleans; {fix}")

    try:
        return np.array(array, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"{name}: an array of {array.dtype} values; pass an array of "
                        f"floats") from None


def _shaped(name, array, data):
    """``array`` as ``(T, N)`` over ``data``'s rows and instruments, ``(T,)`` taken for
    one instrument."""
    rows, width = len(data), len(data.names)
    given = array.shape

    if array.ndim == 1 and width == 1:
        array = array[:, None]

    if array.ndim == 1 and len(array) == rows:
        raise ValueError(f"{name}: a (T,) array for a universe of {width} instruments; "
                         f"pass a (T, {width}) array")

    if array.shape != (rows, width):
        raise ValueError(f"{name}: an array of shape {given} for a Data of {rows:,} rows "
                         f"and {width} instrument{'s' if width > 1 else ''}; pass one of "
                         f"({rows}, {width})")

    return array


def _finite(name, rows, data):
    """Refuse an infinite value among ``rows``, naming its row and instrument."""
    bad = np.isinf(rows)
    if bad.any():
        t, k = map(int, np.argwhere(bad)[0])
        raise ValueError(f"{name}: {rows[t, k]} at row {t:,} ({_when(data.time[t])}) "
                         f"for {data.names[k]}; give a finite value, or NaN to hold")


def _table(name, given, data):
    """A table's values on ``data``'s clock and instruments: its rows matched by their
    times, in a datetime index or column or a ``timestamp`` column of milliseconds,
    dated as the Data dates daily bars, and its columns by name, a pandas series being
    a table of one column, named for the Data's one instrument where it carries no
    name. A row the table leaves out is NaN where no instrument printed a bar, which
    the account never sees, and refused where one did."""
    names = data.names

    if type(given).__module__.split(".")[0] == "pandas" and not hasattr(given,
                                                                         "columns"):
        label = given.name if given.name is not None or len(names) > 1 else names[0]
        if label is None:
            raise ValueError(f"{name}: an unnamed pandas series for a universe of "
                             f"{len(names)} instruments; name it after one of "
                             f"{', '.join(names)}, or pass a frame of them")
        given = given.to_frame(name=label)

    try:
        clocks, numbers, others = _columns(given)
    except (TypeError, ValueError) as error:
        raise _renamed(name, error) from None

    numbers = {str(label): values for label, values in numbers.items()}
    others  = {str(label): held for label, held in others.items()}
    if not clocks and getattr(numbers.get("timestamp"), "dtype",
                              np.dtype(float)).kind == "i":
        clocks["timestamp"] = (numbers.pop("timestamp") * 1_000_000, "UTC")

    if len(clocks) > 1:
        raise ValueError(f"{name}: a table with times in {', '.join(map(str, clocks))}; "
                         f"keep each row's bar open time alone")

    if not clocks:
        raise ValueError(f"{name}: a table with no times; give each row's bar open time "
                         f"in a datetime index or column or a timestamp column of "
                         f"milliseconds, or pass its values as an array")

    times, zone = next(iter(clocks.values()))
    times = np.asarray(times, dtype=np.int64)
    if data._root.step >= DAY:
        try:
            times = _dated(times, zone)
        except ValueError as error:
            raise _renamed(name, error) from None

    for label in names:
        if label in others:
            raise ValueError(f"{name}: a table's column {label} holds {others[label]}; "
                             f"give numbers")

    missing = [label for label in names if label not in numbers]
    if missing:
        raise ValueError(f"{name}: a table with no column for {missing[0]}; give one "
                         f"for each of {', '.join(names)}")

    extra = [label for label in numbers if label not in names]
    if extra:
        raise ValueError(f"{name}: a table's column {extra[0]}, which the data does not "
                         f"hold; give columns for {', '.join(names)} alone")

    clock = data.time
    at    = np.searchsorted(clock, times)
    off   = (at >= len(clock)) | (clock[np.minimum(at, len(clock) - 1)] != times)
    if off.any():
        when = _when(times[np.flatnonzero(off)[0]])
        raise ValueError(f"{name}: a table row at {when}, which is not on the data's "
                         f"clock; give rows at the bars' open times")

    counts = np.bincount(at, minlength=len(clock))
    if (counts > 1).any():
        when = _when(clock[np.flatnonzero(counts > 1)[0]])
        raise ValueError(f"{name}: a table with two rows at {when}; give one a bar")

    absent = (counts == 0) & data.available.any(axis=1)
    if absent.any():
        when = _when(clock[np.flatnonzero(absent)[0]])
        raise ValueError(f"{name}: a table with no row at {when}, a bar the data's "
                         f"instruments printed; give a row for each, NaN to hold")

    rows = np.full((len(clock), len(names)), np.nan)
    for k, label in enumerate(names):
        rows[at, k] = np.asarray(numbers[label], dtype=np.float64)

    return rows


def _barrier(name, given, data):
    """A barrier as the run takes it: None, a return above zero for every row, or
    ``(T, N)`` of them from an array or a table, NaN for none."""
    if given is None:
        return None

    if getattr(given, "dtype", np.dtype(float)).kind == "b" or isinstance(given, bool):
        raise TypeError(f"{name}: {given!r}; give a return from the entry above 0, such "
                        f"as 0.04")

    if isinstance(given, (int, float, np.number)) or getattr(given, "ndim", None) == 0:
        value = float(given)
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name}: {value!r}; give a return from the entry above 0, "
                             f"such as 0.04")
        return value

    if _table_like(given):
        values = _table(name, given, data)
    elif _array_like(given):
        values = _shaped(name, _floats(name, _host(given, name)), data)
    else:
        raise TypeError(f"{name}: {_kind(given)}; give a return from the entry above 0, "
                        f"or a (T, N) array of them")

    bad = ~(np.isnan(values) | (np.isfinite(values) & (values > 0.0)))
    if bad.any():
        t, k = map(int, np.argwhere(bad)[0])
        raise ValueError(f"{name}: {values[t, k]} at row {t:,} ({_when(data.time[t])}) "
                         f"for {data.names[k]}; give a return above 0, or NaN for none")

    return values


def _seen(rows, printed):
    """The rows the account sees, one for each row any instrument printed from the
    first, each holding the values of the rows after it that none printed, a later
    one's replacing an earlier one's, and the row each value came from."""
    kept   = np.flatnonzero(printed)
    values = rows[kept]
    chosen = np.broadcast_to(kept[:, None], values.shape).copy()

    if len(kept) < len(rows):
        at = np.cumsum(printed) - 1
        for t in np.flatnonzero(~printed):
            if at[t] < 0:
                continue

            given = ~np.isnan(rows[t])
            values[at[t], given] = rows[t, given]
            chosen[at[t], given] = t

    return np.ascontiguousarray(values), chosen


def _annualised(data):
    """The bars a year of the whole Data ``data`` is drawn from, counted on the rows its
    instruments printed, so every slice of it annualises alike (Specification,
    Annualisation)."""
    root    = data._root
    printed = root.available[:, list(data._cols)].any(axis=1)
    return periods_per_year(np.ascontiguousarray(root.time[printed]))


def _described(rows, sizing):
    """What a provenance says of an array policy."""
    unit = "weights" if sizing == "weight" else "quantities"
    return f"a ({rows.shape[0]}, {rows.shape[1]}) array of {unit}"


def _shown(barrier):
    """What a provenance says of a barrier."""
    if barrier is None or isinstance(barrier, float):
        return barrier
    return f"a {barrier.shape} array"
