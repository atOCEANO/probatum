"""The causality check: a policy whose decisions do not repeat on the same bars, or whose
decisions on bars that had closed change when the bars after them are cut away, is
refused before a search trusts it (ADR 0032; Specification, The causality check)."""

import copy
import math

import numpy as np

from ._backtest import (_array_like, _barrier, _host, _kind, _markets, _played, _rows,
                        _sized, _table_like)
from ._data import _before, _when
from ._errors import LookaheadError, PolicyError, ProbatumError, ProbatumValueError
from ._probatum import Engine
from ._strategy import Strategy, _play, _Run, _warmup
from ._venue import _run

GIVEN = ("policy", "stop", "take_profit", "tolerance", "sizing")
# the relative bounds a float is held to, float64's and a float32 a strategy passes
WIDE, SINGLE = 1e-9, 8.0 * float(np.finfo(np.float32).eps)
FLOAT32 = float(np.finfo(np.float32).max)
CAUSAL  = "decide from the bars up to each close"
SEEDED  = "seed every random draw the policy makes, so the check can compare its runs"


def _read(given, data):
    """What a function returned, weights, read as backtest reads them, an array or a
    table of them or a dict of backtest's arguments whose policy is one: its rows, its
    barriers, its sizing and its tolerance."""
    sizing, tolerance, barriers = "weight", 0.0, {"stop": None, "take_profit": None}

    if isinstance(given, dict):
        unknown = [key for key in given if key not in GIVEN]
        if unknown:
            raise PolicyError(f"policy: returned a dict holding {unknown[0]!r}; hold "
                              f"{', '.join(GIVEN)} alone")
        if "policy" not in given:
            raise PolicyError("policy: returned a dict with no policy; give it the "
                              "weights as \"policy\"")

        sizing, tolerance = given.get("sizing", sizing), given.get("tolerance", tolerance)
        barriers = {name: given.get(name) for name in barriers}
        given    = given["policy"]

    if not (_table_like(given) or _array_like(given)):
        raise PolicyError(f"policy: returned {_kind(given)}; return weights as an array "
                          f"or a table, a dict of backtest's arguments, or a Strategy")

    rows = _rows(given, data)
    _sized(sizing, tolerance)
    barriers = {name: _barrier(name, value, data) for name, value in barriers.items()}

    return rows, barriers, sizing, tolerance


def _scales(given, rows, barriers):
    """The relative bound the weights a function returned are held to, and each of its
    barriers, by the float each was returned in, ``rows`` and ``barriers`` the values
    the reader read of them."""
    held = given if isinstance(given, dict) else {"policy": given}
    return (_scale(held.get("policy"), "policy", rows),
            {name: _scale(held.get(name), name, barriers[name])
             for name in ("stop", "take_profit")})


def _scale(given, name, read):
    """The relative bound ``given``, an array or a table returned as ``name`` whose
    values the reader read as ``read``, is held to: float64's, or eight times the
    epsilon of the narrowest float it holds, a table read through Arrow holding float32
    where every value it holds is one."""
    if type(given).__module__.split(".")[0] == "pandas":
        kinds = [getattr(kind, "numpy_dtype", kind) for kind in (
            given.dtypes if hasattr(given, "columns") else [given.dtype])]
    elif _table_like(given):
        # Arrow hands a float32 column over as float64 values, so they say what it was
        values = read[~np.isnan(read)]
        single = values.size and (values.astype(np.float32) == values).all()
        kinds  = [np.dtype(np.float32)] if single else []
    elif _array_like(given):
        kinds = [_host(given, name).dtype]
    else:
        kinds = []

    eps = [np.finfo(kind).eps for kind in kinds
           if isinstance(kind, np.dtype) and kind.kind == "f"]
    widest = float(np.finfo(np.float64).eps)
    return WIDE if max(eps, default=widest) <= widest else 8.0 * float(max(eps))


class _Weights:
    """A function's weights on a Data's rows, read as backtest reads them, with the
    relative bound a float of them is held to and each barrier array's."""


    def __init__(self, rows, barriers, sizing, tolerance, scale, scales=None):
        self.rows, self.barriers = rows, barriers
        self.sizing, self.tolerance, self.scale = sizing, float(tolerance), scale
        self.scales = scales or {}


    def first(self, printed):
        named = printed[~np.isnan(self.rows[printed]).all(axis=1)]
        return int(named[0]) if len(named) else None


    def on(self, row, k=None):
        """The weights of ``row``, or instrument ``k``'s, None where one holds."""
        if k is not None:
            value = float(self.rows[row, k])
            return None if math.isnan(value) else value

        return tuple(None if math.isnan(value) else value
                     for value in self.rows[row].tolist())


    def empty(self, row, k=None):
        return bool(np.isnan(self.rows[row] if k is None else self.rows[row, k]).all())


    def changes(self, rows, before, k):
        """The positions in ``rows`` where instrument ``k``'s weight changes to a value
        from the one of the row before, ``before`` before the first."""
        values = self.rows[rows, k]
        last   = np.concatenate([[np.nan if before is None else self.rows[before, k]],
                                 values[:-1]])
        return np.flatnonzero(~np.isnan(values) & (values != last)).tolist()


class _Calls:
    """A Strategy's calls on its account, row by row, and the warm-up its init set."""


    def __init__(self, calls, warmup):
        self.calls  = {row: tuple(made) for row, made in calls.items()}
        self.warmup = warmup


    def first(self, printed):
        later = printed[printed >= self.warmup]
        return int(later[0]) if len(later) else None


    def on(self, row, k=None):
        """The calls of ``row``, or those that concern instrument ``k``."""
        made = self.calls.get(row, ())
        if k is None or not made:
            return made

        return tuple(part for part in (_about(call, k) for call in made)
                     if part is not None)


    def empty(self, row, k=None):
        return not self.on(row, k)


    def changes(self, rows, before, k):
        """The positions in ``rows`` where the calls that concern instrument ``k``
        change to some from those of the row before, ``before`` before the first."""
        out, last = [], () if before is None else self.on(before, k)
        for at, row in enumerate(rows):
            now = self.on(row, k)
            if now and now != last:
                out.append(at)
            last = now

        return out


def _about(call, k):
    """What of ``call`` concerns instrument ``k``: all of a call that names it or names
    none, a target's weight for it, and nothing of a call that names another."""
    verb, args = call
    if verb == "target" and isinstance(args[0], dict):
        return (verb, (args[0][k], *args[1:])) if k in args[0] else None

    if verb in ("order", "order_value", "bracket", "close", "cancel_all"):
        return call if args[-1] is None or args[-1] == k else None

    return call


def _refused(error):
    """Mark ``error`` as a refusal of the decisions a policy made, by a verb or by the
    reader, which a cut reports as a lookahead, where an error the policy's own code
    raises stays its own."""
    error._refused = True


class _Recorder:
    """An engine that writes each verb it is called with into ``calls``, under the row
    of ``run``'s decision, and is the engine in everything else."""

    VERBS = frozenset(("order", "order_value", "bracket", "close", "cancel",
                       "cancel_all", "replace", "target", "transfer"))


    def __init__(self, engine):
        self.engine, self.run, self.calls = engine, None, {}


    def __getattr__(self, name):
        found = getattr(self.engine, name)
        if name not in self.VERBS:
            return found

        def called(*args):
            self.calls.setdefault(self.run.row, []).append((name, args))
            try:
                return found(*args)
            except ProbatumError as error:
                _refused(error)
                raise

        return called


class _Checked(_Run):
    """A strategy's run in the check, whose verbs mark what they refuse."""


def _marking(verb):
    def marked(self, *args):
        try:
            return verb(self, *args)
        except ProbatumError as error:
            _refused(error)
            raise

    return marked


for _verb in ("order", "value", "bracket", "close", "cancel", "cancel_all", "replace",
              "target", "transfer"):
    setattr(_Checked, _verb, _marking(getattr(_Run, _verb)))


def _calls(strategy, data, venue, last):
    """The calls ``strategy`` makes on ``data`` at ``venue``, on an engine of its own,
    at every close from its warm-up on, with ``last`` at its last close too."""
    run    = _run(venue, data)
    engine = _Recorder(Engine(run.candles, report=True, **run.settings))
    played = engine.run = _Checked(engine, data, run.start, _markets(run))
    _play(strategy, played, last=last)

    return _Calls(engine.calls, _warmup(strategy))


def _decided(policy, kind, params, data, venue, run, last=False):
    """What ``policy`` decides with ``params`` on ``data``: a Strategy's calls, at its
    last close too with ``last``, or a function's weights, each run built afresh and a
    Strategy copied before it runs, as a trial builds it."""
    if kind == "class":
        return _calls(copy.deepcopy(policy(**params)), data, venue, last)

    given = policy(data, **params)
    if isinstance(given, Strategy):
        return _calls(copy.deepcopy(given), data, venue, last)

    try:
        rows, barriers, sizing, tolerance = _read(given, data)
        # the refusal a trial's run would make of what it holds
        _played(rows, barriers, sizing, tolerance, data, run)
    except ProbatumError as error:
        _refused(error)
        raise

    return _Weights(rows, barriers, sizing, tolerance, *_scales(given, rows, barriers))


def _check(policy, kind, params, data, venue, run):
    """Check ``policy`` with ``params`` on ``data``, the bars a search runs on, for
    decisions on bars that had not closed: True where it was checked, False where it
    decided nothing at or after its warm-up on a row before its last printed one, and
    refused where its decisions do not repeat or a cut changes them."""
    printed = np.flatnonzero(data.available.any(axis=1))
    whole   = _whole(lambda: _decided(policy, kind, params, data, venue, run))
    first   = whole.first(printed)
    rows    = [] if first is None else [int(row) for row in printed[:-1] if row >= first]

    if not any(not whole.empty(row) for row in rows):
        return False

    again  = _whole(lambda: _decided(policy, kind, params, data, venue, run))
    differ = _difference(whole, again, len(data), data)
    if differ is not None:
        raise ProbatumValueError(_said(differ, "in one run on the in-sample bars",
                                       "in the next", SEEDED))

    for cut in _cuts(whole, rows, printed, len(data.names)):
        end = _when(data.time[cut])
        # the rows up to the next printed one, whose values a run plays at the cut's
        # close
        stop    = int(printed[np.searchsorted(printed, cut, side="right")])
        shorter = _before(data, stop)

        try:
            given = _decided(policy, kind, params, shorter, venue, run, last=True)
        except Exception as error:
            if getattr(error, "_refused", False):
                raise LookaheadError(f"policy: on the bars up to {end} its "
                                     f"decisions are refused ({error}), and on the "
                                     f"in-sample bars they are not; {CAUSAL}") from error
            error.add_note(f"raised by the policy on the bars up to {end}, in search's "
                           f"causality check")
            raise

        differ = _difference(whole, given, stop, data)
        if differ is not None:
            raise LookaheadError(_said(differ, "on the in-sample bars",
                                       f"on the bars up to {end}", CAUSAL))

    return True


def _whole(decide):
    """``decide()`` on the in-sample bars, an error it raises carrying a note that names
    them."""
    try:
        return decide()
    except Exception as error:
        error.add_note("raised by the policy on the in-sample bars, in search's "
                       "causality check")
        raise


def _cuts(whole, rows, printed, width):
    """The rows the check cuts the bars after: five targets spread over ``rows``, the
    printed rows from the first the policy decides on to the one before the last, the
    k-th on instrument (k - 1) mod ``width``, each moved to the first of them at or
    after it where that instrument's decision changes to one that is not empty and
    differs from the one its cut before ended on, failing that to the first where it
    changes to one that is not empty, and failing that kept; each is cut with the
    printed row before it, where the decision it changes from held, each cut once and
    only where it holds two printed rows."""
    at     = int(np.searchsorted(printed, rows[0]))
    before = int(printed[at - 1]) if at else None

    changes, ended, cuts = {}, {}, set()
    for k in range(1, 6):
        j, target = (k - 1) % width, k * len(rows) // 6
        if j not in changes:
            changes[j] = whole.changes(rows, before, j)
        later = [change for change in changes[j] if change >= target]
        moved = next((change for change in later
                      if whole.on(rows[change], j) != ended.get(j)),
                     later[0] if later else target)
        ended[j] = whole.on(rows[moved], j)
        cuts.update(rows[max(moved - 1, 0):moved + 1])

    return sorted(cut for cut in cuts if np.searchsorted(printed, cut, side="right") >= 2)


def _near(a, b, scale):
    """Whether floats ``a`` and ``b`` are one within ``scale`` of the larger, and never
    a bound below 1e-12."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)

    return a == b or abs(a - b) <= max(scale * max(abs(a), abs(b)), 1e-12)


def _single(value):
    """Whether ``value`` is a float32's value exactly."""
    return (math.isfinite(value) and abs(value) <= FLOAT32
            and float(np.float32(value)) == value)


def _same(a, b):
    """Whether two arguments of a Strategy's calls are one: floats within the bound, a
    float32's where both are float32 values, and anything else equal."""
    if isinstance(a, float) and isinstance(b, float):
        return _near(a, b, SINGLE if _single(a) and _single(b) else WIDE)

    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)

    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(map(_same, a, b))

    return type(a) is type(b) and a == b


def _apart(whole, other, scale):
    """The first row and column at which arrays ``whole`` and ``other`` differ, the
    NaN of one meeting a value of the other or two values further apart than ``scale``
    of the largest magnitude ``whole``'s column holds, or None."""
    mine, theirs = np.isnan(whole), np.isnan(other)
    largest = np.abs(np.where(mine, 0.0, whole)).max(axis=0) if len(whole) else 0.0
    bound   = np.maximum(scale * largest, 1e-12)
    apart   = (mine != theirs) | (~mine & ~theirs & (np.abs(np.where(mine, 0.0, whole)
                                                            - np.where(theirs, 0.0, other))
                                                     > bound))

    hits = np.argwhere(apart)
    return None if not len(hits) else tuple(int(k) for k in hits[0])


def _difference(whole, other, rows, data):
    """The first difference between ``whole``'s decisions and ``other``'s on the first
    ``rows`` rows of ``data``, as what differs, the time of the row it differs on, None
    for a setting of every row, and the two values, or None."""
    names, time = data.names, data.time

    if isinstance(whole, _Calls) != isinstance(other, _Calls):
        kinds = ("a Strategy", "weights") if isinstance(whole, _Calls) else (
            "weights", "a Strategy")
        return "policy", None, *kinds

    if isinstance(whole, _Calls):
        for row in sorted(set(whole.calls) | set(other.calls)):
            mine, theirs = whole.on(row), other.on(row)
            # most rows match exactly, which needs no bound and compares in C
            if row < rows and mine != theirs and not _same(mine, theirs):
                return ("calls", _when(time[row]), _shown(mine, names),
                        _shown(theirs, names))
        return None

    hit = _apart(whole.rows[:rows], other.rows, whole.scale)
    if hit is not None:
        row, k = hit
        return (f"decision for {names[k]}", _when(time[row]), float(whole.rows[row, k]),
                float(other.rows[row, k]))

    for name, mine in whole.barriers.items():
        theirs = other.barriers[name]
        if isinstance(mine, np.ndarray) and isinstance(theirs, np.ndarray):
            hit = _apart(mine[:rows], theirs, whole.scales.get(name, WIDE))
            if hit is not None:
                row, k = hit
                return (f"{name} for {names[k]}", _when(time[row]), float(mine[row, k]),
                        float(theirs[row, k]))
        elif not (mine is theirs is None or isinstance(mine, float)
                  and isinstance(theirs, float) and _near(mine, theirs, WIDE)):
            return name, None, _plain(mine), _plain(theirs)

    if whole.sizing != other.sizing:
        return "sizing", None, whole.sizing, other.sizing

    if not _near(whole.tolerance, other.tolerance, WIDE):
        return "tolerance", None, whole.tolerance, other.tolerance

    return None


def _plain(barrier):
    """What a message shows of a barrier: its value, or that it is an array."""
    return f"an array of {barrier.shape}" if isinstance(barrier, np.ndarray) else barrier


def _said(differ, mine, theirs, fix):
    """The refusal of the difference ``differ``, the whole run's ``mine`` and the other
    run's ``theirs``."""
    what, when, a, b = differ
    if what == "policy":
        return f"policy: it returns {a} {mine} and {b} {theirs}; {fix}"

    if what == "calls":
        return f"policy: at {when} it calls {a} {mine} and {b} {theirs}; {fix}"

    at = "" if when is None else f" at {when}"
    return f"policy: its {what}{at} is {a!r} {mine} and {b!r} {theirs}; {fix}"


def _shown(calls, names):
    """A row's calls as the account's verbs that made them, or nothing."""
    if not calls:
        return "nothing"

    return " and ".join(_call(verb, args, names) for verb, args in calls)


def _call(verb, args, names):
    """One call of a Strategy on its account, as the verb and arguments it gave."""
    if verb == "order":
        side, size, kind, price, trigger, reduce_only, post_only, tif, k = args
        given = [repr(size)] + ([f"limit={price!r}"] if kind == "limit" else []) + (
            [f"stop={trigger!r}"] if kind == "stop" else []) + (
            [f"tif={tif!r}"] if tif != "gtc" else []) + (
            ["post_only=True"] if post_only else []) + (
            ["reduce_only=True"] if reduce_only else [])
        return f"{side}({', '.join(given)}) on {names[k]}"

    if verb == "order_value":
        side, amount, reduce_only, tif, k = args
        given = [repr(amount)] + ([f"tif={tif!r}"] if tif != "ioc" else []) + (
            ["reduce_only=True"] if reduce_only else [])
        return f"{side}_value({', '.join(given)}) on {names[k]}"

    if verb == "bracket":
        stop, take_profit, k = args
        return f"bracket(stop={stop!r}, take_profit={take_profit!r}) on {names[k]}"

    if verb in ("close", "cancel_all") and args[0] is not None:
        return f"{verb}() on {names[args[0]]}"

    if verb == "cancel_all":
        return "cancel_all()"

    if verb == "cancel":
        return f"cancel(order {args[0]})"

    if verb == "replace":
        order, size, price, trigger = args
        given = [f"order {order}"] + [f"{name}={value!r}" for name, value in (
            ("price", price), ("trigger", trigger), ("size", size)) if value is not None]
        return f"replace({', '.join(given)})"

    if verb == "target":
        weights, tolerance, transfer = args
        shown = (repr(weights) if not isinstance(weights, dict) else
                 "{" + ", ".join(f"{names[k]!r}: {weight!r}"
                                 for k, weight in weights.items()) + "}")
        given = [shown] + ([f"tolerance={tolerance!r}"] if tolerance else []) + (
            ["transfer=True"] if transfer else [])
        return f"target({', '.join(given)})"

    amount, to = args
    return f"transfer({amount!r}, to={to!r})"
