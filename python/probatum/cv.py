"""``probatum.cv``: the ways of dividing time, each a scikit-learn compatible splitter
built without importing scikit-learn, which leaves out of a split's training rows every
one whose label reaches a test span, and the rows the embargo holds after it (ADR 0035;
Specification, The splitters; The purge and the embargo)."""

import itertools
import math
import os
import warnings

import numpy as np

from ._errors import (ProbatumAttributeError, ProbatumTypeError, ProbatumValueError,
                      ProbatumWarning)
from ._space import _flag, _whole

__all__ = ["CombinatorialPurged", "Holdout", "PurgedKFold", "WalkForward"]

# the mark of a row with no label, as a table marks a missing integer
NONE = np.iinfo(np.int64).min
# the package's files, which a warning skips to name the caller's line
PACKAGE = (os.path.dirname(__file__),)


class _Splitter:
    """What the four splitters share: built once and never changed, each row's label
    end where one was given, the splits laid out on the bars they divide, and the rows
    each split trains and tests on, as scikit-learn's tools read them."""

    __slots__ = ("end",)


    def __setattr__(self, name, value):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a splitter does "
                                     f"not change; build another")


    def __delattr__(self, name):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a splitter does "
                                     f"not change; build another")


    def _set(self, **given):
        for name, value in given.items():
            object.__setattr__(self, name, value)


    def __reduce__(self):
        args = self._args()
        return _built, (type(self), args, self._keywords(), self.end)


    def __copy__(self):
        return self


    def __deepcopy__(self, memo):
        return self


    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        if (self._args(), self._keywords()) != (other._args(), other._keywords()):
            return False

        if self.end is None or other.end is None:
            return self.end is other.end
        return self.end.shape == other.end.shape and np.array_equal(self.end, other.end)


    def __hash__(self):
        end = self.end
        return hash((type(self), self._args(), tuple(self._keywords().items()),
                     None if end is None else (end.shape, end.tobytes())))


    def __repr__(self):
        given = [repr(value) for value in self._args()]
        given += [f"{name}={value!r}" for name, value in self._keywords().items()
                  if value != self._defaults()[name]]
        if self.end is not None:
            given.append(f"end of shape {self.end.shape}")

        return f"{type(self).__name__}({', '.join(given)})"


    def split(self, X, y=None, groups=None):
        """Each split as ``(train, test)``, the ascending int64 positions of the rows of
        ``X`` it trains and tests on, every split checked before the first is given.
        ``X`` holds the rows in the long, time-major form an estimator reads, row ``r``
        bar ``r // N`` of instrument ``r % N``, as many as ``end`` has, and without
        ``end`` each of its rows is a bar. ``y`` is not read, and ``groups``, which
        scikit-learn passes on, is ignored with a warning where it is given."""
        bars, width = self._shape(X)

        if groups is not None:
            warnings.warn(f"groups passed to split: ignored by {type(self).__name__}, "
                          f"which divides by time", ProbatumWarning,
                          skip_file_prefixes=PACKAGE)

        ends, labelled = self._ends(bars, width)
        layout  = self._layout(bars)
        embargo = self._embargo(ends, labelled, bars)

        for number, (tests, trains) in enumerate(layout):
            train, test = _rows(tests, trains, ends, labelled, embargo)
            named = self._named(number, tests, bars)

            if not test.any():
                raise ProbatumValueError(f"{named}: no row to test on, as none of "
                                         f"its rows holds a label that ends by the "
                                         f"last bar; give them labels that do")
            if not train.any():
                raise ProbatumValueError(f"{named}: no row left to train on once the "
                                         f"purge{self._embargoed(embargo)} took the "
                                         f"rest; give {self._fix()}")
        return _given(layout, ends, labelled, embargo)


    def get_n_splits(self, X=None, y=None, groups=None):
        """How many splits ``split`` gives, for the rows of ``X`` where it is given."""
        if X is not None:
            bars, _ = self._shape(X)
        else:
            bars = None if self.end is None else self.end.shape[0]

        return self._count(bars)


    def _shape(self, X):
        """The bars and the instruments ``X``'s rows hold, refusing a count of rows
        other than ``end``'s."""
        shape = getattr(X, "shape", None)
        if shape is not None and len(shape) and isinstance(shape[0], int):
            rows = shape[0]
        else:
            try:
                rows = len(X)
            except TypeError:
                raise ProbatumTypeError(f"X: {type(X).__name__}, which has no rows; "
                                        f"give the rows an estimator reads") from None

        if self.end is None:
            return rows, 1

        bars, width = self.end.shape[0], _width(self.end)
        if rows != bars * width:
            raise ProbatumValueError(f"X: {rows:,} rows where this splitter was built "
                                     f"for {bars * width:,}; build it for the rows it "
                                     f"splits")

        return bars, width


    def _ends(self, bars, width):
        """Each row's label end as ``(bars, width)`` and which rows have a label that
        ends by the last bar; without ``end``, each row a label of no length."""
        if self.end is None:
            ends = np.broadcast_to(np.arange(bars, dtype=np.int64)[:, None], (bars, 1))
            return ends, np.ones((bars, 1), dtype=bool)

        ends = self.end.reshape(bars, width)
        return ends, (ends != NONE) & (ends < bars)


    def _embargo(self, ends, labelled, bars):
        """The bars embargoed after each span's reach: ``embargo`` where it was given,
        and otherwise the most bars any label held, ``end - t - 1``, 0 with none."""
        keywords = self._keywords()
        if "embargo" not in keywords:
            return 0
        if keywords["embargo"] is not None:
            return keywords["embargo"]

        if self.end is None or not labelled.any():
            return 0

        held = ends - np.arange(bars, dtype=np.int64)[:, None] - 1
        return int(held[labelled].max())


    def _named(self, number, tests, bars):
        return f"split {number}"


    def _embargoed(self, embargo):
        return f" and an embargo of {embargo:,} bars" if embargo else ""


    def _fix(self):
        return "more bars, or labels that end sooner"


    def _keywords(self):
        return {}


    def _defaults(self):
        return {}


def _built(kind, args, keywords, end):
    """A splitter again from what it pickled as."""
    return kind(*args, **keywords, end=end)


def _width(end):
    return 1 if end.ndim == 1 else end.shape[1]


def _given(layout, ends, labelled, embargo):
    for tests, trains in layout:
        train, test = _rows(tests, trains, ends, labelled, embargo)
        yield (np.flatnonzero(train.reshape(-1)).astype(np.int64),
               np.flatnonzero(test.reshape(-1)).astype(np.int64))


def _rows(tests, trains, ends, labelled, embargo):
    """The rows a split trains and tests on, as ``(bars, width)`` masks: a test span's
    rows with a label, and its training ranges' rows with one, but those whose label
    reaches a span from before it and those from the bar after a span to its reach, the
    latest end of a label that tests in it, and the ``embargo`` after it
    (Specification, The purge and the embargo)."""
    t     = np.arange(ends.shape[0], dtype=np.int64)[:, None]
    train = np.zeros(ends.shape, dtype=bool)
    test  = np.zeros(ends.shape, dtype=bool)
    gone  = np.zeros(ends.shape, dtype=bool)

    for lo, hi in trains:
        train[lo:hi] = True

    for a, b in tests:
        test[a:b + 1] = True

        # a label censored or running past the bars tests nothing, so reaches nothing
        marked = ends[a:b + 1][labelled[a:b + 1]]
        reach  = max(b, int(marked.max())) if marked.size else b
        gone |= (t < a) & (ends >= a)
        gone[b + 1:reach + embargo + 1] = True

    return train & labelled & ~gone, test & labelled


def _spans(groups, edges):
    """The bar spans of adjacent ``groups``, each from its first group's first bar to
    its last group's last."""
    out = []
    for g in groups:
        if out and out[-1][2] == g - 1:
            out[-1] = (out[-1][0], edges[g + 1] - 1, g)
        else:
            out.append((edges[g], edges[g + 1] - 1, g))

    return [(a, b) for a, b, _ in out]


def _others(tests, bars):
    """The ranges of bars outside ``tests``, as half-open ``(lo, hi)``."""
    out, at = [], 0
    for a, b in tests:
        if a > at:
            out.append((at, a))
        at = b + 1

    if at < bars:
        out.append((at, bars))

    return out


def _edges(bars, n):
    """The edges of ``n`` groups of ``bars`` bars in order, as scikit-learn's KFold
    sizes folds: the first ``bars % n`` hold a bar more than the rest."""
    q, m = divmod(bars, n)
    sizes = [q + 1] * m + [q] * (n - m)
    return [0] + list(itertools.accumulate(sizes))


def _ends_of(end):
    """``end`` as the int64 array a splitter keeps, a copy, refusing what holds no bar
    positions or a row whose label ends at or before its own bar."""
    if end is None:
        return None

    values = np.asarray(end)

    if values.dtype.kind not in "iu":
        raise ProbatumTypeError(f"end: an array of {values.dtype}; give each row's "
                                f"label end as an int64 bar, the least int64 for a "
                                f"row with no label")

    if values.ndim not in (1, 2):
        raise ProbatumValueError(f"end: an array of shape {values.shape}; give (T,) "
                                 f"for one instrument or (T, N)")

    if not values.size:
        raise ProbatumValueError("end: no bars; give each bar's label end")

    if values.dtype.kind == "u" and values.max() > np.iinfo(np.int64).max:
        raise ProbatumValueError(f"end: {int(values.max())}, past the largest int64; "
                                 f"give each row's label end as an int64 bar")

    out   = np.array(values, dtype=np.int64)
    t     = np.arange(out.shape[0], dtype=np.int64)
    early = (out <= (t if out.ndim == 1 else t[:, None])) & (out != NONE)
    if early.any():
        at    = np.argwhere(early)[0]
        row   = int(at[0])
        place = (f"row {row}" if out.ndim == 1
                 else f"row {row} of instrument {int(at[1])}")
        hint = (", and a (T, N) grid flattened reads its rows as bars, so pass it as "
                "(T, N)" if out.ndim == 1 else "")
        raise ProbatumValueError(f"end: {place} ends at {int(out[tuple(at)])}, at or "
                                 f"before its own bar; give the bar its trade closed "
                                 f"on, from the next on, or the least int64 for a row "
                                 f"with no label{hint}")

    out.flags.writeable = False
    return out


def _counted(name, value, least, unit="", most=None):
    """Refuse ``value``, given as ``name``, where it is no whole number from ``least``,
    or past ``most``."""
    fix = f"give a whole number{unit} from {least}" + (
        "" if most is None else f" to {most}")
    if not _whole(value):
        raise ProbatumTypeError(f"{name}: {value!r}; {fix}")
    if value < least or most is not None and value > most:
        raise ProbatumValueError(f"{name}: {value}; {fix}")

    return int(value)


def _embargo_of(embargo):
    if embargo is None:
        return None
    return _counted("embargo", embargo, 0, " of bars")


def _bars(name, value):
    """Refuse ``value``, given as ``name``, where it is neither a share in (0, 1) of the
    bars nor a whole number of them from 1."""
    if isinstance(value, (bool, np.bool_)):
        raise ProbatumTypeError(f"{name}: {value!r}; give a share of the bars above 0 "
                                f"and below 1, or a number of them from 1")

    if _whole(value):
        return _counted(name, value, 1, " of bars")

    if isinstance(value, (float, np.floating)):
        if not 0.0 < value < 1.0:
            raise ProbatumValueError(f"{name}: {value!r}; give a share of the bars "
                                     f"above 0 and below 1, or a number of them from 1")
        return float(value)

    raise ProbatumTypeError(f"{name}: {value!r}; give a share of the bars above 0 and "
                            f"below 1, or a number of them from 1")


class Holdout(_Splitter):
    """One split: the last ``tail`` bars tested, a share in (0, 1) of them rounded half
    to even or a whole number of them, and the bars before trained on, every training
    row whose label reaches the tail purged."""

    __slots__ = ("tail",)


    def __init__(self, tail, end=None):
        self._set(tail=_bars("tail", tail), end=_ends_of(end))
        if self.end is not None:
            self._layout(self.end.shape[0])


    def _args(self):
        return (self.tail,)


    def _tail(self, bars):
        """The bars ``tail`` holds out of ``bars``, refusing a tail of none or of
        all."""
        held = self.tail if isinstance(self.tail, int) else round(bars * self.tail)

        if held < 1:
            raise ProbatumValueError(f"tail: {self.tail!r} of {bars:,} bars rounds to "
                                     f"0; give a larger share or a number of bars "
                                     f"from 1")

        if held >= bars:
            raise ProbatumValueError(f"tail: {self.tail!r} of {bars:,} bars, which "
                                     f"leaves none to train on; give fewer")

        return held


    def _layout(self, bars):
        held = self._tail(bars)
        return [([(bars - held, bars - 1)], [(0, bars - held)])]


    def _paths(self, bars):
        held = self._tail(bars)
        return [[(bars - held, bars - 1, 0)]]


    def _count(self, bars):
        return 1


class WalkForward(_Splitter):
    """Windows from the first bar, ``floor((T - train) / test)`` of them, each testing
    the next ``test`` bars and the last running to the end, each trained on the
    ``train`` bars before it, or with ``anchored`` on every bar before it, both counted
    in bars; bars appended move only the last window."""

    __slots__ = ("train", "test", "anchored")


    def __init__(self, train, test, anchored=False, end=None):
        self._set(
            train    = _counted("train", train, 1, " of bars"),
            test     = _counted("test", test, 1, " of bars"),
            anchored = _flag("anchored", anchored),
            end      = _ends_of(end),
        )
        if self.end is not None:
            self._layout(self.end.shape[0])


    def _args(self):
        return self.train, self.test


    def _keywords(self):
        return {"anchored": self.anchored}


    def _defaults(self):
        return {"anchored": False}


    def _windows(self, bars):
        """Each window's first and last test bar, refusing a layout with none."""
        count = (bars - self.train) // self.test

        if count < 1:
            raise ProbatumValueError(f"train: {self.train:,} and test: {self.test:,} "
                                     f"for {bars:,} bars, which leave no window; give "
                                     f"fewer bars to train on or to test")

        firsts = [self.train + j * self.test for j in range(count)]
        lasts  = [first + self.test - 1 for first in firsts[:-1]] + [bars - 1]
        return list(zip(firsts, lasts))


    def _layout(self, bars):
        return [([(first, last)], [(0 if self.anchored else first - self.train, first)])
                for first, last in self._windows(bars)]


    def _paths(self, bars):
        windows = self._windows(bars)
        return [[(first, last, j) for j, (first, last) in enumerate(windows)]]


    def _count(self, bars):
        if bars is None:
            raise ProbatumValueError("X: none, and a WalkForward built without end "
                                     "counts its windows on the rows it splits; pass "
                                     "X, or build it with end=")

        return len(self._windows(bars))


    def _named(self, number, tests, bars):
        first, last = tests[0]
        return f"window {number}, testing bars {first:,} to {last:,}"


    def _fix(self):
        return "a longer train, or labels that end sooner"


class PurgedKFold(_Splitter):
    """``k`` groups of bars in order, as scikit-learn's KFold sizes folds, each split
    testing one and training on the others, every training row whose label reaches the
    group purged and the ``embargo`` bars after its labels' reach removed, an embargo of
    None the longest label held."""

    __slots__ = ("k", "embargo")


    def __init__(self, k, embargo=None, end=None):
        self._set(
            k       = _counted("k", k, 2),
            embargo = _embargo_of(embargo),
            end     = _ends_of(end),
        )
        if self.end is not None:
            self._layout(self.end.shape[0])


    def _args(self):
        return (self.k,)


    def _keywords(self):
        return {"embargo": self.embargo}


    def _defaults(self):
        return {"embargo": None}


    def _groups(self, bars):
        if self.k > bars:
            raise ProbatumValueError(f"k: {self.k} for {bars:,} bars, which leaves a "
                                     f"group with none; give k up to {bars:,}")

        return _edges(bars, self.k)


    def _layout(self, bars):
        edges = self._groups(bars)
        out   = []
        for i in range(self.k):
            tests = [(edges[i], edges[i + 1] - 1)]
            out.append((tests, _others(tests, bars)))

        return out


    def _paths(self, bars):
        edges = self._groups(bars)
        return [[(edges[i], edges[i + 1] - 1, i) for i in range(self.k)]]


    def _count(self, bars):
        return self.k


    def _named(self, number, tests, bars):
        return f"split {number}, testing group {number}"


    def _fix(self):
        return "more bars, a smaller k or a shorter embargo"


class CombinatorialPurged(_Splitter):
    """``groups`` groups of bars in order, as PurgedKFold makes them, every combination
    of ``test_groups`` of them tested in turn, in the order itertools.combinations
    lists them, each split training on the other groups with the purge and the
    embargo of PurgedKFold at each run of adjacent test groups; its ``C(groups - 1,
    test_groups - 1)`` paths take each group from the p-th split that tests it."""

    __slots__ = ("groups", "test_groups", "embargo")


    def __init__(self, groups, test_groups, embargo=None, end=None):
        groups = _counted("groups", groups, 2)
        self._set(
            groups      = groups,
            test_groups = _counted("test_groups", test_groups, 1, most=groups - 1),
            embargo     = _embargo_of(embargo),
            end         = _ends_of(end),
        )
        if self.end is not None:
            self._layout(self.end.shape[0])


    def _args(self):
        return self.groups, self.test_groups


    def _keywords(self):
        return {"embargo": self.embargo}


    def _defaults(self):
        return {"embargo": None}


    def _edges(self, bars):
        if self.groups > bars:
            raise ProbatumValueError(f"groups: {self.groups} for {bars:,} bars, which "
                                     f"leaves a group with none; give groups up to "
                                     f"{bars:,}")

        return _edges(bars, self.groups)


    def _tested(self):
        return list(itertools.combinations(range(self.groups), self.test_groups))


    def _layout(self, bars):
        edges = self._edges(bars)
        out   = []
        for chosen in self._tested():
            tests = _spans(chosen, edges)
            out.append((tests, _others(tests, bars)))

        return out


    def _paths(self, bars):
        edges, tested = self._edges(bars), self._tested()
        count = math.comb(self.groups - 1, self.test_groups - 1)
        paths = [[None] * self.groups for _ in range(count)]

        for g in range(self.groups):
            testing = [number for number, chosen in enumerate(tested) if g in chosen]
            for p, number in enumerate(testing):
                paths[p][g] = (edges[g], edges[g + 1] - 1, number)

        return paths


    def _count(self, bars):
        return math.comb(self.groups, self.test_groups)


    def _named(self, number, tests, bars):
        chosen = self._tested()[number]
        named  = (f"group {chosen[0]}" if len(chosen) == 1 else "groups " + ", ".join(
            str(g) for g in chosen[:-1]) + f" and {chosen[-1]}")
        return f"split {number}, testing {named}"


    def _fix(self):
        return "more bars, fewer test groups or a shorter embargo"
