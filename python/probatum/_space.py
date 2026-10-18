"""What ``search`` searches: each parameter's range, an `Int`, a `Float` or a
`Categorical`, checked as it is built and never changed after, with the lattice of
values it holds (Specification, Samplers)."""

import collections.abc
import decimal
import math

import numpy as np

from ._errors import ProbatumAttributeError, ProbatumTypeError, ProbatumValueError

# the whole numbers a trials table's int64 column holds
WHOLE = range(-2**63, 2**63)


def _whole(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _number(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


def _flag(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise ProbatumTypeError(f"{name}: {value!r}; give True or False")
    return bool(value)


class _Range:
    """What the three ranges share: built once, compared and hashed by what they hold,
    and pickled as the call that builds them."""

    __slots__ = ()


    def __setattr__(self, name, value):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a range does not "
                                     f"change; build another")


    def __delattr__(self, name):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a range does not "
                                     f"change; build another")


    def __eq__(self, other):
        return type(other) is type(self) and self._args() == other._args()


    def __hash__(self):
        return hash((type(self).__name__, self._args()))


    def __reduce__(self):
        return type(self), self._args()


    def _set(self, **given):
        for name, value in given.items():
            object.__setattr__(self, name, value)


class Int(_Range):
    """A whole number from ``low`` to ``high``, both included, on the lattice
    ``low + k * step``, the ``step`` a whole number from 1 that divides the span; with
    ``log`` it is drawn evenly in its logarithm, which needs a ``low`` from 1 and a step
    of 1."""

    __slots__ = ("low", "high", "step", "log")


    def __init__(self, low, high, step=1, log=False):
        for name, value in (("low", low), ("high", high), ("step", step)):
            if not _whole(value):
                raise ProbatumTypeError(f"{name}: {value!r}; give a whole number, as an "
                                        f"int")

        log = _flag("log", log)
        low, high, step = int(low), int(high), int(step)

        for name, value in (("low", low), ("high", high)):
            if value not in WHOLE:
                raise ProbatumValueError(_beyond(name, value))

        if high < low:
            raise ProbatumValueError(f"high: {high}, below low, {low}; give a high from "
                                     f"{low}")

        if step < 1:
            raise ProbatumValueError(f"step: {step}; give a whole number from 1")

        if (high - low) % step:
            raise ProbatumValueError(f"step: {step}, which does not divide the span from "
                                     f"{low} to {high}, {high - low:,}; give a step that "
                                     f"does")

        if log and low < 1:
            raise ProbatumValueError(f"low: {low} on a log scale; give a low from 1")
        if log and step != 1:
            raise ProbatumValueError(f"step: {step} on a log scale; leave it at 1")

        self._set(low=low, high=high, step=step, log=log)


    def _args(self):
        return self.low, self.high, self.step, self.log


    def _size(self):
        """The lattice's points."""
        return (self.high - self.low) // self.step + 1


    def _at(self, k):
        """The lattice's point ``k``."""
        return self.low + k * self.step


    def __repr__(self):
        given = ([] if self.step == 1 else [f"step={self.step}"]) + (
            ["log=True"] if self.log else [])
        return f"Int({', '.join([str(self.low), str(self.high), *given])})"


class Float(_Range):
    """A number from ``low`` to ``high``: with no ``step`` the interval ``[low, high)``,
    and with one the lattice ``low + k * step``, its last point ``high`` exactly, the
    step above 0 dividing the span as their decimals do; with ``log`` it is drawn evenly
    in its logarithm, which needs a ``low`` above 0 and no step."""

    __slots__ = ("low", "high", "step", "log", "_n")


    def __init__(self, low, high, step=None, log=False):
        given = [("low", low), ("high", high)] + ([] if step is None else [("step", step)])
        for name, value in given:
            if not _number(value):
                raise ProbatumTypeError(f"{name}: {value!r}; give a number, such as 0.5")
            if not math.isfinite(value):
                raise ProbatumValueError(f"{name}: {value!r}; give a finite number")

        log = _flag("log", log)
        low, high = float(low), float(high)

        if not low < high:
            raise ProbatumValueError(f"high: {high!r}, not above low, {low!r}; give a "
                                     f"high above low")

        n = None
        if step is not None:
            step = float(step)
            if step <= 0.0:
                raise ProbatumValueError(f"step: {step!r}; give a step above 0")

            # the span tested on the decimals the numbers print as, as Optuna tests it,
            # so it never moves high
            span = decimal.Decimal(str(high)) - decimal.Decimal(str(low))
            try:
                left = span % decimal.Decimal(str(step))
            except decimal.InvalidOperation:
                raise ProbatumValueError(f"step: {step!r}, which divides the span from "
                                         f"{low!r} to {high!r} into more points than can "
                                         f"be counted; give a coarser step") from None
            if left:
                raise ProbatumValueError(f"step: {step!r}, which does not divide the span "
                                         f"from {low!r} to {high!r}; give a step that "
                                         f"does")

            n = int(span / decimal.Decimal(str(step)))

        if log and low <= 0.0:
            raise ProbatumValueError(f"low: {low!r} on a log scale; give a low above 0")
        if log and step is not None:
            raise ProbatumValueError(f"step: {step!r} on a log scale; leave it out")

        self._set(low=low, high=high, step=step, log=log, _n=n)


    def _args(self):
        return self.low, self.high, self.step, self.log


    def _size(self):
        """The lattice's points, None with no step."""
        return None if self._n is None else self._n + 1


    def _at(self, k):
        """The lattice's point ``k``, its last ``high`` exactly."""
        return self.high if k == self._n else self.low + k * self.step


    def __repr__(self):
        given = ([] if self.step is None else [f"step={self.step!r}"]) + (
            ["log=True"] if self.log else [])
        return f"Float({', '.join([repr(self.low), repr(self.high), *given])})"


def _plain(value):
    """``value``, a choice, as the plain Python value a trial receives."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if _whole(value):
        return int(value)
    if _number(value):
        return float(value)
    raise ProbatumTypeError(f"choices: {value!r}, a {type(value).__name__}; give None, "
                            f"bools, numbers or strings")


def _beyond(name, value):
    """The refusal of ``value``, a whole number given as ``name`` that a trials table
    cannot hold."""
    return (f"{name}: {value}, beyond the 64-bit integers a trials table holds; give "
            f"one from {WHOLE.start:,} to {WHOLE.stop - 1:,}")


def _one(a, b):
    """Whether choices ``a`` and ``b`` are one: equal, or both NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


class Categorical(_Range):
    """One of ``choices``, a list or a tuple of None, bools, numbers and strings, no two
    of them equal, so 1, 1.0 and True are one choice, each a plain Python value."""

    __slots__ = ("choices",)


    def __init__(self, choices):
        if (isinstance(choices, (str, bytes))
                or not isinstance(choices, collections.abc.Sequence)):
            raise ProbatumTypeError(f"choices: {choices!r}; give a list of them, such as "
                                    f"[\"ema\", \"sma\"]")

        kept = tuple(_plain(choice) for choice in choices)
        if not kept:
            raise ProbatumValueError("choices: none; give at least one")

        for choice in kept:
            if type(choice) is int and choice not in WHOLE:
                raise ProbatumValueError(_beyond("choices", choice))

        for k, choice in enumerate(kept):
            for other in kept[:k]:
                if _one(choice, other):
                    raise ProbatumValueError(f"choices: {other!r} and {choice!r}, which "
                                             f"are one choice; keep one")

        self._set(choices=kept)


    def _args(self):
        return (list(self.choices),)


    def __eq__(self, other):
        # 1, 1.0 and True hand a trial three different values, and NaN is one choice
        if type(other) is not type(self) or len(other.choices) != len(self.choices):
            return False
        return all(type(a) is type(b) and _one(a, b)
                   for a, b in zip(self.choices, other.choices))


    def __hash__(self):
        return hash((type(self).__name__, len(self.choices)))


    def _size(self):
        return len(self.choices)


    def _at(self, k):
        return self.choices[k]


    def __repr__(self):
        return f"Categorical({list(self.choices)!r})"
