"""How ``search`` draws its trials' parameters: probatum's random and grid samplers on
named streams, and Optuna's TPE with every argument it keeps set (ADR 0032;
Specification, Samplers). Each asks for trial ``number`` in turn and is told its
outcome."""

import contextlib
import math

import numpy as np

from ._errors import MissingDependency, ProbatumValueError
from ._probatum import exp, ln
from ._space import Categorical, Float, Int
from ._streams import stream

# the most points numpy draws an index among
MOST = 2**63 - 1
# every argument Optuna 5.0 keeps; the eight it deprecates stay at the recorded
# version's defaults (ADR 0032)
TPE = dict(n_startup_trials=10, n_ei_candidates=24, multivariate=True, group=False,
           constant_liar=True)


def _drawn(given, rng):
    """A value of the range ``given`` drawn from ``rng`` as the random sampler draws
    one, a log scale through libm's logarithm and exponential."""
    if isinstance(given, Categorical):
        return given.choices[int(rng.integers(0, len(given.choices)))]

    if isinstance(given, Int) and given.log:
        low, high = ln(given.low - 0.5), ln(given.high + 0.5)
        value = round(exp(low + rng.random() * (high - low)))
        return min(max(value, given.low), given.high)

    size = given._size()
    if size is not None:
        return given._at(int(rng.integers(0, size)))

    if given.log:
        low, high = ln(given.low), ln(given.high)
        value = exp(low + rng.random() * (high - low))
    else:
        value = given.low + rng.random() * (given.high - given.low)

    return min(max(value, given.low), math.nextafter(given.high, -math.inf))


class _Random:
    """probatum's random sampler: each trial's parameters drawn in the space's order
    from the stream named search random."""

    name = "random"


    def __init__(self, space, trials, seed):
        for name, given in space.items():
            size = None if isinstance(given, Int) and given.log else given._size()
            if size is not None and size > MOST:
                raise ProbatumValueError(f"space: {name} holds {size:,} points, more "
                                         f"than sampler=\"random\" draws among, 2**63 - "
                                         f"1; give it a narrower range")

        self.space, self.count = space, trials
        self.rng = stream(seed, "search random")


    def ask(self, number):
        return {name: _drawn(given, self.rng) for name, given in self.space.items()}


    def tell(self, number, trial):
        pass


    def arguments(self):
        return {"stream": "search random"}


class _Grid:
    """probatum's grid sampler: the product of each parameter's values in the space's
    order, the last changing fastest, every point once where it holds no more than
    ``trials``, and otherwise ``trials`` of them drawn without replacement from the
    stream named search grid, run in that order."""

    name = "grid"


    def __init__(self, space, trials, seed):
        sizes = []
        for name, given in space.items():
            size = given._size()
            if size is None:
                raise ProbatumValueError(f"space: {name} is {given!r}, which has no "
                                         f"step to list its values; give it one, or "
                                         f"search it with sampler=\"random\"")
            sizes.append(size)

        size = math.prod(sizes)
        if size <= trials:
            points = range(size)
        elif size > MOST:
            raise ProbatumValueError(f"space: a grid of {size:,} points, more than "
                                     f"sampler=\"grid\" draws among, 2**63 - 1; give it "
                                     f"fewer, or search it with sampler=\"random\"")
        else:
            points = np.sort(stream(seed, "search grid").choice(size, trials,
                                                                replace=False))

        self.space, self.sizes, self.points = space, sizes, points
        self.count = len(points)


    def ask(self, number):
        index, digits = int(self.points[number]), []
        for size in reversed(self.sizes):
            index, digit = divmod(index, size)
            digits.append(digit)

        return {name: given._at(k)
                for (name, given), k in zip(self.space.items(), reversed(digits))}


    def tell(self, number, trial):
        pass


    def arguments(self):
        return {"stream": "search grid", "points": math.prod(self.sizes)}


def _optuna():
    """Optuna, 5.0 or later, which TPE needs."""
    try:
        import optuna
    except ImportError:
        raise MissingDependency("sampler: \"tpe\" needs optuna, which is not "
                                "installed; pip install \"optuna>=5.0\"") from None

    if int(optuna.__version__.split(".")[0]) < 5:
        raise MissingDependency(f"sampler: \"tpe\" needs optuna 5.0 or later, and "
                                f"{optuna.__version__} is installed; pip install "
                                f"\"optuna>=5.0\"")

    return optuna


@contextlib.contextmanager
def _quiet(optuna):
    """Optuna's logger at warnings, put back as it was after."""
    level = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        yield
    finally:
        optuna.logging.set_verbosity(level)


def _distribution(optuna, given):
    """The range ``given`` as Optuna's distribution of it."""
    kinds = optuna.distributions
    if isinstance(given, Int):
        return kinds.IntDistribution(given.low, given.high, log=given.log,
                                     step=given.step)

    if isinstance(given, Float):
        return kinds.FloatDistribution(given.low, given.high, log=given.log,
                                       step=given.step)

    return kinds.CategoricalDistribution(given.choices)


def _onto(given, value):
    """``value``, which TPE proposed for ``given``, on probatum's lattice: a stepped
    Float's point at the nearest index, so TPE and the random sampler search one
    lattice, and a Float with no step held below its high."""
    if isinstance(given, Int):
        return int(value)

    if isinstance(given, Float):
        size = given._size()
        if size is not None:
            k = round((float(value) - given.low) / given.step)
            return given._at(min(max(k, 0), size - 1))
        return min(max(float(value), given.low), math.nextafter(given.high, -math.inf))

    return value


class _Tpe:
    """Optuna's TPE with every argument Optuna 5.0 keeps set, seeded with the first
    draw below 2**32 of the stream named search tpe, and told a floored or failed
    trial as failed."""

    name = "tpe"


    def __init__(self, space, trials, seed, direction):
        optuna = self.optuna = _optuna()
        self.seed = int(stream(seed, "search tpe").integers(0, 2**32))
        self.space, self.count = space, trials
        self.distributions = {name: _distribution(optuna, given)
                              for name, given in space.items()}

        with _quiet(optuna):
            sampler    = optuna.samplers.TPESampler(**TPE, seed=self.seed)
            self.study = optuna.create_study(sampler=sampler, direction=direction)

        self.asked = {}


    def ask(self, number):
        with _quiet(self.optuna):
            asked = self.study.ask(self.distributions)

        self.asked[number] = asked
        return {name: _onto(given, asked.params[name])
                for name, given in self.space.items()}


    def tell(self, number, trial):
        asked = self.asked.pop(number)

        with _quiet(self.optuna):
            if trial.state == "complete":
                self.study.tell(asked, trial.score)
            else:
                self.study.tell(asked, state=self.optuna.trial.TrialState.FAIL)


    def arguments(self):
        return dict(TPE, seed=self.seed, optuna=self.optuna.__version__)
