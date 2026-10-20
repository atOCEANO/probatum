"""``probatum.stats``: how far a run's statistics can be trusted, read from a Result or
from any series of returns: the Sharpe's interval by the delta method over a long-run
variance with Newey and West's weights, its probability of beating a benchmark and the
track record that would show it, and the long-run Sharpe (ADR 0031; Specification,
Inference)."""

import math
import statistics

import numpy as np

from ._errors import ProbatumAttributeError, ProbatumTypeError, ProbatumValueError
from ._space import _flag, _number, _whole

__all__ = ["Interval", "Probability", "TrackRecord", "autocorrelation", "effective_sample",
           "min_track_record_length", "probabilistic_sharpe", "sharpe", "sharpe_interval"]

NORMAL   = statistics.NormalDist()
ROUNDING = 1e-20     # a standardised influence's variance below this is rounding


class _Frozen:
    """What the module's results share: set once, compared and hashed by what they
    hold, and pickled as the values they were made from."""

    __slots__ = ()


    def __setattr__(self, name, value):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a result does not "
                                     f"change; compute another")


    def __delattr__(self, name):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a result does not "
                                     f"change; compute another")


    def __eq__(self, other):
        return type(other) is type(self) and self._args() == other._args()


    def __hash__(self):
        return hash((type(self).__name__, self._args()))


    def __reduce__(self):
        return type(self), self._args()


    def __repr__(self):
        return f"{type(self).__name__}({self})"


    def _args(self):
        return tuple(getattr(self, name) for name in self.__slots__)


    def _set(self, **given):
        for name, value in given.items():
            object.__setattr__(self, name, value)


class Interval(_Frozen):
    """A statistic's interval: ``statistic``, its name; ``estimate``, ``low`` and
    ``high``, in its units; ``level``; ``method``, how it was found; ``count``, the
    returns it read; and ``lag`` and ``floored``, the long-run variance's lag and
    whether its floor at lag zero bound. Its text says each."""

    __slots__ = ("statistic", "estimate", "low", "high", "level", "method", "count", "lag",
                 "floored")


    def __init__(self, statistic, estimate, low, high, level, method, count, lag,
                 floored):
        self._set(
            statistic = statistic,
            estimate  = float(estimate),
            low       = float(low),
            high      = float(high),
            level     = float(level),
            method    = method,
            count     = int(count),
            lag       = int(lag),
            floored   = bool(floored),
        )


    def __str__(self):
        floor = " floored at lag zero" if self.floored else ""
        return (f"{self.statistic} {self.estimate:.2f}, two-sided {_percent(self.level)} "
                f"interval {self.low:.2f} to {self.high:.2f}, {self.method}, Newey-West "
                f"lag {self.lag}{floor}, {self.count:,} returns")


class Probability(_Frozen):
    """The probabilistic Sharpe: ``value``, the one-sided probability that the Sharpe is
    above ``benchmark``, an annual Sharpe; ``lag`` and ``floored``, its variance's; and
    ``count``, the returns it read."""

    __slots__ = ("value", "benchmark", "lag", "floored", "count")


    def __init__(self, value, benchmark, lag, floored, count):
        self._set(
            value     = float(value),
            benchmark = float(benchmark),
            lag       = int(lag),
            floored   = bool(floored),
            count     = int(count),
        )


    def __str__(self):
        return (f"probabilistic sharpe {self.value:.3f} against "
                f"{_against(self.benchmark)}, one-sided")


class TrackRecord(_Frozen):
    """The minimum track record: ``observations``, the returns a Sharpe of ``sharpe``
    needs for a one-sided ``confidence`` that it is above ``benchmark``, an annual
    Sharpe, and ``years``, those over the annualisation, both None where the Sharpe is
    not above it; ``series_years``, the years the series holds; and ``lag``,
    ``floored`` and ``count`` as a Probability's."""

    __slots__ = ("observations", "years", "series_years", "sharpe", "confidence",
                 "benchmark", "lag", "floored", "count")


    def __init__(self, observations, years, series_years, sharpe, confidence, benchmark,
                 lag, floored, count):
        self._set(
            observations = None if observations is None else float(observations),
            years        = None if years is None else float(years),
            series_years = float(series_years),
            sharpe       = float(sharpe),
            confidence   = float(confidence),
            benchmark    = float(benchmark),
            lag          = int(lag),
            floored      = bool(floored),
            count        = int(count),
        )


    def __str__(self):
        if self.years is None:
            return (f"minimum track record none: the sharpe {self.sharpe:.2f} is not "
                    f"above {_against(self.benchmark)}")

        return (f"minimum track record {_years(self.years)} years for "
                f"{_percent(self.confidence)} one-sided confidence against "
                f"{_against(self.benchmark)}; this series has {_years(self.series_years)}")


def _percent(level):
    return f"{level * 100.0:.10g}%"


def _against(benchmark):
    return "zero" if benchmark == 0.0 else f"{benchmark:.2f}"


def _years(years):
    return f"{years:.1f}" if years >= 1.0 else f"{years:.2f}"


def sharpe(returns, periods_per_year=None, risk_free=None, long_run=False):
    """The Sharpe of ``returns``, a Result's, which is its record's, or a plain series'
    at ``periods_per_year`` against an annual ``risk_free``: the excess returns' mean
    over their deviation, annualised by the square root of the bars a year. With
    ``long_run`` the deviation is the long-run one, the Bartlett sum of the returns'
    autocovariances at Newey and West's 1994 lag, not floored, which tells what the
    returns' autocorrelation does to a Sharpe the square root of time scales
    (Specification, What a statistic reads; The long-run Sharpe)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    long_run = _flag("long_run", long_run)

    if x.size < 2:
        return 0.0

    if not long_run:
        return _ranked(x.mean(), x.std(ddof=1), math.sqrt(rate))

    centred = x - x.mean()
    m       = _lag(centred)
    spread  = _bartlett(_autocovariances(centred, m), m)
    return _ranked(x.mean(), math.sqrt(max(spread, 0.0)), math.sqrt(rate))


def sharpe_interval(returns, periods_per_year=None, risk_free=None, level=0.95):
    """The Sharpe's two-sided interval at ``level``, as an Interval, read as
    ``sharpe`` reads ``returns``: its standard error is the long-run variance of the
    Sharpe's influence series at Newey and West's 1994 lag, floored at its value at lag
    zero, over the returns less one, so autocorrelation widens it and never narrows it.
    Fewer than 4 returns, returns that never vary and returns whose Sharpe's influence
    is zero on every bar have none (Specification, The Sharpe's variance; Samples with
    no interval)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    level = _fraction("level", level)

    variance, lag, floored = _variance(x)
    error = math.sqrt(variance / (x.size - 1))
    z     = NORMAL.inv_cdf((1.0 + level) / 2.0)
    estimate, root = _per_bar(x), math.sqrt(rate)

    return Interval("sharpe", estimate * root, (estimate - z * error) * root,
                    (estimate + z * error) * root, level, "delta method", x.size, lag,
                    floored)


def probabilistic_sharpe(returns, benchmark=0.0, periods_per_year=None, risk_free=None):
    """The one-sided probability that the Sharpe of ``returns`` is above
    ``benchmark``, an annual Sharpe, as a Probability, from the variance
    ``sharpe_interval`` reads (Specification, The probabilistic Sharpe and the minimum
    track record)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    benchmark = _finite("benchmark", benchmark, "an annual Sharpe, such as 0.0")

    variance, lag, floored = _variance(x)
    gap   = _per_bar(x) - benchmark / math.sqrt(rate)
    value = NORMAL.cdf(gap * math.sqrt(x.size - 1) / math.sqrt(variance))

    return Probability(value, benchmark, lag, floored, x.size)


def min_track_record_length(returns, benchmark=0.0, confidence=0.95, periods_per_year=None,
                            risk_free=None):
    """The returns at which the Sharpe of ``returns`` and its variance would give a
    one-sided ``confidence`` that it is above ``benchmark``, an annual Sharpe, as a
    TrackRecord in returns and in years, None where the Sharpe is not above the
    benchmark (Specification, The probabilistic Sharpe and the minimum track
    record)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    benchmark  = _finite("benchmark", benchmark, "an annual Sharpe, such as 0.0")
    confidence = _fraction("confidence", confidence)

    variance, lag, floored = _variance(x)
    gap    = _per_bar(x) - benchmark / math.sqrt(rate)
    needed = (None if gap <= 0.0
              else 1.0 + variance * (NORMAL.inv_cdf(confidence) / gap) ** 2)

    return TrackRecord(needed, None if needed is None else needed / rate, x.size / rate,
                       _per_bar(x) * math.sqrt(rate), confidence, benchmark, lag, floored,
                       x.size)


def autocorrelation(returns, lag=1):
    """The autocorrelation of ``returns``, a Result's or a plain series, at ``lag``: 1
    at lag 0, and 0 where the returns never vary or number no more than ``lag + 1``. No
    interval reads it (Specification, What a statistic reads)."""
    if not _whole(lag):
        raise ProbatumTypeError(f"lag: {lag!r}; give a whole number of bars, from 0")

    if lag < 0:
        raise ProbatumValueError(f"lag: {lag}; give a whole number of bars, from 0")

    x = _values(returns)
    if lag == 0:
        return 1.0

    centred = x - x.mean() if x.size else x
    below   = float(centred @ centred)
    if x.size <= lag + 1 or below == 0.0:
        return 0.0

    return float(centred[lag:] @ centred[:-lag]) / below


def effective_sample(returns):
    """emsl's count of independent returns in ``returns``, ``n * (1 - a) / (1 + a)`` at
    the first autocorrelation held to [0, 0.99], at least 2, and the count itself below
    4 returns. No interval reads it (Specification, What a statistic reads)."""
    x = _values(returns)
    if x.size < 4:
        return float(x.size)

    a = min(max(autocorrelation(x, 1), 0.0), 0.99)
    return max(2.0, x.size * (1.0 - a) / (1.0 + a))


def _read(returns, periods_per_year, risk_free, name="returns"):
    """The excess returns a statistic reads and the bars a year it annualises by: a
    Result's after its first row, at its own annualisation and risk-free rate, or a
    plain series by position at the rate it is given, each refused by the ``name`` it
    was passed as (Specification, What a statistic reads)."""
    # _result reads this module's types, so it is imported here and not at the top
    from ._result import Result
    if isinstance(returns, Result):
        given = returns.provenance
        if periods_per_year is not None:
            raise ProbatumTypeError(f"periods_per_year: given beside a Result, which runs "
                                    f"at {given['periods_per_year']}; leave it out")
        if risk_free is not None:
            raise ProbatumTypeError(f"risk_free: given beside a Result, which ran at "
                                    f"{given['risk_free']}; leave it out")

        rate = given["periods_per_year"]
        if rate is None:
            raise ProbatumValueError(f"{name}: a Result whose clock is too short to "
                                     f"annualise; run it over more than one row")
        return np.asarray(returns.returns[1:]) - given["risk_free"] / rate, rate

    if periods_per_year is None:
        raise ProbatumTypeError("periods_per_year: missing for a plain return series; "
                                "pass 365 for daily crypto returns")

    rate = _finite("periods_per_year", periods_per_year, "the bars a year, such as 365")
    if rate <= 0.0:
        raise ProbatumValueError(f"periods_per_year: {periods_per_year!r}; give the bars "
                                 f"a year, such as 365")

    rate_free = 0.0 if risk_free is None else _finite(
        "risk_free", risk_free, "the annual rate as a finite number, such as 0.04")
    return _series(returns, name) - rate_free / rate, rate


def _values(returns):
    """The returns a Result's record reads, or a plain series', at no annualisation."""
    from ._result import Result
    return (np.asarray(returns.returns[1:]) if isinstance(returns, Result)
            else _series(returns))


def _series(returns, name="returns", first=0):
    """A plain return series passed as ``name``: a 1-D array, a list or a pandas Series
    read by position, finite from position ``first``."""
    if isinstance(returns, (str, bytes, dict)) or not hasattr(returns, "__len__"):
        raise ProbatumTypeError(f"{name}: {type(returns).__name__}; pass a Result, or a "
                                f"1-D array, a list or a pandas Series of returns")

    try:
        values = np.asarray(returns, dtype=np.float64)
    except (TypeError, ValueError):
        raise ProbatumTypeError(f"returns: a {type(returns).__name__} that holds no "
                                f"numbers; pass a 1-D series of returns") from None

    if values.ndim != 1:
        raise ProbatumValueError(f"returns: an array of shape {values.shape}; pass one "
                                 f"series, 1-D")

    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise ProbatumValueError(f"returns: position {bad[0]} holds {values[bad[0]]}; a "
                                 f"return series holds finite numbers")

    return values


def _finite(name, value, fix):
    if not _number(value):
        raise ProbatumTypeError(f"{name}: {value!r}; give {fix}")
    if not math.isfinite(value):
        raise ProbatumValueError(f"{name}: {value!r}; give {fix}")
    return float(value)


def _fraction(name, value):
    value = _finite(name, value, "a fraction between 0 and 1, such as 0.95")
    if not 0.0 < value < 1.0:
        raise ProbatumValueError(f"{name}: {value!r}; give a fraction between 0 and 1, "
                                 f"such as 0.95")

    return value


def _ranked(reward, risk, annualizer):
    """A reward over a risk that can be zero, ranked as the record ranks it: infinite
    for a positive reward over no risk, and zero for any other (ADR 0030)."""
    if risk > 0.0:
        return float(reward / risk * annualizer)
    return math.inf if reward > 0.0 else 0.0


def _per_bar(x):
    return float(x.mean() / x.std(ddof=1))


def _variance(x):
    """The Sharpe's variance, its lag and whether its floor bound, refused for a
    sample that cannot support it (Specification, Samples with no interval)."""
    if x.size < 4:
        raise ProbatumValueError(f"returns: {x.size} of them; an interval needs at "
                                 f"least 4")

    if np.all(x == x[0]):
        raise ProbatumValueError(f"returns: all {x.size} are the same; an interval "
                                 f"needs returns that vary")

    variance, lag, floored = _floored(_influence(x))
    if variance < ROUNDING:
        raise ProbatumValueError(_still())

    return variance, lag, floored


def _still():
    """Why returns whose Sharpe's influence is zero on every bar have no interval: the
    influence is a quadratic in each standardised return, so they hold two values."""
    return ("returns: two values whose Sharpe's influence is zero on every bar, so it "
            "has no standard error; an interval needs returns that take a third")


def _influence(x):
    """The Sharpe's influence series on the moments with ``n`` in every divisor."""
    mu = x.mean()
    s  = math.sqrt(float(((x - mu) ** 2).mean()))
    u  = (x - mu) / s
    return u - (mu / s) / 2.0 * (u * u - 1.0)


def _autocovariances(z, upto):
    n = z.size
    return np.array([float(z[j:] @ z[:n - j]) / n if j < n else 0.0
                     for j in range(upto + 1)])


def _lag(z):
    """Newey and West's 1994 lag for the Bartlett kernel, on a series whose mean is
    zero."""
    n  = z.size
    p  = math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))
    g  = _autocovariances(z, p)
    s0 = g[0] + 2.0 * g[1:].sum()
    s1 = 2.0 * (np.arange(1, p + 1) * g[1:]).sum()
    if s0 <= 0.0:
        return 0

    return min(math.floor(1.1447 * ((s1 / s0) ** 2 * n) ** (1.0 / 3.0)), n - 1)


def _bartlett(g, m):
    weights = 1.0 - np.arange(1, m + 1) / (m + 1.0)
    return float(g[0] + 2.0 * (weights * g[1:m + 1]).sum())


def _floored(z):
    """The Bartlett sum at the 1994 lag on ``z``, floored at its value at lag zero,
    with that lag and whether the floor bound."""
    m      = _lag(z)
    g      = _autocovariances(z, m)
    summed = _bartlett(g, m)
    return max(summed, float(g[0])), m, summed < g[0]
