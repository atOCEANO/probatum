"""``probatum.stats``: how far a run's statistics can be trusted, read from a Result or
from any series of returns: the Sharpe's interval by the delta method over a long-run
variance with Newey and West's weights, its probability of beating a benchmark and the
track record that would show it, the long-run Sharpe, and every other statistic's
interval by a bootstrap of the returns or of the round trips (ADR 0031; Specification,
Inference)."""

import hashlib
import math
import statistics

import numpy as np

from ._errors import ProbatumAttributeError, ProbatumTypeError, ProbatumValueError
from ._space import _flag, _number, _whole
from ._streams import stream

__all__ = ["Interval", "Probability", "TrackRecord", "autocorrelation", "bootstrap",
           "effective_sample", "min_track_record_length", "probabilistic_sharpe", "sharpe",
           "sharpe_interval"]

NORMAL   = statistics.NormalDist()
ROUNDING = 1e-20     # a standardised influence's variance below this is rounding
NUDGE    = 1e-9      # a rank this near a whole number is that number
CEILING  = 1e12      # what a CAGR saturates at, as the record's does
FLOOR    = 60        # the closed round trips a trade statistic's interval needs
SPAN     = 500       # the returns a bootstrap of them needs
BLOCKS   = 100       # the mean blocks those returns need to hold

# the statistics a stationary bootstrap of the returns serves: each one's words in a
# text, its unit, whether a text prints it as a loss, and how the resamples are read,
# studentized on a pivot, at their percentiles, or as the middle of the resampled
# histories, which is no interval
RETURNS = {
    "total_return_pct":          ("total return", "%", False, "studentized"),
    "net_profit_pct":            ("net profit", "%", False, "studentized"),
    "cagr_pct":                  ("cagr", "%", False, "studentized"),
    "volatility_pct":            ("volatility", "%", False, "studentized"),
    "sortino":                   ("sortino", "", False, "studentized"),
    "value_at_risk":             ("value at risk", "%", True, "percentile"),
    "conditional_value_at_risk": ("conditional value at risk", "%", True, "studentized"),
    "max_drawdown_pct":          ("max drawdown", "%", True, "range"),
    "calmar":                    ("calmar", "", False, "range"),
}
# the statistics read on the growth a bar, which a return of -100% leaves undefined
GROWTH = ("total_return_pct", "net_profit_pct", "cagr_pct")
# the statistics a bootstrap of the closed round trips serves: words, unit and method
TRADES = {
    "win_rate":      ("win rate", "", "percentile bootstrap of round trips"),
    "profit_factor": ("profit factor", "",
                      "bias-corrected and accelerated bootstrap of round trips"),
    "avg_trade_pct": ("average trade", "%", "studentized bootstrap of round trips"),
}


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
    ``high``, in its units and with its sign, a drawdown and a value at risk positive
    losses; ``level``; ``method``, how it was found; ``count``, the returns or round
    trips it read; ``lag`` and ``floored``, the delta method's lag and whether its
    floor at lag zero bound; ``block``, a stationary bootstrap's mean block; and
    ``kind``, ``"interval"``, or ``"range"`` for the middle of the resampled histories,
    which is no confidence interval. Its text says each, a loss as the negative return
    it is."""

    __slots__ = ("statistic", "estimate", "low", "high", "level", "method", "count", "lag",
                 "floored", "block", "kind")


    def __init__(self, statistic, estimate, low, high, level, method, count, lag=None,
                 floored=None, block=None, kind="interval"):
        self._set(
            statistic = statistic,
            estimate  = float(estimate),
            low       = float(low),
            high      = float(high),
            level     = float(level),
            method    = method,
            count     = int(count),
            lag       = None if lag is None else int(lag),
            floored   = None if floored is None else bool(floored),
            block     = None if block is None else float(block),
            kind      = kind,
        )


    def __str__(self):
        words, unit, loss = _shown(self.statistic)
        flip = -1.0 if loss else 1.0
        low, high = sorted((flip * self.low + 0.0, flip * self.high + 0.0))
        what = (f"; middle {_percent(self.level)} of resampled histories"
                if self.kind == "range" else f", two-sided {_percent(self.level)} interval")

        how = self.method
        if self.lag is not None:
            how += f", Newey-West lag {self.lag}"
            how += " floored at lag zero" if self.floored else ""
        if self.block is not None:
            how += f", block {self.block:.1f}"

        noun = "round trips" if self.statistic in TRADES else "returns"
        return (f"{words} {flip * self.estimate + 0.0:.2f}{unit}{what} {low:.2f}{unit} to "
                f"{high:.2f}{unit}, {how}, {self.count:,} {noun}")


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


def _shown(statistic):
    """A statistic's words in a text, its unit and whether the text prints it as a
    loss."""
    if statistic in RETURNS:
        return RETURNS[statistic][:3]
    if statistic in TRADES:
        return TRADES[statistic][0], TRADES[statistic][1], False
    return statistic, "", False


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


def bootstrap(returns, statistic, periods_per_year=None, risk_free=None, level=0.95,
              resamples=2000, seed=None):
    """``statistic``'s interval at ``level``, as an Interval, by a bootstrap of
    ``returns``, read as ``sharpe`` reads them. Total return, net profit, CAGR,
    volatility, Sortino, value at risk and conditional value at risk take a stationary
    block bootstrap of the returns, its mean block the longest Politis and White's rule
    finds on the returns, their squared deviations or their tail, each studentized on
    its own standard error but value at risk, whose interval is percentile, and maximum
    drawdown and Calmar the middle of those resampled histories, which is no confidence
    interval. A Result's win rate, profit factor and average trade take a bootstrap of
    its closed round trips. A bootstrap of the returns needs 500 of them in 100 of its
    blocks, and one of the round trips 60 of them. The ``resamples`` come from a stream
    seeded by the values they resample, or by ``seed``, so an interval repeats on every
    read with no seed (Specification, Intervals of the returns; Intervals of the round
    trips; Samples with no interval)."""
    if not isinstance(statistic, str):
        raise ProbatumTypeError(f"statistic: {statistic!r}; give one of "
                                f"{', '.join([*RETURNS, *TRADES])}")
    if statistic not in RETURNS and statistic not in TRADES:
        raise ProbatumValueError(f"statistic: {statistic!r}; give one of "
                                 f"{', '.join([*RETURNS, *TRADES])}")

    level = _fraction("level", level)

    if not _whole(resamples):
        raise ProbatumTypeError(f"resamples: {resamples!r}; give a whole number from 2")
    if resamples < 2:
        raise ProbatumValueError(f"resamples: {resamples}; give a whole number from 2")

    if seed is not None and not _whole(seed):
        raise ProbatumTypeError(f"seed: {seed!r}; give a whole number from 0, or None")
    if seed is not None and seed < 0:
        raise ProbatumValueError(f"seed: {seed}; give a whole number from 0, or None")

    if _rank(resamples, (1.0 - level) / 2.0) == _rank(resamples, (1.0 + level) / 2.0):
        raise ProbatumValueError(f"resamples: {resamples} at a level of {level:g}, which "
                                 f"puts both bounds on one resample; give more")

    if statistic in TRADES:
        return _trade_interval(returns, statistic, periods_per_year, risk_free, level,
                               resamples, seed)

    x, rate = _read(returns, periods_per_year, risk_free)
    _supports(x)

    made, said = _return_core(_values(returns), x, rate, (statistic,), level, resamples,
                              seed)[statistic]
    if made is None:
        raise ProbatumValueError(said)

    return made


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
        raise ProbatumTypeError(f"{name}: a {type(returns).__name__} that holds no "
                                f"numbers; pass a 1-D series of returns") from None

    if values.ndim != 1:
        raise ProbatumValueError(f"{name}: an array of shape {values.shape}; pass one "
                                 f"series, 1-D")

    bad = np.flatnonzero(~np.isfinite(values[first:])) + first
    if bad.size:
        raise ProbatumValueError(f"{name}: position {bad[0]} holds {values[bad[0]]}; a "
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


def _supports(x):
    """Refuse returns that cannot support an interval: fewer than 4, or returns that
    never vary (Specification, Samples with no interval)."""
    if x.size < 4:
        raise ProbatumValueError(f"returns: {x.size} of them; an interval needs at "
                                 f"least 4")

    if np.all(x == x[0]):
        raise ProbatumValueError(f"returns: all {x.size} are the same; an interval "
                                 f"needs returns that vary")


def _variance(x):
    """The Sharpe's variance, its lag and whether its floor bound, refused where it is
    zero."""
    _supports(x)

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


def _trade_interval(returns, statistic, periods_per_year, risk_free, level, resamples,
                    seed):
    """A trade statistic's interval from a bootstrap of a Result's closed round trips:
    the win rate's percentile, the profit factor's bias-corrected and accelerated and
    the average trade's studentized (Specification, Intervals of the round trips)."""
    from ._result import Result
    if not isinstance(returns, Result):
        raise ProbatumValueError(f"statistic: {statistic} reads round trips, which a plain "
                                 f"series does not hold; pass the Result")
    _read(returns, periods_per_year, risk_free)

    trades  = returns.trades
    pnl     = np.asarray(trades["pnl"][~trades["open"]], dtype=np.float64)
    opening = float(returns.equity[0])
    count   = pnl.size
    won, lost = np.count_nonzero(pnl > 0.0), np.count_nonzero(pnl < 0.0)

    if count < FLOOR:
        raise ProbatumValueError(f"round trips: {count} closed; an interval of "
                                 f"{statistic} needs at least {FLOOR}")

    if statistic == "win_rate" and not 0 < won < count:
        raise ProbatumValueError(f"win_rate: {won} of {count} round trips won; an "
                                 f"interval needs one that won and one that did not")

    if statistic == "profit_factor" and (won == 0 or lost < 2):
        raise ProbatumValueError(f"profit_factor: {won} round trips won and {lost} lost; "
                                 f"an interval needs one that won and two that lost")

    if statistic == "avg_trade_pct" and np.all(pnl == pnl[0]):
        raise ProbatumValueError("avg_trade_pct: every round trip made the same; an "
                                 "interval needs pnl that varies")

    estimate, low, high = _trade_bounds(pnl, opening, statistic, level, resamples,
                                        _drawn(pnl, "stats trades", seed))
    if low == high:
        raise ProbatumValueError(_met(statistic, low))

    return Interval(statistic, estimate, low, high, level, TRADES[statistic][2], count)


def _return_core(r, x, rate, names, level, resamples, seed):
    """Each of the return statistics ``names``' interval, from one stationary block
    bootstrap of returns ``r`` that vary, whose excess returns are ``x``: its Interval
    beside None, or None beside why the sample cannot support one (Specification,
    Intervals of the returns; Samples with no interval)."""
    whole  = _resampled(r, x, rate, np.arange(r.size)[None, :])
    block  = _return_block(r)
    said   = {name: _unsupported(r, name, float(whole[name][0]), block) for name in names}
    pivots = {}

    for name in names:
        if said[name] is None and RETURNS[name][3] == "studentized":
            theta, psi = _pivot(name, r[None, :], x[None, :])
            m     = _lag(psi[0])
            error = math.sqrt(float(_long_run(psi, m)[0]) / (r.size - 1))
            if error > 0.0:
                pivots[name] = float(theta[0]), error, m
            else:
                said[name] = (f"{name}: no standard error, as the returns never vary in "
                              f"what it reads; an interval needs returns that do")

    held  = [name for name in names if said[name] is None]
    found = {name: [] for name in held}

    if held:
        for part in _stationary(r.size, block, resamples, _drawn(r, "stats returns", seed)):
            taken, excess = r[part], x[part]
            plain = (None if all(name in pivots for name in held)
                     else _resampled(r, x, rate, part))
            for name in held:
                found[name].append(_ratios(name, taken, excess, pivots[name][0],
                                           pivots[name][2])
                                   if name in pivots else plain[name])

    low_q, high_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    made = {name: (None, said[name]) for name in names}
    for name in held:
        values, how = np.concatenate(found[name]), RETURNS[name][3]
        if name in pivots:
            theta, error, _ = pivots[name]
            low, high = sorted(_unpivoted(name, theta - _ordered(values, q) * error,
                                          r.size, rate) for q in (high_q, low_q))
        else:
            low, high = _ordered(values, low_q), _ordered(values, high_q)

        if how != "range" and low == high:
            made[name] = None, _met(name, low)
            continue

        method = ("stationary block bootstrap" if how == "range"
                  else f"{how} stationary block bootstrap")
        made[name] = Interval(
            name,
            float(whole[name][0]),
            low,
            high,
            level,
            method,
            r.size,
            block = block,
            kind  = "range" if how == "range" else "interval",
        ), None

    return made


def _unsupported(r, statistic, estimate, block):
    """Why the returns ``r``, resampled in a mean block of ``block``, cannot support an
    interval of ``statistic``, whose estimate on them is ``estimate``, or None where
    they can."""
    if math.isinf(estimate):
        return (f"{statistic}: inf, a reward over no measured risk; an estimate ranked "
                f"at infinity has no interval")

    if statistic in GROWTH and r.min() <= -1.0:
        return (f"{statistic}: a return of {r.min() * 100.0:.2f}%, which leaves no "
                f"equity to grow; growth has no interval past a loss of all of it")

    if r.size < SPAN:
        return (f"returns: {r.size:,} of them; a bootstrap of returns needs at least "
                f"{SPAN}")

    if r.size < BLOCKS * block:
        # the block rounded up and the count down, so neither prints as meeting the
        # floor it misses
        shown = math.ceil(block * 10.0) / 10.0
        held  = math.floor(r.size / block * 10.0) / 10.0
        return (f"returns: {r.size:,} in blocks of {shown:.1f}, {held:.1f} of them; a "
                f"bootstrap of returns needs at least {BLOCKS} blocks")

    if statistic == "conditional_value_at_risk":
        k      = (r.size + 19) // 20
        lowest = np.partition(r, k - 1)[:k]
        if np.all(lowest == lowest[0]):
            return (f"{statistic}: the {k} lowest returns are all "
                    f"{lowest[0] * 100.0:.2f}%; an interval needs a tail that varies")

    return None


def _met(statistic, value):
    """Why an interval of ``statistic`` whose resamples put both its bounds on ``value``
    has none (Specification, Samples with no interval)."""
    _, unit, loss = _shown(statistic)
    shown = -value + 0.0 if loss else value + 0.0
    return (f"{statistic}: its resamples put both bounds on {shown:.2f}{unit}; an "
            f"interval needs resamples that differ between them")


def _pivot(statistic, r, x):
    """What a studentized interval of ``statistic`` is read on, along each row of the
    returns ``r`` whose excess returns are ``x``, and its influence series, with ``n``
    in every divisor: the growth a bar for total return, net profit and CAGR, the log
    of the deviation for volatility, the Sortino a bar, and the mean of the ``k``
    lowest returns for conditional value at risk."""
    count = r.shape[1]

    # a resample whose returns never vary, or never fall below the rate, has a pivot
    # or an influence its ratios read as a move with no error to measure it
    with np.errstate(divide="ignore", invalid="ignore"):
        if statistic in GROWTH:
            grown = np.log1p(r)
            g     = grown.mean(axis=1, keepdims=True)
            return g[:, 0], grown - g

        if statistic == "volatility_pct":
            centred = r - r.mean(axis=1, keepdims=True)
            s2      = (centred * centred).mean(axis=1, keepdims=True)
            return (0.5 * np.log(s2[:, 0] * count / (count - 1.0)),
                    (centred * centred - s2) / (2.0 * s2))

        if statistic == "sortino":
            mu  = x.mean(axis=1, keepdims=True)
            low = np.minimum(x, 0.0) ** 2
            d2  = low.mean(axis=1, keepdims=True)
            d   = np.sqrt(d2)
            return (_ranked_all(mu[:, 0], d[:, 0], 1.0),
                    (x - mu) / d - mu / (2.0 * d * d2) * (low - d2))

    k      = (count + 19) // 20
    lowest = np.partition(r, k - 1, axis=1)
    q      = lowest[:, k - 1:k]
    tail   = lowest[:, :k].mean(axis=1, keepdims=True)
    return tail[:, 0], q + (r - q) * (r <= q) * (count / k) - tail


def _ratios(statistic, r, x, estimate, m):
    """Each resample's studentized statistic: its pivot's move from the sample's
    ``estimate`` over its own standard error, at the sample's lag ``m``, infinite in
    the sign of a move it has no error to measure, and 0 where it did not move."""
    theta, psi = _pivot(statistic, r, x)

    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.sqrt(_long_run(psi, m) / (r.shape[1] - 1))
        t     = np.where(np.isfinite(error) & (error > 0.0), (theta - estimate) / error,
                         np.sign(theta - estimate) * math.inf)

    return np.where(np.isnan(t), 0.0, t)


def _unpivoted(statistic, theta, count, rate):
    """A bound of ``statistic`` from its pivot's bound ``theta``, on ``count`` returns at
    ``rate`` bars a year."""
    with np.errstate(over="ignore"):
        if statistic in ("total_return_pct", "net_profit_pct"):
            return float(np.expm1(count * theta)) * 100.0
        if statistic == "cagr_pct":
            return min(float(np.expm1(rate * theta)), CEILING) * 100.0
        if statistic == "volatility_pct":
            return float(np.exp(theta)) * math.sqrt(rate) * 100.0

    if statistic == "sortino":
        return theta * math.sqrt(rate)
    return -theta * 100.0


def _long_run(psi, m):
    """Each row's Bartlett sum of the influence series ``psi`` at lag ``m``, floored at
    its value at lag zero. ``n * (m + 1)`` times the sum is the sum of the squared
    sums of every window of ``m + 1`` rows, the rows outside the series zero, so a
    resample costs its length at any lag."""
    rows, count = psi.shape
    prefix = np.concatenate([np.zeros((rows, 1)), np.cumsum(psi, axis=1)], axis=1)
    starts = np.arange(-m, count)
    sums   = prefix[:, np.minimum(starts + m + 1, count)] - prefix[:, np.maximum(starts, 0)]
    summed = (sums * sums).sum(axis=1) / (count * (m + 1.0))
    return np.maximum(summed, (psi * psi).sum(axis=1) / count)


def _trade_bounds(pnl, opening, statistic, level, resamples, rng):
    """A trade statistic's estimate and bounds from ``resamples`` draws of the round
    trips' ``pnl``, each as many trips drawn evenly with replacement."""
    count = pnl.size
    taken = pnl[rng.integers(0, count, (resamples, count))]
    low_q, high_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0

    if statistic == "win_rate":
        values = (taken > 0.0).mean(axis=1)
        return (np.count_nonzero(pnl > 0.0) / count, _ordered(values, low_q),
                _ordered(values, high_q))

    if statistic == "profit_factor":
        gains, losses = np.where(pnl > 0.0, pnl, 0.0), np.where(pnl < 0.0, -pnl, 0.0)
        estimate = gains.sum() / losses.sum()
        values   = _ranked_all(np.where(taken > 0.0, taken, 0.0).sum(axis=1),
                               np.where(taken < 0.0, -taken, 0.0).sum(axis=1), 1.0)
        return (estimate, *_corrected(values, estimate,
                                      (gains.sum() - gains) / (losses.sum() - losses),
                                      low_q, high_q))

    m, s, root = pnl.mean(), pnl.std(ddof=1), math.sqrt(count)
    means, spreads = taken.mean(axis=1), taken.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(spreads > 0.0, (means - m) / (spreads / root),
                     np.sign(means - m) * math.inf)
    t = np.where(np.isnan(t), 0.0, t)
    return (m / opening * 100.0, (m - _ordered(t, high_q) * s / root) / opening * 100.0,
            (m - _ordered(t, low_q) * s / root) / opening * 100.0)


def _corrected(values, estimate, jackknife, low_q, high_q):
    """The bias-corrected and accelerated bounds of resampled ``values`` around
    ``estimate``, its acceleration read from the ``jackknife``, each value with one
    observation left out (Efron 1987)."""
    below = (np.count_nonzero(values < estimate)
             + 0.5 * np.count_nonzero(values == estimate)) / values.size
    if not 0.0 < below < 1.0:
        raise ProbatumValueError(f"resamples: {values.size:,}, every one of them on one "
                                 f"side of the profit factor's estimate; give more, since "
                                 f"its bias correction needs some on each side")

    z0     = NORMAL.inv_cdf(below)
    away   = jackknife.mean() - jackknife
    a      = float((away ** 3).sum() / (6.0 * float((away ** 2).sum()) ** 1.5))
    bounds = []
    for q in (low_q, high_q):
        z = NORMAL.inv_cdf(q)
        bounds.append(_ordered(values, NORMAL.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))))

    return tuple(bounds)


def _drawn(values, name, seed):
    """The stream ``name`` seeded by a 64-bit BLAKE2b digest of the float64 ``values``
    it resamples, or by ``seed`` where one is given (ADR 0021)."""
    if seed is None:
        blob = np.ascontiguousarray(values, dtype="<f8").tobytes()
        seed = int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "little")

    return stream(seed, name)


def _block_lengths(y):
    """The stationary and the circular block lengths that Politis and White's rule gives
    ``y``, as Patton's opt_block_length_REV_dec07 computes them, a length held to at
    least 1, and 1 for a series too short or too flat for the rule to read."""
    count = y.size
    m_max = math.ceil(math.sqrt(count)) + 5
    b_max = math.ceil(min(3.0 * math.sqrt(count), count / 3.0))
    if count < m_max + 3 or np.all(y == y[0]):
        return 1.0, 1

    h     = 2.0 * math.sqrt(math.log10(count) / count)
    later = y[m_max:] - y[m_max:].mean()
    rho   = np.empty(m_max)

    for j in range(1, m_max + 1):
        side   = y[m_max - j:count - j]
        side   = side - side.mean()
        bottom = float(later @ later) * float(side @ side)
        if bottom == 0.0:
            return 1.0, 1
        rho[j - 1] = float(later @ side) / math.sqrt(bottom)

    small = np.abs(rho) < h
    m_hat = next((j for j in range(1, m_max - 3) if small[j - 1:j + 4].all()), None)
    if m_hat is None:
        m_hat = int(np.flatnonzero(np.abs(rho) > h)[-1]) + 1

    top  = min(2 * m_hat, m_max)
    head = y[top:] - y[top:].mean()
    r    = np.empty(top + 1)
    for j in range(top + 1):
        side = y[top - j:count - j]
        r[j] = float(head @ (side - side.mean())) / (head.size - 1)

    v        = np.arange(1, top + 1) / top
    w        = np.where(v < 0.5, 1.0, 2.0 * (1.0 - v))
    spectral = float(r[0] + 2.0 * (w * r[1:]).sum())

    if spectral == 0.0:
        raw = math.inf
    else:
        raw = (abs(2.0 * float((w * np.arange(1, top + 1) * r[1:]).sum()) / spectral)
               ** (2.0 / 3.0) * count ** (1.0 / 3.0))

    return (max(1.0, min(raw, b_max)),
            max(1, math.ceil(min(1.5 ** (1.0 / 3.0) * raw, b_max))))


def _return_block(r):
    """The mean block of a stationary bootstrap of the returns ``r``: the longest
    stationary length of the returns, their squared deviations from their mean and the
    indicator of a return at or below the value at risk's."""
    k      = (r.size + 19) // 20
    cutoff = np.partition(r, k - 1)[k - 1]
    return max(_block_lengths(series)[0] for series in (
        r, (r - r.mean()) ** 2, (r <= cutoff).astype(np.float64)))


def _stationary(count, block, resamples, rng):
    """The rows of ``resamples`` stationary resamples of ``count`` returns, some at a
    time: each first row drawn evenly, and each next the one after the last, wrapping
    past the last return, with probability ``1 - 1/block``, and drawn evenly again
    otherwise (Politis and Romano 1994)."""
    chunk = max(1, 2_000_000 // count)
    at    = np.arange(count)

    for first in range(0, resamples, chunk):
        rows  = min(chunk, resamples - first)
        fresh = rng.random((rows, count)) < 1.0 / block
        fresh[:, 0] = True
        starts = rng.integers(0, count, (rows, count))
        began  = np.maximum.accumulate(np.where(fresh, at, 0), axis=1)
        yield (np.take_along_axis(starts, began, axis=1) + (at - began)) % count


def _resampled(r, x, rate, rows):
    """Each return statistic on the resamples ``rows`` of the returns ``r``, whose
    excess returns are ``x``, computed as the record computes it, a resampled equity
    that reaches zero moving no further."""
    taken, excess = r[rows], x[rows]
    grown = np.cumprod(1.0 + taken, axis=1)
    gone  = grown <= 0.0
    for row in np.flatnonzero(gone.any(axis=1)):
        at = int(gone[row].argmax())
        grown[row, at + 1:] = grown[row, at]

    final = grown[:, -1]
    with np.errstate(over="ignore", invalid="ignore"):
        growth = np.where(final > 0.0, np.minimum(
            np.power(np.where(final > 0.0, final, 1.0), rate / r.size) - 1.0, CEILING),
            -1.0)

    path   = np.concatenate([np.ones((len(grown), 1)), grown], axis=1)
    peak   = np.maximum.accumulate(path, axis=1)
    fall   = np.minimum((peak - path) / peak, 1.0).max(axis=1)
    k      = (r.size + 19) // 20
    lowest = np.partition(taken, k - 1, axis=1)[:, :k]
    total  = (final - 1.0) * 100.0

    return {"total_return_pct": total, "net_profit_pct": total, "cagr_pct": growth * 100.0,
            "volatility_pct": taken.std(axis=1, ddof=1) * math.sqrt(rate) * 100.0,
            "sortino": _ranked_all(excess.mean(axis=1),
                                   np.sqrt((np.minimum(excess, 0.0) ** 2).mean(axis=1)),
                                   math.sqrt(rate)),
            "max_drawdown_pct": fall * 100.0,
            "calmar": np.where(growth < 0.0, growth * fall, _ranked_all(growth, fall, 1.0)),
            "value_at_risk": -lowest.max(axis=1) * 100.0,
            "conditional_value_at_risk": -lowest.mean(axis=1) * 100.0}


def _ranked_all(reward, risk, annualizer):
    """``_ranked`` over arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(risk > 0.0, reward / np.where(risk > 0.0, risk, 1.0) * annualizer,
                        np.where(reward > 0.0, math.inf, 0.0))


def _ordered(values, q):
    """The value of rank ``_rank(B, q)`` among the ``B`` values."""
    rank = _rank(values.size, q)
    return float(np.partition(values, rank - 1)[rank - 1])


def _rank(count, q):
    """The rank ``ceil(count * q)``, a rank within 1e-9 of a whole number being that
    number, the first at least (Specification, Intervals of the returns)."""
    return max(1, math.ceil(count * q - NUDGE))
