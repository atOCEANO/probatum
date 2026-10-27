"""A small, pinned set of indicators.

Thirty functions, chosen to cover what a bar-level strategy usually reaches for
and stopped there. This is not an attempt to be ta-lib: an indicator library is
mostly somebody's opinion about warm-up and smoothing, and a hundred of those
opinions is a hundred chances to disagree with the chart you have been reading
for years.

Three rules hold for every function here, and they are the reason it lives in the
library rather than in your notebook:

**One value a bar, each instrument over its own.** Every function takes a series
of shape ``(T,)``, or a universe of shape ``(T, N)`` with a column for each
instrument, its time along ``axis``, the first by default, and returns float64 of
the same shape, aligned so that entry ``i`` is bar ``i``, with the warm-up as
``NaN``. Never a shorter array. Each column runs over its own values: a value that
is ``NaN`` or infinite is a row the column does not have, so a window, a smoother,
a lookback and a crossing carry across it, and the output there is ``NaN``. A
Data's fields are ``NaN`` exactly where an instrument printed no bar, so a closure
neither breaks an indicator nor restarts it, and a series too short for its
warm-up is all warm-up rather than an error, only one of no bars being refused, so
a cut of a series that has a bar gives the first rows of the whole. The same array
goes straight into a rule and into ``probatum.chart``, a column at a time, with no
padding decision in between. The two functions that answer a yes-or-no question,
``crossover`` and ``crossunder``, return booleans and pad with ``False`` rather
than ``NaN``, because ``bool(nan)`` is ``True`` and a warm-up that reads as a
signal is the one shape of this rule that would be dangerous (ADR 0037).

**The convention is written down.** Where implementations differ, and they differ
most on how an exponential average is seeded and which smoothing an oscillator
uses, this module says which one it picked in the function's own docstring. A
curve that quietly disagrees with the chart you trust is worse than no curve.

**Nothing here knows about the engine.** These are functions of arrays. They hold
no state, take no engine, need no pandas, and the chart still computes nothing
(ADR 0042): you compute once in ``init``, trade on that array, and hand the same
object to the chart, so the line on screen cannot drift from the line that made
the decision.

    class Cross(probatum.Strategy):
        def init(self, data):
            self.fast = probatum.ta.ema(data.close, 20)
            self.slow = probatum.ta.ema(data.close, 60)
            self.warmup = 60
"""

from __future__ import annotations

import math

import numpy as np

from ._errors import DataError, ProbatumTypeError, ProbatumValueError

__all__ = [
    "sma",
    "ema",
    "wma",
    "hma",
    "vwma",
    "vwap",
    "rsi",
    "macd",
    "stoch",
    "willr",
    "cci",
    "roc",
    "atr",
    "natr",
    "true_range",
    "bbands",
    "keltner",
    "stdev",
    "donchian",
    "supertrend",
    "adx",
    "obv",
    "mfi",
    "zscore",
    "shift",
    "change",
    "highest",
    "lowest",
    "crossover",
    "crossunder",
    "Bands",
    "Channel",
    "Macd",
    "Stochastic",
    "Trend",
]


def _shown(line):
    return f"{line.shape[0]} bars" if line.ndim == 1 else f"shape {line.shape}"


class Bands:
    """Three lines around a middle: an upper, a middle and a lower."""

    __slots__ = ("upper", "middle", "lower")


    def __init__(self, upper, middle, lower):
        self.upper  = upper
        self.middle = middle
        self.lower  = lower


    def __repr__(self):
        return f"Bands({_shown(self.middle)})"


class Channel:
    """A band around a middle, plus which side of it the price sits on."""

    __slots__ = ("upper", "middle", "lower", "direction")


    def __init__(self, upper, middle, lower, direction):
        self.upper     = upper
        self.middle    = middle
        self.lower     = lower
        self.direction = direction


    def __repr__(self):
        return f"Channel({_shown(self.middle)})"


class Trend:
    """A directional reading and the two strengths behind it."""

    __slots__ = ("adx", "plus", "minus")


    def __init__(self, adx, plus, minus):
        self.adx   = adx
        self.plus  = plus
        self.minus = minus


    def __repr__(self):
        return f"Trend({_shown(self.adx)})"


class Macd:
    """The convergence line, its signal, and the gap between them."""

    __slots__ = ("line", "signal", "histogram")


    def __init__(self, line, signal, histogram):
        self.line      = line
        self.signal    = signal
        self.histogram = histogram


    def __repr__(self):
        return f"Macd({_shown(self.line)})"


class Stochastic:
    """The raw oscillator and its smoothing."""

    __slots__ = ("k", "d")


    def __init__(self, k, d):
        self.k = k
        self.d = d


    def __repr__(self):
        return f"Stochastic({_shown(self.k)})"


def _number(value):
    """Whether ``value`` is a real number and no bool."""
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


def _length(length, name, least=1):
    """``length``, the argument ``name``, as a whole number of bars from ``least``."""
    # a fractional length is a mistake, not a request to round one way or the
    # other, and silently flooring it answered a question nobody asked
    if not _number(length):
        raise ProbatumTypeError(f"{name}: {length!r}; give a whole number of bars "
                                f"from {least}")

    whole = (isinstance(length, (int, np.integer))
             or (math.isfinite(length) and float(length).is_integer()))
    if not whole or length < least:
        raise ProbatumValueError(f"{name}: {length!r}; give a whole number of bars "
                                 f"from {least}")

    return int(length)


def _width(width, name, above=False):
    """``width``, the argument ``name``, as a finite number of at least 0, or above it
    where ``above`` says so."""
    # a negative width turns a band inside out, upper below lower, and every
    # comparison written against it then reads backwards
    fix = "give a width above 0" if above else "give a width of at least 0"
    if not _number(width):
        raise ProbatumTypeError(f"{name}: {width!r}; {fix}")

    width = float(width)
    if not math.isfinite(width) or width < 0.0 or (above and width == 0.0):
        raise ProbatumValueError(f"{name}: {width}; {fix}")

    return width


def _axis(axis, ndim):
    """``axis``, the time axis of an array of ``ndim`` dimensions, counted from 0."""
    if not isinstance(axis, (int, np.integer)) or isinstance(axis, (bool, np.bool_)):
        raise ProbatumTypeError(f"axis: {axis!r}; give the time axis as a whole number")

    if not -ndim <= axis < ndim:
        axes = "0" if ndim == 1 else "0 or 1"
        raise ProbatumValueError(f"axis: {axis} for an array of {ndim} "
                                 f"dimension{'s' * (ndim > 1)}; give {axes}")

    return int(axis) % ndim


def _blank(size):
    return np.full(size, np.nan, dtype=np.float64)


def _over(kernel, series, axis, lines=1, flag=False):
    """``kernel`` run on each column of ``series``, a dict from each argument's name to
    its values, along ``axis``: on the rows where every one of them has a value, in
    order, each result written back on its row, and NaN, or False for a crossing, on
    the rows a column does not have, so a column with no values never reaches the
    kernel. One array of the series' shape, or a tuple of ``lines`` of them."""
    arrays = {}
    for name, values in series.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            raise DataError(f"{name}: {values!r}; give a series of bars")
        if array.ndim > 2:
            raise DataError(f"{name}: an array of shape {array.shape}; give a (T,) "
                            f"series or a (T, N) array, a column for each instrument")
        arrays[name] = array

    (first, shape), *others = [(name, array.shape) for name, array in arrays.items()]
    for name, other in others:
        if other != shape:
            owner = f"{first}'" if first.endswith("s") else f"{first}'s"
            raise DataError(f"{name}: shape {other} beside {owner} {shape}; give "
                            f"series of one shape")

    axis = _axis(axis, len(shape))
    rows = shape[axis]
    if not rows:
        raise DataError(f"{first}: no bars; give a series that has some")

    # time first and a column for each instrument, as views of what was given
    columns = [np.moveaxis(array, axis, 0).reshape(rows, -1)
               for array in arrays.values()]

    width = columns[0].shape[1]
    out   = [np.zeros((rows, width), dtype=bool) if flag else
             np.full((rows, width), np.nan) for _ in range(lines)]

    for k in range(width):
        own = np.isfinite(columns[0][:, k])
        for column in columns[1:]:
            own &= np.isfinite(column[:, k])
        if not own.any():
            continue
        start = int(own.argmax())
        # a column whose values run unbroken to its end, as a warm-up leaves one, goes
        # over as a slice, sparing the copy and the scatter a mask costs, so a kernel
        # never writes to what it is handed, which can be the caller's own array
        pick  = slice(start, None) if own[start:].all() else own
        given = kernel(*(np.ascontiguousarray(column[pick, k]) for column in columns))
        for line, values in zip(out, given if lines > 1 else (given,)):
            line[pick, k] = values

    moved = np.moveaxis(arrays[first], axis, 0).shape
    out   = [np.ascontiguousarray(np.moveaxis(line.reshape(moved), 0, axis))
             for line in out]

    return tuple(out) if lines > 1 else out[0]


def _windows(values, length):
    # every full window of `length`, as a (size - length + 1, length) view, and none
    # where the series is shorter than one, as an empty view of one column, since
    # numpy cannot size an empty one of `length` columns once that passes its
    # largest index. A NaN inside a window carries through to its result, which is
    # how a line built from a series' changes, such as mfi's flows, takes its warm-up
    if values.size < length:
        return np.empty((0, 1))
    return np.lib.stride_tricks.sliding_window_view(values, length)


def _smooth(values, alpha, length):
    # one exponential pass, seeded from the simple average of the first full
    # window rather than from the first value. Seeding from the first value lets a
    # single opening print steer the curve for hundreds of bars, and the simple
    # average is what TradingView and Wilder both use.
    #
    # A value that is not finite is skipped, as everywhere in this module, and the
    # output there is NaN. A smoother runs on a column's own values, so the only ones
    # it meets lead the lines it is handed, rsi's first change, adx's first moves and
    # dx's warm-up, longer where the bars open with no direction, and what follows them
    # is, bit for bit, what the line without them gives.
    #
    # The pass is vectorized rather than looped per bar. A per-bar Python loop
    # measured 191 ms for one EMA over 100k candles against 18 ms for the simple
    # average of the same length, which made two EMAs computed in `init` 42% of an
    # entire backtest trial, paid again on every trial of every search
    out    = _blank(values.size)
    finite = np.isfinite(values)
    given  = values[finite]
    if given.size < length:
        return out

    smoothed = _blank(given.size)
    smoothed[length - 1] = float(given[:length].mean())
    smoothed[length:]    = _decay(given[length:], smoothed[length - 1], alpha)
    out[finite]          = smoothed

    return out


def _block(keep):
    # how many bars one cumulative sum may span before its own weights overflow:
    # keep**-block held under about 1e150, ten orders of magnitude inside the
    # largest finite double. A function rather than a line inside `_decay` so a
    # test can read the size the module actually uses instead of recomputing the
    # formula beside it and agreeing with itself
    return max(1, int(150.0 / -math.log10(keep)))


def _decay(values, seed, alpha):
    """One exponential pass over a stretch that has no gaps in it.

    ``out[k] = alpha * values[k] + (1 - alpha) * out[k - 1]`` unrolled is
    ``keep**(k + 1) * seed`` plus a decaying sum of the values, and written as one
    cumulative sum that needs ``keep**-k``. Which overflows to infinity within a
    few thousand bars at any ordinary length, taking most of the output with it
    and saying nothing: at ``length`` 20 over 100k bars the one-line version
    returns 7,045 finite values out of 99,981. So the sum is taken in blocks short
    enough that the largest weight stays well inside the float range, each seeded
    from the last value of the block before it. That is the whole reason this is
    not a one-liner, and the block size has to come from ``alpha`` rather than
    being picked.
    """
    keep = 1.0 - alpha
    if keep <= 0.0:
        # alpha of exactly 1 keeps nothing, so every value is its own average
        return values * alpha

    block   = max(1, min(values.size, _block(keep)))
    out     = np.empty(values.size, dtype=np.float64)
    carried = seed

    for start in range(0, values.size, block):
        chunk  = values[start:start + block]
        powers = keep ** np.arange(1.0, chunk.size + 1.0)
        # sum_{j <= k} alpha * values[j] * keep**(k - j), by dividing the weights
        # out of a cumulative sum and multiplying them back in
        out[start:start + chunk.size] = (
            np.cumsum(alpha * chunk / powers) * powers + carried * powers
        )
        carried = out[start + chunk.size - 1]

    return out


def sma(values, length, axis=0):
    """The simple moving average over ``length`` bars; ``values`` is ``(T,)`` or
    ``(T, N)`` along ``axis``, and the average has its shape."""
    length = _length(length, "length")

    def column(values):
        out = _blank(values.size)
        out[length - 1:] = _windows(values, length).mean(axis=-1)
        return out

    return _over(column, {"values": values}, axis)


def ema(values, length, axis=0):
    """The exponential moving average over ``length`` bars, of the shape of
    ``values``.

    Smoothed at ``2 / (length + 1)`` and seeded from the simple average of the
    first ``length`` values, which is TradingView's convention. Seeding from the
    first value instead, as some implementations do, lets one opening print steer
    the curve for hundreds of bars.
    """
    length = _length(length, "length")
    return _over(lambda values: _smooth(values, 2.0 / (length + 1.0), length),
                 {"values": values}, axis)


def wma(values, length, axis=0):
    """The linearly weighted moving average, of the shape of ``values``: the newest
    bar counts ``length`` times as much as the oldest.
    """
    length = _length(length, "length")

    def column(values):
        out = _blank(values.size)
        if values.size < length:
            # before its weights, which a length no series reaches would size
            return out

        weights = np.arange(1.0, length + 1.0)
        windows = _windows(values, length)
        summed  = np.zeros(len(windows))
        # one weight at a time, in order, which is how numpy's matmul sums many
        # windows; it hands a single one to BLAS, which sums in another order, so a
        # series of one window gave its bar other bits than a longer series gives there
        for k, weight in enumerate(weights / weights.sum()):
            summed += windows[:, k] * weight

        out[length - 1:] = summed
        return out

    return _over(column, {"values": values}, axis)


def vwap(high, low, close, volume, length, axis=0):
    """The volume-weighted average price over a rolling window of ``length`` bars,
    of the shape of the four series, which share one.

    Rolling, not anchored to a session, because the engine has bars and no notion
    of a trading day. Priced on the typical price ``(high + low + close) / 3``.
    A window whose volume is zero has no weighted price, so it is NaN.
    """
    length = _length(length, "length")

    def column(high, low, close, volume):
        typical = (high + low + close) / 3.0
        paid    = _windows(typical * volume, length).sum(axis=-1)
        traded  = _windows(volume, length).sum(axis=-1)
        out     = _blank(close.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[length - 1:] = np.where(traded > 0.0, paid / traded, np.nan)

        return out

    return _over(column, {"high": high, "low": low, "close": close, "volume": volume},
                 axis)


def rsi(values, length=14, axis=0):
    """The relative strength index over ``length`` bars, 0 to 100, of the shape of
    ``values``.

    Wilder's smoothing, ``1 / length`` rather than ``2 / (length + 1)``, which is
    what TradingView's ``ta.rsi`` uses and what the indicator was defined with.

    Rising with no losses at all is 100. Going **nowhere**, with neither gains nor
    losses, is NaN: there is no ratio of strength to weakness when both are zero,
    and the alternative conventions each pick a side of a coin that was never
    tossed. Reporting 100 for a dead flat market, as some do, is a live trap for the
    rule this exists to be traded on, since ``rsi > 70`` then fires on a market
    that has not moved at all.
    """
    length = _length(length, "length")

    def column(values):
        change = np.diff(values, prepend=np.nan)
        gain   = _smooth(np.maximum(change, 0.0), 1.0 / length, length)
        loss   = _smooth(-np.minimum(change, 0.0), 1.0 / length, length)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(loss > 0.0, 100.0 - 100.0 / (1.0 + gain / loss), 100.0)

        moved = (gain > 0.0) | (loss > 0.0)
        return np.where(np.isfinite(gain) & np.isfinite(loss) & moved, out, np.nan)

    return _over(column, {"values": values}, axis)


def macd(values, fast=12, slow=26, signal=9, axis=0):
    """Moving average convergence divergence, as ``.line``, ``.signal`` and
    ``.histogram``, each of the shape of ``values``.

    The line is the fast exponential average less the slow one; the signal is an
    exponential average of the line; the histogram is the gap. Each has its own
    warm-up, and the signal warms up after the line does.
    """
    fast, slow = _length(fast, "fast"), _length(slow, "slow")
    signal = _length(signal, "signal")
    if not fast < slow:
        raise ProbatumValueError(f"slow: {slow}, not above fast's {fast}; give fast "
                                 f"below slow")

    def column(values):
        line     = ema(values, fast) - ema(values, slow)
        smoothed = ema(line, signal)
        return line, smoothed, line - smoothed

    return Macd(*_over(column, {"values": values}, axis, lines=3))


def stoch(high, low, close, length=14, smooth=3, axis=0):
    """The stochastic oscillator, as ``.k`` and ``.d``, both 0 to 100 and of the
    shape of the three series.

    ``k`` is where the close sits inside the last ``length`` bars' range, and
    ``d`` is its simple average over ``smooth`` bars, over the bars ``k`` has a
    value on. A window with no range at all has no position inside it, so ``k`` is
    NaN there rather than a fifty, and ``d`` averages across it.

    Both run 0 to 100 for well-formed bars. A close outside its own bar's high
    and low is not well-formed, and produces a value outside that range rather
    than being clamped, because a clamp would hide the only evidence that the
    data is wrong.
    """
    length, smooth = _length(length, "length"), _length(smooth, "smooth")

    def column(high, low, close):
        top    = _blank(close.size)
        bottom = _blank(close.size)
        top[length - 1:]    = _windows(high, length).max(axis=-1)
        bottom[length - 1:] = _windows(low, length).min(axis=-1)

        span = top - bottom
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(span > 0.0, (close - bottom) / span * 100.0, np.nan)

        return k, sma(k, smooth)

    return Stochastic(*_over(column, {"high": high, "low": low, "close": close}, axis,
                             lines=2))


def roc(values, length, axis=0):
    """The rate of change over ``length`` bars, in percent, of the shape of
    ``values``."""
    length = _length(length, "length")

    def column(values):
        out  = _blank(values.size)
        past = values[:-length]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[length:] = np.where(past != 0.0, (values[length:] / past - 1.0) * 100.0,
                                    np.nan)

        return out

    return _over(column, {"values": values}, axis)


def true_range(high, low, close, axis=0):
    """The true range of each bar, of the shape of the three series: its own range,
    or the gap from the previous close if that is wider.

    The first bar has no previous close, so it is its own high less its low, and a
    bar after one the series does not have takes the close of the last bar before
    it, so a gap through a closure widens the range by what it moved.
    """
    def column(high, low, close):
        previous = np.roll(close, 1)
        previous[0] = np.nan

        spans = np.vstack([high - low, np.abs(high - previous),
                           np.abs(low - previous)])
        out = spans.max(axis=0)

        # the first has no previous close by construction rather than by accident,
        # and so is its own range
        out[0] = high[0] - low[0]
        return out

    return _over(column, {"high": high, "low": low, "close": close}, axis)


def atr(high, low, close, length=14, axis=0):
    """The average true range over ``length`` bars, of the shape of the three
    series.

    Wilder's smoothing of the true range, ``1 / length``, matching ``rsi`` and
    TradingView's ``ta.atr``.
    """
    length = _length(length, "length")

    def column(high, low, close):
        return _smooth(true_range(high, low, close), 1.0 / length, length)

    return _over(column, {"high": high, "low": low, "close": close}, axis)


def stdev(values, length, axis=0):
    """The rolling standard deviation over ``length`` bars, of the shape of
    ``values``.

    Population, dividing by ``length`` rather than ``length - 1``, because that is
    what ``bbands`` is defined against and a band that disagrees with every chart
    by a factor of ``sqrt(n / (n - 1))`` is worse than no band.
    """
    length = _length(length, "length")

    def column(values):
        out = _blank(values.size)
        out[length - 1:] = _windows(values, length).std(axis=-1)
        return out

    return _over(column, {"values": values}, axis)


def bbands(values, length=20, deviations=2.0, axis=0):
    """Bollinger bands, as ``.upper``, ``.middle`` and ``.lower``, each of the shape
    of ``values``.

    A simple moving average, with a band ``deviations`` population standard
    deviations either side of it.
    """
    length     = _length(length, "length")
    deviations = _width(deviations, "deviations")

    def column(values):
        middle = sma(values, length)
        spread = stdev(values, length) * deviations
        return middle + spread, middle, middle - spread

    return Bands(*_over(column, {"values": values}, axis, lines=3))


def donchian(high, low, length=20, axis=0):
    """The Donchian channel, as ``.upper``, ``.middle`` and ``.lower``, each of the
    shape of the two series: the highest high and the lowest low of the last
    ``length`` bars, and their midpoint.

    Both edges include the current bar, so a breakout of ``upper`` cannot happen
    on the bar that set it. Compare against the previous bar's edge if that is
    what you mean.
    """
    length = _length(length, "length")

    def column(high, low):
        top    = _blank(high.size)
        bottom = _blank(low.size)
        top[length - 1:]    = _windows(high, length).max(axis=-1)
        bottom[length - 1:] = _windows(low, length).min(axis=-1)
        return top, (top + bottom) / 2.0, bottom

    return Bands(*_over(column, {"high": high, "low": low}, axis, lines=3))


def zscore(values, length, axis=0):
    """How many rolling standard deviations the current value sits from its own
    rolling mean, of the shape of ``values``. A window with no deviation at all has
    no position, so it is NaN, not a zero.
    """
    length = _length(length, "length")

    def column(values):
        spread = stdev(values, length)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(spread > 0.0, (values - sma(values, length)) / spread,
                            np.nan)

    return _over(column, {"values": values}, axis)


def _lag(bars):
    """``bars``, a lag, as a whole number of bars from 0."""
    if _number(bars) and bars < 0:
        raise ProbatumValueError(f"bars: {bars!r}, which would read the future; give a "
                                 f"whole number of bars from 0")
    return _length(bars, "bars", least=0)


def shift(values, bars=1, axis=0):
    """``values`` moved ``bars`` of its own bars into the future, of its shape, so a
    column's ``i``-th value is carried to its ``i + bars``-th; the first ``bars``
    are the warm-up.

    The safe way to look back. Written by hand, ``values[i - bars]`` on an early
    bar is a NEGATIVE index, which numpy resolves from the END of the array, so
    the rule reads a price from the far future, produces a spectacular result and
    raises nothing at all; ``Strategy.warmup`` closes that only when the declared
    warm-up is at least as deep as the lookback, and the natural error is declaring
    the indicator's warm-up and then reaching further back than it inside ``next``.
    It also reads rows, and on a universe the row before an instrument's bar can be
    one it printed nothing on. Computed once here, the array is padded rather than
    wrapped and lags each instrument's own bars, and there is no index to get wrong
    (ADR 0037). A series whose NaN means "hold" is lagged with numpy, since every
    NaN here is a row the series does not have.

    A negative ``bars`` is refused rather than read as a lead: shifting the other
    way is lookahead by construction, and this module has no honest use for it.
    """
    bars = _lag(bars)

    def column(values):
        if bars == 0:
            return values.copy()

        out = _blank(values.size)
        if bars < values.size:
            out[bars:] = values[:-bars]

        return out

    return _over(column, {"values": values}, axis)


def change(values, bars=1, axis=0):
    """How much ``values`` moved over its last ``bars`` bars, in its own units, of
    its shape."""
    bars = _lag(bars)
    return _over(lambda values: values - shift(values, bars), {"values": values}, axis)


def highest(values, length, axis=0):
    """The highest value of the last ``length`` bars, the current one included, of
    the shape of ``values``."""
    length = _length(length, "length")

    def column(values):
        out = _blank(values.size)
        out[length - 1:] = _windows(values, length).max(axis=-1)
        return out

    return _over(column, {"values": values}, axis)


def lowest(values, length, axis=0):
    """The lowest value of the last ``length`` bars, the current one included, of
    the shape of ``values``."""
    length = _length(length, "length")

    def column(values):
        out = _blank(values.size)
        out[length - 1:] = _windows(values, length).min(axis=-1)
        return out

    return _over(column, {"values": values}, axis)


def _level(other):
    """``other``, a crossing's second argument, as a level, or None where it is a
    series."""
    if np.ndim(other) > 0:
        return None

    if isinstance(other, np.ndarray):
        other = other.item()

    if not _number(other):
        raise ProbatumTypeError(f"other: {other!r}; give a level, or a series of the "
                                f"values' shape")

    level = float(other)
    if not math.isfinite(level):
        raise ProbatumValueError(f"other: {level}; give a finite level, or a series")

    return level


def _crossing(values, other, above, axis):
    """Where ``values`` crosses ``other``, above it or below it: compared on the rows
    both have, each against the last before it."""
    level = _level(other)

    def column(values, other):
        out = np.zeros(values.size, dtype=bool)
        if above:
            out[1:] = (values[1:] > other[1:]) & (values[:-1] <= other[:-1])
        else:
            out[1:] = (values[1:] < other[1:]) & (values[:-1] >= other[:-1])

        return out

    if level is None:
        return _over(column, {"values": values, "other": other}, axis, flag=True)

    return _over(lambda values: column(values, np.full(values.size, level)),
                 {"values": values}, axis, flag=True)


def crossover(values, other, axis=0):
    """``True`` on each bar where ``values`` crosses above ``other``: at or below
    it on the bar before, above it on this one, of the shape of ``values``.

    ``other`` is a second series of the same shape or a plain number, since crossing
    a level is the commoner case. Each pair is compared on the bars both have, so a
    cross across a bar one of them does not have is reported on the first after it.
    Returns **bool**, not float, and a bar whose comparison cannot be made, the very
    first or one either of them does not have, is ``False``. That is the one
    deliberate exception to this module's NaN warm-up: ``bool(nan)`` is ``True`` in
    Python, so a float warm-up would fire the rule on exactly the bars where nothing
    is known.
    """
    return _crossing(values, other, True, axis)


def crossunder(values, other, axis=0):
    """``True`` on each bar where ``values`` crosses below ``other``. The mirror
    of ``crossover``, and boolean for the same reason.
    """
    return _crossing(values, other, False, axis)


def hma(values, length, axis=0):
    """Hull's moving average, of the shape of ``values``: two weighted averages
    differenced and smoothed again, which turns far faster than a simple average of
    the same length for the same amount of noise.

    ``wma(2 * wma(values, length / 2) - wma(values, length), sqrt(length))``, with
    both inner lengths floored to whole bars, which is the published definition.
    """
    length = _length(length, "length")
    half   = max(1, length // 2)
    root   = max(1, int(math.sqrt(length)))

    def column(values):
        raw = 2.0 * wma(values, half) - wma(values, length)
        return wma(raw, root)

    return _over(column, {"values": values}, axis)


def vwma(values, volume, length, axis=0):
    """The volume-weighted moving average of ``values`` over ``length`` bars, of the
    shape of the two series.

    ``sma`` treats a bar nobody traded as equal to one everybody did. A window
    whose volume is zero has no weighted average, so it is NaN.
    """
    length = _length(length, "length")

    def column(values, volume):
        paid   = _windows(values * volume, length).sum(axis=-1)
        traded = _windows(volume, length).sum(axis=-1)
        out    = _blank(values.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[length - 1:] = np.where(traded > 0.0, paid / traded, np.nan)

        return out

    return _over(column, {"values": values, "volume": volume}, axis)


def willr(high, low, close, length=14, axis=0):
    """Williams %R over ``length`` bars, -100 to 0, of the shape of the three
    series.

    Where the close sits inside the recent range, read from the top: 0 is the
    high of the window and -100 the low. The same measurement ``stoch`` makes and
    the opposite sign convention, which is why both exist rather than one. A
    window with no range at all has no position inside it, so it is NaN.
    """
    length = _length(length, "length")

    def column(high, low, close):
        top, bottom = highest(high, length), lowest(low, length)
        span = top - bottom
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(span > 0.0, (close - top) / span * 100.0, np.nan)

    return _over(column, {"high": high, "low": low, "close": close}, axis)


def cci(high, low, close, length=20, axis=0):
    """The commodity channel index over ``length`` bars, of the shape of the three
    series.

    How far the typical price ``(high + low + close) / 3`` sits from its own
    average, in units of its mean absolute deviation, scaled by the conventional
    0.015 so that most readings land inside -100 to 100. Mean ABSOLUTE deviation,
    not standard deviation, which is the definition and is what makes it react
    differently from ``zscore`` on a fat-tailed series. A window with no deviation
    at all is NaN.
    """
    length = _length(length, "length")

    def column(high, low, close):
        typical = (high + low + close) / 3.0
        frames  = _windows(typical, length)
        with np.errstate(invalid="ignore"):
            spread = np.abs(frames - frames.mean(axis=-1, keepdims=True)).mean(axis=-1)

        deviation = _blank(close.size)
        deviation[length - 1:] = spread

        middle = sma(typical, length)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(deviation > 0.0,
                            (typical - middle) / (0.015 * deviation), np.nan)

    return _over(column, {"high": high, "low": low, "close": close}, axis)


def natr(high, low, close, length=14, axis=0):
    """The average true range as a percent of the close, of the shape of the three
    series.

    The number to size a stop with across instruments and across time: an ATR of
    400 means one thing on a coin at 90,000 and another on one at 800, and the
    same strategy run over two years of a trending market is comparing two
    different sizes without this.
    """
    length = _length(length, "length")

    def column(high, low, close):
        ranges = atr(high, low, close, length)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(close > 0.0, ranges / close * 100.0, np.nan)

    return _over(column, {"high": high, "low": low, "close": close}, axis)


def keltner(high, low, close, length=20, atr_length=10, multiplier=2.0, axis=0):
    """Keltner channels, as ``.upper``, ``.middle`` and ``.lower``, each of the shape
    of the three series.

    An exponential average with bands ``multiplier`` average true ranges either
    side. Where ``bbands`` measures the spread of the closes, this measures the
    spread of the BARS, so a session of wide ranges closing in one place widens
    this and not that. It is also the band an ATR-sized stop belongs on, which is
    the question ``atr`` and a round trip's ``worst`` excursion both point at and
    neither could answer.
    """
    length, atr_length = _length(length, "length"), _length(atr_length, "atr_length")
    multiplier = _width(multiplier, "multiplier")

    def column(high, low, close):
        middle = ema(close, length)
        spread = atr(high, low, close, atr_length) * multiplier
        return middle + spread, middle, middle - spread

    return Bands(*_over(column, {"high": high, "low": low, "close": close}, axis,
                        lines=3))


def supertrend(high, low, close, length=10, multiplier=3.0, axis=0):
    """The supertrend, as ``.upper``, ``.lower``, ``.middle`` (the line in force)
    and ``.direction`` (``1.0`` while long, ``-1.0`` while short), each of the shape
    of the three series.

    An ATR band that ratchets: each edge may only tighten toward price while the
    trend holds, and the line flips when the close closes through the far one.
    Read ``direction`` rather than comparing the close against ``middle``, since
    the line is the edge in force and the comparison is what set it.

    The ratchet starts on the first bar with a band, once the average true range
    has warmed up, and carries across a bar the series does not have, as every
    smoother here does.
    """
    length     = _length(length, "length")
    multiplier = _width(multiplier, "multiplier", above=True)

    def column(high, low, close):
        ranges       = atr(high, low, close, length)
        middle_price = (high + low) / 2.0
        raw_upper    = middle_price + multiplier * ranges
        raw_lower    = middle_price - multiplier * ranges
        size         = close.size

        upper, lower, line = _blank(size), _blank(size), _blank(size)
        direction = _blank(size)
        running   = False

        for i in range(size):
            if not (np.isfinite(raw_upper[i]) and np.isfinite(raw_lower[i])
                    and np.isfinite(close[i])):
                running = False
                continue

            if not running:
                upper[i], lower[i], direction[i] = raw_upper[i], raw_lower[i], 1.0
                running = True
            else:
                upper[i] = (raw_upper[i] if raw_upper[i] < upper[i - 1]
                            or close[i - 1] > upper[i - 1] else upper[i - 1])
                lower[i] = (raw_lower[i] if raw_lower[i] > lower[i - 1]
                            or close[i - 1] < lower[i - 1] else lower[i - 1])
                if direction[i - 1] > 0.0:
                    direction[i] = -1.0 if close[i] < lower[i] else 1.0
                else:
                    direction[i] = 1.0 if close[i] > upper[i] else -1.0

            line[i] = lower[i] if direction[i] > 0.0 else upper[i]

        return upper, line, lower, direction

    return Channel(*_over(column, {"high": high, "low": low, "close": close}, axis,
                          lines=4))


def adx(high, low, close, length=14, axis=0):
    """The average directional index, as ``.adx``, ``.plus`` and ``.minus``, each of
    the shape of the three series.

    How strongly the market is trending, without saying which way: ``plus`` and
    ``minus`` are the two directional strengths and ``adx`` is how far apart they
    are, smoothed. Above 20 to 25 is the usual reading for "there is a trend
    here", and it is the filter a moving-average cross is conventionally paired
    with, since a cross in a range is the most reliable way to lose money slowly.

    Wilder's smoothing throughout, matching ``rsi`` and ``atr``. Bars that open a
    series with no directional movement either way have no ``dx``, so the ADX warms
    up after them.
    """
    length = _length(length, "length")

    def column(high, low, close):
        up         = high - shift(high, 1)
        down       = shift(low, 1) - low
        plus_move  = np.where((up > down) & (up > 0.0), up, 0.0)
        minus_move = np.where((down > up) & (down > 0.0), down, 0.0)

        # a bar with no previous is not a zero move, it is an unknown one
        missing        = ~(np.isfinite(up) & np.isfinite(down))
        plus_move      = np.where(missing, np.nan, plus_move)
        minus_move     = np.where(missing, np.nan, minus_move)
        alpha          = 1.0 / length
        ranges         = _smooth(true_range(high, low, close), alpha, length)
        plus_smoothed  = _smooth(plus_move, alpha, length)
        minus_smoothed = _smooth(minus_move, alpha, length)

        with np.errstate(divide="ignore", invalid="ignore"):
            plus  = np.where(ranges > 0.0, plus_smoothed / ranges * 100.0, np.nan)
            minus = np.where(ranges > 0.0, minus_smoothed / ranges * 100.0, np.nan)
            total = plus + minus
            dx    = np.where(total > 0.0, np.abs(plus - minus) / total * 100.0, np.nan)

        return _smooth(dx, alpha, length), plus, minus

    return Trend(*_over(column, {"high": high, "low": low, "close": close}, axis,
                        lines=3))


def obv(close, volume, axis=0):
    """On-balance volume, of the shape of the two series: the running total of each
    bar's volume, signed by the direction the close moved.

    The one thing in this module that reads participation rather than price. A
    price making new highs on falling on-balance volume is the divergence it
    exists to show.

    The total carries across a bar the series does not have, on the same rule the
    exponential averages use, from the close of the last bar before it.
    """
    def column(close, volume):
        out = _blank(close.size)

        # bar zero has no previous close at all, which is a warm-up rather than a
        # zero, and the total starts flat after it
        running = 0.0
        for i in range(1, close.size):
            if close[i] > close[i - 1]:
                running += volume[i]
            elif close[i] < close[i - 1]:
                running -= volume[i]
            out[i] = running

        return out

    return _over(column, {"close": close, "volume": volume}, axis)


def mfi(high, low, close, volume, length=14, axis=0):
    """The money flow index over ``length`` bars, 0 to 100, of the shape of the four
    series.

    ``rsi`` weighted by what actually traded: the same up-against-down ratio, on
    the typical price times the volume rather than on the price change alone. A
    move on no volume moves this hardly at all, which is the whole point of
    reading it beside ``rsi``.

    A window with flows in one direction only is 100 or 0. A window with no flow
    at all is NaN, for the reason ``rsi`` gives: there is no ratio of strength to
    weakness when both are zero.
    """
    length = _length(length, "length")

    def column(high, low, close, volume):
        typical  = (high + low + close) / 3.0
        flow     = typical * volume
        rising   = typical > shift(typical, 1)
        falling  = typical < shift(typical, 1)
        known    = np.isfinite(flow) & np.isfinite(shift(typical, 1))
        positive = np.where(known, np.where(rising, flow, 0.0), np.nan)
        negative = np.where(known, np.where(falling, flow, 0.0), np.nan)
        up       = _blank(close.size)
        down     = _blank(close.size)
        up[length - 1:]   = _windows(positive, length).sum(axis=-1)
        down[length - 1:] = _windows(negative, length).sum(axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(down > 0.0, 100.0 - 100.0 / (1.0 + up / down), 100.0)

        moved = (up > 0.0) | (down > 0.0)
        return np.where(np.isfinite(up) & np.isfinite(down) & moved, out, np.nan)

    return _over(column, {"high": high, "low": low, "close": close, "volume": volume},
                 axis)
