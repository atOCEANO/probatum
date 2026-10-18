"""A Data, the bars a run reads: N instruments over T bars on one clock (ADR 0004,
ADR 0005), and the helpers the older entry points still turn their candles into the
engine's arrays with.

A Data copies its bars once into a buffer it owns, bar-major, so one bar across every
instrument is contiguous for the step, and every field and slice is a read-only view
of it. A slice keeps the whole buffer, so the bars before it are there for features
and an agent's window to warm up on. Frames are read through the Arrow C interface in
the binding, and a pandas frame by its own methods where pyarrow is missing or cannot
convert one of its columns, so no frame library is imported to read one (ADR 0044).
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import datetime
import hashlib
import inspect
import json
import os
import pathlib
import re
import time
import warnings
import zoneinfo

import numpy as np

from . import _ipc
from ._probatum import arrow_columns, periods_per_year

_OHLCV = ["open", "high", "low", "close", "volume"]


def to_ohlcv(data):
    """Return a ``(T, 5)`` float64 OHLCV array from a numpy array, a pandas
    DataFrame with the columns open/high/low/close/volume, or a path to a parquet
    file.

    **Volume is in the units an order's size is in**, base units or contracts on an
    inverse perpetual, because the liquidity budget is a share of it and the
    market-impact term divides a fill's size by it. A feed shipping quote volume
    instead inflates the budget by roughly the price, so ``budget`` stops binding
    and impact goes to zero: no error, no warning, just a better backtest than the
    market would have given. The router's frames carry base ``volume`` and a
    separate ``volume_usd``, and only the first is read.
    """
    if isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(
                f"data arrived with shape {arr.shape}; an array must be (T, 5) "
                f"OHLCV, one row per bar"
            )
        return arr

    frame   = _as_frame(data)
    missing = [c for c in _OHLCV if c not in frame.columns]
    if missing:
        raise ValueError(f"data is missing columns {missing}; need {_OHLCV}")

    _validate_index(frame)

    return np.ascontiguousarray(frame[_OHLCV].to_numpy(dtype=np.float64))


def _validate_index(frame):
    # the spec's input contract: a real (datetime) index must be sorted ascending
    # and unique. A default RangeIndex carries no timestamps, so it is exempt
    import pandas as pd
    idx = frame.index
    if isinstance(idx, pd.RangeIndex):
        return

    if not idx.is_monotonic_increasing:
        stamps = idx.to_numpy()
        at     = int(np.argmax(stamps[1:] <= stamps[:-1])) + 1
        raise ValueError(
            f"data index falls back at row {at}, from {stamps[at - 1]} to "
            f"{stamps[at]}; sort the frame by its index before passing it"
        )

    if not idx.is_unique:
        repeats = idx[idx.duplicated()]
        raise ValueError(
            f"data index repeats {len(repeats)} of its {len(idx)} stamps, the "
            f"first being {repeats[0]}; one row per bar, so drop or aggregate them"
        )


def to_float2d(data):
    """Return a ``(T, F)`` float64 array from a numpy array or a pandas DataFrame."""
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float64)

    frame = _as_frame(data)
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float64))


def index_of(data):
    """Return the row index of a DataFrame or parquet input as a numpy array, for
    stamping trades with times. Returns None for a numpy array, or when the frame
    carries only a default integer range index (no real timestamps to attach).
    """
    if isinstance(data, np.ndarray):
        return None

    import pandas as pd
    frame = _as_frame(data)
    if isinstance(frame.index, pd.RangeIndex):
        return None

    return frame.index.to_numpy()


def prepare(data):
    """Return ``(ohlcv_array, index_or_None)`` from one read of ``data``.

    ``to_ohlcv`` and ``index_of`` each route a parquet path through ``_as_frame``, so
    calling both decoded the file twice, doubling the load time and peak memory of
    the most convenient entry point and leaving room for the two reads to disagree
    if the file changed between them.
    """
    if isinstance(data, np.ndarray):
        return to_ohlcv(data), None

    frame = _as_frame(data)
    return to_ohlcv(frame), index_of(frame)


def _datetimes(index):
    # the index as UTC datetime64, or None when it is not a clock. A tz-AWARE
    # DatetimeIndex hands back object dtype holding Timestamps rather than
    # datetime64, and a tz-aware index is what an exchange feed produces, so that
    # is the common path here and not the odd one
    if index is None:
        return None

    stamps = np.asarray(index)
    if stamps.dtype.kind == "O":
        try:
            stamps = _pandas().DatetimeIndex(stamps)
        except (TypeError, ValueError, ImportError):
            return None
        if stamps.tz is not None:
            stamps = stamps.tz_convert("UTC").tz_localize(None)
        stamps = stamps.to_numpy()

    if stamps.dtype.kind != "M":
        return None
    return stamps


def _epoch_seconds(index):
    stamps = _datetimes(index)
    if stamps is None:
        return None
    return stamps.astype("datetime64[s]").astype("int64").astype(np.float64)


def epoch_nanoseconds(index):
    """Return ``index`` as UTC epoch nanoseconds, the engine's clock, or ``None``
    when it is not a clock."""
    stamps = _datetimes(index)
    if stamps is None:
        return None
    return np.ascontiguousarray(stamps.astype("datetime64[ns]").astype(np.int64))


def bars_per_year(index):
    """Return the bars a year the clock of ``index`` holds, or ``None``.

    ``None`` when there is no datetime index to read, which is every numpy input
    and every frame carrying only a default range index, and for a single bar.

    The rows less one, per 365 days of their span (ADR 0004): a market that never
    closes gets its interval's count, 8760 for hourly bars, and one that closes or
    misses bars gets the bars it actually has, a stock's daily bars about 252.
    """
    times = epoch_nanoseconds(index)
    if times is None:
        return None
    return periods_per_year(times)


def annualization(given, index):
    """Resolve ``periods_per_year``: what the caller asked for, else the bars a year
    the candles' own clock holds, else a warned fallback of 365.

    The fallback used to be the default, silently. On the hourly candles this
    library is mostly pointed at that understates sharpe by a factor of about
    4.9 and cagr by 25, because the exponent reads a year of bars as 24 years,
    and with a non-zero ``risk_free`` it can report a negative sharpe for a
    profitable strategy.
    """
    if given is not None:
        return float(given)

    inferred = bars_per_year(index)
    if inferred is not None:
        return inferred

    warnings.warn(
        "periods_per_year could not be read from the candles, because they carry "
        "no datetime index, so sharpe, sortino, volatility, cagr and calmar are "
        "annualized as though each bar were a day. Pass periods_per_year= (8760 "
        "for hourly bars, 105_120 for five-minute) or a frame with a "
        "DatetimeIndex",
        stacklevel=3,
    )
    return 365.0


def _as_frame(data):
    # the type check comes before the import: on a pandas-free install an
    # unsupported argument used to report "pandas is required", sending the caller
    # after a dependency that would not have helped
    if isinstance(data, str) or hasattr(data, "__fspath__"):
        return _pandas().read_parquet(data)

    if type(data).__module__.split(".")[0] == "pandas":
        frame = _pandas().DataFrame
        if isinstance(data, frame):
            return data

    raise TypeError(
        f"unsupported data type {type(data).__name__}; "
        "pass a numpy array, a pandas DataFrame, or a parquet path"
    )


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for DataFrame or parquet input; pip install pandas pyarrow"
        ) from exc

    return pd


SECOND    = 1_000_000_000
HOUR      = 3_600 * SECOND
DAY       = 86_400 * SECOND
INTERVALS = {
    "1m":  60 * SECOND,
    "3m":  180 * SECOND,
    "5m":  300 * SECOND,
    "15m": 900 * SECOND,
    "30m": 1_800 * SECOND,
    "1h":  3_600 * SECOND,
    "2h":  7_200 * SECOND,
    "4h":  14_400 * SECOND,
    "6h":  21_600 * SECOND,
    "8h":  28_800 * SECOND,
    "12h": 43_200 * SECOND,
    "1d":  DAY,
    "3d":  3 * DAY,
    "1w":  7 * DAY,
}
MARKETS = ("spot", "linear", "inverse")
# what an Arrow format a frame cannot give bars in holds, for a refusal to name
ARROW = {
    "u":  "text",
    "U":  "text",
    "vu": "text",
    "b":  "booleans",
    "z":  "bytes",
    "Z":  "bytes",
    "vz": "bytes",
    "n":  "nulls",
    "g":  "floats",
    "f":  "floats",
    "e":  "floats",
    "l":  "integers",
    "i":  "integers",
    "s":  "integers",
    "c":  "integers",
    "L":  "integers",
    "I":  "integers",
    "S":  "integers",
    "C":  "integers",
}
FUNDING = np.dtype([("time", np.int64), ("instrument", np.int64), ("rate", np.float64)])
# an instrument's fields as instruments.arrow holds them
FIELDS = {
    "symbol":       "text",
    "market":       "text",
    "venue":        "text",
    "quote":        "text",
    "contract":     "float",
    "base":         "text",
    "min_notional": "float",
    "max_qty":      "float",
}

Gap         = collections.namedtuple("Gap", "instrument after before missing")
Gap.__doc__ = """A place an instrument's bars jump by more than the interval: its bar
before the jump opened at ``after``, its next at ``before``, and ``missing`` bars of the
interval fit between them."""

# one instrument's bars as a source gave them, before they are placed on the clock
Part = collections.namedtuple("Part", "times bars instrument attrs zone source")


@dataclasses.dataclass(frozen=True)
class Instrument:
    """One instrument of a Data, keyed by its venue, its market and its symbol, with
    the currency its price is quoted in, for an inverse contract the dollars one
    contract is worth, and from the router's symbol information its base asset, the
    smallest notional the venue takes in an order, in the quote currency, and the
    largest size it takes in a limit order, in base units, or contracts on an inverse
    contract. A field the source does not say is None, and a market left None is the
    venue's to give when the Data runs."""

    symbol: str | None = None
    market: str | None = None
    venue: str | None = None
    quote: str | None = None
    contract: float | None = None
    base: str | None = None
    min_notional: float | None = None
    max_qty: float | None = None


    @property
    def settlement(self):
        """The asset its account settles in: its base asset for an inverse contract,
        which is settled in its coin, its quote on any other market, and None where
        its market is the venue's to give."""
        if self.market is None:
            return None
        return self.base if self.market == "inverse" else self.quote


class _Root:
    """The arrays a Data and every view of it share."""


    def __init__(self, time, bars, instruments, interval):
        self.time        = _frozen(time)
        self.bars        = _frozen(bars)
        self.available   = _frozen(~np.isnan(bars[:, :, 3]))
        self.instruments = tuple(instruments)
        self.names       = _display(self.instruments)
        self.interval    = interval
        self.step        = INTERVALS[interval]
        self.funding     = _frozen(np.zeros(0, dtype=FUNDING))
        self.provenance  = {}
        self.rates       = None


    def __setstate__(self, state):
        # an array comes back from a pickle writable, and the bars must not be
        self.__dict__.update(state)

        for name in ("time", "bars", "available", "funding", "rates"):
            if state.get(name) is not None:
                _frozen(state[name])


class Data:
    """N instruments over T bars on one clock, from a frame, a dict of frames keyed
    by symbol, or arrays with ``time=``, each bar's open time.

    A frame is anything that exports the Arrow C stream or array interface (polars,
    pyarrow, DuckDB, pandas with pyarrow), or a pandas frame, with ``open``, ``high``,
    ``low``, ``close`` and ``volume`` columns and the bars' open times in a datetime
    index or column, or in a ``timestamp`` column of milliseconds since 1970. An array
    is ``(T, 5)`` for one instrument or ``(T, N, 5)`` for N, NaN in all five values
    where an instrument printed no bar, and ``time`` is datetimes or int64
    nanoseconds since 1970. ``symbol`` names a single frame's instrument, or each of
    an array's N, an instrument left unnamed taking its position as its name;
    ``market`` is ``"spot"``, ``"linear"`` or ``"inverse"``, the venue's to give where
    it is left out; and ``interval`` is read from a router frame or from the median
    gap between bars where it is not given. A router frame names its venue, market,
    symbol and interval itself."""


    def __init__(self, data, symbol=None, market=None, time=None, interval=None):
        if isinstance(data, Data):
            if any(given is not None for given in (symbol, market, time, interval)):
                raise TypeError("data: a Data with settings beside it; build the Data "
                                "with them, or pass it alone")
            self.__dict__.update(data.__dict__)
            return

        if isinstance(data, dict):
            if symbol is not None:
                raise TypeError(f"symbol: {symbol!r} beside a dict of frames, whose "
                                f"keys name its instruments; leave it out")

            parts = []
            for name, source in data.items():
                if not isinstance(name, str):
                    raise TypeError(f"data: a key of {name!r}; key each frame by its "
                                    f"symbol")
                parts += _parts(source, name, market, time)
        else:
            parts = _parts(data, symbol, market, time)

        declared = {part.attrs["interval"] for part in parts
                    if part.attrs.get("interval")}
        if len(declared) > 1:
            raise ValueError(f"data: frames of {', '.join(sorted(declared))} bars; "
                             f"give every instrument bars on one interval")

        if interval is not None and declared - {interval}:
            raise ValueError(f"interval: {interval!r} for frames of {declared.pop()} "
                             f"bars; leave it out, the frames name their own")

        root = _clock(parts, interval or (declared.pop() if declared else None))
        root.provenance = dict(
            source=", ".join(sorted({part.source for part in parts})),
            venue=_venues(root), interval=root.interval,
            warnings=[w for part in parts for w in part.attrs.get("warnings", [])])

        self._root        = root
        self._rows        = (0, len(root.time))
        self._cols        = tuple(range(len(root.instruments)))
        self._fingerprint = None


    @classmethod
    def _of(cls, root, rows, cols):
        view              = object.__new__(cls)
        view._root        = root
        view._rows        = rows
        view._cols        = tuple(cols)
        view._fingerprint = None

        return view


    def __copy__(self):
        return self


    def __deepcopy__(self, memo):
        # a Data never changes, so a copy of one is the one itself; a deep copy of its
        # arrays would be writable under the fingerprint taken of them
        return self


    @classmethod
    def from_router(cls, router, exchange, market, interval, symbols, start, end=None,
                    funding=False):
        """The bars of ``symbols`` on ``exchange``'s ``market`` at ``interval``, from
        the first bar at or after ``start`` to the last at or before ``end`` that has
        closed, fetched through ``router``, an exchange-router ``Router``, and with
        ``funding`` the funding events that charge them. ``start`` and ``end`` are
        times or the text of one, a text end taking in the whole period it names.
        Each instrument takes the router's own name for its symbol, its base asset and
        its order limits from the router's symbol information, which a symbol the
        router does not list now goes without, as a warning names; two symbols the
        router names as one are refused.

        One router call walks back from its newest bar for at most a hundred requests
        to the venue, so the calls are chained back to ``start``. A call that brings
        nothing earlier is asked once more, and where it brings nothing again the
        venue's history is taken to begin there, which the provenance records and a
        warning names, as another names bars that end before ``end``. A symbol the
        router failed on is refused rather than left out, and so is a funding series
        that misses part of its instrument's span: one that begins more than a cycle
        after its first bar, ends more than a cycle before its last bar ends, misses
        events between or holds one with no rate, so a run never charges nothing in
        silence."""
        fetch = getattr(router, "get_candles", None)
        if fetch is None:
            raise TypeError(f"router: {type(router).__name__}, which has no "
                            f"get_candles; pass an exchange-router Router")

        if inspect.iscoroutinefunction(fetch):
            raise TypeError("router: an AsyncRouter, whose calls return coroutines; "
                            "pass Router.local(...) or Router.service(...)")

        if getattr(router, "get_symbol_info", None) is None:
            raise TypeError(f"router: {type(router).__name__}, which has no "
                            f"get_symbol_info; pass an exchange-router Router")

        if market not in MARKETS:
            raise ValueError(f"market: {market!r}; use {', '.join(MARKETS)}")
        if interval not in INTERVALS:
            raise ValueError(f"interval: {interval!r}; use one of "
                             f"{', '.join(INTERVALS)}")

        if isinstance(symbols, str) or not hasattr(symbols, "__iter__"):
            raise TypeError(f"symbols: {symbols!r}; give a list of symbols")

        symbols = list(symbols)
        if not symbols:
            raise ValueError("symbols: none; name at least one symbol")

        twice = [symbol for symbol in symbols if symbols.count(symbol) > 1]
        if twice:
            raise ValueError(f"symbols: {symbols!r} names {twice[0]} twice; name each "
                             f"once")

        if funding and market == "spot":
            raise ValueError("funding: True for the spot market, which charges none; "
                             "leave it out")

        step  = INTERVALS[interval]
        now   = _now()
        first = _period(start, "start")[0]

        # a venue serves the bar still open as though it were done, so the newest bar
        # taken is the last to have closed whatever the end
        last = now - step
        if end is not None:
            stop = _period(end, "end")[1] - 1
            if stop < first:
                raise ValueError(f"end: {end!r}, before the start, {start!r}; give an "
                                 f"end after the start")
            last = min(last, stop)

        if last < first:
            raise ValueError(f"start: {start!r}, where no {interval} bar has closed "
                             f"yet; give an earlier start")

        initial = {}
        if len(symbols) > 1:
            batch = router.candles_many(exchange, market, symbols, interval=interval,
                                        limit=(last - first) // step + 1,
                                        start=last // 1_000_000, verbose=False)

            if batch.failed:
                symbol, error = next(iter(batch.failed.items()))
                raise ValueError(f"symbols: the router failed on {symbol}: "
                                 f"{getattr(error, 'detail', error)}; leave it out, or "
                                 f"fetch it again once the router can serve it")

            initial = batch.data()

        parts, begins, ends, unlisted, notes = [], {}, {}, [], []
        for symbol in symbols:
            def call(upper, pieces, symbol=symbol):
                if symbol in initial:
                    return initial.pop(symbol)
                return router.get_candles(exchange, market, symbol, interval=interval,
                                          limit=(upper - first) // step + 1,
                                          start=upper // 1_000_000, verbose=False)

            pieces, begun = _walk(call, first, last,
                                  lambda frame: _bars_of(frame, symbol, market))
            if not pieces:
                raise ValueError(f"symbols: {symbol} has no {interval} bars on "
                                 f"{exchange} {market} over the span asked; check the "
                                 f"symbol, the market and the span")

            part = pieces[0][2]
            listed, said = _listed(router, exchange, market, symbol)
            if listed is None:
                unlisted.append((symbol, said))
            else:
                part = part._replace(instrument=dataclasses.replace(part.instrument,
                                                                    **listed))

            name = part.instrument.symbol
            if begun is not None and begun > first:
                begins[name] = _when(begun)

            times, bars = _joined(pieces)
            if times[-1] + step <= last:
                ends[name] = _when(times[-1])

            parts.append(part._replace(times=times, bars=bars))
            notes += [w for piece in pieces for w in piece[2].attrs.get("warnings", [])]

        held = [part.instrument.symbol for part in parts]
        same = [typed for typed, name in zip(symbols, held) if held.count(name) > 1]
        if same:
            raise ValueError(f"symbols: {same[0]!r} and {same[1]!r} are one "
                             f"instrument, {held[symbols.index(same[0])]}; name it once")

        root = _clock(parts, interval)

        if funding:
            events, told = _funding(router, exchange, market, root, step)
            root.funding = _frozen(events)
            notes += told

        root.provenance = dict(source="router", venue=exchange, interval=interval,
                               fetched=_when(now), warnings=list(dict.fromkeys(notes)))
        if begins:
            root.provenance["history"] = begins

        data = cls._of(root, (0, len(root.time)), range(len(root.instruments)))

        if begins:
            symbol, when = next(iter(begins.items()))
            more = (f", and those of {_count(len(begins) - 1, 'more instrument')} "
                    f"begin late too" if len(begins) > 1 else "")
            warnings.warn(f"data: {symbol}'s bars begin at {when}, after the "
                          f"start{more}; the venue serves none earlier, see "
                          f"data.provenance[\"history\"]", stacklevel=2)

        if ends:
            symbol, when = next(iter(ends.items()))
            more = (f", and those of {_count(len(ends) - 1, 'more instrument')} end "
                    f"early too" if len(ends) > 1 else "")
            warnings.warn(f"data: {symbol}'s bars end at {when}, before the "
                          f"end{more}; the venue served none later", stacklevel=2)

        if unlisted:
            (symbol, said), more = unlisted[0], len(unlisted) - 1
            others = f", nor {_count(more, 'more instrument')}" if more else ""
            whose  = "their base assets" if more else "its base asset"
            warnings.warn(f"data: the router does not list {symbol} now, saying "
                          f"{said!r}{others}, as for a symbol delisted or paused, so "
                          f"{whose} and order limits are unknown; see data.instruments",
                          stacklevel=2)

        gaps = data.gaps()
        if gaps:
            gap  = gaps[0]
            more = f", one of {len(gaps)} gaps" if len(gaps) > 1 else ""
            warnings.warn(f"data: {gap.instrument} misses {_count(gap.missing, 'bar')} "
                          f"after {_when(gap.after.astype(np.int64))}{more}; see "
                          f"data.gaps()", stacklevel=2)

        return data


    def _take(self, array):
        # a run of instruments in order is a view of the buffer, any other choice a
        # read-only copy
        start, stop = self._rows
        cols = self._cols
        if cols == tuple(range(cols[0], cols[-1] + 1)):
            return array[start:stop, cols[0]:cols[-1] + 1]
        return _frozen(array[start:stop][:, list(cols)])


    @property
    def time(self):
        """Each bar's open time, ``(T,)`` int64 nanoseconds since 1970, UTC."""
        start, stop = self._rows
        return self._root.time[start:stop]


    @property
    def bars(self):
        """``(T, N, 5)``, each instrument's open, high, low, close and volume, NaN in
        all five where it printed no bar."""
        return self._take(self._root.bars)


    @property
    def opens(self):
        """``(T, N)``, NaN where an instrument printed no bar."""
        return self.bars[:, :, 0]


    @property
    def highs(self):
        """``(T, N)``, NaN where an instrument printed no bar."""
        return self.bars[:, :, 1]


    @property
    def lows(self):
        """``(T, N)``, NaN where an instrument printed no bar."""
        return self.bars[:, :, 2]


    @property
    def closes(self):
        """``(T, N)``, NaN where an instrument printed no bar."""
        return self.bars[:, :, 3]


    @property
    def volumes(self):
        """``(T, N)`` in base units, or in contracts for an inverse instrument, NaN
        where an instrument printed no bar."""
        return self.bars[:, :, 4]


    @property
    def available(self):
        """``(T, N)`` bool, whether each instrument printed a bar at each time."""
        return self._take(self._root.available)


    @property
    def open(self):
        """``(T,)``, the open of a Data of one instrument."""
        return self._one("open")


    @property
    def high(self):
        """``(T,)``, the high of a Data of one instrument."""
        return self._one("high")


    @property
    def low(self):
        """``(T,)``, the low of a Data of one instrument."""
        return self._one("low")


    @property
    def close(self):
        """``(T,)``, the close of a Data of one instrument."""
        return self._one("close")


    @property
    def volume(self):
        """``(T,)``, the volume of a Data of one instrument."""
        return self._one("volume")


    def _one(self, field):
        if len(self._cols) != 1:
            raise AttributeError(f"data.{field}: the universe has {len(self._cols)} "
                                 f"instruments; use data.{field}s or "
                                 f"data[\"{self.names[0]}\"].{field}")
        return getattr(self, field + "s")[:, 0]


    @property
    def instruments(self):
        """Each instrument's `Instrument`, in the order of the columns."""
        return tuple(self._root.instruments[i] for i in self._cols)


    @property
    def names(self):
        """Each instrument's short name, by which the Data is indexed: its symbol,
        with its market before it where another market holds the same symbol, and
        its venue before that where another venue does."""
        return tuple(self._root.names[i] for i in self._cols)


    @property
    def interval(self):
        """The clock's interval, by the router's name for it, such as ``"1h"``."""
        return self._root.interval


    @property
    def funding(self):
        """The funding events its bars own, at their real times: a structured array of
        ``time``, ``instrument``, the column the event charges, and ``rate``."""
        events = self._root.funding
        start, stop = self._rows

        owners = _owners(self._root, events["time"])
        kept   = (owners >= start) & (owners < stop) & np.isin(events["instrument"],
                                                               self._cols)

        out   = events[kept].copy()
        place = {i: k for k, i in enumerate(self._cols)}
        out["instrument"] = [place[i] for i in out["instrument"]]
        return _frozen(out)


    @property
    def funding_rate(self):
        """``(T, N)``, the last funding rate known at each bar's end, NaN before the
        first, for features."""
        root = self._root
        if root.rates is None:
            rates  = np.full(root.bars.shape[:2], np.nan)
            events = root.funding
            for owner, instrument, rate in zip(_owners(root, events["time"]),
                                               events["instrument"], events["rate"]):
                rates[owner:, instrument] = rate
            root.rates = _frozen(rates)

        return self._take(root.rates)


    @property
    def fingerprint(self):
        """A 128-bit blake2b over everything a run on these bars can read, written as
        ``"b2:"`` and 32 hex characters: the clock and its interval, the bars, which of
        them printed, the funding events and the instruments (ADR 0005). A slice's
        covers the bars before it as well, which its features and an agent's window
        warm up on, and where it starts among them. It is taken once."""
        if self._fingerprint is None:
            self._fingerprint = _digest(self)

        return self._fingerprint


    @property
    def provenance(self):
        """Where the bars came from: the source, the venue, the instruments, the
        interval, the first and the last bar, the warnings that came with them and the
        fingerprint."""
        out = copy.deepcopy(self._root.provenance)
        out["instruments"] = [":".join(key for key in _key(instrument) if key) or name
                              for instrument, name in zip(self.instruments, self.names)]
        if len(self):
            out.update(first=_when(self.time[0]), last=_when(self.time[-1]))

        out["fingerprint"] = self.fingerprint
        return out


    def __len__(self):
        start, stop = self._rows
        return stop - start


    def __getitem__(self, key):
        if isinstance(key, str):
            names = self.names
            if key not in names:
                raise KeyError(f"data: no instrument {key!r}; use one of "
                               f"{', '.join(names)}")
            return Data._of(self._root, self._rows, [self._cols[names.index(key)]])

        if isinstance(key, slice):
            if key.step is not None:
                raise TypeError(f"data: a step of {key.step!r}; slice bars without one")

            start, _ = self._rows
            first = start + (0 if key.start is None
                             else _row(self, key.start, True, "start"))
            last = start + (len(self) if key.stop is None
                            else _row(self, key.stop, False, "stop"))
            return Data._of(self._root, (first, max(first, last)), self._cols)

        raise TypeError(f"data: indexed by {key!r}; use an instrument's name or a "
                        f"slice of times")


    def split(self, when):
        """The bars before ``when`` and the bars from it on, as two Datas: ``when`` is
        a time, the text of a period, whose first bar starts the second Data, or a
        position."""
        start, stop = self._rows
        row = start + _row(self, when, True, "when")
        return (Data._of(self._root, (start, row), self._cols),
                Data._of(self._root, (row, stop), self._cols))


    def after(self, when):
        """The bars from ``when`` on, a time, the text of a period or a position."""
        return self.split(when)[1]


    def select(self, names):
        """The instruments ``names``, in that order."""
        if isinstance(names, str):
            raise TypeError(f"names: {names!r}; give a list of names, or index the "
                            f"Data by one")

        names = list(names)
        own   = self.names
        if not names:
            raise ValueError("names: none; name at least one instrument")

        missing = [name for name in names if name not in own]
        if missing:
            raise KeyError(f"names: no instrument {missing[0]!r}; use some of "
                           f"{', '.join(own)}")

        twice = [name for name in names if names.count(name) > 1]
        if twice:
            raise ValueError(f"names: {names!r} names {twice[0]} twice; name each once")

        return Data._of(self._root, self._rows,
                        [self._cols[own.index(name)] for name in names])


    def describe(self):
        """What a run on these bars assumes (Specification, Assumptions), as the
        `Assumptions` rows of the Data a run lists: its bars, gaps, funding events,
        history, delisting, universe and prices, the router's graded venue, with no
        cost."""
        from . import _assumptions, _result
        return _result.Assumptions._of(_assumptions.bars(self))


    def gaps(self):
        """Every place an instrument's bars jump by more than the interval, as `Gap`
        records: on a market that never closes, each is an outage or a hole in the
        data."""
        step = self._root.step
        out  = []
        for column, name in enumerate(self.names):
            times = self.time[self.available[:, column]]
            for k in np.flatnonzero(np.diff(times) > step):
                after, before = int(times[k]), int(times[k + 1])
                out.append(Gap(name, np.datetime64(after, "ns"),
                               np.datetime64(before, "ns"),
                               (before - after) // step - 1))

        return out


    def join(self, other):
        """This Data's instruments and ``other``'s on one clock, for a universe that
        holds both, such as a venue's spot and perpetual markets."""
        other = Data(other)
        if other.interval != self.interval:
            raise ValueError(f"other: {other.interval} bars beside {self.interval}; "
                             f"join bars on one interval")

        keys  = {_key(i) for i in self.instruments}
        twice = [i for i in other.instruments if _key(i) in keys]
        if twice:
            raise ValueError(f"other: holds {_label(twice[0])} too; join Datas of "
                             f"different instruments")

        parts = [Part(view.time[view.available[:, k]],
                      view.bars[view.available[:, k], k], view.instruments[k], {}, "",
                      "join")
                 for view in (self, other) for k in range(len(view._cols))]
        root = _clock(parts, self.interval, placed=True)

        merged, shift = [], 0
        for view in (self, other):
            events = view.funding.copy()
            events["instrument"] += shift
            merged.append(events)
            shift += len(view._cols)

        root.funding    = _frozen(np.sort(np.concatenate(merged), order="time"))
        root.provenance = dict(
            source="join", venue=_venues(root), interval=root.interval,
            warnings=(self._root.provenance["warnings"]
                      + other._root.provenance["warnings"]))

        return Data._of(root, (0, len(root.time)), range(len(root.instruments)))


    def resample(self, interval):
        """These instruments' bars at ``interval``, a whole multiple of this Data's
        (ADR 0006): each coarse bar the first open, the highest high, the lowest low and
        the last close of its bars and the sum of their volumes, available where any of
        them was. Coarse bars align to 00:00 UTC and weeks to Monday, moved onto this
        Data's own steps, and one these bars do not cover to its end is dropped, since
        its close is not yet a close, while the first holds the bars from these bars'
        first, wherever in it they begin. Funding events keep their stamps, the
        instruments their names, and a slice's coarse bars keep the ones before it as
        the slice keeps its bars, reading nothing after its end."""
        root = self._root
        if interval not in INTERVALS:
            raise ValueError(f"interval: {interval!r}; use one of "
                             f"{', '.join(INTERVALS)}")

        coarse = INTERVALS[interval]
        if coarse % root.step:
            raise ValueError(f"interval: {interval!r} for {root.interval} bars; give a "
                             f"whole multiple of {root.interval}")

        start, stop = self._rows
        if not self.available.any():
            raise ValueError("data: no bars to resample; resample a Data that has some")

        origin  = 4 * DAY if interval == "1w" else 0
        anchor  = origin + (int(root.time[0]) - origin) % root.step
        covered = int(root.time[stop - 1]) + root.step
        parts   = []
        for k in self._cols:
            printed = root.available[:stop, k]
            times, bars = root.time[:stop][printed], root.bars[:stop, k][printed]
            ids   = (times - anchor) // coarse
            whole = anchor + (ids + 1) * coarse <= covered
            times, bars, ids = times[whole], bars[whole], ids[whole]
            if len(ids):
                first = np.flatnonzero(np.diff(ids, prepend=ids[0] - 1))
                last  = np.append(first[1:], len(ids)) - 1
                bars  = np.column_stack([bars[first, 0],
                                         np.maximum.reduceat(bars[:, 1], first),
                                         np.minimum.reduceat(bars[:, 2], first),
                                         bars[last, 3],
                                         np.add.reduceat(bars[:, 4], first)])
                times = anchor + ids[first] * coarse

            parts.append(Part(times, bars, root.instruments[k], {}, "", "resample"))

        if not any(len(part.times) for part in parts):
            raise ValueError(f"interval: {interval!r}, and these bars cover no "
                             f"{interval} bar to its end; resample bars that do")

        out       = _clock(parts, interval, placed=True)
        out.names = self.names
        events    = Data._of(root, (0, stop), self._cols).funding
        events    = np.sort(events[_owners(out, events["time"]) < len(out.time)],
                            order=["time", "instrument"])

        out.funding    = _frozen(events)
        out.provenance = copy.deepcopy(root.provenance)
        out.provenance.update(interval=interval, resampled_from=root.interval)

        # the coarse bar a slice opens inside began among the bars before it, so it is
        # one of those the slice keeps behind it
        cut = anchor + (int(root.time[start]) - anchor) // coarse * coarse
        if start and root.time[start - 1] >= cut:
            cut += coarse

        first = int(np.searchsorted(out.time, cut, side="left"))
        return Data._of(out, (first, len(out.time)), range(len(out.instruments)))


    def asof(self, values, when=None):
        """``values`` from another clock put on this Data's: row t holds the last of
        them known by the end of bar t, and NaN before the first (ADR 0006). ``when``
        is a Data, or anything a Data is built from, whose values are known at the ends
        of its bars, or the times at which each row of ``values`` became known; left
        out, ``values`` is a frame or a series. A frame of bars is read as a Data is,
        each value known at its bar's end, a column of a router candle frame at its
        candle's end, and any other frame or series at its index or its time column.
        Where ``values`` holds a column for each of ``when``'s instruments, an
        instrument that printed no bar there gave no new value, so its last one
        stands."""
        named, shift = "when", 0
        if when is None:
            named = "values"
            values, known, shift = _stamped(values)
        else:
            if hasattr(when, "columns"):
                when = Data(when)
            if isinstance(when, Data):
                known, shift = when.time, when._root.step
            else:
                known, _ = _times(when, "when")

        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0 or len(values) != len(known):
            raise ValueError(f"values: {len(values) if values.ndim else 'one value'} "
                             f"for {len(known)} times; give one row of values per time")

        missing = np.flatnonzero(known == np.iinfo(np.int64).min)
        if len(missing):
            raise ValueError(f"{named}: no time at row {missing[0]}; give each row the "
                             f"time it became known")

        back = np.flatnonzero(np.diff(known) < 0)
        if len(back):
            raise ValueError(f"{named}: times that fall back at row {back[0] + 1}; "
                             f"sort the rows by the time each became known")

        known = known + shift
        ends  = self.time + self._root.step
        out   = np.full((len(self),) + values.shape[1:], np.nan)
        width = len(when._cols) if isinstance(when, Data) else 0

        if width and (values.ndim == 1 and width == 1
                      or values.ndim > 1 and values.shape[1] == width):
            printed = when.available
            for k in range(width):
                own  = np.flatnonzero(printed[:, k])
                rows = np.searchsorted(known[own], ends, side="right") - 1
                hit  = rows >= 0
                if values.ndim == 1:
                    out[hit] = values[own[rows[hit]]]
                else:
                    out[hit, k] = values[own[rows[hit]], k]

            return out

        rows = np.searchsorted(known, ends, side="right") - 1
        out[rows >= 0] = values[rows[rows >= 0]]
        return out


    def save(self, path):
        """Write these bars into the directory ``path``, made where it is missing, as
        three Arrow IPC files any Arrow reader opens (ADR 0005): ``bars.arrow``, a row
        for each bar and instrument, the instrument given as its row of
        ``instruments.arrow`` and the values null where it printed no bar, with the
        interval, the provenance and the fingerprint on its schema;
        ``instruments.arrow``; and ``funding.arrow``. A slice saves the bars before it
        as well, which its features warm up on, and loads back as the same slice. Each
        file is written under another name and then put in place, the bars last."""
        if not len(self):
            raise ValueError("data: no bars to save; save a Data that has some")

        folder = pathlib.Path(path)
        if folder.exists() and not folder.is_dir():
            raise ValueError(f"path: {str(path)!r} is a file; name a directory")
        folder.mkdir(parents=True, exist_ok=True)

        start, stop = self._rows
        seen, width = Data._of(self._root, (0, stop), self._cols), len(self._cols)
        bars    = seen.bars.reshape(stop * width, 5)
        printed = seen.available.reshape(-1)
        record  = dict(format=1, interval=self.interval, start=start,
                       fingerprint=self.fingerprint, provenance=self._root.provenance)

        listed = [("name", "text", list(self.names), None)]
        for field, kind in FIELDS.items():
            values = [getattr(instrument, field) for instrument in self.instruments]
            given  = np.array([value is not None for value in values])
            if kind == "float":
                values = np.array([np.nan if v is None else v for v in values])
            listed.append((field, kind, values, given))

        events    = seen.funding
        positions = np.tile(np.arange(width), stop)
        files     = {
            "instruments.arrow": (listed, {}),
            "funding.arrow":     ([("time", "time", events["time"], None),
                                   ("instrument", "int", events["instrument"], None),
                                   ("rate", "float", events["rate"], None)], {}),
            "bars.arrow":        ([("time", "time", np.repeat(seen.time, width), None),
                                   ("instrument", "int", positions, None)]
                                  + [(field, "float", bars[:, k], printed)
                                     for k, field in enumerate(_OHLCV)],
                                  {"probatum": json.dumps(record)}),
        }

        for name, (columns, metadata) in files.items():
            _ipc.write(folder / f"{name}.partial", columns, metadata)

        for name in files:
            os.replace(folder / f"{name}.partial", folder / name)


    @classmethod
    def load(cls, path):
        """The Data ``save`` wrote into the directory ``path``, a slice with the bars
        before it, its bars checked as a Data's are, and refused where they do not give
        the fingerprint they were saved under (ADR 0005)."""
        folder = pathlib.Path(path)
        bars, metadata = _saved(path, folder, "bars.arrow", dict(
            time="time", instrument="int", **{field: "float" for field in _OHLCV}))
        listed, _ = _saved(path, folder, "instruments.arrow",
                           dict(name="text", **FIELDS))
        events, _ = _saved(path, folder, "funding.arrow",
                           dict(time="time", instrument="int", rate="float"))

        fix = "give a directory a Data was saved into"
        try:
            record = json.loads(metadata.get("probatum", ""))
        except ValueError:
            record = None

        if (not isinstance(record, dict) or record.get("format") != 1
                or record.get("interval") not in INTERVALS
                or not isinstance(record.get("start"), int)
                or not isinstance(record.get("provenance"), dict)):
            raise ValueError(f"path: {str(path)!r} holds bars this version of probatum "
                             f"did not save; {fix}")

        names, time = listed["name"][1], bars["time"][1]
        width = len(names)
        rows  = len(time) // width if width else 0

        if (not width or len(time) != rows * width or not 0 <= record["start"] < rows
                or not bars["time"][2].all()
                or not np.array_equal(bars["instrument"][1],
                                      np.tile(np.arange(width), rows))
                or not (time.reshape(rows, width) == time[::width, None]).all()):
            raise ValueError(f"path: the rows of {str(path)!r}'s bars.arrow are not "
                             f"the ones save writes; {fix}")

        def value_of(name, k):
            kind, values, present, _ = listed[name]
            if not present[k]:
                return None
            return float(values[k]) if kind == "float" else values[k]

        instruments = [Instrument(**{name: value_of(name, k) for name in FIELDS})
                       for k in range(width)]
        values = np.stack([np.where(bars[field][2], bars[field][1], np.nan)
                           for field in _OHLCV], axis=1).reshape(rows, width, 5)
        clock = np.array(time[::width])

        for k in range(width):
            _check(names[k], clock, values[:, k])

        off = np.flatnonzero((clock - clock[0]) % INTERVALS[record["interval"]])
        if len(off):
            raise ValueError(f"path: the bars in {str(path)!r} open at "
                             f"{_when(clock[off[0]])}, between the "
                             f"{record['interval']} steps from {_when(clock[0])}; {fix}")

        owners = events["instrument"][1]
        if (np.any((owners < 0) | (owners >= width)) or not events["time"][2].all()
                or np.any(events["time"][1] == np.iinfo(np.int64).min)
                or not np.isfinite(events["rate"][1]).all()):
            raise ValueError(f"path: {str(path)!r}'s funding.arrow holds an event with "
                             f"no time, rate or instrument of these bars; {fix}")

        root       = _Root(clock, values, instruments, record["interval"])
        root.names = tuple(names)
        funding    = np.zeros(len(owners), dtype=FUNDING)
        funding["time"], funding["instrument"] = events["time"][1], owners
        funding["rate"] = events["rate"][1]
        root.funding    = _frozen(funding)
        root.provenance = record["provenance"]
        data            = cls._of(root, (record["start"], rows), range(width))

        if data.fingerprint != record.get("fingerprint"):
            raise ValueError(f"path: the bars in {str(path)!r} give the fingerprint "
                             f"{data.fingerprint}, not the {record.get('fingerprint')} "
                             f"they were saved under, so they changed since or another "
                             f"version of probatum saved them; build the Data again "
                             f"from where its bars came from")

        return data


    def __repr__(self):
        what = (self.names[0] if len(self._cols) == 1
                else f"{len(self._cols)} instruments")
        if not len(self):
            return f"Data({what}, {self.interval}, no bars)"
        return (f"Data({what}, {self.interval}, {len(self)} bars from "
                f"{_when(self.time[0])} to {_when(self.time[-1])})")


def _frozen(array):
    array.flags.writeable = False
    return array


def _digest(data):
    """``data``'s fingerprint: blake2b of 16 bytes over a line naming the layout, the
    interval, the rows before ``data`` and its own and its instruments, then the clock,
    the bars, the availability and the funding events of all those rows, as
    little-endian arrays with every zero unsigned and the events in order of time and
    instrument, then the instruments as JSON."""
    start, stop = data._rows

    seen   = Data._of(data._root, (0, stop), data._cols)
    events = np.sort(seen.funding, order=["time", "instrument"])
    out    = hashlib.blake2b(digest_size=16)

    out.update(f"probatum.data 1 {data.interval} {start} {stop - start} "
               f"{len(data._cols)}\n".encode())
    out.update(np.ascontiguousarray(seen.time, dtype="<i8"))
    out.update(np.ascontiguousarray(seen.bars + 0.0, dtype="<f8"))
    out.update(np.ascontiguousarray(seen.available, dtype=np.uint8))
    out.update(np.ascontiguousarray(events["time"], dtype="<i8"))
    out.update(np.ascontiguousarray(events["instrument"], dtype="<i8"))
    out.update(np.ascontiguousarray(events["rate"] + 0.0, dtype="<f8"))
    out.update(json.dumps([dataclasses.astuple(i) for i in data.instruments],
                          separators=(",", ":")).encode())

    return "b2:" + out.hexdigest()


def _saved(path, folder, name, kinds):
    """The columns and the metadata of the file ``name`` in ``folder``, the directory
    ``path`` names, refused where it is missing, unreadable or without the columns of
    ``kinds`` that save writes."""
    fix = "give a directory a Data was saved into"
    if not (folder / name).is_file():
        raise ValueError(f"path: {str(path)!r} holds no {name}; {fix}")

    try:
        columns, metadata = _ipc.read(folder / name)
    except _ipc.Refused as error:
        raise ValueError(f"path: {str(path)!r}'s {name} {error}; {fix}") from None

    for column, kind in kinds.items():
        if columns.get(column, ("",))[0] != kind:
            raise ValueError(f"path: {str(path)!r}'s {name} has no {kind} column "
                             f"{column}; {fix}")

    return columns, metadata


def _key(instrument):
    return instrument.venue, instrument.market, instrument.symbol


def _label(instrument):
    named = ":".join(part for part in _key(instrument) if part)
    return named or "an unnamed instrument"


def _display(instruments):
    """Each instrument's short name: its symbol, with its market where another
    market holds the same symbol, and its venue too where another venue does, either
    left out where the bars do not say it, or its position where it has no symbol."""
    names = []
    for k, instrument in enumerate(instruments):
        if instrument.symbol is None:
            names.append(str(k))
            continue

        same   = [i for i in instruments if i.symbol == instrument.symbol]
        venues = {i.venue for i in same}
        name   = instrument.symbol
        if instrument.market and (len({i.market for i in same}) > 1 or len(venues) > 1):
            name = f"{instrument.market}:{name}"
        if instrument.venue and len(venues) > 1:
            name = f"{instrument.venue}:{name}"
        names.append(name)

    return tuple(names)


def _venues(root):
    return ", ".join(sorted({i.venue for i in root.instruments if i.venue})) or None


def _count(n, noun):
    """``n`` of ``noun``, such as 1 bar or 2 bars."""
    return f"{n} {noun}{'s' * (n != 1)}"


def _when(ns):
    """A time as a reader reads it, to the minute, or to the second where it has
    seconds."""
    text = str(np.datetime64(int(ns), "ns").astype("datetime64[s]")).replace("T", " ")
    return text[:-3] if text.endswith(":00") else text


def _period(when, name):
    """The start and the end of the period ``when`` names, in nanoseconds since 1970:
    text names the year, month, day, hour, minute or second it gives, and a time
    names itself. ``name`` is what the caller called it."""
    if isinstance(when, str):
        text = when.strip().replace(" ", "T")
        try:
            # numpy reads no text and "NaT" as a time that is not there
            start = np.datetime64(text) if text and text.lower() != "nat" else None
        except ValueError:
            start = None

        if start is None:
            raise ValueError(f"{name}: {when!r} is not a time; give one such as "
                             f"\"2024-06\" or \"2024-06-01 08:00\"")

        end = start + np.timedelta64(1, np.datetime_data(start.dtype)[0])
        return (int(start.astype("datetime64[ns]").astype(np.int64)),
                int(end.astype("datetime64[ns]").astype(np.int64)))

    if isinstance(when, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: {when!r}; give a time or the text of one")

    if isinstance(when, datetime.datetime) and when.tzinfo is not None:
        when = when.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    try:
        ns = int(np.datetime64(when, "ns").astype(np.int64))
    except (TypeError, ValueError):
        raise TypeError(f"{name}: {when!r}; give a time or the text of one") from None

    if ns == np.iinfo(np.int64).min:
        raise ValueError(f"{name}: {when!r} is not a time; give one such as "
                         f"\"2024-06\" or \"2024-06-01 08:00\"")

    return ns, ns + 1


def _row(data, bound, start, name):
    """The row ``bound`` falls on: a position, or the first row of a time or a period
    where ``start``, and the row after its last where not. ``name`` is what the
    caller called it."""
    if isinstance(bound, (int, np.integer)) and not isinstance(bound, bool):
        rows = range(len(data))
        return rows[bound:].start if start else rows[:bound].stop

    first, end = _period(bound, name)
    return int(np.searchsorted(data.time, first if start else end, side="left"))


def _owners(root, times):
    """The row that owns each funding event at ``times``: the first whose bar ends at
    or after it, a stamp within a second of a boundary taken as the boundary."""
    ends = root.time + root.step
    return np.searchsorted(ends, np.asarray(times) - SECOND, side="left")


def _parts(source, symbol, market, time):
    """The bars ``source`` holds, as a `Part` for each instrument."""
    attrs  = getattr(source, "attrs", None)
    attrs  = dict(attrs) if isinstance(attrs, dict) and "exchange" in attrs else {}
    origin = "router" if attrs else "frame"
    pandas = (type(source).__module__.split(".")[0] == "pandas"
              and hasattr(source, "columns"))

    if hasattr(source, "__arrow_c_stream__") or hasattr(source, "__arrow_c_array__"):
        try:
            columns = arrow_columns(source)
        except Exception as error:
            # pandas exports its stream through pyarrow, which converts every column,
            # so without pyarrow, or with a column it cannot convert, the frame is
            # read by its own methods below
            if not pandas or not (isinstance(error, ImportError)
                                  or type(error).__module__.startswith("pyarrow")):
                raise
        else:
            times, bars, zone = _from_columns(columns, time)
            return [_part(times, bars, zone, symbol, market, attrs, origin)]

    if pandas:
        times, bars, zone = _from_pandas(source, time)
        return [_part(times, bars, zone, symbol, market, attrs, origin)]

    try:
        array = np.from_dlpack(source) if hasattr(source, "__dlpack__") else source
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"data: a {type(source).__name__}; pass a frame, a dict of "
                        f"frames or an array of bars") from None

    if time is None:
        raise TypeError("time: none for an array of bars; pass time=, each bar's open "
                        "time")

    times, zone = _times(time)
    if array.ndim == 2 and array.shape[1] == 5:
        return [_part(times, array, zone, symbol, market, {}, "array")]

    if array.ndim == 3 and array.shape[2] == 5:
        width = array.shape[1]
        names = [None] * width if symbol is None else symbol
        if (not isinstance(names, (list, tuple)) or len(names) != width
                or not all(name is None or isinstance(name, str) for name in names)):
            raise ValueError(f"symbol: {symbol!r} for {width} instruments; give a list "
                             f"of {width} names")

        twice = [name for name in names if name is not None and names.count(name) > 1]
        if twice:
            raise ValueError(f"symbol: {symbol!r} names {twice[0]} twice; name each "
                             f"once")

        return [_part(times, array[:, k], zone, names[k], market, {}, "array")
                for k in range(width)]

    raise ValueError(f"data: an array of shape {array.shape}; give (T, 5) bars of one "
                     f"instrument or (T, N, 5) of N")


def _part(times, bars, zone, symbol, market, attrs, origin):
    """One instrument's `Part`, its symbol and market checked against what a router
    frame says of itself."""
    if not (symbol is None or isinstance(symbol, str)):
        raise TypeError(f"symbol: {symbol!r} for one instrument's bars; give its name")

    if attrs:
        named = attrs.get("symbol")
        if symbol is not None and named is not None and symbol != named:
            raise ValueError(f"symbol: {symbol!r} for a frame of {named}; leave it "
                             f"out, the frame names itself")
        symbol = symbol or named
        kind   = attrs.get("market_type")
        if market is not None and kind is not None and market != kind:
            raise ValueError(f"market: {market!r} for a frame of the {kind} market; "
                             f"leave it out, the frame names itself")
        market = market or kind
        if market == "inverse" and attrs.get("volume_unit") not in (None, "contract"):
            raise ValueError(f"data: {symbol}'s volume is in {attrs['volume_unit']}, "
                             f"and an inverse contract's volume is counted in "
                             f"contracts; give its volume in contracts")

    if market is not None and market not in MARKETS:
        raise ValueError(f"market: {market!r}; use {', '.join(MARKETS)}")

    if len(times) != len(bars):
        raise ValueError(f"time: {len(times)} times for {len(bars)} bars; give each "
                         f"bar its open time")

    # the router writes an empty quote or size where it holds no symbol information
    contract   = attrs.get("contract_size") or None if market == "inverse" else None
    instrument = Instrument(
        symbol   = symbol,
        market   = market,
        venue    = attrs.get("exchange"),
        quote    = attrs.get("quote") or None,
        contract = None if contract is None else float(contract),
    )

    return Part(times, np.asarray(bars, dtype=np.float64), instrument, attrs, zone,
                origin)


def _from_columns(columns, time):
    """The open times, the bars and the zone of a frame read as Arrow ``columns``."""
    named = {name: (kind, values, detail) for name, kind, values, detail in columns}
    bars  = _five(named, lambda name: named[name][1],
                  lambda name: named[name][0] in ("float", "int"),
                  lambda name: _described(named[name][0], named[name][2]))

    if time is not None:
        times, zone = _times(time)
        return times, bars, zone

    clocks = [name for name, (kind, _, _) in named.items() if kind in ("time", "date")]
    if len(clocks) > 1:
        raise ValueError(f"data: times in {', '.join(clocks)}; keep the bars' open "
                         f"times alone, or pass time=")

    if clocks:
        kind, values, zone = named[clocks[0]]
        return values, bars, zone if kind == "time" else ""

    if named.get("timestamp", ("",))[0] == "int":
        return named["timestamp"][1] * 1_000_000, bars, "UTC"

    raise ValueError("data: no time column; give the bars' open times in a datetime "
                     "column or index, a timestamp column of milliseconds, or time=")


def _described(kind, detail):
    """What an Arrow column of ``kind`` and format ``detail`` holds, for a refusal to
    name."""
    if kind in ("time", "date"):
        return f"{kind}s"
    if detail.startswith("dictionary of "):
        return f"categories of {_described('other', detail[len('dictionary of '):])}"
    return ARROW.get(detail, detail)


def _from_pandas(frame, time):
    """The open times, the bars and the zone of a pandas frame read by its own
    methods, where pyarrow is missing or cannot convert one of its columns, its
    times taken from the places an export would hold them."""
    bars = _five(frame.columns,
                 lambda name: frame[name].to_numpy(dtype=np.float64, na_value=np.nan),
                 lambda name: frame[name].dtype.kind in "fiu",
                 lambda name: f"values of {frame[name].dtype}")

    if time is not None:
        times, zone = _times(time)
        return times, bars, zone

    clocks = {name: frame[name] for name in frame.columns
              if frame[name].dtype.kind == "M"}
    if getattr(frame.index.dtype, "kind", "") == "M":
        clocks = {"the index": frame.index, **clocks}

    if len(clocks) > 1:
        raise ValueError(f"data: times in {', '.join(map(str, clocks))}; keep the "
                         f"bars' open times alone, or pass time=")

    if clocks:
        times, zone = _times(next(iter(clocks.values())))
        return times, bars, zone

    if "timestamp" in frame.columns and frame["timestamp"].dtype.kind in "iu":
        return frame["timestamp"].to_numpy(dtype=np.int64) * 1_000_000, bars, "UTC"

    raise ValueError("data: no time column; give the bars' open times in a datetime "
                     "column or index, a timestamp column of milliseconds, or time=")


def _five(columns, values, numeric, kind):
    """The ``(n, 5)`` bars of a frame whose ``columns`` hold them."""
    missing = [name for name in _OHLCV if name not in columns]
    if missing:
        raise ValueError(f"data: no {', '.join(missing)} column; give open, high, low, "
                         f"close and volume")

    for name in _OHLCV:
        if not numeric(name):
            raise ValueError(f"data: column {name} holds {kind(name)}; give numbers")

    return np.column_stack([np.asarray(values(name), dtype=np.float64)
                            for name in _OHLCV])


def _times(time, name="time"):
    """``time``, the argument ``name``, as int64 nanoseconds since 1970, and the zone it
    carries, which a pandas index or series holds on its array."""
    zone = ""
    time = getattr(time, "array", time)
    if getattr(time, "tz", None) is not None:
        zone = time.tz
        time = time.tz_convert("UTC").tz_localize(None)

    values = np.asarray(time)
    if values.dtype.kind == "O":
        try:
            values = values.astype("datetime64[ns]")
        except (TypeError, ValueError):
            pass

    if values.dtype.kind == "M":
        return values.astype("datetime64[ns]").astype(np.int64), zone

    if values.dtype.kind in "iu":
        return values.astype(np.int64), zone

    raise TypeError(f"{name}: values of {values.dtype}; give datetimes or int64 "
                    f"nanoseconds since 1970")


def _columns(source):
    """A pandas frame's or series' columns, or those an Arrow stream or array exports:
    the times among them by name, a pandas datetime index as ``"the index"``, each as
    int64 nanoseconds since 1970 and the zone it carries; the numbers, a pandas column
    of integers with no gap as int64 and every other as float64 with NaN in its gaps;
    and what each other column holds."""
    clocks, numbers, others = {}, {}, {}
    if type(source).__module__.split(".")[0] == "pandas":
        if getattr(source.index.dtype, "kind", "") == "M":
            clocks["the index"] = _times(source.index)
        columns = ({name: source[name] for name in source.columns}
                   if hasattr(source, "columns") else {source.name: source})
        for name, column in columns.items():
            if column.dtype.kind == "M":
                clocks[name] = _times(column)
            elif column.dtype.kind in "iu" and not column.hasnans:
                numbers[name] = column.to_numpy(dtype=np.int64)
            elif column.dtype.kind in "fiu":
                numbers[name] = column.to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                others[name] = f"values of {column.dtype}"
    else:
        for name, kind, values, detail in arrow_columns(source):
            if kind in ("time", "date"):
                clocks[name] = (values, detail if kind == "time" else "")
            elif kind in ("float", "int"):
                numbers[name] = values
            else:
                others[name] = _described(kind, detail)

    return clocks, numbers, others


def _stamped(source):
    """The values a frame or a series holds, ``(K,)`` for one column and ``(K, C)`` for
    C, the time at which each row became known and what to add to reach it: a frame of
    bars is read as a Data is, each bar known at its end, and a column of a router
    candle frame at its candle's end; anything else at its datetime index or its one
    time column, or a ``timestamp`` column of milliseconds where it has neither."""
    pandas = type(source).__module__.split(".")[0] == "pandas"
    if not (pandas or hasattr(source, "columns")):
        raise TypeError(f"values: a {type(source).__name__}, which carries no times; "
                        f"pass when=, the time each row became known")

    attrs = getattr(source, "attrs", None)
    attrs = attrs if isinstance(attrs, dict) else {}

    # a router candle frame, or a column of one, says what its volume is counted in
    if "volume_unit" in attrs and attrs.get("interval") not in INTERVALS:
        raise ValueError(f"values: candles of {attrs.get('interval')!r}, an interval "
                         f"no Data runs on; give them on one of {', '.join(INTERVALS)}")

    clocks, numbers, _ = _columns(source)
    if hasattr(source, "columns") and {"open", "high", "low", "close"} <= set(numbers):
        # bars' times are their opens, and what a bar holds is known at its end
        bars    = Data(source)
        printed = ~np.isnan(np.column_stack([np.asarray(numbers[name], dtype=np.float64)
                                             for name in _OHLCV])).all(axis=1)
        if not clocks:
            numbers.pop("timestamp", None)
        values = np.column_stack([np.asarray(column, dtype=np.float64)[printed]
                                  for column in numbers.values()])
        values = values[:, 0] if values.shape[1] == 1 else values
        return values, bars.time, bars._root.step

    if len(clocks) > 1:
        raise ValueError(f"values: times in {', '.join(map(str, clocks))}; keep the "
                         f"time each row became known alone, or pass when=")

    if clocks:
        known = next(iter(clocks.values()))[0]
    elif getattr(numbers.get("timestamp"), "dtype", np.dtype(float)).kind == "i":
        known = numbers.pop("timestamp") * 1_000_000
    else:
        raise ValueError("values: no times; give the time each row became known in a "
                         "datetime index or column, or pass when=")

    if not numbers:
        raise ValueError("values: no column of numbers; give one beside the times")

    values = np.column_stack([np.asarray(column, dtype=np.float64)
                              for column in numbers.values()])
    shift = INTERVALS[attrs["interval"]] if "volume_unit" in attrs else 0
    return (values[:, 0] if values.shape[1] == 1 else values), known, shift


def _clock(parts, interval, placed=False):
    """The root of the bars ``parts`` hold, each instrument's bars checked and placed
    on one clock of ``interval``, read from the gaps between bars where not given."""
    shown = _display([part.instrument for part in parts])
    names = [name if part.instrument.symbol is not None else f"instrument {name}"
             for part, name in zip(parts, shown)]

    checked = []
    for part, name in zip(parts, names):
        times = np.asarray(part.times, dtype=np.int64)
        _check(name, times, part.bars)
        printed = ~np.isnan(part.bars).all(axis=1)
        checked.append(part._replace(times=times[printed], bars=part.bars[printed]))

    if not any(len(part.times) for part in checked):
        raise ValueError("data: no bars; give at least one")

    if interval is None:
        interval = _interval(checked)
    if interval not in INTERVALS:
        raise ValueError(f"interval: {interval!r}; use one of {', '.join(INTERVALS)}")

    step = INTERVALS[interval]

    if step >= DAY and not placed:
        checked = [part._replace(times=_dated(part.times, part.zone))
                   for part in checked]
        for part, name in zip(checked, names):
            repeat = np.flatnonzero(np.diff(part.times) == 0)
            if len(repeat):
                day = np.datetime64(int(part.times[repeat[0]]), "ns").astype("M8[D]")
                raise ValueError(f"data: {name} has two bars dated {day}; give each "
                                 f"bar a date of its own")

    times = np.unique(np.concatenate([part.times for part in checked]))
    off   = np.flatnonzero((times - times[0]) % step)
    if len(off):
        late  = times[off[0]]
        owner = next(name for part, name in zip(checked, names) if late in part.times)
        raise ValueError(f"data: {owner} opens a bar at {_when(late)}, between the "
                         f"{interval} steps from {_when(times[0])}; give every "
                         f"instrument bars that open on one interval's steps")

    bars = np.full((len(times), len(checked), 5), np.nan)
    for column, part in enumerate(checked):
        bars[np.searchsorted(times, part.times), column] = part.bars

    return _Root(times, bars, [part.instrument for part in parts], interval)


def _check(name, times, bars):
    """Refuse ``name``'s bars where a bar is missing some of its values, holds one that
    is not finite, does not contain its open and close, or has a price at or below
    zero or a negative volume, and where a time is missing, repeats or falls back
    (ADR 0007)."""
    missing = np.flatnonzero(times == np.iinfo(np.int64).min)
    if len(missing):
        raise ValueError(f"data: {name}'s bar at row {missing[0]} has no open time; "
                         f"give every bar its open time")

    back = np.flatnonzero(np.diff(times) <= 0)
    if len(back):
        row = int(back[0]) + 1
        if times[row] == times[row - 1]:
            raise ValueError(f"data: {name} has two bars at {_when(times[row])}; give "
                             f"each bar a time of its own")
        raise ValueError(f"data: {name}'s bars fall back at row {row}, from "
                         f"{_when(times[row - 1])} to {_when(times[row])}; sort them "
                         f"by time")

    gone    = np.isnan(bars)
    partial = np.flatnonzero(gone.any(axis=1) & ~gone.all(axis=1))
    if len(partial):
        row     = int(partial[0])
        missing = [field for field, lost in zip(_OHLCV, gone[row]) if lost]
        raise ValueError(f"data: {name}'s bar at {_when(times[row])} misses its "
                         f"{', '.join(missing)}; give all five values, or none where "
                         f"it printed no bar")

    printed  = ~gone.all(axis=1)
    infinite = np.flatnonzero(printed & np.isinf(bars).any(axis=1))
    if len(infinite):
        raise ValueError(f"data: {name}'s bar at {_when(times[infinite[0]])} holds an "
                         f"infinite value; give finite values")

    opened, high, low, closed, volume = (bars[:, k] for k in range(5))
    with np.errstate(invalid="ignore"):
        wrong = np.flatnonzero(printed & ~((low <= opened) & (opened <= high)
                                           & (low <= closed) & (closed <= high)))
        nought   = np.flatnonzero(printed & (low <= 0.0))
        negative = np.flatnonzero(printed & (volume < 0.0))

    if len(wrong):
        row = wrong[0]
        raise ValueError(f"data: {name}'s bar at {_when(times[row])} is not a bar: "
                         f"open {opened[row]:g}, high {high[row]:g}, low {low[row]:g}, "
                         f"close {closed[row]:g}; need low <= open <= high and low <= "
                         f"close <= high, so check the column order")

    if len(nought):
        raise ValueError(f"data: {name}'s bar at {_when(times[nought[0]])} has a low "
                         f"of {low[nought[0]]:g}; give prices above 0")

    if len(negative):
        raise ValueError(f"data: {name}'s bar at {_when(times[negative[0]])} has a "
                         f"volume of {volume[negative[0]]:g}; give a volume of 0 or "
                         f"more")


def _interval(checked):
    """The interval the median gap between the bars gives: the shortest of the
    instruments' medians, since one that prints seldom still prints on the clock's
    steps, each the lower of the two middle gaps where there is an even number,
    since a market that closes has fewer long gaps than short ones. Gaps of about a
    day or more are measured between the dates the bars carry, since days that open
    at local midnight are an hour short or long of a day where the clocks change."""
    found = []
    for part in checked:
        if len(part.times) > 1:
            gap = _median(np.diff(part.times))
            if gap >= DAY - HOUR:
                gap = _median(np.diff(_dated(part.times, part.zone)))
            found.append(gap)

    if not found:
        raise ValueError("data: one bar, from which no interval can be read; pass "
                         "interval=")

    gap = min(found)
    for name, step in INTERVALS.items():
        if step == gap:
            return name

    raise ValueError(f"data: bars {_span(gap)} apart at the median, which is no "
                     f"interval the clock runs on; give bars on one of "
                     f"{', '.join(INTERVALS)}, or pass interval=")


def _median(gaps):
    """The middle of ``gaps``, the lower of the two middle ones where there is an even
    number."""
    return int(np.sort(gaps)[(len(gaps) - 1) // 2])


def _span(ns):
    """A length of time as the router writes an interval, such as 1h30m, with any
    part of a second in nanoseconds."""
    days, rest = divmod(ns, DAY)
    hours, rest = divmod(rest, HOUR)
    minutes, rest = divmod(rest, 60 * SECOND)
    seconds, rest = divmod(rest, SECOND)

    shown = [f"{n}{unit}" for n, unit in ((days, "d"), (hours, "h"), (minutes, "m"),
                                         (seconds, "s"), (rest, "ns")) if n]
    return "".join(shown) or "0s"


def _dated(times, zone):
    """Daily or longer bars placed at 00:00 UTC on the date each carries in its own
    time zone, so a day stamped at midnight in New York stays on its date. ``zone``
    is a pandas time zone, or an Arrow one: a name or an offset such as -05:00."""
    if str(zone) in ("", "UTC", "Etc/UTC", "+00:00"):
        return times - times % DAY

    if isinstance(zone, str):
        offset = re.fullmatch(r"([+-])(\d\d):(\d\d)", zone)
        if offset:
            sign, hours, minutes = offset.groups()
            zone = datetime.timezone(int(sign + "1") * datetime.timedelta(
                hours=int(hours), minutes=int(minutes)))
        else:
            try:
                zone = zoneinfo.ZoneInfo(zone)
            except (zoneinfo.ZoneInfoNotFoundError, ValueError):
                raise ValueError(f"data: times in the zone {zone!r}, which this "
                                 f"machine does not know; give them in UTC") from None

    epoch = datetime.date(1970, 1, 1)
    return np.array([(datetime.datetime.fromtimestamp(int(t) / SECOND, zone).date()
                      - epoch).days * DAY for t in times], dtype=np.int64)


def features(data, values, fill=None):
    """The ``(T, N, F)`` features ``values`` for ``data``: the array itself, or a
    function of a Data that returns one, given the bars before a slice as well so
    that it can warm up on them. A feature is finite wherever its instrument is
    available once its warm-up, the values before its first finite one, is past,
    counted from the first bar the feature was given; one that is not is refused
    unless ``fill`` is a finite number to put in its place or ``"previous"``, the
    instrument's last value before it."""
    if not (fill is None or isinstance(fill, str) and fill == "previous"
            or isinstance(fill, (int, float, np.integer, np.floating))
            and not isinstance(fill, bool) and np.isfinite(fill)):
        raise ValueError(f"fill: {fill!r}; give a finite number or \"previous\"")

    start = 0
    if callable(values):
        start, stop = data._rows
        data   = Data._of(data._root, (0, stop), data._cols)
        values = values(data)

    values = np.asarray(values, dtype=np.float64)
    shape  = (len(data), len(data._cols))
    if values.ndim != 3 or values.shape[:2] != shape:
        raise ValueError(f"features: shape {values.shape} for {shape[0]} bars of "
                         f"{shape[1]} instruments; give ({shape[0]}, {shape[1]}, F)")

    out       = values.copy()
    available = data.available

    for instrument in range(shape[1]):
        rows = np.flatnonzero(available[:, instrument])
        for feature in range(values.shape[2]):
            series = out[rows, instrument, feature]
            finite = np.isfinite(series)
            if not finite.any():
                continue

            warm = int(np.argmax(finite))
            bad  = np.flatnonzero(~finite[warm:]) + warm
            if fill is None:
                # a value lost in the bars before a slice is no matter, as the slice
                # never reads it
                read = bad[rows[bad] >= start]
                if len(read):
                    row, lost = int(rows[read[0]]), series[read[0]]
                    value = "NaN" if np.isnan(lost) else str(lost)
                    raise ValueError(f"features: {value} at bar {row - start:,} "
                                     f"({_when(data.time[row])}) for "
                                     f"{data.names[instrument]}, feature {feature}; "
                                     f"pass fill=")
                continue

            for k in bad:
                series[k] = series[k - 1] if fill == "previous" else fill
            out[rows, instrument, feature] = series

    return _frozen(out[start:])


def _now():
    """Nanoseconds since 1970, now."""
    return time.time_ns()


def _walk(call, first, upper, read):
    """Every row from ``first`` to ``upper``, in nanoseconds since 1970, that ``call``
    gives walking back from ``upper``, as the pieces the calls brought, newest first,
    and the earliest row where the venue's history begins after ``first``. ``call``
    takes the newest time wanted and the pieces brought so far, and ``read`` turns
    what it returns into the rows' times, their values and what else the frame says.
    A call that brings nothing earlier is asked once more, and where it brings
    nothing again the venue's history begins at the earliest row."""
    pieces, asked = [], False
    while upper >= first:
        times, values, carried = read(call(upper, pieces))
        kept = (times >= first) & (times <= upper)
        if not kept.any():
            if asked:
                break
            asked = True
            continue

        asked = False
        pieces.append((times[kept], values[kept], carried))
        upper = int(times[kept].min()) - 1_000_000

    begun = None
    if pieces and upper >= first:
        begun = min(int(piece[0].min()) for piece in pieces)

    return pieces, begun


def _joined(pieces):
    """The rows a walk's ``pieces`` hold, in time order, a row two calls both brought
    taken once."""
    times  = np.concatenate([piece[0] for piece in pieces])
    values = np.concatenate([piece[1] for piece in pieces])
    order  = np.argsort(times, kind="stable")
    times, values = times[order], values[order]

    once = np.concatenate([[True], np.diff(times) != 0])
    return times[once], values[once]


def _bars_of(frame, symbol, market):
    """The open times and the bars of a router candle frame, with its `Part`."""
    if not len(frame):
        return np.zeros(0, dtype=np.int64), np.zeros((0, 5)), None

    [part] = _parts(frame, symbol, market, None)
    return np.asarray(part.times, dtype=np.int64), part.bars, part


def _listed(router, exchange, market, symbol):
    """What the router's symbol information says of ``symbol``, and None: the router's
    own name for it, which the presets key it by however it was typed; its base asset;
    the smallest notional the venue takes in an order, in the quote currency, left out
    on an inverse contract, whose minimum the router gives in dollars where the engine
    counts one in the coin; and the largest size it takes in a limit order, in base
    units, or contracts there. Where the router does not list the symbol now, which it
    answers as a bad request, it is None and what the router said."""
    try:
        info = router.get_symbol_info(exchange, market, symbol)
    except Exception as error:
        if getattr(error, "status", None) != 400:
            raise
        return None, getattr(error, "detail", str(error))

    def number(name):
        value = info.get(name)
        return None if value is None else float(value)

    return dict(symbol=info.get("symbol") or symbol,
                base=info.get("base_asset") or None,
                min_notional=None if market == "inverse" else number("min_notional"),
                max_qty=number("max_qty")), None


def _rates_of(frame):
    """The times and the rates of a router funding frame, and the warnings it came
    with."""
    told = list(frame.attrs.get("warnings", []))
    if not len(frame):
        return np.zeros(0, dtype=np.int64), np.zeros(0), told

    columns = None
    if hasattr(frame, "__arrow_c_stream__"):
        try:
            columns = arrow_columns(frame)
        except ImportError:
            pass

    if columns is None:
        times, _ = _times(frame.index)
        rates = frame["rate"].to_numpy(dtype=np.float64, na_value=np.nan)
        return times, rates, told

    times = [values for _, kind, values, _ in columns if kind == "time"]
    rates = [values for name, kind, values, _ in columns
             if name == "rate" and kind in ("float", "int")]
    if len(times) != 1 or not rates:
        raise ValueError("data: a funding frame without one time column and a rate; "
                         "pass the router's own frame")

    return times[0], np.asarray(rates[0], dtype=np.float64), told


def _funding(router, exchange, market, root, step):
    """The funding events that charge each instrument of ``root``, fetched back as
    its bars were and owned by its bars, and the warnings the router gave with them;
    refused where the series misses part of its instrument's span, which a run would
    otherwise charge nothing for."""
    events, notes = [], []
    for column, (instrument, name) in enumerate(zip(root.instruments, root.names)):
        printed = root.time[root.available[:, column]]
        opened, ended = int(printed[0]), int(printed[-1]) + step

        def call(upper, pieces, symbol=instrument.symbol):
            # the events the span holds at the cycle the last ones brought, or at eight
            # hours before any came: Binance reads the count as a window of that many
            # cycles and returns the oldest events in it, and a paging venue reports a
            # count it cannot fill as a short result
            gaps  = np.diff(pieces[-1][0]) if pieces else ()
            cycle = max(_median(gaps), HOUR) if len(gaps) else 8 * HOUR
            return router.get_funding_rate(exchange, market, symbol,
                                           limit=(upper - opened) // cycle + 1,
                                           start=upper // 1_000_000, verbose=False)

        pieces, _ = _walk(call, opened, ended + SECOND, _rates_of)
        notes += [w for piece in pieces for w in piece[2]]
        times, rates = (_joined(pieces) if pieces
                        else (np.zeros(0, np.int64), np.zeros(0)))
        owned = (times > opened + SECOND) & (times <= ended + SECOND)
        times, rates = times[owned], rates[owned]
        if not len(times):
            if ended - opened > DAY:
                raise ValueError(f"funding: {name} has no funding events over its "
                                 f"bars, from {_when(opened)} to {_when(ended)}; load "
                                 f"it without funding")
            continue

        missing = np.flatnonzero(np.isnan(rates))
        if len(missing):
            raise ValueError(f"funding: {name}'s event at {_when(times[missing[0]])} "
                             f"has no rate; load it without funding, or after that "
                             f"time")

        spacing = np.diff(times)
        head, tail = (int(spacing[0]), int(spacing[-1])) if len(spacing) else (DAY, DAY)
        if times[0] - opened > head + SECOND:
            raise ValueError(f"funding: {name}'s funding history begins at "
                             f"{_when(times[0])}, more than a cycle after its first "
                             f"bar at {_when(opened)}; start the bars at "
                             f"{_when(times[0])} or load them without funding")

        if ended - times[-1] > tail + SECOND:
            raise ValueError(f"funding: {name}'s funding history ends at "
                             f"{_when(times[-1])}, more than a cycle before its last "
                             f"bar ends at {_when(ended)}; end the bars by "
                             f"{_when(times[-1])} or load them without funding")

        # a cycle may change, so a hole is a spacing more than half as long again as
        # both its neighbours
        around = np.maximum(np.concatenate([[0], spacing[:-1]]),
                            np.concatenate([spacing[1:], [0]]))
        hole = np.flatnonzero(spacing > 1.5 * around) if len(spacing) > 1 else []
        if len(hole):
            k = hole[0]
            raise ValueError(f"funding: {name}'s funding history misses the events "
                             f"between {_when(times[k])} and {_when(times[k + 1])}; "
                             f"load the bars before or after them, or without funding")

        events += [(int(t), column, float(r)) for t, r in zip(times, rates)]

    return np.sort(np.array(events, dtype=FUNDING), order="time"), notes
