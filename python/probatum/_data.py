"""Input helpers: turn a numpy array, a pandas DataFrame, or a parquet path into
the numeric arrays the engine takes. pandas and pyarrow are optional; they are
imported only when a DataFrame or a parquet path is passed.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._probatum import periods_per_year

_OHLCV = ["open", "high", "low", "close", "volume"]


def to_ohlcv(data):
    """Return a ``(T, 5)`` float64 OHLCV array from a numpy array, a pandas
    DataFrame with the columns open/high/low/close/volume, or a path to a parquet
    file.

    **Volume is in base units**, the same units an order's size is in, because the
    liquidity budget compares one against the other and the market-impact term
    divides one by the other. A feed shipping quote volume instead inflates the
    budget by roughly the price, so ``budget`` stops binding and impact goes to
    zero: no error, no warning, just a better backtest than the market would have
    given. The router's frames carry base ``volume`` and a separate
    ``volume_usd``, and only the first is read.
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
