"""What one run produced, whatever its policy was: its equity and returns as arrays and
its records as tables any Arrow reader takes, saved as files any tool reads
(ADR 0026)."""

import copy
import functools
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import sys

import numpy as np

from . import _ipc
from ._probatum import Table as _Held
from ._venue import _settlement

# the format a saved Result is written in, which this version reads, as every later
# one will
FORMAT = 1
# each table a Result holds: its columns, of a kind each, and whether each can hold a
# null, as the binding builds them and a saved Result's files hold them
SCHEMAS = {
    "ledger": [("bar", "int", False), ("time", "time", False)]
    + [(name, "float", True) for name in (
        "equity", "spot_cash", "margin_cash", "gross_exposure", "net_exposure",
        "leverage", "margin_used", "maintenance", "fees", "funding", "spread",
        "slippage", "impact", "liquidation", "dust", "equity_in_quote")],
    "holdings": [("bar", "int", False), ("time", "time", False),
                 ("instrument", "text", False)]
    + [(name, "float", True) for name in (
        "position", "entry", "mark", "notional", "unrealized", "weight", "drift")],
    "fills": [("bar", "int", False), ("time", "time", False),
              ("instrument", "text", False), ("order", "int", True),
              ("side", "text", False)]
    + [(name, "float", True) for name in (
        "size", "price", "reference", "spread", "slippage", "impact", "fee",
        "adjustment")]
    + [("taker", "bool", False), ("reason", "text", False)],
    "orders": [("bar", "int", False), ("time", "time", False), ("kind", "text", False),
               ("order", "int", True), ("instrument", "text", True),
               ("size", "float", True), ("price", "float", True),
               ("reason", "text", False)],
    "trades": [("instrument", "text", False), ("side", "text", False)]
    + [(name, "float", True) for name in ("size", "entry", "exit")]
    + [("open_bar", "int", False), ("close_bar", "int", True),
       ("open_time", "time", False), ("close_time", "time", True)]
    + [(name, "float", True) for name in (
        "realized", "fees", "funding", "pnl", "ret", "best", "worst")]
    + [("reason", "text", True), ("open", "bool", False)],
}
# the statistics a run reports, in the order the binding gives them, the counts among
# them integers
STATS = ("total_return_pct", "net_profit_pct", "cagr_pct", "sharpe", "sortino", "calmar",
         "max_drawdown_pct", "volatility_pct", "exposure_pct", "win_rate",
         "profit_factor", "num_trades", "avg_trade_pct", "num_fills", "funding_paid")
COUNTS = ("num_trades", "num_fills")
# the order log's kinds of event, in the order an order's life meets them
KINDS = ("placed", "armed", "clamped", "refused", "not_filled", "expired", "cancelled",
         "filled")
BARS   = ("bar", "open_bar", "close_bar")
STORED = {
    "float": "float",
    "int":   "int64",
    "time":  "time",
    "bool":  "bool",
    "text":  "large_text",
}


def _frozen(array):
    array = np.asarray(array)
    array.flags.writeable = False
    return array


def _imported(name, method):
    """The library ``name``, imported for ``method``, which needs it."""
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(f"{method}: {name} is not installed; pip install {name}") from None


def _outside(said, given, rows):
    """The refusal of row ``given``, as ``said`` names it, of a table of ``rows`` rows."""
    if not rows:
        return f"{said} {given} of a table with no rows; check len(table) first"
    return (f"{said} {given} of a table of {rows:,} rows; give one from {-rows:,} to "
            f"{rows - 1:,}")


class Table:
    """One of a run's tables, which any Arrow reader takes through the C stream
    interface, as ``pyarrow.table(table)``, ``polars.DataFrame(table)`` or a DuckDB
    query that names it, with nothing imported by probatum.

    ``table[name]``, or ``table.name``, is a column as a read-only numpy array: floats,
    NaN where a value is missing; integers as int64 and times as int64 nanoseconds since
    1970, plain where none can be missing and masked where one can, so a missing value
    is masked rather than read; booleans; and text as objects, None where it is missing.
    ``table[i]`` is row ``i`` as a dict, from the end where ``i`` is negative, None
    where a value is missing, and a slice, a boolean mask or an array of positions gives
    a table of those rows, a masked position refused, since it names no row; iterating
    gives the rows in order, ``len(table)`` counts them and ``columns`` names the
    columns. ``to_pandas()`` and ``to_polars()`` make a frame of it. Two tables are
    equal where their columns and their values are."""

    __slots__ = ("_held", "_kinds", "_read")


    def __init__(self, held):
        self._held  = held
        self._kinds = {name: (kind, nullable) for name, kind, nullable in held.kinds}
        self._read  = {}


    def __copy__(self):
        return self


    def __deepcopy__(self, memo):
        return self


    @property
    def columns(self):
        """The columns' names, in order."""
        return tuple(self._kinds)


    def __len__(self):
        return len(self._held)


    def _column(self, name, kept=True):
        """Column ``name``, kept for the next read unless ``kept`` is false, as a
        conversion that reads every column once asks."""
        if name not in self._kinds:
            raise KeyError(f"table: no column {name!r}; use one of "
                           f"{', '.join(self._kinds)}")

        read = self._read.get(name)
        if read is None:
            kind, nullable = self._kinds[name]
            masked  = nullable and kind in ("int", "time")
            missing = None
            if masked:
                valid = self._held.valid(name)
                if valid is not None:
                    missing = ~valid
                    missing.flags.writeable = False

            read = (self._held.column(name), masked, missing)
            if kept:
                self._read[name] = read

        values, masked, missing = read
        if masked:
            # each read a masked array of its own over a read-only mask, so no caller's
            # change to one reaches the next
            return np.ma.MaskedArray(values, mask=np.ma.nomask if missing is None
                                     else missing, copy=False)
        return values


    def _rows(self, rows):
        return type(self)(self._held.take(np.ascontiguousarray(rows, dtype=np.int64)))


    def __getitem__(self, key):
        if isinstance(key, str):
            return self._column(key)

        rows = len(self._held)

        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            at = int(key) + rows if key < 0 else int(key)
            if not 0 <= at < rows:
                raise IndexError(_outside("row:", int(key), rows))
            return self._held.row(at)

        if isinstance(key, slice):
            return self._rows(np.arange(*key.indices(rows), dtype=np.int64))

        if isinstance(key, np.ma.MaskedArray) and key.dtype.kind in "iu":
            masked = np.ma.count_masked(key)
            if masked:
                raise IndexError(f"key: positions of which {masked:,} are masked, and a "
                                 f"masked position names no row; give key.compressed()")

        picked = (key.filled(False) if isinstance(key, np.ma.MaskedArray)
                  else np.asarray(key))

        if picked.dtype == bool and picked.ndim == 1:
            if len(picked) != rows:
                raise IndexError(f"key: a mask of {len(picked):,} rows for a table of "
                                 f"{rows:,}; give one a row")
            return self._rows(np.flatnonzero(picked))

        if picked.ndim == 1 and (picked.dtype.kind in "iu" or not picked.size):
            outside = (picked < -rows) | (picked >= rows)
            if outside.any():
                raise IndexError(_outside("key: position", int(picked[outside][0]), rows))
            at = picked.astype(np.int64)
            return self._rows(np.where(at < 0, at + rows, at))

        raise TypeError(f"key: {key!r}; give a column's name, a row's position, a slice "
                        f"of rows, or a boolean mask or positions of them")


    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        try:
            return self._column(name)
        except KeyError:
            raise AttributeError(f"table: no column {name!r}; use one of "
                                 f"{', '.join(self.columns)}") from None


    def __iter__(self):
        held = self._held
        return (held.row(at) for at in range(len(held)))


    def __contains__(self, item):
        if isinstance(item, str):
            raise TypeError(f"{item!r} in table: a table holds rows, as dicts; ask "
                            f"{item!r} in table.columns")
        return any(row == item for row in self)


    def __eq__(self, other):
        return type(other) is type(self) and self._held == other._held

    __hash__ = None


    def __repr__(self):
        return f"{type(self).__name__}({len(self):,} rows: {', '.join(self.columns)})"


    def __reduce__(self):
        return _rebuilt, (type(self), [tuple(part) for part in self._held.parts()])


    def __arrow_c_stream__(self, requested_schema=None):
        return self._held.__arrow_c_stream__(requested_schema)


    def to_pandas(self):
        """The table as a pandas DataFrame, made from its columns with no pyarrow: times
        as UTC datetimes, NaT where one is missing, and an integer column that can miss
        a value as pandas' nullable Int64."""
        pd  = _imported("pandas", "to_pandas")
        out = {}

        for name, kind, nullable, values, valid, _ in self._held.parts():
            if kind == "time":
                out[name] = pd.DatetimeIndex(values.view("M8[ns]")).tz_localize("UTC")
            elif kind == "int" and nullable:
                missing = (np.zeros(len(values), dtype=bool) if valid is None
                           else ~valid)
                out[name] = pd.arrays.IntegerArray(values.copy(), missing)
            elif kind == "text":
                out[name] = self._column(name, kept=False)
            else:
                out[name] = values

        return pd.DataFrame(out)


    def to_polars(self):
        """The table as a polars DataFrame, read through the Arrow C stream."""
        return _imported("polars", "to_polars").DataFrame(self)


class OrderLog(Table):
    """The order log, a `Table` whose ``events(kind)`` keeps one kind of its events."""

    __slots__ = ()


    def events(self, kind):
        """The events of ``kind``, one of placed, armed, clamped, refused, not_filled,
        expired, cancelled and filled, as an order log of their own, in order."""
        if kind not in KINDS:
            raise ValueError(f"kind: {kind!r}; use one of {', '.join(KINDS)}")
        return self[self._held.equal("kind", kind)]


def _rebuilt(kind, parts):
    """A `Table` of the type ``kind`` made again from its ``parts``, as pickling does."""
    return kind(_Held(parts))


def _tables(held):
    """The binding's tables ``held``, by name, as a Result holds them."""
    return {name: (OrderLog if name == "orders" else Table)(held[name])
            for name in SCHEMAS}


class Result:
    """What one run recorded, on the rows its account saw from the first: ``time``,
    each row's open in int64 nanoseconds since 1970, with ``equity`` and ``returns``,
    each ``(T,)``; the tables ``ledger``, a row a bar, ``holdings``, a row a bar and
    instrument, ``fills``, the fill log, ``orders``, the order log, and ``trades``, the
    round trips (Specification, What a run records); ``stats``, the statistics at the
    Data's annualisation and the run's risk-free rate, None where the clock is too
    short to annualise; ``policy``, the copy of the policy the run used; ``bust``,
    whether the account's equity reached zero; ``liquidations``, the number of bars it
    was liquidated on; and ``provenance``, what the run was given and where it ran. A
    row's return is its equity over the row before's, less one, and nothing once the
    equity is gone. ``save(path)`` writes it as files any tool reads, and
    ``Result.load(path)`` reads it back."""


    def __init__(self, time, equity, tables, stats, policy, bust, provenance):
        self.time   = _frozen(np.array(time, dtype=np.int64))
        self.equity = _frozen(np.array(equity, dtype=np.float64))
        before      = self.equity[:-1]
        grown       = np.divide(self.equity[1:], before, out=np.ones_like(before),
                                where=before > 0.0)
        self.returns    = _frozen(np.concatenate([[0.0], grown - 1.0]))
        self.ledger     = tables["ledger"]
        self.holdings   = tables["holdings"]
        self.fills      = tables["fills"]
        self.orders     = tables["orders"]
        self.trades     = tables["trades"]
        self.stats      = stats
        self.policy     = _frozen(policy) if isinstance(policy, np.ndarray) else policy
        self.bust       = bool(bust)
        self.provenance = provenance

        hit               = self.fills._held.equal("reason", "liquidation")
        self.liquidations = len(np.unique(self.fills["bar"][hit]))


    @classmethod
    def _of(cls, held, stats, policy, bust, provenance):
        """The Result of a run whose tables the binding built, ``held``."""
        tables = _tables(held)
        ledger = tables["ledger"]
        return cls(ledger["time"], ledger["equity"], tables, stats, policy, bust,
                   provenance)


    def __repr__(self):
        given = self.provenance
        return (f"Result({given['policy']} on {given['venue']}, {len(self.time):,} bars, "
                f"equity {self.equity[0]:,.2f} to {self.equity[-1]:,.2f})")


    def __copy__(self):
        copied = object.__new__(Result)
        copied.__dict__.update(self.__dict__)
        return copied


    def __deepcopy__(self, memo):
        # its arrays and tables never change, so a copy shares them
        copied = memo[id(self)] = self.__copy__()

        for name in ("stats", "policy", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                setattr(copied, name, copy.deepcopy(value, memo))

        return copied


    def __reduce__(self):
        # a pickled Strategy's copy or loop is described in the provenance, not kept
        policy = self.policy if isinstance(self.policy, np.ndarray) else None
        tables = {name: getattr(self, name) for name in SCHEMAS}
        return Result, (self.time, self.equity, tables, self.stats, policy, self.bust,
                        self.provenance)


    def to_pandas(self):
        """``equity`` and ``returns`` as a pandas DataFrame on a UTC DatetimeIndex named
        time."""
        pd    = _imported("pandas", "to_pandas")
        index = pd.DatetimeIndex(self.time.view("M8[ns]"), name="time").tz_localize("UTC")
        return pd.DataFrame({"equity": self.equity, "returns": self.returns}, index=index)


    def to_polars(self):
        """``time``, in UTC, ``equity`` and ``returns`` as a polars DataFrame."""
        pl = _imported("polars", "to_polars")
        return pl.DataFrame(_Held([("time", "time", False, self.time, None, None),
                                   ("equity", "float", True, self.equity, None, None),
                                   ("returns", "float", True, self.returns, None,
                                    None)]))


    def to_dict(self):
        """This Result as plain data: ``time``, ``equity`` and ``returns`` as lists,
        each table as a dict of its columns as lists, None where a value is missing,
        and ``stats``, a ratio infinite where it earned something against no measured
        risk, ``policy`` where it is an array, ``bust``, ``liquidations`` and
        ``provenance``."""
        out = dict(time=self.time.tolist(), equity=self.equity.tolist(),
                   returns=self.returns.tolist())

        for name in SCHEMAS:
            table = getattr(self, name)
            out[name] = {column: _listed(table._column(column, kept=False))
                         for column in table.columns}

        out.update(stats=None if self.stats is None else dict(self.stats),
                   policy=(_listed(self.policy) if isinstance(self.policy, np.ndarray)
                           else None),
                   bust=self.bust, liquidations=self.liquidations,
                   provenance=json.loads(json.dumps(self.provenance)))

        return out


    def save(self, path):
        """Write this Result into the directory ``path``, made where it is missing, as
        files any tool reads (ADR 0026): ``result.json``, the manifest, holding the
        format it is written in, the version of probatum that wrote it, the provenance,
        whether the policy was an array and whether the account went bust; ``time.npy``,
        ``equity.npy`` and ``returns.npy``; ``policy.npy`` where the policy was an
        array; each table as an Arrow IPC file, ``ledger.arrow``, ``holdings.arrow``,
        ``fills.arrow``, ``orders.arrow`` and ``trades.arrow``; and ``stats.arrow``, the
        statistics as one row, none where the clock was too short to annualise. Every
        file is written under another name first, all of them removed again where one
        cannot be written, then the manifest already there and any policy.npy go, the
        files are put in place, and the manifest last, so a save that an error or a
        killed process stops leaves either the Result that was there or a directory with
        no manifest, which load refuses."""
        folder = pathlib.Path(path)
        if folder.exists() and not folder.is_dir():
            raise ValueError(f"path: {str(path)!r} is a file; name a directory")
        folder.mkdir(parents=True, exist_ok=True)

        array  = self.policy if isinstance(self.policy, np.ndarray) else None
        arrays = dict(time=self.time, equity=self.equity, returns=self.returns)
        if array is not None:
            arrays["policy"] = array
        files = ([f"{name}.npy" for name in arrays] + [f"{name}.arrow" for name in SCHEMAS]
                 + ["stats.arrow", "result.json"])
        stats    = self.stats or {}
        manifest = dict(format=FORMAT, version=_version(),
                        policy=None if array is None else "array", bust=self.bust,
                        provenance=self.provenance)

        try:
            for name, values in arrays.items():
                with open(folder / f"{name}.npy.partial", "wb") as file:
                    np.save(file, np.ascontiguousarray(values), allow_pickle=False)
            for name in SCHEMAS:
                _ipc.write(folder / f"{name}.arrow.partial", _stored(getattr(self, name)),
                           {})
            _ipc.write(folder / "stats.arrow.partial",
                       [(key, "int64" if key in COUNTS else "float",
                         np.array([stats[key]] if stats else [],
                                  dtype=np.int64 if key in COUNTS else np.float64),
                         None, False) for key in STATS], {})
            with open(folder / "result.json.partial", "w", encoding="utf-8") as file:
                json.dump(manifest, file, allow_nan=False, indent=1)
        except BaseException:
            for name in files:
                (folder / f"{name}.partial").unlink(missing_ok=True)
            raise

        (folder / "result.json").unlink(missing_ok=True)
        (folder / "policy.npy").unlink(missing_ok=True)
        for name in files:
            os.replace(folder / f"{name}.partial", folder / name)


    @classmethod
    def load(cls, path):
        """The Result ``save`` wrote into the directory ``path``, read by its format,
        never by the class that wrote it, so a Result one version saves opens in the
        next. Refused are a directory with no manifest, a format this version does not
        read, and files that are not the ones save writes or do not agree with one
        another. The policy comes back for an array run, read-only, and as None for any
        other, which its provenance still describes."""
        folder = pathlib.Path(path)
        fix    = "give a directory a Result was saved into"
        if not (folder / "result.json").is_file():
            raise ValueError(f"path: {str(path)!r} holds no result.json; {fix}")

        try:
            with open(folder / "result.json", encoding="utf-8") as file:
                manifest = json.load(file, parse_constant=_constant)
        except ValueError:
            manifest = None

        given = manifest.get("format") if isinstance(manifest, dict) else None
        if type(given) is int and given != FORMAT:
            raise ValueError(f"path: {str(path)!r} holds a Result saved in format "
                             f"{given} by probatum {manifest.get('version')}, and this "
                             f"version, {_version()}, reads format {FORMAT}; load it "
                             f"with the version that saved it, or a later one")

        if (type(given) is not int or not isinstance(manifest.get("provenance"), dict)
                or not isinstance(manifest.get("bust"), bool)
                or manifest.get("policy", "") not in (None, "array")
                or not _described(manifest["provenance"])):
            raise ValueError(f"path: {str(path)!r}'s result.json is not a manifest save "
                             f"writes; {fix}")

        provenance = manifest["provenance"]
        arrays     = {name: _array(path, folder, name, fix)
                      for name in ("time", "equity", "returns")}
        if manifest["policy"] == "array":
            arrays["policy"] = _array(path, folder, "policy", fix)

        time, equity, policy = arrays["time"], arrays["equity"], arrays.get("policy")
        if (time.dtype != np.int64 or equity.dtype != np.float64 or time.ndim != 1
                or not len(time) or equity.shape != time.shape
                or arrays["returns"].shape != time.shape
                or (policy is not None and (policy.dtype != np.float64
                                            or policy.ndim not in (1, 2)))):
            raise ValueError(f"path: the arrays of {str(path)!r} are not a row each of "
                             f"one clock as save writes them; {fix}")

        rows   = len(time)
        tables = {name: _loaded(path, folder, name, fix) for name in SCHEMAS}
        ledger, holdings = tables["ledger"], tables["holdings"]
        bars = [table[column] for table in tables.values() for column in BARS
                if column in table.columns]
        if (len(ledger) != rows or not np.array_equal(ledger["time"], time)
                or not np.array_equal(ledger["equity"], equity)
                or any(np.ma.any((column < 0) | (column >= rows)) for column in bars)
                or any(not np.array_equal(table["time"], time[table["bar"]])
                       for table in (holdings, tables["fills"], tables["orders"]))
                or not _laid_out(holdings, rows)):
            raise ValueError(f"path: the tables of {str(path)!r} do not hold the rows of "
                             f"its time.npy and equity.npy; {fix}")

        stats = _stats(path, folder, fix)
        if stats is not None and (
                stats["num_fills"] != len(tables["fills"])
                or stats["num_trades"] != np.count_nonzero(~tables["trades"]["open"])):
            raise ValueError(f"path: {str(path)!r}'s stats.arrow does not count the fills "
                             f"and the round trips its tables hold; {fix}")

        loaded = cls(time, equity, tables, stats, policy, manifest["bust"], provenance)
        if not np.array_equal(loaded.returns, arrays["returns"]):
            raise ValueError(f"path: {str(path)!r}'s returns.npy does not hold the returns "
                             f"its equity.npy gives; {fix}")

        return loaded


def _listed(values):
    """``values``, a column or an array, as a list, None where a value is missing."""
    if isinstance(values, np.ma.MaskedArray) or values.dtype.kind != "f":
        return values.tolist()
    return np.where(np.isnan(values), None, values.astype(object)).tolist()


@functools.cache
def _version():
    try:
        return importlib.metadata.version("probatum")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _constant(token):
    raise ValueError(f"{token} is not JSON")


def _stored(table):
    """``table``'s columns as `_ipc.write` takes them."""
    out = []
    for name, kind, nullable, values, valid, data in table._held.parts():
        if kind == "text":
            values = (values, data)
        out.append((name, STORED[kind], values, valid, nullable))

    return out


def _array(path, folder, name, fix):
    """The array ``name``.npy of ``folder``, the directory ``path`` names, refused where
    it is missing or is no array of numbers."""
    file = folder / f"{name}.npy"
    if not file.is_file():
        raise ValueError(f"path: {str(path)!r} holds no {name}.npy; {fix}")

    try:
        with open(file, "rb") as handle:
            values = np.load(handle, allow_pickle=False)
    except (ValueError, EOFError):
        values = None

    if not isinstance(values, np.ndarray):
        raise ValueError(f"path: {str(path)!r}'s {name}.npy is not an array of numbers "
                         f"save writes; {fix}")

    return values


def _read(path, folder, name, fix):
    """The columns of the Arrow IPC file ``name`` of ``folder``, the directory ``path``
    names, refused where it is missing or unreadable."""
    file = folder / name
    if not file.is_file():
        raise ValueError(f"path: {str(path)!r} holds no {name}; {fix}")

    try:
        columns, _ = _ipc.read(file)
    except _ipc.Refused as error:
        raise ValueError(f"path: {str(path)!r}'s {name} {error}; {fix}") from None

    return columns


def _loaded(path, folder, name, fix):
    """The table ``name``.arrow of ``folder``, the directory ``path`` names, refused
    where it lacks a column save writes or holds a null in a column that holds none,
    its columns in this version's order; a column a later version added comes back
    after them as it was written."""
    columns  = _read(path, folder, f"{name}.arrow", fix)
    schema   = SCHEMAS[name]
    kinds    = {stored: kind for kind, stored in STORED.items()}
    nullable = {column: can for column, _, can in schema}

    for column, kind, _ in schema:
        if columns.get(column, ("",))[0] != STORED[kind]:
            raise ValueError(f"path: {str(path)!r}'s {name}.arrow has no {kind} column "
                             f"{column}; {fix}")

    parts = []
    for column in [column for column, _, _ in schema] + [
            column for column in columns if column not in nullable]:
        stored, values, present, written = columns[column]
        if stored not in kinds:
            raise ValueError(f"path: {str(path)!r}'s {name}.arrow holds a column "
                             f"{column} of a kind save does not write; {fix}")

        can = nullable.get(column, written)
        if not (can or present.all()):
            raise ValueError(f"path: {str(path)!r}'s {name}.arrow holds a null in "
                             f"{column}, a column that holds none; {fix}")

        data = None
        if stored == "large_text":
            values, data = values

        parts.append((column, kinds[stored], can, values,
                      None if present.all() else present, data))

    try:
        held = _Held(parts)
    except ValueError:
        raise ValueError(f"path: {str(path)!r}'s {name}.arrow is not a table save "
                         f"writes; {fix}") from None

    return (OrderLog if name == "orders" else Table)(held)


def _stats(path, folder, fix):
    """The statistics ``stats.arrow`` holds, None where it holds no row, refused where
    one is missing; a statistic a later version added is left out."""
    columns = _read(path, folder, "stats.arrow", fix)
    counts  = {len(columns[key][1]) for key in STATS if key in columns}

    if (any(columns.get(key, ("",))[0] != ("int64" if key in COUNTS else "float")
            for key in STATS) or len(counts) != 1 or counts.pop() > 1
            or not all(columns[key][2].all() for key in STATS)):
        raise ValueError(f"path: {str(path)!r}'s stats.arrow is not the row of "
                         f"statistics save writes; {fix}")

    if not len(columns[STATS[0]][1]):
        return None
    return {key: (int if key in COUNTS else float)(columns[key][1][0]) for key in STATS}


def _described(provenance):
    """Whether ``provenance`` holds what a Result's repr reads of it."""
    return "venue" in provenance and "policy" in provenance


def _laid_out(holdings, rows):
    """Whether ``holdings`` hold a row a bar and instrument over ``rows`` bars, each
    bar's instruments in one order, as a run records them."""
    width = len(holdings) // rows
    names = holdings["instrument"]

    return (width > 0 and len(holdings) == rows * width
            and np.array_equal(holdings["bar"], np.repeat(np.arange(rows), width))
            and bool((names.reshape(rows, width) == names[:width]).all()))


def _environment():
    """What a provenance says of where a run ran: the version of every installed
    distribution behind a module it had loaded, read from their metadata so nothing is
    imported to ask, the Python version, the platform and the tags of the wheel
    probatum was installed from."""
    owners   = _owners()
    packages = {}

    # a name sys.modules holds as None is one whose import was blocked or failed
    for name in {name.partition(".")[0] for name, module in list(sys.modules.items())
                 if module is not None}:
        for owner in owners.get(name, ()):
            version = _version_of(owner)
            if version is not None:
                packages[owner] = version

    wheel = _wheel()
    return dict(packages=dict(sorted(packages.items())),
                python=platform.python_version(), platform=_platform(),
                wheel=None if wheel is None else list(wheel))


@functools.cache
def _owners():
    return importlib.metadata.packages_distributions()


@functools.cache
def _version_of(owner):
    try:
        return importlib.metadata.version(owner)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.cache
def _platform():
    return platform.platform()


@functools.cache
def _wheel():
    """The tags of the wheel probatum was installed from, its ABI among them, or None
    where it came from none."""
    try:
        said = importlib.metadata.distribution("probatum").read_text("WHEEL")
    except importlib.metadata.PackageNotFoundError:
        return None

    tags = tuple(line.partition(":")[2].strip() for line in (said or "").splitlines()
                 if line.startswith("Tag:"))
    return tags or None


def _provenance(data, venue, markets, policy, risk_free, rate, seed=None, **given):
    """What a Result's provenance records of a run of ``policy``, described, on
    ``data`` at ``venue``, a Venue, its instruments on ``markets``, at the annual rate
    ``risk_free`` and the annualisation ``rate``, with ``seed`` and anything ``given``
    of the policy's settings."""
    streams = {"intrabar": "ChaCha20"} if venue._intrabar == "random" else {}
    # the binding calls a linear perpetual perp, where every other surface says linear
    named = ["linear" if market == "perp" else market for market in markets]

    return dict(data=data.provenance, venue=repr(venue), venue_name=venue.name,
                markets=named, settlement=_settlement(data.instruments, markets),
                policy=policy, **given, seed=seed, streams=streams,
                risk_free=float(risk_free), periods_per_year=rate, version=_version(),
                **_environment())
