"""What one run produced, whatever its policy was: its equity and returns as arrays, its
records as tables any Arrow reader takes, saved as files any tool reads, and books kept
at several venues added into one (ADR 0026)."""

import collections.abc
import copy
import functools
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import sys
import textwrap

import numpy as np

from . import _assumptions, _ipc
from ._data import DAY, INTERVALS, _chosen, _when
from ._errors import (MissingDependency, ProbatumAttributeError, ProbatumIndexError,
                      ProbatumKeyError, ProbatumTypeError, ProbatumValueError)
from ._probatum import Table as _Held, ln, stats_of
from ._transitions import Transitions
from ._venue import _settlement
from .stats import Statistics, _closed, _percent, sharpe, time_under_water

# the format a saved Result is written in, which this version reads, as every later
# one will
FORMAT = 1
# each table a Result holds: its columns, of a kind each, and whether each can hold a
# null, as the binding builds them, the assumptions and the baselines as the Result
# builds them, and as a saved Result's files hold them
SCHEMAS = {
    "ledger": [("bar", "int", False), ("time", "time", False)]
    + [(name, "float", True) for name in (
        "equity", "spot_cash", "margin_cash", "gross_exposure", "net_exposure",
        "leverage", "margin_used", "maintenance", "fees", "funding", "spread",
        "slippage", "impact", "liquidation", "dust", "equity_in_quote")],
    "holdings": [("bar", "int", False), ("time", "time", False),
                 ("instrument", "text", False)]
    + [(name, "float", True) for name in (
        "position", "entry", "mark", "notional", "unrealized", "weight", "drift",
        "pnl")],
    "fills": [("bar", "int", False), ("time", "time", False),
              ("instrument", "text", False), ("order", "int", True),
              ("side", "text", False)]
    + [(name, "float", True) for name in (
        "size", "price", "reference", "spread", "slippage", "impact", "fee",
        "adjustment", "lifted", "paid_in")]
    + [("taker", "bool", False), ("reason", "text", False)],
    "orders": [("bar", "int", False), ("time", "time", False), ("kind", "text", False),
               ("order", "int", True), ("instrument", "text", True),
               ("side", "text", True), ("size", "float", True), ("price", "float", True),
               ("reason", "text", False)],
    "trades": [("instrument", "text", False), ("side", "text", False)]
    + [(name, "float", True) for name in ("size", "entry", "exit")]
    + [("open_bar", "int", False), ("close_bar", "int", True),
       ("open_time", "time", False), ("close_time", "time", True)]
    + [(name, "float", True) for name in (
        "realized", "fees", "funding", "pnl", "ret", "best", "worst")]
    + [("reason", "text", True), ("open", "bool", False)],
    "assumptions": [(name, "text", False) for name in ("name", "value", "source", "grade")]
    + [("direction", "text", True), ("cost", "float", True)],
    "baselines": [("bar", "int", False), ("time", "time", False)]
    + [(name, "float", True) for name in ("cash", "buy_and_hold", "frictionless")],
}
# the tables a book holds of its own, which name no venue
OWN = ("ledger", "baselines")
# the engine's records only a run at the full level keeps, which one below it refuses
# by name, as it does its baselines (Specification, Recording levels)
LOGGED = ("holdings", "fills", "orders")
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
# a book's ledger: what each account counts as its equity is held, what it counts only
# inside its own span, and the costs that fall on the rows they were paid on
HELD    = ("equity", "spot_cash", "margin_cash", "equity_in_quote")
SPANNED = ("gross_exposure", "net_exposure", "margin_used", "maintenance")
COSTS   = ("fees", "funding", "spread", "slippage", "impact", "liquidation", "dust")
# the assets a book counts at par, one for another, where it names one of them its own
DOLLARS = frozenset(("USD", "USDT", "USDC", "USDD", "TUSD", "FDUSD", "BUSD", "DAI",
                     "PYUSD", "USDE", "USDP", "USD1", "RLUSD"))


def _frozen(array):
    array = np.asarray(array)
    array.flags.writeable = False
    return array


def _imported(name, method):
    """The library ``name``, imported for ``method``, which needs it."""
    try:
        return importlib.import_module(name)
    except ImportError:
        raise MissingDependency(f"{method}: {name} is not installed; pip install "
                                f"{name}") from None


def _outside(said, given, rows):
    """The refusal of row ``given``, as ``said`` names it, of a table of ``rows`` rows."""
    if not rows:
        return f"{said} {given} of a table with no rows; check len(table) first"
    return (f"{said} {given} of a table of {rows:,} rows; give one from {-rows:,} to "
            f"{rows - 1:,}")


def _returns(equity):
    """Each row's return on ``equity``: its equity over the row before's, less one, and
    nothing once the equity is gone, the first row's nothing."""
    before = equity[:-1]
    grown  = np.divide(equity[1:], before, out=np.ones_like(before), where=before > 0.0)
    return np.concatenate([[0.0], grown - 1.0])


def _money(value):
    """An amount as a summary prints it: whole from a hundred, to three figures below,
    as a coin's are, and to the eighth decimal where three figures would take an
    exponent, so an amount too small for that prints as 0, never as -0."""
    if abs(value) >= 100.0:
        return f"{value:,.0f}"

    shown = f"{value:.3g}"
    if "e" in shown:
        shown = f"{value:.8f}".rstrip("0").rstrip(".")
    return "0" if shown == "-0" else shown


def _row(label, value, said):
    """One line of a summary, its value right-aligned under the others and its words
    wrapped at 88 columns beneath themselves."""
    return textwrap.fill(f"  {label:<16}{value:>8}   {said}", width=88,
                         subsequent_indent=" " * 29, break_long_words=False,
                         break_on_hyphens=False)


def _recorded(provenance):
    """The level the run ``provenance`` describes was recorded at, full where it names
    none, as a book's, a loop's and a run's saved before the levels do."""
    return provenance.get("record", "full")


def _unkept(result, said, what):
    """The refusal of ``said``, which reads ``what`` a run below the full level does not
    keep, of ``result``."""
    return (f"{said}: a run recorded at {_recorded(result.provenance)!r} keeps {what}; "
            f"run it with record=\"full\"")


def _untaken(result, said):
    """The refusal of ``said``, which reads the transitions ``result`` did not record."""
    return (f"{said}: a run recorded at {_recorded(result.provenance)!r} keeps no "
            f"transitions; backtest an agent with record=\"transitions\"")


class Table:
    """One of a run's tables, which any Arrow reader takes through the C stream
    interface, as ``pyarrow.table(table)``, ``polars.DataFrame(table)`` or a DuckDB
    query that names it, with nothing imported by probatum.

    ``table[name]``, or ``table.name``, is a column as a read-only numpy array: floats,
    NaN where a value is missing; integers as int64, times as int64 nanoseconds since
    1970 and booleans, plain where none can be missing and masked where one can, so a
    missing value is masked rather than read; and text as objects, None where it is
    missing. ``table[i]`` is row ``i`` as a dict, from the end where ``i`` is negative,
    None where a value is missing, and a slice, a boolean mask or an array of positions
    gives a table of those rows, a masked position refused, since it names no row;
    iterating gives the rows in order, ``len(table)`` counts them and ``columns`` names
    the columns. ``to_pandas()`` and ``to_polars()`` make a frame of it. Two tables are
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
            raise ProbatumKeyError(f"table: no column {name!r}; use one of "
                                   f"{', '.join(self._kinds)}")

        read = self._read.get(name)
        if read is None:
            kind, nullable = self._kinds[name]
            masked  = nullable and kind in ("int", "time", "bool")
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
                raise ProbatumIndexError(_outside("row:", int(key), rows))
            return self._held.row(at)

        if isinstance(key, slice):
            return self._rows(np.arange(*key.indices(rows), dtype=np.int64))

        if isinstance(key, np.ma.MaskedArray) and key.dtype.kind in "iu":
            masked = np.ma.count_masked(key)
            if masked:
                raise ProbatumIndexError(f"key: positions of which {masked:,} are "
                                         f"masked, and a masked position names no row; "
                                         f"give key.compressed()")

        picked = (key.filled(False) if isinstance(key, np.ma.MaskedArray)
                  else np.asarray(key))

        if picked.dtype == bool and picked.ndim == 1:
            if len(picked) != rows:
                raise ProbatumIndexError(f"key: a mask of {len(picked):,} rows for a "
                                         f"table of {rows:,}; give one a row")
            return self._rows(np.flatnonzero(picked))

        if picked.ndim == 1 and (picked.dtype.kind in "iu" or not picked.size):
            outside = (picked < -rows) | (picked >= rows)
            if outside.any():
                raise ProbatumIndexError(_outside("key: position",
                                                  int(picked[outside][0]), rows))
            at = picked.astype(np.int64)
            return self._rows(np.where(at < 0, at + rows, at))

        raise ProbatumTypeError(f"key: {key!r}; give a column's name, a row's "
                                f"position, a slice of rows, or a boolean mask or "
                                f"positions of them")


    def __getattr__(self, name):
        if name.startswith("_"):
            raise ProbatumAttributeError(name)

        try:
            return self._column(name)
        except KeyError:
            raise ProbatumAttributeError(f"table: no column {name!r}; use one of "
                                         f"{', '.join(self.columns)}") from None


    def __iter__(self):
        held = self._held
        return (held.row(at) for at in range(len(held)))


    def __contains__(self, item):
        if isinstance(item, str):
            raise ProbatumTypeError(f"{item!r} in table: a table holds rows, as dicts; "
                                    f"ask {item!r} in table.columns")
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
        as UTC datetimes, NaT where one is missing, and an integer or a boolean column
        that can miss a value as pandas' nullable Int64 or boolean."""
        pd  = _imported("pandas", "to_pandas")
        out = {}

        for name, kind, nullable, values, valid, _ in self._held.parts():
            if kind == "time":
                out[name] = pd.DatetimeIndex(values.view("M8[ns]")).tz_localize("UTC")
            elif kind in ("int", "bool") and nullable:
                missing = (np.zeros(len(values), dtype=bool) if valid is None
                           else ~valid)
                made = (pd.arrays.IntegerArray if kind == "int"
                        else pd.arrays.BooleanArray)
                out[name] = made(values.copy(), missing)
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
        _chosen("kind", kind, KINDS, f"use one of {', '.join(KINDS)}")
        return self[self._held.equal("kind", kind)]


class Assumptions(Table):
    """What a run assumed (Specification, Assumptions), a `Table` of a row for each
    modelling choice: its ``name``, its ``value``, its ``source``, its ``grade``,
    ``venue``, ``simplified`` or ``optimistic``, the ``direction`` its error leans where
    it is simplified, and its ``cost`` where the run's ledger and fill log measure it,
    or below the full level its ledger and round trips, in the asset the account
    settles in. Its repr shows every row, and
    ``table[table.grade == "optimistic"]`` keeps the optimistic ones."""

    __slots__ = ()


    @classmethod
    def _of(cls, rows):
        """The table of ``rows``, each its columns' values in order, None where one is
        missing."""
        columns = list(zip(*rows)) or [()] * len(SCHEMAS["assumptions"])
        parts   = []

        for (name, kind, nullable), values in zip(SCHEMAS["assumptions"], columns):
            if kind == "text":
                ends, valid, data = _text(values)
                parts.append((name, kind, nullable, ends, valid, data))
            else:
                floats = np.array([np.nan if value is None else value for value in values],
                                  dtype=np.float64)
                parts.append((name, kind, nullable, floats, None, None))

        return cls(_Held(parts))


    def __repr__(self):
        lines = [list(self.columns)] + [
            ["" if value is None else f"{value:,.6g}" if isinstance(value, float)
             else value for value in row.values()] for row in self]
        widths = [max(len(line[k]) for line in lines) for k in range(len(lines[0]))]
        return "\n".join("  ".join(text.ljust(width) for text, width in zip(line, widths))
                         .rstrip() for line in lines)


CLASSES = {"orders": OrderLog, "assumptions": Assumptions}


def _rebuilt(kind, parts):
    """A `Table` of the type ``kind`` made again from its ``parts``, as pickling does."""
    return kind(_Held(parts))


def _text(values):
    """A text column of ``values`` as the binding takes one: each row's end in its bytes
    after a leading zero, the rows that hold a text, None where every row does, and the
    bytes."""
    encoded = [b"" if value is None else value.encode() for value in values]
    ends    = np.cumsum([0] + [len(each) for each in encoded], dtype=np.int64)
    valid   = (None if all(value is not None for value in values)
               else np.array([value is not None for value in values]))
    return ends, valid, np.frombuffer(b"".join(encoded), dtype=np.uint8)


def _tables(held):
    """The binding's tables ``held``, by name, as a Result holds them."""
    return {name: CLASSES.get(name, Table)(table) for name, table in held.items()}


def _baselined(time, cash, held, frictionless):
    """The baselines of a Result on the rows ``time``: ``cash`` on every row, and on
    each the equity ``held`` and ``frictionless`` give."""
    rows    = len(time)
    columns = dict(bar=np.arange(rows, dtype=np.int64),
                   time=np.ascontiguousarray(time, dtype=np.int64),
                   cash=np.full(rows, cash, dtype=np.float64),
                   buy_and_hold=np.ascontiguousarray(held, dtype=np.float64),
                   frictionless=np.ascontiguousarray(frictionless, dtype=np.float64))

    return Table(_Held([(name, kind, nullable, columns[name], None, None)
                        for name, kind, nullable in SCHEMAS["baselines"]]))


class _Record:
    """What a run's Result keeps of its run, which its assumptions and its baselines
    read, shared by every copy of the Result: the ``venue``, the ``data``, and the
    Engine's ``settings`` with ``start``, the row of the Data's first bar among the
    Engine's, as `_run` gave them; the ``token`` of a run whose account handed out
    handles, which explain checks one against; the ``seed`` of a loop the caller
    stepped; and the ``policy`` to play again without costs, an array's arguments, the
    copy of a Strategy taken at the call or an agent's run, a function of the change to
    its settings that returns its equity, until the baselines are ``computed``, with
    whether that play is ``playing``."""

    __slots__ = ("venue", "data", "settings", "start", "token", "seed", "policy",
                 "computed", "playing")


    def __init__(self, venue, data, run, token=None, seed=None, policy=None):
        self.venue, self.data = venue, data
        self.settings, self.start = run.settings, run.start
        self.token, self.seed, self.policy = token, seed, policy
        self.computed, self.playing = None, False


class InstrumentResult:
    """One instrument's record of a Result, as ``result[name]`` gives it: its ``name``;
    ``position``, ``weight`` and ``pnl``, each ``(T,)`` on the Result's rows and
    read-only, the weight over its account's own equity and the pnl what it has made
    since the first row (Specification, The ledger); and its own ``fills`` and
    ``trades``, the rows of the Result's that are its."""

    __slots__ = ("name", "position", "weight", "pnl", "fills", "trades")


    def __init__(self, name, position, weight, pnl, fills, trades):
        self.name     = name
        self.position = _frozen(position)
        self.weight   = _frozen(weight)
        self.pnl      = _frozen(pnl)
        self.fills    = fills
        self.trades   = trades


    def __repr__(self):
        return (f"InstrumentResult({self.name}, {len(self.pnl):,} bars, pnl "
                f"{self.pnl[-1]:,.2f})")


class Result:
    """What one run recorded, on the rows its account saw from the first: ``time``,
    each row's open in int64 nanoseconds since 1970, with ``equity`` and ``returns``,
    each ``(T,)``; the tables ``ledger``, a row a bar, ``holdings``, a row a bar and
    instrument, ``fills``, the fill log, ``orders``, the order log, and ``trades``, the
    round trips (Specification, What a run records); ``costs``, the totals of the
    ledger's costs; ``assumptions``, what the run assumed, graded; ``stats``, the
    statistics at the Data's annualisation and the run's risk-free rate, None where the
    clock is too short to annualise; ``policy``, the copy of the policy the run used;
    ``bust``, whether the account's equity reached zero; ``liquidations``, the number
    of bars it was liquidated on, each account of a book counted apart; and
    ``provenance``, what the run was given and where it ran. A row's return is its
    equity over the row before's, less one, and nothing once the equity is gone.
    ``result[name]`` is one instrument's record, and with one instrument the Result
    itself answers ``position``, ``weight`` and ``pnl``. A run recorded at the account
    level keeps its ledger, its round trips, its costs and its statistics, the full
    level's, its assumptions priced where those measure them, and no policy, and refuses
    by name its holdings, its fills, its orders, its baselines, an instrument's record
    and ``explain``; an agent's run recorded at the transitions level keeps the same
    and ``transitions``, each step of the agent's episode, which ``export("arrow",
    path)`` writes. Its repr names every assumption graded optimistic, and a level below the
    full one. ``save(path)`` writes it as files any tool reads, and
    ``Result.load(path)`` reads it back."""

    # a Result is no sequence of its instruments, which it names rather than counts
    __iter__ = None


    def __init__(self, time, equity, tables, stats, policy, bust, provenance,
                 record=None, transitions=None):
        self._transitions = transitions
        self.time         = _frozen(np.array(time, dtype=np.int64))
        self.equity       = _frozen(np.array(equity, dtype=np.float64))
        self.returns      = _frozen(_returns(self.equity))
        self.ledger       = tables["ledger"]
        self._logged      = {name: tables.get(name) for name in LOGGED}
        self.trades       = tables["trades"]

        # a run's assumptions and baselines are read from its record when first asked
        # for and a book's baselines from its accounts'; tables holds them otherwise, an
        # unpickled Result's baselines only where they were read before it was pickled
        self._assumptions = tables.get("assumptions")
        self._baselines   = tables.get("baselines")
        self._record      = record
        self._parts       = None
        self.stats        = None if stats is None else Statistics(
            stats, self.returns[1:], _closed(self.trades), self.equity[0],
            provenance["periods_per_year"], provenance["risk_free"])

        self.policy     = _frozen(policy) if isinstance(policy, np.ndarray) else policy
        self.bust       = bool(bust)
        self.provenance = provenance

        # with no fill log, the bars it was liquidated on are the close bars of the round
        # trips forced closes ended, each on the bar it closed out
        source, bar = ((self.fills, "bar") if self._logged["fills"] is not None
                       else (self.trades, "close_bar"))
        held   = source._held
        forced = held.take(np.flatnonzero(held.equal("reason", "liquidation")))
        venue  = (forced.column("venue") if "venue" in source.columns
                  else np.zeros(len(forced)))
        self.liquidations = len(set(zip(venue.tolist(), forced.column(bar).tolist())))


    @classmethod
    def _of(cls, held, stats, policy, bust, provenance, record, transitions=None):
        """The Result of a run whose tables the binding built, ``held``, which keeps
        ``record`` of the run and an agent's ``transitions`` where it recorded them; a
        run below the full level reads its assumptions now and keeps nothing to read
        them, or baselines, from (ADR 0027)."""
        tables = _tables(held)
        ledger = tables["ledger"]
        result = cls(ledger["time"], ledger["equity"], tables, stats, policy, bust,
                     provenance, record, transitions)
        if _recorded(provenance) != "full":
            result.assumptions
            result._record = None

        return result


    @property
    def holdings(self):
        """The holdings, a row a bar and instrument, which a run below the full level
        does not keep."""
        return self._logged_table("holdings", "no holdings")


    @property
    def fills(self):
        """The fill log, which a run below the full level does not keep."""
        return self._logged_table("fills", "no fill log")


    @property
    def orders(self):
        """The order log, which a run below the full level does not keep."""
        return self._logged_table("orders", "no order log")


    def _logged_table(self, name, what):
        table = self._logged[name]
        if table is None:
            raise ProbatumAttributeError(_unkept(self, f"result.{name}", what))
        return table


    @property
    def transitions(self):
        """Each step of an agent's episode, a row each, as `Transitions`, which only an
        agent's backtest recorded at ``"transitions"`` keeps (Specification,
        Transitions)."""
        if self._transitions is None:
            raise ProbatumAttributeError(_untaken(self, "result.transitions"))
        return self._transitions


    def export(self, format, path):
        """Write the transitions, the one record this writes, as one Arrow IPC file at
        ``path``, as `Transitions.export` writes them; ``save`` writes the rest."""
        if self._transitions is None:
            raise ProbatumAttributeError(_untaken(self, "result.export"))
        self._transitions.export(format, path)


    def _table(self, name):
        """Table ``name`` as the run keeps it, None where its level keeps no part of it,
        and the baselines only where they are computed, so no policy is played."""
        if name in self._logged:
            return self._logged[name]
        return self._ready() if name == "baselines" else getattr(self, name)


    def __repr__(self):
        given = self.provenance
        what  = (f"a book of {' and '.join(given['results'])}" if "results" in given
                 else f"{given['policy']} on {given['venue']}")
        said = (f"Result({what}, {len(self.time):,} bars, equity {self.equity[0]:,.2f} "
                f"to {self.equity[-1]:,.2f}")
        if _recorded(given) != "full":
            said += f", recorded at {_recorded(given)!r}"

        table = self.assumptions
        named = [f"{row['name']} {row['value']}"
                 + (f" at {row['venue']}" if "venue" in row else "")
                 for row in table[table.grade == "optimistic"]]
        if named:
            said += (f"; {len(named)} optimistic assumption{'s' * (len(named) != 1)}: "
                     f"{'; '.join(named)}")

        return said + ")"


    def summary(self):
        """The run in a few lines of text: what ran, at which venue and on what bars,
        its span and opening balance, its return, its Sharpe with the interval and the
        lag beside the long-run Sharpe where the two differ by more than 0.1, its
        maximum drawdown as a loss with the longest time under water, its round trips,
        its costs and its optimistic assumptions; a book names its accounts, margined
        apart with no capital moving between them, and a run below the full level names
        its level (Specification, Summaries)."""
        given    = self.provenance
        book     = given.get("results")
        accounts = list(book.values()) if book else [given]
        names    = []

        for account in accounts:
            for name in (account.get("data") or {}).get("instruments", []):
                if name.split(":", 1)[-1] not in names:
                    names.append(name.split(":", 1)[-1])

        what = (f"a book of {' and '.join(book)}" if book
                else f"{given['policy']}, {given['venue_name'] or 'a venue built by hand'}")
        where    = names[0] if len(names) == 1 else f"{len(names)} instruments"
        interval = (accounts[0].get("data") or {}).get("interval", "")
        lines    = [f"{what}, {where} {interval}".rstrip()]

        first, last = (np.datetime64(int(self.time[at]), "ns").astype("datetime64[D]")
                       for at in (0, -1))
        span = (f"{first} to {last}, {len(self.time):,} bars, opening balance "
                f"{_money(self.equity[0])}")
        span += f" {given['settlement']}" if given.get("settlement") else ""
        if _recorded(given) != "full":
            span += f", recorded at the {_recorded(given)} level"
        lines.append(span)

        stats = self.stats
        if stats is None:
            lines.append(_row("statistics", "", "none, the clock too short to annualise"))
        else:
            lines.extend(self._rows(stats))

        paid = sorted(((name, value) for name, value in self.costs.items() if value),
                      key=lambda item: -abs(item[1]))
        lines.append(_row("costs", _money(-sum(self.costs.values())),
                          ", ".join(f"{name} {_money(-value)}"
                                    for name, value in paid) or "none"))

        table = self.assumptions
        named = [f"{row['name']} {row['value']}"
                 + (f" at {row['venue']}" if "venue" in row else "")
                 for row in table[table.grade == "optimistic"]]
        lines.append(_row("assumptions", "", f"{len(named)} optimistic: {', '.join(named)}"
                          if named else "none optimistic"))

        if book:
            lines.append(_row("accounts", "", f"{' and '.join(book)}, margined apart; no "
                              f"capital moves between them"))

        return "\n".join(lines)


    def report(self, path):
        """Write the run's tear sheet to ``path`` as one HTML file with no JavaScript,
        its folders made where they are missing, and return ``path``: its headline
        numbers with their intervals, each naming its method; its equity and drawdown
        beside its baselines, its returns and each month's, its rolling Sharpe and
        volatility, its exposure, each instrument's weight and pnl where it holds
        several, its costs and its round trips, as static figures; the three baselines
        with the Sharpe difference against buying and holding; its assumptions, the
        optimistic ones named at the top; and its provenance. It needs matplotlib,
        which the plot extra installs (ADR 0041)."""
        from .report import _written
        return _written("report", path, "sheet", self)


    def _rows(self, stats):
        """The summary's lines that read the run's record of statistics."""
        lines = []
        lines.append(_row("return", f"{stats['total_return_pct']:+.1f}%",
                          f"CAGR {stats['cagr_pct']:+.1f}%, volatility "
                          f"{stats['volatility_pct']:.1f}%, in the market "
                          f"{stats['exposure_pct']:.0f}% of bars"))

        held = stats.interval("sharpe")
        if held is None:
            count = len(self.time) - 1
            told  = (f"no interval: {count} returns, where one needs 4" if count < 4
                     else "no interval: the returns never vary"
                     if np.all(self.returns[1:] == self.returns[1])
                     else "no interval: its influence is zero on every bar")
        else:
            told = (f"{_percent(held.level)} interval {held.low:.2f} to {held.high:.2f}, "
                    f"delta method, Newey-West lag {held.lag}"
                    + (" floored at lag zero" if held.floored else ""))

        lines.append(_row("sharpe", f"{stats['sharpe']:.2f}", told))

        longer = sharpe(self, long_run=True)
        if abs(longer - stats["sharpe"]) > 0.1:
            lines.append(_row("long-run sharpe", f"{longer:.2f}",
                              "the mean over the returns' long-run deviation, which "
                              "their autocorrelation moves"))

        lines.append(_row("max drawdown", f"{-stats['max_drawdown_pct'] + 0.0:.1f}%",
                          f"{time_under_water(self).longest_days:,.0f} days under water"))
        lines.append(_row("round trips", f"{stats['num_trades']:,}",
                          f"win rate {stats['win_rate']:.0%}, profit factor "
                          f"{stats['profit_factor']:.2f}"))

        return lines


    @property
    def assumptions(self):
        """What the run assumed (Specification, Assumptions), as an `Assumptions` table:
        the venue's settings each instrument ran, the engine's own rules and the Data's,
        each with its value, its source and its grade, and its cost where the run's
        ledger and fill log measure it, read when first asked for, or below the full
        level where its ledger and round trips do, read as the run ends. A book's are
        its accounts', in the order they were given, each row naming its venue."""
        table = self._assumptions
        if table is None:
            table = self._assumptions = Assumptions._of(_assumptions.ran(self._record,
                                                                         self))
        return table


    @property
    def baselines(self):
        """The run set beside three baselines on its own data and venue (Specification,
        Baselines), as a table of a row per row of the Result: ``bar``, ``time``,
        ``cash``, the opening equity, ``buy_and_hold``, an equal share of the opening
        balance in every instrument bought at its first close and held, and
        ``frictionless``, the same policy with the costs of trading removed and every
        rule kept, missing for a loop the caller stepped. They are computed when first
        read, the policy played again then, an agent's as its backtest runs, so a Result
        pickled before they were computed refuses them, as a run below the full level
        does, and a book's are its accounts' summed as its equity is."""
        if _recorded(self.provenance) != "full":
            raise ProbatumAttributeError(_unkept(self, "result.baselines",
                                                 "no copy of its policy to play them"))

        table = self._ready()
        if table is None:
            from . import _baselines
            table = self._baselines = _baselined(self.time, *_baselines.read(self))
            # a book lets its accounts' records go once it holds their sum
            self._parts = None

        return table


    def _ready(self):
        """The baselines where they are computed, None where reading them would play a
        policy."""
        record = self._record
        if self._baselines is None and record is not None and record.computed:
            rows = len(self.time)
            held, frictionless = record.computed
            self._baselines = _baselined(self.time, self.equity[0], held[:rows],
                                         frictionless[:rows])

        return self._baselines


    def explain(self, item):
        """How one fill's price or one order's life came about, as text: ``item`` is a
        row of the fill log, a row of the order log, which tells that row's order, or an
        `Order` this run's account handed out, which a Result loaded, unpickled or
        combined refuses, a book's rows naming their venue (Python_API,
        Transparency); a run below the full level has none of them to tell."""
        from . import _explain
        if self._logged["fills"] is None:
            raise ProbatumValueError(_unkept(self, "explain",
                                             "no fill log or order log to tell"))
        return _explain.told(self, item)


    @property
    def costs(self):
        """The run's costs (Specification, Costs): the totals of the ledger's cost
        columns in its order, ``fees``, ``funding``, ``spread``, ``slippage``,
        ``impact``, ``liquidation`` and ``dust``, each a compensated sum, positive where
        the account paid it, in the asset it settles in, and measured from each fill's
        reference, so the move from a decision's close to the next open is price PnL.
        Each read is a dict of its own."""
        totals = self.__dict__.get("_costs")
        if totals is None:
            totals = self._costs = {name: math.fsum(self.ledger[name]) for name in COSTS}
        return dict(totals)


    def _instruments(self):
        """The instruments the holdings name, each once in the order they first appear,
        with its venue first in a book."""
        holdings = self.holdings
        names    = holdings["instrument"].tolist()
        if "venue" in holdings.columns:
            return list(dict.fromkeys(zip(holdings["venue"].tolist(), names)))
        return list(dict.fromkeys(names))


    def __getitem__(self, key):
        """One instrument's record, as an `InstrumentResult`: ``result[name]``, and in a
        book ``result[venue, name]``, or ``result[name]`` where one venue alone holds
        it. A book's series are carried from each row of the account's to the next
        inside its span, the position and the weight zero outside it and the pnl zero
        before its first row and held after its last."""
        if self._logged["holdings"] is None:
            raise ProbatumValueError(_unkept(self, f"result[{key!r}]", "no holdings, "
                                             "which an instrument's record reads"))

        holdings = self.holdings
        book     = "venue" in holdings.columns

        if (book and isinstance(key, tuple) and len(key) == 2
                and all(isinstance(part, str) for part in key)):
            venue, name = key
        elif isinstance(key, str):
            venue, name = None, key
        else:
            fix = ('its venue and its name, as result["binance", "BTCUSDT"]' if book
                   else 'its name, as result["BTCUSDT"]')
            raise ProbatumTypeError(f"result[{key!r}]: an instrument is named by a "
                                    f"string; give {fix}")

        held   = holdings._held
        picked = held.equal("instrument", name)

        if book:
            venues = {at for at, named in self._instruments() if named == name}
            if venue is None and len(venues) > 1:
                listed = sorted(venues)
                raise ProbatumKeyError(f"result[{name!r}]: {name} is held at "
                                       f"{', '.join(listed[:-1])} and {listed[-1]}; "
                                       f"give its venue, as result[\"{listed[0]}\", "
                                       f"\"{name}\"]")

            venue  = venue if venue is not None else next(iter(venues), None)
            picked = picked & held.equal("venue", "" if venue is None else venue)

        if not picked.any():
            known = ", ".join(repr(given) for given in self._instruments())
            raise ProbatumKeyError(f"result[{key!r}]: the run holds no such "
                                   f"instrument; use one of {known}")

        rows   = np.flatnonzero(picked)
        bars   = holdings["bar"][rows]
        at     = np.searchsorted(bars, np.arange(len(self.time)), side="right") - 1
        inside = (at >= 0) & (np.arange(len(self.time)) <= bars[-1])
        last   = np.maximum(at, 0)

        def carried(column, outside):
            values = holdings[column][rows][last]
            return np.where(inside, values, outside(values))

        position = carried("position", np.zeros_like)
        weight   = carried("weight", np.zeros_like)
        pnl      = np.where(at >= 0, holdings["pnl"][rows][last], 0.0)
        own      = {"instrument": name} if venue is None else {
            "venue":      venue,
            "instrument": name,
        }

        def theirs(table):
            mask = np.ones(len(table), dtype=bool)
            for column, value in own.items():
                mask &= table._held.equal(column, value)
            return table[mask]

        return InstrumentResult(name, position, weight, pnl, theirs(self.fills),
                                theirs(self.trades))


    def _one(self, field):
        """``field`` of the one instrument the run holds."""
        if self._logged["holdings"] is None:
            raise ProbatumAttributeError(_unkept(self, f"result.{field}", "no holdings"))

        instruments = self._instruments()
        if len(instruments) != 1:
            first = instruments[0]
            named = (f"\"{first[0]}\", \"{first[1]}\"" if isinstance(first, tuple)
                     else f"\"{first}\"")
            raise ProbatumAttributeError(f"result.{field}: the run holds "
                                         f"{len(instruments)} instruments; use "
                                         f"result[{named}].{field}")

        return getattr(self[instruments[0]], field)


    @property
    def position(self):
        """The one instrument's position, ``(T,)``, where the run holds one."""
        return self._one("position")


    @property
    def weight(self):
        """The one instrument's weight, ``(T,)``, over the account's equity, where the
        run holds one."""
        return self._one("weight")


    @property
    def pnl(self):
        """What the one instrument has made, ``(T,)``, where the run holds one."""
        return self._one("pnl")


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

        copied._transitions = copy.deepcopy(self._transitions, memo)
        return copied


    def __reduce__(self):
        # a pickled Strategy's copy or loop is described in the provenance, not kept,
        # nor the record its assumptions are read from, which are read first; pickling
        # plays no policy, so it carries the baselines only once they are computed
        policy = self.policy if isinstance(self.policy, np.ndarray) else None
        tables = {name: self._table(name) for name in SCHEMAS}
        stats  = None if self.stats is None else dict(self.stats)
        return Result, (self.time, self.equity, tables, stats, policy, self.bust,
                        self.provenance, None, self._transitions)


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
        the baselines None until they are computed, since this plays no policy,
        ``costs``, and ``stats``, a ratio infinite where it earned something against no
        measured risk, ``policy`` where it is an array, ``bust``, ``liquidations``,
        ``provenance`` and ``transitions``, its columns as lists beside its
        ``metadata``, None where the run kept none."""
        out = dict(time=self.time.tolist(), equity=self.equity.tolist(),
                   returns=self.returns.tolist())

        for name in SCHEMAS:
            table = self._table(name)
            out[name] = None if table is None else {
                column: _listed(table._column(column, kept=False))
                for column in table.columns}

        steps = self._transitions
        out["transitions"] = None if steps is None else {
            **{column: _listed(steps[column]) for column in steps.columns},
            "metadata": json.loads(json.dumps(steps.metadata))}

        out.update(costs=self.costs,
                   stats=None if self.stats is None else dict(self.stats),
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
        ``fills.arrow``, ``orders.arrow``, ``trades.arrow``, ``assumptions.arrow`` and
        ``baselines.arrow``, the last two read first where the run has not read them
        yet; ``stats.arrow``, the statistics as one row, none where the clock was too
        short to annualise; and ``transitions.arrow`` where an agent's run kept them, as
        ``export`` writes them. A run below the full level writes the tables it keeps
        alone. A Result pickled before its baselines were read, or a book of one, is
        refused before anything is written. Every file is written under another name
        first, all of them removed again where one cannot be written, then the manifest
        already there, any policy.npy and the tables this Result does not write go, the
        files are put in place, and the manifest last, so a save that an error or a
        killed process stops leaves either the Result that was there or a directory
        with no manifest, which load refuses."""
        full = _recorded(self.provenance) == "full"
        if full:
            try:
                self.baselines
            except AttributeError as error:
                raise ProbatumValueError(f"{error}, and save writes them, so it wrote "
                                         f"nothing") from None

        written = [name for name in SCHEMAS if full or self._table(name) is not None]
        folder  = pathlib.Path(path)
        if folder.exists() and not folder.is_dir():
            raise ProbatumValueError(f"path: {str(path)!r} is a file; name a directory")
        folder.mkdir(parents=True, exist_ok=True)

        array  = self.policy if isinstance(self.policy, np.ndarray) else None
        arrays = dict(time=self.time, equity=self.equity, returns=self.returns)
        if array is not None:
            arrays["policy"] = array

        steps = ["transitions.arrow"] if self._transitions is not None else []
        files = ([f"{name}.npy" for name in arrays] + [f"{name}.arrow" for name in written]
                 + steps + ["stats.arrow", "result.json"])

        stats    = self.stats or {}
        manifest = dict(format=FORMAT, version=_version(),
                        policy=None if array is None else "array", bust=self.bust,
                        provenance=self.provenance)

        try:
            for name, values in arrays.items():
                with open(folder / f"{name}.npy.partial", "wb") as file:
                    np.save(file, np.ascontiguousarray(values), allow_pickle=False)
            for name in written:
                _ipc.write(folder / f"{name}.arrow.partial", _stored(self._table(name)),
                           {})
            if steps:
                self._transitions.export("arrow", folder / "transitions.arrow.partial")
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
        stale = (["policy.npy"] + [f"{name}.arrow" for name in SCHEMAS
                                   if name not in written]
                 + ([] if steps else ["transitions.arrow"]))
        for name in stale:
            (folder / name).unlink(missing_ok=True)

        for name in files:
            os.replace(folder / f"{name}.partial", folder / name)


    @classmethod
    def load(cls, path):
        """The Result ``save`` wrote into the directory ``path``, read by its format,
        never by the class that wrote it, so a Result one version saves opens in the
        next. Refused are a directory with no manifest, a format this version does not
        read, and files that are not the ones save writes or do not agree with one
        another. The policy comes back for an array run recorded at the full level,
        read-only, and as None for any other, which its provenance still describes."""
        folder = pathlib.Path(path)
        fix    = "give a directory a Result was saved into"
        if not (folder / "result.json").is_file():
            raise ProbatumValueError(f"path: {str(path)!r} holds no result.json; {fix}")

        try:
            with open(folder / "result.json", encoding="utf-8") as file:
                manifest = json.load(file, parse_constant=_constant)
        except ValueError:
            manifest = None

        given = manifest.get("format") if isinstance(manifest, dict) else None
        if type(given) is int and given != FORMAT:
            raise ProbatumValueError(f"path: {str(path)!r} holds a Result saved in "
                                     f"format {given} by probatum "
                                     f"{manifest.get('version')}, and this version, "
                                     f"{_version()}, reads format {FORMAT}; load it "
                                     f"with the version that saved it, or a later one")

        if (type(given) is not int or not isinstance(manifest.get("provenance"), dict)
                or not isinstance(manifest.get("bust"), bool)
                or manifest.get("policy", "") not in (None, "array")
                or not _described(manifest["provenance"])
                or _recorded(manifest["provenance"]) not in ("full", "account",
                                                              "transitions")):
            raise ProbatumValueError(f"path: {str(path)!r}'s result.json is not a "
                                     f"manifest save writes; {fix}")

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
                                            or policy.ndim not in (1, 2, 3)))):
            raise ProbatumValueError(f"path: the arrays of {str(path)!r} are not a row "
                                     f"each of one clock as save writes them; {fix}")

        rows = len(time)
        book = "results" in provenance

        # a run below the full level is saved with the tables it keeps alone
        full   = _recorded(provenance) == "full"
        tables = {name: _loaded(path, folder, name, book, fix) for name in SCHEMAS
                  if full or name not in LOGGED + ("baselines",)}
        ledger = tables["ledger"]
        bars   = [table[column] for table in tables.values() for column in BARS
                  if column in table.columns]
        if (len(ledger) != rows or not np.array_equal(ledger["time"], time)
                or not np.array_equal(ledger["equity"], equity)
                or any(np.ma.any((column < 0) | (column >= rows)) for column in bars)
                or any(not np.array_equal(table["time"], time[table["bar"]])
                       for name, table in tables.items() if name in LOGGED)
                or not (book or not full or _laid_out(tables["holdings"], rows))):
            raise ProbatumValueError(f"path: the tables of {str(path)!r} do not hold "
                                     f"the rows of its time.npy and equity.npy; {fix}")

        baselines = tables.get("baselines")
        if baselines is not None and (
                len(baselines) != rows or not np.array_equal(baselines["time"], time)
                or not (baselines["cash"] == equity[0]).all()
                or not all(np.isnan(value) or value == equity[0] for value in (
                    baselines["buy_and_hold"][0], baselines["frictionless"][0]))):
            raise ProbatumValueError(f"path: {str(path)!r}'s baselines.arrow does not "
                                     f"open at its equity.npy's first row on each of "
                                     f"its rows; {fix}")

        stats = _stats(path, folder, fix)
        if stats is not None and (
                (full and stats["num_fills"] != len(tables["fills"]))
                or stats["num_trades"] != np.count_nonzero(~tables["trades"]["open"])):
            raise ProbatumValueError(f"path: {str(path)!r}'s stats.arrow does not "
                                     f"count the fills and the round trips its tables "
                                     f"hold; {fix}")

        steps = (_stepped(path, folder, fix) if _recorded(provenance) == "transitions"
                 else None)
        # an agent's rows are its run's from its first close, each paid what its spec's
        # reward pays on the run's own equity
        if steps is not None and (
                steps.metadata["spec"] != provenance.get("spec")
                or steps["reward"].tolist() != _earned(steps.metadata["reward"],
                                                       equity[:len(steps) + 1].tolist())):
            raise ProbatumValueError(f"path: {str(path)!r}'s transitions.arrow does not "
                                     f"hold the steps of the agent its result.json "
                                     f"describes; {fix}")

        loaded = cls(time, equity, tables, stats, policy, manifest["bust"], provenance,
                     transitions=steps)
        if not np.array_equal(loaded.returns, arrays["returns"]):
            raise ProbatumValueError(f"path: {str(path)!r}'s returns.npy does not hold "
                                     f"the returns its equity.npy gives; {fix}")

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
    raise ProbatumValueError(f"{token} is not JSON")


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
        raise ProbatumValueError(f"path: {str(path)!r} holds no {name}.npy; {fix}")

    try:
        with open(file, "rb") as handle:
            values = np.load(handle, allow_pickle=False)
    except (ValueError, EOFError):
        values = None

    if not isinstance(values, np.ndarray):
        raise ProbatumValueError(f"path: {str(path)!r}'s {name}.npy is not an array of "
                                 f"numbers save writes; {fix}")

    return values


def _read(path, folder, name, fix):
    """The columns of the Arrow IPC file ``name`` of ``folder``, the directory ``path``
    names, refused where it is missing or unreadable."""
    file = folder / name
    if not file.is_file():
        raise ProbatumValueError(f"path: {str(path)!r} holds no {name}; {fix}")

    try:
        columns, _ = _ipc.read(file)
    except _ipc.Refused as error:
        raise ProbatumValueError(f"path: {str(path)!r}'s {name} {error}; "
                                 f"{fix}") from None

    return columns


def _earned(reward, equity):
    """What each step from one value of ``equity`` to the next pays under ``reward``, an
    agent spec's record of a builtin reward, as an environment works it (Specification,
    Reward and termination)."""
    floor, eta = reward["floor"] * equity[0], reward.get("eta")
    a = b = 0.0
    paid = []

    for before, after in zip(equity[:-1], equity[1:]):
        r = ln(max(after, floor) / before)
        if eta is None:
            paid.append(r)
            continue

        da, db = r - a, r * r - b
        spread = b - a * a
        paid.append((b * da - a * db / 2.0) / (spread * math.sqrt(spread))
                    if spread > 0.0 else 0.0)
        a, b = a + eta * da, b + eta * db

    return paid


def _stepped(path, folder, fix):
    """The transitions ``transitions.arrow`` of ``folder`` holds, the directory ``path``
    names, refused where it is missing or holds none export writes."""
    file = folder / "transitions.arrow"
    if not file.is_file():
        raise ProbatumValueError(f"path: {str(path)!r} holds no transitions.arrow; {fix}")

    try:
        return Transitions._read(file)
    except ProbatumValueError as error:
        raise ProbatumValueError(f"path: {str(path)!r}'s transitions.arrow {error}; "
                                 f"{fix}") from None


def _loaded(path, folder, name, book, fix):
    """The table ``name``.arrow of ``folder``, the directory ``path`` names, refused
    where it lacks a column save writes or holds a null in a column that holds none, a
    book's with its venue first but for its ledger and its baselines, which are the
    book's own, its columns in this version's order; a column a later version added
    comes back after them as it was written."""
    columns  = _read(path, folder, f"{name}.arrow", fix)
    joined   = book and name not in OWN
    schema   = ([("venue", "text", False)] if joined else []) + SCHEMAS[name]
    kinds    = {stored: kind for kind, stored in STORED.items()}
    nullable = {column: can for column, _, can in schema}

    for column, kind, _ in schema:
        if columns.get(column, ("",))[0] != STORED[kind]:
            raise ProbatumValueError(f"path: {str(path)!r}'s {name}.arrow has no "
                                     f"{kind} column {column}; {fix}")

    parts = []
    for column in [column for column, _, _ in schema] + [
            column for column in columns if column not in nullable]:
        stored, values, present, written = columns[column]
        if stored not in kinds:
            raise ProbatumValueError(f"path: {str(path)!r}'s {name}.arrow holds a "
                                     f"column {column} of a kind save does not write; "
                                     f"{fix}")

        can = nullable.get(column, written)
        if not (can or present.all()):
            raise ProbatumValueError(f"path: {str(path)!r}'s {name}.arrow holds a null "
                                     f"in {column}, a column that holds none; {fix}")

        data = None
        if stored == "large_text":
            values, data = values

        parts.append((column, kinds[stored], can, values,
                      None if present.all() else present, data))

    try:
        held = _Held(parts)
    except ValueError as error:
        # the table's own refusal says which column or length failed, which this one
        # leaves out
        raise ProbatumValueError(f"path: {str(path)!r}'s {name}.arrow is not a table "
                                 f"save writes; {fix}") from error

    return CLASSES.get(name, Table)(held)


def _stats(path, folder, fix):
    """The statistics ``stats.arrow`` holds, None where it holds no row, refused where
    one is missing; a statistic a later version added is left out."""
    columns = _read(path, folder, "stats.arrow", fix)
    counts  = {len(columns[key][1]) for key in STATS if key in columns}

    if (any(columns.get(key, ("",))[0] != ("int64" if key in COUNTS else "float")
            for key in STATS) or len(counts) != 1 or counts.pop() > 1
            or not all(columns[key][2].all() for key in STATS)):
        raise ProbatumValueError(f"path: {str(path)!r}'s stats.arrow is not the row of "
                                 f"statistics save writes; {fix}")

    if not len(columns[STATS[0]][1]):
        return None
    return {key: (int if key in COUNTS else float)(columns[key][1][0]) for key in STATS}


def _described(provenance):
    """Whether ``provenance`` holds what a Result's repr, combine and explain read of
    it, each account's in a book."""
    if "results" in provenance:
        parts = provenance["results"]
        return isinstance(parts, dict) and all(
            key in provenance for key in ("settlement", "risk_free", "periods_per_year")
        ) and all(isinstance(part, dict) and all(
            key in part for key in ("markets", "settlement", "contract"))
            for part in parts.values())

    data = provenance.get("data")
    return isinstance(data, dict) and "interval" in data and all(
        key in provenance for key in ("venue", "venue_name", "policy", "markets",
                                      "settlement", "contract", "risk_free",
                                      "periods_per_year"))


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
    named    = ["linear" if market == "perp" else market for market in markets]
    contract = data.instruments[0].contract if named == ["inverse"] else None

    return dict(data=data.provenance, venue=repr(venue), venue_name=venue.name,
                markets=named, settlement=_settlement(data.instruments, markets),
                contract=contract, policy=policy, **given, seed=seed, streams=streams,
                risk_free=float(risk_free), periods_per_year=rate, version=_version(),
                **_environment())


def combine(results, settlement=None):
    """One book of ``results``, Results from accounts at different venues, each named by
    its venue's name, or a mapping of a name to each, every account margined on its own
    and no capital moved between them.

    The book's clock is every time any of them has a row, on one interval's steps. An
    account counts its opening balance before its first row, its last row's between
    its rows, and its last equity after its end, and the book's equity is their sum in
    the order given, so its returns and statistics are those of the sum, computed as a
    run's are. Its ledger is the book's, a row per row of its clock: each account's
    cash held as its equity is, its exposure, margin and maintenance counted only inside
    its own span, its costs on the rows they were paid on, and the leverage gross
    exposure over equity. Its baselines are its own too, its accounts' summed as its
    equity is. Its other tables are theirs joined, each row with its venue first and
    every bar counted on the book's clock, in order of bar, the round trips closed in
    the order they closed and then those still open, and the assumptions in the order
    the Results are given (Specification, Combining books).

    ``settlement`` names the dollar asset a book of accounts kept in dollar assets, such
    as USDT and USDC, is kept in, each counted at par (Specification, Combining books).
    Refused are Results kept in different assets but dollar ones, or in different
    dollar assets without it; an inverse contract's account beside an account of
    another kind, whose equity in the quote is not in dollars; Results on bars of
    different intervals or off one interval's steps, or at different risk-free rates;
    a run recorded below the full level, which keeps no holdings, fill log or order
    log to join;
    and a book already combined. A column a later version added to a table is left out
    of the book's, since this version cannot tell what it counts."""
    if isinstance(results, Result):
        raise ProbatumTypeError("results: one Result; give a list of them, or a "
                                "mapping of a name to each")

    if isinstance(results, collections.abc.Mapping):
        named = list(results.items())
        for name, _ in named:
            if not isinstance(name, str):
                raise ProbatumTypeError(f"results: a name {name!r}; name each Result "
                                        f"with a string")
            if not name:
                raise ProbatumValueError("results: a name ''; give each Result a name")
    elif isinstance(results, collections.abc.Iterable):
        named = [(None, part) for part in results]
    else:
        raise ProbatumTypeError(f"results: {results!r}; give a list of Results, or a "
                                f"mapping of a name to each")

    if not named:
        raise ProbatumValueError("results: none; give the Results to combine")

    for k, (name, part) in enumerate(named):
        if not isinstance(part, Result):
            raise ProbatumTypeError(f"results: a {type(part).__name__}; give Results, "
                                    f"as backtest returns them")
        if "results" in part.provenance:
            raise ProbatumValueError("results: a book already combined; combine the "
                                     "Results it was made of")
        if _recorded(part.provenance) != "full":
            raise ProbatumValueError(_unkept(part, "results", "no holdings, fill log or "
                                             "order log to join"))

        if name is None:
            name = part.provenance["venue_name"]
            if name is None:
                raise ProbatumValueError("results: a Result at a venue with no name; "
                                         "name each, as combine({\"a\": first, \"b\": "
                                         "second})")
            named[k] = (name, part)

    labels = [name for name, _ in named]
    for name in labels:
        if labels.count(name) > 1:
            raise ProbatumValueError(f"results: two at {name}; name each, as "
                                     f"combine({{\"a\": first, \"b\": second}})")

    parts = [part for _, part in named]
    kept  = _kept(named, settlement)

    for key, what, fix in (("interval", "bars of", "run them on one interval"),
                           ("risk_free", "risk-free rates of", "run them at one rate")):
        seen = list(dict.fromkeys(part.provenance["data"]["interval"] if key == "interval"
                                  else part.provenance[key] for part in parts))
        if len(seen) > 1:
            raise ProbatumValueError(f"results: {what} {seen[0]} and {seen[1]}; {fix}")

    clock = np.unique(np.concatenate([part.time for part in parts]))
    step  = INTERVALS.get(parts[0].provenance["data"]["interval"])

    if step is not None:
        off = np.flatnonzero((clock - clock[0]) % step)
        if len(off):
            name = next(name for name, part in named if clock[off[0]] in part.time)
            raise ProbatumValueError(f"results: {name} opens a row at "
                                     f"{_when(clock[off[0]])}, between the "
                                     f"{parts[0].provenance['data']['interval']} steps "
                                     f"from {_when(clock[0])}; run them on bars that "
                                     f"open on one interval's steps")

    book, held, rows, funding = {}, np.zeros(len(clock), dtype=bool), [], 0.0
    for part in parts:
        at     = np.maximum(np.searchsorted(part.time, clock, side="right") - 1, 0)
        within = (clock >= part.time[0]) & (clock <= part.time[-1])
        place  = np.searchsorted(clock, part.time)
        ledger = part.ledger

        for name in HELD + SPANNED + COSTS:
            if name in HELD:
                values = ledger[name][at]
            elif name in SPANNED:
                values = np.where(within, ledger[name][at], 0.0)
            else:
                values = np.zeros(len(clock))
                values[place] = ledger[name]
            book[name] = values if name not in book else book[name] + values

        width   = len(part.holdings) // len(part.time)
        holding = (np.abs(part.holdings["position"].reshape(len(part.time), width))
                   > 0.0).any(axis=1)
        held |= within & holding[at]
        rows.append(place)
        funding = funding + (0.0 if part.stats is None else part.stats["funding_paid"])

    equity   = book["equity"]
    leverage = np.divide(book["gross_exposure"], equity, out=np.zeros(len(clock)),
                         where=equity > 0.0)
    columns = dict(bar=np.arange(len(clock), dtype=np.int64), time=clock,
                   leverage=leverage, **book)
    tables = {"ledger": Table(_Held([(name, kind, nullable, columns[name], None, None)
                                     for name, kind, nullable in SCHEMAS["ledger"]]))}

    for name in SCHEMAS:
        if name not in OWN:
            tables[name] = CLASSES.get(name, Table)(
                _joined([getattr(part, name) for part in parts], labels, rows, name))

    trades    = tables["trades"]
    rate      = _annualised(parts, clock)
    risk_free = parts[0].provenance["risk_free"]
    stats     = None if rate is None else stats_of(
        float(equity[0]), np.ascontiguousarray(equity[1:]),
        np.ascontiguousarray(trades["pnl"][~trades["open"]]), rate, float(risk_free),
        int(held[1:].sum()), len(tables["fills"]), funding)

    provenance = dict(results={label: copy.deepcopy(part.provenance)
                               for label, part in zip(labels, parts)},
                      settlement=kept, risk_free=risk_free, periods_per_year=rate,
                      version=_version(), **_environment())

    # a book plays no policy: it keeps each account's baselines where they are
    # computed, and otherwise the record they are read from, until its own are read
    given = []
    for label, part in zip(labels, parts):
        ready = part._ready()
        given.append((label, part.time, part.equity[0],
                       None if ready is not None else part._record,
                       None if ready is None else (ready["buy_and_hold"],
                                                   ready["frictionless"])))

    book = Result(clock, equity, tables, stats, None, bool((equity[1:] <= 0.0).any()),
                  provenance)
    book._parts = given
    if all(computed is not None for *_, computed in given):
        book.baselines

    return book


def _kept(named, settlement):
    """The asset a book of the Results ``named`` is kept in: the one asset its accounts
    are kept in, or the dollar asset ``settlement`` names where they are kept in dollar
    assets, None where none says, refusing a book whose accounts are kept in assets
    that do not add."""
    if settlement is not None and not (isinstance(settlement, str) and settlement):
        error = ProbatumValueError if isinstance(settlement, str) else ProbatumTypeError
        raise error(f"settlement: {settlement!r}; name the asset the book is kept in, "
                    f"such as \"USDT\"")

    coins = [name for name, part in named if part.provenance["markets"] == ["inverse"]]
    if coins and len(coins) < len(named):
        other = next(name for name, _ in named if name not in coins)
        raise ProbatumValueError(f"results: {coins[0]} trades an inverse contract and "
                                 f"{other} does not, so their equity in the quote does "
                                 f"not add; combine each kind on its own")

    assets = list(dict.fromkeys(part.provenance["settlement"] for _, part in named))

    if coins:
        if len(assets) > 1 or (assets[0] is None and len(coins) > 1):
            raise ProbatumValueError(f"results: accounts kept in coins that are not "
                                     f"one known coin, {', '.join(map(str, assets))}; "
                                     f"combine accounts kept in one coin")
        if settlement not in (None, assets[0]):
            raise ProbatumValueError(f"settlement: {settlement!r} for accounts kept in "
                                     f"their coin; a book of a coin is kept in that "
                                     f"coin, so leave settlement out")
        return assets[0]

    known = [asset for asset in assets if asset is not None]
    odd   = [asset for asset in known if asset not in DOLLARS]
    if odd:
        if len(assets) > 1:
            other = next(asset for asset in assets if asset != odd[0])
            raise ProbatumValueError(f"results: accounts kept in {odd[0]} and "
                                     f"{other or 'an asset their bars do not name'}, "
                                     f"which do not add; combine accounts kept in one "
                                     f"asset")
        if settlement not in (None, odd[0]):
            raise ProbatumValueError(f"settlement: {settlement!r} for accounts kept in "
                                     f"{odd[0]}, which is no dollar asset to count at "
                                     f"par; leave settlement out")
        return odd[0]

    if settlement is not None:
        if known and settlement not in DOLLARS:
            raise ProbatumValueError(f"settlement: {settlement!r} for accounts kept in "
                                     f"{' and '.join(known)}; name the dollar asset "
                                     f"the book is kept in, as \"{known[0]}\"")
        return settlement

    if len(known) > 1:
        raise ProbatumValueError(f"results: accounts kept in {known[0]} and "
                                 f"{known[1]}; pass settlement= to count them at par "
                                 f"in one, as settlement=\"{known[0]}\"")

    return known[0] if known else None


def _annualised(parts, clock):
    """The bars a year a book of ``parts`` on ``clock`` is annualised at: theirs where
    every part that has one agrees, and otherwise its rows less one per 365 days of the
    time some part covers, so time between two parts' spans counts for neither."""
    rates = list(dict.fromkeys(part.provenance["periods_per_year"] for part in parts
                               if part.provenance["periods_per_year"] is not None))
    if len(rates) == 1:
        return rates[0]

    spans = sorted((int(part.time[0]), int(part.time[-1])) for part in parts)
    covered, end = 0, None
    for first, last in spans:
        if end is None or first > end:
            covered += last - first
            end = last
        elif last > end:
            covered += last - end
            end = last

    if len(clock) < 2 or covered <= 0:
        return None
    return (len(clock) - 1) * 365.0 / (covered / DAY)


def _joined(tables, labels, rows, name):
    """``tables``, one of each book's, as one table of the binding, each row with its
    book's venue first and every bar counted on the book's clock through ``rows``, each
    book's rows on it, the rows in order of bar, for the round trips ``name`` names,
    the closed ones in the order they closed and then those still open, and for the
    assumptions, in the order of the books. Each part's columns are found by name and
    the book's are this version's, so a column a later version added is left out."""
    pieces  = [{part[0]: part for part in table._held.parts()} for table in tables]
    counts  = [len(table) for table in tables]
    encoded = [label.encode() for label in labels]
    ends    = np.concatenate([[0], np.cumsum(np.repeat([len(b) for b in encoded], counts))])
    data    = np.frombuffer(b"".join(b * count for b, count in zip(encoded, counts)),
                            dtype=np.uint8)

    joined = [("venue", "text", False, ends.astype(np.int64, copy=False), None, data)]
    for column, kind, nullable in SCHEMAS[name]:
        values = [piece[column][3] for piece in pieces]
        valid  = [piece[column][4] for piece in pieces]
        if column in BARS:
            values = [own[given] if mask is None else
                      np.where(mask, own[np.where(mask, given, 0)], given)
                      for own, given, mask in zip(rows, values, valid)]

        if kind == "text":
            shifts = np.cumsum([0] + [len(piece[column][5]) for piece in pieces[:-1]])
            values = np.concatenate([[0]] + [given[1:] + shift
                                             for given, shift in zip(values, shifts)])
            text = np.concatenate([piece[column][5] for piece in pieces])
        else:
            values, text = np.concatenate(values), None

        mask = (None if all(given is None for given in valid) else np.concatenate(
            [np.ones(count, dtype=bool) if given is None else given
             for given, count in zip(valid, counts)]))
        joined.append((column, kind, nullable, np.ascontiguousarray(values), mask, text))

    held = _Held(joined)
    del joined

    if name == "assumptions":
        return held

    if name == "trades":
        _, _, _, closed, valid, _ = next(part for part in held.parts()
                                         if part[0] == "close_bar")
        key = closed if valid is None else np.where(valid, closed, np.iinfo(np.int64).max)
    else:
        key = held.column("bar")

    return held.take(np.argsort(key, kind="stable").astype(np.int64, copy=False))
