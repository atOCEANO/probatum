"""``labels``: what one trade of a notional decided at each row's close makes through the
fill model a backtest runs, for a model to fit on (ADR 0036; Specification, Labels)."""

import math

import numpy as np

from ._backtest import _barrier, _real, _repeatable
from ._data import Data, _chosen, _frozen
from ._errors import ProbatumAttributeError, ProbatumTypeError, ProbatumValueError
from ._probatum import Engine, run_labels
from ._venue import _market, _run

SIDES = ("long", "short")
# each exit by the code run_labels gives it, no label first and censored last
EXITS    = ("", "stop", "take_profit", "horizon", "delist", "liquidation", "censored")
CENSORED = len(EXITS) - 1
# the end of a row with no label, as the splitters read it
NONE = np.iinfo(np.int64).min


class Labels:
    """What `labels` found on a Data, each field ``(T, N)`` and read-only: ``ret``, each
    trade's return net of every cost, NaN where it has none or is censored; ``exit``,
    what left its account flat, "stop", "take_profit", "horizon", "delist" or
    "liquidation", "censored", or "" where the row has no label; ``bars``, the bars
    from its entry's fill to its end; ``end``, the bar whose price closed it, at ``T``
    or later where it is censored and the least int64 where it has none, as the
    splitters of `probatum.cv` read it; and ``weight``, its average uniqueness among
    its instrument's labels that have a return, NaN elsewhere. It keeps what it was
    made with: ``horizon``, ``notional``, ``side``, ``stop`` and ``take_profit``,
    ``venue``, and the Data's ``names`` and ``fingerprint``."""

    __slots__ = ("ret", "exit", "bars", "end", "weight", "horizon", "notional", "side",
                 "stop", "take_profit", "venue", "names", "fingerprint")


    def __init__(self, ret, exit, bars, end, weight, horizon, notional, side, stop,
                 take_profit, venue, names, fingerprint):
        given = dict(ret=ret, exit=exit, bars=bars, end=end, weight=weight,
                     horizon=horizon, notional=notional, side=side, stop=stop,
                     take_profit=take_profit, venue=venue, names=names,
                     fingerprint=fingerprint)
        for name, value in given.items():
            # an array comes back from a pickle writable
            if isinstance(value, np.ndarray):
                _frozen(value)
            object.__setattr__(self, name, value)


    def __setattr__(self, name, value):
        raise ProbatumAttributeError(f"Labels.{name}: labels do not change; make others "
                                     f"with probatum.labels")


    def __delattr__(self, name):
        raise ProbatumAttributeError(f"Labels.{name}: labels do not change; make others "
                                     f"with probatum.labels")


    def __reduce__(self):
        return Labels, tuple(getattr(self, name) for name in self.__slots__)


    def __repr__(self):
        rows, width = self.ret.shape
        what = self.names[0] if width == 1 else f"{width} instruments"
        return (f"Labels({what} over {rows} bars, {self.side} {self.notional:,} at a "
                f"horizon of {self.horizon})")


def labels(data, venue, horizon, notional, stop=None, take_profit=None, side="long"):
    """What one trade of ``notional`` decided at the close of each row of ``data`` makes
    on each of its instruments at ``venue``, a Venue or a preset's name, through the
    fill model a backtest runs, as `Labels` (ADR 0036; Specification, Labels).

    Each label is a fresh account on the Data's clock trading its instrument alone,
    its balance ``notional`` in the currency it settles in, an inverse contract's
    coin. At the row's close it buys ``notional``, or sells it for ``side="short"``,
    to fill at the next open; places a bracket at ``stop`` and ``take_profit``,
    returns from its average entry, at the close of the bar its entry filled on; and
    closes what is left at the close of the row ``horizon`` bars on, or of its
    instrument's last printed row before it, and again at each close after while some
    is left. ``stop`` and ``take_profit`` are each None, a return above 0 or a
    ``(T, N)`` array of them read on the label's row, NaN for none. A venue that
    decides at random is refused, as `backtest` refuses one."""
    data = data if isinstance(data, Data) else Data(data)
    held = _repeatable(venue, "labels", "label")

    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool):
        raise ProbatumTypeError(f"horizon: {horizon!r}; give a whole number of bars "
                                f"from 1")

    if horizon < 1:
        raise ProbatumValueError(f"horizon: {horizon}; give a whole number of bars from 1")

    if not _real(notional):
        raise ProbatumTypeError(f"notional: {notional!r}; give the amount each label "
                                f"trades, a positive number")

    if not (math.isfinite(notional) and notional > 0.0):
        raise ProbatumValueError(f"notional: {notional!r}; give a positive amount, in the "
                                 f"currency the account settles in")

    _chosen("side", side, SIDES, "use \"long\" or \"short\"")
    barriers = [_barrier(name, given, data)
                for name, given in (("stop", stop), ("take_profit", take_profit))]

    horizon, notional = int(horizon), float(notional)
    names   = data.names
    origin  = held.name or "venue"
    markets = [_market(instrument, name, held, origin)
               for instrument, name in zip(data.instruments, names)]

    spot  = [k for k, market in enumerate(markets) if market == "spot"]
    perps = [k for k, market in enumerate(markets) if market != "spot"]
    if side == "short" and spot:
        raise ProbatumValueError(f"side: \"short\", and {names[spot[0]]} is spot and "
                                 f"cannot be short; label it \"long\", or a perpetual")

    count, width = len(data), len(names)
    ret    = np.full((count, width), np.nan)
    codes  = np.zeros((count, width), dtype=np.uint8)
    bars   = np.zeros((count, width), dtype=np.int64)
    end    = np.full((count, width), NONE, dtype=np.int64)
    weight = np.full((count, width), np.nan)

    # the rows the Data's runs keep, those some instrument printed, in order
    rows = np.flatnonzero(data.available.any(axis=1))

    for wallet, columns in (("spot", spot), ("margin", perps)):
        if not columns:
            continue

        balance = ({name: notional if name == wallet else 0.0
                    for name in ("spot", "margin")} if spot and perps else notional)
        run    = _run(held, data, balance=balance)
        engine = Engine(run.candles, report=True, **run.settings)
        at     = np.full(count, -1, dtype=np.int64)
        at[rows] = run.start + np.arange(len(rows))

        made = []
        for k in columns:
            printed = np.flatnonzero(data.available[:, k])
            starts  = printed[:-1]
            # a label whose horizon runs past the last row is censored unrun
            late = starts + horizon + 1 >= count
            codes[starts[late], k] = CENSORED
            end[starts[late], k]   = starts[late] + horizon + 1
            starts = starts[~late]
            lasts  = printed[np.searchsorted(printed, starts + horizon, side="right") - 1]
            made.append((np.full(len(starts), k), starts, lasts))

        held_by, starts, lasts = (np.concatenate(parts) for parts in zip(*made))
        read = [barrier if barrier is None or isinstance(barrier, float) else
                np.ascontiguousarray(barrier[starts, held_by]).reshape(-1, 1)
                for barrier in barriers]
        made_ret, made_code, entered, ended = run_labels(
            engine, held_by, at[starts], at[lasts], notional,
            "buy" if side == "long" else "sell", *read)

        ret[starts, held_by]   = made_ret
        codes[starts, held_by] = made_code
        censored = made_code == CENSORED
        end[starts[censored], held_by[censored]] = np.maximum(
            starts[censored] + horizon + 1, count)
        closed = (made_code > 0) & ~censored
        t, i   = starts[closed], held_by[closed]
        first = rows[entered[closed] - run.start]
        last  = rows[ended[closed] - run.start]
        end[t, i], bars[t, i] = last, last - first

        for k in columns:
            own = i == k
            weight[t[own], k] = _uniqueness(first[own], last[own], count)

    shown = [barrier if barrier is None or isinstance(barrier, float) else _frozen(barrier)
             for barrier in barriers]
    return Labels(ret, np.array(EXITS)[codes], bars, end, weight, horizon, notional, side,
                  *shown, held, names, data.fingerprint)


def _uniqueness(first, end, count):
    """Each label's average uniqueness among one instrument's labels, label ``j``
    spanning the rows ``first[j]`` to ``end[j]`` of ``count``: the mean over its span
    of one over how many labels span each row (Lopez de Prado 2018, chapter 4)."""
    spanning = np.zeros(count + 1)
    np.add.at(spanning, first, 1.0)
    np.add.at(spanning, end + 1, -1.0)
    spanning = np.cumsum(spanning[:-1])
    unique   = np.divide(1.0, spanning, out=np.zeros(count), where=spanning > 0.0)
    shares   = np.concatenate(([0.0], np.cumsum(unique)))
    return (shares[end + 1] - shares[first]) / (end - first + 1)
