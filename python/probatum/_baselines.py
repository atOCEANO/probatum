"""A run set beside three baselines on its own data and venue (Specification,
Baselines): holding cash, buying and holding an equal share of every instrument, and
the same policy with the costs of trading removed and every rule kept, each read when
first asked for, an agent's as its backtest runs."""

import os
import warnings

import numpy as np

from ._backtest import _seen
from ._errors import ProbatumAttributeError, ProbatumWarning
from ._probatum import Engine, run_array
from ._strategy import Strategy, _play, _Run
from ._venue import _run

# the Engine's settings that charge funding, which the run without costs leaves out
FUNDING = ("funding_hours", "funding_interval", "funding_rate", "funding_times",
           "funding_rates", "funding_instruments")
UNPICKLED = ("result.baselines: not computed before this Result was pickled; read "
             "result.baselines before pickling it, or save it in the process that ran "
             "it")
# the package's files, which a warning skips to name the caller's line on every path
# that reads the baselines
PACKAGE = (os.path.dirname(__file__),)


def read(result):
    """``result``'s baselines as columns of its rows: its record's, or for a book its
    accounts' summed as its equity is."""
    if result._parts is not None:
        return _book(result)

    record = result._record
    if record is None:
        raise ProbatumAttributeError(UNPICKLED)

    held, frictionless = computed(record)
    rows = len(result.time)
    return result.equity[0], held[:rows], frictionless[:rows]


def computed(record):
    """What ``record``'s run would have held bought and held and run without its costs,
    on each of its rows, computed once and kept in the record its copies share."""
    if record.computed is None:
        held = _equity(record, _bought(record))
        record.computed = held, _frictionless(record)

    return record.computed


def _engine(record, change):
    """A binding Engine on ``record``'s Data and venue, its settings as ``change``
    leaves them, and the row its first bar is on; the candles are let go once it holds
    them."""
    run      = _run(record.venue, record.data, record.seed)
    settings = change(dict(run.settings))
    return Engine(run.candles, report=True, **settings), run.start


def _equity(record, played):
    """The equity of each row of ``record``'s run as ``played`` plays it on an Engine,
    from its opening balance."""
    change, play = played
    engine, start = _engine(record, change)
    play(engine, start)

    curve   = engine.equity_curve()
    balance = record.settings["quote"]
    opening = sum(balance.values()) if isinstance(balance, dict) else balance

    return np.concatenate([[opening], curve])


def _bought(record):
    """How buying and holding is played: an equal share of the opening balance in each
    of the Data's instruments, the quantity it buys at the instrument's first printed
    close, or the contracts it is worth there, asked for at every close from then on,
    an account of two wallets split between them in the universe's proportion."""
    data, settings = record.data, record.settings
    specs   = settings.get("instruments", [settings])
    width   = len(specs)
    balance = settings["quote"]
    opening = sum(balance.values()) if isinstance(balance, dict) else balance

    closes, available = data.closes, data.available
    rows = np.full(closes.shape, np.nan)
    last = len(data) - 1

    for k, spec in enumerate(specs):
        printed = np.flatnonzero(available[:, k])
        # an instrument with no close before the last row, where nothing is decided,
        # holds its share in cash
        if not len(printed) or printed[0] >= last:
            continue

        close = closes[printed[0], k]
        if spec["market"] == "inverse":
            rows[printed[0]:, k] = opening * close / (width * spec["contract"])
        else:
            rows[printed[0]:, k] = opening / width / close

    def change(settings):
        if isinstance(settings["quote"], dict):
            spot   = sum(spec["market"] == "spot" for spec in specs)
            larger = max(spot, width - spot)

            # the larger wallet's share is the product and the smaller the rest, so the
            # two add back to the balance exactly; a universe of one kind's is the whole
            # balance, which the product can round past
            share = opening if larger == width else opening * larger / width
            if 2 * spot >= width:
                settings["quote"] = {"spot": share, "margin": opening - share}
            else:
                settings["quote"] = {"spot": opening - share, "margin": share}

        return settings

    values, _ = _seen(rows, available.any(axis=1))

    def play(engine, start):
        run_array(engine, values, "quantity", 0.0, None, None, start=start)

    return change, play


def _costless(settings):
    """``settings`` with the costs of trading removed and every rule kept: the maker
    and taker fees, each instrument's own among them, the liquidation fee, the spread,
    the slippage, the impact and the funding."""
    for name in FUNDING:
        settings.pop(name, None)

    settings.update(fee_maker=0.0, fee_taker=0.0, spread=None, slippage_bps=0.0,
                    impact=0.0)

    if "liquidation" in settings:
        settings["liquidation_fee"] = 0.0

    if "instruments" in settings:
        settings["instruments"] = [dict(spec, fee_maker=0.0, fee_taker=0.0)
                                   for spec in settings["instruments"]]

    return settings


def _frictionless(record):
    """The run's policy played again without the costs of trading: an array's rows,
    sizing, tolerance and barriers, or the copy of a Strategy taken at the call,
    played once, or an agent run again in an environment of its own; missing for a loop
    the caller stepped, and where the play raises or did not finish, with a warning
    that says so."""
    missing = np.full(len(record.settings["times"]) - record.start, np.nan)

    # the policy leaves the record before its play, so it is played at most once, and
    # a play an interrupt stopped leaves the record saying so
    policy, record.policy = record.policy, None
    if policy is None:
        if record.playing:
            warnings.warn("baselines: the run without costs stopped before it "
                          "finished, so its column is missing; run backtest again to "
                          "read it", ProbatumWarning, skip_file_prefixes=PACKAGE)
        return missing

    if isinstance(policy, Strategy):
        def play(engine, start):
            _play(policy, _Run(engine, record.data, start, _markets(record.settings)))
    elif callable(policy):
        play = None
    else:
        def play(engine, start):
            run_array(engine, *policy, start=start)

    record.playing = True
    try:
        # an agent's run is its own, given the settings to change
        equity = policy(_costless) if play is None else _equity(record, (_costless, play))
    except Exception as error:
        warnings.warn(f"baselines: the run without costs raised "
                      f"{type(error).__name__}: {error}; its column is missing",
                      ProbatumWarning, skip_file_prefixes=PACKAGE)
        equity = missing

    record.playing = False
    return equity


def _markets(settings):
    return [spec["market"] for spec in settings.get("instruments", [settings])]


def _book(book):
    """A book's baselines: each account's on the book's clock, its opening balance
    before its first row and its last value after its end, summed in the order the
    accounts were given, a column an account cannot give missing on every row."""
    clock = book.time
    cash, held, frictionless = None, None, None

    for label, time, opening, record, given in book._parts:
        if given is None:
            if record is None:
                raise ProbatumAttributeError(f"book.baselines: {label} was pickled "
                                             f"before its baselines were read; read "
                                             f"them on it before combining it")
            given = tuple(column[:len(time)] for column in computed(record))

        at = np.maximum(np.searchsorted(time, clock, side="right") - 1, 0)
        if cash is None:
            cash, held, frictionless = opening, given[0][at], given[1][at]
        else:
            cash, held = cash + opening, held + given[0][at]
            frictionless = frictionless + given[1][at]

    return cash, held, frictionless
