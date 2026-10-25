"""``Engine``, the loop the caller owns: an account at a venue on a Data, stepped bar by
bar, with the account's verbs and state on the engine itself (ADR 0002)."""

import copy
import operator
import weakref

from . import _probatum
from ._backtest import _annualised, _markets, _rate
from ._data import Data
from ._result import Result, _Record, _provenance
from ._strategy import Account, Bar, _handed, _Run
from ._venue import _run, _venue_of

# what a pickle of an engine's run leaves out: the engine and the handles, kept apart,
# what the Data gives again, and what is read again when first asked for
UNKEPT = ("engine", "handles", "token", "available", "bars", "time", "rates", "state",
          "events", "trips")


class Engine(Account):
    """An account at ``venue``, a Venue or a preset's name, on ``data``, a Data or
    anything one is built from, opened at the Data's first bar, for a loop the caller
    owns.

    ``reset()`` opens the account again and ``step()`` moves it to the next bar, each
    returning the `Bar` just closed; ``done()`` is true at the last bar, which no bar
    follows to fill on; and ``result()`` returns what the account has recorded, as
    `backtest` returns it. Between steps the caller decides through the account's
    verbs and reads its state, on the engine itself, as a Strategy's ``next`` reads
    its account, ``events`` holding what happened to its orders since the last step.
    ``seed`` names the stream a venue that decides what a bar cannot show at random
    draws from, which runs on across a reset, so each episode draws where the last
    left off (Specification, What a bar cannot show). ``clone()`` copies the account as
    it stands, and a pickle moves it to another process of this version
    (Specification, Clones and pickles).
    """

    __slots__ = ("_venue", "_seed", "_markets", "_record", "__weakref__")


    def __init__(self, data, venue, seed=None):
        data    = data if isinstance(data, Data) else Data(data)
        held    = _venue_of(venue)
        run     = _run(held, data, seed)
        engine  = _probatum.Engine(run.candles, report=True, **run.settings)
        markets = _markets(run)

        super().__init__(_Run(engine, data, run.start, markets))

        # a handle points back at its run, so the run is ended when the engine goes
        weakref.finalize(self, self._run.end).atexit = False
        self._run.named = "engine"
        self._venue     = held

        # the binding read the seed as an integer, and the provenance keeps it as one
        self._seed    = None if seed is None else operator.index(seed)
        self._markets = markets
        self._record  = _Record(held, data, run, self._run.token, self._seed)


    def __repr__(self):
        run  = self._run
        held = run.names[0] if run.width == 1 else f"{run.width} instruments"
        return f"Engine({held} on {self._venue!r}, row {run.row:,})"


    def reset(self):
        """Open the account again at the first bar and return that bar."""
        run = self._run
        run.reset()
        return Bar(run, run.row)


    def step(self):
        """Fill what was decided at the last close on the next bar and mark the account
        at its close, returning that bar; at the last bar the account stays where it
        is."""
        run = self._run
        if not run.engine.done():
            run.step()

        run.begin()
        return Bar(run, run.row)


    def done(self):
        """Whether the account is at the last bar."""
        return self._run.engine.done()


    def clone(self):
        """A copy of the account as it stands, which steps on as this one would: the
        engine and all it recorded copied, the Data shared, and handles of its own for
        the orders placed so far, which refuses this engine's."""
        record       = copy.copy(self._record)
        run          = self._run.cloned()
        record.token = run.token

        return _held(run, self._venue, self._seed, self._markets, record)


    def __copy__(self):
        return self.clone()


    def __deepcopy__(self, memo):
        return self.clone()


    def __reduce__(self):
        run  = self._run
        kept = {name: value for name, value in run.__dict__.items() if name not in UNKEPT}
        return _restored, (run.engine, kept, run.held(), self._venue, self._seed,
                           self._markets, self._record)


    def result(self, risk_free=0.0):
        """What the account has recorded from its first bar to the current one, as
        `backtest` returns it, its statistics at ``risk_free``, an annual rate."""
        _rate(risk_free)

        run        = self._run
        rate       = _annualised(run.data)
        stats      = None if rate is None else run.engine.stats(rate, float(risk_free))
        provenance = _provenance(run.data, self._venue, self._markets,
                                 "the caller's loop", risk_free, rate, seed=self._seed)

        return Result._of(run.engine.tables(run.start, list(run.names)), stats, None,
                          run.engine.is_bust(), provenance, self._record)


def _held(run, venue, seed, markets, record):
    """An Engine over ``run``, with what it was made with, ended as its run is once
    nothing holds it."""
    engine = Engine.__new__(Engine)
    Account.__init__(engine, run)
    weakref.finalize(engine, run.end).atexit = False
    engine._venue, engine._seed, engine._markets, engine._record = (venue, seed, markets,
                                                                    record)

    return engine


def _restored(inner, kept, handles, venue, seed, markets, record):
    """The Engine a pickle holds: its run rebuilt over the binding's engine ``inner``,
    which a process of another version refuses, its handles over that run, and what
    the Data gives read from it again."""
    run = _Run.__new__(_Run)
    run.__dict__.update(kept)

    data       = run.data
    run.engine = inner
    run.token  = record.token

    run.available, run.bars, run.time = data.available, data.bars, data.time
    run.rates, run.state, run.events, run.trips = None, None, None, None
    run.handles = {order: _handed(fields, run) for order, fields in handles.items()}

    return _held(run, venue, seed, markets, record)
