"""``Engine``, the loop the caller owns: an account at a venue on a Data, stepped bar by
bar, with the account's verbs and state on the engine itself (ADR 0002)."""

from . import _probatum
from ._backtest import _annualised, _markets, _number, _recorded
from ._backtester import _version
from ._data import Data
from ._result import Result
from ._strategy import Account, Bar, _Run
from ._venue import _run, _venue_of


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
    left off (Specification, What a bar cannot show).
    """

    __slots__ = ("_venue", "_seed")


    def __init__(self, data, venue, seed=None):
        data   = data if isinstance(data, Data) else Data(data)
        held   = _venue_of(venue)
        run    = _run(held, data, seed)
        engine = _probatum.Engine(run.candles, report=True, **run.settings)

        super().__init__(_Run(engine, data, run.start, _markets(run)))
        self._run.named = "engine"
        self._venue     = held
        self._seed      = seed


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


    def result(self, risk_free=0.0):
        """What the account has recorded from its first bar to the current one, as
        `backtest` returns it, its statistics at ``risk_free``, an annual rate."""
        if not _number(risk_free):
            raise ValueError(f"risk_free: {risk_free!r}; give the annual rate as a "
                             f"finite number, such as 0.04")

        run        = self._run
        rate       = _annualised(run.data)
        stats      = None if rate is None else run.engine.stats(rate, float(risk_free))
        provenance = dict(data=run.data.provenance, venue=repr(self._venue),
                          policy="the caller's loop", seed=self._seed,
                          risk_free=risk_free, periods_per_year=rate, version=_version())

        return Result(*_recorded(run.engine, run.start), stats, None,
                      run.engine.is_bust(), provenance)
