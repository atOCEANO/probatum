"""probatum: embeddable market simulation library.

The compiled Rust core is ``probatum._probatum``, whose engine the Backtester and rl
drive. ``Engine`` is an account at a venue on a Data for a loop the caller owns, opened
with ``reset`` and moved one bar at a time with ``step``. ``Batch`` runs many
independent envs over one shared series and steps them in parallel with the GIL
released. ``chart`` renders a frame, your own arrays and a run into one
self-contained HTML document for a notebook cell or a file, computing nothing itself.
``Venue`` holds a venue's rules, and ``venues`` the presets, each one a Venue.
``backtest`` runs a policy on a Data at a venue and returns the ``Result`` it
recorded, ``Strategy`` is a policy that decides at each close, and ``combine`` adds
Results from accounts at different venues into one book. ``search`` searches a
policy's parameters over ``Int``, ``Float`` and ``Categorical`` ranges on the bars
before a holdout, and returns the ``Study`` of every trial. ``validate`` runs that
search on the training bars of each split a splitter of ``cv`` gives, trades each
split's winner on its test span and returns the ``Study`` of the out-of-sample paths.
``stats`` says how far a run's statistics can be trusted, read from a Result or any
series of returns: the Sharpe's interval, its probability of beating a benchmark and
the track record that would show it.
Every error probatum raises is a ``ProbatumError`` and the built-in it means, and
every warning a ``ProbatumWarning``, outside the modules carried from emsl, which raise
the built-ins and warn with ``UserWarning`` or ``RuntimeWarning``.
"""

from ._errors import (DataError, LookaheadError, MissingDependency, PolicyError,
                      ProbatumAttributeError, ProbatumError, ProbatumIndexError,
                      ProbatumKeyError, ProbatumNotImplementedError, ProbatumRuntimeError,
                      ProbatumTypeError, ProbatumValueError, ProbatumWarning, SpecError,
                      VenueError)
from ._probatum import Batch
from ._data import Data, Instrument, to_ohlcv
from ._venue import Venue
from ._space import Categorical, Float, Int
from . import cv, metrics, plot, rl, stats, ta, venues
from ._backtester import Backtester, BacktestResult
from ._strategy import Strategy
from ._engine import Engine
from ._result import Result, combine
from ._backtest import backtest
from ._search import Study, search
from ._validate import validate
from .market import Market
from ._chart import chart, chart_defaults

__all__ = [
    "BacktestResult", "Backtester", "Batch", "Categorical", "Data", "Engine", "Float",
    "Instrument", "Int", "Market", "Result", "Strategy", "Study", "Venue", "backtest",
    "chart", "chart_defaults", "combine", "search", "to_ohlcv", "validate",
    "cv", "metrics", "plot", "rl", "stats", "ta", "venues",
    "ProbatumError", "ProbatumValueError", "ProbatumTypeError", "ProbatumKeyError",
    "ProbatumIndexError", "ProbatumAttributeError", "ProbatumRuntimeError",
    "ProbatumNotImplementedError", "DataError", "VenueError", "SpecError",
    "LookaheadError", "PolicyError", "MissingDependency", "ProbatumWarning",
]

try:
    from importlib.metadata import version as _version

    __version__ = _version("probatum")
except Exception:
    __version__ = "0.0.0"


def __getattr__(name):
    # sb3 is reachable as `probatum.sb3` but is not imported with the package, because
    # it needs stable_baselines3 and almost no install has it. The export table
    # listed it for four releases while `probatum.sb3` raised AttributeError, and the
    # test that would have caught that carried an exemption for this one name
    if name == "sb3":
        # import_module rather than `from . import sb3`, which resolves the name
        # through getattr on this very module and recurses until the stack ends
        import importlib
        return importlib.import_module(".sb3", __name__)

    raise ProbatumAttributeError(f"module 'probatum' has no attribute {name!r}")
