"""probatum: embeddable market simulation library.

The compiled Rust core is ``probatum._probatum``, whose engine the Backtester and rl
drive. ``Engine`` is an account at a venue on a Data for a loop the caller owns, opened
with ``reset`` and moved one bar at a time with ``step``. ``Batch`` runs many
independent envs over one shared series and steps them in parallel with the GIL
released. ``Data.chart`` and ``Result.chart`` draw bars, a run on them and your own
arrays as one self-contained HTML document for a notebook cell or a file, computing
nothing, with the marks ``chart.marks`` holds and the look ``chart.defaults`` sets.
``Venue`` holds a venue's rules, and ``venues`` the presets, each one a Venue.
``backtest`` runs a policy on a Data at a venue and returns the ``Result`` it
recorded, ``Strategy`` is a policy that decides at each close, and ``combine`` adds
Results from accounts at different venues into one book. ``search`` searches a
policy's parameters over ``Int``, ``Float`` and ``Categorical`` ranges on the bars
before a holdout, and returns the ``Study`` of every trial, which a directory given as
its ``storage`` keeps as it runs, so the same call resumes it there, and which ``save``
and ``Study.load`` write and read as open files. ``labels`` gives, as
``Labels``, what one trade of a notional decided at each row's close makes through the
same fill model, for a model to fit on. ``validate`` runs that search, or fits a model
to its labels, on the training bars of each split a splitter of ``cv`` gives, trades
each split's winner on its test span and returns the ``Study`` of the out-of-sample
paths. ``capacity`` runs a policy again at several balances, and ``cost_curve`` with
every cost its venue charges for trading multiplied together, where ``breakeven_bps``
finds the round-trip cost at which it stops making money. ``env`` steps accounts at a
venue on a Data together in Rust as a Gymnasium vector environment, each in episodes
drawn from a stream of its own, and ``rl`` holds it with one environment of it; both
load Gymnasium when first reached, which nothing else here imports but the backtest of
an agent trained there, which ``save`` writes beside the spec it trained with.
``stats`` says how far a run's statistics can be trusted, read from a Result or any
series of returns: the Sharpe's interval, its probability of beating a benchmark and
the track record that would show it; what else describes a run, its drawdowns, its
statistics over time, its round trips and where its money went, with ``compare``
setting runs side by side; and how far a search's winner can, judged across all its
trials. ``Result.report`` and ``Study.report`` write a run's tear sheet and a search's
or a validation's report, each one HTML file of static figures that ``report`` draws
from what they hold, with the plot extra's matplotlib.
Every error probatum raises is a ``ProbatumError`` and the built-in it means, and
every warning a ``ProbatumWarning``, outside the modules still carried from emsl,
which raise the built-ins and warn with ``UserWarning`` or ``RuntimeWarning``.
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
from . import cv, metrics, stats, ta, venues
from ._backtester import Backtester, BacktestResult
from ._strategy import Strategy
from ._engine import Engine
from ._result import Result, combine
from .stats import compare
from ._backtest import backtest
from ._agent import save
from ._costs import Capacity, CostCurve, breakeven_bps, capacity, cost_curve
from ._labels import Labels, labels
from ._search import Study, search
from ._validate import validate
from .market import Market
from . import chart

__all__ = [
    "BacktestResult", "Backtester", "Batch", "Capacity", "Categorical", "CostCurve",
    "Data", "Engine", "Float", "Instrument", "Int", "Labels", "Market", "Result",
    "Strategy", "Study", "Venue",
    "backtest", "breakeven_bps", "capacity", "combine", "compare", "cost_curve",
    "labels", "save", "search", "to_ohlcv", "validate",
    "chart", "cv", "metrics", "stats", "ta", "venues",
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
    # rl, env, sb3 and torch are imported when first reached rather than with the
    # package, and left out of __all__ so a star import reaches none of them, since they
    # import Gymnasium, which a backtest loads only to run an agent (ADR 0044), and sb3
    # and torch the frameworks a user brings. import_module rather than
    # `from . import rl`, which resolves the name through getattr on this very module
    # and recurses until the stack ends
    import importlib
    if name in ("rl", "sb3", "torch"):
        return importlib.import_module(f".{name}", __name__)

    if name == "env":
        return importlib.import_module("._env", __name__).env

    raise ProbatumAttributeError(f"module 'probatum' has no attribute {name!r}")
