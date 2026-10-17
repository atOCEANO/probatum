"""probatum: embeddable market simulation library.

The compiled Rust core is ``probatum._probatum``, whose engine the Backtester and rl
drive. ``Engine`` is an account at a venue on a Data for a loop the caller owns, opened
with ``reset`` and moved one bar at a time with ``step``. ``Batch`` runs many
independent envs over one shared series and steps them in parallel with the GIL
released, and ``tune`` searches a strategy's parameters by running many backtests
across cores. ``chart`` renders a frame, your own arrays and a run into one
self-contained HTML document for a notebook cell or a file, computing nothing itself.
``Venue`` holds a venue's rules, and ``venues`` the presets, each one a Venue.
``backtest`` runs a policy on a Data at a venue and returns the ``Result`` it
recorded, and ``Strategy`` is a policy that decides at each close.
"""

from ._probatum import Batch
from ._data import Data, Instrument, to_ohlcv
from ._venue import Venue
from . import metrics, plot, rl, ta, venues
from ._backtester import Backtester, BacktestResult
from ._strategy import Strategy
from ._engine import Engine
from ._result import Result
from ._backtest import backtest
from .market import Market
from ._tune import tune
from ._walk import walk_forward
from ._chart import chart, chart_defaults

__all__ = [
    "BacktestResult", "Backtester", "Batch", "Data", "Engine", "Instrument", "Market",
    "Result", "Strategy", "Venue", "backtest", "chart", "chart_defaults", "to_ohlcv",
    "tune", "walk_forward", "metrics", "plot", "rl", "ta", "venues",
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

    raise AttributeError(f"module 'probatum' has no attribute {name!r}")
