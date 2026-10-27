"""``probatum.report``: a run's tear sheet and a search's or a validation's report, each
one self-contained HTML file of static figures, text and tables with no JavaScript,
which ``Result.report`` and ``Study.report`` write, and each figure of them alone as a
matplotlib Figure, which ``Result.figure`` and ``Study.figure`` hand out. All of it
draws what a Result or a Study holds and what ``probatum.stats`` reads of it, so a
redesign cannot move a number; it needs matplotlib, which the plot extra installs
(ADR 0041)."""

import importlib
import os
import pathlib

from .._errors import (MissingDependency, ProbatumKeyError, ProbatumTypeError,
                       ProbatumValueError)

# the figures of a run's tear sheet, of a search's report and of a validation's, by
# the name a figure is asked for, in the order the page draws them
FIGURES = {
    "sheet":  ("equity", "drawdown", "returns", "monthly", "rolling_sharpe",
               "rolling_volatility", "exposure", "weights", "pnl", "costs", "trips"),
    "trials": ("trials", "importance"),
    "paths":  ("paths", "windows"),
}
# why a figure draws nothing of what it is given, where it can
EMPTY = {
    "returns":            "a run of one row has no returns",
    "rolling_sharpe":     "the clock is too short to annualise, or a quarter of a year "
                          "of bars leaves a window fewer than 2 returns; read "
                          "stats.rolling_sharpe over a window of your own",
    "rolling_volatility": "the clock is too short to annualise, or a quarter of a year "
                          "of bars leaves a window fewer than 2 returns; read "
                          "stats.rolling_volatility over a window of your own",
    "weights":            "it sets several instruments apart, and the run holds one or "
                          "keeps no holdings; read result.weight",
    "pnl":                "it sets several instruments apart, and the run holds one or "
                          "keeps no holdings; read result.pnl",
    "trips":              "the run closed no round trip",
    "trials":             "no trial completed with a score",
    "paths":              "no path scored",
    "windows":            "no window scored both in sample and out of it, as none of a "
                          "model's fits scores in sample",
}


def _plotted(said):
    """matplotlib, refused, naming ``said``, with the line that installs it where it is
    missing."""
    try:
        return importlib.import_module("matplotlib")
    except ImportError:
        raise MissingDependency(f"{said}: matplotlib is not installed; pip install "
                                f"\"probatum[plot]\"") from None


def _written(said, path, page, *given):
    """Write the page ``page`` builds of ``given`` to ``path``, its folders made where
    they are missing, and return ``path``, refusing a path that is no path and, naming
    ``said``, a missing matplotlib with the line that installs it."""
    if not isinstance(path, (str, os.PathLike)):
        raise ProbatumTypeError(f"path: {type(path).__name__}; give a file path, such as "
                                f"\"run.html\"")
    _plotted(said)

    from . import _sheet, _study
    text   = {"sheet": _sheet.page, "study": _study.page}[page](*given)
    target = pathlib.Path(path)
    if target.parent != pathlib.Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(text, encoding="utf-8", newline="\n")
    return path


def _figure(name, page, given):
    """The figure ``name`` of the report ``page`` of ``given``, as a matplotlib Figure
    drawn at the reports' look, refusing a name the report does not draw and a figure
    that draws nothing of ``given``."""
    names = FIGURES[page]
    if not isinstance(name, str):
        raise ProbatumTypeError(f"name: {name!r}; give a figure's name, one of "
                                f"{', '.join(names)}")
    if name not in names:
        raise ProbatumKeyError(f"name: no figure {name!r}; use one of {', '.join(names)}")

    matplotlib = _plotted("figure")
    from . import _figures, _look
    with matplotlib.rc_context(_look.RC):
        figure = getattr(_figures, name)(given)

    if figure is None:
        raise ProbatumValueError(f"name: {name!r} draws nothing here, since {EMPTY[name]}")
    return figure
