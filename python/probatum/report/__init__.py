"""``probatum.report``: a run's tear sheet and a search's or a validation's report, each
one self-contained HTML file of static figures, text and tables with no JavaScript,
which ``Result.report`` and ``Study.report`` write. All of it draws what a Result or a
Study holds and what ``probatum.stats`` reads of it, so a redesign cannot move a number;
it needs matplotlib, which the plot extra installs (ADR 0041)."""

import importlib
import os
import pathlib

from .._errors import MissingDependency, ProbatumTypeError


def _written(said, path, page, *given):
    """Write the page ``page`` builds of ``given`` to ``path``, its folders made where
    they are missing, and return ``path``, refusing a path that is no path and, naming
    ``said``, a missing matplotlib with the line that installs it."""
    if not isinstance(path, (str, os.PathLike)):
        raise ProbatumTypeError(f"path: {type(path).__name__}; give a file path, such as "
                                f"\"run.html\"")
    try:
        importlib.import_module("matplotlib")
    except ImportError:
        raise MissingDependency(f"{said}: matplotlib is not installed; pip install "
                                f"\"probatum[plot]\"") from None

    from . import _sheet, _study
    text   = {"sheet": _sheet.page, "study": _study.page}[page](*given)
    target = pathlib.Path(path)
    if target.parent != pathlib.Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(text, encoding="utf-8", newline="\n")
    return path
