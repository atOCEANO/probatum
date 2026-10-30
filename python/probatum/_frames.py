"""How a table, a record and a summary show, in a notebook and as text, as pandas frames
(ADR 0041): the frame its ``to_pandas()`` gives, beside it a frame of one column of what
holds for every row, and under them a frame of the reasons a row has no value or no
interval, by the number its row gives. Without pandas, which a frame needs, a table and
a record show their fields."""

import html
import importlib
import math

import numpy as np

from ._errors import MissingDependency

# the frames side by side where the cell is wide enough, and the reasons under them
BESIDE = ('<div style="display: flex; flex-wrap: wrap; align-items: flex-start; '
          'gap: 0 24px">{}</div>')
UNDER  = '<div style="margin-top: 4px">{}</div>'


def pandas(said=None):
    """pandas, or None where it is not installed, refused as a MissingDependency that
    names ``said`` where ``said`` is given, since it needs pandas."""
    try:
        return importlib.import_module("pandas")
    except ImportError:
        if said is None:
            return None
        raise MissingDependency(f"{said}: pandas is not installed; pip install "
                                f"pandas") from None


def typed(frame):
    """``frame``'s columns each as the kind its values are: a truth or a whole number
    kept so where another row has none, which pandas would make an object or a float,
    and a number otherwise."""
    for name in frame.columns:
        values = [value for value in frame[name] if value is not None]
        if not values:
            continue
        if all(isinstance(value, (bool, np.bool_)) for value in values):
            frame[name] = frame[name].astype("boolean")
        elif all(isinstance(value, (int, np.integer)) for value in values):
            frame[name] = frame[name].astype("Int64")
        elif all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            frame[name] = frame[name].astype(np.float64)
    return frame


def said(value):
    """``value`` as a frame of what holds shows it: a number to the places pandas shows,
    with the trailing zeros left off, or to that many significant figures where it is
    too small for them, and any other as its text."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        places = pandas().get_option("display.precision")
        value  = float(value) + 0.0
        if not math.isfinite(value):
            return str(value)
        if value and abs(value) < 10.0 ** -places:
            return f"{value:.{places}g}"
        shown = f"{value:,.{places}f}".rstrip("0")
        return shown + "0" if shown.endswith(".") else shown
    return str(value)


def held(pairs):
    """``pairs``, (name, value), as a frame of one column, ``value``, each as ``said``
    gives it, or None where there are none."""
    pairs = list(pairs)
    if not pairs:
        return None
    return pandas().DataFrame({"value": [said(value) for _, value in pairs]},
                              index=[name for name, _ in pairs])


def shown(frame, columns, figure):
    """``frame``, whose last columns are ``low``, ``high``, ``level``, ``method`` and
    ``why_none``, as it shows: each number of ``columns`` as ``figure`` gives it for its
    row, the level as ``said`` does, nothing where there is none, and each reason the
    number of its row of the frame of reasons beside it, None where it holds none."""
    frame = frame.copy()
    for column in columns:
        frame[column] = [figure(value, row)
                         for row, value in zip(frame.index, frame[column])]
    frame["level"] = ["" if math.isnan(level) else said(level) for level in frame["level"]]

    reasons = list(dict.fromkeys(why for why in frame["why_none"] if why))
    frame["why_none"] = [str(reasons.index(why) + 1) if why else ""
                         for why in frame["why_none"]]
    if not reasons:
        return frame, None
    return frame, pandas().DataFrame({"why_none": reasons},
                                     index=range(1, len(reasons) + 1))


def html_of(frame, beside=None, under=None):
    """``frame`` as a notebook shows it, ``beside`` it what holds for every row and
    ``under`` them the reasons, each cell whole, or the three as text where pandas is
    set to show frames in a notebook as text."""
    pd = pandas()
    if not pd.get_option("display.notebook_repr_html"):
        return f"<pre>{html.escape(text_of(frame, beside, under))}</pre>"

    with pd.option_context("display.max_colwidth", None):
        parts = "".join(f"<div>{part._repr_html_()}</div>" for part in (frame, beside)
                        if part is not None)
        shown = BESIDE.format(parts)
        if under is not None:
            shown += UNDER.format(under._repr_html_())
    return shown


def text_of(frame, beside=None, under=None):
    """``frame`` as text, ``beside``, what holds for every row, and ``under``, the
    reasons, after it, each cell whole."""
    with pandas().option_context("display.max_colwidth", None):
        return "\n\n".join(repr(part) for part in (frame, beside, under)
                           if part is not None)
