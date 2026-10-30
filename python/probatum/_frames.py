"""How a table, a record and a summary show, in a notebook and as text: as the pandas
frame its ``to_pandas()`` gives, what holds for every row said once under it as names
and values, and why a row has no value or no interval numbered under it (ADR 0041).
Without pandas, which a frame needs, a table and a record show their fields."""

import html
import importlib
import math

import numpy as np

from ._errors import MissingDependency

# a line under a frame, a little apart from it and faded, in the cell's own ink
UNDER = ('<div style="margin-top: 6px; font-size: 0.92em; opacity: 0.75; '
         'white-space: normal">{}</div>')


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
    """One setting's value: a number to the places pandas shows, 6 without it, with the
    trailing zeros left off, or to that many significant figures where it is too small
    for them, and any other as its text."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        found  = pandas()
        places = 6 if found is None else found.get_option("display.precision")
        value  = float(value) + 0.0
        if not math.isfinite(value):
            return str(value)
        if value and abs(value) < 10.0 ** -places:
            return f"{value:.{places}g}"
        shown = f"{value:,.{places}f}".rstrip("0")
        return shown + "0" if shown.endswith(".") else shown
    return str(value)


def line(settings):
    """``settings``, (name, value) pairs, as one line, a value of None left out."""
    return "; ".join(f"{name} {said(value)}" for name, value in settings if value is not None)


def noted(frame):
    """``frame``, whose last columns are ``level``, ``method`` and ``why_none``, as it
    shows, beside the lines under it: its level dropped where every interval it holds
    shares one, which is said under it, and each reason numbered in ``why_none``, the
    reasons under it, each once."""
    reasons = list(dict.fromkeys(why for why in frame["why_none"] if why))
    shown   = frame.copy()
    shown["why_none"] = [str(reasons.index(why) + 1) if why else ""
                         for why in frame["why_none"]]

    lines  = []
    levels = frame["level"].dropna().unique()
    if len(levels) <= 1:
        shown = shown.drop(columns="level")
        lines.append(line([("level", level) for level in levels]))

    return shown, lines + [f"{number}. {why}" for number, why in enumerate(reasons, start=1)]


def html_of(frame, lines):
    return frame._repr_html_() + "".join(UNDER.format(html.escape(text)) for text in lines
                                         if text)


def text_of(frame, lines):
    return "\n".join([repr(frame)] + [text for text in lines if text])
