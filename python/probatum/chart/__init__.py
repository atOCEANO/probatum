"""``probatum.chart``: a Data's bars and a run on them as one self-contained HTML
document, which ``Data.chart`` and ``Result.chart`` build and ``defaults`` sets the
look of; ``probatum.chart.marks`` holds the marks it draws.

Three steps and no fourth: build a compact spec dict, substitute it and the
vendored renderer into a template, and hand the string to a notebook cell or a
file. A cell's output is stored in the notebook, so a chart drawn this way is
still there, and still interactive, when the notebook is reopened with no kernel
behind it. There is no server, no build step and no bundler (ADR 0042).

This module computes nothing: every number it draws is the Data's, the Result's or
what probatum.stats reads of the Result, its drawdown among them. It does not import
pandas, pyarrow or IPython at module level.
"""

import copy
import functools
import html
import json
import math
import os
import pathlib
import re
import warnings

import numpy as np

from .._data import INTERVALS, SECOND, Data
from .._errors import (DataError, MissingDependency, ProbatumKeyError,
                       ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError,
                       ProbatumWarning)
from .._result import Result, Table
from ..stats import drawdown as _falls
from .marks import _MAX_STOPS, Line, Panel, _Mark, _color, _finite, _whole

__all__ = ["Chart", "defaults", "marks"]

PACKAGE = (os.path.dirname(os.path.dirname(__file__)),)

_STATIC = pathlib.Path(__file__).parent.parent / "_static"

# an explicit ordered tuple rather than a directory glob: nothing here executes at
# load, so the order is documentation, but a glob would silently ship any file
# that happened to land in _static (ADR 0042)
_ASSETS = (
    "core.js",
    "primitives/band.js",
    "primitives/background.js",
    "primitives/callout.js",
    "legend.js",
    "trades.js",
    "controls.js",
)

_PLACEHOLDER = re.compile(r"__[A-Z_]+__")

# a value that lands inside a <style> block or a canvas call rather than in the
# JSON payload, so escaping cannot save it and it is refused instead. A title is
# not in here: that one is escaped on both paths it travels
_UNSAFE = re.compile(r"[<>\";{}]")

_WEIGHTS = {"price": 5.0, "volume": 1.0, "equity": 2.0, "drawdown": 1.5}
_AUTO    = ("price", "volume", "equity", "drawdown")

# a notebook stores a cell's output verbatim, so a chart's data is paid for once
# in the .ipynb and again in every copy of it. Measured rather than guessed from
# the bar count, because the width of a row depends on how many series are on it
_BIG_BYTES = 8_000_000

# the candles rise in #089981 and fall in #f23645, both filled, in either theme: the
# one pair that clears 3:1 on both grounds and stands 11.1 apart for a deuteranope, so
# a round trip's win and loss and the equity's two sides take it too. The rest is the
# site's ink at its strengths, the muted text at 4.5:1 on its ground or better, and s1
# to s4 are the series handed out in order to lines that named no colour, s1 the ink
_PALETTES = {
    "dark":  {
        "surface": "#070d13", "plane": "#05090e", "grid": "#131c26",
        "axis": "#1f2a36", "ink": "#e9ebee", "ink2": "#aab1bb",
        "muted": "#8a939e", "hairline": "rgba(255,255,255,0.16)",
        "up": "#089981", "down": "#f23645", "win": "#089981", "loss": "#f23645",
        "s1": "#e9ebee", "s2": "#f2a33c", "s3": "#d8b4ff", "s4": "#8b97a5",
        "selected": ["rgba(8,153,129,0.24)", "rgba(8,153,129,0.10)",
                     "rgba(8,153,129,0.02)"],
    },
    "light": {
        "surface": "#ffffff", "plane": "#eef1f5", "grid": "#e3e8ee",
        "axis": "#c8d0d9", "ink": "#070d13", "ink2": "#48525e",
        "muted": "#5f6874", "hairline": "rgba(7,13,19,0.14)",
        "up": "#089981", "down": "#f23645", "win": "#089981", "loss": "#f23645",
        "s1": "#070d13", "s2": "#a15600", "s3": "#7a3fc4", "s4": "#6b7684",
        "selected": ["rgba(8,153,129,0.20)", "rgba(8,153,129,0.09)",
                     "rgba(8,153,129,0.02)"],
    },
}
# the colours a palette may name beside those, each its candle's own where it names none
_ADDED = ("wick_up", "wick_down", "volume_up", "volume_down")
_HEX   = re.compile(r"#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")
_RGB   = re.compile(r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*"
                    r"(?:,\s*(\d*\.?\d+)\s*)?\)")

# an empty palette rather than None, so the settings defaults() hands back put a
# session's palette back as they put back the rest
_DEFAULTS = {"theme": "auto", "height": 660, "palette": {}, "drawdown": "under",
             "price": "candles"}

# "auto" is not a third palette, it is a deferral: both of them ship in every
# document already, so the answer is read in the browser, where Python is not
# standing when a saved chart is opened a month later
_THEMES = ("dark", "light", "auto")
# the price as filled candles, candles whose rising bodies are their outline, OHLC bars
# or a line of the closes
_PRICES = ("candles", "hollow", "bars", "line")

# A background nobody gave a fill. Flat alpha at full height with a hard edge
# either side is the banding recipe, so this runs strongest at the floor and is
# gone before the ceiling: ground under the candles rather than a pane they sit
# inside. The edges stay hard, since a span covers exactly the bars it names
_BACKGROUND_FILL = ["rgba(139,151,165,0.20)", "rgba(139,151,165,0.02)"]


def _theme_mode(value):
    if value not in _THEMES:
        raise ProbatumValueError(f"theme: {value!r}; give 'dark', 'light' or 'auto', "
                                 f"which opens on the notebook's theme or else the "
                                 f"reader's system's")

    return value


def _price(value):
    if value not in _PRICES:
        raise ProbatumValueError(f"price: {value!r}; give 'candles', 'hollow' for a rising "
                                 f"candle drawn as its outline, 'bars' or 'line'")

    return value


def _palette(palette):
    if not isinstance(palette, dict):
        raise ProbatumTypeError(f"palette: {palette!r}; give a dict of 'dark' or 'light' "
                                f"to the colours each overrides")

    known  = sorted(set(_PALETTES["dark"]) | set(_ADDED))
    merged = {}
    for mode, values in palette.items():
        if mode not in _PALETTES:
            raise ProbatumValueError(f"palette: a key of {mode!r}; key the colours by "
                                     f"'dark' or 'light'")
        if not isinstance(values, dict):
            raise ProbatumTypeError(f"palette: {mode!r} to {values!r}; give {mode!r} a "
                                    f"dict of colour names to colours")

        kept = {}
        for key, value in values.items():
            if key not in known:
                raise ProbatumKeyError(f"palette: {key!r} under {mode!r}, a colour the "
                                       f"chart does not draw; use one of {', '.join(known)}")
            # a list only where the chart's own colour is one, a selected trade's fill,
            # whose stops one colour gives as a list of one
            listed = isinstance(_PALETTES["dark"].get(key), list)
            if listed and isinstance(value, str):
                value = [value]
            for colour in value if listed and isinstance(value, (list, tuple)) else [value]:
                if not isinstance(colour, str) or _channels(colour) is None:
                    raise ProbatumValueError(f"palette: {colour!r} for {key!r}; give a "
                                             f"colour as #rrggbb, #rrggbbaa or rgb(r, g, b)")
            kept[key] = _safe_value(value, "palette")
        merged[mode] = kept

    return merged


def _channels(value):
    """``value``'s red, green and blue and its opacity, or None for a colour the chart
    does not read."""
    if _HEX.fullmatch(value):
        digits = value[1:]
        if len(digits) < 6:
            digits = "".join(digit * 2 for digit in digits)
        parts = [int(digits[at:at + 2], 16) for at in range(0, len(digits), 2)]
        return (*parts[:3], parts[3] / 255.0 if len(parts) == 4 else 1.0)

    found = _RGB.fullmatch(value)
    if found is None:
        return None

    red, green, blue = (int(found.group(k)) for k in (1, 2, 3))
    opacity          = 1.0 if found.group(4) is None else float(found.group(4))
    if max(red, green, blue) > 255 or opacity > 1.0:
        return None
    return red, green, blue, opacity


def _faded(value, share):
    red, green, blue, opacity = _channels(value)
    return f"rgba({red},{green},{blue},{round(opacity * share, 3)})"


# How a run's drawdown is drawn. It is a fall from the running peak either way, so
# the choice is where to put it rather than what it says. "under" shades between
# the equity curve and its own peak on the equity panel: one axis, one pane, and
# the trough drawn directly beneath the high that caused it. "panel" is the pane it
# used to get, which is worth having when the equity axis is linear, because a
# shaded fall in quote currency draws a late twenty percent taller than an early
# one and a percentage pane does not. False draws neither
_DRAWDOWN = ("under", "panel", False)


def _drawdown_mode(value):
    # False by identity, since 0 == False would take a 0 for no drawdown while the
    # legend went on reading one
    if value is not False and not (isinstance(value, str) and value in _DRAWDOWN):
        raise ProbatumValueError(f"drawdown: {value!r}; give 'under' to shade the fall "
                                 f"on the equity panel, 'panel' for a pane of its own, "
                                 f"or False for neither")

    return value

# the four that decide whether a strategy is worth keeping. Formatting lives here
# rather than in the renderer, so the seam holds: Python says what the numbers
# are and what they read as, and the JavaScript only draws the pairs
_HEADLINE = ("total_return_pct", "sharpe", "max_drawdown_pct", "num_trades")

_STAT_LABELS = {
    "total_return_pct": ("return", "{:+.1f}%"),
    "net_profit_pct":   ("net", "{:+.1f}%"),
    "cagr_pct":         ("cagr", "{:+.1f}%"),
    "sharpe":           ("sharpe", "{:.2f}"),
    "sortino":          ("sortino", "{:.2f}"),
    "calmar":           ("calmar", "{:.2f}"),
    "max_drawdown_pct": ("max drawdown", "{:.1f}%"),
    "volatility_pct":   ("vol", "{:.1f}%"),
    "exposure_pct":     ("exposure", "{:.1f}%"),
    "win_rate":         ("win rate", "{:.0%}"),
    "profit_factor":    ("profit factor", "{:.2f}"),
    "avg_trade_pct":    ("avg trade", "{:+.2f}%"),
    "num_trades":       ("round trips", "{:,.0f}"),
    "num_fills":        ("fills", "{:.0f}"),
    "funding_paid":     ("funding", "{:+.2f}"),
}

_FONT = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace'

_bundled = None


@functools.lru_cache(maxsize=None)
def _asset(name):
    return (_STATIC / name).read_text(encoding="utf-8")


def _bundle():
    global _bundled
    if _bundled is None:
        _bundled = "\n".join(
            "// ---- " + name + "\n" + _asset(name) for name in _ASSETS
        )

    return _bundled


def _ipython():
    try:
        from IPython.display import HTML, display
    except ImportError:
        raise MissingDependency("show: IPython is not installed; pip install ipython, "
                                "or write the chart to a file with save(path)") from None

    return display, HTML


def _in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


def _safe(value, where):
    text = str(value)

    if _UNSAFE.search(text):
        raise ProbatumValueError(f"{where}: {value!r}; give one without <, >, quotes, "
                                 f"semicolons or braces")

    return text


def _safe_value(value, where):
    # 'selected' is a list of gradient stops rather than one colour, and str() on
    # a list is a repr carrying no character _UNSAFE looks for, so it would pass
    # the check and ship as one long string where the renderer needs three entries
    if isinstance(value, (list, tuple)):
        return [_safe(item, where) for item in value]

    return _safe(value, where)


def _height(value):
    value = _whole(value, "height")

    if value < 120:
        raise ProbatumValueError(f"height: {value}; give at least 120 pixels, the room "
                                 f"an axis needs")

    return value


def _start_for(size, num_bars, label, ahead=0):
    # the bar index a series of this length begins at (ADR 0042)
    if size == num_bars:
        return 0

    if size == num_bars - 1:
        return 1

    if ahead and size == num_bars + ahead:
        return 0

    lengths = f"{num_bars}, one a bar, or {num_bars - 1}, drawn from bar 1 as a diff is"
    if ahead:
        lengths += f", or {num_bars + ahead}, running {ahead} bars past the last candle"

    raise ProbatumValueError(f"{label}: {size} values for {num_bars} bars; give {lengths}")


def _align(values, num_bars, label, places=8, ahead=0):
    # a length T array starts at bar 0 and a length T-1 array at bar 1, as a diff
    # does. That second case is not a claim about when a value could first be
    # known: a number recorded inside Strategy.next was known on bar i, and the
    # right repair for one of those is to pad it to T. It is that drawing a short
    # array early is the visual signature of lookahead (ADR 0042). The leading
    # non-finite run folds into i0, so a rolling warm-up costs no bytes and the
    # alignment stays exact; an interior gap stays in place as None, because a hole
    # in the middle is information (ADR 0042)
    start  = _start_for(len(values), num_bars, label, ahead)
    finite = np.isfinite(values)
    if not finite.any():
        return start, []

    lead = int(np.argmax(finite))
    kept = finite[lead:]

    # rounding through numpy and letting tolist build the Python floats is the whole
    # of the work when nothing is missing, which is the ordinary case: an equity
    # curve, a volume column and any rolling feature are gapless once warmed up.
    # Only a series with a hole in it pays for the per-bar None substitution.
    # Unlike a price, a volume or a mean is not tick-aligned, so a value sitting on
    # a tie can round the other way from round(); places is two past what the panel
    # displays, so the disagreement is a unit of something already below reading
    rounded = np.round(values[lead:], places)
    if kept.all():
        return start + lead, rounded.tolist()

    tail = [
        None if not ok else v
        for v, ok in zip(rounded.tolist(), kept.tolist())
    ]
    return start + lead, tail


def _track(values, num_bars, label, places=8, ahead=0):
    # rounded to two places past what the panel displays. A price is stored as a
    # double and a naive round(x, 8) keeps every one of its thirteen significant
    # digits, which on a 100k bar chart is megabytes of notebook nobody can read
    i0, v = _align(np.asarray(values, dtype=np.float64), num_bars, label, places, ahead)
    return {"i0": i0, "v": v}


def _encode_color(value, num_bars, label, ahead=0):
    if value is None or isinstance(value, str):
        return value

    start   = _start_for(len(value), num_bars, label, ahead)
    palette = [None]
    seen    = {}
    keys    = []

    for item in value:
        if item is None:
            keys.append(0)
            continue

        at = seen.get(item)
        if at is None:
            at = len(palette)
            palette.append(item)
            seen[item] = at
        keys.append(at)

    return {"i0": start, "p": palette, "k": keys}


def _truthy(value):
    # a NaN is not a condition being met, for the reason at_bar's docstring gives.
    # pandas.NA answers neither way and raises on bool(), and a value that cannot
    # answer the question is by definition not a condition being met
    if value is None:
        return False

    if isinstance(value, float) and math.isnan(value):
        return False

    try:
        return bool(value)
    except TypeError:
        return False


def _spans(values, fill, num_bars, label, ahead=0):
    start = _start_for(len(values), num_bars, label, ahead)

    if isinstance(fill, dict):
        table = {}
        fills = []
        # the names, in the order the fills are in, so a named background can say
        # which region the crosshair is over. A mask has no names to send, and
        # sends none rather than a list of nulls
        keys = []

        for key, stops in fill.items():
            # a label mapped to nothing shades nothing, which is what an absent
            # key already means. It must not reach fills: the renderer builds its
            # gradient cache by mapping over every entry
            if stops is None:
                continue

            table[key] = len(fills)
            fills.append(stops)
            keys.append(str(key))

        def at(value):
            try:
                return table.get(value)
            except TypeError:
                return None
    else:
        fills = [fill if fill else list(_BACKGROUND_FILL)]
        keys  = None

        def at(value):
            return 0 if _truthy(value) else None

    spans   = []
    open_at = -1
    key     = None

    for offset in range(len(values) + 1):
        here = at(values[offset]) if offset < len(values) else None
        if here != key:
            if key is not None and open_at >= 0:
                spans.append([start + open_at, start + offset, key])
            key     = here
            open_at = offset if here is not None else -1

    return spans, fills, keys


def _extent(values):
    arr    = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return None

    return float(finite.min()), float(finite.max())


def _digits(extent):
    if extent is None:
        return 2

    span = extent[1] - extent[0]
    if span <= 0:
        span = abs(extent[1]) or 1.0

    places = int(math.ceil(-math.log10(span / 1000.0)))
    return max(0, min(8, places))


def _edges(mark, num_bars, index=0, ahead=0):
    # every value a mark puts on its panel's axis, as (label, values, start),
    # where start is the bar the first entry is drawn on or None for a value that
    # is not per-bar. One accessor, because the panel extent and the log guard
    # both need this answer and used to disagree: a band's lower edge moved
    # neither, and a level moved the extent but not the guard
    label = _label(mark, index)

    if mark.kind in ("line", "histogram"):
        start = _start_for(len(mark.values), num_bars, label, ahead)
        out   = [(label, mark.values, start)]
        if mark.kind == "histogram":
            out.append((f"{label} base", np.full(1, mark.base), None))
        return out

    if mark.kind == "band":
        start = _start_for(len(mark.upper), num_bars, label, ahead)
        out   = [(f"{label} upper", mark.upper, start)]
        if mark.lower is None:
            out.append((f"{label} lower", np.full(1, mark.level), None))
        else:
            below = _start_for(len(mark.lower), num_bars, label, ahead)
            out.append((f"{label} lower", mark.lower, below))
        return out

    if mark.kind == "level":
        return [(label, np.full(1, mark.value), None)]

    return []


def _place(extent, price):
    # overlay unless doing so costs the candles more than half the price panel.
    # Range overlap was the obvious rule and it carries no information about
    # units: an oscillator bounded 0 to 100 overlaps a coin priced 20 to 90, rides
    # the candles and stretches the axis. This rule cannot flatten them, because
    # the threshold is the guarantee (ADR 0042)
    if extent is None or price is None or price[1] <= price[0]:
        return True

    span = max(price[1], extent[1]) - min(price[0], extent[0])
    return span <= 0 or (price[1] - price[0]) / span >= 0.5


def _dead(values):
    arr    = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return "every value is NaN"

    if float(finite.min()) == float(finite.max()):
        return f"every value is {float(finite.min())}"

    return None


def _panel_key(name, index):
    # a panel created from a mark's label is an identity as well as a header, and
    # controls.js writes it into a DOM attribute. The derived form is stripped
    # rather than refused, because the caller wrote a legend label and not a panel
    # name; a panel= they typed themselves still raises. A label that collides
    # with an automatic panel falls back too, so a series named "price" cannot
    # land on the candles _place just refused it from
    key = _UNSAFE.sub("", name or "").strip()
    return key if key and key not in _AUTO else f"series {index}"


def _label(mark, index):
    # an unnamed mark reports its position, because every bare array is wrapped as
    # a Line and "Line" is the same word for all four of the ones just passed
    if mark.name:
        return f"{type(mark).__name__} {mark.name!r}"
    return f"series {index}"


def _flatten(arg):
    # the one ambiguous case: a list whose every element is a number is one
    # series, and anything else is a list of series
    if isinstance(arg, (list, tuple)):
        if arg and all(isinstance(x, (int, float, np.number)) for x in arg):
            return [arg]

        out = []
        for item in arg:
            out.extend(_flatten(item))
        return out

    return [arg]


def _collect(given):
    marks = []

    for arg in given:
        for item in _flatten(arg):
            if item is None:
                continue
            if isinstance(item, _Mark):
                marks.append(item)
            elif isinstance(item, Result):
                # the chart draws one run, its own, and another run's equity is a
                # series on the same rows
                raise ProbatumTypeError("marks: a Result; draw another run's equity as "
                                        "Line(other.equity, \"its name\", "
                                        "panel=\"equity\")")
            else:
                marks.append(Line(item))

    return marks


def defaults(theme=None, height=None, palette=None, drawdown=None, price=None):
    """Set how every later chart looks, and return the previous settings.

    Explicit keywords rather than ``**options``, so ``defaults(them="dark")`` is a
    ``TypeError`` at the call rather than a key stored and never read. Only
    appearance lives here, never what gets drawn. A document is built when the
    chart is, so changing a default afterwards changes no chart that already
    exists and no file already saved.

    ``height`` is the height of a notebook cell, in pixels. A saved file sizes
    itself to the window instead, so one file is right on any screen. There is no
    width anywhere: a chart fills whatever contains it.

    ``theme`` and ``palette`` are both here and on each chart, and mean the same
    thing in either place. ``theme="auto"``, the default, opens each chart on the
    notebook's own theme, or else on whichever of the two palettes the reader's
    system asks for.

    ``drawdown`` is ``"under"``, ``"panel"`` or ``False``, and it is here as well
    as on ``Result.chart`` so a session that wants the separate pane back gets it
    in one line rather than at every call site. ``price`` is ``"candles"``,
    ``"hollow"``, ``"bars"`` or ``"line"``, as on each chart.
    """
    previous = dict(_DEFAULTS)
    pending  = {}

    if theme is not None:
        pending["theme"] = _theme_mode(theme)
    if height is not None:
        pending["height"] = _height(height)
    if drawdown is not None:
        pending["drawdown"] = _drawdown_mode(drawdown)
    if palette is not None:
        pending["palette"] = _palette(palette)
    if price is not None:
        pending["price"] = _price(price)

    # applied only once every argument has passed. A call that raised halfway
    # through used to leave the theme set and the height not, and the caller never
    # got the previous values back to restore from
    _DEFAULTS.update(pending)
    return previous


def _assign(marks, ohlc):
    price = (float(np.nanmin(ohlc[:, 2])), float(np.nanmax(ohlc[:, 1])))
    names = []

    for index, mark in enumerate(marks):
        if mark.panel is not None:
            names.append(mark.panel)
            continue

        if mark.kind in ("line", "histogram", "band"):
            values = mark.upper if mark.kind == "band" else mark.values
            extent = _extent(values)
            if _place(extent, price):
                names.append("price")
                if extent is not None:
                    # the price extent widens as each overlay is accepted, so an
                    # early one can push a later one out. Order dependent, and
                    # deterministic for a given argument order
                    price = (min(price[0], extent[0]), max(price[1], extent[1]))
            else:
                names.append(_panel_key(mark.name, index))
        else:
            # a level, a marker or a background with no panel= goes on the price
            # panel, and a level is the one that can vanish there: the renderer
            # draws it with createPriceLine, which the autoscale never reads, so
            # Level(70) beside a coin at 61,000 is off screen with nothing said.
            # The oscillator idiom that puts a line and its thresholds together
            # is the most common chart in the world and it failed in silence
            if mark.kind == "level" and not _within(mark.value, price):
                warnings.warn(f"Level({mark.value:g}): no panel=, so it went on the "
                              f"price panel, where the candles run {price[0]:g} to "
                              f"{price[1]:g} and it cannot be seen; give it the panel= "
                              f"of the series it belongs with", ProbatumWarning,
                              skip_file_prefixes=PACKAGE)

            names.append("price")

    return names


def _within(value, extent):
    # generous, because a level just outside the range is a target and belongs on
    # the chart; a level orders of magnitude away is on the wrong panel
    low, high = extent
    room = (high - low) or abs(high) or 1.0
    return low - room <= value <= high + room


def _order(names, config, has_volume, has_result, drawdown="under"):
    order = ["price"]
    if has_volume:
        order.append("volume")

    for name in names:
        if name not in order:
            order.append(name)

    if has_result:
        # the drawdown earns a pane of its own only when it was asked for one.
        # Shaded on the equity panel it is a band rather than a panel, so it never
        # enters this list and nothing downstream has to know the difference
        auto = ("equity", "drawdown") if drawdown == "panel" else ("equity",)
        for name in auto:
            if name not in order:
                order.append(name)

    # panels= fixes the relative order of the panels it names without moving them
    # relative to the panels it does not: take the slots those panels already
    # occupy and write them back in the listed order. Naming one panel therefore
    # provably changes no order at all
    listed = [p.name for p in config if p.name in order]
    if len(listed) > 1:
        slots = sorted(order.index(name) for name in listed)
        for slot, name in zip(slots, listed):
            order[slot] = name

    return order


def _build_panels(order, config, marks, names, num_bars, ohlc, vol, equity, dd,
                  ahead=0):
    by_name = {p.name: p for p in config}

    if by_name.get("price") is not None and not by_name["price"].show:
        raise ProbatumValueError("panels: Panel('price', show=False), and the price "
                                 "panel carries the candles; hide the other panels "
                                 "instead, or give a shorter height")

    hidden = {p.name for p in config if not p.show}
    panels = []

    for name in order:
        if name in hidden:
            continue

        cfg  = by_name.get(name)
        pool = []

        if name == "price":
            pool.append((float(np.nanmin(ohlc[:, 2])), float(np.nanmax(ohlc[:, 1]))))
        if name == "volume" and vol is not None:
            pool.append(_extent(vol))
        if name == "equity" and equity is not None:
            pool.append(_extent(equity))
        if name == "drawdown" and dd is not None:
            pool.append(_extent(dd))

        for index, (mark, on) in enumerate(zip(marks, names)):
            if on != name:
                continue
            pool.extend(
                _extent(values)
                for _, values, _ in _edges(mark, num_bars, index, ahead)
            )

        pool   = [p for p in pool if p is not None]
        extent = (min(p[0] for p in pool), max(p[1] for p in pool)) if pool else None

        panel = {
            # controls.js writes this into a DOM attribute, which the payload
            # escaping cannot reach because the script reads the payload as a
            # literal, whose escapes give back the original
            "name": _safe(name, "panel"),
            "weight": (float(cfg.weight) if cfg and cfg.weight is not None
                       else _WEIGHTS.get(name, 1.5)),
            "scale": cfg.scale if cfg else "linear",
            "digits": _digits(extent),
        }
        if cfg is not None and cfg.range is not None:
            panel["range"] = [cfg.range[0], cfg.range[1]]
        if name == "price":
            panel["candles"] = True
        if name == "volume" and vol is not None:
            panel["volume"] = True

        panels.append(panel)

    return panels, hidden


def _check_log(panels, marks, names, ohlc, tracks, ahead=0):
    num_bars = len(ohlc)

    for panel in panels:
        if panel["scale"] != "log":
            continue

        pools = []
        if panel.get("candles"):
            pools.append(("the candles", ohlc[:, 2], 0))
        for key, values in tracks.items():
            if key == panel["name"] and values is not None:
                start = _start_for(len(values), num_bars, key, ahead)
                pools.append((key, values, start))
        for index, (mark, on) in enumerate(zip(marks, names)):
            if on == panel["name"]:
                pools.extend(_edges(mark, num_bars, index, ahead))

        for label, values, start in pools:
            arr = np.asarray(values, dtype=np.float64)
            bad = np.flatnonzero(np.isfinite(arr) & (arr <= 0))
            if not bad.size:
                continue

            # the index into the array is not the bar it is drawn on: a T-1 series
            # starts at bar 1, and naming a bar one early is exactly the confusion
            # the alignment contract exists to prevent (ADR 0042)
            at    = int(bad[0])
            where = "" if start is None else f" at bar {start + at}"
            raise ProbatumValueError(f"panels: {panel['name']!r} is logarithmic, and "
                                     f"{label} reaches {float(arr[at])}{where}; give it "
                                     f"scale='linear', or shift the series above zero")


def _focus(value, num_bars, rows, ahead=0):
    last = num_bars + ahead - 1.0

    if value is None:
        return None

    if isinstance(value, dict) and "open_bar" in value:
        # a round trip's bars are the run's, which rows places on the Data's, one
        # still open running to the run's last
        if "close_bar" not in value:
            raise ProbatumTypeError(f"focus: {value!r}; give a round trip of the run, as "
                                    f"result.trades[0]")
        trip = [value["open_bar"],
                len(rows) - 1 if value["close_bar"] is None else value["close_bar"]]
        if not all(isinstance(bar, (int, np.integer)) and 0 <= bar < len(rows)
                   for bar in trip):
            raise ProbatumValueError(f"focus: a round trip on bars {value['open_bar']} to "
                                     f"{value['close_bar']} of a run of {len(rows)}; give "
                                     f"one of the run's own round trips")
        opened, closed = (int(rows[bar]) for bar in trip)
        pad    = max(18.0, max(1, closed - opened) * 0.7)
        low, high = opened - pad, closed + pad
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        # the last N bars, plus whatever was projected past them: asking for a
        # window and being shown less than the chart holds is never what was meant
        if value < 1:
            raise ProbatumValueError(f"focus: {value}; give a number of bars from 1")
        low, high = num_bars - int(value), last
    else:
        try:
            low, high = value
        except (TypeError, ValueError):
            raise ProbatumTypeError(f"focus: {value!r}; give a round trip, a (start, "
                                    f"end) pair of bars or a number of bars") from None
        low, high = _finite(low, "focus"), _finite(high, "focus")

    # clamped rather than sliced, so every marker stays on its own bar whatever
    # was asked for. The cap goes last: widening to a one bar minimum first and
    # capping second used to emit a range one bar past the end of the axis
    high = min(max(float(high), float(low) + 1.0), last)
    low  = max(0.0, min(float(low), high - 1.0))
    return [low, high]


def _trades(result, name, rows):
    # the round trips the run made on the instrument drawn, each on the Data's rows of
    # the run's bars it opened and closed on, one still open standing at the run's last
    # bar with its pnl so far. Columns rather than rows, so a run of many trips builds
    # its list from arrays
    trades = result.trades
    kept   = np.flatnonzero(trades["instrument"] == name)
    if not kept.size:
        return []

    still  = np.asarray(trades["open"])[kept]
    opened = trades["open_bar"][kept]
    closed = np.where(still, len(rows) - 1, np.ma.getdata(trades["close_bar"])[kept])
    money  = [[round(value, 8) if math.isfinite(value) else None
               for value in trades[column][kept].tolist()]
              for column in ("size", "entry", "exit", "fees", "funding", "pnl", "ret",
                             "best", "worst")]

    out = []
    for number, (start, end, bars, side, size, entry, exit, fees, funding, pnl, ret, best,
                 worst, reason, held) in enumerate(zip(
                     rows[opened].tolist(), rows[closed].tolist(),
                     (closed - opened).tolist(), trades["side"][kept].tolist(), *money,
                     trades["reason"][kept].tolist(), still.tolist()), 1):
        trade = {"i": number, "in": start, "out": end, "side": side, "size": size,
                 "px_in": entry, "px_out": exit, "fees": fees, "fund": funding,
                 "net": pnl, "ret": ret, "best": best, "worst": worst, "bars": bars,
                 "why": "open" if held else reason or ""}

        # only when it is set, so an ordinary run spends no bytes on it. A forced
        # close is the one row a reader must not mistake for a decision
        if held:
            trade["open"] = True
        if reason == "liquidation":
            trade["liq"] = True

        out.append(trade)

    return out


def _headline(result, chosen, name, several):
    stats = result.stats
    if chosen is None:
        keys = _HEADLINE
    elif isinstance(chosen, str) or not hasattr(chosen, "__iter__"):
        raise ProbatumTypeError(f"stats: {chosen!r}; give a list of the names of the "
                                f"run's statistics")
    else:
        keys = list(chosen)

    out = []
    for key in keys:
        if key not in stats:
            raise ProbatumKeyError(f"stats: {key!r}, which the run does not report; use "
                                   f"one of {', '.join(stats)}")

        value = float(stats[key])
        label, form = _STAT_LABELS.get(key, (key, "{:.2f}"))
        shown = form.format(value) if math.isfinite(value) else "n/a"
        out.append([label, _tripped(result, name, several, shown)
                    if key == "num_trades" else shown])

    return out


def _tripped(result, name, several, shown):
    """The round trips a headline counts, ``shown``: one instrument's of the account's
    where the account trades several, and those still open."""
    trades = result.trades
    still  = np.asarray(trades["open"])
    mine   = np.asarray(trades["instrument"] == name) if several else ~np.zeros_like(still)
    if several:
        shown = f"{int((mine & ~still).sum()):,} of {shown}"

    held = int((mine & still).sum())
    return shown + (f", {held:,} open" if held else "")


def _cell(value):
    # str() on a float is 1.9400000000000002, and this table is read rather than
    # computed with. Six significant digits keeps a sharpe short and a basis point
    # from rounding to zero, which a fixed number of decimal places cannot do both
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "n/a"
    if value is None:
        return ""
    # pandas' missing values read as a missing float does, not as their names
    if type(value).__name__ in ("NAType", "NaTType"):
        return "n/a"
    return str(value)


def _notes(notes):
    """Turn ``notes`` into the header and rows the panel under the plot draws.

    A pandas or polars DataFrame, a pyarrow Table, or a list of rows whose first row
    is the header. A DataFrame's index is not drawn: drawing it whenever it looked
    meaningful would be a rule nobody could predict, and ``reset_index()`` says so at
    the call site instead.
    """
    if isinstance(notes, Table):
        # its rows are dicts, whose own list is their keys, so read as rows it would
        # repeat the header on every line
        raise ProbatumTypeError("notes: a Table; give table.to_pandas(), or rows whose "
                                "first names the columns")

    library = type(notes).__module__.split(".")[0]
    if library == "pandas":
        head = [_cell(c) for c in notes.columns]
        rows = [[_cell(v) for v in row] for row in notes.itertuples(index=False)]
    elif library == "polars":
        # a polars or pyarrow table iterates its columns, which read as rows would
        # draw the table on its side
        head = [_cell(c) for c in notes.columns]
        rows = [[_cell(v) for v in row] for row in notes.rows()]
    elif library == "pyarrow":
        head = [_cell(c) for c in notes.column_names]
        rows = [[_cell(v) for v in row.values()] for row in notes.to_pylist()]
    else:
        readable = not isinstance(notes, (str, dict)) and hasattr(notes, "__iter__")
        table    = list(notes) if readable else None
        if table is None or any(isinstance(row, (str, dict))
                                or not hasattr(row, "__iter__") for row in table):
            # a string, a dict of columns or a row that is no list of cells would be
            # read a letter or a key at a time
            raise ProbatumTypeError(f"notes: {notes!r}; give a DataFrame, or rows whose "
                                    f"first names the columns")
        if not table:
            return None
        head = [_cell(c) for c in table[0]]
        rows = [[_cell(v) for v in row] for row in table[1:]]

    if not head:
        raise ProbatumValueError("notes: no columns; give a DataFrame with columns, or "
                                 "rows whose first names them")

    for index, row in enumerate(rows):
        if len(row) != len(head):
            raise ProbatumValueError(f"notes: row {index} of {len(row)} cells under a "
                                     f"header of {len(head)}; give every row the "
                                     f"header's cells")

    if not rows:
        return None

    return {"head": head, "rows": rows}


def _theme(mode, palette):
    """Both palettes as the document carries them: the caller's colours over the
    chart's, the wicks and the volume the bodies' where the caller names none, and the
    washes faded here from whatever form a colour came in, so the renderer composes no
    alpha of its own (ADR 0042)."""
    out = {"mode": mode, "font": _FONT}
    for key in ("dark", "light"):
        merged = dict(_PALETTES[key])
        if palette and key in palette:
            merged.update(palette[key])

        merged.setdefault("wick_up", merged["up"])
        merged.setdefault("wick_down", merged["down"])
        merged.setdefault("volume_up", _faded(merged["up"], 0.4))
        merged.setdefault("volume_down", _faded(merged["down"], 0.4))
        merged.update(ddTop=_faded(merged["loss"], 0.0), ddBottom=_faded(merged["loss"], 0.267),
                      edge=_faded(merged["ink"], 0.07), edgeClear=_faded(merged["ink"], 0.0))
        out[key] = merged

    return out


def _payload(spec):
    """``spec`` as the JSON its document carries, warned of where a notebook would
    struggle to carry it."""
    payload = json.dumps(spec, separators=(",", ":"), allow_nan=False)

    if len(payload) > _BIG_BYTES:
        # gzip was measured before writing this: 3.1x raw, and 2.3x once base64
        # armours it back into a document, which turns 45 MB into 17 MB and solves
        # nothing while costing an async bootstrap and a modern-browser floor.
        # Fewer bars is the only real answer, so the warning names one
        bars = spec.get("n", 0)
        keep = max(1_000, 10 ** (len(str(bars // 20)) - 1) * 2)
        warnings.warn(f"chart: {len(payload) // 1_000_000} MB over {bars:,} bars, which "
                      f"a notebook stores in every copy of itself, so it draws but opens "
                      f"slowly; chart fewer bars, as data[-{keep:_}:], or the bars "
                      f"data.resample makes coarser", ProbatumWarning,
                      skip_file_prefixes=PACKAGE)

    # json.dumps does not escape <, so a series named "</script>" would close the
    # script element and take the document with it
    return (
        payload.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _render(payload, title):
    values = {
        "__TITLE__":   html.escape(title or "probatum chart"),
        "__CSS__":     _asset("chart.css"),
        "__LIBRARY__": _asset("lightweight-charts.js"),
        "__BUNDLE__":  _bundle(),
        "__SPEC__":    payload,
    }

    # one re.sub pass rather than chained .replace(): the vendored renderer is
    # 196 KB of minified JavaScript, and a chained replace would happily expand a
    # placeholder that appeared inside text already substituted
    return _PLACEHOLDER.sub(lambda m: values[m.group(0)], _asset("chart.html"))


class Chart:
    """One built chart: the document, and the spec it was built from. A notebook draws
    it where it is a cell's last expression, and ``show()`` anywhere in a cell."""


    def __init__(self, spec, title):
        self._spec     = spec
        self._title    = title
        # made with the chart rather than with its document, so a chart too heavy for
        # a notebook is warned of at the line that built it: by the time a cell's last
        # expression is drawn, no line of the caller's is left to name
        self._payload  = _payload(spec)
        self._document = None


    def __repr__(self):
        return f"Chart(bars={self._spec['n']}, panels={len(self._spec['panels'])})"


    def spec(self):
        """The underlying document as a plain dict.

        A fresh copy each call, so a caller can pull it apart without changing
        what gets drawn. This is what the tests assert against, because a spec
        says what is drawn where a screenshot only says it looked the same.
        """
        return copy.deepcopy(self._spec)


    def save(self, path):
        """Write one self-contained HTML file and return ``path``.

        For when somebody wants the picture without the notebook. The renderer,
        the styles and the data are all inside the one file, so it opens in a
        browser on a machine that has no Python. Missing parent directories are
        created.

        The document sizes itself to the window rather than to ``height``, which
        is a notebook cell setting. That is what makes one file right on a laptop
        and on a wide monitor, and it is what lets another page embed it in an
        iframe of the page's own choosing.
        """
        if not isinstance(path, (str, os.PathLike)):
            raise ProbatumTypeError(f"path: {path!r}; give the file's path, a str or a "
                                    f"pathlib.Path")

        target = pathlib.Path(path)
        if target.parent != pathlib.Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(self._html(), encoding="utf-8", newline="\n")
        return path


    def show(self, height=None):
        """Draw the chart in the current notebook cell.

        ``height`` overrides the cell height for this one display, without
        rebuilding anything: the document is already built and only the frame
        around it reads a height. Width is always the width of the cell, because
        a cell is a container of unknown size and a pinned width is how a chart
        ends up cut off on one screen and short of the edge on another.

        An iframe with ``srcdoc`` rather than a bare ``HTML()``: JupyterLab strips
        script tags out of raw HTML output, while an iframe is its own document
        and always runs, and a cell's output is stored inside the ``.ipynb``, so
        the chart survives the notebook being saved, closed and reopened with no
        kernel and no network.

        Displays rather than returning, so a chart drawn inside a loop draws every
        time instead of only on the last expression, and returns ``None``. Raises a
        ``ProbatumRuntimeError`` outside a notebook, where there is nothing to draw
        into and a silent no-op is the expensive kind of nothing.
        """
        if not _in_notebook():
            raise ProbatumRuntimeError("show: no notebook here to draw into; write the "
                                       "chart to a file with save(path)")

        display, HTML = _ipython()
        display(HTML(self._framed(
            self._spec["height"] if height is None else _height(height)
        )))


    def _repr_html_(self):
        """The chart as a notebook draws it where it is a cell's last expression, in the
        frame ``show()`` draws it in, at the chart's own height."""
        return self._framed(self._spec["height"])


    def _framed(self, height):
        # wrapped, because IPython warns on output that opens with a bare iframe
        # and points at IPython.display.IFrame, which takes a src url and so
        # cannot carry its document inline, which is the whole point of this one
        srcdoc = html.escape(self._html(), quote=True)
        return (f'<div class="probatum-chart">'
                f'<iframe srcdoc="{srcdoc}" style="width:100%;height:{height}px;'
                f'border:0"></iframe></div>')


    def _html(self):
        if self._document is None:
            # the payload is held by the document from here, and kept twice it would
            # double what a heavy chart holds
            self._document, self._payload = _render(self._payload, self._title), None

        return self._document


def _ran(result, data):
    """The Data ``result`` ran on and the Data's row of each of the run's: its own, or
    ``data``, held to the fingerprint the run recorded, for a Result that keeps none."""
    given = result.provenance
    if "results" in given:
        raise ProbatumValueError(f"result: a book of {' and '.join(given['results'])}, "
                                 f"whose accounts ran on bars of their own; chart each "
                                 f"account's Result")

    record = result._record
    if data is None:
        if record is None:
            raise ProbatumValueError("data: none, and a Result loaded, unpickled or "
                                     "recorded below the full level keeps no Data; give "
                                     "the Data the run read")
        data = record.data
    elif not isinstance(data, Data):
        raise ProbatumTypeError(f"data: {type(data).__name__}; give the Data the run "
                                f"read")
    elif data.fingerprint != given["data"]["fingerprint"]:
        raise DataError(f"data: bars of fingerprint {data.fingerprint}, where the run "
                        f"read {given['data']['fingerprint']}; give the Data the run read")

    # the run's rows found by their times, since a run recorded only rows one of its
    # instruments printed, from its first or, a validation's piece, a later one, and
    # a Result taken part way holds them up to the current one
    return data, np.searchsorted(data.time, result.time)


def _column(data, instrument):
    """The column of the instrument whose candles are drawn: ``instrument``, or the
    Data's one where it is None."""
    names = data.names
    if instrument is None:
        if len(names) != 1:
            raise ProbatumValueError(f"instrument: none, for a Data of {len(names)} "
                                     f"instruments; name the one whose candles to draw, "
                                     f"as instrument=\"{names[0]}\"")
        return 0

    if not isinstance(instrument, str):
        raise ProbatumTypeError(f"instrument: {instrument!r}; give an instrument's name, "
                                f"as instrument=\"{names[0]}\"")
    if instrument not in names:
        raise ProbatumKeyError(f"instrument: no instrument {instrument!r}; use one of "
                               f"{', '.join(names)}")

    return names.index(instrument)


# the array each kind of mark is read from, whose length its other edge and its
# anchors share
_PRIMARY = {"line": "values", "histogram": "values", "band": "upper", "markers": "mask",
            "background": "values"}


def _per_bar(mark):
    """Each array of ``mark`` that holds a value a bar, beside the word that names it:
    the one it is read from, and a colour or a fill given one a bar."""
    out   = [("", getattr(mark, _PRIMARY[mark.kind]))] if mark.kind in _PRIMARY else []
    color = getattr(mark, "color", None)
    fill  = getattr(mark, "fill", None)
    if isinstance(color, list):
        out.append((" color", color))
    if isinstance(fill, list) and len(fill) > _MAX_STOPS:
        out.append((" fill", fill))

    return out


def _unplaced(marks, candle_color, rows, count):
    """Refuse an array of ``marks`` or ``candle_color`` as long as the ``rows`` a run
    recorded, where those are not all ``count`` rows of its Data."""
    # a run's own arrays hold a value a row it recorded, and drawn as the Data's they
    # land off their bars, without complaint where one row short reads as a diff.
    # Its rows from bar 1 to the last are the one case a diff and the run's array
    # are drawn alike
    if len(rows) == count - 1 and rows[0] == 1:
        return

    given = [(_label(mark, index) + word, values)
             for index, mark in enumerate(marks) for word, values in _per_bar(mark)]
    if candle_color is not None and np.ndim(candle_color) == 1:
        given.append(("candle_color", candle_color))

    for label, values in given:
        if len(values) == len(rows):
            diff = (", and a diff as numpy.diff(x, prepend=numpy.nan)"
                    if len(rows) == count - 1 else "")
            raise ProbatumValueError(f"{label}: {len(rows)} values for {count} bars, as "
                                     f"many as the rows the run recorded, which are not "
                                     f"all the Data's; give {count}, one a bar, an array "
                                     f"of the run's placed on the Data's rows at "
                                     f"numpy.searchsorted(data.time, result.time){diff}")


def _placed(values, rows, count):
    """``values``, one for each of a run's rows, on the ``count`` rows of its Data,
    ``rows`` being each one's, and NaN on the rows the run never saw."""
    out = np.full(count, np.nan)
    out[rows] = values
    return out


def _candles(ohlc, shown):
    """The candles as the document carries them: rounded to the precision their own
    range needs, and null where the instrument printed no bar."""
    # numpy's own tolist() here, never a comprehension over the candles. They are the
    # largest thing in the document by far, and rounding each of the four values in
    # Python was two thirds of the cost of a chart: tolist() already materialises T
    # lists, and the comprehension built T more on top of them. The rounding was
    # tried without and rejected: it costs nothing on exchange data, whose reprs are
    # already short, and 2.6x on bars somebody computed. numpy rounds by scaling, so
    # it can break a tie the other way from round(); a price is a whole number of
    # ticks, which puts it nowhere near a tie, and 2.48M real prices came out
    # identical. The precision comes from the candles' own range and NOT from the
    # price panel's digits, which every mark sharing that panel contributes to: a
    # Level(70) beside a coin priced 0.000012 dragged the panel from eight digits to
    # two and rounded every candle to 0.0, and one Line reaching a million took XRP
    # from 4990 distinct candles to 52. What is displayed is a panel setting; what is
    # stored must depend on the data alone
    rows = np.round(ohlc, _digits(_extent(ohlc)) + 2).tolist()
    if not shown.all():
        rows = [row if printed else None for row, printed in zip(rows, shown.tolist())]

    return rows


def _drawn(data, given, instrument=None, result=None, rows=None, panels=None, focus=None,
           candle_color=None, theme=None, palette=None, height=None, title=None,
           future=0, stats=None, trades=True, drawdown=None, notes=None, price=None):
    """The chart of ``data``'s bars of ``instrument`` with the marks ``given`` and, where
    it is given, ``result``, a run on them whose rows are the Data's ``rows``, as
    `Data.chart` and `Result.chart` describe."""
    num_bars = len(data)
    if not num_bars:
        raise DataError("data: no bars; chart a Data that holds at least one")

    k      = _column(data, instrument)
    name   = data.names[k]
    bars   = data.bars[:, k]
    shown  = data.available[:, k]
    if not shown.any():
        raise DataError(f"instrument: {name} printed no bar in the Data; chart a span "
                        f"it trades in")

    rows   = np.arange(num_bars) if rows is None else rows
    times  = data.time // SECOND
    future = _whole(future, "future")
    if future < 0:
        raise ProbatumValueError(f"future: {future}; give a whole number of bars from 0")

    if future:
        # the axis grows and the candles do not. Everything that paints a candle
        # loops the candle count, so nothing can invent a bar that never traded
        step  = INTERVALS[data.interval] // SECOND
        times = np.concatenate(
            [times, times[-1] + step * np.arange(1, future + 1, dtype=np.int64)]
        )

    marks = _collect(given)
    if len(rows) != num_bars:
        _unplaced(marks, candle_color, rows, num_bars)

    try:
        config = list(panels or [])
    except TypeError:
        raise ProbatumTypeError(f"panels: {panels!r}; give a list of Panel objects of "
                                f"probatum.chart.marks") from None

    seen = set()
    for panel in config:
        if not isinstance(panel, Panel):
            raise ProbatumTypeError(f"panels: {panel!r}; give Panel objects of "
                                    f"probatum.chart.marks")
        if panel.name in seen:
            # list.index returns the first match, so a repeated name yields the
            # same slot twice and the second write clobbers the first, deleting a
            # panel and duplicating another
            raise ProbatumValueError(f"panels: {panel.name!r} twice; configure a panel "
                                     f"with one Panel, or the second wins on some "
                                     f"settings and the first on others")
        seen.add(panel.name)

    ohlc   = bars[:, :4]
    volume = bars[:, 4]
    dead   = _dead(volume)
    if dead is not None:
        warnings.warn(f"volume: {dead}, so no volume panel is drawn; a dead feed draws "
                      f"as a plausible flat chart", ProbatumWarning,
                      skip_file_prefixes=PACKAGE)
        volume = None

    dd_mode = _drawdown_mode(
        drawdown if drawdown is not None else _DEFAULTS["drawdown"]
    )

    equity = None
    dd     = None
    peak   = None
    if result is not None:
        # the run's own equity from the balance it opened with on row 0, and the fall
        # from its running peak as probatum.stats reads it, whose deepest is the
        # record's maximum drawdown, so the drawn fall and the reported one cannot
        # disagree. The peak is the running maximum that fall is measured against,
        # which the shading runs up to
        equity = result.equity
        dd     = 0.0 - _falls(result)
        peak   = np.maximum.accumulate(equity)
        if len(rows) != num_bars:
            equity, dd, peak = (_placed(values, rows, num_bars)
                                for values in (equity, dd, peak))

    names = _assign(marks, ohlc)
    order = _order(names, config, volume is not None, result is not None, dd_mode)
    panels_out, hidden = _build_panels(
        order, config, marks, names, num_bars, ohlc, volume, equity, dd,
        future,
    )

    _check_log(panels_out, marks, names, ohlc, {
        "equity":   equity,
        "drawdown": dd,
        "volume":   volume,
    }, future)

    shown_digits = {p["name"]: p["digits"] for p in panels_out}

    def places(name, fallback=4):
        return shown_digits.get(name, fallback) + 2

    series = []
    for index, (mark, on) in enumerate(zip(marks, names)):
        if on in hidden:
            continue

        label = _label(mark, index)
        entry = {"kind": mark.kind, "panel": on}
        if mark.name:
            entry["name"] = mark.name

        if mark.kind in ("line", "histogram"):
            digits = _digits(_extent(mark.values))
            entry.update(
                _track(mark.values, num_bars, label, digits + 2, future)
            )
            entry["color"] = _encode_color(
                mark.color, num_bars, f"{label} color", future
            )
            entry["digits"] = digits
            if mark.kind == "line":
                entry["width"] = mark.width
                entry["style"] = mark.style
                entry["step"]  = mark.step
                if mark.fill:
                    entry["fill"] = mark.fill
            else:
                entry["base"] = mark.base
        elif mark.kind == "band":
            # each edge to the places a line of its values takes, so a band between
            # two named lines ships their very numbers, which the legend finds them by
            entry.update(
                _track(mark.upper, num_bars, f"{label} upper",
                       _digits(_extent(mark.upper)) + 2, future)
            )
            if mark.lower is None:
                entry["level"] = mark.level
            else:
                lower = _track(
                    mark.lower, num_bars, f"{label} lower",
                    _digits(_extent(mark.lower)) + 2, future
                )
                # both edges are one shape, so they share one start
                start = max(entry["i0"], lower["i0"])
                entry["v"]  = entry["v"][start - entry["i0"]:]
                entry["v2"] = lower["v"][start - lower["i0"]:]
                entry["i0"] = start

            stops = mark.fill or ["rgba(139,151,165,0.14)"]
            if len(stops) > 3:
                # one colour per bar, which the renderer draws by splitting the
                # shape where the colour changes rather than by one gradient
                entry["fill"] = _encode_color(
                    stops, num_bars, f"{label} fill", future
                )
            else:
                # only="below" runs downward on screen, and flipping it here is
                # what lets the renderer always paint stop 0 on the reference edge
                if mark.only == "below" and len(stops) > 1:
                    stops = list(reversed(stops))
                entry["fill"] = stops

            entry["only"]  = mark.only
            entry["color"] = _encode_color(
                mark.color, num_bars, f"{label} color", future
            )
        elif mark.kind == "level":
            entry["value"] = mark.value
            entry["width"] = mark.width
            entry["style"] = mark.style
            if mark.color:
                entry["color"] = mark.color
        elif mark.kind == "marker":
            if not 0 <= mark.bar < num_bars + future:
                where = (f"{num_bars} bars" if not future else
                         f"{num_bars} bars and {future} projected past them")
                raise ProbatumValueError(f"Marker bar: {mark.bar}, outside a chart of "
                                         f"{where}; give a bar on it")

            # both kinds ship a list, so the renderer holds one primitive and one
            # loop rather than one attached primitive per glyph
            entry["kind"]   = "marker"
            entry["bars"]   = [mark.bar]
            entry["shape"]  = mark.shape
            entry["offset"] = mark.offset
            if mark.value is not None:
                entry["values"] = [mark.value]
            if mark.text:
                entry["text"] = mark.text
            if mark.color:
                entry["color"] = mark.color
        elif mark.kind == "markers":
            # one flatnonzero, used for both the bars and their anchors, so the
            # two lists cannot come out of step
            start = _start_for(mark.mask.size, num_bars, label, future)
            where = np.flatnonzero(mark.mask)
            entry["kind"]   = "marker"
            entry["bars"]   = (where + start).tolist()
            entry["shape"]  = mark.shape
            entry["offset"] = mark.offset
            if mark.value is not None:
                if np.ndim(mark.value) == 0:
                    entry["values"] = [float(mark.value)] * where.size
                else:
                    # a null anchor falls back to the bar's own extreme, which is
                    # what an absent one does, so a glyph on a bar its anchor had
                    # not warmed up for lands on the candle rather than vanishing
                    picked = np.asarray(mark.value)[where]
                    entry["values"] = [
                        None if not np.isfinite(v) else round(float(v), 8)
                        for v in picked.tolist()
                    ]

            if mark.text:
                entry["text"] = mark.text
            if mark.color:
                entry["color"] = mark.color
        elif mark.kind == "background":
            spans, fills, keys = _spans(
                mark.values, mark.fill, num_bars, label, future
            )
            entry["spans"] = spans
            entry["fills"] = fills
            if keys is not None and mark.name is not None:
                entry["keys"] = keys

        entry = {k: v for k, v in entry.items() if v is not None}
        series.append(entry)

    # the fall from the peak, shaded on the equity panel between the curve and its
    # own running maximum. A band rather than a second pane, so the two readings
    # share one axis and one set of units and the trough is drawn directly beneath
    # the high that caused it. The band never inverts, because a running maximum is
    # never below the series it is taken from. Not drawn when a drawdown panel
    # already exists: a mark carrying panel="drawdown" creates one the same way any
    # other name does, and the fall would then be drawn twice, once in its own pane
    # and once shaded onto the equity
    has_dd_panel = any(p["name"] == "drawdown" for p in panels_out)
    if (dd_mode == "under" and not has_dd_panel
            and equity is not None and peak is not None
            and "equity" not in hidden and "drawdown" not in hidden):
        upper = _track(peak, num_bars, "drawdown peak", places("equity"))
        lower = _track(equity, num_bars, "equity", places("equity"))
        at    = max(upper["i0"], lower["i0"])
        series.append({
            "kind": "band", "panel": "equity", "name": "drawdown",
            "i0": at,
            "v": upper["v"][at - upper["i0"]:],
            "v2": lower["v"][at - lower["i0"]:],
            # translucent grey rather than a theme key, for the same reason the default
            # band fill is one: a band's colours are baked into the document and a
            # theme toggle repaints series rather than primitives. Denser at the
            # curve, where the fall is measured to, fading toward the peak it fell
            # from
            "fill": ["rgba(138,147,158,0.26)", "rgba(138,147,158,0.03)"],
        })

    mode = _theme_mode(theme) if theme is not None else _DEFAULTS["theme"]
    # the call's palette replaces the session's rather than merging into it, the
    # same way its theme does, so one call site reads as one appearance instead of
    # as the difference between two
    tint = _palette(palette) if palette is not None else _DEFAULTS["palette"]

    spec = {
        "schema": 7,
        "n": num_bars,
        # numpy's own tolist(), never a comprehension, for the reason _candles gives
        "t": times.tolist(),
        # the interval is the Data's own, as the router names it
        "interval": data.interval,
        "ohlc": _candles(ohlc, shown),
        "candles": {"panel": "price",
                    "style": _DEFAULTS["price"] if price is None else _price(price)},
        # the instrument the candles are, which the price pane's legend names
        "symbol": name,
        "panels": panels_out,
        "series": series,
        "trades": _trades(result, name, rows) if result is not None and trades else [],
        "theme": _theme(mode, tint),
        "height": _height(height) if height is not None else _DEFAULTS["height"],
    }

    if title is None and result is not None and len(data.names) > 1:
        # the equity and the headline are the account's, which nothing else says
        title = f"{name}, one of the account's {len(data.names)} instruments"
    if title is not None:
        spec["title"] = str(title)
    if notes is not None:
        table = _notes(notes)
        if table is not None:
            spec["notes"] = table

    if volume is not None and "volume" not in hidden:
        spec["vol"] = _track(volume, num_bars, "volume", places("volume"))

    if candle_color is not None:
        # the one colour that arrives as raw user input rather than through a
        # mark, so it needs the same coercion every other colour already gets: a
        # NaN entry would otherwise become a palette entry and fail at save()
        spec["candles"]["color"] = _encode_color(
            _color(candle_color, "candle_color"), num_bars, "candle_color", future
        )

    if equity is not None and "equity" not in hidden:
        spec["equity"] = _track(equity, num_bars, "equity", places("equity"))
    # the track ships whether the fall has a pane or a band, because the legend
    # reads the number off it either way; only drawdown=False drops it
    if dd is not None and dd_mode is not False and "drawdown" not in hidden:
        spec["drawdown"] = _track(dd, num_bars, "drawdown", places("drawdown"))

    if result is not None and result.stats is not None:
        # a stat is legitimately infinite (profit_factor with no losing trade), and
        # JSON carries no infinity, so one becomes null rather than a crash on the
        # way out
        spec["stats"] = {
            key: (float(v) if math.isfinite(float(v)) else None)
            for key, v in result.stats.items()
        }
        headline = _headline(result, stats, name, len(data.names) > 1)
        if headline:
            spec["headline"] = headline
    elif stats:
        raise ProbatumValueError("stats: the run has no statistics, its clock too short "
                                 "to annualise; leave stats out")

    window = _focus(focus, num_bars, rows, future)
    if window is not None:
        spec["focus"] = window

    # the coerced title, not the argument: html.escape wants a string and a
    # non-string one used to reach it from inside save(), long after this returned
    return Chart(spec, spec.get("title"))
