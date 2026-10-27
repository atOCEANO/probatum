"""The reports' figures, a function a panel, each drawing what a Result or a Study holds
and what probatum.stats reads of it, so a redesign cannot move a number (ADR 0041).
Each is a matplotlib Figure made with no pyplot, so no backend or global state is
touched, at the look _look sets, its plotted data readable from its artists; one that
does not apply to what it is given, as weights to a run of one instrument, is None."""

import io
import math
import re

import matplotlib
import numpy as np
from matplotlib import colors, dates, ticker
from matplotlib.figure import Figure

from .. import stats
from .._errors import ProbatumAttributeError, ProbatumValueError
from .._result import COSTS
from . import _look

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
          "Dec")
# the steps a log axis of money is marked at, within each power of ten
STEPS = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)


def drawn(make, *given):
    """``make(*given)`` drawn at the reports' look, and its SVG, or None for both where
    it does not apply."""
    with matplotlib.rc_context(_look.RC):
        figure = make(*given)
        return figure, None if figure is None else _svg(figure)


def _svg(figure):
    """A figure's SVG to hold inline in a page: without its declarations and metadata,
    which a page does not need and which carry nothing a reader sees."""
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    text = buffer.getvalue()
    text = re.sub(r"<\?xml[^>]*>\s*|<!DOCTYPE[^>]*>\s*|<!--.*?-->\s*", "", text, flags=re.S)
    return re.sub(r"\s*<metadata>.*?</metadata>", "", text, flags=re.S)


def _canvas(height, width=_look.WIDTH, left=64.0, right=24.0, top=34.0, bottom=30.0):
    """A figure of one axes, its size and margins in pixels."""
    figure = Figure(figsize=(width / _look.DPI, height / _look.DPI), dpi=_look.DPI)
    figure.subplots_adjust(left=left / width, right=1.0 - right / width,
                           top=1.0 - top / height, bottom=bottom / height)
    return figure, figure.add_subplot()


def _when(result):
    return result.time.view("M8[ns]")


def _dated(ax, when):
    locator = dates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
    ax.set_xlim(when[0], when[-1])


def _legend(ax, count):
    # one series needs none, its title naming it
    if count > 1:
        ax.legend(ncols=min(count, 4), loc="lower left", bbox_to_anchor=(0.0, 1.0),
                  borderaxespad=0.0, handlelength=1.6, columnspacing=1.2)


def _percent(value, _=None):
    return f"{value:.0f}%" if value == round(value) else f"{value:.1f}%"


def _compact(value, _=None):
    """An amount on an axis: 950, 12K, 1.5M."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            shown = f"{value / size:.1f}".rstrip("0").rstrip(".")
            return f"{shown}{unit}"
    return f"{value:.0f}" if abs(value) >= 1.0 else f"{value:.3g}"


def _logged(ax, series):
    """A log axis of money, marked at the steps between its least and its largest
    value, or where fewer than two fall between them at a linear axis' marks."""
    finite = np.concatenate([values[np.isfinite(values)] for values in series])
    low, high = float(finite.min()), float(finite.max())
    powers = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
    ax.set_yscale("log")
    marks = [step * 10.0 ** power for power in powers for step in STEPS
             if low <= step * 10.0 ** power <= high]
    if len(marks) < 2:
        marks = [mark for mark in ticker.MaxNLocator(4).tick_values(low, high)
                 if low <= mark <= high and mark > 0.0]

    ax.yaxis.set_major_locator(ticker.FixedLocator(marks))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_compact))


def _baselines(result):
    """The run's baselines, None where it keeps none to read: a run below the full
    level, or one loaded or unpickled before they were computed."""
    try:
        return result.baselines
    except (ProbatumAttributeError, ProbatumValueError):
        return None


def _histogram(ax, values, color, label):
    """``values`` counted into bins of one width, drawn as bars."""
    bins = int(min(60, max(10, round(math.sqrt(values.size)))))
    counts, edges = np.histogram(values, bins=bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=color,
           linewidth=0.0, label=label)
    ax.axvline(0.0, color=_look.AXIS, linewidth=_look.THIN)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_percent))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))


def window(result):
    """The rolling panels' window: a quarter of a year of the run's bars, at most half
    its returns, None where its clock is too short to annualise or that leaves fewer
    than 2."""
    rate = result.provenance["periods_per_year"]
    if rate is None:
        return None

    size = min(round(rate / 4.0), (len(result.time) - 1) // 2)
    return size if size >= 2 else None


def equity(result):
    """The run's equity beside its baselines, buying and holding, the same policy
    without the costs of trading, and holding cash, on a log scale while every value is
    above zero."""
    figure, ax = _canvas(_look.HEIGHT)
    when   = _when(result)
    held   = _baselines(result)
    series = []
    if held is not None:
        series.append(("cash", held["cash"], _look.MUTED, _look.THIN, None))
        series.append(("buy and hold", held["buy_and_hold"], _look.HOLD, _look.LINE, None))
        if np.all(np.isfinite(held["frictionless"])):
            series.append(("without costs", held["frictionless"], _look.STRATEGY,
                           _look.THIN, (3.0, 3.0)))
    series.append(("strategy", result.equity, _look.STRATEGY, _look.LINE, None))

    for label, values, color, width, dashes in series:
        line, = ax.plot(when, values, color=color, linewidth=width, label=label)
        if dashes:
            line.set_dashes(dashes)

    values = [np.asarray(values, dtype=np.float64) for _, values, *_ in series]
    if all(np.all(each[np.isfinite(each)] > 0.0) for each in values):
        _logged(ax, values)
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_compact))

    ax.set_title("Equity", pad=26.0 if len(series) > 1 else 10.0)
    _legend(ax, len(series))
    _dated(ax, when)
    return figure


def drawdown(result):
    """The run's fall below its running peak, in percent, beside buying and holding's."""
    figure, ax = _canvas(_look.SHORT)
    when  = _when(result)
    held  = _baselines(result)
    count = 1
    if held is not None and np.all(np.isfinite(held["buy_and_hold"])):
        hold = -stats._fallen(np.asarray(held["buy_and_hold"], dtype=np.float64))
        ax.plot(when, hold, color=_look.HOLD, linewidth=_look.THIN, label="buy and hold")
        count += 1

    falls = -stats.drawdown(result)
    ax.fill_between(when, falls, 0.0, color=_look.STRATEGY, alpha=_look.WASH,
                    linewidth=0.0)
    ax.plot(when, falls, color=_look.STRATEGY, linewidth=_look.LINE, label="strategy")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_percent))
    ax.set_title("Drawdown", pad=26.0 if count > 1 else 10.0)
    _legend(ax, count)
    _dated(ax, when)
    return figure


def returns(result):
    """How the run's returns a row spread, in percent."""
    if len(result.time) < 2:
        return None

    figure, ax = _canvas(_look.SHORT)
    _histogram(ax, np.asarray(result.returns[1:]) * 100.0, _look.STRATEGY, "returns")
    ax.set_title("Returns a row")
    return figure


def monthly(result):
    """The run's return in each calendar month in UTC, a row a year, in percent."""
    table  = stats.period_returns(result, by="month")
    names  = table["period"].tolist()
    years  = sorted({int(name[:4]) for name in names})
    values = np.full((len(years), 12), np.nan)
    for name, value in zip(names, np.asarray(table["return_pct"]).tolist()):
        values[years.index(int(name[:4])), int(name[5:]) - 1] = value

    figure, ax = _canvas(60.0 + 26.0 * len(years), left=52.0, top=40.0, bottom=10.0)
    reach = float(np.nanmax(np.abs(values))) or 1.0
    ramp  = colors.LinearSegmentedColormap.from_list(
        "returns", [_look.LOSS, _look.MIDDLE, _look.GAIN])
    ax.imshow(np.ma.masked_invalid(values), cmap=ramp, vmin=-reach, vmax=reach,
              aspect="auto")

    for (row, column), value in np.ndenumerate(values):
        if not math.isnan(value):
            # ink on a light cell, the surface on a dark one
            shade = colors.rgb_to_hsv(ramp((value + reach) / (2.0 * reach))[:3])[2]
            ink   = _look.INK if abs(value) < 0.6 * reach or shade > 0.9 else _look.SURFACE
            ax.text(column, row, f"{value:+.1f}", ha="center", va="center", color=ink,
                    fontsize=_look.SIZE - 1.0)

    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(len(years)), [str(year) for year in years])
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Monthly returns, %", pad=24.0)
    return figure


def rolling_sharpe(result):
    """The record's Sharpe of the window of returns ending at each row, a window with
    no risk, whose Sharpe is infinite, left as a gap."""
    size = window(result)
    if size is None:
        return None

    figure, ax = _canvas(_look.SHORT)
    when   = _when(result)
    values = stats.rolling_sharpe(result, size)
    ax.axhline(0.0, color=_look.AXIS, linewidth=_look.THIN)
    ax.plot(when, np.where(np.isinf(values), np.nan, values), color=_look.STRATEGY,
            linewidth=_look.LINE, label="strategy")
    ax.set_title(f"Rolling Sharpe, {size:,} bars")
    _dated(ax, when)
    return figure


def rolling_volatility(result):
    """The volatility of the window of returns ending at each row, a year of it, in
    percent."""
    size = window(result)
    if size is None:
        return None

    figure, ax = _canvas(_look.SHORT)
    when = _when(result)
    ax.plot(when, stats.rolling_volatility(result, size), color=_look.STRATEGY,
            linewidth=_look.LINE, label="strategy")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_percent))
    ax.set_title(f"Rolling volatility, {size:,} bars")
    _dated(ax, when)
    return figure


def exposure(result):
    """The run's gross and net exposure over its equity at each row's close."""
    figure, ax = _canvas(_look.SHORT)
    when   = _when(result)
    ledger = result.ledger
    equity = np.asarray(ledger["equity"], dtype=np.float64)
    net    = np.divide(np.asarray(ledger["net_exposure"], dtype=np.float64), equity,
                       out=np.zeros_like(equity), where=equity > 0.0)
    ax.axhline(0.0, color=_look.AXIS, linewidth=_look.THIN)
    ax.plot(when, ledger["leverage"], color=_look.STRATEGY, linewidth=_look.LINE,
            label="gross")
    ax.plot(when, net, color=_look.HOLD, linewidth=_look.THIN, label="net")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:.1f}x"))
    ax.set_title("Exposure over equity", pad=26.0)
    _legend(ax, 2)
    _dated(ax, when)
    return figure


def _instruments(result):
    """A run's instruments, each with its label and its record, None for a run of one
    or one that keeps no holdings."""
    try:
        held = result._instruments()
    except ProbatumAttributeError:
        return None
    if len(held) < 2:
        return None

    return [(" ".join(key) if isinstance(key, tuple) else key, result[key])
            for key in held]


def _each(result, field, title, formatter):
    """``field`` of each instrument over the rows, each in the colour of its place in
    the run, those past the palette's last but one summed as others."""
    held = _instruments(result)
    if held is None:
        return None

    figure, ax = _canvas(_look.HEIGHT)
    when  = _when(result)
    shown = len(_look.SERIES) if len(held) <= len(_look.SERIES) else len(_look.SERIES) - 1
    for (name, record), color in zip(held[:shown], _look.SERIES):
        ax.plot(when, getattr(record, field), color=color, linewidth=_look.LINE,
                label=name)
    if len(held) > shown:
        others = np.sum([getattr(record, field) for _, record in held[shown:]], axis=0)
        ax.plot(when, others, color=_look.SERIES[-1], linewidth=_look.LINE,
                label=f"{len(held) - shown} others")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
    ax.set_title(title, pad=26.0 if len(held) <= 4 else 44.0)
    _legend(ax, min(len(held), len(_look.SERIES)))
    _dated(ax, when)
    return figure


def weights(result):
    """Each instrument's weight over its account's equity at each row's close."""
    return _each(result, "weight", "Weights", lambda value, _: f"{value:.2f}")


def pnl(result):
    """What each instrument has made since the first row."""
    return _each(result, "pnl", "What each instrument made", _compact)


def costs(result):
    """What each cost took from the run, in the asset it settles in."""
    figure, ax = _canvas(56.0 + 24.0 * len(COSTS), left=96.0, top=34.0, bottom=28.0)
    found  = stats.decompose(result)
    values = [getattr(found, name) for name in COSTS]
    ax.barh(range(len(COSTS)), values, height=0.5, color=_look.STRATEGY, linewidth=0.0,
            label="paid")
    ax.set_yticks(range(len(COSTS)), COSTS)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_compact))
    ax.set_title("Costs paid")
    return figure


def trips(result):
    """How the closed round trips' returns spread, each its net pnl over its size times
    its entry, in percent."""
    trades = result.trades
    closed = ~np.asarray(trades["open"])
    if not closed.any():
        return None

    figure, ax = _canvas(_look.SHORT)
    _histogram(ax, np.asarray(trades["ret"], dtype=np.float64)[closed] * 100.0,
               _look.STRATEGY, "round trips")
    ax.set_title("Round trips' returns")
    return figure


def trials(study):
    """A search's complete trials' scores in the order they were asked, the winner
    marked."""
    table    = study.trials
    score    = np.asarray(table["score"], dtype=np.float64)
    complete = (np.asarray(table["state"]) == "complete") & np.isfinite(score)
    if not complete.any():
        return None

    figure, ax = _canvas(_look.HEIGHT)
    numbers = np.asarray(table["number"])
    ax.scatter(numbers[complete], score[complete], s=16.0, color=_look.STRATEGY,
               linewidths=0.0, label="trials")
    best = numbers == study._winner.number
    ax.scatter(numbers[best], score[best], s=64.0, color=_look.HOLD, linewidths=0.0,
               label="winner")
    ax.set_title(f"Each trial's {study.provenance['objective']}", pad=26.0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(8, integer=True))
    _legend(ax, 2)
    return figure


def importance(study):
    """Each parameter's share of the variation in the trials' scores its values
    explain, beside the share chance gives."""
    found = stats.importance(study)
    names = list(found.names)
    figure, ax = _canvas(68.0 + 30.0 * len(names), left=120.0, top=40.0, bottom=28.0)
    ax.barh(range(len(names)), found.importance, height=0.5, color=_look.STRATEGY,
            linewidth=0.0, label="importance")
    ax.scatter(found.chance, range(len(names)), marker="|", s=200.0, color=_look.INK,
               linewidths=_look.LINE, label="chance")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Parameter importance", pad=26.0)
    _legend(ax, 2)
    return figure


def paths(study):
    """A validation's paths' scores, a bar a path."""
    table = study.paths
    score = np.asarray(table["score"], dtype=np.float64)
    if not np.isfinite(score).any():
        return None

    figure, ax = _canvas(_look.SHORT)
    ax.bar(np.asarray(table["number"]), np.where(np.isfinite(score), score, 0.0),
           width=0.6, color=_look.STRATEGY, linewidth=0.0, label="paths")
    ax.axhline(0.0, color=_look.AXIS, linewidth=_look.THIN)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(8, integer=True))
    ax.set_title(f"Each path's {study.provenance['objective']}")
    return figure


def windows(study):
    """Each window of a validation: what its fit scored in sample against what the
    window then scored, the line where they are equal; a model's fits score nothing in
    sample, so its validation has none."""
    fits = study.fits
    if "fitted" not in fits.columns:
        return None

    table   = study.windows
    earned  = np.asarray(table["score"], dtype=np.float64)
    fitted  = dict(zip(np.asarray(fits["number"]).tolist(),
                       np.asarray(fits["fitted"], dtype=np.float64).tolist()))
    scored  = np.array([fitted.get(fit, math.nan)
                        for fit in np.asarray(table["fit"]).tolist()])
    present = np.isfinite(scored) & np.isfinite(earned)
    if not present.any():
        return None

    figure, ax = _canvas(_look.HEIGHT)
    low  = float(min(scored[present].min(), earned[present].min()))
    high = float(max(scored[present].max(), earned[present].max()))
    ax.plot([low, high], [low, high], color=_look.AXIS, linewidth=_look.THIN)
    ax.scatter(scored[present], earned[present], s=24.0, color=_look.STRATEGY,
               linewidths=0.0, label="windows")
    ax.set_xlabel("in sample")
    ax.set_ylabel("out of sample")
    ax.set_title("In sample against out of sample")
    return figure


SHEET = (equity, drawdown, returns, monthly, rolling_sharpe, rolling_volatility, exposure,
         weights, pnl, costs, trips)
