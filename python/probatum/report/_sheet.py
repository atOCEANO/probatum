"""A run's tear sheet: its headline numbers with their intervals, each naming its
method; its figures; its costs and its round trips; the three baselines beside it with
the Sharpe difference against buying and holding; its assumptions, the optimistic ones
named at the top; and its provenance (ADR 0041; Specification, What a run records)."""

import json
import math

import numpy as np

from .. import stats
from .._errors import ProbatumAttributeError, ProbatumValueError
from .._result import _recorded, _returns
from . import _figures, _page

# the record's statistics in the order the headline reads them
HEADLINE = ("total_return_pct", "cagr_pct", "sharpe", "sortino", "calmar",
            "max_drawdown_pct", "volatility_pct", "value_at_risk",
            "conditional_value_at_risk", "exposure_pct", "num_trades", "win_rate",
            "profit_factor", "avg_trade_pct", "num_fills", "funding_paid")
# the words and the units of those with no interval, which stats has no text for
WORDS = {
    "exposure_pct": ("in the market", "%"),
    "num_trades":   ("round trips", ""),
    "num_fills":    ("fills", ""),
    "funding_paid": ("funding paid", ""),
}


def page(result):
    """The tear sheet of ``result`` as one HTML page."""
    figures = _page.Figures()
    lines   = result.summary().splitlines()
    named   = _optimistic(result)

    body = f"<h1>{_page.escape(lines[0])}</h1>\n" + _page.paragraph(lines[1])
    if named:
        body += _page.paragraph(f"{len(named)} optimistic assumption"
                                f"{'s' * (len(named) != 1)}: {'; '.join(named)}", "warn")

    body += _page.section("Headline", _headline(result))
    body += _page.section("Equity", figures.held(_figures.equity, result)
                          + figures.held(_figures.drawdown, result)
                          + _page.paragraph(str(stats.time_under_water(result))))
    body += _page.section("Returns", figures.held(_figures.returns, result)
                          + figures.held(_figures.monthly, result))
    body += _page.section("Over time", figures.held(_figures.rolling_sharpe, result)
                          + figures.held(_figures.rolling_volatility, result)
                          + figures.held(_figures.exposure, result))
    body += _page.section("Instruments", figures.held(_figures.weights, result)
                          + figures.held(_figures.pnl, result))
    body += _page.section("Costs", figures.held(_figures.costs, result)
                          + _page.paragraph(str(stats.decompose(result))))
    body += _page.section("Round trips", _trips(result, figures))
    body += _page.section("Baselines", _baselines(result))
    body += _page.section("Assumptions", _assumptions(result))
    body += _page.section("Provenance", _page.folded(
        "what the run was given and where it ran",
        _page.preformatted(json.dumps(result.provenance, indent=2, default=str))))

    return _page.document(lines[0], body)


def _optimistic(result):
    table = result.assumptions
    return [f"{row['name']} {row['value']}" + (f" at {row['venue']}" if "venue" in row
                                                else "")
            for row in table[table.grade == "optimistic"]]


def _headline(result):
    """The record's statistics, each with its interval and how it was found, or why it
    has none."""
    held = result.stats
    if held is None:
        return _page.paragraph("none: the clock is too short to annualise")

    rows = []
    for name in HEADLINE:
        if name in ("value_at_risk", "conditional_value_at_risk"):
            value = getattr(stats, name)(result)
        else:
            value = held[name]

        if name in WORDS:
            words, unit = WORDS[name]
            rows.append((words, _number(value, unit), "", ""))
            continue

        words, unit, loss = stats._shown(name)
        interval = held.interval(name) if name in held else _bootstrapped(result, name)
        rows.append((words, _number(-value if loss else value, unit),
                     *_bounds(result, name, interval, unit, loss)))

    return _page.table(("statistic", "value", "95% interval", "how"), rows, (1, 2))


def _bootstrapped(result, name):
    """The interval of a statistic the record does not hold, or None where its sample
    supports none."""
    try:
        return stats.bootstrap(result, name)
    except ProbatumValueError:
        return None


def _bounds(result, name, interval, unit, loss):
    """An interval's bounds and how it was found, or why it has none."""
    if interval is None:
        return "none", _why(result, name)

    low, high = sorted(((-1.0 if loss else 1.0) * bound + 0.0
                        for bound in (interval.low, interval.high)))
    how = interval.method
    if interval.lag is not None:
        how += f", Newey-West lag {interval.lag}"
        how += " floored at lag zero" if interval.floored else ""
    if interval.block is not None:
        how += f", block {interval.block:.1f}"
    if interval.kind == "range":
        how = f"middle of resampled histories, {how}"

    return f"{_number(low, unit)} to {_number(high, unit)}", how


def _why(result, name):
    """Why the sample supports no interval of ``name``."""
    try:
        if name == "sharpe":
            stats.sharpe_interval(result)
        else:
            stats.bootstrap(result, name)
    except ProbatumValueError as error:
        return str(error)

    return "none at this level"


def _number(value, unit):
    if value is None:
        return "none"
    if isinstance(value, int):
        return f"{value:,}"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value + 0.0:,.2f}{unit}"


def _trips(result, figures):
    """The closed round trips' statistics, by side, and how their returns spread."""
    found = stats.trade_stats(result)
    body  = _page.paragraph(str(found))
    if found.trades:
        split = stats.long_short_split(result)
        body += _page.table(
            ("side", "round trips", "win rate", "payoff", "expectancy", "rows held"),
            [(side, f"{held.trades:,}",
              f"{held.wins / held.trades:.0%}" if held.trades else "",
              _number(held.payoff, ""), _number(held.expectancy, ""),
              f"{held.avg_bars_held:,.1f}") for side, held in split.items()],
            (1, 2, 3, 4, 5))
    return body + figures.held(_figures.trips, result)


def _baselines(result):
    """The run beside its three baselines, each one's return, Sharpe and maximum
    drawdown, with the Sharpe difference against buying and holding by both tests."""
    try:
        held = result.baselines
    except (ProbatumAttributeError, ProbatumValueError) as error:
        return _page.paragraph(f"none: {error}")

    rate = result.provenance["periods_per_year"]
    free = result.provenance["risk_free"]
    rows = []
    for name, equity in (("strategy", result.equity), ("cash", held["cash"]),
                         ("buy and hold", held["buy_and_hold"]),
                         ("without costs", held["frictionless"])):
        equity = np.asarray(equity, dtype=np.float64)
        if not np.all(np.isfinite(equity)):
            rows.append((name, "none", "", ""))
            continue

        # read as the record reads a run's, nothing once the equity is gone, and over
        # the returns' own deviation, which a series that never varies has as 0
        moved  = _returns(equity)[1:]
        sharpe = math.nan if rate is None else stats._recorded_sharpe(moved, rate, free)
        rows.append((name, _number((equity[-1] / equity[0] - 1.0) * 100.0, "%"),
                     _number(sharpe, ""), _number(-stats._fallen(equity).max() + 0.0, "%")))

    body = _page.table(("run", "return", "sharpe", "max drawdown"), rows, (1, 2, 3))
    try:
        apart = stats.sharpe_difference(result)
    except ProbatumValueError as error:
        return body + _page.paragraph(f"sharpe difference against buying and holding: "
                                      f"none, {error}")

    return (body + _page.paragraph(f"against buying and holding: {apart.delta}")
            + _page.paragraph(str(apart.bootstrap)))


def _assumptions(result):
    """What the run assumed, every row folded beneath its count."""
    table = result.assumptions
    rows  = [(row["name"], row["value"], row["source"], row["grade"],
              row["direction"] or "", "" if row["cost"] is None
              or (isinstance(row["cost"], float) and math.isnan(row["cost"]))
              else f"{row['cost']:,.2f}") for row in table]
    held  = _page.table(("name", "value", "source", "grade", "direction", "cost"), rows,
                        (5,))
    return _page.folded(f"{len(rows)} assumptions, {len(_optimistic(result))} optimistic",
                        held, open_=_recorded(result.provenance) == "full")
