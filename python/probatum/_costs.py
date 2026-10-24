"""``capacity``, ``cost_curve`` and ``breakeven_bps``: one policy run again where only
its balance or its costs differ, each run a backtest of it, read for what its size and
its costs did to it (ADR 0045; Specification, Capacity and costs)."""

import collections.abc
import math
import warnings

import numpy as np

from ._backtest import _backtested, _real
from ._data import Data
from ._errors import ProbatumTypeError, ProbatumValueError, ProbatumWarning, VenueError
from ._result import _money
from ._venue import _balance, _python, _run, _venue_of
from .stats import PACKAGE, _Frozen

# what a run pays for trading, the costs a sweep multiplies
CHARGED = ("fees", "spread", "slippage", "impact")
SCALES  = (0.0, 0.5, 1.0, 2.0, 4.0)
# the most halvings of a break-even's bracket
HALVINGS = 60

Measured = collections.namedtuple(
    "Measured", "total_return_pct sharpe num_trades costs cost_bps turnover held")


class Capacity(_Frozen):
    """How a policy held up as its account grew, a run a balance in the order given:
    ``balances``; ``total_return_pct`` and ``sharpe``; ``impact``, what impact took, in
    the asset the account settles in; ``held``, the share of the orders placed that the
    liquidity budget held; ``cost_bps``, the round-trip cost its fills paid, in basis
    points; and ``turnover``, a year of its traded notional over its mean equity
    (Specification, Capacity)."""

    __slots__ = ("balances", "total_return_pct", "sharpe", "impact", "held", "cost_bps",
                 "turnover")


    def __init__(self, balances, total_return_pct, sharpe, impact, held, cost_bps,
                 turnover):
        self._set(
            balances         = tuple(_Wallets(balance) if isinstance(balance, dict)
                                     else balance for balance in balances),
            total_return_pct = _floats(total_return_pct),
            sharpe           = _floats(sharpe),
            impact           = _floats(impact),
            held             = _floats(held),
            cost_bps         = _floats(cost_bps),
            turnover         = _floats(turnover),
        )


    def __repr__(self):
        return f"Capacity({_counted(self.balances, 'balance')})"


    def __str__(self):
        header = ("balance", "return", "sharpe", "impact paid", "held by the budget",
                  "round trip", "turnover")
        rows   = [(_opened(balance), _cell(gain, "{:+.1f}%"), _cell(sharpe, "{:.2f}"),
                   _money(-impact + 0.0), _cell(held, "{:.0%}"),
                   _cell(cost, "{:.1f} bps"), _cell(turnover, "{:,.1f}"))
                  for balance, gain, sharpe, impact, held, cost, turnover in zip(
                      self.balances, self.total_return_pct, self.sharpe, self.impact,
                      self.held, self.cost_bps, self.turnover)]

        return _table(header, rows)


class CostCurve(_Frozen):
    """What a policy kept as its costs grew, a run a scale in the order given:
    ``scales``, the multiples of every cost its venue charges for trading;
    ``cost_bps``, the round-trip cost each run's fills paid, in basis points;
    ``total_return_pct`` and ``sharpe``; ``num_trades``, its round trips; ``fees``,
    ``spread``, ``slippage`` and ``impact``, what each took, in the asset the account
    settles in; and ``turnover``, a year of its traded notional over its mean equity
    (Specification, Cost sweeps)."""

    __slots__ = ("scales", "cost_bps", "total_return_pct", "sharpe", "num_trades",
                 "fees", "spread", "slippage", "impact", "turnover")


    def __init__(self, scales, cost_bps, total_return_pct, sharpe, num_trades, fees,
                 spread, slippage, impact, turnover):
        self._set(
            scales           = _floats(scales),
            cost_bps         = _floats(cost_bps),
            total_return_pct = _floats(total_return_pct),
            sharpe           = _floats(sharpe),
            num_trades       = tuple(int(count) for count in num_trades),
            fees             = _floats(fees),
            spread           = _floats(spread),
            slippage         = _floats(slippage),
            impact           = _floats(impact),
            turnover         = _floats(turnover),
        )


    def __repr__(self):
        return f"CostCurve({_counted(self.scales, 'scale')})"


    def __str__(self):
        header = ("scale", "round trip", "return", "sharpe", "trades", "fees", "spread",
                  "slippage", "impact", "turnover")
        rows   = [(f"{scale:g}", _cell(cost, "{:.1f} bps"), _cell(gain, "{:+.1f}%"),
                   _cell(sharpe, "{:.2f}"), f"{trades:,}",
                   *(_money(-paid + 0.0) for paid in costs), _cell(turnover, "{:,.1f}"))
                  for scale, cost, gain, sharpe, trades, turnover, *costs in zip(
                      self.scales, self.cost_bps, self.total_return_pct, self.sharpe,
                      self.num_trades, self.turnover, self.fees, self.spread,
                      self.slippage, self.impact)]

        return _table(header, rows)


def capacity(policy, data, venue, balances, sizing="weight", stop=None,
             take_profit=None, tolerance=0.0, risk_free=0.0):
    """How ``policy`` holds up as its account grows: a backtest of it on ``data`` at
    ``venue``, a Venue or a preset's name, with the account opening at each of
    ``balances`` in turn, each a number or ``{"spot": s, "margin": m}`` as a Venue's
    balance is, and the rest of backtest's settings, as a `Capacity` of each run's
    return, Sharpe, impact paid, share of orders the liquidity budget held, round-trip
    cost and turnover (ADR 0045; Specification, Capacity). A weight is a share of
    equity, so its positions grow with the balance; a quantity does not, and
    ``sizing="quantity"`` is refused, as is, before any run, a balance the account
    cannot open with."""
    if isinstance(sizing, str) and sizing == "quantity":
        raise ProbatumValueError("sizing: \"quantity\", whose positions do not grow "
                                 "with the balance; give weights, which capacity "
                                 "scales")

    balances = _listed("balances", balances, _opening, "give the opening balances to "
                       "run at, such as [10_000.0, 100_000.0, 1_000_000.0]")

    data = data if isinstance(data, Data) else Data(data)
    _shaped(balances, _venue_of(venue), data)

    runs = [_measured(_backtested(policy, data, venue, sizing, stop, take_profit,
                                  tolerance, "full", risk_free, balance=balance), data)
            for balance in balances]

    return Capacity(balances, [run.total_return_pct for run in runs],
                    [run.sharpe for run in runs], [run.costs["impact"] for run in runs],
                    [run.held for run in runs], [run.cost_bps for run in runs],
                    [run.turnover for run in runs])


def cost_curve(policy, data, venue, scales=SCALES, sizing="weight", stop=None,
               take_profit=None, tolerance=0.0, risk_free=0.0):
    """What ``policy`` keeps as its costs grow: a backtest of it on ``data`` at
    ``venue``, a Venue or a preset's name, with every cost the venue charges for
    trading multiplied by each of ``scales`` in turn, each fee rate above 0, maker and
    taker, the spread, fixed or estimated, the slippage and the impact coefficient, and
    the rest of backtest's settings, as a `CostCurve` of each run's round-trip cost,
    return, Sharpe, round trips, each cost paid and turnover (ADR 0045; Specification,
    Cost sweeps). A rebate, a fee rate below 0, the liquidity budget, funding and a
    liquidation fee the venue sets are not moved, one it does not set being the taker
    rate and moving as that does; and a venue that charges none of the four on the
    data's instruments is refused."""
    scales = _listed("scales", scales, _scale, "give the multiples of the venue's "
                     "costs to run at, numbers from 0, such as (0.0, 1.0, 2.0)")
    data = data if isinstance(data, Data) else Data(data)
    _charged(venue, data)

    runs = [_measured(_backtested(policy, data, venue, sizing, stop, take_profit,
                                  tolerance, "full", risk_free, costs=scale), data)
            for scale in scales]

    return CostCurve(scales, [run.cost_bps for run in runs],
                     [run.total_return_pct for run in runs],
                     [run.sharpe for run in runs], [run.num_trades for run in runs],
                     *([run.costs[name] for run in runs] for name in CHARGED),
                     [run.turnover for run in runs])


def breakeven_bps(policy, data, venue, ceiling=100.0, precision=0.05, sizing="weight",
                  stop=None, take_profit=None, tolerance=0.0, risk_free=0.0):
    """The round-trip cost, in basis points, at which ``policy`` stops making money on
    ``data`` at ``venue``: what its fills paid where its return reaches zero as every
    cost the venue charges for trading is multiplied, the multiple bisected from 0 and
    ``ceiling`` until the two runs that bracket it pay within ``precision`` basis
    points of each other, the mean of the two, or the lower one's where the upper
    filled nothing, with a warning where no scale between them brings them that
    close. None where it makes nothing with no costs at all, and where it still makes
    money at ``ceiling``, that run's cost, with a warning; the costs change which
    fills happen, so this finds a crossing rather than the crossing (ADR 0045;
    Specification, Cost sweeps)."""
    ceiling   = _positive("ceiling", ceiling, "give the largest multiple of the "
                          "venue's costs to try, such as 100")
    precision = _positive("precision", precision, "give the basis points the answer "
                          "may be off by, such as 0.05")

    data = data if isinstance(data, Data) else Data(data)
    _charged(venue, data)

    def ran(scale):
        run = _measured(_backtested(policy, data, venue, sizing, stop, take_profit,
                                    tolerance, "full", risk_free, costs=scale), data)
        return run.total_return_pct > 0.0, run.cost_bps

    gains, low_cost = ran(0.0)
    if not gains:
        return None

    gains, high_cost = ran(ceiling)
    if gains:
        warnings.warn(f"ceiling: {ceiling:g}, and the policy still makes money there, "
                      f"paying {high_cost:.1f} basis points a round trip; its "
                      f"break-even lies past it, so give a higher ceiling to find it",
                      ProbatumWarning, skip_file_prefixes=PACKAGE)
        return high_cost

    low, high = 0.0, ceiling
    for _ in range(HALVINGS):
        middle = (low + high) / 2.0
        # a middle at an end is a bracket no scale lies inside
        if abs(high_cost - low_cost) <= precision or middle in (low, high):
            break

        gains, cost = ran(middle)
        if gains:
            low, low_cost = middle, cost
        else:
            high, high_cost = middle, cost

    # a run past the break-even that filled nothing paid no cost to set beside the
    # last that made money
    if math.isnan(high_cost):
        return low_cost

    if abs(high_cost - low_cost) > precision:
        warnings.warn(f"precision: {precision:g}, and the two runs that bracket the "
                      f"break-even, {high - low:.3g} of a scale apart, pay "
                      f"{low_cost:.1f} and {high_cost:.1f} basis points a round trip, "
                      f"the cost jumping there; the break-even given is their mean, "
                      f"and a cost_curve at scales about {low:.4g} shows the jump",
                      ProbatumWarning, skip_file_prefixes=PACKAGE)

    return (low_cost + high_cost) / 2.0


def _measured(result, data):
    """What a sweep reads of a run's `Result`, at the full level: its return, its
    Sharpe and its round trips, each cost it paid, the round-trip cost its fills paid,
    its turnover and the share of its orders the liquidity budget held (Specification,
    What a run of them reads)."""
    fills = result.fills
    size  = np.abs(np.asarray(fills["size"], dtype=np.float64))
    price = np.asarray(fills["price"], dtype=np.float64)
    first = data.instruments[0]

    # an inverse account holds its one contract, whose notional is in its coin
    notional = (size * first.contract / price if first.market == "inverse"
                else size * price)
    traded   = math.fsum(notional.tolist())
    costs    = result.costs
    paid     = math.fsum(costs[name] for name in CHARGED)
    equity   = np.asarray(result.equity, dtype=np.float64)
    mean     = float(equity.mean()) if len(equity) else 0.0
    rate     = result.provenance["periods_per_year"]
    turnover = (traded / mean * rate / (len(equity) - 1)
                if rate is not None and len(equity) > 1 and mean > 0.0 else math.nan)

    placed, named = set(), set()
    for event in result.orders:
        if event["kind"] == "placed":
            placed.add(event["order"])
        if "the liquidity budget" in (event["reason"] or ""):
            named.add(event["order"])

    stats = result.stats
    return Measured(
        total_return_pct = math.nan if stats is None else stats["total_return_pct"],
        sharpe           = math.nan if stats is None else stats["sharpe"],
        num_trades       = 0 if stats is None else stats["num_trades"],
        costs            = costs,
        cost_bps         = 2.0 * 10_000.0 * paid / traded if traded > 0.0 else math.nan,
        turnover         = turnover,
        held             = len(placed & named) / len(placed) if placed else 0.0,
    )


def _charged(venue, data):
    """Refuse a sweep of ``venue``'s costs where it charges no fee, spread, slippage or
    impact on ``data``'s instruments, since no multiple of them moves a cost."""
    settings = _run(venue, data).settings
    fees     = any(spec.get(fee, 0.0) > 0.0
                   for spec in (settings, *settings.get("instruments", []))
                   for fee in ("fee_maker", "fee_taker"))
    if (fees or settings["spread"] not in (None, 0.0) or settings["slippage_bps"] != 0.0
            or settings["impact"] != 0.0):
        return

    named = venue if isinstance(venue, str) else (venue.name or "the venue")
    raise VenueError(f"venue: {named} charges no fee, spread, slippage or impact on "
                     f"{', '.join(data.names)}, so no multiple of its costs moves one; "
                     f"give a venue that charges some, such as probatum.venues."
                     f"binance_perp")


def _listed(name, given, read, fix):
    """``given``, the levels a sweep runs at, passed as ``name``, each read by
    ``read``, refused where it is no sequence or holds none."""
    if isinstance(given, (str, bytes, dict)) or not hasattr(given, "__iter__"):
        raise ProbatumTypeError(f"{name}: {given!r}; {fix}")

    levels = [read(value) for value in given]
    if not levels:
        raise ProbatumValueError(f"{name}: none; {fix}")

    return levels


def _opening(balance):
    """One of capacity's balances, as a Venue reads and holds one."""
    try:
        return _balance(_python(balance))
    except (ProbatumTypeError, VenueError) as error:
        raise type(error)(f"balances: {balance!r}; give each opening cash as a number "
                          f"from 0, or {{\"spot\": s, \"margin\": m}} for an account "
                          f"that holds both wallets") from None


def _shaped(balances, venue, data):
    """Refuse a balance of ``balances`` the account cannot open with on ``data`` at
    ``venue``: one wallet's where its instruments trade spot and perpetuals, which use
    both, and two where it trades an inverse contract, whose account holds its coin in
    one."""
    markets = [instrument.market or venue.kind for instrument in data.instruments]

    for balance in balances:
        if (not isinstance(balance, dict) and "spot" in markets
                and any(market not in ("spot", None) for market in markets)):
            raise VenueError(f"balances: {balance!r}, one wallet's, and the data's "
                             f"spot and perpetual instruments use both; give each as "
                             f"{{\"spot\": s, \"margin\": m}}")

        if isinstance(balance, dict) and markets == ["inverse"]:
            raise VenueError(f"balances: {balance!r}, the cash of two wallets, and an "
                             f"inverse contract's account holds its coin in one; give "
                             f"each as a number of coins")


def _scale(scale):
    """One of a cost sweep's scales, a number from 0."""
    if not _real(scale):
        raise ProbatumTypeError(f"scales: {scale!r}; give multiples of the venue's "
                                f"costs, numbers from 0")

    if not (math.isfinite(scale) and scale >= 0.0):
        raise ProbatumValueError(f"scales: {scale!r}; give multiples of the venue's "
                                 f"costs from 0")

    return float(scale)


def _positive(name, value, fix):
    """``value``, passed as ``name``, refused unless it is a finite number above 0."""
    if not _real(value):
        raise ProbatumTypeError(f"{name}: {value!r}; {fix}")

    if not (math.isfinite(value) and value > 0.0):
        raise ProbatumValueError(f"{name}: {value!r}; {fix}")

    return float(value)


def _floats(values):
    """A record's column of numbers, as floats."""
    return tuple(float(value) for value in values)


def _counted(levels, unit):
    """How many levels a record holds, as its repr names them."""
    return f"{len(levels):,} {unit}{'s' * (len(levels) != 1)}"


class _Wallets(collections.abc.Mapping):
    """A balance of two wallets as a Capacity holds it: read as the dict it was given,
    and set once and hashed, as the record is."""

    __slots__ = ("_cash",)


    def __init__(self, cash):
        self._cash = dict(cash)


    def __getitem__(self, wallet):
        return self._cash[wallet]


    def __iter__(self):
        return iter(self._cash)


    def __len__(self):
        return len(self._cash)


    def __hash__(self):
        return hash(frozenset(self._cash.items()))


    def __reduce__(self):
        return _Wallets, (self._cash,)


    def __repr__(self):
        return repr(self._cash)


def _opened(balance):
    """A balance as a table prints it: an amount, or each wallet's."""
    if isinstance(balance, _Wallets):
        return f"{_money(balance['spot'])} spot, {_money(balance['margin'])} margin"
    return _money(balance)


def _cell(value, form):
    """A number as a table prints it, "none" where it is NaN."""
    return "none" if math.isnan(value) else form.format(value)


def _table(header, rows):
    """``rows`` of text under ``header``, each column as wide as its widest cell, the
    first left-aligned and the rest right-aligned."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    return "\n".join("   ".join([cells[0].ljust(widths[0])]
                                + [cell.rjust(width)
                                   for cell, width in zip(cells[1:], widths[1:])])
                     for cells in (header, *rows))
