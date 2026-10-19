"""One fill's price or one order's life told in words (Python_API, Transparency): a fill
as its reference and each part that moved it to its price, then its fee, and an order
as the events the order log holds of it, each with its numbers."""

import math

import numpy as np

from ._data import _when
from ._errors import ProbatumTypeError, ProbatumValueError
from ._result import DOLLARS
from ._strategy import Order

FIX = ("pass a row of result.fills or result.orders, or an Order the run's account "
       "handed out")


def told(result, item):
    """The text that tells ``item`` of ``result``: a row of its fill log, a row of its
    order log, which tells that row's order, or an `Order` its run handed out."""
    book = "venue" in result.fills.columns

    if isinstance(item, Order):
        if book:
            raise ProbatumTypeError(f"item: {item!r}, a handle on one account of the "
                                    f"book; pass a row of book.orders, which names its "
                                    f"venue")

        token = None if result._record is None else result._record.token
        if token is None or item._run.token is not token:
            raise ProbatumValueError(f"item: {item!r}, of a run this Result does not "
                                     f"hold; pass a row of this Result's order log")
        return _order(result, item.id, None)

    if isinstance(item, dict) and "reference" in item:
        return _fill(result, _found(result.fills, item, "fill log"), item)

    if isinstance(item, dict) and "kind" in item:
        if item.get("order") is None:
            raise ProbatumValueError("item: an event of no order, such as a "
                                     "transfer's clamp; pass an event of an order, or "
                                     "a fill")
        _found(result.orders, item, "order log")
        return _order(result, item["order"], item.get("venue"))

    raise ProbatumTypeError(f"item: {item!r}; {FIX}")


def _found(table, row, named):
    """The position of the first row of ``table`` equal to ``row`` in every column,
    refused where it holds none."""
    if set(row) == set(table.columns):
        for at in np.flatnonzero(np.asarray(table["bar"]) == row["bar"]):
            if table[int(at)] == row:
                return int(at)

    raise ProbatumValueError(f"item: a row this Result's {named} does not hold; pass "
                             f"one of its rows")


def _account(result, venue):
    """The provenance of the account ``venue`` names in a book, or of the run."""
    given = result.provenance
    return given if venue is None else given["results"][venue]


def _rows(table, venue, mask):
    """The rows of ``table`` that ``mask`` keeps, of the account ``venue`` names in a
    book."""
    if venue is not None:
        mask = mask & table._held.equal("venue", venue)

    return [int(at) for at in np.flatnonzero(mask)]


def _assumed(result, venue):
    """The values of the Result's assumptions by name, of the account ``venue`` names in
    a book."""
    return {row["name"]: row["value"] for row in result.assumptions
            if row.get("venue") == venue}


def _fill(result, n, fill):
    """The fill at position ``n`` of the fill log, ``fill``, as its reference, each part
    of its price, what the margin bound did and its fee."""
    venue   = fill.get("venue")
    account = _account(result, venue)
    sign    = 1.0 if fill["side"] == "buy" else -1.0
    price, reference = fill["price"], fill["reference"]
    adjustment, lifted, paid = fill["adjustment"], fill["lifted"], fill["paid_in"]
    places  = _places(reference)
    inverse = account["markets"] == ["inverse"]

    # an inverse contract's account is kept in its coin, whatever its bars name
    asset = account["settlement"]
    coin  = inverse or asset is not None and asset not in DOLLARS

    # on an inverse contract a cost's rate is over the price it was paid at, before
    # the bar held it, since its parts were charged in the coin's reciprocal
    base = price - adjustment if inverse else reference

    rows = _assumed(result, venue)
    taker, reason = fill["taker"], fill["reason"]
    # a taker pays its parts, a sale on the book among them, and a maker, a takeover
    # and a delisting's close pay none
    paying = (None if taker else "none in a takeover" if reason == "liquidation"
              else "none at a delisting" if reason == "delist" else "none for a maker")

    def bps(part):
        return part / base * 10_000.0

    spread = rows["spread"]
    if paying is not None:
        spread_said = paying
    elif spread.startswith(("fixed", "estimated")):
        word        = "fixed" if spread.startswith("fixed") else "estimated"
        spread_said = f"half of {_bps(2.0 * bps(fill['spread']))} bps, {word}"
    else:
        spread_said = spread

    slippage_said = paying or (rows["slippage"] if rows["slippage"] == "not modelled"
                               else f"{_bps(bps(fill['slippage']))} bps")
    impact_said = paying or rows["impact"]
    edge        = "high" if sign > 0.0 else "low"
    held        = adjustment - lifted
    lines       = [("reference", _reference(result, n, fill), _price(reference, places)),
                   ("spread", spread_said, _signed(sign * fill["spread"], places)),
                   ("slippage", slippage_said, _signed(sign * fill["slippage"], places)),
                   ("impact", impact_said, _signed(sign * fill["impact"], places)),
                   ("held in the bar", "closed before the bar opened" if reason == "delist"
                    else f"held to the bar's {edge}" if held != 0.0
                    else f"the bar's {edge} was not reached", _signed(held, places))]

    if lifted != 0.0 or paid != 0.0:
        lines += [("margin bound", "lifted toward where equity is zero",
                   _signed(lifted, places)),
                  ("margin bound", "paid in for what the bar could not hold",
                   _money(paid, coin))]

    size, fee = fill["size"], fill["fee"]
    notional = size * account["contract"] / price if inverse else size * price
    if fee == 0.0:
        fee_said = "none"
    else:
        rate = (f"liquidation fee {_rate(fee / notional)}"
                if reason == "liquidation" and not taker
                else f"{'taker' if taker else 'maker'} {_rate(fee / notional)}")
        fee_said = f"{rate} on {_money(notional, coin)}"

    lines.append(("fee", fee_said, _money(fee, coin)))

    who  = "" if venue is None else f"{venue} "
    head = (f"fill {n}, {who}{fill['instrument']} {fill['side']} {_size(size)} at "
            f"{_price(price, places)}, {_when(result.time[fill['bar']])}")
    return "\n".join([head] + _laid(lines))


def _reference(result, n, fill):
    """What fill ``n``'s reference is, by its reason: the open, its order's own price,
    what traded to it, or a delisted instrument's last close."""
    reason = fill["reason"]
    if reason in ("market", "rebalance"):
        return "open of the bar"

    if reason == "liquidation":
        return ("where the sale began" if fill["taker"]
                else "where the account was closed out")

    if reason == "delist":
        return "its instrument's last close"

    if reason not in ("limit", "take_profit", "stop"):
        raise ProbatumValueError(f"item: a fill of reason {reason}, which explain does "
                                 f"not tell; pass a fill of another reason")

    asked = _asked(result, n, fill)
    if reason == "stop":
        return "its trigger" if fill["reference"] == asked else (
            "past its trigger, where the bar stood")

    if fill["reference"] == asked:
        return "the limit's price"

    fills, venue = result.fills, fill.get("venue")
    earlier = _rows(fills, venue, np.asarray(fills["bar"]) == fill["bar"])
    if any(at < n and fills[at]["instrument"] == fill["instrument"] for at in earlier):
        return ("what traded before it since the last close, or since the fill inside "
                "its bar")

    return "what traded since the last close"


def _asked(result, n, fill):
    """The price fill ``n``'s order asked as it filled: its limit's or its stop's, on
    the last event that placed or armed it before the event that logged the fill."""
    venue, order = fill.get("venue"), fill["order"]
    fills, orders = result.fills, result.orders

    own   = _rows(fills, venue, np.ma.filled(fills["order"] == order, False))
    count = own.index(n)
    asked = None

    for at in _rows(orders, venue, np.ma.filled(orders["order"] == order, False)):
        event = orders[at]
        if event["kind"] in ("placed", "armed"):
            asked = event["price"]
        elif event["kind"] == "filled":
            if not count:
                return asked
            count -= 1

    return asked


def _order(result, order, venue):
    """Order ``order``'s life, of the account ``venue`` names in a book: a head from
    the event that placed it, or refused it where none did, then each of its events."""
    orders, fills = result.orders, result.fills
    held   = np.ma.filled(orders["order"] == order, False)
    events = [orders[at] for at in _rows(orders, venue, held)]
    if not events:
        raise ProbatumValueError(f"item: order {order}, which this Result's order log "
                                 f"does not hold; pass one of its rows")

    filled = iter(_rows(fills, venue, np.ma.filled(fills["order"] == order, False)))
    lines  = []
    for event in events:
        kind = event["kind"]
        if kind == "filled":
            said = (f"{_size(event['size'])} at {_price(event['price'])}, fill "
                    f"{next(filled)}")
        elif kind in ("placed", "armed"):
            asking = _asking(event)
            joint  = " " if asking.startswith("at ") else ", "
            said   = joint.join(part for part in (event["reason"], asking) if part)
        else:
            said = event["reason"]

        lines.append((_when(result.time[event["bar"]]), kind, said))

    who    = "" if venue is None else f"{venue} "
    placed = next((event for event in events if event["kind"] == "placed"), None)
    if placed is None:
        refused = events[0]
        head    = " ".join(part for part in (refused["instrument"], refused["side"],
                                             _sized(refused)) if part)
        head = f"order {order}, {who}{head}, refused"
    else:
        head = " ".join(part for part in (placed["side"], placed["reason"],
                                          _asking(placed, side=False)) if part)
        head = f"order {order}, {who}{placed['instrument']} {head}"

    width = max(len(kind) for _, kind, _ in lines)
    return "\n".join([head] + [f"  {when}  {kind:<{width}}  {said}"
                               for when, kind, said in lines])


def _asking(event, side=True):
    """What an event's order asks: its side, where ``side``, its size and its price,
    each where the event holds one."""
    parts = [event["side"] if side else None, _sized(event)]
    if event["price"] is not None:
        parts.append(f"at {_price(event['price'])}")

    return " ".join(part for part in parts if part)


def _sized(event):
    return None if event["size"] is None else _size(event["size"])


def _laid(lines):
    """``lines``, each a label, a text and a number, laid out in three columns."""
    widths = [max(len(line[k]) for line in lines) for k in range(3)]
    return [f"  {label:<{widths[0]}}  {said:<{widths[1]}}  {number:>{widths[2]}}"
            for label, said, number in lines]


def _places(price):
    """The decimals a fill's prices print to: two from 1 up, else enough for five
    significant figures, and two for a price at or below zero, which a limit or a stop
    on a perpetual can be placed at."""
    if not 0.0 < price < 1.0:
        return 2

    return 4 - math.floor(math.log10(price))


def _price(value, places=None):
    """A price at ``places`` decimals, its own where None, with thousands separated."""
    return f"{value:,.{_places(value) if places is None else places}f}"


def _signed(value, places):
    """A part of a price with its sign, zero as +0.00."""
    said = f"{value:+,.{places}f}"
    return "+" + said[1:] if float(said.replace(",", "")) == 0.0 else said


def _money(value, coin):
    """Money to the cent, or where ``coin`` says it is kept in a coin, to five
    significant figures."""
    places = 2
    if coin and value != 0.0:
        places = max(2, 4 - math.floor(math.log10(abs(value))))

    return f"{value:,.{places}f}"


def _rate(rate):
    """A rate as a percentage, to three significant figures."""
    return f"{rate * 100.0:.3g}%"


def _bps(bps):
    """Basis points to one decimal, or below one to three significant figures, a hair
    below one that rounds to it read as one."""
    said = f"{bps:.3g}"
    return f"{bps:.1f}" if float(said) >= 1.0 else said


def _size(size):
    """A size to twelve significant figures in fixed point, its trailing zeros dropped
    and its thousands separated, with its sign, and an infinite one as Python prints
    it, as a refused order's can be."""
    if not math.isfinite(size):
        return f"{size}"

    said = np.format_float_positional(abs(size), precision=12, unique=False,
                                      fractional=False, trim="-")
    whole, dot, part = said.partition(".")
    return f"{'-' if size < 0.0 else ''}{int(whole):,}{dot}{part}"
