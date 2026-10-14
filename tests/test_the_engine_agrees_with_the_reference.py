"""The engine against the reference simulator, one random case at a time.

The reference is written from the Specification and never from the engine, so a case
the two part company on is a question about a rule, and the run prints it. A case is
compared up to the first bar on which the reference applied a rule the engine has
yet to take up.
Every gate stage runs a short comparison; the slow stage runs it at length, where
every rule the two share has to be reached as well. A rule random cases reach too
rarely to rely on has a case built for it.
"""

import pytest

from reference import differential
from reference.simulator import Bar

# the rules both simulators apply, each of which a comparison at length must reach
SHARED = [
    "a bar clipped",
    "a bracket armed",
    "a bracket armed by a resting fill",
    "a bracket armed on a short",
    "a bracket cancelled by another's leg",
    "a bracket cancelled with its position",
    "a bracket leg filled",
    "a bracket leg filled on the bar a resting fill armed it",
    "a bracket leg filled on the bar it armed",
    "a fill held at a tier's bound",
    "a fill held at the last tier's bound",
    "a fill held to the margin cap",
    "a flip",
    "a flip's old side on the check",
    "a flip split by a target",
    "a fok order filling nothing",
    "a forced close of a leg that joined during the bar",
    "a growing fill refused at no equity",
    "a leg joining the check after the open",
    "a limit filled as a maker",
    "a limit filled as a taker",
    "a limit on the book its bar opened through",
    "a liquidation at the maintenance trigger",
    "a liquidation before the bar opens",
    "a liquidation by the check",
    "a market order",
    "a market order filling nothing",
    "a position's exits reordered",
    "a post_only limit its first bar opened through",
    "a post_only order refused at the last close",
    "a reduce_only order held to the position",
    "a reduce_only order with nothing to reduce",
    "a replace",
    "a sizeless leg sized at a position the bar's fills moved",
    "a spot buy held to its cash",
    "a spot sale held to the holding",
    "a stop triggered",
    "a takeover",
    "a taker's price held inside the bar",
    "a target",
    "a target leg skipped below the tolerance",
    "a target replaced in its decision",
    "a target's shortfall taken back",
    "a target's spot shortfall moved across",
    "a transfer",
    "a transfer clamped",
    "an OCO partner cancelled",
    "an account over the cap kept from growing",
    "an exit priced from its position's fill",
    "an ioc or fok limit expiring after one bar",
    "an order trading against the account's own refused",
    "book then backstop falling back to takeover",
    "funding at the close of its row",
    "funding between rows",
    "funding inside a bar counted toward the check",
    "funding inside a bar counted toward the path carried in",
    "an inverse contract filled",
    "slippage paid",
]


def built(*actions, shaped):
    """A case built by hand: three quiet bars at 100 and then `shaped`, an (open,
    high, low, close), on a perpetual with 100 of margin at 10x and no fees,
    slippage or funding, with an action decided at each quiet bar's close."""
    bars  = [Bar(100.0, 100.0, 100.0, 100.0, 1e6)] * 3 + [Bar(*shaped, 1e6)]
    times = [k * differential.HOUR for k in range(len(bars))]
    cfg   = dict(market="perp", quote=100.0, idle=0.0, fee_taker=0.0, fee_maker=0.0,
                 slippage_bps=0.0, max_open_orders=8, leverage=10.0, funding=("none",),
                 tiers=None, tiers_by="notional", intrabar="pessimistic")

    return bars, times, cfg, list(actions)


def at(kind, side, size, price):
    """An action at a close of 100 whose price comes out at `price`."""
    if kind in ("stop", "stop_entry"):
        base, step = 0.8, 0.4
    else:
        base, step = (0.8 if side == "buy" else 0.99), 0.21

    return kind, side, size, (price / 100.0 - base) / step


OPENED = "an exit held back from a position the bar's orders opened"


def debited_at_the_close():
    """A long of 9 from 100 on 100 of margin, through a bar down to 99 its path stays
    clear of, whose close pays a funding debit of 20% on its 900 of notional, 180,
    which takes equity to -80."""
    bars, times, cfg, actions = built(("market", "buy", 9.0, 0.0), None, None,
                                      shaped=(100.0, 101.0, 99.0, 100.0))
    cfg.update(funding=("every_bars", 3, 0.2))
    return bars, times, cfg, actions


def bounded_flat():
    """The Specification's margin bound: 50.5 of margin behind a long of 10 at 100,
    a taker fee of 0.1%, and a market sale on a bar that trades at 95 throughout,
    which the bound pays 0.45 into, leaving the account flat and its wallet at
    zero."""
    bars, times, cfg, actions = built(("market", "buy", 10.0, 0.0), None,
                                      ("market", "sell", 10.0, 0.0),
                                      shaped=(95.0, 95.0, 95.0, 95.0))
    cfg.update(quote=51.5, fee_taker=0.001, leverage=0.0)
    return bars, times, cfg, actions


def flipped_at_the_low():
    """A long of 10 from 100 on 480 of margin, at 5% maintenance and no cap, sold 20
    by a stop at its bar's low of 90, so the short joins the path where the long
    leaves it, at its end, and runs on alone to the high of 120."""
    bars, times, cfg, actions = built(None, ("market", "buy", 10.0, 0.0),
                                      at("stop_entry", "sell", 20.0, 90.0),
                                      shaped=(100.0, 120.0, 90.0, 110.0))
    cfg.update(quote=480.0, leverage=0.0, tiers=[(1e12, 0.0, 0.05, 0.0)])
    return bars, times, cfg, actions


def joined_past_its_trigger():
    """55 of margin at 50x and 1% maintenance, a buy limit of 10 at 95 placed at the
    second bar's close and a reduce-only stop at 90 at the third's, on a bar from 100
    down to 85: the long the limit opens passes its trigger, 90.40, before the stop
    could close it."""
    bars, times, cfg, actions = built(None, at("limit", "buy", 10.0, 95.0),
                                      at("stop", "sell", 10.0, 90.0),
                                      shaped=(100.0, 100.0, 85.0, 92.0))
    cfg.update(quote=55.0, leverage=0.0, tiers=[(1e12, 50.0, 0.01, 0.0)])
    return bars, times, cfg, actions


def past_its_trigger_at_the_open():
    """100 of margin with no cap and 1% maintenance buying 150 at an open of 100: a
    requirement of 150 against equity of 100 where the check's path starts, which
    closes the account there."""
    bars, times, cfg, actions = built(None, None, ("market", "buy", 150.0, 0.0),
                                      shaped=(100.0, 101.0, 99.0, 100.0))
    cfg.update(leverage=0.0, tiers=[(1e12, 0.0, 0.01, 0.0)])
    return bars, times, cfg, actions


def fees_below_zero():
    """5 of margin with no cap and a taker fee of 1%: a buy of 3 at 100 pays 3 and
    leaves 2, and its sale at the next open pays 3 more, the bound stopping at the
    entry, which leaves -1 with nothing held."""
    bars, times, cfg, actions = built(None, ("market", "buy", 3.0, 0.0),
                                      ("market", "sell", 3.0, 0.0),
                                      shaped=(100.0, 100.0, 100.0, 100.0))
    cfg.update(quote=5.0, fee_taker=0.01, leverage=0.0)
    return bars, times, cfg, actions


def joined_on_the_favourable_side():
    """An inverse account of 20 coins on contracts of 100 dollars, with no cap or
    maintenance, short 50 from 100 and selling 50 more by a stop at 90, below its
    bar's open, so the new leg runs from 90 and the carried one from the open, each
    to the high of 130 in the reciprocal of the price, where both are liquidated at
    prices a path read in dollars would put elsewhere."""
    bars, times, cfg, actions = built(None, ("market", "sell", 50.0, 0.0),
                                      at("stop_entry", "sell", 50.0, 90.0),
                                      shaped=(100.0, 130.0, 80.0, 120.0))
    cfg.update(market="inverse", contract=100.0, quote=20.0, leverage=0.0)
    return bars, times, cfg, actions


def sold_in_its_coin():
    """An inverse account of 1 coin on contracts of 100 dollars, at 1% maintenance
    with no cap, a taker fee of 0.06% and 5 bps of slippage, long 10 from 100 for a
    fee of 0.006, on a bar down to 91.60: its trigger is where 10.994 - 1,000 / p
    falls to 10 / p, 91.87, and its backstop level, two thirds of the requirement,
    91.57, lies below the low, so the long is sold on the book from the trigger."""
    bars, times, cfg, actions = built(("market", "buy", 10.0, 0.0), None, None,
                                      shaped=(100.0, 100.5, 91.6, 95.0))
    cfg.update(market="inverse", contract=100.0, quote=1.0, leverage=0.0,
               fee_taker=0.0006, slippage_bps=5.0, tiers=[(1e12, 0.0, 0.01, 0.0)],
               liquidation=("book_then_backstop", 2.0 / 3.0))
    return bars, times, cfg, actions


def joined_and_sold():
    """A long of 5 from 100 on 100 of margin, at 1% maintenance with no cap, a taker
    fee of 0.06% and 5 bps of slippage, and a buy limit of 5 at 98 the bar fills on
    its way down to 89.75: the two legs reach the trigger together at 89.93, where
    10p - 890.3 falls to 0.1p, short of the backstop level, two thirds of the
    requirement, at 89.63, and each is sold on the book there, the second taking
    what the position still holds."""
    bars, times, cfg, actions = built(("market", "buy", 5.0, 0.0), None,
                                      at("limit", "buy", 5.0, 98.0),
                                      shaped=(100.0, 100.0, 89.75, 92.0))
    cfg.update(leverage=0.0, fee_taker=0.0006, slippage_bps=5.0,
               tiers=[(1e12, 0.0, 0.01, 0.0)],
               liquidation=("book_then_backstop", 2.0 / 3.0))

    return bars, times, cfg, actions


def changed(case, **settings):
    """`case` with `settings` in place of its own."""
    bars, times, cfg, actions = case
    cfg.update(settings)
    return bars, times, cfg, actions


def crossed_as_it_arms():
    """A bid at 101 placed at the second bar's close, a third bar that trades nothing,
    so the bid still rests, and at its close a bracket placed flat with a market
    entry and a take-profit at 101 alone."""
    bars, times, cfg, actions = built(None, ("limit", "buy", 1.0, 1.0),
                                      ("bracket", "buy", 1.0, 1.0),
                                      shaped=(100.0, 102.0, 99.0, 100.0))
    bars[2] = Bar(100.0, 100.0, 100.0, 100.0, 0.0)
    return bars, times, cfg, actions


# cases built where random cases seldom reach a rule, each with the rule it reaches
BUILT = {
    # a buy opens a long a take-profit and a stop both close, the take-profit below
    # the close the bar reaches after the fill, and the stop, the worse, fills first
    # though typed last
    "a stop and a take-profit of a long the bar opened": (
        "the exits of a position the bar's orders opened reordered",
        built(at("limit", "buy", 1.0, 95.0), at("reduce_limit", "sell", 1.0, 110.0),
              at("stop", "sell", 1.0, 90.0), shaped=(100.0, 112.0, 85.0, 111.0))),
    # a sale above the open would close the long the bar's buy opened, which the
    # path without it takes to its trigger
    "a sale above the open on a long the bar opened": (OPENED, built(
        None, at("limit", "buy", 10.0, 95.0), at("limit", "sell", 10.0, 105.0),
        shaped=(100.0, 106.0, 80.0, 95.0))),
    # the same with a reduce-only sale below the open, above where the bar's buy
    # opened the long, which meets the bar from that fill and so needs a close above
    # it to reach it
    "a sale below the open on a long the bar opened lower": (OPENED, built(
        None, at("limit", "buy", 10.0, 95.0), at("reduce_limit", "sell", 10.0, 98.0),
        shaped=(100.0, 101.0, 80.0, 99.0))),
    # the short a flip opens at the long's extreme is tested without the long beside
    # it, and stays clear of its trigger
    "a flip at the low of a maintained long": (
        "a flip's old side on the check", flipped_at_the_low()),
    # a bar the path survives, closed at its closes by the debit charged there
    "a debit at the close that takes the account to its trigger": (
        "a liquidation at the closes", debited_at_the_close()),
    # a buy limit of 10 at 98.90 placed while flat with a take-profit at 102.66 after
    # it: the entry fills on the way down to 88, which liquidates the long, and the
    # take-profit it armed, which the bar reaches after the fill only on its way back
    # up to the close, is held back
    "a take-profit a limit entry armed on a liquidating bar": (
        "a leg a resting fill armed held back on a liquidating bar",
        built(None, None, ("bracket", "buy", 10.0, 0.9),
              shaped=(100.0, 104.0, 88.0, 103.0))),
    # the same bracket of 1, on a bar that reached 104 before the entry and closes at
    # 100: after the fill the bar goes no higher than that close, so the take-profit
    # rests
    "a take-profit the bar reached only before its limit entry": (
        "an exit the bar reached only before its position's fill",
        built(None, None, ("bracket", "buy", 1.0, 0.9),
              shaped=(100.0, 104.0, 95.0, 100.0))),
    # a sell limit at 105 opens a short inside a bar that reached 98 and closes at
    # 104: the low came before the fill, so a reduce-only buy at 102 rests
    "a take-profit of a short the bar reached only before its entry": (
        "an exit the bar reached only before its position's fill",
        built(at("limit", "sell", 1.0, 105.0), None,
              at("reduce_limit", "buy", 1.0, 102.0),
              shaped=(100.0, 106.0, 98.0, 104.0))),
    # on spot the buy at 95 makes the holding inside a bar from 100, and a sale at 97,
    # which only that holding can fill, meets the bar from there and rests
    "a spot sale the bar reached only before its holding was bought": (
        "an exit the bar reached only before its position's fill",
        changed(built(at("limit", "buy", 1.0, 95.0), None,
                      at("limit", "sell", 1.0, 97.0),
                      shaped=(100.0, 100.5, 94.0, 95.5)), market="spot")),
    # a long of 1 a sell of 2 at 105 flips short inside the bar, and a reduce-only
    # buy at 101 the bar reached only before the flip rests
    "a take-profit of a short a flip opened inside the bar": (
        "an exit the bar reached only before its position's fill",
        built(("market", "buy", 1.0, 0.0), at("limit", "sell", 2.0, 105.0),
              at("reduce_limit", "buy", 1.0, 101.0),
              shaped=(100.0, 106.0, 98.0, 103.0))),
    # a buy stop at 105 opens a long on a bar from 100, and a reduce-only stop at 102
    # fills at 102, where the price crossed it after the entry, and not at the open
    "a stop between the open and a breakout long's entry": (
        "an exit priced from its position's fill",
        built(at("stop_entry", "buy", 1.0, 105.0), at("stop", "sell", 1.0, 102.0),
              None, shaped=(100.0, 108.0, 98.0, 103.0))),
    # under the optimistic setting the high came after a buy at 95, and a reduce-only
    # sale at 98 decided at the last close reaches it as a maker, where the bar's open
    # of 100 would have made it a taker
    "a take-profit after a limit entry under the optimistic setting": (
        "an exit priced from its position's fill",
        changed(built(at("limit", "buy", 1.0, 95.0), None,
                      at("reduce_limit", "sell", 1.0, 98.0),
                      shaped=(100.0, 101.0, 94.0, 95.5)), intrabar="optimistic")),
    # a bracket placed while flat rests unarmed through two bars beside its limit
    # entry at 92.18, and a market buy arms it at the open of 100, through its
    # take-profit at 93.77, which then fills there as a taker on the way down to 93
    "a take-profit armed through the open it arms at": (
        "a limit filled as a taker",
        built(("bracket", "buy", 1.0, 0.58), None, ("market", "buy", 1.0, 0.0),
              shaped=(100.0, 101.0, 93.0, 95.0))),
    # a bid at 101 rests through a bar that trades nothing, and a bracket placed flat
    # beside a market entry arms its take-profit at 101 on that entry's fill, a sell
    # at the account's own bid, so it is cancelled as it arms
    "a take-profit armed against the account's own bid": (
        "a take-profit cancelled as it armed against the account's own",
        crossed_as_it_arms()),
    # a bracket on a long of 1, its stop at 95 and its take-profit at 105, and the
    # stop moved to 94 and given a size of 2: it sells the 1 held and goes with it
    "a stop given more than its long": (
        "a sized leg gone with the position its fill closed",
        built(("market", "buy", 1.0, 0.0), ("bracket", "buy", 1.0, 15.0 / 19.0),
              ("replace", "buy", 4.0, 0.35), shaped=(100.0, 101.0, 93.0, 95.0))),
    # a sale the bound pays into, which leaves the account flat at zero
    "a sale the bound pays into that leaves the account flat": (
        "a wallet a fill left within rounding of zero set to zero", bounded_flat()),
    # a long a limit opens inside the bar is liquidated at its trigger, which the bar
    # reaches before the reduce-only stop beyond it
    "a long opened inside the bar past its trigger before its stop": (
        "a bar clipped for a leg that joined it", joined_past_its_trigger()),
    # a long bought at the open already past its trigger, where the check's path
    # starts
    "a long past its trigger where it opened": (
        "a check closing the account at the open", past_its_trigger_at_the_open()),
    # a sale the bound stops at its entry, whose fee leaves the wallet below zero
    "a sale whose fee leaves the margin wallet below zero": (
        "a wallet that fees left below zero set to zero", fees_below_zero()),
    # two legs of one inverse perpetual on segments of their own, which the path
    # moves in step in the reciprocal of the price
    "an inverse short with a leg joined on the favourable side": (
        "a liquidation by the check", joined_on_the_favourable_side()),
    # book then backstop, which random cases seldom take to a sale: an inverse long,
    # and a long whose carried leg and the leg the bar joined are sold together
    "an inverse long sold on the book": (
        "book then backstop selling on the book", sold_in_its_coin()),
    "a long with a leg joined during the bar sold on the book": (
        "book then backstop selling on the book", joined_and_sold()),
}


@pytest.mark.parametrize("case", sorted(BUILT))
def test_a_case_built_for_a_rule_agrees_and_reaches_it(case):
    rule, (bars, times, cfg, actions) = BUILT[case]
    rows, records, sim = differential.run(bars, times, cfg, actions)

    assert differential.disagreement(rows, records, sim, cfg) is None
    assert sim.reached[rule]


def check(cases, seed):
    out    = differential.compare(cases, seed)
    report = differential.render(out, cases)

    assert not out.disagreed, report
    assert out.compared >= cases // 3, report

    idle = [kind for kind in differential.TRACKED if not out.reached[kind]]
    assert not idle, f"never met a live position or order: {idle}\n{report}"

    return out


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_random_cases_agree_on_position_equity_cash_and_funding(seed):
    check(120, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42, 1337, 20261005, 99991])
def test_random_cases_agree_at_length_and_reach_every_rule_both_apply(seed):
    out = check(500, seed)

    assert not [rule for rule in SHARED if not out.rules[rule]]
