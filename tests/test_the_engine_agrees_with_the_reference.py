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
    "a bracket cancelled with its position",
    "a bracket leg filled",
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
    "an ioc or fok limit expiring after one bar",
    "an order trading against the account's own refused",
    "funding at the close of its row",
    "funding between rows",
    "funding inside a bar counted toward the check",
    "funding inside a bar counted toward the path carried in",
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


# cases built where random cases seldom reach a rule, each with the rule it reaches
BUILT = {
    # a buy opens a long a take-profit and a stop both close, and the stop, the
    # worse, fills first though typed last
    "a stop and a take-profit of a long the bar opened": (
        "the exits of a position the bar's orders opened reordered",
        built(at("limit", "buy", 1.0, 95.0), at("reduce_limit", "sell", 1.0, 110.0),
              at("stop", "sell", 1.0, 90.0), shaped=(100.0, 112.0, 85.0, 100.0))),
    # a sale above the open would close the long the bar's buy opened, which the
    # path without it takes to its trigger
    "a sale above the open on a long the bar opened": (OPENED, built(
        None, at("limit", "buy", 10.0, 95.0), at("limit", "sell", 10.0, 105.0),
        shaped=(100.0, 106.0, 80.0, 95.0))),
    # the same with the sale below the open, above where the bar's buy opened it
    "a sale below the open on a long the bar opened lower": (OPENED, built(
        None, at("limit", "buy", 10.0, 95.0), at("reduce_limit", "sell", 10.0, 98.0),
        shaped=(100.0, 101.0, 80.0, 90.0))),
    # the short a flip opens at the long's extreme is tested without the long beside
    # it, and stays clear of its trigger
    "a flip at the low of a maintained long": (
        "a flip's old side on the check", flipped_at_the_low()),
    # a bar the path survives, closed at its closes by the debit charged there
    "a debit at the close that takes the account to its trigger": (
        "a liquidation at the closes", debited_at_the_close()),
    # a sale the bound pays into, which leaves the account flat at zero
    "a sale the bound pays into that leaves the account flat": (
        "a wallet a fill left within rounding of zero set to zero", bounded_flat()),
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
