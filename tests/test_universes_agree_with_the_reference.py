"""The engine against the reference over universes of several instruments.

The comparison `test_the_engine_agrees_with_the_reference` makes, over two to five
spot and perpetual instruments sharing a clock with holes in it and one margin
wallet. Every gate stage runs a short comparison, and the slow stage runs it at
length, where every rule the two share across instruments has to be reached as well.
A rule random universes reach too rarely to rely on has a universe built for it.
"""

import collections

import pytest

from reference import differential, universe_differential
from reference.simulator import Bar

# the rules both simulators apply across several instruments, each of which a
# comparison at length must reach
SHARED = [
    "a bar clipped",
    "a bar clipped across several perpetuals",
    "a bar clipped for a leg that joined it",
    "a bracket armed",
    "a bracket armed by a resting fill",
    "a bracket armed on a short",
    "a bracket cancelled by a liquidation",
    "a bracket cancelled with its position",
    "a bracket leg filled",
    "a bracket leg filled on the bar a resting fill armed it",
    "a bracket leg filled on the bar it armed",
    "a bracket waiting through a row without its bar",
    "a corner or the closes reaching the trigger",
    "a favourable exit held back on a liquidating bar",
    "a fill held at a tier's bound",
    "a fill held at the last tier's bound",
    "a fill held to the margin cap",
    "a flip",
    "a flip filled on the gap",
    "a flip's old side on the check",
    "a flip split by a target",
    "a fok order filling nothing",
    "a growing fill refused at no equity",
    "a held perp read at its last mark",
    "a leg joining the check after the open",
    "a limit filled as a maker",
    "a limit filled as a taker",
    "a limit on the book its bar opened through",
    "a liquidation at the maintenance trigger",
    "a liquidation before the bar opens",
    "a liquidation by the check",
    "a liquidation of several perpetuals",
    "a market order",
    "a market order filling nothing",
    "a position held through a row without its bar",
    "a post_only limit its first bar opened through",
    "a post_only order refused at the last close",
    "a reduce_only order held to the position",
    "a reduce_only order with nothing to reduce",
    "a reducing fill bounded at a fee of its own",
    "a replace",
    "a round trip closed",
    "a sale on the book and its backstop level both reached",
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
    "a tier above the first",
    "a transfer",
    "a transfer clamped",
    "an OCO partner cancelled",
    "an account over the cap kept from growing",
    "an exit held back from a position the bar's orders opened",
    "an ioc or fok limit expiring after one bar",
    "an order trading against the account's own refused",
    "an order waiting for its instrument's next bar",
    "book then backstop falling back to takeover",
    "book then backstop selling on the book",
    "funding at the close of its row",
    "funding between rows",
    "funding inside a bar counted toward the check",
    "funding inside a bar counted toward the path carried in",
    "slippage paid",
    "the cap with the other legs at the fill's fraction",
    "the margin bound beside another perp",
    "the margin bound lifting a fill inside its bar",
    "the margin bound paying what the bar cannot hold",
    "tiers by quantity",
]


MARGINED = [
    "a bar clipped across several perpetuals",
    "a liquidation at the maintenance trigger",
    "a liquidation of several perpetuals",
    "a tier above the first",
]


def bitcoin_and_ether(closes, intrabar):
    """The Specification's account at two instruments, built as a universe: 10,060 of
    margin buys 1 BTC at 60,000 and sells 20 ETH at 3,000 for fees of 30 each, then
    BTC falls to 54,000 and ETH rises to 3,300 on one bar, closing at `closes`."""
    def flat(price):
        return Bar(price, price, price, price, 1e6)

    columns = [
        [flat(60_000.0)] * 3 + [Bar(60_000.0, 60_000.0, 54_000.0, closes[0], 1e6)],
        [flat(3_000.0)] * 3 + [Bar(3_000.0, 3_300.0, 3_000.0, closes[1], 1e6)],
    ]

    times = [k * differential.HOUR for k in range(4)]
    cfg   = dict(kinds=["linear", "linear"], quote=10_060.0, balance=10_060.0,
                 fee_taker=0.0005, fee_maker=0.0002, slippage_bps=0.0,
                 max_open_orders=8, leverage=0.0, funding=("none",),
                 tiers=[(None, "notional")] * 2, intrabar=intrabar)

    actions = [(0, "market", "buy", 1.0, 0.0), (1, "market", "sell", 20.0, 0.0), None]

    return times, ["linear", "linear"], columns, cfg, actions


def closed_out_on_two(units):
    """Longs of `units` and of 10 on two perpetuals from 100, on the margin that
    leaves the second's closing fee once the first is marked at 90, with a taker fee
    of 0.2%, closed by one target on a bar where the first opens at 90, which the
    bound lifts to 90.18 to leave that fee, and the second at 100, which the fee
    takes to the zero exact arithmetic gives; the buy decided at that close then
    has nothing to back it."""
    def flat(price):
        return Bar(price, price, price, price, 1e9)

    margin  = 10.0 * units + 2.0
    quote   = margin + 0.2 * units + 2.0
    columns = [
        [flat(100.0)] * 3 + [Bar(90.0, 95.0, 90.0, 92.0, 1e9), flat(92.0)],
        [flat(100.0)] * 3 + [Bar(100.0, 105.0, 100.0, 103.0, 1e9), flat(103.0)],
    ]

    times = [k * differential.HOUR for k in range(5)]
    cfg   = dict(kinds=["linear", "linear"], quote=quote, balance=quote,
                 fee_taker=0.002, fee_maker=0.0, slippage_bps=0.0, max_open_orders=8,
                 leverage=0.0, funding=("none",), tiers=[(None, "notional")] * 2,
                 intrabar="pessimistic")

    actions = [(0, "market", "buy", units, 0.0), (1, "market", "buy", 10.0, 0.0),
               (0, "weights", "buy", {0: 0.0, 1: 0.0}, 0.0),
               (0, "market", "buy", 5.0, 0.0), None]

    return times, ["linear", "linear"], columns, cfg, actions


def waiting_and_unlisted():
    """Two perpetuals, the second printing its first bar on the third row and none on
    the fourth and fifth: a target on the first row names it before it prints, and
    its leg is refused, and a target on the fourth row replaces the leg the third
    row's left waiting for its next bar."""
    def flat(price):
        return Bar(price, price, price, price, 1e6)

    columns = [
        [flat(100.0)] * 7,
        [None, None, flat(10.0), None, None, flat(10.0), flat(10.0)],
    ]

    times = [k * differential.HOUR for k in range(7)]
    cfg   = dict(kinds=["linear", "linear"], quote=10_000.0, balance=10_000.0,
                 fee_taker=0.0, fee_maker=0.0, slippage_bps=0.0, max_open_orders=8,
                 leverage=0.0, funding=("none",), tiers=[(None, "notional")] * 2,
                 intrabar="pessimistic")

    actions = [(0, "weights", "buy", {0: 0.2, 1: 0.2}, 0.0), None,
               (0, "weights", "buy", {1: 0.5}, 0.0),
               (0, "weights", "buy", {1: 0.5}, 0.0), None, None]

    return times, ["linear", "linear"], columns, cfg, actions


def beside(first, actions, margin=100.0):
    """Two perpetuals on three quiet rows at 100 and then `first`, an (open, high,
    low, close), on the first and a quiet row on the second, on `margin` with no cap,
    fees or slippage; `actions` are decided at the quiet closes, a stop's and a
    limit's price set by the `u` that puts it at the price named."""
    columns = [
        [Bar(100.0, 100.0, 100.0, 100.0, 1e6)] * 3 + [Bar(*first, 1e6)],
        [Bar(100.0, 100.0, 100.0, 100.0, 1e6)] * 4,
    ]

    times = [k * differential.HOUR for k in range(4)]
    cfg   = dict(kinds=["linear", "linear"], quote=margin, balance=margin,
                 fee_taker=0.0, fee_maker=0.0, slippage_bps=0.0, max_open_orders=8,
                 leverage=0.0, funding=("none",), tiers=[(None, "notional")] * 2,
                 intrabar="pessimistic")

    return times, ["linear", "linear"], columns, cfg, actions


# universes built where random ones seldom reach a rule, each with the rule it reaches
BUILT = {
    "a target naming an instrument before its first bar": (
        "a target leg on an instrument yet to print", waiting_and_unlisted()),
    "a target replacing the leg an earlier one left waiting": (
        "a waiting target leg replaced", waiting_and_unlisted()),
    "two perps closed out where the margin ran out": (
        "a wallet a fill left within rounding of zero set to zero",
        closed_out_on_two(10.0)),
    # the first close is a million units, whose rounding the second's own fill is
    # far too small to measure
    "two perps closed out where the margin ran out, the first a million units": (
        "a wallet a fill left within rounding of zero set to zero",
        closed_out_on_two(1e6)),
    "every corner and the closes clear of the trigger": (
        "a path inside the bars that never reaches the trigger",
        bitcoin_and_ether((57_000.0, 3_150.0), "optimistic")),
    "the closes at the trigger": (
        "a corner or the closes reaching the trigger",
        bitcoin_and_ether((55_000.0, 3_250.0), "optimistic")),
    # beside a long on the second, a buy stop at 105 opens a long on the first on a
    # bar from 100, and a reduce-only stop at 102 fills at 102, where the price crossed
    # it after the entry, and not at the open
    "a stop between the open and a breakout long's entry, beside another perp": (
        "an exit priced from its position's fill",
        beside((100.0, 108.0, 98.0, 103.0),
               [(1, "market", "buy", 0.2, 0.0), (0, "stop_entry", "buy", 0.2, 0.625),
                (0, "stop", "sell", 0.2, 0.55)])),
    # a long of 7 on the first, beside one of 5 on the second, on 100 of margin runs
    # out below 86 on a bar down to 85, and a buy of 2 at 95 decided before it grows
    # the first on the way, so the forced close takes the first in two legs, the one
    # it carried and the one it grew
    "a forced close of a long grown on its liquidating bar": (
        "a forced close of a leg that joined during the bar",
        beside((100.0, 100.0, 85.0, 86.0),
               [(0, "market", "buy", 7.0, 0.0), (1, "market", "buy", 5.0, 0.0),
                (0, "limit", "buy", 2.0, (0.95 - 0.8) / 0.21)])),
}


@pytest.mark.parametrize("case", sorted(BUILT))
def test_a_universe_built_for_a_rule_agrees_and_reaches_it(case):
    rule, built = BUILT[case]
    counted = collections.defaultdict(collections.Counter)
    records, sim = universe_differential.run(built, counted)

    assert universe_differential.disagreement(records, sim) is None
    assert sim.reached[rule]


def check(cases, seed):
    out    = universe_differential.compare(cases, seed)
    report = universe_differential.render(out, cases)

    assert not out.disagreed, report
    assert out.compared >= cases // 3, report

    idle = [kind for kind in differential.TRACKED if not out.reached[kind]]

    assert not idle, f"never met a live position or order: {idle}\n{report}"

    return out


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_random_universes_agree_on_every_record(seed):
    check(60, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42, 1337, 20261005, 99991])
def test_random_universes_agree_at_length_and_reach_every_rule_both_apply(seed):
    out = check(360, seed)

    assert not [rule for rule in SHARED if not out.rules[rule]]
    # the universes drawn around shared margin reach its trigger and its tiers on
    # their own, where random ones seldom hold more than two perps at once
    assert not [rule for rule in MARGINED if not out.margined[rule]]
