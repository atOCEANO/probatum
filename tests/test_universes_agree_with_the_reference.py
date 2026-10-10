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
    "a corner or the closes reaching the trigger",
    "a favourable exit held back on a liquidating bar",
    "a fill held at a tier's bound",
    "a fill held at the last tier's bound",
    "a fill held to the margin cap",
    "a flip",
    "a flip filled on the gap",
    "a flip's old side on the check",
    "a fok order filling nothing",
    "a forced close of a leg that joined during the bar",
    "a growing fill refused at no equity",
    "a held perp read at its last mark",
    "a leg joining the check after the open",
    "a limit filled as a maker",
    "a limit filled as a taker",
    "a liquidation at the maintenance trigger",
    "a liquidation before the bar opens",
    "a liquidation by the check",
    "a liquidation of several perpetuals",
    "a market order",
    "a market order filling nothing",
    "a position held through a row without its bar",
    "a post_only order refused at the last close",
    "a reduce_only order held to the position",
    "a reduce_only order with nothing to reduce",
    "a reducing fill bounded at a fee of its own",
    "a replace",
    "a round trip closed",
    "a spot buy held to its cash",
    "a spot sale held to the holding",
    "a stop triggered",
    "a takeover",
    "a taker's price held inside the bar",
    "a target",
    "a target replaced in its decision",
    "a target's spot shortfall moved across",
    "a tier above the first",
    "a transfer",
    "a transfer clamped",
    "an account over the cap kept from growing",
    "an exit held back from a position the bar's orders opened",
    "an ioc or fok limit expiring after one bar",
    "an order waiting for its instrument's next bar",
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


# universes built where random ones seldom reach a rule, each with the rule it reaches
BUILT = {
    "every corner and the closes clear of the trigger": (
        "a path inside the bars that never reaches the trigger",
        bitcoin_and_ether((57_000.0, 3_150.0), "optimistic")),
    "the closes at the trigger":                        (
        "a corner or the closes reaching the trigger",
        bitcoin_and_ether((55_000.0, 3_250.0), "optimistic")),
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
