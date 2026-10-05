"""The engine against the reference simulator, one random case at a time.

The reference is written from the Specification and never from the engine, so a case
the two part company on is a question about a rule, and the run prints it. A case is
compared only where the reference applied no rule the engine has yet to take up.
Every gate stage runs a short comparison; the slow stage runs it at length, where
every rule the two share has to be reached as well.
"""

import pytest

from reference import differential

# the rules both simulators apply, each of which a comparison at length must reach
SHARED = [
    "a bar clipped",
    "a fill held to the margin cap",
    "a flip",
    "a flip's old side on the check",
    "a fok order filling nothing",
    "a growing fill refused at no equity",
    "a leg joining the check after the open",
    "a limit filled as a maker",
    "a limit filled as a taker",
    "a liquidation before the bar opens",
    "a liquidation by the check",
    "a market order",
    "a market order filling nothing",
    "a position's exits reordered",
    "a post_only order refused at the last close",
    "a reduce_only order held to the position",
    "a reduce_only order with nothing to reduce",
    "a replace",
    "a spot buy held to its cash",
    "a spot sale held to the holding",
    "a stop triggered",
    "a takeover",
    "a taker's price held inside the bar",
    "an account over the cap kept from growing",
    "an ioc or fok limit expiring after one bar",
    "funding at the close of its row",
    "slippage paid",
]


def check(cases, seed):
    out    = differential.compare(cases, seed)
    report = differential.render(out, cases)

    assert not out.disagreed, report
    assert out.compared >= cases // 2, report

    idle = [kind for kind in differential.TRACKED if not out.reached[kind]]
    assert not idle, f"never met a live position or order: {idle}\n{report}"

    return out


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_random_cases_agree_on_position_equity_cash_and_funding(seed):
    check(60, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42, 1337, 20261005, 99991])
def test_random_cases_agree_at_length_and_reach_every_rule_both_apply(seed):
    out = check(500, seed)

    assert not [rule for rule in SHARED if not out.rules[rule]]
