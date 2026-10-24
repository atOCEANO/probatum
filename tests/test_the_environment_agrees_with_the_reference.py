"""Every environment of a batch against a reference episode, step by step.

What an agent trains on is the environment's observation, reward, state and flags,
worked in Rust for the whole batch at once. Each environment over a random universe
is held to a reference episode opened at the same row on the same slots, written from
the Specification and knowing nothing of the engine: its slots shown or hidden as
they enter, print and delist, its weights decoded from every kind of action, its
orders resting whole ticks off the close, and its episodes ended at a liquidation, at
the last bar or at their length.
"""

import pytest

from reference import environment_differential


def check(cases, seed):
    out    = environment_differential.compare(cases, seed)
    report = environment_differential.render(out, cases)

    assert not out.disagreed, report
    assert out.compared >= cases // 2, report
    # a pass over episodes that never liquidated, rested an order or hid a slot
    # proves nothing about those paths
    assert all(out.reached.values()), report


@pytest.mark.parametrize("seed", [7, 42])
def test_every_environment_agrees_with_a_reference_episode(seed):
    check(40, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [7, 42, 20261024])
def test_every_environment_agrees_at_length(seed):
    check(200, seed)
