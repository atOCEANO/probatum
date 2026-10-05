"""The batch against one reference per env, each run from that env's own offset.

Every reinforcement learning number comes out of the batch, and an episode that
starts part way into the series still funds on the bars the series funds on, which
no comparison that starts every env at bar zero can see.
"""

import pytest

from reference import batch_differential


def check(cases, seed):
    out    = batch_differential.compare(cases, seed)
    report = batch_differential.render(out, cases)

    assert not out.disagreed, report
    assert out.compared >= cases // 2, report
    # a pass over envs that never traded, funded, died or ran out of bars proves
    # nothing about those paths
    assert all(out.reached.values()), report


@pytest.mark.parametrize("seed", [7, 42])
def test_every_env_agrees_with_a_reference_started_at_its_offset(seed):
    check(40, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [7, 42, 20261005])
def test_every_env_agrees_at_length(seed):
    check(200, seed)
