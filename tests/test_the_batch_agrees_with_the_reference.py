"""The batch against one reference per env, each started at that env's own offset.

Every reinforcement learning number comes out of the batch, and an episode that
starts part way into the series still funds on the bars the series funds on, which
no comparison that starts every env at bar zero can see.
"""

import pytest

from reference import batch_differential


@pytest.mark.parametrize("seed", [7, 42])
def test_every_env_agrees_with_a_reference_started_at_its_offset(seed):
    assert batch_differential.main(["40", str(seed)]) == 0


@pytest.mark.slow
@pytest.mark.parametrize("seed", [7, 42, 20260814])
def test_every_env_agrees_at_length(seed):
    assert batch_differential.main(["200", str(seed)]) == 0
