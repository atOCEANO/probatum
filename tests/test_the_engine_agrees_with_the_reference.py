"""The engine against the reference simulator, one random case at a time.

The reference is written from the decisions and never from the engine, so a case
the two part company on is a question about a rule, and the run prints it. Every
gate stage runs a short comparison; the slow stage runs it at length.
"""

import pytest

from reference import differential


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_random_cases_agree_on_position_equity_and_funding(seed):
    assert differential.main(["60", str(seed)]) == 0


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42, 1337, 20260813, 99991])
def test_random_cases_agree_at_length(seed):
    assert differential.main(["500", str(seed)]) == 0
