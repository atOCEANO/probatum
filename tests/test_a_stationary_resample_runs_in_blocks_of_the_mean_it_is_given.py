"""A stationary resample runs on in blocks whose mean is the length it is given: each
first row drawn evenly, each next the one after the last with probability 1 - 1/b,
wrapping past the last return, and drawn evenly again otherwise, from a stream that the
values it resamples seed, so it repeats with no seed (Specification, Intervals of the
returns).

The agreement test hands these rows to the reference, which takes them as given, so
the rows' own law is held here.
"""

import numpy as np

from probatum import stats

COUNT, RESAMPLES = 500, 2_000
VALUES = np.random.default_rng(11).normal(0.0, 0.01, COUNT)


def rows(block, seed=None):
    rng = stats._drawn(VALUES, "stats returns", seed)
    return np.concatenate(list(stats._stationary(COUNT, block, RESAMPLES, rng)))


def test_a_row_runs_on_from_the_last_with_probability_1_minus_1_over_b():
    drawn = rows(4.0)
    on    = (drawn[:, 1:] == (drawn[:, :-1] + 1) % COUNT).mean()

    # a fresh draw lands on the next row one time in COUNT, which runs it on too
    expected = 0.75 + 0.25 / COUNT
    spread   = np.sqrt(expected * (1.0 - expected) / drawn[:, 1:].size)

    assert abs(on - expected) < 4.0 * spread


def test_a_block_of_1_draws_every_row_afresh():
    drawn = rows(1.0)
    on    = (drawn[:, 1:] == (drawn[:, :-1] + 1) % COUNT).mean()

    assert abs(on - 1.0 / COUNT) < 4.0 * np.sqrt(1.0 / COUNT / drawn[:, 1:].size)


def test_the_first_rows_are_drawn_evenly_and_the_last_row_runs_on_to_the_first():
    drawn     = rows(4.0)
    counts    = np.bincount(drawn.ravel(), minlength=COUNT)
    expected  = drawn.size / COUNT
    statistic = ((counts - expected) ** 2 / expected).sum()

    assert statistic < (COUNT - 1) + 6.0 * np.sqrt(2.0 * (COUNT - 1))

    wrapped = (drawn[:, :-1] == COUNT - 1) & (drawn[:, 1:] == 0)
    assert wrapped.sum() > 0.5 * 0.75 * drawn[:, 1:].size / COUNT


def test_the_same_values_draw_the_same_rows_and_a_seed_draws_others():
    assert np.array_equal(rows(4.0), rows(4.0))
    assert not np.array_equal(rows(4.0), rows(4.0, seed=7))
    assert np.array_equal(rows(4.0, seed=7), rows(4.0, seed=7))
