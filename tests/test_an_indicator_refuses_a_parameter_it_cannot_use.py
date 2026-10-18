"""An indicator refuses what it cannot compute over and says why: a length that is not
a whole number of bars from 1, or longer than the series, a band of negative width, a
fast average no faster than the slow one, a shift into the future, series of
different lengths, a level that is not one, a whole frame for a column and an empty
series.
"""

import numpy as np
import pytest

from probatum import ta
from test_every_indicator_gives_one_value_a_bar import RAMP


def test_a_length_longer_than_the_series_says_both_numbers():
    with pytest.raises(ValueError) as excinfo:
        ta.sma(RAMP, 50)

    assert "50" in str(excinfo.value) and "20" in str(excinfo.value)


def test_a_length_below_one_is_refused():
    with pytest.raises(ValueError):
        ta.sma(RAMP, 0)


def test_a_whole_frame_handed_in_place_of_a_column_is_refused():
    frame = np.column_stack([RAMP, RAMP + 1.0, RAMP - 1.0, RAMP, RAMP])

    with pytest.raises(ValueError) as excinfo:
        ta.sma(frame, 5)

    assert "not the whole frame" in str(excinfo.value)
    # a single column with a trailing axis is still a single column
    assert np.allclose(ta.sma(RAMP.reshape(-1, 1), 5), ta.sma(RAMP, 5), equal_nan=True)


def test_a_fractional_length_is_a_mistake_not_a_rounding_request():
    with pytest.raises(ValueError) as excinfo:
        ta.sma(RAMP, 3.9)

    assert "whole number of bars" in str(excinfo.value)


def test_rsi_needs_more_bars_than_its_length():
    # the first bar makes no change, so a length equal to the series returned an
    # entirely NaN array rather than saying anything
    with pytest.raises(ValueError) as excinfo:
        ta.rsi(RAMP, 20)

    assert "more bars than its length" in str(excinfo.value)
    assert np.isfinite(ta.rsi(RAMP, 19)[-1])


def test_a_negative_band_width_is_refused():
    with pytest.raises(ValueError) as excinfo:
        ta.bbands(RAMP, 5, deviations=-2.0)

    assert "at least 0" in str(excinfo.value)


def test_the_smoothing_length_is_named_by_the_function_that_took_it():
    with pytest.raises(ValueError) as excinfo:
        ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5, smooth=999)

    assert "stoch" in str(excinfo.value)


def test_mismatched_series_lengths_are_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        ta.stoch(RAMP, RAMP[:5], RAMP, 3)

    assert "one length" in str(excinfo.value)


def test_macd_needs_the_fast_leg_to_be_faster():
    with pytest.raises(ValueError) as excinfo:
        ta.macd(RAMP, 26, 12)

    assert "fast below slow" in str(excinfo.value)


def test_an_empty_series_is_refused():
    with pytest.raises(ValueError):
        ta.sma(np.array([]), 1)


def test_a_shift_cannot_be_asked_to_read_the_future():
    with pytest.raises(ValueError) as excinfo:
        ta.shift(RAMP, -1)

    assert "read the future" in str(excinfo.value)

    with pytest.raises(ValueError):
        ta.shift(RAMP, 1.5)


def test_a_cross_takes_a_level_or_a_second_series():
    fast = ta.sma(RAMP, 3)

    with pytest.raises(ValueError):
        ta.crossover(fast, np.arange(5.0))          # a series of the wrong length
    with pytest.raises(ValueError):
        ta.crossover(fast, np.nan)                  # a level that is not one
