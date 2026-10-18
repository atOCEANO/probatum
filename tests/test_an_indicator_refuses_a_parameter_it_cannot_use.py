"""An indicator refuses a parameter it cannot use and says why, naming the argument as
it was typed: a length that is not a whole number of bars from 1, a band of negative
width, a fast average no faster than the slow one, a shift into the future, a level that
is not one and an axis the array lacks, each a ProbatumValueError, or a
ProbatumTypeError where it is not a number at all, or an axis not a whole number; and
it refuses as a DataError a series it cannot run on, one of no bars, a single value,
an array of more than two axes and series of different shapes.
"""

import math
import re

import numpy as np
import pytest

from probatum import DataError, ProbatumTypeError, ProbatumValueError, ta
from test_every_indicator_gives_one_value_a_bar import RAMP

HIGH, LOW = RAMP + 1.0, RAMP - 1.0

CASES = [
    # a fractional length is a mistake, not a request to round one way or the
    # other, and silently flooring it answered a question nobody asked
    (lambda: ta.sma(RAMP, 3.9), ProbatumValueError,
     "length: 3.9; give a whole number of bars from 1"),
    (lambda: ta.sma(RAMP, 0), ProbatumValueError,
     "length: 0; give a whole number of bars from 1"),
    (lambda: ta.sma(RAMP, math.inf), ProbatumValueError, "length: inf; give a whole"),
    (lambda: ta.sma(RAMP, "5"), ProbatumTypeError, "length: '5'; give a whole number"),
    (lambda: ta.sma(RAMP, True), ProbatumTypeError, "length: True; give a whole"),
    (lambda: ta.sma(RAMP, None), ProbatumTypeError, "length: None; give a whole"),
    (lambda: ta.stoch(HIGH, LOW, RAMP, 5, smooth=0), ProbatumValueError,
     "smooth: 0; give a whole number of bars from 1"),
    (lambda: ta.macd(RAMP, 12, 26, signal=2.5), ProbatumValueError, "signal: 2.5;"),
    (lambda: ta.macd(RAMP, 26, 12), ProbatumValueError,
     "slow: 12, not above fast's 26; give fast below slow"),
    (lambda: ta.keltner(HIGH, LOW, RAMP, atr_length=0), ProbatumValueError,
     "atr_length: 0;"),
    (lambda: ta.bbands(RAMP, 5, deviations=-2.0), ProbatumValueError,
     "deviations: -2.0; give a width of at least 0"),
    (lambda: ta.bbands(RAMP, 5, deviations="2"), ProbatumTypeError,
     "deviations: '2'; give a width of at least 0"),
    (lambda: ta.keltner(HIGH, LOW, RAMP, multiplier=-1.0), ProbatumValueError,
     "multiplier: -1.0; give a width of at least 0"),
    (lambda: ta.supertrend(HIGH, LOW, RAMP, multiplier=0.0), ProbatumValueError,
     "multiplier: 0.0; give a width above 0"),
    (lambda: ta.shift(RAMP, -1), ProbatumValueError,
     "bars: -1, which would read the future; give a whole number of bars from 0"),
    (lambda: ta.shift(RAMP, 1.5), ProbatumValueError,
     "bars: 1.5; give a whole number of bars from 0"),
    (lambda: ta.change(RAMP, -1), ProbatumValueError,
     "bars: -1, which would read the future"),
    (lambda: ta.crossover(RAMP, np.nan), ProbatumValueError,
     "other: nan; give a finite level, or a series"),
    (lambda: ta.crossover(RAMP, "2"), ProbatumTypeError,
     "other: '2'; give a level, or a series"),
    (lambda: ta.sma(RAMP, 5, axis=1), ProbatumValueError,
     "axis: 1 for an array of 1 dimension; give 0"),
    (lambda: ta.sma(RAMP, 5, axis=1.0), ProbatumTypeError,
     "axis: 1.0; give the time axis as a whole number"),
    (lambda: ta.sma(np.array([]), 1), DataError,
     "values: no bars; give a series that has some"),
    (lambda: ta.sma(5.0, 1), DataError, "values: 5.0; give a series of bars"),
    # the value as it came, not the float numpy made of it
    (lambda: ta.sma(None, 1), DataError, "values: None; give a series of bars"),
    (lambda: ta.sma(np.zeros((20, 3, 5)), 5), DataError,
     "values: an array of shape (20, 3, 5); give a (T,) series or a (T, N) array"),
    (lambda: ta.stoch(RAMP, RAMP[:5], RAMP, 3), DataError,
     "low: shape (5,) beside high's (20,); give series of one shape"),
    (lambda: ta.crossover(ta.sma(RAMP, 3), np.arange(5.0)), DataError,
     "other: shape (5,) beside values' (20,)"),
    # a column with a trailing axis is not the series it came from
    (lambda: ta.crossover(RAMP.reshape(-1, 1), RAMP), DataError,
     "other: shape (20,) beside values' (20, 1)"),
]


@pytest.mark.parametrize("call, error, says", CASES)
def test_what_an_indicator_cannot_use_is_refused_naming_the_argument(call, error, says):
    with pytest.raises(error, match=re.escape(says)) as caught:
        call()

    # a parameter's wrong value is the root's class, and a series' a DataError
    assert caught.type is error
