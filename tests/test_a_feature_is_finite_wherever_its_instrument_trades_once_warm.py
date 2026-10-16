"""A feature is the user's (T, N, F) array, or a function of a Data that returns one.
It must be finite wherever its instrument printed a bar once its warm-up is past,
counted from the first bar the feature was given, and a value that is not is refused,
naming the bar, the instrument and the feature, unless the caller passes a finite
number or "previous" to fill it with.
"""

import re

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum._data import features


def frame(close, rows=range(6)):
    closes = close + np.arange(6, dtype=np.float64)
    index  = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    return pd.DataFrame({
        "open":   closes,
        "high":   closes + 1.0,
        "low":    closes - 1.0,
        "close":  closes,
        "volume": 1_000.0,
    }, index=index).iloc[rows]


def universe():
    return probatum.Data({"BTCUSDT": frame(100.0), "ETHUSDT": frame(10.0, range(2, 6))})


def values():
    out = np.ones((6, 2, 2))
    out[:2, 0, 0] = np.nan          # BTCUSDT's first feature warms up over two bars
    out[:3, 1, :] = np.nan          # ETHUSDT prints from row 2, warming up one bar more
    return out


def test_a_warm_up_and_a_bar_with_no_print_pass_as_they_are():
    out = features(universe(), values())

    assert np.isnan(out[:2, 0, 0]).all() and np.isnan(out[:3, 1, :]).all()
    assert not out.flags.writeable


def test_a_value_lost_after_the_warm_up_is_refused_naming_where():
    lost = values()
    lost[4, 1, 1] = np.nan

    with pytest.raises(ValueError, match=re.escape(
            "features: NaN at bar 4 (2024-01-01 04:00) for ETHUSDT, feature 1; "
            "pass fill=")):
        features(universe(), lost)

    lost[4, 1, 1] = np.inf
    with pytest.raises(ValueError, match=re.escape("features: inf at bar 4")):
        features(universe(), lost)


def test_fill_puts_a_number_or_the_value_before_it_in_its_place():
    lost = values()
    lost[3, 0, 1] = 7.0
    lost[4, 0, 1] = np.nan

    assert features(universe(), lost, fill=0.0)[4, 0, 1]         == 0.0
    assert features(universe(), lost, fill=np.int64(2))[4, 0, 1] == 2.0
    assert features(universe(), lost, fill="previous")[4, 0, 1]  == 7.0


def test_a_bar_an_instrument_did_not_print_is_passed_over_and_not_filled_from():
    # ETHUSDT prints no bar at 03:00, so its value there is never read, and the value
    # lost at 04:00 is filled from 02:00, its last bar
    data = probatum.Data({
        "BTCUSDT": frame(100.0),
        "ETHUSDT": frame(10.0, [0, 1, 2, 4, 5]),
    })
    lost = np.ones((6, 2, 1))
    lost[:, 1, 0] = [10.0, 11.0, 12.0, np.nan, np.nan, 15.0]

    with pytest.raises(ValueError, match=re.escape("features: NaN at bar 4")):
        features(data, lost)

    lost[4, 1, 0] = 14.0

    assert np.isnan(features(data, lost)[3, 1, 0])

    lost[4, 1, 0] = np.nan
    filled = features(data, lost, fill="previous")

    assert filled[4, 1, 0] == 12.0 and np.isnan(filled[3, 1, 0])


def counted(history):
    # each bar's position in the bars the function was given, lost at 00:00, which
    # warms the feature up, and at 03:00
    out = np.arange(len(history), dtype=np.float64)[:, None, None] * np.ones((1, 2, 1))
    out[[0, 3]] = np.nan
    return out


def test_a_function_is_judged_from_the_first_bar_it_was_given():
    with pytest.raises(ValueError, match=re.escape(
            "features: NaN at bar 0 (2024-01-01 03:00) for BTCUSDT, feature 0; "
            "pass fill=")):
        features(universe()[3:], counted)

    np.testing.assert_array_equal(features(universe()[3:], counted, fill="previous"),
                                  features(universe(), counted, fill="previous")[3:])
    assert features(universe()[4:], counted)[:, 0, 0].tolist() == [4.0, 5.0]


@pytest.mark.parametrize("given, fill, says", [
    (np.ones((6, 2)), None, "features: shape (6, 2) for 6 bars of 2 instruments; give "
                            "(6, 2, F)"),
    (lambda data: np.ones((2, 2, 1)), None,
     "features: shape (2, 2, 1) for 4 bars of 2 instruments; give (4, 2, F)"),
    (np.ones((6, 2, 1)), "last", "fill: 'last'; give a finite number or \"previous\""),
    (np.ones((6, 2, 1)), True, "fill: True; give a finite number or \"previous\""),
    (np.ones((6, 2, 1)), float("nan"),
     "fill: nan; give a finite number or \"previous\""),
])
def test_a_feature_of_the_wrong_shape_or_a_fill_it_cannot_use_is_refused(given, fill,
                                                                          says):
    with pytest.raises(ValueError, match=re.escape(says)):
        features(universe()[2:4] if callable(given) else universe(), given, fill=fill)
