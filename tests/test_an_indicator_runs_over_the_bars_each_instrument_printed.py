"""An indicator runs each instrument over the bars it printed. A series is (T,) or a
universe (T, N) along an axis, and every function returns the shape it was given, each
column the function of that column's own values: a value that is NaN or infinite is a
row the column does not have, NaN in the output there and False for a crossing, so a
window, a smoother, a lookback and a crossing carry across it rather than restarting,
and a series too short for its warm-up is all warm-up.
"""

import warnings

import numpy as np
import pytest

import probatum
from probatum import ta
from test_every_indicator_gives_one_value_a_bar import FLAGS, LINES, RAMP, ramp_bars
from test_nothing_reads_a_bar_that_has_not_closed import bars, universe

DAY = 86_400_000_000_000


@pytest.mark.parametrize("seed", range(4))
def test_each_column_is_the_function_of_its_instrument_s_own_bars(seed):
    # the universe lists late, misses a tenth of its bars and delists early, each
    # instrument at random, so every rule of the skip is met on the way; one more
    # lists late and then misses nothing, which goes to each function as a slice of
    # its column from its first value rather than through a mask
    given, _ = universe(seed)
    listed = np.full((len(given), 1, 5), np.nan)
    listed[37:, 0] = bars(np.random.default_rng(seed + 100), len(given) - 37)
    given = np.concatenate([given, listed], axis=1)

    high, low, close, volume = (given[:, :, field] for field in (1, 2, 3, 4))
    printed = ~np.isnan(close)

    for name, call in {**LINES, **FLAGS}.items():
        whole = call(high, low, close, volume)
        for k in range(close.shape[1]):
            rows  = printed[:, k]
            alone = call(high[rows, k], low[rows, k], close[rows, k], volume[rows, k])
            for line, (mine, own) in enumerate(zip(whole, alone)):
                assert mine.shape == close.shape, f"{name} line {line}"
                np.testing.assert_array_equal(mine[rows, k], own, err_msg=(
                    f"{name} line {line} of instrument {k} is not its own bars'"))
                skipped = mine[~rows, k]
                if mine.dtype == bool:
                    assert not skipped.any(), f"{name} fired where {k} printed nothing"
                else:
                    assert np.isnan(skipped).all(), f"{name} wrote where {k} had none"


def test_time_may_run_along_either_axis():
    given, _ = universe(1)
    close, high, low = given[:, :, 3], given[:, :, 1], given[:, :, 2]

    np.testing.assert_array_equal(ta.sma(close.T, 5, axis=1), ta.sma(close, 5).T)
    np.testing.assert_array_equal(ta.sma(close.T, 5, axis=-1), ta.sma(close, 5).T)

    lines = ta.bbands(close.T, 5, axis=1)
    np.testing.assert_array_equal(lines.upper, ta.bbands(close, 5).upper.T)

    np.testing.assert_array_equal(ta.atr(high.T, low.T, close.T, 5, axis=1),
                                  ta.atr(high, low, close, 5).T)
    np.testing.assert_array_equal(ta.crossover(close.T, 100.0, axis=1),
                                  ta.crossover(close, 100.0).T)


def test_a_frame_of_columns_is_a_column_each_and_one_column_keeps_its_axis():
    frame = np.column_stack([RAMP, RAMP + 1.0, RAMP - 1.0, RAMP, RAMP])
    out   = ta.sma(frame, 5)

    assert out.shape == (20, 5)
    for k in range(5):
        np.testing.assert_array_equal(out[:, k], ta.sma(frame[:, k], 5))

    column = ta.sma(RAMP.reshape(-1, 1), 5)

    assert column.shape == (20, 1)
    np.testing.assert_array_equal(column, ta.sma(RAMP, 5)[:, None])


def test_a_column_with_no_values_is_all_nan_beside_one_that_has_some():
    pair = np.column_stack([RAMP, np.full(20, np.nan)])
    out  = ta.ema(pair, 5)

    np.testing.assert_array_equal(out[:, 0], ta.ema(RAMP, 5))
    assert np.isnan(out[:, 1]).all()
    assert not ta.crossover(pair, 10.0)[:, 1].any()


def test_a_series_shorter_than_its_warm_up_is_all_warm_up():
    # every cut of a series gives the first rows of the whole, as the causality check
    # compares them, so a series too short for its length gives warm-up, where emsl
    # refused one longer than the series and an rsi as long as it
    assert ta.sma(RAMP, 50).shape == (20,) and np.isnan(ta.sma(RAMP, 50)).all()
    assert np.isnan(ta.rsi(RAMP, 20)).all() and np.isfinite(ta.rsi(RAMP, 19)[-1])

    swing = ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5, smooth=999)

    np.testing.assert_array_equal(swing.k, ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5).k)
    assert np.isnan(swing.d).all()

    # a length no series reaches sizes nothing: numpy holds no empty array of 2**63
    # columns, and wma would build and walk a weight for every bar of the length
    assert np.isnan(ta.sma(RAMP, 2**63)).all()
    assert np.isnan(ta.wma(RAMP, 10**6)).all()


def test_each_line_is_warm_up_on_as_many_values_as_its_first_value_needs():
    # the bar each line first gives a value on, as the inventory pins it: a column of
    # that many values is all warm-up, and one more gives a value on its last
    firsts = {
        "sma":        [4],
        "ema":        [4],
        "wma":        [4],
        "rsi":        [5],
        "roc":        [5],
        "stdev":      [4],
        "zscore":     [4],
        "atr":        [4],
        "vwap":       [4],
        "macd":       [5, 7, 7],
        "stoch":      [4, 6],
        "bbands":     [4, 4, 4],
        "donchian":   [4, 4, 4],
        "hma":        [6],
        "vwma":       [4],
        "willr":      [4],
        "cci":        [4],
        "natr":       [4],
        "keltner":    [4, 4, 4],
        "highest":    [4],
        "lowest":     [4],
        "shift":      [3],
        "change":     [3],
        "supertrend": [4, 4, 4, 4],
        "adx":        [9, 5, 5],
        "obv":        [1],
        "mfi":        [5],
    }

    for name, lines in firsts.items():
        for line, first in enumerate(lines):
            short = LINES[name](*(series[:first] for series in ramp_bars()))[line]
            assert np.isnan(short).all(), f"{name} line {line} gave a value early"
            enough = LINES[name](*(series[:first + 1] for series in ramp_bars()))[line]
            assert np.isfinite(enough[-1]), f"{name} line {line} never gave one"

    # a bar on its own is its own range, and a crossing needs two
    assert ta.true_range([10.0], [8.0], [9.0]).tolist() == [2.0]
    assert ta.crossover([1.0], 0.0).tolist() == [False]


def test_a_missing_bar_is_nan_in_every_output_and_carried_across():
    # the whole bar goes missing, which is what a hole in an OHLCV feed looks like
    high, low, close, volume = ramp_bars()
    for series in (high, low, close, volume):
        series[10] = np.nan

    for name, call in {**LINES, **FLAGS}.items():
        for index, line in enumerate(call(high, low, close, volume)):
            missing = line[10]
            assert (not missing) if line.dtype == bool else np.isnan(missing), (
                f"{name} line {index} wrote a value on the missing bar")

    # a lag counts the series' own bars, so the bar after the hole reads the one
    # before it, where emsl carried the hole three rows on
    lagged = ta.shift(close, 3)
    assert np.isnan(lagged[10]) and lagged[13] == 10.0 and lagged[14] == 12.0


def test_a_bar_after_a_missing_close_takes_its_range_from_the_close_before():
    # a gap through the missing bar widens the next one's range by what it moved,
    # which is the clean series' answer, where emsl gave NaN
    high  = np.array([100.0, 100.0, 130.0, 130.0])
    low   = np.array([99.0, 99.0, 129.0, 129.0])
    close = np.array([100.0, np.nan, 130.0, 130.0])
    out   = ta.true_range(high, low, close)

    assert np.isnan(out[1]) and out[2] == 30.0

    # and a bar missing its high alone is a bar the series does not have
    out = ta.true_range([10.0, np.nan, 12.0, 13.0], [8.0, 9.0, 10.0, 11.0],
                        [9.0, 10.0, 11.0, 12.0])
    np.testing.assert_array_equal(out, [2.0, np.nan, 3.0, 2.0])


def test_an_infinity_is_a_bar_the_series_does_not_have():
    for hurt in (np.inf, -np.inf):
        np.testing.assert_array_equal(ta.sma([1.0, 2.0, hurt, 4.0], 2),
                                      [np.nan, 1.5, np.nan, 3.0])


def test_nothing_warns_on_a_gappy_or_infinite_input():
    # a numpy RuntimeWarning is the module leaking its workings; a gap is an
    # answer and should arrive as one
    high, low, close, volume = ramp_bars()

    for hurt in (np.nan, np.inf, -np.inf):
        for name, call in LINES.items():
            damaged = close.copy()
            damaged[10] = hurt
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                call(high, low, damaged, volume)


def test_a_recursive_indicator_seeds_from_its_first_values_across_a_gap():
    # the seed is the mean of the first five values the series has, the hole at bar
    # 2 skipped, where emsl waited for five clean bars in a row
    holed = RAMP.copy()
    holed[2] = np.nan
    out = ta.ema(holed, 5)

    assert np.isnan(out[:5]).all()
    assert out[5] == pytest.approx((1.0 + 2.0 + 4.0 + 5.0 + 6.0) / 5.0)
    assert np.isfinite(ta.macd(holed, 3, 6, 3).signal[-1])
    assert np.isfinite(ta.rsi(holed, 5)[-1])
    assert np.isfinite(ta.atr(holed + 1.0, holed - 1.0, holed, 5)[-1])


def test_a_gap_after_the_seed_costs_only_its_own_bar():
    # the average carries across the hole on the next value, where emsl restarted
    # its warm-up and gave nothing for the next five bars
    long  = np.arange(1.0, 61.0)
    holed = long.copy()
    holed[30] = np.nan
    out   = ta.ema(holed, 5)
    alpha = 2.0 / 6.0

    assert np.isfinite(out[29]) and np.isnan(out[30])
    assert out[31] == pytest.approx(alpha * long[31] + (1.0 - alpha) * out[29],
                                    rel=1e-12)
    assert np.isfinite(out[31:]).all()


def test_a_cross_across_a_missing_value_fires_on_the_value_after_it():
    # 1 then 3 across the hole is a cross of 2, reported on the 3
    a = np.array([1.0, np.nan, 3.0, 4.0])
    assert ta.crossover(a, 2.0).tolist() == [False, False, True, False]
    assert ta.crossover(np.array([1.0, 1.5, 3.0, 4.0]), 2.0).tolist() == [
        False, False, True, False
    ]

    # a series compared with another skips the bars either lacks
    assert ta.crossover([1.0, 1.0, 3.0], [2.0, np.nan, 2.0]).tolist() == [
        False, False, True]


def test_obv_carries_its_total_across_a_missing_bar():
    close = np.array([10.0, 11.0, np.nan, 13.0, 14.0])
    out   = ta.obv(close, np.full(5, 100.0))

    np.testing.assert_array_equal(out, [np.nan, 100.0, np.nan, 200.0, 300.0])


def test_stoch_s_d_averages_across_a_window_with_no_range():
    # two flat bars give k no position, and d averages over the bars k has, where
    # emsl's d stayed NaN through every window that held one
    closes = np.array([9.0, 10.0, 10.0, 11.0, 13.0])
    given  = np.column_stack([closes, [10.0, 10.0, 10.0, 12.0, 13.0],
                              [8.0, 10.0, 10.0, 10.0, 11.0], closes, np.ones(5)])

    data  = probatum.Data(given, time=np.arange(5) * 3_600_000_000_000, interval="1h")
    swing = ta.stoch(data.high, data.low, data.close, 2, 2)

    np.testing.assert_array_equal(swing.k, [np.nan, 100.0, np.nan, 50.0, 100.0])
    np.testing.assert_array_equal(swing.d, [np.nan, np.nan, np.nan, 75.0, 75.0])


def test_a_stock_s_three_bar_average_spans_the_weekend_beside_a_coin_that_trades_it():
    # Thursday to Monday: the coin prints every day and the stock skips Saturday and
    # Sunday, so its average on Monday is Thursday's, Friday's and Monday's
    coin   = np.arange(10.0, 15.0)
    stock  = np.array([100.0, 103.0, np.nan, np.nan, 109.0])
    closes = np.column_stack([coin, stock])
    given  = np.stack([closes, closes, closes, closes, np.where(np.isnan(closes),
                                                                 np.nan, 1.0)], axis=2)

    data = probatum.Data(
        given,
        time     = np.arange(5) * DAY,
        symbol   = ["COIN", "SPY"],
        interval = "1d",
    )
    average = ta.sma(data.closes, 3)

    np.testing.assert_array_equal(average[:, 0], [np.nan, np.nan, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(average[:, 1], [np.nan] * 4 + [104.0])
