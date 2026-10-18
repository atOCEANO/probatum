"""Each indicator holds the convention its docstring states, checked against a value
worked by hand rather than against another library, which would only prove the two
agree: how an average is seeded and smoothed, what a window with nothing to say gives,
how far a lookback reaches, and when a crossing fires.
"""

import math

import numpy as np
import pytest

from probatum import ta
from test_every_indicator_gives_one_value_a_bar import RAMP, ramp_bars


def test_sma_is_the_mean_of_the_window():
    out = ta.sma(RAMP, 5)

    assert np.isnan(out[3])
    assert out[4]  == pytest.approx(3.0)     # (1+2+3+4+5)/5
    assert out[19] == pytest.approx(18.0)    # (16+17+18+19+20)/5


def test_ema_seeds_from_the_simple_average_not_the_first_value():
    # seeded at bar 4 with (1+2+3+4+5)/5 = 3, then alpha = 2/6 = 1/3
    out = ta.ema(RAMP, 5)

    assert np.isnan(out[3])
    assert out[4] == pytest.approx(3.0)
    assert out[5] == pytest.approx(3.0 + (6.0 - 3.0) / 3.0)          # 4.0
    assert out[6] == pytest.approx(4.0 + (7.0 - 4.0) / 3.0)          # 5.0


def test_wma_leans_on_the_newest_bar():
    # weights 1..5 over values 1..5: (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15
    out = ta.wma(RAMP, 5)

    assert out[4] == pytest.approx(55.0 / 15.0)
    assert out[10] > ta.sma(RAMP, 5)[10]


def test_rsi_is_a_hundred_when_nothing_ever_falls():
    out = ta.rsi(RAMP, 5)

    assert out[-1] == pytest.approx(100.0)
    assert ta.rsi(RAMP[::-1].copy(), 5)[-1] == pytest.approx(0.0)


def test_rsi_uses_wilders_smoothing():
    # up 1 every bar except one fall of 5 at bar 10, checked against Wilder by hand
    values = RAMP.copy()
    values[10:] -= 6.0
    out = ta.rsi(values, 5)

    # seeded at bar 5 from the first five changes, all +1, so gain 1 loss 0
    assert out[5] == pytest.approx(100.0)

    # the fall of 5 enters at bar 10: gain 1*(4/5) = 0.8, loss 5/5 = 1.0
    gain, loss = 1.0 * 0.8, 5.0 / 5.0
    assert out[10] == pytest.approx(100.0 - 100.0 / (1.0 + gain / loss))


def test_true_range_takes_the_widest_of_the_three_spans():
    high  = np.array([10.0, 12.0, 20.0])
    low   = np.array([8.0, 11.0, 19.0])
    close = np.array([9.0, 11.5, 19.5])
    out   = ta.true_range(high, low, close)

    assert out[0] == pytest.approx(2.0)   # 10 - 8, no previous close
    assert out[1] == pytest.approx(3.0)   # |12 - 9| beats 12 - 11
    assert out[2] == pytest.approx(8.5)   # |20 - 11.5| beats 20 - 19


def test_stdev_is_population_so_bbands_match_every_chart():
    # 1..5 has a population deviation of sqrt(2), a sample one of sqrt(2.5)
    assert ta.stdev(RAMP, 5)[4] == pytest.approx(math.sqrt(2.0))

    bands = ta.bbands(RAMP, 5, deviations=2.0)

    assert bands.middle[4] == pytest.approx(3.0)
    assert bands.upper[4]  == pytest.approx(3.0 + 2.0 * math.sqrt(2.0))
    assert bands.lower[4]  == pytest.approx(3.0 - 2.0 * math.sqrt(2.0))


def test_roc_is_a_percentage_of_the_bar_it_looks_back_to():
    out = ta.roc(RAMP, 5)

    assert np.isnan(out[4])
    assert out[5] == pytest.approx((6.0 / 1.0 - 1.0) * 100.0)


def test_donchian_brackets_the_window_and_sits_between():
    channel = ta.donchian(RAMP + 1.0, RAMP - 1.0, 5)

    assert channel.upper[4]  == pytest.approx(6.0)   # highest high of 2..6
    assert channel.lower[4]  == pytest.approx(0.0)   # lowest low of 0..4
    assert channel.middle[4] == pytest.approx(3.0)


def test_stoch_is_where_the_close_sits_in_the_range():
    swing = ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5, 3)

    # close 5, range 0 to 6, so (5 - 0) / 6
    assert swing.k[4] == pytest.approx(5.0 / 6.0 * 100.0)
    assert 0.0 <= swing.k[-1] <= 100.0


def test_vwap_weights_by_volume_and_prices_on_the_typical():
    # the fixture that used to stand here had high, low and close spaced so that
    # (h + l + c) / 3, the close alone and the midpoint were all the SAME array,
    # so it could not rule out the two things its comment said it ruled out.
    # These are chosen so all three differ on every bar
    high     = np.array([12.0, 30.0, 34.0, 60.0])
    low      = np.array([6.0, 16.0, 26.0, 36.0])
    close    = np.array([11.0, 17.0, 33.0, 37.0])
    typical  = (high + low + close) / 3.0
    midpoint = (high + low) / 2.0

    assert not np.allclose(typical, close) and not np.allclose(typical, midpoint)

    volume   = np.array([1.0, 1.0, 1.0, 97.0])
    out      = ta.vwap(high, low, close, volume, 4)
    expected = float((typical * volume).sum() / volume.sum())

    assert out[3] == pytest.approx(expected)
    # and it is neither of the two implementations the typical price is not
    assert out[3] != pytest.approx(float((close * volume).sum() / volume.sum()))
    assert out[3] != pytest.approx(float((midpoint * volume).sum() / volume.sum()))
    assert np.isnan(ta.vwap(high, low, close, np.zeros(4), 4)[3])


def test_stoch_smooths_k_into_d():
    high  = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    low   = np.full(6, 0.0)
    close = np.array([5.0, 6.0, 3.0, 9.0, 7.0, 12.0])
    swing = ta.stoch(high, low, close, 2, 2)

    # bar 1: window high 11, low 0, close 6 -> 54.5454...
    assert swing.k[1] == pytest.approx(6.0 / 11.0 * 100.0)
    # bar 2: window high 12, low 0, close 3 -> 25.0
    assert swing.k[2] == pytest.approx(25.0)
    assert swing.d[2] == pytest.approx((6.0 / 11.0 * 100.0 + 25.0) / 2.0)


def test_zscore_has_no_position_in_a_window_with_no_deviation():
    flat = np.full(20, 7.0)

    assert np.isnan(ta.zscore(flat, 5)[-1])
    assert ta.zscore(RAMP, 5)[-1] == pytest.approx(
        (20.0 - 18.0) / math.sqrt(2.0)
    )


def test_macd_is_the_gap_between_two_averages_and_its_own_average():
    # worked by hand rather than recomputed with the code under test, which would
    # pass for any ema at all, right or wrong. On the ramp, both averages sit
    # exactly on their own window's mean once warm, so the line is the distance
    # between the two window centres: (6-1)/2 - (3-1)/2 = 1.5
    lines = ta.macd(RAMP, 3, 6, 3)

    assert np.isnan(lines.line[4])            # the slow leg needs 6 bars
    assert lines.line[5]  == pytest.approx(1.5)
    assert lines.line[19] == pytest.approx(1.5)
    # the signal is an average of a line that is constant at 1.5, so it is too
    assert lines.signal[19]    == pytest.approx(1.5)
    assert lines.histogram[19] == pytest.approx(0.0)

    first_line = int(np.flatnonzero(np.isfinite(lines.line))[0])

    assert np.isnan(lines.signal[first_line])
    assert first_line == 5 and np.isfinite(lines.signal[first_line + 2])


def test_a_market_going_nowhere_has_no_relative_strength():
    # neither gains nor losses is not "maximally overbought": reporting 100 there
    # fires every `rsi > 70` rule on a market that has not moved
    assert np.isnan(ta.rsi(np.full(20, 7.0), 5)[-1])
    assert np.isnan(ta.rsi(np.zeros(20), 5)[-1])
    assert ta.rsi(RAMP, 5)[-1] == pytest.approx(100.0)


def test_shift_pads_rather_than_wrapping():
    # the whole reason it exists: values[i - n] on an early bar is a negative
    # index, which numpy resolves from the END of the array (ADR 0037)
    out = ta.shift(RAMP, 3)

    assert np.isnan(out[:3]).all()
    assert np.array_equal(out[3:], RAMP[:-3])
    # written by hand this reads the last bar of the series instead
    assert RAMP[0 - 3] == RAMP[17] == 18.0
    assert ta.shift(RAMP, 0).tolist() == RAMP.tolist()
    assert np.isnan(ta.shift(RAMP, 20)).all()


def test_change_is_the_move_over_the_lookback():
    out = ta.change(RAMP, 4)

    assert np.isnan(out[:4]).all()
    assert out[10] == pytest.approx(4.0)          # a ramp of one per bar


def test_highest_and_lowest_include_the_bar_they_are_read_on():
    # the same rule donchian states, so a breakout of `highest` cannot happen on
    # the bar that set it
    high = ta.highest(RAMP, 5)
    low  = ta.lowest(RAMP, 5)

    assert np.isnan(high[:4]).all() and np.isnan(low[:4]).all()
    assert high[4] == 5.0 and low[4] == 1.0
    assert high[19] == 20.0 and low[19] == 16.0


def test_a_cross_fires_on_the_bar_it_happens_and_only_then():
    values = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    up     = ta.crossover(values, 2.0)
    down   = ta.crossunder(values, 2.0)

    assert up.tolist()   == [False, False, True, False, False, False, True]
    assert down.tolist() == [False, False, False, False, True, False, False]
    # touching the level and coming back is a cross; touching and staying is not
    assert ta.crossover(np.array([1.0, 2.0, 2.0]), 2.0).tolist() == [False, False, False]


def jagged(n=60, seed=5):
    # a varying true range, because on a straight ramp every smoothing constant
    # converges to the same number and a test on one cannot tell them apart
    rng    = np.random.default_rng(seed)
    close  = 100.0 + np.cumsum(rng.normal(0.0, 1.5, n))
    spread = rng.uniform(0.2, 4.0, n)
    return close + spread, close - spread, close


def test_atr_smooths_at_wilders_rate_and_not_the_exponential_one():
    # the difference is 1/n against 2/(n+1), which on a jagged series is a
    # different number every bar and on a smooth one is the same number
    high, low, close = jagged()

    ranges = ta.true_range(high, low, close)
    out    = ta.atr(high, low, close, 5)
    seed   = float(ranges[:5].mean())

    def rolled(alpha):
        running = seed
        for i in range(5, len(close)):
            running = alpha * ranges[i] + (1.0 - alpha) * running

        return running

    assert out[-1] == pytest.approx(rolled(1.0 / 5.0), rel=1e-12)
    assert out[-1] != pytest.approx(rolled(2.0 / 6.0), rel=1e-6)
    assert out[4] == pytest.approx(seed)          # the seed is the window mean


def test_rsi_smooths_at_wilders_rate_too():
    _h, _l, close = jagged()

    out    = ta.rsi(close, 5)
    change = np.diff(close, prepend=np.nan)
    gains, losses = np.maximum(change, 0.0), -np.minimum(change, 0.0)

    def rolled(values, alpha):
        running = float(values[1:6].mean())
        for i in range(6, len(close)):
            running = alpha * values[i] + (1.0 - alpha) * running

        return running

    up, down = rolled(gains, 1.0 / 5.0), rolled(losses, 1.0 / 5.0)

    assert out[-1] == pytest.approx(100.0 - 100.0 / (1.0 + up / down), rel=1e-12)


def test_willr_is_the_stochastic_read_from_the_top():
    rng   = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 80))
    high, low = close + 1.0, close - 1.0

    percent_r = ta.willr(high, low, close, 14)
    raw_k     = ta.stoch(high, low, close, 14, 1).k
    both      = np.isfinite(percent_r) & np.isfinite(raw_k)

    assert both.any()
    assert np.allclose(percent_r[both], raw_k[both] - 100.0)
    assert (percent_r[both] <= 0.0).all() and (percent_r[both] >= -100.0).all()


def test_cci_divides_by_the_mean_absolute_deviation_not_the_standard_one():
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    high = low = close = values

    out      = ta.cci(high, low, close, 5)
    window   = values                                   # typical == values here
    mad      = float(np.abs(window - window.mean()).mean())
    expected = (values[-1] - window.mean()) / (0.015 * mad)

    assert out[4] == pytest.approx(expected)
    assert mad != pytest.approx(float(window.std()))    # the two really differ


def test_natr_is_the_range_as_a_share_of_price():
    high, low, close, _v = ramp_bars()

    ranges = ta.atr(high, low, close, 5)
    out    = ta.natr(high, low, close, 5)
    both   = np.isfinite(out)

    assert np.allclose(out[both], ranges[both] / close[both] * 100.0)


def test_keltner_bands_the_bars_where_bbands_bands_the_closes():
    # a session of wide ranges closing in one place widens keltner and not bbands,
    # which is the reason both exist
    close = np.full(60, 100.0)
    high  = close + np.where(np.arange(60) >= 30, 10.0, 1.0)
    low   = close - np.where(np.arange(60) >= 30, 10.0, 1.0)
    belt  = ta.keltner(high, low, close, 10, 10, 2.0)
    bands = ta.bbands(close, 10, 2.0)

    assert belt.upper[-1] - belt.lower[-1] > belt.upper[25] - belt.lower[25]
    assert bands.upper[-1] - bands.lower[-1] == pytest.approx(0.0)
    assert belt.middle[-1] == pytest.approx(ta.ema(close, 10)[-1])


def test_adx_reads_a_clean_trend_as_a_trend_and_a_range_as_none():
    trending = np.arange(1.0, 81.0)
    strong   = ta.adx(trending + 1.0, trending - 1.0, trending, 14)

    assert strong.adx[-1] > 90.0
    assert strong.plus[-1] > strong.minus[-1]

    chop = 100.0 + np.tile([0.0, 1.0], 40)
    weak = ta.adx(chop + 1.0, chop - 1.0, chop, 14)

    assert weak.adx[-1] < strong.adx[-1]

    finite = np.isfinite(strong.adx)

    # a perfect one-way ramp reaches the ceiling exactly, so allow the last ulp
    assert (strong.adx[finite] >= 0.0).all()
    assert (strong.adx[finite] <= 100.0 + 1e-9).all()


def test_a_falling_market_puts_the_strength_on_the_other_side():
    falling = np.arange(80.0, 0.0, -1.0)
    trend   = ta.adx(falling + 1.0, falling - 1.0, falling, 14)

    assert trend.minus[-1] > trend.plus[-1]


def test_supertrend_flips_when_price_closes_through_the_far_rail():
    up    = np.arange(1.0, 61.0)
    close = np.concatenate([up, up[-1] - np.arange(1.0, 41.0) * 3.0])
    high, low = close + 1.0, close - 1.0

    rails  = ta.supertrend(high, low, close, 10, 3.0)
    finite = np.isfinite(rails.direction)

    assert set(np.unique(rails.direction[finite])) <= {1.0, -1.0}
    assert rails.direction[59] == 1.0                    # still long at the top
    assert rails.direction[-1] == -1.0                   # short by the end

    # the line in force is the near rail, never the far one
    long_bars = finite & (rails.direction == 1.0)

    assert np.allclose(rails.middle[long_bars], rails.lower[long_bars])


def test_obv_signs_the_volume_by_the_direction_of_the_close():
    close  = np.array([10.0, 11.0, 11.0, 9.0, 12.0])
    volume = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    out    = ta.obv(close, volume)

    assert np.isnan(out[0])                              # no previous close
    assert out[1] == pytest.approx(200.0)                # up, add
    assert out[2] == pytest.approx(200.0)                # unchanged, hold
    assert out[3] == pytest.approx(-200.0)               # down, subtract
    assert out[4] == pytest.approx(300.0)                # up, add


def test_mfi_weights_the_move_by_what_traded():
    rng   = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 60))
    high, low = close + 1.0, close - 1.0

    quiet  = ta.mfi(high, low, close, np.full(60, 1.0), 14)
    loud   = ta.mfi(high, low, close, np.where(close > 100.0, 1000.0, 1.0), 14)
    finite = np.isfinite(quiet) & np.isfinite(loud)

    assert finite.any()
    assert not np.allclose(quiet[finite], loud[finite])
    assert (loud[finite] >= 0.0).all() and (loud[finite] <= 100.0).all()

    # a market that only rises is all inflow
    rising = np.arange(1.0, 41.0)

    assert ta.mfi(rising + 1.0, rising - 1.0, rising, np.full(40, 10.0), 14)[-1] == 100.0


def test_hma_turns_faster_than_the_average_it_is_built_from():
    close = np.concatenate([np.full(40, 100.0), np.full(40, 120.0)])
    hull  = ta.hma(close, 16)
    plain = ta.sma(close, 16)

    # ten bars after the step, the hull is nearer the new level
    assert abs(hull[50] - 120.0) < abs(plain[50] - 120.0)
    assert np.isnan(hull[0])


def test_vwma_leans_on_the_bars_that_traded():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    heavy  = ta.vwma(values, np.array([1.0, 1.0, 1.0, 97.0]), 4)

    assert heavy[3] == pytest.approx((10.0 + 20.0 + 30.0 + 40.0 * 97.0) / 100.0)
    assert heavy[3] != pytest.approx(float(values.mean()))
    assert np.isnan(ta.vwma(values, np.zeros(4), 4)[3])
