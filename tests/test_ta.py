"""probatum.ta over a missing bar: a gap in the input stays a gap in every output,
costs a recursive indicator its warm-up rather than the rest of the series, and raises
nothing and warns of nothing.
"""

import warnings

import numpy as np
import pytest

from probatum import ta
from test_every_indicator_gives_one_value_a_bar import LINES, RAMP, ramp_bars


def test_a_gap_in_the_input_stays_a_gap_in_every_output():
    # the claim is made module-wide, so it is checked module-wide: no function may
    # absorb a missing bar, and none may raise over one either
    # the whole bar goes missing, which is what a gap in an OHLCV feed actually
    # looks like. Holing only the close leaves donchian untouched, correctly, since
    # it reads the high and the low and never the close
    high, low, close, volume = ramp_bars()
    for series in (high, low, close, volume):
        series[10] = np.nan

    for name, call in LINES.items():
        if name == "shift":
            # a lag MOVES its values, so it moves the gap with them. Checked
            # below rather than exempted, since a shift that lost the gap would
            # be the same defect as one that bridged it
            continue

        for index, line in enumerate(call(high, low, close, volume)):
            assert np.isnan(line[10]), (
                f"{name} line {index} passed straight through the gap"
            )

    lagged = ta.shift(close, 3)
    assert np.isnan(lagged[13]), "the shift lost the gap it was carrying"
    assert np.isfinite(lagged[10]), "the shift held the gap where it was not"


def test_a_missing_close_makes_the_true_range_a_gap_not_the_bars_own_range():
    # taking the widest span while ignoring the missing ones quietly returned high
    # less low, so a gapping bar reported a fraction of its real range and every
    # ATR-sized stop built on it came out far too tight
    high  = np.array([100.0, 100.0, 130.0, 130.0])
    low   = np.array([99.0, 99.0, 129.0, 129.0])
    close = np.array([100.0, np.nan, 130.0, 130.0])
    out   = ta.true_range(high, low, close)

    assert np.isnan(out[2]), f"the gap was swallowed and reported {out[2]}"

    clean = ta.true_range(high, low, np.array([100.0, 100.0, 130.0, 130.0]))

    assert clean[2] == pytest.approx(30.0)   # 130 - 100, the real range


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


def test_a_recursive_indicator_waits_out_a_gap_rather_than_raising():
    # ema, rsi, atr and macd all seed from a window, and a single missing bar in
    # the warm-up used to raise, from `ema`, about a bar the caller of `macd` had
    # never heard of. It costs warm-up now, as it does everywhere else
    holed = RAMP.copy()
    holed[2] = np.nan

    out = ta.ema(holed, 5)

    assert np.isnan(out[:7]).all()      # no clean window of 5 ends before bar 7
    assert out[7] == pytest.approx(RAMP[3:8].mean())
    assert np.isfinite(ta.macd(holed, 3, 6, 3).signal[-1])
    assert np.isfinite(ta.rsi(holed, 5)[-1])
    assert np.isfinite(ta.atr(holed + 1.0, holed - 1.0, holed, 5)[-1])


def test_a_gap_after_the_seed_costs_warm_up_rather_than_the_rest_of_the_series():
    # a recurrence cannot carry across a missing bar, and the strict reading is
    # that everything after one is undefined. That reading makes a year of candles
    # with a single hole produce nothing from the hole onward, which is useless
    # rather than rigorous: the next clean window seeds a fresh run instead
    long  = np.arange(1.0, 61.0)
    holed = long.copy()
    holed[30] = np.nan

    out = ta.ema(holed, 5)

    assert np.isfinite(out[29])         # fine before the hole
    assert np.isnan(out[30])            # the hole itself
    assert np.isnan(out[31:35]).all()   # and while it looks for a clean window
    # bars 31 to 35 are the first five clean ones after it, so the seed lands there
    assert out[35] == pytest.approx(long[31:36].mean())
    assert np.isfinite(out[-1]), "one missing bar killed the rest of the series"

    for call in (lambda v: ta.rsi(v, 5),
                 lambda v: ta.atr(v + 1.0, v - 1.0, v, 5),
                 lambda v: ta.macd(v, 3, 6, 3).signal):
        assert np.isfinite(call(holed)[-1])


def test_a_cross_against_a_missing_value_is_not_a_cross():
    a = np.array([1.0, np.nan, 3.0, 4.0])

    assert ta.crossover(a, 2.0).tolist() == [False, False, False, False]
    assert ta.crossover(np.array([1.0, 1.5, 3.0, 4.0]), 2.0).tolist() == [
        False, False, True, False
    ]


def test_obv_restarts_after_a_gap_rather_than_absorbing_it():
    close  = np.array([10.0, 11.0, np.nan, 13.0, 14.0])
    volume = np.full(5, 100.0)
    out    = ta.obv(close, volume)

    assert np.isnan(out[2]) and np.isnan(out[3])         # the gap and its successor
    assert out[4] == pytest.approx(100.0)                # a fresh run, starting flat
