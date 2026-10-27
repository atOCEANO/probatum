"""Every one of probatum.ta's thirty indicators gives one value a bar, aligned so entry
i is bar i, with the warm-up as NaN and never a shorter array, so the array a rule
trades on goes onto the chart with no padding: a function of several lines returns
them by name, and the two crossings return booleans, False through the warm-up.
"""

import numpy as np
import pytest

import probatum
from probatum import ta

# 1, 2, 3, ... 20: every rolling answer over it can be worked out on paper
RAMP = np.arange(1.0, 21.0)


# every function, as one call over (high, low, close, volume). Multi-line results
# are split so each line is checked in its own right
LINES = {
    "sma": lambda h, l, c, v: [ta.sma(c, 5)],
    "ema": lambda h, l, c, v: [ta.ema(c, 5)],
    "wma": lambda h, l, c, v: [ta.wma(c, 5)],
    "rsi": lambda h, l, c, v: [ta.rsi(c, 5)],
    "roc": lambda h, l, c, v: [ta.roc(c, 5)],
    "stdev": lambda h, l, c, v: [ta.stdev(c, 5)],
    "zscore": lambda h, l, c, v: [ta.zscore(c, 5)],
    "atr": lambda h, l, c, v: [ta.atr(h, l, c, 5)],
    "true_range": lambda h, l, c, v: [ta.true_range(h, l, c)],
    "vwap": lambda h, l, c, v: [ta.vwap(h, l, c, v, 5)],
    "macd": lambda h, l, c, v: list(vars_of(ta.macd(c, 3, 6, 3))),
    "stoch": lambda h, l, c, v: list(vars_of(ta.stoch(h, l, c, 5, 3))),
    "bbands": lambda h, l, c, v: list(vars_of(ta.bbands(c, 5))),
    "donchian": lambda h, l, c, v: list(vars_of(ta.donchian(h, l, 5))),
    "hma": lambda h, l, c, v: [ta.hma(c, 6)],
    "vwma": lambda h, l, c, v: [ta.vwma(c, v, 5)],
    "willr": lambda h, l, c, v: [ta.willr(h, l, c, 5)],
    "cci": lambda h, l, c, v: [ta.cci(h, l, c, 5)],
    "natr": lambda h, l, c, v: [ta.natr(h, l, c, 5)],
    "keltner": lambda h, l, c, v: list(vars_of(ta.keltner(h, l, c, 5, 5))),
    "highest": lambda h, l, c, v: [ta.highest(c, 5)],
    "lowest": lambda h, l, c, v: [ta.lowest(c, 5)],
    "shift": lambda h, l, c, v: [ta.shift(c, 3)],
    "change": lambda h, l, c, v: [ta.change(c, 3)],
    "supertrend": lambda h, l, c, v: list(vars_of(ta.supertrend(h, l, c, 5))),
    "adx": lambda h, l, c, v: list(vars_of(ta.adx(h, l, c, 5))),
    "obv": lambda h, l, c, v: [ta.obv(c, v)],
    "mfi": lambda h, l, c, v: [ta.mfi(h, l, c, v, 5)],
}


# the crossings are boolean by design, so they take the same causality checks and
# not the NaN ones (ADR 0037)
FLAGS = {
    "crossover": lambda h, l, c, v: [ta.crossover(ta.sma(c, 3), ta.sma(c, 5))],
    "crossunder": lambda h, l, c, v: [ta.crossunder(ta.sma(c, 3), 100.0)],
}


def vars_of(bundle):
    return [getattr(bundle, name) for name in bundle.__slots__]


def ramp_bars():
    # copies, every one: a caller that hole-punches these must not reach back and
    # corrupt the module-level fixture for every test that runs after it
    return RAMP + 1.0, RAMP - 1.0, RAMP.copy(), np.full(20, 10.0)


EVERY = {name: (lambda call=call: call(*ramp_bars())[0]) for name, call in LINES.items()}


def test_every_function_returns_one_value_per_bar():
    # the alignment contract: never a shorter array, so the same object goes into
    # a rule and into a chart with no padding decision in between
    for name, call in EVERY.items():
        out = call()
        assert out.shape == (20,), f"{name} returned {out.shape}"
        assert out.dtype == np.float64, f"{name} returned {out.dtype}"


def test_the_multi_line_functions_return_named_lines_not_tuples():
    bands = ta.bbands(RAMP, 5)
    assert bands.upper.shape == bands.middle.shape == bands.lower.shape == (20,)

    lines = ta.macd(RAMP, 3, 6, 3)
    assert lines.line.shape == lines.signal.shape == lines.histogram.shape == (20,)

    swing = ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5, 3)
    assert swing.k.shape == swing.d.shape == (20,)

    channel = ta.donchian(RAMP + 1.0, RAMP - 1.0, 5)
    assert channel.upper.shape == channel.lower.shape == (20,)

    high, low, close, _volume = ramp_bars()
    belt = ta.keltner(high, low, close, 5, 5)
    assert belt.upper.shape == belt.middle.shape == belt.lower.shape == (20,)

    trend = ta.adx(high, low, close, 5)
    assert trend.adx.shape == trend.plus.shape == trend.minus.shape == (20,)

    rails = ta.supertrend(high, low, close, 5)
    assert rails.direction.shape == rails.middle.shape == (20,)


def test_the_crossings_are_boolean_and_their_warm_up_is_false():
    # the one deliberate exception to the NaN rule. bool(nan) is True, so a float
    # warm-up would fire a rule on exactly the bars where nothing is known
    for name, call in FLAGS.items():
        out = call(*ramp_bars())[0]
        assert out.shape == (20,), name
        assert out.dtype == bool, f"{name} returned {out.dtype}"
        assert not bool(out[0]), f"{name} fired on bar 0"

    # and the trap itself, stated so nobody restores the float version
    assert bool(np.float64("nan"))


def test_the_warm_up_is_a_gap_and_the_rest_is_not():
    for name, call in EVERY.items():
        out = call()
        if name == "true_range":
            continue  # the first bar is its own range, so nothing is missing

        assert np.isnan(out[0]), f"{name} produced a value on bar 0"
        assert np.isfinite(out[-1]), f"{name} never warmed up"


def test_each_line_first_gives_a_value_on_the_bar_its_convention_names():
    # at LINES' lengths, by hand: a window of L fills on bar L - 1, a change needs one
    # bar before it, so rsi, roc, mfi and adx's two strengths wait one bar more, and a
    # smoother of a line that has its own warm-up waits out both: macd's signal at
    # slow + signal - 2, stoch's d at L + smooth - 2, hma at L + floor(sqrt(L)) - 2
    # and the ADX itself at 2L - 1
    firsts = {
        "sma":        [4],
        "ema":        [4],
        "wma":        [4],
        "rsi":        [5],
        "roc":        [5],
        "stdev":      [4],
        "zscore":     [4],
        "atr":        [4],
        "true_range": [0],
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

    assert firsts.keys() == LINES.keys()

    for name, call in LINES.items():
        got = [int(np.flatnonzero(np.isfinite(line))[0]) for line in call(*ramp_bars())]
        assert got == firsts[name], f"{name} first gives a value on {got}"


def test_a_pandas_series_works_without_pandas_being_a_dependency():
    pd = pytest.importorskip("pandas")

    assert np.allclose(ta.sma(pd.Series(RAMP), 5), ta.sma(RAMP, 5), equal_nan=True)


class Cross(probatum.Strategy):
    """Long while the fast average is over the slow one."""


    def init(self, data):
        self.fast   = ta.ema(data.close, 5)
        self.slow   = ta.ema(data.close, 15)
        self.warmup = 15


    def next(self, bar, account):
        i = bar.i
        if account.position == 0.0 and self.fast[i] > self.slow[i]:
            account.buy(1.0)
        elif account.position > 0.0 and self.fast[i] < self.slow[i]:
            account.close()


def test_an_indicator_drives_a_strategy_and_draws_on_the_chart():
    rng   = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0.1, 1.0, 120))
    data  = probatum.Data(np.column_stack([close, close + 1.0, close - 1.0, close,
                                           rng.uniform(500.0, 1500.0, 120)]),
                          symbol="BTCUSDT", market="linear",
                          time=1_735_689_600 * 10**9
                          + np.arange(120, dtype=np.int64) * 3_600 * 10**9)

    result = probatum.backtest(Cross(), data, "frictionless")
    fast   = result.policy.fast

    # the same array the rule traded on goes straight onto the chart, no padding
    chart = result.chart(probatum.chart.marks.Line(fast, "EMA 5"))

    assert len(fast) == len(data)
    assert chart.spec()["series"][0]["kind"] == "line"
    assert chart.spec()["series"][0]["v"][-1] == pytest.approx(fast[-1])


def test_ta_is_exported_from_the_package():
    assert probatum.ta is ta
    assert "ta" in probatum.__all__

    kinds = [n for n in ta.__all__ if isinstance(getattr(ta, n), type)]
    assert len(ta.__all__) - len(kinds) == 30
    assert set(kinds) == {"Bands", "Channel", "Macd", "Stochastic", "Trend"}
