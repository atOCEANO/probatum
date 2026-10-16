"""Coarser bars and outside series arrive when they were known (ADR 0006). `resample`
builds bars of a whole multiple of the interval, aligned to 00:00 UTC and weeks to
Monday on the Data's own steps, and drops a coarse bar the data does not cover to its
end; `asof` puts each value on the row by whose end it was known, never on the row
whose open time matches it, and reads a frame of bars, from wherever it came, at each
bar's end.
"""

import datetime
import re

import numpy as np
import pandas as pd
import polars
import pyarrow as pa
import pytest

import probatum
from probatum._data import features

HOUR  = 3_600_000_000_000
START = 1_704_067_200_000_000_000      # 2024-01-01 00:00 UTC, a Monday


def frame(n=24, start="2024-01-01", freq="h", close=100.0, closes=None):
    closes = close + np.arange(n, dtype=np.float64) if closes is None else closes
    index  = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="datetime")
    return pd.DataFrame({
        "open":   closes - 0.5,
        "high":   closes + 1.0,
        "low":    closes - 1.0,
        "close":  closes,
        "volume": 10.0 + np.arange(n),
    }, index=index)


def routed(market, close=100.0, interval="1h", **settings):
    out = frame(close=close, **settings)
    out.attrs.update(exchange="binance", market_type=market, symbol="BTCUSDT",
                     interval=interval, volume_unit="base", warnings=[])
    return out


def hours(data):
    return ((data.time - START) / HOUR).tolist()


def test_a_coarse_bar_takes_the_first_open_the_extremes_the_last_close_and_the_volume():
    # within each four hours the highest high is the second bar's and the lowest low
    # the third's
    zigzag = 100.0 + np.tile([0.0, 3.0, -2.0, 1.0], 6)
    data   = probatum.Data(frame(closes=zigzag), symbol="BTCUSDT")
    four   = data.resample("4h")

    assert four.interval == "4h" and hours(four) == [0, 4, 8, 12, 16, 20]
    assert four.bars[0, 0].tolist()          == [99.5, 104.0, 97.0, 101.0, 46.0]
    assert four.provenance["resampled_from"] == "1h"
    assert data.resample("1h").fingerprint   == data.fingerprint


def test_a_coarse_bar_is_available_where_any_of_its_bars_was():
    data = probatum.Data({
        "BTCUSDT": frame(),
        "ETHUSDT": frame(close=10.0).iloc[[1, 2, 9, 10, 11]],
    })
    four = data.resample("4h")

    assert four.available[:, 1].tolist() == [True, False, True, False, False, False]

    eth, hourly = four["ETHUSDT"], data["ETHUSDT"]
    assert (eth.open[0], eth.close[0]) == (hourly.open[1], hourly.close[2])
    assert eth.volume[2] == hourly.volumes[9:12, 0].sum()


def test_coarser_bars_keep_the_names_of_their_instruments():
    both = probatum.Data(routed("spot")).join(probatum.Data(routed("linear", 101.0)))
    perp = both.select(["linear:BTCUSDT"]).resample("4h")

    assert perp.names == ("linear:BTCUSDT",)
    assert perp["linear:BTCUSDT"].close[0] == 104.0


@pytest.mark.parametrize("source, interval, opens", [
    # hourly bars at half past the hour give coarse bars at half past too
    (frame(start="2024-01-01 00:30"), "4h", [0.5, 4.5, 8.5, 12.5, 16.5, 20.5]),
    # 4-hour bars from 02:00 give days from 02:00, so none straddles two days
    (frame(n=12, start="2024-01-01 02:00", freq="4h"), "1d", [2, 26]),
    # daily bars from a Wednesday give weeks from the Monday before, and the third
    # week, which the bars stop inside, is dropped
    (frame(n=14, start="2024-01-03", freq="D"), "1w", [0, 168]),
    # three days run from 1970-01-01, so the first holds 1 and 2 January, and the
    # fourth, from 9 January, is not covered to its end
    (frame(n=9, freq="D"), "3d", [-24, 48, 120]),
    # weeks of hourly bars at half past start on Monday at half past
    (frame(n=192, start="2024-01-01 00:30"), "1w", [0.5]),
])
def test_coarse_bars_align_to_midnight_and_monday_on_the_data_s_own_steps(source,
                                                                          interval,
                                                                          opens):
    coarse = probatum.Data(source, symbol="X").resample(interval)

    assert hours(coarse) == opens


def test_a_coarse_bar_the_bars_stop_inside_is_dropped():
    four = probatum.Data(frame(n=22), symbol="X").resample("4h")

    assert hours(four) == [0, 4, 8, 12, 16]


def test_a_slice_s_coarse_bars_keep_the_ones_before_it_and_read_nothing_after_it():
    data = probatum.Data(frame(), symbol="X")
    late = data["2024-01-01 06:00":"2024-01-01 17:00"]
    four = late.resample("4h")

    # 06:00 opens inside the bar from 04:00, which began before the slice, and the
    # slice ends at 18:00, inside the bar from 16:00
    assert hours(four) == [8, 12]
    np.testing.assert_array_equal(four.bars, data.resample("4h").bars[2:4])

    seen = []
    features(four, lambda history: seen.append(len(history))
             or np.zeros((len(history), 1, 1)))

    assert seen == [4]

    # a slice that opens on a coarse bar's start, or after a gap that leaves the bar
    # it opens inside without bars before it, keeps that bar
    assert hours(data["2024-01-01 08:00":].resample("4h"))[0] == 8

    gapped = probatum.Data(frame().drop(frame().index[6:9]), symbol="X")
    assert hours(gapped["2024-01-01 09:00":].resample("4h"))[0] == 8


@pytest.mark.parametrize("make, says", [
    (lambda: probatum.Data(frame(), symbol="X").resample("7m"),
     "interval: '7m'; use one of 1m"),
    (lambda: probatum.Data(frame(), symbol="X").resample("30m"),
     "interval: '30m' for 1h bars; give a whole multiple of 1h"),
    (lambda: probatum.Data(frame(n=24, freq="8h"), symbol="X").resample("12h"),
     "interval: '12h' for 8h bars; give a whole multiple of 8h"),
    (lambda: probatum.Data(frame(n=3), symbol="X").resample("1d"),
     "interval: '1d', and these bars cover no 1d bar to its end; resample bars that "
     "do"),
    (lambda: probatum.Data(frame(), symbol="X")[:0].resample("4h"),
     "data: no bars to resample; resample a Data that has some"),
    (lambda: probatum.Data(frame(), symbol="X")[5:5].resample("4h"),
     "data: no bars to resample; resample a Data that has some"),
])
def test_a_resample_it_cannot_make_is_refused(make, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        make()


def test_a_4_hour_bar_reaches_the_hourly_clock_at_its_end():
    # the Specification's worked example: the bar from 00:00 ends at 04:00, as the
    # hourly bar from 03:00 does, so its value first appears on the 03:00 row
    hourly = probatum.Data(frame(n=12), symbol="BTCUSDT")
    four   = hourly.resample("4h")
    on     = hourly.asof(np.array([1.0, 2.0, 3.0]), four)

    assert np.isnan(on[:3]).all()
    assert on[3:].tolist() == [1.0] * 4 + [2.0] * 4 + [3.0]
    np.testing.assert_array_equal(hourly.asof([1.0, 2.0, 3.0], frame(n=3, freq="4h")),
                                  on)


def test_an_instrument_that_printed_no_coarse_bar_keeps_its_last_value():
    # ETHUSDT prints nothing from 04:00 to 08:00, so the bar from 04:00 brings no
    # close of its, and its close from the bar before stands until 12:00
    data = probatum.Data({"BTCUSDT": frame(n=12),
                          "ETHUSDT": frame(n=12, close=10.0).iloc[[0, 1, 2, 3, 8, 9,
                                                                   10, 11]]})
    four = data.resample("4h")

    assert four.available[:, 1].tolist() == [True, False, True]

    on = data.asof(four.closes, four)

    assert on[3:11, 1].tolist() == [13.0] * 8 and on[11, 1] == 21.0
    assert on[3:11, 0].tolist() == [103.0] * 4 + [107.0] * 4

    eth = data["ETHUSDT"].asof(four["ETHUSDT"].close, four["ETHUSDT"])
    np.testing.assert_array_equal(eth, on[:, 1])


def test_a_value_reaches_the_first_row_by_whose_end_it_was_known():
    hourly = probatum.Data(frame(n=8), symbol="BTCUSDT")
    known  = pd.DatetimeIndex(["2024-01-01 02:30", "2024-01-01 05:00"], tz="UTC")
    on     = hourly.asof([7.0, 8.0], known)

    assert np.isnan(on[:2]).all() and on[2:].tolist() == [7.0, 7.0] + [8.0] * 4

    pair   = probatum.Data({"BTCUSDT": frame(n=8), "ETHUSDT": frame(n=8, close=10.0)})
    coarse = pair.resample("4h")

    assert pair.asof(coarse.closes, coarse).shape == (8, 2)


def test_a_frame_or_a_series_carries_its_own_times():
    hourly = probatum.Data(frame(n=8), symbol="BTCUSDT")
    known  = pd.date_range("2024-01-01 01:30", periods=3, freq="2h", tz="UTC")
    table  = pd.DataFrame({
        "oi":     [1.0, 2.0, 3.0],
        "oi_usd": [10.0, 20.0, 30.0],
        "unit":   ["coin"] * 3,
    }, index=known)

    on = hourly.asof(table)

    assert on.shape == (8, 2) and np.isnan(on[0]).all()
    assert on[1:].tolist() == [[1.0, 10.0]] * 2 + [[2.0, 20.0]] * 2 + [[3.0, 30.0]] * 3
    np.testing.assert_array_equal(hourly.asof(table["oi"]), on[:, 0])
    np.testing.assert_array_equal(hourly.asof(polars.from_pandas(table.reset_index())),
                                  on)

    stamped = table.reset_index(drop=True).assign(timestamp=known.as_unit("ms").asi8)
    np.testing.assert_array_equal(hourly.asof(stamped), on)


def test_bars_from_any_source_are_known_at_their_ends():
    hourly       = probatum.Data(frame(n=12), symbol="BTCUSDT")
    candles      = frame(n=3, freq="4h")
    milliseconds = candles.reset_index(drop=True).assign(
        timestamp=candles.index.as_unit("ms").asi8)

    for source in (candles, milliseconds, polars.from_pandas(candles.reset_index())):
        on = hourly.asof(source)
        assert on.shape == (12, 5) and np.isnan(on[:3]).all()
        assert on[3, 3] == candles.close.iloc[0] and on[7, 3] == candles.close.iloc[1]


def test_a_router_candle_column_is_known_at_its_end_and_any_other_series_at_its_index():
    hourly  = probatum.Data(frame(n=12), symbol="BTCUSDT")
    candles = frame(n=3, freq="4h")
    plain   = hourly.asof(candles["close"])

    candles.attrs.update(exchange="binance", market_type="linear", symbol="BTCUSDT",
                         interval="4h", volume_unit="base", warnings=[])
    on = hourly.asof(candles["close"])

    assert plain[0] == candles.close.iloc[0]
    assert np.isnan(on[:3]).all() and on[3] == candles.close.iloc[0]
    np.testing.assert_array_equal(hourly.asof(candles)[:, 3], on)


def monthly():
    out = frame(n=2, freq="MS")
    out.attrs.update(exchange="binance", market_type="linear", symbol="BTCUSDT",
                     interval="1M", volume_unit="base", warnings=[])
    return out


@pytest.mark.parametrize("values, when, error, says", [
    ([1.0, 2.0], "four", ValueError,
     "values: 2 for 3 times; give one row of values per time"),
    (5.0, "four", ValueError, "values: one value for 3 times"),
    ([1.0, 2.0], pd.DatetimeIndex(["2024-01-01 05:00", "2024-01-01 02:00"], tz="UTC"),
     ValueError, "when: times that fall back at row 1; sort the rows by the time each "
                 "became known"),
    ([1.0, 2.0], np.array(["2024-01-01", "NaT"], dtype="datetime64[ns]"), ValueError,
     "when: no time at row 1; give each row the time it became known"),
    ([1.0, 2.0], ["2024-01-01 02:00", "2024-01-01 05:00"], TypeError,
     "when: values of <U16; give datetimes or int64 nanoseconds since 1970"),
    (pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-01 05:00",
                                                   "2024-01-01 02:00"], tz="UTC")),
     None, ValueError, "values: times that fall back at row 1; sort the rows"),
    ([1.0], None, TypeError,
     "values: a list, which carries no times; pass when=, the time each row became "
     "known"),
    (pa.array([1.0]), None, TypeError, "values: a DoubleArray, which carries no times"),
    (pd.DataFrame({"x": [1.0]}), None, ValueError,
     "values: no times; give the time each row became known in a datetime index or "
     "column, or pass when="),
    (pd.DataFrame({"unit": ["coin"]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC")),
     None, ValueError, "values: no column of numbers; give one beside the times"),
    (polars.DataFrame({"a": [datetime.datetime(2024, 1, 1)],
                       "b": [datetime.datetime(2024, 1, 1)], "x": [1.0]}),
     None, ValueError, "values: times in a, b; keep the time each row became known "
                       "alone, or pass when="),
    (monthly()["close"], None, ValueError,
     "values: candles of '1M', an interval no Data runs on; give them on one of 1m"),
    (monthly(), None, ValueError,
     "values: candles of '1M', an interval no Data runs on"),
])
def test_values_it_cannot_place_are_refused(values, when, error, says):
    hourly = probatum.Data(frame(n=12), symbol="BTCUSDT")
    when   = hourly.resample("4h") if isinstance(when, str) else when

    with pytest.raises(error, match=re.escape(says)):
        hourly.asof(values, when)
