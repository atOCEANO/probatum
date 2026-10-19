"""A Data reads a frame of any library, a dict of frames or arrays onto one clock: the
times at which any instrument printed a bar, whole intervals apart, a daily bar
placed on its own date in whatever zone it was stamped (ADR 0004). A pandas frame
reads alike whether pyarrow exports it or its own methods do. Every bar is checked as
it arrives, as the binding checks it, and a refusal names the instrument and the time
(ADR 0007).
"""

import datetime
import re

import dateutil.tz
import numpy as np
import pandas as pd
import polars
import pyarrow as pa
import pytest

import probatum
from probatum import DataError, ProbatumTypeError

HOUR  = 3_600_000_000_000
START = 1_704_067_200_000_000_000      # 2024-01-01 00:00 UTC


def frame(n=6, start="2024-01-01", freq="h", close=None, tz="UTC"):
    close = np.linspace(100.0, 105.0, n) if close is None else np.asarray(close)
    index = pd.date_range(start, periods=n, freq=freq, tz=tz, name="datetime")

    return pd.DataFrame({
        "open":   close,
        "high":   close + 1.0,
        "low":    close - 1.0,
        "close":  close,
        "volume": 1_000.0,
    }, index=index)


def routed(symbol="BTCUSDT", market="linear", source=None, **attrs):
    out = frame() if source is None else source
    out.attrs.update(dict(exchange="binance", market_type=market, symbol=symbol,
                          interval="1h", quote="USDT", warnings=["a warning"]), **attrs)
    return out


def without_pyarrow(monkeypatch):
    def missing(self, requested_schema=None):
        raise ImportError("pyarrow is not installed")

    monkeypatch.setattr(pd.DataFrame, "__arrow_c_stream__", missing)


def dates(data):
    return [str(np.datetime64(int(t), "ns").astype("datetime64[D]")) for t in data.time]


def test_a_router_frame_names_its_venue_market_symbol_and_interval():
    data = probatum.Data(routed())

    assert data.names == ("BTCUSDT",)
    assert data.instruments == (probatum.Instrument(
        symbol = "BTCUSDT",
        market = "linear",
        venue  = "binance",
        quote  = "USDT",
    ),)
    assert data.interval               == "1h"
    assert data.provenance["warnings"] == ["a warning"]
    assert data.provenance["source"]   == "router"


@pytest.mark.parametrize("source", [
    lambda f: f,
    lambda f: pa.Table.from_pandas(f),
    lambda f: polars.from_pandas(f.reset_index()),
    lambda f: f.reset_index(drop=True).assign(timestamp=f.index.as_unit("ms").asi8),
])
def test_every_kind_of_frame_gives_the_same_bars(source):
    data = probatum.Data(source(frame()), symbol="BTCUSDT")

    assert data.time.tolist() == [START + k * HOUR for k in range(6)]
    np.testing.assert_array_equal(data.close, np.linspace(100.0, 105.0, 6))


@pytest.mark.parametrize("source", [
    lambda: frame(tz="America/New_York"),
    lambda: frame(tz="America/New_York").reset_index(),
])
def test_a_pandas_frame_without_pyarrow_is_read_by_its_own_methods(monkeypatch, source):
    without_pyarrow(monkeypatch)
    data = probatum.Data(source(), symbol="SPY")

    assert data.time[0] == START + 5 * HOUR
    np.testing.assert_array_equal(data.close, np.linspace(100.0, 105.0, 6))


def test_a_column_pyarrow_cannot_convert_leaves_the_frame_to_pandas():
    source = frame().assign(note=[1, "x", 2, 3, 4, 5])

    assert len(probatum.Data(source, symbol="X")) == 6


def test_arrays_take_their_times_from_time():
    bars = frame()[["open", "high", "low", "close", "volume"]].to_numpy()
    one  = probatum.Data(bars, time=frame().index, symbol="BTCUSDT")
    two  = probatum.Data(
        np.stack([bars, bars], axis=1),
        time   = frame().index.tz_localize(None).to_numpy(),
        symbol = ["BTCUSDT", "ETHUSDT"],
    )

    hours = [START + k * HOUR for k in range(6)]
    assert one.time.tolist() == two.time.tolist() == hours
    assert two.names == ("BTCUSDT", "ETHUSDT")

    unnamed = probatum.Data(np.stack([bars, bars], axis=1), time=frame().index)
    assert unnamed.names == ("0", "1")


def test_a_missing_time_in_a_column_the_clock_does_not_read_is_no_matter():
    source = frame().assign(settled=lambda f: f.index.to_series().where(f.close > 101))

    assert len(probatum.Data(source, symbol="X", time=source.index)) == 6


def test_the_clock_is_the_bars_that_exist():
    data = probatum.Data({"BTCUSDT": frame(), "ETHUSDT": frame().iloc[[1, 4]]})

    assert len(data) == 6
    assert data.available[:, 1].tolist() == [False, True, False, False, True, False]
    assert np.isnan(data.closes[0, 1]) and data.closes[1, 1] == 101.0


def test_a_day_stamped_at_midnight_in_new_york_stays_on_its_date():
    days = frame(3, start="2024-01-04", freq="B", tz="America/New_York")
    data = probatum.Data(days, symbol="SPY", market="spot")

    assert data.interval == "1d"
    assert [str(np.datetime64(int(t), "ns"))[:16] for t in data.time] == [
        "2024-01-04T00:00", "2024-01-05T00:00", "2024-01-08T00:00"]


def test_days_keep_their_dates_across_a_change_of_clocks():
    # New York's midnights of 9, 10 and 11 March 2024 are 24 and then 23 hours apart
    data = probatum.Data(frame(3, start="2024-03-09", freq="D", tz="America/New_York"),
                         symbol="SPY")

    assert data.interval == "1d"
    assert dates(data)   == ["2024-03-09", "2024-03-10", "2024-03-11"]


@pytest.mark.parametrize("tz", [datetime.timezone(datetime.timedelta(hours=-5)),
                                dateutil.tz.gettz("America/New_York")])
@pytest.mark.parametrize("pyarrow", [True, False])
def test_a_day_in_a_fixed_offset_or_a_dateutil_zone_stays_on_its_date(monkeypatch, tz,
                                                                        pyarrow):
    if not pyarrow:
        without_pyarrow(monkeypatch)

    data = probatum.Data(frame(3, start="2024-01-04", freq="B", tz=tz), symbol="SPY")

    assert dates(data) == ["2024-01-04", "2024-01-05", "2024-01-08"]


def test_a_zoned_series_given_as_time_keeps_its_zone():
    days = frame(3, start="2024-01-04", freq="B", tz="Asia/Tokyo")
    bars = days[["open", "high", "low", "close", "volume"]].to_numpy()
    data = probatum.Data(bars, time=days.index.to_series(), symbol="7203")

    assert dates(data) == ["2024-01-04", "2024-01-05", "2024-01-08"]


def test_the_interval_is_the_median_gap_or_the_one_given_or_declared():
    assert probatum.Data(frame(freq="4h"), symbol="X").interval        == "4h"
    assert probatum.Data(frame(1), symbol="X", interval="1h").interval == "1h"

    # bars two hours apart under a frame that declares hourly ones are hourly bars
    # with gaps, not two-hour bars
    declared = probatum.Data(routed(source=frame(freq="2h")))
    assert declared.interval == "1h" and len(declared.gaps()) == 5


@pytest.mark.parametrize("make, says", [
    (lambda: probatum.Data(frame(1), symbol="X"),
     "data: one bar, from which no interval can be read; pass interval="),
    (lambda: probatum.Data(frame(0), symbol="X"), "data: no bars; give at least one"),
    (lambda: probatum.Data({}, interval="1h"), "data: no bars; give at least one"),
    (lambda: probatum.Data(frame(freq="7min"), symbol="X"),
     "data: bars 7m apart at the median, which is no interval the clock runs on"),
    (lambda: probatum.Data(np.ones((3, 5)), time=np.arange(3) * 3_600),
     "data: bars 3600ns apart at the median"),
    (lambda: probatum.Data({"BTCUSDT": routed(),
                            "ETHUSDT": routed("ETHUSDT", interval="4h")}),
     "data: frames of 1h, 4h bars; give every instrument bars on one interval"),
    (lambda: probatum.Data(routed(), interval="4h"),
     "interval: '4h' for frames of 1h bars; leave it out, the frames name their own"),
    (lambda: probatum.Data({"A": frame(), "B": frame(start="2024-01-01 00:30")}),
     "data: B opens a bar at 2024-01-01 00:30, between the 1h steps from 2024-01-01 "
     "00:00"),
    (lambda: probatum.Data(frame(), symbol="X", interval="1M"),
     "interval: '1M'; use one of 1m, 3m"),
    (lambda: probatum.Data(frame().iloc[[0, 2, 1]], symbol="X"),
     "data: X's bars fall back at row 2, from 2024-01-01 02:00 to 2024-01-01 01:00; "
     "sort them by time"),
    (lambda: probatum.Data(pd.concat([frame(), frame()]).sort_index(), symbol="X"),
     "data: X has two bars at 2024-01-01 00:00; give each bar a time of its own"),
    (lambda: probatum.Data(frame(), symbol="X", interval="1d"),
     "data: X has two bars dated 2024-01-01; give each bar a date of its own"),
    (lambda: probatum.Data(np.ones((2, 5)), time=np.array(["2024-01-01", "NaT"],
                                                          dtype="datetime64[ns]")),
     "data: instrument 0's bar at row 1 has no open time; give every bar its open "
     "time"),
    (lambda: probatum.Data(pa.table({
        "time": pa.array([START, START + 24 * HOUR],
                         pa.timestamp("ns", "Mars/Olympus")),
        **{name: [1.0, 1.0] for name in ("open", "high", "low", "close", "volume")}}),
        symbol="X"),
     "data: times in the zone 'Mars/Olympus', which this machine does not know; give "
     "them in UTC"),
])
def test_a_clock_the_bars_cannot_keep_is_refused(make, says):
    with pytest.raises(DataError, match=re.escape(says)):
        make()


def broken(row, **values):
    out = frame()
    for column, value in values.items():
        out.iloc[row, out.columns.get_loc(column)] = value
    return out


@pytest.mark.parametrize("bars, says", [
    (broken(2, close=np.nan),
     "data: X's bar at 2024-01-01 02:00 misses its close; give all five values"),
    (broken(3, high=np.inf),
     "data: X's bar at 2024-01-01 03:00 holds an infinite value"),
    (broken(1, high=50.0),
     "data: X's bar at 2024-01-01 01:00 is not a bar: open 101, high 50, low 100, "
     "close 101; need low <= open <= high"),
    (broken(4, low=0.0), "data: X's bar at 2024-01-01 04:00 has a low of 0; give "
                         "prices above 0"),
    (broken(5, volume=-1.0), "data: X's bar at 2024-01-01 05:00 has a volume of -1; "
                             "give a volume of 0 or more"),
])
def test_a_bar_the_engine_cannot_simulate_is_refused_where_it_sits(bars, says):
    with pytest.raises(DataError, match=re.escape(says)):
        probatum.Data(bars, symbol="X")


def test_a_row_with_no_bar_is_an_instrument_that_did_not_print():
    data = probatum.Data(broken(2, open=np.nan, high=np.nan, low=np.nan, close=np.nan,
                                volume=np.nan), symbol="X")

    assert len(data) == 5 and START + 2 * HOUR not in data.time


@pytest.mark.parametrize("make, error, says", [
    (lambda: probatum.Data(frame().drop(columns=["volume"]), symbol="X"), DataError,
     "data: no volume column; give open, high, low, close and volume"),
    (lambda: probatum.Data(frame().assign(close="x"), symbol="X"), DataError,
     "data: column close holds text; give numbers"),
    (lambda: probatum.Data(frame().assign(close=lambda f: pd.Categorical(f.close)),
                           symbol="X"), DataError,
     "data: column close holds categories of floats; give numbers"),
    (lambda: probatum.Data(frame().assign(close=lambda f: f.index), symbol="X"),
     DataError, "data: column close holds times; give numbers"),
    (lambda: probatum.Data(frame().reset_index().assign(
        datetime=lambda f: f.datetime.where(f.index != 2)), symbol="X"), DataError,
     "data: X's bar at row 2 has no open time; give every bar its open time"),
    (lambda: probatum.Data(frame().reset_index(drop=True), symbol="X"), DataError,
     "data: no time column; give the bars' open times in a datetime column or index"),
    (lambda: probatum.Data(frame().reset_index().assign(
        close_time=lambda f: f.datetime), symbol="X"), DataError,
     "data: times in datetime, close_time; keep the bars' open times alone"),
    (lambda: probatum.Data(frame().assign(close_time=lambda f: f.index), symbol="X"),
     DataError, "data: times in close_time, datetime; keep the bars' open times "
                 "alone"),
    (lambda: probatum.Data(np.ones((3, 5))), ProbatumTypeError,
     "time: none for an array of bars; pass time=, each bar's open time"),
    (lambda: probatum.Data(np.ones((3, 5)), time=["2024-01-01"] * 3), ProbatumTypeError,
     "time: values of <U10; give datetimes or int64 nanoseconds since 1970"),
    (lambda: probatum.Data(np.ones((3, 5)), time=np.arange(2)), DataError,
     "time: 2 times for 3 bars; give each bar its open time"),
    (lambda: probatum.Data(np.ones((3, 4)), time=np.arange(3)), DataError,
     "data: an array of shape (3, 4); give (T, 5) bars of one instrument"),
    (lambda: probatum.Data(np.ones((3, 2, 5)), time=np.arange(3), symbol="X"),
     ProbatumTypeError, "symbol: 'X' for 2 instruments; give a list of 2 names"),
    (lambda: probatum.Data(np.ones((3, 2, 5)), time=np.arange(3), symbol=["X"]),
     DataError, "symbol: ['X'] for 2 instruments; give a list of 2 names"),
    (lambda: probatum.Data(np.ones((3, 2, 5)), time=np.arange(3), symbol=["X", "X"]),
     DataError, "symbol: ['X', 'X'] names X twice; name each once"),
    (lambda: probatum.Data(frame(), symbol=["X"]), ProbatumTypeError,
     "symbol: ['X'] for one instrument's bars; give its name"),
    (lambda: probatum.Data({1: frame()}), ProbatumTypeError,
     "data: a key of 1; key each frame by its symbol"),
    (lambda: probatum.Data(probatum.Data(frame(), symbol="X"), symbol="Y"),
     ProbatumTypeError,
     "data: a Data with settings beside it; build the Data with them, or pass it "
     "alone"),
    (lambda: probatum.Data(routed(), symbol="ETHUSDT"), DataError,
     "symbol: 'ETHUSDT' for a frame of BTCUSDT; leave it out, the frame names itself"),
    (lambda: probatum.Data(routed(), market="spot"), DataError,
     "market: 'spot' for a frame of the linear market; leave it out"),
    (lambda: probatum.Data(routed(market="inverse", volume_unit="base")), DataError,
     "data: BTCUSDT's volume is in base, and an inverse contract's volume is counted "
     "in contracts"),
    (lambda: probatum.Data(frame(), symbol="X", market="futures"), DataError,
     "market: 'futures'; use spot, linear, inverse"),
    (lambda: probatum.Data(frame(), symbol="X", market=5), ProbatumTypeError,
     "market: 5; use spot, linear, inverse"),
    (lambda: probatum.Data({"A": frame(), "B": frame().assign()}, symbol="C"),
     ProbatumTypeError, "symbol: 'C' beside a dict of frames, whose keys name its "
                "instruments; leave it out"),
])
def test_a_source_it_cannot_read_bars_from_is_refused(make, error, says):
    with pytest.raises(error, match=re.escape(says)):
        make()
