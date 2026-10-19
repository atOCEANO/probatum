"""A Data owns its bars: it copies them once, so no later edit to its source changes
them, and every field and every slice of it is a read-only view (ADR 0005). It slices
on time, by text naming a period or by position, and on instruments, by their short
names, and joins two markets on one clock.
"""

import copy
import pickle
import re

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import (DataError, ProbatumAttributeError, ProbatumKeyError,
                      ProbatumTypeError)

HOUR = 3_600_000_000_000


def frame(n=48, start="2024-01-01", close=100.0, market=None, symbol=None):
    closes = close + np.arange(n, dtype=np.float64)
    index  = pd.date_range(start, periods=n, freq="h", tz="UTC", name="datetime")
    out    = pd.DataFrame({
        "open":   closes,
        "high":   closes + 1.0,
        "low":    closes - 1.0,
        "close":  closes,
        "volume": 1_000.0,
    }, index=index)
    if market:
        out.attrs.update(exchange="binance", market_type=market, symbol=symbol,
                         interval="1h", warnings=[])

    return out


def universe():
    return probatum.Data({"BTCUSDT": frame(), "ETHUSDT": frame(close=10.0).iloc[5:]})


def test_the_bars_are_copied_once_and_every_view_is_read_only():
    source = frame()
    data   = probatum.Data(source, symbol="BTCUSDT")

    source.iloc[0, 3] = 999.0

    assert data.close[0] == 100.0

    for field in (data.time, data.bars, data.closes, data.close, data.available,
                  data[1:5].close, universe().select(["ETHUSDT", "BTCUSDT"]).closes):
        assert not field.flags.writeable
        with pytest.raises(ValueError):
            field[0] = 0


def test_a_copy_or_a_pickle_of_a_data_is_as_read_only_as_the_data():
    data = universe()

    assert copy.copy(data) is data and copy.deepcopy(data) is data

    again = pickle.loads(pickle.dumps(data))

    assert again.fingerprint == data.fingerprint

    for field in (again.bars, again.closes, again.time, again.available):
        with pytest.raises(ValueError):
            field[0] = 0


def test_one_instrument_reads_in_the_singular_and_several_in_the_plural():
    data = universe()

    assert data.closes.shape == (48, 2)
    assert data["ETHUSDT"].close.shape == (48,)

    with pytest.raises(ProbatumAttributeError, match=re.escape(
            'data.close: the universe has 2 instruments; use data.closes or '
            'data["BTCUSDT"].close')):
        data.close

    with pytest.raises(ProbatumKeyError, match=re.escape(
            "data: no instrument 'SOLUSDT'; use one of BTCUSDT, ETHUSDT")):
        data["SOLUSDT"]


def test_text_names_a_period_and_a_slice_takes_both_ends_whole():
    data = universe()
    day  = data["2024-01-02":"2024-01-02"]

    assert len(day) == 24 and day.time[0] == data.time[24]

    hours = data["2024-01-01 05":"2024-01-01 06"]

    assert (hours.time - data.time[0]).tolist() == [5 * HOUR, 6 * HOUR]
    assert len(data[:"2024-01-01"]) == 24 and len(data["2024-01-02":]) == 24
    assert len(data[10:20]) == 10 and len(data[-5:]) == 5


def test_split_and_after_part_the_bars_at_a_time_or_a_position():
    train, test = universe().split("2024-01-02")

    assert (len(train), len(test)) == (24, 24)
    assert len(universe().after("2024-01-02 12:00")) == 12
    assert len(universe().after(np.datetime64("2024-01-02T12:00"))) == 12

    train, test = universe().split(40)

    assert (len(train), len(test))   == (40, 8)
    assert len(universe().after(-5)) == 5


def test_a_slice_keeps_the_bars_before_it_for_a_feature_to_warm_up_on():
    from probatum._data import features
    late = universe()["2024-01-02":]
    seen = []

    def lagged(history):
        seen.append(len(history))
        return np.stack([np.roll(history.closes, 1, axis=0)], axis=2)

    out = features(late, lagged)

    assert seen         == [48]
    assert out[0, 0, 0] == late.closes[0, 0] - 1.0


def test_select_takes_instruments_in_the_order_named():
    data = universe()
    pair = data.select(["ETHUSDT", "BTCUSDT"])

    assert pair.names == ("ETHUSDT", "BTCUSDT")
    np.testing.assert_array_equal(pair.closes[:, 1], data.closes[:, 0])


@pytest.mark.parametrize("take, error, says", [
    (lambda data: data["soon":], DataError, "start: 'soon' is not a time; give one"),
    (lambda data: data[:"NaT"], DataError, "stop: 'NaT' is not a time"),
    (lambda data: data.split(""), DataError, "when: '' is not a time"),
    (lambda data: data.split(None), ProbatumTypeError,
     "when: None; give a time or the text of one"),
    (lambda data: data.after(1.5), ProbatumTypeError,
     "when: 1.5; give a time or the text of one"),
    (lambda data: data[::2], ProbatumTypeError,
     "data: a step of 2; slice bars without one"),
    (lambda data: data[["BTCUSDT"]], ProbatumTypeError,
     "data: indexed by ['BTCUSDT']; use an instrument's name or a slice of times"),
    (lambda data: data.select(["X"]), ProbatumKeyError, "names: no instrument 'X'"),
    (lambda data: data.select("BTCUSDT"), ProbatumTypeError,
     "names: 'BTCUSDT'; give a list of names, or index the Data by one"),
    (lambda data: data.select(None), ProbatumTypeError,
     "names: None; give a list of names, or index the Data by one"),
    (lambda data: data.select([]), DataError, "names: none; name at least one"),
    (lambda data: data.select(["BTCUSDT", "BTCUSDT"]), DataError,
     "names: ['BTCUSDT', 'BTCUSDT'] names BTCUSDT twice; name each once"),
])
def test_a_slice_or_a_selection_it_cannot_make_is_refused(take, error, says):
    with pytest.raises(error, match=re.escape(says)):
        take(universe())


def test_a_join_puts_two_markets_on_one_clock_under_their_market_names():
    spot = probatum.Data(frame(market="spot", symbol="BTCUSDT"))
    perp = probatum.Data(frame(close=101.0, market="linear", symbol="BTCUSDT").iloc[2:])
    both = spot.join(perp)

    assert both.names == ("spot:BTCUSDT", "linear:BTCUSDT")
    assert both.available[:, 1].tolist()   == [False, False] + [True] * 46
    assert both["linear:BTCUSDT"].close[2] == 103.0

    with pytest.raises(DataError, match=re.escape(
            "other: holds binance:spot:BTCUSDT too; join Datas of different "
            "instruments")):
        spot.join(spot)

    four = probatum.Data(frame()[::4], symbol="X")
    with pytest.raises(DataError, match=re.escape("other: 4h bars beside 1h")):
        spot.join(four)

    with pytest.raises(DataError, match=re.escape(
            "other: instrument 0's bar at 2024-01-01 00:00 is not a bar")):
        spot.join(frame().assign(high=50.0))


def test_a_name_leaves_out_what_the_bars_do_not_say():
    plain = probatum.Data(frame(), symbol="BTCUSDT")
    both  = plain.join(probatum.Data(frame(market="linear", symbol="BTCUSDT")))

    assert both.names == ("BTCUSDT", "binance:linear:BTCUSDT")


def test_the_provenance_names_the_instruments_and_is_the_caller_s_own_copy():
    spot = probatum.Data(frame(market="spot", symbol="BTCUSDT"))
    both = spot.join(probatum.Data(frame(market="linear", symbol="BTCUSDT")))

    assert both.provenance["instruments"] == ["binance:spot:BTCUSDT",
                                              "binance:linear:BTCUSDT"]
    assert both["linear:BTCUSDT"].provenance["instruments"] == [
        "binance:linear:BTCUSDT"]
    assert universe().provenance["instruments"] == ["BTCUSDT", "ETHUSDT"]

    both.provenance["warnings"].append("an edit")

    assert both.provenance["warnings"] == []


def test_gaps_list_where_an_instrument_stops_printing():
    source = frame().drop(frame().index[10:13])
    gaps   = probatum.Data(source, symbol="BTCUSDT").gaps()

    assert len(gaps) == 1

    gap = gaps[0]

    assert (gap.instrument, gap.missing) == ("BTCUSDT", 3)
    assert gap.after  == np.datetime64("2024-01-01T09:00", "ns")
    assert gap.before == np.datetime64("2024-01-01T13:00", "ns")


def test_a_data_shows_what_it_holds():
    assert repr(universe()) == ("Data(2 instruments, 1h, 48 bars from 2024-01-01 00:00 "
                                "to 2024-01-02 23:00)")
    assert repr(universe()["ETHUSDT"][:0]) == "Data(ETHUSDT, 1h, no bars)"
