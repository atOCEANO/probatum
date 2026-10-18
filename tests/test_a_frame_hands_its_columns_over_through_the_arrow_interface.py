"""A frame hands its columns over through the Arrow C stream or array interface,
read in Rust, so polars, pyarrow and pandas with pyarrow reach probatum without it
importing any of them (ADR 0044): numbers as float64, an integer column missing no
value as int64, times and dates as nanoseconds since 1970 with their zone, and any
other column by its format alone. Nothing in a column no one reads can refuse the
frame, so a missing time comes back as NaT for the clock to refuse.
"""

import datetime
import re

import numpy as np
import pandas as pd
import polars
import pyarrow as pa
import pytest

from probatum import DataError, ProbatumTypeError
from probatum._probatum import arrow_columns

NAT = np.iinfo(np.int64).min


def columns(data):
    return {name: (kind, values, detail)
            for name, kind, values, detail in arrow_columns(data)}


def test_each_kind_of_column_comes_back_as_its_kind():
    table = pa.table({
        "close":    pa.array([1.5, 2.5], pa.float64()),
        "low":      pa.array([1.0, 2.5], pa.float32()),
        "count":    pa.array([2**62, 3], pa.int64()),
        "small":    pa.array([4, 5], pa.uint8()),
        "time":     pa.array([1_700_000_000_000, 1_700_003_600_000],
                             pa.timestamp("ms", tz="America/New_York")),
        "day":      pa.array([datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]),
        "long_day": pa.array([datetime.date(2024, 1, 2)] * 2, pa.date64()),
        "symbol":   pa.array(["BTCUSDT", "ETHUSDT"]),
        "side":     pa.array(["buy", "sell"]).dictionary_encode(),
    })

    got = columns(table)

    assert got["close"][0] == "float" and got["close"][1].tolist() == [1.5, 2.5]
    assert got["low"][0] == "float" and got["low"][1].tolist() == [1.0, 2.5]
    assert got["count"][0] == "int" and got["count"][1].tolist() == [2**62, 3]
    assert got["small"][0] == "int" and got["small"][1].tolist() == [4, 5]
    assert got["time"][0] == "time" and got["time"][2] == "America/New_York"
    assert got["time"][1].tolist() == [1_700_000_000_000_000_000,
                                       1_700_003_600_000_000_000]

    day = np.datetime64("2024-01-02", "ns").astype(np.int64)
    assert got["day"][0] == "date" and got["day"][1][0] == day
    assert got["long_day"][0] == "date" and got["long_day"][1][0] == day
    assert got["symbol"][0] == "other" and got["symbol"][1] is None
    assert got["symbol"][2] == "u"
    assert got["side"][0] == "other" and got["side"][2] == "dictionary of u"


def test_batches_join_in_order_and_a_slice_starts_where_it_was_cut():
    table = pa.Table.from_batches([pa.record_batch({"close": [1.0, None, 3.0]}),
                                   pa.record_batch({"close": [4.0, 5.0, 6.0]})])
    close = columns(table.slice(1, 4))["close"][1]

    assert np.isnan(close[0]) and close[1:].tolist() == [3.0, 4.0, 5.0]


class OnlyAnArray:
    """A producer that exports the array interface and no stream."""


    def __init__(self, array):
        self.array = array


    def __arrow_c_array__(self, requested_schema=None):
        return self.array.__arrow_c_array__(requested_schema)


def test_a_table_cut_or_missing_a_row_of_its_own_is_read_where_it_stands():
    rows = pa.StructArray.from_arrays([pa.array([1.0, 2.0, 3.0, 4.0])], ["close"],
                                      mask=pa.array([False, False, True, False]))
    close = columns(OnlyAnArray(rows.slice(1)))["close"][1]

    assert close[0] == 2.0 and np.isnan(close[1]) and close[2] == 4.0


def test_an_integer_column_missing_a_value_or_past_int64_comes_back_as_floats():
    kind, volume, _ = columns(pa.table({"volume": pa.array([1, None, 3])}))["volume"]
    assert kind == "float" and np.isnan(volume[1]) and volume[2] == 3.0

    kind, huge, _ = columns(pa.table({"id": pa.array([2**64 - 1, 1], pa.uint64())}))["id"]
    assert kind == "float" and huge.tolist() == [float(2**64 - 1), 1.0]


def test_a_missing_time_comes_back_as_nat():
    got = columns(pa.table({"time": pa.array([0, None], pa.timestamp("ms"))}))

    assert got["time"][1].tolist() == [0, NAT]


class Spent:
    """A producer that hands over a capsule whose stream was read already."""


    def __init__(self, table):
        self.capsule = table.__arrow_c_stream__()
        pa.RecordBatchReader.from_stream(self).read_all()


    def __arrow_c_stream__(self, requested_schema=None):
        return self.capsule


class Wrong:

    def __arrow_c_stream__(self, requested_schema=None):
        return {"not": "a capsule"}


@pytest.mark.parametrize("data, error, says", [
    (pa.table({"time": pa.array([2**62], pa.timestamp("s"))}), DataError,
     'data: column "time" holds 4611686018427387904 ticks of 1000000000 nanoseconds, '
     'past what nanoseconds since 1970 can count'),
    (Spent(pa.table({"close": [1.0]})), DataError,
     "data: an Arrow stream already released; pass the frame again"),
    (Wrong(), ProbatumTypeError,
     "data: __arrow_c_stream__ gave a dict, not a capsule named arrow_array_stream"),
    ([1.0, 2.0], ProbatumTypeError,
     "data: a list has neither the Arrow C stream nor the array interface"),
])
def test_what_cannot_be_read_as_a_table_is_refused(data, error, says):
    with pytest.raises(error, match=re.escape(says)):
        arrow_columns(data)


def test_the_array_interface_is_read_where_there_is_no_stream():
    got = columns(OnlyAnArray(pa.record_batch({"close": [1.0, 2.0]})))

    assert got["close"][1].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("data", [pa.array([1.0, None]),
                                  pa.chunked_array([[1.0], [None]]),
                                  OnlyAnArray(pa.array([1.0, None]))])
def test_an_array_of_anything_but_a_table_is_one_column(data):
    [(name, kind, values, detail)] = arrow_columns(data)

    assert (name, kind, detail) == ("", "float", "")
    assert values[0] == 1.0 and np.isnan(values[1])


def test_a_pandas_frame_hands_its_datetime_index_over_as_a_time_column():
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC", name="datetime")
    got   = columns(pd.DataFrame({"close": [1.0, 2.0]}, index=index))

    assert got["datetime"][0] == "time" and got["datetime"][2] == "UTC"
    assert got["datetime"][1].tolist() == index.as_unit("ns").asi8.tolist()
    assert got["close"][1].tolist()    == [1.0, 2.0]


def test_a_polars_frame_hands_its_columns_over():
    times = polars.datetime_range(datetime.datetime(2024, 1, 1),
                                  datetime.datetime(2024, 1, 1, 1), "1h",
                                  time_zone="UTC", eager=True)

    got = columns(polars.DataFrame({"time": times, "close": [1.0, 2.0]}))

    assert got["time"][0] == "time" and got["time"][2] == "UTC"
    assert got["time"][1].tolist() == [1_704_067_200_000_000_000,
                                       1_704_070_800_000_000_000]
    assert got["close"][1].tolist() == [1.0, 2.0]
