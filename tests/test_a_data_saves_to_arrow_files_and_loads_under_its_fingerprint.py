"""A Data saves to a directory the user names, as three Arrow IPC files any Arrow reader
opens, and loads back under the same fingerprint, a slice with the bars before it; it
is refused where its bars no longer give that fingerprint or are not bars a Data holds
(ADR 0005). probatum writes and reads the files itself, so saving needs no pyarrow.
"""

import json
import re

import numpy as np
import pandas as pd
import polars
import pyarrow as pa
import pytest

import probatum
from probatum import DataError, ProbatumValueError
from probatum._data import features


def frame(n=24, close=100.0, start="2024-01-01"):
    closes = close + np.arange(n, dtype=np.float64)
    index  = pd.date_range(start, periods=n, freq="h", tz="UTC", name="datetime")

    return pd.DataFrame({"open": closes, "high": closes + 1.0, "low": closes - 1.0,
                         "close": closes, "volume": 1_000.0 + np.arange(n)},
                        index=index)


def routed(symbol, close, rows, market="linear", **attrs):
    out = frame(close=close).iloc[rows]
    out.attrs.update(dict(exchange="binance", market_type=market, symbol=symbol,
                          interval="1h", quote="USDT", warnings=[f"served {symbol}"]),
                     **attrs)
    return out


def universe():
    # a coin listed late and missing a bar, and one named in characters past ASCII
    late = "币安人生USDT"
    return probatum.Data({
        "BTCUSDT": routed("BTCUSDT", 100.0, slice(None)),
        late:      routed(late, 5.0, [3, 4, 6, 7, 8]),
    })


def rewritten(path, change):
    table = change(pa.ipc.open_file(path).read_all())
    with pa.ipc.new_file(path, table.schema) as writer:
        writer.write_table(table)


def test_a_data_comes_back_as_it_was_saved(tmp_path):
    data = universe()
    data.save(tmp_path / "pair")
    loaded = probatum.Data.load(tmp_path / "pair")

    assert loaded.fingerprint == data.fingerprint
    np.testing.assert_array_equal(loaded.bars, data.bars)
    assert loaded.time.tolist() == data.time.tolist()
    assert (loaded.names, loaded.instruments, loaded.interval) == (data.names,
                                                                  data.instruments,
                                                                  data.interval)
    assert loaded.provenance == data.provenance
    assert not loaded.bars.flags.writeable
    assert sorted(path.name for path in (tmp_path / "pair").iterdir()) == [
        "bars.arrow", "funding.arrow", "instruments.arrow"]


def test_a_data_of_one_instrument_holds_its_clock_apart_from_the_file_it_read(tmp_path):
    probatum.Data(frame(), symbol="BTCUSDT", market="linear").save(tmp_path / "one")

    # a view of the file's bytes would keep all of them to hold the clock
    assert probatum.Data.load(tmp_path / "one")._root.time.flags.owndata


def test_a_slice_saves_the_bars_before_it_and_loads_back_as_the_same_slice(tmp_path):
    late = universe()["币安人生USDT"][5:9]
    late.save(tmp_path / "late")
    loaded = probatum.Data.load(tmp_path / "late")

    assert len(loaded) == 4 and loaded.names == ("币安人生USDT",)
    assert loaded.fingerprint == late.fingerprint

    # a count of the bars printed so far reads the bars before the slice
    counted = lambda history: 1.0 * np.cumsum(history.available, axis=0)[:, :, None]

    assert features(loaded, counted)[:, 0, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    np.testing.assert_array_equal(features(loaded, counted), features(late, counted))


def test_a_selection_loads_back_under_its_own_names(tmp_path):
    spot = probatum.Data(routed("BTCUSDT", 100.0, slice(None), market="spot"))
    both = spot.join(probatum.Data(routed("BTCUSDT", 101.0, slice(None))))
    both.select(["linear:BTCUSDT"]).save(tmp_path / "perp")

    assert probatum.Data.load(tmp_path / "perp").names == ("linear:BTCUSDT",)

    flipped = universe().select(["币安人生USDT", "BTCUSDT"])
    flipped.save(tmp_path / "flipped")
    loaded = probatum.Data.load(tmp_path / "flipped")

    assert loaded.names == flipped.names and loaded.fingerprint == flipped.fingerprint


def test_every_field_of_every_instrument_comes_back(tmp_path):
    inverse = routed("BTCUSD_PERP", 100.0, slice(None), market="inverse", quote="USD",
                     volume_unit="contract", contract_size=100.0)
    data = probatum.Data({"BTCUSD_PERP": inverse, "PLAIN": frame(close=50.0)})
    data.save(tmp_path / "mixed")
    loaded = probatum.Data.load(tmp_path / "mixed")

    assert loaded.instruments == data.instruments
    assert loaded.instruments[0].contract == 100.0

    listed = pa.ipc.open_file(tmp_path / "mixed" / "instruments.arrow").read_all()

    assert listed["venue"].to_pylist()    == ["binance", None]
    assert listed["contract"].to_pylist() == [100.0, None]


def test_any_arrow_reader_opens_the_files(tmp_path):
    data = universe()
    data.save(tmp_path / "pair")
    bars = pa.ipc.open_file(tmp_path / "pair" / "bars.arrow").read_all()

    assert bars.schema.field("time").type       == pa.timestamp("ns", tz="UTC")
    assert bars.schema.field("instrument").type == pa.int32()
    assert all(field.nullable for field in bars.schema)
    assert [bars.schema.field(f).type for f in ("open", "close")] == [pa.float64()] * 2
    assert bars.num_rows == 2 * len(data)
    assert bars["instrument"].to_pylist()[:4] == [0, 1, 0, 1]
    assert bars["close"].null_count == int((~data.available).sum())

    record = json.loads(bars.schema.metadata[b"probatum"])

    assert (record["fingerprint"], record["interval"]) == (data.fingerprint, "1h")

    listed = polars.read_ipc(tmp_path / "pair" / "instruments.arrow")

    assert listed["name"].to_list()        == ["BTCUSDT", "币安人生USDT"]
    assert listed["contract"].null_count() == 2

    closes = polars.read_ipc(tmp_path / "pair" / "bars.arrow")["close"].to_numpy()

    np.testing.assert_array_equal(closes, data.closes.reshape(-1))

    funding = pa.ipc.open_file(tmp_path / "pair" / "funding.arrow").read_all()

    assert funding.num_rows == 0

    # the stream inside the file ends with its marker, before the footer
    raw    = (tmp_path / "pair" / "bars.arrow").read_bytes()
    footer = int.from_bytes(raw[-10:-6], "little")

    assert raw[-18 - footer:-10 - footer] == b"\xff\xff\xff\xff\x00\x00\x00\x00"


def test_bars_changed_after_they_were_saved_are_refused(tmp_path):
    universe().save(tmp_path / "pair")

    def nudged(table):
        closes = table["close"].to_numpy()
        closes[0] += 1e-9
        return table.set_column(table.schema.get_field_index("close"), "close",
                                pa.array(closes))

    rewritten(tmp_path / "pair" / "bars.arrow", nudged)

    with pytest.raises(DataError, match=re.escape(
            f"path: the bars in {str(tmp_path / 'pair')!r} give the fingerprint b2:")):
        probatum.Data.load(tmp_path / "pair")


def test_saved_bars_that_are_not_bars_are_refused_by_their_path(tmp_path):
    universe().save(tmp_path / "pair")

    def crossed(table):
        return table.set_column(table.schema.get_field_index("high"), "high",
                                pa.array(table["low"].to_numpy() - 1.0))

    rewritten(tmp_path / "pair" / "bars.arrow", crossed)

    with pytest.raises(DataError, match=re.escape(
            "path: BTCUSDT's bar at 2024-01-01 00:00 is not a bar: open 100, high 98")):
        probatum.Data.load(tmp_path / "pair")


def test_files_written_by_another_hand_are_read_too(tmp_path):
    # pyarrow writes the same columns back with its own alignment, no bitmap where
    # nothing is missing, and zeros, not NaN, under the bars an instrument did not
    # print, which the bitmap alone marks
    data = universe()
    data.save(tmp_path / "pair")

    def nulled(table):
        for name in ("open", "high", "low", "close", "volume"):
            values = [None if np.isnan(v) else v for v in table[name].to_numpy()]
            table  = table.set_column(table.schema.get_field_index(name), name,
                                      pa.array(values, pa.float64()))

        return table

    rewritten(tmp_path / "pair" / "bars.arrow", nulled)
    for name in ("instruments.arrow", "funding.arrow"):
        rewritten(tmp_path / "pair" / name, lambda table: table)

    assert probatum.Data.load(tmp_path / "pair").fingerprint == data.fingerprint


def damaged(path):
    raw = bytearray((path / "bars.arrow").read_bytes())
    raw[-10:-6] = bytes(4)
    (path / "bars.arrow").write_bytes(raw)


@pytest.mark.parametrize("change, says", [
    (None, "holds no bars.arrow; give a directory a Data was saved into"),
    (lambda path: (path / "bars.arrow").write_bytes(b"not arrow"),
     "'s bars.arrow is not an Arrow IPC file; give a directory a Data was saved into"),
    (damaged, "'s bars.arrow is not an Arrow IPC file this reader can read"),
    (lambda path: rewritten(path / "bars.arrow", lambda table: table.set_column(
        1, "instrument", table["instrument"].dictionary_encode())),
     "'s bars.arrow holds a column 'instrument' of a type this reader does not read"),
    (lambda path: rewritten(path / "bars.arrow",
                            lambda table: table.drop_columns(["volume"])),
     "'s bars.arrow has no float column volume; give a directory a Data was saved "
     "into"),
    (lambda path: rewritten(path / "bars.arrow",
                            lambda table: table.replace_schema_metadata(None)),
     "holds bars this version of probatum did not save; give a directory a Data was "
     "saved into"),
])
def test_a_load_it_cannot_make_is_refused(tmp_path, change, says):
    path = tmp_path / "pair"
    if change is not None:
        universe().save(path)
        change(path)

    with pytest.raises(ProbatumValueError, match=re.escape(says)) as caught:
        probatum.Data.load(path)

    # a file that is not what save writes is no fault in bars
    assert caught.type is ProbatumValueError


@pytest.mark.parametrize("make, error, says", [
    (lambda path: universe()[:0].save(path), DataError,
     "data: no bars to save; save a Data that has some"),
    (lambda path: (path.write_text("x"), universe().save(path)), ProbatumValueError,
     "is a file; name a directory"),
])
def test_a_save_it_cannot_make_is_refused(tmp_path, make, error, says):
    with pytest.raises(error, match=re.escape(says)) as caught:
        make(tmp_path / "pair")

    assert caught.type is error
