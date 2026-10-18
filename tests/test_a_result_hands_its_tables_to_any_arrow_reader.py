"""A run's tables reach polars, pyarrow, DuckDB and pandas through the Arrow C stream
interface with nothing imported by probatum. Each holds the run's records, every bar
counted from the Result's first row, each instrument by its name and every value the
run did not record a null, each column of the type its schema names, and a table reads
like the lists of records it replaced. Every read is a stream of its own, which the
reader keeps after the table and the Result are gone; and the provenance says what the
run loaded and where it ran without importing anything to ask.
"""

import copy
import gc
import json
import math
import pickle
import platform
import re
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import polars
import pyarrow
import pyarrow.compute
import pytest

import probatum
from probatum._result import SCHEMAS, OrderLog, Table
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, bars, one, pair, weights)
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import busy, gapped
from test_every_depth_gives_one_ledger import rows_of, tabled

ARROW = {
    "float": pyarrow.float64(),
    "int":   pyarrow.int64(),
    "time":  pyarrow.timestamp("ns", "UTC"),
    "bool":  pyarrow.bool_(),
    "text":  pyarrow.large_string(),
}
POLARS = {
    "float": polars.Float64,
    "int":   polars.Int64,
    "time":  polars.Datetime("ns", "UTC"),
    "bool":  polars.Boolean,
    "text":  polars.String,
}
DUCKDB = {
    "float": "DOUBLE",
    "int":   "BIGINT",
    "time":  "TIMESTAMP WITH TIME ZONE",
    "bool":  "BOOLEAN",
    "text":  "VARCHAR",
}


def inverse():
    """One coin of a BTCUSD perpetual, half of it short at 50,000, as the router gives
    an inverse contract's bars."""
    frame = bars(level=50_000.0)
    frame.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                       interval="1h", quote="USD", contract_size=100.0, warnings=[],
                       volume_unit="contract")

    return probatum.backtest(np.full(40, -0.5), probatum.Data(frame),
                             probatum.Venue(balance=1.0))


def takeover():
    """Ten times levered into a fall of a tenth, closed out on two instruments, so a
    fill names no order."""
    rows = np.full((40, 2), np.nan)
    rows[3] = 5.0

    return probatum.backtest(rows, pair(), VENUE.replace(leverage=0.0))


def clamped():
    """The binding's engine after a transfer larger than the wallet it leaves, which the
    order log holds with no order and no instrument."""
    closes  = 100.0 + np.arange(12, dtype=np.float64)
    candles = np.stack([np.column_stack([closes, closes + 1.0, closes - 1.0, closes,
                                         np.full(12, 1e6)])] * 2, axis=1)

    times  = np.arange(12, dtype=np.int64) * 3_600_000_000_000
    engine = probatum._probatum.Engine(
        candles,
        report      = True,
        times       = times,
        quote       = {"spot": 1_000.0, "margin": 500.0},
        instruments = [{"market": "spot"}, {"market": "perp"}],
    )

    engine.reset()
    engine.target({1: 0.3}, 0.0, False)
    engine.step()
    engine.transfer(5_000.0, "spot")
    engine.step()

    return engine


def runs():
    """Results of every kind of record: a universe of NaN holds, a slice whose rows
    count from 10, an inverse contract, a takeover, and a run that fills nothing."""
    data = pair()
    return {
        "pair":     probatum.backtest(weights(), data, VENUE),
        "slice":    probatum.backtest(weights(n=50, width=1), gapped()[10:], VENUE),
        "inverse":  inverse(),
        "takeover": takeover(),
        "idle":     probatum.backtest(np.full((40, 2), np.nan), data, VENUE),
    }


@pytest.fixture(scope="module")
def results():
    return runs()


@pytest.mark.parametrize("seed", range(6))
def test_every_table_holds_the_binding_s_records_of_its_run(seed):
    strategy, engine, _ = busy(seed)

    assert tabled(engine, strategy.names) == rows_of(engine, strategy.names)

    names  = ["cash", "perp"]
    held   = clamped()
    events = rows_of(held, names)["orders"]

    assert any(event["order"] is None and event["instrument"] is None
               for event in events)
    assert tabled(held, names) == rows_of(held, names)
    assert tabled(held, names, start=1)["fills"][0]["bar"] == 0


def test_every_bar_counts_from_the_result_s_first_row(results):
    for result in results.values():
        assert result.ledger["bar"].tolist() == list(range(len(result.time)))

        for name in ("holdings", "fills", "orders"):
            table = getattr(result, name)
            np.testing.assert_array_equal(result.time[table["bar"]], table["time"])

        trades = result.trades
        np.testing.assert_array_equal(result.time[trades.open_bar], trades.open_time)
        closed = trades[~trades.open]
        np.testing.assert_array_equal(result.time[closed.close_bar], closed.close_time)

    sliced = results["slice"]

    assert len(sliced.time) == np.count_nonzero(gapped()[10:].available[:, 0])


def test_a_value_the_run_did_not_record_is_a_null_in_every_reader(results):
    for result in results.values():
        for name, schema in SCHEMAS.items():
            table = getattr(result, name)
            rows  = list(table)
            arrow = pyarrow.table(table)
            frame = polars.DataFrame(table)

            for column, kind, nullable in schema:
                missing = sum(row[column] is None for row in rows)
                assert arrow[column].null_count == frame[column].null_count() == missing
                assert nullable or not missing
                if kind == "float":
                    assert not pyarrow.compute.any(
                        pyarrow.compute.is_nan(arrow[column])).as_py()

    trades = results["pair"].trades

    assert trades[-1]["open"] and trades[-1]["close_bar"] is None
    assert trades[-1]["close_time"] is None and trades[-1]["reason"] is None

    forced = results["takeover"].fills[results["takeover"].fills.reason == "liquidation"]

    assert len(forced) == 2 and all(fill["order"] is None for fill in forced)


def test_each_column_has_the_type_its_schema_names(results):
    for name, schema in SCHEMAS.items():
        table = getattr(results["pair"], name)
        assert table.columns == tuple(column for column, _, _ in schema)

        expected = pyarrow.schema([pyarrow.field(column, ARROW[kind], nullable)
                                   for column, kind, nullable in schema])
        assert pyarrow.table(table).schema == expected
        assert dict(polars.DataFrame(table).schema) == {column: POLARS[kind]
                                                        for column, kind, _ in schema}

        scanned   = table
        described = duckdb.sql("describe select * from scanned").fetchall()
        assert [(row[0], row[1]) for row in described] == [
            (column, DUCKDB[kind]) for column, kind, _ in schema]


def test_every_reader_reads_the_rows_a_table_holds(results):
    result = results["pair"]

    for name in SCHEMAS:
        table = getattr(result, name)
        rows  = list(table)
        times = [column for column, kind, _ in SCHEMAS[name] if kind == "time"]
        arrow = pyarrow.table(table)
        for column in times:
            arrow = arrow.set_column(arrow.column_names.index(column), column,
                                     arrow[column].cast(pyarrow.int64()))

        assert arrow.to_pylist() == rows

        frame = polars.DataFrame(table).with_columns(
            [polars.col(column).cast(polars.Int64) for column in times])
        assert frame.to_dicts() == rows

        epoch = ", ".join(f"epoch_ns({column}) as {column}" if column in times
                          else f'"{column}"' for column, _, _ in SCHEMAS[name])
        scanned = table
        fetched = duckdb.sql(f"select {epoch} from scanned").fetchall()
        assert [dict(zip(table.columns, row)) for row in fetched] == rows

        pandas = table.to_pandas()
        for column, kind, nullable in SCHEMAS[name]:
            values = pandas[column]
            if kind == "time":
                assert [None if pd.isna(value) else value.value for value in values] == [
                    row[column] for row in rows]
            else:
                assert [None if pd.isna(value) else value
                        for value in values.astype(object)] == [row[column] for row in rows]

        assert table.to_polars().equals(polars.DataFrame(table))

    trades = result.trades.to_pandas()

    assert str(trades["close_bar"].dtype) == "Int64"
    assert str(trades["open_bar"].dtype)  == "int64"
    assert str(trades["open_time"].dtype) == "datetime64[ns, UTC]"

    framed = result.to_pandas()

    assert framed.index.name == "time" and str(framed.index.tz) == "UTC"
    assert framed.index.asi8.tolist() == result.time.tolist()
    np.testing.assert_array_equal(framed["equity"], result.equity)
    np.testing.assert_array_equal(framed["returns"], result.returns)

    shown = result.to_polars()

    assert dict(shown.schema) == {
        "time":    polars.Datetime("ns", "UTC"),
        "equity":  polars.Float64,
        "returns": polars.Float64,
    }
    np.testing.assert_array_equal(shown["returns"].to_numpy(), result.returns)


def test_a_table_streams_anew_each_read_and_a_spent_capsule_is_refused(results):
    fills = results["pair"].fills

    assert pyarrow.table(fills).equals(pyarrow.table(fills))

    joined = duckdb.sql("select count(*) from fills a join fills b on a.bar = b.bar")

    assert joined.fetchone()[0] >= len(fills)

    capsule = fills.__arrow_c_stream__()
    pyarrow.RecordBatchReader._import_from_c_capsule(capsule).read_all()

    with pytest.raises(pyarrow.ArrowInvalid):
        pyarrow.RecordBatchReader._import_from_c_capsule(capsule)

    for _ in range(50):
        fills.__arrow_c_stream__()
    gc.collect()

    assert len(pyarrow.table(fills)) == len(fills)


def test_what_a_reader_keeps_outlives_the_table_and_the_result():
    result   = probatum.backtest(weights(), pair(), VENUE)
    rows     = list(result.trades)
    reader   = pyarrow.RecordBatchReader.from_stream(result.trades)
    frame    = polars.DataFrame(result.fills)
    expected = len(result.fills)
    column   = result.holdings["position"]

    del result
    gc.collect()

    assert len(reader.read_all()) == len(rows)
    assert frame.height == expected and frame["price"].sum() > 0.0
    assert np.isfinite(column).any()


def test_a_requested_schema_is_met_or_refused(results):
    trades = results["pair"].trades
    own    = pyarrow.table(trades).schema

    assert pyarrow.RecordBatchReader.from_stream(trades, schema=own).read_all().equals(
        pyarrow.table(trades))

    renamed = own.set(0, own.field(0).with_name("symbol"))
    with pytest.raises(ValueError, match="requested_schema: the columns symbol, side"):
        trades.__arrow_c_stream__(renamed.__arrow_c_schema__())

    spent = own.__arrow_c_schema__()
    pyarrow.Schema._import_from_c_capsule(spent)

    with pytest.raises(ValueError, match="requested_schema: a schema already released; "
                                         "pass a live one, or None"):
        trades.__arrow_c_stream__(spent)
    with pytest.raises(TypeError, match="requested_schema: a int, not a capsule named "
                                        "arrow_schema"):
        trades.__arrow_c_stream__(42)


def test_a_run_that_fills_nothing_has_empty_tables_of_the_same_schema(results):
    idle = results["idle"]

    for name in ("fills", "orders", "trades"):
        table = getattr(idle, name)
        assert len(table) == 0 and not table and list(table) == []
        assert pyarrow.table(table).schema == pyarrow.table(
            getattr(results["pair"], name)).schema
        assert polars.DataFrame(table).height == 0
        scanned = table
        assert duckdb.sql("select count(*) from scanned").fetchone()[0] == 0
        assert len(table.to_pandas()) == 0

    assert len(idle.holdings) == 2 * len(idle.time)


def test_a_result_taken_mid_run_keeps_its_rows():
    engine = probatum.Engine(pair(), VENUE)
    bar    = engine.reset()

    while bar.i < 20:
        engine.target({"BTCUSDT": 0.5 if bar.i % 6 < 3 else -0.5})
        bar = engine.step()

    early = engine.result()
    kept  = json.dumps(early.to_dict())

    while not engine.done():
        engine.target({"ETHUSDT": 0.3})
        engine.step()

    engine.reset()

    assert json.dumps(early.to_dict()) == kept
    assert len(early.time) == 21


def test_a_table_reads_by_column_row_slice_and_mask(results):
    trades = results["pair"].trades
    count  = len(trades)

    assert trades and count > 3 and trades.columns[0] == "instrument"

    pnl = trades["pnl"]

    assert pnl is trades.pnl and not pnl.flags.writeable
    with pytest.raises(ValueError):
        pnl[0] = 1.0

    assert trades.instrument.dtype == object and not trades.open.flags.writeable
    assert trades[0] == next(iter(trades)) and trades[-1] == list(trades)[-1]
    with pytest.raises(IndexError, match=f"row: {count} of a table of {count} rows"):
        trades[count]

    assert list(trades[1:3]) == list(trades)[1:3]
    assert len(trades[trades.open]) == trades.open.sum() > 0
    assert type(trades[trades.open]) is Table
    assert list(trades[np.array([0, -1])]) == [trades[0], trades[-1]]

    late = trades[trades.close_bar > 20]

    assert len(late) and all(row["close_bar"] > 20 for row in late)

    close = trades.close_bar

    assert isinstance(close, np.ma.MaskedArray) and close.dtype == np.int64
    np.testing.assert_array_equal(np.ma.getmaskarray(close), trades.open)
    with pytest.raises(TypeError, match=r"'pnl' in table: a table holds rows, as dicts; "
                                        r"ask 'pnl' in table.columns"):
        "pnl" in trades

    assert trades[0] in trades
    assert trades == results["pair"].trades and trades != results["idle"].trades
    assert repr(trades).startswith(f"Table({count} rows: instrument, side, size")
    with pytest.raises(KeyError, match="table: no column 'symbol'; use one of "
                                       "instrument, side"):
        trades["symbol"]
    with pytest.raises(AttributeError, match="table: no column 'symbol'"):
        trades.symbol
    with pytest.raises(TypeError, match="key: 1.5; give a column's name"):
        trades[1.5]
    with pytest.raises(IndexError, match="key: a mask of 2 rows"):
        trades[np.array([True, False])]

    assert list(trades[::-2]) == list(trades)[::-2]
    assert len(trades[[]]) == 0 and trades[[]].columns == trades.columns
    with pytest.raises(IndexError, match=f"key: position {-count - 1} of a table of "
                                         f"{count} rows; give one from {-count} to "
                                         f"{count - 1}"):
        trades[np.array([0, -count - 1])]
    with pytest.raises(IndexError, match=re.escape(
            "row: 0 of a table with no rows; check len(table) first")):
        results["idle"].trades[0]


def test_a_column_that_can_miss_a_value_is_masked_and_a_masked_position_names_no_row(
        results):
    trades, ledger = results["pair"].trades, results["pair"].ledger

    assert type(trades.open_bar) is np.ndarray and type(ledger["bar"]) is np.ndarray

    for column in (trades.close_bar, trades.close_time):
        assert isinstance(column, np.ma.MaskedArray)
        np.testing.assert_array_equal(np.ma.getmaskarray(column), trades.open)

    order = results["takeover"].fills.order

    assert isinstance(order, np.ma.MaskedArray) and np.ma.count_masked(order) == 2
    with pytest.raises(IndexError, match=f"key: positions of which {trades.open.sum()} are "
                                         f"masked, and a masked position names no row"):
        ledger[trades.close_bar]

    closed = trades[~trades.open]

    assert list(ledger[closed.close_bar]) == [ledger[int(bar)] for bar in closed.close_bar]
    assert list(ledger[trades.close_bar.compressed()]) == list(ledger[closed.close_bar])


def test_positions_of_any_width_of_integer_read_the_rows_they_name():
    data   = probatum.Data(bars(n=300), symbol="BTCUSDT", market="linear")
    ledger = probatum.backtest(np.full(300, 0.5), data, VENUE).ledger

    assert list(ledger[np.array([1, -1], dtype=np.int8)]) == [ledger[1], ledger[-1]]
    assert list(ledger[np.array([299], dtype=np.uint16)]) == [ledger[299]]
    with pytest.raises(IndexError, match="key: position 18446744073709551615 of a table"):
        ledger[np.array([2**64 - 1], dtype=np.uint64)]


def test_a_column_and_every_reader_share_the_table_s_memory(results):
    fills = results["pair"].fills
    price = fills["price"]

    assert np.shares_memory(price, fills._held.column("price"))
    assert pyarrow.table(fills)["price"].chunk(0).buffers()[1].address == price.ctypes.data

    names = results["pair"].holdings["instrument"]

    assert names[0] == "BTCUSDT" and names[0] is names[2]


def test_the_binding_s_table_names_what_it_refuses_and_the_fix(results):
    held = results["pair"].trades._held
    rows = len(held)

    with pytest.raises(KeyError, match="name: no column \"symbol\"; use one of instrument"):
        held.column("symbol")
    with pytest.raises(IndexError, match=f"row: {rows} of a table of {rows} rows; give one "
                                         f"below {rows}"):
        held.row(rows)
    with pytest.raises(IndexError, match=f"indices: {rows} of a table of {rows} rows; give "
                                         f"positions from 0 to {rows - 1}"):
        held.take(np.array([rows]))

    engine = probatum._probatum.Engine(np.ones((3, 5)), report=True)
    engine.reset()

    with pytest.raises(ValueError, match="times: none, so no row of a table has a time; "
                                         "build the engine with times="):
        engine.tables(0, ["BTCUSDT"])

    # numpy lets a view put any byte in a bool array, and any byte but 0 reads as true
    flags = np.array([2, 0, 1], dtype=np.uint8).view(bool)
    made  = probatum._probatum.Table([("x", "bool", False, flags, None, None),
                                      ("y", "float", True, np.ones(3), flags, None)])

    assert made.column("x").tolist() == made.valid("y").tolist() == [True, False, True]


def test_a_round_trip_still_open_has_the_exit_of_what_closed_of_it():
    rows = np.full(40, np.nan)
    rows[3], rows[8] = 1.0, 0.5

    result = probatum.backtest(rows, one(), VENUE)
    last, sale = result.trades[-1], result.fills[-1]

    assert last["open"] and last["close_bar"] is None and last["reason"] is None
    assert sale["side"] == "sell" and last["exit"] == sale["price"]


def test_an_order_log_keeps_one_kind_of_event_and_refuses_another(results):
    orders = results["pair"].orders
    filled = orders.events("filled")

    assert type(filled) is OrderLog and len(filled) == len(results["pair"].fills)
    assert set(filled.kind) == {"filled"}
    assert list(filled)     == [row for row in orders if row["kind"] == "filled"]
    with pytest.raises(ValueError, match="kind: 'fill'; use one of placed, armed"):
        orders.events("fill")


def test_a_table_and_a_result_pickle_and_copy_as_they_were(results):
    result = results["pair"]

    for name in SCHEMAS:
        table = getattr(result, name)
        assert pickle.loads(pickle.dumps(table)) == table
        assert copy.deepcopy(table) is table and copy.copy(table) is table

    again = pickle.loads(pickle.dumps(result))

    np.testing.assert_array_equal(again.equity, result.equity)
    assert again.stats == result.stats and again.provenance == result.provenance
    assert again.ledger == result.ledger and not again.policy.flags.writeable

    for kept in (result, again, copy.deepcopy(result)):
        assert not any(getattr(kept, name).flags.writeable
                       for name in ("time", "equity", "returns"))

    class Held(probatum.Strategy):
        """A buy whose handle it keeps, which holds the run's engine."""


        def next(self, bar, account):
            if bar.i == 3:
                self.order = account.buy(1.0)

    ran  = probatum.backtest(Held(), one(), VENUE)
    kept = pickle.loads(pickle.dumps(ran))

    assert kept.policy is None and kept.trades == ran.trades

    # a copy needs no pickling, so it keeps the copy of the Strategy the run used
    assert copy.copy(ran).policy is ran.policy

    deep = copy.deepcopy(ran)

    assert type(deep.policy) is Held and deep.policy is not ran.policy
    assert deep.trades is ran.trades and deep.provenance == ran.provenance


def test_to_dict_is_plain_data_with_none_where_a_value_is_missing(results):
    result = results["takeover"]
    out    = result.to_dict()
    text   = json.dumps(out, allow_nan=False)

    assert json.loads(text) == out
    assert out["liquidations"] == result.liquidations == 1
    assert out["fills"]["order"].count(None) == 2
    assert out["equity"] == result.equity.tolist()
    assert results["pair"].to_dict()["trades"]["close_time"][-1] is None


def test_the_provenance_names_the_venue_its_markets_settlement_and_seed():
    frame = bars()
    frame.attrs.update(exchange="binance", market_type="linear", symbol="BTCUSDT",
                       interval="1h", quote="USDT", warnings=[])
    data  = probatum.Data(frame)
    venue = probatum.venues.binance_perp.replace(funding="none")

    given = probatum.backtest(np.full(40, 0.5), data, venue).provenance

    assert (given["venue_name"], given["markets"], given["settlement"]) == (
        "binance_perp", ["linear"], "USDT")
    assert (given["seed"], given["streams"]) == (None, {})
    assert given["python"] == platform.python_version()

    # every suite runs on an installed wheel, and each run is handed a list of its own
    assert given["wheel"] and all(isinstance(tag, str) for tag in given["wheel"])

    given["wheel"].append("edited")
    again = probatum.backtest(np.full(40, 0.5), data, venue).provenance

    assert again["wheel"] == given["wheel"][:-1]

    engine = probatum.Engine(data, venue.replace(intrabar="random"), seed=np.int64(7))
    engine.reset()
    ran = engine.result().provenance

    assert (ran["seed"], ran["streams"]) == (7, {"intrabar": "ChaCha20"})
    assert type(ran["seed"]) is int
    assert json.loads(json.dumps(ran)) == ran


def test_a_run_reads_and_writes_its_tables_without_importing_a_reader(tmp_path):
    script = f"""
import json, sys
for name in ("pyarrow", "polars", "pandas", "duckdb"):
    sys.modules[name] = None
import numpy as np
import probatum
close = 100.0 + np.arange(50, dtype=np.float64)
bars = np.column_stack([close, close + 1.0, close - 1.0, close, np.full(50, 1e3)])
hours = np.arange(50, dtype=np.int64) * 3_600_000_000_000
data = probatum.Data(bars, symbol="BTCUSDT", market="linear", time=hours)
result = probatum.backtest(np.full(50, 0.5), data, "frictionless")
result.trades.__arrow_c_stream__()
result.save({str(tmp_path)!r})
loaded = probatum.Result.load({str(tmp_path)!r})
import importlib.metadata as metadata
packages = result.provenance["packages"]
assert packages["numpy"] == metadata.version("numpy"), packages
assert not {{"pyarrow", "polars", "pandas", "duckdb"}} & set(packages), packages
assert loaded.provenance == result.provenance
print(json.dumps(sorted(packages)))
"""

    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert "numpy" in json.loads(done.stdout)


def test_to_pandas_needs_no_pyarrow():
    script = """
import sys
sys.modules["pyarrow"] = None
import numpy as np
import probatum
close = 100.0 + np.sin(np.arange(60, dtype=np.float64)) * 5.0
bars = np.column_stack([close, close + 1.0, close - 1.0, close, np.full(60, 1e3)])
hours = np.arange(60, dtype=np.int64) * 3_600_000_000_000
data = probatum.Data(bars, symbol="BTCUSDT", market="linear", time=hours)
rows = np.where(np.arange(60) % 10 < 5, 1.0, 0.0)
frame = probatum.backtest(rows, data, "frictionless").trades.to_pandas()
assert str(frame["close_bar"].dtype) == "Int64", frame.dtypes
assert str(frame["close_time"].dtype) == "datetime64[ns, UTC]", frame.dtypes
assert frame["close_bar"].isna().sum() == frame["open"].sum()
"""

    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("seed", range(6))
def test_what_each_instrument_made_adds_up_to_the_change_in_equity(seed):
    # on every row, and an instrument that ends flat on a run with no forced close or
    # dust has made what its round trips did, its funding included
    _, engine, _ = busy(seed)
    ledger = engine.ledger()
    pnl, equity = ledger["pnl"], ledger["equity"]
    scale = 1e-9 * (1.0 + np.abs(equity).max())

    np.testing.assert_allclose(pnl.sum(axis=1), equity - equity[0], rtol=0.0, atol=scale)

    if ledger["liquidation"].any() or ledger["dust"].any():
        return

    for k in range(pnl.shape[1]):
        if ledger["position"][-1, k] == 0.0:
            made = sum(trip["pnl"] for trip in engine.trades() if trip["instrument"] == k)
            assert made == pytest.approx(pnl[-1, k], abs=scale)
