"""A Result saves as a directory of files any tool reads, a JSON manifest, numpy arrays
and Arrow IPC tables, and loads back by its format exactly as it was, whoever wrote the
files since: every array, table and statistic, its provenance, and its policy where that
was an array. A format this version does not read, files save does not write and files
that disagree with one another are refused, and a save that stops partway leaves either
the Result that was there or a directory load refuses, and nothing of its own where it
failed.
"""

import json
import os
import pathlib

import numpy as np
import polars
import pyarrow
import pyarrow.ipc
import pytest

import probatum
from probatum._result import SCHEMAS
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, bars, one, pair, weights)
from test_a_result_hands_its_tables_to_any_arrow_reader import inverse, takeover

TABLES = [f"{name}.arrow" for name in SCHEMAS] + ["stats.arrow"]
ARRAYS = ["time.npy", "equity.npy", "returns.npy"]


class Twice(probatum.Strategy):
    """A long of BTCUSDT every twelve bars, closed six bars later."""


    def next(self, bar, account):
        if bar.i % 12 == 2:
            account["BTCUSDT"].buy(1.0)
        elif bar.i % 12 == 8:
            account["BTCUSDT"].close()


def made():
    """A Result of each kind a save meets: an array run, a Strategy, a loop stepped by
    hand with a numpy seed, a run that fills nothing, an inverse contract, a takeover,
    a winner whose profit factor is infinite, and a run of one row, too short for
    statistics."""
    data     = pair()
    array    = probatum.backtest(weights(), data, VENUE)
    strategy = probatum.backtest(Twice(), data, VENUE)
    engine   = probatum.Engine(data, VENUE.replace(intrabar="random"), seed=np.int64(3))
    bar      = engine.reset()

    while not engine.done():
        if bar.i % 5 == 1:
            engine.target({"BTCUSDT": 0.4, "ETHUSDT": -0.2})
        bar = engine.step()

    rows = np.full(40, np.nan)
    rows[14], rows[23] = 1.0, 0.0
    winner = probatum.backtest(rows, one(), VENUE)
    single = probatum.Data(bars(n=1), symbol="BTCUSDT", market="linear", interval="1h")

    return dict(array=array, strategy=strategy, loop=engine.result(),
                idle=probatum.backtest(np.full((40, 2), np.nan), data, VENUE),
                inverse=inverse(), takeover=takeover(), winner=winner,
                single=probatum.backtest(np.full(1, 0.5), single, VENUE))


@pytest.fixture(scope="module")
def results():
    return made()


def same(loaded, result):
    for name in ("time", "equity", "returns"):
        assert getattr(loaded, name).tobytes() == getattr(result, name).tobytes()
        assert not getattr(loaded, name).flags.writeable

    for name in SCHEMAS:
        assert getattr(loaded, name) == getattr(result, name), name

    assert loaded.stats == result.stats
    if result.stats is not None:
        assert [type(value) for value in loaded.stats.values()] == [
            type(value) for value in result.stats.values()]

    assert (loaded.bust, loaded.liquidations) == (result.bust, result.liquidations)
    assert loaded.provenance == result.provenance
    assert repr(loaded)      == repr(result)


@pytest.mark.parametrize("kind", ["array", "strategy", "loop", "idle", "inverse",
                                  "takeover", "winner", "single"])
def test_a_result_comes_back_as_it_was_saved(kind, results, tmp_path):
    result = results[kind]
    result.save(tmp_path)
    loaded = probatum.Result.load(tmp_path)

    same(loaded, result)
    if isinstance(result.policy, np.ndarray):
        np.testing.assert_array_equal(loaded.policy, result.policy)
        assert not loaded.policy.flags.writeable
    else:
        assert loaded.policy is None

    assert results["winner"].stats["profit_factor"] == float("inf")
    assert results["single"].stats is None


def test_the_directory_holds_open_files_any_tool_reads(results, tmp_path):
    result = results["array"]
    result.save(tmp_path)

    assert sorted(os.listdir(tmp_path)) == sorted(TABLES + ARRAYS
                                                  + ["policy.npy", "result.json"])

    for name in ARRAYS + ["policy.npy"]:
        np.load(tmp_path / name, allow_pickle=False)

    def refused(token):
        raise ValueError(token)

    manifest = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"),
                          parse_constant=refused)

    assert (manifest["format"], manifest["policy"], manifest["bust"]) == (1, "array",
                                                                         False)

    for name in SCHEMAS:
        written = pyarrow.ipc.open_file(tmp_path / f"{name}.arrow").read_all()
        written.validate(full=True)
        assert written.equals(pyarrow.table(getattr(result, name)))
        assert polars.read_ipc(tmp_path / f"{name}.arrow").equals(
            polars.DataFrame(getattr(result, name)))

    stats = pyarrow.ipc.open_file(tmp_path / "stats.arrow").read_all().to_pylist()

    assert stats == [result.stats]

    results["winner"].save(tmp_path / "winner")
    kept = pyarrow.ipc.open_file(tmp_path / "winner" / "stats.arrow").read_all()

    assert kept["profit_factor"].to_pylist() == [float("inf")]


def rewritten(path, name, change=None):
    """The table ``name`` of the directory ``path`` written again by pyarrow in batches
    of three rows, every float rebuilt from its values with nothing under a null, and
    ``change`` made to it."""
    file  = path / f"{name}.arrow"
    table = pyarrow.ipc.open_file(file).read_all()

    for k, field in enumerate(table.schema):
        if pyarrow.types.is_floating(field.type):
            rebuilt = pyarrow.array(table[field.name].to_pylist(), pyarrow.float64())
            table   = table.set_column(k, field, rebuilt)

    if change is not None:
        table = change(table)

    with pyarrow.ipc.new_file(file, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=3):
            writer.write_batch(batch)


def test_files_written_again_by_another_hand_load_the_same(results, tmp_path):
    result = results["takeover"]
    result.save(tmp_path)

    for name in SCHEMAS:
        rewritten(tmp_path, name, lambda table: table.select(table.column_names[::-1]))

    np.save(tmp_path / "equity.npy", np.load(tmp_path / "equity.npy"))
    manifest = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    (tmp_path / "result.json").write_text(json.dumps(manifest, sort_keys=True),
                                          encoding="utf-8")

    loaded = probatum.Result.load(tmp_path)

    same(loaded, result)
    assert np.isnan(loaded.trades["exit"][loaded.trades.open]).all()


def test_a_column_a_later_version_adds_comes_back_beside_the_ones_this_one_reads(
        results, tmp_path):
    result = results["array"]
    result.save(tmp_path)

    def added(table):
        count = len(table)
        table = table.append_column("venue_fee_tier", pyarrow.array(
            ["vip0"] * count, pyarrow.large_string()))
        table = table.append_column(pyarrow.field("seq", pyarrow.int64(), nullable=False),
                                    pyarrow.array(range(count), pyarrow.int64()))
        return table.append_column("rebate", pyarrow.array(
            [None] + [0.5] * (count - 1), pyarrow.float64()))

    rewritten(tmp_path, "fills", added)
    rewritten(tmp_path, "stats", lambda table: table.append_column(
        "kelly", pyarrow.array([0.25], pyarrow.float64())))

    loaded = probatum.Result.load(tmp_path)
    fills  = loaded.fills

    assert fills.columns[-3:] == ("venue_fee_tier", "seq", "rebate")
    assert set(fills.venue_fee_tier) == {"vip0"} and fills.seq.tolist() == list(
        range(len(fills)))
    assert type(fills.seq) is np.ndarray and np.isnan(fills.rebate[0])

    written = pyarrow.table(fills).schema

    assert not written.field("seq").nullable and written.field("rebate").nullable
    assert [{key: row[key] for key in result.fills.columns} for row in fills] == list(
        result.fills)
    assert loaded.stats == result.stats


def test_a_result_saved_again_writes_the_same_bytes(results, tmp_path):
    result = results["strategy"]
    result.save(tmp_path / "a")
    probatum.Result.load(tmp_path / "a").save(tmp_path / "b")
    result.save(tmp_path / "c")

    for name in TABLES + ARRAYS + ["result.json"]:
        first = (tmp_path / "a" / name).read_bytes()
        assert first == (tmp_path / "b" / name).read_bytes() == (
            tmp_path / "c" / name).read_bytes(), name


def manifest_of(path, **changed):
    """The manifest of the directory ``path`` written again with ``changed``, a key of
    None left out."""
    file     = path / "result.json"
    manifest = json.loads(file.read_text(encoding="utf-8"))
    manifest.update(changed)
    file.write_text(json.dumps({key: value for key, value in manifest.items()
                                if value is not None}), encoding="utf-8")


def test_a_format_this_version_does_not_read_is_refused(results, tmp_path):
    results["array"].save(tmp_path)
    manifest_of(tmp_path, format=2, version="9.9.9")

    with pytest.raises(ValueError, match=r"holds a Result saved in format 2 by probatum "
                                         r"9\.9\.9, and this version, .*, reads format "
                                         r"1; load it with the version that saved it"):
        probatum.Result.load(tmp_path)

    for given in (0, "1", 1.0, True, None):
        manifest_of(tmp_path, format=given)
        with pytest.raises(ValueError, match="result.json is not a manifest save writes"
                           if not isinstance(given, int) or isinstance(given, bool)
                           else "saved in format 0"):
            probatum.Result.load(tmp_path)

    (tmp_path / "result.json").write_text('{"format": 1, "bust": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="is not a manifest save writes"):
        probatum.Result.load(tmp_path)


CLOCK    = "do not hold the rows of its time.npy and equity.npy"
MANIFEST = "result.json is not a manifest save writes"


@pytest.mark.parametrize("broken, said", [
    ("no manifest", "holds no result.json"),
    ("no policy", MANIFEST),
    ("another policy", MANIFEST),
    ("an empty provenance", MANIFEST),
    ("no table", "holds no orders.arrow"),
    ("no column", "fills.arrow has no int column order"),
    ("another kind", "fills.arrow has no int column order"),
    ("a short curve", "are not a row each of one clock"),
    ("float times", "are not a row each of one clock"),
    ("a single time", "are not a row each of one clock"),
    ("a policy of text", "are not a row each of one clock"),
    ("objects", "equity.npy is not an array of numbers save writes"),
    ("an empty array", "time.npy is not an array of numbers save writes"),
    ("a zip", "time.npy is not an array of numbers save writes"),
    ("other returns", "returns.npy does not hold the returns"),
    ("another run's ledger", CLOCK),
    ("other times", CLOCK),
    ("a bar past the clock", CLOCK),
    ("a bar at the clock's end", CLOCK),
    ("a bar before the clock", CLOCK),
    ("holdings by instrument", CLOCK),
    ("fills off their bars", CLOCK),
    ("two rows of statistics", "stats.arrow is not the row of statistics"),
    ("a count of another kind", "stats.arrow is not the row of statistics"),
    ("a missing statistic", "stats.arrow is not the row of statistics"),
    ("another run's statistics", "stats.arrow does not count the fills and the round "
                                 "trips its tables hold"),
    ("a null bar", "fills.arrow holds a null in bar, a column that holds none"),
    ("a misaligned body", "ledger.arrow holds a buffer off the 8-byte boundary the "
                          "format puts it on"),
])
def test_files_save_does_not_write_are_refused(broken, said, results, tmp_path,
                                               monkeypatch):
    results["array"].save(tmp_path)
    other = tmp_path / "other"
    results["strategy"].save(other)
    time = np.load(tmp_path / "time.npy")

    if broken == "no manifest":
        (tmp_path / "result.json").unlink()
    elif broken == "no policy":
        manifest_of(tmp_path, policy=None)
    elif broken == "another policy":
        manifest_of(tmp_path, policy="strategy")
    elif broken == "an empty provenance":
        manifest_of(tmp_path, provenance={})
    elif broken == "no table":
        (tmp_path / "orders.arrow").unlink()
    elif broken == "no column":
        rewritten(tmp_path, "fills", lambda table: table.drop_columns(["order"]))
    elif broken == "another kind":
        rewritten(tmp_path, "fills", lambda table: table.set_column(
            3, "order", table["order"].cast(pyarrow.float64())))
    elif broken == "a short curve":
        np.save(tmp_path / "equity.npy", np.load(tmp_path / "equity.npy")[:-1])
    elif broken == "float times":
        np.save(tmp_path / "time.npy", time.astype(float))
    elif broken == "a single time":
        np.save(tmp_path / "time.npy", time[0])
    elif broken == "a policy of text":
        np.save(tmp_path / "policy.npy", np.array(["hold"] * len(time)))
    elif broken == "objects":
        np.save(tmp_path / "equity.npy", np.array([object()] * 40), allow_pickle=True)
    elif broken == "an empty array":
        (tmp_path / "time.npy").write_bytes(b"")
    elif broken == "a zip":
        with open(tmp_path / "time.npy", "wb") as file:
            np.savez(file, time=time)
    elif broken == "other returns":
        np.save(tmp_path / "returns.npy", np.load(tmp_path / "returns.npy") + 1e-9)
    elif broken == "another run's ledger":
        os.replace(other / "ledger.arrow", tmp_path / "ledger.arrow")
    elif broken == "other times":
        np.save(tmp_path / "time.npy", time + 1)
    elif broken.startswith("a bar "):
        bar = {
            "a bar past the clock":     999,
            "a bar at the clock's end": len(time),
            "a bar before the clock":   -1,
        }[broken]
        rewritten(tmp_path, "fills", lambda table: table.set_column(
            0, "bar", pyarrow.array([bar] * len(table), pyarrow.int64())))
    elif broken == "holdings by instrument":
        rewritten(tmp_path, "holdings", lambda table: table.sort_by("instrument"))
    elif broken == "fills off their bars":
        rewritten(tmp_path, "fills", lambda table: table.set_column(
            1, "time", pyarrow.array(np.asarray(table["time"].cast(pyarrow.int64())) + 1,
                                     pyarrow.timestamp("ns", "UTC"))))
    elif broken == "two rows of statistics":
        rewritten(tmp_path, "stats", lambda table: pyarrow.concat_tables([table, table]))
    elif broken == "a count of another kind":
        rewritten(tmp_path, "stats", lambda table: table.set_column(
            11, "num_trades", table["num_trades"].cast(pyarrow.float64())))
    elif broken == "a missing statistic":
        rewritten(tmp_path, "stats", lambda table: table.set_column(
            3, "sharpe", pyarrow.array([None], pyarrow.float64())))
    elif broken == "another run's statistics":
        os.replace(other / "stats.arrow", tmp_path / "stats.arrow")
    elif broken == "a null bar":
        rewritten(tmp_path, "fills", lambda table: table.set_column(
            0, pyarrow.field("bar", pyarrow.int64()),
            pyarrow.array([None] + table["bar"].to_pylist()[1:], pyarrow.int64())))
    elif broken == "a misaligned body":
        framed, calls = probatum._ipc._framed, []

        def shifted(flatbuffer):
            # the record batch's metadata four bytes longer, so its body starts off the
            # boundary
            calls.append(flatbuffer)
            return framed(flatbuffer) + bytes(4 if len(calls) == 2 else 0)

        monkeypatch.setattr(probatum._ipc, "_framed", shifted)
        probatum._ipc.write(tmp_path / "ledger.arrow",
                            probatum._result._stored(results["array"].ledger), {})

    with pytest.raises(ValueError, match=said):
        probatum.Result.load(tmp_path)


def test_text_whose_offsets_run_past_its_bytes_is_refused(tmp_path):
    # a reader that joins batches would splice the next one's bytes into this one's
    past = (np.array([0, 20], dtype=np.int64), np.frombuffer(b"abc", dtype=np.uint8))

    probatum._ipc.write(tmp_path / "x.arrow", [("reason", "large_text", past, None)], {})

    with pytest.raises(probatum._ipc.Refused, match="holds text in 'reason' whose "
                                                    "offsets run past its bytes"):
        probatum._ipc.read(tmp_path / "x.arrow")


def test_a_save_that_stops_partway_leaves_the_result_that_was_there_or_none(
        results, tmp_path, monkeypatch):
    array, strategy = results["array"], results["strategy"]
    array.save(tmp_path)

    def broken(*given, **told):
        raise OSError("the disk is full")

    with monkeypatch.context() as patched:
        patched.setattr(probatum._ipc, "write", broken)
        with pytest.raises(OSError):
            array.save(tmp_path)

    same(probatum.Result.load(tmp_path), array)
    # what the save that failed wrote, its policy.npy among it, went with it
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".partial")]

    calls   = []
    replace = os.replace

    def stopped(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise OSError("the disk is full")
        replace(source, target)

    with monkeypatch.context() as patched:
        patched.setattr(probatum._result.os, "replace", stopped)
        with pytest.raises(OSError):
            strategy.save(tmp_path)

    with pytest.raises(ValueError, match="holds no result.json"):
        probatum.Result.load(tmp_path)

    strategy.save(tmp_path)

    assert not (tmp_path / "policy.npy").exists()
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".partial")]
    same(probatum.Result.load(tmp_path), strategy)
    with pytest.raises(ValueError, match="path: .* is a file; name a directory"):
        array.save(tmp_path / "result.json")
