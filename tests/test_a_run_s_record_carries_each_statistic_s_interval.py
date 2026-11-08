"""A run's record of its statistics reads as the dict it replaced and carries each
statistic's interval: the one probatum.stats computes from the Result, computed when
first read and kept, None where a statistic has none or its sample cannot support one,
and the same after a pickle, a save and a load (Specification, Inference).
"""

import math
import pickle
from collections.abc import Mapping

import numpy as np
import pytest

import probatum
from probatum import (ProbatumAttributeError, ProbatumKeyError, ProbatumTypeError,
                      ProbatumValueError, stats)


def run(held):
    rows  = held.size
    close = 100.0 + np.random.default_rng(9).normal(0.0, 1.0, rows).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(rows, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * 3_600_000_000_000,
    )

    return probatum.backtest(np.where(held, 1.0, 0.0), data,
                             probatum.venues.bybit_perp.replace(funding="none"))


@pytest.fixture(scope="module")
def traded():
    return run(np.random.default_rng(9).random(1_000) < 0.5)


def test_the_record_reads_as_the_dict_it_replaced(traded):
    record = traded.stats

    assert isinstance(record, Mapping) and isinstance(record, stats.Statistics)
    assert record == dict(record) and dict(record) == record and len(record) == 15
    assert record.sharpe == record["sharpe"] and record.get("sharp") is None
    assert pytest.approx(dict(record), nan_ok=True) == record

    with pytest.raises(ProbatumAttributeError, match="a run's record does not change; "
                       "run again for another"):
        record.sharpe = 1.0
    with pytest.raises(TypeError):
        record["sharpe"] = 1.0

    with pytest.raises(ProbatumAttributeError, match="Statistics.sharp: no such "
                       "statistic; give one of total_return_pct"):
        record.sharp


def test_each_statistic_s_interval_is_the_module_s_and_is_kept(traded):
    record = traded.stats

    assert record.interval("sharpe") == stats.sharpe_interval(traded)
    for name in ("total_return_pct", "sortino", "max_drawdown_pct", "calmar", "volatility_pct",
                 "win_rate", "profit_factor", "avg_trade_pct"):
        assert record.interval(name) == stats.bootstrap(traded, name)

    assert record.interval("total_return_pct") is record.interval("total_return_pct")
    assert record.interval("sharpe", level=0.9) == stats.sharpe_interval(traded, level=0.9)

    # a level so narrow that both bounds of 2,000 resamples are one of them
    assert record.interval("volatility_pct", level=1e-13) is None
    assert record.interval("win_rate", level=1e-13) is None
    assert record.interval("max_drawdown_pct", level=1e-13) is None

    for name in ("exposure_pct", "num_trades", "num_fills", "funding_paid"):
        assert record.interval(name) is None

    with pytest.raises(ProbatumKeyError, match="name: 'sharp'; give one of"):
        record.interval("sharp")
    with pytest.raises(ProbatumTypeError, match=r"name: \['sharpe'\]; give one of"):
        record.interval(["sharpe"])
    with pytest.raises(ProbatumValueError, match="level: 1.5; give a fraction"):
        record.interval("sharpe", level=1.5)


def test_a_sample_that_cannot_support_an_interval_gives_none():
    few = run(np.arange(400) % 40 < 20)

    assert few.stats["num_trades"] < stats.FLOOR and few.returns.size - 1 < stats.SPAN
    assert few.stats.interval("cagr_pct") is None
    assert few.stats.interval("win_rate") is None
    assert few.stats.interval("avg_trade_pct") is None
    assert few.stats.interval("sharpe") == stats.sharpe_interval(few)


def test_the_record_holds_none_where_its_number_would_describe_nothing():
    idle   = run(np.zeros(400, dtype=bool))
    record = idle.stats
    none   = ("cagr_pct", "sharpe", "sortino", "calmar", "win_rate", "profit_factor",
              "avg_trade_pct")

    assert [name for name in record if math.isnan(record[name])] == list(none)
    assert all(record.interval(name) is None for name in none)
    assert (record["total_return_pct"], record["num_trades"]) == (0.0, 0)
    assert record._values["sharpe"] == 0.0 and pickle.loads(pickle.dumps(record)) == record
    assert record._withheld == {
        "cagr_pct": "cagr_pct: 400 rows, 0.0457 of a year at 8,760 a year; a rate "
                    "compounded to a year needs a year of rows",
        **dict.fromkeys(("sharpe", "sortino", "calmar"), "returns: all 399 are the same; "
                        "a ratio needs returns that vary"),
        **dict.fromkeys(("win_rate", "profit_factor", "avg_trade_pct"), "round trips: "
                        "none closed; a rate of round trips needs one that closed"),
    }

    kept, short = (run(np.zeros(rows, dtype=bool)) for rows in (8_760, 8_759))

    assert kept.stats["cagr_pct"] == 0.0 and math.isnan(short.stats["cagr_pct"])


def test_the_intervals_are_the_same_after_a_pickle_a_save_and_a_load(traded, tmp_path):
    before = {name: traded.stats.interval(name) for name in ("cagr_pct", "win_rate")}
    again  = pickle.loads(pickle.dumps(traded))

    assert again.stats == traded.stats
    assert {name: again.stats.interval(name) for name in before} == before

    record = pickle.loads(pickle.dumps(traded.stats))

    assert record == traded.stats and record.interval("cagr_pct") == before["cagr_pct"]

    traded.save(tmp_path / "run")
    loaded = probatum.Result.load(tmp_path / "run")

    assert {name: loaded.stats.interval(name) for name in before} == before
