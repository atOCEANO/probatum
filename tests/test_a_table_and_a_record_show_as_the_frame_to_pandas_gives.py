"""A table, a record and a run's statistics show, in a notebook and as text, as the
pandas frame their to_pandas() gives: what holds for every row is said once under it,
and why a statistic has no value or no interval is numbered under it. Without pandas a
table and a record show their fields (ADR 0041).
"""

import math
import sys

import numpy as np
import pytest

import probatum
from probatum import MissingDependency, _frames, stats

HOUR = 3_600_000_000_000


def run(rows, ending=False):
    close = 100.0 + np.random.default_rng(9).normal(0.0, 1.0, rows).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(rows, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * HOUR,
    )

    held = np.arange(rows) % 40 < 20
    held[-5:] = ending
    return probatum.backtest(np.where(held, 1.0, 0.0), data, "frictionless")


@pytest.fixture(scope="module")
def traded():
    return run(1_000)


def test_a_table_shows_as_its_frame_its_counts_plain_where_none_is_missing(traded):
    trips = traded.trades
    frame = trips.to_pandas()

    assert repr(trips) == repr(frame) and trips._repr_html_() == frame._repr_html_()
    assert frame["close_bar"].dtype == np.int64 and frame["open"].dtype == bool

    held = run(1_000, ending=True).trades.to_pandas()

    assert str(held["close_bar"].dtype) == "Int64" and held["close_bar"].isna().sum() == 1


def test_a_record_of_one_row_is_named_by_its_statistic_or_its_class(traded):
    found = traded.stats.interval("sharpe")
    frame = found.to_pandas()

    assert list(frame.index) == ["sharpe"]
    assert list(frame.columns) == ["estimate", "low", "high", "level", "method", "count",
                                   "lag", "floored", "kind"]
    assert frame.loc["sharpe", "estimate"] == found.estimate and repr(found) == repr(frame)
    assert list(stats.time_under_water(traded).to_pandas().index) == ["UnderWater"]


def test_a_record_s_lists_are_rows_by_what_numbers_them_the_rest_under_them():
    decay = stats.Decay([0, 2], [0, HOUR], [2 * HOUR, 3 * HOUR], [1.8, 1.2], [0.6, 1.4],
                        [1.2, -0.2], 0.5, 0.7, "sharpe", "maximize")
    frame = decay.to_pandas()

    assert (frame.index.name, list(frame.index)) == ("window", [0, 2])
    assert list(frame.columns) == ["first", "last", "fitted", "earned", "gaps"]
    assert str(frame["first"].dtype) == "datetime64[ns, UTC]"
    assert repr(decay).splitlines()[-1] == ("mean 0.5; error 0.7; objective sharpe; "
                                            "direction maximize")

    held = stats.Consistency([0.81, 1.22, -0.3, 0.94], "stretches", 0.0, 0.75, "sharpe",
                             "maximize")

    assert held.to_pandas()["scores"].tolist() == [0.81, 1.22, -0.3, 0.94]
    assert repr(held).splitlines()[-1] == ("unit stretches; break_even 0.0; share 0.75; "
                                           "objective sharpe; direction maximize")


def test_a_record_of_two_tests_is_a_row_each_what_they_share_said_under_it(traded):
    apart = stats.sharpe_difference(traded)
    frame = apart.to_pandas()

    assert list(frame.index) == ["delta", "bootstrap"]
    assert {"statistic", "estimate", "level", "count", "kind"}.isdisjoint(frame.columns)
    assert frame.loc["delta", "p"] == apart.delta.p
    assert frame.loc["bootstrap", "block"] == apart.bootstrap.block
    assert repr(apart).splitlines()[-1].endswith(
        f"; statistic sharpe difference; level 0.95; count {apart.delta.count:,}; kind "
        f"interval")

    check = stats.RealityCheck(0.045, 0.048, 0.010297, 2.173125, 2, 3.13, 1000, 40, 0,
                               "cash", False)
    shown = check.to_pandas()

    assert list(shown.index) == ["reality check", "spa"]
    assert shown.to_dict("list") == {"p": [0.045, 0.048], "statistic": [0.010297, 2.173125]}
    assert repr(check).splitlines()[-1] == ("best 2; block 3.13; resamples 1,000; trials "
                                            "40; excluded 0; benchmark cash; descriptive "
                                            "False")


def test_a_run_s_statistics_show_each_beside_its_interval_and_why_it_has_none(traded):
    record = traded.stats
    frame  = record.to_pandas()
    sharpe = record.interval("sharpe")

    assert (frame.index.name, list(frame.index)) == ("statistic", list(record))
    assert list(frame.columns) == ["value", "low", "high", "level", "method", "why_none"]
    assert frame.loc["sharpe"].tolist()[:4] == [record["sharpe"], sharpe.low, sharpe.high,
                                                0.95]
    assert frame.loc["sharpe", "method"] == (
        f"delta method, Newey-West lag {sharpe.lag}"
        + (", floored at lag zero" if sharpe.floored else ""))
    assert math.isnan(frame.loc["cagr_pct", "value"])
    assert frame.loc["cagr_pct", "why_none"] == record._withheld["cagr_pct"]
    assert frame.loc["total_return_pct", "why_none"].startswith("returns: 999 in blocks of")
    assert frame.loc["exposure_pct", "why_none"] == ""

    reasons = list(dict.fromkeys(why for why in frame["why_none"] if why))
    lines   = ["level 0.95"] + [f"{k}. {why}" for k, why in enumerate(reasons, start=1)]

    assert repr(record).splitlines()[-len(lines):] == lines
    assert all(line in record._repr_html_() for line in lines[:1])


def test_a_setting_shows_to_pandas_places_and_one_too_small_for_them_in_full(monkeypatch):
    settings = [("level", 0.95), ("rate", 3e-7), ("dust", -1e-9), ("zero", -0.0)]
    said     = "level 0.95; rate 3e-07; dust -1e-09; zero 0.0"

    assert _frames.line(settings) == said

    # a summary's lines are built with it, and need no pandas to be
    monkeypatch.setitem(sys.modules, "pandas", None)

    assert _frames.line(settings) == said


def test_without_pandas_a_table_and_a_record_show_their_fields(traded, monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    held = stats.Consistency([1.0], "windows", 0.0, 1.0, "sharpe", "maximize")

    assert repr(traded.trades) == (f"Table({len(traded.trades):,} rows: "
                                   f"{', '.join(traded.trades.columns)})")
    assert traded.trades._repr_html_() is None and held._repr_html_() is None
    assert repr(held) == ("Consistency(scores=(1.0,), unit='windows', break_even=0.0, "
                          "share=1.0, objective='sharpe', direction='maximize')")
    assert repr(traded.stats).startswith("Statistics({'total_return_pct': ")

    with pytest.raises(MissingDependency, match="to_pandas: pandas is not installed; pip "
                                                "install pandas"):
        held.to_pandas()
    with pytest.raises(MissingDependency, match="to_pandas: pandas is not installed"):
        traded.stats.to_pandas()
