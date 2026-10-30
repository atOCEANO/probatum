"""A table, a record and a run's statistics show, in a notebook and as text, as the
pandas frame their to_pandas() gives: what holds for every row in a frame of one column
beside it, and why a statistic has no value or no interval in a frame under them, by
the number its row gives. Without pandas a table and a record show their fields (ADR
0041).
"""

import html
import math
import sys

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import MissingDependency, _frames, stats
from probatum._result import _money

HOUR = 3_600_000_000_000


def beside(record):
    """What ``record`` shows beside its frame, each name to its value as it reads."""
    return _frames.held(record._frame()[1])["value"].to_dict()


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


def test_a_record_s_lists_are_rows_by_what_numbers_them_the_rest_beside_them():
    decay = stats.Decay([0, 2], [0, HOUR], [2 * HOUR, 3 * HOUR], [1.8, 1.2], [0.6, 1.4],
                        [1.2, -0.2], 0.5, 0.7, "sharpe", "maximize")
    frame = decay.to_pandas()
    held  = _frames.held(decay._frame()[1])
    shown = decay._repr_html_()

    assert (frame.index.name, list(frame.index)) == ("window", [0, 2])
    assert list(frame.columns) == ["first", "last", "fitted", "earned", "gaps"]
    assert str(frame["first"].dtype) == "datetime64[ns, UTC]"
    assert held["value"].to_dict() == {"mean": "0.5", "error": "0.7", "objective": "sharpe",
                                       "direction": "maximize"}
    assert repr(decay) == f"{frame!r}\n\n{held!r}"
    assert shown.index(frame._repr_html_()) < shown.index(held._repr_html_())

    # a holdout's one gap has no standard error, a field the record holds none of
    once = stats.Decay(["holdout"], [0], [HOUR], [1.8], [0.6], [1.2], 1.2, None, "sharpe",
                       "maximize")

    assert list(beside(once)) == ["mean", "objective", "direction"]

    held = stats.Consistency([0.81, 1.22, -0.3, 0.94], "stretches", 0.0, 0.75, "sharpe",
                             "maximize")

    assert held.to_pandas()["scores"].tolist() == [0.81, 1.22, -0.3, 0.94]
    assert beside(held) == {"unit": "stretches", "break_even": "0.0", "share": "0.75",
                            "objective": "sharpe", "direction": "maximize"}


def test_a_record_of_two_tests_is_a_row_each_what_they_share_beside_it(traded):
    apart = stats.sharpe_difference(traded)
    frame = apart.to_pandas()

    assert list(frame.index) == ["delta", "bootstrap"]
    assert {"statistic", "estimate", "level", "count", "kind"}.isdisjoint(frame.columns)
    assert frame.loc["delta", "p"] == apart.delta.p
    assert frame.loc["bootstrap", "block"] == apart.bootstrap.block
    assert list(beside(apart).items())[-4:] == [
        ("statistic", "sharpe difference"), ("level", "0.95"),
        ("count", f"{apart.delta.count:,}"), ("kind", "interval")]

    # one series against itself reversed, two equal Sharpes whose tests both read 1
    steps = np.tile([0.015625, -0.0078125, 0.0078125, -0.00390625], 256)
    equal = stats.sharpe_difference(steps, steps[::-1], periods_per_year=365).to_pandas()

    assert equal["p"].tolist() == [1.0, 1.0]

    # a difference built by hand whose two tests read at two levels shows both
    wider = stats.Interval(*(0.9 if name == "level" else value for name, value in
                             zip(stats.Interval.__slots__, apart.bootstrap._args())))
    mixed = stats.Difference(apart.estimate, apart.delta, wider).to_pandas()

    assert mixed["level"].tolist() == [0.95, 0.9] and "statistic" not in mixed.columns

    check = stats.RealityCheck(0.045, 0.048, 0.010297, 2.173125, 2, 3.13, 1000, 40, 0,
                               "cash", False)
    shown = check.to_pandas()

    assert list(shown.index) == ["reality check", "spa"]
    assert shown.to_dict("list") == {"p": [0.045, 0.048], "statistic": [0.010297, 2.173125]}
    assert beside(check) == {"best": "2", "block": "3.13", "resamples": "1,000",
                             "trials": "40", "excluded": "0", "benchmark": "cash",
                             "descriptive": "False"}


def test_the_sides_of_a_run_s_round_trips_keep_every_column_whatever_they_share(traded):
    held = stats.trade_stats(traded)
    both = stats.Sides(held, held).to_pandas()

    assert list(both.index) == ["long", "short"]
    assert list(both.columns) == list(stats.TradeStats.__slots__)
    assert both.loc["short", "trades"] == held.trades


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


def test_a_run_s_statistics_show_their_numbers_as_a_summary_does_over_their_reasons(
        traded):
    record               = traded.stats
    frame                = record.to_pandas()
    sharpe               = record.interval("sharpe")
    shown, held, reasons = record._shown()
    why                  = list(dict.fromkeys(why for why in frame["why_none"] if why))

    assert shown.loc[["num_trades", "num_fills"], "value"].tolist() == [
        f"{record['num_trades']:,.0f}", f"{record['num_fills']:,.0f}"]
    assert shown.loc["funding_paid", "value"] == _money(record["funding_paid"])
    assert shown.loc["sharpe", ["value", "low", "high", "level"]].tolist() == [
        f"{record['sharpe']:,.2f}", f"{sharpe.low:,.2f}", f"{sharpe.high:,.2f}", "0.95"]
    assert held is None and reasons.to_dict("list") == {"why_none": why}
    assert list(reasons.index) == list(range(1, len(why) + 1))
    assert shown.loc["cagr_pct", "why_none"] == str(
        why.index(record._withheld["cagr_pct"]) + 1)
    assert shown.loc["exposure_pct", "why_none"] == ""

    with pd.option_context("display.max_colwidth", None):
        page = record._repr_html_()

        assert repr(record) == f"{shown!r}\n\n{reasons!r}"
        assert page.index(shown._repr_html_()) < page.index(reasons._repr_html_())


def test_where_pandas_shows_frames_as_text_a_notebook_shows_a_record_as_its_text(traded):
    under = stats.time_under_water(traded)
    # a unit named with markup, which the page escapes
    named = stats.Consistency([1.0], "<a & 'b'>", 0.0, 1.0, "sharpe", "maximize")

    with pd.option_context("display.notebook_repr_html", False):
        for shown, said in ((traded.stats, traded.stats), (under, under), (named, named),
                            (traded, traded.summary())):
            assert shown._repr_html_() == f"<pre>{html.escape(repr(said))}</pre>"


def test_a_value_beside_a_frame_shows_to_pandas_places_and_a_tiny_one_in_full():
    values = [(0.95, "0.95"), (1.0, "1.0"), (12_345.5, "12,345.5"), (3e-7, "3e-07"),
              (-1e-9, "-1e-09"), (-0.0, "0.0"), (1_000, "1,000"), (False, "False"),
              ("cash", "cash")]

    assert [_frames.said(value) for value, _ in values] == [said for _, said in values]


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
