"""Tests for probatum.chart: it builds one compact document from a Data's bars, your
arrays and a run on them, holds the alignment contract, and refuses the inputs that
would draw a lie.

Everything here reads ``Chart.spec()`` rather than pixels. The gate has no browser,
and a spec assertion is the sharper test anyway: it says what is drawn, where a
screenshot only says it looked the same.
"""

import html.parser
import json
import pathlib
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

import probatum
from probatum import (DataError, ProbatumKeyError, ProbatumRuntimeError,
                      ProbatumTypeError, ProbatumValueError, ProbatumWarning, stats)
from probatum.chart.marks import (
    Background, Band, Histogram, Level, Line, Marker, Markers, Panel, ramp,
)

HOUR = 3_600 * 10**9
# 2024-01-01 at midnight, UTC
START = 1_704_067_200 * 10**9
FIELDS = ("open", "high", "low", "close", "volume")


def table(n=8, **columns):
    close = 100.0 + np.arange(n, dtype=np.float64)
    given = {
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
        # varied, so the default bars do not trip the dead-feed warning on every
        # test in the file
        "volume": 1000.0 + np.arange(n, dtype=np.float64),
    }
    given.update(columns)
    return np.column_stack([np.asarray(given[name], dtype=np.float64)
                            for name in FIELDS])


def bars(n=8, step=HOUR, **columns):
    """``n`` bars of one instrument from 2024-01-01, ``step`` apart, closing a point
    higher each bar unless ``columns`` say otherwise."""
    return probatum.Data(table(n, **columns), symbol="BTCUSDT", market="linear",
                         time=START + np.arange(n, dtype=np.int64) * step)


class BuyThenClose(probatum.Strategy):
    """A buy of one at the first close and a close at the fifth."""


    def next(self, bar, account):
        if bar.i == 0:
            account.buy(1.0)
        elif bar.i == 4:
            account.close()


class BuyAtOpen(probatum.Strategy):
    """A buy of fifty at the first close, held to the end."""


    def next(self, bar, account):
        if bar.i == 0:
            account.buy(50.0)


def sank(n=6):
    # bars that fall hard on the bar the first order fills on, then recover
    return bars(
        n,
        open=[100.0, 110.0, 104.0, 106.0, 108.0, 110.0][:n],
        high=[112.0, 112.0, 108.0, 110.0, 112.0, 114.0][:n],
        low=[98.0, 78.0, 98.0, 100.0, 102.0, 104.0][:n],
        close=[110.0, 80.0, 106.0, 108.0, 110.0, 112.0][:n],
    )


def run(data=None):
    return probatum.backtest(BuyThenClose(), bars(8) if data is None else data,
                             "frictionless")


def crashed():
    """Five times the balance held through a fall of 40%, which liquidates it."""
    rows  = 60
    close = 100.0 * np.concatenate([np.full(10, 1.0), np.linspace(1.0, 0.6, 20),
                                    np.full(30, 0.6)])
    data  = bars(rows, open=close, high=close * 1.001, low=close * 0.999, close=close)
    venue = probatum.venues.binance_perp.replace(funding="none")
    return data, probatum.backtest(np.full(rows, 500.0), data, venue, sizing="quantity")


def pair(late=0, n=10):
    """``n`` rows of BTCUSDT and ETHUSDT on one clock, ETHUSDT a tenth of the price and
    printing no bar on its first ``late`` rows."""
    btc = table(n)
    eth = table(n) / 10.0
    eth[:late] = np.nan
    return probatum.Data(np.stack([btc, eth], axis=1), symbol=["BTCUSDT", "ETHUSDT"],
                         market="linear",
                         time=START + np.arange(n, dtype=np.int64) * HOUR)


def everything():
    """One chart carrying every mark kind, so the golden covers the whole union."""
    data   = bars(40)
    result = run(data)
    close  = data.close
    fast   = np.convolve(close, np.full(5, 0.2))[:40]
    fast[:4] = np.nan
    held   = np.zeros(len(data), dtype=bool)

    for trade in result.trades:
        last = len(data) - 1 if trade["open"] else trade["close_bar"]
        held[trade["open_bar"]:last + 1] = True

    return result.chart(
        [
            # the ramp and the fill are mutually exclusive: an area carries one
            # line colour and would drop the ramp while the legend went on
            # reporting it, so the pair is refused. Both halves still
            # appear in the golden, on two marks
            Line(fast, "sma 5", color=ramp(close, "#000000", "#ffffff")),
            Line(fast - 2.0, "sma 5 band", fill="#11223344"),
            Histogram(close - 100.0, "delta", panel="flow"),
            Band(close + 1.0, close - 1.0, "channel", fill="#11223344"),
            Band(close, 120.0, only="above", panel="flow", fill=("#00000000", "#000000ff")),
            Level(110.0, "mid"),
            Marker(3, text="here", shape="arrow_up", offset=20, value=105.0,
                   color="#ff5470"),
            Markers(mask=held, shape="arrow_down", offset=-16, value=fast),
            Background(held, fill="#11223344"),
            # named and labelled, which is the only shape that ships `keys`: the
            # mask above covers the unnamed one, and "out" maps to nothing so the
            # absent-label path is in here too
            Background(np.where(held, "in", "out"), "regime", fill={"in": "#2fe0a822"}),
        ],
        candle_color=np.where(held, "#4d9fff", None),
        panels=[Panel("flow", weight=1.5, range=(-5, 50))],
        title="everything",
        focus=(2, 30),
        notes=[["window", "sharpe"], ["1", 1.94], ["2", -0.31]],
    )


def test_the_shared_time_axis_has_one_entry_per_bar():
    spec = bars(8).chart().spec()

    assert spec["n"]      == 8
    assert len(spec["t"]) == 8


def test_the_time_axis_is_the_data_s_clock_in_utc_seconds():
    spec = bars(4).chart().spec()

    assert spec["t"][0] == 1704067200
    assert np.diff(spec["t"]).tolist() == [3_600] * 3


def test_a_full_length_series_starts_at_bar_zero():
    spec = bars(8).chart(Line(np.arange(8, dtype=float), "f")).spec()

    assert spec["series"][0]["i0"] == 0


def test_a_series_one_short_is_drawn_from_bar_one():
    spec = bars(8).chart(Line(np.arange(7, dtype=float), "f")).spec()

    assert spec["series"][0]["i0"]     == 1
    assert len(spec["series"][0]["v"]) == 7


def test_a_series_of_any_other_length_names_both_numbers():
    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(Line(np.arange(4, dtype=float), "f"))

    message = str(excinfo.value)
    assert "4" in message and "8" in message and "7" in message


def test_a_leading_nan_run_becomes_an_offset_rather_than_a_run_of_nulls():
    values = np.arange(8, dtype=float)
    values[:3] = np.nan

    spec = bars(8).chart(Line(values, "f")).spec()

    assert spec["series"][0]["i0"]     == 3
    assert len(spec["series"][0]["v"]) == 5


def test_an_interior_nan_becomes_a_gap_rather_than_a_dropped_row():
    values = np.arange(8, dtype=float)
    values[4] = np.nan

    spec = bars(8).chart(Line(values, "f")).spec()

    assert spec["series"][0]["i0"] == 0
    assert spec["series"][0]["v"][4] is None
    assert len(spec["series"][0]["v"]) == 8


def test_a_short_series_that_also_has_a_gap_still_starts_at_bar_one():
    values = np.arange(7, dtype=float)
    values[:2] = np.nan

    spec = bars(8).chart(Line(values, "f")).spec()

    # 2 leading gaps on an array that already starts at bar 1
    assert spec["series"][0]["i0"]     == 3
    assert len(spec["series"][0]["v"]) == 5
    assert spec["series"][0]["v"][0]   == 2.0


def test_a_short_series_with_an_interior_gap_keeps_its_offset():
    values = np.arange(7, dtype=float)
    values[3] = np.nan

    spec = bars(8).chart(Line(values, "f")).spec()

    assert spec["series"][0]["i0"] == 1
    assert spec["series"][0]["v"][3] is None
    assert len(spec["series"][0]["v"]) == 7


def test_an_all_nan_series_draws_nothing_rather_than_raising():
    spec = bars(8).chart(Line(np.full(8, np.nan), "f")).spec()

    assert spec["series"][0]["v"] == []


def test_the_equity_panel_is_the_run_s_equity_from_its_opening_balance():
    result = run()
    spec   = result.chart().spec()

    assert spec["equity"]["i0"] == 0
    assert spec["equity"]["v"]  == pytest.approx(result.equity.tolist())
    assert spec["equity"]["v"][0] == result.equity[0]
    # the drawdown starts on the same bar, so the crosshair reads both
    assert spec["drawdown"]["i0"] == 0
    assert spec["drawdown"]["v"][0] == 0.0


def test_a_run_that_ends_where_it_started_reads_zero_on_the_equity_panel():
    # the number the legend prints on hover and the axis a percent panel asks for
    # are both taken from the track's first value, which has to be the balance the
    # run opened with: the order decided on bar 0 fills at bar 1's open of 100, and
    # bar 1 then closes at 80, so the account is down a tenth on its first move.
    # It recovers to exactly where it began
    close = np.array([100.0, 80.0, 85.0, 90.0, 95.0, 100.0, 100.0, 100.0])
    opens = np.concatenate(([100.0], close[:-1]))
    data  = bars(8, open=opens, high=np.maximum(opens, close) + 1.0,
                 low=np.minimum(opens, close) - 1.0, close=close,
                 volume=np.full(8, 1000.0) + np.arange(8))

    result = probatum.backtest(BuyAtOpen(), data, "frictionless")
    spec   = result.chart().spec()
    base, last = spec["equity"]["v"][0], spec["equity"]["v"][-1]

    assert base == result.equity[0]
    assert (last / base - 1.0) * 100.0 == pytest.approx(
        result.stats["total_return_pct"], abs=1e-9
    )


def test_a_colour_array_of_the_wrong_length_names_both_numbers():
    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(candle_color=["#ffffff"] * 5)

    assert "candle_color" in str(excinfo.value)


def test_a_short_colour_array_is_drawn_from_bar_one_like_any_other_series():
    spec = bars(8).chart(candle_color=["#ffffff"] * 7).spec()

    assert spec["candles"]["color"]["i0"] == 1


def test_no_non_finite_value_reaches_the_document():
    values = np.arange(8, dtype=float)
    values[2] = np.inf
    values[5] = np.nan

    spec = run().chart(Line(values, "f")).spec()

    json.dumps(spec, allow_nan=False)


def test_an_infinite_stat_reads_n_a_rather_than_breaking_the_document():
    # profit_factor with no losing trade is infinite, and JSON carries no infinity
    spec = run().chart(stats=["profit_factor"]).spec()

    assert spec["headline"] == [["profit factor", "n/a"]]
    json.dumps(spec, allow_nan=False)


def test_a_bar_the_instrument_printed_none_on_is_left_empty():
    # ETHUSDT lists three rows after the universe opens, so its first three candles
    # are whitespace that keeps their place on the axis, and its volume starts later
    spec = pair(late=3).chart(instrument="ETHUSDT").spec()

    assert spec["n"] == 10 and spec["ohlc"][:3] == [None, None, None]
    assert spec["ohlc"][3] == pytest.approx([10.3, 10.4, 10.2, 10.3])
    assert spec["vol"]["i0"] == 3
    json.dumps(spec, allow_nan=False)


def test_the_overlay_threshold_is_pinned_on_both_sides():
    # ADR 0042 calls the half-panel threshold the guarantee rather than a tuning
    # parameter, and the suite only ever exercised ratios of 1.0, 0.08 and 0.002,
    # so every value from 0.09 to 1.0 would have passed. bars(8) spans 99 to 108,
    # so a flat series at V merges to a span of V - 99 and the ratio is 9 / that
    overlays = bars(8).chart(Line(np.full(8, 116.0), "just inside")).spec()

    assert [p["name"] for p in overlays["panels"]] == ["price", "volume"]

    apart = bars(8).chart(Line(np.full(8, 118.0), "just outside")).spec()

    assert "just outside" in [p["name"] for p in apart["panels"]]


def test_an_accepted_overlay_widens_the_price_extent_for_the_next_one():
    near, far = Line(np.full(8, 116.0)), Line(np.full(8, 124.0))

    # near overlays and takes price to 99..116, after which far fits too
    together = bars(8).chart([near, far]).spec()

    assert [p["name"] for p in together["panels"]] == ["price", "volume"]

    # judged first, against 99..108, far does not fit and takes its own panel
    apart = bars(8).chart([Line(np.full(8, 124.0)), Line(np.full(8, 116.0))]).spec()

    assert len(apart["panels"]) == 3


def test_an_unnamed_band_is_placed_by_its_upper_edge():
    # one edge has to decide it, and the choice is not arbitrary: the upper edge
    # is what a band shares with the candles it might sit on. Swapping it for the
    # lower edge passed every test in the file
    band = Band(np.full(8, 116.0), np.full(8, 60.0), fill="#11223344")

    spec = bars(8).chart(band).spec()

    # by the upper edge this overlays; by the lower it would take its own panel
    assert [p["name"] for p in spec["panels"]] == ["price", "volume"]


def test_price_is_always_the_first_panel():
    spec = bars(8).chart(Line(np.arange(8, dtype=float), "f")).spec()

    assert spec["panels"][0]["name"] == "price"
    assert spec["panels"][0]["candles"] is True


def test_a_bare_array_on_the_price_scale_overlays_the_candles():
    spec = bars(8).chart(np.full(8, 104.0)).spec()

    assert spec["series"][0]["panel"] == "price"


def test_a_bare_array_that_would_flatten_the_candles_gets_its_own_panel():
    # the bars live around 100 and this lives around 5000, so overlaying would
    # leave the candles a sliver
    spec = bars(8).chart(np.full(8, 5000.0)).spec()

    assert spec["series"][0]["panel"] != "price"


def test_naming_a_panel_overrides_the_placement_rule():
    spec = bars(8).chart(Line(np.full(8, 5000.0), "f", panel="price")).spec()

    assert spec["series"][0]["panel"] == "price"


def test_configuring_one_panel_leaves_every_other_panel_where_it_was():
    marks = [Line(np.full(8, 5000.0), "a", panel="one"),
             Line(np.full(8, 6000.0), "b", panel="two")]

    plain = [p["name"] for p in run().chart(marks).spec()["panels"]]
    tuned = run().chart(marks, panels=[Panel("two", weight=3)]).spec()["panels"]

    assert [p["name"] for p in tuned] == plain


def test_listing_two_panels_fixes_their_order_relative_to_each_other():
    marks = [Line(np.full(8, 5000.0), "a", panel="one"),
             Line(np.full(8, 6000.0), "b", panel="two")]

    names = [p["name"] for p in bars(8).chart(
        marks, panels=[Panel("two"), Panel("one")]
    ).spec()["panels"]]

    assert names.index("two") < names.index("one")


def test_a_volume_column_draws_a_volume_panel():
    spec = bars(8, volume=np.arange(8, dtype=float) + 1.0).chart().spec()

    assert any(p.get("volume") for p in spec["panels"])
    assert "vol" in spec


def test_a_hidden_panel_and_everything_on_it_leave_the_document():
    data = bars(8, volume=np.arange(8, dtype=float) + 1.0)
    spec = data.chart(panels=[Panel("volume", show=False)]).spec()

    assert not any(p["name"] == "volume" for p in spec["panels"])
    assert "vol" not in spec


def test_the_price_panel_cannot_be_hidden():
    with pytest.raises(ProbatumValueError):
        bars(8).chart(panels=[Panel("price", show=False)])


def test_a_pinned_range_reaches_the_panel():
    spec = bars(8).chart(Line(np.arange(8, dtype=float), "f", panel="x"),
                         panels=[Panel("x", range=(0, 100))]).spec()

    panel = [p for p in spec["panels"] if p["name"] == "x"][0]
    assert panel["range"] == [0.0, 100.0]


def test_a_result_adds_an_equity_panel_and_shades_the_drawdown_onto_it():
    # a run that falls on its first fill and recovers, so the shading opens
    spec  = probatum.backtest(BuyAtOpen(), sank(), "frictionless").chart().spec()
    names = [p["name"] for p in spec["panels"]]

    assert "equity" in names and "drawdown" not in names

    shade = [s for s in spec["series"]
             if s["kind"] == "band" and s["panel"] == "equity"]

    assert len(shade) == 1
    # the upper edge is the running maximum of the lower one, so the shape can touch
    # but never cross, and it opens only where the equity fell
    upper, lower = shade[0]["v"], shade[0]["v2"]
    assert upper == np.maximum.accumulate(lower).tolist()
    assert max(u - v for u, v in zip(upper, lower)) > 0.0


def test_the_drawdown_track_ships_even_with_no_panel_of_its_own():
    # the legend reads the number off it and prints it beside the shading
    assert "drawdown" in run().chart().spec()


def test_asking_for_a_drawdown_panel_gives_one_and_stops_shading_the_equity():
    spec = run().chart(drawdown="panel").spec()

    assert "drawdown" in [p["name"] for p in spec["panels"]]
    assert not [s for s in spec["series"] if s["kind"] == "band"]


def test_turning_the_drawdown_off_leaves_neither_a_panel_nor_a_shade():
    spec = run().chart(drawdown=False).spec()

    assert "drawdown" not in [p["name"] for p in spec["panels"]]
    assert "drawdown" not in spec
    assert not [s for s in spec["series"] if s["kind"] == "band"]


def test_hiding_the_drawdown_panel_still_removes_the_shading():
    # the documented off switch predates the shading and has to keep working
    spec = run().chart(panels=[Panel("drawdown", show=False)]).spec()

    assert not [s for s in spec["series"] if s["kind"] == "band"]
    assert "drawdown" not in spec


def test_a_mark_that_creates_a_drawdown_panel_stops_the_shading():
    # panel= creates a panel the same way any other name does, so the fall can end
    # up with a pane without anyone asking for one; it must not then also be shaded
    # onto the equity, which would draw the same number twice on one chart
    spec = run().chart(Level(-1.0, "watch", panel="drawdown")).spec()

    assert "drawdown" in [p["name"] for p in spec["panels"]]
    assert not [s for s in spec["series"] if s["kind"] == "band"]


def test_a_drawdown_mode_outside_the_three_is_told_what_they_are():
    with pytest.raises(ProbatumValueError) as excinfo:
        run().chart(drawdown="pane")

    message = str(excinfo.value)
    assert "'under'" in message and "'panel'" in message and "'pane'" in message


def test_a_drawdown_of_zero_is_refused_rather_than_read_as_false():
    # 0 == False, and a 0 taken for False drew neither a shade nor a pane while the
    # legend went on reading the drawdown
    for call in (lambda: run().chart(drawdown=0),
                 lambda: probatum.chart.defaults(drawdown=0)):
        with pytest.raises(ProbatumValueError):
            call()


def test_the_drawdown_mode_can_be_set_for_a_whole_session():
    previous = probatum.chart.defaults(drawdown="panel")

    try:
        names = [p["name"] for p in run().chart().spec()["panels"]]
        assert "drawdown" in names
    finally:
        probatum.chart.defaults(**{k: v for k, v in previous.items() if v is not None})


def test_marks_given_in_a_list_are_the_marks_given_alone():
    values = np.arange(8, dtype=float) + 100.0
    result = run()

    alone  = result.chart(Line(values, "f"), Level(104.0), title="marks").spec()
    listed = result.chart([Line(values, "f"), [Level(104.0)]], title="marks").spec()

    assert alone == listed
    assert [s["name"] for s in alone["series"] if s.get("name")] == ["f", "drawdown"]


def test_a_result_among_the_marks_is_refused_and_told_how_to_draw_its_equity():
    with pytest.raises(ProbatumTypeError) as excinfo:
        run().chart(run())

    assert "Line(other.equity" in str(excinfo.value)

    # which is a line on the equity panel, on the same rows
    other = run()
    spec  = run().chart(Line(other.equity, "other", panel="equity")).spec()

    assert [s["panel"] for s in spec["series"] if s.get("name") == "other"] == ["equity"]


def test_a_candle_ships_the_price_that_was_in_the_bars():
    # the payload is rounded by numpy, which rounds by scaling and can therefore
    # break a tie the other way from round(). A price is a whole number of ticks,
    # which is nowhere near a tie, and this pins that: a candle a cent away from
    # the bars it was drawn from is the one thing this layer must never do
    data = bars(
        4,
        open=[111.035, 104.695, 116.185, 0.00012345],
        high=[112.045, 112.005, 118.015, 0.00012355],
        low=[98.005, 78.045, 98.025, 0.00012335],
        close=[110.025, 80.015, 106.035, 0.00012345],
    )

    spec    = data.chart().spec()
    candles = data.bars[:, 0, :4]

    from probatum.chart import _digits, _extent
    wanted = [
        [round(float(x), _digits(_extent(candles)) + 2) for x in row]
        for row in candles
    ]

    assert spec["ohlc"] == wanted


def test_what_else_shares_the_price_panel_cannot_change_the_candles():
    close = np.linspace(0.000012, 0.000019, 40)
    data  = bars(40, open=close, high=close * 1.01, low=close * 0.99, close=close)

    alone = data.chart().spec()["ohlc"]
    with pytest.warns(ProbatumWarning):
        beside = data.chart(Level(70.0)).spec()["ohlc"]

    assert alone == beside
    assert len({row[3] for row in alone}) == 40


def test_a_level_that_cannot_be_seen_on_the_price_panel_says_so():
    with pytest.warns(ProbatumWarning) as caught:
        bars(8).chart(Line(np.linspace(0.0, 100.0, 8), "rsi"), Level(70.0))

    assert "panel=" in str(caught[0].message)
    # named at the line that drew the chart
    assert caught[0].filename == __file__


def test_a_level_near_the_candles_is_not_second_guessed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bars(8).chart(Level(112.0))


def test_the_drawdown_panel_agrees_with_the_record():
    # the fill lands on bar 1's open and the bar closes far below it, so the first
    # move of the equity is already under the opening balance
    result = probatum.backtest(BuyAtOpen(), sank(), "frictionless")
    values = result.chart().spec()["drawdown"]["v"]
    worst  = min(v for v in values if v is not None)

    assert result.stats["max_drawdown_pct"] > 0.0
    assert abs(worst + result.stats["max_drawdown_pct"]) < 1e-9


def test_the_drawdown_is_the_fall_probatum_stats_reads():
    data, result = crashed()
    drawn = result.chart().spec()["drawdown"]

    assert drawn["i0"] == 0
    assert drawn["v"] == pytest.approx((0.0 - stats.drawdown(result)).tolist(), abs=1e-6)
    # no fall is a zero the legend prints as 0.00%, and a negated one as -0.00%
    assert not any(np.signbit(value) for value in drawn["v"] if value == 0.0)


def test_a_drawdown_is_capped_at_a_total_loss():
    # an account that loses everything has fallen 100%, and no run is deeper
    data, result = crashed()

    assert result.bust
    assert min(result.chart(stats=[]).spec()["drawdown"]["v"]) == -100.0


def test_a_result_that_keeps_no_data_takes_it_by_its_fingerprint():
    data = bars(8)
    kept = probatum.backtest(BuyThenClose(), data, "frictionless", record="account")

    with pytest.raises(ProbatumValueError) as excinfo:
        kept.chart()

    assert "data" in str(excinfo.value)
    assert kept.chart(data=data).spec() == run(data).chart().spec()

    with pytest.raises(DataError) as excinfo:
        kept.chart(data=bars(8, volume=2_000.0 + np.arange(8.0)))

    assert "fingerprint" in str(excinfo.value)

    with pytest.raises(ProbatumTypeError):
        kept.chart(data=table(8))


def test_a_loaded_result_charts_as_the_run_did_once_given_its_data(tmp_path):
    data   = bars(8)
    result = run(data)
    result.save(tmp_path / "run")
    loaded = probatum.Result.load(tmp_path / "run")

    with pytest.raises(ProbatumValueError):
        loaded.chart()

    assert loaded.chart(data=data).spec() == result.chart().spec()


def test_a_book_is_refused():
    book = probatum.combine({"one": run(), "two": run()})

    with pytest.raises(ProbatumValueError) as excinfo:
        book.chart()

    assert "book" in str(excinfo.value)


class Pairs(probatum.Strategy):
    """A buy of BTCUSDT at the first close and of ETHUSDT at the third, both closed at
    the sixth."""


    def next(self, bar, account):
        if bar.i == 0:
            account["BTCUSDT"].buy(1.0)
        elif bar.i == 2:
            account["ETHUSDT"].buy(10.0)
        elif bar.i == 5:
            account["BTCUSDT"].close()
            account["ETHUSDT"].close()


def test_a_universe_draws_the_instrument_it_is_asked_for_and_its_trips_alone():
    data   = pair()
    result = probatum.backtest(Pairs(), data, "frictionless")

    with pytest.raises(ProbatumValueError) as excinfo:
        result.chart()

    assert 'instrument="BTCUSDT"' in str(excinfo.value)

    with pytest.raises(ProbatumKeyError):
        result.chart(instrument="SOLUSDT")

    with pytest.raises(ProbatumTypeError):
        result.chart(instrument=1)

    spec = result.chart(instrument="ETHUSDT").spec()

    assert spec["ohlc"][0] == pytest.approx(data.bars[0, 1, :4].tolist())
    assert [(t["in"], t["out"], t["size"]) for t in spec["trades"]] == [(3, 6, 10.0)]
    # the equity is the whole account's
    assert spec["equity"]["v"] == pytest.approx(result.equity.tolist())


def test_one_instrument_of_several_is_named_and_counted_as_the_account_s_share():
    data   = pair()
    result = probatum.backtest(Pairs(), data, "frictionless")
    spec   = result.chart(instrument="ETHUSDT").spec()
    total  = int((~np.asarray(result.trades["open"])).sum())
    mine   = int(sum(1 for trip in result.trades
                     if trip["instrument"] == "ETHUSDT" and not trip["open"]))

    assert spec["symbol"] == "ETHUSDT"
    assert spec["title"] == "ETHUSDT, one of the account's 2 instruments"
    assert dict(spec["headline"])["round trips"] == f"{mine:,} of {total:,}"
    assert result.chart(instrument="ETHUSDT", title="mine").spec()["title"] == "mine"


def test_a_trip_still_open_stands_at_the_last_bar_with_its_pnl_so_far():
    result = probatum.backtest(BuyAtOpen(), bars(8), "frictionless")
    spec   = result.chart().spec()
    [trip] = result.trades

    assert trip["open"]
    assert spec["trades"] == [{
        "i": 1, "in": trip["open_bar"], "out": 7, "side": "long", "size": 50.0,
        "px_in": trip["entry"], "px_out": None, "fees": trip["fees"],
        "fund": trip["funding"], "net": trip["pnl"], "ret": round(trip["ret"], 8),
        "best": round(trip["best"], 8), "worst": round(trip["worst"], 8),
        "bars": 7 - trip["open_bar"], "why": "open", "open": True,
    }]
    assert dict(spec["headline"])["round trips"] == "0, 1 open"


def test_the_price_is_candles_hollow_candles_bars_or_a_line():
    data = bars(8)

    assert data.chart().spec()["candles"]["style"] == "candles"
    assert [data.chart(price=style).spec()["candles"]["style"]
            for style in ("hollow", "bars", "line")] == ["hollow", "bars", "line"]

    with pytest.raises(ProbatumValueError, match="price: 'area'; give 'candles', 'hollow' "
                       "for a rising candle drawn as its outline, 'bars' or 'line'"):
        data.chart(price="area")

    previous = probatum.chart.defaults(price="line")
    try:
        assert data.chart().spec()["candles"]["style"] == "line"
    finally:
        probatum.chart.defaults(**previous)

    assert data.chart().spec()["candles"]["style"] == "candles"


def test_a_palette_names_the_wicks_and_the_volume_and_refuses_what_it_cannot_draw():
    data  = bars(8)
    given = {"dark": {"up": "#2962ff", "volume_down": "rgba(1, 2, 3, 0.5)"}}
    theme = data.chart(palette=given).spec()["theme"]["dark"]

    assert (theme["up"], theme["wick_up"], theme["volume_up"]) == (
        "#2962ff", "#2962ff", "rgba(41,98,255,0.4)")
    assert theme["volume_down"] == "rgba(1, 2, 3, 0.5)"

    with pytest.raises(ProbatumKeyError, match="palette: 'rising' under 'dark', a colour "
                       "the chart does not draw; use one of axis, down, "):
        data.chart(palette={"dark": {"rising": "#000000"}})
    with pytest.raises(ProbatumValueError, match=r"palette: 'red' for 'up'; give a colour "
                       r"as #rrggbb, #rrggbbaa or rgb\(r, g, b\)"):
        data.chart(palette={"light": {"up": "red"}})
    # a list is a selected trade's fill, the one colour the chart's own palette lists
    with pytest.raises(ProbatumValueError, match=r"palette: \['#ffffff'\] for 'up'; give a "
                       r"colour"):
        data.chart(palette={"dark": {"up": ["#ffffff"]}})

    listed = data.chart(palette={"dark": {"selected": ["#ffffff", "#000000"]}})
    one    = data.chart(palette={"dark": {"selected": "#ffffff"}})

    assert listed.spec()["theme"]["dark"]["selected"] == ["#ffffff", "#000000"]
    assert one.spec()["theme"]["dark"]["selected"] == ["#ffffff"]

    # the settings handed back put the session's palette back, none at all included
    previous = probatum.chart.defaults(palette={"dark": {"s1": "#ffffff"}})
    probatum.chart.defaults(**previous)

    assert data.chart().spec()["theme"]["dark"]["s1"] == "#e9ebee"


def test_a_panel_named_for_its_scale_keeps_its_own_share():
    shares = [{p["name"]: p["weight"] for p in run().chart(panels=[given]).spec()["panels"]}
              for given in (Panel("equity", scale="percent"), Panel("equity", weight=3))]

    assert (shares[0]["price"], shares[0]["equity"], shares[1]["equity"]) == (5.0, 2.0, 3.0)


def test_a_marker_draws_every_glyph_it_names():
    shapes = ("triangle_up", "triangle_down", "diamond", "cross", "flag")

    assert [bars(8).chart(Marker(3, shape=shape)).spec()["series"][0]["shape"]
            for shape in shapes] == list(shapes)


def test_a_missing_value_in_notes_reads_as_a_missing_float_does():
    pd    = pytest.importorskip("pandas")
    frame = pd.DataFrame({"bar": pd.array([1, None], dtype="Int64"),
                          "at": pd.to_datetime([0, None], utc=True), "x": [0.5, np.nan]})

    assert bars(8).chart(notes=frame).spec()["notes"]["rows"][1] == ["n/a", "n/a", "n/a"]


class Late(probatum.Strategy):
    """A buy of one at the close of the Data's row ``start`` and a close three rows
    later."""


    def __init__(self, start=4):
        self.start = start


    def next(self, bar, account):
        if bar.i == self.start:
            account.buy(1.0)
        elif bar.i == self.start + 3:
            account.close()


def test_a_run_on_rows_its_account_never_saw_draws_its_records_on_the_data_s_rows():
    # ETHUSDT alone prints from row 3, so its run's rows are the Data's from 3 and
    # its round trip's bars count from there: its entry at the run's bar 2 is the
    # Data's row 5
    data   = pair(late=3)["ETHUSDT"]
    result = probatum.backtest(Late(), data, "frictionless")
    spec   = result.chart().spec()

    assert len(result.time) == 7 and spec["n"] == 10
    assert [(t["in"], t["out"], t["bars"]) for t in spec["trades"]] == [(5, 8, 3)]
    assert spec["equity"]["i0"] == 3
    assert spec["equity"]["v"] == pytest.approx(result.equity.tolist())
    assert spec["drawdown"]["i0"] == 3


def test_a_result_taken_part_way_through_a_run_draws_the_rows_it_holds():
    # an Engine's result() holds the rows up to its current one, here the Data's
    # first 21, and the bars after them carry no equity
    engine = probatum.Engine(bars(40), "frictionless")
    bar    = engine.reset()

    while bar.i < 20:
        if bar.i in (2, 15):
            engine.buy(1.0)
        elif bar.i == 6:
            engine.close()
        bar = engine.step()

    result = engine.result()
    spec   = result.chart().spec()

    assert len(result.time) == 21 and spec["n"] == 40
    assert spec["equity"]["v"][:21] == pytest.approx(result.equity.tolist())
    assert spec["equity"]["v"][21:] == [None] * 19
    assert [(t["in"], t["out"], t.get("open", False)) for t in spec["trades"]] == [
        (3, 7, False), (16, 20, True)]

    # the trip still open runs to the run's last row, 20, not to the Data's
    held = result.trades[-1]
    assert held["open"] and result.chart(focus=held).spec()["focus"] == [0.0, 38.0]


def test_an_array_of_a_run_s_rows_is_refused_where_they_are_not_all_the_data_s():
    # ETHUSDT printed no bar on row 5, so a run on it alone records 9 of the Data's
    # 10 rows, and its equity, one short, would pass as a diff and draw every value
    # before the hole a bar late
    btc  = table(10)
    eth  = table(10) / 10.0
    eth[5] = np.nan
    data = probatum.Data(np.stack([btc, eth], axis=1), symbol=["BTCUSDT", "ETHUSDT"],
                         market="linear",
                         time=START + np.arange(10, dtype=np.int64) * HOUR)["ETHUSDT"]
    result = probatum.backtest(BuyThenClose(), data, "frictionless")
    spec   = result.chart().spec()

    assert len(result.time) == 9 and spec["equity"]["v"][5] is None

    with pytest.raises(ProbatumValueError) as excinfo:
        result.chart(Line(result.equity, "again", panel="equity"))

    assert "searchsorted" in str(excinfo.value) and "prepend" in str(excinfo.value)

    with pytest.raises(ProbatumValueError):
        result.chart(candle_color=["#4d9fff"] * 9)

    # a colour a bar is read by position too, and a colour that is no list is read
    # as the chart reads one anywhere
    with pytest.raises(ProbatumValueError) as excinfo:
        result.chart(Line(np.full(10, 1.0), "flat", color=["#2fe0a8"] * 9))

    assert "Line 'flat' color" in str(excinfo.value)

    with pytest.raises(ProbatumValueError) as excinfo:
        result.chart(Band(np.full(10, 2.0), 1.0, "shade", fill=["#2fe0a8"] * 9))

    assert "Band 'shade' fill" in str(excinfo.value)

    with pytest.raises(ProbatumValueError):
        result.chart(candle_color=5)

    rows   = np.searchsorted(data.time, result.time)
    placed = np.full(len(data), np.nan)
    placed[rows] = result.equity
    drawn  = result.chart(Line(placed, "again", panel="equity")).spec()
    [again] = [s for s in drawn["series"] if s.get("name") == "again"]

    assert again["v"] == drawn["equity"]["v"]


def test_an_array_of_a_run_s_rows_is_drawn_where_it_lacks_only_the_first_row():
    # a diff and the run's array are both drawn from bar 1, where its rows start
    data   = pair(late=1)["ETHUSDT"]
    result = probatum.backtest(BuyThenClose(), data, "frictionless")
    drawn  = result.chart(Line(result.equity, "again", panel="equity")).spec()
    [again] = [s for s in drawn["series"] if s.get("name") == "again"]

    assert again["i0"] == 1


def test_focus_of_a_round_trip_frames_it_on_the_data_s_rows():
    # ETHUSDT prints from row 30, so the trip opened on the run's bar 5 is framed
    # around the Data's row 35, eighteen bars of room either side
    data   = pair(late=30, n=60)["ETHUSDT"]
    result = probatum.backtest(Late(start=34), data, "frictionless")
    trip   = result.trades[0]

    assert (trip["open_bar"], trip["close_bar"]) == (5, 8)
    assert result.chart(focus=trip).spec()["focus"] == [17.0, 56.0]


def test_a_logarithmic_panel_whose_data_reaches_zero_names_the_bar():
    values = np.arange(8, dtype=float)          # values[0] is 0.0

    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(Line(values, "f", panel="x"), panels=[Panel("x", scale="log")])

    assert "bar 0" in str(excinfo.value)


def test_a_dead_volume_column_draws_no_panel_and_warns():
    with pytest.warns(ProbatumWarning) as caught:
        spec = bars(8, volume=np.zeros(8)).chart().spec()

    assert not any(p.get("volume") for p in spec["panels"])
    # named at the line that drew the chart
    assert caught[0].filename == __file__


def test_a_chart_too_heavy_for_a_notebook_warns_at_the_line_that_builds_it(monkeypatch):
    # its payload is made with the chart, so the warning names that line wherever the
    # document is drawn later: a cell's display has no line of the caller's to name
    monkeypatch.setattr(probatum.chart, "_BIG_BYTES", 1_000)

    with pytest.warns(ProbatumWarning, match="opens slowly") as caught:
        bars(8).chart()

    assert caught[0].filename == __file__


def test_a_marker_outside_the_bars_is_refused():
    with pytest.raises(ProbatumValueError):
        bars(8).chart(Marker(99))


def test_a_per_bar_colour_becomes_a_palette_and_indices():
    colors = ["#ff0000"] * 4 + ["#00ff00"] * 4
    spec   = bars(8).chart(candle_color=colors).spec()
    tint   = spec["candles"]["color"]

    assert tint["p"] == [None, "#ff0000", "#00ff00"]
    assert tint["k"] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_the_first_palette_entry_is_always_the_absent_colour():
    spec = bars(8).chart(candle_color=["#ff0000"] * 8).spec()

    assert spec["candles"]["color"]["p"][0] is None


def test_a_colour_array_of_object_dtype_with_none_entries_encodes():
    # numpy.where(mask, ramp(...), None) is the ordinary way to write one of
    # these and it lands as an object array holding str and None
    mask = np.array([True, False] * 4)

    spec = bars(8).chart(candle_color=np.where(mask, "#ff0000", None)).spec()

    assert spec["candles"]["color"]["k"] == [1, 0] * 4


def test_a_background_becomes_spans_rather_than_one_entry_per_bar():
    mask  = np.array([False, True, True, True, False, False, True, False])
    spec  = bars(8).chart(Background(mask, fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert entry["spans"] == [[1, 4, 0], [6, 7, 0]]


def test_a_background_label_with_no_fill_shades_nothing():
    labels = np.array(["calm", "wild", "calm", "wild",
                       "calm", "calm", "calm", "calm"], dtype=object)

    spec  = bars(8).chart(Background(labels, fill={"wild": "#ff0000"})).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert entry["spans"] == [[1, 2, 0], [3, 4, 0]]


def test_notes_take_a_frame_and_leave_its_index_where_it_was():
    # drawing the index whenever it looked meaningful would be a rule nobody
    # could predict from outside; reset_index() says so at the call site instead
    pd    = pytest.importorskip("pandas")
    notes = pd.DataFrame({"window": [1, 2], "sharpe": [1.94, -0.31]},
                         index=["first", "second"])

    spec = bars(8).chart(notes=notes).spec()

    assert spec["notes"]["head"] == ["window", "sharpe"]
    assert spec["notes"]["rows"] == [["1", "1.94"], ["2", "-0.31"]]


def test_notes_take_rows_with_the_first_naming_the_columns():
    spec = bars(8).chart(notes=[["a", "b"], [1, 2], [3, 4]]).spec()

    assert spec["notes"] == {"head": ["a", "b"], "rows": [["1", "2"], ["3", "4"]]}


def test_a_note_row_that_does_not_match_its_header_names_both_counts():
    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(notes=[["a", "b"], [1, 2, 3]])

    assert "3 cells" in str(excinfo.value) and "of 2" in str(excinfo.value)


def test_notes_read_a_polars_frame_by_its_rows():
    # a polars frame iterates its columns, which read as rows drew the table on its
    # side
    pl    = pytest.importorskip("polars")
    notes = pl.DataFrame({"window": [1, 2], "sharpe": [1.94, -0.31]})

    spec = bars(8).chart(notes=notes).spec()

    assert spec["notes"] == {"head": ["window", "sharpe"],
                             "rows": [["1", "1.94"], ["2", "-0.31"]]}


def test_notes_read_a_pyarrow_table_by_its_rows():
    pa    = pytest.importorskip("pyarrow")
    notes = pa.table({"window": [1, 2], "sharpe": [1.94, -0.31]})

    spec = bars(8).chart(notes=notes).spec()

    assert spec["notes"] == {"head": ["window", "sharpe"],
                             "rows": [["1", "1.94"], ["2", "-0.31"]]}


def test_what_a_chart_cannot_read_is_refused_as_probatum_s_own_error():
    # each of these failed inside the chart with a built-in error naming neither
    # the argument nor its repair, or read a string a letter at a time
    for call in (lambda: bars(8).chart(panels=Panel("equity")),
                 lambda: bars(8).chart(palette="dark"),
                 lambda: bars(8).chart(palette={"dark": "#ffffff"}),
                 lambda: bars(8).chart(notes=5),
                 lambda: bars(8).chart(notes="a note"),
                 lambda: bars(8).chart(notes={"window": [1, 2]}),
                 lambda: bars(8).chart(notes=["ab", "cd"]),
                 lambda: bars(8).chart(notes=[["a", "b"], 5]),
                 lambda: run().chart(stats=5),
                 lambda: run().chart(stats="sharpe"),
                 lambda: bars(8).chart().save(5)):
        with pytest.raises(ProbatumTypeError):
            call()


def test_notes_with_no_rows_draw_no_table_rather_than_a_header_over_nothing():
    assert "notes" not in bars(8).chart(notes=[["a", "b"]]).spec()
    assert "notes" not in bars(8).chart(notes=[]).spec()


def test_a_float_in_a_note_is_read_rather_than_reproduced():
    # str(1.94) is fine and str(0.1 + 0.2) is 0.30000000000000004, and this table
    # is looked at rather than computed with. Six significant digits keeps a
    # sharpe short and a basis point off zero, which fixed decimals cannot do both
    spec = bars(8).chart(notes=[["x"], [0.1 + 0.2], [0.000123456], [float("nan")]]).spec()

    assert spec["notes"]["rows"] == [["0.3"], ["0.000123456"], ["n/a"]]


def test_a_table_as_notes_is_refused_and_told_how_to_give_it():
    # a Table's rows are dicts, whose own list is their keys, so read as rows it
    # would print its header on every line
    with pytest.raises(ProbatumTypeError) as excinfo:
        bars(8).chart(notes=run().trades)

    assert "to_pandas()" in str(excinfo.value)


def test_a_background_with_no_fill_of_its_own_is_ground_rather_than_a_band():
    # full height at one flat alpha with a hard edge either side is the banding
    # recipe, and a run of shaded stretches then reads as a rendering artifact
    # rather than as regions that mean something. Stops run bottom to top, so the
    # default is strongest at the floor and gone before the ceiling
    mask  = np.array([False, True, True, True, False, False, True, False])
    spec  = bars(8).chart(Background(mask)).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert len(entry["fills"][0]) == 2
    assert entry["fills"][0][0] != entry["fills"][0][1]


def test_a_background_given_one_colour_is_left_flat():
    # the default has an opinion about what a background is; a caller who named a
    # colour asked for that colour, and one stop means flat on every other mark
    mask  = np.array([False, True, True, True, False, False, True, False])
    spec  = bars(8).chart(Background(mask, fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert entry["fills"] == [["#112233"]]


def test_a_named_background_ships_the_names_of_the_regions_it_shaded():
    labels = np.array(["calm", "wild", "calm", "wild",
                       "calm", "calm", "calm", "calm"], dtype=object)

    spec  = bars(8).chart(Background(labels, "regime", fill={"wild": "#ff0000"})).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert entry["name"] == "regime"
    # in fill order and only the ones that shade, because the index in a span
    # points into `fills` and the two lists have to stay parallel
    assert entry["keys"] == ["wild"]


def test_an_unnamed_background_spends_no_bytes_saying_so():
    labels = np.array(["calm", "wild"] * 4, dtype=object)
    spec   = bars(8).chart(Background(labels, fill={"wild": "#ff0000"})).spec()
    entry  = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert "keys" not in entry
    assert "name" not in entry


def test_a_named_mask_background_sends_no_names_because_it_has_none():
    # a mask says where, never what, so its legend row is the swatch and the name
    # the caller gave the whole shading
    mask = np.array([False, True, True, False, False, False, False, False])

    spec  = bars(8).chart(Background(mask, "in a trade", fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert entry["name"] == "in a trade"
    assert "keys" not in entry


def test_a_nan_in_a_background_mask_shades_nothing():
    # numpy.array([nan]).astype(bool) is True, so a mask padded with NaN would
    # otherwise report the condition it was looking for on its last bar
    mask  = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan], dtype=object)
    spec  = bars(8).chart(Background(mask, fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert entry["spans"] == [[1, 2, 0]]


def test_a_band_with_a_scalar_edge_carries_a_level_rather_than_a_second_array():
    spec  = bars(8).chart(Band(np.arange(8, dtype=float), 3.0, panel="x")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]

    assert entry["level"] == 3.0
    assert "v2" not in entry


def test_a_band_with_two_edges_ships_them_at_one_start_and_one_length():
    upper = np.arange(8, dtype=float)
    lower = np.arange(8, dtype=float) - 5.0
    lower[:3] = np.nan

    spec  = bars(8).chart(Band(upper, lower, panel="x")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]

    assert entry["i0"] == 3
    assert len(entry["v"]) == len(entry["v2"]) == 5


def test_a_band_between_two_named_lines_ships_the_numbers_the_lines_do():
    # each line spans 95 where the panel, which the level stretches, spans 120, so a
    # line keeps a place the panel does not; the legend finds a band's lines by value
    upper = np.linspace(50.0, 145.0, 8) + 0.123456
    lower = upper - 1.0
    spec  = bars(8).chart(Line(upper, "upper", panel="x"), Line(lower, "lower", panel="x"),
                          Band(upper, lower, "channel", panel="x"),
                          Level(170.0, panel="x")).spec()
    band  = [s for s in spec["series"] if s["kind"] == "band"][0]
    lines = [s for s in spec["series"] if s["kind"] == "line"]

    assert (band["v"], band["v2"]) == (lines[0]["v"], lines[1]["v"])


def test_a_below_band_ships_its_fill_flipped_so_stop_zero_is_the_reference_edge():
    stops = ("#00000000", "#000000ff")

    above = bars(8).chart(Band(
        np.arange(8, dtype=float),
        3.0,
        only  = "above",
        panel = "x",
        fill  = stops,
    )).spec()
    below = bars(8).chart(Band(
        np.arange(8, dtype=float),
        3.0,
        only  = "below",
        panel = "x",
        fill  = stops,
    )).spec()

    up   = [s for s in above["series"] if s["kind"] == "band"][0]
    down = [s for s in below["series"] if s["kind"] == "band"][0]

    assert up["fill"]   == list(stops)
    assert down["fill"] == list(reversed(stops))


def test_a_trade_row_carries_the_keys_the_renderer_reads_and_no_more():
    result = run()
    spec   = result.chart().spec()
    [trip] = result.trades

    assert spec["trades"] == [{
        "i": 1, "in": trip["open_bar"], "out": trip["close_bar"], "side": "long",
        "size": 1.0, "px_in": trip["entry"], "px_out": trip["exit"],
        "fees": trip["fees"], "fund": trip["funding"], "net": trip["pnl"],
        "ret": round(trip["ret"], 8), "best": round(trip["best"], 8),
        "worst": round(trip["worst"], 8), "bars": trip["close_bar"] - trip["open_bar"],
        "why": trip["reason"],
    }]


def test_the_interval_is_the_data_s_own():
    assert bars(8).chart().spec()["interval"] == "1h"

    # a day missing from a daily clock leaves it daily
    days = probatum.Data(np.delete(table(9), 4, axis=0), symbol="BTCUSDT",
                         market="linear",
                         time=np.delete(START + np.arange(9, dtype=np.int64) * 24 * HOUR,
                                        4))

    assert days.chart().spec()["interval"] == "1d"


def test_focus_moves_the_viewport_and_leaves_every_bar_in_the_data():
    spec = bars(8).chart(focus=(2, 5)).spec()

    assert spec["focus"] == [2.0, 5.0]
    assert spec["n"] == 8 and len(spec["t"]) == 8


def test_focus_bounds_are_clamped_rather_than_refused():
    spec = bars(8).chart(focus=(2, 900)).spec()

    assert spec["focus"][1] <= 7.0


def test_focus_of_an_integer_means_the_last_n_bars():
    spec = bars(8).chart(focus=3).spec()

    assert spec["focus"] == [5.0, 7.0]


def test_focus_of_a_trade_frames_that_trade():
    result = run()
    spec   = result.chart(focus=result.trades[0]).spec()

    assert spec["focus"][0] <= result.trades[0]["open_bar"]


def test_focus_of_a_round_trip_still_open_frames_it_to_the_last_bar():
    result = probatum.backtest(BuyAtOpen(), bars(60), "frictionless")
    [held] = result.trades

    assert held["open"]

    spec = result.chart(focus=held).spec()

    assert spec["focus"][0] <= held["open_bar"]
    assert spec["focus"][1] == 59.0


def test_focus_of_anything_else_is_refused():
    with pytest.raises(ProbatumTypeError):
        bars(8).chart(focus="last week")

    with pytest.raises(ProbatumValueError):
        bars(8).chart(focus=(2.0, float("nan")))

    # a flag is no number of bars, and none of them frames nothing
    with pytest.raises(ProbatumTypeError):
        bars(8).chart(focus=True)

    with pytest.raises(ProbatumValueError):
        bars(8).chart(focus=0)


def test_focus_refuses_a_round_trip_the_run_did_not_make():
    later = probatum.backtest(Late(start=20), bars(40), "frictionless").trades[0]

    with pytest.raises(ProbatumValueError):
        run().chart(focus=later)

    with pytest.raises(ProbatumTypeError):
        run().chart(focus={"open_bar": 1})


def test_two_charts_of_the_same_inputs_produce_the_same_document():
    def once():
        return run().chart(Line(np.arange(8, dtype=float), "f"))._html()

    assert once() == once()


def test_a_series_name_containing_a_closing_script_tag_cannot_break_the_document():
    document = bars(8).chart(Line(np.arange(8, dtype=float), "</script><b>x"))._html()

    assert "</script><b>x" not in document
    assert document.count("</script>") == 2


def test_a_title_containing_markup_is_escaped_in_the_title_element():
    document = bars(8).chart(title="<b>hi</b>")._html()

    assert "<title>&lt;b&gt;hi&lt;/b&gt;</title>" in document


def test_the_saved_file_carries_its_renderer_and_loads_no_url(tmp_path):
    path     = run().chart().save(tmp_path / "run.html")
    document = pathlib.Path(path).read_text(encoding="utf-8")

    assert "TradingView" in document or "Lightweight Charts" in document
    assert 'src="http' not in document
    assert "src='http" not in document
    # the one link is the renderer's credit, which its licence asks the page to give
    assert re.findall(r'href="(http[^"]*)"', document) == ["https://www.tradingview.com/"]


def test_save_returns_the_path_it_was_given(tmp_path):
    target = tmp_path / "nested" / "run.html"

    assert bars(8).chart().save(target) == target


class Tags(html.parser.HTMLParser):
    """The start tags of what it is fed, each with its attributes as a browser reads
    them."""


    def __init__(self):
        super().__init__()
        self.tags = []


    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_a_chart_left_as_a_cell_s_last_line_draws_its_own_document_in_a_frame():
    # a notebook calls _repr_html_ on what a cell leaves, and the frame carries the
    # whole document inline, at the chart's own height. Parsed rather than matched,
    # because a quote left unescaped ends the attribute inside the document
    built = run().chart(height=480)
    read  = Tags()
    read.feed(built._repr_html_())

    assert read.tags == [
        ("div",    {"class": "probatum-chart"}),
        ("iframe", {"srcdoc": built._html(),
                    "style":  "width:100%;height:480px;border:0"}),
    ]


def test_show_outside_a_notebook_points_at_save():
    with pytest.raises(ProbatumRuntimeError) as excinfo:
        bars(8).chart().show()

    assert "save" in str(excinfo.value)


def test_defaults_returns_the_previous_settings_and_takes_only_known_keywords():
    previous = probatum.chart.defaults(height=700)

    try:
        assert bars(8).chart().spec()["height"] == 700
        assert bars(8).chart(height=300).spec()["height"] == 300
    finally:
        probatum.chart.defaults(**{k: v for k, v in previous.items() if v is not None})

    with pytest.raises(TypeError):
        probatum.chart.defaults(them="dark")


def test_both_palettes_travel_so_the_theme_toggle_works_offline():
    theme = bars(8).chart().spec()["theme"]

    assert set(theme["dark"]) == set(theme["light"])
    assert theme["mode"]      == "auto"


def test_auto_reaches_the_document_undecided():
    # the choice is made in the browser, because the author of a chart is not its
    # reader and Python is not there when a saved file is opened
    theme = bars(8).chart(theme="auto").spec()["theme"]

    assert theme["mode"]      == "auto"
    assert set(theme["dark"]) == set(theme["light"])


def test_a_theme_outside_the_three_is_refused_by_both_entry_points():
    for call in (lambda: bars(8).chart(theme="midnight"),
                 lambda: probatum.chart.defaults(theme="midnight")):
        with pytest.raises(ProbatumValueError):
            call()


def test_a_palette_on_the_call_tints_that_chart_and_leaves_the_next_one_alone():
    tinted = bars(8).chart(palette={"dark": {"s1": "#abcdef"}}).spec()
    plain  = bars(8).chart().spec()

    assert tinted["theme"]["dark"]["s1"] == "#abcdef"
    assert plain["theme"]["dark"]["s1"] != "#abcdef"
    # an override merges into the base palette rather than replacing it, so
    # branding one colour keeps the other thirteen
    assert tinted["theme"]["dark"]["s2"] == plain["theme"]["dark"]["s2"]
    assert tinted["theme"]["light"]      == plain["theme"]["light"]


def test_a_palette_on_the_call_replaces_the_session_one_rather_than_merging():
    previous = probatum.chart.defaults()

    try:
        probatum.chart.defaults(palette={"dark": {"s1": "#111111", "s2": "#222222"}})
        theme = bars(8).chart(palette={"dark": {"s1": "#333333"}}).spec()["theme"]

        # s1 is the call's and s2 is the base rather than the session's, because
        # one call site has to read as one appearance and not as the difference
        # between two of them
        assert theme["dark"]["s1"] == "#333333"
        assert theme["dark"]["s2"] != "#222222"
    finally:
        probatum.chart._DEFAULTS["palette"] = previous["palette"]


def test_a_palette_is_validated_the_same_way_on_the_call_as_on_the_session():
    for call in (lambda: bars(8).chart(palette={"midnight": {"s1": "#fff"}}),
                 lambda: probatum.chart.defaults(palette={"midnight": {"s1": "#fff"}})):
        with pytest.raises(ProbatumValueError):
            call()

    with pytest.raises(ProbatumValueError):
        bars(8).chart(palette={"dark": {"s1": "</style>"}})


def test_every_vendored_asset_ships_inside_the_wheel():
    # the gate copies no source tree, so there is no python/probatum to resolve
    # against here and this reads each asset out of the installed package.
    # maturin ships python-source with no include key, so a non-.py file rides
    # along by convention, and a packaging regression is invisible until a user
    # calls save()
    from probatum import chart
    names = ("chart.html", "chart.css", "lightweight-charts.js") + chart._ASSETS

    for name in names:
        assert chart._asset(name).strip()


def test_the_vendored_renderer_keeps_its_licence():
    from probatum import chart

    assert "Apache" in chart._asset("lightweight-charts.js")[:2000]
    assert "Apache" in chart._asset("LICENSE-lightweight-charts")


def test_the_renderer_s_notice_and_link_sit_in_the_footer_as_its_licence_asks():
    # the licence asks for its notice and a link to TradingView on the page; the logo
    # over the plot would give the link alone, so the footer gives both (ADR 0042)
    from probatum import chart

    assert "attributionLogo: false" in chart._asset("core.js")
    assert ('<a class="credit" href="https://www.tradingview.com/" target="_blank" '
            'rel="noopener">TradingView Lightweight Charts&trade; Copyright (c) 2026 '
            'TradingView, Inc.</a>') in " ".join(chart._asset("chart.html").split())


def test_the_manifest_lists_every_script_in_the_static_directory():
    # the bundle is built from an explicit tuple so a stray file cannot ship;
    # this is the other half of that bargain, catching a file added to _static
    # and never wired in
    from probatum import chart
    on_disk = sorted(
        str(p.relative_to(chart._STATIC)).replace("\\", "/")
        for p in chart._STATIC.rglob("*.js")
        if p.name != "lightweight-charts.js"
    )

    assert on_disk == sorted(chart._ASSETS)


def test_no_top_level_javascript_name_is_declared_twice():
    # the assets are concatenated into one scope, so a repeated top-level const
    # throws at load and takes the whole chart with it. There is no bundler and
    # no javascript engine in the gate, so the check is done on the text
    from probatum import chart
    pattern = re.compile(r"^(?:const|let|var|function|class)\s+(\w+)", re.M)
    seen    = {}

    for name in chart._ASSETS:
        for declared in pattern.findall(chart._asset(name)):
            assert declared not in seen, f"{declared} in {seen.get(declared)} and {name}"
            seen[declared] = name


def test_every_asset_opts_into_strict_mode():
    # in development the assets load as separate script tags and in the wheel
    # they load inside one iife; a per-file directive is what makes the two agree
    from probatum import chart

    for name in chart._ASSETS:
        assert chart._asset(name).lstrip().startswith('"use strict";')


def test_no_colour_is_written_into_the_javascript():
    # python decides what is drawn and javascript decides only how it is painted,
    # so a colour on that side is a decision in the wrong half. Full transparency
    # is not a colour, it is a way of drawing nothing
    from probatum import chart
    pattern = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\(")

    for name in chart._ASSETS:
        for hit in pattern.findall(chart._asset(name)):
            assert hit == "rgba(", f"{name} carries a colour: {hit}"

    for text in (chart._asset(n) for n in chart._ASSETS):
        for hit in re.findall(r"rgba\([^)]*\)", text):
            assert hit == "rgba(0,0,0,0)", f"a javascript asset carries {hit}"


def test_no_alpha_is_composed_in_the_javascript():
    # the pattern above cannot see this shape: an alpha byte concatenated onto a
    # theme colour carries no # of its own, so `t.up + "66"` walked past it for
    # the life of the layer, in two files that had to agree on the byte. Choosing
    # an alpha is choosing a colour, and it belongs on the python side (ADR 0042)
    from probatum import chart

    for name in chart._ASSETS:
        hit = re.search(r'\+\s*"[0-9a-fA-F]{2}"', chart._asset(name))
        assert hit is None, f"{name} composes a colour: {hit.group(0)}"


def test_importing_probatum_pulls_in_no_optional_dependency():
    # pandas is duck-typed and IPython is imported inside show(); a stray
    # module-level import would make probatum unimportable for anyone without them
    code = ("import sys, probatum; "
            "print(int('pandas' in sys.modules), int('IPython' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "0"]


def test_a_run_puts_four_numbers_above_the_chart():
    spec = run().chart(title="a run").spec()

    assert spec["title"] == "a run"

    labels = [row[0] for row in spec["headline"]]
    assert labels == ["return", "sharpe", "max drawdown", "round trips"]


def test_the_headline_is_preformatted_so_the_renderer_decides_nothing():
    spec = run().chart().spec()

    for label, value in spec["headline"]:
        assert isinstance(label, str) and isinstance(value, str)


def test_a_non_finite_stat_reads_as_not_available_rather_than_breaking():
    spec = run().chart(stats=["profit_factor"]).spec()

    assert spec["headline"] == [["profit factor", "n/a"]]

    json.dumps(spec, allow_nan=False)


def test_stats_chooses_which_numbers_are_shown():
    spec = run().chart(stats=["sharpe", "exposure_pct"]).spec()

    assert [row[0] for row in spec["headline"]] == ["sharpe", "exposure"]


def test_an_empty_stats_list_shows_none():
    assert "headline" not in run().chart(stats=[]).spec()


def test_stats_naming_a_key_the_run_does_not_report_is_refused():
    with pytest.raises(ProbatumKeyError) as excinfo:
        run().chart(stats=["alpha"])

    assert "'alpha'" in str(excinfo.value) and "sharpe" in str(excinfo.value)


def test_a_run_with_no_statistics_draws_none_and_refuses_any_asked_for():
    # one bar is too short a clock to annualise, so the run reports no statistics
    one   = probatum.Data(table(1), symbol="BTCUSDT", market="linear", time=[START],
                          interval="1h")
    short = probatum.backtest(np.ones(1), one, "frictionless")

    assert short.stats is None

    # one bar's volume never varies, which is a dead feed's
    with pytest.warns(ProbatumWarning):
        spec = short.chart().spec()

        with pytest.raises(ProbatumValueError, match="stats: the run has no statistics, "
                           "its clock too short to annualise"):
            short.chart(stats=["sharpe"])

    assert "headline" not in spec and "stats" not in spec
    assert spec["equity"]["v"] == [short.equity[0]]


def test_a_chart_with_no_title_and_no_run_has_no_headline():
    spec = bars(8).chart().spec()

    assert "headline" not in spec and "title" not in spec


def test_the_headline_is_written_as_text_rather_than_as_markup():
    from probatum import chart
    controls = chart._asset("controls.js")

    assert "textContent" in controls

    document = bars(8).chart(title="<b>hi</b>")._html()

    assert "<b>hi</b>" not in document


def test_future_lengthens_the_axis_and_leaves_the_candles_alone():
    spec = bars(8).chart(future=3).spec()

    assert spec["n"]         == 8
    assert len(spec["t"])    == 11
    assert len(spec["ohlc"]) == 8


def test_the_projected_bars_continue_the_interval():
    spec = bars(8).chart(future=3).spec()
    gaps = np.diff(spec["t"])

    assert set(gaps.tolist()) == {3_600}


def test_a_series_running_past_the_last_candle_is_accepted_only_with_future():
    ahead = np.arange(11, dtype=float)

    with pytest.raises(ProbatumValueError):
        bars(8).chart(Line(ahead, "cloud"))

    spec = bars(8).chart(Line(ahead, "cloud"), future=3).spec()

    assert spec["series"][0]["i0"]     == 0
    assert len(spec["series"][0]["v"]) == 11


def test_the_length_error_names_the_projected_length_when_future_is_set():
    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(Line(np.arange(5, dtype=float), "f"), future=3)

    message = str(excinfo.value)
    assert "11" in message and "past the last candle" in message


def test_the_two_ordinary_lengths_still_work_with_future_set():
    spec = bars(8).chart(Line(np.arange(8, dtype=float), "a"),
                         Line(np.arange(7, dtype=float), "b"), future=3).spec()

    assert spec["series"][0]["i0"] == 0
    assert spec["series"][1]["i0"] == 1


def test_a_band_can_be_projected_which_is_the_whole_point_of_a_cloud():
    upper = np.arange(11, dtype=float) + 10
    lower = np.arange(11, dtype=float)

    spec  = bars(8).chart(Band(upper, lower, "cloud", panel="x"), future=3).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]

    assert entry["i0"] == 0
    assert len(entry["v"]) == len(entry["v2"]) == 11


def test_a_run_can_be_drawn_without_its_trades():
    with_them = run().chart().spec()
    without   = run().chart(trades=False).spec()

    assert with_them["trades"]
    assert without["trades"] == []
    assert without["equity"] == with_them["equity"]


def test_hiding_a_panel_drops_its_data_and_not_just_its_pixels():
    # drawdown="panel" so the pane is there to be hidden. Without it the default
    # carries no drawdown pane either, and the panel-list assertion below would
    # hold just as well against a chart that never hid anything
    shown  = run().chart(drawdown="panel").spec()
    hidden = run().chart(drawdown="panel", panels=[Panel("drawdown", show=False)]).spec()

    assert [p["name"] for p in shown["panels"]] == [
        "price", "volume", "equity", "drawdown",
    ]
    assert "drawdown" in shown
    assert "drawdown" not in hidden
    assert [p["name"] for p in hidden["panels"]] == ["price", "volume", "equity"]


def test_markers_draws_one_entry_holding_every_bar_the_mask_is_true_on():
    mask    = np.array([False, True, False, True, True, False, False, False])
    spec    = bars(8).chart(Markers(mask=mask)).spec()
    entries = [s for s in spec["series"] if s["kind"] == "marker"]

    assert len(entries)       == 1
    assert entries[0]["bars"] == [1, 3, 4]


def test_markers_one_bar_short_lands_a_bar_later_like_every_other_mark():
    mask = np.array([False, True, False, False, False, False, False])
    spec = bars(8).chart(Markers(mask=mask)).spec()

    assert [s for s in spec["series"] if s["kind"] == "marker"][0]["bars"] == [2]


def test_markers_anchors_each_glyph_to_its_own_bar_when_given_an_array():
    mask   = np.array([False, True, False, True, False, False, False, False])
    values = np.arange(8, dtype=float) + 100.0

    spec  = bars(8).chart(Markers(mask=mask, value=values)).spec()
    entry = [s for s in spec["series"] if s["kind"] == "marker"][0]

    assert entry["bars"]   == [1, 3]
    assert entry["values"] == [101.0, 103.0]


def test_markers_refuses_a_float_mask_because_nan_is_truthy():
    # how a comparison arrives once a warm-up NaN has been through it, and
    # astype(bool) would put a glyph on the one bar nothing could be decided on
    with pytest.raises(ProbatumTypeError) as excinfo:
        Markers(mask=np.array([0.0, float("nan"), 1.0]))

    assert "NaN is truthy" in str(excinfo.value)


def test_a_marker_may_sit_in_the_projected_room():
    spec = bars(8).chart(Marker(9), future=3).spec()

    assert [s for s in spec["series"] if s["kind"] == "marker"][0]["bars"] == [9]

    with pytest.raises(ProbatumValueError):
        bars(8).chart(Marker(11), future=3)


def test_focus_reaches_the_projected_room():
    assert "focus" not in bars(8).chart(future=3).spec()

    windowed = bars(8).chart(future=3, focus=4).spec()

    assert windowed["focus"] == [4.0, 10.0]


def test_a_data_of_one_bar_projects_on_its_own_interval():
    # one bar has no gap to read an interval off, so the Data is given one, and
    # the axis grows by it
    one = probatum.Data(table(1), symbol="BTCUSDT", market="linear",
                        time=[START], interval="4h")

    with pytest.warns(ProbatumWarning):
        spec = one.chart(future=2).spec()

    assert np.diff(spec["t"]).tolist() == [14_400, 14_400]


def test_future_must_be_zero_or_positive():
    with pytest.raises(ProbatumValueError):
        bars(8).chart(future=-1)

    with pytest.raises(ProbatumTypeError):
        bars(8).chart(future="3")


def test_the_renderer_lets_the_cursor_reach_the_projected_room():
    # spec.n counts candles and spec.t counts the axis; the crosshair must
    # travel the axis or the projection cannot be read
    from probatum import chart

    assert "spec.t.length - 1" in chart._asset("core.js")
    assert "i < SPEC.n" in chart._asset("legend.js")


def test_a_non_string_title_does_not_crash_inside_save(tmp_path):
    bars(8).chart(title=20260803).save(tmp_path / "run.html")


def test_a_background_label_mapped_to_nothing_ships_no_fill():
    # a null in fills breaks the renderer's gradient cache on every repaint, even
    # when no bar carries the label
    labels = np.array(["calm"] * 7 + ["wild"], dtype=object)
    spec   = bars(8).chart(Background(labels, fill={"wild": "#ff0000", "calm": None})).spec()
    entry  = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert entry["fills"] == [["#ff0000"]]
    assert entry["spans"] == [[7, 8, 0]]


def test_a_data_with_no_bars_is_refused_at_a_guard():
    with pytest.raises(DataError) as excinfo:
        bars(8)[:0].chart()

    assert "no bars" in str(excinfo.value)


def test_an_instrument_that_printed_no_bar_is_refused():
    # the rows before ETHUSDT listed
    with pytest.raises(DataError) as excinfo:
        pair(late=3)[:3].chart(instrument="ETHUSDT")

    assert "ETHUSDT printed no bar" in str(excinfo.value)


def test_a_pandas_nullable_mask_does_not_crash_a_background():
    # bool(pandas.NA) raises rather than answering, and a value that cannot answer
    # the question is not a condition being met
    pd   = pytest.importorskip("pandas")
    mask = pd.array([True, False, True, None, True, False, True, False],
                    dtype="boolean")

    spec  = bars(8).chart(Background(mask, fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]

    assert [span[2] for span in entry["spans"]] == [0, 0, 0, 0]


def test_a_nan_in_candle_color_is_read_as_no_colour_rather_than_crashing():
    tint = np.array(["#2fe0a8"] * 7 + [np.nan], dtype=object)
    spec = bars(8).chart(candle_color=tint).spec()

    assert spec["candles"]["color"]["k"][-1] == 0

    json.dumps(spec, allow_nan=False)


def test_a_per_bar_band_colour_becomes_a_palette_like_every_other_mark():
    colors = ["#ff0000"] * 4 + ["#00ff00"] * 4
    spec   = bars(8).chart(
        Band(np.arange(8, dtype=float), 3.0, panel="x", color=colors)
    ).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]

    assert entry["color"]["p"] == [None, "#ff0000", "#00ff00"]


def test_a_line_with_a_fill_becomes_an_area_that_is_actually_drawn():
    from probatum import chart
    spec = bars(8).chart(
        Line(np.arange(8, dtype=float), "f", fill=("#00000000", "#000000ff"))
    ).spec()

    assert spec["series"][0]["fill"] == ["#00000000", "#000000ff"]

    core = chart._asset("core.js")
    assert "spec.fill" in core, "the renderer must read the fill it is given"


def test_a_line_refuses_a_per_bar_fill_and_names_the_substitute():
    with pytest.raises(ProbatumTypeError) as excinfo:
        Line(np.arange(8, dtype=float), fill=["#ff0000"] * 8)

    assert "Band" in str(excinfo.value)


def test_a_per_bar_band_fill_becomes_a_palette():
    colors = ["#ff0000"] * 4 + ["#00ff00"] * 4
    spec   = bars(8).chart(
        Band(np.arange(8, dtype=float), 3.0, panel="x", fill=colors)
    ).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]

    assert entry["fill"]["p"] == [None, "#ff0000", "#00ff00"]
    assert entry["fill"]["k"] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_a_per_bar_band_fill_of_the_wrong_length_raises():
    with pytest.raises(ProbatumValueError):
        bars(8).chart(Band(
            np.arange(8, dtype=float),
            3.0,
            panel = "x",
            fill  = ["#ff0000"] * 5,
        ))


def test_a_gradient_of_three_stops_is_still_a_gradient():
    stops = ("#000000", "#111111", "#222222")
    spec  = bars(8).chart(Band(
        np.arange(8, dtype=float),
        3.0,
        panel = "x",
        fill  = stops,
    )).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]

    assert entry["fill"] == list(stops)


def test_the_band_primitive_cuts_its_shape_where_a_per_bar_fill_changes():
    # one canvas path carries one fillStyle, so a two coloured region is two
    # shapes that have to meet rather than one gradient
    from probatum import chart
    band = chart._asset("primitives/band.js")

    assert "perBar" in band
    assert "colorAt(spec.fill" in band


def test_the_band_primitive_reads_the_palette_it_is_handed():
    # a palette object assigned to strokeStyle is a canvas no-op, so a per-bar
    # band colour used to be accepted, encoded correctly, and then painted in
    # the default colour with nothing anywhere saying so
    from probatum import chart

    assert "colorAt(" in chart._asset("primitives/band.js")


def test_a_per_bar_band_colour_of_the_wrong_length_raises():
    with pytest.raises(ProbatumValueError):
        bars(8).chart(Band(np.arange(8, dtype=float), 3.0, panel="x",
                           color=["#ff0000"] * 3))


def test_volume_keeps_the_precision_its_own_panel_advertises():
    spec  = bars(8, volume=np.linspace(0.00432, 0.00711, 8)).chart().spec()
    panel = [p for p in spec["panels"] if p.get("volume")][0]

    assert len(set(spec["vol"]["v"])) == 8, "rounding flattened the column"
    assert panel["digits"] >= 4


def test_a_panel_name_carrying_markup_is_refused():
    # controls.js writes the name into a DOM attribute, which payload escaping
    # cannot reach because the script reads the payload as a literal, whose escapes
    # give back the original
    with pytest.raises(ProbatumValueError):
        bars(8).chart(Line(np.arange(8, dtype=float), panel='"><script>'))


def test_focus_never_names_a_bar_outside_the_data():
    assert bars(8).chart(focus=1).spec()["focus"] == [6.0, 7.0]
    assert bars(8).chart(focus=(7, 7)).spec()["focus"][1] <= 7.0


def test_naming_a_panel_twice_in_panels_is_refused():
    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(panels=[Panel("equity", weight=2), Panel("equity", scale="log")])

    assert "twice" in str(excinfo.value)


def test_panels_takes_panel_objects_alone():
    with pytest.raises(ProbatumTypeError):
        bars(8).chart(panels=["equity"])


def test_a_chart_and_the_defaults_enforce_the_same_height_floor():
    for call in (lambda: bars(8).chart(height=40),
                 lambda: probatum.chart.defaults(height=40)):
        with pytest.raises(ProbatumValueError):
            call()


def test_a_height_is_a_whole_number_of_pixels():
    with pytest.raises(ProbatumTypeError):
        bars(8).chart(height="480")

    for height in (480.5, float("inf")):
        with pytest.raises(ProbatumValueError):
            bars(8).chart(height=height)


def test_height_sizes_a_notebook_cell_and_a_saved_file_uses_the_window():
    # one file has to be right on a laptop and on a wide monitor, and another
    # page embedding it should size it from the outside
    tall  = bars(8).chart(height=900)
    short = bars(8).chart(height=200)

    assert tall.spec()["height"] == 900 and short.spec()["height"] == 200
    assert "height: 100%" in tall._html()
    # the number reaches the document only as a value show() reads back out
    assert '"height":900' in tall._html()
    assert len(tall._html()) == len(short._html())


def test_a_list_valued_palette_entry_is_validated_rather_than_stringified():
    # str(["#a", "#b"]) contains nothing _UNSAFE looks for, so the repr used to
    # pass the check and ship where the renderer needs three entries
    previous = probatum.chart.defaults()

    try:
        probatum.chart.defaults(palette={"dark": {"selected": ["#000000", "#111111"]}})
        theme = bars(8).chart().spec()["theme"]

        assert theme["dark"]["selected"] == ["#000000", "#111111"]
        with pytest.raises(ProbatumValueError):
            probatum.chart.defaults(palette={"dark": {"selected": ["</style>"]}})
    finally:
        probatum.chart.defaults(theme=previous["theme"], height=previous["height"])
        probatum.chart._DEFAULTS["palette"] = previous["palette"]


def test_defaults_changes_nothing_when_one_argument_is_bad():
    previous = probatum.chart.defaults()

    with pytest.raises(ProbatumValueError):
        probatum.chart.defaults(theme="light", height=40)

    assert probatum.chart.defaults() == previous


def test_the_log_guard_names_the_bar_a_short_series_is_drawn_on():
    # a T-1 series starts at bar 1, so the array index is one less than the bar
    values = np.arange(1, 8, dtype=float)
    values[2] = 0.0

    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(Line(values, "f", panel="x"), panels=[Panel("x", scale="log")])

    assert "bar 3" in str(excinfo.value)


def test_the_log_guard_sees_a_level_and_a_bands_scalar_edge():
    with pytest.raises(ProbatumValueError):
        bars(8).chart(Line(np.arange(1, 9, dtype=float), "f", panel="x"),
                      Level(0.0, panel="x"), panels=[Panel("x", scale="log")])

    with pytest.raises(ProbatumValueError):
        bars(8).chart(Band(np.arange(1, 9, dtype=float), 0.0, panel="y"),
                      panels=[Panel("y", scale="log")])


def test_a_bands_lower_edge_widens_the_panel_extent():
    upper = np.full(8, 10.0)
    lower = np.full(8, -500.0)

    spec  = bars(8).chart(Band(upper, lower, panel="x")).spec()
    panel = [p for p in spec["panels"] if p["name"] == "x"][0]

    assert panel["digits"] <= 1


def test_panels_configures_and_never_creates():
    names = [p["name"] for p in
             bars(8).chart(panels=[Panel("momentom", weight=3)]).spec()["panels"]]

    assert "momentom" not in names


def test_a_mark_named_after_an_automatic_panel_does_not_land_on_it():
    # _place has already refused this from the candles; letting the label become
    # a registry key would put it back on them
    spec = bars(8).chart(Line(np.full(8, 5000.0), "price")).spec()

    assert spec["series"][0]["panel"] != "price"


def test_a_label_carrying_markup_is_stripped_when_it_becomes_a_panel_name():
    # the caller wrote a legend label, not a panel name, so the derived key is
    # cleaned rather than refused; the label itself still travels intact
    spec = bars(8).chart(Line(np.full(8, 5000.0), "</script>squeeze")).spec()

    assert spec["series"][0]["name"] == "</script>squeeze"

    panel = spec["series"][0]["panel"]
    assert not set(panel) & set("<>\";{}")
    assert "squeeze" in panel


def test_a_wrong_length_bare_array_says_which_one_it_was():
    with pytest.raises(ProbatumValueError) as excinfo:
        bars(8).chart(np.arange(8, dtype=float), np.arange(3, dtype=float))

    assert "series 1" in str(excinfo.value)


def key_paths(node, prefix=""):
    if isinstance(node, dict):
        out = []
        for key in node:
            out += key_paths(node[key], f"{prefix}.{key}" if prefix else key)
        return out

    if isinstance(node, list):
        if not node:
            return [prefix + "[]"]

        # every element, not the first: series is a union of six mark kinds, and
        # reading element zero recorded the keys of a Line and nothing else, so
        # renaming a key on any of the other five passed this test untouched
        out = []
        for item in node:
            out += key_paths(item, prefix + "[]")
        return out

    return [prefix]


def test_the_document_carries_exactly_the_keys_the_renderer_reads():
    # the javascript half cannot be run in this gate and its whole contract is
    # the key set, so a renamed key would pass every other test here and draw
    # nothing. Regenerate it with dev/golden.py, in the same commit that bumps
    # the spec's schema number
    path   = pathlib.Path(__file__).parent / "chart_schema.json"
    golden = json.loads(path.read_text(encoding="utf-8"))

    assert sorted(set(key_paths(everything().spec()))) == golden


def test_a_marker_and_a_markers_carry_the_same_keywords_into_the_document():
    # one primitive draws both, so a keyword one branch writes and the other drops
    # is a difference with no cause. The golden could not catch this: it records
    # the keys that were there, so an absent one reads as the contract
    mask = np.zeros(8, dtype=bool)
    mask[3] = True

    spec = bars(8).chart([
        Marker(3, text="label", color="#ff0000"),
        Markers(mask, text="label", color="#ff0000"),
    ]).spec()
    single, plural = spec["series"][0], spec["series"][1]

    assert single["text"] == plural["text"] == "label"
    assert single["color"] == plural["color"] == "#ff0000"
    assert set(single) == set(plural)


def test_a_marker_given_neither_keyword_spends_no_bytes_on_them():
    spec  = bars(8).chart(Marker(3)).spec()
    entry = spec["series"][0]

    assert "text" not in entry and "color" not in entry


def test_a_forced_close_is_marked_in_the_trade_payload():
    # the golden covers everything() and everything() never dies, so this key is
    # conditional and has to be asserted here instead. A forced close reads as an
    # ordinary exit otherwise, and the leverage that caused it goes unread
    data, dead = crashed()

    assert dead.bust

    rows = dead.chart().spec()["trades"]

    assert any(row.get("liq") for row in rows)

    ordinary = run().chart().spec()["trades"]

    assert ordinary
    assert not any("liq" in row for row in ordinary)


def test_the_drawdown_panel_falls_from_the_running_peak_not_from_the_balance():
    # the peak is the account's running maximum from the opening balance on. A run
    # that never makes a new high before its worst bar measures the same either
    # way, so this one doubles and then gives it all back
    close  = np.array([100.0, 100.0, 200.0, 100.0])
    result = probatum.backtest(BuyAtOpen(), bars(4, open=close, high=close + 1.0,
                                                 low=close - 1.0, close=close),
                               "frictionless")

    # 50 units bought at 100, so the account rides to 15,000 and back to 10,000
    assert result.equity.tolist() == pytest.approx([10_000.0, 10_000.0, 15_000.0,
                                                    10_000.0])

    fallen = result.chart().spec()["drawdown"]["v"]

    # a third given back from the peak, where an anchor on the opening balance
    # reports a run that never fell at all. The tolerance is the track's own
    # display rounding and nothing looser: the error being hunted is 33 points
    assert min(fallen) == pytest.approx(-100.0 / 3.0, abs=1e-3)
    assert result.stats["max_drawdown_pct"] == pytest.approx(100.0 / 3.0, abs=1e-9)


def test_a_line_ships_the_values_it_was_handed_not_the_panels_rounding():
    feature = np.full(8, 1.2345e-05)
    spec    = bars(8).chart(Line(feature, "tiny", panel="price")).spec()
    line    = next(s for s in spec["series"] if s.get("name") == "tiny")

    assert line["v"][0] == pytest.approx(1.2345e-05, rel=1e-9)
    assert line["v"][0] != 0.0
