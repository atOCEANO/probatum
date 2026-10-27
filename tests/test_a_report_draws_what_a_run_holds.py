"""A report draws what a Result or a Study holds and what probatum.stats reads of it,
and nothing of its own (ADR 0041): every report renders as one page with no script,
the same bytes each time, and each figure's plotted data, read back from its artists,
is the run's, as is each figure handed out alone. No pixel is compared, so the look
can change without a test moving.
"""

import re
import sys

import numpy as np
import pytest
from matplotlib.figure import Figure

import probatum
from probatum import (MissingDependency, ProbatumKeyError, ProbatumTypeError,
                      ProbatumValueError, report, stats)
from probatum.report import _figures

VENUE = probatum.venues.binance_perp.replace(funding="none")


def bars(rows, seed):
    close = 100.0 * np.exp(np.random.default_rng(seed).normal(0.0, 0.01, rows).cumsum())
    return np.column_stack([close, close * 1.003, close * 0.997, close, np.full(rows, 1e3)])


# four hours apart, so 900 rows cross five calendar months
TIME = 1_700_000_000 * 10**9 + np.arange(900, dtype=np.int64) * 4 * 3_600 * 10**9


@pytest.fixture(scope="module")
def data():
    return probatum.Data(bars(900, 1), symbol="BTCUSDT", market="linear", time=TIME)


def crossing(data, fast, slow):
    up   = np.asarray(probatum.ta.ema(data.close, fast)).reshape(-1)
    down = np.asarray(probatum.ta.ema(data.close, slow)).reshape(-1)
    return np.where(np.isnan(down), np.nan, np.where(up > down, 1.0, -0.5))


@pytest.fixture(scope="module")
def result(data):
    return probatum.backtest(crossing(data, 12, 48), data, VENUE, tolerance=0.05)


@pytest.fixture(scope="module")
def pair():
    held = probatum.Data(np.stack([bars(900, 1), bars(900, 2)], axis=1),
                         symbol=["BTCUSDT", "ETHUSDT"], market="linear", time=TIME)
    trend = crossing(probatum.Data(bars(900, 1), symbol="BTCUSDT", market="linear",
                                   time=TIME), 12, 48)
    return probatum.backtest(np.column_stack([trend, -trend]) * 0.5, held, VENUE,
                             tolerance=0.05)


SPACE = {"fast": probatum.Int(5, 40), "slow": probatum.Int(50, 200, step=10)}


@pytest.fixture(scope="module")
def study(data):
    return probatum.search(crossing, SPACE, data, VENUE, sampler="random", trials=30,
                           holdout=0.3, seed=7)


@pytest.fixture(scope="module")
def paths(data):
    return probatum.validate(crossing, data, VENUE, probatum.cv.WalkForward(300, 100),
                             space=SPACE, search={"sampler": "random", "trials": 8},
                             holdout=0.2, seed=3)


def drawn(make, given):
    return _figures.drawn(make, given)[0]


def lines(figure):
    return {line.get_label(): line for line in figure.axes[0].lines
            if not line.get_label().startswith("_")}


def test_every_report_renders_as_one_page_with_no_script(tmp_path, data, result, pair,
                                                          study, paths):
    kept = probatum.backtest(crossing(data, 12, 48), data, VENUE, tolerance=0.05,
                             record="account")
    null = probatum.search(crossing, SPACE, data, VENUE, sampler="random", trials=30,
                           holdout=0.3, seed=8)
    # a run of one instrument draws no weights, and one kept at the account level
    # keeps no baselines to draw beside it
    written = [(result.report(tmp_path / "run.html"), 9),
               (pair.report(tmp_path / "pair.html"), 11),
               (kept.report(tmp_path / "kept.html"), 9),
               (study.report(tmp_path / "search.html", null=null), 2),
               (paths.report(tmp_path / "nested" / "paths.html"), 2)]

    for path, figures in written:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<!doctype html>") and text.endswith("</html>\n")
        assert "<script" not in text and text.count("<figure>") == figures

    said = (tmp_path / "search.html").read_text(encoding="utf-8")
    assert "Parameter importance" in said and "Deflated Sharpe" in said
    assert "keeps no copy of its policy to play them" in (tmp_path / "kept.html").read_text(
        encoding="utf-8")


def test_a_run_that_loses_everything_still_writes_its_tear_sheet(tmp_path):
    # five times the balance held through a fall of 40% leaves the account at zero for
    # its last 80 rows, whose returns are nothing rather than zero over zero
    rows  = 120
    close = 100.0 * np.concatenate([np.full(20, 1.0), np.linspace(1.0, 0.6, 40),
                                    np.full(60, 0.6)])
    bars  = np.column_stack([close, close * 1.001, close * 0.999, close, np.full(rows, 1e3)])
    crash = probatum.Data(bars, symbol="BTCUSDT", market="linear", time=TIME[:rows])
    bust  = probatum.backtest(np.full(rows, 500.0), crash, VENUE, sizing="quantity")

    assert bust.bust and np.count_nonzero(bust.equity == 0.0) > 1
    text = bust.report(tmp_path / "bust.html").read_text(encoding="utf-8")
    assert text.count("<figure>") == 9
    assert (f'<td>strategy</td><td class="number">-100.00%</td><td class="number">'
            f'{bust.stats["sharpe"]:,.2f}</td>') in text


def test_a_baseline_that_never_varies_has_a_sharpe_of_0_at_any_rate(tmp_path, data):
    held = probatum.backtest(crossing(data, 12, 48), data, VENUE, tolerance=0.05,
                             risk_free=0.04)
    text = held.report(tmp_path / "rate.html").read_text(encoding="utf-8")

    assert ('<td>cash</td><td class="number">0.00%</td><td class="number">0.00</td>'
            in text)


def test_a_report_writes_the_same_bytes_every_time(tmp_path, result):
    first, second = result.report(tmp_path / "a.html"), result.report(tmp_path / "b.html")

    assert first.read_bytes() == second.read_bytes()


def test_a_page_s_figures_keep_their_ids_apart_and_reach_only_their_own(tmp_path, result):
    text = result.report(tmp_path / "run.html").read_text(encoding="utf-8")
    ids  = [found.split('"', 1)[0] for found in text.split(' id="')[1:]]

    assert len(ids) > 100 and len(ids) == len(set(ids))

    reached = re.findall(r'href="#([^"]+)"|url\(#([^)]+)\)', text)
    assert reached and {a or b for a, b in reached} <= set(ids)


def test_the_equity_and_the_drawdown_plot_the_run_and_its_baselines(result):
    held   = result.baselines
    when   = result.time.view("M8[ns]")
    equity = lines(drawn(_figures.equity, result))

    assert list(equity) == ["cash", "buy and hold", "without costs", "strategy"]
    for label, values in (("strategy", result.equity), ("cash", held["cash"]),
                          ("buy and hold", held["buy_and_hold"]),
                          ("without costs", held["frictionless"])):
        assert np.array_equal(equity[label].get_ydata(), values)
        assert np.array_equal(equity[label].get_xdata(), when)

    falls = lines(drawn(_figures.drawdown, result))
    assert np.array_equal(falls["strategy"].get_ydata(), -stats.drawdown(result))
    assert np.array_equal(falls["buy and hold"].get_ydata(),
                          -stats._fallen(np.asarray(held["buy_and_hold"])))


def counted(shown, values):
    """Whether bars ``shown`` are the histogram of ``values`` into as many bins."""
    counts, edges = np.histogram(values, bins=len(shown))
    return ([bar.get_height() for bar in shown] == counts.tolist()
            and [bar.get_x() for bar in shown] == edges[:-1].tolist())


def test_the_returns_and_each_month_plot_the_run_s_own(result):
    shown = drawn(_figures.returns, result).axes[0].patches
    assert counted(shown, result.returns[1:] * 100.0)
    assert sum(bar.get_height() for bar in shown) == len(result.time) - 1

    image = drawn(_figures.monthly, result).axes[0].images[0].get_array()
    table = stats.period_returns(result, by="month")
    assert image.count() == len(table) and len(table) > 4
    assert image.compressed() == pytest.approx(table["return_pct"], rel=1e-12)


def test_the_rolling_series_and_the_exposure_plot_what_stats_and_the_ledger_hold(result):
    size  = _figures.window(result)
    ratio = stats.rolling_sharpe(result, size)

    assert size == min(round(result.provenance["periods_per_year"] / 4.0),
                       (len(result.time) - 1) // 2)
    assert np.array_equal(lines(drawn(_figures.rolling_sharpe, result))["strategy"]
                          .get_ydata(), np.where(np.isinf(ratio), np.nan, ratio),
                          equal_nan=True)
    assert np.array_equal(lines(drawn(_figures.rolling_volatility, result))["strategy"]
                          .get_ydata(), stats.rolling_volatility(result, size),
                          equal_nan=True)

    shown  = lines(drawn(_figures.exposure, result))
    ledger = result.ledger
    assert np.array_equal(shown["gross"].get_ydata(), ledger["leverage"])
    assert np.array_equal(shown["net"].get_ydata(),
                          ledger["net_exposure"] / ledger["equity"])


def test_the_costs_and_the_round_trips_plot_what_the_run_paid_and_closed(result):
    paid = [bar.get_width() for bar in drawn(_figures.costs, result).axes[0].patches]
    assert paid == list(result.costs.values())

    shown  = drawn(_figures.trips, result).axes[0].patches
    closed = ~np.asarray(result.trades["open"])
    assert counted(shown, np.asarray(result.trades["ret"])[closed] * 100.0)
    assert sum(bar.get_height() for bar in shown) == result.stats["num_trades"]


def test_each_instrument_s_weight_and_pnl_are_plotted_where_a_run_holds_several(result,
                                                                                pair):
    assert drawn(_figures.weights, result) is None and drawn(_figures.pnl, result) is None

    for make, field in ((_figures.weights, "weight"), (_figures.pnl, "pnl")):
        shown = lines(drawn(make, pair))
        assert list(shown) == ["BTCUSDT", "ETHUSDT"]
        for name, line in shown.items():
            assert np.array_equal(line.get_ydata(), getattr(pair[name], field))


def test_a_study_s_figures_plot_its_trials_and_its_paths(study, paths):
    table    = study.trials
    score    = np.asarray(table["score"], dtype=np.float64)
    complete = (np.asarray(table["state"]) == "complete") & np.isfinite(score)
    points   = drawn(_figures.trials, study).axes[0].collections

    assert points[0].get_offsets().tolist() == np.column_stack(
        [np.asarray(table["number"])[complete], score[complete]]).tolist()
    assert points[1].get_offsets().tolist() == [[study._winner.number, study.best_score]]

    found = study.importance()
    axes  = drawn(_figures.importance, study).axes[0]
    assert [bar.get_width() for bar in axes.patches] == list(found.importance)
    assert axes.collections[0].get_offsets()[:, 0].tolist() == list(found.chance)

    scores = np.asarray(paths.paths["score"], dtype=np.float64)
    scored = [bar.get_height() for bar in drawn(_figures.paths, paths).axes[0].patches]
    assert scored == np.where(np.isfinite(scores), scores, 0.0).tolist()

    fitted = dict(zip(np.asarray(paths.fits["number"]).tolist(),
                      np.asarray(paths.fits["fitted"], dtype=np.float64).tolist()))
    pairs  = [(fitted[fit], score) for fit, score in zip(
        paths.windows["fit"].tolist(), np.asarray(paths.windows["score"]).tolist())]
    shown  = drawn(_figures.windows, paths).axes[0].collections[0].get_offsets()
    assert shown.tolist() == [[a, b] for a, b in pairs
                              if np.isfinite(a) and np.isfinite(b)]


def plotted(figure):
    """Every value ``figure`` plots, axes by axes: its lines' points, its bars' corners,
    its collections' offsets and paths, and its images' cells."""
    out = []
    for ax in figure.axes:
        out += [line.get_xydata() for line in ax.lines]
        out += [[bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()]
                for bar in ax.patches]
        for collection in ax.collections:
            out.append(collection.get_offsets())
            out += [path.vertices for path in collection.get_paths()]
        out += [image.get_array().filled(np.nan) for image in ax.images]

    return out


def test_each_figure_is_handed_out_alone_as_its_report_draws_it(pair, study, paths):
    for given, kind in ((pair, "sheet"), (study, "trials"), (paths, "paths")):
        for name in report.FIGURES[kind]:
            alone, paged = given.figure(name), drawn(getattr(_figures, name), given)
            values, shown = plotted(alone), plotted(paged)

            assert isinstance(alone, Figure) and values, name
            assert len(values) == len(shown) and all(
                np.array_equal(np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64), equal_nan=True)
                for a, b in zip(values, shown)), name

            # at the reports' look, which is set while a figure is drawn
            title, page = alone.axes[0].title, paged.axes[0].title
            assert alone.get_size_inches().tolist() == paged.get_size_inches().tolist()
            assert (title.get_fontsize(), title.get_fontname()) == (
                page.get_fontsize(), page.get_fontname())


def test_a_figure_no_report_draws_or_one_that_draws_nothing_is_refused(result, study):
    with pytest.raises(ProbatumValueError, match="name: 'weights' draws nothing here, "
                       "since it sets several instruments apart"):
        result.figure("weights")
    with pytest.raises(ProbatumKeyError, match="name: no figure 'sharpe'; use one of "
                       "equity, drawdown, returns"):
        result.figure("sharpe")
    with pytest.raises(ProbatumKeyError, match="name: no figure 'paths'; use one of "
                       "trials, importance"):
        study.figure("paths")
    with pytest.raises(ProbatumTypeError, match="name: 1; give a figure's name"):
        result.figure(1)


def test_a_report_without_matplotlib_names_the_line_that_installs_it(tmp_path, result,
                                                                      monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(MissingDependency, match="report: matplotlib is not installed; "
                       "pip install \"probatum\\[plot\\]\""):
        result.report(tmp_path / "run.html")
    assert not (tmp_path / "run.html").exists()

    with pytest.raises(MissingDependency, match="figure: matplotlib is not installed; "
                       "pip install \"probatum\\[plot\\]\""):
        result.figure("equity")


def test_a_path_that_is_no_path_is_refused(result):
    with pytest.raises(ProbatumTypeError, match="path: int; give a file path"):
        result.report(3)
