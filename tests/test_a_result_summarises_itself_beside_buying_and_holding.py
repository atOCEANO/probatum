"""A Result's summary is a frame of its headline statistics as its record holds them,
its costs and the bars it was liquidated on, beside buying and holding over the same
rows and each one's interval or why it has none, and its context, what ran, where, on
which bars and from which balance, and its assumptions; it shows each number to its
row's places, its context in a frame beside it and its reasons numbered in a frame
under them (Specification, Summaries).
"""

import pickle
import sys

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import MissingDependency, stats
from probatum._result import HOLDING, SUMMARY, Summary, _money

HOUR  = 3_600_000_000_000
VENUE = probatum.venues.bybit_perp.replace(funding="none")
FREE  = probatum.Venue(balance=10_000.0, fees={"maker": 0.0, "taker": 0.0})
LEANS = ("order limits, fills in the bar, an instrument with no bar, auto-deleveraging, a "
         "flip held back, universe")


def data(rows, seed=9):
    close = 100.0 + np.random.default_rng(seed).normal(0.0, 1.0, rows).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(rows, 1e3)])

    return probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * HOUR,
    )


def stepped(steps):
    """Bars whose closes move by ``steps``, each opening at the close before, so a run
    held at a weight of one earns each step to rounding."""
    close = 100.0 * np.concatenate([[1.0], np.cumprod(steps)])
    opens = np.concatenate([[close[0]], close[:-1]])
    high  = np.maximum(opens, close)
    low   = np.minimum(opens, close)
    bars  = np.column_stack([opens, high, low, close, np.full(close.size, 1e9)])

    return probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(close.size, dtype=np.int64) * HOUR,
    )


@pytest.fixture(scope="module")
def traded():
    weights = np.where(np.arange(1_000) % 20 < 10, 1.0, 0.0)
    return probatum.backtest(weights, data(1_000), VENUE)


def test_a_run_s_summary_is_its_record_beside_buying_and_holding(traded):
    frame  = traded.summary().to_pandas()
    record = traded.stats

    assert list(frame.index) == [*SUMMARY, "costs"]
    assert list(frame.columns) == ["run", "buy_and_hold", "low", "high", "level", "method",
                                   "why_none"]
    assert frame["run"][list(SUMMARY)].tolist() == pytest.approx(
        [record[name] for name in SUMMARY], nan_ok=True)
    assert frame.loc["costs", "run"] == pytest.approx(sum(traded.costs.values()))

    held  = np.asarray(traded.baselines["buy_and_hold"])
    given = traded.provenance
    moved = held[1:] / held[:-1] - 1.0

    assert frame.loc["total_return_pct", "buy_and_hold"] == pytest.approx(
        (held[-1] / held[0] - 1.0) * 100.0, rel=1e-12)
    assert frame.loc["max_drawdown_pct", "buy_and_hold"] == pytest.approx(
        np.max(1.0 - held / np.maximum.accumulate(held)) * 100.0, rel=1e-12)
    assert frame.loc["sharpe", "buy_and_hold"] == pytest.approx(stats.sharpe(
        moved, periods_per_year=given["periods_per_year"], risk_free=given["risk_free"]),
        rel=1e-9)
    assert frame["buy_and_hold"].drop(list(HOLDING)).isna().all()

    sharpe = record.interval("sharpe")

    assert frame.loc["sharpe", ["low", "high", "level"]].tolist() == [sharpe.low, sharpe.high,
                                                                     0.95]
    assert frame.loc["cagr_pct", "why_none"] == record._withheld["cagr_pct"]
    assert frame.loc["win_rate", "why_none"] == ("round trips: 50 closed; an interval of "
                                                 "them needs at least 60")


def test_its_context_is_what_ran_its_bars_its_balance_and_its_assumptions(traded):
    assert list(traded.summary().context.items()) == [
        ("policy", "a (1000, 1) array of weights"), ("venue", "bybit_perp"),
        ("data", "BTCUSDT 1h"), ("first", "1970-01-01"), ("last", "1970-02-11"),
        ("rows", 1_000), ("balance", "10,000"), ("assumptions", 37),
        ("graded optimistic", "funding"), ("lean optimistic", LEANS)]

    one = probatum.Data(
        np.array([[100.0, 100.5, 99.5, 100.0, 1e3]]),
        symbol   = "BTCUSDT",
        market   = "linear",
        time     = np.zeros(1, dtype=np.int64),
        interval = "1h",
    )
    short = probatum.backtest(np.ones(1), one, VENUE).summary()

    assert list(short.to_pandas().index) == ["costs"]
    assert list(short.context.items())[3:8] == [
        ("first", "1970-01-01 00:00"), ("last", "1970-01-01 01:00"), ("rows", 1),
        ("balance", "10,000"), ("statistics", "none, the clock too short to annualise")]


def test_it_shows_each_number_to_its_row_s_places_beside_its_context_over_its_reasons(
        traded):
    found                   = traded.summary()
    frame                   = found.to_pandas()
    shown, context, reasons = found._shown()
    record                  = traded.stats

    assert shown.loc["num_trades", "run"] == "50"
    assert shown.loc["sharpe", "run"] == f"{record['sharpe']:,.2f}"
    assert shown.loc["costs", "run"] == _money(sum(traded.costs.values()))
    assert shown.loc["cagr_pct", "run"] == "" and shown.loc["cagr_pct", "level"] == ""
    assert shown.loc["sharpe", "level"] == "0.95"
    assert list(context.index) == list(found.context) and context["value"]["rows"] == "1,000"

    why = list(dict.fromkeys(why for why in frame["why_none"] if why))

    assert reasons.to_dict("list") == {"why_none": why}
    assert list(reasons.index) == list(range(1, len(why) + 1))
    assert shown.loc["cagr_pct", "why_none"] == str(
        why.index(frame.loc["cagr_pct", "why_none"]) + 1)

    with pd.option_context("display.max_colwidth", None):
        parts = [shown, context, reasons]
        shows = found._repr_html_()
        at    = [shows.index(part._repr_html_()) for part in parts]

        assert repr(found) == "\n\n".join(repr(part) for part in parts) and at == sorted(at)
    assert traded._repr_html_() == shows


@pytest.mark.parametrize("steps, weight, why", [
    # 2% on one bar in five and 0.5% on the others, the one proportion of two values
    # whose Sharpe's influence is zero on every bar
    (np.where(np.arange(1, 401) % 5 == 0, 1.02, 1.005), 1.0,
     "returns: two values whose Sharpe's influence is zero on every bar"),
    (np.full(400, 1.005), 0.0, "returns: all 400 are the same; a ratio needs returns"),
    (np.array([1.01, 0.99, 1.02]), 1.0, "returns: 3 of them; an interval needs at least 4")])
def test_a_sharpe_with_no_interval_says_why(steps, weight, why):
    run = probatum.backtest(np.full(steps.size + 1, weight), stepped(steps), FREE)

    assert run.summary().to_pandas().loc["sharpe", "why_none"].startswith(why)


def test_a_liquidated_run_counts_the_bars_it_was_liquidated_on():
    steps = np.concatenate([np.full(20, 1.001), [0.85], np.full(20, 1.001)])
    found = probatum.backtest(np.full(steps.size + 1, 10.0), stepped(steps), VENUE).summary()

    assert list(found.to_pandas().index[-2:]) == ["costs", "liquidations"]
    assert found._shown()[0].loc["liquidations", "run"] == "1"


def test_a_coin_and_a_book_name_what_they_hold():
    closes = 30_000.0 + np.random.default_rng(9).normal(0.0, 100.0, 200).cumsum()
    frame  = pd.DataFrame({"open": closes, "high": closes + 50.0, "low": closes - 50.0,
                           "close": closes, "volume": np.full(200, 1e3)},
                          index=pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC",
                                              name="datetime"))
    frame.attrs.update(dict(exchange="binance", market_type="inverse", symbol="BTCUSD",
                            interval="1h", quote="USD", warnings=[],
                            volume_unit="contract", contract_size=100.0))

    coin  = probatum.backtest(np.where(np.arange(200) % 20 < 10, 1.0, 0.0),
                              probatum.Data(frame), probatum.Venue(balance=0.5))
    found = coin.summary()

    assert [found.context[name] for name in ("policy", "venue", "data", "rows",
                                             "balance")] == [
        "a (200, 1) array of weights", "built by hand", "inverse:BTCUSD 1h", 200, "0.5"]
    assert found.context["first"].startswith("2024-01-01")
    assert found.to_pandas().loc["costs", "run"] == coin.costs["fees"] > 0.0
    assert found._shown()[0].loc["costs", "run"] == f"{coin.costs['fees']:.3g}"

    book = probatum.combine({"a": coin}).summary()

    assert list(book.context.items())[:3] == [("accounts", "a"), ("margined", "apart"),
                                              ("data", "inverse:BTCUSD 1h")]


def test_an_amount_is_whole_from_a_hundred_three_figures_below_and_never_minus_0():
    printed = {
        12_345.6: "12,346",
        99.96:    "100",
        -0.00612: "-0.00612",
        5e-5:     "0.00005",
        1.234e-5: "0.00001234",
        -1e-9:    "0",
        -0.0:     "0",
    }

    assert {amount: _money(amount) for amount in printed} == printed


class Marked(probatum.Strategy):
    """A strategy that decides nothing, named with markup."""


    def __repr__(self):
        return "Marked<&>"


    def next(self, bar, account):
        pass


def test_a_policy_named_with_markup_is_escaped_in_a_notebook():
    shown = probatum.backtest(Marked(), data(60), VENUE)._repr_html_()

    assert "<td>Marked&lt;&amp;&gt;</td>" in shown and "Marked<&>" not in shown


def test_a_book_names_its_accounts_and_a_run_its_level_below_the_full_one(traded):
    other = probatum.backtest(np.where(np.arange(1_000) % 30 < 15, 1.0, 0.0), data(1_000),
                              VENUE)
    book  = probatum.combine({"A": traded, "B": other}).summary()

    assert list(book.context.items())[:3] == [("accounts", "A and B"), ("margined", "apart"),
                                              ("data", "BTCUSDT 1h")]

    kept  = probatum.backtest(np.where(np.arange(1_000) % 20 < 10, 1.0, 0.0), data(1_000),
                              VENUE, record="account")
    found = kept.summary()

    assert list(found.context.items())[5:8] == [("rows", 1_000), ("balance", "10,000"),
                                                ("recorded", "account")]
    assert found.to_pandas()["buy_and_hold"].isna().all()


def test_two_summaries_are_equal_where_their_rows_and_context_are(traded, tmp_path):
    traded.summary()
    traded.save(tmp_path / "run")

    assert traded.summary() == probatum.Result.load(tmp_path / "run").summary()
    assert traded.summary() == pickle.loads(pickle.dumps(traded)).summary()
    assert traded.summary() != probatum.backtest(np.ones(1_000), data(1_000), VENUE).summary()

    found = traded.summary()
    again = Summary(found._named, found._rows, found.context.items())
    moved = Summary(found._named, found._rows, {**found.context, "rows": 2}.items())

    assert again == found and moved != found


def test_without_pandas_a_summary_shows_its_rows_and_its_context(traded, monkeypatch):
    found = traded.summary()
    monkeypatch.setitem(sys.modules, "pandas", None)

    assert repr(found).startswith("Summary({'total_return_pct': (")
    assert repr(found).endswith(f", context={found.context!r})")
    assert found._repr_html_() is None

    with pytest.raises(MissingDependency, match="to_pandas: pandas is not installed"):
        found.to_pandas()
