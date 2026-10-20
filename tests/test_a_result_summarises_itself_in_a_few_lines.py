"""A Result's summary says in a few lines what ran, where and on what bars, its span and
opening balance, its return, its Sharpe with the interval and the lag, or why it has
none, beside the long-run Sharpe where the two differ by more than 0.1, its maximum
drawdown as a loss with the longest time under water, its round trips, its costs, each
whole from a hundred and to three figures below, and its optimistic assumptions, and it
names a book's accounts as margined apart and a level below the full one (Specification,
Summaries).
"""

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import stats
from probatum._result import _money

HOUR  = 3_600_000_000_000
VENUE = probatum.venues.binance_perp.replace(funding="none")
FREE  = probatum.Venue(balance=10_000.0, fees={"maker": 0.0, "taker": 0.0})


def data(rows, seed=9, persist=0.0):
    moves = np.random.default_rng(seed).normal(0.0, 1.0, rows)
    for t in range(1, rows):
        moves[t] += persist * moves[t - 1]

    close = 100.0 + moves.cumsum()
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


def line(label, value, said):
    return f"  {label:<16}{value:>8}   {said}"


def longest_under_water(result):
    """The longest stretch from a row at the running peak to the first row back at or
    above it, or to the last row, in days, found the slow way."""
    equity, time = result.equity, result.time
    longest, peak, at = 0, equity[0], 0

    for row in range(1, len(equity)):
        if equity[row] >= peak:
            if row - at > 1 or equity[row - 1] < peak:
                longest = max(longest, time[row] - time[at])
            peak, at = equity[row], row

    if equity[-1] < peak:
        longest = max(longest, time[-1] - time[at])

    return longest / (24 * HOUR)


@pytest.fixture(scope="module")
def traded():
    weights = np.where(np.arange(1_000) % 20 < 10, 1.0, 0.0)
    return probatum.backtest(weights, data(1_000), VENUE)


def test_a_run_s_summary_reads_its_record_line_by_line(traded):
    record = traded.stats
    lines  = traded.summary().split("\n")

    assert lines[:2] == ["a (1000, 1) array of weights, binance_perp, BTCUSDT 1h",
                         "1970-01-01 to 1970-02-11, 1,000 bars, opening balance 10,000"]
    assert lines[2] == line("return", f"{record['total_return_pct']:+.1f}%",
                            f"CAGR {record['cagr_pct']:+.1f}%, volatility "
                            f"{record['volatility_pct']:.1f}%, in the market "
                            f"{record['exposure_pct']:.0f}% of bars")

    held = record.interval("sharpe")
    said = " ".join(lines[3:5]).split()
    assert said[:2] == ["sharpe", f"{record['sharpe']:.2f}"]
    assert " ".join(said[2:]) == (f"95% interval {held.low:.2f} to {held.high:.2f}, delta "
                                  f"method, Newey-West lag {held.lag}"
                                  + " floored at lag zero" * held.floored)

    shown = traded.summary()
    assert (f"max drawdown    {-record['max_drawdown_pct']:7.1f}%   "
            f"{longest_under_water(traded):,.0f} days under water") in shown
    assert line("round trips", "50", f"win rate {record['win_rate']:.0%}, profit factor "
                f"{record['profit_factor']:.2f}") in shown

    paid = sorted(((k, v) for k, v in traded.costs.items() if v), key=lambda kv: -abs(kv[1]))
    assert line("costs", _money(-sum(traded.costs.values())),
                ", ".join(f"{k} {_money(-v)}" for k, v in paid)) in shown
    assert shown.endswith(line("assumptions", "", "1 optimistic: funding not charged"))


def test_the_long_run_sharpe_is_shown_where_it_parts_from_the_sharpe_by_more_than_0_1():
    shown = []
    for persist in (0.0, 0.6):
        run    = probatum.backtest(np.ones(2_000), data(2_000, seed=4, persist=persist), VENUE)
        parted = abs(stats.sharpe(run, long_run=True) - run.stats["sharpe"]) > 0.1
        assert ("  long-run sharpe" in run.summary()) == parted
        shown.append(parted)

    assert True in shown


@pytest.mark.parametrize("steps, weight, why", [
    # 2% on one bar in five and 0.5% on the others, the one proportion of two values
    # whose Sharpe's influence is zero on every bar
    (np.where(np.arange(1, 401) % 5 == 0, 1.02, 1.005), 1.0,
     "its influence is zero on every bar"),
    (np.full(400, 1.005), 0.0, "the returns never vary"),
    (np.array([1.01, 0.99, 1.02]), 1.0, "3 returns, where one needs 4")])
def test_a_sharpe_with_no_interval_says_why(steps, weight, why):
    run = probatum.backtest(np.full(steps.size + 1, weight), stepped(steps), FREE)

    assert line("sharpe", f"{run.stats['sharpe']:.2f}",
                f"no interval: {why}") in run.summary().split("\n")


def test_a_coin_a_venue_built_by_hand_and_a_run_too_short_for_statistics_read_plainly():
    closes = 30_000.0 + np.random.default_rng(9).normal(0.0, 100.0, 200).cumsum()
    frame  = pd.DataFrame({"open": closes, "high": closes + 50.0, "low": closes - 50.0,
                           "close": closes, "volume": np.full(200, 1e3)},
                          index=pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC",
                                              name="datetime"))
    frame.attrs.update(dict(exchange="binance", market_type="inverse", symbol="BTCUSD",
                            interval="1h", quote="USD", warnings=[],
                            volume_unit="contract", contract_size=100.0))

    coin = probatum.backtest(np.where(np.arange(200) % 20 < 10, 1.0, 0.0),
                             probatum.Data(frame), probatum.Venue(balance=0.5))
    lines = coin.summary().split("\n")

    assert lines[0].startswith("a (200, 1) array of weights, a venue built by hand, ")
    assert lines[1].endswith(", 200 bars, opening balance 0.5")

    fees = coin.costs["fees"]
    assert line("costs", f"{-fees:.3g}", f"fees {-fees:.3g}") in lines

    book = probatum.combine({"a": coin}).summary().split("\n")
    assert book[1].endswith(", 200 bars, opening balance 0.5")
    assert line("costs", f"{-fees:.3g}", f"fees {-fees:.3g}") in book

    one = probatum.Data(
        np.array([[100.0, 100.5, 99.5, 100.0, 1e3]]),
        symbol   = "BTCUSDT",
        market   = "linear",
        time     = np.zeros(1, dtype=np.int64),
        interval = "1h",
    )
    short = probatum.backtest(np.ones(1), one, VENUE)

    assert short.stats is None
    assert short.summary().split("\n")[2:] == [
        line("statistics", "", "none, the clock too short to annualise"),
        line("costs", "0", "none"),
        line("assumptions", "", "1 optimistic: funding not charged")]


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


def test_a_book_names_its_accounts_and_a_run_its_level_below_the_full_one(traded):
    other = probatum.backtest(np.where(np.arange(1_000) % 30 < 15, 1.0, 0.0), data(1_000),
                              VENUE)

    book  = probatum.combine({"A": traded, "B": other})
    lines = book.summary().split("\n")

    assert lines[0] == "a book of A and B, BTCUSDT 1h"
    assert lines[-1] == line("accounts", "", "A and B, margined apart; no capital moves "
                             "between them")

    kept = probatum.backtest(np.where(np.arange(1_000) % 20 < 10, 1.0, 0.0), data(1_000),
                             VENUE, record="account")
    assert kept.summary().split("\n")[1].endswith(", recorded at the account level")
