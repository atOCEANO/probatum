"""``combine`` adds Results from accounts at different venues into one book on the times
any of them has a row: each account held at its opening balance before its first row
and at its last equity after its end, the sum taken in the order given, the book's
ledger its own, its other tables theirs joined with each row's venue, and its
statistics those of the sum, computed as a run's are. One book combined is itself, two
copies are twice it, and what does not add, in assets, intervals, grids or rates, is
refused, naming the fix.
"""

import re

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import Venue
from probatum._result import SCHEMAS
from test_a_data_from_the_router_is_every_bar_it_asked_for import Router
from test_a_data_from_the_router_is_every_bar_it_asked_for import bars as rows
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, bars, pair, weights)
from test_a_result_hands_its_tables_to_any_arrow_reader import inverse, takeover

FREE = {"maker": 0.0, "taker": 0.0}
HOUR = 3_600_000_000_000
DAY  = 24 * HOUR


def routed(quote="USDT", start="2024-01-01", freq="h"):
    """A perpetual's 40 bars as the router gives them, quoted in ``quote``."""
    frame = bars(start=start, freq=freq)
    frame.attrs.update(exchange="binance", market_type="linear", symbol=f"BTC{quote}",
                       interval={"h": "1h", "4h": "4h"}[freq], quote=quote,
                       warnings=[])
    return probatum.Data(frame)


def without_venue(result):
    """``result``'s joined tables as rows, each with its venue left out."""
    return {name: [{key: value for key, value in row.items() if key != "venue"}
                   for row in getattr(result, name)]
            for name in SCHEMAS if name != "ledger"}


FUNDED = VENUE.replace(funding={"every": 8, "rate": 0.0001})


@pytest.mark.parametrize("make", [
    lambda: probatum.backtest(weights(), pair(), VENUE),
    lambda: probatum.backtest(weights(), pair(), FUNDED, risk_free=0.04),
    takeover, inverse])
def test_one_book_combined_is_itself(make):
    result = make()
    book   = probatum.combine({"a": result})

    for name in ("time", "equity", "returns"):
        assert getattr(book, name).tobytes() == getattr(result, name).tobytes()
    assert book.ledger == result.ledger
    assert book.stats  == result.stats
    assert (book.bust, book.liquidations) == (result.bust, result.liquidations)
    for name, rows in without_venue(book).items():
        assert rows == list(getattr(result, name))
        assert set(getattr(book, name)["venue"]) <= {"a"}
    assert book.provenance["results"] == {"a": result.provenance}
    assert book.provenance["results"]["a"] is not result.provenance


def test_two_copies_of_a_book_are_twice_it():
    result = takeover()
    book   = probatum.combine({"a": result, "b": result})

    assert book.equity.tobytes()  == (2.0 * result.equity).tobytes()
    assert book.returns.tobytes() == result.returns.tobytes()
    for key in ("total_return_pct", "cagr_pct", "sharpe", "sortino", "calmar",
                "max_drawdown_pct", "volatility_pct", "exposure_pct", "win_rate"):
        assert book.stats[key] == result.stats[key], key
    for key in ("num_trades", "num_fills", "funding_paid"):
        assert book.stats[key] == 2 * result.stats[key], key
    assert book.stats["avg_trade_pct"] == pytest.approx(result.stats["avg_trade_pct"] / 2)
    assert book.liquidations == 2 * result.liquidations == 2
    assert len(book.fills) == 2 * len(result.fills)
    assert book.ledger["fees"].tobytes() == (2.0 * result.ledger["fees"]).tobytes()
    assert book.bust == result.bust


def test_the_sum_is_taken_in_the_order_given():
    data = pair()
    flat = {name: probatum.backtest(np.full((40, 2), np.nan), data,
                                    VENUE.replace(balance=balance))
            for name, balance in (("a", 0.1), ("b", 0.2), ("c", 0.3))}

    assert probatum.combine(flat).equity[0] == (0.1 + 0.2) + 0.3 == 0.6000000000000001

    backwards = probatum.combine(dict(reversed(flat.items())))

    assert backwards.equity[0] == (0.3 + 0.2) + 0.1 == 0.6


def test_a_book_holds_each_account_s_opening_before_it_and_its_last_after_it():
    data   = pair()
    costly = FUNDED.replace(fees={"maker": 0.0002, "taker": 0.0006})
    early  = probatum.backtest(weights(), data, VENUE)
    late   = probatum.backtest(weights(n=30), data["2024-01-01 10:00":],
                               costly.replace(balance={"spot": 1_000.0, "margin": 4_000.0}))

    rows = weights(n=31)
    # a decision at the last close but one, so a fill lands on the account's last row
    rows[29] = [0.7, -0.3]
    ended = probatum.backtest(rows, data[:31], costly.replace(balance=2_000.0))
    book  = probatum.combine({"early": early, "late": late, "ended": ended})

    np.testing.assert_array_equal(book.time, data.time)

    def spread(result, name, before, after):
        """``result``'s column ``name`` on the book's 40 rows: ``before`` ahead of its
        first row, its own values, and ``after`` past its last."""
        values = result.ledger[name]
        if len(values) == 30:
            return np.concatenate([np.full(10, before), values])
        return np.concatenate([values, np.full(9, after)])

    np.testing.assert_array_equal(
        book.equity, early.equity + spread(late, "equity", 5_000.0, None)
        + spread(ended, "equity", None, ended.equity[-1]))

    ledger = book.ledger

    assert len(ledger) == len(book.time) and "venue" not in ledger.columns
    np.testing.assert_array_equal(ledger["equity"], book.equity)

    # cash is held as equity is, exposure and margin count inside their account's span
    # alone, and a cost falls on the row it was paid on
    for name in ("spot_cash", "margin_cash", "equity_in_quote"):
        held = (early.ledger[name] + spread(late, name, late.ledger[name][0], None)
                + spread(ended, name, None, ended.ledger[name][-1]))
        np.testing.assert_array_equal(ledger[name], held, err_msg=name)
    for name in ("gross_exposure", "net_exposure", "margin_used", "fees", "funding"):
        counted = (early.ledger[name] + spread(late, name, 0.0, None)
                   + spread(ended, name, None, 0.0))
        np.testing.assert_array_equal(ledger[name], counted, err_msg=name)

    leverage = np.divide(ledger["gross_exposure"], book.equity, out=np.zeros(40),
                         where=book.equity > 0.0)

    np.testing.assert_array_equal(ledger["leverage"], leverage)
    assert book.stats["funding_paid"] == (early.stats["funding_paid"]
                                          + late.stats["funding_paid"]
                                          + ended.stats["funding_paid"])


def test_the_tables_name_each_row_s_venue_and_count_bars_on_the_book_s_clock():
    data   = pair()
    first  = probatum.backtest(weights(), data, VENUE)
    second = probatum.backtest(weights(n=30, seed=5), data["2024-01-01 10:00":], VENUE)
    book   = probatum.combine({"first": first, "second": second})

    for name in ("holdings", "fills", "orders"):
        table = getattr(book, name)
        np.testing.assert_array_equal(book.time[table["bar"]], table["time"])
        assert (np.diff(table["bar"]) >= 0).all()
        for venue, part, shift in (("first", first, 0), ("second", second, 10)):
            mine = table[table["venue"] == venue]
            assert [{**{key: value for key, value in row.items() if key != "venue"},
                     "bar": row["bar"] - shift} for row in mine] == list(
                getattr(part, name))

    placed = book.orders.events("placed")
    keys   = list(zip(placed["venue"], placed["order"].tolist()))

    assert len(set(keys)) == len(keys) and len({order for _, order in keys}) < len(keys)

    trades = book.trades
    closed = trades[~trades["open"]]

    assert (np.diff(closed["close_bar"]) >= 0).all()
    assert list(trades["open"]) == sorted(trades["open"])
    np.testing.assert_array_equal(book.time[closed.close_bar], closed.close_time)
    np.testing.assert_array_equal(book.time[trades.open_bar], trades.open_time)

    for venue, part, shift in (("first", first, 0), ("second", second, 10)):
        mine = []
        for row in trades[trades["venue"] == venue]:
            del row["venue"]
            row["open_bar"] -= shift
            if row["close_bar"] is not None:
                row["close_bar"] -= shift
            mine.append(row)

        assert mine == list(part.trades)


def test_the_book_annualises_as_its_parts_where_they_agree_and_over_their_spans():
    data     = pair()
    first    = probatum.backtest(weights(), data, VENUE)
    second   = probatum.backtest(weights(seed=4), data, VENUE)
    together = probatum.combine({"a": first, "b": second})

    assert together.provenance["periods_per_year"] == first.provenance[
        "periods_per_year"] == 8_760.0

    gappy = probatum.Data(
        bars(start="2024-03-01").drop(
            pd.Timestamp("2024-03-01 05:00", tz="UTC")),
        symbol   = "BTCUSDT",
        market   = "linear",
        interval = "1h",
    )
    apart = {
        "a": probatum.backtest(np.full(40, 0.5), routed(), VENUE),
        "b": probatum.backtest(np.full(39, 0.5), gappy, VENUE),
    }

    assert len({part.provenance["periods_per_year"] for part in apart.values()}) == 2

    book = probatum.combine(apart)

    # each covers 39 hours, two months apart, and the months between count for neither
    assert book.provenance["periods_per_year"] == (len(book.time) - 1) * 365.0 / (
        2 * 39 * HOUR / DAY)

    def hourly(rows, n):
        """A run on the hourly bars ``rows`` of ``n``, the Data told its interval."""
        data = probatum.Data(
            bars(n=n).iloc[rows],
            symbol   = "BTCUSDT",
            market   = "linear",
            interval = "1h",
        )
        return probatum.backtest(np.full(len(data), 0.5), data, VENUE)

    # A at 8,760 from 00:00 to 03:00 and B at 4,380 at 00:00, 02:00, 04:00 and 06:00:
    # the book's five steps over the six hours they cover
    a, b = hourly(slice(None), 4), hourly(slice(None, None, 2), 7)

    assert (a.provenance["periods_per_year"], b.provenance["periods_per_year"]) == (
        8_760.0, 4_380.0)
    assert probatum.combine({"a": a, "b": b}).provenance["periods_per_year"] == 7_300.0

    # a span inside another counts once
    whole, inside = hourly(slice(None), 7), hourly(slice(2, None, 2), 5)

    assert probatum.combine({"a": whole, "b": inside}).provenance[
        "periods_per_year"] == 8_760.0


def test_a_book_is_bust_where_its_sum_reaches_zero_and_not_where_one_account_does():
    gone = takeover()
    book = probatum.combine({"gone": gone, "flat": probatum.backtest(
        np.full((40, 2), np.nan), pair(), VENUE)})

    assert gone.bust and not book.bust
    assert book.liquidations == gone.liquidations == 1


def test_what_combine_cannot_add_is_refused():
    data   = pair()
    result = probatum.backtest(weights(), data, VENUE)
    alpha  = probatum.backtest(weights(), data, Venue(name="alpha", fees=FREE))
    usdt   = probatum.backtest(np.full(40, 0.5), routed("USDT"), VENUE)
    usdc   = probatum.backtest(np.full(40, 0.5), routed("USDC"), VENUE)
    cases  = [
        ([], ValueError, "results: none; give the Results to combine"),
        (result, TypeError, "results: one Result; give a list of them"),
        (None, TypeError, "results: None; give a list of Results, or a mapping of a "
                          "name to each"),
        ([result], ValueError, "results: a Result at a venue with no name; name each, "
                               "as combine({\"a\": first, \"b\": second})"),
        ([alpha, "x"], TypeError, "results: a str; give Results"),
        ({1: result}, TypeError, "results: a name 1; name each Result with a string"),
        ({"": result}, TypeError, "results: a name ''; name each Result with a string"),
        ([alpha, alpha], ValueError, "results: two at alpha; name each, as "
                                     "combine({\"a\": first, \"b\": second})"),
        ([probatum.combine({"a": result})], ValueError,
         "results: a book already combined; combine the Results it was made of"),
        ({"coin": inverse(), "dollars": result}, ValueError,
         "results: coin trades an inverse contract and dollars does not, so their equity "
         "in the quote does not add; combine each kind on its own"),
        ({"two": inverse(), "coins": inverse()}, ValueError,
         "results: accounts kept in coins that are not one known coin"),
        ({"usdt": usdt, "usdc": usdc}, ValueError,
         "results: accounts kept in USDT and USDC; pass settlement= to count them at "
         "par in one, as settlement=\"USDT\""),
        ({"hourly": result, "four": probatum.backtest(
            np.full(40, 0.5), routed(freq="4h"), VENUE)}, ValueError,
         "results: bars of 1h and 4h; run them on one interval"),
        ({"a": result, "b": probatum.backtest(weights(), data, VENUE, risk_free=0.04)},
         ValueError, "results: risk-free rates of 0.0 and 0.04; run them at one rate"),
        ({"a": result, "b": probatum.backtest(np.full(40, 0.5), routed(
            start="2024-01-01 00:30"), VENUE)}, ValueError,
         "results: b opens a row at 2024-01-01 00:30, between the 1h steps from "
         "2024-01-01 00:00; run them on bars that open on one interval's steps"),
    ]

    for given, error, said in cases:
        with pytest.raises(error, match=re.escape(said)):
            probatum.combine(given)

    par = probatum.combine({"usdt": usdt, "usdc": usdc}, settlement="USDT")

    assert par.provenance["settlement"] == "USDT"
    np.testing.assert_array_equal(par.equity, usdt.equity + usdc.equity)
    assert probatum.combine({"usdt": usdt, "usdc": usdc}, settlement="USDC").provenance[
        "settlement"] == "USDC"

    for given in (1, ""):
        with pytest.raises(TypeError, match=f"settlement: {given!r}; name the asset the "
                                            f"book is kept"):
            probatum.combine({"coin": inverse()}, settlement=given)

    with pytest.raises(ValueError, match=re.escape(
            "settlement: 'USDT' for accounts kept in their coin; a book of a coin is kept "
            "in that coin, so leave settlement out")):
        probatum.combine({"coin": inverse()}, settlement="USDT")

    named = probatum.combine([alpha, probatum.backtest(
        weights(), data, Venue(name="beta", fees=FREE))])

    assert list(named.provenance["results"]) == ["alpha", "beta"]


def test_accounts_kept_in_an_asset_other_than_a_dollar_add_only_to_their_own():
    usdt  = probatum.backtest(np.full(40, 0.5), routed("USDT"), VENUE)
    usdc  = probatum.backtest(np.full(40, 0.5), routed("USDC"), VENUE)
    eur   = probatum.backtest(np.full(40, 0.5), routed("EUR"), VENUE)
    frame = bars(level=0.05)
    frame.attrs.update(exchange="binance", market_type="spot", symbol="ETHBTC",
                       interval="1h", quote="BTC", warnings=[])
    btc = probatum.backtest(np.full(40, 0.5), probatum.Data(frame),
                            Venue(balance=1.0, fees=FREE))

    for given, other in (({"btc": btc, "usdt": usdt}, "BTC and USDT"),
                         ({"usdt": usdt, "eur": eur}, "EUR and USDT")):
        # refused with or without a settlement, which no refusal offers
        for settlement in (None, "USDT"):
            with pytest.raises(ValueError, match=re.escape(
                    f"results: accounts kept in {other}, which do not add; combine "
                    f"accounts kept in one asset")):
                probatum.combine(given, settlement=settlement)

    with pytest.raises(ValueError, match=re.escape(
            "settlement: 'USDT' for accounts kept in BTC, which is no dollar asset to "
            "count at par; leave settlement out")):
        probatum.combine({"btc": btc, "again": btc}, settlement="USDT")

    with pytest.raises(ValueError, match=re.escape(
            "settlement: 'BTC' for accounts kept in USDT and USDC; name the dollar asset "
            "the book is kept in, as \"USDT\"")):
        probatum.combine({"usdt": usdt, "usdc": usdc}, settlement="BTC")

    twice = probatum.combine({"btc": btc, "again": btc}, settlement="BTC")

    assert twice.provenance["settlement"] == "BTC"
    np.testing.assert_array_equal(twice.equity, 2.0 * btc.equity)
    assert probatum.combine({"usdt": usdt, "again": usdt}).provenance[
        "settlement"] == "USDT"

    with pytest.raises(ValueError, match=re.escape(
            "results: coin trades an inverse contract and btc does not, so their equity "
            "in the quote does not add")):
        probatum.combine({"coin": inverse(), "btc": btc})

    # an inverse contract from the router names its coin, which a frame's does not
    data = probatum.Data.from_router(
        Router({"BTCUSD": rows(48)}, market="inverse"), "binance", "inverse", "1h",
        ["BTCUSD"], start="2024-01-01", end="2024-01-02 23:00")
    coin = probatum.backtest(np.full(48, -0.5), data, Venue(balance=1.0))

    assert probatum.combine({"a": coin, "b": coin}).provenance["settlement"] == "BTC"

    with pytest.raises(ValueError, match=re.escape(
            "results: accounts kept in coins that are not one known coin, None, BTC")):
        probatum.combine({"unnamed": inverse(), "btc": coin})


def test_the_binding_s_statistics_refuse_a_year_of_no_bars():
    with pytest.raises(ValueError, match="^periods_per_year: 0; give the bars a year as "
                                         "a finite number above 0$"):
        probatum._probatum.stats_of(1.0, np.ones(2), np.zeros(0), 0.0, 0.0, 0, 0, 0.0)


def test_combine_leaves_its_parts_as_they_were():
    result = takeover()
    before = (result.equity.tobytes(), list(result.fills), dict(result.stats),
              repr(result.provenance))

    probatum.combine({"a": result, "b": result})

    assert before == (result.equity.tobytes(), list(result.fills), dict(result.stats),
                      repr(result.provenance))
