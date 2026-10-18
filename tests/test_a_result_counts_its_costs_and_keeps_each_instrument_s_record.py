"""A Result's costs are the totals of its ledger's cost columns, each a compensated sum,
and with the price PnL measured from each fill's reference they account for every
change in equity. Each instrument's record is its own position, weight, pnl, fills and
round trips, read from the holdings and carried in a book from each of its account's
rows to the next.
"""

import math

import numpy as np
import pandas as pd
import pytest

import probatum
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, bars, pair, weights)
from test_a_result_hands_its_tables_to_any_arrow_reader import runs

NAMES = ["fees", "funding", "spread", "slippage", "impact", "liquidation", "dust"]
# every cost a fill or a row can charge, the impact included
CHARGED = VENUE.replace(fees={"maker": 0.0002, "taker": 0.0006},
                        fills={"spread": {"fixed": 5.0}, "slippage": {"fixed": 2.0},
                               "impact": {"sqrt": 0.5}},
                        funding={"every": 8, "rate": 0.0001})


@pytest.fixture(scope="module")
def results():
    return runs()


def test_the_costs_are_the_compensated_totals_of_the_ledger_s_columns(results):
    for name, result in results.items():
        costs = result.costs
        assert list(costs) == NAMES, name
        for cost in NAMES:
            assert costs[cost] == math.fsum(result.ledger[cost]), (name, cost)

        costs["fees"] = None
        assert result.costs["fees"] is not None


def test_a_spot_buy_its_cash_holds_counts_the_hair_it_leaves_as_dust():
    # 1,000 pays for 9.99001 of a buy of 20 at 100 with a fee of 0.1%, and the
    # wallet a hair below zero is set to zero, the hair a negative dust adjustment
    frame = bars(n=4).assign(open=100.0, high=101.0, low=99.0, close=100.0)
    data  = probatum.Data(frame, symbol="BTCUSDT", market="spot")
    venue = probatum.Venue(
        kind    = "spot",
        balance = 1_000.0,
        fees    = {"maker": 0.001, "taker": 0.001},
    )

    result = probatum.backtest(np.array([20.0, np.nan, np.nan, np.nan]), data, venue,
                               sizing="quantity")

    assert result.ledger["dust"].any()
    assert result.costs["dust"] == math.fsum(result.ledger["dust"]) < 0.0


def test_the_costs_and_the_price_pnl_account_for_every_change_in_equity():
    result = probatum.backtest(weights(), pair(), CHARGED)
    costs  = result.costs

    assert all(costs[name] != 0.0 for name in ("fees", "funding", "spread", "slippage",
                                               "impact")), costs

    rows, width = len(result.time), 2
    held     = result.holdings
    position = held["position"].reshape(rows, width)
    mark     = held["mark"].reshape(rows, width)
    names    = list(held["instrument"][:width])
    moved    = np.zeros(rows)

    moved[1:] = (position[:-1] * np.diff(mark, axis=0)).sum(axis=1)
    for fill in result.fills:
        signed = fill["size"] if fill["side"] == "buy" else -fill["size"]
        k      = names.index(fill["instrument"])
        moved[fill["bar"]] += signed * (mark[fill["bar"], k] - fill["reference"])

    paid   = sum(np.asarray(result.ledger[name]) for name in NAMES)
    change = np.diff(result.equity)
    scale  = 1e-9 * (1.0 + np.abs(result.equity).max())

    np.testing.assert_allclose(change, moved[1:] - paid[1:], rtol=0.0, atol=scale)


def test_the_fees_and_the_funding_are_the_fill_log_s_and_the_round_trips():
    result = probatum.backtest(weights(), pair(), CHARGED)
    costs, scale = result.costs, 1e-9 * (1.0 + abs(result.equity).max())

    assert costs["fees"]    == pytest.approx(math.fsum(result.fills["fee"]), abs=scale)
    assert costs["funding"] == pytest.approx(result.stats["funding_paid"], abs=scale)

    trades = result.trades
    assert costs["fees"]    == pytest.approx(math.fsum(trades["fees"]), abs=scale)
    assert costs["funding"] == pytest.approx(math.fsum(trades["funding"]), abs=scale)


def test_each_instrument_s_record_is_its_holdings_fills_and_round_trips(results):
    result = results["pair"]
    rows   = len(result.time)

    for k, name in enumerate(["BTCUSDT", "ETHUSDT"]):
        own = result[name]
        assert own.name == name
        for field in ("position", "weight", "pnl"):
            column = result.holdings[field].reshape(rows, 2)[:, k]
            assert np.array_equal(getattr(own, field), column), (name, field)
            assert not getattr(own, field).flags.writeable

        assert len(own.fills) and set(own.fills["instrument"]) == {name}
        assert set(own.trades["instrument"]) == {name}

    assert len(result["BTCUSDT"].fills) + len(result["ETHUSDT"].fills) == len(
        result.fills)
    np.testing.assert_allclose(result["BTCUSDT"].pnl + result["ETHUSDT"].pnl,
                               result.equity - result.equity[0], rtol=0.0, atol=1e-6)


def test_one_instrument_s_run_answers_for_it_and_several_s_names_the_one_to_ask(results):
    alone = results["inverse"]
    for field in ("position", "weight", "pnl"):
        assert np.array_equal(getattr(alone, field), getattr(alone["BTCUSD"], field))

    several = results["pair"]
    with pytest.raises(AttributeError, match=r"result\.pnl: the run holds 2 "
                                             r"instruments; use result\[\"BTCUSDT\"\]\.pnl"):
        several.pnl

    with pytest.raises(KeyError, match="no such instrument; use one of 'BTCUSDT', "
                                       "'ETHUSDT'"):
        several["SOLUSDT"]

    with pytest.raises(TypeError, match="an instrument is named by a string"):
        several[0]

    with pytest.raises(TypeError):
        iter(several)


def test_a_book_carries_an_account_s_instrument_through_a_row_it_has_none_of():
    # C prints at 00:00, 01:00 and 03:00 and buys 10 at 01:00, and D prints every
    # hour, so at the book's 02:00 C's long, its weight and what it has made are
    # carried from its 01:00 row
    hours = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta([0, 1, 3], unit="h")
    gappy = bars(n=4).iloc[[0, 1, 3]].set_axis(hours)
    c     = probatum.backtest(np.array([10.0, np.nan, np.nan]),
                              probatum.Data(gappy, symbol="BTCUSDT", market="linear",
                                            interval="1h"),
                              VENUE.replace(leverage=0.0), sizing="quantity")

    d = probatum.backtest(np.full(4, np.nan),
                          probatum.Data(bars(n=4), symbol="ETHUSDT", market="linear"),
                          VENUE)

    book = probatum.combine({"C": c, "D": d})
    held = book["BTCUSDT"]

    assert held.position.tolist() == [0.0, 10.0, 10.0, 10.0]
    assert held.pnl[2] == held.pnl[1] == c.pnl[1]
    assert held.weight[2] == held.weight[1] == c.weight[1]
    assert held.pnl[3] == c.pnl[2]


def test_a_book_names_every_venue_that_holds_a_name_and_asks_for_two_strings():
    d = probatum.backtest(np.full(4, np.nan),
                          probatum.Data(bars(n=4), symbol="ETHUSDT", market="linear"),
                          VENUE)

    book = probatum.combine({"C": d, "A": d, "B": d})

    with pytest.raises(KeyError, match="ETHUSDT is held at A, B and C; give its venue, "
                                       "as result\\[\"A\", \"ETHUSDT\"\\]"):
        book["ETHUSDT"]

    for key in ((0, "ETHUSDT"), ("A", 0), (None, "ETHUSDT")):
        with pytest.raises(TypeError, match="an instrument is named by a string; give its "
                                            "venue and its name"):
            book[key]

    assert book["B", "ETHUSDT"].name == "ETHUSDT"
