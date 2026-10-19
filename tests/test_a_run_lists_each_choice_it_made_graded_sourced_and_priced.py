"""A run lists each modelling choice it made as a row of its assumptions: its value, the
input that decided it, a grade of venue, simplified or optimistic with the way a
simplified one leans, and what it cost where the ledger and the fill log measure it. A
row reads the value each instrument ran rather than the option's word, an engine rule
that cannot act reads its off form, a venue and a Data describe theirs alone, and the
summary line names every row graded optimistic.
"""

import copy
import math
import pickle

import numpy as np
import pandas as pd
import pytest

import probatum
import probatum._data
from test_a_data_from_the_router_is_every_bar_it_asked_for import (
    HOUR, START, Router, fetch, settlements)
from test_a_data_from_the_router_is_every_bar_it_asked_for import bars as served
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, bars, pair, weights)
from test_a_result_counts_its_costs_and_keeps_each_instrument_s_record import CHARGED
from test_a_result_hands_its_tables_to_any_arrow_reader import runs

VENUE_ROWS = ["balance", "fees", "leverage", "maintenance", "liquidation",
              "liquidation fee", "max open orders", "limit fills", "window", "liquidity",
              "spread", "slippage", "impact", "sizes", "order limits", "intrabar",
              "funding"]
ENGINE_ROWS = ["margin", "spot borrowing", "liquidation trigger", "funding price",
               "fills in the bar", "margin bound", "an instrument with no bar",
               "liquidation in slices", "auto-deleveraging", "a flip held back",
               "funding paid inside a bar", "funding counted inside a bar",
               "a leg opened on the favourable side"]
DATA_ROWS = ["bars", "gaps", "funding events", "history", "delisting", "universe",
             "prices"]
PRICED = ["fees", "liquidation fee", "funding", "spread", "slippage", "impact",
          "margin bound", "liquidation", "sizes"]
PERP  = probatum.venues.binance_perp
TAKEN = f"binance_perp, taken {PERP.taken}"
# the rows only a perpetual has, and the engine's that act only on one
PERPETUAL = ["leverage", "maintenance", "liquidation", "liquidation fee", "funding"]
MARGINED  = ["margin", "liquidation trigger", "funding price", "margin bound",
             "liquidation in slices", "auto-deleveraging", "a flip held back",
             "funding paid inside a bar", "funding counted inside a bar",
             "a leg opened on the favourable side"]


def read(table):
    """``table``'s rows by name, each its value, source, grade, direction and cost."""
    return {row["name"]: (row["value"], row["source"], row["grade"], row["direction"],
                          row["cost"]) for row in table}


def graded(table):
    """``table``'s rows by name, each its value, grade and direction."""
    return {name: (value, grade, way) for name, (value, _, grade, way, _)
            in read(table).items()}


def routed(venue=PERP, symbols=("BTCUSDT",), **asked):
    """A run on ``venue`` of a long half the equity in each of ``symbols``, on 48
    hourly router bars with funding every 8 hours."""
    held    = {symbol: served(48, close=100.0 * (k + 1)) for k, symbol in enumerate(symbols)}
    funding = {symbol: settlements() for symbol in symbols}
    data    = fetch(Router(held, funding), symbols, funding=True, **asked)

    return probatum.backtest(np.full((48, len(symbols)), 0.5 / len(symbols)), data, venue)


def framed(venue, symbols=("BTCUSDT",), n=40, freq="h", market="linear"):
    """A run on ``venue`` of a long half the equity in each of ``symbols`` on frames."""
    data = probatum.Data({symbol: bars(n=n, freq=freq) for symbol in symbols},
                         market=market)
    return probatum.backtest(np.full((n, len(symbols)), 0.5 / len(symbols)), data, venue)


@pytest.fixture(scope="module")
def results():
    return runs()


def test_a_run_lists_every_row_the_specification_names_in_its_order(results):
    for name, result in results.items():
        table = result.assumptions
        assert type(table) is probatum._result.Assumptions
        assert table.columns == ("name", "value", "source", "grade", "direction",
                                 "cost")
        assert list(table["name"]) == VENUE_ROWS + ENGINE_ROWS + DATA_ROWS, name
        for row in table:
            assert row["grade"] in ("venue", "simplified", "optimistic"), row
            # a direction belongs to a simplified row alone, and a cost to the rows the
            # ledger and the fill log measure
            assert (row["direction"] is not None) == (row["grade"] == "simplified"), row
            if row["name"] not in PRICED:
                assert row["cost"] is None, row

    assert list(VENUE.describe()["name"])  == VENUE_ROWS + ENGINE_ROWS
    assert list(pair().describe()["name"]) == DATA_ROWS

    optimistic = results["pair"].assumptions
    kept       = optimistic[optimistic.grade == "optimistic"]

    assert type(kept) is probatum._result.Assumptions and set(kept["grade"]) == {
        "optimistic"}


def test_a_preset_run_assumes_nothing_optimistic_until_moved_to_touch_and_no_spread():
    # the Specification's example: binance_perp has no optimistic assumption, and moved
    # to limits that fill on touch and no spread it has two, each set by the user
    result = routed()
    table  = result.assumptions

    assert not len(table[table.grade == "optimistic"])
    assert "optimistic" not in repr(result)

    moved = routed(PERP.replace(fills={"limit": "touch", "spread": "none"}))
    table = moved.assumptions
    kept  = table[table.grade == "optimistic"]

    assert [(row["name"], row["value"], row["source"]) for row in kept] == [
        ("limit fills", "on touch", "set by the user"),
        ("spread", "not modelled", "set by the user")]
    assert repr(moved).endswith("; 2 optimistic assumptions: limit fills on touch; "
                                "spread not modelled)")


def test_a_preset_run_reads_each_row_from_the_preset_it_ran():
    rows    = read(routed().assumptions)
    fetched = rows["bars"][1].removeprefix("the router, fetched ")

    assert rows["balance"][:3] == ("10,000 USDT", "plain", "venue")
    assert rows["fees"][:4]    == ("maker 0.02%, taker 0.05%", TAKEN, "venue", None)
    for name in ("leverage", "maintenance"):
        assert rows[name][:3] == ("each instrument's tier table", TAKEN, "venue")
    assert rows["liquidation"][:3] == ("by takeover", TAKEN, "venue")
    assert rows["liquidation fee"][:3] == ("unused, as maintenance is kept", TAKEN,
                                           "venue")
    assert rows["max open orders"][:4] == ("8 an instrument", "plain", "simplified",
                                           "pessimistic")
    assert rows["limit fills"][:4] == ("through, half a step past the limit", TAKEN,
                                       "simplified", "either")
    assert rows["window"][:3] == ("24 bars", "plain", "simplified")

    calibrated = "binance_perp, a starting calibration"
    assert rows["liquidity"][:2] == ("10% of the trailing volume a bar", calibrated)
    assert rows["spread"][:2] == ("estimated, which errs high on the most liquid "
                                  "perpetuals", TAKEN)
    assert rows["slippage"][:2] == ("1 bp", calibrated)
    assert rows["impact"][:2]   == ("sqrt, k = 0.1", calibrated)
    assert rows["sizes"][:3]    == ("on the lot and the price rule", TAKEN, "venue")
    assert rows["order limits"][:3] == ("a minimum notional and a maximum size",
                                        f"the router, fetched {fetched}", "venue")
    assert rows["intrabar"][:4]      == ("pessimistic", "plain", "simplified", "pessimistic")
    assert rows["funding"][:3]       == ("from the data", TAKEN, "venue")
    assert rows["bars"][0]           == "1h, 2024-01-01 00:00 to 2024-01-02 23:00, from binance"
    assert rows["funding events"][0] == "6"
    assert rows["universe"][:4] == ("the user's list of 1 instrument",
                                    f"the router, fetched {fetched}", "simplified",
                                    "optimistic")


# each row's grade and direction on binance_perp's router bars of a perpetual, as the
# Specification's tables give them
GRADED = {
    "balance":                             ("venue", None),
    "fees":                                ("venue", None),
    "leverage":                            ("venue", None),
    "maintenance":                         ("venue", None),
    "liquidation":                         ("venue", None),
    "liquidation fee":                     ("venue", None),
    "max open orders":                     ("simplified", "pessimistic"),
    "limit fills":                         ("simplified", "either"),
    "window":                              ("simplified", "either"),
    "liquidity":                           ("simplified", "either"),
    "spread":                              ("simplified", "either"),
    "slippage":                            ("simplified", "either"),
    "impact":                              ("simplified", "either"),
    "sizes":                               ("venue", None),
    "order limits":                        ("venue", None),
    "intrabar":                            ("simplified", "pessimistic"),
    "funding":                             ("venue", None),
    "margin":                              ("venue", None),
    "spot borrowing":                      ("venue", None),
    "liquidation trigger":                 ("simplified", "pessimistic"),
    "funding price":                       ("simplified", "either"),
    "fills in the bar":                    ("simplified", "optimistic"),
    "margin bound":                        ("venue", None),
    "an instrument with no bar":           ("simplified", "optimistic"),
    "liquidation in slices":               ("simplified", "pessimistic"),
    "auto-deleveraging":                   ("simplified", "optimistic"),
    "a flip held back":                    ("simplified", "optimistic"),
    "funding paid inside a bar":           ("venue", None),
    "funding counted inside a bar":        ("venue", None),
    "a leg opened on the favourable side": ("venue", None),
    "bars":                                ("venue", None),
    "gaps":                                ("venue", None),
    "funding events":                      ("venue", None),
    "history":                             ("venue", None),
    "delisting":                           ("venue", None),
    "universe":                            ("simplified", "optimistic"),
    "prices":                              ("venue", None),
}


def test_every_row_takes_the_grade_the_specification_gives_it():
    rows = graded(routed().assumptions)

    assert {name: row[1:] for name, row in rows.items()} == GRADED

    # the forms that run does not reach
    forms = {(VENUE.replace(max_open_orders=16), "max open orders"): (
                 "16 an instrument", "simplified", "either"),
             (VENUE, "sizes"): ("not rounded or bounded", "optimistic", None),
             (VENUE.replace(leverage=0.0), "leverage"): ("not capped", "optimistic",
                                                         None),
             (VENUE.replace(fees={"taker": 0.0005}), "liquidation fee"): (
                 "the taker rate", "simplified", "either")}

    for (venue, name), form in forms.items():
        assert graded(framed(venue).assumptions)[name] == form, name

    data   = probatum.Data(bars(n=8), symbol="BTCUSDT", market="linear")
    looped = probatum.Engine(data, VENUE.replace(intrabar="random"), seed=1).result()

    assert graded(looped.assumptions)["intrabar"] == ("random", "simplified", "either")


def test_each_row_reads_the_value_its_option_gave_and_not_its_word():
    zero = VENUE.replace(fills={"spread": {"fixed": 0.0}, "slippage": {"fixed": 0.0},
                                "impact": {"sqrt": 0.0}},
                         funding={"every": 8, "rate": 0.0},
                         tiers=[{"up_to": 1e12, "leverage": 0.0, "rate": 0.01}])
    rows = graded(framed(zero).assumptions)

    for name in ("spread", "slippage", "impact"):
        assert rows[name] == ("not modelled", "optimistic", None), name

    assert rows["funding"] == ("not charged", "optimistic", None)
    for name in ("funding price", "funding paid inside a bar",
                 "funding counted inside a bar"):
        assert rows[name] == ("no funding charged", "venue", None), name

    assert rows["leverage"]    == ("not capped", "optimistic", None)
    assert rows["maintenance"] == ("each instrument's tier table", "venue", None)
    assert rows["fees"]        == ("not charged", "optimistic", None)

    some = VENUE.replace(fees={"taker": 0.0005},
                         fills={"spread": {"fixed": 1.0}, "slippage": {"fixed": 2.0},
                                "impact": {"linear": 0.1}},
                         funding={"every_bars": 3, "rate": 0.0001})
    rows = graded(framed(some).assumptions)

    assert rows["spread"]   == ("fixed at 1 bp", "simplified", "either")
    assert rows["slippage"] == ("2 bps", "simplified", "either")
    assert rows["impact"]   == ("linear, k = 0.1", "simplified", "either")
    assert rows["funding"]  == ("every 3 bars at 0.01%", "simplified", "either")
    assert rows["fees"]     == ("maker 0%, taker 0.05%", "venue", None)
    assert rows["leverage"] == ("capped at 10x", "simplified", "either")

    hourly = graded(framed(some.replace(funding={"every": 8, "rate": -0.0001},
                                        leverage=12.5)).assumptions)
    assert hourly["funding"]  == ("every 8 hours at -0.01%", "simplified", "either")
    assert hourly["leverage"] == ("capped at 12.5x", "simplified", "either")


def test_a_tier_table_comes_before_the_flat_leverage_and_maintenance_before_the_fee(
        monkeypatch):
    # every fetch at one time, so the runs differ only where their venues do
    monkeypatch.setattr(probatum._data, "_now", lambda: START + 50 * HOUR)
    base = routed()

    for moved in (PERP.replace(leverage=0.0), PERP.replace(leverage=20.0),
                  PERP.replace(liquidation_fee=0.0), PERP.replace(liquidation_fee=0.005)):
        assert routed(moved).assumptions == base.assumptions, repr(moved)

    rows = graded(framed(VENUE.replace(liquidation_fee=0.005)).assumptions)
    assert rows["liquidation fee"] == ("0.5%", "simplified", "either")

    rows = graded(framed(VENUE.replace(liquidation_fee=0.0)).assumptions)
    assert rows["liquidation fee"] == ("not charged", "optimistic", None)

    rows = read(framed(probatum.Venue(fees={"maker": 0.0, "taker": 0.0})).assumptions)
    assert rows["liquidation fee"][:3] == ("not charged", "set by the user", "optimistic")


def test_an_instrument_the_venue_gives_no_tiers_is_capped_flat_and_keeps_no_maintenance():
    venue = PERP.replace(funding="none",
                         instruments={"XYZUSDT": {"lot": 0.001, "tick": 0.1}})
    rows = read(framed(venue, ("BTCUSDT", "XYZUSDT")).assumptions)

    assert rows["leverage"][:4] == (
        "each instrument's tier table; XYZUSDT capped at 10x",
        f"{TAKEN}; XYZUSDT leverage plain", "simplified", "either")
    assert rows["maintenance"][:3] == ("each instrument's tier table; XYZUSDT not kept",
                                       f"{TAKEN}; XYZUSDT plain", "optimistic")
    assert rows["liquidation fee"][:4] == (
        "unused, as maintenance is kept; XYZUSDT the taker rate", TAKEN, "simplified",
        "either")
    assert rows["fees"][:2] == ("maker 0.02%, taker 0.05%", TAKEN)

    alone = graded(framed(venue, ("XYZUSDT",)).assumptions)
    assert alone["maintenance"]     == ("not kept", "optimistic", None)
    assert alone["liquidation fee"] == ("the taker rate", "simplified", "either")


def test_through_with_no_price_rule_fills_on_touch():
    rows = graded(framed(VENUE.replace(fills={"limit": "through"})).assumptions)

    assert rows["limit fills"] == ("on touch", "optimistic", None)

    ticked = VENUE.replace(fills={"limit": "through"}, tick=0.01)
    assert graded(framed(ticked).assumptions)["limit fills"] == (
        "through, half a step past the limit", "simplified", "either")


def test_the_fees_row_names_the_instruments_that_pay_their_own():
    bybit = probatum.venues.bybit_perp
    own   = bybit.to_dict()["instruments"]
    venue = bybit.replace(funding="none")
    rows  = read(framed(venue, ("BTCUSDT", "1000TAGUSDT")).assumptions)
    taken = f"bybit_perp, taken {bybit.taken}"

    assert rows["fees"][:3] == ("maker 0.02%, taker 0.055%; 1000TAGUSDT maker 0.04%, "
                                "taker 0.11%", taken, "venue")
    assert read(framed(venue, ("1000TAGUSDT",)).assumptions)["fees"][0] == (
        "maker 0.04%, taker 0.11%")

    departing = sum("fees" in table for table in own.values())
    assert departing > 100
    assert read(bybit.describe())["fees"][:2] == (
        f"maker 0.02%, taker 0.055%; {departing:,} instruments maker 0.04%, taker 0.11%",
        taken)

    legs = probatum.Data(bars(), symbol="BTCUSDT", market="spot").join(
        probatum.Data(bars(), symbol="BTCUSDT", market="linear"))
    binance = probatum.venues.binance.replace(
        balance={"spot": 5_000.0, "margin": 5_000.0}, funding="none")
    rows = read(probatum.backtest(np.full((40, 2), 0.25), legs, binance).assumptions)

    assert rows["fees"][:2] == ("spot maker 0.1%, taker 0.1%; linear maker 0.02%, "
                                "taker 0.05%", f"binance, taken {binance.taken}")
    assert rows["balance"][0]        == "5,000 spot and 5,000 margin"
    assert rows["spot borrowing"][0] == "not modelled, a spot buy held to the cash"

    held = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "1000TAGUSDT",
            "4STOCKUSDT", "4USDT", "ACUUSDT")
    assert read(framed(venue, held).assumptions)["fees"][0] == (
        "maker 0.02%, taker 0.055%; 1000TAGUSDT, 4STOCKUSDT, 4USDT and ACUUSDT maker "
        "0.04%, taker 0.11%")


def test_a_combined_preset_reads_its_perpetual_rows_over_the_perpetuals_alone():
    legs = probatum.Data(bars(), symbol="BTCUSDT", market="spot").join(
        probatum.Data(bars(), symbol="BTCUSDT", market="linear"))
    binance = probatum.venues.binance.replace(
        balance={"spot": 5_000.0, "margin": 5_000.0}, funding="none")

    taken    = f"binance, taken {binance.taken}"
    expected = {"leverage": ("each instrument's tier table", taken, "venue", None),
                "maintenance": ("each instrument's tier table", taken, "venue", None),
                "liquidation fee": ("unused, as maintenance is kept", taken, "venue",
                                    None)}

    for venue in (binance, binance.replace(leverage=0.0),
                  binance.replace(liquidation_fee=0.0)):
        rows = read(probatum.backtest(np.full((40, 2), 0.25), legs, venue).assumptions)
        assert {name: rows[name][:4] for name in expected} == expected, repr(venue)

    described = read(probatum.venues.binance.describe())
    assert {name: described[name][:4] for name in expected} == expected
    assert described["fees"][:2] == ("spot maker 0.1%, taker 0.1%; linear maker 0.02%, "
                                     "taker 0.05%", taken)
    assert described["spot borrowing"][0] == "not modelled, a spot buy held to the cash"


def test_a_row_is_sourced_from_the_input_that_decided_each_instrument_s_value():
    rows = read(PERP.replace(fees={"taker": 0.0004}).describe())

    assert rows["fees"][:2] == ("maker 0.02%, taker 0.04%",
                                f"{TAKEN}; taker set by the user")

    base, wider = read(PERP.describe()), read(PERP.replace(fills={"window": 48})
                                              .describe())
    assert wider["window"][:2] == ("48 bars", "set by the user")
    assert {name: row for name, row in wider.items() if name != "window"} == {
        name: row for name, row in base.items() if name != "window"}

    hyperliquid = probatum.venues.hyperliquid_perp
    rows        = read(hyperliquid.replace(backstop=0.5).describe())

    assert rows["liquidation"][:2] == (
        "on the book, then by backstop at 1/2",
        f"hyperliquid_perp, taken {hyperliquid.taken}; backstop set by the user")
    assert read(hyperliquid.describe())["liquidation"][0] == (
        "on the book, then by backstop at 2/3")

    frictionless = read(probatum.venues.frictionless.describe())
    assert not any("calibration" in source for _, source, *_ in frictionless.values())
    assert frictionless["slippage"][1] == (
        f"frictionless, taken {probatum.venues.frictionless.taken}")

    plain = read(probatum.Venue().describe())
    assert {name: plain[name][1] for name in VENUE_ROWS} == dict.fromkeys(VENUE_ROWS,
                                                                          "plain")
    assert {plain[name][1] for name in ENGINE_ROWS} == {"the engine"}

    bybit = probatum.venues.bybit_perp
    split = bybit.replace(funding="none", fees={"taker": 0.0004},
                          instruments={"1000TAGUSDT": {"fees": {"maker": 0.0005}}})
    rows = read(framed(split, ("BTCUSDT", "1000TAGUSDT")).assumptions)

    assert rows["fees"][1] == (f"bybit_perp, taken {bybit.taken}; BTCUSDT taker and "
                               f"1000TAGUSDT maker set by the user")

    lacking = PERP.replace(funding="none", fills={"sizes": "off"})
    rows    = read(framed(lacking, ("XYZUSDT",)).assumptions)

    assert rows["limit fills"][:3] == ("on touch", f"{TAKEN}; tick plain", "optimistic")


def test_order_limits_read_where_each_limit_came_from(monkeypatch):
    monkeypatch.setattr(probatum._data, "_now", lambda: START + 50 * HOUR)
    router = "the router, fetched 2024-01-03 02:00"
    rows   = read(routed().assumptions)

    assert rows["order limits"][:3] == ("a minimum notional and a maximum size", router,
                                        "venue")

    capped = PERP.replace(instruments={"BTCUSDT": {"max_qty": 50.0}})
    assert read(routed(capped).assumptions)["order limits"][:2] == (
        "a minimum notional and a maximum size", f"{router}; max_qty set by the user")

    tabled = PERP.replace(funding="none", instruments={
        "BTCUSDT": {"min_notional": 5.0, "max_qty": 100.0}})
    assert read(framed(tabled).assumptions)["order limits"][:3] == (
        "a minimum notional and a maximum size", "set by the user", "venue")

    bare = read(framed(PERP.replace(funding="none")).assumptions)
    assert bare["order limits"][:4] == ("none given", "plain", "simplified",
                                        "optimistic")

    half = PERP.replace(funding="none", instruments={"BTCUSDT": {"max_qty": 100.0}})
    assert graded(framed(half).assumptions)["order limits"] == (
        "a maximum size, no minimum notional", "simplified", "optimistic")

    off = read(framed(VENUE).assumptions)
    assert off["order limits"][:3] == ("sizes are off", "plain", "venue")
    assert read(PERP.describe())["order limits"][:3] == (
        "from the Data's instruments", "the Data", "venue")


def hair():
    """A spot buy of 20 at 100 its 1,000 of cash holds to 9.99001 at a fee of 0.1%,
    which leaves the wallet a hair below zero, set to zero as dust."""
    frame = bars(n=4).assign(open=100.0, high=101.0, low=99.0, close=100.0)
    venue = probatum.Venue(
        kind    = "spot",
        balance = 1_000.0,
        fees    = {"maker": 0.001, "taker": 0.001},
    )

    return probatum.backtest(np.array([20.0, np.nan, np.nan, np.nan]),
                             probatum.Data(frame, symbol="BTCUSDT", market="spot"),
                             venue, sizing="quantity")


def test_the_rows_with_a_cost_hold_the_run_s_costs_and_sum_to_them(results):
    assert hair().costs["dust"] < 0.0

    for name, result in {**results, "hair": hair(),
                         "charged": probatum.backtest(weights(), pair(),
                                                      CHARGED)}.items():
        rows, costs = read(result.assumptions), result.costs
        cost     = {row: rows[row][4] for row in PRICED}
        scale    = 1e-9 * (1.0 + np.abs(result.equity).max())
        fills    = result.fills
        takeover = math.fsum(fills["fee"][(fills["reason"] == "liquidation")
                                          & ~fills["taker"]])
        perpetual = rows["margin"][0] != "no perpetual"

        assert cost["liquidation fee"] == (takeover if perpetual else None), name
        assert cost["fees"] + takeover == pytest.approx(costs["fees"], abs=scale), name
        for column in ("spread", "slippage", "impact"):
            assert cost[column] == costs[column], (name, column)
        assert cost["funding"] == (costs["funding"] if perpetual else None), name
        assert cost["sizes"] == costs["dust"], name
        if perpetual:
            assert cost["margin bound"] + cost["liquidation"] == pytest.approx(
                costs["liquidation"], abs=scale), name

        given = [value for value in cost.values() if value is not None]
        assert math.fsum(given) == pytest.approx(math.fsum(costs.values()), abs=scale)

    charged = read(probatum.backtest(weights(), pair(), CHARGED).assumptions)
    assert all(charged[row][4] != 0.0 for row in ("fees", "funding", "spread",
                                                   "slippage", "impact"))


def test_the_liquidation_fee_row_costs_the_10_005_the_close_paid():
    # the Specification's one-instrument example with no maintenance: a long of 0.5 BTC
    # from 60,000 on a wallet of 10,000 after its fee, and a bar that opens at 41,500
    # with a low of 39,000 closes it at 40,020.01, the fee taking the 10.005 left
    frame = pd.DataFrame({"open": [60_000.0, 60_000.0, 41_500.0],
                          "high": [60_000.0, 60_000.0, 41_500.0],
                          "low": [60_000.0, 60_000.0, 39_000.0],
                          "close": [60_000.0, 60_000.0, 39_500.0],
                          "volume": 1_000.0},
                         index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))

    venue  = probatum.Venue(balance=10_015.0, fees={"maker": 0.0002, "taker": 0.0005})
    result = probatum.backtest(np.array([0.5, np.nan, np.nan]),
                               probatum.Data(frame, symbol="BTCUSDT", market="linear"),
                               venue, sizing="quantity")

    assert result.liquidations == 1

    rows = read(result.assumptions)
    assert rows["liquidation fee"][4] == pytest.approx(10.005, abs=5e-4)
    assert rows["fees"][4]            == pytest.approx(15.0, abs=1e-9)
    assert rows["liquidation"][4]     == pytest.approx(0.0, abs=1e-6)
    assert rows["margin bound"][4]    == 0.0


def test_the_margin_bound_row_costs_minus_the_0_45_it_paid_in():
    # the Specification's margin bound example: a wallet of 50.5 holding a long of 10
    # from 100 sells on a bar that trades at 95 throughout, below where its equity less
    # the fee is zero, and the 0.45 the bar cannot hold is paid in
    frame = pd.DataFrame({"open": [100.0, 100.0, 95.0], "high": [100.0, 100.0, 95.0],
                          "low": [100.0, 100.0, 95.0], "close": [100.0, 100.0, 95.0],
                          "volume": 1_000.0},
                         index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    venue = probatum.Venue(
        balance  = 51.5,
        fees     = {"maker": 0.0, "taker": 0.001},
        leverage = 50.0,
    )

    result = probatum.backtest(np.array([10.0, 0.0, np.nan]),
                               probatum.Data(frame, symbol="XYZUSDT", market="linear"),
                               venue, sizing="quantity")

    assert result.liquidations == 0

    rows = read(result.assumptions)
    assert rows["margin bound"][4]     == pytest.approx(-0.45, abs=0.005)
    assert rows["liquidation"][4]      == pytest.approx(0.0, abs=1e-9)
    assert result.costs["liquidation"] == pytest.approx(-0.45, abs=0.005)


def test_the_margin_bound_row_costs_what_its_lift_and_its_pay_in_gave():
    # the same sale on a bar that opens at 95 and reaches 95.02: the bound lifts it
    # 0.02 to the high, 0.2 over its 10, and pays in the 0.2502 the bar cannot hold
    frame = pd.DataFrame({"open": [100.0, 100.0, 95.0], "high": [100.0, 100.0, 95.02],
                          "low": [100.0, 100.0, 95.0], "close": [100.0, 100.0, 95.0],
                          "volume": 1_000.0},
                         index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    venue = probatum.Venue(
        balance  = 51.5,
        fees     = {"maker": 0.0, "taker": 0.001},
        leverage = 50.0,
    )

    result = probatum.backtest(np.array([10.0, 0.0, np.nan]),
                               probatum.Data(frame, symbol="XYZUSDT", market="linear"),
                               venue, sizing="quantity")

    sale = result.fills[-1]
    assert sale["lifted"] == pytest.approx(0.02) and sale["paid_in"] > 0.0

    rows = read(result.assumptions)
    assert rows["margin bound"][4] == pytest.approx(-0.4502, abs=1e-4)
    assert rows["margin bound"][4] == pytest.approx(
        -(sale["size"] * sale["lifted"] + sale["paid_in"]), rel=1e-12)
    assert rows["liquidation"][4] == pytest.approx(0.0, abs=1e-9)


def test_on_an_inverse_contract_the_margin_bound_costs_the_coin_its_lift_gave():
    # a long of 600 contracts of 100 dollars from 60,000 on a wallet of 0.0527607 BTC
    # after its fee sells on a bar that opens at 57,000, below where its equity less
    # the fee is zero, about 57,050, which the bar's high of 57,100 holds, so the bound
    # lifts the sale inside the bar and pays nothing in, and the coin it gave is all
    # the liquidation charge holds
    frame = pd.DataFrame({"open": [60_000.0, 60_000.0, 57_000.0],
                          "high": [60_000.0, 60_000.0, 57_100.0],
                          "low": [60_000.0, 60_000.0, 57_000.0],
                          "close": [60_000.0, 60_000.0, 57_000.0],
                          "volume": 1e6},
                         index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    frame.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                       interval="1h", quote="USD", contract_size=100.0, warnings=[],
                       volume_unit="contract")
    venue = probatum.Venue(
        balance  = 0.0537607,
        fees     = {"maker": 0.0, "taker": 0.001},
        leverage = 0.0,
    )

    result = probatum.backtest(np.array([600.0, 0.0, np.nan]), probatum.Data(frame),
                               venue, sizing="quantity")

    sale = result.fills[-1]
    assert result.liquidations == 0 and sale["paid_in"] == 0.0
    assert 57_000.0 < sale["price"] < 57_100.0 and sale["lifted"] > 0.0

    given = 600.0 * 100.0 * (1.0 / (sale["price"] - sale["lifted"]) - 1.0 / sale["price"])
    rows  = read(result.assumptions)
    assert rows["margin bound"][4] == pytest.approx(-given, rel=1e-12)
    assert rows["liquidation"][4]  == pytest.approx(0.0, abs=1e-11)


def test_an_engine_rule_that_cannot_act_reads_its_off_form():
    spot = probatum.venues.binance_spot
    rows = read(framed(spot, market="spot").assumptions)

    for name in PERPETUAL:
        assert rows[name][0] == "no perpetual" and rows[name][2:] == (
            "venue", None, None), name

    for name in MARGINED:
        assert rows[name] == ("no perpetual", "the engine", "venue", None, None), name

    assert rows["spot borrowing"][:3] == ("not modelled, a spot buy held to the cash",
                                          "the engine", "venue")
    # what an instrument with no bar hides is a liquidation, which a spot run has none of
    assert rows["an instrument with no bar"] == ("held at its last mark", "the engine",
                                                 "venue", None, None)

    rows = graded(framed(VENUE).assumptions)
    assert rows["spot borrowing"] == ("no spot instrument", "venue", None)
    assert rows["an instrument with no bar"] == ("held at its last mark", "simplified",
                                                 "optimistic")
    assert rows["a flip held back"] == ("held whole, the side it opens with it",
                                        "simplified", "optimistic")
    assert rows["a leg opened on the favourable side"] == (
        "no exception under the pessimistic setting", "venue", None)

    hopeful = graded(framed(VENUE.replace(intrabar="optimistic")).assumptions)
    assert hopeful["intrabar"] == ("optimistic", "optimistic", None)
    assert hopeful["a flip held back"] == ("none, under the optimistic setting", "venue",
                                           None)
    assert hopeful["a leg opened on the favourable side"] == (
        "runs to its adverse extreme, under the optimistic setting", "simplified",
        "pessimistic")
    assert read(spot.describe())["leverage"][0] == "no perpetual"


def daily(venue, n=5):
    return framed(venue, n=n, freq="D")


EIGHT = VENUE.replace(funding={"every": 8, "rate": 0.0001})


def test_funding_inside_a_bar_is_counted_on_the_run_s_own_rows():
    # on daily bars 08:00 and 16:00 fall inside each day and the next midnight ends it
    rows = graded(daily(EIGHT).assumptions)

    assert rows["funding paid inside a bar"] == (
        "paid on what the bar's fills left, 10 events inside a bar", "simplified",
        "either")
    assert rows["funding counted inside a bar"] == (
        "counted on the size the path holds, 10 events inside a bar", "simplified",
        "optimistic")

    hopeful = graded(daily(EIGHT.replace(intrabar="optimistic")).assumptions)
    assert hopeful["funding counted inside a bar"][2] == "pessimistic"

    # hourly bars end on every stamp, and every_bars charges at a close
    for quiet in (framed(EIGHT), daily(EIGHT.replace(funding={"every_bars": 1,
                                                               "rate": 0.0001}))):
        rows = graded(quiet.assumptions)
        for name in ("funding paid inside a bar", "funding counted inside a bar"):
            assert rows[name] == ("no event fell inside a bar", "venue", None), name
        assert rows["funding price"] == ("the close, not the mark", "simplified",
                                         "either")

    # four-hour bars from 02:00 hold the 8-hourly stamps every other bar
    times  = pd.date_range("2024-01-01 02:00", periods=12, freq="4h", tz="UTC")
    four   = bars(n=12).set_axis(times)
    result = probatum.backtest(np.full(12, 0.5),
                               probatum.Data(four, symbol="BTCUSDT", market="linear"),
                               EIGHT)

    assert graded(result.assumptions)["funding paid inside a bar"][0] == (
        "paid on what the bar's fills left, 6 events inside a bar")


def test_an_engine_mid_run_counts_the_rows_it_has_stepped():
    data   = probatum.Data(bars(n=5, freq="D"), symbol="BTCUSDT", market="linear")
    engine = probatum.Engine(data, EIGHT.replace(intrabar="random"), seed=7)

    engine.step()
    engine.step()

    rows = graded(engine.result().assumptions)
    assert rows["funding paid inside a bar"][0] == (
        "paid on what the bar's fills left, 6 events inside a bar")
    assert rows["funding counted inside a bar"][2] == "either"
    assert rows["a flip held back"][0] == "held whole, the side it opens with it"
    assert rows["a leg opened on the favourable side"][0] == (
        "runs to its adverse extreme, under the optimistic setting")


def test_a_stamp_within_a_second_of_a_row_end_is_on_it_whatever_the_gap():
    # 07:59:59.997 is 08:00, the end of the 07:00 bar, and 08:00:00.003 is the end of
    # 07:00 too where the 08:00 bar is missing
    early = START + np.arange(1, 7, dtype=np.int64) * 8 * HOUR - 3_000_000
    data  = fetch(Router({"BTCUSDT": served(48)}, {"BTCUSDT": (early, np.full(6, 1e-4))}),
                  funding=True)
    rows = graded(probatum.backtest(np.full(48, 0.5), data, PERP).assumptions)

    assert rows["funding paid inside a bar"][0] == "no event fell inside a bar"

    times, values = served(48)
    holed = (np.delete(times, 8), np.delete(values, 8, axis=0))
    late  = START + np.arange(1, 7, dtype=np.int64) * 8 * HOUR + 3_000_000

    with pytest.warns(UserWarning, match="misses 1 bar"):
        data = fetch(Router({"BTCUSDT": holed}, {"BTCUSDT": (late, np.full(6, 1e-4))}),
                     funding=True)

    rows = graded(probatum.backtest(np.full(47, 0.5), data, PERP).assumptions)
    assert rows["funding paid inside a bar"][0] == "no event fell inside a bar"
    assert graded(PERP.describe())["funding paid inside a bar"] == (
        "paid on what the bar's fills left", "simplified", "either")

    every = VENUE.replace(funding={"every_bars": 8, "rate": 0.0001})
    assert graded(every.describe())["funding paid inside a bar"] == (
        "no event falls inside a bar", "venue", None)


def test_router_bars_are_the_venue_s_own_where_every_instrument_names_its_venue(
        monkeypatch):
    monkeypatch.setattr(probatum._data, "_now", lambda: START + 50 * HOUR)
    router = Router({"BTCUSDT": served(48), "ETHUSDT": served(48, close=10.0)},
                    {"BTCUSDT": settlements(), "ETHUSDT": settlements()})
    data = fetch(router, ("BTCUSDT",), funding=True)
    rows = graded(probatum.backtest(np.full(48, 0.5), data, PERP).assumptions)

    for name in ("bars", "prices", "delisting"):
        assert rows[name][1] == "venue", name

    assert rows["delisting"][0] == "no delisting time; the router serves none"
    assert rows["prices"][0]    == "as the venue printed them"

    # another exchange's bars are not this venue's, while a venue built by hand, even
    # one named for a preset, keeps the grade the Data gives
    bybit = probatum.venues.bybit_perp
    assert graded(probatum.backtest(np.full(48, 0.5), data, bybit).assumptions)[
        "bars"][1:] == ("simplified", "either")

    named = probatum.Venue(name="bybit_perp", funding="data")
    assert graded(probatum.backtest(np.full(48, 0.5), data, named).assumptions)[
        "bars"][1:] == ("venue", None)

    joined = data.join(fetch(router, ("ETHUSDT",), funding=True))
    rows   = read(probatum.backtest(np.full((48, 2), 0.25), joined, PERP).assumptions)

    assert rows["bars"][1:3]       == ("joined Datas", "venue")
    assert rows["order limits"][1] == "the router"

    mixed = data.join(probatum.Data(bars(n=48), symbol="XYZUSDT", market="linear"))
    rows  = read(mixed.describe())

    assert rows["bars"][1:4] == ("joined Datas", "simplified", "either")
    assert rows["prices"][2] == "simplified"
    assert rows["delisting"][0] == "no delisting time; every instrument's bars run to " \
                                   "the last"
    assert read(data.describe())["bars"][1:3] == (
        "the router, fetched 2024-01-03 02:00", "venue")
    assert read(data.resample("4h").describe())["bars"][1] == (
        "the router, fetched 2024-01-03 02:00, resampled from 1h")


def test_the_data_rows_read_its_gaps_its_funding_events_and_where_its_bars_end():
    frame  = bars(n=40)
    gapped = probatum.Data({
        "BTCUSDT": frame.drop(frame.index[[5, 6, 20]]),
        "ETHUSDT": frame.iloc[:30],
    }, market="linear")
    rows = read(gapped.describe())
    gaps = gapped.gaps()

    assert len(gaps)       == 2
    assert rows["gaps"][0] == "2, the longest 2 hours"
    assert rows["funding events"][0] == "none"
    assert rows["delisting"][:4] == ("no delisting time; ETHUSDT's bars end "
                                     "2024-01-02 05:00", "a frame", "simplified",
                                     "either")
    assert rows["universe"][0] == "the user's list of 2 instruments"
    assert rows["history"][0]  == "from each instrument's first bar"
    assert rows["bars"][0]     == "1h, 2024-01-01 00:00 to 2024-01-02 15:00"

    late = START + 10 * HOUR + HOUR * np.arange(38, dtype=np.int64)
    _, values = served(38)

    with pytest.warns(UserWarning, match="begin"):
        data = fetch(Router({"BTCUSDT": (late, values)}))

    rows = read(data.describe())
    assert rows["history"][0] == ("from each instrument's first bar; the router's "
                                  "history of BTCUSDT begins at 2024-01-01 10:00")
    assert rows["funding events"][0] == "none"

    funded = fetch(Router({"BTCUSDT": served(48)}, {"BTCUSDT": settlements()}),
                   funding=True)
    assert read(funded.describe())["funding events"][0] == f"{len(funded.funding):,}"

    # a router universe's late start is its own instrument's alone, and bars the
    # router stops serving early leave a position held at its last mark as any do
    times, values = served(48)
    held = {"BTCUSDT": (times, values), "ETHUSDT": (times[10:30], values[10:30])}

    with pytest.warns(UserWarning):
        universe = fetch(Router(held), ("BTCUSDT", "ETHUSDT"))

    rows = read(universe.describe())
    assert rows["history"][0] == ("from each instrument's first bar; the router's "
                                  "history of ETHUSDT begins at 2024-01-01 10:00")
    assert rows["delisting"][0] == ("no delisting time; ETHUSDT's bars end "
                                    "2024-01-02 05:00")
    assert rows["delisting"][2:4] == ("simplified", "either")

    alone = read(universe.select(["BTCUSDT"]).describe())
    assert alone["history"][0] == "from each instrument's first bar"
    assert alone["delisting"][:3:2] == ("no delisting time; the router serves none",
                                        "venue")


def test_the_delisting_row_names_each_delisting_time_the_venue_gives():
    # ETH's bars end at 05:00 on the second day, where the venue delists it at 06:00,
    # and SOL's end at 19:00 on the first with no delisting time
    frame = bars(n=40)
    data  = probatum.Data({
        "BTCUSDT": frame,
        "ETHUSDT": frame.iloc[:30],
        "SOLUSDT": frame.iloc[:20],
    }, market="linear")
    venue = probatum.Venue(
        kind        = "linear",
        instruments = {"ETHUSDT": {"delist": "2024-01-02 06:00"}},
    )
    rows = read(probatum.backtest(np.full((40, 3), 0.2), data, venue).assumptions)

    assert rows["delisting"][:4] == (
        "ETHUSDT delisted 2024-01-02 06:00; no delisting time otherwise; SOLUSDT's "
        "bars end 2024-01-01 19:00", "set by the user; a frame", "simplified", "either")

    pair = data.select(["BTCUSDT", "ETHUSDT"])
    rows = read(probatum.backtest(np.full((40, 2), 0.25), pair, venue).assumptions)

    assert rows["delisting"][:4] == ("ETHUSDT delisted 2024-01-02 06:00",
                                     "set by the user", "venue", None)

    # alone, no row of ETH's own reaches the time
    eth   = data.select(["ETHUSDT"])
    alone = read(probatum.backtest(np.full(len(eth), 0.5), eth, venue).assumptions)

    assert alone["delisting"][0] == ("ETHUSDT delisted 2024-01-02 06:00, after the "
                                     "last row")


def test_a_venue_describes_its_rows_from_the_venue_alone():
    rows = read(PERP.describe())

    assert rows["balance"][0] == "10,000"
    assert all(cost is None for *_, cost in rows.values())

    listed = probatum.Venue(
        kind        = "linear",
        instruments = {
            "BTCUSDT": {"tiers": [{"up_to": 1e12, "leverage": 20.0, "rate": 0.005}]}},
    )
    described = read(listed.describe())

    assert described["leverage"][0] == "each instrument's tier table"

    ran = read(framed(listed, ("ETHUSDT",)).assumptions)
    assert ran["leverage"][0] == "capped at 10x"
    assert list(described)    == VENUE_ROWS + ENGINE_ROWS
    for name in ("liquidation", "max open orders", "window", "liquidity", "spread",
                 "slippage", "impact", "sizes", "intrabar") + tuple(ENGINE_ROWS[:-3]):
        assert described[name][:4] == ran[name][:4], name

    kinds = read(probatum.Venue(kinds={
        "spot":   {"fees": {"maker": 0.001, "taker": 0.001}},
        "linear": {},
    }).describe())
    assert kinds["fees"][0] == ("spot maker 0.1%, taker 0.1%; linear maker 0.02%, "
                                "taker 0.06%")
    assert kinds["fees"][1] == "set by the user; linear plain"


def test_a_book_joins_its_accounts_assumptions_in_the_order_they_were_given(tmp_path):
    first  = framed(VENUE)
    second = framed(PERP.replace(funding="none"))
    book   = probatum.combine({"B": second, "A": first})
    table  = book.assumptions

    assert type(table) is probatum._result.Assumptions
    assert table.columns[0] == "venue"

    count = len(first.assumptions)
    assert list(table["venue"]) == ["B"] * count + ["A"] * count
    for k, part in enumerate((second, first)):
        assert [{key: row[key] for key in part.assumptions.columns}
                for row in table[k * count:(k + 1) * count]] == list(part.assumptions)

    assert "spread not modelled at A" in repr(book)

    book.save(tmp_path / "book")
    assert probatum.Result.load(tmp_path / "book").assumptions == table


def test_the_assumptions_go_wherever_the_result_goes(tmp_path):
    result    = probatum.backtest(weights(), pair(), CHARGED)
    unpickled = pickle.loads(pickle.dumps(result))

    assert unpickled._record is None
    assert unpickled.assumptions == result.assumptions

    result.save(tmp_path / "run")
    assert (tmp_path / "run" / "assumptions.arrow").is_file()
    assert probatum.Result.load(tmp_path / "run").assumptions == result.assumptions
    assert copy.copy(result).assumptions is result.assumptions
    assert copy.deepcopy(result).assumptions == result.assumptions

    plain = result.to_dict()["assumptions"]
    assert plain["name"] == list(result.assumptions["name"])

    fees = VENUE_ROWS.index("fees")
    assert plain["cost"][fees] == result.assumptions[fees]["cost"] == result.costs[
        "fees"]

    shown = repr(result.assumptions).splitlines()
    assert shown[0].split() == ["name", "value", "source", "grade", "direction", "cost"]
    assert len(shown)       == 1 + len(result.assumptions)
