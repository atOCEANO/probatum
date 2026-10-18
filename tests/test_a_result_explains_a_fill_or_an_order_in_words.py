"""A Result tells one fill's price or one order's life in words: a fill as its
reference, each part that moved it to its price and its fee, every number reading back
to its field at the precision it prints and the price lines adding up to the price, and
an order as the events its log holds. A handle is told only by the Result of the run
that handed it out, and a book's rows name their venue.
"""

import copy
import math
import pickle
import re

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import ProbatumTypeError, ProbatumValueError
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, pair, weights)
from test_a_result_counts_its_costs_and_keeps_each_instrument_s_record import CHARGED
from test_a_result_hands_its_tables_to_any_arrow_reader import runs
from test_a_run_lists_each_choice_it_made_graded_sourced_and_priced import hair

PRICES = ("reference", "spread", "slippage", "impact", "held in the bar")


def frame(rows, volume=1e9):
    """Hourly bars from 1 January 2024, each row its open, high, low and close."""
    index = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame([list(row) for row in rows],
                        columns=["open", "high", "low", "close"],
                        index=index).assign(volume=volume)


def data(rows, symbol="BTCUSDT", market="linear", volume=1e9):
    return probatum.Data(frame(rows, volume), symbol=symbol, market=market)


class Script(probatum.Strategy):
    """At the close of each row ``plan`` names, its steps, each a name and a call of
    the account and the bar, keeping what each returns under its name."""


    def __init__(self, plan):
        self.plan = plan


    def init(self, data):
        self.kept = {}


    def next(self, bar, account):
        for name, step in self.plan.get(bar.i, ()):
            self.kept[name] = step(account, bar)


def scripted(plan, rows, venue=VENUE, **given):
    return probatum.backtest(Script(plan), data(rows, **given), venue)


def number(text):
    """A printed number and the decimals it was printed to."""
    _, _, part = text.partition(".")
    return float(text.replace(",", "")), len(part)


def read(result, k):
    """Fill ``k``'s text, checked: the head names the first fill equal to it, every
    printed number reads back to its field within half a unit of its last decimal,
    and the price lines add up to the printed price within half a unit for each."""
    fill = result.fills[k]
    text = result.explain(fill)
    head, *body = text.splitlines()

    first = next(at for at in range(len(result.fills)) if result.fills[at] == fill)
    assert head.startswith(f"fill {first}, "), (head, k)

    found = re.fullmatch(r"fill \d+, (?:(\S+) )?(\S+) (buy|sell) (\S+) at (\S+), (.+)",
                         head)
    assert found is not None, head

    _, instrument, side, size, price, when = found.groups()
    assert (instrument, side) == (fill["instrument"], fill["side"])
    assert float(size.replace(",", "")) == pytest.approx(fill["size"], rel=1e-11)

    printed, places = number(price)
    unit = 10.0 ** -places
    assert abs(printed - fill["price"]) <= 0.5 * unit + 1e-9 * abs(fill["price"])

    sign   = 1.0 if side == "buy" else -1.0
    fields = {
        "reference":       fill["reference"],
        "spread":          sign * fill["spread"],
        "slippage":        sign * fill["slippage"],
        "impact":          sign * fill["impact"],
        "held in the bar": fill["adjustment"] - fill["lifted"],
    }

    lines = [re.split(r" {2,}", line.strip()) for line in body]
    assert [label for label, *_ in lines][:5] == list(PRICES)

    total, summed = 0.0, 0
    for label, said, shown in lines:
        value, decimals = number(shown)
        if label in PRICES or said.startswith("lifted"):
            assert decimals == places, (label, shown)
            field = fields.get(label, fill["lifted"])
            assert abs(value - field) <= 0.5 * unit + 1e-9 * abs(fill["price"]), (
                label, shown, field)
            total, summed = total + value, summed + 1
        elif said.startswith("paid in"):
            assert abs(value - fill["paid_in"]) <= 0.5 * 10.0 ** -decimals
        else:
            assert label == "fee"
            assert abs(value - fill["fee"]) <= 0.5 * 10.0 ** -decimals

    assert abs(total - printed) <= (summed + 1) / 2.0 * unit + 1e-9 * abs(printed), text
    assert lines[-1][0] == "fee"

    return text


def costs_example():
    # the Specification's costs example: a fixed spread and slippage of 1 bp and a
    # taker fee of 0.05%, a buy of 0.1 at the 60,000 open and a sale at the 60,400 one
    rows = [(60_000.0, 60_000.0, 60_000.0, 60_000.0), (60_000.0, 60_600.0, 59_900.0,
                                                       60_500.0),
            (60_400.0, 60_500.0, 60_300.0, 60_450.0)]
    venue = probatum.Venue(
        balance = 10_000.0,
        fees    = {"maker": 0.0, "taker": 0.0005},
        fills   = {"spread": {"fixed": 1.0}, "slippage": {"fixed": 1.0}},
        funding = {"every_bars": 1, "rate": 0.0001},
    )

    return probatum.backtest(np.array([0.1, 0.0, np.nan]), data(rows), venue,
                             sizing="quantity")


def test_the_costs_example_reads_its_reference_each_cost_and_its_fee():
    result = costs_example()

    assert result.explain(result.fills[0]) == "\n".join([
        "fill 0, BTCUSDT buy 0.1 at 60,009.00, 2024-01-01 01:00",
        "  reference        open of the bar                 60,000.00",
        "  spread           half of 1.0 bps, fixed              +3.00",
        "  slippage         1.0 bps                             +6.00",
        "  impact           not modelled                        +0.00",
        "  held in the bar  the bar's high was not reached      +0.00",
        "  fee              taker 0.05% on 6,000.90              3.00"])
    assert result.explain(result.fills[1]) == "\n".join([
        "fill 1, BTCUSDT sell 0.1 at 60,390.94, 2024-01-01 02:00",
        "  reference        open of the bar                60,400.00",
        "  spread           half of 1.0 bps, fixed             -3.02",
        "  slippage         1.0 bps                            -6.04",
        "  impact           not modelled                       +0.00",
        "  held in the bar  the bar's low was not reached      +0.00",
        "  fee              taker 0.05% on 6,039.09             3.02"])


def inverse_charged():
    """A buy and a sale of an inverse contract at a spread of 10 bps and a slippage of
    50 bps, as the router gives its bars."""
    rows = [(60_000.0, 60_000.0, 60_000.0, 60_000.0), (60_000.0, 61_000.0, 59_000.0,
                                                       60_500.0),
            (60_500.0, 61_500.0, 59_500.0, 61_000.0), (61_000.0, 61_000.0, 61_000.0,
                                                       61_000.0)]
    given = frame(rows)
    given.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                       interval="1h", quote="USD", contract_size=100.0, warnings=[],
                       volume_unit="contract")

    venue = probatum.Venue(
        balance = 1.0,
        fees    = {"maker": 0.0002, "taker": 0.0005},
        fills   = {"spread": {"fixed": 10.0}, "slippage": {"fixed": 50.0}},
    )

    return probatum.backtest(np.array([300.0, 0.0, np.nan, np.nan]), probatum.Data(given),
                             venue, sizing="quantity")


def held():
    """A buy at an open the bar never trades above, its slippage held to the high."""
    rows = [(100.0, 100.0, 100.0, 100.0), (100.0, 100.0, 98.0, 99.0),
            (99.0, 99.0, 99.0, 99.0)]
    venue = VENUE.replace(fills={"slippage": {"fixed": 50.0}})

    return probatum.backtest(np.array([1.0, np.nan, np.nan]), data(rows), venue,
                             sizing="quantity")


def lifted():
    """The Specification's margin bound example on a bar that opens at 95 and reaches
    95.02: the sale is lifted to the high and the rest paid in."""
    rows = [(100.0, 100.0, 100.0, 100.0), (100.0, 100.0, 100.0, 100.0),
            (95.0, 95.02, 95.0, 95.0)]
    venue = probatum.Venue(
        balance  = 51.5,
        fees     = {"maker": 0.0, "taker": 0.001},
        leverage = 50.0,
    )

    return probatum.backtest(np.array([10.0, 0.0, np.nan]), data(rows, "XYZUSDT"), venue,
                             sizing="quantity")


def stories():
    """A Strategy whose orders tell every kind of story: a market entry its budget
    clamps, a bracket placed on the long the limit left, a post_only limit whose
    replace is refused, that fills in part, moves and fills, a value order, and a limit
    left resting."""
    rows = [(100.0, 100.5, 99.5, 100.0)] * 3 + [
        (100.0, 100.2, 98.0, 99.0), (99.0, 99.2, 98.0, 98.6), (98.6, 98.8, 98.2, 98.5),
        (98.5, 101.0, 98.3, 100.5), (100.5, 103.0, 100.0, 102.0),
        (102.0, 102.5, 90.0, 95.0), (95.0, 96.0, 94.0, 95.5)] + [
        (95.5, 96.0, 95.0, 95.5)] * 4

    plan = {
        0: [("rest", lambda account, bar: account.buy(3.0, limit=99.0, post_only=True))],
        1: [("refused", lambda account, bar: account.replace(account.orders[0],
                                                             price=101.0))],
        3: [("moved", lambda account, bar: account.replace(account.orders[0],
                                                           price=98.5))],
        5: [("entry", lambda account, bar: account.buy(5.0)),
            ("exits", lambda account, bar: account.bracket(stop=92.0,
                                                           take_profit=102.5))],
        9: [("value", lambda account, bar: account.buy_value(150.0)),
            ("left", lambda account, bar: account.sell(0.5, limit=150.0))],
    }

    venue = VENUE.replace(fees={"maker": 0.0002, "taker": 0.0005},
                          fills={"liquidity": {"budget": 0.1}, "spread": {"fixed": 2.0},
                                 "slippage": {"fixed": 1.0}, "impact": {"sqrt": 0.5}})

    return scripted(plan, rows, venue, volume=20.0)


def test_every_fill_prints_numbers_that_read_back_and_add_up_to_its_price():
    results = {**runs(), "charged": probatum.backtest(weights(), pair(), CHARGED),
               "inverse charged": inverse_charged(), "held": held(), "lifted": lifted(),
               "stories": stories(), "hair": hair(), "costs": costs_example()}
    results["book"] = probatum.combine({"A": results["costs"], "B": results["held"]})

    for name, result in results.items():
        assert len(result.fills) or name == "idle", name
        for k in range(len(result.fills)):
            read(result, k)


def test_a_stop_reads_its_trigger_or_where_the_bar_stood_past_it():
    # the Specification's example: a sell stop at 95 fills at 95 on a bar that opens
    # at 96 and trades down to 94, and at 93 on a bar that opens at 93
    plan = {
        0: [("long", lambda account, bar: account.buy(1.0))],
        1: [("stop", lambda account, bar: account.sell(1.0, stop=95.0))],
    }
    flat = [(100.0, 100.0, 100.0, 100.0)] * 2

    for opened, price, said in ((96.0, 95.0, "its trigger"),
                                (93.0, 93.0, "past its trigger, where the bar stood")):
        bar    = (opened, max(opened, 94.0), min(opened, 94.0), 94.0)
        result = scripted(plan, flat + [bar] * 2)
        stop   = result.fills[-1]
        assert (stop["reason"], stop["price"]) == ("stop", price)
        assert reference_of(result, len(result.fills) - 1) == said


def lines_of(result, k):
    """Fill ``k``'s text, checked, as its label, text and number a line."""
    return [re.split(r" {2,}", line.strip()) for line in read(result, k).splitlines()]


def reference_of(result, k):
    return lines_of(result, k)[1][1]


def test_a_limit_reads_its_price_or_what_traded_to_it():
    # a buy limit at 99 placed at a close of 100 fills at 99 on a bar that opens at
    # 98.50, and one at 110 fills at 100, the last close, on a bar that opens at 96
    # with a high of 98, since nothing from that close through the bar reached 110
    flat = [(100.0, 100.0, 100.0, 100.0)]

    for limit, opened, high, price, said in (
            (99.0, 98.5, 99.5, 99.0, "the limit's price"),
            (110.0, 96.0, 98.0, 100.0, "what traded since the last close")):
        plan   = {0: [("limit", lambda account, bar, at=limit: account.buy(1.0, limit=at))]}
        result = scripted(plan, flat + [(opened, high, 95.0, 97.0)] * 2)
        assert result.fills[0]["price"] == price
        assert reference_of(result, 0)  == said

    # a take-profit its entry's fill opened through on the same bar reads what traded
    # since that fill
    plan = {0: [("entry", lambda account, bar: account.buy(1.0)),
                ("exits", lambda account, bar: account.bracket(take_profit=99.0))]}

    result = scripted(plan, flat + [(100.0, 100.5, 99.5, 100.0)] * 2)
    leg    = result.fills[1]

    assert leg["reason"] == "take_profit" and leg["reference"] != 99.0
    assert reference_of(result, 1) == ("what traded before it since the last close, or "
                                       "since the fill inside its bar")


def half_a_bitcoin(bar, venue):
    """The Specification's one-instrument account: 10,000 USDT after the entry's fee,
    a long of 0.5 BTC from 60,000, then ``bar``."""
    rows = [(60_000.0,) * 4, (60_000.0,) * 4, bar, bar]
    return probatum.backtest(np.array([0.5, np.nan, np.nan, np.nan]), data(rows), venue,
                             sizing="quantity")


def test_a_forced_close_reads_where_the_account_was_closed_out_or_the_sale_began():
    fees   = {"maker": 0.0002, "taker": 0.0005}
    bare   = probatum.Venue(balance=10_015.0, fees=fees)
    result = half_a_bitcoin((41_500.0, 41_500.0, 39_000.0, 39_500.0), bare)
    close  = result.fills[-1]

    assert close["reason"] == "liquidation" and not close["taker"]

    lines = lines_of(result, len(result.fills) - 1)
    assert lines[1][1] == "where the account was closed out"
    assert lines[2][1] == "none in a takeover"
    assert lines[-1][1].startswith("liquidation fee 0.05% on 20,010.0")
    assert lines[-1][2] == "10.01"

    kept   = [{"up_to": 1e12, "leverage": 20.0, "rate": 0.005}]
    slide  = (41_000.0, 41_200.0, 40_150.0, 40_500.0)
    result = half_a_bitcoin(slide, probatum.Venue(
        balance = 10_015.0,
        fees    = fees,
        tiers   = kept,
    ))

    assert lines_of(result, len(result.fills) - 1)[-1][1:] == ["none", "0.00"]

    # a sale on the book is a taker's, and pays the spread and the slippage it sold at
    book = probatum.Venue(
        balance     = 10_015.0,
        fees        = fees,
        tiers       = kept,
        liquidation = "book_then_backstop",
        backstop    = 2.0 / 3.0,
        fills       = {"spread": {"fixed": 4.0}, "slippage": {"fixed": 2.0}},
    )

    result = half_a_bitcoin(slide, book)
    sale   = result.fills[-1]

    assert sale["reason"] == "liquidation" and sale["taker"]

    lines = lines_of(result, len(result.fills) - 1)
    assert lines[1][1] == "where the sale began"
    assert [line[1] for line in lines[2:4]] == ["half of 4.0 bps, fixed", "2.0 bps"]
    assert lines[-1][1].startswith("taker 0.05% on ")


def test_an_inverse_contract_prints_the_venue_s_rates():
    result = inverse_charged()

    for k in range(2):
        lines = lines_of(result, k)
        assert lines[2][1] == "half of 10.0 bps, fixed"
        assert lines[3][1] == "50.0 bps"
        assert lines[-1][1].startswith("taker 0.05% on 0.")
        # the fee is in the coin, to five significant figures
        assert re.fullmatch(r"0\.000\d{5}", lines[-1][2]), lines[-1]


def test_a_price_the_bar_held_and_one_the_margin_bound_lifted_say_so():
    result = held()

    assert lines_of(result, 0)[5][1:] == ["held to the bar's high", "-0.50"]

    result = lifted()
    sale   = result.fills[-1]

    assert sale["lifted"] == pytest.approx(0.02) and sale["paid_in"] == pytest.approx(
        0.2502, abs=1e-4)
    assert result.costs["liquidation"] == pytest.approx(-0.4502, abs=1e-4)

    lines = lines_of(result, len(result.fills) - 1)
    assert [line[1:] for line in lines[5:]] == [
        ["the bar's low was not reached", "+0.00"],
        ["lifted toward where equity is zero", "+0.02"],
        ["paid in for what the bar could not hold", "0.25"],
        ["taker 0.1% on 950.20", "0.95"]]


def test_a_fee_reads_its_own_rate_and_a_price_below_one_its_five_figures():
    venue  = probatum.venues.bybit_perp.replace(funding="none")
    cheap  = [(0.05, 0.051, 0.049, 0.05)] * 4
    result = probatum.backtest(np.array([1_000.0, 0.0, np.nan, np.nan]),
                               data(cheap, "1000TAGUSDT"), venue, sizing="quantity")

    # bars that never move estimate no spread and no deviation, so only the slippage
    # of 1 bp moves the price, at 0.05 a twentieth of a hundredth of a cent
    assert read(result, 0) == "\n".join([
        "fill 0, 1000TAGUSDT buy 1,000 at 0.050005, 2024-01-01 01:00",
        "  reference        open of the bar                  0.050000",
        "  spread           half of 0 bps, estimated        +0.000000",
        "  slippage         1.0 bps                         +0.000005",
        "  impact           sqrt, k = 0.1                   +0.000000",
        "  held in the bar  the bar's high was not reached  +0.000000",
        "  fee              taker 0.11% on 50.01                 0.06"])


def test_the_specification_s_fills_read_as_it_works_them():
    # a buy of 0.141 BTC at an open of 21,012.00, its spread fixed at 0.6 bps, 1 bp of
    # slippage, and sqrt(0.1) impact on a window of returns of 0.4% and 24,800 BTC a bar
    c      = 21_012.0
    window = [(x,) * 4 for x in [c if k % 2 == 0 else c * math.exp(0.004)
                                 for k in range(24)]]

    rows  = window + [(c, 21_100.0, 21_000.0, 21_050.0), (21_050.0,) * 4]
    venue = probatum.Venue(
        fees  = {"maker": 0.0002, "taker": 0.0005},
        fills = {"spread": {"fixed": 0.6}, "slippage": {"fixed": 1.0},
                 "impact": {"sqrt": 0.1}},
    )

    asked = np.full(26, np.nan)
    asked[23] = 0.141
    result = probatum.backtest(asked, data(rows, volume=24_800.0), venue,
                               sizing="quantity")

    assert read(result, 0) == "\n".join([
        "fill 0, BTCUSDT buy 0.141 at 21,014.75, 2024-01-02 00:00",
        "  reference        open of the bar                 21,012.00",
        "  spread           half of 0.6 bps, fixed              +0.63",
        "  slippage         1.0 bps                             +2.10",
        "  impact           sqrt, k = 0.1                       +0.02",
        "  held in the bar  the bar's high was not reached      +0.00",
        "  fee              taker 0.05% on 2,963.08              1.48"])

    # the six bars of the spread estimate's example, a spread of 0.1027%, half of it
    # 0.05 on a buy whose reference is 100.40
    six = [(100.0, 100.5, 99.8, 100.2), (100.3, 100.6, 100.0, 100.0),
           (100.1, 100.6, 99.9, 100.4), (100.5, 100.8, 100.1, 100.1),
           (100.2, 100.7, 100.0, 100.5), (100.6, 100.9, 100.3, 100.3),
           (100.4, 100.6, 100.2, 100.5), (100.5,) * 4]

    asked = np.full(8, np.nan)
    asked[5] = 1.0
    result = probatum.backtest(asked, data(six),
                               probatum.Venue(fills={"spread": "estimated"}),
                               sizing="quantity")

    assert lines_of(result, 0)[1:3] == [
        ["reference", "open of the bar", "100.40"],
        ["spread", "half of 10.3 bps, estimated", "+0.05"]]

    # a rate below 1 bp keeps three figures, and one a hair below 1 bp reads as one
    for given, said in ((0.45, "0.45 bps"), (0.9996, "1.0 bps")):
        venue  = VENUE.replace(fills={"slippage": {"fixed": given}})
        result = probatum.backtest(np.array([1.0, np.nan, np.nan]),
                                   data([(100.0, 101.0, 99.0, 100.0)] * 3), venue,
                                   sizing="quantity")
        assert lines_of(result, 0)[3][1] == said


def story(result, name):
    """The order log's text of the order the script kept under ``name``."""
    return result.explain(result.policy.kept[name]).splitlines()


def test_an_order_s_story_is_each_of_its_events():
    result = stories()
    rest   = story(result, "rest")

    assert rest[0] == "order 1, BTCUSDT buy limit 3 at 99.00"

    kinds = [line.split()[2] for line in rest[1:]]
    assert kinds[:2] == ["placed", "refused"]
    assert kinds.count("filled") >= 2 and "placed" in kinds[2:]
    assert rest[2].endswith(result.orders.events("refused")[0]["reason"])

    for line in rest[1:]:
        if line.split()[2] == "filled":
            k    = int(line.rsplit(" ", 1)[1])
            fill = result.fills[k]
            assert fill["order"] == 1 and reference_of(result, k) == "the limit's price"

    moved = [line for line in rest[1:] if line.split()[2] == "placed"][1]
    # a replace places the whole order again, what it filled and what remains together
    assert re.search(r"placed +replace, buy 3 at 98\.50$", moved), moved

    exits  = result.policy.kept["exits"]
    stop   = result.explain(exits.stop).splitlines()
    profit = result.explain(exits.take_profit).splitlines()
    assert stop[0] == f"order {exits.stop.id}, BTCUSDT sell stop at 92.00"

    value = story(result, "value")
    assert value[0] == f"order {result.policy.kept['value'].id}, BTCUSDT buy value of 150"

    left = story(result, "left")
    assert len(left) == 2 and left[1].endswith("limit, sell 0.5 at 150.00")

    for row in result.orders:
        if row["kind"] != "placed":
            continue

        lines = result.explain(row).splitlines()
        named = [int(line.rsplit(" ", 1)[1]) for line in lines if " filled " in line]
        own   = [k for k in range(len(result.fills)) if result.fills[k]["order"] ==
                 row["order"]]
        assert named == own, lines

    assert profit[0] == f"order {exits.take_profit.id}, BTCUSDT sell take_profit at 102.50"


def test_a_bracket_placed_while_flat_shows_its_side_once_it_arms():
    # the Specification's example: a buy limit at 95 placed while flat, with a bracket
    # at 90 and 110 placed after it, meets a bar that opens at 100 and trades down to
    # 85, and the entry fills at 95, the bracket arms and its stop closes it at 90
    plan = {0: [("entry", lambda account, bar: account.buy(1.0, limit=95.0)),
                ("exits", lambda account, bar: account.bracket(stop=90.0,
                                                               take_profit=110.0))]}
    rows = [(100.0, 100.0, 100.0, 100.0), (100.0, 100.0, 85.0, 88.0),
            (88.0, 88.0, 88.0, 88.0)]

    result = scripted(plan, rows)
    entry  = lines_of(result, 0)

    assert result.fills[0]["taker"] is False
    assert [line[1] for line in entry[2:5]] == ["none for a maker"] * 3
    assert story(result, "entry") == [
        "order 1, BTCUSDT buy limit 1 at 95.00",
        "  2024-01-01 00:00  placed  limit, buy 1 at 95.00",
        "  2024-01-01 01:00  filled  1 at 95.00, fill 0"]

    exits = result.policy.kept["exits"]
    told  = result.explain(exits.stop).splitlines()
    assert told[:3] == ["order 2, BTCUSDT stop at 90.00",
                        "  2024-01-01 00:00  placed  stop at 90.00",
                        "  2024-01-01 01:00  armed   stop, sell at 90.00"]
    assert told[3] == "  2024-01-01 01:00  filled  1 at 90.00, fill 1"
    assert reference_of(result, 1) == "its trigger"

    told = result.explain(exits.take_profit).splitlines()
    assert told[0] == "order 3, BTCUSDT take_profit at 110.00"
    assert told[-1].split(None, 3)[2:] == ["cancelled", "its OCO partner filled"]


def test_the_specification_s_clamps_tell_their_stories():
    # a spot wallet of 1,000 USDT buys 20 at an open of 100 with a taker fee of 0.1%,
    # and its cash pays for 9.99001
    spot = probatum.Venue(
        kind    = "spot",
        balance = 1_000.0,
        fees    = {"maker": 0.001, "taker": 0.001},
    )

    flat   = [(100.0, 100.0, 100.0, 100.0)] * 3
    result = scripted({0: [("buy", lambda account, bar: account.buy(20.0))]}, flat, spot,
                      market="spot")

    clamp = result.orders.events("clamped")[0]["reason"]
    assert clamp.startswith("the spot wallet's cash allows 9.99")
    assert story(result, "buy") == [
        "order 1, BTCUSDT buy market 20",
        "  2024-01-01 00:00  placed   market, buy 20",
        f"  2024-01-01 01:00  clamped  {clamp}",
        "  2024-01-01 01:00  filled   9.99000999001 at 100.00, fill 0",
        "  2024-01-01 01:00  expired  a market order takes one bar"]

    # under budget(0.1) with a trailing volume of 50 the bar holds 5, so five market
    # orders of 2 fill 2, 2 and 1, and the last two nothing
    plan   = {1: [(k, lambda account, bar: account.buy(2.0)) for k in range(5)]}
    result = scripted(plan, [(100.0, 100.0, 100.0, 100.0)] * 4,
                      VENUE.replace(fills={"liquidity": {"budget": 0.1}}), volume=50.0)

    told = [story(result, k) for k in range(5)]
    assert [len([line for line in lines if " filled " in line]) for lines in told] == [
        1, 1, 1, 0, 0]
    assert told[2][2].endswith("clamped  the liquidity budget allows 1 of the 2 asked")
    assert "not_filled" in told[3][2]

    # with a lot of 0.001 a buy of 0.0006 is refused as it is placed, and with a
    # maximum of 1.0005 a buy of 5 is held to 1.000
    sized = VENUE.replace(fills={"sizes": "on"}, lot=0.001, tick=0.01, max_qty=1.0005)
    plan  = {0: [("tiny", lambda account, bar: account.buy(0.0006)),
                 ("large", lambda account, bar: account.buy(5.0))]}

    result = scripted(plan, flat, sized)
    tiny   = story(result, "tiny")

    assert tiny[0] == f"order {result.policy.kept['tiny'].id}, BTCUSDT buy 0.0006, refused"

    large = result.policy.kept["large"].id
    assert story(result, "large") == [
        f"order {large}, BTCUSDT buy market 1",
        "  2024-01-01 00:00  placed   market, buy 1",
        "  2024-01-01 00:00  clamped  the maximum order quantity allows 1 of the 5 asked",
        "  2024-01-01 01:00  filled   1 at 100.00, fill 0"]


def test_an_order_no_engine_could_fill_tells_its_story_too():
    # a size below zero or past every float is refused as the call gave it, and a
    # limit at or below zero, which a perpetual's book takes, rests there
    plan = {0: [("minus", lambda account, bar: account.buy(-0.5)),
                ("inf", lambda account, bar: account.buy(math.inf)),
                ("zero", lambda account, bar: account.buy(1.0, limit=0.0)),
                ("below", lambda account, bar: account.sell(1.0, stop=-0.5))]}

    result = scripted(plan, [(100.0, 100.0, 100.0, 100.0)] * 3)

    assert story(result, "minus")[0] == "order 1, BTCUSDT buy -0.5, refused"
    assert story(result, "inf")[0]   == "order 2, BTCUSDT buy inf, refused"
    assert story(result, "zero") == ["order 3, BTCUSDT buy limit 1 at 0.00",
                                      "  2024-01-01 00:00  placed  limit, buy 1 at 0.00"]
    assert story(result, "below") == ["order 4, BTCUSDT sell stop 1 at -0.50",
                                       "  2024-01-01 00:00  placed  stop, sell 1 at -0.50"]

    # a bracket refused while flat has neither side nor size to name
    plan = {0: [("rest", lambda account, bar: account.buy(1.0, limit=90.0)),
                ("exits", lambda account, bar: account.bracket(stop=80.0,
                                                               take_profit=120.0))]}

    result = scripted(plan, [(100.0, 100.0, 100.0, 100.0)] * 3,
                      VENUE.replace(max_open_orders=1))

    stop = result.explain(result.policy.kept["exits"].stop).splitlines()
    assert stop[0] == "order 2, BTCUSDT, refused"


class Restless(probatum.Strategy):
    """At closes ``seed`` draws, a market, limit, stop or value order of either side,
    a bracket, a limit moved or an order cancelled, each bracket's legs kept."""


    def __init__(self, seed):
        self.seed = seed


    def init(self, data):
        self.rng = np.random.default_rng(self.seed)
        self.legs, self.limits, self.handles = set(), [], []


    def next(self, bar, account):
        rng, close = self.rng, bar.close
        verb, side = int(rng.integers(8)), "buy" if rng.random() < 0.5 else "sell"
        size, place = float(rng.choice([0.5, 1.0, 2.0])), getattr(account, side)
        sign = 1.0 if side == "buy" else -1.0

        if verb == 0:
            self.handles.append(place(size))
        elif verb == 1:
            self.limits.append(place(size, limit=close * (1.0 - 0.01 * sign)))
        elif verb == 2:
            self.handles.append(place(size, stop=close * (1.0 + 0.01 * sign)))
        elif verb == 3:
            held  = 1.0 if account.position >= 0.0 else -1.0
            exits = account.bracket(stop=close * (1.0 - 0.03 * held),
                                    take_profit=close * (1.0 + 0.03 * held))
            self.legs |= {leg.id for leg in exits if leg is not None}
        elif verb == 4 and self.limits:
            account.replace(self.limits[-1], price=close * (1.0 - 0.005 * sign))
        elif verb == 5 and self.handles + self.limits:
            held = self.handles + self.limits
            account.cancel(held[int(rng.integers(len(held)))])
        elif verb == 6:
            self.handles.append(getattr(account, f"{side}_value")(close * size))

        self.handles = [handle for handle in self.handles if handle is not None]
        self.limits  = [handle for handle in self.limits if handle is not None]


def restless(seed):
    rng    = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 80)))
    opens  = np.concatenate([[100.0], closes[:-1]])
    rows   = [(o, max(o, c) * 1.004, min(o, c) * 0.996, c) for o, c in zip(opens, closes)]
    venue  = VENUE.replace(fees={"maker": 0.0002, "taker": 0.0005}, lot=0.001, tick=0.01,
                           fills={"sizes": "on", "spread": {"fixed": 2.0},
                                  "slippage": {"fixed": 1.0}})

    return probatum.backtest(Restless(seed), data(rows), venue)


# the events after which an order has no life left but a replace's refusal
ENDS = ("expired", "cancelled", "not_filled")


def test_every_story_agrees_with_the_logs_it_is_told_from():
    told = stories()
    runs = [restless(seed) for seed in range(6)]

    assert sum(len(run.policy.legs) for run in runs) and sum(
        len(run.fills) for run in runs) > 20

    for result in runs + [told]:
        fills, policy = result.fills, result.policy
        legs = (policy.legs if isinstance(policy, Restless) else
                {leg.id for leg in policy.kept["exits"] if leg is not None})
        orders = sorted({row["order"] for row in result.orders if row["order"] is not None})

        for order in orders:
            first  = next(row for row in result.orders if row["order"] == order)
            events = [line.split(None, 3) for line in
                      result.explain(first).splitlines()[1:]]
            own    = [k for k in range(len(fills)) if fills[k]["order"] == order]
            filled = [event[3] for event in events if event[2] == "filled"]
            assert len(filled) == len(own), (order, events)
            for said, k in zip(filled, own):
                size, _, where = said.partition(" at ")
                assert float(size.replace(",", "")) == pytest.approx(fills[k]["size"],
                                                                     rel=1e-11)
                assert where.endswith(f", fill {k}")

            ended = False
            for _, _, kind, *said in events:
                said = said[0] if said else ""
                if ended:
                    assert kind == "refused" and said.startswith("replace: "), events
                ended = ended or kind in ENDS or (kind == "refused"
                                                  and not said.startswith("replace: "))

            if any(event[2] == "armed" for event in events):
                assert order in legs, (order, events)


def test_only_the_run_that_handed_out_a_handle_tells_it(tmp_path):
    result = stories()
    handle = result.policy.kept["rest"]
    told   = result.explain(handle)

    assert result.explain(copy.deepcopy(handle)) == told

    copied = copy.deepcopy(result)
    assert copied.policy.kept["rest"] is not handle
    assert copied.explain(copied.policy.kept["rest"]) == told
    assert result.explain(copied.policy.kept["rest"]) == told
    assert result.explain(next(row for row in result.orders if row["order"] == 1)) == told

    other = stories()
    with pytest.raises(ProbatumValueError, match="of a run this Result does not hold"):
        other.explain(handle)

    unpickled = pickle.loads(pickle.dumps(result))
    with pytest.raises(ProbatumValueError, match="of a run this Result does not hold"):
        unpickled.explain(handle)

    result.save(tmp_path / "run")
    loaded = probatum.Result.load(tmp_path / "run")
    with pytest.raises(ProbatumValueError, match="of a run this Result does not hold"):
        loaded.explain(handle)

    book = probatum.combine({"A": result, "B": held()})
    with pytest.raises(ProbatumTypeError, match="pass a row of book.orders"):
        book.explain(handle)
    with pytest.raises(ProbatumTypeError, match="pass a row of result.fills"):
        result.explain(0)

    changed = dict(result.fills[0], price=1.0)
    with pytest.raises(ProbatumValueError, match="fill log does not hold"):
        result.explain(changed)
    with pytest.raises(ProbatumValueError, match="order log does not hold"):
        result.explain(dict(result.orders[0], bar=99))


def test_a_transfer_s_clamp_names_no_order_to_tell():
    venue = VENUE.replace(balance={"spot": 1_000.0, "margin": 500.0})
    rows  = [(100.0, 100.0, 100.0, 100.0)] * 3
    both  = probatum.Data(frame(rows), symbol="BTCUSDT", market="spot").join(
        probatum.Data(frame(rows), symbol="BTCUSDT", market="linear"))

    plan   = {0: [("moved", lambda account, bar: account.transfer(2_000.0, to="margin"))]}
    result = probatum.backtest(Script(plan), both, venue)
    clamp  = next(row for row in result.orders if row["order"] is None)

    with pytest.raises(ProbatumValueError, match="an event of no order"):
        result.explain(clamp)


def texts_of(result):
    """The text of every fill and of every order's event the Result can tell."""
    return [result.explain(row) for row in [*result.fills, *result.orders]
            if "reference" in row or row["order"] is not None]


def test_the_text_is_the_same_after_save_load_a_pickle_and_in_a_book(tmp_path):
    for k, result in enumerate((stories(), restless(0))):
        texts     = texts_of(result)
        unpickled = pickle.loads(pickle.dumps(result))
        # save reads the baselines first, which plays the policy again
        result.save(tmp_path / f"run{k}")
        loaded = probatum.Result.load(tmp_path / f"run{k}")
        assert texts_of(unpickled) == texts_of(loaded) == texts_of(result) == texts

    result = stories()
    book   = probatum.combine({"A": result, "B": held()})
    texts  = texts_of(book)

    book.baselines
    assert texts_of(book) == texts

    n, fill = next((k, row) for k, row in enumerate(book.fills) if row["venue"] == "A")
    told = book.explain(fill).splitlines()
    assert told[0].startswith(f"fill {n}, A BTCUSDT buy ")
    assert told[1:] == result.explain(result.fills[0]).splitlines()[1:]

    event = next(row for row in book.orders if row["venue"] == "B")
    assert book.explain(event).splitlines()[0].startswith("order 1, B BTCUSDT ")
