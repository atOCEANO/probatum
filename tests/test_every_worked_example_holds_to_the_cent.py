"""Every worked example on the Specification page, held to the cent.

Each test is one example, named for what it shows, and runs it through the reference
simulator written from that page, or through probatum where the rule is the Python
layer's own, a run's tables and a book's sum. The engine is held to the examples it
models by its own tests, which name the same numbers.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import ProbatumKeyError

from reference import statistics
from reference.environment import Episode, limit, logged
from reference.simulator import (Bar, Instrument, Simulator, Venue, edge, labels,
                                 periods_per_year, play_array, run, uniqueness)

HOUR = 3_600 * 10**9
DAY  = 24 * HOUR


def flat(price, volume=1_000.0):
    return Bar(price, price, price, price, volume)


def hourly(rows):
    return [i * HOUR for i in range(rows)]


# one tier at 20x with a maintenance rate of 0.5% and no amount
MAINTAINED = [(1e12, 20.0, 0.005, 0.0)]


def half_a_bitcoin(later, script=None, tiers=None, times=None, interval=None,
                   events=None, **venue):
    """The page's account: a margin wallet of 10,000 USDT holding a long of 0.5 BTC
    from 60,000 from row 1, its entry's taker fee of 15.00 taken from a balance of
    10,015, and then the bars in ``later``. The entry is a limit on a bar that opens
    on it, so a venue's slippage never moves it."""
    bars   = [[flat(60_000.0)], [flat(60_000.0)]] + [[bar] for bar in later]
    script = dict(script or {})
    script[0] = [("buy", dict(instrument=0, size=0.5, limit=60_000.0))]
    settings = dict(balance=10_015.0, maker=0.0002, taker=0.0005)
    settings.update(venue)

    return run(times or hourly(len(bars)), bars, [Instrument("BTCUSDT", tiers=tiers)],
               Venue(**settings), script=script, interval=interval, funding=events)


def btcusd(later, script, balance, events=None, **venue):
    """An inverse account on contracts of 100 dollars with a maintenance rate of
    0.5%, no fees unless asked, on two rows at 60,000 and then the bars in
    ``later``."""
    bars     = [[flat(60_000.0)], [flat(60_000.0)]] + [[bar] for bar in later]
    settings = dict(balance=balance, maker=0.0, taker=0.0)
    settings.update(venue)
    inverse = Instrument("BTCUSD", kind="inverse", contract=100.0, tiers=MAINTAINED)

    return run(hourly(len(bars)), bars, [inverse], Venue(**settings), script=script,
               funding=events)


def test_hourly_bars_over_a_year_annualise_at_8760():
    assert periods_per_year(hourly(8_761)) == 8_760.0


def test_253_daily_bars_from_one_2_january_to_the_next_annualise_at_252():
    days = [0] + list(range(1, 252)) + [365]

    assert periods_per_year([d * DAY for d in days]) == pytest.approx(252.0, abs=1e-9)


def test_a_3_bar_average_on_monday_averages_thursday_friday_and_monday():
    # 1 January 1970 was a Thursday, and a stock prints no bar over the weekend
    closes = np.array([100.0, 101.0, 98.5])
    bars   = np.column_stack([closes, closes + 1.0, closes - 1.0, closes,
                              np.full(3, 1e3)])
    data = probatum.Data(
        bars,
        symbol   = "SPY",
        market   = "spot",
        time     = [0, DAY, 4 * DAY],
        interval = "1d",
    )

    assert data.time.tolist() == [0, DAY, 4 * DAY]

    average = probatum.ta.sma(data.close, 3)
    assert np.isnan(average[1]) and average[2] == pytest.approx(299.5 / 3.0)


def test_orders_decided_at_fridays_close_fill_at_mondays_open():
    # 1 January 1970 was a Thursday, so these rows are Thursday, Friday and Monday
    times = [0, DAY, 4 * DAY]
    bars  = [[flat(100.0)], [flat(101.0)], [Bar(98.0, 99.0, 97.0, 98.5, 1_000.0)]]
    sim   = run(times, bars, [Instrument("SPY", kind="spot")],
                Venue(maker=0.0, taker=0.0), interval=DAY,
                script={0: [("buy", dict(instrument=0, size=10.0))],
                        1: [("buy", dict(instrument=0, size=5.0)),
                            ("sell", dict(instrument=0, size=10.0, stop=100.0))]})

    assert [(f["time"], f["price"], f["reason"]) for f in sim.fills] == [
        (DAY, 101.0, "market"), (4 * DAY, 98.0, "market"), (4 * DAY, 98.0, "stop")]


def test_a_delisting_closes_the_position_at_the_last_close_with_no_fee():
    times = [10 * HOUR, 11 * HOUR, 12 * HOUR]
    bars  = [[flat(48.0), flat(10.0)],
             [Bar(49.0, 51.0, 48.5, 50.0, 1_000.0), flat(10.0)],
             [None, flat(10.0)]]
    sim = run(times, bars,
              [Instrument("OLD", delist=12 * HOUR), Instrument("NEW")],
              Venue(maker=0.0, taker=0.0),
              script={0: [("buy", dict(instrument=0, size=2.0))],
                      1: [("buy", dict(instrument=0, size=1.0))]})
    closing = sim.fills[-1]

    assert closing["reason"] == "delist"
    assert closing["time"]   == 12 * HOUR
    assert closing["price"]  == 50.0
    assert closing["size"]   == 2.0
    assert closing["fee"]    == 0.0
    assert sim.position[0]   == 0.0
    assert [e["reason"] for e in sim.events if e["kind"] == "cancelled"] == ["delist"]


def test_a_target_of_half_on_ten_thousand_buys_0_083_of_bitcoin():
    sim = run(hourly(2), [[flat(60_000.0)], [flat(60_000.0)]],
              [Instrument("BTCUSDT", lot=0.001)], Venue(sizes=True),
              script={0: [("target", dict(weights=0.5))]})
    [order] = sim.orders.values()

    assert order["side"]   == "buy"
    assert order["size"]   == pytest.approx(0.083, abs=1e-12)
    assert sim.position[0] == pytest.approx(0.083, abs=1e-12)


def test_a_target_of_half_held_through_a_close_of_66_000_drifts_by_minus_0_0238():
    bars = [[flat(60_000.0)], [Bar(60_000.0, 66_000.0, 60_000.0, 66_000.0, 1_000.0)]]
    sim  = run(hourly(2), bars, [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0),
               script={0: [("target", dict(weights=0.5))]})

    assert math.isnan(sim.ledger[0]["instruments"][0]["drift"])
    assert sim.ledger[1]["equity"] == pytest.approx(10_500.0)

    leg = sim.ledger[1]["instruments"][0]
    assert leg["weight"] == pytest.approx(5_500.0 / 10_500.0)
    assert leg["drift"]  == pytest.approx(-0.0238, abs=5e-5)


def test_one_coin_targeting_minus_one_of_an_inverse_perpetual_wants_600_contracts():
    sim = btcusd([], {0: [("target", dict(weights=-1.0))]}, balance=1.0)

    assert sim.position[0] == pytest.approx(-600.0, abs=1e-9)


def test_a_spot_target_with_transfer_moves_its_shortfall_of_4_005_across():
    sim = run(hourly(2), [[flat(60_000.0)], [flat(60_000.0)]],
              [Instrument("BTCUSDT", kind="spot")],
              Venue(balance={"spot": 1_000.0, "margin": 9_000.0}, taker=0.001),
              script={0: [("target", dict(weights=0.5, transfer=True))]})

    assert sim.transfers   == []
    assert sim.margin_cash == pytest.approx(9_000.0 - 4_005.0, abs=1e-6)
    assert sim.position[0] == pytest.approx(0.083333, abs=5e-7)
    assert sim.spot_cash   == pytest.approx(0.0, abs=1e-6)


def test_a_flip_realizes_the_old_side_and_opens_the_rest_at_the_fill():
    sim = run(hourly(3), [[flat(100.0)], [flat(100.0)], [flat(110.0)]],
              [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0),
              script={0: [("buy", dict(instrument=0, size=1.0))],
                      1: [("sell", dict(instrument=0, size=1.5))]})
    [closed, opened] = sim.round_trips()

    assert closed["pnl"]   == pytest.approx(10.0, abs=0.005)
    assert sim.position[0] == pytest.approx(-0.5)
    assert sim.entry[0]    == 110.0
    assert opened["side"] == "short" and opened["open"]


def test_six_fills_make_two_closed_round_trips_and_an_open_one():
    rows   = 32
    script = {
        9:  [("buy", dict(instrument=0, size=1.0))],
        11: [("buy", dict(instrument=0, size=1.0))],
        14: [("sell", dict(instrument=0, size=0.5))],
        19: [("sell", dict(instrument=0, size=1.5))],
        24: [("sell", dict(instrument=0, size=1.0))],
        29: [("buy", dict(instrument=0, size=2.0))],
    }
    sim = run(hourly(rows), [[flat(100.0)] for _ in range(rows)],
              [Instrument("BTCUSDT")], script=script)

    assert [row["instruments"][0]["position"] for row in sim.ledger][9:31] == [
        0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0,
        0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
    ]

    trips = sim.round_trips()
    assert [(t["side"], t["open_bar"], t["close_bar"], t["size"], t["open"])
            for t in trips] == [("long", 10, 20, 2.0, False),
                                ("short", 25, 30, 1.0, False),
                                ("long", 30, None, 1.0, True)]
    assert sum(1 for t in trips if not t["open"]) == 2
    assert len(sim.ledger)         == rows
    assert sim.ledger[0]["equity"] == 10_000.0


def test_a_spot_buy_is_clamped_to_what_its_cash_pays_for():
    sim = run(hourly(2), [[flat(100.0)], [flat(100.0)]],
              [Instrument("BTCUSDT", kind="spot")],
              Venue(balance=1_000.0, taker=0.001),
              script={0: [("buy", dict(instrument=0, size=20.0))]})

    assert sim.position[0] == pytest.approx(9.99001, abs=5e-6)
    assert [e["kind"] for e in sim.events] == ["placed", "clamped", "filled", "expired"]

    clamp = sim.events[1]
    assert clamp["reason"] == (f"the spot wallet's cash allows {clamp['size']} of the "
                               f"20.0 asked")


def test_a_transfer_out_of_margin_moves_its_cash_but_never_its_unrealized_profit():
    venue = Venue(
        balance  = {"spot": 0.0, "margin": 600.0},
        maker    = 0.0,
        taker    = 0.0,
        leverage = 7.0,
    )
    sim = run(hourly(4), [[flat(100.0), flat(1.0)], [flat(100.0), flat(1.0)],
                          [flat(140.0), flat(1.0)], [flat(140.0), flat(1.0)]],
              [Instrument("BTCUSDT"), Instrument("USDCUSDT", kind="spot")], venue,
              script={0: [("buy", dict(instrument=0, size=10.0))],
                      2: [("transfer", dict(amount=1_000.0, to="spot"))]})

    assert sim.cap_requirement() == pytest.approx(200.0)
    assert sim.margin_cash       == pytest.approx(0.0, abs=1e-9)
    assert sim.spot_cash         == pytest.approx(600.0)

    [clamp] = [e for e in sim.events if e["kind"] == "clamped"]
    assert clamp["size"] == pytest.approx(600.0)


def test_with_the_cap_off_a_transfer_takes_a_losing_margin_wallet_to_its_trigger():
    # and a perpetual buy after the takeover finds the margin wallet with no equity
    sim = run(hourly(5), [[flat(100.0)], [flat(100.0)], [flat(95.0)], [flat(95.0)],
                          [flat(95.0)]],
              [Instrument("BTCUSDT")], Venue(balance=1_000.0, maker=0.0, taker=0.0,
                                             leverage=0.0),
              script={0: [("buy", dict(instrument=0, size=10.0))],
                      2: [("transfer", dict(amount=2_000.0, to="spot"))],
                      3: [("buy", dict(instrument=0, size=1.0))]})
    [clamp] = [e for e in sim.events if e["kind"] == "clamped"]

    assert clamp["size"] == 950.0
    assert clamp["reason"] == ("transfer: the margin wallet's free cash allows 950.0 of "
                               "the 2000.0 asked")
    assert sim.position[0] == 0.0
    assert (sim.spot_cash, sim.margin_cash) == (950.0, 0.0)

    [close] = [f for f in sim.fills if f["order"] is None]
    assert (close["bar"], close["price"]) == (3, 95.0)

    [missed] = [e for e in sim.events if e["kind"] == "not_filled"]
    assert (missed["order"], missed["reason"]) == (
        2, "nothing could fill: the margin wallet's equity allows none")


def test_a_fill_pays_its_size_times_its_price_times_the_fee():
    sim = run(hourly(2), [[flat(21_000.0)], [flat(21_014.75)]],
              [Instrument("BTCUSDT")], Venue(taker=0.0005),
              script={0: [("buy", dict(instrument=0, size=0.141))]})
    [fill] = sim.fills

    assert fill["size"] * fill["price"] == pytest.approx(2_963.08, abs=0.005)
    assert fill["fee"] == pytest.approx(1.48, abs=0.005)


def test_a_spot_buy_pays_its_own_0_1_percent_beside_a_perpetual_at_the_venues_0_05():
    sim = run(hourly(2), [[flat(60_000.0), flat(60_000.0)]] * 2,
              [Instrument("spot:BTCUSDT", kind="spot", maker=0.001, taker=0.001),
               Instrument("linear:BTCUSDT")],
              Venue(balance={"spot": 40_000.0, "margin": 40_000.0}, maker=0.0002,
                    taker=0.0005),
              script={0: [("buy", dict(instrument=0, size=0.5)),
                          ("sell", dict(instrument=1, size=0.5))]})

    assert [fill["fee"] for fill in sim.fills] == pytest.approx([30.0, 15.0])


@pytest.mark.parametrize("bar, price", [(Bar(96.0, 96.5, 94.0, 95.5, 1_000.0), 95.0),
                                        (Bar(93.0, 94.0, 92.0, 93.5, 1_000.0), 93.0)])
def test_a_sell_stop_fills_at_its_trigger_or_at_an_open_that_gapped_through_it(bar,
                                                                                 price):
    sim = run(hourly(3), [[flat(100.0)], [flat(100.0)], [bar]],
              [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0),
              script={0: [("buy", dict(instrument=0, size=1.0))],
                      1: [("sell", dict(instrument=0, size=1.0, stop=95.0))]})

    assert sim.fills[-1]["price"] == price
    assert sim.position[0]        == 0.0


@pytest.mark.parametrize("intrabar, price, cancelled",
                         [("pessimistic", 95.0, "take_profit"),
                          ("optimistic", 110.0, "stop")])
def test_a_bar_reaching_both_legs_of_a_bracket_fills_the_worse_first_by_default(
        intrabar, price, cancelled):
    wide = Bar(100.0, 111.0, 94.0, 100.0, 1_000.0)
    sim  = run(hourly(3), [[flat(100.0)], [flat(100.0)], [wide]],
               [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0, intrabar=intrabar),
               script={0: [("buy", dict(instrument=0, size=1.0))],
                       1: [("bracket", dict(instrument=0, stop=95.0,
                                            take_profit=110.0))]})

    assert sim.fills[-1]["price"] == price
    assert sim.position[0]        == 0.0

    [gone] = [o for o in sim.orders.values() if o["status"] == "cancelled"]
    assert gone["leg"] == cancelled


@pytest.mark.parametrize("later, taker", [
    ([Bar(98.5, 99.5, 98.0, 99.0, 1_000.0)], True),
    ([Bar(100.0, 100.5, 98.0, 99.0, 1_000.0)], False),
    ([flat(100.0), Bar(98.5, 99.5, 98.0, 99.0, 1_000.0)], False),
])
def test_a_limit_is_a_taker_only_when_its_first_bar_opens_through_it(later, taker):
    bars = [[flat(100.0)]] + [[bar] for bar in later]
    sim  = run(hourly(len(bars)), bars, [Instrument("BTCUSDT")],
               script={0: [("buy", dict(instrument=0, size=1.0, limit=99.0))]})
    [fill] = sim.fills

    assert fill["price"] == 99.0
    assert fill["taker"] is taker


def test_a_post_only_buy_at_the_close_and_a_buy_crossing_its_own_sell_are_refused():
    sim = run(hourly(2), [[flat(100.0)], [flat(100.0)]], [Instrument("BTCUSDT")],
              script={0: [("buy", dict(instrument=0, size=1.0, limit=100.0,
                                       post_only=True)),
                          ("sell", dict(instrument=0, size=1.0, limit=100.5)),
                          ("buy", dict(instrument=0, size=1.0, limit=101.0))]})

    assert [(e["order"], e["kind"]) for e in sim.events] == [
        (1, "refused"), (2, "placed"), (3, "refused")]


@pytest.mark.parametrize("limit, low, fills", [("touch", 99.40, True),
                                               ("through", 99.40, False),
                                               ("through", 99.398, False),
                                               ("through", 99.394, True),
                                               ("through", 99.39, True)])
def test_a_buy_limit_at_99_40_fills_on_touch_and_through_half_a_tick_past(limit, low,
                                                                          fills):
    sim = run(hourly(2), [[flat(100.0)], [Bar(100.0, 100.2, low, 99.8, 1_000.0)]],
              [Instrument("BTCUSDT", tick=0.01)], Venue(limit=limit),
              script={0: [("buy", dict(instrument=0, size=1.0, limit=99.40))]})

    assert bool(sim.fills) is fills


def test_a_buy_limit_at_110_that_no_trade_since_the_close_reached_fills_at_100():
    sim = run(hourly(2), [[flat(100.0)], [Bar(96.0, 98.0, 95.0, 97.0, 1_000.0)]],
              [Instrument("BTCUSDT")],
              script={0: [("buy", dict(instrument=0, size=1.0, limit=110.0))]})
    [fill] = sim.fills

    assert (fill["price"], fill["taker"]) == (100.0, True)


def test_a_replace_keeps_its_id_and_takes_the_first_free_slot():
    sim = run(hourly(2), [[flat(100.0)], [flat(100.0)]], [Instrument("BTCUSDT")],
              script={0: [("buy", dict(instrument=0, size=1.0, limit=95.0)),
                          ("buy", dict(instrument=0, size=1.0, limit=94.0)),
                          ("cancel", dict(order_id=1)),
                          ("replace", dict(order_id=2, price=92.0)),
                          ("buy", dict(instrument=0, size=1.0, limit=93.0))]})

    assert [(o["id"], o["price"]) for o in sim.slots] == [(2, 92.0), (3, 93.0)]


def test_an_ioc_limit_expires_after_its_first_bar_and_a_partial_fok_fills_nothing():
    sim = run(hourly(2), [[flat(100.0)], [Bar(100.0, 100.5, 97.0, 99.0, 1_000.0)]],
              [Instrument("BTCUSDT", kind="spot")], Venue(balance=300.0),
              script={0: [("buy", dict(instrument=0, size=1.0, limit=96.0, tif="ioc")),
                          ("buy", dict(instrument=0, size=5.0, tif="fok"))]})

    assert sim.fills == []
    assert [sim.orders[k]["status"] for k in (1, 2)] == ["expired", "expired"]

    [fok] = [e["reason"] for e in sim.events if e["kind"] == "not_filled"]
    assert fok.startswith("fok: the spot wallet's cash allows ")
    assert fok.endswith(" of the 5.0 asked")


def test_a_bracket_placed_with_its_entry_arms_when_the_entry_fills():
    sim = run(hourly(2), [[flat(100.0)], [Bar(100.0, 101.0, 99.0, 100.0, 1_000.0)]],
              [Instrument("BTCUSDT")],
              script={0: [("buy", dict(instrument=0, size=1.0)),
                          ("bracket", dict(instrument=0, stop=95.0,
                                           take_profit=110.0))]})
    armed = [(e["bar"], e["reason"]) for e in sim.events if e["kind"] == "armed"]

    assert armed == [(1, "stop"), (1, "take_profit")]
    assert [o["status"] for o in sim.orders.values()] == ["filled", "resting",
                                                          "resting"]


def test_a_bracket_placed_after_a_limit_entry_stops_it_on_the_bar_it_fills():
    sim = run(hourly(2), [[flat(100.0)], [Bar(100.0, 100.0, 85.0, 88.0, 1_000.0)]],
              [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0),
              script={0: [("buy", dict(instrument=0, size=1.0, limit=95.0)),
                          ("bracket", dict(instrument=0, stop=90.0,
                                           take_profit=110.0))]})

    assert [(f["price"], f["reason"]) for f in sim.fills] == [(95.0, "limit"),
                                                             (90.0, "stop")]
    assert sim.position[0] == 0.0


@pytest.mark.parametrize("intrabar, fills", [
    ("pessimistic", [(95.0, "limit")]),
    ("optimistic", [(95.0, "limit"), (97.0, "take_profit")]),
])
def test_a_take_profit_reached_only_before_its_limit_entry_needs_the_optimistic_way(
        intrabar, fills):
    sim = run(hourly(2), [[flat(100.0)], [Bar(100.0, 100.5, 94.0, 95.5, 1_000.0)]],
              [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0, intrabar=intrabar),
              script={0: [("buy", dict(instrument=0, size=1.0, limit=95.0)),
                          ("bracket", dict(instrument=0, stop=90.0,
                                           take_profit=97.0))]})

    assert [(f["price"], f["reason"]) for f in sim.fills] == fills
    assert not any(f["taker"] for f in sim.fills)


@pytest.mark.parametrize("intrabar", ["pessimistic", "optimistic"])
def test_a_stop_leg_of_a_breakout_long_fills_where_the_price_crossed_it_after_the_entry(
        intrabar):
    sim = run(hourly(2), [[flat(100.0)], [Bar(100.0, 108.0, 98.0, 103.0, 1_000.0)]],
              [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0, intrabar=intrabar),
              script={0: [("buy", dict(instrument=0, size=1.0, stop=105.0)),
                          ("bracket", dict(instrument=0, stop=102.0,
                                           take_profit=110.0))]})

    assert [(f["price"], f["reason"]) for f in sim.fills] == [(105.0, "stop"),
                                                             (102.0, "stop")]


def test_a_bracket_armed_on_a_long_is_cancelled_when_the_long_flips_short():
    sim = run(hourly(4), [[flat(100.0)], [flat(100.0)], [flat(100.0)],
                          [Bar(100.0, 111.0, 94.0, 100.0, 1_000.0)]],
              [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0),
              script={0: [("buy", dict(instrument=0, size=1.0))],
                      1: [("bracket", dict(instrument=0, stop=95.0,
                                           take_profit=110.0)),
                          ("sell", dict(instrument=0, size=2.0))]})

    assert sim.position[0] == -1.0

    legs = [o for o in sim.orders.values() if o["leg"] is not None]
    assert [(o["side"], o["status"]) for o in legs] == [("sell", "cancelled"),
                                                        ("sell", "cancelled")]
    assert [(e["bar"], e["reason"]) for e in sim.events
            if e["kind"] == "cancelled"] == [(2, "its position closed")] * 2


def test_a_stop_reached_on_a_bar_with_no_volume_stays_armed_for_the_next():
    sim = run(hourly(4), [[flat(100.0)], [flat(100.0)],
                          [Bar(99.0, 99.5, 94.0, 96.0, 0.0)],
                          [Bar(96.0, 96.5, 94.5, 95.5, 1_000.0)]],
              [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0),
              script={0: [("buy", dict(instrument=0, size=1.0))],
                      1: [("sell", dict(instrument=0, size=1.0, stop=95.0))]})

    assert [(f["bar"], f["price"]) for f in sim.fills] == [(1, 100.0), (3, 95.0)]


def test_a_taker_buy_of_0_141_pays_spread_slippage_and_impact_up_to_21_014_75():
    # 24 returns of +a and -a have a sample deviation of exactly 0.004
    a      = 0.004 * math.sqrt(23.0 / 24.0)
    closes = [21_000.0]
    for k in range(24):
        closes.append(closes[-1] * math.exp(a if k % 2 == 0 else -a))

    bars = [[Bar(c, c, c, c, 24_800.0)] for c in closes]
    # the bar's own volume, which the window before it leaves out
    bars.append([Bar(21_012.0, 21_080.0, 21_000.0, 21_050.0, 30_000.0)])
    venue = Venue(
        taker    = 0.0005,
        spread   = ("fixed", 0.6),
        slippage = 1.0,
        impact   = ("sqrt", 0.1),
    )
    sim = run(hourly(len(bars)), bars, [Instrument("BTCUSDT")], venue,
              script={24: [("buy", dict(instrument=0, size=0.141))]})
    [fill] = sim.fills

    assert fill["spread"]   == pytest.approx(0.63, abs=0.005)
    assert fill["slippage"] == pytest.approx(2.10, abs=0.005)
    assert fill["impact"]   == pytest.approx(0.02, abs=0.005)
    assert fill["price"]    == pytest.approx(21_014.75, abs=0.005)

    moved = 0.6e-4 / 2.0 + 1e-4 + 0.1 * 0.004 * math.sqrt(0.141 / 24_800.0)
    assert fill["price"] == pytest.approx(21_012.0 * (1.0 + moved), rel=1e-12)
    assert fill["size"] * fill["price"] == pytest.approx(2_963.08, abs=0.005)
    assert fill["fee"] == pytest.approx(1.48, abs=0.005)


def test_a_budget_of_a_tenth_of_50_lets_five_orders_of_2_fill_2_2_and_1():
    # the bar they fill on trades 80 of its own, which the window before it leaves out
    bars = [[flat(100.0, volume=50.0)] for _ in range(25)]
    bars.append([flat(100.0, volume=80.0)])
    script = {24: [("buy", dict(instrument=0, size=2.0)) for _ in range(5)]}
    sim    = run(hourly(len(bars)), bars, [Instrument("BTCUSDT")], Venue(budget=0.1),
                 script=script)

    assert [f["size"] for f in sim.fills] == pytest.approx([2.0, 2.0, 1.0])
    assert [e["kind"] for e in sim.events if e["order"] in (4, 5)] == [
        "placed", "placed", "not_filled", "not_filled"]

    held = [e["reason"] for e in sim.events if e["kind"] in ("clamped", "not_filled")]
    assert held == ["the liquidity budget allows 1.0 of the 2.0 asked"] + [
        "nothing could fill: the liquidity budget allows none"] * 2


def test_six_bars_estimate_a_spread_of_0_1027_percent():
    six = [Bar(100.0, 100.5, 99.8, 100.2, 1_000.0),
           Bar(100.3, 100.6, 100.0, 100.0, 1_000.0),
           Bar(100.1, 100.6, 99.9, 100.4, 1_000.0),
           Bar(100.5, 100.8, 100.1, 100.1, 1_000.0),
           Bar(100.2, 100.7, 100.0, 100.5, 1_000.0),
           Bar(100.6, 100.9, 100.3, 100.3, 1_000.0)]

    assert edge(six) ** 2 == pytest.approx(1.0553e-6, rel=1e-4)
    assert edge(six)      == pytest.approx(0.001027, abs=5e-7)

    bars = [[bar] for bar in six] + [[Bar(100.40, 100.6, 100.3, 100.5, 1_000.0)]]
    sim  = run(hourly(len(bars)), bars, [Instrument("BTCUSDT")],
               Venue(spread="estimated", window=6),
               script={5: [("buy", dict(instrument=0, size=1.0))]})

    assert sim.fills[0]["spread"] == pytest.approx(0.05, abs=0.005)


def test_with_sizes_on_a_budget_cut_fill_rounds_down_to_the_lot():
    # 0.0756 and 0.0006 round down where the nearest lot would round them up
    bars  = [[flat(100.0, volume=0.0756)]] + [[Bar(100.0, 100.5, 98.5, 99.5, 0.0756)]]
    venue = Venue(sizes=True)
    sim   = run(hourly(2), bars, [Instrument("BTCUSDT", lot=0.001)], venue,
                script={0: [("buy", dict(instrument=0, size=0.2, limit=99.0)),
                            ("buy", dict(instrument=0, size=0.0006))]})
    [fill] = sim.fills

    assert fill["size"] == pytest.approx(0.075, abs=1e-12)

    resting = sim.orders[1]
    assert resting["status"] == "resting"
    assert resting["size"] - resting["filled"] == pytest.approx(0.125, abs=1e-12)
    assert sim.orders[2]["status"] == "refused"


@pytest.mark.parametrize("low, fills", [(99.98, False), (99.9, True)])
def test_five_figures_with_a_cap_of_one_decimal_rest_99_987_at_100_0(low, fills):
    # the cap leaves a step of 0.1 at 100.0, where five figures alone would keep
    # 99.987 and step by 0.001
    bars  = [[flat(101.0)], [Bar(101.0, 101.0, low, 100.5, 1_000.0)]]
    venue = Venue(sizes=True, limit="through")
    btc   = Instrument("BTC", sigfigs=5, decimals=1)
    sim   = run(hourly(2), bars, [btc], venue,
                script={0: [("buy", dict(instrument=0, size=1.0, limit=99.987)),
                            ("sell", dict(instrument=0, size=1.0, limit=123_456.7))]})

    assert (sim.orders[1]["price"], sim.orders[2]["price"]) == (100.0, 123_457.0)
    assert [fill["order"] for fill in sim.fills] == ([1] if fills else [])


@pytest.mark.parametrize("exempt", [False, True])
def test_a_minimum_of_10_refuses_0_05_at_150_and_a_maximum_of_1_0005_holds_5_to_1(exempt):
    inst = Instrument(
        "X",
        lot                = 0.001,
        min_notional       = 10.0,
        max_qty            = 1.0005,
        reduce_only_exempt = exempt,
    )
    sim = run(hourly(2), [[flat(150.0)], [flat(150.0)]], [inst], Venue(sizes=True),
              script={0: [("buy", dict(instrument=0, size=0.05)),
                          ("sell", dict(instrument=0, size=0.05, reduce_only=True)),
                          ("buy", dict(instrument=0, size=5.0))]})

    assert sim.orders[1]["status"] == "refused"
    assert (sim.orders[2]["status"] != "refused") is exempt
    assert sim.orders[3]["size"] == pytest.approx(1.0, abs=1e-12)

    # the hold is logged after the placement it holds
    held = [(event["kind"], event["size"]) for event in sim.events if event["order"] == 3]
    assert held == [("placed", 1.0), ("clamped", 1.0), ("filled", 1.0)]


def test_a_holding_of_0_0004_under_a_lot_of_0_001_is_swept_at_250_as_dust_of_0_1():
    bars = [[flat(150.0)], [Bar(200.0, 260.0, 190.0, 250.0, 1_000.0)]]
    sim  = Simulator(hourly(2), bars, [Instrument("X", kind="spot", lot=0.001)],
                     Venue(sizes=True))
    sim.position[0], sim.entry[0] = 0.0004, 100.0
    sim.step()

    assert sim.position[0]        == 0.0
    assert sim.ledger[-1]["dust"] == pytest.approx(0.1, abs=1e-12)


def costs_example():
    """The costs example's quantities, Data and venue: a tenth of a bitcoin bought and
    sold with a fixed spread and slippage of 1 bp, a taker fee of 0.05% and 0.01% of
    funding at every close, on bars at 60,000, then 60,000 to 60,500, then 60,400 to
    60,450."""
    frame = hourly_bars([60_000.0, 60_000.0, 60_400.0], [60_000.0, 60_500.0, 60_450.0],
                        range(3)).assign(high=[60_000.0, 60_600.0, 60_500.0],
                                         low=[60_000.0, 59_900.0, 60_300.0])
    venue = probatum.Venue(
        balance = 10_000.0,
        fees    = {"maker": 0.0, "taker": 0.0005},
        fills   = {"spread": {"fixed": 1.0}, "slippage": {"fixed": 1.0}},
        funding = {"every_bars": 1, "rate": 0.0001},
    )

    return (np.array([0.1, 0.0, np.nan]),
            probatum.Data(frame, symbol="BTCUSDT", market="linear"), venue)


def tenth(record="full"):
    """The costs example recorded at ``record``."""
    policy, data, venue = costs_example()
    return probatum.backtest(policy, data, venue, sizing="quantity", record=record)


def test_a_buy_and_a_sale_of_a_tenth_of_a_bitcoin_cost_8_430997():
    result = tenth()

    assert [fill["price"] for fill in result.fills] == pytest.approx([60_009.0,
                                                                      60_390.94])

    costs = result.costs
    assert costs == pytest.approx(dict(fees=6.019997, funding=0.605, spread=0.602,
                                       slippage=1.204, impact=0.0, liquidation=0.0,
                                       dust=0.0), abs=1e-9)
    assert math.fsum(costs.values()) == pytest.approx(8.430997, abs=1e-9)
    assert result.equity.tolist() == pytest.approx([10_000.0, 10_045.49455,
                                                    10_031.569003], abs=1e-9)
    # the price PnL, 40.00, less the costs is what the instrument made
    assert result.pnl[-1] == pytest.approx(40.0 - 8.430997, abs=1e-9)


def test_the_costs_example_at_the_account_level_prices_its_fees_and_not_its_bound():
    full, account = tenth(), tenth("account")

    assert (len(account.ledger), len(account.trades)) == (3, 1)
    assert account.stats == full.stats

    costs = [{row["name"]: row["cost"] for row in result.assumptions}
             for result in (full, account)]
    for cost in costs:
        assert cost["fees"] == pytest.approx(6.019997, abs=1e-9)
        assert cost["liquidation fee"] == 0.0
    assert [costs[0]["margin bound"], costs[0]["liquidation"]] == [0.0, 0.0]
    assert [costs[1]["margin bound"], costs[1]["liquidation"]] == [None, None]


def test_the_costs_round_trip_pays_13_bps_and_26_at_twice_and_breaks_even_at_65_4():
    policy, data, venue = costs_example()
    curve = probatum.cost_curve(policy, data, venue, scales=(1.0, 2.0, 5.03, 5.04),
                                sizing="quantity")

    assert [round(cost, 1) for cost in curve.cost_bps[:2]] == [13.0, 26.0]
    assert math.fsum([curve.fees[1], curve.spread[1], curve.slippage[1]]) == (
        pytest.approx(15.651988, abs=1e-9))
    assert curve.total_return_pct[2] > 0.0 > curve.total_return_pct[3]

    found = probatum.breakeven_bps(policy, data, venue, sizing="quantity")
    assert round(found, 1) == 65.4


def test_a_weight_of_1_at_twice_the_balance_buys_half_of_it_held_by_the_budget():
    # a trailing volume of 5 BTC under budget(0.1) holds a bar's fills to 0.5 BTC
    frame = hourly_bars([60_000.0] * 3, [60_000.0] * 3, range(3)).assign(volume=5.0)
    data  = probatum.Data(frame, symbol="BTCUSDT", market="linear")
    venue = probatum.Venue(
        fees  = {"maker": 0.0, "taker": 0.0},
        fills = {"liquidity": {"budget": 0.1}},
    )
    weights = np.array([1.0, np.nan, np.nan])
    held    = probatum.capacity(weights, data, venue, [30_000.0, 60_000.0]).held

    assert held == (0.0, 1.0)

    result = probatum.backtest(weights, data, venue.replace(balance=60_000.0))

    assert [fill["size"] for fill in result.fills] == [0.5]

    [clamp] = [event["reason"] for event in result.orders if event["kind"] == "clamped"]
    assert clamp.startswith("the liquidity budget allows 0.5 of the 1")


def test_a_bar_that_carries_half_a_bitcoin_and_buys_a_tenth_gains_266_0815():
    bars = [[flat(60_000.0, volume=1e6)], [flat(60_000.0, volume=1e6)],
            [Bar(60_300.0, 60_600.0, 60_250.0, 60_500.0, 1e6)]]
    venue = Venue(
        taker    = 0.0005,
        spread   = ("fixed", 6.0 / 60_300.0 * 10_000.0),
        slippage = 1.0,
    )
    sim = run(hourly(3), bars, [Instrument("BTCUSDT")], venue,
              script={0: [("buy", dict(instrument=0, size=0.5))],
                      1: [("buy", dict(instrument=0, size=0.1))]})
    fill = sim.fills[-1]

    assert fill["price"] == pytest.approx(60_309.03, abs=0.005)

    row = sim.ledger[2]
    assert row["spread"]   == pytest.approx(0.30, abs=1e-9)
    assert row["slippage"] == pytest.approx(0.603, abs=1e-9)
    assert row["fees"]     == pytest.approx(3.0155, abs=5e-5)
    assert row["equity"] - sim.ledger[1]["equity"] == pytest.approx(266.0815, abs=5e-5)


def test_a_stamp_three_milliseconds_after_8_is_charged_on_the_bar_that_ends_at_8():
    sim = run(hourly(10), [[flat(60_000.0)] for _ in range(10)],
              [Instrument("BTCUSDT")], Venue(funding="data"),
              script={0: [("buy", dict(instrument=0, size=0.5))]},
              funding=[(8 * HOUR + 3_000_000, 0, 0.0001)])

    assert [round(row["funding"], 2) for row in sim.ledger] == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]


@pytest.mark.parametrize("source, rows", [
    (("every", 8, 0.0001), [7, 15, 23]),
    (("every_bars", 3, 0.0001), [3, 6, 9, 12, 15, 18, 21, 24]),
])
def test_each_source_stamps_its_events_where_the_page_says(source, rows):
    sim = run(hourly(25), [[flat(100.0)] for _ in range(25)],
              [Instrument("BTCUSDT")], Venue(funding=source),
              script={0: [("buy", dict(instrument=0, size=1.0))]})

    assert [t for t, row in enumerate(sim.ledger) if row["funding"] != 0.0] == rows


@pytest.mark.parametrize("stopped, paid", [(True, 0.0), (False, 0.0001 * 101.0)])
def test_an_event_inside_a_daily_bar_is_paid_by_the_position_after_its_fills(
        stopped, paid):
    decisions = [("buy", dict(instrument=0, size=1.0))]
    if stopped:
        decisions.append(("sell", dict(instrument=0, size=1.0, stop=95.0)))

    bars = [[flat(100.0)], [Bar(100.0, 102.0, 94.0, 101.0, 1_000.0)], [flat(101.0)]]
    sim  = run([0, DAY, 2 * DAY], bars, [Instrument("BTCUSDT")],
               Venue(maker=0.0, taker=0.0, funding="data"), script={0: decisions},
               funding=[(DAY + 8 * HOUR, 0, 0.0001)])

    assert sim.ledger[1]["funding"] == pytest.approx(paid, abs=1e-12)


def test_an_event_during_a_closure_is_charged_at_the_next_open_at_the_last_mark():
    # 1 January 1970 was a Thursday, so these rows are Thursday, Friday and Monday
    bars = [[flat(60_000.0)], [Bar(60_000.0, 60_000.0, 41_000.0, 41_000.0, 1_000.0)],
            [flat(41_000.0)]]
    sim = run([0, DAY, 4 * DAY], bars, [Instrument("BTCUSDT")],
              Venue(maker=0.0, taker=0.0, funding="data"), interval=DAY,
              script={0: [("buy", dict(instrument=0, size=0.5))]},
              funding=[(2 * DAY + 8 * HOUR, 0, 0.0005)])

    assert sim.ledger[1]["funding"] == 0.0
    assert sim.ledger[1]["equity"]  == pytest.approx(500.0)
    assert sim.ledger[2]["funding"] == pytest.approx(10.25)


def seven_oclock(rate, stamp=8 * HOUR, intrabar="pessimistic", liquidation="takeover"):
    return half_a_bitcoin([Bar(60_000.0, 60_000.0, 40_210.0, 41_000.0, 1_000.0)],
                          tiers=MAINTAINED, times=[5 * HOUR, 6 * HOUR, 7 * HOUR],
                          events=[(stamp, 0, rate)], funding="data", intrabar=intrabar,
                          liquidation=liquidation)


def test_a_debit_charged_after_the_check_leaves_489_75_against_102_50():
    sim = seven_oclock(0.0005)
    row = sim.ledger[-1]

    assert row["funding"] == pytest.approx(10.25)
    assert row["equity"]  == pytest.approx(489.75)
    assert sim.maintenance_of(0, 0.5, 20_500.0) == pytest.approx(102.50)
    assert row["liquidation"] == 0.0


@pytest.mark.parametrize("liquidation, fee, kept, charge", [
    ("takeover", 0.0, 0.0, 90.00),
    (("book_then_backstop", 2.0 / 3.0), 10.25, 79.75, 0.0),
])
def test_a_debit_of_410_closes_the_account_at_the_close_charging_90_or_keeping_79_75(
        liquidation, fee, kept, charge):
    sim   = seven_oclock(0.02, liquidation=liquidation)
    close = sim.fills[-1]

    assert (close["reason"], close["price"]) == ("liquidation", 41_000.0)
    assert close["fee"]                  == pytest.approx(fee)
    assert sim.ledger[-1]["funding"]     == pytest.approx(410.0)
    assert sim.ledger[-1]["liquidation"] == pytest.approx(charge)
    assert sim.margin_cash               == pytest.approx(kept)


@pytest.mark.parametrize("intrabar, liquidated", [("pessimistic", True),
                                                  ("optimistic", False)])
def test_a_debit_inside_the_bar_counts_toward_its_check_only_when_pessimistic(
        intrabar, liquidated):
    sim = seven_oclock(0.0005, stamp=7 * HOUR + 30 * 60 * 10**9, intrabar=intrabar)
    row = sim.ledger[-1]

    assert row["funding"] == pytest.approx(10.25)
    if liquidated:
        assert sim.fills[-1]["price"] == pytest.approx(40_221.61, abs=0.005)
        assert row["liquidation"]     == pytest.approx(100.55, abs=0.005)
    else:
        assert sim.fills[-1]["reason"] == "limit"
        assert row["equity"] == pytest.approx(489.75)


def test_a_debit_inside_the_bar_clips_it_above_a_stop_at_40_215_which_never_fills():
    stop = dict(instrument=0, size=0.5, stop=40_215.0, reduce_only=True)
    sim  = half_a_bitcoin([Bar(60_000.0, 60_000.0, 40_210.0, 41_000.0, 1_000.0)],
                          script={1: [("sell", stop)]}, tiers=MAINTAINED,
                          times=[5 * HOUR, 6 * HOUR, 7 * HOUR],
                          events=[(7 * HOUR + 30 * 60 * 10**9, 0, 0.0005)], funding="data")

    assert [fill["reason"] for fill in sim.fills] == ["limit", "liquidation"]
    assert sim.fills[-1]["price"] == pytest.approx(40_221.61, abs=0.005)


def test_a_debit_over_a_weekend_moves_the_trigger_onto_mondays_gap():
    # 7 January 1970 was a Wednesday, so the rows run Wednesday to Friday and Monday
    sim = half_a_bitcoin([Bar(60_000.0, 60_000.0, 41_000.0, 41_000.0, 1_000.0),
                          Bar(40_150.0, 40_600.0, 40_000.0, 40_400.0, 1_000.0)],
                         tiers=MAINTAINED, times=[6 * DAY, 7 * DAY, 8 * DAY, 11 * DAY],
                         interval=DAY, events=[(9 * DAY + 8 * HOUR, 0, 0.0005)],
                         funding="data")

    assert sim.ledger[2]["equity"] == pytest.approx(500.0)

    close = sim.fills[-1]
    assert (close["bar"], close["reason"]) == (3, "liquidation")
    assert close["price"]               == pytest.approx(40_221.61, abs=0.005)
    assert sim.ledger[3]["funding"]     == pytest.approx(10.25)
    assert sim.ledger[3]["liquidation"] == pytest.approx(100.55, abs=0.005)


def test_twenty_times_ten_thousand_lets_a_buy_of_5_bitcoin_fill_3_33333():
    sim = run(hourly(2), [[flat(60_000.0)], [flat(60_000.0)]], [Instrument("BTCUSDT")],
              Venue(taker=0.0005, leverage=20.0),
              script={0: [("buy", dict(instrument=0, size=5.0))]})

    assert sim.fills[0]["size"] == pytest.approx(3.33333, abs=5e-6)
    assert [e["kind"] for e in sim.events] == ["placed", "clamped", "filled", "expired"]

    size = sim.fills[0]["size"]
    assert sim.events[1]["reason"] == f"the margin cap allows {size} of the 5.0 asked"


def test_a_reduce_only_sale_of_3_on_a_long_of_1_fills_1_and_on_nothing_fills_none():
    bars  = [[flat(60_000.0)] for _ in range(4)]
    venue = Venue(balance=10_000.0, maker=0.0, taker=0.0, leverage=20.0)
    sim   = run(hourly(4), bars, [Instrument("BTCUSDT")], venue,
                script={0: [("buy", dict(instrument=0, size=1.0))],
                        1: [("sell", dict(instrument=0, size=3.0, reduce_only=True))],
                        2: [("sell", dict(instrument=0, size=3.0, reduce_only=True))]})

    held = [(e["order"], e["reason"]) for e in sim.events
            if e["kind"] in ("clamped", "not_filled")]
    assert held == [(2, "reduce_only allows 1.0 of the 3.0 asked"),
                    (3, "nothing could fill: reduce_only allows none")]

    # without reduce_only the sale flips the account short by 2, which the cap allows
    flip = run(hourly(3), bars[:3], [Instrument("BTCUSDT")], venue,
               script={0: [("buy", dict(instrument=0, size=1.0))],
                       1: [("sell", dict(instrument=0, size=3.0))]})

    assert flip.position[0] == -2.0


def bitcoin_and_ether(row_two, decisions, ether=None, **venue):
    """Two perpetuals in one margin wallet, BTCUSDT at 20x and ETHUSDT at 10x with no
    maintenance, on two rows at 60,000 and 3,000 and then ``row_two``, ETHUSDT at its
    own taker rate ``ether`` where given."""
    bars     = [[flat(60_000.0), flat(3_000.0)], [flat(60_000.0), flat(3_000.0)], row_two]
    settings = dict(balance=10_000.0, maker=0.0, taker=0.0)
    settings.update(venue)

    return run(hourly(3), bars,
               [Instrument("BTCUSDT", tiers=[(1e12, 20.0, 0.0, 0.0)]),
                Instrument("ETHUSDT", tiers=[(1e12, 10.0, 0.0, 0.0)], taker=ether)],
               Venue(**settings), script=decisions)


def test_a_long_bitcoin_leaves_room_for_23_333_ether_sold_at_the_open():
    sim = bitcoin_and_ether([flat(60_000.0), flat(3_000.0)],
                            {0: [("buy", dict(instrument=0, size=1.0))],
                             1: [("sell", dict(instrument=1, size=30.0))]})

    assert sim.fills[-1]["size"] == pytest.approx(23.333, abs=5e-4)
    assert [e["kind"] for e in sim.events if e["order"] == 2] == [
        "placed", "clamped", "filled", "expired"]


def test_ether_sold_at_its_high_marks_bitcoin_at_its_low_and_fits_3_94():
    row_two = [Bar(60_000.0, 60_000.0, 54_000.0, 55_000.0, 1_000.0),
               Bar(3_000.0, 3_300.0, 3_000.0, 3_200.0, 1_000.0)]

    sim = bitcoin_and_ether(row_two,
                            {0: [("buy", dict(instrument=0, size=1.0))],
                             1: [("sell", dict(instrument=1, size=30.0,
                                               limit=3_300.0))]})

    sale = sim.fills[-1]
    assert sale["price"] == 3_300.0
    assert sale["size"]  == pytest.approx(13_000.0 / 3_300.0, rel=1e-9)
    assert sale["size"]  == pytest.approx(3.94, abs=0.005)


def maintained_pair(row_two, **venue):
    """A long of 1 BTC from 60,000 and of 20 ETH from 3,000 on 10,000 USDT of margin,
    both at 20x, BTC keeping 0.5% maintenance and ETH 1%, with no fees, and then
    ``row_two``."""
    bars = [[flat(60_000.0), flat(3_000.0)], [flat(60_000.0), flat(3_000.0)], row_two]
    return run(hourly(3), bars,
               [Instrument("BTCUSDT", tiers=[(1e12, 20.0, 0.005, 0.0)]),
                Instrument("ETHUSDT", tiers=[(1e12, 20.0, 0.01, 0.0)])],
               Venue(balance=10_000.0, maker=0.0, taker=0.0, **venue),
               script={0: [("buy", dict(instrument=0, size=1.0)),
                           ("buy", dict(instrument=1, size=20.0))]})


def test_a_takeover_gives_each_instrument_its_maintenance_of_the_831_234_left():
    sim = maintained_pair([Bar(60_000.0, 60_000.0, 54_000.0, 54_000.0, 1_000.0),
                           Bar(3_000.0, 3_000.0, 2_700.0, 2_700.0, 1_000.0)])

    closes = [fill for fill in sim.fills if fill["reason"] == "liquidation"]
    assert [fill["price"] for fill in closes] == pytest.approx([55_415.62, 2_770.78],
                                                               abs=0.005)
    assert sim.ledger[2]["liquidation"] == pytest.approx(831.234, abs=5e-4)

    pnl = [row["pnl"] for row in sim.ledger[2]["instruments"]]
    assert pnl      == pytest.approx([-4_861.461, -5_138.539], abs=5e-4)
    assert sum(pnl) == pytest.approx(-10_000.0, abs=1e-6)


def test_a_debit_at_the_closes_below_the_requirement_gives_each_its_share_of_484():
    # the bars fall 7% and close at their lows, which the path never takes to the
    # trigger, and 1% of funding at the closes takes equity from 1,600 to 484 against
    # a requirement of 837
    sim = maintained_pair([Bar(60_000.0, 60_000.0, 55_800.0, 55_800.0, 1_000.0),
                           Bar(3_000.0, 3_000.0, 2_790.0, 2_790.0, 1_000.0)],
                          funding=("every_bars", 2, 0.01))

    assert sim.ledger[2]["funding"]     == pytest.approx(1_116.0)
    assert sim.ledger[2]["liquidation"] == pytest.approx(484.0)

    pnl = [row["pnl"] for row in sim.ledger[2]["instruments"]]
    assert pnl == pytest.approx([-4_200.0 - 558.0 - 161.333, -4_200.0 - 558.0 - 322.667],
                                abs=5e-4)


def test_binance_tiers_ask_1_200_either_side_of_300_000():
    tiers = [(300_000.0, 125.0, 0.004, 0.0), (800_000.0, 100.0, 0.005, 300.0)]
    sim   = run(hourly(1), [[flat(60_000.0)]], [Instrument("BTCUSDT", tiers=tiers)])

    assert sim.maintenance_of(0, 5.0, 300_000.0)  == pytest.approx(1_200.0)
    assert sim.maintenance_of(0, 5.0, 300_000.01) == pytest.approx(1_200.0, abs=1e-4)
    # inside the bound's billionth the notional is read in the second tier, as the
    # maintenance of a notional table reads its bounds exactly
    assert sim.maintenance_of(0, 5.0, 300_000.00015) == pytest.approx(
        0.005 * 300_000.00015 - 300.0, abs=1e-9)

    broken = [(300_000.0, 125.0, 0.004, 0.0), (800_000.0, 100.0, 0.005, 250.0)]
    with pytest.raises(ValueError, match="continuity"):
        Instrument("BTCUSDT", tiers=broken)
    with pytest.raises(ValueError, match="initial rate"):
        Instrument("BTCUSDT", tiers=[(1e12, 125.0, 0.008, 0.0)])


def test_hyperliquids_open_btc_table_keeps_3_125_000_on_200_000_000():
    tiers = [(150_000_000.0, 40.0, 0.0125, 0.0), (math.inf, 20.0, 0.025, 1_875_000.0)]
    sim   = run(hourly(1), [[flat(60_000.0)]], [Instrument("BTCUSDC", tiers=tiers)])

    assert sim.maintenance_of(0, 1.0, 200_000_000.0) == pytest.approx(3_125_000.0)

    with pytest.raises(ValueError, match="not finite"):
        Instrument("BTCUSDC", tiers=[(math.inf, 40.0, 0.0125, 0.0),
                                     (2e8, 20.0, 0.025, 0.0)])


FALL = Bar(41_500.0, 41_500.0, 39_000.0, 39_500.0, 1_000.0)


def test_with_no_maintenance_the_account_closes_at_40_020_01_and_ends_at_zero():
    sim   = half_a_bitcoin([FALL])
    close = sim.fills[-1]

    assert close["reason"] == "liquidation"
    assert close["price"]  == pytest.approx(40_020.01, abs=0.005)
    assert close["fee"]    == pytest.approx(10.005, abs=5e-4)
    assert sim.ledger[-1]["liquidation"] == pytest.approx(0.0, abs=1e-6)
    assert sim.margin_cash == 0.0


def test_a_stop_at_41_000_fills_on_the_clipped_bar_and_keeps_489_75():
    sim = half_a_bitcoin([FALL], {1: [("sell", dict(instrument=0, size=0.5,
                                                    stop=41_000.0))]})

    assert [(f["reason"], f["price"]) for f in sim.fills] == [("limit", 60_000.0),
                                                              ("stop", 41_000.0)]
    assert sim.margin_cash == pytest.approx(489.75)


@pytest.mark.parametrize("intrabar, last, cash", [("pessimistic", "liquidation", 0.0),
                                                  ("optimistic", "limit", 995.80)])
def test_a_limit_at_42_000_is_held_back_on_a_liquidating_bar_unless_optimistic(
        intrabar, last, cash):
    wide = Bar(41_500.0, 42_500.0, 39_000.0, 39_500.0, 1_000.0)
    sim  = half_a_bitcoin([wide], {1: [("sell", dict(instrument=0, size=0.5,
                                                     limit=42_000.0))]},
                          intrabar=intrabar)

    assert sim.fills[-1]["reason"] == last
    assert sim.margin_cash         == pytest.approx(cash)
    if last == "liquidation":
        assert sim.fills[-1]["price"]  == pytest.approx(40_020.01, abs=0.005)
        assert sim.orders[2]["status"] == "cancelled"


def test_a_gap_through_the_clip_point_liquidates_the_account_before_the_bar_opens():
    sim = half_a_bitcoin([Bar(60_000.0, 60_000.0, 40_500.0, 40_500.0, 1_000.0),
                          Bar(40_010.0, 40_600.0, 39_000.0, 39_500.0, 1_000.0)],
                         {2: [("sell", dict(instrument=0, size=0.5, stop=40_300.0))]})

    assert [(f["bar"], f["reason"]) for f in sim.fills] == [(1, "limit"),
                                                            (3, "liquidation")]
    assert sim.fills[-1]["price"]  == pytest.approx(40_020.01, abs=0.005)
    assert sim.orders[2]["status"] == "cancelled"


SLIDE = Bar(41_000.0, 41_200.0, 40_150.0, 40_500.0, 1_000.0)


def test_takeover_at_half_a_percent_maintenance_charges_the_100_50_left():
    sim   = half_a_bitcoin([SLIDE], tiers=MAINTAINED)
    close = sim.fills[-1]

    assert close["reason"] == "liquidation"
    assert close["price"]  == pytest.approx(40_201.01, abs=0.005)
    assert close["fee"]    == 0.0
    assert sim.ledger[-1]["liquidation"] == pytest.approx(100.50, abs=0.005)
    assert sim.margin_cash == 0.0


@pytest.mark.parametrize("low, slippage, kept, charge", [
    (40_150.0, 0.0, 90.45, 0.0),
    (40_140.0, 0.0, 90.45, 0.0),
    (40_130.0, 0.0, 0.0, 100.50),
    (40_150.0, 10.0, 70.36, 0.0),
    (40_150.0, 20.0, 0.0, 100.50),
    (40_170.0, 20.0, 74.96, 0.0),
])
def test_book_then_backstop_sells_at_the_trigger_unless_it_would_reach_the_backstop(
        low, slippage, kept, charge):
    bar = Bar(41_000.0, 41_200.0, low, 40_500.0, 1_000.0)
    sim = half_a_bitcoin([bar], tiers=MAINTAINED, slippage=slippage,
                         liquidation=("book_then_backstop", 2.0 / 3.0))

    assert sim.fills[-1]["reason"]       == "liquidation"
    assert sim.margin_cash               == pytest.approx(kept, abs=0.005)
    assert sim.ledger[-1]["liquidation"] == pytest.approx(charge, abs=0.005)


@pytest.mark.parametrize("intrabar, kept, charge", [("pessimistic", 0.0, 100.50),
                                                    ("optimistic", 90.45, 0.0)])
def test_a_low_past_the_backstop_level_takes_the_account_over_unless_optimistic(
        intrabar, kept, charge):
    bar = Bar(41_000.0, 41_200.0, 40_130.0, 40_500.0, 1_000.0)
    sim = half_a_bitcoin([bar], tiers=MAINTAINED, intrabar=intrabar,
                         liquidation=("book_then_backstop", 2.0 / 3.0))

    assert sim.reached["a sale on the book and its backstop level both reached"]
    assert sim.margin_cash == pytest.approx(kept, abs=0.005)
    assert sim.ledger[-1]["liquidation"] == pytest.approx(charge, abs=0.005)


LONG_BITCOIN_SHORT_ETHER = {0: [("buy", dict(instrument=0, size=1.0)),
                                ("sell", dict(instrument=1, size=20.0))]}


def falling_bitcoin_rising_ether(closes):
    return [Bar(60_000.0, 60_000.0, 54_000.0, closes[0], 1_000.0),
            Bar(3_000.0, 3_300.0, 3_000.0, closes[1], 1_000.0)]


def test_a_long_bitcoin_and_short_ether_are_both_closed_at_the_close_point():
    sim = bitcoin_and_ether(falling_bitcoin_rising_ether((57_000.0, 3_150.0)),
                            LONG_BITCOIN_SHORT_ETHER, balance=10_060.0, taker=0.0005)

    assert sim.ledger[1]["equity"] == pytest.approx(10_000.0)

    closes = [(f["instrument"], f["price"]) for f in sim.fills
              if f["reason"] == "liquidation"]
    assert [i for i, _ in closes] == [0, 1]
    assert closes[0][1]           == pytest.approx(55_030.0, abs=0.005)
    assert closes[1][1]           == pytest.approx(3_248.50, abs=0.005)
    assert sim.ledger[-1]["fees"] == pytest.approx(60.0, abs=0.005)
    assert sim.margin_cash        == 0.0


def test_an_ether_short_at_its_own_taker_rate_moves_the_close_point_to_0_8256():
    sim = bitcoin_and_ether(falling_bitcoin_rising_ether((57_000.0, 3_150.0)),
                            LONG_BITCOIN_SHORT_ETHER, ether=0.001, balance=10_090.0,
                            taker=0.0005)

    assert sim.ledger[1]["equity"] == pytest.approx(10_000.0)

    prices = [f["price"] for f in sim.fills if f["reason"] == "liquidation"]
    assert prices                 == pytest.approx([55_046.24, 3_247.69], abs=0.005)
    assert sim.ledger[-1]["fees"] == pytest.approx(92.48, abs=0.005)
    assert sim.margin_cash        == 0.0


@pytest.mark.parametrize("closes, liquidated", [((57_000.0, 3_150.0), False),
                                                ((55_000.0, 3_250.0), True)])
def test_the_optimistic_setting_liquidates_only_when_every_path_reaches_the_trigger(
        closes, liquidated):
    sim = bitcoin_and_ether(falling_bitcoin_rising_ether(closes),
                            LONG_BITCOIN_SHORT_ETHER, balance=10_060.0, taker=0.0005,
                            intrabar="optimistic")

    assert (sim.liquidations == 1) is liquidated
    if liquidated:
        prices = [f["price"] for f in sim.fills if f["reason"] == "liquidation"]
        assert prices == pytest.approx([55_030.0, 3_248.50], abs=0.005)
    else:
        assert sim.ledger[-1]["equity"] == pytest.approx(4_000.0)


def test_a_sale_past_the_close_point_fills_at_95_and_books_0_45_back():
    sim = run(hourly(3), [[flat(100.0)], [flat(100.0)], [flat(95.0)]],
              [Instrument("XYZUSDT")],
              Venue(balance=51.5, maker=0.0, taker=0.001, leverage=50.0),
              script={0: [("buy", dict(instrument=0, size=10.0))],
                      1: [("close", dict(instrument=0))]})

    assert sim.ledger[1]["margin_cash"] == pytest.approx(50.5)

    sale = sim.fills[-1]
    assert sale["price"]                 == 95.0
    assert sim.ledger[-1]["liquidation"] == pytest.approx(-0.45, abs=0.005)
    assert sim.margin_cash               == pytest.approx(0.0, abs=1e-9)


SHORT_ONE_COIN = {0: [("sell", dict(instrument=0, size=600.0))]}


def test_a_coin_short_its_own_value_is_worth_60_000_dollars_at_every_price():
    later = [flat(30_000.0), flat(120_000.0),
             Bar(120_000.0, 600_000.0, 110_000.0, 500_000.0, 1_000.0)]

    sim = btcusd(later, SHORT_ONE_COIN, balance=1.0)

    for row in sim.ledger[2:]:
        close = row["instruments"][0]["mark"]
        assert row["equity"] == pytest.approx(60_000.0 / close, rel=1e-12)
        assert row["equity_in_quote"] == pytest.approx(60_000.0, rel=1e-12)
        assert sim.maintenance_of(0, 600.0, 60_000.0 / close) == pytest.approx(
            300.0 / close, rel=1e-12)

    assert sim.liquidations == 0


def test_a_coin_short_its_own_value_collects_0_0001_bitcoin_a_funding_event():
    sim = btcusd([flat(60_000.0)], SHORT_ONE_COIN, balance=1.0, funding="data",
                 events=[(3 * HOUR, 0, 0.0001)])

    assert sim.ledger[2]["funding"] == pytest.approx(-0.0001, rel=1e-12)
    assert sim.margin_cash == pytest.approx(1.0001, rel=1e-12)


def test_half_a_coin_long_600_contracts_from_60_000_is_liquidated_at_40_200():
    sim = btcusd([Bar(41_000.0, 41_000.0, 40_000.0, 40_500.0, 1_000.0)],
                 {0: [("buy", dict(instrument=0, size=600.0))]}, balance=0.5)

    close = sim.fills[-1]
    assert close["reason"]               == "liquidation"
    assert close["price"]                == pytest.approx(40_200.0, abs=0.005)
    assert sim.ledger[-1]["liquidation"] == pytest.approx(0.0074627, abs=5e-8)
    assert 300.0 / 40_200.0              == pytest.approx(0.0074627, abs=5e-8)


def test_two_buys_of_300_contracts_average_48_000_and_gain_0_05_at_50_000():
    later = [flat(40_000.0), flat(50_000.0)]
    sim   = btcusd(later, {
        0: [("buy", dict(instrument=0, size=300.0))],
        1: [("buy", dict(instrument=0, size=300.0))],
    }, balance=1.0)

    row = sim.ledger[-1]["instruments"][0]
    assert row["entry"] == pytest.approx(48_000.0, rel=1e-12)
    assert row["upnl"]  == pytest.approx(0.05, rel=1e-12)

    lots = [30_000.0 * (1.0 / f["price"] - 1.0 / 50_000.0) for f in sim.fills]
    assert lots == pytest.approx([-0.10, 0.15], rel=1e-12)


def test_a_value_order_buys_60_contracts_and_a_fee_is_paid_in_the_coin():
    sim = btcusd([flat(50_000.0)], {0: [("buy", dict(instrument=0, size=600.0))],
                                    1: [("buy_value", dict(instrument=0,
                                                           amount=6_000.0))]},
                 balance=1.0, taker=0.0005)

    first, second = sim.fills
    assert first["fee"]   == pytest.approx(0.0005, rel=1e-12)
    assert second["size"] == pytest.approx(60.0, rel=1e-12)


def walked(opens, closes, volume=1e9):
    """Bars that open and close where given, their range the two."""
    return [Bar(o, max(o, c), min(o, c), c, volume) for o, c in zip(opens, closes)]


def played(bars, instruments, venue, rows):
    """The equity on each row of ``rows`` of quantities played as an array is, which
    is how buying and holding asks for its shares."""
    sim = play_array(Simulator(hourly(len(bars)), bars, instruments, venue), rows,
                     sizing="quantity")
    return sim, [row["equity"] for row in sim.ledger]


def test_the_costs_example_bought_and_held_reads_10_075_82_and_without_costs_10_040():
    # a sixth of a bitcoin bought at the close of 60,000 and held, and the run's
    # quantities of 0.1, 0 and NaN on the venue with its costs removed
    bars = [[bar] for bar in (Bar(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e9),
                              Bar(60_000.0, 60_600.0, 59_900.0, 60_500.0, 1e9),
                              Bar(60_400.0, 60_500.0, 60_300.0, 60_450.0, 1e9))]
    venue = Venue(
        maker    = 0.0,
        taker    = 0.0005,
        spread   = ("fixed", 1.0),
        slippage = 1.0,
        funding  = ("every_bars", 1, 0.0001),
    )

    sim, equity = played(bars, [Instrument("BTCUSDT")], venue,
                         [[10_000.0 / 60_000.0]] * 3)

    assert sim.fills[0]["price"] == pytest.approx(60_009.0, abs=1e-9)
    assert sim.fills[0]["fee"]   == pytest.approx(5.00075, abs=1e-9)
    assert [row["funding"] for row in sim.ledger[1:]] == pytest.approx(
        [1.008333, 1.0075], abs=5e-7)
    assert equity == pytest.approx([10_000.0, 10_075.82, 10_066.48], abs=0.005)

    _, equity = played(bars, [Instrument("BTCUSDT")], Venue(maker=0.0, taker=0.0),
                       [[0.1], [0.0], [math.nan]])
    assert equity == pytest.approx([10_000.0, 10_050.0, 10_040.0], abs=1e-9)


@pytest.mark.parametrize("kind, equity", [("linear", [10_000.0, 10_289.90, 9_789.90]),
                                          ("spot", [10_000.0, 10_286.74, 9_792.19])])
def test_buying_and_holding_100_at_101_reads_9_789_90_and_on_spot_98_910990_of_it(
        kind, equity):
    bars = [[bar] for bar in walked([100.0, 101.0, 103.0], [100.0, 104.0, 99.0])]

    sim, held = played(bars, [Instrument("BTCUSDT", kind=kind)],
                       Venue(maker=0.0, taker=0.001), [[100.0]] * 3)

    assert held == pytest.approx(equity, abs=0.005)
    # on spot the rest of the 100 is asked again at the next close and fills nothing
    assert [fill["size"] for fill in sim.fills] == pytest.approx(
        [100.0 if kind == "linear" else 98.910990], abs=5e-7)


def test_a_late_lister_bought_on_the_opening_balance_s_share_reads_10_150():
    a = walked([100.0, 101.0, 103.0, 98.0], [100.0, 104.0, 99.0, 102.0])
    b = [None, None] + walked([50.0, 51.0], [50.0, 52.0])

    # A's share at its first close, and B's at its own, on row 2
    rows = [[50.0, math.nan], [50.0, math.nan], [50.0, 100.0], [50.0, 100.0]]
    pair = [Instrument("A"), Instrument("B")]

    _, equity = played([list(row) for row in zip(a, b)], pair,
                       Venue(maker=0.0, taker=0.0), rows)

    assert equity == pytest.approx([10_000.0, 10_150.0, 9_900.0, 10_150.0], abs=1e-9)


@pytest.mark.parametrize("liquidation, kept", [
    ("takeover", 0.0), (("book_then_backstop", 2.0 / 3.0), 100.50)])
def test_without_costs_the_long_is_closed_at_40_201_01_and_on_the_book_keeps_100_50(
        liquidation, kept):
    # the run without costs carries the same long into the bar on the same 10,000
    sim = half_a_bitcoin([SLIDE], tiers=MAINTAINED, balance=10_000.0, maker=0.0,
                         taker=0.0, liquidation=liquidation)

    assert sim.fills[-1]["reason"] == "liquidation"
    assert sim.fills[-1]["price"]  == pytest.approx(40_201.01, abs=0.005)
    assert sim.margin_cash         == pytest.approx(kept, abs=0.005)


def test_one_coin_bought_and_held_as_600_contracts_is_taken_over_at_30_120():
    bars = [[flat(60_000.0)], [flat(60_000.0)],
            [Bar(60_000.0, 60_000.0, 30_000.0, 31_000.0, 1_000.0)], [flat(31_000.0)]]
    inverse = Instrument(
        "BTCUSD",
        kind     = "inverse",
        contract = 100.0,
        tiers    = [(1e12, 20.0, 0.004, 0.0)],
    )

    sim, equity = played(bars, [inverse], Venue(balance=1.0, maker=0.0, taker=0.0),
                         [[1.0 * 60_000.0 / 100.0]] * 4)

    close = sim.fills[-1]
    assert close["reason"] == "liquidation"
    assert close["price"]  == pytest.approx(30_120.0, abs=1e-6)
    assert equity          == pytest.approx([1.0, 1.0, 0.0, 0.0], abs=1e-12)


def hourly_bars(opens, closes, hours):
    """Bars that open and close where given, at the given hours of 1 January 2024."""
    opens, closes = np.array(opens, dtype=np.float64), np.array(closes, dtype=np.float64)
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=h)
                              for h in hours])

    return pd.DataFrame({
        "open":   opens,
        "high":   np.maximum(opens, closes),
        "low":    np.minimum(opens, closes),
        "close":  closes,
        "volume": 1e9,
    }, index=index)


def test_three_rows_of_two_instruments_record_six_holdings_and_null_what_they_lack():
    data = probatum.Data({
        "BTCUSDT": hourly_bars([100.0, 100.0, 101.0], [100.0, 101.0, 102.0], range(3)),
        "ETHUSDT": hourly_bars([10.0, 10.0, 10.0], [10.0, 10.0, 10.0], range(3)),
    },
        market="linear")

    class Long(probatum.Strategy):

        def next(self, bar, account):
            if bar.i == 0:
                account.target({"BTCUSDT": 0.5})

    result = probatum.backtest(Long(), data, "frictionless")

    assert len(result.ledger) == 3

    holdings = list(result.holdings)
    assert [(row["bar"], row["instrument"]) for row in holdings] == [
        (0, "BTCUSDT"), (0, "ETHUSDT"), (1, "BTCUSDT"), (1, "ETHUSDT"), (2, "BTCUSDT"),
        (2, "ETHUSDT")]
    assert [row["drift"] is None for row in holdings] == [True, True, False, True,
                                                         False, True]

    [fill] = list(result.fills)
    assert fill["bar"] == 1 and fill["order"] is not None

    [trip] = list(result.trades)
    assert trip["open"] and (trip["close_bar"], trip["close_time"], trip["exit"],
                             trip["reason"]) == (None, None, None, None)


def test_two_accounts_listed_and_ended_apart_add_into_15_350():
    venue = probatum.venues.frictionless
    first = probatum.Data(
        hourly_bars([100.0, 100.0, 101.0, 100.5],
                    [100.0, 101.0, 100.5, 102.0], range(4)),
        symbol = "COINUSDT",
        market = "linear",
    )
    second = probatum.Data(
        hourly_bars([50.0, 50.0, 51.5], [50.0, 49.0, 52.0], [1, 3, 4]),
        symbol   = "COINUSDT",
        market   = "linear",
        interval = "1h",
    )

    a = probatum.backtest(np.array([100.0, np.nan, np.nan, np.nan]), first,
                          venue.replace(balance=10_000.0), sizing="quantity")
    b = probatum.backtest(np.array([100.0, 0.0, np.nan]), second,
                          venue.replace(balance=5_000.0), sizing="quantity")

    assert a.equity.tolist() == [10_000.0, 10_100.0, 10_050.0, 10_200.0]
    assert b.equity.tolist() == [5_000.0, 4_900.0, 5_150.0]

    book = probatum.combine({"A": a, "B": b})
    assert book.equity.tolist() == [15_000.0, 15_100.0, 15_050.0, 15_100.0, 15_350.0]

    stats = book.stats
    assert stats["total_return_pct"] == pytest.approx(350.0 / 15_000.0 * 100.0)
    assert stats["exposure_pct"]     == 75.0
    assert stats["num_trades"] == 1 and stats["num_fills"] == 3
    assert stats["avg_trade_pct"]    == pytest.approx(1.0)
    assert stats["max_drawdown_pct"] == pytest.approx(50.0 / 15_100.0 * 100.0)

    fills = book.fills[book.fills["venue"] == "B"]
    assert fills["bar"].tolist() == [3, 4]

    # each account's instrument on the book's rows: carried inside its span, its
    # position and weight nothing outside it, and its pnl held after its end
    coin = book["A", "COINUSDT"]
    assert (coin.pnl.tolist(), coin.position.tolist(), coin.weight.tolist()) == (
        [0.0, 100.0, 50.0, 200.0, 200.0], [0.0, 100.0, 100.0, 100.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0])

    coin = book["B", "COINUSDT"]
    assert (coin.pnl.tolist(), coin.position.tolist(), coin.weight.tolist()) == (
        [0.0, 0.0, 0.0, -100.0, 150.0], [0.0, 0.0, 0.0, 100.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0])
    assert coin.fills["bar"].tolist() == [3, 4]

    with pytest.raises(ProbatumKeyError, match="COINUSDT is held at A and B"):
        book["COINUSDT"]

    assert (a.provenance["periods_per_year"], b.provenance["periods_per_year"]) == (
        8_760.0, 5_840.0)
    assert book.provenance["periods_per_year"] == 4 * 365 / (4 / 24) == 8_760.0


def test_lots_that_sum_a_hair_past_a_tier_s_bound_stay_in_the_tier_it_closes():
    tiers = [(100.0, 10.0, 0.005, 0.0), (200.0, 5.0, 0.01, 0.0),
             (math.inf, 3.0, 0.02, 0.0)]

    sim = Simulator(hourly(1), [[flat(100.0)]],
                    [Instrument("BTCUSDT", tiers=tiers, tiers_by="quantity")])

    held = 0.1 * 1071 + 0.1 * 818 + 0.1 * 111
    assert held == 200.00000000000003
    assert sim.leverage_of(0, held, held * 100.0)   == 5.0
    assert sim.leverage_of(0, 200.1, 200.1 * 100.0) == 3.0


def test_a_seam_cancels_a_stop_and_a_waiting_buy_and_a_flatten_closes_the_long():
    # window 1 decides to row 9 and window 2 from row 10, where ETH prints no bar
    bars = [[flat(100.0), flat(50.0)] for _ in range(13)]
    bars[10][1] = None

    instruments = [Instrument("BTCUSDT"), Instrument("ETHUSDT")]
    held        = {
        5: [("buy", dict(instrument=0, size=1.0))],
        8: [("sell", dict(instrument=0, size=1.0, stop=95.0))],
    }

    cancelled = run(hourly(13), bars, instruments, script={
        **held, 9: [("buy", dict(instrument=1, size=1.0))],
        10: [("seam", dict(flatten=False))]})

    seams = [(event["bar"], event["order"]) for event in cancelled.events
             if event["kind"] == "cancelled" and event["reason"] == "seam"]
    assert seams == [(10, 3), (10, 2)]
    assert cancelled.position == [1.0, 0.0]
    assert [fill["instrument"] for fill in cancelled.fills] == [0]

    flattened = run(hourly(13), bars, instruments, script={
        **held, 9: [("seam", dict(flatten=True))], 10: [("seam", dict(flatten=False))]})

    [close] = [fill for fill in flattened.fills if fill["bar"] == 10]
    assert (close["side"], close["size"], close["reason"]) == ("sell", 1.0, "seam")
    assert flattened.position == [0.0, 0.0]
    assert [(event["kind"], event["reason"]) for event in flattened.events
            if event["bar"] == 9] == [("cancelled", "seam"), ("placed", "seam")]


MONTHLY = [0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01]


def test_eight_monthly_returns_have_a_sharpe_of_1_73_whose_interval_is_floored_at_lag_0():
    assert statistics.sharpe(MONTHLY, 12) == pytest.approx(0.5 * math.sqrt(12.0))

    z = statistics.influence(MONTHLY)
    assert [round(14.0 * math.sqrt(3.5) * v, 9) for v in z] == [
        19.0, -29.0, 27.0, -9.0, 7.0, -53.0, 31.0, 7.0]

    g = statistics.autocovariances(z, 2)
    assert [round(v * 5_488.0, 9) for v in g]  == [5_880.0, -3_437.0, 1_286.0]
    assert round(statistics.bartlett(g, 2), 4) == 0.3926

    variance, lag, floored = statistics.floored(z)
    assert (round(variance, 4), lag, floored) == (1.0714, 2, True)

    low, high, lag, floored = statistics.sharpe_interval(MONTHLY, 12)
    assert (round(low, 2), round(high, 2), lag, floored) == (-0.92, 4.39, 2, True)


def test_their_probabilistic_sharpe_is_0_899_and_their_track_record_12_6_returns():
    assert round(statistics.probabilistic_sharpe(MONTHLY, 12), 3) == 0.899

    needed = statistics.min_track_record(MONTHLY, 12)
    assert (round(needed, 1), round(needed / 12, 2)) == (12.6, 1.05)
    assert round(statistics.probabilistic_sharpe(MONTHLY, 12, benchmark=1.0), 3) == 0.705

    needed = statistics.min_track_record(MONTHLY, 12, benchmark=1.0)
    assert (round(needed, 1), round(needed / 12, 2)) == (65.9, 5.49)


def test_their_long_run_sharpe_is_2_95_a_year_at_lag_1():
    centre  = statistics.mean(MONTHLY)
    centred = [v - centre for v in MONTHLY]

    assert [round(v * 1e4, 9) for v in statistics.autocovariances(centred, 2)] == [
        3.5, -2.125, 0.875]
    assert statistics.lag(centred) == 1
    assert round(statistics.long_run_sharpe(MONTHLY, 12), 2) == 2.95


def test_a_correlogram_with_no_lag_significant_reads_r_1_and_r_0_alone():
    assert round(2.0 * math.sqrt(3.0 / 1_000.0), 3) == 0.110
    assert round(statistics.stationary(0.1 / 1.1, 1_000), 2) == 2.02

    noise = np.random.Generator(np.random.PCG64(31)).standard_normal(1_000).tolist()
    rho   = [statistics.correlation(noise[37:], noise[37 - j:1_000 - j]) for j in range(1, 38)]
    assert max(abs(r) for r in rho) < 2.0 * math.sqrt(3.0 / 1_000.0)

    r      = [statistics.covariance(noise[2:], noise[2 - j:1_000 - j]) for j in range(3)]
    length = statistics.stationary(2.0 * r[1] / (r[0] + 2.0 * r[1]), 1_000)
    assert statistics.block_lengths(noise)[0] == max(1.0, min(length, 95))


def test_a_difference_of_0_05_a_bar_over_1000_normal_bars_has_a_p_value_of_0_115():
    variance = 2.0 - 2.0 * 0.5 + (0.01 + 0.0025 - 2.0 * 0.005 * 0.25) / 2.0
    error    = math.sqrt(variance / 999)

    assert (round(variance, 4), round(error, 4)) == (1.005, 0.0317)
    assert round(2.0 * statistics.NORMAL.cdf(-0.05 / error), 3) == 0.115
    assert round(statistics.bootstrap_p([1.0] * 113 + [0.0] * 1_887, 0.5), 3) == 0.057


def test_a_strategy_of_0_5_percent_and_0_8_of_its_benchmark_has_an_alpha_of_6_percent():
    benchmark = [-0.02, -0.01, 0.0, 0.01, 0.02]
    strategy  = [-0.01, -0.005, 0.005, 0.015, 0.02]
    found     = statistics.alpha_beta(strategy, benchmark, 12)

    assert (round(found["alpha"], 9), round(found["beta"], 9), found["lag"]) == (
        0.06, 0.8, 0)
    assert (round(found["alpha_error"] / 12 * 100, 3), round(found["beta_error"], 3)) == (
        0.063, 0.04)

    lagged = statistics.alpha_beta(strategy, benchmark, 12, given=1)
    assert (round(lagged["alpha_error"] / 12 * 100, 3), round(lagged["beta_error"], 3)) == (
        0.049, 0.028)
    assert (round(found["tracking_error"] * 100, 2),
            round(found["information_ratio"], 2)) == (1.22, 4.9)


def test_three_trips_of_20_and_minus_10_twice_end_at_9720_in_every_order():
    # each trip returns its pnl on the equity it opened on in the run's own order
    pnl, opened_on = [2_000.0, -1_200.0, -1_080.0], [10_000.0, 12_000.0, 10_800.0]

    for order in itertools.permutations(range(3)):
        ended = 10_000.0 * math.prod(1.0 + pnl[i] / opened_on[i] for i in order)
        fall, run = statistics.trade_path(pnl, opened_on, order)
        together = abs(order.index(1) - order.index(2)) == 1
        assert round(ended, 6)       == 9_720.0
        assert (round(fall, 6), run) == ((19.0, 2) if together else (10.0, 1))


def test_three_of_four_stretches_score_above_0_and_three_windows_decay_by_1_07():
    assert statistics.stretches(1, 1_000, 4) == [(1, 250), (251, 500), (501, 750),
                                                 (751, 1_000)]
    assert statistics.gained([0.81, 1.22, -0.30, 0.94], "sharpe", "maximize") == 3

    gaps, mean = statistics.decay([1.8, 1.2, 2.0], [0.6, 1.4, -0.2], "maximize")
    assert ([round(gap, 9) for gap in gaps], round(mean, 2)) == ([1.2, -0.2, 2.2], 1.07)


def test_four_trials_on_four_blocks_overfit_and_lose_in_four_of_six_halvings():
    blocks = [(0.4, 0.1, -0.2, 0.3), (-0.1, 0.5, 0.2, -0.3), (0.2, -0.4, 0.6, 0.1),
              (0.35, 0.45, 0.05, 0.15)]
    rows   = [[value for value in trial for _ in range(2)] for trial in blocks]

    assert round(math.log(3 / 5 / (2 / 5)), 2) == 0.41
    assert round(math.log(1 / 4), 2)           == -1.39
    assert statistics.overfitting(rows, statistics.mean, blocks=4) == pytest.approx(
        (4 / 6, 4 / 6))


def test_four_trials_on_four_bars_give_white_0_2_and_spa_0_6_on_five_resamples():
    f = [[1.875, -1.125, 2.875, -0.125], [1.0, 1.0, -2.0, 2.0],
         [-3.0, -2.0, -4.0, -3.0], [1.0, -1.0, 1.0, -2.0]]
    drawn = [[1, 2, 3, 0], [0, 0, 1, 2], [3, 0, 1, 1], [2, 2, 2, 3], [0, 1, 0, 1]]

    assert [statistics.kernel_variance(row, 0.5) for row in f] == [75 / 64, 11 / 8,
                                                                   19 / 64, 199 / 256]
    assert round(math.sqrt(2.0 * math.log(math.log(4.0))), 4) == 0.8083
    assert statistics.reality_check(f, drawn, 0.5) == pytest.approx((0.2, 0.6))


def test_three_trials_near_a_winner_leave_none_within_0_05_and_half_within_0_1():
    ranges = [(5, 45, False), (100, 300, False)]
    trials = [((25, 200), "complete", 1.41), ((27, 210), "complete", 1.2),
              ((28, 200), "complete", 0.4), ((35, 240), "complete", 1.1)]
    shares = statistics.plateau(trials, ranges, 0, 1.0)

    assert (round(math.hypot(0.05, 0.05), 4), round(math.hypot(0.25, 0.2), 2)) == (
        0.0707, 0.32)
    assert shares[:2] == [None, 0.5] and shares[6] == pytest.approx(2 / 3)


def test_a_label_closes_at_its_horizon_or_at_its_stop_and_the_last_rows_are_censored():
    # 10,000 buys 100 at row 1's open of 100.0; row 2 opens at 99.0 and trades down to
    # 94.0, and row 3 opens at 103.0
    bars  = [[flat(100.0)], [flat(100.0)], [Bar(99.0, 99.0, 94.0, 98.0, 1_000.0)]]
    bars += [[flat(103.0)] for _ in range(7)]
    times = hourly(len(bars))
    plain = Venue(maker=0.0, taker=0.0)
    held  = labels(times, bars, [Instrument("BTCUSDT")], plain, 2, 10_000.0)

    assert (held["ret"][0][0], held["exit"][0][0], held["end"][0][0],
            held["bars"][0][0]) == (pytest.approx(0.03), "horizon", 3, 2)

    stopped = labels(times, bars, [Instrument("BTCUSDT")], plain, 2, 10_000.0,
                     stop=0.05)
    assert (stopped["ret"][0][0], stopped["exit"][0][0], stopped["end"][0][0],
            stopped["bars"][0][0]) == (pytest.approx(-0.05), "stop", 2, 1)
    assert [held["exit"][t][0] for t in (7, 8, 9)] == ["censored", "censored", ""]
    assert [held["end"][t][0] for t in (7, 8)]     == [10, 11]
    assert math.isnan(held["ret"][7][0]) and held["end"][9][0] == -2 ** 63


def episode(columns, window, start, kinds=None, **given):
    """An episode of the reference environment over `columns`, one instrument's bars
    each, with no fees, opened at row `start`."""
    rows        = len(columns[0])
    kinds       = kinds or ["linear"] * len(columns)
    instruments = [Instrument(f"I{k}", kind=kind) for k, kind in enumerate(kinds)]
    bars        = [[column[t] for column in columns] for t in range(rows)]
    settings    = dict(columns=columns, features=None, window=window, actions="box",
                       max_leverage=1.0, max_offset=20, reward="log", tolerance=0.0,
                       episode_length=None, balance=10_000.0)
    settings.update(given)

    def make(at):
        return Simulator(hourly(rows), bars, instruments, Venue(maker=0.0, taker=0.0),
                         start=at)

    return Episode(make, start, range(len(columns)), settings)


def test_one_instrument_s_candles_in_a_window_of_64_are_328_values():
    seen = episode([[flat(100.0) for _ in range(70)]], 64, 63)

    assert len(seen.observe()) == 64 * 1 * 5 + 4 * 1 + 4 == 328


def test_a_bar_reads_its_prices_over_the_last_close_and_its_volume_over_the_mean():
    seen = episode([[flat(100.0, 100.0), Bar(100.0, 110.0, 95.0, 105.0, 300.0)]], 2, 1)

    assert seen.market(0)[1] == pytest.approx([0.952381, 1.047619, 0.904762, 1.0, 1.5],
                                              abs=5e-7)


def test_an_action_of_0_3_is_a_weight_of_0_6_and_two_are_scaled_by_their_sum():
    one = episode([[flat(100.0)] * 3], 1, 1, max_leverage=2.0)

    assert one.decode([0.3]) == pytest.approx({0: 0.6})

    two = episode([[flat(100.0)] * 3, [flat(50.0)] * 3], 1, 1, max_leverage=2.0)
    assert two.decode([0.8, -0.6]) == pytest.approx({0: 1.142857, 1: -0.857143},
                                                    abs=5e-7)


def test_a_takeover_from_2_000_is_rewarded_minus_3_and_a_sale_to_1_500_minus_0_29():
    assert episode([[flat(100.0)] * 3], 1, 1).floor  == 100.0
    assert round(logged(0.0, 2_000.0, 100.0), 2)     == -3.00
    assert round(logged(1_500.0, 2_000.0, 100.0), 2) == -0.29


def test_an_order_action_of_0_25_at_half_its_offset_buys_0_1_btc_ten_ticks_below():
    asked = limit(0.25, 0.5, 10_000.0, 50_000.0, Instrument("BTCUSDT", tick=0.1), 2.0,
                  20)

    assert asked == ("buy", pytest.approx(0.1), pytest.approx(49_999.0))


def test_three_overlapping_labels_weigh_0_6111_0_4444_and_0_6111():
    weights = uniqueness([(0, 1, 3), (1, 2, 4), (2, 3, 5)])

    assert [weights[t] for t in range(3)] == pytest.approx([11 / 18, 4 / 9, 11 / 18])
    assert round(weights[1], 4) == 0.4444


def test_four_predictions_correlate_at_0_6_in_rank_with_an_r_squared_of_0_4():
    rho, _, r_squared = statistics.forecast([0.01, -0.01, 0.02, 0.0],
                                            [0.02, -0.01, 0.01, -0.02], [0, 1, 2, 3], 1)

    assert (rho, r_squared) == (pytest.approx(0.6), pytest.approx(0.4))
    assert statistics.mid_ranks([0.01, -0.01, 0.02, 0.0]) == [3.0, 1.0, 4.0, 2.0]


def test_an_array_s_seam_places_the_inherited_stop_again_and_a_latch_holds():
    # window 1's weight of 1.0 at row 2 fills at row 3's open, its stop resting at
    # 95.0 from row 3's close; window 2 starts at row 6
    bars = [[flat(100.0)] for _ in range(10)]
    rows = [[math.nan]] * 2 + [[1.0]] + [[math.nan]] * 3 + [[1.0]] * 4
    sim  = play_array(Simulator(hourly(10), bars, [Instrument("BTCUSDT")],
                                Venue(maker=0.0, taker=0.0)), rows, stop=0.05,
                      seams={6})

    seam = [event for event in sim.events if event["reason"] == "seam"]
    assert [(event["bar"], event["kind"]) for event in seam] == [(6, "cancelled")]
    stops = [order for order in sim.orders.values() if order["leg"] == "stop"]
    assert [(order["trigger"], order["status"]) for order in stops] == [
        (95.0, "cancelled"), (95.0, "resting")]

    # a stop that fills on row 5 holds window 2's longs off until a row asks for zero
    falling = [[flat(100.0)] for _ in range(5)] + [[Bar(100.0, 100.0, 90.0, 92.0,
                                                        1_000.0)]]
    falling += [[flat(92.0)] for _ in range(4)]

    sim = play_array(Simulator(hourly(10), falling, [Instrument("BTCUSDT")],
                               Venue(maker=0.0, taker=0.0)), rows, stop=0.05, seams={6})

    assert sim.position[0] == 0.0
    assert [fill["reason"] for fill in sim.fills] == ["rebalance", "stop"]
