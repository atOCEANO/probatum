"""Every worked example on the Specification page, held to the cent.

Each test is one example, named for what it shows, and runs it through the
reference simulator written from that page. The engine joins each test once it
models the rule; until then the reference is held to the page on its own.
"""

import math

import pytest

from reference.simulator import Bar, Instrument, Venue, edge, periods_per_year, run

HOUR = 3_600 * 10**9
DAY  = 24 * HOUR


def flat(price, volume=1_000.0):
    return Bar(price, price, price, price, volume)


def hourly(rows):
    return [i * HOUR for i in range(rows)]


def test_hourly_bars_over_a_year_annualise_at_8760():
    assert periods_per_year(hourly(8_761)) == 8_760.0


def test_253_daily_bars_from_one_2_january_to_the_next_annualise_at_252():
    days = [0] + list(range(1, 252)) + [365]

    assert periods_per_year([d * DAY for d in days]) == pytest.approx(252.0, abs=1e-9)


def test_an_order_decided_at_fridays_close_fills_at_mondays_open():
    thursday = 3 * DAY
    times    = [thursday, thursday + DAY, thursday + 4 * DAY]
    bars     = [[flat(100.0)], [flat(101.0)], [Bar(98.0, 99.0, 97.0, 98.5, 1_000.0)]]
    sim      = run(times, bars, [Instrument("SPY", kind="spot")],
                   Venue(maker=0.0, taker=0.0),
                   script={1: [("buy", dict(instrument=0, size=10.0))]})

    [fill] = sim.fills
    assert fill["time"]  == times[2]
    assert fill["price"] == 98.0


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
              [Instrument("BTCUSDT", lot=0.001)],
              script={0: [("target", dict(weights=0.5))]})
    [order] = sim.orders.values()

    assert order["side"]   == "buy"
    assert order["size"]   == pytest.approx(0.083, abs=1e-12)
    assert sim.position[0] == pytest.approx(0.083, abs=1e-12)


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


def test_a_fill_pays_its_size_times_its_price_times_the_fee():
    sim = run(hourly(2), [[flat(21_000.0)], [flat(21_014.75)]],
              [Instrument("BTCUSDT")], Venue(taker=0.0005),
              script={0: [("buy", dict(instrument=0, size=0.141))]})
    [fill] = sim.fills

    assert fill["size"] * fill["price"] == pytest.approx(2_963.08, abs=0.005)
    assert fill["fee"] == pytest.approx(1.48, abs=0.005)


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
                                               ("through", 99.39, True)])
def test_a_buy_limit_at_99_40_fills_on_touch_and_through_only_a_tick_past(limit, low,
                                                                          fills):
    sim = run(hourly(2), [[flat(100.0)], [Bar(100.0, 100.2, low, 99.8, 1_000.0)]],
              [Instrument("BTCUSDT", tick=0.01)], Venue(limit=limit),
              script={0: [("buy", dict(instrument=0, size=1.0, limit=99.40))]})

    assert bool(sim.fills) is fills


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
    bars.append([Bar(21_012.0, 21_080.0, 21_000.0, 21_050.0, 24_800.0)])
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
    assert fill["size"] * fill["price"] == pytest.approx(2_963.08, abs=0.005)
    assert fill["fee"] == pytest.approx(1.48, abs=0.005)


def test_a_budget_of_a_tenth_of_50_lets_five_orders_of_2_fill_2_2_and_1():
    bars   = [[flat(100.0, volume=50.0)] for _ in range(26)]
    script = {24: [("buy", dict(instrument=0, size=2.0)) for _ in range(5)]}
    sim    = run(hourly(len(bars)), bars, [Instrument("BTCUSDT")], Venue(budget=0.1),
                 script=script)

    assert [f["size"] for f in sim.fills] == pytest.approx([2.0, 2.0, 1.0])
    assert [e["kind"] for e in sim.events if e["order"] in (4, 5)] == [
        "placed", "placed", "not_filled", "not_filled"]


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
    bars  = [[flat(100.0, volume=0.0754)]] + [[Bar(100.0, 100.5, 98.5, 99.5, 0.0754)]]
    venue = Venue(sizes=True)
    sim   = run(hourly(2), bars, [Instrument("BTCUSDT", lot=0.001)], venue,
                script={0: [("buy", dict(instrument=0, size=0.2, limit=99.0)),
                            ("buy", dict(instrument=0, size=0.0004))]})
    [fill] = sim.fills

    assert fill["size"] == pytest.approx(0.075, abs=1e-12)

    resting = sim.orders[1]
    assert resting["status"] == "resting"
    assert resting["size"] - resting["filled"] == pytest.approx(0.125, abs=1e-12)
    assert sim.orders[2]["status"] == "refused"


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
