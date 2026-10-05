"""Every worked example on the Specification page, held to the cent.

Each test is one example, named for what it shows, and runs it through the
reference simulator written from that page. The engine joins each test once it
models the rule; until then the reference is held to the page on its own.
"""

import pytest

from reference.simulator import Bar, Instrument, Venue, periods_per_year, run

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
