"""A venue's sizes hold every order to its instrument's lot, price rule, minimum
notional and largest size, and a limit under `through` fills once its bar trades half
a step of the price rule past it. The Specification's worked examples hold through
the binding, and a rule no venue could hold is refused where it enters.
"""

import re

import numpy as np
import pytest

import probatum


def bars(*rows):
    return np.array(rows, dtype=np.float64)


def test_a_lot_of_0_001_fills_0_075_of_a_budget_of_0_0756_and_refuses_0_0006():
    # both round down where the nearest lot would round them up
    e = probatum._probatum.Engine(
        bars((100.0, 100.0, 100.0, 100.0, 0.0756),
             (100.0, 100.5, 98.5, 99.5, 0.0756)),
        fee_taker = 0.0,
        fee_maker = 0.0,
        sizes     = True,
        lot       = 0.001,
        report    = True,
    )

    e.reset()
    e.limit_buy(0.2, 99.0)

    assert e.market_buy(0.0006) is None

    state = e.step()

    assert state.position == pytest.approx(0.075, abs=1e-12)

    [resting] = state.open_orders
    assert resting["remaining"] == pytest.approx(0.125, abs=1e-12)

    [refused] = [event for event in e.order_log() if event["kind"] == "refused"]
    assert refused["reason"] == "size: 0.0006 rounds to nothing at a lot of 0.001"


@pytest.mark.parametrize("limit, low, fills", [("touch", 99.40, True),
                                               ("through", 99.40, False),
                                               ("through", 99.398, False),
                                               ("through", 99.394, True),
                                               ("through", 99.39, True)])
def test_a_buy_limit_at_99_40_fills_on_touch_and_through_half_a_tick_past(limit, low,
                                                                          fills):
    e = probatum._probatum.Engine(
        bars((100.0, 100.0, 100.0, 100.0, 1e6),
             (100.0, 100.2, low, 99.8, 1e6)),
        limit = limit,
        tick  = 0.01,
    )

    e.reset()
    e.limit_buy(1.0, 99.40)

    assert (e.step().position == 1.0) is fills


@pytest.mark.parametrize("low, fills", [(99.98, False), (99.9, True)])
def test_five_figures_with_a_cap_of_one_decimal_rest_99_987_at_100_0(low, fills):
    e = probatum._probatum.Engine(
        bars((101.0, 101.0, 101.0, 101.0, 1e6),
             (101.0, 101.0, low, 100.5, 1e6)),
        quote    = 1e6,
        sizes    = True,
        limit    = "through",
        sigfigs  = 5,
        decimals = 1,
    )

    e.reset()
    e.limit_buy(1.0, 99.987)
    e.limit_sell(1.0, 123_456.7)
    prices = [order["price"] for order in e.step().open_orders]

    assert prices == ([123_457.0] if fills else [100.0, 123_457.0])


@pytest.mark.parametrize("exempt", [False, True])
def test_a_minimum_of_10_refuses_0_05_at_150_and_a_maximum_of_1_0005_holds_5_to_1(exempt):
    e = probatum._probatum.Engine(
        bars((150.0, 150.0, 150.0, 150.0, 1e6),
             (150.0, 150.0, 150.0, 150.0, 1e6)),
        market             = "perp",
        sizes              = True,
        lot                = 0.001,
        min_notional       = 10.0,
        max_qty            = 1.0005,
        reduce_only_exempt = exempt,
        report             = True,
    )

    e.reset()

    assert e.market_buy(0.05) is None

    spared = e.stop("sell", 0.05, 100.0, reduce_only=True)

    assert (spared is not None) is exempt

    held = e.market_buy(5.0)
    [clamp] = [event for event in e.order_log() if event["kind"] == "clamped"]

    assert (clamp["order"], clamp["size"]) == (held, 1.0)


def test_a_sell_under_through_needs_half_a_tick_above_it():
    for high, fills in [(100.60, False), (100.604, False), (100.606, True)]:
        e = probatum._probatum.Engine(
            bars((100.0, 100.0, 100.0, 100.0, 1e6),
                 (100.2, high, 100.0, 100.4, 1e6)),
            market = "perp",
            limit  = "through",
            tick   = 0.01,
        )
        e.reset()
        e.limit_sell(1.0, 100.60)
        assert (e.step().position == -1.0) is fills, high


def test_an_inverse_limit_takes_half_a_dollar_tick_through():
    # a dollar buy at 49 needs a dollar low of 48.5 under a tick of 1, and a dollar
    # sell at 51 a high of 51.5, each read on the dollar price quoted
    cases = [("buy", 49.0, (50.0, 50.0, 48.6, 50.0), 0.0),
             ("buy", 49.0, (50.0, 50.0, 48.5, 50.0), 10.0),
             ("sell", 51.0, (50.0, 51.4, 50.0, 50.0), 0.0),
             ("sell", 51.0, (50.0, 51.5, 50.0, 50.0), -10.0)]

    for side, price, bar, position in cases:
        e = probatum._probatum.Engine(
            bars((50.0, 50.0, 50.0, 50.0, 1e6), (*bar, 1e6)),
            market   = "inverse",
            contract = 100.0,
            quote    = 1.0,
            leverage = 0.0,
            limit    = "through",
            tick     = 1.0,
        )
        e.reset()
        (e.limit_buy if side == "buy" else e.limit_sell)(10.0, price)
        assert e.step().position == position, (side, bar)


def test_a_tick_rounds_a_price_at_half_a_tick_to_even():
    e = probatum._probatum.Engine(
        bars((100.0, 100.0, 100.0, 100.0, 1e6),
             (100.0, 100.0, 100.0, 100.0, 1e6)),
        market = "perp",
        sizes  = True,
        tick   = 0.5,
    )

    e.reset()
    e.limit_buy(1.0, 98.25)
    e.limit_sell(1.0, 102.75)
    prices = sorted(order["price"] for order in e.step().open_orders)

    assert prices == [98.0, 103.0]


def test_a_replace_and_a_bracket_take_the_grid_and_a_target_its_lot_and_transfer():
    e = probatum._probatum.Engine(
        bars(*[(100.0, 100.0, 100.0, 100.0, 1e6)] * 3),
        market    = "spot",
        quote     = {"spot": 1_000.0, "margin": 9_000.0},
        fee_taker = 0.0,
        sizes     = True,
        tick      = 0.01,
        lot       = 0.001,
        max_qty   = 1.0,
        report    = True,
    )

    e.reset()
    order = e.limit_buy(0.5, 90.004)

    assert e.replace(order, price=90.106) == order

    # a target of half the equity wants 50 at the close of 100, which the maximum
    # holds to 1, so the spot wallet's 1,000 covers it and nothing moves across
    e.target(0.5, transfer=True)
    state = e.step()

    assert (state.position, state.spot_cash, state.margin_cash) == (1.0, 900.0, 9_000.0)

    [resting] = state.open_orders
    assert resting["price"] == pytest.approx(90.11, abs=1e-12)


def test_an_inverse_lot_is_in_contracts_and_its_minimum_notional_in_the_coin():
    # a buy of 0.4 contracts of 100 dollars rounds to no lot of 1, and 5 contracts at
    # 50,000 are worth 0.01 coin, below a minimum of 0.02 coin
    e = probatum._probatum.Engine(
        bars((50_000.0, 50_000.0, 50_000.0, 50_000.0, 1e6),
             (50_000.0, 50_000.0, 50_000.0, 50_000.0, 1e6)),
        market       = "inverse",
        contract     = 100.0,
        quote        = 1.0,
        sizes        = True,
        lot          = 1.0,
        min_notional = 0.02,
        report       = True,
    )

    e.reset()

    assert e.market_buy(0.4) is None
    assert e.market_buy(5.0) is None
    assert e.market_buy(10.0) is not None

    reasons = [event["reason"] for event in e.order_log() if event["kind"] == "refused"]
    assert reasons == ["size: 0.4 rounds to nothing at a lot of 1",
                       "size: 5 is below the minimum notional of 0.02"]

    assert e.step().position == 10.0


@pytest.mark.parametrize("settings, says", [
    (dict(lot=-1.0), "lot: -1; give the lot every size is a whole number of, from 0"),
    (dict(tick=float("nan")), "tick: NaN; give the step prices sit on, from 0"),
    (dict(min_notional=-5.0),
     "min_notional: -5; give the smallest notional an order may have, from 0"),
    (dict(max_qty=0.0), "max_qty: 0; give the largest size an order may have, above 0"),
    (dict(lot=1.0, max_qty=0.5), "max_qty: 0.5; give a maximum of at least the lot of 1"),
    (dict(sigfigs=0, decimals=1), "sigfigs: 0; give a whole number from 1 to 22"),
    (dict(sigfigs=5, decimals=2.0), "decimals: 2.0; give a whole number from 0 to 22"),
    (dict(sigfigs=5), "sigfigs: 5 without decimals; give the decimal cap with it"),
    (dict(decimals=2), "decimals: 2 without sigfigs; give the significant figures"),
    (dict(sigfigs=5, decimals=2, tick=0.5),
     "sigfigs: 5 beside a tick of 0.5; give one price rule"),
    (dict(limit="Cross"), 'limit: "Cross" is not a fill rule; use touch or through'),
])
def test_a_rule_no_venue_could_hold_is_refused(settings, says):
    candles = bars((100.0, 100.0, 100.0, 100.0, 1.0))

    with pytest.raises(ValueError, match=re.escape(says)):
        probatum._probatum.Engine(candles, **settings)

    with pytest.raises(ValueError, match=re.escape(says)):
        probatum.Batch(candles, 2, **settings)


def test_the_significant_figure_rule_is_refused_on_an_inverse_contract():
    with pytest.raises(ValueError, match=re.escape(
            "sigfigs: 5 for an inverse contract, whose prices take a tick; give its tick")):
        probatum._probatum.Engine(
            bars((100.0, 100.0, 100.0, 100.0, 1.0)),
            market   = "inverse",
            contract = 100.0,
            quote    = 1.0,
            sigfigs  = 5,
            decimals = 1,
        )


def test_a_universe_takes_each_instruments_rules_in_its_own_dict():
    universe    = np.full((2, 2, 5), 100.0)
    instruments = [dict(market="perp", lot=1.0), dict(market="perp", lot=0.5)]
    venue       = dict(market="perp", sizes=True, instruments=instruments)
    script      = {0: [("buy", dict(instrument=0, size=0.6)),
                       ("buy", dict(instrument=1, size=0.6))]}

    log = probatum._probatum.run(universe, venue, script)["order_log"]

    assert [(event["order"], event["kind"]) for event in log if event["bar"] == 0] == [
        (1, "refused"), (2, "placed")]

    with pytest.raises(ValueError, match=re.escape(
            "lot: given beside instruments; give each instrument its own there")):
        probatum._probatum.Engine(universe, market="perp", lot=1.0, instruments=instruments)

    with pytest.raises(ValueError, match=re.escape(
            "instruments[1]: lot: 'half'; give a number")):
        probatum._probatum.Engine(
            universe,
            market      = "perp",
            instruments = [dict(market="perp"), dict(lot="half")],
        )
