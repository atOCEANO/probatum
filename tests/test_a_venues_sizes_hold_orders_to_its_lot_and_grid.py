"""A venue's sizes hold every order to its instrument's lot, price rule, minimum
notional and largest size, and a limit under `through` fills once its bar trades a
step of the price rule past it. The Specification's worked examples hold through the
binding, and a rule no venue could hold is refused where it enters.
"""

import re

import numpy as np
import pytest

import probatum


def bars(*rows):
    return np.array(rows, dtype=np.float64)


def test_a_lot_of_0_001_fills_0_075_of_a_budget_of_0_0754_and_refuses_0_0004():
    e = probatum.Engine(
        bars((100.0, 100.0, 100.0, 100.0, 0.0754),
             (100.0, 100.5, 98.5, 99.5, 0.0754)),
        fee_taker = 0.0,
        fee_maker = 0.0,
        sizes     = True,
        lot       = 0.001,
        report    = True,
    )

    e.reset()
    e.limit_buy(0.2, 99.0)

    assert e.market_buy(0.0004) is None

    state = e.step()
    assert state.position == pytest.approx(0.075, abs=1e-12)

    [resting] = state.open_orders
    assert resting["remaining"] == pytest.approx(0.125, abs=1e-12)

    [refused] = [event for event in e.order_log() if event["kind"] == "refused"]
    assert refused["reason"] == "size: 0.0004 rounds to nothing at a lot of 0.001"


@pytest.mark.parametrize("limit, low, fills", [("touch", 99.40, True),
                                               ("through", 99.40, False),
                                               ("through", 99.39, True)])
def test_a_buy_limit_at_99_40_fills_on_touch_and_through_only_a_tick_past(limit, low,
                                                                          fills):
    e = probatum.Engine(
        bars((100.0, 100.0, 100.0, 100.0, 1e6),
             (100.0, 100.2, low, 99.8, 1e6)),
        limit = limit,
        tick  = 0.01,
    )
    e.reset()

    e.limit_buy(1.0, 99.40)

    assert (e.step().position == 1.0) is fills


@pytest.mark.parametrize("low, fills", [(1_234.6, False), (1_234.5, True)])
def test_five_figures_with_a_cap_of_one_decimal_rest_1_234_56_at_1_234_6(low, fills):
    e = probatum.Engine(
        bars((1_300.0, 1_300.0, 1_300.0, 1_300.0, 1e6),
             (1_300.0, 1_300.0, low, 1_250.0, 1e6)),
        quote    = 1e6,
        sizes    = True,
        limit    = "through",
        sigfigs  = 5,
        decimals = 1,
    )

    e.reset()
    e.limit_buy(1.0, 1_234.56)
    e.limit_sell(1.0, 123_456.7)
    prices = [order["price"] for order in e.step().open_orders]

    assert prices == ([123_457.0] if fills else [1_234.6, 123_457.0])


def test_an_inverse_lot_is_in_contracts_and_its_minimum_notional_in_the_coin():
    # a buy of 0.4 contracts of 100 dollars rounds to no lot of 1, and 5 contracts at
    # 50,000 are worth 0.01 coin, below a minimum of 0.02 coin
    e = probatum.Engine(
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
        probatum.Engine(candles, **settings)

    with pytest.raises(ValueError, match=re.escape(says)):
        probatum.Batch(candles, 2, **settings)


def test_the_significant_figure_rule_is_refused_on_an_inverse_contract():
    with pytest.raises(ValueError, match=re.escape(
            "sigfigs: 5 for an inverse contract, whose prices take a tick; give its tick")):
        probatum.Engine(
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
        probatum.Engine(universe, market="perp", lot=1.0, instruments=instruments)

    with pytest.raises(ValueError, match=re.escape(
            "instruments[1]: lot: 'half'; give a number")):
        probatum.Engine(
            universe,
            market      = "perp",
            instruments = [dict(market="perp"), dict(lot="half")],
        )
