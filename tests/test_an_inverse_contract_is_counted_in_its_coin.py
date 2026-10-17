"""An inverse perpetual is quoted in dollars and settled in its coin: its account holds
that one contract, counts everything in the coin, and values its equity in dollars
beside it. The Specification's worked examples of inverse contracts hold through the
engine, and so do a taker's costs, each its rate of the coin notional.
"""

import math
import re

import numpy as np
import pytest

import probatum

HOUR = 3_600_000_000_000

# one tier with a maintenance rate of 0.5% and no leverage cap, wide enough for any
# position here
MAINTAINED = [(1e9, 0.0, 0.005, 0.0)]


def bars(*rows):
    return np.array([[*row, 1e9] for row in rows], dtype=np.float64)


def engine(candles, **settings):
    venue = dict(market="inverse", contract=100.0, quote=1.0, fee_taker=0.0,
                 fee_maker=0.0, leverage=0.0, tiers=MAINTAINED, report=True,
                 times=np.arange(len(candles), dtype=np.int64) * HOUR)
    venue.update(settings)
    return probatum.Engine(candles, **venue)


def test_a_coin_held_against_a_short_of_its_value_is_a_synthetic_dollar():
    # a wallet of 1 BTC with a short of 600 contracts from 60,000: its equity is
    # 60,000 / p coins, worth 60,000 dollars at every price, never liquidated, and an
    # event of 0.01% at a close of 60,000 pays it 0.0001 BTC
    candles = bars((60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 120_000.0, 30_000.0, 30_000.0),
                   (30_000.0, 120_000.0, 30_000.0, 120_000.0))

    e = engine(candles, funding_times=np.array([2 * HOUR], dtype=np.int64),
               funding_rates=np.array([0.0001]))
    e.reset()
    e.market_sell(600.0)
    state = e.step()

    assert (state.position, state.avg_entry) == (-600.0, 60_000.0)
    assert state.funding_paid == pytest.approx(-0.0001, abs=1e-15)

    for _ in range(2):
        state = e.step()

    ledger = e.ledger()
    closes = candles[:, 3]
    held   = ledger["equity"][1:] - 0.0001

    np.testing.assert_allclose(held * closes[1:], 60_000.0, rtol=1e-12)
    np.testing.assert_allclose(ledger["equity_in_quote"], ledger["equity"] * closes,
                               rtol=1e-12)
    assert not e.is_bust() and ledger["liquidation"].sum() == 0.0


@pytest.mark.parametrize("low, liquidated", [(40_199.0, True), (40_201.0, False)])
def test_a_long_on_half_a_coin_is_liquidated_at_40_200(low, liquidated):
    # a wallet of 0.5 BTC with a long of 600 contracts from 60,000 has equity
    # 1.5 - 60,000 / p and a requirement of 300 / p, which meet at 40,200, where both
    # are 0.0074627 BTC
    candles = bars((60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 60_000.0, low, 50_000.0))

    e = engine(candles, quote=0.5)
    e.reset()
    e.market_buy(600.0)
    e.step()
    state = e.step()
    fills = e.fills()

    if not liquidated:
        assert state.position == 600.0 and len(fills) == 1
        return

    close = fills[-1]

    assert (close["reason"], close["side"]) == ("liquidation", "sell")
    assert close["size"]  == 600.0
    assert close["price"] == pytest.approx(40_200.0, rel=1e-9)
    assert e.ledger()["liquidation"][-1] == pytest.approx(1.5 - 60_000.0 / 40_200.0,
                                                          rel=1e-9)
    assert state.position == 0.0 and abs(state.margin_cash) < 1e-12


def test_two_buys_enter_at_the_harmonic_mean_and_a_value_order_buys_contracts():
    # 300 contracts at 60,000 and 300 at 40,000, the second bought for 30,000 dollars,
    # enter at 48,000, and at 50,000 the long has made 0.05 BTC, the sum of the fills'
    # own, -0.10 and 0.15
    candles = bars((60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (40_000.0, 50_000.0, 40_000.0, 50_000.0))

    e = engine(candles)
    e.reset()

    assert e.qty_from_quote(6_000.0) == 60.0

    e.market_buy(300.0)
    e.step()

    assert e.order_value("buy", -50.0) is None
    assert e.order_log()[-1]["reason"] == "amount: -50 cannot fill"

    e.order_value("buy", 30_000.0)
    state = e.step()

    assert state.position       == 600.0
    assert state.avg_entry      == pytest.approx(48_000.0, rel=1e-12)
    assert state.unrealized_pnl == pytest.approx(0.05, rel=1e-12)


@pytest.mark.parametrize("settings, price, fee, slippage", [
    # a taker fee of 0.05% on 600 contracts filled at 60,000, 1 BTC of notional
    (dict(fee_taker=0.0005), 60_000.0, 0.0005, 0.0),
    # 10 bps of slippage moves the reciprocal of the price, so the buy fills at
    # 60,000 / 0.999 and pays 0.001 BTC beyond its reference
    (dict(slippage_bps=10.0), 60_000.0 / 0.999, 0.0, 0.001),
])
def test_a_taker_fee_and_slip_are_each_their_rate_of_the_coin_notional(settings, price,
                                                                       fee, slippage):
    candles = bars((60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 61_000.0, 59_000.0, 60_000.0))

    e = engine(candles, **settings)
    e.reset()
    e.market_buy(600.0)
    e.step()
    [fill] = e.fills()

    assert fill["price"]    == pytest.approx(price, rel=1e-12)
    assert fill["slippage"] == pytest.approx(price - 60_000.0, abs=1e-9)
    assert fill["fee"]      == pytest.approx(fee, rel=1e-12)
    assert e.ledger()["slippage"][1] == pytest.approx(slippage, rel=1e-9)


def test_a_short_of_one_coin_s_worth_is_a_target_of_minus_one():
    # with 1 BTC at a close of 60,000, a weight of -1 is 600 contracts of 100 dollars
    candles = bars((60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 60_000.0, 60_000.0, 60_000.0))

    e = engine(candles)
    e.reset()
    e.target(-1.0)

    assert e.step().position        == pytest.approx(-600.0, rel=1e-12)
    assert e.ledger()["weight"][-1] == pytest.approx(-1.0, rel=1e-12)


def test_the_views_and_the_state_show_the_bars_as_printed():
    candles = bars((60_000.0, 61_000.0, 59_000.0, 60_500.0),
                   (60_500.0, 62_000.0, 60_000.0, 61_000.0))

    e = engine(candles)

    np.testing.assert_array_equal(e.data, candles)
    np.testing.assert_array_equal(e.closes, candles[:, 3])

    e.reset()
    state = e.step()

    assert (state.bar_high, state.bar_low) == pytest.approx((62_000.0, 60_000.0))
    assert state.bar_volume == pytest.approx(1e9)
    np.testing.assert_array_equal(e.observation(2), candles)


def test_a_batch_of_inverse_accounts_trades_contracts():
    candles = bars((60_000.0, 60_000.0, 60_000.0, 60_000.0),
                   (60_000.0, 60_000.0, 50_000.0, 50_000.0))

    batch = probatum.Batch(
        candles,
        2,
        market    = "inverse",
        contract  = 100.0,
        quote     = 1.0,
        fee_taker = 0.0,
        leverage  = 0.0,
    )
    batch.reset_all()
    states = batch.step_all(np.array([600.0, -300.0]))

    assert [state.position for state in states] == [600.0, -300.0]

    snapshot = batch.snapshot()

    np.testing.assert_allclose(snapshot["position"], [600.0, -300.0])
    np.testing.assert_allclose(snapshot["mark_price"], [50_000.0, 50_000.0])
    np.testing.assert_array_equal(batch.observe(1)[:, 0], candles[[1, 1]])


@pytest.mark.parametrize("settings, says", [
    (dict(contract=None), "contract: none, and an inverse contract is worth"),
    (dict(contract=0.0), "contract: 0; give the dollars one contract is worth"),
    (dict(market="perp"), 'contract: 100 for a perp market; give it with market='),
    (dict(quote={"margin": 1.0}), "quote: a dict of wallets for an inverse contract"),
])
def test_an_inverse_contract_the_engine_cannot_hold_is_refused(settings, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        engine(bars((60_000.0, 60_000.0, 60_000.0, 60_000.0)), **settings)


def test_a_price_with_no_reciprocal_a_universe_and_a_transfer_are_refused():
    with pytest.raises(ValueError, match="candles: a low of 0.0+1 at row 1, whose "):
        engine(bars((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1e-310, 1.0)))

    with pytest.raises(ValueError, match="market: an inverse contract for a universe"):
        probatum.Engine(np.ones((2, 1, 5)), market="inverse", contract=100.0)

    e = engine(bars((60_000.0, 60_000.0, 60_000.0, 60_000.0)))
    with pytest.raises(ValueError, match="transfer: an inverse contract's account"):
        e.transfer(0.5, "spot")

    with pytest.raises(ValueError, match="tiers\\[0\\]: up_to 17976931348623157"):
        engine(bars((60_000.0, 60_000.0, 60_000.0, 60_000.0)),
               tiers=[(1.7976931348623157e308, 20.0, 0.0, 0.0)], tiers_by="quantity")


@pytest.mark.parametrize("place, says", [
    (lambda e: e.stop("sell", 600.0, 0.0, reduce_only=True), "trigger: 0, which an"),
    (lambda e: e.limit_buy(600.0, -5.0), "price: -5, which an inverse contract"),
    (lambda e: e.bracket(stop=0.0), "stop: 0, which an inverse contract"),
    (lambda e: e.order("buy", 1.0, "limit", price=1e-310), "which an inverse contract"),
])
def test_a_price_with_no_reciprocal_is_refused(place, says):
    # a dollar price at or below zero has no reciprocal, and a stop at zero would
    # trigger at once in the engine's terms where a bar can never reach it
    e = engine(bars((60_000.0, 60_000.0, 60_000.0, 60_000.0)))
    e.reset()

    with pytest.raises(ValueError, match=re.escape(says)):
        place(e)

    script = {0: [("buy", {"size": 1.0, "limit": 0.0})]}
    venue  = dict(market="inverse", contract=100.0, quote=1.0)

    with pytest.raises(ValueError, match=re.escape("script[0][0]: limit 0, which")):
        probatum._probatum.run(bars((60_000.0,) * 4, (60_000.0,) * 4), venue, script)


def test_a_price_comes_back_as_it_was_quoted_and_a_flat_position_reads_flat():
    # a reciprocal taken twice lands a float off 49 and 56.31, where the fewest
    # digits find them again
    candles = bars((49.0, 49.0, 49.0, 49.0), (49.0, 57.0, 48.0, 56.31))
    e       = engine(candles)
    state   = e.reset()

    assert math.copysign(1.0, state.position) == 1.0

    e.limit_buy(10.0, 49.0)
    state = e.step()

    assert (state.avg_entry, state.mark_price, state.bar_close) == (49.0, 56.31, 56.31)
    assert e.fills()[0]["price"]       == 49.0
    assert e.ledger()["mark"].tolist() == [49.0, 56.31]


def test_a_value_order_that_is_no_amount_is_refused_for_its_amount():
    # an inverse contract's dollars buy its contracts only where they are an amount,
    # and a market order's time in force is read in any case
    script = {0: [("buy_value", {"amount": -50.0}),
                  ("sell_value", {"amount": 50.0, "tif": "IOC"})]}
    venue  = dict(market="inverse", contract=100.0, quote=1.0)
    played = probatum._probatum.run(bars((60_000.0,) * 4, (60_000.0,) * 4), venue, script)

    assert [(event["kind"], event["reason"]) for event in played["order_log"][:2]] == [
        ("refused", "amount: -50 cannot fill"), ("placed", "market")]
