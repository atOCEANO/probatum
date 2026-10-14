"""A fill reads the window before its bar: the budget a bar holds is a share of the
trailing volume, shared by every order on it, and a taker pays half the spread the
window shows and an impact scaled by its volatility. A limit fills at the price the
market traded since the last close where it was placed through the market. The
Specification's worked examples hold through the binding.
"""

import re

import numpy as np
import pytest

import probatum


def bars(*rows):
    return np.array(rows, dtype=np.float64)


def test_a_budget_of_a_tenth_of_50_holds_5_for_five_orders_of_2():
    # five market orders of 2 BTC under budget(0.1) with a trailing volume of 50 fill
    # 2, 2 and 1, and the last two fill nothing
    e = probatum.Engine(
        bars((100.0, 100.0, 100.0, 100.0, 50.0),
             (100.0, 100.0, 100.0, 100.0, 50.0)),
        fee_taker = 0.0,
        budget    = 0.1,
        report    = True,
    )

    e.reset()
    for _ in range(5):
        e.market_buy(2.0)
    state = e.step()

    assert [fill["size"] for fill in e.fills()] == [2.0, 2.0, 1.0]
    assert state.position == 5.0

    reasons = [event["reason"] for event in e.order_log() if event["bar"] == 1]
    assert reasons.count("nothing could fill on its bar") == 2


def test_the_six_bars_spread_moves_a_buy_from_100_40_up_by_0_05():
    six = [(100.0, 100.5, 99.8, 100.2), (100.3, 100.6, 100.0, 100.0),
           (100.1, 100.6, 99.9, 100.4), (100.5, 100.8, 100.1, 100.1),
           (100.2, 100.7, 100.0, 100.5), (100.6, 100.9, 100.3, 100.3)]
    candles = bars(*[(*row, 1e6) for row in six], (100.40, 100.9, 100.3, 100.5, 1e6))
    e       = probatum.Engine(
        candles,
        fee_taker = 0.0,
        window    = 6,
        spread    = "estimated",
        report    = True,
    )

    e.reset()
    for _ in range(5):
        e.step()

    e.market_buy(1.0)
    e.step()

    [fill] = e.fills()
    assert fill["spread"] == pytest.approx(0.05, abs=0.005)
    assert fill["price"]  == pytest.approx(100.40 + fill["spread"], rel=1e-12)


def test_a_limit_placed_through_the_market_fills_where_it_traded():
    # a buy limit at 110 placed at a close of 100, on a next bar that opens at 96 with
    # a high of 98, fills as a taker at 100: no trade since that close reached 110
    e = probatum.Engine(
        bars((100.0, 100.0, 100.0, 100.0, 1e6),
             (96.0, 98.0, 95.0, 97.0, 1e6)),
        fee_taker = 0.0,
        fee_maker = 0.0,
        report    = True,
    )
    e.reset()
    e.limit_buy(1.0, 110.0)
    e.step()

    [fill] = e.fills()
    assert (fill["price"], fill["taker"]) == (100.0, True)


@pytest.mark.parametrize("settings, says", [
    (dict(budget=0.0),
     "budget: 0; give the share of the trailing volume a bar's fills may take"),
    (dict(window=0),
     "window: 0; give the number of bars before a fill its window reads, from 1"),
    (dict(spread=-1.0), 'spread: -1.0; give basis points from 0, or "estimated"'),
    (dict(spread="Wide"), 'spread: "Wide" is not a spread; give basis points'),
    (dict(impact_model="Cubic"), 'impact_model: "Cubic" is not a model; use linear or'),
])
def test_a_fill_model_the_engine_cannot_hold_is_refused(settings, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        probatum.Engine(bars((100.0, 100.0, 100.0, 100.0, 1.0)), **settings)

    with pytest.raises(ValueError, match=re.escape(says)):
        probatum.Batch(bars((100.0, 100.0, 100.0, 100.0, 1.0)), 2, **settings)
