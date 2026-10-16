"""A venue that books then backstops sells a liquidated account's legs on the book as
a taker from the trigger, and the trader keeps what the sale leaves, unless the path
reaches the backstop level or the sale's own costs would take equity below it, where
the account is taken over as it is everywhere else.
"""

import re

import numpy as np
import pytest

import probatum

HOUR = 3_600_000_000_000

# one tier at 20x keeping a maintenance rate of 0.5%
MAINTAINED = [(1e12, 20.0, 0.005, 0.0)]

BOOKED = dict(liquidation="book_then_backstop", backstop=2.0 / 3.0)


def bars(*rows):
    return np.array([[*row, 1e6] for row in rows], dtype=np.float64)


def half_a_bitcoin(*later, **settings):
    """The Specification's account at one instrument: 10,015 of margin buys 0.5 BTC
    with a limit at 60,000 that fills as a taker for 15, then the bars `later`."""
    candles = bars((60_000.0,) * 4, (60_000.0,) * 4, *later)
    venue   = dict(market="perp", quote=10_015.0, fee_taker=0.0005, fee_maker=0.0002,
                   leverage=10.0, tiers=MAINTAINED, report=True,
                   times=np.arange(len(candles), dtype=np.int64) * HOUR)
    venue.update(settings)

    e = probatum.Engine(candles, **venue)
    e.reset()
    e.limit_buy(0.5, 60_000.0)
    e.step()

    return e


@pytest.mark.parametrize("settings, low, fee, kept, charge", [
    (BOOKED, 40_150.0, 10.05, 90.45, 0.0),
    (dict(), 40_150.0, 0.0, 0.0, 100.50),
    # a low past the backstop level hands the account to the backstop, unless the
    # optimistic setting has the sale finish first
    (BOOKED, 40_130.0, 0.0, 0.0, 100.50),
    (dict(BOOKED, intrabar="optimistic"), 40_130.0, 10.05, 90.45, 0.0),
])
def test_the_long_sold_at_40_201_01_keeps_90_45_where_takeover_charges_it(
        settings, low, fee, kept, charge):
    # the trigger is where 0.5p - 20,000 falls to 0.0025p, and the backstop level is
    # two thirds of the requirement, at 40,133.78
    e     = half_a_bitcoin((41_000.0, 41_200.0, low, 40_500.0), **settings)
    state = e.step()
    sale  = e.fills()[-1]

    assert sale["reason"]    == "liquidation"
    assert sale["price"]     == pytest.approx(40_201.01, abs=0.005)
    assert sale["fee"]       == pytest.approx(fee, abs=0.005)
    assert state.margin_cash == pytest.approx(kept, abs=0.005)
    assert e.ledger()["liquidation"][-1] == pytest.approx(charge, abs=0.005)


@pytest.mark.parametrize("settings, kept, charge", [
    (BOOKED, 88.90, 0.0),
    (dict(), 0.0, 99.0),
])
def test_a_debit_at_the_close_that_reaches_the_trigger_sells_the_long_there(
        settings, kept, charge):
    # the bar stays clear of the trigger and closes at 40,400 with equity of 200
    # against a requirement of 101; a debit of 0.5% on the 20,200 held takes 101,
    # which leaves 99, above the backstop level of 67.33, and a sale at the close
    # pays the fee of 10.10 out of it
    e = half_a_bitcoin((41_000.0, 41_200.0, 40_300.0, 40_400.0), (40_400.0,) * 4,
                       funding_times=np.array([3 * HOUR], dtype=np.int64),
                       funding_rates=np.array([0.005]), **settings)

    state = e.step()
    sale  = e.fills()[-1]

    assert (sale["reason"], sale["price"]) == ("liquidation", 40_400.0)
    assert state.margin_cash               == pytest.approx(kept, abs=1e-9)
    assert e.ledger()["liquidation"][-1]   == pytest.approx(charge, abs=1e-9)


@pytest.mark.parametrize("settings, says", [
    (dict(liquidation="Backstop_Then_Book"),
     'liquidation: "Backstop_Then_Book" is not a model; use takeover or'),
    (dict(liquidation="book_then_backstop"),
     "backstop: none, and book_then_backstop takes over where equity falls to a "
     "share of the requirement; pass backstop="),
    (dict(liquidation="book_then_backstop", backstop=1.0),
     "backstop: 1; give the share of the requirement its level sits at, above 0"),
    (dict(backstop=0.5), 'backstop: 0.5 for takeover; give it with liquidation='),
    (dict(BOOKED, tiers=None),
     "liquidation: book_then_backstop with no maintenance in any perp's tier table"),
    (dict(liquidation_fee=-0.001),
     "liquidation_fee: -0.001; give the rate on its notional a close at the close "
     "point pays, from 0 and below 1"),
    (dict(liquidation_fee=1.0), "liquidation_fee: 1; give the rate on its notional"),
    (dict(liquidation_fee=float("nan")), "liquidation_fee: NaN; give the rate"),
])
def test_a_liquidation_the_venue_cannot_hold_is_refused(settings, says):
    venue = dict(market="perp", tiers=MAINTAINED)
    venue.update(settings)

    with pytest.raises(ValueError, match=re.escape(says)):
        probatum.Engine(bars((100.0,) * 4), **venue)


def test_a_universe_a_batch_and_a_spot_table_with_no_perp_maintenance_are_refused():
    flat = dict(instruments=[dict(market="perp", tiers=[(1e12, 20.0, 0.0, 0.0)])] * 2)
    spot = dict(quote={"spot": 1_000.0, "margin": 1_000.0},
                instruments=[dict(market="spot", tiers=MAINTAINED),
                             dict(market="perp")])

    for universe in (flat, spot):
        with pytest.raises(ValueError, match="liquidation: book_then_backstop with no"):
            probatum.Engine(np.full((2, 2, 5), 100.0), **universe, **BOOKED)

    with pytest.raises(ValueError, match="liquidation: book_then_backstop with no"):
        probatum.Batch(bars((100.0,) * 4), 2, market="perp", **BOOKED)
