"""A sweep runs one policy again where only its balance or its costs differ: capacity
opens the account at each balance as a replaced venue would, a cost sweep multiplies
every fee rate above 0, the spread, fixed or estimated, the slippage and the impact
coefficient together, and each run is the backtest of a fresh copy of the policy, read
for its round-trip cost, its turnover and the orders the liquidity budget held as the
reference reads them. The break-even is bisected where the return reaches zero, and
every refusal names what to give instead.
"""

import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import ProbatumTypeError, ProbatumValueError, ProbatumWarning, VenueError
from probatum import _costs
from reference import statistics

HOUR  = 3_600 * 10**9
START = 1_704_067_200 * 10**9
FEES  = {"maker": 0.0002, "taker": 0.0005}
VENUE = probatum.Venue(
    fees  = FEES,
    fills = {"spread": {"fixed": 2.0}, "slippage": {"fixed": 1.0},
             "impact": {"linear": 0.5},
             "liquidity": {"budget": 0.05}},
)


def walked(count=600, width=1, seed=5, drift=0.0):
    """``width`` linear perpetuals of random walks of ``drift`` a bar on an hourly
    clock, with volumes that vary, as a Data."""
    rng    = np.random.default_rng(seed)
    close  = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, (count, width)), axis=0))
    opened = np.vstack([close[:1], close[:-1]])
    bars   = np.stack([opened, np.maximum(opened, close) * 1.002,
                       np.minimum(opened, close) * 0.998, close,
                       rng.uniform(1e3, 3e3, (count, width))], axis=-1)

    return probatum.Data(
        bars,
        symbol   = [f"C{k}USDT" for k in range(width)],
        market   = "linear",
        time     = START + np.arange(count) * HOUR,
        interval = "1h",
    )


def trend(data, fast=12, slow=48):
    """Long while each instrument's fast average is above its slow one and short below,
    NaN while the slow one warms up."""
    up, down = probatum.ta.ema(data.closes, fast), probatum.ta.ema(data.closes, slow)
    held     = np.where(np.isnan(down), np.nan, np.where(up > down, 1.0, -1.0))
    return held / data.closes.shape[1]


def inverse():
    """An inverse contract of 100 dollars on a random walk from 20,000, as a Data."""
    walked = np.random.default_rng(4).normal(0.0, 0.01, 300)
    close  = 20_000.0 * np.exp(np.cumsum(walked))
    opened = np.r_[close[0], close[:-1]]
    frame  = pd.DataFrame({"open": opened, "high": np.maximum(opened, close) * 1.01,
                           "low": np.minimum(opened, close) * 0.99, "close": close,
                           "volume": 1e6},
                          index=pd.to_datetime(START + np.arange(300) * HOUR, utc=True))
    frame.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                       interval="1h", quote="USD", warnings=[], volume_unit="contract",
                       contract_size=100.0)

    return probatum.Data(frame)


# a walk the trend earns on, until its costs pass about four times the venue's
DATA   = walked(drift=0.001)
POLICY = trend(DATA)


def read(result, data):
    """What the reference reads of ``result``: its round-trip cost, its turnover and the
    share of its orders the liquidity budget held."""
    contract = {name: instrument.contract for name, instrument in
                zip(data.names, data.instruments) if instrument.market == "inverse"}
    notionals = [abs(fill["size"]) * contract[fill["instrument"]] / fill["price"]
                 if fill["instrument"] in contract
                 else abs(fill["size"]) * fill["price"] for fill in result.fills]
    costs = math.fsum(result.costs[name]
                      for name in ("fees", "spread", "slippage", "impact"))

    cost = statistics.round_trip_cost(notionals, costs)
    turn = statistics.turnover(math.fsum(notionals), result.equity.tolist(),
                               result.provenance["periods_per_year"])
    return cost, turn, statistics.held(list(result.orders))


def test_each_balance_runs_as_the_venue_with_that_balance_would():
    balances = [10_000.0, 1_000_000.0, {"spot": 0.0, "margin": 50_000.0}]
    found    = probatum.capacity(POLICY, DATA, VENUE, balances)

    assert found.balances == tuple(balances)

    for k, balance in enumerate(balances):
        result = probatum.backtest(POLICY, DATA, VENUE.replace(balance=balance))
        cost, turnover, held = read(result, DATA)
        assert (found.total_return_pct[k], found.sharpe[k], found.impact[k]) == (
            result.stats["total_return_pct"], result.stats["sharpe"],
            result.costs["impact"])
        assert found.held[k]     == held
        assert found.cost_bps[k] == pytest.approx(cost, rel=1e-12)
        assert found.turnover[k] == pytest.approx(turnover, rel=1e-12)

    # the larger book pays more impact and has more of its orders held
    assert found.impact[1] > found.impact[0] and found.held[1] > found.held[0] > 0.0


def test_a_balance_of_two_wallets_reads_numpy_s_numbers_and_holds_still():
    given = {"spot": np.int64(0), "margin": np.float32(50_000.0)}
    found = probatum.capacity(POLICY, DATA, VENUE, [given])
    held  = probatum.capacity(POLICY, DATA, VENUE, [{"spot": 0.0, "margin": 50_000.0}])

    assert found == held and hash(found) == hash(held)
    assert found.balances[0] == {"spot": 0.0, "margin": 50_000.0}

    with pytest.raises(TypeError):
        found.balances[0]["margin"] = 1.0

    assert pickle.loads(pickle.dumps(found)) == found
    # the balance it holds opens an account again, and compares and hashes in any order
    assert probatum.capacity(POLICY, DATA, VENUE, found.balances) == found
    assert VENUE.replace(balance=found.balances[0]).to_dict()["balance"] == {
        "spot":   0.0,
        "margin": 50_000.0,
    }

    turned = probatum.Capacity([{"margin": 50_000.0, "spot": 0.0}], *found._args()[1:])
    assert turned == found and hash(turned) == hash(found)


def test_a_balance_the_account_cannot_open_with_is_refused_before_any_run():
    walk = walked(width=2)
    spot = probatum.Data(
        walk.bars[:, 0],
        symbol   = "C0USDT",
        market   = "spot",
        time     = walk.time,
        interval = "1h",
    )
    perp = probatum.Data(
        walk.bars[:, 1],
        symbol   = "C1USDT",
        market   = "linear",
        time     = walk.time,
        interval = "1h",
    )
    data = spot.join(perp)

    # a run before the refusal would raise the strategy's own error
    with pytest.raises(VenueError, match=r"^balances: 10000.0, one wallet's, and the "
                       r"data's spot and perpetual instruments use both; give each as "
                       r"\{\"spot\": s, \"margin\": m\}$"):
        probatum.capacity(Raising(), data, VENUE, [{"spot": 5_000.0, "margin": 5_000.0},
                                                   10_000.0])

    with pytest.raises(VenueError, match=r"^balances: \{'spot': 0.5, 'margin': 0.5\}, "
                       r"the cash of two wallets, and an inverse contract's account "
                       r"holds its coin in one; give each as a number of coins$"):
        probatum.capacity(Raising(), inverse(), VENUE,
                          [0.5, {"spot": 0.5, "margin": 0.5}])


def test_each_scale_multiplies_every_cost_the_venue_charges_for_trading():
    # two instruments, the second charged a fee of its own, which moves with the rest
    data  = walked(width=2)
    own   = {"linear:C1USDT": {"fees": {"maker": 0.0, "taker": 0.001}}}
    venue = VENUE.replace(instruments=own)
    found = probatum.cost_curve(trend(data), data, venue, scales=(0.0, 0.5, 3.0))

    for k, scale in enumerate((0.0, 0.5, 3.0)):
        moved = venue.replace(
            fees        = {"maker": 0.0002 * scale, "taker": 0.0005 * scale},
            fills       = {"spread": {"fixed": 2.0 * scale},
                           "slippage": {"fixed": 1.0 * scale},
                           "impact": {"linear": 0.5 * scale}},
            instruments = {"linear:C1USDT": {"fees": {"maker": 0.0,
                                                      "taker": 0.001 * scale}}})
        result = probatum.backtest(trend(data), data, moved)
        cost, turnover, _ = read(result, data)
        assert (found.total_return_pct[k], found.sharpe[k], found.num_trades[k]) == (
            result.stats["total_return_pct"], result.stats["sharpe"],
            result.stats["num_trades"])
        assert [getattr(found, name)[k] for name in ("fees", "spread", "slippage",
                                                     "impact")] == [
            result.costs[name] for name in ("fees", "spread", "slippage", "impact")]
        assert found.cost_bps[k] == pytest.approx(cost, rel=1e-12)
        assert found.turnover[k] == pytest.approx(turnover, rel=1e-12)

    assert found.cost_bps[0] == 0.0 and found.cost_bps[2] > found.cost_bps[1] > 0.0


def test_a_scale_multiplies_the_spread_a_venue_estimates():
    # the Specification's six bars of a spread estimate over and over, quantities and
    # volumes no fill comes near, so each fill is the same at every scale and pays its
    # estimate times the scale
    pattern = [(100.0, 100.5, 99.8, 100.2), (100.3, 100.6, 100.0, 100.0),
               (100.1, 100.6, 99.9, 100.4), (100.5, 100.8, 100.1, 100.1),
               (100.2, 100.7, 100.0, 100.5), (100.6, 100.9, 100.3, 100.3)]
    bars = np.array([[*bar, 1e6] for bar in pattern] * 50, dtype=np.float64)
    data = probatum.Data(
        bars,
        symbol   = "C0USDT",
        market   = "linear",
        time     = START + np.arange(len(bars)) * HOUR,
        interval = "1h",
    )

    venue    = probatum.Venue(
        fees  = {"maker": 0.0, "taker": 0.0},
        fills = {"spread": "estimated"},
    )
    quantity = np.where(np.arange(len(data)) % 40 == 0, 1.0, np.nan)
    quantity[np.arange(len(data)) % 40 == 20] = 0.0
    found = probatum.cost_curve(quantity, data, venue, scales=(1.0, 3.0),
                                sizing="quantity")

    assert found.spread[0] > 0.0
    assert found.spread[1]   == pytest.approx(3.0 * found.spread[0], rel=1e-12)
    assert found.cost_bps[1] == pytest.approx(3.0 * found.cost_bps[0], rel=1e-2)


def test_the_break_even_is_the_cost_where_the_return_reaches_zero():
    found = probatum.breakeven_bps(POLICY, DATA, VENUE, precision=0.01)
    curve = probatum.cost_curve(POLICY, DATA, VENUE, scales=(0.0, 1.0, 2.0, 4.0, 8.0))
    gains = [gain > 0.0 for gain in curve.total_return_pct]
    lost  = gains.index(False)

    assert gains[:lost] == [True] * lost and lost > 0
    assert curve.cost_bps[lost - 1] < found < curve.cost_bps[lost]


def test_a_policy_that_makes_nothing_has_no_break_even_and_one_past_the_ceiling_warns():
    flat = np.zeros(len(DATA))
    assert probatum.breakeven_bps(flat, DATA, VENUE) is None

    said = (r"ceiling: 0.25, and the policy still makes money there, paying \d+\.\d "
            r"basis points a round trip; its break-even lies past it, so give a higher "
            r"ceiling to find it")
    with pytest.warns(ProbatumWarning, match=said):
        found = probatum.breakeven_bps(POLICY, DATA, VENUE, ceiling=0.25)

    assert found == probatum.cost_curve(POLICY, DATA, VENUE, scales=(0.25,)).cost_bps[0]


def bisected(monkeypatch, cost):
    """The scales a break-even ran at, a fake run at each making money below 37 and
    paying ``cost(scale)`` basis points a round trip, and the break-even it found."""
    ran = []

    def backtested(*given, costs, **settings):
        ran.append(costs)
        return costs

    def measured(scale):
        return _costs.Measured(1.0 if scale < 37.0 else -1.0, 0.0, 1, {}, cost(scale),
                               0.0, 0.0)

    monkeypatch.setattr(_costs, "_backtested", backtested)
    monkeypatch.setattr(_costs, "_measured", measured)

    return ran, probatum.breakeven_bps(POLICY, DATA, VENUE)


def test_a_break_even_where_the_cost_jumps_warns_and_runs_no_scale_twice(monkeypatch):
    said = (r"^precision: 0.05, and the two runs that bracket the break-even, \S+ of a "
            r"scale apart, pay 370.0 and 410.0 basis points a round trip, the cost "
            r"jumping there; the break-even given is their mean, and a cost_curve at "
            r"scales about 37 shows the jump$")
    with pytest.warns(ProbatumWarning, match=said):
        ran, found = bisected(monkeypatch, lambda scale: 10.0 * scale
                              + 40.0 * (scale >= 37.0))

    assert len(ran) == len(set(ran)) and found == pytest.approx(390.0)


def test_a_break_even_past_which_nothing_fills_is_the_last_cost_paid(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, found = bisected(monkeypatch, lambda scale: 10.0 * scale if scale < 37.0
                            else math.nan)

    assert found == pytest.approx(370.0)


class Resting(probatum.Strategy):
    """A strategy that rests a buy 1% under each fortieth close while flat and a sell 1%
    over each twentieth after it while long, so its fills are a maker's."""


    def next(self, bar, account):
        if bar.i % 40 == 0 and account.position == 0.0:
            account.buy(10.0, limit=bar.close * 0.99)
        elif bar.i % 40 == 20 and account.position > 0.0:
            account.sell(10.0, limit=bar.close * 1.01)


def test_a_maker_s_fee_moves_with_the_scale_as_a_taker_s_does():
    venue = probatum.Venue(fees=FEES)
    found = probatum.cost_curve(Resting(), DATA, venue, scales=(3.0,))
    held  = probatum.backtest(Resting(), DATA,
                              venue.replace(fees={"maker": 0.0002 * 3.0,
                                                  "taker": 0.0005 * 3.0}))

    assert not any(fill["taker"] for fill in held.fills) and len(held.fills) > 4
    assert found.fees[0] == held.costs["fees"] > 0.0


def test_a_maker_s_rebate_is_the_venue_s_payment_and_stays_at_every_scale():
    venue = probatum.Venue(fees={"maker": -0.0001, "taker": 0.0005})
    found = probatum.cost_curve(Resting(), DATA, venue, scales=(0.0, 3.0))
    held  = probatum.backtest(Resting(), DATA, venue)

    assert not any(fill["taker"] for fill in held.fills)
    assert found.fees[0] == found.fees[1] == held.costs["fees"] < 0.0


class Raising(probatum.Strategy):
    """A strategy that raises at its first bar."""


    def next(self, bar, account):
        raise RuntimeError("the strategy ran")


class Counting(probatum.Strategy):
    """A strategy that keeps every bar it saw in a list and goes long on its tenth and
    flat on its twentieth, so a copy that kept another run's bars would trade
    otherwise."""


    def __init__(self):
        self.seen = []


    def next(self, bar, account):
        self.seen.append(bar.i)

        if len(self.seen) == 10:
            account.target(1.0)
        elif len(self.seen) == 20:
            account.target(0.0)


def test_each_level_runs_a_fresh_copy_of_a_strategy():
    given = Counting()
    found = probatum.capacity(given, DATA, VENUE, [10_000.0, 10_000.0])
    curve = probatum.cost_curve(given, DATA, VENUE, scales=(1.0, 1.0))

    assert found.total_return_pct[0] == found.total_return_pct[1] != 0.0
    assert curve.cost_bps[0] == curve.cost_bps[1] > 0.0
    assert given.seen == []


def test_an_inverse_contract_counts_what_it_traded_in_its_coin():
    data   = inverse()
    venue  = probatum.Venue(balance=0.5, fees=FEES, fills={"spread": {"fixed": 2.0}})
    policy = trend(data)
    found  = probatum.cost_curve(policy, data, venue, scales=(1.0,))
    cost, turnover, _ = read(probatum.backtest(policy, data, venue), data)

    assert found.cost_bps[0] == pytest.approx(cost, rel=1e-12)
    assert found.turnover[0] == pytest.approx(turnover, rel=1e-12)
    # five basis points of fee and one of half the spread each way
    assert found.cost_bps[0] == pytest.approx(12.0, rel=1e-3)


def test_a_sweep_shows_a_row_a_level_and_a_run_that_trades_nothing_reads_nan():
    found = probatum.capacity(POLICY, DATA, VENUE, [10_000.0, 1_000_000.0])
    frame = found.to_pandas()

    assert (frame.index.name, list(frame.index)) == ("balance", [10_000.0, 1_000_000.0])
    assert list(frame.columns) == ["total_return_pct", "sharpe", "impact", "held", "cost_bps",
                                   "turnover"]
    assert repr(found) == repr(frame)

    curve = probatum.cost_curve(np.zeros(len(DATA)), DATA, VENUE, scales=(1.0,))

    assert curve.to_pandas().index.name == "scale"
    assert math.isnan(curve.cost_bps[0]) and math.isnan(curve.to_pandas()["cost_bps"][1.0])


@pytest.mark.parametrize("call, error, said", [
    (lambda: probatum.capacity(POLICY, DATA, VENUE, [10_000.0], sizing="quantity"),
     ProbatumValueError, "sizing: \"quantity\", whose positions do not grow with the "
                         "balance; give weights, which capacity scales"),
    (lambda: probatum.capacity(POLICY, DATA, VENUE, 10_000.0), ProbatumTypeError,
     "balances: 10000.0; give the opening balances to run at, such as [10_000.0, "
     "100_000.0, 1_000_000.0]"),
    (lambda: probatum.capacity(POLICY, DATA, VENUE, []), ProbatumValueError,
     "balances: none; give the opening balances to run at"),
    (lambda: probatum.capacity(POLICY, DATA, VENUE, [-1.0]), VenueError,
     "balances: -1.0; give each opening cash as a number from 0, or {\"spot\": s, "
     "\"margin\": m} for an account that holds both wallets"),
    (lambda: probatum.capacity(POLICY, DATA, VENUE, ["big"]), ProbatumTypeError,
     "balances: 'big'; give each opening cash as a number from 0"),
    (lambda: probatum.cost_curve(POLICY, DATA, VENUE, scales=2.0), ProbatumTypeError,
     "scales: 2.0; give the multiples of the venue's costs to run at, numbers from 0, "
     "such as (0.0, 1.0, 2.0)"),
    (lambda: probatum.cost_curve(POLICY, DATA, VENUE, scales=()), ProbatumValueError,
     "scales: none; give the multiples of the venue's costs to run at"),
    (lambda: probatum.cost_curve(POLICY, DATA, VENUE, scales=(1.0, -0.5)),
     ProbatumValueError, "scales: -0.5; give multiples of the venue's costs from 0"),
    (lambda: probatum.cost_curve(POLICY, DATA, VENUE, scales=("high",)),
     ProbatumTypeError, "scales: 'high'; give multiples of the venue's costs, numbers "
                        "from 0"),
    (lambda: probatum.cost_curve(POLICY, DATA, "frictionless"), VenueError,
     "venue: frictionless charges no fee, spread, slippage or impact on C0USDT, so no "
     "multiple of its costs moves one; give a venue that charges some, such as "
     "probatum.venues.binance_perp"),
    (lambda: probatum.breakeven_bps(POLICY, DATA, probatum.Venue(fees={"maker": 0.0,
                                                                       "taker": 0.0})),
     VenueError, "venue: the venue charges no fee, spread, slippage or impact"),
    (lambda: probatum.cost_curve(POLICY, DATA, probatum.Venue(fees={"maker": -0.0001,
                                                                    "taker": 0.0})),
     VenueError, "venue: the venue charges no fee, spread, slippage or impact"),
    (lambda: probatum.breakeven_bps(POLICY, DATA, VENUE, ceiling=0.0),
     ProbatumValueError, "ceiling: 0.0; give the largest multiple of the venue's costs "
                         "to try, such as 100"),
    (lambda: probatum.breakeven_bps(POLICY, DATA, VENUE, precision="fine"),
     ProbatumTypeError, "precision: 'fine'; give the basis points the answer may be "
                        "off by, such as 0.05"),
])
def test_a_sweep_refuses_what_it_cannot_run_with_the_fix(call, error, said):
    with pytest.raises(error) as raised:
        call()

    assert str(raised.value).startswith(said)
    assert type(raised.value) is error
