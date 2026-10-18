"""A run is set beside three baselines on its own data and venue: holding cash, buying
and holding an equal share of every instrument, and the same policy with the costs of
trading removed and every rule kept, computed when first read. Buying and holding is
the run of the quantities it asks for, the run without costs is the run less its costs
where its fills cannot differ, a book sums its accounts' as it sums their equity, and a
Result pickled before they were read cannot give them.
"""

import copy
import math
import pickle

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import ProbatumAttributeError, ProbatumValueError, ProbatumWarning
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE, bars, pair
from test_a_result_hands_its_tables_to_any_arrow_reader import runs

COSTS = ("fees", "funding", "spread", "slippage", "impact")


def frame(rows, start="2024-01-01", volume=1e9):
    """Hourly bars from ``start``, each row its open, high, low and close."""
    index = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame([list(row) for row in rows],
                        columns=["open", "high", "low", "close"],
                        index=index).assign(volume=volume)


def walked(opens, closes):
    """Rows that open and close where given, their range the two."""
    return [(o, max(o, c), min(o, c), c) for o, c in zip(opens, closes)]


def held_rows(data, opening, width=None):
    """The quantities buying and holding asks for: an equal share of ``opening`` in
    each instrument at its first printed close, or on an inverse contract the contracts
    it is worth there, from that row on."""
    closes, available = data.closes, data.available
    width = width or closes.shape[1]
    rows  = np.full(closes.shape, np.nan)

    for k, instrument in enumerate(data.instruments):
        first = np.flatnonzero(available[:, k])[0]
        close = closes[first, k]
        rows[first:, k] = (opening * close / (width * instrument.contract)
                           if instrument.market == "inverse"
                           else opening / width / close)

    return rows


def test_the_costs_example_sets_the_run_beside_its_three_baselines():
    # the Specification's example: a buy of 0.1 at 60,000 sold at 60,400 that costs
    # 8.430997 in all, beside a sixth of a bitcoin bought and held at 60,009.00
    rows = [(60_000.0,) * 4, (60_000.0, 60_600.0, 59_900.0, 60_500.0),
            (60_400.0, 60_500.0, 60_300.0, 60_450.0)]
    venue = probatum.Venue(
        balance = 10_000.0,
        fees    = {"maker": 0.0, "taker": 0.0005},
        fills   = {"spread": {"fixed": 1.0}, "slippage": {"fixed": 1.0}},
        funding = {"every_bars": 1, "rate": 0.0001},
    )

    result = probatum.backtest(np.array([0.1, 0.0, np.nan]),
                               probatum.Data(frame(rows), symbol="BTCUSDT",
                                             market="linear"), venue, sizing="quantity")

    assert result.to_dict()["baselines"] is None

    table = result.baselines
    assert table.columns == ("bar", "time", "cash", "buy_and_hold", "frictionless")
    assert np.array_equal(table["time"], result.time)
    assert table["cash"].tolist() == [10_000.0] * 3
    assert table["buy_and_hold"] == pytest.approx([10_000.0, 10_075.82425, 10_066.483417],
                                                  abs=1e-6)
    assert table["frictionless"] == pytest.approx([10_000.0, 10_050.0, 10_040.0],
                                                  abs=1e-9)
    assert table["frictionless"][-1] - result.equity[-1] == pytest.approx(8.430997,
                                                                          abs=1e-9)
    assert result.to_dict()["baselines"]["cash"] == [10_000.0] * 3


@pytest.mark.parametrize("market, equity", [
    ("linear", [10_000.0, 10_289.9, 9_789.9]),
    ("spot", [10_000.0, 10_286.74, 9_792.19])])
def test_buying_and_holding_buys_the_quantity_its_share_buys_at_the_first_close(
        market, equity):
    # the Specification's example: closes of 100, 104 and 99 and opens of 100, 101
    # and 103 with a taker fee of 0.1%, where on spot the cash clamp buys 98.910990
    venue = probatum.Venue(kind=market, fees={"maker": 0.0, "taker": 0.001})
    data  = probatum.Data(
        frame(walked([100.0, 101.0, 103.0], [100.0, 104.0, 99.0])),
        symbol = "BTCUSDT",
        market = market,
    )

    result = probatum.backtest(np.full(3, np.nan), data, venue)

    assert result.baselines["buy_and_hold"] == pytest.approx(equity, abs=0.005)


def test_a_late_lister_is_bought_on_the_opening_balance_s_share():
    # A opens at 100, 101, 103 and 98 and closes at 100, 104, 99 and 102, and B prints
    # from row 2 at a close of 50, then opens at 51 and closes at 52
    a      = frame(walked([100.0, 101.0, 103.0, 98.0], [100.0, 104.0, 99.0, 102.0]))
    b      = frame(walked([50.0, 51.0], [50.0, 52.0]), start="2024-01-01 02:00")
    data   = probatum.Data({"A": a, "B": b}, market="linear")
    result = probatum.backtest(np.full((4, 2), np.nan), data, VENUE)

    assert result.baselines["buy_and_hold"].tolist() == [10_000.0, 10_150.0, 9_900.0,
                                                         10_150.0]


def test_buying_and_holding_is_the_run_of_the_quantities_it_asks_for():
    charged = VENUE.replace(fees={"maker": 0.0002, "taker": 0.0006},
                            fills={"spread": {"fixed": 3.0}, "slippage": {"fixed": 2.0},
                                   "impact": {"sqrt": 0.5}},
                            funding={"every": 8, "rate": 0.0001})
    late = probatum.Data({"BTCUSDT": bars(), "ETHUSDT": bars(level=10.0).iloc[7:]},
                         market="linear")

    cases = runs()
    cases["late"] = probatum.backtest(np.full((40, 2), np.nan), late, charged)

    for name, result in cases.items():
        record = result._record
        data, venue = record.data, record.venue
        opening = result.equity[0]
        asked   = probatum.backtest(held_rows(data, opening), data, venue,
                                    sizing="quantity")

        assert np.array_equal(result.baselines["buy_and_hold"], asked.equity), name


def test_an_inverse_long_bought_and_held_is_taken_over_near_a_halving():
    # 1 BTC of contracts worth 100 dollars at a maintenance rate of 0.4% from 60,000 is
    # 600 contracts, which a takeover closes at 60,000 * 1.004 / 2 = 30,120
    given = frame([(60_000.0,) * 4, (60_000.0,) * 4, (60_000.0, 60_000.0, 30_000.0,
                                                      31_000.0), (31_000.0,) * 4])
    given.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                       interval="1h", quote="USD", contract_size=100.0, warnings=[],
                       volume_unit="contract")
    venue = probatum.Venue(
        balance = 1.0,
        fees    = {"maker": 0.0, "taker": 0.0},
        tiers   = [{"up_to": 1e12, "leverage": 20.0, "rate": 0.004}],
    )

    result = probatum.backtest(np.full(4, np.nan), probatum.Data(given), venue)
    held   = result.baselines["buy_and_hold"]

    assert held[:2].tolist() == [1.0, 1.0] and held[2:].tolist() == [0.0, 0.0]

    shown = probatum.backtest(np.full(4, 600.0), probatum.Data(given), venue,
                              sizing="quantity")

    assert np.array_equal(held, shown.equity)
    assert shown.fills[-1]["price"] == pytest.approx(30_120.0, abs=1e-6)


def test_the_run_without_costs_is_the_run_less_its_costs_where_its_fills_are_its_own():
    # bybit's 1000TAGUSDT at its own fees beside BTCUSDT at the venue's, on-lot sizes
    # in the first tiers, a budget that holds one fill back a bar and funding every 4
    # bars: nothing the costs change moves a fill, so the gap is the costs; the spread
    # is fixed, as these bars' range estimates none
    venue = probatum.venues.bybit_perp.replace(funding={"every_bars": 4, "rate": 0.0001},
                                               fills={"spread": {"fixed": 2.0}})
    data = probatum.Data({
        "BTCUSDT":     bars().assign(volume=200.0),
        "1000TAGUSDT": bars().assign(volume=200.0),
    }, market="linear")

    rows = np.full((40, 2), np.nan)
    rows[3]  = [0.5, 30.0]
    rows[20] = [0.2, 10.0]
    rows[30] = [0.0, 0.0]

    result = probatum.backtest(rows, data, venue, sizing="quantity")

    clamps = result.orders.events("clamped")
    assert len(clamps) and not result.liquidations
    assert result.costs["dust"] == result.costs["liquidation"] == 0.0

    ledger = result.ledger
    paid   = np.array([math.fsum(column) for column in zip(*[ledger[name] for name in COSTS])])
    gap    = result.baselines["frictionless"] - result.equity
    scale  = 1e-9 * (1.0 + np.abs(result.equity).max())

    np.testing.assert_allclose(gap, np.cumsum(paid), rtol=0.0, atol=scale)
    assert all(result.costs[name] != 0.0 for name in COSTS)


def test_a_run_that_charges_nothing_is_its_own_run_without_costs():
    results = {name: result for name, result in runs().items() if name != "inverse"}

    # a tier table that caps the long holds the run without costs to it as well
    capped = VENUE.replace(tiers=[{"up_to": 500.0, "leverage": 2.0, "rate": 0.01}])
    rows   = np.full((40, 2), np.nan)
    rows[3] = [3.0, 0.0]

    results["capped"] = probatum.backtest(rows, pair(), capped)

    assert results["capped"].orders.events("clamped")
    for name, result in results.items():
        assert np.array_equal(result.baselines["frictionless"], result.equity), name


def half_a_bitcoin(venue, policy):
    """A long of 0.5 BTC its limit buys at 60,000 as a maker, so it costs nothing, and a
    bar that slides from an open of 41,000 to a low of 40,150, below the trigger of a
    maintenance rate of 0.5%, 40,201.01."""
    rows = [(60_100.0,) * 4, (60_100.0, 60_100.0, 59_900.0, 60_000.0),
            (41_000.0, 41_200.0, 40_150.0, 40_500.0), (40_500.0,) * 4]
    data = probatum.Data(frame(rows), symbol="BTCUSDT", market="linear")

    return probatum.backtest(policy, data, venue)


class Entry(probatum.Strategy):

    def next(self, bar, account):
        if bar.i == 0:
            account.buy(0.5, limit=60_000.0)


@pytest.mark.parametrize("model, kept", [("takeover", 0.0), ("book_then_backstop", 90.45)])
def test_a_forced_close_takes_what_it_takes_from_the_run_without_costs_too(model, kept):
    # the run and its baseline are both closed at 40,201.01; under takeover both lose
    # the 100.50 the wallet holds there, and on the book the baseline keeps all of it
    # where the run pays the sale's taker fee of 10.05
    venue = probatum.Venue(
        fees        = {"maker": 0.0, "taker": 0.0005},
        liquidation = model,
        backstop    = 2.0 / 3.0 if model != "takeover" else None,
        tiers       = [{"up_to": 1e12, "leverage": 20.0, "rate": 0.005}],
    )

    result = half_a_bitcoin(venue, Entry())

    assert result.liquidations == 1
    assert result.equity[-1]   == pytest.approx(kept, abs=0.005)
    assert result.baselines["frictionless"][-1] == pytest.approx(
        0.0 if model == "takeover" else 100.5, abs=0.005)


PLAYS = []


class Counted(probatum.Strategy):
    """Buys once its counter, set as it is built, reaches ``at``, and rests a buy limit
    at 95, which a later bar's low touches but does not trade through; each copy it
    plays is counted."""


    def __init__(self, at=3):
        self.at    = at
        self.count = 0


    def init(self, data):
        PLAYS.append(self)


    def next(self, bar, account):
        self.count += 1
        if self.count == self.at:
            account.buy(1.0)
            account.buy(1.0, limit=95.0)


def test_a_strategy_plays_again_once_the_copy_taken_at_the_call():
    through = VENUE.replace(fills={"limit": "through"}, tick=0.01)
    rows    = walked(100.0 + np.arange(12), 101.0 + np.arange(12))
    rows[8] = (107.0, 109.0, 95.0, 108.0)

    data     = probatum.Data(frame(rows), symbol="BTCUSDT", market="linear")
    strategy = Counted()

    PLAYS.clear()
    result = probatum.backtest(strategy, data, through)

    # the caller's instance moves after the call and the copy the run keeps was taken
    # before it did
    strategy.at = 5
    copies      = [copy.copy(result), copy.deepcopy(result), result]

    assert len(PLAYS) == 1

    tables = [each.baselines for each in copies]

    assert len(PLAYS) == 2 and PLAYS[-1] is not result.policy
    for table in tables:
        assert np.array_equal(table["frictionless"], result.equity)
    assert result.policy.count == len(result.time) - 1
    # the limit at 95 rested through the bar whose low touched it, as the rule says
    assert len(result.fills) == 1 and len(result.orders.events("filled")) == 1


def test_a_loop_the_caller_stepped_has_no_policy_to_play_again():
    data   = probatum.Data(bars(n=12), symbol="BTCUSDT", market="linear")
    venue  = VENUE.replace(intrabar="random")
    engine = probatum.Engine(data, venue, seed=7)

    engine.buy(1.0)
    for _ in range(4):
        engine.step()
    early = engine.result()

    while not engine.done():
        engine.step()
    late = engine.result()

    assert len(early.time) == 5 and len(late.time) == 12
    for result in (early, late):
        table = result.baselines
        assert np.isnan(table["frictionless"]).all()
        assert (table["cash"] == result.equity[0]).all()
    assert np.array_equal(early.baselines["buy_and_hold"],
                          late.baselines["buy_and_hold"][:5])

    # under the random setting a baseline draws from the start of the seed's stream
    again = probatum.Engine(data, venue, seed=7)
    while not again.done():
        again.step()

    assert np.array_equal(again.result().baselines["buy_and_hold"],
                          late.baselines["buy_and_hold"])


@pytest.mark.parametrize("spot, linear, balance", [
    (2, 1, {"spot": 7_000.0, "margin": 3_000.0}),
    (1, 2, {"spot": 7_000.0, "margin": 3_000.0}),
    # a universe of one kind's wallet takes the whole balance, which 1,900.9 * 3 / 3
    # rounds past
    (3, 0, {"spot": 1_900.9, "margin": 0.0}),
    (0, 3, {"spot": 0.0, "margin": 1_900.9})])
def test_two_wallets_hold_each_leg_s_share_of_the_balance(spot, linear, balance):
    names = [f"S{k}USDT" for k in range(spot)] + [f"L{k}USDT" for k in range(linear)]
    parts = [probatum.Data(bars(level=10.0 * (k + 1)), symbol=name,
                           market="spot" if k < spot else "linear")
             for k, name in enumerate(names)]
    data = parts[0]
    for part in parts[1:]:
        data = data.join(part)

    venue  = VENUE.replace(balance=balance, fees={"maker": 0.001, "taker": 0.001})
    result = probatum.backtest(np.full((40, len(names)), np.nan), data, venue)

    width, total = len(names), sum(balance.values())
    larger = max(spot, linear)
    share  = total if larger == width else total * larger / width
    split  = ({"spot": share, "margin": total - share} if spot > linear else
              {"spot": total - share, "margin": share})

    asked = probatum.backtest(held_rows(data, total), data,
                              venue.replace(balance=split), sizing="quantity")

    assert np.array_equal(result.baselines["buy_and_hold"], asked.equity)


STOP = {"now": False, "error": None}


class Stopped(probatum.Strategy):

    def next(self, bar, account):
        if STOP["now"]:
            raise STOP["error"]

        if bar.i == 2:
            account.buy(1.0)


def test_a_run_without_costs_that_raises_or_stops_leaves_its_column_missing():
    data = probatum.Data(bars(), symbol="BTCUSDT", market="linear")
    STOP.update(now=False)

    raised  = probatum.backtest(Stopped(), data, VENUE)
    stopped = probatum.backtest(Stopped(), data, VENUE)

    try:
        STOP.update(now=True, error=RuntimeError("the model went away"))
        with pytest.warns(ProbatumWarning,
                          match="raised RuntimeError: the model went away") as caught:
            table = raised.baselines

        assert caught.pop(ProbatumWarning).filename == __file__
        assert np.isnan(table["frictionless"]).all()
        assert not np.isnan(table["buy_and_hold"]).any()

        STOP.update(error=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            stopped.baselines
    finally:
        STOP.update(now=False)

    # the copy that stopped is gone, so it is not played a second time
    with pytest.warns(ProbatumWarning, match="stopped before it finished") as caught:
        table = stopped.baselines

    assert caught.pop(ProbatumWarning).filename == __file__
    assert np.isnan(table["frictionless"]).all()
    assert stopped.baselines is table


def test_a_result_pickled_before_its_baselines_were_read_cannot_give_or_save_them(
        tmp_path):
    result = probatum.backtest(np.full(40, 0.5), probatum.Data(
        bars(),
        symbol = "BTCUSDT",
        market = "linear",
    ), VENUE)

    early = pickle.loads(pickle.dumps(result))

    with pytest.raises(ProbatumAttributeError, match="not computed before this Result "
                                                     "was pickled; read "
                                                     "result.baselines before"):
        early.baselines
    assert early.to_dict()["baselines"] is None
    with pytest.raises(ProbatumValueError, match="so it wrote nothing"):
        early.save(tmp_path / "early")
    assert not (tmp_path / "early").exists()

    book = probatum.combine({"A": early, "B": result})

    with pytest.raises(ProbatumAttributeError, match="A was pickled before its "
                                                     "baselines"):
        book.baselines
    with pytest.raises(ProbatumValueError, match="A was pickled before"):
        book.save(tmp_path / "book")
    assert not (tmp_path / "book").exists()

    # save reads them first, and a Result pickled after they were read carries them
    result.save(tmp_path / "run")

    assert result._ready() is not None

    later = pickle.loads(pickle.dumps(result))

    assert later.baselines == result.baselines

    loaded = probatum.Result.load(tmp_path / "run")

    assert loaded.baselines == result.baselines


def test_a_book_sums_its_accounts_baselines_as_it_sums_their_equity(tmp_path):
    # the Specification's two accounts on the frictionless preset, which charges
    # nothing, so each runs without costs as it ran
    venue = probatum.venues.frictionless
    first = probatum.Data(
        frame(walked([100.0, 100.0, 101.0, 100.5],
                     [100.0, 101.0, 100.5, 102.0])),
        symbol = "COINUSDT",
        market = "linear",
    )
    hours  = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta([1, 3, 4], unit="h")
    second = probatum.Data(
        frame(walked([50.0, 50.0, 51.5], [50.0, 49.0, 52.0]))
        .set_axis(hours),
        symbol   = "COINUSDT",
        market   = "linear",
        interval = "1h",
    )
    a = probatum.backtest(np.array([100.0, np.nan, np.nan, np.nan]), first,
                          venue.replace(balance=10_000.0), sizing="quantity")
    b = probatum.backtest(np.array([100.0, 0.0, np.nan]), second,
                          venue.replace(balance=5_000.0), sizing="quantity")
    book = probatum.combine({"A": a, "B": b})

    assert book._parts is not None and book.to_dict()["baselines"] is None

    table = book.baselines

    assert book._parts is None
    assert "venue" not in table.columns and np.array_equal(table["time"], book.time)
    assert (table["cash"] == book.equity[0]).all()
    assert np.array_equal(table["frictionless"], book.equity)
    assert table["buy_and_hold"].tolist() == [15_000.0, 15_100.0, 15_050.0, 15_100.0,
                                              15_400.0]

    book.save(tmp_path / "book")

    assert probatum.Result.load(tmp_path / "book").baselines == table

    again = probatum.combine({"A": a, "B": b})

    assert again._parts is None and again.baselines == table

    looped = probatum.Engine(first, venue.replace(balance=10_000.0)).result()
    mixed  = probatum.combine({"A": looped, "B": b})

    assert np.isnan(mixed.baselines["frictionless"]).all()

    # three accounts add in the order given, as their equity does, to the bit
    idle = {name: probatum.backtest(np.full(4, np.nan), first,
                                    venue.replace(balance=balance))
            for name, balance in (("A", 0.1), ("B", 0.2), ("C", 0.3))}
    three = probatum.combine(idle)

    assert three.equity[0] != 0.3 + 0.2 + 0.1

    table = three.baselines

    assert (table["cash"] == three.equity[0]).all()
    assert table["buy_and_hold"][0] == table["frictionless"][0] == three.equity[0]
