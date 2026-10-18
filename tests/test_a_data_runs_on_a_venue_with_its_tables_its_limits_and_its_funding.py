"""A Data runs on a venue as the Engine's settings: each instrument takes the venue's
tables, keyed by its symbol, and the order limits the router gave it where its own
table gives none; the funding comes from the Data where the venue charges what the
data holds; a slice brings the bars before it for its window to read; and an account
the venue cannot hold is refused before it runs.
"""

import math
import re

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import DataError, ProbatumTypeError, Venue, VenueError
from probatum._venue import _run
from test_a_data_from_the_router_is_every_bar_it_asked_for import (Router, bars, fetch,
                                                                   settlements)

HOUR    = 3_600_000_000_000
PAIR    = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
FUNDING = {"BTCUSDT": settlements(), "ETHUSDT": settlements(rate=-0.0002)}


def frame(n=48, close=100.0):
    closes = close + np.arange(n, dtype=np.float64)
    index  = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="datetime")

    return pd.DataFrame({"open": closes, "high": closes + 1.0, "low": closes - 1.0,
                         "close": closes, "volume": 1_000.0 + np.arange(n)},
                        index=index)


def routed(symbol, market="linear", quote="USDT", exchange="binance", **attrs):
    """A frame that names itself as the router's do."""
    out = frame()
    out.attrs.update(dict(exchange=exchange, market_type=market, symbol=symbol,
                          interval="1h", quote=quote, warnings=[]), **attrs)
    return out


def test_a_universe_from_the_router_runs_on_a_preset_with_every_table_and_limit():
    data = fetch(Router(PAIR, FUNDING), PAIR, funding=True)
    run  = _run("binance_perp", data)
    btc, eth = run.settings["instruments"]
    table = probatum.venues.binance_perp.to_dict()["instruments"]["BTCUSDT"]

    assert (btc["market"], btc["tick"], btc["lot"]) == ("perp", table["tick"],
                                                       table["lot"])
    assert btc["tiers"][0][:3] == (table["tiers"][0]["up_to"],
                                   table["tiers"][0]["leverage"],
                                   table["tiers"][0]["rate"])
    assert (btc["min_notional"], btc["max_qty"], btc["reduce_only_exempt"]) == (
        5.0, 1_000.0, True)
    assert "fee_taker" not in btc
    assert (run.settings["fee_maker"], run.settings["fee_taker"]) == (0.0002, 0.0005)
    assert run.candles.shape == (48, 2, 5) and run.start == 0
    np.testing.assert_array_equal(run.settings["times"], data.time)
    assert run.settings["funding_times"].tolist() == data.funding["time"].tolist()
    assert run.settings["funding_instruments"]    == data.funding["instrument"].tolist()
    assert (run.settings["quote"], run.settings["limit"], run.settings["spread"],
            run.settings["intrabar"]) == (10_000.0, "through", "estimated",
                                          "pessimistic")

    engine = probatum._probatum.Engine(run.candles, **run.settings)
    engine.reset()
    engine.market_buy(1.0)
    for _ in range(12):
        state = engine.step()

    assert state["position"] == 1.0 and state["funding_paid"] != 0.0


def test_the_venue_s_own_limit_stands_over_the_router_s():
    data  = fetch(Router(PAIR), PAIR)
    venue = probatum.venues.binance_perp.replace(funding="none", min_notional=50.0)

    for spec in _run(venue, data).settings["instruments"]:
        assert (spec["min_notional"], spec["max_qty"]) == (50.0, 1_000.0)


def test_a_selection_runs_on_the_rows_its_instruments_printed():
    # ETH lists six hours after BTC and misses an hour later, so ETH alone has rows
    # of the wider clock it printed nothing on, which the run leaves out
    times, values = bars(48, close=10.0)
    eth = (np.delete(times[6:], [10]), np.delete(values[6:], [10], axis=0))

    with pytest.warns(UserWarning, match="ETHUSDT"):
        data = fetch(Router({"BTCUSDT": bars(48), "ETHUSDT": eth}),
                     ["BTCUSDT", "ETHUSDT"])

    alone = data["ETHUSDT"]["2024-01-01 12:00":]
    run   = _run(probatum.venues.binance_perp.replace(funding="none"), alone)

    assert run.candles.shape == (41, 1, 5) and run.start == 6

    printed = data["ETHUSDT"].available[:, 0]
    np.testing.assert_array_equal(run.settings["times"], data.time[printed])

    probatum._probatum.Engine(run.candles, **run.settings).reset()


def test_an_instrument_s_own_table_keeps_its_limit_over_the_router_s():
    data  = fetch(Router(PAIR), PAIR)
    venue = probatum.venues.binance_perp.replace(
        funding="none", instruments={"BTCUSDT": {"min_notional": 50.0}})

    btc, eth = _run(venue, data).settings["instruments"]

    assert (btc["min_notional"], btc["max_qty"]) == (50.0, 1_000.0)
    assert (eth["min_notional"], eth["max_qty"]) == (5.0, 1_000.0)


def test_a_slice_brings_the_bars_before_it_and_only_its_own_funding():
    data = fetch(Router(PAIR, FUNDING), PAIR, funding=True)
    late = data["2024-01-02":]
    run  = _run("binance_perp", late)

    assert run.start == 24 and run.candles.shape == (48, 2, 5)
    np.testing.assert_array_equal(run.candles[24:], late.bars)
    assert run.settings["funding_times"].tolist() == late.funding["time"].tolist()
    assert len(run.settings["funding_times"])     == 6


def test_a_venue_built_by_hand_runs_bars_from_a_file_on_its_plain_rules():
    spy = probatum.Data(frame(), symbol="SPY", market="spot")
    run = _run(Venue(fees={"maker": 0.0, "taker": 0.0005}), spy)

    assert run.candles.shape == (48, 1, 5)
    assert (run.settings["market"], run.settings["fee_taker"],
            run.settings["sizes"]) == ("spot", 0.0005, False)
    assert "liquidation" not in run.settings

    probatum._probatum.Engine(run.candles, **run.settings).reset()

    perp  = probatum.Data(frame(), symbol="BTCUSDT", market="linear")
    every = _run(Venue(funding={"every_bars": 8, "rate": 0.0001}), perp).settings

    assert (every["funding_interval"], every["funding_rate"]) == (8, 0.0001)

    drawn = _run(Venue(intrabar="random"), perp, seed=7)

    assert (drawn.settings["intrabar"], drawn.settings["seed"]) == ("random", 7)

    probatum._probatum.Engine(drawn.candles, **drawn.settings).reset()


def test_the_funding_a_run_reports_is_the_total_of_its_payments():
    # four thousand payments on a random walk, whose plain running sum lands six ulps
    # from their exact sum, where the compensated total the statistics report keeps
    # to it
    closes = 100.0 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0, 0.01, 4_000)))
    index  = pd.date_range("2024-01-01", periods=4_000, freq="h", tz="UTC", name="datetime")
    walk   = pd.DataFrame({
        "open":   closes,
        "high":   closes * 1.001,
        "low":    closes * 0.999,
        "close":  closes,
        "volume": 1e6,
    }, index=index)
    perp   = probatum.Data(walk, symbol="BTCUSDT", market="linear")
    venue  = Venue(funding={"every_bars": 1, "rate": 0.0001})
    result = probatum.backtest(np.ones(4_000), perp, venue, sizing="quantity")
    paid   = result.ledger["funding"].tolist()

    assert sum(payment != 0.0 for payment in paid) == 3_999
    assert result.stats["funding_paid"] == pytest.approx(math.fsum(paid), rel=2 ** -52,
                                                         abs=0.0)


def test_an_inverse_contract_runs_alone_on_its_own_bars_in_its_coin():
    router = Router({"BTCUSD": bars(48)}, {"BTCUSD": settlements()}, market="inverse")
    data   = probatum.Data.from_router(router, "binance", "inverse", "1h", ["BTCUSD"],
                                       start="2024-01-01", end="2024-01-02 23:00",
                                       funding=True)

    run = _run(probatum.venues.binance_inverse.replace(balance=1.0), data)

    assert run.candles.shape == (48, 5)
    assert (run.settings["market"], run.settings["contract"],
            run.settings["quote"]) == ("inverse", 100.0, 1.0)

    probatum._probatum.Engine(run.candles, **run.settings).reset()


def two(*frames):
    """A Data of the frames ``frames``, joined one by one."""
    data = probatum.Data(frames[0])
    for each in frames[1:]:
        data = data.join(probatum.Data(each))
    return data


@pytest.mark.parametrize("venue, data, error, says", [
    (Venue(kind="linear"), two(routed("BTCUSDT"), routed("ETHUSDT", exchange="bybit")),
     DataError, "data: instruments at binance and bybit, and an account is at one "
                "venue"),
    (Venue(kind="linear"), two(routed("BTCUSDT"), routed("BTCUSDC", quote="USDC")),
     DataError, "data: BTCUSDT settles in USDT and BTCUSDC settles in USDC, and an "
                "account's cash is one asset; run them on their own"),
    (Venue(balance={"spot": 1.0, "margin": 1.0}),
     two(routed("ETHBTC", market="spot", quote="BTC"), routed("BTCUSDT")), DataError,
     "data: BTCUSDT settles in USDT and ETHBTC is quoted in BTC"),
    (Venue(), two(routed("BTCUSDT", market="spot"), routed("BTCUSDT")), VenueError,
     "balance: 10000.0, one wallet's, and the data's spot and perpetual instruments "
     "use both; replace(balance={\"spot\": s, \"margin\": m})"),
    ("binance_perp", two(routed("BTCUSDT")), DataError,
     "funding: binance_perp charges the funding the data holds, and the data holds "
     "none for BTCUSDT; fetch the bars with funding=True, or "
     "replace(funding=\"none\")"),
    (probatum.venues.binance_perp.replace(funding="none"), two(routed("FOOUSDT")),
     VenueError, "venue: binance_perp gives FOOUSDT no lot and no tick, and its sizes "
                 "hold every order to them"),
    ("binance_perp", probatum.Data(frame(), market="linear"), DataError,
     "data: an unnamed instrument; pass probatum.Data(frame, symbol=...)"),
    (probatum.venues.binance_perp.replace(fills={"sizes": "off"}, funding="none"),
     probatum.Data(frame(), market="linear"), DataError,
     "data: an unnamed instrument; pass probatum.Data(frame, symbol=...)"),
    ("binance_perp", probatum.Data(frame(), symbol="", market="linear"), DataError,
     "data: an unnamed instrument; pass probatum.Data(frame, symbol=...)"),
    (probatum.venues.hyperliquid_perp.replace(funding="none"), two(routed("FOOUSDC")),
     VenueError, "venue: hyperliquid_perp gives FOOUSDC significant figures and no "
                 "decimals, the cap its price rule needs; give FOOUSDC its decimals in "
                 "instruments"),
    (probatum.venues.hyperliquid_perp.replace(funding="none", decimals=4, lot=1.0),
     two(routed("FOOUSDC")), VenueError,
     "venue: hyperliquid_perp books then backstops, and gives FOOUSDC no tier table "
     "that keeps maintenance; give FOOUSDC its tiers in instruments"),
    (probatum.venues.binance.replace(
        fills={"sizes": "off"}, balance={"spot": 1.0, "margin": 1.0}),
     two(routed("BTCUSD", market="inverse", quote="USD", volume_unit="contract",
                contract_size=100.0)), VenueError,
     "venue: binance lists spot and linear instruments, and BTCUSD is inverse"),
    ("binance_perp", two(routed("BTCUSDT", market="spot")), VenueError,
     "venue: binance_perp lists linear instruments, and BTCUSDT is spot; run it on a "
     "venue of its kind"),
    (Venue(), probatum.Data(frame(), symbol="BTCUSDT"), DataError,
     "data: BTCUSDT names no market, and venue is not of one kind; build the Data "
     "with market="),
    (Venue(), probatum.Data(frame(), symbol="BTCUSD", market="inverse"), DataError,
     "data: BTCUSD is an inverse contract whose bars do not give the dollars a "
     "contract is worth"),
    (Venue(balance={"spot": 1.0, "margin": 1.0}),
     two(routed("BTCUSD", market="inverse", quote="USD", volume_unit="contract",
                contract_size=100.0)), VenueError,
     "balance: {'spot': 1.0, 'margin': 1.0}, the cash of two wallets, and an inverse "
     "contract's account holds its coin in one; give the coin as a number"),
    ("binance_futures", two(routed("BTCUSDT")), VenueError,
     "venue: 'binance_futures' is not a preset; use one of"),
    (5, two(routed("BTCUSDT")), ProbatumTypeError,
     "venue: 5; pass a Venue, or a preset's name"),
    (Venue(), two(routed("BTCUSDT"))[:0], DataError,
     "data: no bars to run; give a Data that has"),
])
def test_an_account_the_venue_cannot_hold_is_refused_before_it_runs(venue, data, error,
                                                                    says):
    with pytest.raises(error, match=re.escape(says)):
        _run(venue, data)
