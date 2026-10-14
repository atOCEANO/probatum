"""Cross-version smoke tests for the probatum Engine over the single-env step.

These run against the installed abi3 wheel on each Python in the Docker gate, so
they double as the "works on most Python versions" check and the behavioral check
for the Python boundary: numpy candles in, a typed state out, and the cross-bar
no-lookahead rule holding through the binding.
"""

import re

import numpy as np
import pytest

import probatum


def test_version_is_single_sourced_from_the_build():
    # resolved from the wheel metadata (the Cargo.toml workspace version), not a
    # hardcoded string, so it can never drift from the built artifact
    assert probatum.__version__ != "0.0.0"
    assert re.match(r"^\d+\.\d+\.\d+", probatum.__version__)


def series():
    # open, high, low, close, volume
    return np.array(
        [
            [100.0, 160.0, 90.0, 150.0, 1000.0],
            [200.0, 260.0, 190.0, 250.0, 1000.0],
            [300.0, 360.0, 290.0, 350.0, 1000.0],
        ],
        dtype=np.float64,
    )


def engine(**overrides):
    params = dict(market="spot", quote=10_000.0, fee_taker=0.0, fee_maker=0.0)
    params.update(overrides)

    return probatum.Engine(series(), **params)


def test_reset_starts_flat_at_the_first_bar():
    e = engine()
    s = e.reset()

    assert s["tick_index"]  == 0
    assert s["position"]    == 0.0
    assert s["quote"]       == 10_000.0
    assert s["bar_open"]    == 100.0
    assert s["open_orders"] == []


def test_the_state_is_one_record_read_by_attribute_or_by_key():
    e     = engine()
    first = e.reset()
    e.limit_buy(1.0, 50.0)
    s = e.step()

    for name in ("tick_index", "position", "equity", "spot_cash", "margin_cash",
                 "bar_close", "funding_paid", "open_orders"):
        assert getattr(s, name) == s[name]

    assert s.open_orders[0]["price"] == 50.0
    assert first == engine().reset() and first != s

    with pytest.raises(KeyError):
        s["entry"]

    with pytest.raises(AttributeError):
        s.position = 1.0


def test_market_order_fills_on_the_next_open_not_the_decision_bar():
    e = engine()
    e.reset()
    oid = e.market_buy(1.0)

    assert isinstance(oid, int)

    s = e.step()

    assert s["tick_index"] == 1
    assert s["position"]   == 1.0
    assert s["avg_entry"]  == 200.0  # next bar open, not 150 or 250


def test_close_realizes_spot_pnl():
    e = engine()
    e.reset()
    e.market_buy(2.0)
    s1 = e.step()  # long 2 @ 200

    assert s1["position"] == 2.0

    e.close()
    s2 = e.step()  # closes at bar 2 open 300

    assert s2["position"] == 0.0
    assert s2["equity"]   == 10_200.0  # bought 2 @ 200, sold 2 @ 300, +200


def test_done_only_at_the_last_bar():
    e = engine()
    e.reset()

    assert not e.done()

    e.step()

    assert not e.done()

    e.step()

    assert e.done()


def test_limit_rests_in_open_orders_then_cancels():
    e = engine()
    e.reset()
    oid = e.limit_buy(1.0, 50.0)  # 50 is never reached (lows are 90/190/290)
    s1  = e.step()

    assert s1["position"]         == 0.0
    assert len(s1["open_orders"]) == 1

    order = s1["open_orders"][0]
    assert order["id"]    == oid
    assert order["kind"]  == "limit"
    assert order["side"]  == "buy"
    assert order["price"] == 50.0

    assert e.cancel(oid)

    s2 = e.step()

    assert s2["open_orders"] == []


def test_stats_none_without_reporting_and_dict_with():
    off = engine()
    off.reset()
    off.step()

    assert off.stats() is None

    on = engine(report=True)
    on.reset()
    on.step()
    st = on.stats(periods_per_year=365.0)

    assert isinstance(st, dict)
    assert "sharpe" in st
    assert "max_drawdown_pct" in st


def test_a_fill_is_logged_with_its_reference_and_what_it_paid_beyond_it():
    off = engine()
    off.reset()

    assert off.fills() is None

    e = engine(report=True, slippage_bps=10.0, fee_taker=0.0005,
               times=np.array([0, HOUR, 2 * HOUR], dtype=np.int64))
    e.reset()
    order = e.market_buy(1.0)
    e.step()
    [fill] = e.fills()

    assert (fill["bar"], fill["time"], fill["order"])    == (1, HOUR, order)
    assert (fill["side"], fill["reason"], fill["taker"]) == ("buy", "market", True)
    assert fill["reference"] == 200.0                  # the open it filled at
    assert fill["slippage"]  == pytest.approx(0.2)     # 10 bps of it
    assert fill["price"]     == pytest.approx(200.2)
    assert fill["fee"]       == pytest.approx(200.2 * 0.0005)
    assert fill["spread"] == fill["impact"] == fill["adjustment"] == 0.0


def test_the_ledger_has_a_row_per_bar_from_the_opening_balance():
    off = engine()
    off.reset()

    assert off.ledger() is None

    e = engine(report=True, fee_taker=0.001)
    e.reset()
    e.market_buy(2.0)
    e.step()                                        # 2 at 200, a fee of 0.4
    e.step()
    ledger = e.ledger()

    assert list(ledger["bar"]) == [0, 1, 2]
    assert ledger["time"] is None                   # no times, no clock
    assert ledger["equity"][0]      == 10_000.0
    assert list(ledger["fees"])     == pytest.approx([0.0, 0.4, 0.0])
    assert list(ledger["position"]) == [0.0, 2.0, 2.0]
    assert list(ledger["mark"])     == [150.0, 250.0, 350.0]


def test_the_order_log_follows_each_order_and_keeps_a_refused_one():
    off = engine()
    off.reset()

    assert off.order_log() is None

    e = engine(report=True, quote=1_000.0, max_open_orders=1)
    e.reset()
    rested = e.limit_buy(1.0, 90.0)

    assert e.limit_buy(1.0, 80.0) is None           # the book's one slot is taken
    assert e.replace(rested, price=95.0) == rested  # the same order, moved

    bought = e.market_buy(20.0)                     # 1,000 pays for 5 at 200
    e.step()
    log = [(event["order"], event["kind"]) for event in e.order_log()]

    assert log == [(rested, "placed"), (rested + 1, "refused"), (rested, "placed"),
                   (bought, "placed"), (bought, "clamped"), (bought, "filled"),
                   (bought, "expired")]

    clamp = e.order_log()[4]

    assert (clamp["size"], clamp["reason"]) == (5.0, "5 of the 20 asked fits")


def test_a_tier_table_holds_a_perp_where_its_margin_stops_carrying_it():
    # 20x up to 100,000 of notional and 10x after: 10,000 of margin holds a buy of
    # 5 at 60,000 at the bound
    bars = np.full((2, 5), 60_000.0)
    bars[:, 4] = 1e6
    e = probatum.Engine(
        bars,
        market    = "perp",
        quote     = 10_000.0,
        fee_taker = 0.0,
        tiers     = [(100_000.0, 20.0, 0.0, 0.0), (1e6, 10.0, 0.0, 0.0)],
    )

    e.reset()
    e.market_buy(5.0)

    assert e.step().position == pytest.approx(100_000.0 / 60_000.0)


@pytest.mark.parametrize("tiers, tiers_by, says", [
    ([(300_000.0, 125.0, 0.004, 0.0), (800_000.0, 100.0, 0.005, 250.0)], "notional",
     "continuity needs 300"),
    ([(10.0, 20.0, 0.05, 0.0)], "quantity", "reaches the initial rate of 0.05"),
    ([(10.0, 20.0, 0.0, 0.0), (5.0, 10.0, 0.0, 0.0)], "notional",
     "is not above the bound before it"),
    ([(10.0, 10.0, 0.0, 0.0), (50.0, 20.0, 0.0, 0.0)], "quantity",
     "is above the tier before it"),
    ([(10.0, 20.0, 0.02, 0.0), (50.0, 10.0, 0.01, 0.0)], "quantity",
     "a maintenance rate of 0.01 is below the tier before it"),
    ([(300_000.0, 125.0, 0.004, 30.0)], "notional", "in the first tier"),
    ([(10.0, 20.0, float("nan"), 0.0)], "quantity", "rate NaN is not a finite number"),
    ([(10.0, 20.0, -0.01, 0.0)], "quantity", "rate -0.01 is negative"),
    ([(10.0, 20.0, 0.0)], "notional", "3 numbers"),
    ([(10.0, 20.0, 0.0, 0.0)], "contracts", "is not a basis"),
    (None, "contracts", 'tiers_by: "contracts" is not a basis; use notional'),
])
def test_a_tier_table_the_loader_cannot_trust_is_refused(tiers, tiers_by, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        probatum.Engine(series(), market="perp", tiers=tiers, tiers_by=tiers_by)


def test_an_intrabar_setting_it_does_not_know_is_refused():
    says = 'intrabar: "neutral" is not a setting; use pessimistic, optimistic or random'

    for build in (probatum.Engine, lambda *a, **k: probatum.Batch(*a, num_envs=2, **k)):
        with pytest.raises(ValueError, match=re.escape(says)):
            build(series(), intrabar="neutral")


def test_a_random_setting_without_a_seed_is_refused():
    says = 'seed: none, and intrabar="random" draws from a stream named from it; pass seed='
    for build in (probatum.Engine, lambda *a, **k: probatum.Batch(*a, num_envs=2, **k)):
        with pytest.raises(ValueError, match=re.escape(says)):
            build(series(), intrabar="random")

    with pytest.raises(ValueError, match=re.escape(says)):
        probatum._probatum.run(series()[:, None, :], {"intrabar": "random"}, {})


def test_a_seed_that_is_not_a_whole_number_from_0_is_refused():
    for build in (probatum.Engine, lambda *a, **k: probatum.Batch(*a, num_envs=2, **k)):
        for seed, shown in ((-1, "-1"), (1.5, "1.5"), ("7", "'7'")):
            says = f"seed: {shown}; give a whole number from 0"
            with pytest.raises(ValueError, match=re.escape(says)):
                build(series(), intrabar="random", seed=seed)


def test_a_random_bar_draws_one_way_for_a_seed_and_both_ways_over_seeds():
    # a long of 1 from 100 meets a bar reaching both its stop at 90 and its target
    # at 110, and each seed's stream books one of the two, the same one every run
    candles = np.array([
        [100.0, 100.0 + reach, 100.0 - reach, 100.0, 1000.0]
        for reach in (1.0, 1.0, 20.0, 1.0)
    ])
    script = {
        0: [("buy", {"size": 1.0})],
        1: [
            ("sell", {"size": 1.0, "stop": 90.0, "reduce_only": True}),
            ("sell", {"size": 1.0, "limit": 110.0, "reduce_only": True}),
        ],
    }

    def exit_at(seed):
        venue = dict(fee_taker=0.0, fee_maker=0.0, intrabar="random", seed=seed)
        return probatum._probatum.run(candles, venue, script)["trades"][0]["exit"]

    exits = [exit_at(seed) for seed in range(32)]

    assert exits      == [exit_at(seed) for seed in range(32)]
    assert set(exits) == {90.0, 110.0}
    # as wide as the entropy NumPy draws for a SeedSequence given none
    assert exit_at(2**128 - 1) in (90.0, 110.0)


@pytest.mark.parametrize("weights", [0.5, [0.5], np.array([0.5]), {0: 0.5}])
def test_a_target_takes_one_number_one_weight_each_or_a_dict(weights):
    # half of 10,000 at the close of 150, bought at the next open
    e = engine(report=True)
    e.reset()
    [leg] = e.target(weights)

    assert e.step().position == pytest.approx(0.5 * 10_000.0 / 150.0)
    assert [fill["order"] for fill in e.fills()] == [leg]


@pytest.mark.parametrize("weights, tolerance, says", [
    (-0.5, 0.0, "weights: -0.5 for instrument 0, which trades spot and cannot be short; "
                "trade a perpetual instead"),
    ([0.5, 0.5], 0.0, "weights: 2 for the 1 the engine holds; give one per instrument"),
    ({3: 0.5}, 0.0, "weights: 3 is not an instrument the engine holds"),
    (float("nan"), 0.0, "weights: NaN for instrument 0 is not a weight; give a finite number"),
    ("half", 0.0, "weights: 'half' is not a weight; give a number, one per instrument"),
    (0.5, -0.1, "tolerance: -0.1 is not a fraction of equity; give one from 0"),
])
def test_a_target_the_engine_cannot_take_is_refused(weights, tolerance, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        engine().target(weights, tolerance=tolerance)


def test_a_universe_reads_an_array_of_weights_one_per_instrument():
    # a quarter of 10,000 on each of two perps, at closes of 150 and 15, the second
    # trading the first's volume so no fill is held to a bar's
    pair = np.stack([series(), series() / 10.0], axis=1)
    pair[:, 1, 4] = 1_000.0

    venue  = dict(market="perp", fee_taker=0.0, leverage=0.0)
    played = probatum._probatum.run(pair, venue, {
        0: [("target", {"weights": np.array([0.25, 0.25])})],
    })

    sizes = sorted((fill["instrument"], fill["size"]) for fill in played["fills"])
    assert sizes == [(0, pytest.approx(2_500.0 / 150.0)), (1, pytest.approx(2_500.0 / 15.0))]


def test_a_target_in_a_script_books_rebalances_and_the_drift_they_leave():
    played = probatum._probatum.run(series(), dict(fee_taker=0.0), {
        0: [("target", {"weights": 0.5})],
    })

    assert [fill["reason"] for fill in played["fills"]] == ["rebalance"]

    placed = [event["reason"] for event in played["order_log"] if event["kind"] == "placed"]
    assert placed == ["rebalance"]

    drift, weight = played["ledger"]["drift"], played["ledger"]["weight"]
    assert np.isnan(drift[0])
    assert drift[1:] == pytest.approx(0.5 - weight[1:])

    with pytest.raises(ValueError, match=re.escape("script[0][0]: target has no weights")):
        probatum._probatum.run(series(), None, {0: [("target", {})]})


def test_a_script_played_in_one_call_records_what_stepping_by_hand_records():
    venue  = dict(market="perp", quote=10_000.0, fee_taker=0.0006, leverage=5.0)
    played = probatum._probatum.run(series(), venue, {
        0: [("buy", {"size": 2.0}), ("sell", {"size": 1.0, "limit": 255.0})],
        1: [("close", {})],
    })

    e = probatum.Engine(series(), report=True, **venue)
    e.reset()
    e.market_buy(2.0)
    e.limit_sell(1.0, 255.0)
    e.step()
    e.close()
    e.step()

    assert played["fills"]     == e.fills()
    assert played["order_log"] == e.order_log()
    assert played["trades"]    == e.trades()

    by_hand = e.ledger()
    assert played["ledger"].keys() == by_hand.keys()
    for name, column in by_hand.items():
        np.testing.assert_array_equal(played["ledger"][name], column)


def test_a_script_names_an_order_by_its_placement_or_as_the_oldest_resting():
    # the first bid is cancelled by its placement and the replace moves the third
    # into the slot that frees, so the oldest still resting is the second
    bids   = [("buy", {"size": 1.0, "limit": price}) for price in (50.0, 40.0, 30.0)]
    played = probatum._probatum.run(series(), None, {
        0: bids,
        1: [("cancel", {"order": 0}), ("replace", {"order": 2, "price": 35.0}),
            ("cancel", {"order": "oldest"})],
    })

    log       = played["order_log"]
    placed    = [event["order"] for event in log if event["kind"] == "placed"]
    cancelled = [event["order"] for event in log if event["kind"] == "cancelled"]

    assert cancelled == [placed[0], placed[1]]

    # with one slot the second bid is refused, and naming it cancels nothing
    played = probatum._probatum.run(series(), {"max_open_orders": 1}, {
        0: bids[:2],
        1: [("cancel", {"order": 1})],
    })

    assert [event["kind"] for event in played["order_log"]] == ["placed", "refused"]


def test_a_bracket_arms_on_its_entry_and_its_first_fill_cancels_the_other_leg():
    # placed beside its entry, the legs arm as sells when the entry fills at 200,
    # and the bar's high of 260 reaches the take-profit at 255, which closes the
    # long and cancels the stop
    e = engine(report=True)
    e.reset()
    e.market_buy(1.0)
    stop, take_profit = e.bracket(stop=150.0, take_profit=255.0)
    s = e.step()

    assert (s.position, s.open_orders) == (0.0, [])

    fills = [(fill["order"], fill["price"], fill["reason"]) for fill in e.fills()]
    assert fills[1:] == [(take_profit, 255.0, "take_profit")]

    log = [(event["order"], event["kind"], event["reason"]) for event in e.order_log()]
    assert (stop, "armed", "stop") in log
    assert log[-1] == (stop, "cancelled", "its OCO partner filled")


def test_a_bracket_leg_has_no_side_until_it_arms_and_no_size_of_its_own():
    e = engine()
    e.reset()
    e.bracket(stop=50.0)
    [leg] = e.step().open_orders

    assert (leg["side"], leg["size"], leg["remaining"]) == (None, None, None)

    with pytest.raises(ValueError, match="bracket has no stop and no take_profit; give"):
        e.bracket()

    with pytest.raises(ValueError, match="take_profit must be a finite number, got NaN"):
        e.bracket(take_profit=float("nan"))


def test_a_bracket_in_a_script_is_two_placements_its_stop_then_its_take_profit():
    # the entry is placement 0, the stop 1 and the take-profit 2: the take-profit
    # is cancelled by its placement and the stop moved up to 295, where the last
    # bar's low of 290 fills it
    played = probatum._probatum.run(series(), None, {
        0: [("buy", {"size": 1.0}), ("bracket", {"stop": 150.0, "take_profit": 400.0})],
        1: [("cancel", {"order": 2}), ("replace", {"order": 1, "trigger": 295.0})],
    })

    log = played["order_log"]
    stop, take_profit = [event["order"] for event in log if event["kind"] == "placed"][1:3]
    fills = [(fill["order"], fill["price"], fill["reason"]) for fill in played["fills"]]

    assert fills[1:] == [(stop, 295.0, "stop")]

    cancelled = [(event["order"], event["reason"]) for event in log
                 if event["kind"] == "cancelled"]
    assert cancelled == [(take_profit, "cancelled by the account")]


def test_a_universe_with_an_instrument_nobody_trades_records_what_the_other_alone_does():
    # a second instrument printing every row and never traded changes nothing the
    # first records, and the ledger gives each instrument a column of its own
    venue  = dict(market="perp", quote=10_000.0, fee_taker=0.0006, leverage=5.0)
    script = {0: [("buy", {"size": 2.0, "instrument": 0}),
                  ("sell", {"size": 1.0, "limit": 255.0, "instrument": 0})],
              1: [("close", {"instrument": 0})]}
    pair   = np.stack([series(), series() / 10.0], axis=1)
    alone  = probatum._probatum.run(series(), venue, script)
    played = probatum._probatum.run(pair, venue, script)

    for record in ("fills", "order_log", "trades"):
        assert played[record] == alone[record]

    for name, column in alone["ledger"].items():
        mine = played["ledger"][name]
        if name in ("position", "entry", "mark", "notional", "unrealized", "weight", "drift"):
            assert mine.shape == (3, 2)
            mine = mine[:, 0]
        np.testing.assert_array_equal(mine, column)


def test_an_instrument_with_no_bar_on_a_row_waits_for_its_next():
    # the second instrument prints nothing on the second row, so a buy decided at
    # the first close fills at its third row's open, on instrument 1
    pair = np.stack([series(), series() / 10.0], axis=1)
    pair[1, 1] = np.nan

    played = probatum._probatum.run(pair, dict(market="spot"),
                                    {0: [("buy", {"size": 1.0, "instrument": 1})]})

    [fill] = played["fills"]
    assert (fill["bar"], fill["instrument"], fill["price"]) == (2, 1, 30.0)
    assert played["ledger"]["mark"][1, 1] == 15.0


def test_a_universe_s_instruments_take_the_venue_s_tiers_by_where_they_name_none():
    # by quantity the 20x tier carries its bound of 10 and the 5x tier only 2.5
    # at 200 on 100 of margin, so a buy of 15 is held at 10; by notional it would
    # be held at 2.5
    tiers = [(10.0, 20.0, 0.0, 0.0), (1e9, 5.0, 0.0, 0.0)]
    pair  = np.stack([series(), series() / 10.0], axis=1)
    venue = dict(market="perp", quote=100.0, fee_taker=0.0, tiers_by="quantity",
                 instruments=[{"tiers": tiers}, {}])

    played = probatum._probatum.run(
        pair, venue, {0: [("buy", {"size": 15.0, "instrument": 0})]})

    assert played["fills"][0]["size"] == 10.0


def test_a_universe_of_one_keeps_an_instrument_column_per_instrument():
    played = probatum._probatum.run(series()[:, None, :], None, {})

    assert played["ledger"]["position"].shape == (3, 1)
    assert played["ledger"]["equity"].shape   == (3,)


@pytest.mark.parametrize("change, venue, script, says", [
    (lambda c: c.__setitem__((1, 0, 2), np.nan), {}, {},
     "row 1 gives instrument 0 NaN in some of its five values"),
    (lambda c: c.__setitem__((1, 0, 1), np.inf), {}, {},
     "row 1 gives instrument 0 a value of inf"),
    (lambda c: c.__setitem__((1, slice(None)), np.nan), {}, {},
     "row 1 holds no instrument's bar"),
    (None, {}, {0: [("buy", {"size": 1.0})]}, "buy names no instrument"),
    (None, {}, {0: [("buy", {"size": 1.0, "instrument": 2})]},
     "instrument 2 is not one the engine holds"),
    (None, {"instruments": [{"market": "perp"}]}, {}, "instruments: 1 for the 2"),
    (None, {"instruments": [{"market": "spot"}, {"market": "perp"}]}, {},
     "quote: 10000 funds one wallet and this universe uses both"),
    (None, {"instruments": [{"kind": "spot"}, {}]}, {}, '"kind" is not a setting'),
    (None, {"instruments": [{1: "spot"}, {}]}, {}, "1 is not a setting's name"),
    (None, {"instruments": [{"market": 1}, {}]}, {},
     "instruments[0]: market 1 is not a string"),
    (None, {"instruments": [{}, {"tiers": 3.0}]}, {},
     "instruments[1]: tiers 3.0 is not a list of rows"),
    (None, {"instruments": [{"tiers_by": "contracts"}, {}]}, {},
     'instruments[0]: tiers_by "contracts" is not a basis; use notional or quantity'),
    (None, {"funding_instruments": [None]}, {}, "given without funding_times"),
    (None, {"funding_instruments": [-1]}, {}, "row 0 names instrument -1, of 2"),
    (None, {"fee": 0.001}, {}, "Engine.__new__() got an unexpected keyword argument 'fee'"),
])
def test_a_universe_the_engine_cannot_play_is_refused(change, venue, script, says):
    pair = np.stack([series(), series() / 10.0], axis=1)
    if change is not None:
        change(pair)

    error = TypeError if "keyword" in says else ValueError
    with pytest.raises(error, match=re.escape(says)):
        probatum._probatum.run(pair, venue, script)


def test_a_universe_s_book_is_bounded_in_all():
    # each instrument may ask 4,096 slots, and 257 of them would allocate past the
    # bound up front
    with pytest.raises(ValueError, match="is 1052672 slots, past 1048576; lower it"):
        probatum._probatum.run(np.ones((3, 257, 5)), {"max_open_orders": 4096}, {})


def universe_pair():
    """Two instruments, the second printing nothing on the second row."""
    pair = np.stack([series(), series() / 10.0], axis=1)
    pair[1, 1] = np.nan
    return pair


def test_a_universe_engine_shows_its_candles_with_an_instrument_axis():
    pair = universe_pair()
    eng  = probatum.Engine(pair)

    eng.reset()
    eng.step()

    assert eng.shape == (3, 2, 5)
    np.testing.assert_array_equal(eng.data, pair)
    np.testing.assert_array_equal(eng.closes, pair[:, :, 3])

    window = eng.observation(5)
    np.testing.assert_array_equal(window, pair[:2])
    assert not window.flags.writeable


def test_a_universe_batch_records_for_each_env_what_its_own_run_does():
    # each env targets weights of its own, the second instrument's leg waiting for
    # its next bar, and records what a run of those targets does
    pair  = universe_pair()
    venue = dict(market="perp", quote=10_000.0, fee_taker=0.0006, leverage=5.0)
    rows  = ([0.5, -0.5], [1.0, 0.25])
    batch = probatum.Batch(pair, 2, report=True, **venue)
    batch.reset_all()
    batch.target(np.array(rows))
    batch.advance()

    assert batch.observe(2).shape == (2, 2, 2, 5)
    np.testing.assert_array_equal(batch.observe(2)[0, 1, 1], np.zeros(5))

    batch.advance()
    snap = batch.snapshot()

    assert snap["positions"].shape == snap["marks"].shape == (2, 2)

    for env, weights in enumerate(rows):
        script = {0: [("target", {"weights": weights})]}
        played = probatum._probatum.run(pair, venue, script)
        mine   = batch.records(env)

        # an open round trip has no exit, NaN, which repr compares as the same
        assert mine["fills"]        == played["fills"]
        assert repr(mine["trades"]) == repr(played["trades"])
        for name, column in played["ledger"].items():
            np.testing.assert_array_equal(mine["ledger"][name], column)
        np.testing.assert_array_equal(snap["positions"][env],
                                      played["ledger"]["position"][-1])

    for env in (2, -1):
        with pytest.raises(ValueError, match=f"env: {env} of 2; name one counted"):
            batch.records(env)


@pytest.mark.parametrize("market, weights, says", [
    ("perp", np.zeros((3, 2)),
     "weights: shape (3, 2) for 2 envs of 2 instruments; give (2, 2)"),
    ("perp", np.zeros(2),
     "weights: an array of 1 dimensions for 2 envs of 2 instruments; give (2, 2)"),
    ("perp", [[0.5, 0.0], [0.5, 0.0]],
     "weights: a list; give a (2, 2) float64 array, one row per env"),
    ("perp", np.zeros((2, 2), dtype=np.int64),
     "weights: an array of int64; give float64 values"),
    ("perp", np.array([[0.5, np.nan], [0.0, 0.0]]),
     "weights: NaN for instrument 1 of env 0; give finite weights"),
    ("spot", np.array([[0.5, 0.0], [-0.5, 0.0]]),
     "weights: -0.5 for instrument 0 of env 1, which trades spot and cannot be short"),
])
def test_a_batch_refuses_weights_it_cannot_place_before_any_env_trades(market, weights,
                                                                          says):
    batch = probatum.Batch(universe_pair(), 2, market=market)
    batch.reset_all()

    with pytest.raises(ValueError, match=re.escape(says)):
        batch.target(weights)

    batch.advance()

    assert not batch.snapshot()["positions"].any()


@pytest.mark.parametrize("candles, settings, says", [
    (series(), {"instruments": [{}]},
     "instruments: given with one instrument's (T, 5)"),
    (np.zeros(5), {},
     "candles: an array of 1 dimensions; pass (T, 5) for one instrument"),
    (series().tolist(), {},
     "candles: a list; pass a float64 array, (T, 5) for one instrument"),
    (series().astype(np.int64), {},
     "candles: an array of int64; pass float64 values"),
    (np.zeros((0, 5)), {}, "candles: no rows; give at least one bar"),
    (np.zeros((3, 2, 4)), {},
     "candles: 4 values to a bar; give 5, open, high, low, close and volume"),
    (np.zeros((0, 2, 5)), {},
     "candles: 0 rows of 2 instruments; give at least one of each"),
])
def test_candles_the_engine_cannot_read_are_refused(candles, settings, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        probatum.Engine(candles, **settings)


def test_an_empty_batch_over_a_universe_keeps_its_instrument_axis():
    batch = probatum.Batch(universe_pair(), 0)

    assert batch.observe(2).shape == (0, 2, 2, 5)
    assert batch.snapshot()["positions"].shape == (0, 2)


@pytest.mark.parametrize("script, says", [
    ({0: [("hold", {})]}, '"hold" is not a verb'),
    ({0: [("buy", {"size": 1.0, "price": 99.0})]}, "buy takes no price"),
    ({0: [("buy", {1: 1.0})]}, "1 is not an argument's name"),
    ({0: [("buy", {"size": 1.0, "limit": 99.0, "stop": 98.0})]},
     "; give one or the other"),
    ({0: [("buy", {"size": 1.0, "limit": float("nan")})]}, "limit NaN is not a price"),
    ({0: [("cancel", {"order": 0})]}, "is not one of the 0 placements before it"),
    ({0: [("buy", {"size": 1.0}), ("cancel", {"order": "1"})]},
     "order '1' is not one of the 1 placements before it"),
    ({0: [("transfer", {"amount": -1.0})]}, "amount: -1 is negative"),
    ({0: [("bracket", {})]}, "bracket has no stop and no take_profit; give either or both"),
    ({0: [("bracket", {"stop": 99.0, "size": 1.0})]}, "bracket takes no size"),
    ({0: None}, "bar 0 holds None; give it a list of actions"),
    ({2: [("close", {})]}, "bar 2 has no bar after it to fill on"),
    ({2**64 - 1: [("close", {})]}, "has no bar after it to fill on"),
])
def test_a_script_the_engine_cannot_play_is_refused(script, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        probatum._probatum.run(series(), None, script)


def test_observation_is_a_zero_copy_readonly_view():
    e = engine()
    e.reset()  # tick 0
    obs = e.observation(1)

    assert obs.shape       == (1, 5)
    assert obs.dtype       == np.float64
    assert obs[0].tolist() == [100.0, 160.0, 90.0, 150.0, 1000.0]  # bar 0
    assert obs.flags.owndata is False
    assert obs.flags.writeable is False

    with pytest.raises(ValueError):
        obs[0, 0] = 1.0


def test_observation_grows_with_tick_and_clamps_at_warmup():
    e = engine()
    e.reset()  # tick 0

    assert e.observation(5).shape == (1, 5)  # only one bar exists yet, clamped

    e.step()  # tick 1
    obs = e.observation(2)

    assert obs.shape       == (2, 5)
    assert obs[0].tolist() == [100.0, 160.0, 90.0, 150.0, 1000.0]   # bar 0
    assert obs[1].tolist() == [200.0, 260.0, 190.0, 250.0, 1000.0]  # bar 1


def test_observation_outlives_the_dropped_engine():
    import gc

    e = engine()
    e.reset()
    e.step()  # tick 1

    obs      = e.observation(2)
    expected = [
        [100.0, 160.0, 90.0, 150.0, 1000.0],
        [200.0, 260.0, 190.0, 250.0, 1000.0],
    ]

    del e  # the view must keep the engine (and its Arc buffer) alive
    gc.collect()

    assert obs.tolist() == expected


def test_batch_matches_single_engines_and_tracks_done():
    n = 4
    b = probatum.Batch(series(), num_envs=n, market="spot", fee_taker=0.0, fee_maker=0.0)

    assert len(b)     == n
    assert b.num_envs == n

    b.reset_all()

    assert not b.done()

    # each env buys a different size, applied as a batched action array
    actions = np.array([0.1 * (i + 1) for i in range(n)], dtype=np.float64)
    states  = b.step_all(actions)

    assert len(states) == n
    for i, s in enumerate(states):
        # fills at bar 1 open (200), so position is exactly the action size
        assert s["tick_index"] == 1
        assert abs(s["position"] - 0.1 * (i + 1)) < 1e-12
        assert s["avg_entry"] == 200.0

    b.step_all()  # step with no new orders

    assert b.done()


def test_batch_step_without_actions_holds_position():
    b = probatum.Batch(series(), num_envs=2, fee_taker=0.0, fee_maker=0.0)
    b.reset_all()

    b.step_all(np.array([1.0, 2.0], dtype=np.float64))
    states = b.step_all()  # no actions: positions carry, no new fills

    assert states[0]["position"] == 1.0
    assert states[1]["position"] == 2.0


def test_batch_rejects_mismatched_action_length():
    b = probatum.Batch(series(), num_envs=3)
    b.reset_all()

    with pytest.raises(ValueError):
        b.step_all(np.array([1.0, 2.0], dtype=np.float64))  # length 2 != 3 envs


def test_batch_refuses_offsets_it_cannot_start_from():
    b = probatum.Batch(series(), num_envs=2)

    with pytest.raises(ValueError, match="offsets: -1 for env 1; give a row from 0"):
        b.reset_at_each(np.array([0, -1]))

    with pytest.raises(ValueError, match="offsets: 3 for 2 envs; give one row per env"):
        b.reset_at_each(np.array([0, 1, 2]))

    with pytest.raises(ValueError, match="offsets: a strided array; give a contiguous"):
        b.reset_at_each(np.array([0, 9, 1, 9])[::2])


def test_batch_set_features_rejects_a_short_matrix():
    # features must carry one row per candle; a shorter matrix would index out of
    # bounds in the gather, so set_features rejects it rather than panicking across
    # the FFI
    b = probatum.Batch(series(), num_envs=2)  # series() has 3 bars
    b.reset_all()

    with pytest.raises(ValueError):
        b.set_features(np.zeros((2, 4), dtype=np.float64))  # 2 rows != 3 bars


class BuyAndHold:
    """Buys once on the first bar, then holds, through the engine's own API."""


    def __init__(self):
        self.entered = False


    def next(self, state, engine):
        if not self.entered:
            engine.market_buy(1.0)
            self.entered = True


class RecordsInit:

    def __init__(self):
        self.inited = False


    def init(self, engine):
        self.inited = True


    def next(self, state, engine):
        pass


def test_run_drives_a_python_strategy_that_places_orders():
    e     = engine()
    final = e.run(BuyAndHold())

    assert final["tick_index"] == 2    # drove to the last bar
    assert final["position"]   == 1.0  # bought 1 at bar 1 open, held
    # spot: quote 10000 - 200 spent, base 1 marked at the last close 350
    assert final["equity"] == 10_150.0


def test_run_calls_init_when_present():
    strat = RecordsInit()
    engine().run(strat)

    assert strat.inited is True


class BuyThenClose:
    """Buys on the first bar, then closes once it holds a position."""


    def next(self, state, engine):
        if state["tick_index"] == 0:
            engine.market_buy(1.0)
        elif state["position"] != 0.0:
            engine.close()


def test_run_with_reporting_yields_stats():
    e = engine(report=True)
    e.run(BuyAndHold())
    st = e.stats()

    assert isinstance(st, dict)
    assert "total_return_pct" in st
    assert st["total_return_pct"] > 0.0  # bought and held into a rising series


def test_closed_trades_flow_into_stats():
    e = engine(report=True)
    e.run(BuyThenClose())
    st = e.stats()

    assert st["num_trades"]    == 1             # bought 1 @ 200, closed @ 300
    assert st["win_rate"]      == 1.0
    assert st["profit_factor"] == float("inf")  # a win and no losses


def test_market_impact_worsens_a_large_fill():
    # buy 100 into a 1000-volume bar with impact 0.5: slip = 0.5 * 0.1 = 5%
    e = probatum.Engine(
        series(),
        market    = "spot",
        quote     = 1_000_000.0,
        fee_taker = 0.0,
        fee_maker = 0.0,
        impact    = 0.5,
    )
    e.reset()
    e.market_buy(100.0)
    s = e.step()

    assert abs(s["avg_entry"] - 210.0) < 1e-9  # bar 1 open 200 * 1.05


def test_perp_funding_charges_a_held_position():
    # funding every bar at 0.001; a held long pays it, seen as a quote debit
    e = probatum.Engine(
        series(),
        market           = "perp",
        quote            = 10_000.0,
        fee_taker        = 0.0,
        fee_maker        = 0.0,
        funding_rate     = 0.001,
        funding_interval = 1,
    )
    e.reset()
    e.market_buy(1.0)
    s = e.step()  # long 1 @ 200; funding at bar 1 close 250: pay 1 * 250 * 0.001

    assert abs(s["quote"] - (10_000.0 - 0.25)) < 1e-9


def test_perp_leverage_caps_the_position():
    e = probatum.Engine(
        series(),
        market    = "perp",
        quote     = 100.0,
        fee_taker = 0.0,
        fee_maker = 0.0,
        leverage  = 2.0,
    )
    e.reset()
    e.market_buy(10.0)  # wants 10, but 2x on 100 equity at price 200 caps it at 1
    s = e.step()

    assert s["position"] == 1.0


def test_perp_leverage_defaults_to_a_finite_cap():
    # no leverage kwarg: the shipped 10x default bounds an oversized perp buy
    e = probatum.Engine(series(), market="perp", quote=100.0, fee_taker=0.0, fee_maker=0.0)
    e.reset()
    e.market_buy(10.0)  # 10x on 100 equity at price 200 caps the notional at 5 units
    s = e.step()

    assert s["position"] == 5.0


def test_per_env_cost_randomization_varies_costs_across_envs():
    # a per-env override array gives env1 a 1% taker fee and env0 none; both buy
    # 1 at bar 1 open 200, so only their quote after the fee differs
    b = probatum.Batch(
        series(),
        num_envs          = 2,
        market            = "spot",
        fee_taker         = 0.0,
        fee_taker_per_env = np.array([0.0, 0.01], dtype=np.float64),
    )

    b.reset_all()
    states = b.step_all(np.array([1.0, 1.0], dtype=np.float64))

    assert states[0]["quote"] == 9_800.0  # 10000 - 200
    assert states[1]["quote"] == 9_798.0  # 10000 - 200 - 0.01 * 200


def test_per_env_cost_override_length_must_match_num_envs():
    with pytest.raises(ValueError):
        probatum.Batch(
            series(),
            num_envs=3,
            fee_taker_per_env=np.array([0.0, 0.01], dtype=np.float64),  # length 2 != 3
        )


def test_order_reduce_only_market_only_shrinks():
    e = engine()
    e.reset()

    e.market_buy(1.0)
    e.step()  # long 1 @ 200
    e.order("buy", 1.0, type="market", reduce_only=True)  # cannot grow the long
    s = e.step()

    assert s["position"] == 1.0


def test_order_reduce_only_stop_loss_closes_a_long():
    # rise in, then a fall so a protective sell stop triggers and closes the long
    data = np.array(
        [
            [100.0, 110.0, 90.0, 100.0, 1000.0],
            [100.0, 110.0, 90.0, 100.0, 1000.0],
            [80.0, 85.0, 75.0, 80.0, 1000.0],
        ],
        dtype=np.float64,
    )
    e = probatum.Engine(data, market="spot", quote=10_000.0, fee_taker=0.0, fee_maker=0.0)

    e.reset()

    e.market_buy(1.0)
    e.step()  # long 1 @ bar 1 open 100
    e.order("sell", 1.0, type="stop", trigger=85.0, reduce_only=True)
    s = e.step()  # bar 2 low 75 crosses 85; the reduce-only stop closes the long

    assert s["position"] == 0.0


def test_order_ioc_limit_does_not_rest():
    e = engine()
    e.reset()

    e.order("buy", 1.0, type="limit", price=1.0, tif="IOC")  # 1.0 never reached
    s = e.step()

    assert s["open_orders"] == []  # IOC expired instead of resting


def test_order_validates_required_fields_and_tif():
    e = engine()
    e.reset()

    with pytest.raises(ValueError):
        e.order("buy", 1.0, type="limit")  # a limit needs a price

    with pytest.raises(ValueError):
        e.order("sell", 1.0, type="stop")  # a stop needs a trigger

    with pytest.raises(ValueError):
        e.order("buy", 1.0, tif="AON")  # unknown time-in-force


def test_cancel_all_drops_resting_orders():
    e = engine()
    e.reset()

    e.limit_buy(1.0, 50.0)
    e.limit_buy(1.0, 40.0)

    assert e.cancel_all() == 2

    s = e.step()

    assert s["open_orders"] == []


def test_qty_helpers_size_from_weight_and_quote():
    e = engine()
    e.reset()  # bar 0 close 150, equity 10000, flat

    assert abs(e.qty_from_weight(0.5) - 5_000.0 / 150.0) < 1e-9
    assert abs(e.qty_from_quote(300.0) - 2.0) < 1e-9


def test_stats_include_exposure_and_net_profit():
    e = engine(report=True)
    e.run(BuyAndHold())
    st = e.stats()

    assert "net_profit_pct" in st and "exposure_pct" in st
    assert abs(st["net_profit_pct"] - st["total_return_pct"]) < 1e-9
    assert 0.0 <= st["exposure_pct"] <= 100.0
    assert st["exposure_pct"] > 0.0  # BuyAndHold holds through the run


def test_open_order_dict_has_remaining_and_status():
    e = engine()
    e.reset()

    e.limit_buy(2.0, 50.0)  # rests, never reached
    s     = e.step()
    order = s["open_orders"][0]

    assert order["remaining"] == 2.0  # nothing filled yet
    assert order["status"]    == "resting"


def test_engine_shape_reports_the_candle_count():
    e = engine()

    assert e.shape == (3, 5)  # series() has 3 bars


def test_data_is_a_full_series_readonly_view():
    e = engine()
    d = e.data

    assert d.shape == (3, 5)  # the whole series, not a window
    assert d.flags.writeable is False
    assert d[:, 3].tolist() == [150.0, 250.0, 350.0]  # column 3 is close

    with pytest.raises(ValueError):
        d[0, 0] = 1.0


def test_bad_market_raises():
    says = 'market: "Options" is not a market; use spot or perp'

    for candles in (series(), universe_pair()):
        with pytest.raises(ValueError, match=re.escape(says)):
            probatum.Engine(candles, market="Options")


def test_candles_must_have_five_columns():
    with pytest.raises(ValueError, match="candles: 4 columns; give 5, open, high, low"):
        probatum.Engine(np.zeros((3, 4), dtype=np.float64))


def test_non_finite_candles_are_rejected():
    bad = series()
    bad[1, 2] = np.nan  # a NaN low on bar 1

    with pytest.raises(ValueError, match="candles: NaN at row 1; clean NaN"):
        probatum.Engine(bad)


def test_non_finite_config_scalars_are_rejected():
    # the candle array was always checked for non-finite values; every other knob
    # crossing the boundary poisons the same things and is now checked too (ADR 0007)
    for kwargs in (
        {"quote": float("nan")},
        {"fee_taker": float("nan")},
        {"fee_maker": float("inf")},
        {"slippage_bps": float("nan")},
        {"max_fill_fraction": float("nan")},
        {"leverage": float("nan")},
        {"impact": float("nan")},
        {"funding_rate": float("nan")},
    ):
        with pytest.raises(ValueError):
            probatum.Engine(series(), **kwargs)
        with pytest.raises(ValueError):
            probatum.Batch(series(), num_envs=2, **kwargs)


def test_out_of_range_per_env_costs_are_rejected_like_the_scalars():
    # the per-env arrays were checked for contiguity and length and nothing else,
    # so they were the one way past every guard below. A fee of -2.0
    # reached an env: on a perp, four buys of five units took equity from 10,000
    # to 14,000, minted out of a rebate larger than the notional
    for name, bad in (
        ("fee_taker_per_env", -2.0),
        ("fee_maker_per_env", -2.0),
        ("slippage_bps_per_env", -5.0),
        ("impact_per_env", -5.0),
    ):
        with pytest.raises(ValueError):
            probatum.Batch(series(), num_envs=2,
                       **{name: np.array([0.0, bad], dtype=np.float64)})

        # and the good value beside it still builds, so the guard is a range check
        # rather than a refusal to accept the argument at all
        probatum.Batch(series(), num_envs=2,
                   **{name: np.array([0.0, 0.5], dtype=np.float64)})


def test_out_of_range_config_scalars_are_rejected():
    for kwargs in (
        {"quote": -1.0},
        {"slippage_bps": -1.0},  # slippage is adverse, never favourable
        {"impact": -0.5},
        {"leverage": -1.0},
        {"max_fill_fraction": 0.0},
        {"max_fill_fraction": -0.5},
        {"fee_maker": -1.0},  # a rebate cannot exceed the notional
        {"fee_taker": -2.0},
        {"max_open_orders": 0},
        {"max_open_orders": 10_000},
    ):
        with pytest.raises(ValueError):
            probatum.Engine(series(), **kwargs)


def test_a_zero_window_raises_instead_of_panicking():
    # the batched gathers chunk by the window and rayon asserts a nonzero chunk, so
    # this used to surface as a PanicException, which `except Exception` misses
    b = probatum.Batch(series(), num_envs=2)
    b.reset_all()

    with pytest.raises(ValueError):
        b.observe(0)

    b.set_features(np.zeros((3, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        b.observe_features(0)


def test_non_finite_prices_and_triggers_are_rejected():
    e = engine()
    e.reset()

    with pytest.raises(ValueError):
        e.limit_buy(1.0, float("nan"))

    with pytest.raises(ValueError):
        e.limit_sell(1.0, float("inf"))

    with pytest.raises(ValueError):
        e.stop("sell", 1.0, float("nan"))

    with pytest.raises(ValueError):
        e.order("buy", 1.0, type="limit", price=float("nan"))


def test_stats_reject_a_non_positive_annualization():
    e = engine(report=True)
    e.reset()
    e.step()

    for ppy in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            e.stats(periods_per_year=ppy)


HOUR = 3_600 * 10**9


def test_candles_on_a_clock_count_their_own_bars_a_year():
    # three bars an hour apart but for a closure after the second: two steps over
    # five hours, so a year holds 3504 of them (ADR 0004)
    times    = np.array([0, HOUR, 5 * HOUR], dtype=np.int64)
    on_clock = engine(times=times).periods_per_year()

    assert on_clock == pytest.approx(2 * 8760.0 / 5)

    batch = probatum.Batch(series(), num_envs=2, times=times)

    assert batch.periods_per_year() == on_clock
    assert probatum._probatum.periods_per_year(times) == on_clock


def test_candles_without_times_have_no_annualization_to_report():
    assert engine().periods_per_year() is None
    assert probatum._probatum.periods_per_year(np.array([0], dtype=np.int64)) is None


def hourly(rows):
    return np.full((rows, 5), 100.0), np.arange(rows, dtype=np.int64) * HOUR


def test_funding_every_eight_hours_is_charged_at_the_close_of_the_bar_ending_then():
    # the 08:00 event belongs to the bar that opened at 07:00, and its venue-data
    # twin stamped a hair later lands on the same bar (ADR 0020)
    candles, times = hourly(10)
    sources = [
        dict(funding_hours=8.0, funding_rate=0.001),
        dict(funding_times=np.array([8 * HOUR + 3_000_000], dtype=np.int64),
             funding_rates=np.array([0.001])),
    ]

    for source in sources:
        eng = probatum.Engine(
            candles,
            market    = "perp",
            quote     = 10_000.0,
            fee_taker = 0.0,
            times     = times,
            **source,
        )

        eng.reset()
        eng.market_buy(1.0)
        paid = [eng.step()["funding_paid"] for _ in range(8)]

        assert paid == [0.0] * 6 + [pytest.approx(0.1)] * 2, source


def test_funding_named_on_the_one_instrument_is_charged_as_funding_named_on_none():
    candles, times = hourly(10)
    stamps = dict(funding_times=np.array([8 * HOUR], dtype=np.int64),
                  funding_rates=np.array([0.001]))

    paid = []
    for named in ({}, {"funding_instruments": [0]}, {"funding_instruments": [None]}):
        eng = probatum.Engine(
            candles,
            market    = "perp",
            quote     = 10_000.0,
            fee_taker = 0.0,
            times     = times,
            **stamps,
            **named,
        )
        eng.reset()
        eng.market_buy(1.0)
        paid.append([eng.step()["funding_paid"] for _ in range(8)])

    assert paid[0] == paid[1] == paid[2]

    with pytest.raises(ValueError, match=re.escape("row 0 names instrument 1, of 1")):
        probatum.Engine(
            candles,
            market              = "perp",
            times               = times,
            funding_instruments = [1],
            **stamps,
        )


def test_a_funding_source_is_refused_without_its_clock_or_beside_another():
    candles, times = hourly(4)

    with pytest.raises(ValueError, match="times: none"):
        probatum.Engine(candles, market="perp", funding_hours=8.0)

    with pytest.raises(ValueError, match="funding: more than one source"):
        probatum.Engine(
            candles,
            market           = "perp",
            times            = times,
            funding_hours    = 8.0,
            funding_interval = 3,
        )

    with pytest.raises(ValueError, match="funding_times: given without funding_rates"):
        probatum.Batch(
            candles,
            num_envs      = 1,
            times         = times,
            funding_times = np.array([HOUR], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="funding_rates: 2 for 1 funding_times"):
        probatum.Engine(
            candles,
            market        = "perp",
            times         = times,
            funding_times = np.array([HOUR], dtype=np.int64),
            funding_rates = np.array([0.1, 0.1]),
        )


def test_a_clock_out_of_order_or_short_of_a_bar_is_refused():
    with pytest.raises(ValueError, match="times: row 2 opens at 3600000000000"):
        engine(times=np.array([0, 2 * HOUR, HOUR], dtype=np.int64))

    with pytest.raises(ValueError, match="times: row 1 opens at 0"):
        probatum.Batch(
            series(),
            num_envs = 1,
            times    = np.array([0, 0, HOUR], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="times: 2 for 3 bars"):
        engine(times=np.array([0, HOUR], dtype=np.int64))


def test_quote_opens_both_wallets_by_name_and_a_transfer_lands_on_the_next_bar():
    e = engine(market="perp", quote={"spot": 300.0, "margin": 1_000.0})
    s = e.reset()

    assert (s["spot_cash"], s["margin_cash"]) == (300.0, 1_000.0)
    assert s["quote"] == s["equity"] == 1_300.0

    e.transfer(300.0, to="margin")
    e.market_buy(1.0)
    s = e.step()

    assert (s["spot_cash"], s["margin_cash"]) == (0.0, 1_300.0)
    assert s["position"] == 1.0


def test_a_wallet_or_a_transfer_the_engine_cannot_use_is_refused():
    with pytest.raises(ValueError, match='quote: a wallet named "savings"'):
        engine(quote={"spot": 1.0, "savings": 1.0})

    with pytest.raises(ValueError, match=r'quote\["margin"\] must be at least 0'):
        engine(market="perp", quote={"margin": -1.0})

    with pytest.raises(ValueError, match="quote: 'all'; give the opening cash"):
        engine(quote="all")

    with pytest.raises(ValueError, match="quote: a wallet named 1; name spot"):
        engine(quote={1: 5.0})

    with pytest.raises(ValueError, match="quote: '5' for the spot wallet; give"):
        engine(quote={"spot": "5"})

    e = engine()

    with pytest.raises(ValueError, match="amount: -5 is negative"):
        e.transfer(-5.0, "margin")

    with pytest.raises(ValueError, match="amount: NaN is not a finite number"):
        e.transfer(float("nan"), "margin")

    with pytest.raises(ValueError, match='to: "savings" is not a wallet'):
        e.transfer(5.0, "savings")


def test_stop_takes_reduce_only_so_it_cannot_open_the_other_side():
    # a fall that crosses the trigger while flat: a protective stop fills nothing,
    # a bare one opens a short on a perp (ADR 0010)
    data = np.array(
        [
            [100.0, 101.0, 99.0, 100.0, 1000.0],
            [100.0, 101.0, 80.0, 85.0, 1000.0],
            [85.0, 86.0, 84.0, 85.0, 1000.0],
        ],
        dtype=np.float64,
    )

    protective = probatum.Engine(data, market="perp", quote=10_000.0, fee_taker=0.0)
    protective.reset()
    protective.stop("sell", 1.0, 95.0, reduce_only=True)

    assert protective.step()["position"] == 0.0

    bare = probatum.Engine(data, market="perp", quote=10_000.0, fee_taker=0.0)
    bare.reset()
    bare.stop("sell", 1.0, 95.0)

    assert bare.step()["position"] == -1.0


def test_order_ids_do_not_restart_on_reset():
    e = engine()
    e.reset()

    first = e.limit_buy(1.0, 50.0)
    e.reset()
    second = e.limit_buy(1.0, 50.0)

    assert first != second
    assert e.cancel(first) is False  # the stale handle names nothing
    assert e.cancel(second) is True


def test_a_fok_market_blocked_by_the_cash_clamp_fills_nothing():
    # the bar has the volume, but the cash affords 50 of the 1000; FOK is all or
    # nothing against every clamp, not just liquidity (ADR 0010)
    e = engine()
    e.reset()

    e.order("buy", 1000.0, type="market", tif="FOK")
    s = e.step()

    assert s["position"] == 0.0
    assert s["quote"]    == 10_000.0


def test_a_trade_carries_both_sides_of_its_fee():
    # charging only the closing fill left about half the round trip attributed to
    # nothing, so the metrics built on it stayed optimistic (ADR 0023)
    e = probatum.Engine(
        series(),
        market    = "spot",
        quote     = 10_000.0,
        fee_taker = 0.01,
        fee_maker = 0.01,
        report    = True,
    )

    e.reset()
    e.market_buy(1.0)
    e.step()  # buy 1 at 200, entry fee 2.0
    e.close()
    e.step()  # sell 1 at 300, exit fee 3.0
    t = e.trades()[0]

    assert abs(t["realized"] - 100.0) < 1e-9
    assert abs(t["fees"] - 5.0) < 1e-9
    assert abs(t["pnl"] - 95.0) < 1e-9


def test_stats_reproduce_from_the_trade_log():
    data = np.array(
        [[100.0 + (i % 7) - 3.0] * 4 + [10_000.0] for i in range(60)], dtype=np.float64
    )

    class PingPong:

        def __init__(self):
            self.hold = 0


        def next(self, state, engine):
            if state["position"] == 0.0:
                engine.market_buy(1.0)
                self.hold = 0
            else:
                self.hold += 1
                if self.hold >= 2:
                    engine.close()

    e = probatum.Engine(
        data,
        market    = "spot",
        quote     = 10_000.0,
        fee_taker = 0.001,
        fee_maker = 0.001,
        report    = True,
    )
    e.run(PingPong())
    st     = e.stats()
    trades = [t for t in e.trades() if not t["open"]]

    assert len(trades) > 3
    assert st["num_trades"] == len(trades)

    wins = sum(1 for t in trades if t["pnl"] > 0.0)

    assert abs(st["win_rate"] - wins / len(trades)) < 1e-12
    for t in trades:
        assert abs(t["pnl"] - (t["realized"] - t["fees"] - t["funding"])) < 1e-12


def test_num_fills_separates_a_dead_feed_from_a_quiet_strategy():
    dead = np.array([[100.0, 100.0, 100.0, 100.0, 0.0]] * 6, dtype=np.float64)
    e    = probatum.Engine(dead, market="spot", report=True)
    e.reset()
    while not e.done():
        e.market_buy(1.0)
        e.step()

    assert e.num_fills() == 0
    assert e.stats()["num_fills"] == 0

    live = engine(report=True)
    live.reset()

    live.market_buy(1.0)
    live.step()

    assert live.num_fills() == 1


def test_replace_moves_one_order_and_will_not_arm_a_second():
    # the trailing-stop trap: stop() per bar rests a new order every time, and the
    # one that fills leaves its siblings live (ADR 0010)
    e = probatum.Engine(series(), market="perp", quote=10_000.0, fee_taker=0.0)
    e.reset()

    first  = e.stop("sell", 1.0, 50.0, reduce_only=True)
    second = e.replace(first, trigger=60.0)
    s      = e.step()

    assert len(s["open_orders"])          == 1
    assert s["open_orders"][0]["id"]      == second
    assert s["open_orders"][0]["trigger"] == 60.0
    assert s["open_orders"][0]["reduce_only"] is True
    assert e.cancel(second)
    assert e.replace(second, trigger=70.0) is None  # gone, so nothing is placed
    assert e.step()["open_orders"] == []


def test_is_bust_reports_a_dead_account():
    e = probatum.Engine(series(), market="spot", quote=10_000.0)
    e.reset()

    assert e.is_bust() is False

    dead = probatum.Engine(series(), market="spot", quote=0.0)
    dead.reset()
    dead.step()

    assert dead.is_bust() is True


def test_trade_metrics_are_net_of_fees():
    # a creeping series whose per-trade gross edge is smaller than the round trip
    # cost: gross it wins every trade, net it loses money (ADR 0030)
    data = np.array(
        [[100.0 + i * 0.01] * 4 + [1000.0] for i in range(40)], dtype=np.float64
    )

    class PingPong:

        def next(self, state, engine):
            if state["position"] == 0.0:
                engine.market_buy(1.0)
            else:
                engine.close()

    e = probatum.Engine(
        data,
        market    = "spot",
        quote     = 10_000.0,
        fee_taker = 0.001,
        fee_maker = 0.001,
        report    = True,
    )
    e.run(PingPong())
    st = e.stats()

    assert st["total_return_pct"] < 0.0
    assert st["win_rate"]      == 0.0
    assert st["profit_factor"] == 0.0
    assert st["avg_trade_pct"] < 0.0


def test_spot_buy_cannot_spend_more_quote_than_held():
    # the Python view of the spot-buy cash clamp: a huge buy fills only what the
    # cash affords, and the balance never goes negative
    e = probatum.Engine(series(), market="spot", quote=10_000.0, fee_taker=0.0, fee_maker=0.0)
    e.reset()
    e.market_buy(1_000.0)  # wants 1000 base at bar 1 open 200, affords 50
    s = e.step()

    assert s["position"] == 50.0
    assert s["quote"]    == 0.0


def costed_bars():
    # bar 1 is the one a decision on bar 0 fills against: its open is the fill
    # price, its volume is what an impact fraction is taken of, and its high sits
    # far enough above both that the taker clamp of ADR 0013 never binds and the
    # slip can be read straight off the price
    return np.array([
        [100.0, 200.0, 90.0, 100.0, 1000.0],
        [100.0, 200.0, 90.0, 100.0, 10.0],
        [100.0, 200.0, 90.0, 100.0, 1000.0],
    ], dtype=np.float64)


def test_a_per_env_slippage_lands_on_slippage_and_not_on_impact():
    # asserted by VALUE. The only thing standing behind these two arrays was
    # np.std(rewards) > 0.0, a property, and crossing the two assignments still
    # spreads the rewards: a slippage draw landing on impact moves the price by
    # size/volume times the number instead of by basis points, which is wrong by
    # orders of magnitude and reads identical to a property
    b = probatum.Batch(
        costed_bars(),
        num_envs             = 2,
        market               = "spot",
        fee_taker            = 0.0,
        fee_maker            = 0.0,
        slippage_bps_per_env = np.array([0.0, 50.0], dtype=np.float64),
        impact_per_env       = np.array([0.0, 0.0], dtype=np.float64),
    )

    b.reset_all()
    states = b.step_all(np.array([1.0, 1.0], dtype=np.float64))

    assert states[0]["quote"] == 9_900.0            # one unit at the open of 100
    assert states[1]["quote"] == 9_899.5            # 50 bps of 100 is 0.5


def test_a_per_env_impact_lands_on_impact_and_not_on_slippage():
    # the mirror. One unit of a ten-volume bar is a tenth of it, so an impact of
    # 5.0 is half the price; the same 5.0 read as slippage_bps would be 5 bps
    b = probatum.Batch(
        costed_bars(),
        num_envs             = 2,
        market               = "spot",
        fee_taker            = 0.0,
        fee_maker            = 0.0,
        slippage_bps_per_env = np.array([0.0, 0.0], dtype=np.float64),
        impact_per_env       = np.array([0.0, 5.0], dtype=np.float64),
    )

    b.reset_all()
    states = b.step_all(np.array([1.0, 1.0], dtype=np.float64))

    assert states[0]["quote"] == 9_900.0
    assert states[1]["quote"] == 9_850.0            # 5.0 * (1 / 10) is half of 100
