"""A Strategy is called at each close from its warm-up but the last, on the rows the
account sees, with the bar just closed and the account that close left. The bar is the
Data's row and its history ends there; the account is the ledger's row; every order is
a live handle that follows the order log, and the events since the last close are the
log's own. A backtest runs a copy of the Strategy, and the Engine stepped by hand
records what a backtest of the same decisions records.
"""

import datetime
import math
import random
import re

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import Venue
from probatum._venue import _run
from reference import universe_differential
from test_a_data_from_the_router_is_every_bar_it_asked_for import (Router, bars, fetch,
                                                                   settlements)
from test_every_depth_gives_one_ledger import case, records, same, through_account

VENUE = Venue(fees={"maker": 0.0002, "taker": 0.0005})
ENDS  = {
    "filled":     "filled",
    "cancelled":  "cancelled",
    "expired":    "expired",
    "not_filled": "expired",
    "refused":    "refused",
}


def frame(n=60, level=100.0, seed=0, start="2024-01-01"):
    """Hourly bars on a walk about `level`."""
    rng    = np.random.default_rng(seed)
    close  = level * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    opened = np.concatenate([[close[0]], close[:-1]])
    index  = pd.date_range(start, periods=n, freq="h", tz="UTC")

    return pd.DataFrame({
        "open":   opened,
        "high":   np.maximum(opened, close) * 1.004,
        "low":    np.minimum(opened, close) * 0.996,
        "close":  close,
        "volume": 1e6,
    }, index=index)


def jumpy(n=60, level=100.0, seed=0):
    """Hourly bars that open away from the close before them, each field its own."""
    rng    = np.random.default_rng(seed)
    close  = level * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    opened = close * np.exp(rng.normal(0.0, 0.004, n))
    index  = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    return pd.DataFrame({
        "open":   opened,
        "high":   np.maximum(opened, close) * 1.004,
        "low":    np.minimum(opened, close) * 0.996,
        "close":  close,
        "volume": rng.uniform(1e5, 1e6, n),
    }, index=index)


def pair():
    return probatum.Data({"BTCUSDT": frame(), "ETHUSDT": frame(level=10.0, seed=1)},
                         market="linear")


def one():
    return probatum.Data(frame(), symbol="BTCUSDT", market="linear")


class Seen(probatum.Strategy):
    """Every row it was called at, and what the bar said there."""


    def __init__(self, warmup=0):
        self.warmup = warmup


    def init(self, data):
        self.data = data
        self.rows = []
        self.bars = []


    def next(self, bar, account):
        self.rows.append(bar.i)
        self.bars.append(bar)


def gapped():
    """BTCUSDT of a universe beside ETHUSDT, which prints rows 2, 3 and 30 where
    BTCUSDT prints none."""
    late = frame(seed=2)
    late.iloc[[2, 3, 30], :] = np.nan

    wide = probatum.Data({"BTCUSDT": late.dropna(), "ETHUSDT": frame(seed=3)},
                         market="linear")

    return wide.select(["BTCUSDT"])


def test_next_runs_at_each_close_from_its_warmup_but_the_last_and_a_row_none_printed():
    data     = gapped()
    strategy = Seen(warmup=5)
    result   = probatum.backtest(strategy, data, VENUE)
    printed  = np.flatnonzero(data.available[:, 0])

    assert result.policy.rows == [i for i in printed[:-1] if i >= 5]
    assert result.policy.data is data and not hasattr(strategy, "rows")
    assert len(result.time) == len(printed)


class Traded(probatum.Strategy):
    """A buy every six rows and a close three rows later, with what the account named."""


    def init(self, data):
        self.placed, self.events, self.trades = [], [], None


    def next(self, bar, account):
        self.events += account.events

        if bar.i % 6 == 0:
            self.placed.append((bar.i, account.buy(1.0)))
        elif bar.i % 6 == 3:
            self.placed.append((bar.i, account.close()))

        self.trades = account.trades


@pytest.mark.parametrize("cut", [0, 10])
def test_the_account_names_the_data_s_rows_with_the_rows_none_printed_left_out(cut):
    data   = gapped()[cut:]
    result = probatum.backtest(Traded(), data, VENUE)
    strategy, printed = result.policy, np.flatnonzero(data.available[:, 0])
    placed = [(row, order) for row, order in strategy.placed if order is not None]

    assert len(placed) > 4 and all(order.placed == row for row, order in placed)
    assert [event.bar for event in strategy.events] == [
        printed[event["bar"]] for event in result.orders[:len(strategy.events)]]

    closed = [trip for trip in result.trades if not trip["open"]][:len(strategy.trades)]
    assert len(closed) > 3
    assert strategy.trades.open_bar.tolist()  == [printed[t["open_bar"]] for t in closed]
    assert strategy.trades.close_bar.tolist() == [printed[t["close_bar"]] for t in closed]


def test_the_bar_is_the_data_s_row_and_its_history_ends_there():
    data     = pair()
    strategy = probatum.backtest(Seen(), data, VENUE).policy

    for bar in strategy.bars:
        i = bar.i
        assert bar.ns == data.time[i]
        assert bar.time == datetime.datetime.fromtimestamp(data.time[i] / 1e9,
                                                           datetime.timezone.utc)
        np.testing.assert_array_equal(bar.closes, data.closes[i])
        np.testing.assert_array_equal(bar.available, data.available[i])
        assert bar["ETHUSDT"].close == data["ETHUSDT"].close[i]
        assert bar["ETHUSDT"].high  == data["ETHUSDT"].high[i]
        assert bar.n                == 2

        window = bar.history(5)
        assert window.time[-1] == data.time[i] and len(window) == min(5, i + 1)
        np.testing.assert_array_equal(window.closes, data.closes[max(0, i - 4):i + 1])
        assert len(bar.history()) == i + 1

    with pytest.raises(AttributeError, match=r"bar.close: the universe has 2 "
                                             r"instruments; use bar.closes or "
                                             r"bar\[\"BTCUSDT\"\].close"):
        strategy.bars[0].close

    with pytest.raises(ValueError, match="n: 0; give history a whole number of bars"):
        strategy.bars[0].history(0)


def test_every_field_of_the_bar_is_the_data_s_at_its_row():
    eth      = jumpy(level=10.0, seed=5)
    late     = eth.drop(eth.index[[7, 8]])
    data     = probatum.Data({"BTCUSDT": jumpy(seed=4), "ETHUSDT": late}, market="linear")
    strategy = probatum.backtest(Seen(), data, VENUE).policy

    assert not data.available[7, 1] and 7 in strategy.rows

    fields = ("open", "high", "low", "close", "volume")
    for bar in strategy.bars:
        for f, field in enumerate(fields):
            values = data.bars[bar.i, :, f]
            np.testing.assert_array_equal(getattr(bar, field + "s"), values)
            for k, name in enumerate(data.names):
                np.testing.assert_array_equal(getattr(bar[name], field), values[k])
        assert [bar[name].available for name in data.names] == list(data.available[bar.i])

    lone = probatum.Data(jumpy(seed=6), symbol="BTCUSDT", market="linear")
    for bar in probatum.backtest(Seen(), lone, VENUE).policy.bars:
        assert [getattr(bar, field) for field in fields] == lone.bars[bar.i, 0].tolist()


def test_a_bar_s_funding_rate_is_the_last_one_known_at_its_end():
    funding = {"BTCUSDT": settlements(), "ETHUSDT": settlements(rate=-0.0002)}
    data    = fetch(Router({"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}, funding),
                    ("BTCUSDT", "ETHUSDT"), funding=True)

    rates = data.funding_rate
    assert np.isnan(rates[0]).all() and not np.isnan(rates[-1]).any()

    for bar in probatum.backtest(Seen(), data, "binance_perp").policy.bars:
        np.testing.assert_array_equal(bar.funding_rates, rates[bar.i])
        np.testing.assert_array_equal(bar["ETHUSDT"].funding_rate, rates[bar.i, 1])

    lone = data.select(["BTCUSDT"])
    for bar in probatum.backtest(Seen(), lone, "binance_perp").policy.bars:
        np.testing.assert_array_equal(bar.funding_rate, rates[bar.i, 0])


def test_a_slice_s_history_reaches_the_bars_before_it():
    whole  = one()
    data   = whole[20:]
    result = probatum.backtest(Seen(), data, VENUE)
    strategy, first = result.policy, result.policy.bars[0]

    assert first.i == 0 and first.close == whole.close[20]
    np.testing.assert_array_equal(result.time, data.time)
    np.testing.assert_array_equal(first.history(5).close, whole.close[16:21])


class Rebalanced(probatum.Strategy):
    """Random weights at every close, and the account it read there."""


    def __init__(self, seed, kinds):
        self.seed  = seed
        self.kinds = kinds


    def init(self, data):
        self.rng   = random.Random(self.seed)
        self.names = data.names
        self.read  = []


    def next(self, bar, account):
        views = [account[name] for name in self.names]
        self.read.append((bar.i, account.equity, account.leverage, account.positions,
                          [view.position for view in views],
                          [view.entry for view in views], [view.upnl for view in views],
                          [view.weight for view in views], account.spot, account.margin,
                          account.trades.pnl, account.equity_curve))

        account.target({name: self.rng.uniform(0.0 if kind == "spot" else -1.0, 1.0)
                        for name, kind in zip(self.names, self.kinds)
                        if self.rng.random() < 0.7})


@pytest.mark.parametrize("seed", range(6))
def test_the_account_is_the_ledger_s_row_its_close_left(seed):
    times, kinds, candles, cfg = case(random.Random(seed))
    settings = universe_differential.venue(cfg)
    strategy = Rebalanced(seed, kinds)
    engine   = through_account(candles, times, settings, kinds, strategy)
    ledger, trades = engine.ledger(), engine.trades()

    assert any(row[1] != ledger["equity"][0] for row in strategy.read)

    spot_columns = [k for k, kind in enumerate(kinds) if kind == "spot"]
    for i, equity, leverage, positions, each, entry, upnl, weight, spot, margin, pnl, \
            curve in strategy.read:
        assert equity == ledger["equity"][i] and leverage == ledger["leverage"][i]
        np.testing.assert_array_equal(positions, ledger["position"][i])
        np.testing.assert_array_equal(each, ledger["position"][i])
        np.testing.assert_array_equal(entry, ledger["entry"][i])
        np.testing.assert_array_equal(upnl, ledger["unrealized"][i])
        np.testing.assert_array_equal(weight, ledger["weight"][i])
        assert (spot.cash, margin.cash) == (ledger["spot_cash"][i],
                                            ledger["margin_cash"][i])

        held = sum(ledger["position"][i][k] * ledger["mark"][i][k] for k in spot_columns
                   if ledger["position"][i][k] != 0.0)
        assert spot.equity                 == pytest.approx(spot.cash + held, rel=1e-12, abs=1e-9)
        assert spot.equity + margin.equity == pytest.approx(equity, rel=1e-12)
        assert margin.requirement          == ledger["maintenance"][i]
        assert margin.available            == max(0.0, margin.equity - ledger["margin_used"][i])

        closed = [trip["pnl"] for trip in trades
                  if trip["close_bar"] is not None and trip["close_bar"] <= i]
        assert pnl.tolist() == closed and not pnl.flags.writeable
        np.testing.assert_array_equal(curve, ledger["equity"][:i + 1])


class Busy(probatum.Strategy):
    """Every verb, on random instruments, written into the script that places the same
    orders, and at each close the handles it holds as it found them, beside the
    engine's open orders and the orders each instrument's view holds."""


    def __init__(self, seed, kinds):
        self.seed  = seed
        self.kinds = kinds


    def init(self, data):
        self.rng   = random.Random(self.seed)
        self.names = data.names

        self.handles, self.events, self.found, self.asked = [], [], [], []
        self.script, self.placements, self.placed = {}, 0, {}


    def take(self, row, verb, placed=(), **given):
        """``verb`` into the script at ``row``, and the placement each of ``placed``
        is there, counted as the script counts them."""
        self.script.setdefault(row, []).append((verb, given))

        for handle in placed:
            if handle is not None:
                self.placed[handle.id] = self.placements
            self.placements += 1


    def next(self, bar, account):
        rng = self.rng
        self.events += account.events

        engine = account._run.engine
        self.found.append((len(self.events), {o["id"]: o for o in engine.orders()},
                           [(h, h.status, h.filled, h.side, h.size, h.price, h.trigger,
                             h.reason) for h in self.handles],
                           [len(account[name].orders) for name in self.names]))

        for _ in range(rng.randint(0, 3)):
            k, before = rng.randrange(len(self.names)), len(self.handles)
            view, close = account[self.names[k]], bar.closes[k]
            if math.isnan(close):
                continue

            spot = self.kinds[k] == "spot"
            size = rng.uniform(0.05, 0.4) * account.equity / close
            side = rng.choice(["buy", "sell"])
            verb = rng.choice(["market", "limit", "stop", "value", "bracket", "close",
                               "cancel", "replace", "cancel_all", "target"])
            placed = [h for h in self.handles if h.id in self.placed]

            if verb == "market":
                given = dict(size=size, reduce_only=spot and side == "sell",
                             tif=rng.choice(["gtc", "ioc", "fok"]))
                order = getattr(view, side)(**given)
                self.take(bar.i, side, [order], instrument=k, **given)
                self.handles.append(order)
            elif verb == "limit":
                away  = rng.uniform(-0.01, 0.03) * (1.0 if side == "buy" else -1.0)
                given = dict(size=size, limit=close * (1.0 - away),
                             post_only=rng.random() < 0.3,
                             reduce_only=spot and side == "sell")
                order = getattr(view, side)(**given)
                self.take(bar.i, side, [order], instrument=k, **given)
                self.handles.append(order)
            elif verb == "stop":
                away  = rng.uniform(0.002, 0.02) * (1.0 if side == "buy" else -1.0)
                given = dict(size=size, stop=close * (1.0 + away),
                             reduce_only=rng.random() < 0.5)
                order = getattr(view, side)(**given)
                self.take(bar.i, side, [order], instrument=k, **given)
                self.handles.append(order)
            elif verb == "value":
                given = dict(amount=size * close, reduce_only=spot and side == "sell")
                order = getattr(view, side + "_value")(**given)
                self.take(bar.i, side + "_value", [order], instrument=k, **given)
                self.handles.append(order)
            elif verb == "bracket":
                long  = view.position >= 0.0
                given = dict(
                    stop=close * (0.98 if long else 1.02) if rng.random() < 0.8 else None,
                    take_profit=close * (1.03 if long else 0.97))
                exits = view.bracket(**given)
                self.take(bar.i, "bracket", list(exits), instrument=k, **given)
                self.handles += [leg for leg in exits if leg is not None]
            elif verb == "close":
                order = view.close()
                self.take(bar.i, "close", [order], instrument=k)
                self.handles.append(order)
            elif verb == "cancel" and placed:
                handle = rng.choice(placed)
                account.cancel(handle)
                self.take(bar.i, "cancel", order=self.placed[handle.id])
            elif verb == "replace" and placed:
                handle = rng.choice(placed)
                if handle.status == "resting":
                    if handle.kind == "stop":
                        moved = dict(trigger=handle.trigger * 1.001)
                    else:
                        moved = dict(price=handle.price * 0.999)
                    account.replace(handle, **moved)
                    self.take(bar.i, "replace", order=self.placed[handle.id], **moved)
            elif verb == "cancel_all":
                if rng.random() < 0.5:
                    account.cancel_all()
                    self.take(bar.i, "cancel_all")
                else:
                    view.cancel_all()
                    self.take(bar.i, "cancel_all", instrument=k)
            elif verb == "target":
                weight = rng.uniform(0.0 if spot else -1.0, 1.0)
                self.handles += account.target({self.names[k]: weight})
                self.take(bar.i, "target", weights={k: weight})

            self.asked += [(k, handle) for handle in self.handles[before:]
                           if handle is not None]
            self.handles = [h for h in self.handles if h is not None]


def expected(order, log, found, filled_in):
    """What a handle should show at a close: an order still open as the engine has it,
    and one gone with the last word of its events before that close."""
    if order in found:
        given = found[order]
        return (given["status"], given["filled"], given["side"], given["size"],
                given["price"], given["trigger"])

    ends = [event["kind"] for event in log if event["order"] == order
            and event["kind"] in ENDS]
    return ENDS[ends[-1]], filled_in(order)


def busy(seed):
    """A universe of the comparison's, the engine after `Busy` ran on it, and what the
    script it wrote runs from."""
    times, kinds, candles, cfg = case(random.Random(seed))
    strategy, venue = Busy(seed, kinds), universe_differential.venue(cfg)
    engine = through_account(candles, times, venue, kinds, strategy)

    return strategy, engine, (candles, venue, times)


@pytest.fixture(scope="module")
def runs():
    return [busy(seed) for seed in range(12)]


@pytest.mark.parametrize("seed", range(12))
def test_every_handle_and_event_follows_the_order_log(seed, runs):
    strategy, engine, _ = runs[seed]
    log, names = engine.order_log(), strategy.names
    sides = {fill["order"]: fill["side"] for fill in engine.fills()}
    seen  = [(e.kind, e.order and e.order.id, e.instrument, e.bar, e.size, e.price,
              e.reason) for e in strategy.events]

    assert seen == [(e["kind"], e["order"], None if e["instrument"] is None
                     else names[e["instrument"]], e["bar"], e["size"], e["price"],
                     e["reason"]) for e in log[:len(seen)]]
    assert all(handle.instrument == names[k] for k, handle in strategy.asked)

    kinds_seen = {event[0] for event in seen}
    assert {"placed", "filled"} <= kinds_seen, kinds_seen

    for count, found, held, counts in strategy.found:
        before = log[:count]
        assert counts == [sum(1 for order in found.values() if order["instrument"] == k)
                          for k in range(len(names))]

        def filled_in(order):
            return sum(e["size"] for e in before if e["order"] == order
                       and e["kind"] == "filled")

        for handle, status, filled, side, size, price, trigger, reason in held:
            assert reason == [e["reason"] for e in before if e["order"] == handle.id][-1]
            want = expected(handle.id, before, found, filled_in)
            if len(want) == 6:
                assert (status, filled, side, size, price, trigger) == want, handle
            else:
                assert status == want[0], (handle, before)
                assert filled == pytest.approx(want[1], rel=1e-12, abs=1e-12), handle
                if filled:
                    assert side == sides[handle.id], handle


@pytest.mark.parametrize("seed", range(12))
def test_the_verbs_place_what_a_script_of_the_same_actions_places(seed, runs):
    strategy, engine, (candles, venue, times) = runs[seed]

    assert strategy.script

    played = probatum._probatum.run(candles, venue, strategy.script, times=times)

    assert same(records(engine), played)


def test_the_handles_show_every_status_an_order_has(runs):
    shown = {status for strategy, *_ in runs for _, _, held, _ in strategy.found
             for _, status, *_ in held}

    assert shown == {"pending", "resting", "filled", "cancelled", "expired",
                     "refused"}, shown


def test_a_refused_order_is_a_handle_with_its_reason():
    data = one()

    class Refused(probatum.Strategy):

        def next(self, bar, account):
            if bar.i == 3:
                self.empty  = account.buy(0.0)
                self.short  = account.sell(1.0, limit=bar.close * 0.5, post_only=True)
                self.bought = account.buy_value(1_000.0)
                self.exits  = account.bracket(take_profit=bar.close * 1.5)
            elif bar.i == 5:
                self.bought_then = (self.bought.status, self.bought.size,
                                    self.bought.filled, account.position)

    strategy = probatum.backtest(Refused(), data, VENUE).policy

    assert (strategy.empty.status, strategy.empty.reason) == ("refused",
                                                              "size: 0 cannot fill")
    assert strategy.short.status == "refused"
    assert strategy.short.reason.startswith("post_only")

    status, size, filled, position = strategy.bought_then
    assert (status, size, filled) == ("filled", None, position)
    assert filled == pytest.approx(1_000.0 / data.open[4], rel=1e-12)
    assert strategy.exits.stop is None
    assert strategy.exits.take_profit.side   == "sell"
    assert strategy.exits.take_profit.placed == 3


def test_an_order_filled_before_it_is_read_holds_what_the_venue_s_rules_left_it():
    venue = VENUE.replace(fills={"sizes": "on"}, tick=0.01, lot=0.001)

    class Rounded(probatum.Strategy):

        def next(self, bar, account):
            if bar.i == 3:
                self.asked  = bar.close * 1.0012345
                self.bid    = account.buy(0.12345, limit=self.asked)
                self.bought = account.buy(0.23456)

    result = probatum.backtest(Rounded(), one(), venue)
    strategy, fills = result.policy, {fill["order"]: fill for fill in result.fills}

    for order in (strategy.bid, strategy.bought):
        assert order.status == "filled"
        assert order.size == order.filled == fills[order.id]["size"]

    assert (strategy.bid.size, strategy.bought.size) == (0.123, 0.234)

    price = strategy.bid.price
    assert price != strategy.asked and price == round(price, 2)


def test_a_bracket_refused_whole_hands_back_each_leg_s_own_refusal():
    venue = VENUE.replace(max_open_orders=1)

    class Full(probatum.Strategy):

        def next(self, bar, account):
            if bar.i == 3:
                self.bid   = account.buy(1.0, limit=bar.close * 0.5)
                self.exits = account.bracket(stop=bar.close * 0.9,
                                             take_profit=bar.close * 1.1)

    strategy = probatum.backtest(Full(), one(), venue).policy
    stop, take_profit = strategy.exits

    assert strategy.bid.status == "resting"
    assert (stop.status, take_profit.status) == ("refused", "refused")
    assert stop.id < take_profit.id
    assert (stop.kind, take_profit.kind) == ("stop", "limit")
    assert stop.reason == take_profit.reason == "max_open_orders: 1 already resting"


def test_each_verb_places_what_it_names_through_the_view_it_is_taken_through():
    class Placed(probatum.Strategy):

        def next(self, bar, account):
            if bar.i == 3:
                btc, eth = account["BTCUSDT"], account["ETHUSDT"]
                self.bid       = btc.buy(0.01, limit=bar.closes[0] * 0.5)
                self.eth       = eth.buy(1.0, limit=bar.closes[1] * 0.5)
                self.dropped   = eth.cancel_all()
                self.moved     = account.replace(self.eth, price=bar.closes[1] * 0.4)
                self.sold      = eth.sell_value(100.0)
                self.reduced   = btc.buy(1.0, reduce_only=True)
                self.cancelled = account.cancel(self.bid), account.cancel(self.bid)

    data     = pair()
    strategy = probatum.backtest(Placed(), data, VENUE).policy

    assert (strategy.dropped, strategy.moved, strategy.cancelled) == (1, None, (True,
                                                                               False))
    assert (strategy.bid.status, strategy.eth.status) == ("cancelled", "cancelled")
    assert (strategy.sold.side, strategy.sold.status) == ("sell", "filled")

    opened = data["ETHUSDT"].open[4]
    assert strategy.sold.filled == pytest.approx(100.0 / opened, rel=1e-12)
    assert (strategy.reduced.side, strategy.reduced.filled) == ("buy", 0.0)


def test_a_target_s_leg_refused_as_it_is_placed_is_handed_back_refused():
    class Crowded(probatum.Strategy):

        def next(self, bar, account):
            if bar.i == 3:
                self.bought = account.buy(1.0)
                self.legs   = account.target(0.5)
            elif bar.i == 4:
                self.events = account.events

    crowded  = VENUE.replace(max_open_orders=1)
    strategy = probatum.backtest(Crowded(), one(), crowded).policy
    [leg] = strategy.legs

    assert (leg.kind, leg.status, leg.placed, leg.side, leg.size) == (
        "market", "refused", 3, None, None)
    assert leg.reason == "max_open_orders: 1 already waiting"
    assert [event.order for event in strategy.events if event.kind == "refused"] == [leg]


def test_a_handle_read_after_the_run_shows_its_order_at_the_end():
    class Last(probatum.Strategy):

        def init(self, data):
            self.end = len(data) - 2


        def next(self, bar, account):
            if bar.i == self.end:
                self.order = account.buy(1.0)
                self.seen  = self.order.status

    strategy = probatum.backtest(Last(), one(), VENUE).policy

    assert (strategy.seen, strategy.order.status) == ("pending", "filled")


def test_a_bracket_leg_its_partner_cancels_on_the_bar_it_armed_shows_the_side_it_took():
    # placed while flat beside a buy, the bracket arms as the buy fills at the open, and
    # the bar reaches the take-profit, whose fill cancels the stop on that bar
    class Entered(probatum.Strategy):

        def next(self, bar, account):
            if bar.i == 3:
                self.bought = account.buy(1.0)
                self.exits  = account.bracket(stop=bar.close * 0.5,
                                              take_profit=bar.close * 1.0001)

    stop, take_profit = probatum.backtest(Entered(), one(), VENUE).policy.exits

    assert (take_profit.status, stop.status) == ("filled", "cancelled")
    assert (take_profit.side, stop.side)     == ("sell", "sell")
    assert stop.reason == "its OCO partner filled"


class Kept(probatum.Strategy):
    """A buy with its bracket, both kept as handles past the run."""


    def next(self, bar, account):
        if bar.i == 3:
            self.bought = account.buy(1.0)
            self.exits  = account.bracket(stop=bar.close * 0.98,
                                          take_profit=bar.close * 1.02)


def test_a_result_s_policy_runs_again_with_the_handles_it_kept():
    data  = one()
    first = probatum.backtest(Kept(), data, VENUE)
    again = probatum.backtest(first.policy, data, VENUE)

    np.testing.assert_array_equal(again.equity, first.equity)
    assert again.policy.bought is not first.policy.bought


class Weighed(probatum.Strategy):
    """A target at each close of the instruments its row names, NaN holding, with the
    tolerance and the transfer it is given."""


    def __init__(self, rows, tolerance=0.0, transfer=False):
        self.rows      = rows
        self.tolerance = tolerance
        self.transfer  = transfer


    def init(self, data):
        self.names = data.names


    def next(self, bar, account):
        named = {name: float(weight) for name, weight in zip(self.names, self.rows[bar.i])
                 if not math.isnan(weight)}
        if named:
            account.target(named, tolerance=self.tolerance, transfer=self.transfer)


def test_a_target_through_the_account_takes_its_tolerance_and_its_transfer():
    rng  = np.random.default_rng(3)
    rows = rng.uniform(-1.0, 1.0, (60, 2))
    rows[rng.random((60, 2)) < 0.35] = np.nan

    data     = pair()
    tolerant = probatum.backtest(Weighed(rows, tolerance=0.3), data, VENUE)
    array    = probatum.backtest(rows, data, VENUE, tolerance=0.3)

    np.testing.assert_array_equal(tolerant.equity, array.equity)

    every = probatum.backtest(Weighed(rows), data, VENUE)

    assert not np.array_equal(tolerant.equity, every.equity)

    both = probatum.Data(frame(), symbol="BTCUSDT", market="spot").join(
        probatum.Data(frame(seed=1), symbol="BTCUSDT", market="linear"))
    venue  = VENUE.replace(balance={"spot": 1_000.0, "margin": 9_000.0})
    legs   = np.column_stack([np.abs(rows[:, 0]), rows[:, 1]])
    moved  = probatum.backtest(Weighed(legs, transfer=True), both, venue)
    run    = _run(venue, both)
    engine = probatum._probatum.Engine(run.candles, report=True, **run.settings)
    engine.reset()

    for row in legs[:-1]:
        named = {k: float(weight) for k, weight in enumerate(row) if not math.isnan(weight)}
        if named:
            engine.target(named, 0.0, True)
        engine.step()

    np.testing.assert_array_equal(moved.equity, engine.ledger()["equity"])

    kept = probatum.backtest(Weighed(legs), both, venue)

    assert not np.array_equal(moved.equity, kept.equity)


def test_a_verb_on_one_instrument_of_several_is_taken_through_its_view():
    data = pair()

    class Asked(probatum.Strategy):

        def __init__(self, call):
            self.call = call


        def next(self, bar, account):
            self.call(bar, account)

    def run(call):
        probatum.backtest(Asked(call), data, VENUE)

    with pytest.raises(AttributeError, match=r"account.buy: the account holds 2 "
                                             r"instruments; use account\[\"BTCUSDT\"\]"
                                             r".buy"):
        run(lambda bar, account: account.buy(1.0))

    with pytest.raises(AttributeError, match=r"account.position: the account holds 2 "
                                             r"instruments; use account.positions or "
                                             r"account\[\"BTCUSDT\"\].position"):
        run(lambda bar, account: account.position)

    with pytest.raises(KeyError, match="account: no instrument 'SOLUSDT'; use one of "
                                       "BTCUSDT, ETHUSDT"):
        run(lambda bar, account: account["SOLUSDT"])
    with pytest.raises(KeyError, match="weights: no instrument 'SOLUSDT'"):
        run(lambda bar, account: account.target({"SOLUSDT": 0.5}))

    with pytest.raises(ValueError, match=r"weights: 0.5, one number, for 2 instruments"):
        run(lambda bar, account: account.target(0.5))

    cases = [
        (lambda bar, account: account["ETHUSDT"].buy(1.0, limit=9.0, stop=11.0),
         ValueError, "buy: a limit of 9.0 and a stop of 11.0; give one or the other"),
        (lambda bar, account: account["ETHUSDT"].buy(1.0, post_only=True), ValueError,
         "post_only: a market order takes liquidity; give limit= to rest one"),
        (lambda bar, account: account["ETHUSDT"].sell(1.0, stop=9.0, tif="ioc"),
         ValueError, "tif: 'ioc' for a stop, which rests until it triggers"),
        (lambda bar, account: account["ETHUSDT"].buy(1.0, tif="day"), ValueError,
         "tif: 'day'; use \"gtc\", \"ioc\" or \"fok\""),
        (lambda bar, account: account["ETHUSDT"].buy("1"), TypeError,
         "size: '1'; give a number of base units"),
        (lambda bar, account: account["ETHUSDT"].buy(1.0, limit=math.inf), ValueError,
         "limit: inf; give a finite price"),
        (lambda bar, account: account["ETHUSDT"].buy_value(10.0, tif="gtc"), ValueError,
         "tif: \"gtc\" is not a market order's"),
        (lambda bar, account: account["ETHUSDT"].bracket(), ValueError,
         "bracket: no stop and no take_profit; give either or both"),
        (lambda bar, account: account.cancel(7), TypeError,
         "order: 7; pass an Order this account's verbs returned"),
        (lambda bar, account: account.replace(account["ETHUSDT"].buy(
            1.0, stop=bar.closes[1] * 1.1), price=9.0), ValueError,
         "price: 9.0 for a stop order, which has none; move a limit by its price and a "
         "stop by its trigger"),
    ]

    for call, kind, says in cases:
        with pytest.raises(kind, match=re.escape(says)):
            run(call)


def test_a_spot_weight_below_zero_is_refused_naming_the_instrument():
    spot = probatum.Data(frame(), symbol="BTCUSDT", market="spot")

    class Short(probatum.Strategy):

        def next(self, bar, account):
            account.target(-0.5)

    with pytest.raises(ValueError, match="weights: -0.5 for BTCUSDT, which is spot and "
                                         "cannot be short; trade a perpetual instead"):
        probatum.backtest(Short(), spot, VENUE)


class Cross(probatum.Strategy):

    def __init__(self, fast=5, slow=20):
        self.fast = fast
        self.slow = slow


    def init(self, data):
        self.warmup    = self.slow
        self.fast_line = probatum.ta.ema(data.close, self.fast)
        self.slow_line = probatum.ta.ema(data.close, self.slow)
        self.calls     = 0


    def next(self, bar, account):
        self.calls += 1
        account.target(1.0 if self.fast_line[bar.i] > self.slow_line[bar.i] else -0.5)


def test_a_backtest_runs_a_copy_of_the_strategy_and_two_runs_start_alike():
    data, strategy = one(), Cross()
    first  = probatum.backtest(strategy, data, VENUE)
    second = probatum.backtest(strategy, data, VENUE)

    assert not hasattr(strategy, "calls") and first.policy is not strategy
    assert first.policy.calls == second.policy.calls == len(data) - 1 - 20
    np.testing.assert_array_equal(first.equity, second.equity)
    assert first.provenance["policy"] == "Cross(fast=5, slow=20)"
    assert repr(first).startswith("Result(Cross(fast=5, slow=20) on probatum.Venue(")


def test_an_array_run_s_settings_and_a_bad_warmup_are_refused_beside_a_strategy():
    data = one()

    for given, says in [
        (dict(stop=0.02), "stop: 0.02 beside a Strategy, which places its own exits; "
                          "place them with account.bracket"),
        (dict(take_profit=0.04), "take_profit: 0.04 beside a Strategy"),
        (dict(sizing="quantity"), "sizing: 'quantity' beside a Strategy, whose verbs "
                                  "say their own units; leave it out"),
        (dict(tolerance=0.1), "tolerance: 0.1 beside a Strategy; pass it to "
                              "account.target"),
        (dict(risk_free=math.nan), "risk_free: nan; give the annual rate"),
    ]:
        with pytest.raises(ValueError, match=re.escape(says)):
            probatum.backtest(Cross(), data, VENUE, **given)

    for warmup in (-1, 2.5, True, "3"):
        with pytest.raises(ValueError, match=re.escape(f"warmup: {warmup!r}; give a "
                                                       f"whole number of bars from 0")):
            probatum.backtest(Seen(warmup=warmup), data, VENUE)


def test_the_engine_stepped_by_hand_records_what_a_backtest_of_its_decisions_does():
    data = pair()

    def decide(bar, account):
        if bar.i % 7 == 3:
            return account.target({"BTCUSDT": 0.5, "ETHUSDT": -0.3})

        if bar.i % 7 == 6:
            account["ETHUSDT"].close()

    class Same(probatum.Strategy):

        def next(self, bar, account):
            decide(bar, account)

    engine, episodes = probatum.Engine(data, VENUE), []
    for _ in range(2):
        bar = engine.reset()
        assert bar.i == 0 and not engine.done() and not engine.events
        assert engine.equity == 10_000.0

        legs = None
        while not engine.done():
            placed = decide(bar, engine)
            legs   = legs or placed
            bar    = engine.step()

        assert bar.i == len(data) - 1 and engine.step().i == bar.i
        assert {leg.status for leg in legs} == {"filled"}

        episodes.append(engine.result(risk_free=0.04))

    backtest = probatum.backtest(Same(), data, VENUE, risk_free=0.04)

    for by_hand in episodes:
        for name in backtest.ledger:
            np.testing.assert_array_equal(by_hand.ledger[name], backtest.ledger[name])
        assert by_hand.policy is None and by_hand.stats == backtest.stats
        # ids run on across a reset, and nothing else does
        assert [{**fill, "order": None} for fill in by_hand.fills] == [
            {**fill, "order": None} for fill in backtest.fills]

    assert (episodes[0].fills, episodes[0].orders, episodes[0].trades) == (
        backtest.fills, backtest.orders, backtest.trades)

    with pytest.raises(AttributeError, match=re.escape(
            "engine.buy: the account holds 2 instruments; use engine[\"BTCUSDT\"].buy")):
        engine.buy(1.0)

    with pytest.raises(KeyError, match="engine: no instrument 'SOLUSDT'"):
        engine["SOLUSDT"]


def test_the_engine_takes_a_seed_where_backtest_refuses_a_venue_that_draws():
    data  = one()
    drawn = VENUE.replace(intrabar="random")

    with pytest.raises(ValueError, match="venue: intrabar is \"random\", which backtest "
                                         "refuses"):
        probatum.backtest(Cross(), data, drawn)

    # the bar after the entry reaches both legs, which the venue orders at random
    wide = frame()
    for column, by in (("high", 1.02), ("low", 0.98)):
        wide.iloc[11, wide.columns.get_loc(column)] *= by
    data = probatum.Data(wide, symbol="BTCUSDT", market="linear")

    def run(seed):
        engine = probatum.Engine(data, drawn, seed=seed)
        bar    = engine.reset()

        while not engine.done():
            if bar.i == 10:
                engine.buy(5.0)
                engine.bracket(stop=bar.close * 0.995, take_profit=bar.close * 1.005)
            bar = engine.step()

        return engine.result()

    assert run(1).ledger["equity"].tolist() == run(1).ledger["equity"].tolist()
    assert run(1).provenance["seed"]        == 1
    assert {run(seed).trades[0]["reason"] for seed in range(16)} == {"stop",
                                                                      "take_profit"}
