"""A validation's path is one run of each window's Strategy in turn, and at a seam the
account passes to the next: every order the outgoing window left is cancelled with the
reason seam, waiting ones first, the incoming window's on_inherit(bar, account) sees the
position with nothing resting and no events, and its round trips and equity curve
start from that close; under flatten the outgoing window's last close closes every
position instead of deciding, its fill reading seam, as the Specification works it.
"""

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum._backtest import _decided
from probatum._validate import _Path, _Walk
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE


def handed(n=20):
    """Two flat instruments over ``n`` hours, ETHUSDT printing no bar at row 10."""
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    def flat(level):
        return pd.DataFrame({
            "open":   level,
            "high":   level * 1.001,
            "low":    level * 0.999,
            "close":  level,
            "volume": 1e6,
        }, index=index)

    eth = flat(10.0).drop(index[10])
    return probatum.Data({"BTCUSDT": flat(100.0), "ETHUSDT": eth}, market="linear")


class Handover(probatum.Strategy):
    """The Specification's worked example: a round trip in BTC closed by row 4, which
    it reads at row 8, a long of 1 BTC from row 6 with a sell stop resting at 95, and a
    buy of ETH at row 9; and what the window after the seam inherits and reads, and the
    long it closes."""


    def next(self, bar, account):
        btc = account["BTCUSDT"]

        if bar.i in (0, 5):
            btc.buy(1.0)
        elif bar.i == 3:
            btc.sell(1.0)
        elif bar.i == 6:
            btc.sell(1.0, stop=95.0)
        elif bar.i == 8:
            self.read = len(account.trades)
        elif bar.i == 9:
            account["ETHUSDT"].buy(1.0)
        elif bar.i == 10:
            self.ordered = hasattr(self, "inherited")
        elif bar.i == 11:
            events     = [event.kind for event in account.events]
            self.later = events, account.equity_curve, len(account.trades)
        elif bar.i == 12 and btc.position > 0.0:
            btc.sell(1.0)
        elif bar.i == 14:
            self.closed = [(trip["open_bar"], trip["close_bar"])
                           for trip in account.trades]


    def on_inherit(self, bar, account):
        self.inherited = dict(row=bar.i, btc=account["BTCUSDT"].position,
                              eth=account["ETHUSDT"].position, orders=account.orders,
                              events=account.events, trades=len(account.trades),
                              curve=account.equity_curve)


def walked(seam):
    """The path of window 1, deciding at rows 0 to 9, and window 2, from row 10 on."""
    return _decided(_Path(Handover, [(0, 9, {}), (10, 18, {})], seam), handed(), VENUE,
                    "weight", None, None, 0.0, 0.0, "full")


def test_a_seam_cancels_what_the_outgoing_window_left_and_hands_over_the_position():
    result    = walked("cancel")
    orders    = result.orders
    placed    = orders[orders["kind"] == "placed"]
    stop      = placed["order"][placed["price"] == 95.0][0]
    eth       = placed["order"][placed["instrument"] == "ETHUSDT"][0]
    cancelled = orders[orders["kind"] == "cancelled"]

    # the buy still waiting for ETH's next bar, then the stop resting on the book
    assert list(cancelled["order"])  == [eth, stop]
    assert list(cancelled["bar"])    == [10, 10]
    assert list(cancelled["reason"]) == ["seam"] * 2

    first, second = result.policy.parts
    # on_inherit hands the position over before the window's first next
    assert second.ordered

    inherited = second.inherited
    assert (inherited["row"], inherited["btc"], inherited["eth"]) == (10, 1.0, 0.0)
    assert inherited["orders"] == [] and inherited["events"] == []

    # the round trip closed at row 4 is the outgoing window's, which read it, and the
    # incoming one reads none of it
    closed = result.trades[~result.trades["open"]]
    assert list(zip(closed["open_bar"], closed["close_bar"])) == [(1, 4), (6, 13)]
    assert first.read == 1 and inherited["trades"] == 0
    np.testing.assert_array_equal(inherited["curve"], result.equity[10:11])

    kinds, curve, trades = second.later
    assert kinds == [] and list(curve) == list(result.equity[10:12]) and trades == 0
    # the long it inherited, opened at row 6 and closed at row 13, is its first
    assert second.closed == [(6, 13)]
    assert not hasattr(first, "inherited") and not hasattr(first, "later")
    assert result.policy.decided == [10, 9]


def test_a_flattening_seam_closes_each_position_instead_of_the_last_decision():
    result = walked("flatten")
    orders = result.orders

    # window 1 does not decide at row 9, so it buys no ETH
    assert not (orders["instrument"] == "ETHUSDT").any()

    cancelled = orders[orders["kind"] == "cancelled"]
    assert list(cancelled["bar"]) == [9] and list(cancelled["reason"]) == ["seam"]

    fills  = result.fills
    closed = fills[fills["reason"] == "seam"]
    assert (list(closed["bar"]), list(closed["side"]), list(closed["size"])) == (
        [10], ["sell"], [1.0])

    second    = result.policy.parts[1]
    inherited = second.inherited
    assert second.ordered
    assert (inherited["row"], inherited["btc"], inherited["orders"]) == (10, 0.0, [])
    assert inherited["trades"] == 0 and second.closed == []
    assert result.policy.decided == [9, 9]


def test_a_window_owns_the_rows_after_its_decisions_as_the_page_works_it():
    printed = np.arange(20)
    owned   = _Walk(0, [(0, 9, 0), (10, 18, 1)], printed).owned

    assert owned == [(1, 10), (11, 19)]


@pytest.mark.parametrize("seam", ["cancel", "flatten"])
def test_each_window_decides_at_its_own_closes_from_its_own_warm_up(seam):
    seen = []

    class Counted(probatum.Strategy):

        def __init__(self, warmup):
            self.warmup = warmup


        def next(self, bar, account):
            seen.append((self.warmup, bar.i))

    windows = [(4, 9, {"warmup": 0}), (10, 14, {"warmup": 12}), (15, 18, {"warmup": 0})]
    result  = _decided(_Path(Counted, windows, seam), handed(), VENUE, "weight", None,
                       None, 0.0, 0.0, "full")

    # nothing decides before the first window, each window from its warm-up, and under
    # flatten no window but the last at its last row
    kept  = list(range(4, 10 if seam == "cancel" else 9))
    later = list(range(12, 15 if seam == "cancel" else 14))

    assert seen == ([(0, row) for row in kept] + [(12, row) for row in later]
                    + [(0, row) for row in range(15, 19)])
    assert result.policy.decided == [len(kept), len(later), 4]
