"""An environment decides and rewards as its settings say.

What the engine works out each step it is handed by the Python layer: an order's
ticks from the venue's tick and max_offset, the Sharpe's rate from the window, the
floor from the opening balance, and the features on the Data's own rows, a slice's
among them. Each is read back here off what a step returns (Specification, The
environment).
"""

import math

import numpy as np
import pytest

import probatum
from test_an_episode_runs_again_as_a_backtest_from_its_start import HOUR, VENUE, universe

PLAIN = probatum.Venue(fees={"maker": 0.0, "taker": 0.0})


def flat(rows, market="linear"):
    """Bars at 100 that open where they close, on an hourly clock."""
    bars  = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (rows, 1))
    times = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * HOUR
    return bars, times


def test_an_order_of_half_its_offset_rests_ten_ticks_below_the_close():
    # the Specification's worked example: 0.25 at max_leverage 2 and an offset of half
    # of 20 ticks, a buy of 0.1 BTC ten ticks of 0.1 below a close of 50,000, which a
    # flat bar never reaches
    bars, times = flat(40)
    data = probatum.Data(
        bars * [500.0, 500.0, 500.0, 500.0, 1.0],
        symbol = "BTCUSDT",
        market = "linear",
        time   = times,
    )
    env  = probatum.env(data, VENUE, window=4, actions="orders", max_leverage=2.0,
                        max_offset=20)

    env.reset(seed=0)
    obs, *_ = env.step(np.array([[[0.25, 0.5]]], dtype=np.float32))

    # past four bars of candles, the slot's weight, PnL, distance and shown flag
    assert obs[0, 4 * 5 + 4:4 * 5 + 6].tolist() == [1.0, 10.0]


def test_an_order_ten_ticks_below_fills_on_a_bar_trading_through_it_in_dollars():
    # the venue fills a limit half a tick past it, so a dip to 49,998.9 fills a buy at
    # 49,999.0 and no limit further off; the last two bars dip, whichever starts
    bars, times = flat(6)
    bars = bars * [500.0, 500.0, 500.0, 500.0, 1.0]
    bars[4:, 2] = 49_998.9

    data = probatum.Data(bars, symbol="BTCUSDT", market="linear", time=times)
    env  = probatum.env(data, VENUE, window=4, actions="orders", max_leverage=2.0,
                        max_offset=20)

    env.reset(seed=0)
    obs, *_ = env.step(np.array([[[0.25, 0.5]]], dtype=np.float32))

    # 0.1 BTC held, about half the equity, and nothing left resting
    assert obs[0, 4 * 5]     == pytest.approx(0.5, rel=1e-3)
    assert obs[0, 4 * 5 + 4] == 0.0


def test_a_decode_s_nan_leaves_its_slot_unnamed_and_its_position_held():
    decided = iter([[0.5, 0.5], [np.nan, 0.0]])

    def decode(actions, state):
        return np.array([next(decided)])

    env = probatum.env(universe(), VENUE, window=8, decode=decode, seed=4)

    env.reset()
    env.step(np.zeros((1, 2), dtype=np.float32))
    obs, *_ = env.step(np.zeros((1, 2), dtype=np.float32))

    # past eight bars of two slots' candles, each slot's weight four values apart
    held, closed = obs[0, 8 * 2 * 5], obs[0, 8 * 2 * 5 + 4]
    assert held == pytest.approx(0.5, abs=0.05) and closed == 0.0


def stepped(reward, steps=30):
    """The rewards and the ends of three environments' steps on the same episodes
    and actions, rewarded by `reward`."""
    env = probatum.env(universe(), VENUE, window=8, num_envs=3, seed=2, reward=reward)
    env.reset()

    actions = np.random.default_rng(4).uniform(-1.0, 1.0, (steps, 3, 2))
    paid, ended = [], []
    for row in actions.astype(np.float32):
        _, got, terminated, truncated, _ = env.step(row)
        paid.append(got)
        ended.append(terminated | truncated)

    return np.array(paid), np.array(ended)


def test_the_sharpe_reward_moves_its_moments_at_two_over_one_more_than_the_window():
    logged, ended = stepped("log")
    paid, _       = stepped("sharpe")

    eta = 2.0 / (8 + 1)
    a, b      = np.zeros(3), np.zeros(3)
    restarted = np.zeros(3, dtype=bool)

    for returned, got, done in zip(logged, paid, ended):
        # a step that restarts an environment pays nothing, and its episode's moments
        # start again from zero
        a, b   = np.where(restarted, 0.0, a), np.where(restarted, 0.0, b)
        da, db = returned - a, returned * returned - b
        spread = b - a * a
        held   = (spread > 0.0) & ~restarted
        want   = np.where(held, (b * da - a * db / 2.0)
                          / np.where(held, spread, 1.0) ** 1.5, 0.0)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)

        a         = np.where(restarted, a, a + eta * da)
        b         = np.where(restarted, b, b + eta * db)
        restarted = done


def test_an_episode_ends_at_the_floor_a_hundredth_of_its_balance():
    # a spot holding bought at 100 that the bar of row 20 takes to 0.5: 50 of the
    # 10,000 balance, below its floor of 100, which no liquidation reached
    bars, times = flat(30)
    bars[20]  = [100.0, 100.0, 0.5, 0.5, 1e6]
    bars[21:] = [0.5, 0.5, 0.5, 0.5, 1e6]

    data = probatum.Data(bars, symbol="BTCUSDT", market="spot", time=times)
    env  = probatum.env(data, PLAIN, window=4, num_envs=16, seed=3, autoreset="disabled")

    _, info = env.reset()
    early   = info["start"] <= 19

    assert early.any() and not early.all()

    ended = np.zeros(16, dtype=bool)
    paid  = np.zeros(16)
    while not ended.all():
        _, got, terminated, truncated, _ = env.step(np.ones((16, 1), dtype=np.float32))
        paid = np.where(terminated & ~ended, got, paid)
        assert not (terminated & ~early).any()
        ended |= terminated | truncated

    assert paid[early] == pytest.approx(math.log(100.0 / 10_000.0), rel=1e-12)


def test_a_step_gives_the_same_results_on_any_number_of_threads():
    def run(threads):
        env = probatum.env(universe(), VENUE, window=8, num_envs=6, threads=threads,
                           actions="orders", seed=3)

        assert env._inner.threads == threads

        obs, _ = env.reset()
        seen    = [obs]
        actions = np.random.default_rng(5).uniform(0.0, 1.0, (40, 6, 2, 2))
        actions[..., 0] = 2.0 * actions[..., 0] - 1.0
        for row in actions.astype(np.float32):
            seen.extend(env.step(row)[:4])

        return seen

    for one, three in zip(run(1), run(3)):
        assert np.array_equal(one, three)


def test_a_reward_reads_a_state_it_cannot_write_into():
    seen = []

    def reward(state):
        seen.append(state)
        return state.log_return

    env = probatum.env(universe(), VENUE, window=8, num_envs=2, seed=1, reward=reward)

    env.reset()
    env.step(np.zeros((2, 2), dtype=np.float32))

    assert not any(getattr(seen[0], name).flags.writeable
                   for name in probatum._env.STATE)


def test_an_infinite_warm_up_is_a_warm_up_and_never_observed():
    # a feature's log of a quantity that starts at zero, as Data.features takes it
    def logged(data):
        rows = np.arange(len(data), dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.broadcast_to(np.log(rows)[:, None, None], (len(data), 2, 1)).copy()

    env = probatum.env(universe(), VENUE, window=8, num_envs=4, features=logged, seed=2)
    obs, info = env.reset()

    assert np.isfinite(obs).all() and (info["start"] >= 8).all()


def test_features_line_up_with_the_bars_of_a_slice():
    # each bar's feature its row of the whole Data, which a function is given with the
    # bars before the slice
    def rows(data):
        return np.broadcast_to(np.arange(len(data), dtype=np.float64)[:, None, None],
                               (len(data), 2, 1)).copy()

    env = probatum.env(universe()[100:], VENUE, window=8, num_envs=4, seed=5,
                       features=rows)
    obs, info = env.reset()

    # the market block is (8, 2, 1): the window's first and last bar of the first slot
    assert obs[:, 0].tolist()     == (100 + info["start"] - 7).astype(np.float32).tolist()
    assert obs[:, 7 * 2].tolist() == (100 + info["start"]).astype(np.float32).tolist()
