"""An episode runs again as a backtest from its start.

An environment's account is a backtest's: its decision at each close fills as a
policy's would from the next open, through the same fills, costs, caps and
liquidation. So an episode, found again from the start its info gives, replays
through probatum.backtest on the Data from that row with the weights it traded, and
the two equities agree to the bit, its slots read from the info where it samples
them (Specification, The environment).
"""

import numpy as np
import pytest

import probatum

HOUR  = 3_600_000_000_000
VENUE = probatum.venues.binance_perp.replace(funding="none")


def walked(price, rows, rng):
    """A random walk's bars from `price`, each opening at the last close."""
    close  = price * np.exp(np.cumsum(rng.normal(0.0, 0.01, rows)))
    opened = np.concatenate([[price], close[:-1]])
    high   = np.maximum(opened, close) * (1.0 + rng.uniform(0.0, 0.004, rows))
    low    = np.minimum(opened, close) * (1.0 - rng.uniform(0.0, 0.004, rows))

    return np.column_stack([opened, high, low, close, rng.uniform(1e3, 1e4, rows)])


def universe(rows=300, seed=3):
    """Two perpetuals on one hourly clock, at the prices their venue's rules fit."""
    rng   = np.random.default_rng(seed)
    times = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * HOUR
    btc   = probatum.Data(
        walked(30_000.0, rows, rng),
        symbol = "BTCUSDT",
        market = "linear",
        time   = times,
    )
    eth   = probatum.Data(
        walked(2_000.0, rows, rng),
        symbol = "ETHUSDT",
        market = "linear",
        time   = times,
    )

    return btc.join(eth)


@pytest.mark.parametrize("sample", [None, 1])
def test_each_episode_s_equity_is_its_backtest_s_to_the_bit(sample):
    data  = universe()
    draws = np.random.default_rng(5)
    traded, equities = [], []

    def decode(actions, state):
        weights = draws.uniform(-1.0, 1.0, state.weights.shape)
        traded.append(weights)
        return weights

    def reward(state):
        equities.append(state.equity.copy())
        return state.log_return

    env = probatum.env(data, VENUE, window=8, num_envs=3, seed=11, decode=decode,
                       reward=reward, sample=sample, episode_length=40,
                       autoreset="disabled", max_leverage=2.0)

    _, info = env.reset()
    active, steps = np.ones(3, dtype=bool), np.zeros(3, dtype=np.int64)
    while active.any():
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += active
        active &= ~(terminated | truncated)

    traded, equities = np.array(traded), np.array(equities)
    for env_of, start in enumerate(info["start"]):
        n       = steps[env_of]
        slots   = info["instruments"][env_of] if sample else [0, 1]
        weights = np.full((len(data) - start, 2), np.nan)
        weights[:n, slots] = traded[:n, env_of]
        result = probatum.backtest(weights, data[int(start):], VENUE)
        assert np.array_equal(result.equity[1:n + 1], equities[:n, env_of])
