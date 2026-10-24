"""Throughput of each way to run probatum, with one simple strategy per surface.

Built and run in the Docker bench image so the numbers come from the same abi3
wheel the gate builds, not a dev checkout:

    docker build --target bench-surfaces -t probatum-bench-surfaces .
    docker run --rm probatum-bench-surfaces

Every number is wall-clock throughput on one machine and one series; treat them as
relative between surfaces, not as absolute hardware figures. The single-env timings
take the best of a few repeats; the batched ones run once over a fixed amount of
work.
"""

import os
import platform
import time

import numpy as np

import probatum
from probatum import Batch
from probatum._probatum import Engine
from probatum import Backtester, Strategy

BARS = 10_000
FAST = 10
SLOW = 50


def make_series(n):
    t      = np.arange(n, dtype=np.float64)
    close  = 100.0 + 20.0 * np.sin(t / 15.0) + 0.05 * t
    high   = close + 1.0
    low    = close - 1.0
    volume = np.full(n, 1000.0)
    return np.column_stack([close, high, low, close, volume])


class SmaCross(Strategy):
    """Fast/slow moving-average crossover: long when fast crosses above slow, flat
    when it crosses back.
    """


    def __init__(self, fast=FAST, slow=SLOW):
        self.fast = int(fast)
        self.slow = int(slow)


    def init(self, engine):
        self.close = engine.data[:, 3]


    def next(self, state, engine):
        i = state["tick_index"]
        if i < self.slow:
            return

        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()

        if state["position"] == 0 and fast > slow:
            engine.market_buy(1.0)
        elif state["position"] > 0 and fast < slow:
            engine.close()


def best_rate(work, amount, repeats=3):
    best = 0.0

    for _ in range(repeats):
        t0 = time.perf_counter()
        work()
        dt   = time.perf_counter() - t0
        best = max(best, amount / dt)

    return best


def bench_engine_raw(data):
    steps = data.shape[0] - 1

    def run():
        eng = Engine(data, market="spot", quote=100_000)
        eng.reset()
        while not eng.done():
            eng.step()

    return best_rate(run, steps)


def bench_engine(data):
    close = data[:, 3]
    steps = data.shape[0] - 1

    def run():
        eng   = Engine(data, market="spot", quote=100_000)
        state = eng.reset()

        while not eng.done():
            i = state["tick_index"]
            if i >= SLOW:
                fast = close[i - FAST:i].mean()
                slow = close[i - SLOW:i].mean()
                if state["position"] == 0 and fast > slow:
                    eng.market_buy(1.0)
                elif state["position"] > 0 and fast < slow:
                    eng.close()

            state = eng.step()

    return best_rate(run, steps)


def bench_backtester(data):
    bt    = Backtester(data, market="spot", quote=100_000)
    steps = data.shape[0] - 1

    return best_rate(lambda: bt.run(SmaCross()), steps)


def weighted(width, bars):
    t      = np.arange(bars, dtype=np.float64)
    closes = np.stack([100.0 + 20.0 * np.sin(t / 15.0 + k) + 0.05 * t
                       for k in range(width)], axis=1)

    candles = np.stack([closes, closes + 1.0, closes - 1.0, closes,
                        np.full_like(closes, 1e6)], axis=-1)

    times = np.arange(bars, dtype=np.int64) * 3_600_000_000_000
    data  = probatum.Data(
        candles,
        time   = times,
        market = "linear",
        symbol = [f"I{k}" for k in range(width)],
    )

    weights = np.full((bars, width), np.nan)
    for k in range(width):
        weights[k % 50::50, k]        = 1.0 / width
        weights[(k + 25) % 50::50, k] = 0.0

    return data, weights


def bench_array(width, bars=50_000):
    data, weights = weighted(width, bars)
    venue = probatum.Venue()

    return best_rate(lambda: probatum.backtest(weights, data, venue), bars)


class Decided(Strategy):
    """The array's weights decided at each close, a target naming the instruments its
    row gives, or with ``every`` the weight last given held by a target at every
    close of one instrument."""


    def __init__(self, weights, every=False):
        self.weights = weights
        self.every   = every


    def init(self, data):
        names      = data.names
        self.given = [{} for _ in range(len(data))]
        for t, k in zip(*np.nonzero(~np.isnan(self.weights))):
            self.given[t][names[k]] = float(self.weights[t, k])

        self.held = 0.0


    def next(self, bar, account):
        given = self.given[bar.i]

        if self.every:
            self.held = given.get("I0", self.held)
            account.target(self.held)
        elif given:
            account.target(given)


def bench_strategy(width, every=False, bars=50_000):
    data, weights = weighted(width, bars)
    venue = probatum.Venue()

    return best_rate(lambda: probatum.backtest(Decided(weights, every), data, venue),
                     bars)


def crossing(data, fast, slow):
    up      = probatum.ta.sma(data.close, fast)
    down    = probatum.ta.sma(data.close, slow)
    weights = np.full(len(data), np.nan)
    weights[probatum.ta.crossover(up, down)]  = 1.0
    weights[probatum.ta.crossunder(up, down)] = 0.0
    return weights


class Crossing(Strategy):
    """``crossing``'s weights decided at each close, a trial at a time."""


    def __init__(self, fast, slow):
        self.fast = fast
        self.slow = slow


    def init(self, data):
        self.weights = crossing(data, self.fast, self.slow)


    def next(self, bar, account):
        if not np.isnan(self.weights[bar.i]):
            account.target(self.weights[bar.i])


def bench_search(policy, trials, n_jobs):
    data, _weights = weighted(1, BARS)
    space = {"fast": probatum.Int(5, 40), "slow": probatum.Int(40, 200)}
    t0    = time.perf_counter()

    probatum.search(policy, space, data, probatum.Venue(), sampler="random",
                    trials=trials, holdout=0, n_jobs=n_jobs, seed=0)
    return trials / (time.perf_counter() - t0)


def bench_env(num_envs, steps):
    data, _weights = weighted(1, BARS)
    env = probatum.env(data, probatum.Venue(), window=32, num_envs=num_envs, seed=0)
    obs, _ = env.reset(seed=0)

    # the close of the window's last bar and of its first, over the last close
    last, first = 31 * 5 + 3, 3

    def run():
        nonlocal obs
        for _ in range(steps):
            actions = np.where(obs[:, last] > obs[:, first], 1.0, -1.0)[:, None]
            obs, _r, _t, _tr, _i = env.step(actions.astype(np.float32))

    return best_rate(run, num_envs * steps, repeats=2)


def bench_batch(data, num_envs, steps):
    batch = Batch(data, num_envs=num_envs, market="perp", quote=10_000)
    batch.reset_all()
    sizes = np.zeros(num_envs)

    def run():
        for _ in range(steps):
            if batch.done():
                batch.reset_all()
            position = batch.snapshot()["position"]
            sizes[:] = np.where(position == 0.0, 0.1, 0.0)
            batch.advance(sizes)

    return best_rate(run, num_envs * steps, repeats=2)


def main():
    cores = os.cpu_count()

    print(f"# probatum surface benchmarks")
    print(f"# python {platform.python_version()}, {platform.machine()}, cpu_count={cores}")
    print(f"# series: {BARS} bars, strategy: sma_cross(fast={FAST}, slow={SLOW})")
    print()

    data = make_series(BARS)

    engine_raw_bps = bench_engine_raw(data)
    engine_bps     = bench_engine(data)
    backtester_bps = bench_backtester(data)

    trials       = 48
    search_array = bench_search(crossing, trials, n_jobs=1)
    search_seq   = bench_search(Crossing, trials, n_jobs=1)
    search_par   = bench_search(Crossing, trials, n_jobs=-1)

    array_one      = bench_array(1)
    array_fifty    = bench_array(50)
    strategy_one   = bench_strategy(1)
    strategy_every = bench_strategy(1, every=True)
    strategy_fifty = bench_strategy(50)

    rl_envs, rl_steps = 1_024, 2_000
    env_eps   = bench_env(num_envs=rl_envs, steps=rl_steps)
    batch_eps = bench_batch(data, num_envs=rl_envs, steps=rl_steps)

    print("surface,metric,value,detail")
    print(f"engine_raw_step,bars_per_sec,{engine_raw_bps:.0f},single env no strategy")
    print(f"engine_own_loop,bars_per_sec,{engine_bps:.0f},single env python loop")
    print(f"backtester,bars_per_sec,{backtester_bps:.0f},single env reporting wrapper")
    print(f"array,bars_per_sec,{array_one:.0f},50000 bars N=1 weights in one call")
    print(f"array,bars_per_sec,{array_fifty:.0f},50000 bars N=50 weights in one call")
    print(f"strategy,bars_per_sec,{strategy_one:.0f},50000 bars N=1 the same weights "
          f"decided at each close")
    print(f"strategy,bars_per_sec,{strategy_every:.0f},50000 bars N=1 a target at every "
          f"close")
    print(f"strategy,bars_per_sec,{strategy_fifty:.0f},50000 bars N=50 the same weights "
          f"decided at each close")
    print(f"search,trials_per_sec,{search_array:.2f},{trials} trials of weights run together")
    print(f"search,trials_per_sec,{search_seq:.2f},n_jobs=1 {trials} trials of a strategy")
    print(f"search,trials_per_sec,{search_par:.2f},n_jobs=-1 {trials} trials of a strategy "
          f"{cores} cores")
    print(f"env,env_steps_per_sec,{env_eps:.0f},{rl_envs} envs x {rl_steps} steps")
    print(f"batch,env_steps_per_sec,{batch_eps:.0f},{rl_envs} envs x {rl_steps} steps")


if __name__ == "__main__":
    main()
