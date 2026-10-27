"""Wheel-level throughput benchmark for probatum.

Covers the paths a user actually runs, not just the raw step:

- single-env ``Engine.step``, which crosses the Python-to-Rust boundary each call;
- batched ``Batch.advance``, which releases the GIL and steps every env in parallel;
- ``backtest`` asking a Python ``Strategy`` at each close (flexible, slow);
- a full ``probatum.env`` step, including the decode, the observation and the reward.

It prints steps/sec and env-steps/sec, the only figures a speed claim in the docs
may cite. Run it inside the Docker bench stage: ``docker build --target bench .``
"""

import time

import numpy as np

from probatum import Batch
from probatum._probatum import Engine


def synthetic(bars):
    i      = np.arange(bars, dtype=np.float64)
    base   = 100.0 + 20.0 * np.sin(i / 25.0)
    volume = np.full(bars, 100_000.0)
    return np.stack([base, base + 2.0, base - 2.0, base, volume], axis=1)


def time_single(data, passes):
    engine = Engine(data, market="perp", quote=1_000_000.0, slippage_bps=2.0)
    bars   = data.shape[0]
    start  = time.perf_counter()

    for _ in range(passes):
        engine.reset()
        while not engine.done():
            engine.market_buy(0.001)
            engine.step()

    elapsed = time.perf_counter() - start
    return (passes * bars) / elapsed


def time_batch(data, num_envs, passes):
    batch = Batch(
        data,
        num_envs     = num_envs,
        market       = "perp",
        quote        = 1_000_000.0,
        slippage_bps = 2.0,
    )

    bars  = data.shape[0]
    sizes = np.full(num_envs, 0.001, dtype=np.float64)
    start = time.perf_counter()

    for _ in range(passes):
        batch.reset_all()
        while not batch.done():
            batch.advance(sizes)

    elapsed = time.perf_counter() - start
    return (passes * bars * num_envs) / elapsed


def time_backtest(data, passes):
    import probatum

    class SmaCross(probatum.Strategy):
        """Long while the 10-bar average is over the 30-bar one."""

        warmup = 30


        def init(self, data):
            self.fast = probatum.ta.sma(data.close, 10)
            self.slow = probatum.ta.sma(data.close, 30)


        def next(self, bar, account):
            up = self.fast[bar.i] > self.slow[bar.i]
            if account.position == 0.0 and up:
                account.buy(1.0)
            elif account.position != 0.0 and not up:
                account.close()

    bars  = probatum.Data(
        data,
        symbol = "BTCUSDT",
        market = "spot",
        time   = np.arange(len(data), dtype=np.int64) * 3_600_000_000_000,
    )
    venue = probatum.Venue(balance=100_000.0, fills={"slippage": {"fixed": 2.0}})
    start = time.perf_counter()

    for _ in range(passes):
        probatum.backtest(SmaCross(), bars, venue)

    elapsed = time.perf_counter() - start
    return (passes * len(data)) / elapsed


def time_rl(data, num_envs, steps):
    import probatum

    bars  = probatum.Data(
        data,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(len(data), dtype=np.int64) * 3_600_000_000_000,
    )
    venue = probatum.Venue(balance=1_000_000.0, fills={"slippage": {"fixed": 2.0}})
    env   = probatum.env(bars, venue, window=64, num_envs=num_envs, copy=False, seed=0)
    env.reset(seed=0)

    rng     = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, (steps, num_envs, 1)).astype(np.float32)
    start   = time.perf_counter()

    for step in range(steps):
        env.step(actions[step])

    elapsed = time.perf_counter() - start
    return (steps * num_envs) / elapsed


def main():
    data = synthetic(2_000)

    print("probatum throughput (synthetic 2000-bar series, honest costs on)")
    print("-- raw stepping --")
    print(f"single-env Engine.step     : {time_single(data, 5):>16,.0f} steps/sec")
    for num_envs in (1_024, 4_096, 8_192):
        rate = time_batch(data, num_envs, 3)
        print(f"batch advance n={num_envs:<5}       : {rate:>16,.0f} env-steps/sec")

    print("-- wrappers --")
    print(f"backtest (python Strategy) : {time_backtest(data, 20):>16,.0f} bar-steps/sec")
    for num_envs in (1_024, 4_096):
        rate = time_rl(data, num_envs, 300)
        print(f"probatum.env step n={num_envs:<5}   : {rate:>16,.0f} env-steps/sec")


if __name__ == "__main__":
    main()
