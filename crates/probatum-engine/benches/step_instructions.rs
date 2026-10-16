//! Instruction counts for the step loop, measured under Valgrind. A count, unlike a
//! timing, does not depend on the machine, so CI compares it with the commit before
//! and fails when it rises by more than 10%. Run with
//! `cargo bench -p probatum-engine --bench step_instructions` where Valgrind and
//! `gungraun-runner` 0.20.0 are installed, or through the `bench-instructions`
//! Docker stage.

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use probatum_core::{Candle, Market};
use probatum_engine::{
    Candles, Engine, EngineConfig, Impact, Intrabar, Limit, Liquidation, Spread,
};

/// The same oscillating series and perp config the wall-clock bench steps over.
fn engine(bars: usize) -> Engine {
    let candles: Vec<Candle> = (0..bars)
        .map(|i| {
            let base = 100.0 + ((i % 40) as f64);
            Candle {
                open: base,
                high: base + 4.0,
                low: base - 4.0,
                close: base + 1.0,
                volume: 100_000.0,
            }
        })
        .collect();
    let config = EngineConfig {
        market: Market::Perp,
        quote: 1_000_000.0,
        idle_cash: 0.0,
        fee_taker: 0.0006,
        fee_maker: 0.0002,
        slippage_bps: 2.0,
        budget: 1.0,
        window: 24,
        max_open_orders: 8,
        report: false,
        max_leverage: 0.0,
        impact: 0.0,
        funding_rate: 0.0,
        funding_interval: 0,
        intrabar: Intrabar::Pessimistic,
        liquidation: Liquidation::Takeover,
        impact_model: Impact::Linear,
        spread: Spread::None,
        limit: Limit::Touch,
        sizes: false,
        liquidation_fee: None,
    };
    Engine::new(Candles::new(candles), config)
}

#[library_benchmark]
#[bench::bars_1000(args = [1_000], setup = engine)]
fn step_flat(mut engine: Engine) -> Engine {
    while !engine.done() {
        black_box(engine.step());
    }
    engine
}

#[library_benchmark]
#[bench::bars_1000(args = [1_000], setup = engine)]
fn step_trading(mut engine: Engine) -> Engine {
    while !engine.done() {
        engine.market_buy(black_box(0.001));
        black_box(engine.step());
    }
    engine
}

library_benchmark_group!(name = step, benchmarks = [step_flat, step_trading]);

main!(library_benchmark_groups = step);
