//! probatum-engine: the bar-level realization of the market simulator. It owns the
//! shared candle series, the next-bar fill model, the single step state machine,
//! and the batched runner. Pure Rust, no Python; it builds on probatum-core's
//! fidelity-neutral primitives.

pub mod batch;
pub mod candles;
pub mod clock;
pub mod engine;
pub mod execution;
pub mod inverse;
pub mod reporter;
pub mod schedule;
pub mod script;
pub mod stats;
pub mod strategy;
pub mod streams;
pub mod sweep;
pub mod window;

pub use batch::{BatchSnapshot, EnvBatch};
pub use candles::{Candles, Instrument};
pub use clock::Clock;
pub use engine::{Engine, EngineConfig, Intrabar, Liquidation, TargetError};
pub use execution::{FillCosts, FillModel, Impact, Spread};
pub use inverse::Units;
pub use reporter::{Reporter, RoundTrip};
pub use schedule::Schedule;
pub use script::{Action, OrderRef, Script};
pub use stats::Stats;
pub use strategy::{run, Strategy};
pub use sweep::{builtin_strategy, run_sweep, SmaCross, StrategyFactory};
pub use window::Trailing;
