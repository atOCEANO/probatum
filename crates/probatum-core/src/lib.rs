//! probatum-core: fidelity-neutral primitives for the market simulation engine.
//!
//! Pure Rust: no candle series, no Python, no parallelism. These primitives are
//! the reuse seam shared across fidelity tiers (the bar engine here, a future
//! tick and L2 engine later). The crate also holds a table of named columns and its
//! export through the Arrow C interfaces, kept here so that Miri, which runs no
//! Python, checks their release.

pub mod account;
pub mod arrow;
pub mod book;
pub mod candle;
pub mod cost;
pub mod enums;
pub mod fill;
pub mod order;
pub mod position;
pub mod state;
pub mod tiers;
pub mod units;

pub use account::Account;
pub use book::RestingOrderBook;
pub use candle::Candle;
pub use cost::{CostModel, FlatCostModel};
pub use enums::{Market, OrderType, Side, TimeInForce, Wallet};
pub use fill::Fill;
pub use order::{Leg, Order, OrderId};
pub use position::Position;
pub use state::State;
pub use tiers::{Tier, TierBasis, TierError, Tiers};
pub use units::{Bps, Notional, Price, Qty};
