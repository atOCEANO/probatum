//! Optional reporting buffers: an equity curve sampled each step, a log of closed
//! trades and a log of fills. Off by default (the engine holds it behind an
//! `Option`) so the RL hot path allocates nothing; the backtester turns it on.
//! Stats are derived from these later.

use probatum_core::{OrderId, Side};

use crate::execution::FillCosts;
use crate::stats::Stats;

/// What made a fill, as the fill log names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillReason {
    Market,
    Limit,
    Stop,
    Liquidation,
}

impl FillReason {
    /// The name the fill log gives it.
    pub const fn name(self) -> &'static str {
        match self {
            FillReason::Market => "market",
            FillReason::Limit => "limit",
            FillReason::Stop => "stop",
            FillReason::Liquidation => "liquidation",
        }
    }
}

/// One fill, with the reference it is measured from and what it paid beyond it
/// (ADR 0022).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillRecord {
    /// The bar the fill landed on.
    pub tick: usize,
    /// The order that filled, or `None` for a forced close.
    pub order: Option<OrderId>,
    pub side: Side,
    pub size: f64,
    pub price: f64,
    /// The reference and the spread, slippage, impact and adjustment, per unit.
    pub costs: FillCosts,
    /// The fee, in quote.
    pub fee: f64,
    /// True when the fill took liquidity. A forced close is the venue's and takes
    /// none of the account's.
    pub taker: bool,
    pub reason: FillReason,
}

/// One closed trade, recorded when a position is reduced or closed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trade {
    pub entry_tick: usize,
    pub exit_tick: usize,
    /// The side of the position that was closed: buy for a long, sell for a short.
    pub side: Side,
    /// Base size closed.
    pub size: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    /// The whole round-trip fee on the closed size, in quote: the share of the
    /// position's entry fee belonging to this size, plus the closing fill's fee on
    /// it. Charging only the exit side understated the cost by about half and let a
    /// strategy whose edge is smaller than its costs read as profitable per trade
    /// (ADR 0023).
    pub fees: f64,
    /// Gross realized price PnL booked on the close, in quote, before fees.
    pub pnl: f64,
    /// `pnl - fees`: what this trade actually added to the account. The trade
    /// statistics are built from this, so they can be reproduced from the log.
    pub net_pnl: f64,
    pub bars_held: usize,
    /// True when the position was force-closed by a liquidation rather than by an
    /// order. A forced close used to be booked nowhere at all, so a death and an
    /// ordinary exit were not merely alike in the log, one of them was absent
    /// (ADR 0017).
    pub liquidated: bool,
}

/// Side buffers for the equity curve and the trade log. Never part of the
/// hot-path state or the RL observation.
#[derive(Clone, Debug, Default)]
pub struct Reporter {
    equity_curve: Vec<f64>,
    trades: Vec<Trade>,
    fills: Vec<FillRecord>,
    /// Steps recorded while holding a nonzero position, for the exposure stat.
    in_position_steps: usize,
}

impl Reporter {
    /// An empty reporter.
    pub fn new() -> Reporter {
        Reporter::default()
    }

    /// Append the equity value for the current step, counting the step as exposure
    /// when a position is held.
    pub fn record_equity(&mut self, equity: f64, in_position: bool) {
        self.equity_curve.push(equity);
        if in_position {
            self.in_position_steps += 1;
        }
    }

    /// Append a closed trade.
    pub fn record_trade(&mut self, trade: Trade) {
        self.trades.push(trade);
    }

    /// Append a fill to the fill log.
    pub fn record_fill(&mut self, fill: FillRecord) {
        self.fills.push(fill);
    }

    /// The equity sampled at each recorded step: one point per bar the engine
    /// actually advanced, so a driver that keeps calling `step` past the last bar
    /// adds none.
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// The closed trades, in the order they closed. A forced close is logged here
    /// like any other, carrying `liquidated` (ADR 0017).
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Every fill in the order it landed, forced closes included (ADR 0022).
    pub fn fills(&self) -> &[FillRecord] {
        &self.fills
    }

    /// Number of recorded equity points, and nothing else. It does not pair with
    /// `is_empty`, which wants the trade log empty too, so a reporter holding a
    /// trade and no equity point answers zero here and `false` there.
    pub fn len(&self) -> usize {
        self.equity_curve.len()
    }

    /// True when no buffer holds anything: no equity point, no trade and no fill.
    pub fn is_empty(&self) -> bool {
        self.equity_curve.is_empty() && self.trades.is_empty() && self.fills.is_empty()
    }

    /// Performance statistics over the recorded curve and trades. `initial` is the
    /// starting equity and `num_fills` the engine's fill counter; see ADR 0030 for
    /// the conventions.
    pub fn stats(
        &self,
        initial: f64,
        periods_per_year: f64,
        risk_free: f64,
        num_fills: usize,
        funding_paid: f64,
    ) -> Stats {
        Stats::compute(
            initial,
            &self.equity_curve,
            &self.trades,
            periods_per_year,
            risk_free,
            self.in_position_steps,
            num_fills,
            funding_paid,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Reporter, Trade};
    use probatum_core::Side;

    fn a_trade() -> Trade {
        Trade {
            entry_tick: 1,
            exit_tick: 5,
            side: Side::Buy,
            size: 2.0,
            entry_price: 100.0,
            exit_price: 110.0,
            fees: 0.1,
            pnl: 20.0,
            net_pnl: 19.9,
            bars_held: 4,
            liquidated: false,
        }
    }

    #[test]
    fn records_the_equity_curve() {
        let mut r = Reporter::new();
        r.record_equity(1000.0, false);
        r.record_equity(1010.0, true);
        assert_eq!(r.equity_curve(), &[1000.0, 1010.0]);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn records_trades() {
        let mut r = Reporter::new();
        r.record_trade(a_trade());
        assert_eq!(r.trades().len(), 1);
        assert_eq!(r.trades()[0].pnl, 20.0);
    }

    #[test]
    fn default_is_empty() {
        assert!(Reporter::new().is_empty());
    }
}
