//! `target`: the legs that bring each instrument it names to a weight of account
//! equity, sized at the last closes and queued as market orders for the next opens,
//! every reducing leg before every growing one so a rebalance is funded by its own
//! sales. The weights asked are kept, and each row records how far the weight it
//! holds has drifted from them (Specification, Target; ADR 0009).

use probatum_core::{Market, Order, OrderId, Side, TimeInForce, Wallet};

use super::path::DUST;
use super::Engine;
use crate::reporter::{EventKind, EventReason};

/// Why a target was refused before it placed anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetError {
    /// A negative weight on a spot instrument, which cannot be short.
    Short { instrument: usize, weight: f64 },
}

/// What `target` keeps between decisions.
#[derive(Clone, Debug)]
pub(super) struct Targets {
    /// The weight the latest target asked of each instrument, NaN where none has.
    pub(super) asked: Vec<f64>,
    /// `asked` as it stood before this decision's first target, which a second in
    /// the same decision restores before it asks again.
    before: Vec<f64>,
    /// The bar whose close the latest target was decided at.
    decided: Option<usize>,
    /// Each leg a target queued that may still be waiting, with the bar it was
    /// decided at.
    legs: Vec<(OrderId, usize)>,
}

impl Targets {
    pub(super) fn new(width: usize) -> Targets {
        Targets {
            asked: vec![f64::NAN; width],
            before: vec![f64::NAN; width],
            decided: None,
            legs: Vec::new(),
        }
    }

    /// Forget every target, as a reset forgets the account.
    pub(super) fn reset(&mut self) {
        self.asked.fill(f64::NAN);
        self.decided = None;
        self.legs.clear();
    }

    /// Whether `id` is a leg a target queued, whose fills the logs name rebalances.
    pub(super) fn placed(&self, id: OrderId) -> bool {
        self.legs.iter().any(|&(leg, _)| leg == id)
    }

    /// Keep only the legs still waiting in `pending`.
    pub(super) fn keep_waiting(&mut self, pending: &[Order]) {
        if !self.legs.is_empty() {
            self.legs
                .retain(|&(id, _)| pending.iter().any(|order| order.id == id));
        }
    }
}

impl Engine {
    /// Bring each instrument `weights` names, each at most once, to its weight of
    /// account equity at the last closes: the leg that takes its position to the
    /// one whose notional there is the weight's share of equity, queued as a market
    /// order for the next open, every reducing leg before every growing one. A leg
    /// whose notional is below `tolerance` of equity is skipped. With `transfer`,
    /// what the spot legs cost at the closes, fees included, beyond what their sales
    /// raise and the spot wallet holds, moves across from the margin wallet at the
    /// start of the next bar. A second target in one decision replaces the first,
    /// its legs and the weights it asked for. Returns the legs queued, or refuses a
    /// negative weight on a spot instrument before placing anything.
    pub fn target(
        &mut self,
        weights: &[(usize, f64)],
        tolerance: f64,
        transfer: bool,
    ) -> Result<Vec<OrderId>, TargetError> {
        for &(i, weight) in weights {
            if self.account.markets[i] == Market::Spot && weight < 0.0 {
                return Err(TargetError::Short {
                    instrument: i,
                    weight,
                });
            }
        }
        if self.targets.decided == Some(self.tick) {
            self.retarget();
        } else {
            self.targets.before.clone_from(&self.targets.asked);
        }
        self.targets.decided = Some(self.tick);
        let mut named = weights.to_vec();
        named.sort_by_key(|&(i, _)| i);
        for &(i, weight) in &named {
            self.targets.asked[i] = weight;
        }
        let equity = self.equity();
        let (mut reducing, mut growing) = (Vec::new(), Vec::new());
        for &(i, weight) in &named {
            let close = self.marks[i];
            if close.is_nan() {
                // an instrument yet to print its first bar has no close to size a
                // leg on, and its leg is refused as any order there is
                growing.push((i, f64::NAN));
                continue;
            }
            let q = self.account.positions[i].qty.get();
            let leg = weight * equity / close - q;
            if leg.abs() * close < tolerance * equity || leg.abs() <= DUST {
                continue;
            }
            let grow = if q != 0.0 && (q + leg) * q < 0.0 {
                reducing.push((i, -q));
                q + leg
            } else if q != 0.0 && (q + leg).abs() < q.abs() {
                reducing.push((i, leg));
                0.0
            } else {
                leg
            };
            if grow != 0.0 {
                growing.push((i, grow));
            }
        }
        let mut queued = Vec::with_capacity(reducing.len() + growing.len());
        for (legs, reduce_only) in [(&reducing, true), (&growing, false)] {
            for &(i, leg) in legs {
                let side = if leg > 0.0 { Side::Buy } else { Side::Sell };
                let tif = TimeInForce::Ioc;
                let reason = EventReason::Rebalance;
                if let Some(id) = self.queue(i, side, leg.abs(), reduce_only, tif, reason) {
                    self.targets.legs.push((id, self.tick));
                    queued.push(id);
                }
            }
        }
        if transfer {
            let mut needed = -self.account.spot_cash;
            for &(i, leg) in reducing.iter().chain(&growing) {
                if self.account.markets[i] == Market::Spot && !leg.is_nan() {
                    let cost = leg * self.marks[i];
                    needed += cost + cost.abs() * self.cost.fee_taker;
                }
            }
            self.transfer(needed, Wallet::Spot);
        }
        Ok(queued)
    }

    /// Cancel the legs this decision's earlier target queued that still wait, and
    /// restore the weights asked before it.
    fn retarget(&mut self) {
        let tick = self.tick;
        let legs = std::mem::take(&mut self.targets.legs);
        for &(id, decided) in &legs {
            if decided != tick {
                self.targets.legs.push((id, decided));
                continue;
            }
            if let Some(at) = self.pending.iter().position(|order| order.id == id) {
                let order = self.pending.remove(at);
                self.log_event(
                    EventKind::Cancelled,
                    Some(id),
                    Some(order.instrument),
                    None,
                    None,
                    EventReason::Retargeted,
                );
            }
        }
        self.targets.asked.clone_from(&self.targets.before);
    }
}

#[cfg(test)]
mod tests {
    use super::TargetError;
    use crate::candles::Candles;
    use crate::engine::fixtures::{cfg, ohlc, perp_cfg, universe};
    use crate::engine::{Engine, EngineConfig};
    use crate::reporter::{EventKind, EventReason, FillReason};
    use probatum_core::{Candle, Market, OrderType, Side, TimeInForce};

    /// A flat perp account of 10,000 with no fees, on a bar at 60,000 and then `next`.
    fn bitcoin(next: Candle) -> Engine {
        let candles = Candles::new(vec![
            ohlc(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e6),
            next,
        ]);
        let config = EngineConfig {
            quote: 10_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(candles, config);
        e.reset();
        e
    }

    #[test]
    fn a_target_of_half_held_through_a_close_of_66_000_drifts_by_minus_0_0238() {
        // the Specification's example: 0.5 of 10,000 at 60,000 buys 0.083333, and a
        // close of 66,000 leaves 10,500 holding 5,500, a weight of 0.5238
        let mut e = bitcoin(ohlc(60_000.0, 66_000.0, 60_000.0, 66_000.0, 1e6));
        let legs = e
            .target(&[(0, 0.5)], 0.0, false)
            .expect("a perp may be targeted");
        assert_eq!(legs.len(), 1);
        let s = e.step();
        assert!((s.position - 0.5 * 10_000.0 / 60_000.0).abs() < 1e-15);
        let reporter = e.reporter().expect("reporting is on");
        assert!(reporter.legs()[0].drift.is_nan());
        let held = reporter.legs()[1];
        assert!((held.weight - 5_500.0 / 10_500.0).abs() < 1e-12);
        assert!((held.drift - (0.5 - 5_500.0 / 10_500.0)).abs() < 1e-12);
        assert_eq!(reporter.fills()[0].reason, FillReason::Rebalance);
    }

    #[test]
    fn every_reducing_leg_is_queued_before_every_growing_one() {
        // a long of 50 on the first perp and a short of 50 on the second: the target
        // grows the first, flips the second, and trims nothing, so the second's
        // close of its short comes first, then the first's buy and the second's buy
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let candles = universe(&[Market::Perp, Market::Perp], vec![vec![flat, flat]; 3]);
        let config = EngineConfig {
            quote: 10_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(candles, config);
        e.reset();
        for (i, side) in [(0, Side::Buy), (1, Side::Sell)] {
            let (market, ioc) = (OrderType::Market, TimeInForce::Ioc);
            e.order_on(i, side, 50.0, market, None, None, false, false, ioc);
        }
        e.step();
        e.target(&[(1, 0.3), (0, 0.8)], 0.0, false)
            .expect("perps may be targeted");
        let queued: Vec<(usize, Side, f64, bool)> = e
            .pending
            .iter()
            .map(|order| {
                (
                    order.instrument,
                    order.side,
                    order.size.get(),
                    order.reduce_only,
                )
            })
            .collect();
        assert_eq!(
            queued,
            vec![
                (1, Side::Buy, 50.0, true),
                (0, Side::Buy, 30.0, false),
                (1, Side::Buy, 30.0, false),
            ]
        );
    }

    #[test]
    fn a_leg_below_the_tolerance_is_skipped() {
        // 0.5 of 10,000 at 60,000 is 5,000 of notional, below a tolerance of 0.6
        // of equity and above one of 0.4
        for (tolerance, legs) in [(0.6, 0), (0.4, 1)] {
            let mut e = bitcoin(ohlc(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e6));
            let queued = e.target(&[(0, 0.5)], tolerance, false).expect("a perp");
            assert_eq!(queued.len(), legs, "tolerance {tolerance}");
        }
    }

    #[test]
    fn a_second_target_in_one_decision_replaces_the_first_and_its_weights() {
        // the first target's leg is cancelled for the later one, and the weight it
        // asked of the instrument the second does not name is forgotten
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let candles = universe(&[Market::Perp, Market::Perp], vec![vec![flat, flat]; 2]);
        let config = EngineConfig {
            quote: 10_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(candles, config);
        e.reset();
        let first = e.target(&[(0, 0.5), (1, 0.2)], 0.0, false).expect("perps");
        let second = e.target(&[(0, -0.3)], 0.0, false).expect("perps");
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let cancelled: Vec<_> = reporter
            .events()
            .iter()
            .filter(|event| event.kind == EventKind::Cancelled)
            .map(|event| (event.order, event.reason))
            .collect();
        assert_eq!(
            cancelled,
            first
                .iter()
                .map(|&id| (Some(id), EventReason::Retargeted))
                .collect::<Vec<_>>()
        );
        assert_eq!(reporter.fills().len(), 1);
        assert_eq!(reporter.fills()[0].order, Some(second[0]));
        // the second row's legs, two to a row
        let legs = &reporter.legs()[2..4];
        assert!((legs[0].drift - (-0.3 - legs[0].weight)).abs() < 1e-12);
        assert!(legs[1].drift.is_nan());
    }

    #[test]
    fn a_spot_target_with_transfer_moves_its_shortfall_of_4_005_across() {
        // the Specification's example: 1,000 in the spot wallet and 9,000 in the
        // margin wallet, a taker fee of 0.1%, and 0.5 of bitcoin on spot at 60,000
        // buys 0.083333 for 5,000.00 and a fee of 5.00, so 4,005.00 moves across
        let flat = ohlc(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e6);
        let config = EngineConfig {
            quote: 1_000.0,
            idle_cash: 9_000.0,
            fee_taker: 0.001,
            report: true,
            ..cfg()
        };
        let mut e = Engine::new(Candles::new(vec![flat, flat]), config);
        e.reset();
        e.target(&[(0, 0.5)], 0.0, true).expect("a long on spot");
        let s = e.step();
        assert!((s.position - 0.5 * 10_000.0 / 60_000.0).abs() < 1e-12);
        assert!((s.margin_cash - (9_000.0 - 4_005.0)).abs() < 1e-6);
        assert!(s.spot_cash.abs() < 1e-6);
    }

    #[test]
    fn a_negative_weight_on_spot_is_refused_before_anything_is_placed() {
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let mut e = Engine::new(Candles::new(vec![flat, flat]), cfg());
        e.reset();
        assert_eq!(
            e.target(&[(0, -0.5)], 0.0, false),
            Err(TargetError::Short {
                instrument: 0,
                weight: -0.5
            })
        );
        assert!(e.pending.is_empty());
    }
}
