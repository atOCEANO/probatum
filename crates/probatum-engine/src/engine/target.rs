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
    /// Where among the transfers this decision's target queued the shortfall it
    /// moves across, which a second target in the decision takes back.
    moved: Option<usize>,
}

impl Targets {
    pub(super) fn new(width: usize) -> Targets {
        Targets {
            asked: vec![f64::NAN; width],
            before: vec![f64::NAN; width],
            decided: None,
            legs: Vec::new(),
            moved: None,
        }
    }

    /// Forget every target, as a reset forgets the account.
    pub(super) fn reset(&mut self) {
        self.asked.fill(f64::NAN);
        self.decided = None;
        self.legs.clear();
        self.moved = None;
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
    /// whose notional is below `tolerance` of equity is skipped before it is split.
    /// With `transfer`, what the spot legs it queued cost at the closes, fees
    /// included, beyond what their sales raise and the spot wallet holds, moves
    /// across from the margin wallet at the start of the next bar. A second target
    /// in one decision replaces the first, its legs, its transfer and the weights it
    /// asked for, and a target in a later decision replaces the legs earlier targets
    /// left waiting on the instruments it names. Returns the legs queued, or refuses
    /// a negative weight on a spot instrument before placing anything.
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
        let tick = self.tick;
        if self.targets.decided == Some(tick) {
            self.cancel_legs(|decided, _| decided == tick);
            if let Some(k) = self.targets.moved.take() {
                self.transfers.remove(k);
            }
            self.targets.asked.clone_from(&self.targets.before);
        } else {
            self.targets.before.clone_from(&self.targets.asked);
            self.targets.moved = None;
        }
        self.targets.decided = Some(tick);
        let mut named = weights.to_vec();
        named.sort_by_key(|&(i, _)| i);
        // a leg an earlier decision left waiting for its instrument's next bar would
        // fill beside this one's, sized from the same position
        self.cancel_legs(|decided, at| decided != tick && named.iter().any(|&(i, _)| i == at));
        for &(i, weight) in &named {
            self.targets.asked[i] = weight;
        }
        let equity = self.equity();
        let (mut reducing, mut growing) = (Vec::new(), Vec::new());
        for &(i, weight) in &named {
            let close = self.marks[i];
            if close.is_nan() {
                // an instrument yet to print its first bar has no close to size a
                // leg on: a weight of zero there asks for nothing, and any other
                // places a leg refused as any order there is
                if weight != 0.0 {
                    growing.push((i, f64::NAN));
                }
                continue;
            }
            let q = self.account.positions[i].qty.get();
            let leg = weight * equity / close - q;
            if leg.abs() <= DUST || leg.abs() * close < tolerance * equity {
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
            if grow.abs() > DUST {
                growing.push((i, grow));
            }
        }
        let mut queued = Vec::with_capacity(reducing.len() + growing.len());
        let mut needed = -self.account.spot_cash;
        for (legs, reduce_only) in [(&reducing, true), (&growing, false)] {
            for &(i, leg) in legs {
                let side = if leg > 0.0 { Side::Buy } else { Side::Sell };
                let tif = TimeInForce::Ioc;
                let reason = EventReason::Rebalance;
                let Some(id) = self.queue(i, side, leg.abs(), reduce_only, tif, reason) else {
                    continue;
                };
                self.targets.legs.push((id, tick));
                queued.push(id);
                if self.account.markets[i] == Market::Spot {
                    let cost = leg * self.marks[i];
                    needed += cost + cost.abs() * self.cost.fee_taker;
                }
            }
        }
        if transfer && needed > 0.0 {
            self.targets.moved = Some(self.transfers.len());
            self.transfer(needed, Wallet::Spot);
        }
        Ok(queued)
    }

    /// Cancel the waiting legs `which` picks, by the bar each was decided at and the
    /// instrument it trades, each logged as replaced by a later target, and forget
    /// those no longer waiting.
    fn cancel_legs(&mut self, which: impl Fn(usize, usize) -> bool) {
        let legs = std::mem::take(&mut self.targets.legs);
        for (id, decided) in legs {
            let Some(at) = self.pending.iter().position(|order| order.id == id) else {
                continue;
            };
            if !which(decided, self.pending[at].instrument) {
                self.targets.legs.push((id, decided));
                continue;
            }
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
}

#[cfg(test)]
mod tests {
    use super::TargetError;
    use crate::candles::Candles;
    use crate::engine::fixtures::{cfg, ohlc, perp_cfg, universe, ABSENT};
    use crate::engine::{Engine, EngineConfig};
    use crate::reporter::{EventKind, EventReason, FillReason};
    use probatum_core::{Candle, Market, OrderType, Side, TimeInForce, Wallet};

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

    /// The Specification's spot account: 1,000 in the spot wallet, 9,000 in the
    /// margin wallet and a taker fee of 0.1%, on two rows of bitcoin at 60,000, with
    /// `max_open_orders` an instrument may queue.
    fn spot_and_margin(max_open_orders: usize) -> Engine {
        let flat = ohlc(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e6);
        let config = EngineConfig {
            quote: 1_000.0,
            idle_cash: 9_000.0,
            fee_taker: 0.001,
            max_open_orders,
            report: true,
            ..cfg()
        };
        let mut e = Engine::new(Candles::new(vec![flat, flat]), config);
        e.reset();
        e
    }

    #[test]
    fn a_second_target_in_a_decision_takes_back_the_first_ones_transfer() {
        // the same target twice moves its 4,005 once, and a second target that moves
        // nothing leaves the first one's in the margin wallet
        for (second, moved) in [(Some(true), 4_005.0), (Some(false), 0.0), (None, 4_005.0)] {
            let mut e = spot_and_margin(8);
            e.target(&[(0, 0.5)], 0.0, true).expect("a long on spot");
            if let Some(transfer) = second {
                let weight = if transfer { 0.5 } else { 0.0 };
                e.target(&[(0, weight)], 0.0, transfer)
                    .expect("a long on spot");
            }
            let s = e.step();
            assert!(
                (s.margin_cash - (9_000.0 - moved)).abs() < 1e-6,
                "{second:?} left {}",
                s.margin_cash
            );
        }
    }

    #[test]
    fn a_refused_leg_moves_no_shortfall_across() {
        // a buy already waits in the one place the queue has, so the target's leg is
        // refused and nothing it would have cost moves
        let mut e = spot_and_margin(1);
        e.market_buy(0.001);
        let legs = e.target(&[(0, 0.5)], 0.0, true).expect("a long on spot");
        assert!(legs.is_empty());
        assert_eq!(e.step().margin_cash, 9_000.0);
    }

    #[test]
    fn a_second_target_takes_back_only_the_first_ones_transfer() {
        // the account's own transfer of 100, made between the two targets, lands
        // beside the second target's 4,005, which saw the same 1,000 of spot cash
        let mut e = spot_and_margin(8);
        e.target(&[(0, 0.5)], 0.0, true).expect("a long on spot");
        e.transfer(100.0, Wallet::Spot);
        e.target(&[(0, 0.5)], 0.0, true).expect("a long on spot");
        let s = e.step();
        let kept = 9_000.0 - 100.0 - 4_005.0;
        assert!(
            (s.margin_cash - kept).abs() < 1e-6,
            "kept {}",
            s.margin_cash
        );
        assert!((s.spot_cash - 100.0).abs() < 1e-6, "spot {}", s.spot_cash);
    }

    #[test]
    fn a_flip_whose_whole_leg_clears_the_tolerance_queues_both_parts() {
        // a long of 4 at 100 on 10,000, 0.04 of equity, asked to -0.04 with a
        // tolerance of 0.05: the whole leg of 8 is 800 of notional, past the 500 the
        // tolerance skips, though each part of 4 is 400
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let config = EngineConfig {
            quote: 10_000.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(Candles::new(vec![flat; 3]), config);
        e.reset();
        e.market_buy(4.0);
        e.step();
        e.target(&[(0, -0.04)], 0.05, false).expect("a perp");
        let queued: Vec<(Side, f64, bool)> = e
            .pending
            .iter()
            .map(|order| (order.side, order.size.get(), order.reduce_only))
            .collect();
        assert_eq!(
            queued,
            vec![(Side::Sell, 4.0, true), (Side::Sell, 4.0, false)]
        );
    }

    #[test]
    fn a_weight_of_zero_on_an_instrument_yet_to_print_places_nothing() {
        // the second perp prints its first bar on the third row: a weight of zero on
        // it asks for nothing, and any other places a leg refused there
        let at = |price: f64| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![at(100.0), ABSENT],
            vec![at(100.0), ABSENT],
            vec![at(100.0), at(10.0)],
        ];
        let config = EngineConfig {
            quote: 10_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        e.reset();
        let refused = |e: &Engine| {
            let reporter = e.reporter().expect("reporting is on");
            reporter
                .events()
                .iter()
                .filter(|event| event.kind == EventKind::Refused)
                .count()
        };
        let legs = e.target(&[(0, 0.5), (1, 0.0)], 0.0, false).expect("perps");
        assert_eq!((legs.len(), refused(&e)), (1, 0));
        e.target(&[(1, 0.3)], 0.0, false).expect("perps");
        assert_eq!(refused(&e), 1);
    }

    /// Two perps, the second at 10 printing no bar on the third and fourth rows, on
    /// 10,000 of margin with no fees, reset.
    fn waiting() -> Engine {
        let at = |price: f64| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![at(100.0), at(10.0)],
            vec![at(100.0), at(10.0)],
            vec![at(100.0), ABSENT],
            vec![at(100.0), ABSENT],
            vec![at(100.0), at(10.0)],
        ];
        let config = EngineConfig {
            quote: 10_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        e.reset();
        e
    }

    #[test]
    fn a_later_target_leaves_a_waiting_leg_on_what_it_does_not_name() {
        // the second perp's buy of 500 waits, and a target of the first alone lets it
        // fill on the fifth row beside its own buy of 20 on the fourth
        let mut e = waiting();
        e.step();
        let first = e.target(&[(1, 0.5)], 0.0, false).expect("a perp");
        e.step();
        let second = e.target(&[(0, 0.2)], 0.0, false).expect("a perp");
        e.step();
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let fills: Vec<_> = reporter
            .fills()
            .iter()
            .map(|fill| (fill.order, fill.size))
            .collect();
        assert_eq!(
            fills,
            vec![(Some(second[0]), 20.0), (Some(first[0]), 500.0)]
        );
    }

    #[test]
    fn a_target_replaces_the_legs_an_earlier_one_left_waiting() {
        // the buy of 500 a target decides on the second row waits; the same target
        // on the third row replaces it with its own, and one buy of 500 fills on the
        // fifth row
        let mut e = waiting();
        e.step();
        let first = e.target(&[(1, 0.5)], 0.0, false).expect("a perp");
        e.step();
        let second = e.target(&[(1, 0.5)], 0.0, false).expect("a perp");
        e.step();
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let fills: Vec<_> = reporter
            .fills()
            .iter()
            .map(|fill| (fill.order, fill.size))
            .collect();
        assert_eq!(fills, vec![(Some(second[0]), 500.0)]);
        let cancelled = reporter
            .events()
            .iter()
            .find(|event| event.kind == EventKind::Cancelled)
            .expect("the first leg cancelled");
        assert_eq!(
            (cancelled.order, cancelled.reason),
            (Some(first[0]), EventReason::Retargeted)
        );
    }
}
