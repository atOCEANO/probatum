//! Brackets: two reduce-only exits placed as one OCO pair, neither with a size of
//! its own, each closing the position it finds when it fills. A leg takes its side
//! from the position it arms on and keeps it, the first fill of either cancels the
//! other, and a bracket goes when the position it armed on closes or flips, so its
//! legs never act on a position they were not placed for (ADR 0010).

use probatum_core::{Leg, Order, OrderId, OrderType, Price, Qty, Side, TimeInForce};

use super::Engine;
use crate::reporter::{EventKind, EventReason};

impl Engine {
    /// Place a bracket on `instrument`: a stop at `stop` and a take-profit at
    /// `take_profit`, each a reduce-only leg with no size of its own, the two one OCO
    /// pair. A leg arms at once on a position held, and otherwise once the entry
    /// fills. Returns each leg's id, `None` for one not asked for or refused.
    pub fn bracket(
        &mut self,
        instrument: usize,
        stop: Option<f64>,
        take_profit: Option<f64>,
    ) -> (Option<OrderId>, Option<OrderId>) {
        let leg = Some(Leg {
            pair: None,
            armed: false,
            sizeless: true,
        });
        // the side is a placeholder until the leg arms, and nothing reads it before
        let unbounded = Qty(f64::INFINITY);
        let stop = stop.and_then(|trigger| {
            let id = self.next_id();
            let order = Order::stop(id, Side::Sell, unbounded, Price(trigger), true);
            self.rest(
                Order {
                    leg,
                    ..order.on(instrument)
                },
                EventReason::Stop,
            )
        });
        let take_profit = take_profit.and_then(|price| {
            let id = self.next_id();
            let order = Order::limit(
                id,
                Side::Sell,
                unbounded,
                Price(price),
                TimeInForce::Gtc,
                false,
                true,
            );
            self.rest(
                Order {
                    leg,
                    ..order.on(instrument)
                },
                EventReason::TakeProfit,
            )
        });
        if let (Some(stop), Some(take_profit)) = (stop, take_profit) {
            for (id, pair) in [(stop, take_profit), (take_profit, stop)] {
                if let Some(leg) = self.book.get_mut(id).and_then(|order| order.leg.as_mut()) {
                    leg.pair = Some(pair);
                }
            }
        }
        self.arm();
        (stop, take_profit)
    }

    /// Arm each bracket leg resting on an instrument whose position is held, on the
    /// side that closes it.
    pub(super) fn arm(&mut self) {
        if self.book.is_empty() {
            return;
        }
        let mut armed = Vec::new();
        for order in self.book.iter_mut() {
            let held = self.account.positions[order.instrument].qty.get();
            if let Some(leg) = order.leg.as_mut().filter(|leg| !leg.armed && held != 0.0) {
                leg.armed = true;
                order.side = if held > 0.0 { Side::Sell } else { Side::Buy };
                armed.push((order.id, order.instrument, order.kind));
            }
        }
        for (id, instrument, kind) in armed {
            let reason = if kind == OrderType::Stop {
                EventReason::Stop
            } else {
                EventReason::TakeProfit
            };
            self.log_event(
                EventKind::Armed,
                Some(id),
                Some(instrument),
                None,
                None,
                reason,
            );
        }
    }

    /// Cancel the armed legs resting on instrument `i`, whose position a fill by
    /// `filling` closed or flipped, but that order and its partner, which the OCO rule
    /// settles.
    pub(super) fn disarm(&mut self, i: usize, filling: OrderId) {
        if self.book.is_empty() {
            return;
        }
        let pair = self
            .book
            .get(filling)
            .and_then(|order| order.leg)
            .and_then(|leg| leg.pair);
        let gone: Vec<OrderId> = self
            .book
            .iter()
            .filter(|order| order.instrument == i && order.leg.is_some_and(|leg| leg.armed))
            .map(|order| order.id)
            .filter(|&id| id != filling && Some(id) != pair)
            .collect();
        for id in gone {
            self.book.cancel(id);
            self.log_event(
                EventKind::Cancelled,
                Some(id),
                Some(i),
                None,
                None,
                EventReason::PositionClosed,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{ohlc, perp_cfg, report_cfg};
    use crate::engine::{Engine, EngineConfig, Intrabar};
    use crate::reporter::{EventKind, EventReason, FillReason, OrderEvent};
    use probatum_core::{Candle, OrderId, Side};

    const FLAT: Candle = Candle {
        open: 100.0,
        high: 100.0,
        low: 100.0,
        close: 100.0,
        volume: 1e6,
    };

    /// A spot engine over a flat bar and then `after`, with reporting on.
    fn spot(after: &[Candle], intrabar: Intrabar) -> Engine {
        let mut bars = vec![FLAT];
        bars.extend_from_slice(after);
        let config = EngineConfig {
            intrabar,
            ..report_cfg()
        };
        let mut e = Engine::new(Candles::new(bars), config);
        e.reset();
        e
    }

    fn events(e: &Engine) -> Vec<(u64, EventKind, EventReason)> {
        let log: &[OrderEvent] = e.reporter().expect("reporting is on").events();
        log.iter()
            .map(|event| (event.order.map_or(0, |id| id.0), event.kind, event.reason))
            .collect()
    }

    fn fills(e: &Engine) -> Vec<(u64, f64, f64, FillReason)> {
        let reporter = e.reporter().expect("reporting is on");
        reporter
            .fills()
            .iter()
            .map(|fill| {
                (
                    fill.order.map_or(0, |id| id.0),
                    fill.size,
                    fill.price,
                    fill.reason,
                )
            })
            .collect()
    }

    fn id(order: Option<OrderId>) -> u64 {
        order.expect("the leg rests").0
    }

    #[test]
    fn a_bar_reaching_both_legs_fills_the_one_the_setting_decides_and_cancels_the_other() {
        // the Specification's example: a long of 1 from 100 holds a bracket with its
        // stop at 95 and its take-profit at 110, and the next bar opens at 100 with a
        // low of 94 and a high of 111. The pessimistic setting fills the stop at 95
        // and cancels the take-profit; the optimistic one fills the take-profit at
        // 110 and cancels the stop
        let wide = ohlc(100.0, 111.0, 94.0, 100.0, 1e6);
        for (intrabar, filled, price) in [
            (Intrabar::Pessimistic, FillReason::Stop, 95.0),
            (Intrabar::Optimistic, FillReason::TakeProfit, 110.0),
        ] {
            let mut e = spot(&[FLAT, wide], intrabar);
            e.market_buy(1.0);
            e.step();
            let (stop, take_profit) = e.bracket(0, Some(95.0), Some(110.0));
            let s = e.step();
            let (won, lost) = if filled == FillReason::Stop {
                (id(stop), id(take_profit))
            } else {
                (id(take_profit), id(stop))
            };
            assert_eq!(fills(&e)[1], (won, 1.0, price, filled));
            assert_eq!(
                events(&e).last().copied(),
                Some((lost, EventKind::Cancelled, EventReason::PartnerFilled))
            );
            assert_eq!(s.position, 0.0);
            assert!(s.open_orders.is_empty());
        }
    }

    #[test]
    fn a_bracket_placed_with_its_entry_arms_when_the_entry_fills() {
        // placed while flat, the legs rest unarmed through the decision, arm on the
        // long the open fills, and the take-profit closes it on the same bar
        let mut e = spot(
            &[ohlc(100.0, 111.0, 99.0, 105.0, 1e6)],
            Intrabar::Pessimistic,
        );
        let entry = e.market_buy(1.0).expect("queued").0;
        let (stop, take_profit) = e.bracket(0, Some(95.0), Some(110.0));
        let (stop, take_profit) = (id(stop), id(take_profit));
        assert!(e
            .current_state()
            .open_orders
            .iter()
            .all(|order| order.leg.is_some_and(|leg| !leg.armed)));
        let s = e.step();
        assert_eq!(
            events(&e),
            vec![
                (entry, EventKind::Placed, EventReason::Market),
                (stop, EventKind::Placed, EventReason::Stop),
                (take_profit, EventKind::Placed, EventReason::TakeProfit),
                (entry, EventKind::Filled, EventReason::Market),
                (stop, EventKind::Armed, EventReason::Stop),
                (take_profit, EventKind::Armed, EventReason::TakeProfit),
                (take_profit, EventKind::Filled, EventReason::TakeProfit),
                (stop, EventKind::Cancelled, EventReason::PartnerFilled),
            ]
        );
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_leg_reached_while_flat_fills_nothing_and_waits() {
        let wide = ohlc(100.0, 111.0, 94.0, 100.0, 1e6);
        let mut e = spot(&[wide], Intrabar::Pessimistic);
        e.bracket(0, Some(95.0), Some(110.0));
        let s = e.step();
        assert!(fills(&e).is_empty());
        assert_eq!(s.open_orders.len(), 2);
    }

    #[test]
    fn a_stop_reached_on_a_bar_of_no_volume_stays_armed_for_the_next() {
        // the Specification's example of the book: the bracket's stop is reached on
        // a bar that printed no volume and fills on the next bar that reaches it
        let mut e = spot(
            &[
                FLAT,
                ohlc(100.0, 100.0, 94.0, 96.0, 0.0),
                ohlc(96.0, 97.0, 94.0, 95.0, 1e6),
            ],
            Intrabar::Pessimistic,
        );
        e.market_buy(1.0);
        e.step();
        let (stop, _) = e.bracket(0, Some(95.0), Some(110.0));
        e.step();
        assert_eq!(fills(&e).len(), 1);
        e.step();
        assert_eq!(fills(&e)[1], (id(stop), 1.0, 95.0, FillReason::Stop));
    }

    #[test]
    fn a_flip_cancels_the_bracket_its_position_armed() {
        // the Specification's example: a bracket armed on a long of 1, its stop at 95
        // and its take-profit at 110, is cancelled when a market sell of 2 flips the
        // account short, where a buy stop at 95 and a buy limit at 110 would have
        // closed the short at once
        let bars = Candles::new(vec![FLAT, FLAT, FLAT, ohlc(100.0, 111.0, 94.0, 100.0, 1e6)]);
        let config = EngineConfig {
            quote: 1_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.0);
        e.step();
        let (stop, take_profit) = e.bracket(0, Some(95.0), Some(110.0));
        let flip = e.market_sell(2.0).expect("queued").0;
        e.step();
        assert_eq!(
            events(&e)[events(&e).len() - 3..],
            [
                (flip, EventKind::Filled, EventReason::Market),
                (id(stop), EventKind::Cancelled, EventReason::PositionClosed),
                (
                    id(take_profit),
                    EventKind::Cancelled,
                    EventReason::PositionClosed
                ),
            ]
        );
        let s = e.step();
        assert_eq!(s.position, -1.0);
        assert_eq!(fills(&e).len(), 2);
    }

    #[test]
    fn a_leg_closes_the_position_it_finds_and_its_remainder_keeps_resting() {
        // armed on a long of 1 that grows to 3 before the stop is reached, the stop
        // closes all 3, a bar's volume capping it at 2 first: the take-profit goes at
        // that first fill, and the stop rests on to close the last 1
        let bars = vec![
            FLAT,
            FLAT,
            ohlc(100.0, 100.0, 94.0, 95.0, 2.0),
            ohlc(95.0, 96.0, 94.0, 95.0, 1e6),
        ];
        let mut e = spot(&bars, Intrabar::Pessimistic);
        e.market_buy(1.0);
        e.step();
        let (stop, take_profit) = e.bracket(0, Some(95.0), Some(110.0));
        e.market_buy(2.0);
        e.step();
        assert_eq!(e.current_state().position, 3.0);
        e.step();
        assert_eq!(fills(&e)[2], (id(stop), 2.0, 95.0, FillReason::Stop));
        assert_eq!(
            events(&e).last().copied(),
            Some((
                id(take_profit),
                EventKind::Cancelled,
                EventReason::PartnerFilled
            ))
        );
        let s = e.step();
        assert_eq!(fills(&e)[3], (id(stop), 1.0, 95.0, FillReason::Stop));
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn a_fill_closing_the_position_cancels_every_other_bracket_on_it() {
        // two brackets on one long and a bar reaching both stops: the worse, the
        // second's at 92, fills first and closes it, which cancels the first's legs
        // as its position closed and the second's take-profit as its partner, so the
        // first's stop, reached on the same bar, never fills
        let mut e = spot(
            &[FLAT, ohlc(100.0, 101.0, 90.0, 95.0, 1e6)],
            Intrabar::Pessimistic,
        );
        e.market_buy(1.0);
        e.step();
        let (first, first_profit) = e.bracket(0, Some(95.0), Some(110.0));
        let (second, second_profit) = e.bracket(0, Some(92.0), Some(120.0));
        e.step();
        assert_eq!(fills(&e).len(), 2);
        assert_eq!(fills(&e)[1].0, id(second));
        assert_eq!(
            events(&e)[events(&e).len() - 3..],
            [
                (id(first), EventKind::Cancelled, EventReason::PositionClosed),
                (
                    id(first_profit),
                    EventKind::Cancelled,
                    EventReason::PositionClosed
                ),
                (
                    id(second_profit),
                    EventKind::Cancelled,
                    EventReason::PartnerFilled
                ),
            ]
        );
    }

    #[test]
    fn a_replace_keeps_a_legs_partner_and_a_size_given_closes_only_that() {
        // the stop moved up to 97 and given a size of 0.5 is still the take-profit's
        // partner: it sells 0.5 at 97 and cancels the take-profit, and rests no more
        let mut e = spot(
            &[FLAT, ohlc(100.0, 101.0, 96.0, 98.0, 1e6)],
            Intrabar::Pessimistic,
        );
        e.market_buy(1.0);
        e.step();
        let (stop, take_profit) = e.bracket(0, Some(95.0), Some(110.0));
        let moved = stop.expect("the leg rests");
        assert_eq!(e.replace(moved, Some(0.5), None, Some(97.0)), stop);
        let s = e.step();
        assert_eq!(fills(&e)[1], (id(stop), 0.5, 97.0, FillReason::Stop));
        assert_eq!(
            events(&e).last().copied(),
            Some((
                id(take_profit),
                EventKind::Cancelled,
                EventReason::PartnerFilled
            ))
        );
        assert_eq!(s.position, 0.5);
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn a_short_arms_its_legs_as_buys() {
        let bars = Candles::new(vec![FLAT, FLAT, ohlc(100.0, 106.0, 99.0, 100.0, 1e6)]);
        let config = EngineConfig {
            quote: 1_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_sell(1.0);
        e.step();
        let (stop, _) = e.bracket(0, Some(105.0), Some(90.0));
        assert!(e
            .current_state()
            .open_orders
            .iter()
            .all(|order| order.side == Side::Buy));
        let s = e.step();
        assert_eq!(fills(&e)[1], (id(stop), 1.0, 105.0, FillReason::Stop));
        assert_eq!(s.position, 0.0);
    }
}
