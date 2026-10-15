//! Brackets: two reduce-only exits placed as one OCO pair, neither with a size of
//! its own, each closing the position it finds when it fills. A bracket arms at once
//! on a position held, and otherwise on the first fill that opens one on its
//! instrument; a leg takes its side from the position it arms on and keeps it, the
//! first fill of either cancels the other, and a bracket goes when the position it
//! armed on closes or flips, so its legs never act on a position they were not
//! placed for (ADR 0010). A take-profit is held to the self-trade rule when it takes
//! its side (ADR 0011).

use probatum_core::{Leg, Order, OrderId, OrderType, Price, Qty, Side, TimeInForce};

use super::Engine;
use crate::reporter::{EventKind, EventReason, Refusal};

/// The reason a leg of `kind` is placed, armed or filled for.
fn reason(kind: OrderType) -> EventReason {
    if kind == OrderType::Stop {
        EventReason::Stop
    } else {
        EventReason::TakeProfit
    }
}

impl Engine {
    /// Place a bracket on `instrument`: a stop at `stop` and a take-profit at
    /// `take_profit`, each a reduce-only leg with no size of its own, the two one OCO
    /// pair. A bracket placed beside a position held arms at once, its take-profit
    /// refused if it would trade against the account's own; one placed while flat
    /// arms on the first fill that opens a position there. Returns each leg's id,
    /// `None` for one not asked for or refused.
    pub fn bracket(
        &mut self,
        instrument: usize,
        stop: Option<f64>,
        take_profit: Option<f64>,
    ) -> (Option<OrderId>, Option<OrderId>) {
        let held = self.account.positions[instrument].qty.get();
        let leg = Some(Leg {
            pair: None,
            armed: held != 0.0,
            sizeless: true,
        });
        // a leg placed while flat has no side yet, and nothing reads this one before
        // it arms
        let side = if held < 0.0 { Side::Buy } else { Side::Sell };
        let unbounded = Qty(f64::INFINITY);
        let stop = stop.and_then(|trigger| {
            let id = self.next_id();
            let order = Order::stop(id, side, unbounded, Price(trigger), true);
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
                side,
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
        if held != 0.0 {
            for (id, kind) in [(stop, OrderType::Stop), (take_profit, OrderType::Limit)] {
                if let Some(id) = id {
                    self.log_event(
                        EventKind::Armed,
                        Some(id),
                        Some(instrument),
                        None,
                        None,
                        reason(kind),
                    );
                }
            }
        }
        (stop, take_profit)
    }

    /// Arm the bracket legs waiting on instrument `i` when the fill just booked there
    /// opened a position, `arming` saying whether a leg rests and the instrument was
    /// flat before it. It runs once the fill and its order are booked, so an entry the
    /// fill completed has left the book and is no resting order of the account's a
    /// take-profit could cross.
    #[inline]
    pub(super) fn arm_if_opened(&mut self, i: usize, arming: bool) {
        if arming && self.account.positions[i].qty.get() != 0.0 {
            self.arm(i);
        }
    }

    /// Arm the bracket legs waiting on instrument `i` on the position a fill has just
    /// opened there, on the side that closes it. A take-profit that would then trade
    /// against the account's own resting limit is cancelled (ADR 0011).
    pub(super) fn arm(&mut self, i: usize) {
        if self.book.legs() == 0 {
            return;
        }
        let held = self.account.positions[i].qty.get();
        let side = if held > 0.0 { Side::Sell } else { Side::Buy };
        let waiting: Vec<OrderId> = self
            .book
            .iter()
            .filter(|order| order.instrument == i && order.leg.is_some_and(|leg| !leg.armed))
            .map(|order| order.id)
            .collect();
        for id in waiting {
            let Some(order) = self.book.get_mut(id) else {
                continue;
            };
            // arming is the leg's placement as an order with a side, so a limit is a
            // taker on the bar it arms on where that bar, from its fill, is at or
            // through it (ADR 0011)
            order.side = side;
            order.on_book = false;
            if let Some(leg) = order.leg.as_mut() {
                leg.armed = true;
            }
            let order = *order;
            self.log_event(
                EventKind::Armed,
                Some(id),
                Some(i),
                None,
                None,
                reason(order.kind),
            );
            if let Some(own) = self.crossed(&order) {
                self.book.cancel(id);
                self.log_event(
                    EventKind::Cancelled,
                    Some(id),
                    Some(i),
                    None,
                    None,
                    EventReason::Refused(Refusal::SelfTrade(own)),
                );
            }
        }
    }

    /// Cancel the armed legs resting on instrument `i`, whose position a fill by
    /// `filling` closed or flipped, or the dust sweep zeroed, but that order, which
    /// goes as its fill leaves it, and its partner, which the OCO rule cancels.
    pub(super) fn disarm(&mut self, i: usize, filling: Option<OrderId>) {
        if self.book.legs() == 0 {
            return;
        }
        let pair = filling
            .and_then(|filling| self.book.get(filling))
            .and_then(|order| order.leg)
            .and_then(|leg| leg.pair);
        let gone: Vec<OrderId> = self
            .book
            .iter()
            .filter(|order| order.instrument == i && order.leg.is_some_and(|leg| leg.armed))
            .map(|order| order.id)
            .filter(|&id| Some(id) != filling && Some(id) != pair)
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
    use crate::reporter::{EventKind, EventReason, FillReason, OrderEvent, Refusal};
    use probatum_core::{Candle, OrderId, OrderType, Side, TimeInForce};

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
        // closes all 3, the bar's budget, half its trailing volume of 4, capping it at
        // 2 first: the take-profit goes at that first fill, and the stop rests on to
        // close the last 1
        let quiet = ohlc(100.0, 100.0, 100.0, 100.0, 4.0);
        let bars = vec![
            quiet,
            quiet,
            quiet,
            ohlc(100.0, 100.0, 94.0, 95.0, 4.0),
            ohlc(95.0, 96.0, 94.0, 95.0, 4.0),
        ];
        let config = EngineConfig {
            budget: 0.5,
            ..report_cfg()
        };
        let mut e = Engine::new(Candles::new(bars), config);
        e.reset();
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

    /// A perp engine over a flat bar and then `after`, with margin to spare.
    fn perp(after: &[Candle]) -> Engine {
        let mut bars = vec![FLAT];
        bars.extend_from_slice(after);
        let config = EngineConfig {
            quote: 1_000.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(Candles::new(bars), config);
        e.reset();
        e
    }

    #[test]
    fn a_bracket_placed_after_a_limit_entry_arms_when_it_fills_and_can_stop_it_there() {
        // the Specification's example of the book: a buy limit at 95 placed while
        // flat, with a bracket at 90 and 110 placed after it, meets a bar that opens
        // at 100 and trades down to 85: the entry fills at 95, the bracket arms on
        // that long, and its stop closes it at 90 on the same bar
        let mut e = spot(
            &[ohlc(100.0, 100.0, 85.0, 88.0, 1e6)],
            Intrabar::Pessimistic,
        );
        let entry = e.limit_buy(1.0, 95.0).expect("rests").0;
        let (stop, take_profit) = e.bracket(0, Some(90.0), Some(110.0));
        let s = e.step();
        assert_eq!(
            fills(&e),
            vec![
                (entry, 1.0, 95.0, FillReason::Limit),
                (id(stop), 1.0, 90.0, FillReason::Stop),
            ]
        );
        assert_eq!(
            events(&e)[3..],
            [
                (entry, EventKind::Filled, EventReason::Limit),
                (id(stop), EventKind::Armed, EventReason::Stop),
                (id(take_profit), EventKind::Armed, EventReason::TakeProfit),
                (id(stop), EventKind::Filled, EventReason::Stop),
                (
                    id(take_profit),
                    EventKind::Cancelled,
                    EventReason::PartnerFilled
                ),
            ]
        );
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_take_profit_the_bar_reached_only_before_its_limit_entry_fills_only_if_optimistic() {
        // the Specification's example of what a bar cannot show: the buy limit at 95
        // fills inside a bar that opens at 100, reaches 100.50 and 94, and closes at
        // 95.50; the high came before the fill under the pessimistic setting, so the
        // take-profit at 97 rests, and after it under the optimistic one, so it fills
        // there as a maker
        let wide = ohlc(100.0, 100.5, 94.0, 95.5, 1e6);
        for (intrabar, held) in [(Intrabar::Pessimistic, 1.0), (Intrabar::Optimistic, 0.0)] {
            let mut e = spot(&[wide], intrabar);
            let entry = e.limit_buy(1.0, 95.0).expect("rests").0;
            let (_, take_profit) = e.bracket(0, Some(90.0), Some(97.0));
            let s = e.step();
            let mut expected = vec![(entry, 1.0, 95.0, FillReason::Limit)];
            if held == 0.0 {
                expected.push((id(take_profit), 1.0, 97.0, FillReason::TakeProfit));
            }
            assert_eq!(fills(&e), expected);
            assert_eq!(s.position, held);
            let reporter = e.reporter().expect("reporting is on");
            assert!(reporter.fills().iter().all(|fill| !fill.taker));
        }
    }

    #[test]
    fn a_stop_leg_of_a_breakout_long_fills_where_the_price_crossed_it_after_the_entry() {
        // the buy stop at 105 fills inside a bar that opens at 100, reaches 108 and 98,
        // and closes at 103: the high came after the fill, which the price had not
        // reached, and so did the low, which works against the long, under either
        // setting, so the stop leg fills at 102, where the price crossed it, and not
        // at the open
        let wide = ohlc(100.0, 108.0, 98.0, 103.0, 1e6);
        for intrabar in [Intrabar::Pessimistic, Intrabar::Optimistic] {
            let mut e = spot(&[wide], intrabar);
            let entry = e.stop(Side::Buy, 1.0, 105.0, false).expect("rests").0;
            let (stop, _) = e.bracket(0, Some(102.0), Some(110.0));
            let s = e.step();
            assert_eq!(
                fills(&e),
                vec![
                    (entry, 1.0, 105.0, FillReason::Stop),
                    (id(stop), 1.0, 102.0, FillReason::Stop),
                ]
            );
            assert_eq!(s.position, 0.0);
        }
    }

    #[test]
    fn the_optimistic_setting_keeps_the_stop_the_check_runs_a_breakout_long_past() {
        // the same breakout on a margin wallet of 6 at 20x: the check runs the long
        // from its fill at 105 down to 98, so its stop at 102 is reached after the
        // fill under the optimistic setting as under the default, and the account
        // keeps 3 where, without it, the long would be liquidated
        for intrabar in [Intrabar::Pessimistic, Intrabar::Optimistic] {
            let bars = Candles::new(vec![FLAT, ohlc(100.0, 108.0, 98.0, 103.0, 1e6)]);
            let config = EngineConfig {
                quote: 6.0,
                max_leverage: 20.0,
                report: true,
                intrabar,
                ..perp_cfg()
            };
            let mut e = Engine::new(bars, config);
            e.reset();
            e.stop(Side::Buy, 1.0, 105.0, false).expect("rests");
            e.bracket(0, Some(102.0), Some(110.0));
            let s = e.step();
            assert_eq!((s.position, s.margin_cash), (0.0, 3.0), "{intrabar:?}");
        }
    }

    #[test]
    fn a_post_only_sale_a_part_of_its_first_bar_opens_through_is_cancelled_there() {
        // the post-only sell at 104 survives the open at 100, and the buy stop at 105
        // then buys the holding it could sell: the part of the bar after that fill
        // opens at 105, through the sale, which would take liquidity there, so it is
        // cancelled rather than filled as a taker
        let mut e = spot(
            &[ohlc(100.0, 108.0, 98.0, 103.0, 1e6)],
            Intrabar::Pessimistic,
        );
        e.stop(Side::Buy, 1.0, 105.0, false).expect("rests");
        let sale = e
            .order(
                Side::Sell,
                1.0,
                OrderType::Limit,
                Some(104.0),
                None,
                false,
                true,
                TimeInForce::Gtc,
            )
            .expect("rests")
            .0;
        let s = e.step();
        assert_eq!(fills(&e).len(), 1);
        assert_eq!(s.position, 1.0);
        assert_eq!(
            events(&e).last().copied(),
            Some((sale, EventKind::Cancelled, EventReason::OpenedThrough))
        );
    }

    #[test]
    fn a_spot_sale_meets_a_holding_a_limit_bought_inside_the_bar_from_that_fill() {
        // a sale on spot can only sell the holding, which the buy at 95 makes inside
        // the bar, so it meets the bar from there: a sell at 97 the bar reached only
        // on its way down from 100 rests, where the whole bar would have filled it
        let mut e = spot(
            &[ohlc(100.0, 100.5, 94.0, 95.5, 1e6)],
            Intrabar::Pessimistic,
        );
        e.limit_buy(1.0, 95.0).expect("rests");
        let sale = e.limit_sell(1.0, 97.0).expect("rests");
        let s = e.step();
        assert_eq!(fills(&e).len(), 1);
        assert_eq!(s.position, 1.0);
        assert!(s.open_orders.iter().any(|order| order.id == sale));
    }

    #[test]
    fn a_spot_sale_meets_a_holding_a_limit_grew_inside_the_bar_from_that_fill() {
        // the same with 1 already held: the buy of 9 at 95 grows the holding, and
        // what it adds traded only after it, so the sale of 10 at 97 rests where the
        // whole bar sold all 10, 9 of them at a price that may have come before
        let mut e = spot(
            &[FLAT, ohlc(100.0, 100.5, 94.0, 95.5, 1e6)],
            Intrabar::Pessimistic,
        );
        e.market_buy(1.0);
        e.step();
        e.limit_buy(9.0, 95.0).expect("rests");
        let sale = e.limit_sell(10.0, 97.0).expect("rests");
        let s = e.step();
        assert_eq!(fills(&e).len(), 2);
        assert_eq!(s.position, 10.0);
        assert!(s.open_orders.iter().any(|order| order.id == sale));
    }

    #[test]
    fn a_leg_arming_is_its_placement_for_maker_and_taker() {
        // the take-profit at 99 rests through a bar while flat, then arms at the open
        // of 100 on the long a market buy opens there: it is through the price it arms
        // at, so it fills at once as a taker, though it had rested through a bar, and
        // at 100, the lowest trade since the last close, none of which came down to it
        let mut e = spot(&[FLAT, FLAT], Intrabar::Pessimistic);
        let (_, take_profit) = e.bracket(0, None, Some(99.0));
        e.step();
        e.market_buy(1.0);
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let fill = reporter.fills().last().expect("the take-profit filled");
        assert_eq!(
            (fill.order, fill.price, fill.taker),
            (take_profit, 100.0, true)
        );
    }

    #[test]
    fn a_leg_in_an_earlier_slot_than_the_fill_that_arms_it_waits_for_the_next_bar() {
        // placed before its entry, the stop takes its turn on the entry's bar before
        // the entry fills, while it has no side, and stops the long on the bar after
        let mut e = spot(
            &[
                ohlc(100.0, 100.0, 85.0, 88.0, 1e6),
                ohlc(88.0, 89.0, 86.0, 87.0, 1e6),
            ],
            Intrabar::Pessimistic,
        );
        let (stop, _) = e.bracket(0, Some(90.0), Some(110.0));
        e.limit_buy(1.0, 95.0).expect("rests");
        assert_eq!(e.step().position, 1.0);
        let s = e.step();
        assert_eq!(fills(&e)[1], (id(stop), 1.0, 88.0, FillReason::Stop));
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_flip_at_the_open_after_a_limit_entry_cancels_the_bracket_it_armed() {
        // the limit entry fills at 95 on a bar that reaches neither leg, so the
        // bracket rests armed on the long, and a market sell of 2 flipping it short
        // at the next open cancels both legs before a buy stop at 90 and a buy limit
        // at 110 could close the short
        let mut e = perp(&[
            ohlc(100.0, 101.0, 94.0, 100.0, 1e6),
            ohlc(100.0, 111.0, 89.0, 100.0, 1e6),
        ]);
        let entry = e.limit_buy(1.0, 95.0).expect("rests").0;
        let (stop, take_profit) = e.bracket(0, Some(90.0), Some(110.0));
        e.step();
        assert_eq!(fills(&e), vec![(entry, 1.0, 95.0, FillReason::Limit)]);
        let flip = e.market_sell(2.0).expect("queued").0;
        let s = e.step();
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
        assert_eq!(s.position, -1.0);
    }

    #[test]
    fn a_leg_given_a_size_goes_with_the_position_its_fill_closes() {
        // the stop given a size of 2 behind a long of 1 sells 1 at 95, all the long
        // there is, and goes with it rather than resting on to sell a later long
        let mut e = spot(
            &[
                FLAT,
                ohlc(100.0, 101.0, 94.0, 96.0, 1e6),
                FLAT,
                ohlc(100.0, 101.0, 94.0, 96.0, 1e6),
            ],
            Intrabar::Pessimistic,
        );
        e.market_buy(1.0);
        e.step();
        let (stop, _) = e.bracket(0, Some(95.0), Some(110.0));
        e.replace(stop.expect("the leg rests"), Some(2.0), None, None);
        let s = e.step();
        assert_eq!(fills(&e)[1], (id(stop), 1.0, 95.0, FillReason::Stop));
        assert!(s.open_orders.is_empty());
        // the 1 it had left to sell is cancelled with the position, so the order log
        // ends its life as it ends every other order's
        let last = events(&e)
            .into_iter()
            .rev()
            .find(|&(order, ..)| order == id(stop))
            .map(|(_, kind, reason)| (kind, reason));
        assert_eq!(
            last,
            Some((EventKind::Cancelled, EventReason::PositionClosed))
        );
        e.market_buy(3.0);
        e.step();
        assert_eq!(e.step().position, 3.0);
    }

    #[test]
    fn a_take_profit_is_not_held_against_the_entry_whose_whole_fill_arms_it() {
        // the bid at 99 fills whole and leaves the book before its bracket arms, so
        // the take-profit, a sell at that same 99, is no self-trade and closes the long
        let mut e = spot(
            &[ohlc(100.0, 100.5, 98.5, 99.5, 1e6)],
            Intrabar::Pessimistic,
        );
        let entry = e.limit_buy(1.0, 99.0).expect("rests").0;
        let (_, take_profit) = e.bracket(0, Some(95.0), Some(99.0));
        let s = e.step();
        assert_eq!(
            fills(&e),
            vec![
                (entry, 1.0, 99.0, FillReason::Limit),
                (id(take_profit), 1.0, 99.0, FillReason::TakeProfit),
            ]
        );
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_take_profit_against_the_accounts_own_bid_is_refused_or_cancelled_as_it_arms() {
        // beside a long, a take-profit at 98 is a sell at or below the account's bid
        // at 99 and is refused when placed; placed while flat it has no side, and it
        // is cancelled when the long a market buy opens arms it
        let quiet = ohlc(100.0, 100.5, 99.5, 100.0, 1e6);
        let mut e = spot(&[quiet, quiet], Intrabar::Pessimistic);
        let bid = e.limit_buy(1.0, 99.0).expect("rests");
        let (_, flat) = e.bracket(0, None, Some(98.5));
        let entry = e.market_buy(1.0).expect("queued").0;
        e.step();
        let crossed = EventReason::Refused(Refusal::SelfTrade(bid));
        assert_eq!(
            events(&e)[events(&e).len() - 3..],
            [
                (entry, EventKind::Filled, EventReason::Market),
                (id(flat), EventKind::Armed, EventReason::TakeProfit),
                (id(flat), EventKind::Cancelled, crossed),
            ]
        );
        assert_eq!(e.bracket(0, None, Some(98.0)), (None, None));
        assert_eq!(
            events(&e).last().map(|event| (event.1, event.2)),
            Some((EventKind::Refused, crossed))
        );
        assert_eq!(e.step().open_orders.len(), 1);
    }
}
