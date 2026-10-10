//! Resolving orders against a bar: the queued market orders at their instruments'
//! opens, then the resting book in slot order, with the exits a wide bar reaches
//! ordered worst first among one instrument's, and its take-profits held back where
//! the bar liquidates the account without them (ADR 0012). An all-or-nothing order
//! is judged against every clamp, and IOC and FOK limits expire after their one
//! bar (ADR 0010). An order on an instrument that printed no bar waits for its
//! next.

use probatum_core::{Fill, Market, OrderId, OrderType, Qty, Side, TimeInForce};

use super::{Engine, CLOSE_EPS};
use crate::candles::printed;
use crate::execution::FillCosts;
use crate::reporter::{EventKind, EventReason, FillReason};

/// How badly a fill on `side` at `price` ends for the position currently held: the
/// PnL per unit it would realize, at its fill price with the slippage it pays, as a
/// position realizes it, so the most adverse sorts first. A fill that does not
/// reduce the position is not part of the question a wide bar asks, and sorts last.
fn adversity(position: f64, entry: f64, side: Side, price: f64) -> f64 {
    let reduces = (position > 0.0 && side == Side::Sell) || (position < 0.0 && side == Side::Buy);
    if !reduces {
        return f64::INFINITY;
    }
    (price - entry) * position.signum()
}

/// A resting order a bar reaches: its id and instrument, the fill the model offers
/// and its costs, its type, `reduce_only` and time in force, and what it still asks.
type Offer = (
    OrderId,
    usize,
    Fill,
    OrderType,
    bool,
    TimeInForce,
    f64,
    FillCosts,
);

impl Engine {
    /// Fill the market orders queued since the last step at their instruments'
    /// opens, each on its bar as the path clipped it, in the order they were
    /// placed. One that cannot fill on this bar is dropped rather than carried to
    /// the next, and one whose instrument printed no bar waits for its next.
    pub(super) fn resolve_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let spare = std::mem::take(&mut self.spare);
        let mut pending = std::mem::replace(&mut self.pending, spare);
        for order in pending.drain(..) {
            let i = order.instrument;
            let bar = self.bar_for(i);
            if !printed(&bar) {
                self.pending.push(order);
                continue;
            }
            let requested = order.remaining().get();
            let Some((fill, costs)) = self.fill_model.fill_market(&order, &bar) else {
                self.log_event(
                    EventKind::NotFilled,
                    Some(order.id),
                    Some(i),
                    None,
                    None,
                    EventReason::NothingFilled,
                );
                continue;
            };
            // FOK is all or nothing against everything that can shrink the fill,
            // not just the bar's liquidity: the margin cap and the spot clamps
            // are applied to a copy first, so an order that could only fill in
            // part books nothing at all (ADR 0010).
            if order.tif == TimeInForce::Fok
                && self.clamp_fill(i, order.reduce_only, &fill).size.get() + CLOSE_EPS < requested
            {
                self.log_event(
                    EventKind::NotFilled,
                    Some(order.id),
                    Some(i),
                    None,
                    None,
                    EventReason::Fok,
                );
                continue;
            }
            let reason = if self.targets.placed(order.id) {
                FillReason::Rebalance
            } else {
                FillReason::Market
            };
            let applied = self.apply_fill_clamped(
                i,
                order.id,
                reason,
                order.reduce_only,
                requested,
                fill,
                costs,
                &bar,
            );
            // a market order lives one bar, and what it did not fill is gone
            if applied <= 0.0 {
                self.log_event(
                    EventKind::NotFilled,
                    Some(order.id),
                    Some(i),
                    None,
                    None,
                    EventReason::NothingFilled,
                );
            } else if applied + CLOSE_EPS < requested {
                self.log_event(
                    EventKind::Expired,
                    Some(order.id),
                    Some(i),
                    None,
                    None,
                    EventReason::MarketBar,
                );
            }
        }
        self.spare = pending;
        self.targets.keep_waiting(&self.pending);
    }

    /// Resolve the resting book against each order's instrument's bar, as the path
    /// clipped it, in slot order.
    ///
    /// A stop is consumed only when it produces a fill. On a bar that trades no
    /// volume the trigger may be crossed and nothing can fill, and treating that as
    /// a trigger would delete a stop-loss precisely on the illiquid data where it
    /// matters; it stays armed for the next bar that can trade instead (ADR 0010).
    ///
    /// A bar that reaches both a take-profit and the liquidation trigger cannot show
    /// which came first either, and the liquidation is the worse exit. So each perp
    /// fill that, when its turn comes, would reduce the position it meets with its
    /// reference on the favourable side of where that position started on the bar is
    /// held back when the check would find the trigger without those fills, and the
    /// liquidation then cancels them; otherwise they fill in turn with the rest
    /// (ADR 0012).
    pub(super) fn resolve_resting(&mut self) {
        if self.book.is_empty() {
            return;
        }
        let fills = self.offers();
        let hold = self.unseen.hold && self.may_hold(&fills) && self.liquidates_without(&fills);
        self.fill_offers(fills, hold);

        // IOC and FOK limits get one bar of exposure, the first their instrument
        // prints, then expire whether or not they filled: they never rest for a
        // later bar (ADR 0010).
        let expiring: Vec<(OrderId, usize, TimeInForce)> = self
            .book
            .iter()
            .filter(|o| {
                o.kind == OrderType::Limit
                    && matches!(o.tif, TimeInForce::Ioc | TimeInForce::Fok)
                    && printed(&self.candles.bar(self.tick, o.instrument))
            })
            .map(|o| (o.id, o.instrument, o.tif))
            .collect();
        for (id, instrument, tif) in expiring {
            self.log_event(
                EventKind::Expired,
                Some(id),
                Some(instrument),
                None,
                None,
                EventReason::OneBar(tif),
            );
            self.book.cancel(id);
        }
    }

    /// The resting orders the bar reaches, each with the fill the model offers, in
    /// slot order.
    fn offers(&self) -> Vec<Offer> {
        let mut fills: Vec<Offer> = Vec::new();
        for order in self.book.iter() {
            let bar = self.bar_for(order.instrument);
            if !printed(&bar) {
                continue;
            }
            let fill = match order.kind {
                OrderType::Limit => self.fill_model.fill_limit(order, &bar),
                OrderType::Stop => self.fill_model.fill_stop(order, &bar),
                OrderType::Market => None,
            };
            if let Some((fill, costs)) = fill {
                fills.push((
                    order.id,
                    order.instrument,
                    fill,
                    order.kind,
                    order.reduce_only,
                    order.tif,
                    order.remaining().get(),
                    costs,
                ));
            }
        }
        fills
    }

    /// A perp fill that reduces the position as it now stands, the price the bar
    /// reached for it on the side of where that position started on the bar that it
    /// gains on: a long's sale above its start, a short's purchase below it. The path
    /// takes a position away from its start, so the bar reached such a price only on
    /// a move back, which it cannot show came before the trigger. The fill's
    /// reference is that price; what the fill pays beyond it was never on the bar.
    fn favourable(&self, &(_, i, fill, .., costs): &Offer) -> bool {
        let (position, start) = (self.account.positions[i].qty.get(), self.starts[i]);
        self.account.markets[i] == Market::Perp
            && match fill.side {
                Side::Sell => position > 0.0 && costs.reference > start,
                Side::Buy => position < 0.0 && costs.reference < start,
            }
    }

    /// Whether any offer could meet a position it reduces on the favourable side of
    /// where that position started: one favourable as the bar's resting orders find
    /// the account, or one reached beyond a fill on its instrument on the other side,
    /// which could open or flip that position there, in whatever order the bar's
    /// exits come to be taken.
    fn may_hold(&self, fills: &[Offer]) -> bool {
        let beyond = |offer: &Offer, at: f64| match offer.2.side {
            Side::Sell => offer.7.reference > at,
            Side::Buy => offer.7.reference < at,
        };
        fills.iter().any(|offer| {
            self.favourable(offer)
                || (self.account.markets[offer.1] == Market::Perp
                    && fills.iter().any(|other| {
                        other.1 == offer.1
                            && other.2.side != offer.2.side
                            && beyond(offer, other.7.reference)
                    }))
        })
    }

    /// Whether the check would find the trigger had the bar's resting orders filled
    /// with its favourable exits held back. Where every perp order among them is one
    /// such exit as the orders find the account, none fills and the account the check
    /// would read is the one standing; otherwise they fill on that account with
    /// nothing recorded, and the account and the book are put back.
    fn liquidates_without(&mut self, fills: &[Offer]) -> bool {
        if fills
            .iter()
            .all(|offer| self.account.markets[offer.1] != Market::Perp || self.favourable(offer))
        {
            return self.reaches_trigger();
        }
        let (account, book) = (self.account.clone(), self.book.clone());
        let (notes, flipped, count) = (self.notes.len(), self.flipped, self.fills);
        let starts = self.starts.clone();
        let reporter = self.reporter.take();
        let held = self.fill_offers(fills.to_vec(), true);
        let liquidates = held && self.reaches_trigger();
        self.account = account;
        self.book = book;
        self.starts = starts;
        self.notes.truncate(notes);
        self.flipped = flipped;
        self.fills = count;
        self.reporter = reporter;
        liquidates
    }

    /// Order the exits of the position the offer at `k` meets worst first among the
    /// slots they hold from `k` on, the position as the fills before it left it.
    fn order_exits(&self, fills: &mut [Offer], k: usize) {
        // A bar that reaches both a stop and a target says nothing about which it
        // reached first, and the answer used to be slot order, which is the order
        // those two lines happen to appear in the strategy: place the stop first
        // and every ambiguous bar books a loss, swap the two lines and every one
        // books a win. They are ordered by how badly they end for the position
        // instead, worst first, which does not depend on how the strategy was
        // typed and is the same pessimism the rest of the engine already applies
        // (ADR 0012). The sort is stable, so anything that is not an exit keeps
        // its slot order, and the exits of different instruments, which close
        // different positions, are never ordered against each other. They are
        // ordered as each comes up, against the position it meets, so the exits of
        // a position the bar's own fills open or flip are ordered too, where a sort
        // made once against the position the bar started with left them in the
        // order they were typed
        let i = fills[k].1;
        let position = self.account.positions[i].qty.get();
        let entry = self.account.positions[i].avg_entry.get();
        let rank = |offer: &Offer| adversity(position, entry, offer.2.side, offer.2.price.get());
        if position == 0.0 || !rank(&fills[k]).is_finite() {
            return;
        }
        // The exits are reordered among the slots the exits already hold, and
        // everything else stays exactly where it was. Sorting the whole list
        // instead moved every non-exit behind every exit, which is a different
        // rule than the one written down and not a harmless one: it let an
        // entry be funded by the proceeds of a sale that may just as well have
        // come after it, so the account ended a wide bar richer than slot order
        // (ADR 0011) would have left it, which is the optimism ADR 0012 exists
        // to remove rather than relocate (ADR 0012)
        let slots: Vec<usize> = (k..fills.len())
            .filter(|&j| fills[j].1 == i && rank(&fills[j]).is_finite())
            .collect();
        if slots.len() < 2 {
            return;
        }
        // the worse first, or under the optimistic setting the better, ties left in
        // slot order either way
        let worse = self.unseen.exits[i];
        let mut ordered: Vec<usize> = slots.clone();
        ordered.sort_by(|&a, &b| {
            let (a, b) = if worse { (a, b) } else { (b, a) };
            rank(&fills[a])
                .partial_cmp(&rank(&fills[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let exits: Vec<_> = ordered.iter().map(|&j| fills[j]).collect();
        for (slot, exit) in slots.iter().zip(exits) {
            fills[*slot] = exit;
        }
    }

    /// Fill the offers in turn, each clamped against the account the ones before it
    /// left and the exits among them ordered as each comes up, with `hold` holding
    /// back each that would reduce the position it meets on the favourable side of
    /// where that position started. Returns whether any was held back.
    fn fill_offers(&mut self, mut fills: Vec<Offer>, hold: bool) -> bool {
        let mut held = false;
        for k in 0..fills.len() {
            self.order_exits(&mut fills, k);
            let offer = fills[k];
            if hold && self.favourable(&offer) {
                held = true;
                continue;
            }
            let (id, i, fill, kind, reduce_only, tif, requested, costs) = offer;
            // FOK limit: all or nothing against every clamp, not just the bar's
            // liquidity, and judged against the account as it stands when this
            // order is reached rather than the one its offer was made on, and left
            // for the expiry sweep that follows to cancel unfilled (ADR 0010).
            if kind == OrderType::Limit
                && tif == TimeInForce::Fok
                && self.clamp_fill(i, reduce_only, &fill).size.get() + CLOSE_EPS < requested
            {
                continue;
            }
            let reason = if kind == OrderType::Stop {
                FillReason::Stop
            } else {
                FillReason::Limit
            };
            let bar = self.bar_for(i);
            let applied =
                self.apply_fill_clamped(i, id, reason, reduce_only, requested, fill, costs, &bar);
            match kind {
                // A stop is consumed by filling and by nothing else, which is what
                // ADR 0010 says and what it used to say only for a bar with no
                // volume at all. Triggering and being cancelled regardless meant a
                // bar carrying a thousandth of a unit filled a token size and threw
                // the rest of the protection away, while a bar carrying exactly
                // nothing kept all of it: on one crash that discontinuity was the
                // difference between exiting a 10-lot and riding 9.999 of it down.
                // It also deleted a breakout stop outright whenever a cash or margin
                // clamp took the fill to zero. So it accumulates and rests like a
                // limit, and the remainder stays armed for the next bar that can
                // actually trade it (ADR 0010).
                OrderType::Stop | OrderType::Limit => {
                    if applied > 0.0 {
                        if let Some(order) = self.book.get_mut(id) {
                            order.filled = Qty(order.filled.get() + applied);
                        }
                    }
                    // retired at the dust epsilon, not at zero, because that is
                    // where `clamp_fill` stops booking: an order whose remaining
                    // lands in between can never fill again and never expires, so
                    // it holds a book slot for the rest of the run. Ten fills of
                    // 0.1 against a size of 1.0 leave 1.1e-16, which is all it
                    // takes. Enough of those and the book is full of orders that
                    // will never trade, and every later placement is refused in
                    // silence, which is the harm ADR 0007 names (ADR 0010)
                    let done = self
                        .book
                        .get(id)
                        .is_some_and(|o| o.remaining().get() <= CLOSE_EPS);
                    if done {
                        self.book.cancel(id);
                    }
                }
                OrderType::Market => {}
            }
        }
        held
    }
}

#[cfg(test)]
mod tests {
    use super::adversity;
    use crate::candles::Candles;
    use crate::engine::fixtures::{
        cfg, crash_series, dip_series, ohlc, perp_cfg, report_cfg, series, universe, ABSENT,
    };
    use crate::engine::{Engine, EngineConfig, Intrabar, CLOSE_EPS};
    use crate::reporter::EventKind;
    use crate::reporter::FillReason;
    use probatum_core::{Candle, Market, OrderType, Side, TimeInForce};

    #[test]
    fn a_gtc_limit_fills_across_bars_under_the_volume_cap() {
        // ADR 0013: a resting limit fills up to the volume cap each bar and keeps its
        // remainder resting for the next, so a large limit completes over several bars
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 50.0, 100.0, 10.0), // decision bar
            ohlc(60.0, 60.0, 40.0, 60.0, 10.0),    // touches the 60 limit, low volume
            ohlc(60.0, 60.0, 40.0, 60.0, 10.0),    // touches again
        ]);
        let cfg = EngineConfig {
            max_fill_fraction: 0.5, // cap each fill at half the bar volume (5 units)
            ..cfg()
        };
        let mut e = Engine::new(candles, cfg);
        e.reset();
        e.limit_buy(9.0, 60.0); // wants 9 at 60; the cap allows 5 per bar
        let s1 = e.step(); // bar 1: fills 5, 4 remain resting
        assert_eq!(s1.position, 5.0);
        assert_eq!(s1.open_orders.len(), 1);
        let s2 = e.step(); // bar 2: fills the remaining 4 and completes
        assert_eq!(s2.position, 9.0);
        assert!(s2.open_orders.is_empty());
    }

    #[test]
    fn an_ioc_limit_expires_after_its_one_bar() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.order(
            Side::Buy,
            1.0,
            OrderType::Limit,
            Some(50.0), // never reached, so it does not fill
            None,
            false,
            false,
            TimeInForce::Ioc,
        );
        let s = e.step(); // its one bar of exposure passes
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty()); // IOC did not rest, unlike a GTC limit
    }

    #[test]
    fn a_fok_limit_fills_fully_or_not_at_all() {
        // bar 1 dips to 94 and trades 1000; a FOK for more than the volume cap
        // cannot fill in full, so it fills nothing and does not rest
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.order(
            Side::Buy,
            2000.0,
            OrderType::Limit,
            Some(95.0),
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn a_fok_limit_fills_when_the_whole_size_is_available() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.order(
            Side::Buy,
            10.0,
            OrderType::Limit,
            Some(95.0),
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 10.0); // 10 within the cap, filled whole at 95
    }

    #[test]
    fn impact_scales_with_the_fraction_actually_filled() {
        // both existing fixtures take exactly a tenth of their bar, so they are
        // degenerate in the one dimension the claim is about and two wrong
        // implementations reproduce them exactly: a hard-coded tenth, and the
        // REQUESTED size over the volume rather than the filled one, which are the
        // same number whenever max_fill_fraction does not bind. Here it binds, so
        // requested, capped and a tenth are three different fractions
        let wide = Candles::new(vec![
            ohlc(100.0, 200.0, 90.0, 100.0, 1000.0),
            ohlc(100.0, 200.0, 90.0, 100.0, 1000.0),
            ohlc(100.0, 200.0, 90.0, 100.0, 1000.0),
        ]);
        // both caps have to BIND on a request of 400, or the fraction is the
        // request's own and the second mutant is indistinguishable again
        for (cap, filled, price) in [(0.25, 250.0, 125.0), (0.2, 200.0, 120.0)] {
            let mut config = cfg();
            config.quote = 1_000_000.0;
            config.impact = 1.0;
            config.max_fill_fraction = cap;
            let mut e = Engine::new(wide.clone(), config);
            e.reset();
            e.market_buy(400.0); // more than either cap allows
            let s = e.step();
            assert_eq!(s.position, filled, "cap {cap} filled the wrong size");
            // impact 1.0 times the FILLED fraction, off an open of 100
            assert_eq!(
                s.quote,
                1_000_000.0 - filled * price,
                "cap {cap} priced the fill at something other than {price}"
            );
        }
    }

    #[test]
    fn a_fok_limit_the_cash_cannot_afford_fills_nothing() {
        // ADR 0010 says all or nothing against EVERY clamp, not just the bar's
        // liquidity, and the resting path judges that with a second copy of the
        // rule the pending path already has. Only the pending one was pinned:
        // a_fok_limit_fills_fully_or_not_at_all is refused by the volume cap
        // before any clamp is consulted, so the case the decision actually names,
        // where the bar HAS the liquidity and a clamp shrinks the fill, was never
        // reached on a limit. Comparing the pre-clamp size here books the five
        // units the quote affords out of the twenty asked
        let mut config = cfg();
        config.quote = 1_000.0;
        config.report = true;
        let mut e = Engine::new(dip_series(), config);
        e.reset();
        e.order(
            Side::Buy,
            20.0,
            OrderType::Limit,
            Some(200.0),
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(
            s.position, 0.0,
            "a FOK the cash cannot cover filled in part"
        );
        assert_eq!(s.quote, 1_000.0);
        assert_eq!(e.num_fills(), 0);
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn a_fok_market_fills_fully_or_nothing() {
        let mut config = cfg();
        config.max_fill_fraction = 0.1; // cap at 100 of a 1000-volume bar
        let mut e = Engine::new(series(), config);
        e.reset();
        e.order(
            Side::Buy,
            250.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 0.0); // 250 over the cap of 100, so FOK fills nothing
    }

    #[test]
    fn limit_rests_then_fills_when_the_bar_reaches_it() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.limit_buy(1.0, 95.0); // placed at bar 0
        let s1 = e.step(); // bar 1 dips to 94, reaching 95
        assert_eq!(s1.position, 1.0);
        assert_eq!(s1.avg_entry, 95.0); // at the limit, maker, no slippage
        assert!(s1.open_orders.is_empty()); // fully filled, removed
    }

    #[test]
    fn sell_stop_triggers_when_a_bar_crosses_it() {
        // a sell stop from flat opens a short, a perp behavior (ADR 0008)
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        e.stop(Side::Sell, 1.0, 95.0, false); // placed at bar 0
        let s1 = e.step(); // bar 1 low 80 crosses 95; open 100 so fills at trigger 95
        assert_eq!(s1.position, -1.0);
        assert_eq!(s1.avg_entry, 95.0);
        assert!(s1.open_orders.is_empty());
    }

    #[test]
    fn a_fok_market_blocked_by_the_cash_clamp_fills_nothing() {
        // the bar has the liquidity, but spot cash affords only 50 of the 1000, and
        // FOK is all or nothing against every clamp, not just volume (ADR 0010)
        let mut e = Engine::new(series(), cfg()); // spot, quote 10_000, bar 1 open 200
        e.reset();
        e.order(
            Side::Buy,
            1000.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0);
    }

    #[test]
    fn a_fok_market_blocked_by_the_leverage_cap_fills_nothing() {
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        // 2x on 100 equity at the bar 1 open of 200 allows 1.0, so a FOK for 10 dies
        e.order(
            Side::Buy,
            10.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        assert_eq!(e.step().position, 0.0);
    }

    #[test]
    fn two_resting_orders_filling_on_one_bar_are_clamped_in_sequence() {
        // the fills are collected against a pre-fill snapshot and applied one at a
        // time, so the second must be clamped against the account the first left
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
            ohlc(100.0, 100.0, 80.0, 100.0, 1_000.0),
        ]);
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(candles, config);
        e.reset();
        e.limit_buy(2.0, 90.0).unwrap();
        e.limit_buy(2.0, 90.0).unwrap();
        let s = e.step();
        // 2x on 100 equity at a fill price of 90 allows 2.222 units in total, so the
        // first limit fills whole and the second is clamped to the remainder
        assert!(
            (s.position - 2.0 * 100.0 / 90.0).abs() < 1e-9,
            "position {}",
            s.position
        );
    }

    #[test]
    fn two_pending_market_orders_are_clamped_in_sequence() {
        let mut e = Engine::new(series(), cfg()); // spot, quote 10_000, bar 1 open 200
        e.reset();
        e.market_buy(40.0);
        e.market_buy(40.0);
        let s = e.step();
        // 10_000 of cash at 200 buys 50 in total, not 80
        assert_eq!(s.position, 50.0);
        assert_eq!(s.quote, 0.0);
    }

    #[test]
    fn a_stop_crossed_on_a_dead_bar_stays_armed() {
        // bar 1 crosses the trigger but trades nothing, so the stop cannot fill. It
        // must survive: consuming it here would delete the protection on exactly the
        // illiquid data where a stop matters, and it fills on bar 2 instead (ADR 0010)
        let candles = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 80.0, 85.0, 0.0), // crosses 95, no volume
            ohlc(85.0, 86.0, 84.0, 85.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, perp_cfg());
        e.reset();
        e.stop(Side::Sell, 1.0, 95.0, false).unwrap();
        let s1 = e.step();
        assert_eq!(s1.position, 0.0, "nothing can fill on a bar with no volume");
        assert_eq!(s1.open_orders.len(), 1, "the stop must still be armed");
        let s2 = e.step(); // bar 2 trades, and the trigger is still crossed
        assert_eq!(s2.position, -1.0);
        assert!(s2.open_orders.is_empty(), "and is consumed once it fills");
    }

    #[test]
    fn ioc_limit_partial_fill_applies_then_cancels_the_rest() {
        // an IOC limit larger than the bar's fillable volume takes what it can this
        // bar and cancels the remainder rather than resting
        let mut config = cfg(); // spot
        config.max_fill_fraction = 0.1; // at most 10% of a bar's volume per order
        let candles = Candles::new(vec![
            ohlc(100.0, 160.0, 90.0, 150.0, 1_000.0),
            ohlc(100.0, 160.0, 40.0, 150.0, 1_000.0), // low 40 reaches the limit at 50
            ohlc(100.0, 160.0, 90.0, 150.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, config);
        e.reset();
        e.place_limit(0, Side::Buy, 500.0, 50.0, false, false, TimeInForce::Ioc);
        let s = e.step(); // reaches 50, fills 100 (10% of 1000), cancels the other 400
        assert_eq!(s.position, 100.0);
        assert!(s.open_orders.is_empty()); // IOC: nothing rests
    }

    /// Where a long of 1 from 100 exits on one wide bar that reaches a stop at 90 and
    /// a target at 110, typed in the order `stop_first` says.
    fn both_reached(stop_first: bool, intrabar: Intrabar) -> f64 {
        both_reached_in(stop_first, intrabar, None)
    }

    /// `both_reached` in env `env` of a batch, or alone with none.
    fn both_reached_in(stop_first: bool, intrabar: Intrabar, env: Option<usize>) -> f64 {
        let candles = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 120.0, 80.0, 100.0, 1000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
        ]);
        let config = EngineConfig {
            intrabar,
            ..report_cfg() // spot, so a sell clamps
        };
        let mut e = Engine::new(candles, config);
        if let Some(env) = env {
            e = e.for_env(env);
        }
        e.reset();
        e.market_buy(1.0);
        e.step();
        if stop_first {
            e.stop(Side::Sell, 1.0, 90.0, true);
            e.limit_sell(1.0, 110.0);
        } else {
            e.limit_sell(1.0, 110.0);
            e.stop(Side::Sell, 1.0, 90.0, true);
        }
        let s = e.step();
        assert_eq!(s.position, 0.0, "stop_first={stop_first}");
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1, "stop_first={stop_first}");
        trades[0].exit
    }

    #[test]
    fn a_bar_reaching_both_a_stop_and_a_target_books_the_stop_either_way() {
        // the engine cannot know which the bar touched first. It used to answer
        // with slot order, so placing the stop before the target booked a loss on
        // every ambiguous bar and swapping the two lines booked a win: the sign of
        // a bracket strategy's result was decided by the order it was typed in
        for stop_first in [true, false] {
            let exit = both_reached(stop_first, Intrabar::Pessimistic);
            assert_eq!(exit, 90.0, "stop_first={stop_first}");
        }
    }

    #[test]
    fn an_optimistic_bar_reaching_both_books_the_target_either_way() {
        // the optimistic setting takes the better exit first, whichever was typed
        // first, which with the pessimistic run brackets what the bar could hold
        for stop_first in [true, false] {
            let exit = both_reached(stop_first, Intrabar::Optimistic);
            assert_eq!(exit, 110.0, "stop_first={stop_first}");
        }
    }

    #[test]
    fn a_random_bar_books_the_stop_or_the_target_as_its_stream_draws() {
        // each seed draws the bar's way, the same way each time it runs, and both
        // ways come up over a few seeds, and over the envs of one seed
        let alone: Vec<f64> = (0..32)
            .map(|seed| both_reached(true, Intrabar::Random(seed)))
            .collect();
        let again: Vec<f64> = (0..32)
            .map(|seed| both_reached(true, Intrabar::Random(seed)))
            .collect();
        assert_eq!(alone, again);
        let envs: Vec<f64> = (0..32)
            .map(|env| both_reached_in(true, Intrabar::Random(7), Some(env)))
            .collect();
        for exits in [alone, envs] {
            assert!(exits.iter().all(|&exit| exit == 90.0 || exit == 110.0));
            assert!(exits.contains(&90.0) && exits.contains(&110.0), "{exits:?}");
        }
    }

    #[test]
    fn ordering_the_exits_leaves_an_unambiguous_bar_alone() {
        // only a bar reaching BOTH legs is ambiguous; one that reaches the target
        // and never the stop still books the target
        let candles = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 120.0, 99.0, 115.0, 1000.0), // reaches 110, never 90
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
        ]);
        let mut e = Engine::new(candles, report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.stop(Side::Sell, 1.0, 90.0, true);
        e.limit_sell(1.0, 110.0);
        e.step();
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit, 110.0);
        assert!(trades[0].pnl > 0.0);
    }

    #[test]
    fn a_straddle_of_limits_at_one_price_never_ends_richer() {
        // ADR 0011 accepts that a buy and a sell resting at the same price can both
        // fill against one wide bar, which is a bar-fidelity artifact. Booking both
        // as makers turned that artifact into a money printer, because a maker rate
        // is legally a rebate: the account came back flat and richer every bar,
        // without bound, and a search over anything placing limits walks into it
        for maker in [-0.0002, 0.0, 0.0002] {
            for market in [Market::Spot, Market::Perp] {
                let config = EngineConfig {
                    market,
                    quote: 10_000.0,
                    fee_maker: maker,
                    fee_taker: 0.0006,
                    max_leverage: 10.0,
                    ..cfg()
                };
                let candles = Candles::new(vec![
                    ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                    ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                    ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                ]);
                let mut e = Engine::new(candles, config);
                e.reset();
                e.place_limit(0, Side::Buy, 1.0, 100.0, false, false, TimeInForce::Gtc);
                e.place_limit(0, Side::Sell, 1.0, 100.0, false, false, TimeInForce::Gtc);
                let s = e.step();
                assert!(s.position.abs() <= CLOSE_EPS, "{market:?} left a position");
                assert!(
                    s.equity <= 10_000.0 + CLOSE_EPS,
                    "{market:?} at maker {maker} ended with {}",
                    s.equity
                );
            }
        }
    }

    #[test]
    fn a_stop_is_consumed_by_filling_and_by_nothing_else() {
        // ADR 0010 binds triggering to filling, and that held only for a bar with no
        // volume at all. A cash clamp that took the fill to zero deleted the stop
        // just as thoroughly, and it is the same protection either way
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 150.0, 100.0, 150.0, 1e6),
            ohlc(150.0, 150.0, 150.0, 150.0, 1e6),
        ]);
        let mut e = Engine::new(
            bars,
            EngineConfig {
                quote: 100.0,
                ..cfg()
            },
        );
        e.reset();
        e.market_buy(1.0);
        e.stop(Side::Buy, 1.0, 110.0, false);
        let state = e.step();
        assert_eq!(e.num_fills(), 1, "the market order took all the cash");
        assert_eq!(state.open_orders.len(), 1, "the breakout stop was deleted");
    }

    #[test]
    fn a_partly_filled_stop_keeps_the_rest_of_its_protection() {
        // consuming a stop on any fill at all made a bar carrying a thousandth of a
        // unit far more dangerous than a bar carrying none: the token fill threw the
        // remaining protection away, while zero volume kept all of it
        for volume in [0.0, 0.001, 1.0] {
            let bars = Candles::new(vec![
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(100.0, 101.0, 80.0, 85.0, volume),
                ohlc(85.0, 86.0, 60.0, 60.0, 1e6),
                ohlc(60.0, 60.0, 60.0, 60.0, 1e6),
            ]);
            let mut e = Engine::new(
                bars,
                EngineConfig {
                    quote: 100_000.0,
                    ..perp_cfg()
                },
            );
            e.reset();
            e.market_buy(10.0);
            e.step();
            e.stop(Side::Sell, 10.0, 95.0, true);
            e.step();
            e.step();
            let state = e.step();
            assert!(
                state.position.abs() <= CLOSE_EPS,
                "volume {volume} left {} riding the crash",
                state.position
            );
        }
    }

    #[test]
    fn the_worst_first_sort_leaves_everything_that_is_not_an_exit_where_it_was() {
        // ADR 0012 says the sort applies only to fills that reduce the position, and
        // ranking a non-exit at infinity moved every one of them behind every exit.
        // That let an entry be funded by the proceeds of a sale that may well have
        // come after it, so the account ended the bar richer than slot order leaves it
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 120.0, 80.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(
            bars,
            EngineConfig {
                quote: 150.0,
                ..cfg()
            },
        );
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.limit_buy(1.0, 90.0); // slot 0, an increase
        e.limit_sell(1.0, 110.0); // slot 1, an exit
        let state = e.step();
        // slot order: the buy is clamped to the 50 of cash on hand, then the sell
        assert!(
            (state.position - 50.0 / 90.0).abs() < 1e-9,
            "{}",
            state.position
        );
        assert!(
            (state.equity - 165.5555555555).abs() < 1e-6,
            "{}",
            state.equity
        );
    }

    #[test]
    fn an_order_filled_down_to_float_dust_gives_its_slot_back() {
        // the fill gate refuses a size at or below the dust epsilon and the retire
        // gate asked for zero, so an order left holding the difference could never
        // trade again and never expired. Ten fills of 0.1 against a size of 1.0
        // leave 1.1e-16 behind, which is enough to hold a slot for the whole run;
        // a strategy resting orders on a long series eventually finds every
        // placement silently refused (ADR 0010)
        let bars: Vec<Candle> = (0..14)
            .map(|_| ohlc(100.0, 100.0, 100.0, 100.0, 0.1))
            .collect();
        let mut e = Engine::new(Candles::new(bars), cfg());
        e.reset();
        e.limit_buy(1.0, 100.0);
        for _ in 0..12 {
            e.step();
        }
        let state = e.current_state();
        assert!(
            (state.position - 1.0).abs() < 1e-9,
            "the order should have filled out: {}",
            state.position
        );
        assert!(
            state.open_orders.is_empty(),
            "a filled order kept its slot with {:?} left",
            state.open_orders.first().map(|o| o.remaining().get())
        );
    }

    #[test]
    fn adversity_ranks_an_exit_by_what_it_realizes_and_sorts_everything_else_last() {
        // the whole of ADR 0012 rests on this eight-line function, and the suite
        // reached it only through the engine, where one wide bar exercises one
        // branch. Every comparison and every term is pinned here instead
        let long_loss = adversity(2.0, 100.0, Side::Sell, 90.0);
        let long_win = adversity(2.0, 100.0, Side::Sell, 110.0);
        assert_eq!(long_loss, -10.0);
        assert_eq!(long_win, 10.0);
        assert!(long_loss < long_win, "a long's worse exit must sort first");

        // a short is the mirror: the HIGHER price is the worse exit
        let short_loss = adversity(-2.0, 100.0, Side::Buy, 110.0);
        let short_win = adversity(-2.0, 100.0, Side::Buy, 90.0);
        assert_eq!(short_loss, -10.0);
        assert_eq!(short_win, 10.0);
        assert!(
            short_loss < short_win,
            "a short's worse exit must sort first"
        );

        // anything that is not an exit sorts last and never reorders (ADR 0012)
        for (position, side) in [
            (2.0, Side::Buy),   // adding to a long
            (-2.0, Side::Sell), // adding to a short
            (0.0, Side::Buy),   // opening from flat
            (0.0, Side::Sell),
        ] {
            assert_eq!(
                adversity(position, 100.0, side, 90.0),
                f64::INFINITY,
                "position {position} with a {side:?} is not an exit"
            );
        }
        assert!(long_loss < f64::INFINITY);
    }

    #[test]
    fn a_stop_and_a_limit_on_one_side_are_ranked_at_their_fill_prices() {
        // a long of 1 with a stop at 95 and a marketable limit at 94.80, on a bar
        // that opens at 96 and falls to 90 with 50 bps of slippage. The stop would
        // realize its fill price, 94.525, worse than the limit's, so it fills first
        // whatever its trigger, and the limit, reduce-only, rests
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(96.0, 97.0, 90.0, 92.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 1_000.0,
            slippage_bps: 50.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.stop(Side::Sell, 1.0, 95.0, true);
        let limit = e.order(
            Side::Sell,
            1.0,
            OrderType::Limit,
            Some(94.8),
            None,
            true,
            false,
            TimeInForce::Gtc,
        );
        let s = e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let last = fills.last().expect("the exit");
        assert_eq!(last.reason, FillReason::Stop);
        assert!((last.price - 94.525).abs() < 1e-9, "sold at {}", last.price);
        assert_eq!(s.position, 0.0);
        let resting: Vec<_> = s.open_orders.iter().map(|order| order.id).collect();
        assert_eq!(resting, vec![limit.expect("the limit rests")]);
    }

    #[test]
    fn the_exits_of_a_position_its_own_bar_opened_are_ordered_worst_first() {
        // a flat account rests a buy at 95 and, reduce-only, a take-profit at 110 and
        // a stop at 90, and a bar from 100 reaches all three. The buy opens a long
        // whose two exits the bar reached in an order it cannot show, so the worse,
        // the stop, fills first whichever was typed first, and the account ends 5
        // down
        for stop_first in [false, true] {
            let bars = Candles::new(vec![
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(100.0, 112.0, 85.0, 100.0, 1e6),
            ]);
            let config = EngineConfig {
                report: true,
                ..perp_cfg()
            };
            let mut e = Engine::new(bars, config);
            e.reset();
            e.limit_buy(1.0, 95.0);
            let take = |e: &mut Engine| {
                e.order(
                    Side::Sell,
                    1.0,
                    OrderType::Limit,
                    Some(110.0),
                    None,
                    true,
                    false,
                    TimeInForce::Gtc,
                )
            };
            if stop_first {
                e.stop(Side::Sell, 1.0, 90.0, true);
                take(&mut e);
            } else {
                take(&mut e);
                e.stop(Side::Sell, 1.0, 90.0, true);
            }
            let s = e.step();
            let fills: Vec<(FillReason, f64)> = e
                .reporter()
                .expect("reporting is on")
                .fills()
                .iter()
                .map(|fill| (fill.reason, fill.price))
                .collect();
            assert_eq!(
                fills,
                vec![(FillReason::Limit, 95.0), (FillReason::Stop, 90.0)],
                "stop_first={stop_first}"
            );
            assert_eq!(s.margin_cash, 95.0, "stop_first={stop_first}");
        }
    }

    #[test]
    fn a_fok_order_every_clamp_leaves_whole_fills_whole_at_any_size() {
        // at 20,000,000 units half an ulp is larger than the dust epsilon, so what
        // filled is compared with what was asked without leaning on it
        for kind in [OrderType::Market, OrderType::Limit] {
            for size in [10.0, 20_000_000.0] {
                let config = EngineConfig {
                    quote: 100_000_000.0,
                    ..report_cfg()
                };
                let flat = ohlc(1.0, 1.0, 1.0, 1.0, 1e8);
                let mut e = Engine::new(Candles::new(vec![flat; 2]), config);
                e.reset();
                e.order(
                    Side::Buy,
                    size,
                    kind,
                    Some(1.0),
                    None,
                    false,
                    false,
                    TimeInForce::Fok,
                );
                let state = e.step();
                assert_eq!(state.position, size, "{kind:?} of {size}");
                let kinds: Vec<EventKind> = e
                    .reporter()
                    .expect("reporting is on")
                    .events()
                    .iter()
                    .map(|event| event.kind)
                    .collect();
                assert_eq!(
                    kinds,
                    vec![EventKind::Placed, EventKind::Filled],
                    "{kind:?} of {size}"
                );
            }
        }
    }

    #[test]
    fn an_ioc_limit_takes_the_first_bar_its_instrument_prints() {
        // the second instrument prints nothing on the second row, so the ioc bid on
        // it is still resting after that row and expires after the third
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(10.0)],
            vec![flat(100.0), ABSENT],
            vec![flat(100.0), flat(10.0)],
        ];
        let mut e = Engine::new(universe(&[Market::Spot, Market::Spot], rows), report_cfg());
        e.reset();
        e.order_on(
            1,
            Side::Buy,
            1.0,
            OrderType::Limit,
            Some(9.0),
            None,
            false,
            false,
            TimeInForce::Ioc,
        );
        assert_eq!(e.step().open_orders.len(), 1);
        assert!(e.step().open_orders.is_empty());
    }

    #[test]
    fn the_exits_of_two_instruments_are_never_ordered_against_each_other() {
        // a wide bar reaches the second instrument's take-profit, in the first slot,
        // and the first instrument's stop, in the second: they close different
        // positions, so they fill in slot order however badly each ends
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(10.0)],
            vec![flat(100.0), flat(10.0)],
            vec![
                ohlc(100.0, 120.0, 80.0, 100.0, 1e6),
                ohlc(10.0, 12.0, 8.0, 10.0, 1e6),
            ],
        ];
        let mut e = Engine::new(universe(&[Market::Spot, Market::Spot], rows), report_cfg());
        let place = |e: &mut Engine, i, side, size, kind, price, trigger| {
            e.order_on(
                i,
                side,
                size,
                kind,
                price,
                trigger,
                false,
                false,
                TimeInForce::Gtc,
            )
        };
        e.reset();
        place(&mut e, 0, Side::Buy, 1.0, OrderType::Market, None, None);
        place(&mut e, 1, Side::Buy, 10.0, OrderType::Market, None, None);
        e.step();
        place(
            &mut e,
            1,
            Side::Sell,
            10.0,
            OrderType::Limit,
            Some(11.0),
            None,
        );
        place(
            &mut e,
            0,
            Side::Sell,
            1.0,
            OrderType::Stop,
            None,
            Some(90.0),
        );
        e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let order: Vec<(usize, f64)> = fills.iter().map(|f| (f.instrument, f.price)).collect();
        assert_eq!(order, vec![(0, 100.0), (1, 10.0), (1, 11.0), (0, 90.0)]);
    }
}
