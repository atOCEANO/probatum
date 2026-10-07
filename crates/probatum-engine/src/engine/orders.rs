//! Placing, moving and cancelling orders. A market order is queued for the next
//! bar's open and a limit or a stop rests in the book until a bar reaches it,
//! and `resolution` fills both. A size, price or trigger that could never fill is
//! refused here rather than given a slot, and the order log says why (ADRs 0007,
//! 0010, 0022).

use probatum_core::{Order, OrderId, OrderType, Price, Qty, Side, TimeInForce};

use super::Engine;
use crate::reporter::{EventKind, EventReason, OrderEvent, Refusal};

impl Engine {
    fn next_id(&mut self) -> OrderId {
        let id = OrderId(self.id_counter);
        self.id_counter += 1;
        id
    }

    /// Add an event to the order log when reporting is on.
    pub(super) fn log_event(
        &mut self,
        kind: EventKind,
        order: Option<OrderId>,
        size: Option<f64>,
        price: Option<f64>,
        reason: EventReason,
    ) {
        if let Some(reporter) = self.reporter.as_mut() {
            reporter.record_event(OrderEvent {
                tick: self.tick,
                kind,
                order,
                size,
                price,
                reason,
            });
        }
    }

    /// Log `id` refused for `why`, and answer the placement with nothing.
    fn refuse(&mut self, id: OrderId, why: Refusal) -> Option<OrderId> {
        self.log_event(
            EventKind::Refused,
            Some(id),
            None,
            None,
            EventReason::Refused(why),
        );
        None
    }

    pub(super) fn place_market(
        &mut self,
        side: Side,
        size: f64,
        reduce_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        // Every placement takes an id, a refused one included, so the order log can
        // name what it refused (ADR 0010).
        let id = self.next_id();
        // The same guard the resting book has carried since ADR 0007, extended to
        // the queue on the day ADR 0010 made the queue a bounded resource too. A
        // size that can never fill used to take a slot anyway, so two of them ahead
        // of a real order refused it: `qty_from_weight` returns exactly zero on a
        // flat mark, which is how a strategy places one without meaning to.
        if !size.is_finite() || size <= 0.0 {
            return self.refuse(id, Refusal::Size(size));
        }
        // The queue holds the same number of orders the resting book does. It was
        // unbounded, and since the volume cap is per order (ADR 0013), one order
        // split into a hundred took a hundred times the cap against a single bar,
        // every slice priced as though it were the only participant. Bounding it
        // does not make the cap honest, it makes the evasion finite; a liquidity
        // budget shared across a bar is the real answer (ADR 0010)
        if self.pending.len() >= self.config.max_open_orders {
            return self.refuse(id, Refusal::Waiting(self.config.max_open_orders));
        }
        let mut order = Order::market(id, side, Qty(size), reduce_only);
        order.tif = tif;
        self.pending.push(order);
        self.log_event(EventKind::Placed, Some(id), None, None, EventReason::Market);
        Some(id)
    }

    /// Why a limit or a stop could not rest as `order`, or `None`: a size, price or
    /// trigger that can never fill, a full book, or a post_only limit that would
    /// cross. `replacing` is the order a replace moves, whose slot has just been given
    /// up (ADRs 0007, 0010).
    fn resting_refusal(&self, order: &Order, replacing: Option<&Order>) -> Option<Refusal> {
        // A size that is not a positive finite number can never fill (ADR 0014), so
        // it is refused a slot rather than resting inert forever. A non-finite price
        // is refused for the same reason: every touch test against it compares with
        // NaN and is false, so the order would camp on a slot until the book filled
        // up and later placements were silently rejected (ADR 0007).
        let size = order.remaining().get();
        if !size.is_finite() || size <= 0.0 {
            return Some(Refusal::Size(size));
        }
        if let Some(price) = order.price.map(Price::get) {
            if !price.is_finite() {
                return Some(Refusal::Price(price));
            }
        }
        if let Some(trigger) = order.trigger.map(Price::get) {
            if !trigger.is_finite() {
                return Some(Refusal::Trigger(trigger));
            }
        }
        if replacing.is_none() && self.book.is_full() {
            return Some(Refusal::Resting(self.config.max_open_orders));
        }
        // Crossing is judged against the CURRENT close, the last price the strategy
        // actually saw. The next bar's open is what this order would really cross,
        // and reading it here would be the lookahead the engine refuses everywhere
        // else
        let crosses = order.price.is_some_and(|price| {
            self.fill_model
                .limit_crosses(order.side, price, Price(self.current_close()))
        });
        // post_only is a rule about how an order enters the book, so a replace that
        // leaves the price where it was is not checked again
        let enters = replacing.is_none_or(|old| old.price != order.price);
        if order.post_only && crosses && enters {
            return Some(Refusal::PostOnly);
        }
        None
    }

    /// Rest `order` in the first free slot, or log why it cannot rest.
    fn rest(&mut self, order: Order, reason: EventReason) -> Option<OrderId> {
        if let Some(why) = self.resting_refusal(&order, None) {
            return self.refuse(order.id, why);
        }
        let id = self.book.place(order).ok()?;
        self.log_event(EventKind::Placed, Some(id), None, None, reason);
        Some(id)
    }

    pub(super) fn place_limit(
        &mut self,
        side: Side,
        size: f64,
        price: f64,
        reduce_only: bool,
        post_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        let id = self.next_id();
        let order = Order::limit(
            id,
            side,
            Qty(size),
            Price(price),
            tif,
            post_only,
            reduce_only,
        );
        self.rest(order, EventReason::Limit)
    }

    fn place_stop(
        &mut self,
        side: Side,
        size: f64,
        trigger: f64,
        reduce_only: bool,
    ) -> Option<OrderId> {
        let id = self.next_id();
        let order = Order::stop(id, side, Qty(size), Price(trigger), reduce_only);
        self.rest(order, EventReason::Stop)
    }

    /// The one order primitive the typed shortcuts wrap. Dispatches on `kind`: a
    /// market ignores `price` and `trigger`, a limit needs a `price`, a stop needs
    /// a `trigger`. `tif` applies to market and limit orders; a stop rests until it
    /// triggers (ADR 0010). `None` if the book is full, the size is not a positive
    /// finite number, a post_only limit would cross, or a limit has no price or a
    /// stop no trigger.
    #[allow(clippy::too_many_arguments)]
    pub fn order(
        &mut self,
        side: Side,
        size: f64,
        kind: OrderType,
        price: Option<f64>,
        trigger: Option<f64>,
        reduce_only: bool,
        post_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        match kind {
            OrderType::Market => self.place_market(side, size, reduce_only, tif),
            OrderType::Limit => self.place_limit(side, size, price?, reduce_only, post_only, tif),
            OrderType::Stop => self.place_stop(side, size, trigger?, reduce_only),
        }
    }

    /// Queue a market buy; it fills on the next bar's open. `None` if the queue is
    /// already holding `max_open_orders` for this bar, or the size is not a
    /// positive finite number.
    pub fn market_buy(&mut self, size: f64) -> Option<OrderId> {
        self.place_market(Side::Buy, size, false, TimeInForce::Ioc)
    }

    /// Queue a market sell; it fills on the next bar's open. `None` if the queue is
    /// already holding `max_open_orders` for this bar, or the size is not a
    /// positive finite number.
    pub fn market_sell(&mut self, size: f64) -> Option<OrderId> {
        self.place_market(Side::Sell, size, false, TimeInForce::Ioc)
    }

    /// Rest a buy limit; it fills when a later bar trades down to `price`. `None`
    /// if the book is full, the size is not a positive finite number, or `price`
    /// is not finite.
    pub fn limit_buy(&mut self, size: f64, price: f64) -> Option<OrderId> {
        self.place_limit(Side::Buy, size, price, false, false, TimeInForce::Gtc)
    }

    /// Rest a sell limit; it fills when a later bar trades up to `price`. `None`
    /// if the book is full, the size is not a positive finite number, or `price`
    /// is not finite.
    pub fn limit_sell(&mut self, size: f64, price: f64) -> Option<OrderId> {
        self.place_limit(Side::Sell, size, price, false, false, TimeInForce::Gtc)
    }

    /// Rest a stop that becomes a market order once `trigger` is crossed. `None`
    /// if the book is full, the size is not a positive finite number, or the
    /// trigger is not finite. `reduce_only` makes it a protective stop that can
    /// only shrink the position, never open one on the other side; a stop-loss
    /// wants it true (ADR 0010).
    pub fn stop(
        &mut self,
        side: Side,
        size: f64,
        trigger: f64,
        reduce_only: bool,
    ) -> Option<OrderId> {
        self.place_stop(side, size, trigger, reduce_only)
    }

    /// Base size for a fraction of current equity, marked at the current close.
    pub fn qty_from_weight(&self, fraction: f64) -> f64 {
        self.account
            .qty_from_weight(fraction, Price(self.current_close()))
            .get()
    }

    /// Base size for a cash amount in quote, at the current close.
    pub fn qty_from_quote(&self, cash: f64) -> f64 {
        self.account
            .qty_from_quote(cash, Price(self.current_close()))
            .get()
    }

    /// Cancel a resting order by id. True if it was found and removed.
    pub fn cancel(&mut self, id: OrderId) -> bool {
        let found = self.book.cancel(id).is_some();
        if found {
            self.log_event(
                EventKind::Cancelled,
                Some(id),
                None,
                None,
                EventReason::CancelledByAccount,
            );
        }
        found
    }

    /// Move a resting order: the same order, under the same id, with whichever of
    /// `size`, `price` and `trigger` are given, in the first free slot. Returns the
    /// id.
    ///
    /// `None` when `id` is not resting or the moved order could not rest, and in
    /// that case NOTHING changes. That is the point of it. Re-placing a trailing
    /// stop with `stop()` each bar rests a new order every time, so the one that
    /// fills leaves its siblings live, and on a perp those open a position on the
    /// other side; a trail written with `replace` cannot leave two orders alive,
    /// and once the stop has filled the replacement silently does nothing instead
    /// of arming a fresh one (ADR 0010).
    pub fn replace(
        &mut self,
        id: OrderId,
        size: Option<f64>,
        price: Option<f64>,
        trigger: Option<f64>,
    ) -> Option<OrderId> {
        let (slot, old) = self.book.cancel_at(id)?;
        let mut moved = old;
        if let Some(size) = size {
            moved.size = Qty(old.filled.get() + size);
        }
        if moved.kind == OrderType::Limit {
            moved.price = price.map(Price).or(old.price);
        } else {
            moved.trigger = trigger.map(Price).or(old.trigger);
        }
        if let Some(why) = self.resting_refusal(&moved, Some(&old)) {
            // The cancel came first, so a replacement refused here would otherwise
            // leave nothing behind and still answer `None`, which this method
            // defines as "nothing happened". A caller trailing a stop reads that as
            // "it has already filled" and stops arming one, so a refused size or a
            // non-finite trigger silently disarmed a live stop-loss. The slot it
            // came from is still free, so putting the old order back cannot fail.
            //
            // Back in that same slot, not wherever `place` would put it. `place`
            // takes the FIRST free slot (ADR 0011), so an order whose neighbour had
            // been cancelled was silently promoted up the queue by a replacement
            // that was REFUSED, and resolution follows slot order: on a spot book
            // with the cash clamp binding, the promoted order took the cash the
            // other one would have had. "Nothing happened" has to include the
            // queue (ADR 0010).
            self.book.restore(slot, old);
            self.log_event(
                EventKind::Refused,
                Some(id),
                None,
                None,
                EventReason::Replacing(why),
            );
            return None;
        }
        self.book.place(moved).ok()?;
        self.log_event(
            EventKind::Placed,
            Some(id),
            None,
            None,
            EventReason::Replace,
        );
        Some(id)
    }

    /// Cancel every resting order, returning how many were dropped. Pending market
    /// orders are not resting and fill on the next step regardless.
    pub fn cancel_all(&mut self) -> usize {
        let ids: Vec<OrderId> = self.book.iter().map(|order| order.id).collect();
        for &id in &ids {
            self.cancel(id);
        }
        ids.len()
    }

    /// Flatten the position with a reduce-only market order. `None` when flat.
    pub fn close(&mut self) -> Option<OrderId> {
        let pos = self.account.position.qty.get();
        if pos == 0.0 {
            return None;
        }
        let (side, size) = if pos > 0.0 {
            (Side::Sell, pos)
        } else {
            (Side::Buy, -pos)
        };
        self.place_market(side, size, true, TimeInForce::Ioc)
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{
        cfg, crash_series, dip_series, levered, ohlc, perp_cfg, report_cfg, series,
    };
    use crate::engine::{Engine, EngineConfig};
    use crate::reporter::{EventKind, EventReason, OrderEvent, Refusal};
    use probatum_core::{OrderId, OrderType, Side, TimeInForce};

    fn events(e: &Engine) -> Vec<(u64, EventKind)> {
        log(e)
            .iter()
            .map(|event| (event.order.map_or(0, |id| id.0), event.kind))
            .collect()
    }

    fn log(e: &Engine) -> &[OrderEvent] {
        e.reporter().expect("reporting is on").events()
    }

    #[test]
    fn a_negative_market_size_is_a_no_op() {
        // size is a positive magnitude; a negative buy must not open a short, move
        // the position, or pay a fee
        let cfg = EngineConfig {
            fee_taker: 0.001,
            ..cfg()
        };
        let mut e = Engine::new(series(), cfg);
        e.reset();
        e.market_buy(-2.0);
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0); // no fill, so no fee charged
    }

    #[test]
    fn a_nan_market_size_is_a_no_op() {
        // f64::min keeps the other operand when one side is NaN, so without the cap
        // guard a NaN size would fill the whole volume cap; it must fill nothing and
        // pay nothing (ADR 0014)
        let cfg = EngineConfig {
            fee_taker: 0.001,
            ..cfg()
        };
        let mut e = Engine::new(series(), cfg);
        e.reset();
        e.market_buy(f64::NAN);
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0); // no fill, so no fee charged
    }

    #[test]
    fn a_non_finite_size_limit_or_stop_is_refused_a_slot() {
        // a NaN-size resting order could never complete (NaN minus any fill stays
        // NaN), so placement refuses it instead of letting it camp on a book slot
        // and refill the cap on every touching bar (ADR 0014)
        let mut e = Engine::new(series(), cfg());
        e.reset();
        assert_eq!(e.limit_buy(f64::NAN, 50.0), None);
        assert_eq!(e.stop(Side::Sell, f64::INFINITY, 90.0, false), None);
        let s = e.step();
        assert!(s.open_orders.is_empty());
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn close_flattens_the_position() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(2.0);
        let s1 = e.step(); // long 2 @ 200
        assert_eq!(s1.position, 2.0);
        e.close();
        let s2 = e.step(); // close fills at bar 2 open (300)
        assert_eq!(s2.position, 0.0);
        // spot: bought 2 @ 200 (spent 400), sold 2 @ 300 (got 600), +200
        assert_eq!(s2.equity, 10_200.0);
    }

    #[test]
    fn a_replacement_keeps_the_side_it_was_placed_on() {
        // the KIND half of ADR 0010 is pinned, by re-resting a stop as a limit and
        // watching the unwrap fail. The SIDE half is asserted nowhere, and flipping
        // it in either branch left every test green: the one fixture that reaches
        // this replaces a stop that is reduce_only against a FLAT position, so the
        // flipped order triggers, gets zeroed by the clamp, and rests on with the
        // id and trigger the assertions read. Here the replacement has to FILL, and
        // a buy and a sell move the position opposite ways
        let candles = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000_000.0),
            ohlc(100.0, 106.0, 99.0, 100.0, 1_000_000.0),
        ]);
        let mut e = Engine::new(candles, cfg());
        e.reset();
        e.market_buy(10.0);
        e.step();
        assert_eq!(e.state().position, 10.0);

        let id = e.limit_sell(5.0, 105.0).expect("the book has room");
        e.replace(id, None, Some(102.0), None)
            .expect("moving a resting limit is not refused here");
        let s = e.step(); // the bar reaches 106, so a sell at 102 fills
        assert_eq!(
            s.position, 5.0,
            "the replacement filled on the wrong side: 15 means it became a buy"
        );
    }

    #[test]
    fn a_replaced_stop_keeps_the_side_it_was_placed_on() {
        // the stop branch is a second copy of the same rule and the limit test
        // above cannot reach it. NOT reduce_only, deliberately: that flag is what
        // let the existing fixture survive a flipped side, because the wrong-way
        // order was zeroed by the clamp before it could move anything, so nothing
        // downstream could tell it apart from an order that never fired
        let candles = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000_000.0),
            ohlc(100.0, 101.0, 94.0, 95.0, 1_000_000.0),
        ]);
        let mut e = Engine::new(candles, cfg());
        e.reset();
        e.market_buy(10.0);
        e.step();
        assert_eq!(e.state().position, 10.0);

        let id = e
            .stop(Side::Sell, 5.0, 95.0, false)
            .expect("the book has room");
        e.replace(id, None, None, Some(98.0))
            .expect("moving a resting stop is not refused here");
        // the bar dips through 98, so a sell stop triggers; a buy stop at 98 would
        // trigger on the same bar too, since its high is above it, and would buy
        let s = e.step();
        assert_eq!(
            s.position, 5.0,
            "the replaced stop fired on the wrong side: 15 means it became a buy"
        );
    }

    #[test]
    fn cancel_all_drops_every_resting_order() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.limit_buy(1.0, 50.0);
        e.limit_buy(1.0, 40.0);
        assert_eq!(e.cancel_all(), 2);
        let s = e.step();
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn qty_helpers_size_from_weight_and_quote() {
        let mut e = Engine::new(series(), cfg());
        e.reset(); // bar 0 close 150, equity 10000, flat
        assert!((e.qty_from_weight(0.5) - (5_000.0 / 150.0)).abs() < 1e-9);
        assert!((e.qty_from_quote(300.0) - 2.0).abs() < 1e-9); // 300 / 150
    }

    #[test]
    fn the_order_primitive_dispatches_by_kind() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        // a limit with no price is rejected
        assert!(e
            .order(
                Side::Buy,
                1.0,
                OrderType::Limit,
                None,
                None,
                false,
                false,
                TimeInForce::Gtc
            )
            .is_none());
        let lim = e.order(
            Side::Buy,
            1.0,
            OrderType::Limit,
            Some(50.0),
            None,
            false,
            false,
            TimeInForce::Gtc,
        );
        let stp = e.order(
            Side::Sell,
            1.0,
            OrderType::Stop,
            None,
            Some(40.0),
            false,
            false,
            TimeInForce::Gtc,
        );
        assert!(lim.is_some() && stp.is_some());
        let s = e.step();
        assert_eq!(s.open_orders.len(), 2); // the GTC limit and the resting stop
    }

    #[test]
    fn untouched_limit_stays_resting_and_can_be_canceled() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        let id = e.limit_buy(1.0, 50.0).unwrap(); // never reached
        let s1 = e.step();
        assert_eq!(s1.position, 0.0);
        assert_eq!(s1.open_orders.len(), 1); // still resting
        assert_eq!(s1.open_orders[0].id, id);
        assert!(e.cancel(id));
        let s2 = e.step();
        assert!(s2.open_orders.is_empty());
    }

    #[test]
    fn post_only_limit_that_would_cross_is_rejected() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset(); // bar 0 close 100
        assert!(e
            .place_limit(Side::Buy, 1.0, 105.0, false, true, TimeInForce::Gtc)
            .is_none()); // post_only that would cross
        assert!(e
            .place_limit(Side::Buy, 1.0, 95.0, false, true, TimeInForce::Gtc)
            .is_some()); // maker, accepted
    }

    #[test]
    fn replace_moves_one_order_and_refuses_to_arm_a_second() {
        // the trailing-stop trap: re-placing with stop() rests a new order every
        // bar, and the one that fills leaves its siblings live (ADR 0010)
        let mut e = Engine::new(dip_series(), perp_cfg());
        e.reset();
        let first = e.stop(Side::Sell, 1.0, 90.0, true).unwrap();
        let second = e.replace(first, None, None, Some(92.0)).unwrap();
        assert_eq!(second, first, "a replace keeps the order's id");
        let s = e.step();
        assert_eq!(s.open_orders.len(), 1, "replace must not leave two live");
        assert_eq!(s.open_orders[0].id, second);
        assert_eq!(s.open_orders[0].trigger.unwrap().get(), 92.0);
        assert!(s.open_orders[0].reduce_only, "flags carry over");

        // once the order is gone, replacing places nothing at all
        assert!(e.cancel(second));
        assert_eq!(e.replace(second, None, None, Some(95.0)), None);
        assert!(e.step().open_orders.is_empty());
    }

    #[test]
    fn close_flattens_a_short_and_records_the_trade() {
        // every other close test runs on a long, leaving the short arm of close()
        // and the pos < 0 half of the reduce_only predicate unexecuted
        let mut config = perp_cfg(); // quote 100, no cap
        config.report = true;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_sell(1.0);
        let s1 = e.step(); // short 1 @ bar 1 open 200
        assert_eq!(s1.position, -1.0);
        e.close();
        let s2 = e.step(); // buys back at bar 2 open 300
        assert_eq!(s2.position, 0.0);
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].side, Side::Sell);
        assert_eq!(trades[0].realized, -100.0); // short 1 from 200 bought back at 300
        let stats = e.stats(365.0, 0.0).unwrap();
        assert_eq!(stats.win_rate, 0.0);
        assert_eq!(stats.profit_factor, 0.0);
    }

    #[test]
    fn order_ids_stay_unique_across_a_reset() {
        // an id handed out in one episode must not name a live order in the next,
        // or a handle carried across a reset cancels somebody else's order (ADR 0010)
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        let first = e.limit_buy(1.0, 50.0).unwrap();
        e.reset();
        let second = e.limit_buy(1.0, 50.0).unwrap();
        assert_ne!(first, second);
        assert!(!e.cancel(first)); // the stale handle matches nothing
        assert!(e.cancel(second));
    }

    #[test]
    fn a_reduce_only_stop_cannot_open_the_other_side() {
        // the shortcut carries the flag now, so a protective stop that outlives the
        // one that closed the position fills nothing instead of opening a short
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        assert!(e.stop(Side::Sell, 1.0, 95.0, true).is_some());
        let s = e.step(); // bar 1 low 80 crosses 95, but there is nothing to reduce
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_non_finite_limit_price_or_stop_trigger_is_refused_a_slot() {
        // every touch test against NaN is false, so such an order never fills and
        // never expires; it would hold a slot until the book was full (ADR 0007)
        let mut e = Engine::new(series(), cfg());
        e.reset();
        assert_eq!(e.limit_buy(1.0, f64::NAN), None);
        assert_eq!(e.limit_sell(1.0, f64::INFINITY), None);
        assert_eq!(e.stop(Side::Sell, 1.0, f64::NAN, false), None);
        let s = e.step();
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn splitting_an_order_cannot_take_more_than_its_slots_of_a_bar() {
        // the volume cap is per order (ADR 0013) and the pending queue was
        // unbounded, so a hundred small market buys took the whole bar while every
        // slice was priced as though it were the only participant (ADR 0010)
        let config = EngineConfig {
            max_open_orders: 3,
            max_fill_fraction: 0.01, // 10 base units per order, against volume 1000
            quote: 1_000_000.0,
            ..cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        let placed = (0..100).filter(|_| e.market_buy(50.0).is_some()).count();
        assert_eq!(placed, 3);
        let s = e.step();
        assert_eq!(s.position, 30.0); // three slots at the 10-unit cap, not a hundred

        // the queue drains with the bar, so the next bar gets its slots back
        assert!(e.market_buy(50.0).is_some());
    }

    #[test]
    fn a_refused_replacement_leaves_the_original_order_resting() {
        // replace cancels before it places, so a replacement the book refuses left
        // nothing behind and still answered None, which this method defines as
        // "nothing happened". A trail reads that as "it already filled" and stops
        // arming one, so the protective stop was silently gone (ADR 0010)
        let mut e = Engine::new(series(), perp_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step();
        let id = e.stop(Side::Sell, 1.0, 90.0, true).expect("stop rests");
        for refused in [
            e.replace(id, None, None, Some(f64::NAN)),
            e.replace(id, Some(0.0), None, None),
        ] {
            assert!(refused.is_none());
        }
        let resting = e.current_state().open_orders;
        assert_eq!(resting.len(), 1);
        assert_eq!(resting[0].id, id);
        assert_eq!(
            resting[0].trigger.expect("a stop keeps its trigger").get(),
            90.0
        );
    }

    #[test]
    fn a_refused_replacement_leaves_the_original_where_it_was_in_the_queue() {
        // ADR 0010 defines a None from `replace` as "nothing happened", and ADR 0010
        // restores the original so that is true of the book too. It was not true of
        // the order's PRIORITY. The restore goes through `place`, which takes the
        // FIRST free slot (ADR 0011), so an order whose neighbour had been cancelled
        // was silently promoted up the queue by a replacement that was refused.
        //
        // Resolution follows slot order, and slot order decides who gets the cash
        // when the spot clamp binds, so the promotion moves a number. Sixty of cash
        // against a 40 and a 50: the one resolved first fills whole and the other
        // takes what is left.
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 35.0, 100.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 60.0,
            ..cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        let first = e.limit_buy(1.0, 10.0).expect("rests in the first slot");
        let kept = e.limit_buy(1.0, 50.0).expect("rests in the second");
        e.step();

        // the first slot is free now, and a refused replacement must not take it
        assert!(e.cancel(first));
        assert_eq!(e.replace(kept, Some(-1.0), None, None), None);
        e.limit_buy(1.0, 40.0)
            .expect("takes the slot the cancel freed");
        e.step();

        // the 40 resolves first and fills whole, leaving 20 of the 60, so the 50
        // fills 0.4 of its 1.0. Promote the 50 and it fills first for the whole 50,
        // leaving 10 against the 40, which is 1.25
        let state = e.current_state();
        assert!(
            (state.position - 1.4).abs() < 1e-9,
            "position {} says the restored order jumped the queue",
            state.position
        );
    }

    #[test]
    fn a_market_size_that_cannot_fill_is_refused_a_queue_slot() {
        // the queue became a bounded resource with ADR 0010 and did not inherit
        // ADR 0007's guard, so two sizes that can never fill refused a real order.
        // qty_from_weight returns exactly zero on a flat mark, which is how a
        // strategy places one without meaning to
        let mut e = Engine::new(
            series(),
            EngineConfig {
                max_open_orders: 2,
                ..cfg()
            },
        );
        e.reset();
        assert!(e.market_buy(f64::NAN).is_none());
        assert!(e.market_buy(-1.0).is_none());
        assert!(e.market_buy(0.0).is_none());
        assert!(e.market_buy(1.0).is_some());
        assert_eq!(e.step().position, 1.0);
    }

    #[test]
    fn a_refused_placement_takes_an_id_and_the_log_says_why() {
        let config = EngineConfig {
            max_open_orders: 1,
            ..report_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        let rested = e.limit_buy(1.0, 90.0).expect("rests");
        assert_eq!(e.limit_buy(1.0, 80.0), None);
        assert_eq!(e.market_buy(-1.0), None);
        let next = e.market_buy(1.0).expect("queued");
        assert_eq!((rested.0, next.0), (1, 4));
        assert_eq!(
            events(&e),
            vec![
                (1, EventKind::Placed),
                (2, EventKind::Refused),
                (3, EventKind::Refused),
                (4, EventKind::Placed),
            ]
        );
        assert_eq!(log(&e)[1].reason, EventReason::Refused(Refusal::Resting(1)));
        assert_eq!(log(&e)[2].reason, EventReason::Refused(Refusal::Size(-1.0)));
    }

    #[test]
    fn the_book_keeps_a_replaced_orders_id_and_hands_it_the_first_free_slot() {
        // the Specification's example of the book: a buy limit at 95 takes the first
        // slot and one at 94 the second; cancelling the first frees its slot, a
        // replace moving the 94 to 92 keeps its id and takes that slot, and a buy
        // limit at 93 placed next takes the second
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        let first = e.limit_buy(1.0, 95.0).expect("rests");
        let second = e.limit_buy(1.0, 94.0).expect("rests");
        assert!(e.cancel(first));
        assert_eq!(e.replace(second, None, Some(92.0), None), Some(second));
        let third = e.limit_buy(1.0, 93.0).expect("rests");
        let book: Vec<(OrderId, f64)> = e
            .current_state()
            .open_orders
            .iter()
            .map(|order| (order.id, order.price.expect("a limit").get()))
            .collect();
        assert_eq!(book, vec![(second, 92.0), (third, 93.0)]);
        let reasons: Vec<EventReason> = log(&e).iter().map(|event| event.reason).collect();
        assert_eq!(
            reasons,
            vec![
                EventReason::Limit,
                EventReason::Limit,
                EventReason::CancelledByAccount,
                EventReason::Replace,
                EventReason::Limit,
            ]
        );
    }

    #[test]
    fn an_ioc_limit_its_bar_does_not_reach_expires_and_a_partial_fok_fills_nothing() {
        // the same example: an ioc buy limit at 96 that its first bar does not reach
        // expires after that bar, and a fok market buy the spot wallet's cash pays
        // for only in part fills nothing
        let config = EngineConfig {
            quote: 1_000.0,
            ..report_cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        let ioc = e
            .order(
                Side::Buy,
                1.0,
                OrderType::Limit,
                Some(96.0),
                None,
                false,
                false,
                TimeInForce::Ioc,
            )
            .expect("rests");
        let fok = e
            .order(
                Side::Buy,
                10.0,
                OrderType::Market,
                None,
                None,
                false,
                false,
                TimeInForce::Fok,
            )
            .expect("queued");
        let s = e.step(); // opens at 200 and trades down to 190: 10 would cost 2,000
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty());
        assert_eq!(
            events(&e),
            vec![
                (ioc.0, EventKind::Placed),
                (fok.0, EventKind::Placed),
                (fok.0, EventKind::NotFilled),
                (ioc.0, EventKind::Expired),
            ]
        );
    }

    #[test]
    fn a_liquidation_logs_every_order_it_cancels() {
        let config = EngineConfig {
            report: true,
            ..levered()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        let entry = e.market_buy(10.0).expect("queued");
        let target = e.limit_sell(1.0, 120.0).expect("rests");
        e.step(); // the long dies at 90 on the way to 80
        assert_eq!(
            events(&e),
            vec![
                (entry.0, EventKind::Placed),
                (target.0, EventKind::Placed),
                (entry.0, EventKind::Filled),
                (target.0, EventKind::Cancelled),
            ]
        );
        assert_eq!(log(&e)[3].reason, EventReason::Liquidation);
    }

    #[test]
    fn a_replace_that_keeps_the_price_is_no_new_placement_for_post_only() {
        // a post_only bid at 99 rests below a close of 100, and the next bar, which
        // trades nothing, closes at 98, above which the bid now sits. Changing its
        // size leaves it where it entered the book; moving it to 98.5 is a new
        // placement that would take liquidity at that close
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 98.0, 98.0, 0.0),
            ohlc(98.0, 98.0, 98.0, 98.0, 1e6),
        ]);
        let mut e = Engine::new(bars, report_cfg());
        e.reset();
        let id = e
            .order(
                Side::Buy,
                1.0,
                OrderType::Limit,
                Some(99.0),
                None,
                false,
                true,
                TimeInForce::Gtc,
            )
            .expect("rests below the close");
        e.step();
        assert_eq!(e.replace(id, Some(2.0), None, None), Some(id));
        assert_eq!(e.replace(id, None, Some(98.5), None), None);
        assert_eq!(
            log(&e).last().map(|event| event.reason),
            Some(EventReason::Replacing(Refusal::PostOnly))
        );
    }
}
