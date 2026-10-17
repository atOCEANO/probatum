//! Placing, moving and cancelling orders. A market order is queued for its
//! instrument's next open and a limit or a stop rests in the book until a bar
//! reaches it, and `resolution` fills both. A size, price or trigger that could
//! never fill is refused here rather than given a slot, and so is an order on an
//! instrument that has not printed its first bar, and the order log says why (ADRs
//! 0007, 0010, 0022).

use probatum_core::{Order, OrderId, OrderType, Price, Qty, Side, TimeInForce};

use super::{Engine, CLOSE_EPS};
use crate::execution::positive;
use crate::reporter::{EventKind, EventReason, OrderEvent, Refusal};

impl Engine {
    pub(super) fn next_id(&mut self) -> OrderId {
        let id = OrderId(self.id_counter);
        self.id_counter += 1;
        id
    }

    /// Add an event to the order log when reporting is on.
    pub(super) fn log_event(
        &mut self,
        kind: EventKind,
        order: Option<OrderId>,
        instrument: Option<usize>,
        size: Option<f64>,
        price: Option<f64>,
        reason: EventReason,
    ) {
        if let Some(reporter) = self.reporter.as_mut() {
            reporter.record_event(OrderEvent {
                tick: self.tick,
                kind,
                order,
                instrument,
                size,
                price,
                reason,
            });
        }
    }

    /// Log `id` on `instrument` refused for `why`, and answer the placement with
    /// nothing.
    fn refuse(&mut self, id: OrderId, instrument: usize, why: Refusal) -> Option<OrderId> {
        self.log_event(
            EventKind::Refused,
            Some(id),
            Some(instrument),
            None,
            None,
            EventReason::Refused(why),
        );
        None
    }

    /// `order`'s price and trigger on its instrument's grid, where the venue's sizes
    /// are on (Specification, Sizes and dust).
    fn on_grid(&self, order: &mut Order) {
        let rule = self.candles.sizes(order.instrument).prices;
        order.price = order.price.map(|price| Price(rule.round(price.get())));
        order.trigger = order
            .trigger
            .map(|trigger| Price(rule.round(trigger.get())));
    }

    /// Hold `order` to its instrument's sizes as it is placed: its size rounded down
    /// to the lot and held to the maximum order quantity rounded down to the lot, the
    /// hold logged as a clamp, and why it is refused where it rounds to no lot, its
    /// notional at the last close falls short of the minimum, which a reduce-only
    /// order escapes where the venue spares it, or a replace held to the maximum
    /// leaves less than a lot to fill (Specification, Sizes and dust).
    fn sized(&mut self, order: &mut Order) -> Option<Refusal> {
        let i = order.instrument;
        let sizes = self.candles.sizes(i);
        let size = order.size.get();
        let rounded = sizes.down(size);
        // a lot past every float, or so small a size over it is, rounds to no number
        if rounded <= 0.0 || !rounded.is_finite() {
            return Some(Refusal::Lot {
                size,
                lot: sizes.lot,
            });
        }
        let most = sizes.down(sizes.max_qty);
        let held = if rounded > most {
            let asked = EventReason::HeldTo { asked: rounded };
            let id = order.id;
            self.log_event(
                EventKind::Clamped,
                Some(id),
                Some(i),
                Some(most),
                None,
                asked,
            );
            most
        } else {
            rounded
        };
        order.size = Qty(held);
        let exempt = order.reduce_only && sizes.reduce_only_exempt;
        if (held * self.marks[i]).abs() < sizes.min_notional && !exempt {
            return Some(Refusal::MinNotional {
                size: held,
                least: sizes.min_notional,
            });
        }
        let left = order.remaining().get();
        if sizes.spent(left, CLOSE_EPS) {
            return Some(Refusal::Size(left));
        }
        None
    }

    pub(super) fn place_market(
        &mut self,
        instrument: usize,
        side: Side,
        size: f64,
        reduce_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        self.queue(
            instrument,
            side,
            size,
            reduce_only,
            tif,
            EventReason::Market,
        )
    }

    /// Queue a market order for its instrument's next open, its placement logged
    /// for `reason`, or log why it cannot wait there.
    #[inline(always)]
    pub(super) fn queue(
        &mut self,
        instrument: usize,
        side: Side,
        size: f64,
        reduce_only: bool,
        tif: TimeInForce,
        reason: EventReason,
    ) -> Option<OrderId> {
        // Every placement takes an id, a refused one included, so the order log can
        // name what it refused (ADR 0010).
        let id = self.next_id();
        // an instrument enters at its first bar, and before it there is no price to
        // fill or check an order against
        if self.marks[instrument].is_nan() {
            return self.refuse(id, instrument, Refusal::Unlisted);
        }
        // The same guard the resting book has carried since ADR 0007, extended to
        // the queue on the day ADR 0010 made the queue a bounded resource too. A
        // size that can never fill used to take a slot anyway, so two of them ahead
        // of a real order refused it: `qty_from_weight` returns exactly zero on a
        // flat mark, which is how a strategy places one without meaning to.
        if !positive(size) {
            return self.refuse(id, instrument, Refusal::Size(size));
        }
        let size = if self.config.sizes {
            let mut sized = Order::market(id, side, Qty(size), reduce_only).on(instrument);
            if let Some(why) = self.sized(&mut sized) {
                return self.refuse(id, instrument, why);
            }
            sized.size.get()
        } else {
            size
        };
        // The queue holds the same number of orders the resting book does, for each
        // instrument (ADR 0010), so a strategy that places without bound cannot make
        // one bar's work unbounded
        let waiting = self
            .pending
            .iter()
            .filter(|order| order.instrument == instrument)
            .count();
        if waiting >= self.config.max_open_orders {
            return self.refuse(
                id,
                instrument,
                Refusal::Waiting(self.config.max_open_orders),
            );
        }
        let mut order = Order::market(id, side, Qty(size), reduce_only).on(instrument);
        order.tif = tif;
        self.pending.push(order);
        self.log_event(
            EventKind::Placed,
            Some(id),
            Some(instrument),
            None,
            None,
            reason,
        );
        Some(id)
    }

    /// Why a limit or a stop could not rest as `order`, or `None`: an instrument
    /// that has not printed its first bar, a size, price or trigger that can never
    /// fill, a full share of the book, a limit that would trade against the
    /// account's own, or a post_only limit that would cross. `replacing` is the
    /// order a replace moves, whose slot has just been given up (ADRs 0007, 0010,
    /// 0011).
    fn resting_refusal(&mut self, order: &mut Order, replacing: Option<&Order>) -> Option<Refusal> {
        let last = self.marks[order.instrument];
        if last.is_nan() {
            return Some(Refusal::Unlisted);
        }
        // A size that is not a positive finite number can never fill (ADR 0014), so
        // it is refused a slot rather than resting inert forever. A non-finite price
        // is refused for the same reason: every touch test against it compares with
        // NaN and is false, so the order would camp on a slot until the book filled
        // up and later placements were silently rejected (ADR 0007).
        let size = order.remaining().get();
        let sizeless = order.leg.is_some_and(|leg| leg.sizeless);
        if !sizeless && !positive(size) {
            return Some(Refusal::Size(size));
        }
        if self.config.sizes && !sizeless {
            if let Some(why) = self.sized(order) {
                return Some(why);
            }
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
        let resting = self
            .book
            .iter()
            .filter(|other| other.instrument == order.instrument)
            .count();
        if replacing.is_none() && resting >= self.config.max_open_orders {
            return Some(Refusal::Resting(self.config.max_open_orders));
        }
        if let Some(own) = self.crossed(order) {
            return Some(Refusal::SelfTrade(own));
        }
        // Crossing is judged against the instrument's last close, the last price the
        // strategy actually saw. The next bar's open is what this order would really
        // cross, and reading it here would be the lookahead the engine refuses
        // everywhere else
        let crosses = order.price.is_some_and(|price| {
            self.fill_model
                .limit_crosses(order.side, price, Price(last))
        });
        // post_only is a rule about how an order enters the book, so a replace that
        // leaves the price where it was is not checked again
        let enters = replacing.is_none_or(|old| old.price != order.price);
        if order.post_only && crosses && enters {
            return Some(Refusal::PostOnly);
        }
        None
    }

    /// The account's own resting limit of the other side on `order`'s instrument that a
    /// limit at `order`'s price reaches, a sell at or below it for a buy and a buy at
    /// or above it for a sell, which a venue refuses to trade against (ADR 0011). A
    /// bracket leg yet to arm has no side, so it is neither checked nor met.
    pub(super) fn crossed(&self, order: &Order) -> Option<OrderId> {
        let sided = |order: &Order| order.leg.is_none_or(|leg| leg.armed);
        let price = order.price.filter(|_| sided(order))?;
        self.book
            .iter()
            .find(|own| {
                own.instrument == order.instrument
                    && sided(own)
                    && own.side != order.side
                    && own
                        .price
                        .is_some_and(|at| self.fill_model.limit_crosses(order.side, price, at))
            })
            .map(|own| own.id)
    }

    /// Rest `order` in the first free slot, its price and trigger on its instrument's
    /// grid where the venue's sizes are on, or log why it cannot rest.
    pub(super) fn rest(&mut self, mut order: Order, reason: EventReason) -> Option<OrderId> {
        if self.config.sizes {
            self.on_grid(&mut order);
        }
        if let Some(why) = self.resting_refusal(&mut order, None) {
            return self.refuse(order.id, order.instrument, why);
        }
        let id = self.book.place(order).ok()?;
        self.log_event(
            EventKind::Placed,
            Some(id),
            Some(order.instrument),
            None,
            None,
            reason,
        );
        Some(id)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn place_limit(
        &mut self,
        instrument: usize,
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
        self.rest(order.on(instrument), EventReason::Limit)
    }

    fn place_stop(
        &mut self,
        instrument: usize,
        side: Side,
        size: f64,
        trigger: f64,
        reduce_only: bool,
    ) -> Option<OrderId> {
        let id = self.next_id();
        let order = Order::stop(id, side, Qty(size), Price(trigger), reduce_only);
        self.rest(order.on(instrument), EventReason::Stop)
    }

    /// The one order primitive the typed shortcuts wrap, on the first instrument.
    /// Dispatches on `kind`: a market ignores `price` and `trigger`, a limit needs a
    /// `price`, a stop needs a `trigger`. `tif` applies to market and limit orders;
    /// a stop rests until it triggers (ADR 0010). `None` if the book is full, the
    /// size is not a positive finite number, a post_only limit would cross, a limit
    /// would trade against the account's own, or a limit has no price or a stop no
    /// trigger.
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
        self.order_on(
            0,
            side,
            size,
            kind,
            price,
            trigger,
            reduce_only,
            post_only,
            tif,
        )
    }

    /// `order` on `instrument`, which also refuses an instrument that has not
    /// printed its first bar.
    #[allow(clippy::too_many_arguments)]
    pub fn order_on(
        &mut self,
        instrument: usize,
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
            OrderType::Market => self.place_market(instrument, side, size, reduce_only, tif),
            OrderType::Limit => {
                self.place_limit(instrument, side, size, price?, reduce_only, post_only, tif)
            }
            OrderType::Stop => self.place_stop(instrument, side, size, trigger?, reduce_only),
        }
    }

    /// Queue a market order on `instrument` for `amount` of its quote currency, whose
    /// size is the amount over its fill price, so it spends or raises that amount
    /// before its fee (Specification, Units). Refused as a market order is, for an
    /// amount that is not a positive finite number, and where the venue's sizes are on
    /// for one below the minimum notional, a reduce-only one spared where the venue
    /// spares it.
    pub fn order_value(
        &mut self,
        instrument: usize,
        side: Side,
        amount: f64,
        reduce_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        let id = self.next_id();
        if self.marks[instrument].is_nan() {
            return self.refuse(id, instrument, Refusal::Unlisted);
        }
        if !positive(amount) {
            return self.refuse(id, instrument, Refusal::Amount(amount));
        }
        if self.config.sizes {
            let sizes = self.candles.sizes(instrument);
            let least = sizes.min_notional;
            if amount < least && !(reduce_only && sizes.reduce_only_exempt) {
                return self.refuse(id, instrument, Refusal::MinAmount { amount, least });
            }
        }
        let waiting = self
            .pending
            .iter()
            .filter(|order| order.instrument == instrument)
            .count();
        if waiting >= self.config.max_open_orders {
            return self.refuse(
                id,
                instrument,
                Refusal::Waiting(self.config.max_open_orders),
            );
        }
        // the size is set as it fills, and what the queue holds in its place fills
        // nothing by itself
        let mut order = Order::market(id, side, Qty(f64::INFINITY), reduce_only).on(instrument);
        order.tif = tif;
        self.pending.push(order);
        self.values.push((id, amount));
        self.log_event(
            EventKind::Placed,
            Some(id),
            Some(instrument),
            None,
            None,
            EventReason::Market,
        );
        Some(id)
    }

    /// Queue a market buy; it fills on the next bar's open. `None` if the queue is
    /// already holding `max_open_orders` for this bar, or the size is not a
    /// positive finite number.
    pub fn market_buy(&mut self, size: f64) -> Option<OrderId> {
        self.place_market(0, Side::Buy, size, false, TimeInForce::Ioc)
    }

    /// Queue a market sell; it fills on the next bar's open. `None` if the queue is
    /// already holding `max_open_orders` for this bar, or the size is not a
    /// positive finite number.
    pub fn market_sell(&mut self, size: f64) -> Option<OrderId> {
        self.place_market(0, Side::Sell, size, false, TimeInForce::Ioc)
    }

    /// Rest a buy limit; it fills when a later bar trades down to `price`. `None`
    /// if the book is full, the size is not a positive finite number, `price` is
    /// not finite, or the account rests a sell at or below it (ADR 0011).
    pub fn limit_buy(&mut self, size: f64, price: f64) -> Option<OrderId> {
        self.place_limit(0, Side::Buy, size, price, false, false, TimeInForce::Gtc)
    }

    /// Rest a sell limit; it fills when a later bar trades up to `price`. `None`
    /// if the book is full, the size is not a positive finite number, `price` is
    /// not finite, or the account rests a buy at or above it (ADR 0011).
    pub fn limit_sell(&mut self, size: f64, price: f64) -> Option<OrderId> {
        self.place_limit(0, Side::Sell, size, price, false, false, TimeInForce::Gtc)
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
        self.place_stop(0, side, size, trigger, reduce_only)
    }

    /// Base size for a fraction of current equity, marked at the first instrument's
    /// current close.
    pub fn qty_from_weight(&self, fraction: f64) -> f64 {
        self.account
            .qty_from_weight(fraction, Price(self.current_close()), &self.marks)
            .get()
    }

    /// Base size for a cash amount in quote, at the first instrument's current
    /// close.
    pub fn qty_from_quote(&self, cash: f64) -> f64 {
        self.account
            .qty_from_quote(cash, Price(self.current_close()))
            .get()
    }

    /// Cancel a resting order by id. True if it was found and removed.
    pub fn cancel(&mut self, id: OrderId) -> bool {
        let Some(order) = self.book.cancel(id) else {
            return false;
        };
        self.log_event(
            EventKind::Cancelled,
            Some(id),
            Some(order.instrument),
            None,
            None,
            EventReason::CancelledByAccount,
        );
        true
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
            // a bracket leg given a size closes that much rather than the position
            if let Some(leg) = moved.leg.as_mut() {
                leg.sizeless = false;
            }
        }
        if moved.kind == OrderType::Limit {
            moved.price = price.map(Price).or(old.price);
        } else {
            moved.trigger = trigger.map(Price).or(old.trigger);
        }
        // the old price and trigger sit on the grid already, which leaves them there
        if self.config.sizes {
            self.on_grid(&mut moved);
        }
        // a limit moved to another price enters the book again, for maker or taker
        // and for post_only (ADR 0011)
        if moved.price != old.price {
            moved.on_book = false;
        }
        if let Some(why) = self.resting_refusal(&mut moved, Some(&old)) {
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
                Some(old.instrument),
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
            Some(old.instrument),
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

    /// Cancel every order resting on `instrument`, returning how many were dropped.
    pub fn cancel_all_on(&mut self, instrument: usize) -> usize {
        let ids: Vec<OrderId> = self
            .book
            .iter()
            .filter(|order| order.instrument == instrument)
            .map(|order| order.id)
            .collect();
        for &id in &ids {
            self.cancel(id);
        }
        ids.len()
    }

    /// The resting order placed first, or `None` with nothing resting. A replace
    /// keeps an order's id, and so its age.
    pub fn oldest_resting(&self) -> Option<OrderId> {
        self.book.iter().map(|order| order.id).min_by_key(|id| id.0)
    }

    /// The orders resting on the book.
    pub fn resting(&self) -> impl Iterator<Item = &Order> {
        self.book.iter()
    }

    /// The market orders waiting for the next open, each beside its amount of the
    /// quote currency where it is a value order, whose fill sets its size.
    pub fn waiting(&self) -> impl Iterator<Item = (&Order, Option<f64>)> {
        self.pending.iter().map(|order| {
            let value = self.values.iter().find(|(id, _)| *id == order.id);
            (order, value.map(|&(_, amount)| amount))
        })
    }

    /// Flatten the first instrument's position with a reduce-only market order.
    /// `None` when flat.
    pub fn close(&mut self) -> Option<OrderId> {
        self.close_on(0)
    }

    /// Flatten `instrument`'s position with a reduce-only market order. `None` when
    /// flat.
    pub fn close_on(&mut self, instrument: usize) -> Option<OrderId> {
        let pos = self.account.positions[instrument].qty.get();
        if pos == 0.0 {
            return None;
        }
        let (side, size) = if pos > 0.0 {
            (Side::Sell, pos)
        } else {
            (Side::Buy, -pos)
        };
        self.place_market(instrument, side, size, true, TimeInForce::Ioc)
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{
        cfg, crash_series, dip_series, levered, ohlc, perp_cfg, report_cfg, series, universe,
        ABSENT,
    };
    use crate::engine::{Engine, EngineConfig};
    use crate::reporter::{EventKind, EventReason, OrderEvent, Refusal};
    use crate::sizes::{PriceRule, Sizes};
    use probatum_core::{Market, OrderId, OrderType, Side, TimeInForce};

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
        // a NaN size must fill nothing and pay nothing, rather than reach the fill as
        // all the budget allows or as a NaN (ADR 0014)
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
            .place_limit(0, Side::Buy, 1.0, 105.0, false, true, TimeInForce::Gtc)
            .is_none()); // post_only that would cross
        assert!(e
            .place_limit(0, Side::Buy, 1.0, 95.0, false, true, TimeInForce::Gtc)
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
    fn splitting_an_order_cannot_take_more_than_the_bars_budget() {
        // the budget is shared by every order on the bar, so a hundred small market
        // buys take no more of it than one large one would (ADR 0013), and the
        // pending queue is bounded as the book is (ADR 0010)
        let config = EngineConfig {
            max_open_orders: 3,
            budget: 0.01, // 10 base units of a trailing volume of 1000, for every order
            quote: 1_000_000.0,
            ..cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        let placed = (0..100).filter(|_| e.market_buy(50.0).is_some()).count();
        assert_eq!(placed, 3);
        let s = e.step();
        assert_eq!(s.position, 10.0); // the first takes the budget, the others nothing

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

    #[test]
    fn a_replace_that_moves_a_limits_price_takes_it_off_the_book() {
        // a bid at 99 rests through a bar that does not reach it, so it is on the
        // book; moved to 99.5 it is a new placement and a taker on a next bar that
        // opens at 98.50, where one whose size alone changed stays a maker
        for (price, taker) in [(Some(99.5), true), (None, false)] {
            let bars = Candles::new(vec![
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(100.0, 100.5, 99.6, 100.0, 1e6),
                ohlc(98.5, 99.5, 98.0, 99.0, 1e6),
            ]);
            let mut e = Engine::new(bars, report_cfg());
            e.reset();
            let id = e.limit_buy(1.0, 99.0).expect("rests below the close");
            e.step();
            assert_eq!(e.replace(id, Some(2.0), price, None), Some(id));
            e.step();
            let fills = e.reporter().expect("reporting is on").fills();
            assert_eq!((fills.len(), fills[0].taker), (1, taker));
        }
    }

    #[test]
    fn a_limit_reaching_the_accounts_own_of_the_other_side_is_refused() {
        // the Specification's example: a buy at 101 placed while the account rests
        // a sell at 100.50 is refused, and so is a bid moved up to it, where a bid
        // below the sell, a buy stop above it and a bid at 101 on another instrument
        // all rest
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let rows = vec![vec![flat, flat], vec![flat, flat]];
        let mut e = Engine::new(universe(&[Market::Spot, Market::Spot], rows), report_cfg());
        e.reset();
        let ask = e.limit_sell(1.0, 100.5).expect("rests above the close");
        let refused = EventReason::Refused(Refusal::SelfTrade(ask));
        assert_eq!(bid(&mut e, 0, 101.0), None);
        assert_eq!(log(&e).last().map(|event| event.reason), Some(refused));
        let below = bid(&mut e, 0, 100.25).expect("rests below the sell");
        assert!(e.stop(Side::Buy, 1.0, 101.0, false).is_some());
        assert!(bid(&mut e, 1, 101.0).is_some());
        assert_eq!(e.replace(below, None, Some(101.0), None), None);
        assert_eq!(
            log(&e).last().map(|event| event.reason),
            Some(EventReason::Replacing(Refusal::SelfTrade(ask)))
        );
    }

    /// Two spot instruments, the second printing its first bar on the second row.
    fn listing_late() -> Engine {
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), ABSENT],
            vec![flat(100.0), flat(10.0)],
            vec![flat(100.0), flat(10.0)],
        ];
        let config = EngineConfig {
            max_open_orders: 1,
            ..report_cfg()
        };
        Engine::new(universe(&[Market::Spot, Market::Spot], rows), config)
    }

    fn bid(e: &mut Engine, instrument: usize, price: f64) -> Option<OrderId> {
        e.order_on(
            instrument,
            Side::Buy,
            1.0,
            OrderType::Limit,
            Some(price),
            None,
            false,
            false,
            TimeInForce::Gtc,
        )
    }

    #[test]
    fn an_order_on_an_instrument_before_its_first_bar_is_refused() {
        let mut e = listing_late();
        e.reset();
        assert_eq!(bid(&mut e, 1, 9.0), None);
        let refused = log(&e)[0];
        assert_eq!(refused.instrument, Some(1));
        assert_eq!(refused.reason, EventReason::Refused(Refusal::Unlisted));
        e.step();
        assert!(bid(&mut e, 1, 9.0).is_some());
    }

    #[test]
    fn max_open_orders_bounds_each_instrument_on_its_own() {
        // one order each: the first instrument's second bid is refused while the
        // second instrument still rests one of its own
        let mut e = listing_late();
        e.reset();
        e.step();
        assert!(bid(&mut e, 0, 90.0).is_some());
        assert!(bid(&mut e, 1, 9.0).is_some());
        assert_eq!(bid(&mut e, 0, 80.0), None);
        let refused = log(&e).last().copied().expect("an event");
        assert_eq!(refused.reason, EventReason::Refused(Refusal::Resting(1)));
    }

    #[test]
    fn a_slot_freed_on_one_instrument_goes_to_the_next_placement_on_any() {
        // one table for the account: the second instrument's bid at 10 takes the first
        // slot and the first instrument's bid at 50 the second; cancelling the 10 frees
        // the first slot, and a bid at 40 on the second instrument takes it, so on a bar
        // that dips through both it resolves first and the 60 of cash buys 1 at 40 and
        // then 0.4 at 50
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let dip = ohlc(100.0, 100.0, 35.0, 100.0, 1e6);
        let rows = vec![vec![flat, flat], vec![flat, flat], vec![dip, dip]];
        let config = EngineConfig {
            quote: 60.0,
            ..report_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Spot, Market::Spot], rows), config);
        e.reset();
        let placeholder = bid(&mut e, 1, 10.0).expect("takes the first slot");
        bid(&mut e, 0, 50.0).expect("takes the second");
        e.step();
        assert!(e.cancel(placeholder));
        bid(&mut e, 1, 40.0).expect("takes the slot the cancel freed");
        e.step();
        let fills: Vec<(usize, f64, f64)> = e
            .reporter()
            .expect("reporting is on")
            .fills()
            .iter()
            .map(|fill| (fill.instrument, fill.size, fill.price))
            .collect();
        assert_eq!(fills.len(), 2);
        assert_eq!((fills[0].0, fills[0].1, fills[0].2), (1, 1.0, 40.0));
        assert_eq!((fills[1].0, fills[1].2), (0, 50.0));
        assert!((fills[1].1 - 0.4).abs() < 1e-12);
    }

    /// An engine at a close of 150 whose instrument's sizes are a lot of 0.001, a tick of
    /// 0.01, a minimum notional of 10 and a maximum order quantity of 1, a reduce-only
    /// order spared the minimum where `exempt`.
    fn sized(exempt: bool) -> Engine {
        let sizes = Sizes {
            lot: 0.001,
            prices: PriceRule::Tick(0.01),
            min_notional: 10.0,
            max_qty: 1.0,
            reduce_only_exempt: exempt,
        };
        let config = EngineConfig {
            sizes: true,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(series().with_sizes(sizes), config);
        e.reset();
        e
    }

    #[test]
    fn with_sizes_on_an_order_is_held_to_the_lot_the_grid_and_the_largest_size() {
        let mut e = sized(false);
        assert!(e.market_buy(0.0004).is_none());
        let held = e
            .market_buy(5.0)
            .expect("held to the largest size and placed");
        let id = e
            .limit_buy(0.2004, 99.404)
            .expect("rests on the lot and the grid");
        let resting = e.book.get(id).expect("resting");
        assert!((resting.size.get() - 0.2).abs() < 1e-15);
        assert!((resting.price.expect("a limit").get() - 99.40).abs() < 1e-9);
        let log = log(&e);
        assert_eq!(
            log[0].reason,
            EventReason::Refused(Refusal::Lot {
                size: 0.0004,
                lot: 0.001
            })
        );
        let clamp = log
            .iter()
            .find(|event| event.order == Some(held) && event.kind == EventKind::Clamped)
            .expect("the hold is logged");
        assert_eq!(clamp.size, Some(1.0));
        assert_eq!(clamp.reason, EventReason::HeldTo { asked: 5.0 });
        assert_eq!(e.pending[0].size.get(), 1.0);
    }

    #[test]
    fn an_order_below_the_minimum_notional_is_refused_and_a_reduce_only_one_spared() {
        // 0.05 at the close of 150 is 7.5, below the minimum of 10
        for exempt in [false, true] {
            let mut e = sized(exempt);
            assert!(e.market_buy(0.05).is_none());
            let reduce = e.order(
                Side::Sell,
                0.05,
                OrderType::Market,
                None,
                None,
                true,
                false,
                TimeInForce::Ioc,
            );
            assert_eq!(reduce.is_some(), exempt, "exempt {exempt}");
            assert_eq!(
                log(&e)[0].reason,
                EventReason::Refused(Refusal::MinNotional {
                    size: 0.05,
                    least: 10.0
                })
            );
        }
    }

    #[test]
    fn a_maximum_off_the_lot_holds_an_order_to_the_lot_below_it() {
        // a maximum of 1.0005 under a lot of 0.001 holds a buy of 5 to 1.000
        let sizes = Sizes {
            lot: 0.001,
            max_qty: 1.0005,
            ..Sizes::default()
        };
        let config = EngineConfig {
            sizes: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(series().with_sizes(sizes), config);
        e.reset();
        e.market_buy(5.0).expect("held and placed");
        assert_eq!(e.pending[0].size.get(), 1.0);
    }

    #[test]
    fn a_replace_that_leaves_less_than_a_lot_to_fill_is_refused() {
        // a buy of 1.0 that a budget of 0.6 filled 0.6 of, replaced by 0.05 more, is
        // 0.65, which rounds to the 0.6 it filled under a lot of 0.1
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 0.6),
            ohlc(100.0, 100.0, 95.0, 100.0, 0.6),
        ])
        .with_sizes(Sizes {
            lot: 0.1,
            ..Sizes::default()
        });
        let config = EngineConfig {
            sizes: true,
            ..report_cfg()
        };
        let mut e = Engine::new(candles, config);
        e.reset();
        let id = e.limit_buy(1.0, 99.0).expect("rests");
        assert!((e.step().position - 0.6).abs() < 1e-12);
        assert_eq!(e.replace(id, Some(0.05), None, None), None);
        let refused = log(&e).last().expect("the refusal is logged").reason;
        assert_eq!(refused, EventReason::Replacing(Refusal::Size(0.0)));
    }

    #[test]
    fn a_value_order_fills_its_amount_over_its_fill_price() {
        // 1,000 at a next open of 200 buys 5, and with 1% of slippage the 4.95 that
        // 1,000 buys at 202
        for (bps, size) in [(0.0, 5.0), (100.0, 1_000.0 / 202.0)] {
            let config = EngineConfig {
                quote: 10_000.0,
                slippage_bps: bps,
                ..report_cfg()
            };
            let mut e = Engine::new(series(), config);
            e.reset();
            let id = e.order_value(0, Side::Buy, 1_000.0, false, TimeInForce::Ioc);
            e.step();
            let fill = e.reporter().expect("reporting is on").fills()[0];
            assert_eq!((fill.order, fill.size), (id, size));
        }
    }

    #[test]
    fn with_sizes_on_a_value_order_is_held_to_the_largest_size_and_the_lot() {
        // at a next open of 200, 100.05 buys the 0.500 of the 0.50025 it reaches that
        // the lot allows, 300 is held to the largest size of 1, 5 falls short of the
        // minimum notional of 10, and with sizes off nothing is refused but an amount
        // that is no positive number
        let mut e = sized(false);
        e.order_value(0, Side::Buy, 100.05, false, TimeInForce::Ioc)
            .expect("queued");
        e.order_value(0, Side::Buy, 300.0, false, TimeInForce::Ioc)
            .expect("queued");
        assert_eq!(
            e.order_value(0, Side::Buy, 5.0, false, TimeInForce::Ioc),
            None
        );
        let refused = log(&e).last().expect("the refusal is logged").reason;
        let short = Refusal::MinAmount {
            amount: 5.0,
            least: 10.0,
        };
        assert_eq!(refused, EventReason::Refused(short));
        e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let sizes: Vec<f64> = fills.iter().map(|fill| fill.size).collect();
        assert_eq!(sizes, [0.5, 1.0]);
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        assert_eq!(
            e.order_value(0, Side::Buy, 0.0, false, TimeInForce::Ioc),
            None
        );
        let refused = log(&e).last().expect("the refusal is logged").reason;
        assert_eq!(refused, EventReason::Refused(Refusal::Amount(0.0)));
    }

    #[test]
    fn a_value_order_is_refused_for_its_amount_first_and_spared_the_minimum_when_reduce_only() {
        // a negative amount is no amount, whatever the minimum notional, and a
        // reduce-only 5 is spared the minimum of 10 where the venue spares it
        let mut e = sized(false);
        assert_eq!(
            e.order_value(0, Side::Buy, -100.0, false, TimeInForce::Ioc),
            None
        );
        let refused = log(&e).last().expect("the refusal is logged").reason;
        assert_eq!(refused, EventReason::Refused(Refusal::Amount(-100.0)));
        assert_eq!(
            e.order_value(0, Side::Sell, 5.0, true, TimeInForce::Ioc),
            None
        );
        let mut e = sized(true);
        assert!(e
            .order_value(0, Side::Sell, 5.0, true, TimeInForce::Ioc)
            .is_some());
    }

    #[test]
    fn an_all_or_nothing_value_order_whose_amount_buys_no_lot_fills_nothing_for_that() {
        // 50 at a next open of 200 buys 0.25, less than a lot of 1, so the bar fills
        // nothing of an order that asks nothing, all or nothing or not
        let sizes = Sizes {
            lot: 1.0,
            prices: PriceRule::Tick(0.01),
            min_notional: 0.0,
            max_qty: 100.0,
            reduce_only_exempt: false,
        };
        let config = EngineConfig {
            sizes: true,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(series().with_sizes(sizes), config);
        e.reset();
        e.order_value(0, Side::Buy, 50.0, false, TimeInForce::Fok)
            .expect("queued");
        e.step();
        let last = log(&e).last().expect("the miss is logged").reason;
        assert_eq!(last, EventReason::NothingFilled);
    }
}
