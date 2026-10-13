//! Where the margin behind the perps runs out, on the path every perp held moves
//! along (Specification, Liquidation). The path carried into a bar is solved before
//! any of its orders resolves: a clip on the gap closed the account before the bar
//! opened, and one inside the bar clips every held perp's bar there, so no order
//! fills past the point the account left the market (ADR 0017). After the fills the
//! path is solved again for the account they left, and the closes are checked once
//! the row's funding is charged. A forced close takes the account over at the clip
//! point and leaves the margin wallet at exactly zero, and no reducing fill books
//! past the point the margin runs out, so an account can end at zero but never
//! owing (ADRs 0018, 0019).

use probatum_core::{Candle, Fill, Market, OrderId, Price, Qty, Side};

use super::ledger::Moment;
use super::path::{PathLeg, Point};
use super::Engine;
use crate::candles::printed;
use crate::execution::FillCosts;
use crate::reporter::{EventKind, EventReason, FillReason, FillRecord};

/// What the check finds for the account as it stands.
enum Found {
    /// The point the account leaves the market at, with the legs and the spans its
    /// path was solved on left in their buffers.
    Clip(Point),
    /// Clear of the trigger with every leg at its adverse extreme, which holds at the
    /// closes until a funding charge moves the account.
    Clear,
    /// No perp held, or a path that never reaches the trigger.
    Nothing,
}

impl Engine {
    /// The rule, 1 to 3: the path of the perps carried into the bar, solved once the
    /// funding between rows is charged, with the funding stamped inside the bar
    /// counted from the open as the check counts it. A clip on the gap, its start
    /// included, means the account was liquidated before the bar opened, and it is
    /// closed there first; one inside the bar clips every held perp's bar at its
    /// price there.
    pub(super) fn solve_carried(&mut self) {
        if self.clipped {
            self.clips.fill(None);
            self.clipped = false;
        }
        if self.account.perps_held() == 0 {
            return;
        }
        // the funding the bar counts comes after the open, so it moves the bar's
        // piece of the path and never the gap's: counted on the gap, a debit would
        // liquidate an account on a payment not yet due
        let wallet = self.account.margin_cash;
        let inside = wallet - self.counted_inside();
        // the lower of the two, compared plainly: `inside` is NaN wherever `wallet`
        // is, and where it alone is the comparison keeps `wallet` as `f64::min` does
        let lower = if inside < wallet { inside } else { wallet };
        // equity less the requirement is concave on the gap and never rises inside
        // the bar, so a path clear of the trigger at the last marks and at the
        // adverse corner, read with the lower of the two wallets, never reaches it;
        // the last closes found the account clear at those marks unless a transfer
        // or a funding charge has moved it since
        if (self.cleared || self.clear_of_trigger(wallet, |i, _| self.marks[i]))
            && self.clear_of_trigger(lower, |i, q| self.adverse(i, q))
        {
            return;
        }
        self.clip_carried(wallet, inside);
    }

    /// The path carried in, solved for an account the corners could not clear, the
    /// wallet `wallet` on the gap and `inside` on the bar. Kept out of line, so the
    /// look that clears most bars stays small where every step makes it.
    #[inline(never)]
    fn clip_carried(&mut self, wallet: f64, inside: f64) {
        let mut path = std::mem::take(&mut self.path);
        self.carried_path(&mut path);
        // the gap is one move for every path, and every path starts the bar at the
        // opens; inside the bars the optimistic way waits for every path to reach
        // the trigger
        let found = self
            .solve(&path, |piece| if piece == 0 { wallet } else { inside })
            .filter(|(trigger, _)| {
                trigger.piece == 0 || trigger.at == 0.0 || self.every_path_reaches(inside)
            });
        if let Some((_, clip)) = found {
            let (from, to) = path.pieces[clip.piece];
            let at = |leg: &PathLeg| leg.start + clip.at * (leg.end - leg.start);
            if clip.piece == 0 {
                let mut closes = std::mem::take(&mut self.closes);
                closes.clear();
                closes.extend(
                    path.legs[from..to]
                        .iter()
                        .map(|leg| (leg.i, leg.q, at(leg))),
                );
                self.take_over(&closes, Moment::BeforeOpen);
                self.closes = closes;
            } else {
                for leg in &path.legs[from..to] {
                    let bar = self.candles.bar(self.tick, leg.i);
                    if printed(&bar) {
                        self.clips[leg.i] = Some(Engine::fenced(&bar, at(leg), leg.q > 0.0));
                        self.clipped = true;
                    }
                }
            }
        }
        self.path = path;
    }

    /// The rule, 5: the path solved again against the raw bars for the account the
    /// fills left, every carried leg from its open and every leg opened during the
    /// bar from its fill price, with the funding stamped inside the bar counted as the
    /// setting decides. On the trigger the account is taken over at the clip point
    /// recomputed for it, a leg yet to join at its fill price, and the charges it
    /// counted are paid.
    pub(super) fn check(&mut self) {
        // asked here as well as in the decision, so a bar with no perp held costs
        // this test and nothing of the decision's frame
        if self.account.perps_held() == 0 {
            return;
        }
        match self.check_point() {
            Found::Clip(clip) => {
                let (a, b) = self.spans[clip.piece];
                let mut closes = std::mem::take(&mut self.closes);
                closes.clear();
                for leg in self.checked.iter().filter(|leg| leg.leave.is_none()) {
                    let (start, end) = leg.ends(a, b);
                    closes.push((leg.i, leg.q, start + clip.at * (end - start)));
                }
                self.pay_counted_inside();
                self.take_over(&closes, Moment::InBar);
                self.closes = closes;
            }
            Found::Clear => self.clear_through = true,
            Found::Nothing => {}
        }
    }

    /// Whether the check would find the trigger for the account as it stands, which
    /// decides whether a bar holds its favourable exits back (The rule, 4).
    pub(super) fn reaches_trigger(&mut self) -> bool {
        matches!(self.check_point(), Found::Clip(_))
    }

    /// What the check finds for the account as it stands, its path solved against
    /// the margin wallet less the charges stamped inside the bar it counts. One
    /// decision serves the check and the exits it holds back, so the two never part.
    #[inline]
    fn check_point(&mut self) -> Found {
        if self.account.perps_held() == 0 {
            return Found::Nothing;
        }
        let wallet = self.account.margin_cash - self.counted_inside();
        // with no flip on the bar the check's path never rises, so one clear of the
        // trigger where every leg ends never reaches it; a flip's old side leaving
        // takes its requirement with it, and the level can step up there
        if !self.flipped && self.clear_of_trigger(wallet, |i, q| self.adverse(i, q)) {
            return Found::Clear;
        }
        self.check_solved(wallet)
    }

    /// The check's path solved against `wallet`, for an account the corners could
    /// not clear. Kept out of line, as `clip_carried` is, so the look that clears
    /// most bars does not pay for the frame the solve needs.
    #[inline(never)]
    fn check_solved(&mut self, wallet: f64) -> Found {
        let mut checked = std::mem::take(&mut self.checked);
        let mut cuts = std::mem::take(&mut self.cuts);
        let mut spans = std::mem::take(&mut self.spans);
        let mut path = std::mem::take(&mut self.path);
        self.build_legs();
        self.check_legs(&mut checked);
        Engine::check_path(&checked, &mut cuts, &mut spans, &mut path);
        let held = checked.iter().any(|leg| leg.leave.is_none());
        // every path starts where the check's does, so an account past the trigger
        // there is liquidated under either setting
        let found = match self.solve(&path, |_| wallet).filter(|(trigger, _)| {
            let start = (trigger.piece, trigger.at) == (0, 0.0);
            held && (start || self.every_path_reaches(wallet))
        }) {
            Some((_, clip)) => Found::Clip(clip),
            None => Found::Nothing,
        };
        self.checked = checked;
        self.cuts = cuts;
        self.spans = spans;
        self.path = path;
        found
    }

    /// Whether every path inside the bars takes the account as it stands, against
    /// `wallet`, to its trigger, which the optimistic way asks before it liquidates
    /// an account of several perps; the pessimistic way and an account of one take
    /// the synchronous path for every path. Every path starts the bar where the
    /// synchronous one does, which the callers ask first, passes each perp's adverse
    /// extreme at some moment, when the others are no better than their favourable
    /// extremes, and ends at the closes, so past that start it is enough that one
    /// such corner or the closes reach the trigger; if none does, a path exists that
    /// never does (Specification, The path).
    fn every_path_reaches(&self, wallet: f64) -> bool {
        if self.unseen.together || self.account.perps_held() < 2 {
            return true;
        }
        let corner = |k: usize| {
            self.margin_level(wallet, |i, q| {
                let bar = self.candles.bar(self.tick, i);
                if !printed(&bar) {
                    self.marks[i]
                } else if (i == k) == (q > 0.0) {
                    bar.low
                } else {
                    bar.high
                }
            })
        };
        (0..self.account.positions.len())
            .filter(|&k| {
                self.account.markets[k] == Market::Perp
                    && self.account.positions[k].qty.get() != 0.0
            })
            .any(|k| corner(k) <= 0.0)
            || self.margin_level(wallet, |i, _| self.close_of(i)) <= 0.0
    }

    /// The rule, 6: once the row's own funding is charged, an account a debit took
    /// to its trigger is closed at the closes, a perp that printed no bar at its
    /// mark (ADR 0020).
    pub(super) fn check_closes(&mut self) {
        if self.account.perps_held() == 0 {
            return;
        }
        // every leg stands at its close no worse than at its adverse extreme, so an
        // account the check found clear there, with no funding charged since, is
        // clear here
        if self.clear_through {
            self.cleared = true;
            return;
        }
        let level = self.margin_level(self.account.margin_cash, |i, _| self.close_of(i));
        self.cleared = level > 0.0;
        if self.cleared {
            return;
        }
        let mut closes = std::mem::take(&mut self.closes);
        closes.clear();
        for (i, position) in self.account.positions.iter().enumerate() {
            let q = position.qty.get();
            if self.account.markets[i] == Market::Perp && q != 0.0 {
                closes.push((i, q, self.close_of(i)));
            }
        }
        self.take_over(&closes, Moment::AtClose);
        self.closes = closes;
    }

    /// The bar as far as this account got: a long sees nothing below `fence` and a
    /// short nothing above it. Clamping is monotone, so the OHLC ordering survives
    /// it, and a bar lying entirely past the fence collapses to the fence itself,
    /// which is the honest picture of a market the account was not in.
    pub(super) fn fenced(bar: &Candle, fence: f64, long: bool) -> Candle {
        let clip = |price: f64| {
            if long {
                price.max(fence)
            } else {
                price.min(fence)
            }
        };
        Candle {
            open: clip(bar.open),
            high: clip(bar.high),
            low: clip(bar.low),
            close: clip(bar.close),
            volume: bar.volume,
        }
    }

    /// Take the account over (The forced close): close every leg in `closes`, each an
    /// instrument, a signed size and a price, at that price, so a perp with a leg
    /// carried into the bar and one opened during it closes in two fills. With no
    /// maintenance each close pays the liquidation fee on its notional, which at the
    /// close point leaves exactly nothing; with maintenance it pays none. Whatever
    /// the margin wallet then holds, positive or negative, is the liquidation charge.
    pub(super) fn take_over(&mut self, closes: &[(usize, f64, f64)], moment: Moment) {
        let maintained = (0..self.account.positions.len()).any(|i| {
            self.account.markets[i] == Market::Perp
                && self.account.positions[i].qty.get() != 0.0
                && self.has_maintenance(i)
        });
        let rate = if maintained { 0.0 } else { self.cost.fee_taker };
        for &(i, q, price) in closes {
            let size = q.abs();
            let fill = Fill {
                side: if q > 0.0 { Side::Sell } else { Side::Buy },
                size: Qty(size),
                price: Price(price),
                is_taker: false,
            };
            let fee = size * price * rate;
            let before = self.before_fill(i);
            self.account.apply_fill(i, &fill, fee);
            self.book_fill(i, size, price, fee, before, FillReason::Liquidation, moment);
            if let Some(reporter) = self.reporter.as_mut() {
                reporter.costs_mut().fees += fee;
                reporter.record_fill(FillRecord {
                    tick: self.tick,
                    instrument: i,
                    order: None,
                    side: fill.side,
                    size,
                    price,
                    costs: FillCosts {
                        reference: price,
                        ..FillCosts::default()
                    },
                    fee,
                    taker: false,
                    reason: FillReason::Liquidation,
                });
            }
        }
        // whatever the wallet still holds once the positions are closed, taken by
        // the venue or absorbed by it, is the liquidation charge (ADR 0018)
        if let Some(reporter) = self.reporter.as_mut() {
            reporter.costs_mut().liquidation += self.account.margin_cash;
        }
        // The bankruptcy price leaves nothing in exact arithmetic and a hair either
        // side of it in floating point, and a hair above zero is equity the cap
        // levers: with the cap off, an account closed out there opened a position
        // again on the next bar. So the wallet ends at exactly zero (ADRs 0015, 0018).
        self.account.margin_cash = 0.0;
        // Every perp order the closed-out account left goes with it. The empty
        // wallet refused them anyway until a transfer could fund it again, after
        // which they filled for an account that had already left the market
        // (ADR 0018). The spot wallet is untouched, and its orders stand.
        let perp = |markets: &[Market], i: usize| markets[i] == Market::Perp;
        if self.reporter.is_some() {
            let left: Vec<(OrderId, usize)> = self
                .pending
                .iter()
                .chain(self.book.iter())
                .filter(|order| perp(&self.account.markets, order.instrument))
                .map(|order| (order.id, order.instrument))
                .collect();
            for (id, instrument) in left {
                self.log_event(
                    EventKind::Cancelled,
                    Some(id),
                    Some(instrument),
                    None,
                    None,
                    EventReason::Liquidation,
                );
            }
        }
        let markets = &self.account.markets;
        self.pending
            .retain(|order| !perp(markets, order.instrument));
        self.book
            .cancel_where(|order| perp(markets, order.instrument));
    }

    /// No fill closes a perp past the point the margin behind it ran out.
    ///
    /// `fenced` clips the whole candle, and it is read once from the position
    /// carried INTO the bar, which is the approximation ADR 0017 accepted on the
    /// grounds that its error runs pessimistic. It does not, in three shapes. A
    /// position GROWN during the bar and exited on the same bar is bounded by a
    /// fence computed for a smaller position, or by no fence at all when the
    /// account entered the bar flat, so a market entry and a protective stop
    /// armed in one decision booked straight through the margin and left the
    /// account owing money. A resting limit prices at its own limit, which
    /// never passes through the taker clamp, so it walks out of the fenced candle
    /// from underneath. And a partial exit that drives quote negative leaves a
    /// residual for the forced close to price off a balance the closed form at
    /// `bankruptcy_price` assumes is positive, which solves to a price on the far
    /// side of the entry: a liquidated WINNER in the trade log, on a dead account.
    /// That third one is the dangerous shape, because the first two are loud and
    /// pessimistic while it flatters. Bounding the fill instead of the bar covers
    /// all three, because the bound is read at the moment the fill lands rather
    /// than a step earlier (ADR 0018).
    ///
    /// Closing at the bankruptcy price leaves the account exactly at its close level
    /// AT that price, the size the fill closes paying its own fee and the rest the
    /// closing fee, so no ordering of fills within a bar can book past it. With other
    /// perps held, the price is solved against the wallet with what they hold beyond
    /// their own closing fees, each at the fraction of its own bar the fill sits at
    /// on its path, so no reducing fill on one leg overdraws the account its other
    /// legs still stand in.
    ///
    /// The lift stops at the position's average entry, past which the close owes
    /// nothing, and inside `candle`, the bar the fill was priced on: a price the bar
    /// never printed is no trade. What the bar cannot hold, the part of the lift past
    /// it on the size closed, net of the fee it would have moved, is returned to be
    /// paid into the margin wallet (Specification, The margin bound).
    pub(super) fn bound_by_margin(&self, i: usize, fill: &mut Fill, candle: &Candle) -> f64 {
        if self.account.markets[i] != Market::Perp {
            return 0.0;
        }
        let position = self.account.positions[i].qty.get();
        let long = position > 0.0;
        let reduces =
            (long && fill.side == Side::Sell) || (position < 0.0 && fill.side == Side::Buy);
        if !reduces {
            return 0.0;
        }
        let fee = self.cost.fee_taker;
        let price = fill.price.get();
        let wallet = self.wallet_beside(i, fee, price);
        let closed = fill.size.get().min(position.abs());
        let rate = if fill.is_taker {
            self.cost.fee_taker
        } else {
            self.cost.fee_maker
        };
        // the size the fill closes pays its own fee and the rest of the position the
        // closing fee, so a maker's exit is held to the close point its own fee sets
        let paid = (closed * rate + (position.abs() - closed) * fee) / position.abs();
        // no close point above zero: a sale leaves the wallet whole at any price,
        // and a purchase overdraws it at every price, so it is lowered all the way
        let star = match self.account.bankruptcy_price(i, paid, wallet) {
            Some(star) => star.get(),
            None if long => return 0.0,
            None => 0.0,
        };
        let entry = self.account.positions[i].avg_entry.get();
        let (goal, lifted) = if long {
            let goal = star.min(entry);
            (goal, price.max(goal.min(candle.high)))
        } else {
            let goal = star.max(entry);
            (goal, price.min(goal.max(candle.low)))
        };
        fill.price = Price(lifted);
        // what closing that part realizes at a price, net of the fill's own fee
        let kept = |p: f64| position.signum() * closed * (p - entry) - closed * p * rate;
        (kept(goal) - kept(lifted)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::{Candles, Instrument};
    use crate::clock::Clock;
    use crate::engine::fixtures::{cfg, crash_series, levered, ohlc, perp_cfg, universe};
    use crate::engine::{Engine, EngineConfig, Intrabar, CLOSE_EPS};
    use crate::reporter::{EventKind, EventReason, FillReason};
    use probatum_core::{
        Candle, Market, OrderType, Side, Tier, TierBasis, Tiers, TimeInForce, Wallet,
    };

    #[test]
    fn leveraged_long_is_liquidated_on_a_crashing_bar() {
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        e.market_buy(10.0); // 10x notional on 100 margin, fills at bar 1 open 100
        let s1 = e.step(); // bar 1 low 80: equity 100 + 10*(80-100) = -100 -> liquidated
        assert_eq!(s1.position, 0.0);
    }

    #[test]
    fn is_bust_is_set_by_liquidation_and_cleared_by_reset() {
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        assert!(!e.is_bust());
        e.market_buy(10.0);
        e.step(); // liquidated on the crash bar
        assert!(e.is_bust());
        e.reset();
        assert!(!e.is_bust());
    }

    #[test]
    fn a_liquidation_cancels_the_orders_the_account_left_behind() {
        // the spot wallet funds the margin wallet again, and a sell limit the
        // closed-out account left resting would have opened a short with it
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 80.0, 85.0, 1e6),
            ohlc(85.0, 120.0, 85.0, 110.0, 1e6),
        ]);
        let config = EngineConfig {
            idle_cash: 500.0,
            ..levered()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(10.0);
        e.limit_sell(1.0, 115.0);
        let s = e.step(); // the long dies at 90 on the way to 80
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty());
        e.transfer(500.0, Wallet::Margin);
        let s = e.step(); // the bar trades through 115
        assert_eq!(s.position, 0.0);
        assert_eq!(s.margin_cash, 500.0);
    }

    /// Perp at 10x with a fee, reporting on, over `crash_series`.
    fn perp_10x() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            report: true,
            ..cfg()
        }
    }

    #[test]
    fn a_stop_armed_with_the_entry_cannot_book_past_the_margin() {
        // the fence is read from the position carried INTO the bar, and an account
        // that entered flat carries none, so the stop reduced a position the
        // margin no longer stood behind and booked the whole distance. Placing a
        // stop was four times worse than placing nothing (ADR 0018)
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(10.0);
        e.stop(Side::Sell, 10.0, 90.0, true);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);

        // and it is no worse than the same bar with nothing resting on it
        let mut bare = Engine::new(crash_series(), perp_10x());
        bare.reset();
        bare.market_buy(10.0);
        let alone = bare.step();
        assert!((s.equity - alone.equity).abs() < CLOSE_EPS);
    }

    #[test]
    fn a_partial_exit_that_empties_the_margin_books_no_winning_liquidation() {
        // the residual was liquidated off a NEGATIVE quote, which solves to a price
        // on the far side of the entry: a profit booked on a bar that never traded
        // there, lifting win_rate and profit_factor while the return read -100
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        // the trigger is BELOW the bankruptcy price of 901/9.99, about 90.19, so
        // the partial exit alone loses more than the whole margin: six units at
        // 82 is 108 against 99 of it, which leaves the quote negative and four
        // units still open for the forced close to price off
        e.market_buy(10.0);
        e.stop(Side::Sell, 6.0, 82.0, true);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);
        for trade in e.reporter().expect("reporting is on").trades() {
            assert!(
                !(trade.reason == Some(FillReason::Liquidation) && trade.pnl > 0.0),
                "liquidated at {} for a profit of {}",
                trade.exit,
                trade.pnl
            );
        }
    }

    #[test]
    fn a_resting_limit_cannot_price_an_exit_past_the_margin() {
        // fill_limit returns the limit itself and never passes through the taker
        // clamp, so it got out from under the fenced candle the trigger respected
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(10.0);
        e.limit_sell(10.0, 20.0);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);
    }

    #[test]
    fn a_healthy_account_prices_its_exit_where_it_asked() {
        // the bound must bite only where the margin has actually run out, or it
        // would quietly improve every ordinary exit
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(1.0); // a tenth of the cap, nowhere near the fence
        e.limit_sell(1.0, 99.0);
        e.step();
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert!(
            (trades[0].exit - 99.0).abs() < 1e-9,
            "exited at {}",
            trades[0].exit
        );
    }

    #[test]
    fn a_liquidation_is_booked_like_any_other_close() {
        // the forced close ran on the account directly and reached none of the
        // engine's bookkeeping: it appeared in no trade row, counted toward no
        // fill, and left the dead position's entry fee standing to be charged
        // against the NEXT position (ADR 0023)
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            report: true,
            ..cfg()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        e.market_buy(10.0); // 1000 of notional on 100 of quote, exactly at the cap
        let s = e.step(); // fills at the open of 100, then the low of 80 busts it
        assert!(e.is_bust());
        assert_eq!(s.position, 0.0);
        assert_eq!(e.num_fills(), 2); // the buy, and the forced close
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
        // closed where the margin ran out, not at the 80 the bar printed. The entry
        // fee left 99 of margin, so (10*100 - 99) / (10 - 10*0.001) = 90.1902
        assert!(
            (trades[0].exit - 901.0 / 9.99).abs() < 1e-9,
            "closed at {}",
            trades[0].exit
        );
        // and the account is left with exactly nothing, never owing (ADR 0018)
        assert!(s.equity.abs() < CLOSE_EPS, "left {}", s.equity);
        assert!(trades[0].fees > 1.0); // the entry fee, plus a real fee on the exit
                                       // and nothing is left over to charge the next position
        assert!(e
            .reporter()
            .expect("reporting is on")
            .open_trades()
            .next()
            .is_none());
    }

    #[test]
    fn a_liquidation_can_never_leave_the_account_owing() {
        // whatever the bar prints past the point the margin is gone, the loss is
        // bounded by the margin: closing at the extreme booked 200 on a 100 account
        for low in [90.0, 80.0, 40.0, 1.0] {
            let config = EngineConfig {
                market: Market::Perp,
                quote: 100.0,
                fee_taker: 0.0006,
                max_leverage: 10.0,
                ..cfg()
            };
            let candles = Candles::new(vec![
                ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                ohlc(100.0, 101.0, low, low, 1000.0),
                ohlc(low, low + 1.0, low - 1.0, low, 1000.0),
            ]);
            let mut e = Engine::new(candles, config);
            e.reset();
            e.market_buy(10.0);
            let s = e.step();
            assert!(s.equity >= -CLOSE_EPS, "a low of {low} left {}", s.equity);
            assert!(
                s.quote >= -CLOSE_EPS,
                "a low of {low} left quote {}",
                s.quote
            );
        }
    }

    #[test]
    fn a_forced_close_leaves_the_wallet_at_exactly_zero() {
        // the bankruptcy price leaves nothing in exact arithmetic and a hair either
        // side of it in floating point, and a hair above zero is equity the cap
        // levers: with the cap off, an account the engine had just closed out
        // opened a position again, which the next bar liquidated (ADRs 0015, 0018)
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 150.0, 50.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        for fee in [0.0006, 0.002] {
            for size in [23.289_100_547_331_138, 38.434_488_167_935_7, 61.7] {
                for long in [true, false] {
                    let config = EngineConfig {
                        market: Market::Perp,
                        quote: 1_000.0,
                        fee_taker: fee,
                        max_leverage: 0.0,
                        ..cfg()
                    };
                    let mut e = Engine::new(bars.clone(), config);
                    e.reset();
                    if long {
                        e.market_buy(size);
                    } else {
                        e.market_sell(size);
                    }
                    let s = e.step();
                    assert!(e.is_bust(), "fee {fee} size {size} long {long}");
                    assert_eq!(s.quote, 0.0, "fee {fee} size {size} long {long}");
                }
            }
        }
    }

    // A first bar at 100, then a bar that opens at 50, far past where a 10x long
    // opened at 100 runs out of margin.
    fn gap_series() -> Candles {
        Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(50.0, 50.0, 20.0, 30.0, 1_000.0),
            ohlc(30.0, 31.0, 29.0, 30.0, 1_000.0),
        ])
    }

    #[test]
    fn closing_on_a_bar_that_gapped_past_the_liquidation_leaves_nothing_owed() {
        // ADR 0018 says bad debt is structurally unreachable rather than clamped,
        // and that rested on the position still being open when the check ran at
        // the end of the bar. An exit order flattened it first, so nothing was left
        // to liquidate and the account booked the whole gap: a 10x long on 100 of
        // margin ended at -400. The run with the exit must land where the run
        // without one does, because the account died before either could matter
        for exit in [false, true] {
            let mut e = Engine::new(gap_series(), levered());
            e.reset();
            e.market_buy(10.0);
            e.step();
            if exit {
                e.close();
            }
            let state = e.step();
            assert_eq!(state.equity, 0.0, "exit={exit} left {}", state.equity);
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades.len(), 1);
            assert_eq!(trades[0].exit, 90.0);
            assert_eq!(
                trades[0].reason,
                Some(FillReason::Liquidation),
                "exit={exit} booked no liquidation"
            );
        }
    }

    #[test]
    fn a_stop_below_the_bankruptcy_price_never_fills_and_one_above_it_does() {
        // the fence is a price, not a moment: coming down from the open the market
        // reaches 95 before 90, so a stop at 95 saves the account and one at 60 is
        // beyond a point it had already been closed at
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 50.0, 60.0, 1_000.0),
            ohlc(60.0, 61.0, 59.0, 60.0, 1_000.0),
        ]);
        for (trigger, want_equity, want_exit, reason) in [
            (95.0, 50.0, 95.0, FillReason::Stop),
            (60.0, 0.0, 90.0, FillReason::Liquidation),
        ] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_buy(10.0);
            e.step();
            e.stop(Side::Sell, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades[0].exit, want_exit, "stop at {trigger}");
            assert_eq!(trades[0].reason, Some(reason), "stop at {trigger}");
        }
    }

    #[test]
    fn a_liquidation_never_books_a_price_outside_the_bar_that_triggered_it() {
        // a partial close on the gap bar drove quote negative, and the bankruptcy
        // solve is derived assuming quote is positive, so it landed on the far side
        // of the entry: the forced close booked a PROFIT at 120 on a bar whose high
        // was 72, and the log showed a liquidated winner
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(70.0, 72.0, 60.0, 65.0, 1e6),
            ohlc(65.0, 66.0, 64.0, 65.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.market_sell(6.0);
        let state = e.step();
        assert_eq!(state.equity, 0.0);
        for trade in e.reporter().expect("reporting is on").trades() {
            assert!(trade.pnl <= 0.0, "a liquidation booked {}", trade.pnl);
            assert!(trade.exit <= 101.0, "exited at {}", trade.exit);
        }
    }

    #[test]
    fn a_wick_that_busts_a_long_liquidates_even_when_the_bar_closes_back_above_it() {
        // ADR 0017 puts the trigger at the bar's ADVERSE EXTREME, the low for a
        // long, because a bar that traded there is a bar the account was closed out
        // on whatever it printed afterwards. Every other liquidation fixture ends
        // at or below its own bust level, so marking the check at bar.close passes
        // all of them; this bar wicks to 85 and closes back at 100, where a
        // close-marked check reads an equity of 100 and lets a dead account trade on.
        //
        // The favourable wick reaches 130, which is TWICE as far from the entry as
        // the adverse one: a check reading whichever extreme is furthest from the
        // entry, rather than the one that is adverse to this side, marks the long
        // at 130 and finds it healthy. On a bar wicking to 85 and 101 the two
        // readings agree, which is why the high is 130 and not 101
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 130.0, 85.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0); // 1000 of notional on 100 of margin, filled at the open of 100
        let state = e.step();
        // margin at the low is 100 + 10 * (85 - 100) = -50, so the account died on
        // the way down and is closed where the margin ran out, at
        // (10 * 100 - 100) / 10 = 90, leaving exactly nothing (ADR 0018)
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        assert_eq!(state.equity, 0.0);
        assert_eq!(state.bar_close, 100.0); // the bar recovered, the account did not
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit, 90.0);
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
    }

    #[test]
    fn a_wick_that_busts_a_short_liquidates_even_when_the_bar_closes_back_below_it() {
        // the mirror of the rule above, and no engine test has ever liquidated a
        // short at all: a short's adverse extreme is the HIGH, so a wick up that
        // takes the margin kills it however calmly the bar ends (ADR 0017). A check
        // reading bar.close, or one that kept the long's low for both sides, reads
        // 100 here and carries a dead short into the next bar. The favourable wick
        // reaches 70, twice as far from the entry as the adverse one, so a check
        // reading the furthest extreme rather than the adverse one marks the short
        // at 70 and finds it richer than ever
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 115.0, 70.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_sell(10.0); // short 10 at the open of 100, on 100 of margin
        let state = e.step();
        // margin at the high is 100 - 10 * (115 - 100) = -50, and the margin runs
        // out at (-10 * 100 - 100) / -10 = 110, which is where it is closed
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        assert_eq!(state.equity, 0.0);
        assert_eq!(state.bar_close, 100.0); // the bar came back, the account did not
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit, 110.0);
        assert_eq!(trades[0].reason, Some(FillReason::Liquidation));
    }

    #[test]
    fn a_bar_that_gaps_up_past_a_shorts_bankruptcy_price_closes_it_at_the_fence() {
        // ADR 0017 works the fence out before the bar's orders are resolved, and it
        // faces the position's OWN direction: a short dies going up, so a bar that
        // OPENS above its bankruptcy price is a bar the account was already closed
        // out on. Every fence fixture is a 10x long, so the short side of that
        // comparison was never taken; with the long's test a close() here books the
        // raw open of 150 and leaves the account owing 400 (ADR 0018)
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(150.0, 180.0, 140.0, 160.0, 1e6),
            ohlc(160.0, 161.0, 159.0, 160.0, 1e6),
        ]);
        for exit in [false, true] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_sell(10.0); // short 10 at 100, on 100 of margin
            e.step();
            if exit {
                e.close();
            }
            let state = e.step();
            // the margin runs out at (-10 * 100 - 100) / -10 = 110 and the bar
            // opened past it, so the exit is priced there and leaves nothing
            assert_eq!(state.position, 0.0, "exit={exit}");
            assert_eq!(state.equity, 0.0, "exit={exit} left {}", state.equity);
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades.len(), 1, "exit={exit}");
            assert_eq!(trades[0].exit, 110.0, "exit={exit}");
            assert_eq!(
                trades[0].reason,
                Some(FillReason::Liquidation),
                "exit={exit} booked no liquidation"
            );
        }
    }

    #[test]
    fn a_shorts_stop_above_the_bankruptcy_price_never_fills_and_one_below_it_does() {
        // the mirror of the long fence: a short's bar is clipped at the fence from
        // ABOVE, so coming up from the open the market reaches 105 before 110 and a
        // buy stop there saves the account, while one at 140 sits beyond a price
        // the account had already been closed at. Clipping a short's bar with the
        // long's max() leaves that 140 reachable and books an exit 30 past the
        // point the margin was gone (ADR 0017)
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 150.0, 99.0, 140.0, 1e6),
            ohlc(140.0, 141.0, 139.0, 140.0, 1e6),
        ]);
        for (trigger, want_equity, want_exit, reason) in [
            (105.0, 50.0, 105.0, FillReason::Stop),
            (140.0, 0.0, 110.0, FillReason::Liquidation),
        ] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_sell(10.0);
            e.step(); // short 10 at 100
            e.stop(Side::Buy, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades[0].exit, want_exit, "stop at {trigger}");
            assert_eq!(trades[0].reason, Some(reason), "stop at {trigger}");
        }
    }

    #[test]
    fn a_long_gapped_past_its_fence_is_closed_before_its_bar_can_fill_a_target() {
        // long 10 from 100 on 100 of margin runs out at 90, and the next bar opens at
        // 85: the account is closed at 90 before that bar opens, so the take-profit
        // at 94 its high of 95 reaches never fills
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 92.0, 93.0, 1e6),
            ohlc(85.0, 95.0, 80.0, 90.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.limit_sell(10.0, 94.0);
        let state = e.step();
        assert!(e.is_bust());
        assert_eq!(state.equity, 0.0);
        let [t] = e.reporter().expect("reporting is on").trades() else {
            panic!("one round trip");
        };
        assert_eq!((t.exit, t.reason), (90.0, Some(FillReason::Liquidation)));
    }

    #[test]
    fn a_close_out_at_the_close_books_the_wallets_deficit_as_a_negative_charge() {
        // a debit of 200 of funding leaves 100 of margin at -100 at the close, the
        // close at 100 realizes nothing, and the deficit the venue absorbs is a
        // charge of -100, so the bar costs 200 - 100 and equity falls by 100
        let config = EngineConfig {
            funding_rate: 1.0,
            funding_interval: 1,
            report: true,
            ..perp_cfg()
        };
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0);
        let mut e = Engine::new(Candles::new(vec![flat; 2]), config);
        e.reset();
        e.market_buy(2.0);
        e.step();
        let ledger = e.reporter().expect("reporting is on").ledger();
        let costs = ledger[1].costs;
        assert_eq!((costs.funding, costs.liquidation), (200.0, -100.0));
        assert_eq!(ledger[1].equity - ledger[0].equity, -100.0);
    }

    #[test]
    fn a_fill_that_grows_the_position_keeps_its_own_price_under_the_margin_bound() {
        // long 10 from 100 with a fee of 1% leaves 90 of margin; at an open of 91.5
        // the account still holds 5, and a buy of 1 there adds at 91.5, so the
        // entry averages to 1,091.5 / 11 and equity is 89.085 - 85
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.01,
            ..cfg()
        };
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(91.5, 91.5, 91.5, 91.5, 1e6),
        ]);
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.market_buy(1.0);
        let state = e.step();
        assert!((state.avg_entry - 1_091.5 / 11.0).abs() < 1e-9);
        assert!(
            (state.equity - 4.085).abs() < 1e-9,
            "equity {}",
            state.equity
        );
    }

    #[test]
    fn two_perps_are_closed_out_together_where_their_margin_runs_out() {
        // 100 of margin behind a long of 1 from 100 and of 10 from 10, on a bar where
        // they fall to 40 and 4 together: the wallet runs out five sixths of the way
        // down, at 50 and 5, so both close there with nothing left to charge, and the
        // perp order goes while the spot one stays
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(10.0), flat(50.0)],
            vec![flat(100.0), flat(10.0), flat(50.0)],
            vec![
                ohlc(100.0, 100.0, 40.0, 50.0, 1e6),
                ohlc(10.0, 10.0, 4.0, 5.0, 1e6),
                flat(50.0),
            ],
        ];
        let config = EngineConfig {
            idle_cash: 1_000.0,
            report: true,
            ..perp_cfg()
        };
        let markets = [Market::Perp, Market::Perp, Market::Spot];
        let mut e = Engine::new(universe(&markets, rows), config);
        let place = |e: &mut Engine, i, side, size, kind, price| {
            e.order_on(
                i,
                side,
                size,
                kind,
                price,
                None,
                false,
                false,
                TimeInForce::Gtc,
            )
        };
        e.reset();
        place(&mut e, 0, Side::Buy, 1.0, OrderType::Market, None);
        place(&mut e, 1, Side::Buy, 10.0, OrderType::Market, None);
        e.step();
        let perp_order = place(&mut e, 0, Side::Sell, 1.0, OrderType::Limit, Some(150.0));
        place(&mut e, 2, Side::Buy, 1.0, OrderType::Limit, Some(40.0));
        let state = e.step();
        assert_eq!((state.margin_cash, state.spot_cash), (0.0, 1_000.0));
        assert_eq!(state.open_orders.len(), 1);
        assert_eq!(state.open_orders[0].instrument, 2);
        let reporter = e.reporter().expect("reporting is on");
        let closes: Vec<(usize, f64)> = reporter
            .fills()
            .iter()
            .filter(|fill| fill.reason == FillReason::Liquidation)
            .map(|fill| (fill.instrument, fill.price))
            .collect();
        let [(0, a), (1, b)] = closes.as_slice() else {
            panic!("A and B closed, in that order: {closes:?}");
        };
        assert!(
            (a - 50.0).abs() < 1e-9 && (b - 5.0).abs() < 1e-9,
            "closed at {a}, {b}"
        );
        assert!(reporter.ledger()[2].costs.liquidation.abs() < 1e-9);
        let cancelled = reporter
            .events()
            .iter()
            .find(|event| event.kind == EventKind::Cancelled)
            .expect("a cancel");
        assert_eq!(
            (cancelled.order, cancelled.reason),
            (perp_order, EventReason::Liquidation)
        );
    }

    #[test]
    fn a_reducing_fill_beside_another_perp_is_bounded_where_that_perp_sits() {
        // 10 of margin behind 0.1 of A from 50 and 1.3 of B from 100. B's stop at 85
        // sits at the end of its adverse move, where A sits at its low of 80, and the
        // wallet with A there holds 13, so the stop is lifted to 90; A is then closed
        // out at 80, where that wallet runs out, a price its bar printed
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(50.0), flat(100.0)],
            vec![flat(50.0), flat(100.0)],
            vec![flat(100.0), flat(100.0)],
            vec![
                ohlc(100.0, 100.0, 80.0, 90.0, 1e6),
                ohlc(100.0, 100.0, 85.0, 86.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: 10.0,
            max_leverage: 10.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        let place = |e: &mut Engine, i, side, size, kind, trigger, reduce_only| {
            let gtc = TimeInForce::Gtc;
            e.order_on(i, side, size, kind, None, trigger, reduce_only, false, gtc)
        };
        e.reset();
        place(&mut e, 0, Side::Buy, 0.1, OrderType::Market, None, false);
        e.step();
        e.step();
        place(&mut e, 1, Side::Buy, 1.3, OrderType::Market, None, false);
        place(
            &mut e,
            1,
            Side::Sell,
            1.3,
            OrderType::Stop,
            Some(85.0),
            true,
        );
        let last = e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let priced = |reason| -> Vec<(usize, f64)> {
            fills
                .iter()
                .filter(|fill| fill.reason == reason)
                .map(|fill| (fill.instrument, fill.price))
                .collect()
        };
        let (stops, closes) = (priced(FillReason::Stop), priced(FillReason::Liquidation));
        let ([(1, stop)], [(0, close)]) = (stops.as_slice(), closes.as_slice()) else {
            panic!("one stop on B and one close of A");
        };
        assert!((stop - 90.0).abs() < 1e-9, "the stop filled at {stop}");
        assert!((close - 80.0).abs() < 1e-9, "A closed at {close}");
        assert_eq!(last.margin_cash, 0.0);
    }

    /// A long of 10 on A from `entry` with a reduce-only sell stop at 98, and a buy
    /// of `b` B at B's open of 50, on a bar where A trades from 100 down to 96 and B
    /// falls to 36, on `margin` with no fees and no cap. Returns the stop's price,
    /// that bar's liquidation charge and the price B is closed out at.
    fn stop_beside(entry: f64, margin: f64, b: f64) -> (f64, f64, f64) {
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(entry), flat(50.0)],
            vec![flat(entry), flat(50.0)],
            vec![
                ohlc(100.0, 105.0, 96.0, 100.0, 1e6),
                ohlc(50.0, 50.0, 36.0, 40.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: margin,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.stop(Side::Sell, 10.0, 98.0, true);
        let ioc = TimeInForce::Ioc;
        e.order_on(
            1,
            Side::Buy,
            b,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let price = |reason| {
            let fill = reporter.fills().iter().find(|fill| fill.reason == reason);
            fill.expect("the fill").price
        };
        let charge = reporter.ledger()[2].costs.liquidation;
        (
            price(FillReason::Stop),
            charge,
            price(FillReason::Liquidation),
        )
    }

    #[test]
    fn the_bound_lifts_a_fill_no_further_than_its_entry_or_its_bar() {
        // B sits at 43 where A's stop at 98 sits on its path, so the stop's close
        // point is 104, and the lift stops at A's entry of 100. From an entry of 110
        // beside 30 B the close point is 111 and the entry 110, but the bar printed
        // no more than 105: the stop fills there, and the 50 the bar could not hold is
        // paid into the wallet, a negative liquidation charge with the lift's 70
        let (stop, _, close) = stop_beside(100.0, 100.0, 20.0);
        assert!((stop - 100.0).abs() < 1e-9, "the stop filled at {stop}");
        assert!((close - 45.0).abs() < 1e-9, "B closed at {close}");
        let (stop, charge, close) = stop_beside(110.0, 200.0, 30.0);
        assert!((stop - 105.0).abs() < 1e-9, "the stop filled at {stop}");
        assert!((charge + 120.0).abs() < 1e-9, "the charge was {charge}");
        assert!((close - 1_300.0 / 30.0).abs() < 1e-9, "B closed at {close}");
    }

    #[test]
    fn a_purchase_with_no_close_point_above_zero_is_lowered_to_the_entry() {
        // 100 of margin behind a short of 10 A from 100, with a buy stop at 110, and
        // 100 B bought at its open of 50. The stop sits halfway up A's bar, where B
        // has fallen to 35, so the wallet beside A holds -1,400 and no purchase of A
        // above zero leaves it whole: the stop is lowered to the entry, and B is
        // closed out where the 100 runs out, at 49
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(50.0)],
            vec![flat(100.0), flat(50.0)],
            vec![
                ohlc(100.0, 120.0, 99.0, 110.0, 1e6),
                ohlc(50.0, 50.0, 20.0, 30.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: 100.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        let (ioc, gtc) = (TimeInForce::Ioc, TimeInForce::Gtc);
        e.reset();
        e.order_on(
            0,
            Side::Sell,
            10.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        e.order_on(
            0,
            Side::Buy,
            10.0,
            OrderType::Stop,
            None,
            Some(110.0),
            true,
            false,
            gtc,
        );
        e.order_on(
            1,
            Side::Buy,
            100.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let priced = |reason| -> Vec<(usize, f64)> {
            fills
                .iter()
                .filter(|fill| fill.reason == reason)
                .map(|fill| (fill.instrument, fill.price))
                .collect()
        };
        let (stops, closes) = (priced(FillReason::Stop), priced(FillReason::Liquidation));
        let ([(0, stop)], [(1, close)]) = (stops.as_slice(), closes.as_slice()) else {
            panic!("one stop on A and one close of B");
        };
        assert_eq!(*stop, 100.0);
        assert!((close - 49.0).abs() < 1e-9, "B closed at {close}");
    }

    #[test]
    fn a_reducing_fill_on_the_gap_reads_the_other_perps_at_their_opens() {
        // 270 of margin and fees of 0.1% behind a long of 10 A from 100, marked 90,
        // and 10 B from 50. A's resting sell at 92 fills on the gap up to an open of
        // 100, so it traded before the bar opened, with B at 50: nothing is lifted,
        // the wallet keeps 189.08, and B is closed where that reaches its close
        // level, 310.92 / 9.99. B at its low of 30 would have lifted the sale to
        // 93.12 and left B at its close level there
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = vec![
            vec![flat(100.0), flat(50.0)],
            vec![flat(100.0), flat(50.0)],
            vec![flat(90.0), flat(50.0)],
            vec![
                ohlc(100.0, 101.0, 95.0, 100.0, 1e6),
                ohlc(50.0, 50.0, 30.0, 40.0, 1e6),
            ],
        ];
        let config = EngineConfig {
            quote: 271.5,
            fee_taker: 0.001,
            fee_maker: 0.001,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(universe(&[Market::Perp, Market::Perp], rows), config);
        let (ioc, gtc) = (TimeInForce::Ioc, TimeInForce::Gtc);
        e.reset();
        e.market_buy(10.0);
        e.order_on(
            1,
            Side::Buy,
            10.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        e.step();
        e.order_on(
            0,
            Side::Sell,
            10.0,
            OrderType::Limit,
            Some(92.0),
            None,
            true,
            false,
            gtc,
        );
        e.step();
        let fills = e.reporter().expect("reporting is on").fills();
        let priced = |reason| -> Vec<(usize, f64)> {
            fills
                .iter()
                .filter(|fill| fill.reason == reason)
                .map(|fill| (fill.instrument, fill.price))
                .collect()
        };
        let (sales, closes) = (priced(FillReason::Limit), priced(FillReason::Liquidation));
        let ([(0, sale)], [(1, close)]) = (sales.as_slice(), closes.as_slice()) else {
            panic!("one sale of A and one close of B");
        };
        assert_eq!(*sale, 92.0);
        assert!((close - 310.92 / 9.99).abs() < 1e-9, "B closed at {close}");
    }

    /// 51.5 of margin behind a long of 10 bought at 100 for a taker's fee of 1, on a
    /// bar down to 95, where the 50.5 left sits between the close point, 95.045, and
    /// the trigger, 94.95; returned after that bar, with `next` to come and a maker's
    /// fee of `maker`.
    fn between_the_close_point_and_the_trigger(next: Candle, maker: f64) -> Engine {
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let bars = vec![
            flat(100.0),
            flat(100.0),
            ohlc(100.0, 100.0, 95.0, 95.0, 1e6),
            next,
        ];
        let config = EngineConfig {
            quote: 51.5,
            fee_taker: 0.001,
            fee_maker: maker,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(Candles::new(bars), config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.step();
        e
    }

    #[test]
    fn a_reducing_fill_is_bounded_at_its_own_fee_on_the_size_it_closes() {
        // the Specification's example: a market sale at 95 pays the taker's fee, its
        // close point is 95.045, past the bar, and the 0.45 the bar cannot hold is
        // paid in, so the wallet ends at zero
        let mut e = between_the_close_point_and_the_trigger(ohlc(95.0, 95.0, 95.0, 95.0, 1e6), 0.0);
        e.market_sell(10.0);
        let taker = e.step();
        let charge = e.reporter().expect("reporting is on").ledger()[3]
            .costs
            .liquidation;
        assert!((charge + 0.45).abs() < 1e-9, "the charge was {charge}");
        assert!(taker.margin_cash.abs() < 1e-9, "kept {}", taker.margin_cash);
        // a limit at 95 above an open of 94.96 fills as a maker for no fee, so its
        // close point is the trigger, below the fill: nothing is lifted and the
        // wallet keeps 0.5, where the taker's fee would have paid it 0.45 more
        let mut e =
            between_the_close_point_and_the_trigger(ohlc(94.96, 95.0, 94.96, 95.0, 1e6), 0.0);
        e.limit_sell(10.0, 95.0);
        let maker = e.step();
        let fill = *e
            .reporter()
            .expect("reporting is on")
            .fills()
            .last()
            .expect("the sale");
        assert_eq!((fill.price, fill.taker, fill.fee), (95.0, false, 0.0));
        assert_eq!(
            e.reporter().expect("reporting is on").ledger()[3]
                .costs
                .liquidation,
            0.0
        );
        assert!(
            (maker.margin_cash - 0.5).abs() < 1e-9,
            "kept {}",
            maker.margin_cash
        );
    }

    #[test]
    fn a_partial_or_dearer_maker_exit_is_held_to_the_zero_its_own_fee_sets() {
        // on a bar from 94.98 to 95.10 the trigger is never reached. A maker's sale
        // of 5 at 94.99 for no fee leaves the other 5 paying the taker's 0.1%, so the
        // equity is zero at 949.5 / 9.995, inside the bar, where the sale is lifted
        let bar = ohlc(94.98, 95.1, 94.97, 95.0, 1e6);
        let mut e = between_the_close_point_and_the_trigger(bar, 0.0);
        e.limit_sell(5.0, 94.99);
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let fill = *reporter.fills().last().expect("the sale");
        let zero = 949.5 / 9.995;
        assert!((fill.price - zero).abs() < 1e-9, "sold at {}", fill.price);
        let charge = reporter.ledger()[3].costs.liquidation;
        assert!(
            (charge + 5.0 * (zero - 94.99)).abs() < 1e-9,
            "the charge was {charge}"
        );
        // a maker's sale of all 10 at 95.10 paying 0.2%, above the taker's fee, has
        // its zero at 949.5 / 9.98, past the bar: the 10 * 0.998 * (zero - 95.10)
        // the bar cannot hold is paid in, net of its own fee, and the wallet ends at
        // zero
        let mut e = between_the_close_point_and_the_trigger(bar, 0.002);
        e.limit_sell(10.0, 95.1);
        let last = e.step();
        let charge = e.reporter().expect("reporting is on").ledger()[3]
            .costs
            .liquidation;
        let paid = 10.0 * 0.998 * (949.5 / 9.98 - 95.1);
        assert!((charge + paid).abs() < 1e-9, "the charge was {charge}");
        assert!(last.margin_cash.abs() < 1e-9, "kept {}", last.margin_cash);
    }

    const HOUR: i64 = 3_600 * 1_000_000_000;
    const DAY: i64 = 24 * HOUR;

    /// Bitcoin at 60,000 on two rows, then the bars in `later`.
    fn bitcoin(later: &[Candle]) -> Vec<Candle> {
        let flat = ohlc(60_000.0, 60_000.0, 60_000.0, 60_000.0, 1e6);
        [&[flat, flat][..], later].concat()
    }

    /// The Specification's account at one instrument: 10,015 of margin and fees of
    /// 0.02% and 0.05%, buying 0.5 BTC with a limit at 60,000 on a bar that opens on
    /// it, a taker's fill of 15 that leaves 10,000; returned after that bar.
    fn half_a_bitcoin(bars: Candles) -> Engine {
        half_a_bitcoin_at(bars, Intrabar::Pessimistic)
    }

    /// `half_a_bitcoin` deciding what its bars cannot show by `intrabar`.
    fn half_a_bitcoin_at(bars: Candles, intrabar: Intrabar) -> Engine {
        let config = EngineConfig {
            quote: 10_015.0,
            fee_maker: 0.0002,
            fee_taker: 0.0005,
            max_leverage: 10.0,
            report: true,
            intrabar,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        let gtc = TimeInForce::Gtc;
        e.order_on(
            0,
            Side::Buy,
            0.5,
            OrderType::Limit,
            Some(60_000.0),
            None,
            false,
            false,
            gtc,
        );
        e.step();
        e
    }

    /// One tier at 20x keeping a maintenance rate of 0.5%.
    fn maintained() -> Tiers {
        let tier = Tier {
            up_to: 1e12,
            leverage: 20.0,
            rate: 0.005,
            amount: 0.0,
        };
        Tiers::new(TierBasis::Notional, vec![tier]).expect("a table")
    }

    const FALL: Candle = Candle {
        open: 41_500.0,
        high: 41_500.0,
        low: 39_000.0,
        close: 39_500.0,
        volume: 1e6,
    };

    #[test]
    fn with_no_maintenance_the_account_closes_at_40_020_01_and_ends_at_zero() {
        // the Specification's example: equity is 0.5p - 20,000 and the fee 0.00025p,
        // which meet at 40,020.01, where the 10.005 left pays the fee
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[FALL])));
        let last = e.step();
        let reporter = e.reporter().expect("reporting is on");
        let close = reporter.fills().last().expect("the close");
        assert_eq!(close.reason, FillReason::Liquidation);
        assert!(
            (close.price - 40_020.01).abs() < 0.005,
            "closed at {}",
            close.price
        );
        assert!((close.fee - 10.005).abs() < 5e-4, "paid {}", close.fee);
        assert!(reporter.ledger()[2].costs.liquidation.abs() < 1e-6);
        assert_eq!(last.margin_cash, 0.0);
    }

    #[test]
    fn a_stop_at_41_000_fills_on_the_clipped_bar_and_keeps_489_75() {
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[FALL])));
        e.stop(Side::Sell, 0.5, 41_000.0, false);
        let last = e.step();
        let reporter = e.reporter().expect("reporting is on");
        let fills: Vec<(FillReason, f64)> = reporter
            .fills()
            .iter()
            .map(|fill| (fill.reason, fill.price))
            .collect();
        assert_eq!(
            fills,
            vec![(FillReason::Limit, 60_000.0), (FillReason::Stop, 41_000.0)]
        );
        assert!(
            (last.margin_cash - 489.75).abs() < 1e-9,
            "kept {}",
            last.margin_cash
        );
    }

    /// The fall with a high of 42,500, which reaches a sell limit at 42,000 above its
    /// open.
    const WIDE: Candle = Candle {
        high: 42_500.0,
        ..FALL
    };

    /// The fills of the run so far, each its reason and its price.
    fn priced(e: &Engine) -> Vec<(FillReason, f64)> {
        let fills = e.reporter().expect("reporting is on").fills();
        fills.iter().map(|fill| (fill.reason, fill.price)).collect()
    }

    #[test]
    fn a_limit_at_42_000_is_held_back_and_the_account_closes_at_40_020_01() {
        // the bar reaches the limit above its open and the clip point below it, and
        // cannot show which came first; without the limit the path reaches the
        // trigger, so the limit is held back and the liquidation cancels it
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[WIDE])));
        let limit = e.limit_sell(0.5, 42_000.0);
        let last = e.step();
        let fills = priced(&e);
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[1].0, FillReason::Liquidation);
        assert!(
            (fills[1].1 - 40_020.01).abs() < 0.005,
            "closed at {}",
            fills[1].1
        );
        let reporter = e.reporter().expect("reporting is on");
        let cancelled = reporter
            .events()
            .iter()
            .find(|event| event.kind == EventKind::Cancelled)
            .expect("the limit cancelled");
        assert_eq!(
            (cancelled.order, cancelled.reason),
            (limit, EventReason::Liquidation)
        );
        assert_eq!(last.margin_cash, 0.0);
    }

    #[test]
    fn a_limit_at_42_000_fills_where_the_bar_stays_clear_of_the_trigger() {
        // the same bar with a low of 40,500 never reaches the trigger, so the limit
        // fills as a maker and the account keeps 10,000 - 9,000 - 4.20
        let shallow = Candle {
            low: 40_500.0,
            close: 41_000.0,
            ..WIDE
        };
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[shallow])));
        e.limit_sell(0.5, 42_000.0);
        let last = e.step();
        assert_eq!(
            priced(&e),
            vec![(FillReason::Limit, 60_000.0), (FillReason::Limit, 42_000.0)]
        );
        assert!(
            (last.margin_cash - 995.8).abs() < 1e-9,
            "kept {}",
            last.margin_cash
        );
    }

    #[test]
    fn under_the_optimistic_setting_the_limit_at_42_000_fills_and_keeps_995_80() {
        // the Specification's example: on the bar down to 39,000 the optimistic
        // setting has the limit fill first, and the account keeps 10,000 - 9,000 -
        // 4.20, as emsl booked it
        let mut e = half_a_bitcoin_at(Candles::new(bitcoin(&[WIDE])), Intrabar::Optimistic);
        e.limit_sell(0.5, 42_000.0);
        let last = e.step();
        assert_eq!(
            priced(&e),
            vec![(FillReason::Limit, 60_000.0), (FillReason::Limit, 42_000.0)]
        );
        assert!(
            (last.margin_cash - 995.8).abs() < 1e-9,
            "kept {}",
            last.margin_cash
        );
    }

    #[test]
    fn an_optimistic_bar_counts_a_credit_stamped_inside_it_toward_its_check() {
        // half a bitcoin with 0.5% maintenance on a bar down to 40,200, below the
        // trigger at 40,201.01, and a credit of 10.25 stamped at 07:30. The
        // pessimistic setting leaves the credit out and closes the account at the
        // trigger; the optimistic one counts it, which moves the trigger to where
        // 0.5p - 19,989.75 falls to 0.0025p, 40,180.40, below the bar, and the
        // account keeps 10,000 - 9,500 + 10.25 at the close
        for (intrabar, kept) in [(Intrabar::Pessimistic, 0.0), (Intrabar::Optimistic, 510.25)] {
            let bars = bitcoin(&[ohlc(60_000.0, 60_000.0, 40_200.0, 41_000.0, 1e6)]);
            let clock = Clock::new(vec![5 * HOUR, 6 * HOUR, 7 * HOUR]);
            let stamp = 7 * HOUR + 30 * 60 * 1_000_000_000;
            let candles = Candles::on_clock(bars, clock)
                .with_funding(&[(stamp, -0.0005)])
                .with_tiers(maintained());
            let mut e = half_a_bitcoin_at(candles, intrabar);
            let last = e.step();
            assert!(
                (last.equity - kept).abs() < 1e-9,
                "{intrabar:?} kept {}",
                last.equity
            );
        }
    }

    #[test]
    fn a_stop_that_saves_the_account_lets_the_limit_beside_it_fill() {
        // the bar resolves without the limit first: a stop of 0.3 at 41,000 leaves
        // 0.2 held on 4,293.85, which stands 93.85 clear at the low, so the bar
        // resolves again with both, the worse exit first, and each fills once
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[WIDE])));
        e.limit_sell(0.2, 42_000.0);
        e.stop(Side::Sell, 0.3, 41_000.0, true);
        let last = e.step();
        assert_eq!(
            priced(&e),
            vec![
                (FillReason::Limit, 60_000.0),
                (FillReason::Stop, 41_000.0),
                (FillReason::Limit, 42_000.0)
            ]
        );
        assert_eq!(e.num_fills(), 3);
        assert_eq!(last.position, 0.0);
        assert!(
            (last.margin_cash - 692.17).abs() < 1e-9,
            "kept {}",
            last.margin_cash
        );
    }

    #[test]
    fn a_stop_too_small_to_save_the_account_still_holds_the_limit_back() {
        // a stop of 0.1 at 41,000 leaves 0.4 held on 8,097.95, which the path still
        // takes to its close point, (24,000 - 8,097.95) / 0.3998 = 39,775.01: the
        // stop fills, the limit is held back, and the rest is closed there
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[WIDE])));
        e.limit_sell(0.4, 42_000.0);
        e.stop(Side::Sell, 0.1, 41_000.0, true);
        let last = e.step();
        let fills = priced(&e);
        assert_eq!(fills.len(), 3);
        assert_eq!(fills[1], (FillReason::Stop, 41_000.0));
        assert_eq!(fills[2].0, FillReason::Liquidation);
        let close = 15_902.05 / 0.3998;
        assert!(
            (fills[2].1 - close).abs() < 1e-6,
            "closed at {}",
            fills[2].1
        );
        assert_eq!(last.margin_cash, 0.0);
    }

    #[test]
    fn a_take_profit_on_a_position_its_own_bar_opened_is_held_back() {
        // a flat 100 rests a buy at 95 and a sell at 105, and the bar opens at 100.
        // The buy opens a long the sell would close above the open, so the sell is
        // held back, and on a bar down to 80 the long, joining the path where 95 lies
        // on it, reaches its trigger at 85 and is closed there. On a bar that stops
        // at 90 the trigger is never reached and both fill, keeping 200
        let cases = [
            (
                80.0,
                vec![(FillReason::Limit, 95.0), (FillReason::Liquidation, 85.0)],
                0.0,
            ),
            (
                90.0,
                vec![(FillReason::Limit, 95.0), (FillReason::Limit, 105.0)],
                200.0,
            ),
        ];
        for (low, fills, kept) in cases {
            let bars = Candles::new(vec![
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(100.0, 106.0, low, 95.0, 1e6),
            ]);
            let config = EngineConfig {
                max_leverage: 10.0,
                report: true,
                ..perp_cfg()
            };
            let mut e = Engine::new(bars, config);
            e.reset();
            e.limit_buy(10.0, 95.0);
            e.limit_sell(10.0, 105.0);
            let last = e.step();
            assert_eq!(priced(&e), fills, "a low of {low}");
            assert_eq!(last.position, 0.0, "a low of {low}");
            assert!(
                (last.margin_cash - kept).abs() < 1e-9,
                "a low of {low} kept {}",
                last.margin_cash
            );
        }
    }

    #[test]
    fn a_take_profit_below_the_open_on_a_long_the_bar_opened_lower_is_held_back() {
        // the buy at 95 opens a long on a bar from 100, and a reduce-only sale at 98,
        // below the open, would close it above where it started: after 95 the bar
        // reached 98 only on its way back up to the close at 98.50, which it cannot
        // show came before the trigger at 85, so the sale is held back and the long
        // closed at 85
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 80.0, 98.5, 1e6),
        ]);
        let config = EngineConfig {
            max_leverage: 10.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.limit_buy(10.0, 95.0);
        e.order(
            Side::Sell,
            10.0,
            OrderType::Limit,
            Some(98.0),
            None,
            true,
            false,
            TimeInForce::Gtc,
        );
        let last = e.step();
        assert_eq!(
            priced(&e),
            vec![(FillReason::Limit, 95.0), (FillReason::Liquidation, 85.0)]
        );
        assert_eq!(last.margin_cash, 0.0);
    }

    #[test]
    fn the_margin_bound_lifts_a_sale_on_the_part_of_its_bar_no_higher_than_that_part() {
        // a long of 1 from a buy at 95 inside a bar from 100, grown by a buy stop at
        // 108 to 2 at 101.50, on a margin wallet of 5 with no cap: a reduce-only sale
        // of 2 at 85 leaves the wallet short, and the bound would lift it to 99, but
        // the part of the bar after the buy goes no higher than its close of 96, so
        // the sale fills at 96 and the 6 the part cannot hold is paid in
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 110.0, 80.0, 96.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 5.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.limit_buy(1.0, 95.0);
        e.stop(Side::Buy, 1.0, 108.0, false);
        e.stop(Side::Sell, 2.0, 85.0, true);
        let s = e.step();
        assert_eq!(
            priced(&e),
            vec![
                (FillReason::Limit, 95.0),
                (FillReason::Stop, 108.0),
                (FillReason::Stop, 96.0)
            ]
        );
        assert_eq!((s.position, s.margin_cash), (0.0, 0.0));
    }

    #[test]
    fn a_stop_above_where_its_long_started_fills_at_the_fill_that_opened_it() {
        // the buy at 95 opens a long inside a bar from 100, and a reduce-only stop at
        // 95.3 is through at that moment, so it fills there, at 95 less 50 bps,
        // 94.525, rather than on a move back up the bar never shows after the fill;
        // the fall to 80 finds the account flat, where its trigger was at 85
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 80.0, 85.0, 1e6),
        ]);
        let config = EngineConfig {
            max_leverage: 10.0,
            slippage_bps: 50.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.limit_buy(10.0, 95.0);
        e.stop(Side::Sell, 10.0, 95.3, true);
        let last = e.step();
        assert_eq!(
            priced(&e),
            vec![(FillReason::Limit, 95.0), (FillReason::Stop, 94.525)]
        );
        assert_eq!(last.position, 0.0);
    }

    #[test]
    fn a_gap_through_the_clip_point_liquidates_the_account_before_the_bar_opens() {
        // the last mark is 40,500 and the next bar opens at 40,010, so the gap
        // crosses the close point, 40,020.01, before the bar opens: the account is
        // closed there and the stop decided at the last close never fills
        let later = [
            ohlc(60_000.0, 60_000.0, 40_500.0, 40_500.0, 1e6),
            ohlc(40_010.0, 40_600.0, 39_000.0, 39_500.0, 1e6),
        ];
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&later)));
        e.step();
        let stop = e.stop(Side::Sell, 0.5, 40_300.0, false);
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let fills: Vec<(usize, FillReason)> = reporter
            .fills()
            .iter()
            .map(|fill| (fill.tick, fill.reason))
            .collect();
        assert_eq!(
            fills,
            vec![(1, FillReason::Limit), (3, FillReason::Liquidation)]
        );
        let close = reporter.fills()[1].price;
        assert!((close - 40_020.01).abs() < 0.005, "closed at {close}");
        let cancelled = reporter
            .events()
            .iter()
            .find(|event| event.kind == EventKind::Cancelled)
            .expect("the stop cancelled");
        assert_eq!(cancelled.order, stop);
    }

    #[test]
    fn takeover_at_half_a_percent_maintenance_charges_the_100_50_left() {
        // the trigger is where 0.5p - 20,000 falls to 0.0025p, 40,201.01, where
        // equity and the requirement are both 100.50; the close pays no fee of its
        // own and the 100.50 is the liquidation charge
        let slide = ohlc(41_000.0, 41_200.0, 40_150.0, 40_500.0, 1e6);
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[slide])).with_tiers(maintained()));
        let last = e.step();
        let reporter = e.reporter().expect("reporting is on");
        let close = reporter.fills().last().expect("the close");
        assert_eq!(close.reason, FillReason::Liquidation);
        assert!(
            (close.price - 40_201.01).abs() < 0.005,
            "closed at {}",
            close.price
        );
        assert_eq!(close.fee, 0.0);
        let charge = reporter.ledger()[2].costs.liquidation;
        assert!((charge - 100.50).abs() < 0.005, "charged {charge}");
        assert_eq!(last.margin_cash, 0.0);
    }

    #[test]
    fn a_trigger_past_a_tiers_bound_is_solved_in_the_tier_it_lies_in() {
        // up to 25,000 of notional at 0.4% and above it at 1% less 150: the long's
        // notional crosses the bound at 50,000 on the way down, and below it the
        // trigger is where 0.5p - 20,000 falls to 0.002p, 40,160.64, which leaves
        // the 80.32 the requirement asks there to be charged
        let tiers = Tiers::new(
            TierBasis::Notional,
            vec![
                Tier {
                    up_to: 25_000.0,
                    leverage: 20.0,
                    rate: 0.004,
                    amount: 0.0,
                },
                Tier {
                    up_to: 1e12,
                    leverage: 10.0,
                    rate: 0.01,
                    amount: 150.0,
                },
            ],
        )
        .expect("a table");
        let fall = ohlc(60_000.0, 60_000.0, 40_000.0, 41_000.0, 1e6);
        let mut e = half_a_bitcoin(Candles::new(bitcoin(&[fall])).with_tiers(tiers));
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        let close = reporter.fills().last().expect("the close");
        assert!(
            (close.price - 20_000.0 / 0.498).abs() < 1e-6,
            "closed at {}",
            close.price
        );
        let charge = reporter.ledger()[2].costs.liquidation;
        assert!(
            (charge - 0.002 * 20_000.0 / 0.498).abs() < 1e-6,
            "charged {charge}"
        );
    }

    /// The account with 0.5% maintenance on rows at 05:00, 06:00 and 07:00, the
    /// last a bar from 60,000 down to 40,210 that closes at 41,000, and one funding
    /// event at `rate` stamped at `stamp`; returned after its last bar.
    fn seven_oclock(rate: f64, stamp: i64) -> Engine {
        let bars = bitcoin(&[ohlc(60_000.0, 60_000.0, 40_210.0, 41_000.0, 1e6)]);
        let clock = Clock::new(vec![5 * HOUR, 6 * HOUR, 7 * HOUR]);
        let candles = Candles::on_clock(bars, clock)
            .with_funding(&[(stamp, rate)])
            .with_tiers(maintained());
        let mut e = half_a_bitcoin(candles);
        e.step();
        e
    }

    #[test]
    fn a_debit_charged_after_the_check_leaves_489_75_against_102_50() {
        let e = seven_oclock(0.0005, 8 * HOUR);
        let row = e.reporter().expect("reporting is on").ledger()[2];
        assert!((row.costs.funding - 10.25).abs() < 1e-9);
        assert!((row.equity - 489.75).abs() < 1e-9, "equity {}", row.equity);
        assert_eq!(row.costs.liquidation, 0.0);
    }

    #[test]
    fn a_debit_of_410_closes_the_account_at_the_close_and_charges_its_90_00() {
        let e = seven_oclock(0.02, 8 * HOUR);
        let reporter = e.reporter().expect("reporting is on");
        let close = reporter.fills().last().expect("the close");
        assert_eq!(
            (close.reason, close.price, close.fee),
            (FillReason::Liquidation, 41_000.0, 0.0)
        );
        let row = reporter.ledger()[2];
        assert!((row.costs.funding - 410.0).abs() < 1e-9);
        assert!((row.costs.liquidation - 90.0).abs() < 1e-9);
        assert_eq!(row.margin_cash, 0.0);
    }

    #[test]
    fn a_debit_inside_the_bar_counts_toward_its_check() {
        // a debit of 10.25 stamped at 07:30 moves the trigger to where 0.5p - 20,010.25
        // falls to 0.0025p, 40,221.61, above the low of 40,210
        let e = seven_oclock(0.0005, 7 * HOUR + 30 * 60 * 1_000_000_000);
        let reporter = e.reporter().expect("reporting is on");
        let close = reporter.fills().last().expect("the close");
        assert!(
            (close.price - 40_221.61).abs() < 0.005,
            "closed at {}",
            close.price
        );
        let row = reporter.ledger()[2];
        assert!((row.costs.funding - 10.25).abs() < 1e-9);
        assert!((row.costs.liquidation - 100.55).abs() < 0.005);
    }

    /// Half a bitcoin with 0.5% maintenance on a 07:00 bar from 60,000 down to `low`
    /// that closes at 41,000, one funding event at `rate` stamped at 07:30, inside it,
    /// and a reduce-only stop to sell the half at `stop`; returned after that bar.
    fn stopped_inside(rate: f64, low: f64, stop: f64, intrabar: Intrabar) -> (Engine, f64) {
        let bars = bitcoin(&[ohlc(60_000.0, 60_000.0, low, 41_000.0, 1e6)]);
        let clock = Clock::new(vec![5 * HOUR, 6 * HOUR, 7 * HOUR]);
        let candles = Candles::on_clock(bars, clock)
            .with_funding(&[(7 * HOUR + 30 * 60 * 1_000_000_000, rate)])
            .with_tiers(maintained());
        let mut e = half_a_bitcoin_at(candles, intrabar);
        e.stop(Side::Sell, 0.5, stop, true);
        let kept = e.step().margin_cash;
        (e, kept)
    }

    #[test]
    fn a_debit_inside_the_bar_clips_it_above_a_stop_a_debit_first_never_reaches() {
        // the debit of 10.25 moves the trigger to 40,221.61, and the path carried in
        // counts it from the open as the check does, so the bar is clipped there and
        // the stop at 40,215 below it never fills: the debit could have come before
        // the fall, and the account closes at its trigger
        let (e, kept) = stopped_inside(0.0005, 40_210.0, 40_215.0, Intrabar::Pessimistic);
        let fills = priced(&e);
        assert_eq!(fills.len(), 2, "{fills:?}");
        assert_eq!(fills[1].0, FillReason::Liquidation);
        assert!(
            (fills[1].1 - 40_221.61).abs() < 0.005,
            "closed at {}",
            fills[1].1
        );
        assert_eq!(kept, 0.0);
    }

    #[test]
    fn an_optimistic_credit_inside_the_bar_leaves_it_whole_for_a_stop_to_fill() {
        // the credit of 10.25 moves the trigger to 40,180.40, below the low of 40,200,
        // so the optimistic path carried in reaches no trigger and clips nothing, and
        // the stop at 40,200.50 fills on the whole bar: 10,000 - 9,899.75 - 10.05
        let (e, kept) = stopped_inside(-0.0005, 40_200.0, 40_200.5, Intrabar::Optimistic);
        assert_eq!(
            priced(&e),
            vec![(FillReason::Limit, 60_000.0), (FillReason::Stop, 40_200.5)]
        );
        assert!((kept - 90.199_875).abs() < 1e-9, "kept {kept}");
    }

    #[test]
    fn a_debit_over_a_weekend_moves_the_trigger_onto_mondays_gap() {
        // 7 January 1970 was a Wednesday, so the rows run Wednesday to Friday and
        // Monday; Saturday's debit is charged at Friday's close of 41,000, which
        // moves the trigger to 40,221.61, and Monday's gap down to 40,150 crosses it
        let bars = bitcoin(&[
            ohlc(60_000.0, 60_000.0, 41_000.0, 41_000.0, 1e6),
            ohlc(40_150.0, 40_600.0, 40_000.0, 40_400.0, 1e6),
        ]);
        let clock = Clock::new(vec![6 * DAY, 7 * DAY, 8 * DAY, 11 * DAY]);
        let candles = Candles::on_clock(bars, clock)
            .with_funding(&[(9 * DAY + 8 * HOUR, 0.0005)])
            .with_tiers(maintained());
        let mut e = half_a_bitcoin(candles);
        e.step();
        e.step();
        let reporter = e.reporter().expect("reporting is on");
        assert!((reporter.ledger()[2].equity - 500.0).abs() < 1e-9);
        let close = reporter.fills().last().expect("the close");
        assert_eq!((close.tick, close.reason), (3, FillReason::Liquidation));
        assert!(
            (close.price - 40_221.61).abs() < 0.005,
            "closed at {}",
            close.price
        );
        let row = reporter.ledger()[3];
        assert!((row.costs.funding - 10.25).abs() < 1e-9);
        assert!((row.costs.liquidation - 100.55).abs() < 0.005);
    }

    #[test]
    fn a_long_bitcoin_and_short_ether_are_both_closed_at_the_close_point() {
        // the Specification's example: 10,000 of margin behind a long of 1 BTC and a
        // short of 20 ETH, on a bar where BTC falls to 54,000 and ETH rises to 3,300.
        // Equity on the path is 10,000 - 12,000s and the closing fee a constant 60,
        // so both close where s is 0.8283, at 55,030.00 and 3,248.50
        let mut e = bitcoin_and_ether((57_000.0, 3_150.0), Intrabar::Pessimistic);
        let last = e.step();
        let reporter = e.reporter().expect("reporting is on");
        let closes: Vec<(usize, f64)> = reporter
            .fills()
            .iter()
            .filter(|fill| fill.reason == FillReason::Liquidation)
            .map(|fill| (fill.instrument, fill.price))
            .collect();
        let [(0, btc), (1, eth)] = closes.as_slice() else {
            panic!("BTC and ETH closed, in that order: {closes:?}");
        };
        assert!((btc - 55_030.0).abs() < 0.005, "BTC closed at {btc}");
        assert!((eth - 3_248.50).abs() < 0.005, "ETH closed at {eth}");
        assert!((reporter.ledger()[2].costs.fees - 60.0).abs() < 0.005);
        assert_eq!(last.margin_cash, 0.0);
    }

    #[test]
    fn the_optimistic_setting_liquidates_bitcoin_and_ether_only_where_every_path_does() {
        // the corners are BTC at 54,000 beside ETH at its low of 3,000 and ETH at
        // 3,300 beside BTC at its high of 60,000, and equity is 4,000 at each and at
        // closes of 57,000 and 3,150: a path exists that never reaches the trigger,
        // and the account survives the bar. Closing at 55,000 and 3,250 every path
        // ends at zero, so it is closed at the synchronous path's clip
        let mut e = bitcoin_and_ether((57_000.0, 3_150.0), Intrabar::Optimistic);
        let last = e.step();
        assert_eq!(e.num_fills(), 2, "nothing but the two entries");
        assert!(
            (last.equity - 4_000.0).abs() < 1e-9,
            "equity {}",
            last.equity
        );
        let mut e = bitcoin_and_ether((55_000.0, 3_250.0), Intrabar::Optimistic);
        let last = e.step();
        let reporter = e.reporter().expect("reporting is on");
        let closes: Vec<f64> = reporter
            .fills()
            .iter()
            .filter(|fill| fill.reason == FillReason::Liquidation)
            .map(|fill| fill.price)
            .collect();
        let [btc, eth] = closes.as_slice() else {
            panic!("BTC and ETH closed: {closes:?}");
        };
        assert!((btc - 55_030.0).abs() < 0.005, "BTC closed at {btc}");
        assert!((eth - 3_248.50).abs() < 0.005, "ETH closed at {eth}");
        assert_eq!(last.margin_cash, 0.0);
    }

    /// The Specification's account at two instruments, 10,060 of margin buying 1
    /// BTC at 60,000 and selling 20 ETH at 3,000 for fees of 30 each, then a bar
    /// where BTC falls to 54,000 and ETH rises to 3,300 and they close at `closes`;
    /// returned after the entry, the bar to come.
    fn bitcoin_and_ether(closes: (f64, f64), intrabar: Intrabar) -> Engine {
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let rows = [
            vec![flat(60_000.0), flat(3_000.0)],
            vec![flat(60_000.0), flat(3_000.0)],
            vec![
                ohlc(60_000.0, 60_000.0, 54_000.0, closes.0, 1e6),
                ohlc(3_000.0, 3_300.0, 3_000.0, closes.1, 1e6),
            ],
        ]
        .concat();
        let at = |leverage| Instrument {
            market: Some(Market::Perp),
            tiers: Tiers::new(
                TierBasis::Notional,
                vec![Tier {
                    up_to: 1e12,
                    leverage,
                    rate: 0.0,
                    amount: 0.0,
                }],
            )
            .expect("a table"),
        };
        let bars = Candles::universe(rows, 2, None).with_instruments(vec![at(20.0), at(10.0)]);
        let config = EngineConfig {
            quote: 10_060.0,
            fee_taker: 0.0005,
            report: true,
            intrabar,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(1.0);
        let ioc = TimeInForce::Ioc;
        e.order_on(
            1,
            Side::Sell,
            20.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            ioc,
        );
        e.step();
        assert!(
            (e.reporter().expect("reporting is on").ledger()[1].equity - 10_000.0).abs() < 1e-9
        );
        e
    }

    #[test]
    fn an_account_past_its_trigger_at_the_opens_is_liquidated_there_under_either_setting() {
        // 99 of margin buys 50 of each of two perps at opens of 100, at 1%
        // maintenance and no cap: a requirement of 100 against equity of 99 where
        // every path starts, though both bars rise to close at 105 and every corner
        // stands clear. The optimistic way closes the account at the opens, as the
        // pessimistic way does
        let maintained = Instrument {
            market: Some(Market::Perp),
            tiers: Tiers::new(
                TierBasis::Notional,
                vec![Tier {
                    up_to: 1e12,
                    leverage: 0.0,
                    rate: 0.01,
                    amount: 0.0,
                }],
            )
            .expect("a table"),
        };
        for intrabar in [Intrabar::Pessimistic, Intrabar::Optimistic] {
            let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
            let rising = ohlc(100.0, 110.0, 99.9, 105.0, 1e6);
            let bars = Candles::universe(vec![flat, flat, rising, rising], 2, None)
                .with_instruments(vec![maintained.clone(), maintained.clone()]);
            let config = EngineConfig {
                quote: 99.0,
                report: true,
                intrabar,
                ..perp_cfg()
            };
            let mut e = Engine::new(bars, config);
            e.reset();
            for i in 0..2 {
                let (kind, ioc) = (OrderType::Market, TimeInForce::Ioc);
                e.order_on(i, Side::Buy, 50.0, kind, None, None, false, false, ioc);
            }
            let last = e.step();
            let closes: Vec<f64> = e
                .reporter()
                .expect("reporting is on")
                .fills()
                .iter()
                .filter(|fill| fill.reason == FillReason::Liquidation)
                .map(|fill| fill.price)
                .collect();
            assert_eq!(closes, [100.0, 100.0], "{intrabar:?}");
            assert_eq!(last.margin_cash, 0.0, "{intrabar:?}");
        }
    }

    #[test]
    fn a_flip_at_the_old_sides_extreme_tests_the_new_side_without_the_old() {
        // a long of 10 from 100 on 480 of margin, at 5% maintenance and no cap, is
        // sold 20 by a stop at the bar's low of 90: the long leaves there and the
        // short joins to run to the high of 120. Alone, the short's level is
        // 1,280 - 10.5p, clear at 120; with the long's requirement still beside it,
        // 1,235 - 10.5p, which closed the short at 117.62
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let tier = Tier {
            up_to: 1e12,
            leverage: 0.0,
            rate: 0.05,
            amount: 0.0,
        };
        let candles = Candles::new(vec![flat, flat, ohlc(100.0, 120.0, 90.0, 110.0, 1e6)])
            .with_tiers(Tiers::new(TierBasis::Notional, vec![tier]).expect("a table"));
        let config = EngineConfig {
            quote: 480.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.stop(Side::Sell, 20.0, 90.0, false);
        let last = e.step();
        assert_eq!(
            priced(&e),
            vec![(FillReason::Market, 100.0), (FillReason::Stop, 90.0)]
        );
        assert_eq!(last.position, -10.0);
        assert!(
            (last.margin_cash - 380.0).abs() < 1e-9,
            "kept {}",
            last.margin_cash
        );
    }

    #[test]
    fn a_reduce_onto_a_quantity_tiers_bound_is_read_in_the_tier_it_lies_in() {
        // 82 of margin behind a long of 3 from 100, the first 1.6 at 1% and above it
        // 5%; the bar falls to 50, which clips it for the 3 carried in at 76.49, and a
        // sale of 1.4 at the open leaves 1.6, in the first tier. A leg shrunk in
        // proportion holds 1.6000000000000003, which the second tier's 5% would read
        // past its trigger; held at the position's own size it keeps 1.2 at the low
        let flat = |price| ohlc(price, price, price, price, 1e6);
        let tiers = Tiers::new(
            TierBasis::Quantity,
            vec![
                Tier {
                    up_to: 1.6,
                    leverage: 20.0,
                    rate: 0.01,
                    amount: 0.0,
                },
                Tier {
                    up_to: 100.0,
                    leverage: 10.0,
                    rate: 0.05,
                    amount: 0.0,
                },
            ],
        )
        .expect("a table");
        let bars = Candles::new(vec![
            flat(100.0),
            flat(100.0),
            ohlc(100.0, 100.0, 50.0, 60.0, 1e6),
        ])
        .with_tiers(tiers);
        let config = EngineConfig {
            quote: 82.0,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(3.0);
        e.step();
        e.market_sell(1.4);
        let last = e.step();
        assert_eq!(last.position, 1.6);
        let row = e.reporter().expect("reporting is on").ledger()[2];
        assert!((row.equity - 18.0).abs() < 1e-9, "equity {}", row.equity);
        assert_eq!(row.costs.liquidation, 0.0);
    }
}
