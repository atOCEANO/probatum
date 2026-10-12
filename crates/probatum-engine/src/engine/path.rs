//! The synchronous adverse path every perp held moves along, and the points on it
//! where the margin runs out (Specification, Liquidation). A path is cut into
//! pieces; on a piece every leg sits at the same fraction of its own move, so margin
//! equity is linear in that fraction and the maintenance requirement piecewise
//! linear, and each crossing is solved in closed form between the tier bounds a
//! leg's notional crosses, where the reference bisects (Decision 0043).

use probatum_core::{Market, TierBasis};

use super::Engine;
use crate::candles::printed;

/// A fill or a remainder smaller than this opens no leg, as it opens no round trip.
pub(super) const DUST: f64 = 1e-9;

/// One position, or a part of it, moving from `start` to `end` along a piece.
#[derive(Clone, Copy, Debug)]
pub(super) struct PathLeg {
    pub(super) i: usize,
    /// Signed: positive for a long.
    pub(super) q: f64,
    pub(super) entry: f64,
    pub(super) start: f64,
    pub(super) end: f64,
}

/// A path: its pieces, each a run of `legs`, the legs of one instrument together.
#[derive(Debug, Default)]
pub(super) struct Path {
    pub(super) legs: Vec<PathLeg>,
    pub(super) pieces: Vec<(usize, usize)>,
}

/// A point on a path: a piece, and the fraction of the way along it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Point {
    pub(super) piece: usize,
    pub(super) at: f64,
}

/// What a position did during the bar, for the check: a part carried into it
/// (`start` none, priced from the open) or opened during it at `start`, joining the
/// path at fraction `join`, and for the old side of a flip the fraction it left at,
/// its entry then the flip's price, since the wallet holds what it realized.
#[derive(Clone, Copy, Debug)]
pub(super) struct HeldLeg {
    pub(super) size: f64,
    pub(super) entry: f64,
    pub(super) start: Option<f64>,
    pub(super) join: f64,
    pub(super) leave: Option<f64>,
    pub(super) long: bool,
}

/// What a perp fill on instrument `i` did: its position from `before`, at average
/// entry `entry`, to `after`, at `price`, `opened` of it a new side.
#[derive(Clone, Copy, Debug)]
pub(super) struct FillNote {
    pub(super) i: usize,
    pub(super) before: f64,
    pub(super) entry: f64,
    pub(super) after: f64,
    pub(super) price: f64,
    pub(super) opened: f64,
}

/// A leg of the check's path: its instrument and signed size, and where along the
/// bar it joins and leaves.
#[derive(Clone, Copy, Debug)]
pub(super) struct CheckLeg {
    pub(super) i: usize,
    pub(super) q: f64,
    pub(super) entry: f64,
    pub(super) frac: f64,
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) leave: Option<f64>,
}

impl CheckLeg {
    /// Its price at fraction `s` of the bar: its start until it joins, then on its
    /// own segment from there to its end.
    fn price_at(&self, s: f64) -> f64 {
        if s <= self.frac || self.frac >= 1.0 {
            return self.start;
        }
        self.start + (s - self.frac) / (1.0 - self.frac) * (self.end - self.start)
    }

    /// Its prices at the two ends of the span from `a` to `b`. On the last span,
    /// which has no length, a leg that joined at the bar's end runs its whole move,
    /// so the new side of a flip at the old side's extreme still reaches its own.
    pub(super) fn ends(&self, a: f64, b: f64) -> (f64, f64) {
        if a == 1.0 && b == 1.0 && self.frac >= 1.0 {
            return (self.start, self.end);
        }
        (self.price_at(a), self.price_at(b))
    }
}

/// Margin equity, the maintenance requirement and the notional held, at a point.
#[derive(Clone, Copy, Debug)]
struct Level {
    equity: f64,
    maintenance: f64,
    notional: f64,
}

impl Engine {
    /// Whether instrument `i`'s tier table asks any maintenance.
    pub(super) fn has_maintenance(&self, i: usize) -> bool {
        self.maintains[i]
    }

    /// The maintenance a position of `size` and `notional` on instrument `i` keeps:
    /// its tier's rate on the notional, less the tier's amount on a notional table.
    pub(super) fn maintenance_of(&self, i: usize, size: f64, notional: f64) -> f64 {
        let tiers = self.candles.tiers(i);
        match tiers.tier(size, notional) {
            Some(tier) if tier.rate > 0.0 && notional > 0.0 => match tiers.basis() {
                TierBasis::Notional => notional * tier.rate - tier.amount,
                TierBasis::Quantity => notional * tier.rate,
            },
            _ => 0.0,
        }
    }

    /// The level on `legs` at fraction `u`, with `wallet` the margin wallet's cash.
    fn level(&self, legs: &[PathLeg], u: f64, wallet: f64) -> Level {
        let mut equity = wallet;
        let mut maintenance = 0.0;
        let mut notional = 0.0;
        let mut k = 0;
        while k < legs.len() {
            let i = legs[k].i;
            let (mut size, mut held) = (0.0, 0.0);
            while k < legs.len() && legs[k].i == i {
                let leg = legs[k];
                let price = leg.start + u * (leg.end - leg.start);
                equity += leg.q * (price - leg.entry);
                size += leg.q.abs();
                held += leg.q.abs() * price;
                k += 1;
            }
            if self.maintains[i] {
                maintenance += self.maintenance_of(i, size, held);
            }
            notional += held;
        }
        Level {
            equity,
            maintenance,
            notional,
        }
    }

    /// Margin equity less the maintenance requirement, with `wallet` the margin
    /// wallet's cash and every perp held at the price `at` gives it for its signed
    /// size.
    pub(super) fn margin_level(&self, wallet: f64, at: impl Fn(usize, f64) -> f64) -> f64 {
        let (mut equity, mut maintenance) = (wallet, 0.0);
        for (i, position) in self.account.positions.iter().enumerate() {
            let q = position.qty.get();
            if q == 0.0 || self.account.markets[i] != Market::Perp {
                continue;
            }
            let price = at(i, q);
            equity += q * (price - position.avg_entry.get());
            if self.maintains[i] {
                maintenance += self.maintenance_of(i, q.abs(), q.abs() * price);
            }
        }
        equity - maintenance
    }

    /// Whether the account stands clear of its trigger with every perp held at the
    /// price `at` gives it, by more than the rounding a path's own sums could leave,
    /// so a solve that reached that point would find no crossing there either. The
    /// path prices each leg across its bar and gap from its entry, so the rounding
    /// scales with those spans; and the legs a reduce shrinks can sum a hair past a
    /// quantity table's bound, where the next tier's rate takes the whole notional,
    /// so such a table is read a hair above the size.
    pub(super) fn clear_of_trigger(&self, wallet: f64, at: impl Fn(usize, f64) -> f64) -> bool {
        let (mut equity, mut maintenance, mut reach) = (wallet, 0.0, wallet.abs());
        for (i, position) in self.account.positions.iter().enumerate() {
            let q = position.qty.get();
            if q == 0.0 || self.account.markets[i] != Market::Perp {
                continue;
            }
            let (price, entry) = (at(i, q), position.avg_entry.get());
            equity += q * (price - entry);
            if self.maintains[i] {
                maintenance += self.maintenance_of(i, q.abs() * (1.0 + 1e-12), q.abs() * price);
            }
            // the largest of three, compared plainly: a printed bar holds no NaN for
            // `f64::max` to skip, and it is the mark alone where the bar is NaN
            let bar = self.candles.bar(self.tick, i);
            let mut span = self.marks[i].abs();
            if printed(&bar) {
                for x in [bar.high.abs(), bar.low.abs()] {
                    if x > span {
                        span = x;
                    }
                }
            }
            reach += q.abs() * (span + entry.abs());
        }
        equity - maintenance > 1e-12 * reach
    }

    /// Instrument `i`'s adverse extreme on the bar for a position of `q`, or its mark
    /// where it printed no bar.
    pub(super) fn adverse(&self, i: usize, q: f64) -> f64 {
        let bar = self.candles.bar(self.tick, i);
        if !printed(&bar) {
            self.marks[i]
        } else if q > 0.0 {
            bar.low
        } else {
            bar.high
        }
    }

    /// The first fraction on `legs` where equity falls to zero, less the
    /// requirement when `maintained`, or none. The requirement rises with every
    /// tier, so equity less it is concave on a piece and crosses at most once from
    /// above; between the bounds a leg's notional crosses it is linear, which makes
    /// the crossing exact by interpolation.
    fn crossing(&self, legs: &[PathLeg], wallet: f64, maintained: bool) -> Option<f64> {
        let gap = |u: f64| {
            let at = self.level(legs, u, wallet);
            if maintained {
                at.equity - at.maintenance
            } else {
                at.equity
            }
        };
        let first = gap(0.0);
        if first <= 0.0 {
            return Some(0.0);
        }
        let last = gap(1.0);
        if last > 0.0 {
            return None;
        }
        let mut bounds = Vec::new();
        if maintained {
            self.notional_bounds(legs, &mut bounds);
        }
        let (mut a, mut ga) = (0.0, first);
        for &b in bounds.iter().chain(std::iter::once(&1.0)) {
            let gb = if b == 1.0 { last } else { gap(b) };
            if gb <= 0.0 {
                return Some(a + ga * (b - a) / (ga - gb));
            }
            (a, ga) = (b, gb);
        }
        Some(1.0)
    }

    /// The fractions in (0, 1) at which an instrument's notional on `legs` crosses a
    /// bound of its notional tier table, sorted.
    fn notional_bounds(&self, legs: &[PathLeg], bounds: &mut Vec<f64>) {
        let mut k = 0;
        while k < legs.len() {
            let i = legs[k].i;
            let (mut from, mut moved) = (0.0, 0.0);
            while k < legs.len() && legs[k].i == i {
                from += legs[k].q.abs() * legs[k].start;
                moved += legs[k].q.abs() * (legs[k].end - legs[k].start);
                k += 1;
            }
            let tiers = self.candles.tiers(i);
            if moved == 0.0 || tiers.basis() != TierBasis::Notional {
                continue;
            }
            let rows = tiers.rows();
            for tier in &rows[..rows.len().saturating_sub(1)] {
                let u = (tier.up_to - from) / moved;
                if u > 0.0 && u < 1.0 {
                    bounds.push(u);
                }
            }
        }
        bounds.sort_by(f64::total_cmp);
    }

    /// The trigger and the clip point on `path`, the margin wallet holding
    /// `wallet(k)` on piece `k`, or none (The trigger, close and clip points). With
    /// maintenance the account leaves at its trigger; with none, from the point it
    /// stays at or past the close level, equity at the closing fee on every notional
    /// held, all the way to the trigger. A leg joining the check's path adds to that
    /// fee, and funding counted from the open moves the wallet, so the level can
    /// step where one piece meets the next.
    pub(super) fn solve(
        &self,
        path: &Path,
        wallet: impl Fn(usize) -> f64,
    ) -> Option<(Point, Point)> {
        let legs = |k: usize| {
            let (from, to) = path.pieces[k];
            &path.legs[from..to]
        };
        let maintained = path.legs.iter().any(|leg| self.has_maintenance(leg.i));
        let trigger = (0..path.pieces.len()).find_map(|k| {
            self.crossing(legs(k), wallet(k), maintained)
                .map(|at| Point { piece: k, at })
        })?;
        if maintained {
            return Some((trigger, trigger));
        }
        let fee = self.cost.fee_taker;
        let close = |k: usize, u: f64| {
            let at = self.level(legs(k), u, wallet(k));
            at.equity - fee * at.notional
        };
        let (mut k, mut s) = (trigger.piece, trigger.at);
        loop {
            let first = close(k, 0.0);
            if first > 0.0 {
                // the close level is linear on a piece and at or below zero at the
                // trigger, so it reaches zero once before it
                let at = (first / (first - close(k, 1.0))).min(s);
                return Some((trigger, Point { piece: k, at }));
            }
            if k == 0 || close(k - 1, 1.0) > 0.0 {
                return Some((trigger, Point { piece: k, at: 0.0 }));
            }
            (k, s) = (k - 1, 1.0);
        }
    }

    /// The path of the perps carried into the bar (The path): from each last mark to
    /// its open, then from the open to its adverse extreme, every one in step. A
    /// perp that printed no bar keeps its mark on both.
    pub(super) fn carried_path(&self, path: &mut Path) {
        path.legs.clear();
        path.pieces.clear();
        for gap in [true, false] {
            let from = path.legs.len();
            for (i, position) in self.account.positions.iter().enumerate() {
                let q = position.qty.get();
                if self.account.markets[i] != Market::Perp || q == 0.0 {
                    continue;
                }
                let bar = self.candles.bar(self.tick, i);
                let mark = self.marks[i];
                let (start, end) = if !printed(&bar) {
                    (mark, mark)
                } else if gap {
                    (mark, bar.open)
                } else {
                    (bar.open, if q > 0.0 { bar.low } else { bar.high })
                };
                path.legs.push(PathLeg {
                    i,
                    q,
                    entry: position.avg_entry.get(),
                    start,
                    end,
                });
            }
            path.pieces.push((from, path.legs.len()));
        }
    }

    /// Note what a perp fill did to its position, for the check's legs, which are
    /// built only on a bar whose check needs its whole path.
    pub(super) fn note(&mut self, note: FillNote) {
        self.flipped |= note.before * note.after < 0.0;
        self.notes.push(note);
    }

    /// The check's record of what each perp did during the bar: one leg carried into
    /// it, from the position the bar's first fill on it found, and then each fill
    /// noted, in turn.
    pub(super) fn build_legs(&mut self) {
        let notes = std::mem::take(&mut self.notes);
        for (i, legs) in self.legs.iter_mut().enumerate() {
            legs.clear();
            if self.account.markets[i] != Market::Perp {
                continue;
            }
            let position = &self.account.positions[i];
            let (q, entry) = notes
                .iter()
                .find(|note| note.i == i)
                .map_or((position.qty.get(), position.avg_entry.get()), |note| {
                    (note.before, note.entry)
                });
            if q != 0.0 {
                legs.push(HeldLeg {
                    size: q.abs(),
                    entry,
                    start: None,
                    join: 0.0,
                    leave: None,
                    long: q > 0.0,
                });
            }
        }
        for note in &notes {
            self.track(note.i, note.before, note.after, note.price, note.opened);
        }
        self.notes = notes;
    }

    /// Carry a perp fill on instrument `i` into the check's legs, the position going
    /// from `before` to `after` at `price`, `opened` of it a new side (The path). A
    /// reduce shrinks every held leg in proportion, as average-entry accounting does;
    /// a flip keeps the old side until the flip's fraction, measured from the flip's
    /// price; an opening fill joins where its price lies on its side's path, and a
    /// flip's new side where the old one left.
    pub(super) fn track(&mut self, i: usize, before: f64, after: f64, price: f64, opened: f64) {
        let mut legs = std::mem::take(&mut self.legs[i]);
        let mut floor = legs.iter().filter_map(|leg| leg.leave).fold(0.0, f64::max);
        let flipped = before * after < 0.0;
        if flipped {
            floor = self.fraction(i, before > 0.0, price);
            for leg in legs.iter_mut().filter(|leg| leg.leave.is_none()) {
                leg.entry = price;
                leg.leave = Some(floor);
            }
        } else if after == 0.0 {
            legs.retain(|leg| leg.leave.is_some());
        } else if after.abs() < before.abs() {
            // in proportion, the last held leg taking what the others leave, so the
            // legs sum to the position and a lone leg holds its size exactly
            let mut last = None;
            let mut rest = after.abs();
            for (k, leg) in legs.iter_mut().enumerate() {
                if leg.leave.is_none() {
                    leg.size = leg.size * after.abs() / before.abs();
                    rest -= leg.size;
                    last = Some(k);
                }
            }
            if let Some(k) = last {
                legs[k].size += rest;
            }
        }
        if opened > DUST && after != 0.0 {
            let join = if flipped {
                floor
            } else {
                self.fraction(i, after > 0.0, price).max(floor)
            };
            legs.push(HeldLeg {
                size: opened,
                entry: price,
                start: Some(price),
                join,
                leave: None,
                long: after > 0.0,
            });
        }
        self.legs[i] = legs;
    }

    /// The check's legs, from the record of what each perp did during the bar: a
    /// carried leg from its open and a leg opened during the bar from its fill
    /// price, each to its side's adverse extreme; a perp that printed no bar stays
    /// at its mark.
    pub(super) fn check_legs(&self, legs: &mut Vec<CheckLeg>) {
        legs.clear();
        for (i, held) in self.legs.iter().enumerate() {
            let bar = self.candles.bar(self.tick, i);
            for leg in held {
                let q = if leg.long { leg.size } else { -leg.size };
                let (frac, start, end) = if !printed(&bar) {
                    (0.0, self.marks[i], self.marks[i])
                } else {
                    let adverse = if leg.long { bar.low } else { bar.high };
                    (leg.join, leg.start.unwrap_or(bar.open), adverse)
                };
                legs.push(CheckLeg {
                    i,
                    q,
                    entry: leg.entry,
                    frac,
                    start,
                    end,
                    leave: leg.leave,
                });
            }
        }
    }

    /// The check's path from `legs`, cut wherever a leg joins or leaves, the spans
    /// of the bar each piece covers in `spans`. A leg that joined at the bar's end
    /// is tested on a last span of no length, which a leg that left there is not on.
    pub(super) fn check_path(
        legs: &[CheckLeg],
        cuts: &mut Vec<f64>,
        spans: &mut Vec<(f64, f64)>,
        path: &mut Path,
    ) {
        spans.clear();
        path.legs.clear();
        path.pieces.clear();
        cuts.clear();
        cuts.extend([0.0, 1.0]);
        cuts.extend(legs.iter().map(|leg| leg.frac));
        cuts.extend(legs.iter().filter_map(|leg| leg.leave));
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        spans.extend(cuts.windows(2).map(|pair| (pair[0], pair[1])));
        if legs.iter().any(|leg| leg.frac >= 1.0) {
            spans.push((1.0, 1.0));
        }
        for &(a, b) in spans.iter() {
            let from = path.legs.len();
            for leg in legs {
                if leg.frac <= a && leg.leave.is_none_or(|leave| leave >= b && leave > a) {
                    let (start, end) = leg.ends(a, b);
                    path.legs.push(PathLeg {
                        i: leg.i,
                        q: leg.q,
                        entry: leg.entry,
                        start,
                        end,
                    });
                }
            }
            path.pieces.push((from, path.legs.len()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CheckLeg, Path, PathLeg, Point};
    use crate::candles::{Candles, Instrument};
    use crate::engine::fixtures::{ohlc, perp_cfg};
    use crate::engine::{Engine, EngineConfig};
    use probatum_core::{Market, Tier, TierBasis, Tiers};

    /// Two perps at a fee of 1% on a bar of their own, B keeping 5% of its notional
    /// by size when `maintained`; the path is built by hand.
    fn engine(maintained: bool) -> Engine {
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 1e6);
        let tiers = if maintained {
            let tier = Tier {
                up_to: 1e12,
                leverage: 10.0,
                rate: 0.05,
                amount: 0.0,
            };
            Tiers::new(TierBasis::Quantity, vec![tier]).expect("a table")
        } else {
            Tiers::default()
        };
        let instruments = vec![
            Instrument {
                market: Some(Market::Perp),
                tiers: Tiers::default(),
            },
            Instrument {
                market: Some(Market::Perp),
                tiers,
            },
        ];
        let candles = Candles::universe(vec![flat, flat], 2, None).with_instruments(instruments);
        let config = EngineConfig {
            fee_taker: 0.01,
            ..perp_cfg()
        };
        Engine::new(candles, config)
    }

    /// A long of 1 A from 100 falling to 95 on the first piece and on to 90 on the
    /// second, where a long of 10 B joins at 95 and falls to 85.
    fn joining() -> Path {
        let a = |start, end| PathLeg {
            i: 0,
            q: 1.0,
            entry: 100.0,
            start,
            end,
        };
        let b = PathLeg {
            i: 1,
            q: 10.0,
            entry: 95.0,
            start: 95.0,
            end: 85.0,
        };
        Path {
            legs: vec![a(100.0, 95.0), a(95.0, 90.0), b],
            pieces: vec![(0, 1), (1, 3)],
        }
    }

    #[test]
    fn the_clip_stops_where_a_joining_leg_adds_its_fee_to_the_close_level() {
        // on 10 of margin equity is 10 - 5u and then 5 - 105u, zero a twenty-first of
        // the way along the second piece. The closing fee on 1,045 of notional takes
        // the close level from 4.05 at the end of the first piece to -5.45 where B
        // joins, so the account leaves at the start of the second
        let (trigger, clip) = engine(false)
            .solve(&joining(), |_| 10.0)
            .expect("liquidated");
        assert_eq!(trigger.piece, 1);
        assert!(
            (trigger.at - 5.0 / 105.0).abs() < 1e-12,
            "trigger at {}",
            trigger.at
        );
        assert_eq!(clip, Point { piece: 1, at: 0.0 });
    }

    #[test]
    fn the_clip_steps_back_across_a_piece_that_stays_past_the_close_level() {
        // on 5.5 the close level is already -0.45 where the first piece ends, so the
        // account leaves on it, where 4.5 - 4.95u reaches zero
        let (trigger, clip) = engine(false)
            .solve(&joining(), |_| 5.5)
            .expect("liquidated");
        assert_eq!(trigger.piece, 1);
        assert_eq!(clip.piece, 0);
        assert!((clip.at - 4.5 / 4.95).abs() < 1e-12, "clip at {}", clip.at);
    }

    #[test]
    fn a_leg_that_joins_with_its_maintenance_can_start_a_piece_past_the_trigger() {
        // B keeps 5% of 950 when it joins, 47.5, which takes 40 of margin from 35 over
        // its trigger at the end of the first piece to 12.5 under it at the start of
        // the second; with maintenance the account leaves at its trigger
        let (trigger, clip) = engine(true)
            .solve(&joining(), |_| 40.0)
            .expect("liquidated");
        assert_eq!(trigger, Point { piece: 1, at: 0.0 });
        assert_eq!(clip, trigger);
    }

    #[test]
    fn a_leg_that_joins_at_the_bars_end_runs_its_move_on_a_last_piece_of_no_length() {
        let legs = [
            CheckLeg {
                i: 0,
                q: 1.0,
                entry: 100.0,
                frac: 0.0,
                start: 100.0,
                end: 90.0,
                leave: None,
            },
            CheckLeg {
                i: 0,
                q: 1.0,
                entry: 90.0,
                frac: 1.0,
                start: 90.0,
                end: 85.0,
                leave: None,
            },
        ];
        let (mut cuts, mut spans, mut path) = (Vec::new(), Vec::new(), Path::default());
        Engine::check_path(&legs, &mut cuts, &mut spans, &mut path);
        assert_eq!(spans, vec![(0.0, 1.0), (1.0, 1.0)]);
        assert_eq!(path.pieces, vec![(0, 1), (1, 3)]);
        let ends: Vec<(f64, f64)> = path.legs.iter().map(|leg| (leg.start, leg.end)).collect();
        assert_eq!(ends, vec![(100.0, 90.0), (90.0, 90.0), (90.0, 85.0)]);
    }
}
