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
const DUST: f64 = 1e-9;

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
        self.candles
            .tiers(i)
            .rows()
            .iter()
            .any(|tier| tier.rate > 0.0)
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
            maintenance += self.maintenance_of(i, size, held);
            notional += held;
        }
        Level {
            equity,
            maintenance,
            notional,
        }
    }

    /// Whether margin equity is at or below the maintenance requirement, or at or
    /// below zero without one, with `legs` where they start.
    pub(super) fn at_trigger(&self, legs: &[PathLeg], wallet: f64) -> bool {
        let at = self.level(legs, 0.0, wallet);
        at.equity - at.maintenance <= 0.0
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

    /// The trigger and the clip point on `path` for `wallet`, or none (The trigger,
    /// close and clip points). With maintenance the account leaves at its trigger;
    /// with none, from the point it stays at or past the close level, equity at the
    /// closing fee on every notional held, all the way to the trigger. A leg joining
    /// the check's path adds to that fee, so the level can step down where one piece
    /// meets the next.
    pub(super) fn solve(&self, path: &Path, wallet: f64) -> Option<(Point, Point)> {
        let legs = |k: usize| {
            let (from, to) = path.pieces[k];
            &path.legs[from..to]
        };
        let maintained = path.legs.iter().any(|leg| self.has_maintenance(leg.i));
        let trigger = (0..path.pieces.len()).find_map(|k| {
            self.crossing(legs(k), wallet, maintained)
                .map(|at| Point { piece: k, at })
        })?;
        if maintained {
            return Some((trigger, trigger));
        }
        let fee = self.cost.fee_taker;
        let close = |k: usize, u: f64| {
            let at = self.level(legs(k), u, wallet);
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

    /// Start the check's record of each perp held: one leg carried into the bar.
    pub(super) fn start_legs(&mut self) {
        for (i, legs) in self.legs.iter_mut().enumerate() {
            legs.clear();
            let position = &self.account.positions[i];
            let q = position.qty.get();
            if self.account.markets[i] == Market::Perp && q != 0.0 {
                legs.push(HeldLeg {
                    size: q.abs(),
                    entry: position.avg_entry.get(),
                    start: None,
                    join: 0.0,
                    leave: None,
                    long: q > 0.0,
                });
            }
        }
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
            for leg in legs.iter_mut().filter(|leg| leg.leave.is_none()) {
                leg.size *= after.abs() / before.abs();
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
    /// is tested on a last span of no length.
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
                if leg.frac <= a && leg.leave.is_none_or(|leave| leave >= b) {
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
