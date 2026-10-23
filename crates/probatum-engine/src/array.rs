//! A run decided in advance as an array: each row the values the account asks of its
//! instruments at that bar's close, weights of equity or the positions wanted, a NaN
//! holding, and every round trip given a bracket at returns from its average entry
//! (Specification, An array; Barriers on an array).

use probatum_core::{OrderId, State};

use crate::engine::Engine;
use crate::inverse::Units;
use crate::strategy::Strategy;

/// The unit an array's values are in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sizing {
    /// Weights of equity, reached as `Engine::target` reaches them.
    Weight,
    /// The position wanted of each instrument, reached as
    /// `Engine::target_quantities` reaches it.
    Quantity,
}

/// A barrier's return from the average entry: none, one for every row, or one per row
/// and instrument, laid out as the array's rows are, NaN where a row sets none.
#[derive(Clone, Debug, PartialEq)]
pub enum Barrier {
    None,
    Every(f64),
    Rows(Vec<f64>),
}

impl Barrier {
    fn at(&self, row: usize, i: usize, width: usize) -> Option<f64> {
        let value = match self {
            Barrier::None => return None,
            Barrier::Every(value) => *value,
            Barrier::Rows(values) => values[row * width + i],
        };
        (!value.is_nan()).then_some(value)
    }
}

/// An array run for `strategy::run_from` to play: `rows[k * width + i]`, in the
/// engine's terms, is what the account asks of instrument `i` at the close of the
/// `k`th bar from the one it opened at.
pub struct Array {
    rows: Vec<f64>,
    width: usize,
    sizing: Sizing,
    tolerance: f64,
    stop: Barrier,
    take_profit: Barrier,
    units: Units,
    /// The bar the account opened at, which the rows count from.
    first: usize,
    /// Each position at the last close.
    held: Vec<f64>,
    /// The row whose decision last named each instrument, and the one whose decision
    /// opened the round trip it holds.
    decided: Vec<Option<usize>>,
    opened: Vec<Option<usize>>,
    /// Each round trip's stop and take-profit, and the average entry they were
    /// priced at.
    legs: Vec<(Option<OrderId>, Option<OrderId>)>,
    priced: Vec<f64>,
    /// The side a barrier filled on, 1.0 or -1.0, and 0.0 where none holds the
    /// instrument flat.
    latched: Vec<f64>,
    /// The instruments the row names and their values, gathered for its target.
    named: Vec<(usize, f64)>,
    /// The rows, counted from the opening, that a validation's window after a path's
    /// first starts on, and those whose close closes every position instead of
    /// deciding, ascending (Specification, Paths and seams).
    seams: Vec<usize>,
    flattens: Vec<usize>,
    /// The legs of each round trip resting at a flattening row's close, which the
    /// seam after it places again on what the close left open.
    carried: Option<Vec<(bool, bool)>>,
}

/// The stop and take-profit of a round trip of `q`, in the engine's terms, at the
/// returns `stop` and `take_profit` from the average entry `entry`, on the quoted price
/// of `units`, each left out where it is None or falls at or below zero.
pub(crate) fn levels(
    units: Units,
    entry: f64,
    q: f64,
    stop: Option<f64>,
    take_profit: Option<f64>,
) -> (Option<f64>, Option<f64>) {
    let entry = units.dollars(entry);
    let side = units.signed(q).signum();
    let at = |factor: f64| (factor > 0.0).then(|| units.price(entry * factor));
    (
        stop.and_then(|stop| at(1.0 - side * stop)),
        take_profit.and_then(|take_profit| at(1.0 + side * take_profit)),
    )
}

impl Array {
    /// The run of `rows` over `width` instruments, a row for every bar from the one
    /// the account opens at, each value in the engine's terms in the unit `sizing`
    /// names, each target held to `tolerance`, with the barriers `stop` and
    /// `take_profit` read on the quoted prices of `units`. Panics unless the rows are
    /// whole and a barrier of rows has one return for each value.
    pub fn new(
        rows: Vec<f64>,
        width: usize,
        sizing: Sizing,
        tolerance: f64,
        stop: Barrier,
        take_profit: Barrier,
        units: Units,
    ) -> Array {
        assert!(
            width > 0 && rows.len() % width == 0,
            "whole rows of the instruments"
        );
        for barrier in [&stop, &take_profit] {
            if let Barrier::Rows(values) = barrier {
                assert_eq!(values.len(), rows.len(), "a return for each value");
            }
        }
        Array {
            rows,
            width,
            sizing,
            tolerance,
            stop,
            take_profit,
            units,
            first: 0,
            held: vec![0.0; width],
            decided: vec![None; width],
            opened: vec![None; width],
            legs: vec![(None, None); width],
            priced: vec![f64::NAN; width],
            latched: vec![0.0; width],
            named: Vec::with_capacity(width),
            seams: Vec::new(),
            flattens: Vec::new(),
            carried: None,
        }
    }

    /// The run with a validation's seams: at each row of `seams`, counted from the
    /// opening, every order not yet filled is cancelled and the targets forgotten, and
    /// each round trip held places again at its average entry the legs of its bracket
    /// still resting, or after a flattening row those resting at its close, its latch
    /// kept, before the row is read; each row of `flattens` closes every position
    /// instead of being read (Specification, Paths and seams).
    pub fn seamed(mut self, mut seams: Vec<usize>, mut flattens: Vec<usize>) -> Array {
        seams.sort_unstable();
        flattens.sort_unstable();
        self.seams = seams;
        self.flattens = flattens;
        self
    }

    /// The stop and take-profit of the round trip instrument `i` holds, `q` in the
    /// engine's terms, at the returns of the row that opened it from the average entry
    /// they are priced at.
    fn levels(&self, i: usize, q: f64) -> (Option<f64>, Option<f64>) {
        let Some(row) = self.opened[i] else {
            return (None, None);
        };
        levels(
            self.units,
            self.priced[i],
            q,
            self.stop.at(row, i, self.width),
            self.take_profit.at(row, i, self.width),
        )
    }

    /// Place instrument `i`'s bracket on the round trip it holds, `q`, at the levels
    /// it is priced at, those of its legs `kept` names alone, returning its legs.
    fn placed(
        &self,
        i: usize,
        q: f64,
        engine: &mut Engine,
        kept: (bool, bool),
    ) -> (Option<OrderId>, Option<OrderId>) {
        let (stop, take_profit) = self.levels(i, q);
        match (stop.filter(|_| kept.0), take_profit.filter(|_| kept.1)) {
            (None, None) => (None, None),
            (stop, take_profit) => engine.bracket(i, stop, take_profit),
        }
    }

    /// Which of each round trip's legs still rest.
    fn resting(&self, engine: &Engine) -> Vec<(bool, bool)> {
        let rests = |leg: Option<OrderId>| {
            leg.is_some_and(|id| engine.resting().any(|order| order.id == id))
        };
        self.legs
            .iter()
            .map(|&(stop, take_profit)| (rests(stop), rests(take_profit)))
            .collect()
    }

    /// See to instrument `i`'s bracket at this close: hold it flat from a barrier's
    /// fill, place one on a round trip that opened on this bar and move one whose
    /// average entry a fill moved.
    fn bracket(&mut self, i: usize, engine: &mut Engine) {
        let q = engine.position(i);
        let (stop, take_profit) = self.legs[i];
        let bracketed = stop.is_some() || take_profit.is_some();
        if bracketed && engine.leg_filled(i) {
            self.latched[i] = self.held[i].signum();
        }
        if q == 0.0 {
            self.legs[i] = (None, None);
        } else if self.held[i] == 0.0 || self.held[i] * q < 0.0 {
            self.opened[i] = self.decided[i];
            self.priced[i] = engine.entry(i);
            self.legs[i] = self.placed(i, q, engine, (true, true));
        } else if bracketed && engine.entry(i) != self.priced[i] {
            self.priced[i] = engine.entry(i);
            let (to_stop, to_take_profit) = self.levels(i, q);
            if let Some(id) = stop {
                engine.replace(id, None, None, to_stop);
            }
            if let Some(id) = take_profit {
                engine.replace(id, None, to_take_profit, None);
            }
        }
        self.held[i] = q;
    }
}

impl Strategy for Array {
    fn init(&mut self, state: &State) {
        self.first = state.tick_index;
        self.held.fill(0.0);
        self.decided.fill(None);
        self.opened.fill(None);
        self.legs.fill((None, None));
        self.priced.fill(f64::NAN);
        self.latched.fill(0.0);
        self.carried = None;
    }

    fn next(&mut self, state: &State, engine: &mut Engine) {
        let k = state.tick_index - self.first;
        let barred = self.stop != Barrier::None || self.take_profit != Barrier::None;
        // the brackets, instrument by instrument, before the row's target
        if barred {
            for i in 0..self.width {
                self.bracket(i, engine);
            }
        }
        if self.flattens.binary_search(&k).is_ok() {
            self.carried = Some(self.resting(engine));
            engine.seam(true);
            self.legs.fill((None, None));
            return;
        }
        if self.seams.binary_search(&k).is_ok() {
            // the legs still resting are placed again alone, so one its partner's fill
            // cancelled stays cancelled
            let kept = match self.carried.take() {
                Some(kept) => kept,
                None => self.resting(engine),
            };
            engine.seam(false);
            for (i, &kept) in kept.iter().enumerate() {
                self.legs[i] = (None, None);
                let q = engine.position(i);
                if barred && q != 0.0 {
                    self.priced[i] = engine.entry(i);
                    self.legs[i] = self.placed(i, q, engine, kept);
                }
            }
        }
        self.named.clear();
        for i in 0..self.width {
            let value = self.rows[k * self.width + i];
            if value.is_nan() {
                continue;
            }
            if self.latched[i] != 0.0 {
                if value != 0.0 && value.signum() == self.latched[i] {
                    continue;
                }
                self.latched[i] = 0.0;
            }
            self.named.push((i, value));
            self.decided[i] = Some(k);
        }
        if !self.named.is_empty() {
            // a negative value on a spot instrument was refused before the run
            let _ = match self.sizing {
                Sizing::Weight => engine.target(&self.named, self.tolerance, false),
                Sizing::Quantity => engine.target_quantities(&self.named, self.tolerance, false),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Array, Barrier, Sizing};
    use crate::candles::Candles;
    use crate::engine::{Engine, EngineConfig, Intrabar, Liquidation};
    use crate::execution::{Impact, Limit, Spread};
    use crate::inverse::Units;
    use crate::reporter::{EventKind, EventReason, FillReason};
    use crate::strategy::{run_from, Strategy};
    use probatum_core::{Candle, Market, OrderType};

    const NAN: f64 = f64::NAN;

    fn at(price: f64) -> Candle {
        bar(price, price, price, price)
    }

    fn bar(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle {
            open,
            high,
            low,
            close,
            volume: 1e9,
        }
    }

    fn engine(bars: Vec<Candle>, quote: f64) -> Engine {
        budgeted(bars, quote, 1.0)
    }

    /// The engine whose fills on each bar take at most `budget` of its trailing volume.
    fn budgeted(bars: Vec<Candle>, quote: f64, budget: f64) -> Engine {
        let config = EngineConfig {
            market: Market::Perp,
            quote,
            idle_cash: 0.0,
            fee_taker: 0.0,
            fee_maker: 0.0,
            slippage_bps: 0.0,
            budget,
            window: 24,
            max_open_orders: 8,
            report: true,
            max_leverage: 0.0,
            impact: 0.0,
            funding_rate: 0.0,
            funding_interval: 0,
            intrabar: Intrabar::Pessimistic,
            liquidation: Liquidation::Takeover,
            impact_model: Impact::Linear,
            spread: Spread::None,
            limit: Limit::Touch,
            sizes: false,
            liquidation_fee: None,
        };
        Engine::new(Candles::new(bars), config)
    }

    fn barriers(rows: Vec<f64>, sizing: Sizing, stop: f64, take_profit: f64) -> Array {
        let barrier = |value: f64| {
            if value.is_nan() {
                Barrier::None
            } else {
                Barrier::Every(value)
            }
        };
        let (stop, take_profit) = (barrier(stop), barrier(take_profit));
        Array::new(rows, 1, sizing, 0.0, stop, take_profit, Units::Linear)
    }

    /// The array through the engine, with each resting exit's kind and price after
    /// each close's decisions.
    fn play(e: &mut Engine, array: &mut Array) -> Vec<Vec<(OrderType, f64)>> {
        let mut state = e.reset();
        array.init(&state);
        let mut seen = Vec::new();
        while !e.done() {
            array.next(&state, e);
            let resting = e.current_state().open_orders.into_iter();
            seen.push(
                resting
                    .map(|order| {
                        let price = order.trigger.or(order.price).expect("an exit's price");
                        (order.kind, price.get())
                    })
                    .collect(),
            );
            state = e.step();
        }
        seen
    }

    fn fills(e: &Engine) -> Vec<(usize, f64, f64, FillReason)> {
        let reporter = e.reporter().expect("reporting is on");
        reporter
            .fills()
            .iter()
            .map(|f| (f.tick, f.side.sign() * f.size, f.price, f.reason))
            .collect()
    }

    #[test]
    fn a_long_rests_its_barriers_from_its_entry_and_moves_them_when_an_add_moves_it() {
        // 5 BTC entered at 20,000 rest a stop at 19,200 and a take-profit at 21,600
        // from the close of the bar the entry filled on, and 5 more at 21,000, an
        // average entry of 20,500, move them to 19,680 and 22,140
        let mut e = engine(
            vec![at(20_000.0), at(20_000.0), at(21_000.0), at(21_000.0)],
            100_000.0,
        );
        let mut array = barriers(vec![1.0, 2.0, NAN, NAN], Sizing::Weight, 0.04, 0.08);
        let seen = play(&mut e, &mut array);
        assert!(seen[0].is_empty());
        assert_eq!(
            seen[1],
            vec![(OrderType::Stop, 19_200.0), (OrderType::Limit, 21_600.0)]
        );
        assert_eq!(
            seen[2],
            vec![(OrderType::Stop, 19_680.0), (OrderType::Limit, 22_140.0)]
        );
        assert_eq!(e.position(0), 10.0);
    }

    #[test]
    fn a_short_rests_its_stop_above_its_entry_and_its_take_profit_below() {
        let mut e = engine(vec![at(20_000.0), at(20_000.0), at(20_000.0)], 100_000.0);
        let mut array = barriers(vec![-1.0, NAN, NAN], Sizing::Weight, 0.04, 0.08);
        let seen = play(&mut e, &mut array);
        assert_eq!(
            seen[1],
            vec![(OrderType::Stop, 20_800.0), (OrderType::Limit, 18_400.0)]
        );
    }

    #[test]
    fn a_barrier_at_or_below_zero_is_not_placed() {
        // a long's stop of 1 lies at zero and one of 1.5 below it, so only its
        // take-profit rests, as it does with no stop at all
        for stop in [1.0, 1.5, NAN] {
            let mut e = engine(vec![at(100.0), at(100.0), at(100.0)], 10_000.0);
            let mut array = barriers(vec![1.0, NAN, NAN], Sizing::Weight, stop, 0.25);
            let seen = play(&mut e, &mut array);
            assert_eq!(seen[1], vec![(OrderType::Limit, 125.0)]);
        }
    }

    #[test]
    fn a_barrier_that_fills_holds_its_instrument_flat_until_a_row_asks_for_zero() {
        // the stop at 96 fills on bar 2 and leaves 9,600; the rows of 1.0 after it buy
        // nothing until the row of zero on bar 4 lets the one on bar 5 buy 96 again,
        // filled on bar 6
        let mut bars = vec![at(100.0), at(100.0), bar(100.0, 100.0, 95.0, 100.0)];
        bars.extend([at(100.0); 5]);
        let mut e = engine(bars, 10_000.0);
        let rows = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, NAN, NAN];
        let mut array = barriers(rows, Sizing::Weight, 0.04, NAN);
        play(&mut e, &mut array);
        assert_eq!(
            fills(&e),
            vec![
                (1, 100.0, 100.0, FillReason::Rebalance),
                (2, -100.0, 96.0, FillReason::Stop),
                (6, 96.0, 100.0, FillReason::Rebalance),
            ]
        );
    }

    #[test]
    fn an_array_opened_part_way_reads_its_first_row_at_the_close_it_opened_at() {
        // the account opens at bar 2, so row 0's weight of 0.5 is decided at bar 2's
        // close and fills at bar 3's open
        let mut e = engine(vec![at(100.0); 5], 10_000.0);
        let mut array = barriers(vec![0.5, NAN, NAN], Sizing::Weight, NAN, NAN);
        run_from(&mut e, &mut array, 2);
        assert_eq!(fills(&e), vec![(3, 50.0, 100.0, FillReason::Rebalance)]);
    }

    #[test]
    fn a_nan_holds_and_the_last_row_is_never_read() {
        let mut e = engine(vec![at(100.0); 4], 10_000.0);
        let mut array = barriers(vec![0.5, NAN, NAN, 1.0], Sizing::Weight, NAN, NAN);
        play(&mut e, &mut array);
        assert_eq!(e.position(0), 50.0);
        assert_eq!(fills(&e).len(), 1);
    }

    #[test]
    fn a_seam_places_the_inherited_stop_again_and_a_flattening_row_closes_instead() {
        // a weight of 1.0 at row 2 fills at row 3's open of 100, its stop resting at 95
        // from row 3's close; window 2 starts at row 6, whose seam cancels the stop
        // and places it again at 95 before the row is read
        let rows = vec![NAN, NAN, 1.0, NAN, NAN, NAN, 1.0, 1.0, 1.0, 1.0];
        let mut e = engine(vec![at(100.0); 10], 10_000.0);
        let mut array = barriers(rows.clone(), Sizing::Weight, 0.05, NAN).seamed(vec![6], vec![]);
        let seen = play(&mut e, &mut array);
        assert_eq!(seen[5], vec![(OrderType::Stop, 95.0)]);
        assert_eq!(seen[6], vec![(OrderType::Stop, 95.0)]);
        let events = e.reporter().expect("reporting is on").events();
        let seam: Vec<(usize, EventKind)> = events
            .iter()
            .filter(|event| event.reason == EventReason::Seam)
            .map(|event| (event.tick, event.kind))
            .collect();
        assert_eq!(seam, vec![(6, EventKind::Cancelled)]);
        assert_eq!(fills(&e), vec![(3, 100.0, 100.0, FillReason::Rebalance)]);
        // with row 5 flattening, its close sells the position at row 6's open, the seam
        // places nothing on the instrument it left flat, and row 6 buys it again at row
        // 7's open
        let mut e = engine(vec![at(100.0); 10], 10_000.0);
        let mut array = barriers(rows, Sizing::Weight, 0.05, NAN).seamed(vec![6], vec![5]);
        let seen = play(&mut e, &mut array);
        assert!(seen[6].is_empty());
        assert_eq!(
            fills(&e),
            vec![
                (3, 100.0, 100.0, FillReason::Rebalance),
                (6, -100.0, 100.0, FillReason::Seam),
                (7, 100.0, 100.0, FillReason::Rebalance),
            ]
        );
    }

    #[test]
    fn a_stop_that_fills_before_a_seam_holds_the_next_window_flat() {
        // the stop at 95 fills on row 5, and window 2's rows of 1.0 from row 6 buy
        // nothing
        let mut bars = vec![at(100.0); 5];
        bars.push(bar(100.0, 100.0, 90.0, 92.0));
        bars.extend([at(92.0); 4]);
        let mut e = engine(bars, 10_000.0);
        let rows = vec![NAN, NAN, 1.0, NAN, NAN, NAN, 1.0, 1.0, 1.0, 1.0];
        let mut array = barriers(rows, Sizing::Weight, 0.05, NAN).seamed(vec![6], vec![]);
        play(&mut e, &mut array);
        assert_eq!(
            fills(&e),
            vec![
                (3, 100.0, 100.0, FillReason::Rebalance),
                (5, -100.0, 95.0, FillReason::Stop),
            ]
        );
    }

    #[test]
    fn a_seam_places_again_only_the_legs_still_resting() {
        // 4 a bar fill 10 in three bars at 100, the bracket resting at 95 and 110; the
        // stop fills 4 on row 4, which cancels the take-profit, and its remainder rests
        // to the seam at row 5, which places the stop again alone, so row 6's 111 takes
        // no profit
        let mut bars = vec![at(100.0); 4];
        bars.push(bar(100.0, 100.0, 94.0, 96.0));
        bars.push(at(100.0));
        bars.extend([at(111.0); 3]);
        let mut e = budgeted(bars, 10_000.0, 4e-9);
        let array = barriers(vec![10.0; 9], Sizing::Quantity, 0.05, 0.10);
        let mut array = array.seamed(vec![5], vec![]);
        let seen = play(&mut e, &mut array);
        assert_eq!(seen[4], vec![(OrderType::Stop, 95.0)]);
        assert_eq!(seen[5], vec![(OrderType::Stop, 95.0)]);
        let held: Vec<(usize, f64, FillReason)> =
            fills(&e).iter().map(|f| (f.0, f.1, f.3)).collect();
        assert_eq!(
            held,
            vec![
                (1, 4.0, FillReason::Rebalance),
                (2, 4.0, FillReason::Rebalance),
                (3, 2.0, FillReason::Rebalance),
                (4, -4.0, FillReason::Stop),
            ]
        );
    }

    #[test]
    fn a_seam_after_a_flattening_row_places_again_the_legs_resting_at_its_close() {
        // 4 a bar fill 10 in three bars at 100, the bracket resting at 95 and 125; row
        // 4's close cancels it and sells the 10, of which row 5's open fills 4, and the
        // seam at row 5 places both legs again on the 6 left
        let mut e = budgeted(vec![at(100.0); 7], 10_000.0, 4e-9);
        let array = barriers(vec![10.0; 7], Sizing::Quantity, 0.05, 0.25);
        let mut array = array.seamed(vec![5], vec![4]);
        let seen = play(&mut e, &mut array);
        assert!(seen[4].is_empty());
        assert_eq!(
            seen[5],
            vec![(OrderType::Stop, 95.0), (OrderType::Limit, 125.0)]
        );
        let held: Vec<(usize, f64, FillReason)> =
            fills(&e).iter().map(|f| (f.0, f.1, f.3)).collect();
        assert_eq!(
            held,
            vec![
                (1, 4.0, FillReason::Rebalance),
                (2, 4.0, FillReason::Rebalance),
                (3, 2.0, FillReason::Rebalance),
                (5, -4.0, FillReason::Seam),
                (6, 4.0, FillReason::Rebalance),
            ]
        );
    }

    #[test]
    fn quantities_reach_the_position_each_row_wants_with_no_drift_once_filled() {
        let mut e = engine(vec![at(100.0); 5], 10_000.0);
        let mut array = barriers(vec![2.0, NAN, 0.5, NAN, NAN], Sizing::Quantity, NAN, NAN);
        play(&mut e, &mut array);
        let held: Vec<(usize, f64)> = fills(&e).iter().map(|f| (f.0, f.1)).collect();
        assert_eq!(held, vec![(1, 2.0), (3, -1.5)]);
        let legs = e.reporter().expect("reporting is on").legs();
        assert!(legs[0].drift.is_nan());
        assert_eq!(
            legs.iter().skip(1).map(|leg| leg.drift).collect::<Vec<_>>(),
            [0.0; 4]
        );
    }
}
