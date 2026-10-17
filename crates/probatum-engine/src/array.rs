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
        }
    }

    /// The stop and take-profit of the round trip instrument `i` holds, `q` in the
    /// engine's terms, at the returns of the row that opened it from the average entry
    /// they are priced at, on the quoted price, each left out where the row sets none
    /// or it falls at or below zero.
    fn levels(&self, i: usize, q: f64) -> (Option<f64>, Option<f64>) {
        let Some(row) = self.opened[i] else {
            return (None, None);
        };
        let entry = self.units.dollars(self.priced[i]);
        let side = self.units.signed(q).signum();
        let at = |factor: f64| (factor > 0.0).then(|| self.units.price(entry * factor));
        let stop = self.stop.at(row, i, self.width);
        let take_profit = self.take_profit.at(row, i, self.width);
        (
            stop.and_then(|stop| at(1.0 - side * stop)),
            take_profit.and_then(|take_profit| at(1.0 + side * take_profit)),
        )
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
            self.legs[i] = match self.levels(i, q) {
                (None, None) => (None, None),
                (stop, take_profit) => engine.bracket(i, stop, take_profit),
            };
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
    }

    fn next(&mut self, state: &State, engine: &mut Engine) {
        let k = state.tick_index - self.first;
        // the brackets, instrument by instrument, before the row's target
        if self.stop != Barrier::None || self.take_profit != Barrier::None {
            for i in 0..self.width {
                self.bracket(i, engine);
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
    use crate::reporter::FillReason;
    use crate::strategy::Strategy;
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
        let config = EngineConfig {
            market: Market::Perp,
            quote,
            idle_cash: 0.0,
            fee_taker: 0.0,
            fee_maker: 0.0,
            slippage_bps: 0.0,
            budget: 1.0,
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
        // take-profit rests
        for stop in [1.0, 1.5] {
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
    fn a_nan_holds_and_the_last_row_is_never_read() {
        let mut e = engine(vec![at(100.0); 4], 10_000.0);
        let mut array = barriers(vec![0.5, NAN, NAN, 1.0], Sizing::Weight, NAN, NAN);
        play(&mut e, &mut array);
        assert_eq!(e.position(0), 50.0);
        assert_eq!(fills(&e).len(), 1);
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
