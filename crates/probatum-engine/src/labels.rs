//! Labels: what one trade of a notional decided at a row's close makes through the
//! fill model, each from a fresh account holding the notional, its bracket placed as
//! an array run places one (Specification, Labels).

use probatum_core::{OrderType, Side, TimeInForce};
use rayon::prelude::*;

use crate::array::levels;
use crate::engine::Engine;
use crate::inverse::Units;
use crate::reporter::{FillReason, Record};

/// What left a label's account flat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    Stop,
    TakeProfit,
    /// The close at the label's horizon.
    Horizon,
    Delist,
    Liquidation,
}

/// What one label's trade made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Label {
    /// Its entry filled nothing by its last row.
    None,
    /// It was still open at the last row.
    Censored,
    /// It was flat again at the close of row `end`, its equity there `ret` of the
    /// notional above it, its entry having filled on row `entered`.
    Closed {
        ret: f64,
        exit: Exit,
        entered: usize,
        end: usize,
    },
}

/// One label's trade on `instrument`: decided at the close of row `start`, closed at
/// the close of row `last` where it is still open and again at each close after while
/// some is left, with the returns `stop` and `take_profit` from its average entry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trade {
    pub instrument: usize,
    pub start: usize,
    pub last: usize,
    pub stop: Option<f64>,
    pub take_profit: Option<f64>,
}

/// What `trade` makes on `engine`, whose account opens holding `notional` and records
/// its fills: at the close of its row a value order of `notional` on `side`, in the
/// venue's terms of `units`, and on an inverse contract one of the dollars `notional`
/// coins are worth at that close; its bracket at the close of the row its entry
/// filled on; and a reduce-only market order closing what is left at the close of
/// its last row, and again at each close after while some is left and none waits.
pub fn label(engine: &mut Engine, trade: Trade, notional: f64, side: Side, units: Units) -> Label {
    let i = trade.instrument;
    engine.reset_at(trade.start);
    let side = units.side(side);
    let tif = TimeInForce::Ioc;
    match units {
        Units::Inverse(value) => {
            let dollars = notional * units.dollars(engine.mark(i));
            let size = units.size(dollars / value);
            engine.order_on(
                i,
                side,
                size,
                OrderType::Market,
                None,
                None,
                false,
                false,
                tif,
            )
        }
        Units::Linear => engine.order_value(i, side, notional, false, tif),
    };
    let mut entered = None;
    while !engine.done() {
        engine.step();
        let r = engine.tick();
        let first = match entered {
            Some(first) => first,
            None if engine.num_fills() > 0 => *entered.insert(r),
            // the entry waits only for its instrument's next bar
            None if r >= trade.last || engine.waiting().next().is_none() => return Label::None,
            None => continue,
        };
        let q = engine.position(i);
        if q == 0.0 {
            return Label::Closed {
                ret: (engine.equity() - notional) / notional,
                exit: exit(engine),
                entered: first,
                end: r,
            };
        }
        if r == first {
            let (stop, take_profit) =
                levels(units, engine.entry(i), q, trade.stop, trade.take_profit);
            if stop.is_some() || take_profit.is_some() {
                engine.bracket(i, stop, take_profit);
            }
        }
        if r >= trade.last && engine.waiting().next().is_none() {
            engine.close_on(i);
        }
    }
    Label::Censored
}

/// The exit the last fill names, a sweep of dust taking the exit of the fill before
/// it, which left the dust.
fn exit(engine: &Engine) -> Exit {
    let fills = engine
        .reporter()
        .expect("a label records its fills")
        .fills();
    let last = fills
        .iter()
        .rev()
        .find(|fill| fill.reason != FillReason::Dust)
        .expect("an entry filled");
    match last.reason {
        FillReason::Stop => Exit::Stop,
        FillReason::TakeProfit => Exit::TakeProfit,
        FillReason::Delist => Exit::Delist,
        FillReason::Liquidation => Exit::Liquidation,
        // the close at the horizon, or the entry itself where its position was dust
        FillReason::Market => Exit::Horizon,
        reason => unreachable!("a label places no order that fills as {reason:?}"),
    }
}

/// Each of `trades` as `label` makes it, on engines of `engine`'s bars and settings
/// that record their fills, in parallel, in the order given.
pub fn labels(
    engine: &Engine,
    trades: &[Trade],
    notional: f64,
    side: Side,
    units: Units,
) -> Vec<Label> {
    trades
        .par_iter()
        .map_init(
            || engine.fresh(Record::Full),
            |each, &trade| label(each, trade, notional, side, units),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{label, labels, Exit, Label, Trade};
    use crate::candles::{Candles, Instrument};
    use crate::clock::Clock;
    use crate::engine::{Engine, EngineConfig, Intrabar, Liquidation};
    use crate::execution::{Impact, Limit, Spread};
    use crate::inverse::Units;
    use probatum_core::{Candle, Market, Side};

    const ABSENT: Candle = Candle {
        open: f64::NAN,
        high: f64::NAN,
        low: f64::NAN,
        close: f64::NAN,
        volume: f64::NAN,
    };

    fn bar(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle {
            open,
            high,
            low,
            close,
            volume: 1e9,
        }
    }

    fn engine(bars: Vec<Candle>) -> Engine {
        Engine::new(Candles::new(bars), config())
    }

    fn config() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 10_000.0,
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
        }
    }

    /// The Specification's bars: 100.0 on rows 0 and 1, row 2 opening at 99.0 and
    /// trading down to 94.0, and 103.0 from row 3 to row 9.
    fn worked() -> Vec<Candle> {
        let flat = |price| bar(price, price, price, price);
        let mut bars = vec![flat(100.0), flat(100.0), bar(99.0, 99.0, 94.0, 98.0)];
        bars.extend((3..10).map(|_| flat(103.0)));
        bars
    }

    fn trade(start: usize, last: usize, stop: Option<f64>) -> Trade {
        Trade {
            instrument: 0,
            start,
            last,
            stop,
            take_profit: None,
        }
    }

    /// The label of `trade` on `e`, 10,000 on `side`.
    fn of(e: &mut Engine, trade: Trade, side: Side) -> Label {
        label(e, trade, 10_000.0, side, Units::Linear)
    }

    #[test]
    fn a_label_closes_at_its_horizon_or_at_its_stop() {
        // 10,000 buys 100 at row 1's open; at a horizon of 2 the close placed at row
        // 2's close fills at row 3's open of 103.0, and a stop at 95.0, resting from
        // row 1's close, fills on row 2
        let mut e = engine(worked());
        let closed = |ret, exit, end| Label::Closed {
            ret,
            exit,
            entered: 1,
            end,
        };
        let horizon = of(&mut e, trade(0, 2, None), Side::Buy);
        assert_eq!(horizon, closed(0.03, Exit::Horizon, 3));
        let stopped = of(&mut e, trade(0, 2, Some(0.05)), Side::Buy);
        assert_eq!(stopped, closed(-0.05, Exit::Stop, 2));
        // a short loses what the long made on the way to row 3
        let short = of(&mut e, trade(0, 2, None), Side::Sell);
        assert_eq!(short, closed(-0.03, Exit::Horizon, 3));
    }

    #[test]
    fn an_inverse_label_buys_the_dollars_its_coins_are_worth() {
        // 0.5 coins at row 0's close of 20,000 are worth 10,000 dollars, 100 contracts
        // of 100 bought at row 1's open; sold at row 3's open of 21,000, they make
        // 10,000 * (1 / 20,000 - 1 / 21,000) coins, a return of 1 / 21
        let flat = |price| bar(price, price, price, price);
        let mut bars = vec![flat(20_000.0); 2];
        bars.extend([flat(21_000.0); 3]);
        let config = EngineConfig {
            quote: 0.5,
            ..config()
        };
        let mut e = Engine::new(Candles::new(bars).inverse(100.0), config);
        let made = label(
            &mut e,
            trade(0, 2, None),
            0.5,
            Side::Buy,
            Units::Inverse(100.0),
        );
        let Label::Closed {
            ret,
            exit,
            entered,
            end,
        } = made
        else {
            panic!("the label closed: {made:?}");
        };
        assert!((ret - 1.0 / 21.0).abs() < 1e-12, "a return of {ret}");
        assert_eq!((exit, entered, end), (Exit::Horizon, 1, 3));
    }

    #[test]
    fn a_label_still_open_at_the_last_row_is_censored() {
        let mut e = engine(worked());
        assert_eq!(of(&mut e, trade(7, 9, None), Side::Buy), Label::Censored);
    }

    #[test]
    fn an_entry_waits_for_its_instrument_s_next_bar_and_a_delisting_closes_it() {
        // the first of two perpetuals prints on rows 0 and 3, rising from 100.0 to
        // 102.0 on row 3, and is delisted at row 5's open, while the second prints
        // every row
        let hour = 3_600_000_000_000;
        let flat = |price| bar(price, price, price, price);
        let first = |t| match t {
            0 => flat(100.0),
            3 => bar(100.0, 102.0, 100.0, 102.0),
            _ => ABSENT,
        };
        let rows: Vec<Candle> = (0..7).flat_map(|t| [first(t), flat(10.0)]).collect();
        let clock = Clock::new((0..7).map(|t| t * hour).collect::<Vec<_>>());
        let perp = Instrument {
            market: Some(Market::Perp),
            ..Instrument::default()
        };
        let delisted = Instrument {
            delist: Some(5 * hour),
            ..perp.clone()
        };
        let bars = Candles::universe(rows, 2, Some(clock)).with_instruments(vec![delisted, perp]);
        let mut e = Engine::new(bars, config());
        // at a horizon of 2 its last printed row is row 0, and the entry waits to row 3
        assert_eq!(of(&mut e, trade(0, 0, None), Side::Buy), Label::None);
        // at a horizon of 3 it buys 100 at row 3's open, and its close, placed at row
        // 3's close, waits until the delisting closes it at its last close of 102.0
        let delisted = Label::Closed {
            ret: 0.02,
            exit: Exit::Delist,
            entered: 3,
            end: 5,
        };
        assert_eq!(of(&mut e, trade(0, 3, None), Side::Buy), delisted);
    }

    #[test]
    fn a_close_that_fills_nothing_is_placed_again_until_the_label_is_flat() {
        // row 3 prints no volume, so the close placed at row 2's close fills nothing
        // there, and the one placed at row 3's close sells at row 4's open of 103.0
        let mut bars = worked();
        bars[3].volume = 0.0;
        let mut e = engine(bars);
        let closed = Label::Closed {
            ret: 0.03,
            exit: Exit::Horizon,
            entered: 1,
            end: 4,
        };
        assert_eq!(of(&mut e, trade(0, 2, None), Side::Buy), closed);
    }

    #[test]
    fn a_short_the_price_doubles_on_is_liquidated() {
        // a short of 10,000 sold at 100.0 on row 1 meets row 2's open of 250.0
        let flat = |price| bar(price, price, price, price);
        let mut e = engine(vec![flat(100.0), flat(100.0), flat(250.0), flat(250.0)]);
        let Label::Closed { exit, end, .. } = of(&mut e, trade(0, 2, None), Side::Sell) else {
            panic!("the short is closed out");
        };
        assert_eq!((exit, end), (Exit::Liquidation, 2));
    }

    #[test]
    fn labels_in_parallel_are_each_label_alone() {
        let trades: Vec<Trade> = (0..7).map(|t| trade(t, t + 2, Some(0.05))).collect();
        let mut alone = engine(worked());
        let each: Vec<Label> = trades
            .iter()
            .map(|&trade| of(&mut alone, trade, Side::Buy))
            .collect();
        let together = labels(
            &engine(worked()),
            &trades,
            10_000.0,
            Side::Buy,
            Units::Linear,
        );
        assert_eq!(together, each);
    }
}
