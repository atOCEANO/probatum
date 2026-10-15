//! A run decided in advance: the actions taken at each bar's close, played through
//! the engine as a strategy. A cancel or a replace names its order by a reference
//! resolved as the run goes, an earlier placement in the script or the oldest order
//! resting, so a placement the engine refused leaves its reference naming nothing,
//! as a strategy left holding no id would find.

use probatum_core::{OrderId, OrderType, Side, State, TimeInForce, Wallet};

use crate::engine::Engine;
use crate::strategy::Strategy;

/// The order a cancel or a replace acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderRef {
    /// What the script's placement `n` made, counting its `Order` and `Close`
    /// actions from 0 in the order they are taken, and a `Bracket` as two, its stop
    /// and then its take-profit.
    Placed(usize),
    /// The oldest order still resting.
    Oldest,
}

/// One decision taken at a bar's close.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Order {
        instrument: usize,
        side: Side,
        size: f64,
        kind: OrderType,
        price: Option<f64>,
        trigger: Option<f64>,
        reduce_only: bool,
        post_only: bool,
        tif: TimeInForce,
    },
    /// A stop and a take-profit as one OCO pair, as `Engine::bracket` places them.
    Bracket {
        instrument: usize,
        stop: Option<f64>,
        take_profit: Option<f64>,
    },
    /// Flatten an instrument's position.
    Close(usize),
    Cancel(OrderRef),
    /// Cancel every resting order, or only an instrument's.
    CancelAll(Option<usize>),
    Replace {
        order: OrderRef,
        size: Option<f64>,
        price: Option<f64>,
        trigger: Option<f64>,
    },
    Transfer {
        amount: f64,
        to: Wallet,
    },
    /// Bring each instrument named to its weight of equity, as `Engine::target`
    /// does.
    Target {
        weights: Vec<(usize, f64)>,
        tolerance: f64,
        transfer: bool,
    },
}

/// The actions to take at each bar's close, `bars[t]` at bar `t` in order, for
/// `strategy::run` to play.
pub struct Script {
    bars: Vec<Vec<Action>>,
    placed: Vec<Option<OrderId>>,
}

impl Script {
    pub fn new(bars: Vec<Vec<Action>>) -> Script {
        Script {
            bars,
            placed: Vec::new(),
        }
    }

    fn resolve(&self, order: OrderRef, engine: &Engine) -> Option<OrderId> {
        match order {
            OrderRef::Placed(n) => self.placed.get(n).copied().flatten(),
            OrderRef::Oldest => engine.oldest_resting(),
        }
    }
}

impl Strategy for Script {
    fn init(&mut self, _state: &State) {
        self.placed.clear();
    }

    fn next(&mut self, state: &State, engine: &mut Engine) {
        let Some(actions) = self.bars.get(state.tick_index) else {
            return;
        };
        for action in actions {
            match *action {
                Action::Order {
                    instrument,
                    side,
                    size,
                    kind,
                    price,
                    trigger,
                    reduce_only,
                    post_only,
                    tif,
                } => {
                    let id = engine.order_on(
                        instrument,
                        side,
                        size,
                        kind,
                        price,
                        trigger,
                        reduce_only,
                        post_only,
                        tif,
                    );
                    self.placed.push(id);
                }
                Action::Bracket {
                    instrument,
                    stop,
                    take_profit,
                } => {
                    let (stop, take_profit) = engine.bracket(instrument, stop, take_profit);
                    self.placed.extend([stop, take_profit]);
                }
                Action::Close(instrument) => self.placed.push(engine.close_on(instrument)),
                Action::Cancel(order) => {
                    if let Some(id) = self.resolve(order, engine) {
                        engine.cancel(id);
                    }
                }
                Action::CancelAll(None) => {
                    engine.cancel_all();
                }
                Action::CancelAll(Some(instrument)) => {
                    engine.cancel_all_on(instrument);
                }
                Action::Replace {
                    order,
                    size,
                    price,
                    trigger,
                } => {
                    if let Some(id) = self.resolve(order, engine) {
                        engine.replace(id, size, price, trigger);
                    }
                }
                Action::Transfer { amount, to } => engine.transfer(amount, to),
                Action::Target {
                    ref weights,
                    tolerance,
                    transfer,
                } => {
                    // a weight it would refuse was refused when the script was read
                    let _ = engine.target(weights, tolerance, transfer);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Action, OrderRef, Script};
    use crate::candles::Candles;
    use crate::engine::{Engine, EngineConfig, Intrabar, Liquidation};
    use crate::execution::{Impact, Limit, Spread};
    use crate::reporter::EventKind;
    use crate::strategy::run;
    use probatum_core::{Candle, Market, OrderId, OrderType, Side, TimeInForce};

    // flat at 100, so a bid at 90 or 80 rests until it is cancelled
    fn engine(max_open_orders: usize) -> Engine {
        let bar = Candle {
            open: 100.0,
            high: 100.0,
            low: 100.0,
            close: 100.0,
            volume: 1e6,
        };
        let config = EngineConfig {
            market: Market::Spot,
            quote: 10_000.0,
            idle_cash: 0.0,
            fee_taker: 0.0,
            fee_maker: 0.0,
            slippage_bps: 0.0,
            budget: 1.0,
            window: 24,
            max_open_orders,
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
        };
        Engine::new(Candles::new(vec![bar; 4]), config)
    }

    fn bid(price: f64) -> Action {
        Action::Order {
            instrument: 0,
            side: Side::Buy,
            size: 1.0,
            kind: OrderType::Limit,
            price: Some(price),
            trigger: None,
            reduce_only: false,
            post_only: false,
            tif: TimeInForce::Gtc,
        }
    }

    fn cancelled(engine: &Engine) -> Vec<OrderId> {
        let reporter = engine.reporter().expect("reporting is on");
        reporter
            .events()
            .iter()
            .filter(|event| event.kind == EventKind::Cancelled)
            .filter_map(|event| event.order)
            .collect()
    }

    #[test]
    fn a_placement_names_the_order_it_made_and_a_refused_one_names_nothing() {
        // one slot: the first bid rests and the second is refused, so cancelling
        // the second placement does nothing and cancelling the first takes it
        let mut e = engine(1);
        let mut script = Script::new(vec![
            vec![bid(90.0), bid(80.0)],
            vec![
                Action::Cancel(OrderRef::Placed(1)),
                Action::Cancel(OrderRef::Placed(0)),
            ],
        ]);
        run(&mut e, &mut script);
        let first = script.placed[0].expect("the first bid rests");
        assert_eq!(script.placed[1], None);
        assert_eq!(cancelled(&e), vec![first]);
    }

    #[test]
    fn the_oldest_resting_order_is_the_first_placed_whatever_its_slot() {
        // cancelling the first of three bids frees slot 0 and the replace moves the
        // third into it, so slot order would give the third where the oldest order
        // still resting is the second
        let mut e = engine(8);
        let mut script = Script::new(vec![
            vec![bid(90.0), bid(80.0), bid(70.0)],
            vec![
                Action::Cancel(OrderRef::Placed(0)),
                Action::Replace {
                    order: OrderRef::Placed(2),
                    size: None,
                    price: Some(75.0),
                    trigger: None,
                },
                Action::Cancel(OrderRef::Oldest),
            ],
        ]);
        let last = run(&mut e, &mut script);
        let placed: Vec<OrderId> = script.placed.iter().map(|id| id.expect("rests")).collect();
        assert_eq!(cancelled(&e), vec![placed[0], placed[1]]);
        assert_eq!(last.open_orders.len(), 1);
        assert_eq!(last.open_orders[0].id, placed[2]);
    }

    #[test]
    fn a_script_played_again_names_the_orders_of_its_new_run() {
        // ids carry on across the reset, so the second run's bid is a new order and
        // its cancel has to name that one rather than the first run's
        let mut e = engine(8);
        let mut script = Script::new(vec![
            vec![bid(90.0)],
            vec![Action::Cancel(OrderRef::Placed(0))],
        ]);
        run(&mut e, &mut script);
        let last = run(&mut e, &mut script);
        assert_eq!(script.placed.len(), 1);
        assert_eq!(cancelled(&e), vec![script.placed[0].expect("rests")]);
        assert!(last.open_orders.is_empty());
    }

    #[test]
    fn an_action_is_taken_at_its_bar_and_fills_on_the_next() {
        let mut e = engine(8);
        let buy = Action::Order {
            instrument: 0,
            side: Side::Buy,
            size: 1.0,
            kind: OrderType::Market,
            price: None,
            trigger: None,
            reduce_only: false,
            post_only: false,
            tif: TimeInForce::Gtc,
        };
        let mut script = Script::new(vec![vec![], vec![buy]]);
        run(&mut e, &mut script);
        let fills = e.reporter().expect("reporting is on").fills();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].tick, 2);
    }
}
