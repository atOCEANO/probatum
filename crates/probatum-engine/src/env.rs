//! The environment: a batch of accounts on one series, each episode an account opened
//! at its start and stepped from its decision at each close, the observation it reads
//! and the reward it earns worked for the whole batch at once, in parallel
//! (Specification, The environment). Which episode each environment runs, its start
//! and its slots, the Python layer draws from that environment's own stream and hands
//! in.

use std::sync::Arc;

use probatum_core::{Candle, Market, OrderId, OrderType, Side, TimeInForce};
use rayon::prelude::*;

use crate::batch::EnvBatch;
use crate::candles::printed;
use crate::engine::{Engine, TargetError};
use crate::inverse::Units;

/// The reward a step pays (Specification, Reward and termination).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reward {
    /// The log return of equity over the step, its equity floored.
    Log,
    /// The differential Sharpe of those log returns, its moments moving at this rate.
    Sharpe(f64),
    /// One the caller works from the state, which pays nothing here.
    Own,
}

/// What a step decides from, every environment's row in turn: each slot's weight, a
/// NaN leaving it unnamed, or each slot's limit as a size and an offset.
#[derive(Clone, Copy, Debug)]
pub enum Decision<'a> {
    Weights(&'a [f64]),
    Orders(&'a [f64]),
}

/// What every environment of a batch shares.
#[derive(Clone, Debug)]
pub struct Settings {
    pub window: usize,
    /// The features, row-major over the rows, the instruments and `width` features
    /// each, or None for the candles; a clone or a fork shares them, as it shares the
    /// bars.
    pub features: Option<Arc<[f64]>>,
    pub width: usize,
    /// Each feature's mean and deviation, which a market value is written over, or
    /// None to write it as it is.
    pub scale: Option<Vec<(f64, f64)>>,
    /// Each instrument's first printed bar with every feature finite, None where it
    /// has none.
    pub onsets: Vec<Option<usize>>,
    pub slots: usize,
    pub max_leverage: f64,
    pub max_offset: f64,
    pub tolerance: f64,
    pub reward: Reward,
    pub orders: bool,
    pub episode_length: Option<usize>,
    /// The equity an episode ends at, 1% of the opening balance.
    pub floor: f64,
    pub units: Units,
    /// Each instrument's tick, which an order's offset counts.
    pub ticks: Vec<f64>,
    /// The threads a step runs on, or None for as many as its work pays for.
    pub threads: Option<usize>,
}

/// The arrays a reward or a decode reads, each environment's place in each: one value
/// in the first nine, and a value a slot in the last two.
pub struct StateBuffers<'a> {
    pub equity: &'a mut [f64],
    pub previous_equity: &'a mut [f64],
    pub log_return: &'a mut [f64],
    pub drawdown: &'a mut [f64],
    pub leverage: &'a mut [f64],
    pub turnover: &'a mut [f64],
    pub fees: &'a mut [f64],
    pub funding: &'a mut [f64],
    pub liquidated: &'a mut [bool],
    pub weights: &'a mut [f64],
    pub positions: &'a mut [f64],
}

/// One environment's episode: its slots and start, and what it keeps between steps.
#[derive(Clone, Debug, Default)]
struct Episode {
    slots: Vec<usize>,
    start: usize,
    steps: usize,
    peak: f64,
    moments: (f64, f64),
    returned: f64,
    ordered: Vec<Option<OrderId>>,
    terminated: bool,
    truncated: bool,
    // what the last step did, which the state reads
    before: f64,
    fees: f64,
    traded: f64,
    funding: f64,
    liquidated: bool,
}

/// A batch of environments over one series.
pub struct Environment {
    batch: EnvBatch,
    settings: Settings,
    episodes: Vec<Episode>,
    // None steps in the caller's own thread
    pool: Option<rayon::ThreadPool>,
}

impl Clone for Environment {
    /// Every environment copied, its engine and its episode, on a pool of its own.
    fn clone(&self) -> Environment {
        Environment::over(
            self.batch.clone(),
            self.settings.clone(),
            self.episodes.clone(),
        )
    }
}

/// One environment's share of a step: its index, its engine and episode, and its
/// places among the rewards and the two flags.
type Stepping<'a> = (
    usize,
    (
        (
            ((&'a mut Engine, &'a mut Episode), &'a mut f64),
            &'a mut bool,
        ),
        &'a mut bool,
    ),
);

/// The threads a step of `envs` environments of `size` floats each runs on where the
/// caller names none: one for each 2^17 floats its observations hold, about the work
/// that pays for waking a thread, and at most half the CPUs (ADR 0040).
fn threads_for(envs: usize, size: usize) -> usize {
    (envs * size)
        .div_ceil(1 << 17)
        .clamp(1, (rayon::current_num_threads() / 2).max(1))
}

/// Whether instrument `i` is shown at the engine's bar: entered, printed and not
/// delisted (Specification, Episodes).
fn shown(s: &Settings, engine: &Engine, i: usize) -> bool {
    let t = engine.tick();
    s.onsets[i].is_some_and(|e| t + 1 >= e + s.window)
        && printed(&engine.candles_all()[t * engine.width() + i])
        && !engine.is_delisted(i)
}

/// Slot instrument `i`'s weight of `equity`, as the venue reckons a position's
/// notional, and 0.0 where equity is at or below zero.
fn weight(s: &Settings, engine: &Engine, i: usize, equity: f64) -> f64 {
    let qty = engine.position(i);
    if qty == 0.0 || equity <= 0.0 {
        return 0.0;
    }
    s.units.signed(qty * engine.mark(i)) / equity
}

/// The step's targets: each shown slot to its weight, a NaN leaving it unnamed.
fn decide(s: &Settings, engine: &mut Engine, episode: &Episode, row: &[f64]) {
    let named: Vec<(usize, f64)> = episode
        .slots
        .iter()
        .zip(row)
        .filter(|&(&i, weight)| !weight.is_nan() && shown(s, engine, i))
        .map(|(&i, &weight)| (i, s.units.signed(weight)))
        .collect();
    if !named.is_empty() {
        // every negative spot weight was refused before any environment stepped
        let _ = engine.target(&named, s.tolerance, false);
    }
}

/// The step's limits: each shown slot's order replacing the one the last step left
/// resting, its notional a share of equity at the last close and its price whole
/// ticks from it on the passive side (Specification, Actions).
fn place(s: &Settings, engine: &mut Engine, episode: &mut Episode, row: &[f64]) {
    let equity = engine.equity();
    let (t, n) = (engine.tick(), engine.width());
    for (j, &i) in episode.slots.iter().enumerate() {
        if !shown(s, engine, i) {
            continue;
        }
        if let Some(id) = episode.ordered[j].take() {
            engine.cancel(id);
        }
        let size = row[2 * j].clamp(-1.0, 1.0);
        let offset = row[2 * j + 1].clamp(0.0, 1.0);
        if size == 0.0 {
            continue;
        }
        let close = engine.candles_all()[t * n + i].close;
        let notional = size.abs() * s.max_leverage * equity;
        let amount = match s.units {
            Units::Inverse(value) => notional * close / value,
            Units::Linear => notional / close,
        };
        let ticks = (offset * s.max_offset).round_ties_even();
        let (side, price) = if size > 0.0 {
            (Side::Buy, close - ticks * s.ticks[i])
        } else {
            (Side::Sell, close + ticks * s.ticks[i])
        };
        // a buy more ticks below the close than it is worth has no price
        if price <= 0.0 {
            continue;
        }
        episode.ordered[j] = engine.order_on(
            i,
            s.units.side(side),
            s.units.size(amount),
            OrderType::Limit,
            Some(s.units.price(price)),
            None,
            false,
            false,
            TimeInForce::Gtc,
        );
    }
}

/// The next bar, and the step's reward and flags (Specification, Reward and
/// termination).
fn advance(s: &Settings, engine: &mut Engine, episode: &mut Episode) -> f64 {
    let before = engine.equity();
    let paid = engine.funding_paid();
    let equity = engine.advance();
    let tally = engine.tally();
    let returned = libm::log(equity.max(s.floor) / before);
    episode.steps += 1;
    episode.peak = episode.peak.max(equity);
    episode.returned = returned;
    episode.before = before;
    episode.fees = tally.fees;
    episode.traded = tally.traded;
    episode.funding = engine.funding_paid() - paid;
    episode.liquidated = tally.liquidated;
    let reward = match s.reward {
        Reward::Log => returned,
        Reward::Sharpe(eta) => {
            let (a, b) = episode.moments;
            let (da, db) = (returned - a, returned * returned - b);
            let spread = b - a * a;
            episode.moments = (a + eta * da, b + eta * db);
            if spread > 0.0 {
                (b * da - a * db / 2.0) / (spread * spread.sqrt())
            } else {
                0.0
            }
        }
        Reward::Own => 0.0,
    };
    episode.terminated = tally.liquidated || equity <= s.floor;
    episode.truncated =
        !episode.terminated && (engine.done() || s.episode_length == Some(episode.steps));
    reward
}

/// Feature `g`'s market value `x` as the observation writes it, over the feature's
/// mean and deviation where it is normalised (Specification, The observation).
fn scaled(s: &Settings, g: usize, x: f64) -> f32 {
    match &s.scale {
        Some(scale) => ((x - scale[g].0) / scale[g].1) as f32,
        None => x as f32,
    }
}

/// The floats of one environment's market block, the observation's first.
fn market_size(s: &Settings) -> usize {
    s.window * s.slots * if s.features.is_some() { s.width } else { 5 }
}

/// Instrument `i`'s mean volume over the printed bars of the window from bar `first`
/// to bar `t`, summed in order, which a candle's volume is written over.
fn window_mean(bars: &[Candle], n: usize, i: usize, first: usize, t: usize) -> f64 {
    let (mut volume, mut count) = (0.0, 0.0);
    for r in first..=t {
        let bar = &bars[r * n + i];
        if printed(bar) {
            volume += bar.volume;
            count += 1.0;
        }
    }
    volume / count
}

/// One environment's observation, written into `row` (Specification, The
/// observation).
fn observed(s: &Settings, engine: &Engine, episode: &Episode, row: &mut [f32]) {
    let (market, account) = row.split_at_mut(market_size(s));
    observed_market(s, engine, episode, market);
    observed_account(s, engine, episode, account);
}

/// One environment's market block, written into `row`.
fn observed_market(s: &Settings, engine: &Engine, episode: &Episode, row: &mut [f32]) {
    let (w, k) = (s.window, s.slots);
    let f = if s.features.is_some() { s.width } else { 5 };
    let (t, n) = (engine.tick(), engine.width());
    let bars = engine.candles_all();
    row.fill(0.0);
    for (j, &i) in episode.slots.iter().enumerate() {
        if !shown(s, engine, i) {
            continue;
        }
        // an instrument shown has entered, so the window lies within the series
        let first = t + 1 - w;
        if let Some(values) = &s.features {
            for r in first..=t {
                if printed(&bars[r * n + i]) {
                    let at = ((r - first) * k + j) * f;
                    for g in 0..f {
                        row[at + g] = scaled(s, g, values[(r * n + i) * f + g]);
                    }
                }
            }
            continue;
        }
        let last = bars[t * n + i].close;
        let mean = window_mean(bars, n, i, first, t);
        for r in first..=t {
            let bar = &bars[r * n + i];
            if printed(bar) {
                let at = ((r - first) * k + j) * 5;
                row[at] = scaled(s, 0, bar.open / last);
                row[at + 1] = scaled(s, 1, bar.high / last);
                row[at + 2] = scaled(s, 2, bar.low / last);
                row[at + 3] = scaled(s, 3, bar.close / last);
                row[at + 4] = if mean > 0.0 {
                    scaled(s, 4, bar.volume / mean)
                } else {
                    0.0
                };
            }
        }
    }
}

/// One environment's account block, written into `row`.
fn observed_account(s: &Settings, engine: &Engine, episode: &Episode, row: &mut [f32]) {
    let reading = engine.reading();
    let equity = reading.equity;
    let mut at = 0;
    let mut weights = 0.0;
    for &i in &episode.slots {
        let held = engine.position(i) != 0.0;
        let own = weight(s, engine, i, equity);
        weights += own;
        let (mark, entry) = (engine.mark(i), engine.entry(i));
        row[at] = own as f32;
        row[at + 1] = if held && equity > 0.0 {
            (engine.unrealized(i) / equity) as f32
        } else {
            0.0
        };
        row[at + 2] = match (held, s.units) {
            (false, _) => 0.0,
            // the engine holds an inverse contract at its price's reciprocal
            (true, Units::Inverse(_)) => (entry / mark - 1.0) as f32,
            (true, Units::Linear) => (mark / entry - 1.0) as f32,
        };
        row[at + 3] = if shown(s, engine, i) { 1.0 } else { 0.0 };
        at += 4;
        if s.orders {
            let quoted = s.units.dollars(mark);
            let (mut resting, mut nearest) = (0.0, f64::INFINITY);
            for order in engine.resting().filter(|order| order.instrument == i) {
                resting += 1.0;
                if let Some(price) = order.price {
                    let away = (s.units.dollars(price.get()) - quoted).abs() / s.ticks[i];
                    nearest = nearest.min(away);
                }
            }
            row[at] = resting as f32;
            row[at + 1] = if nearest.is_finite() {
                nearest as f32
            } else {
                0.0
            };
            at += 2;
        }
    }
    let ratio = if reading.maintenance == 0.0 {
        0.0
    } else if reading.margin_equity <= 0.0 {
        1.0
    } else {
        reading.maintenance / reading.margin_equity
    };
    row[at] = episode.returned as f32;
    row[at + 1] = if equity > 0.0 {
        (reading.gross / equity) as f32
    } else {
        0.0
    };
    row[at + 2] = ratio as f32;
    row[at + 3] = (1.0 - weights) as f32;
}

impl Environment {
    /// Environments over `batch`'s engines, none in an episode until `reset`.
    pub fn new(batch: EnvBatch, settings: Settings) -> Environment {
        let episodes = vec![Episode::default(); batch.len()];
        Environment::over(batch, settings, episodes)
    }

    /// Environment `env` copied `n` times into environments of their own, each
    /// stepping on as it would (ADR 0040).
    pub fn fork(&self, env: usize, n: usize) -> Environment {
        Environment::over(
            self.batch.fork(env, n),
            self.settings.clone(),
            vec![self.episodes[env].clone(); n],
        )
    }

    /// Environments over `batch`'s engines in `episodes`, stepping on a pool as large
    /// as their work pays for.
    fn over(batch: EnvBatch, settings: Settings, episodes: Vec<Episode>) -> Environment {
        let mut env = Environment {
            batch,
            settings,
            episodes,
            pool: None,
        };
        let threads = env
            .settings
            .threads
            .unwrap_or_else(|| threads_for(env.len(), env.size()));
        // a pool the system will not give leaves the step in the caller's thread, to
        // the same results
        if threads > 1 {
            env.pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .ok();
        }
        env
    }

    /// The threads a step runs on, the caller's own counted as one.
    pub fn threads(&self) -> usize {
        self.pool
            .as_ref()
            .map_or(1, rayon::ThreadPool::current_num_threads)
    }

    /// Number of environments.
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// True when there are no environments.
    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    /// The engines, for reading what each environment recorded.
    pub fn engines(&self) -> &[Engine] {
        self.batch.engines()
    }

    /// What the environments share.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The floats of one environment's observation: the market block's window of
    /// each slot's features, then each slot's four account values, six with orders,
    /// and the account's four.
    pub fn size(&self) -> usize {
        let s = &self.settings;
        let f = if s.features.is_some() { s.width } else { 5 };
        s.window * s.slots * f + s.slots * if s.orders { 6 } else { 4 } + 4
    }

    /// Start an episode in each environment `mask` names, at its row of `starts`, a
    /// fresh account with the venue's balance trading its row of `slots`.
    pub fn reset(&mut self, mask: &[bool], starts: &[usize], slots: &[usize]) {
        let k = self.settings.slots;
        // a step resets the few environments whose episodes ended, each in
        // microseconds, which waking a thread for them would cost more than
        let engines = self.batch.engines_mut().iter_mut();
        for (env, (engine, episode)) in engines.zip(&mut self.episodes).enumerate() {
            if !mask[env] {
                continue;
            }
            engine.reset_at(starts[env]);
            let equity = engine.equity();
            *episode = Episode {
                slots: slots[env * k..(env + 1) * k].to_vec(),
                start: starts[env],
                peak: equity,
                before: equity,
                ordered: vec![None; k],
                ..Episode::default()
            };
        }
    }

    /// Each environment's start and the steps its episode has taken.
    pub fn progress(&self) -> Vec<(usize, usize)> {
        self.episodes
            .iter()
            .map(|episode| (episode.start, episode.steps))
            .collect()
    }

    /// Whether each slot of each environment is shown, written into `out`.
    pub fn shown(&self, out: &mut [bool]) {
        let k = self.settings.slots;
        for (env, (engine, episode)) in self.batch.engines().iter().zip(&self.episodes).enumerate()
        {
            for (j, &i) in episode.slots.iter().enumerate() {
                out[env * k + j] = shown(&self.settings, engine, i);
            }
        }
    }

    /// Step each environment `active` names from `decision`, its reward and flags
    /// written to its place in `rewards`, `terminated` and `truncated`; one not
    /// active is left as it stands, its reward 0.0 and its flags as they were.
    /// Refuses before any environment steps where a weight shorts a spot slot, naming
    /// that environment.
    pub fn step(
        &mut self,
        decision: Decision<'_>,
        active: &[bool],
        rewards: &mut [f64],
        terminated: &mut [bool],
        truncated: &mut [bool],
    ) -> Result<(), (usize, TargetError)> {
        let s = &self.settings;
        let k = s.slots;
        if let Decision::Weights(weights) = decision {
            for (env, (engine, episode)) in
                self.batch.engines().iter().zip(&self.episodes).enumerate()
            {
                if !active[env] {
                    continue;
                }
                for (&i, &weight) in episode.slots.iter().zip(&weights[env * k..(env + 1) * k]) {
                    if engine.market(i) == Market::Spot && weight < 0.0 {
                        return Err((
                            env,
                            TargetError::Short {
                                instrument: i,
                                weight,
                            },
                        ));
                    }
                }
            }
        }
        let work = |(env, ((((engine, episode), reward), ended), cut)): Stepping<'_>| {
            if active[env] {
                match decision {
                    Decision::Weights(weights) => {
                        decide(s, engine, episode, &weights[env * k..(env + 1) * k]);
                    }
                    Decision::Orders(actions) => {
                        place(s, engine, episode, &actions[env * k * 2..(env + 1) * k * 2]);
                    }
                }
                *reward = advance(s, engine, episode);
            } else {
                *reward = 0.0;
            }
            *ended = episode.terminated;
            *cut = episode.truncated;
        };
        let (engines, episodes) = (self.batch.engines_mut(), &mut self.episodes);
        match &self.pool {
            Some(pool) => pool.install(|| {
                engines
                    .par_iter_mut()
                    .zip(episodes.par_iter_mut())
                    .zip(rewards.par_iter_mut())
                    .zip(terminated.par_iter_mut())
                    .zip(truncated.par_iter_mut())
                    .enumerate()
                    .for_each(work)
            }),
            None => engines
                .iter_mut()
                .zip(episodes.iter_mut())
                .zip(rewards.iter_mut())
                .zip(terminated.iter_mut())
                .zip(truncated.iter_mut())
                .enumerate()
                .for_each(work),
        }
        Ok(())
    }

    /// Every environment's observation, row `env` of `out` its own, `size()` floats
    /// each.
    pub fn observe(&self, out: &mut [f32]) {
        self.write_each(out, self.size(), observed);
    }

    /// The floats of one environment's account block, the observation's last.
    pub fn account_size(&self) -> usize {
        self.size() - market_size(&self.settings)
    }

    /// Every environment's account block, as `observe` writes it after the market
    /// block, row `env` of `out` its own, `account_size()` floats each.
    pub fn observe_account(&self, out: &mut [f32]) {
        self.write_each(out, self.account_size(), observed_account);
    }

    /// Each environment's row of the series, the bar its engine stands at.
    pub fn ticks(&self) -> Vec<usize> {
        self.batch.engines().iter().map(Engine::tick).collect()
    }

    /// The printed bars every environment reads, row-major over the rows and the
    /// instruments.
    pub fn candles(&self) -> &[Candle] {
        self.batch.engines()[0].candles_all()
    }

    /// Each bar's market cells as the observation writes them, row-major over the rows,
    /// the instruments and the features, 0.0 where its instrument did not print; None
    /// for the candles, whose cells the close at the window's last bar decides.
    pub fn cells(&self) -> Option<Vec<f32>> {
        let s = &self.settings;
        let values = s.features.as_ref()?;
        let bars = self.batch.engines()[0].candles_all();
        let mut out = vec![0.0; values.len()];
        for (at, bar) in bars.iter().enumerate() {
            if printed(bar) {
                for g in 0..s.width {
                    out[at * s.width + g] = scaled(s, g, values[at * s.width + g]);
                }
            }
        }
        Some(out)
    }

    /// Each bar's mean volume over the printed bars of the window ending at it, row-major
    /// over the rows and the instruments, NaN for a bar the window does not fit before
    /// and where the window holds no printed bar.
    pub fn window_means(&self) -> Vec<f64> {
        let w = self.settings.window;
        let engine = &self.batch.engines()[0];
        let (bars, n) = (engine.candles_all(), engine.width());
        let rows = bars.len() / n;
        let mut out = vec![f64::NAN; rows * n];
        for t in w - 1..rows {
            for i in 0..n {
                out[t * n + i] = window_mean(bars, n, i, t + 1 - w, t);
            }
        }
        out
    }

    /// Each environment's `size` floats of `out`, its row, written by `write`.
    fn write_each(
        &self,
        out: &mut [f32],
        size: usize,
        write: fn(&Settings, &Engine, &Episode, &mut [f32]),
    ) {
        let s = &self.settings;
        let work = |((row, engine), episode): ((&mut [f32], &Engine), &Episode)| {
            write(s, engine, episode, row)
        };
        let (engines, episodes) = (self.batch.engines(), &self.episodes);
        match &self.pool {
            Some(pool) => pool.install(|| {
                out.par_chunks_mut(size)
                    .zip(engines.par_iter())
                    .zip(episodes.par_iter())
                    .for_each(work)
            }),
            None => out
                .chunks_mut(size)
                .zip(engines)
                .zip(episodes)
                .for_each(work),
        }
    }

    /// The state each environment's last step left, as a reward or a decode reads
    /// it (Specification, Reward and termination).
    pub fn state(&self, out: StateBuffers<'_>) {
        let s = &self.settings;
        let k = s.slots;
        for (env, (engine, episode)) in self.batch.engines().iter().zip(&self.episodes).enumerate()
        {
            let reading = engine.reading();
            let (equity, before) = (reading.equity, episode.before);
            out.equity[env] = equity;
            out.previous_equity[env] = before;
            out.log_return[env] = episode.returned;
            out.drawdown[env] = 1.0 - equity / episode.peak;
            out.leverage[env] = if equity > 0.0 {
                reading.gross / equity
            } else {
                0.0
            };
            out.turnover[env] = episode.traded / before;
            out.fees[env] = episode.fees / before;
            out.funding[env] = episode.funding / before;
            out.liquidated[env] = episode.liquidated;
            for (j, &i) in episode.slots.iter().enumerate() {
                out.weights[env * k + j] = weight(s, engine, i, equity);
                out.positions[env * k + j] = s.units.contracts(s.units.signed(engine.position(i)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Decision, Environment, Reward, Settings, StateBuffers};
    use crate::batch::EnvBatch;
    use crate::candles::{Candles, Instrument};
    use crate::clock::Clock;
    use crate::engine::{EngineConfig, TargetError};
    use crate::inverse::Units;
    use probatum_core::{Candle, Market, Side, Tier, TierBasis, Tiers};

    const HOUR: i64 = 3_600_000_000_000;

    const ABSENT: Candle = Candle {
        open: f64::NAN,
        high: f64::NAN,
        low: f64::NAN,
        close: f64::NAN,
        volume: f64::NAN,
    };

    fn bar(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
        Candle {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    fn flat(price: f64) -> Candle {
        bar(price, price, price, price, 1_000.0)
    }

    fn settings(window: usize) -> Settings {
        Settings {
            window,
            features: None,
            width: 0,
            scale: None,
            onsets: vec![Some(0)],
            slots: 1,
            max_leverage: 1.0,
            max_offset: 20.0,
            tolerance: 0.0,
            reward: Reward::Log,
            orders: false,
            episode_length: None,
            floor: 100.0,
            units: Units::Linear,
            ticks: vec![0.1],
            threads: None,
        }
    }

    // a perpetual with no fees and a 10,000 balance
    fn perp() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            fee_taker: 0.0,
            fee_maker: 0.0,
            ..EngineConfig::default()
        }
    }

    // one perpetual, opened at row `start`
    fn opened(bars: Vec<Candle>, settings: Settings, start: usize) -> Environment {
        let mut env = Environment::new(EnvBatch::new(1, Candles::new(bars), perp()), settings);
        env.reset(&[true], &[start], &[0]);
        env
    }

    // instruments of `markets` on an hourly clock, a row of bars each, each delisted
    // at its time
    fn universe(rows: Vec<Vec<Candle>>, markets: &[Market], delist: &[Option<i64>]) -> Candles {
        let times: Vec<i64> = (0..rows.len() as i64).map(|t| t * HOUR).collect();
        let bars: Vec<Candle> = rows.into_iter().flatten().collect();
        let specs = markets
            .iter()
            .zip(delist)
            .map(|(&market, &delist)| Instrument {
                market: Some(market),
                delist,
                ..Instrument::default()
            })
            .collect();
        Candles::universe(bars, markets.len(), Some(Clock::new(times))).with_instruments(specs)
    }

    // two perpetuals' environments of two slots each, every one listed from row 0
    fn pair(rows: Vec<Vec<Candle>>, envs: usize, settings: Settings) -> Environment {
        let candles = universe(rows, &[Market::Perp; 2], &[None, None]);
        let settings = Settings {
            onsets: vec![Some(0), Some(0)],
            slots: 2,
            ticks: vec![0.1, 0.01],
            ..settings
        };
        Environment::new(EnvBatch::new(envs, candles, perp()), settings)
    }

    // every buffer of the state, each environment's one value and each slot's value
    fn state_of(env: &Environment) -> ([Vec<f64>; 8], Vec<bool>, Vec<f64>, Vec<f64>) {
        let (n, k) = (env.len(), env.settings().slots);
        let mut values: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; n]);
        let mut liquidated = vec![true; n];
        let (mut weights, mut positions) = (vec![0.0; n * k], vec![0.0; n * k]);
        {
            let [equity, previous_equity, log_return, drawdown, leverage, turnover, fees, funding] =
                &mut values;
            env.state(StateBuffers {
                equity,
                previous_equity,
                log_return,
                drawdown,
                leverage,
                turnover,
                fees,
                funding,
                liquidated: &mut liquidated,
                weights: &mut weights,
                positions: &mut positions,
            });
        }
        (values, liquidated, weights, positions)
    }

    fn stepped(env: &mut Environment, decision: Decision<'_>) -> (f64, bool, bool) {
        let (mut reward, mut terminated, mut truncated) = ([0.0], [false], [false]);
        env.step(
            decision,
            &[true],
            &mut reward,
            &mut terminated,
            &mut truncated,
        )
        .unwrap();
        (reward[0], terminated[0], truncated[0])
    }

    fn observed(env: &Environment) -> Vec<f32> {
        let mut row = vec![0.0f32; env.len() * env.size()];
        env.observe(&mut row);
        row
    }

    #[test]
    fn a_window_of_three_candles_reads_its_prices_over_the_last_close() {
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 110.0, 95.0, 105.0, 300.0),
        ];
        let env = opened(bars, settings(3), 2);
        let mut row = vec![0.0f32; env.size()];
        env.observe(&mut row);
        // three bars of five and the slot's four and the account's four
        assert_eq!(row.len(), 3 * 5 + 4 + 4);
        let last = &row[10..15];
        let mean = (1_000.0 + 1_000.0 + 300.0) / 3.0;
        let expected = [
            100.0 / 105.0,
            110.0 / 105.0,
            95.0 / 105.0,
            1.0,
            300.0 / mean,
        ];
        for (got, want) in last.iter().zip(expected) {
            assert_eq!(*got, want as f32);
        }
        // shown, and flat
        assert_eq!(&row[15..19], &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_weight_of_one_fills_at_the_next_open_and_is_rewarded_its_log_return() {
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 110.0, 100.0, 110.0, 1_000.0),
        ];
        let mut env = opened(bars, settings(1), 1);
        let (reward, terminated, truncated) = stepped(&mut env, Decision::Weights(&[1.0]));
        assert_eq!(reward, libm::log(11_000.0 / 10_000.0));
        assert!(!terminated && truncated, "the series ends at its last bar");
        let mut row = vec![0.0f32; env.size()];
        env.observe(&mut row);
        // 100 held at 110 on 11,000 of equity, 10% from its entry, at 1x
        assert_eq!(
            &row[5..9],
            &[
                (11_000.0 / 11_000.0) as f32,
                (1_000.0 / 11_000.0) as f32,
                0.1f32,
                1.0
            ]
        );
    }

    #[test]
    fn a_takeover_ends_the_episode_rewarded_at_the_floor() {
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 100.0, 75.0, 76.0, 1_000.0),
            flat(76.0),
        ];
        let mut env = opened(bars, settings(1), 1);
        let (reward, terminated, truncated) = stepped(&mut env, Decision::Weights(&[5.0]));
        assert!(terminated && !truncated);
        assert_eq!(reward, libm::log(100.0 / 10_000.0));
        let engine = &env.engines()[0];
        assert!(engine.tally().liquidated);
        // nothing is left, so every value over equity reads zero, never NaN
        assert_eq!(engine.equity(), 0.0);
        let floored = libm::log(100.0 / 10_000.0) as f32;
        assert_eq!(
            &observed(&env)[5..],
            &[0.0, 0.0, 0.0, 1.0, floored, 0.0, 0.0, 1.0]
        );
        let (values, liquidated, weights, _) = state_of(&env);
        assert_eq!((values[4][0], liquidated[0], weights[0]), (0.0, true, 0.0));
    }

    #[test]
    fn a_spot_holding_that_falls_to_the_floor_ends_its_episode_unliquidated() {
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 100.0, 0.5, 0.5, 1_000.0),
            flat(0.5),
        ];
        let config = EngineConfig {
            market: Market::Spot,
            fee_taker: 0.0,
            fee_maker: 0.0,
            ..EngineConfig::default()
        };
        let mut env = Environment::new(EnvBatch::new(1, Candles::new(bars), config), settings(1));
        env.reset(&[true], &[1], &[0]);
        let (reward, terminated, truncated) = stepped(&mut env, Decision::Weights(&[1.0]));
        // 100 held at 0.5 is 50 of equity, below the floor of 100
        assert!(terminated && !truncated);
        assert_eq!(reward, libm::log(100.0 / 10_000.0));
        assert!(!env.engines()[0].tally().liquidated);
    }

    #[test]
    fn an_order_rests_its_offset_in_whole_ticks_below_the_close() {
        let bars = vec![flat(50_000.0), flat(50_000.0), flat(50_000.0)];
        let mut env = opened(
            bars,
            Settings {
                orders: true,
                max_leverage: 2.0,
                ..settings(1)
            },
            1,
        );
        stepped(&mut env, Decision::Orders(&[0.25, 0.5]));
        let resting: Vec<_> = env.engines()[0].resting().collect();
        assert_eq!(resting.len(), 1);
        assert_eq!(resting[0].price.unwrap().get(), 50_000.0 - 10.0 * 0.1);
        assert!((resting[0].size.get() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn a_buy_offset_below_a_price_of_zero_places_none_and_takes_the_last_back() {
        let settings = Settings {
            orders: true,
            max_offset: 2_000.0,
            ..settings(1)
        };
        let mut env = opened(vec![flat(100.0); 4], settings, 1);
        stepped(&mut env, Decision::Orders(&[0.25, 0.0005]));
        assert_eq!(env.engines()[0].resting().count(), 1);
        // 2,000 ticks of 0.1 below a close of 100
        stepped(&mut env, Decision::Orders(&[0.25, 1.0]));
        assert_eq!(env.engines()[0].resting().count(), 0);
    }

    #[test]
    fn an_offset_of_half_a_tick_rounds_to_the_even_tick() {
        let bars = vec![flat(50_000.0), flat(50_000.0), flat(50_000.0)];
        // 0.125 and 0.625 of 20 ticks are two and a half and twelve and a half
        for (offset, ticks) in [(0.125, 2.0), (0.625, 12.0)] {
            let mut env = opened(
                bars.clone(),
                Settings {
                    orders: true,
                    ..settings(1)
                },
                1,
            );
            stepped(&mut env, Decision::Orders(&[0.25, offset]));
            let resting: Vec<_> = env.engines()[0].resting().collect();
            assert_eq!(resting[0].price.unwrap().get(), 50_000.0 - ticks * 0.1);
        }
    }

    #[test]
    fn features_fill_the_market_block_slot_by_slot_on_the_bars_each_printed() {
        let rows = vec![
            vec![flat(100.0), flat(50.0)],
            vec![flat(100.0), ABSENT],
            vec![flat(100.0), flat(50.0)],
        ];
        // each bar's two features a hundred times its row, ten times its instrument and
        // its own place
        let values = (0..3)
            .flat_map(|t| {
                (0..2).flat_map(move |i| (0..2).map(move |g| (100 * t + 10 * i + g) as f64))
            })
            .collect();
        let settings = Settings {
            features: Some(values),
            width: 2,
            onsets: vec![Some(0), Some(0)],
            slots: 2,
            ticks: vec![0.1, 0.1],
            ..settings(2)
        };
        let batch = EnvBatch::new(1, universe(rows, &[Market::Perp; 2], &[None, None]), perp());
        let mut env = Environment::new(batch, settings);
        // the slots in the other order, which the block follows
        env.reset(&[true], &[2], &[1, 0]);
        let row = observed(&env);
        assert_eq!(row.len(), 2 * 2 * 2 + 2 * 4 + 4);
        // row 1, where the second printed nothing, then row 2
        assert_eq!(
            &row[..8],
            &[0.0, 0.0, 100.0, 101.0, 210.0, 211.0, 200.0, 201.0]
        );
        assert_eq!((row[8 + 3], row[12 + 3]), (1.0, 1.0));
    }

    #[test]
    fn a_normalised_market_value_is_written_over_its_mean_and_deviation() {
        let rows = vec![
            vec![flat(100.0), flat(50.0)],
            vec![flat(100.0), ABSENT],
            vec![flat(100.0), flat(50.0)],
        ];
        let values = (0..3)
            .flat_map(|t| {
                (0..2).flat_map(move |i| (0..2).map(move |g| (100 * t + 10 * i + g) as f64))
            })
            .collect();
        let featured = Settings {
            features: Some(values),
            width: 2,
            scale: Some(vec![(100.0, 50.0), (1.0, 4.0)]),
            onsets: vec![Some(0), Some(0)],
            slots: 2,
            ticks: vec![0.1, 0.1],
            ..settings(2)
        };
        let batch = EnvBatch::new(1, universe(rows, &[Market::Perp; 2], &[None, None]), perp());
        let mut env = Environment::new(batch, featured);
        env.reset(&[true], &[2], &[1, 0]);
        // the bar the second slot never printed stays 0.0, every other cell is scaled
        assert_eq!(
            &observed(&env)[..8],
            &[0.0, 0.0, 0.0, 25.0, 2.2, 52.5, 2.0, 50.0]
        );
        // the candles' five columns, a volume of a window that traded nothing left 0.0
        let quiet = bar(100.0, 100.0, 100.0, 100.0, 0.0);
        let candled = Settings {
            scale: Some(vec![
                (1.0, 0.5),
                (1.0, 0.25),
                (0.5, 0.5),
                (1.0, 1.0),
                (1.0, 2.0),
            ]),
            ..settings(1)
        };
        let row = observed(&opened(
            vec![quiet, bar(98.0, 102.0, 96.0, 100.0, 0.0)],
            candled,
            1,
        ));
        assert_eq!(&row[..5], &[-0.04, 0.08, 0.92, 0.0, 0.0]);
    }

    #[test]
    fn a_slot_is_shown_once_entered_on_a_bar_it_printed_until_it_delists() {
        // the second enters at row 1, prints nothing on row 2 and delists at row 4
        let rows = (0..6)
            .map(|t| vec![flat(100.0), if t == 2 { ABSENT } else { flat(50.0) }])
            .collect();
        let settings = Settings {
            onsets: vec![Some(0), Some(1)],
            slots: 2,
            ticks: vec![0.1, 0.1],
            ..settings(2)
        };
        let candles = universe(rows, &[Market::Perp; 2], &[None, Some(4 * HOUR)]);
        let mut env = Environment::new(EnvBatch::new(1, candles, perp()), settings);
        env.reset(&[true], &[1], &[0, 1]);
        let (mut seen, mut held) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            let mut shown = [false; 2];
            env.shown(&mut shown);
            seen.push(shown);
            // a weight of a half for the second, which it trades only where shown,
            // and none named for the first, which holds
            stepped(&mut env, Decision::Weights(&[f64::NAN, 0.5]));
            held.push((env.engines()[0].position(0), env.engines()[0].position(1)));
        }
        let mut shown = [false; 2];
        env.shown(&mut shown);
        seen.push(shown);
        // its window starts before it entered on row 1, it printed nothing on row 2,
        // and row 4 opens at its delisting
        assert_eq!(
            seen,
            vec![[true, false], [true, false], [true, true], [true, false]]
        );
        // bought only at the close of row 3, filled at row 4's open and closed as it
        // delisted
        assert_eq!(held, vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        assert!(env.engines()[0].is_delisted(1));
    }

    #[test]
    fn the_account_block_reads_its_leverage_margin_ratio_and_cash_weight() {
        let tier = Tier {
            up_to: f64::INFINITY,
            leverage: 20.0,
            rate: 0.01,
            amount: 0.0,
        };
        let tiers = Tiers::new(TierBasis::Notional, vec![tier]).expect("a table");
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 110.0, 100.0, 110.0, 1_000.0),
        ];
        let batch = EnvBatch::new(1, Candles::new(bars).with_tiers(tiers), perp());
        let mut env = Environment::new(batch, settings(1));
        env.reset(&[true], &[1], &[0]);
        stepped(&mut env, Decision::Weights(&[1.0]));
        // 100 held at 110: 11,000 of equity and of notional, 110 of it kept
        let expected = [
            1.0,
            (1_000.0 / 11_000.0) as f32,
            0.1,
            1.0,
            libm::log(11_000.0 / 10_000.0) as f32,
            1.0,
            (110.0 / 11_000.0) as f32,
            0.0,
        ];
        assert_eq!(&observed(&env)[5..], &expected);
    }

    #[test]
    fn the_sharpe_reward_is_the_differential_sharpe_of_the_log_returns() {
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 110.0, 100.0, 110.0, 1_000.0),
            bar(110.0, 110.0, 99.0, 99.0, 1_000.0),
            bar(99.0, 108.9, 99.0, 108.9, 1_000.0),
        ];
        let eta = 2.0 / 3.0;
        let settings = Settings {
            reward: Reward::Sharpe(eta),
            ..settings(2)
        };
        let mut env = opened(bars, settings, 1);
        // 100 held throughout: equity 11,000, then 9,900, then 10,890
        let returns = [
            libm::log(11_000.0 / 10_000.0),
            libm::log(9_900.0 / 11_000.0),
            libm::log(10_890.0 / 9_900.0),
        ];
        let (mut a, mut b) = (0.0, 0.0);
        for returned in returns {
            let (reward, ..) = stepped(&mut env, Decision::Weights(&[1.0]));
            let (da, db) = (returned - a, returned * returned - b);
            let spread = b - a * a;
            let expected = if spread > 0.0 {
                (b * da - a * db / 2.0) / spread.powf(1.5)
            } else {
                0.0
            };
            assert!(
                (reward - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{reward} against {expected}"
            );
            (a, b) = (a + eta * da, b + eta * db);
        }
    }

    #[test]
    fn an_episode_is_cut_after_its_length() {
        let settings = Settings {
            episode_length: Some(2),
            ..settings(1)
        };
        let mut env = opened(vec![flat(100.0); 6], settings, 1);
        let held = Decision::Weights(&[f64::NAN]);
        assert_eq!(stepped(&mut env, held), (0.0, false, false));
        assert_eq!(stepped(&mut env, held), (0.0, false, true));
    }

    #[test]
    fn an_environment_left_out_of_a_step_keeps_its_bar_its_position_and_its_flags() {
        let batch = EnvBatch::new(2, Candles::new(vec![flat(100.0); 4]), perp());
        let mut env = Environment::new(batch, settings(1));
        env.reset(&[true, true], &[1, 1], &[0, 0]);
        let (mut rewards, mut ended, mut cut) = ([1.0; 2], [true; 2], [true; 2]);
        env.step(
            Decision::Weights(&[1.0, 1.0]),
            &[true, false],
            &mut rewards,
            &mut ended,
            &mut cut,
        )
        .unwrap();
        assert_eq!((rewards[1], ended[1], cut[1]), (0.0, false, false));
        assert_eq!(env.progress(), vec![(1, 1), (1, 0)]);
        let (first, second) = (&env.engines()[0], &env.engines()[1]);
        assert_eq!(
            (first.position(0), second.tick(), second.position(0)),
            (100.0, 1, 0.0)
        );
    }

    #[test]
    fn a_short_weight_on_a_spot_slot_is_refused_before_any_environment_steps() {
        let config = EngineConfig {
            market: Market::Spot,
            ..perp()
        };
        let batch = EnvBatch::new(2, Candles::new(vec![flat(100.0); 4]), config);
        let mut env = Environment::new(batch, settings(1));
        env.reset(&[true, true], &[1, 1], &[0, 0]);
        let (mut rewards, mut ended, mut cut) = ([0.0; 2], [false; 2], [false; 2]);
        let both = env.step(
            Decision::Weights(&[0.5, -0.5]),
            &[true, true],
            &mut rewards,
            &mut ended,
            &mut cut,
        );
        assert!(matches!(
            both,
            Err((1, TargetError::Short { instrument: 0, .. }))
        ));
        assert_eq!(env.progress(), vec![(1, 0), (1, 0)]);
        // the weight of an environment left out of the step is not read
        let first = env.step(
            Decision::Weights(&[0.5, -0.5]),
            &[true, false],
            &mut rewards,
            &mut ended,
            &mut cut,
        );
        assert!(first.is_ok());
        assert_eq!(env.progress(), vec![(1, 1), (1, 0)]);
    }

    #[test]
    fn the_state_holds_what_the_last_step_traded_paid_and_left() {
        // a taker fee of 0.1% and funding of 1% charged every bar
        let config = EngineConfig {
            fee_taker: 0.001,
            funding_rate: 0.01,
            funding_interval: 1,
            ..perp()
        };
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 100.0, 90.0, 90.0, 1_000.0),
            flat(90.0),
        ];
        let mut env = Environment::new(EnvBatch::new(1, Candles::new(bars), config), settings(1));
        env.reset(&[true], &[1], &[0]);
        // a fresh account: its opening equity, and nothing traded, paid, held or lost
        let (values, liquidated, weights, positions) = state_of(&env);
        let opened = [10_000.0, 10_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(values.map(|value| value[0]), opened);
        assert_eq!(
            (liquidated, weights, positions),
            (vec![false], vec![0.0], vec![0.0])
        );
        stepped(&mut env, Decision::Weights(&[0.5]));
        let (values, liquidated, weights, positions) = state_of(&env);
        let engine = &env.engines()[0];
        let (now, paid) = (engine.equity(), engine.funding_paid());
        // 50 bought at 100 for a fee of 5, held at 90
        let expected = [
            now,
            10_000.0,
            libm::log(now / 10_000.0),
            1.0 - now / 10_000.0,
            50.0 * 90.0 / now,
            5_000.0 / 10_000.0,
            5.0 / 10_000.0,
            paid / 10_000.0,
        ];
        assert_eq!(values.map(|value| value[0]), expected);
        assert!(paid > 0.0);
        assert_eq!(
            (liquidated, weights, positions),
            (vec![false], vec![50.0 * 90.0 / now], vec![50.0])
        );
        // held a bar more: that bar's funding alone, over the equity it opened with
        stepped(&mut env, Decision::Weights(&[f64::NAN]));
        let (values, ..) = state_of(&env);
        let more = env.engines()[0].funding_paid() - paid;
        assert!(more > 0.0);
        assert_eq!(values[7][0], more / now);
    }

    #[test]
    fn each_environment_reads_and_trades_its_own_slots_of_a_universe() {
        // the first flat at 100 with no bar on row 3, the second rising one a row from
        // 50 on a rising volume
        let rows = (0..6)
            .map(|t| {
                let price = 50.0 + t as f64;
                let first = if t == 3 { ABSENT } else { flat(100.0) };
                vec![
                    first,
                    bar(price, price, price, price, 1_000.0 + 100.0 * t as f64),
                ]
            })
            .collect();
        let mut env = pair(rows, 2, settings(2));
        // the second environment opens a row later, its slots the other way round
        env.reset(&[true, true], &[2, 3], &[0, 1, 1, 0]);
        let row = observed(&env);
        let size = env.size();
        assert_eq!(row.len(), 2 * size);
        let first: Vec<f32> = [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [
                51.0 / 52.0,
                51.0 / 52.0,
                51.0 / 52.0,
                51.0 / 52.0,
                1_100.0 / 1_150.0,
            ],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1_200.0 / 1_150.0],
        ]
        .into_iter()
        .flatten()
        .map(|value| value as f32)
        .collect();
        // the first instrument printed nothing on row 3, so the second's slot of it is
        // hidden
        let second: Vec<f32> = [
            [
                52.0 / 53.0,
                52.0 / 53.0,
                52.0 / 53.0,
                52.0 / 53.0,
                1_200.0 / 1_250.0,
            ],
            [0.0; 5],
            [1.0, 1.0, 1.0, 1.0, 1_300.0 / 1_250.0],
            [0.0; 5],
        ]
        .into_iter()
        .flatten()
        .map(|value| value as f32)
        .collect();
        assert_eq!(
            (&row[..20], &row[size..size + 20]),
            (&first[..], &second[..])
        );
        let mut shown = [false; 4];
        env.shown(&mut shown);
        assert_eq!(shown, [true, true, true, false]);
        // the first buys the second instrument, the second environment's first slot
        let (mut rewards, mut ended, mut cut) = ([0.0; 2], [false; 2], [false; 2]);
        let decided = [f64::NAN, 0.5, 0.25, f64::NAN];
        env.step(
            Decision::Weights(&decided),
            &[true, true],
            &mut rewards,
            &mut ended,
            &mut cut,
        )
        .unwrap();
        let (_, _, weights, positions) = state_of(&env);
        let (early, late) = (5_000.0 / 52.0, 2_500.0 / 53.0);
        assert_eq!(positions, vec![0.0, early, late, 0.0]);
        assert_eq!(
            weights,
            vec![0.0, early * 53.0 / 10_000.0, late * 54.0 / 10_000.0, 0.0]
        );
    }

    #[test]
    fn each_environment_places_its_slots_limits_on_the_passive_side_of_their_closes() {
        let settings = Settings {
            orders: true,
            ..settings(1)
        };
        // the second falls a point a row, so each limit reads its own row's close
        let rows = (0..3)
            .map(|t| vec![flat(100.0), flat(50.0 - t as f64)])
            .collect();
        let mut env = pair(rows, 2, settings);
        env.reset(&[true, true], &[1, 1], &[0, 1, 0, 1]);
        // the first buys the first instrument and the second sells the second, each
        // half its offset of 20 ticks off
        let actions = [0.25, 0.5, 0.0, 0.0, 0.0, 0.0, -0.25, 0.5];
        let (mut rewards, mut ended, mut cut) = ([0.0; 2], [false; 2], [false; 2]);
        env.step(
            Decision::Orders(&actions),
            &[true, true],
            &mut rewards,
            &mut ended,
            &mut cut,
        )
        .unwrap();
        let first: Vec<_> = env.engines()[0].resting().collect();
        let second: Vec<_> = env.engines()[1].resting().collect();
        assert_eq!((first.len(), second.len()), (1, 1));
        assert_eq!((first[0].instrument, first[0].side), (0, Side::Buy));
        assert_eq!(first[0].price.unwrap().get(), 100.0 - 10.0 * 0.1);
        assert_eq!((second[0].instrument, second[0].side), (1, Side::Sell));
        assert_eq!(second[0].price.unwrap().get(), 49.0 + 10.0 * 0.01);
        assert!((second[0].size.get() - 2_500.0 / 49.0).abs() < 1e-12);
    }

    #[test]
    fn a_short_spot_weight_is_refused_naming_its_environment_and_instrument() {
        let rows = vec![vec![flat(100.0), flat(50.0)]; 4];
        let candles = universe(rows, &[Market::Spot; 2], &[None, None]);
        let settings = Settings {
            onsets: vec![Some(0), Some(0)],
            slots: 2,
            ticks: vec![0.1, 0.01],
            ..settings(1)
        };
        let config = EngineConfig {
            market: Market::Spot,
            ..perp()
        };
        let mut env = Environment::new(EnvBatch::new(2, candles, config), settings);
        env.reset(&[true, true], &[1, 1], &[0, 1, 1, 0]);
        let (mut rewards, mut ended, mut cut) = ([0.0; 2], [false; 2], [false; 2]);
        // a weight of zero is no short, and the second's short is its second slot's,
        // the first instrument
        let refused = env.step(
            Decision::Weights(&[0.5, 0.0, 0.0, -0.5]),
            &[true, true],
            &mut rewards,
            &mut ended,
            &mut cut,
        );
        assert!(matches!(
            refused,
            Err((1, TargetError::Short { instrument: 0, .. }))
        ));
    }

    #[test]
    fn a_slot_not_yet_listed_reads_zero_where_it_has_no_mark() {
        // the second prints its first bar on row 3
        let rows = (0..4)
            .map(|t| vec![flat(100.0), if t < 3 { ABSENT } else { flat(50.0) }])
            .collect();
        let settings = Settings {
            onsets: vec![Some(0), Some(3)],
            slots: 2,
            ticks: vec![0.1, 0.1],
            ..settings(1)
        };
        let candles = universe(rows, &[Market::Perp; 2], &[None, None]);
        let mut env = Environment::new(EnvBatch::new(1, candles, perp()), settings);
        env.reset(&[true], &[1], &[0, 1]);
        let row = observed(&env);
        // past one bar of two slots' candles and the first slot's four values
        assert_eq!(&row[14..18], &[0.0, 0.0, 0.0, 0.0]);
        assert!(row.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn an_environment_of_no_engines_is_empty() {
        let bars = || Candles::new(vec![flat(100.0); 2]);
        let none = Environment::new(EnvBatch::new(0, bars(), perp()), settings(1));
        let one = Environment::new(EnvBatch::new(1, bars(), perp()), settings(1));
        assert!(none.is_empty());
        assert_eq!(none.len(), 0);
        assert!(!one.is_empty() && one.len() == 1);
    }

    #[test]
    fn an_inverse_slot_reads_its_weight_in_the_coin_and_its_distance_in_dollars() {
        let bars = vec![
            flat(100.0),
            flat(100.0),
            bar(100.0, 110.0, 100.0, 110.0, 1e6),
        ];
        let config = EngineConfig {
            quote: 100.0,
            ..perp()
        };
        let settings = Settings {
            units: Units::Inverse(10.0),
            floor: 1.0,
            ..settings(1)
        };
        let batch = EnvBatch::new(1, Candles::new(bars).inverse(10.0), config);
        let mut env = Environment::new(batch, settings);
        env.reset(&[true], &[1], &[0]);
        let (reward, ..) = stepped(&mut env, Decision::Weights(&[1.0]));
        // 1,000 contracts of 10 dollars bought with 100 coins at 100, held at 110
        let held = 1_000.0 * 10.0 / 110.0;
        let equity = 100.0 + 1_000.0 * 10.0 * (1.0 / 100.0 - 1.0 / 110.0);
        assert!((reward - libm::log(equity / 100.0)).abs() < 1e-12);
        let row = observed(&env);
        assert!((row[5] - (held / equity) as f32).abs() < 1e-6);
        assert!((row[7] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn an_inverse_order_rests_its_contracts_at_a_dollar_price_whole_ticks_off() {
        let config = EngineConfig {
            quote: 100.0,
            ..perp()
        };
        let settings = Settings {
            units: Units::Inverse(10.0),
            orders: true,
            floor: 1.0,
            ..settings(1)
        };
        let batch = EnvBatch::new(1, Candles::new(vec![flat(100.0); 3]).inverse(10.0), config);
        let mut env = Environment::new(batch, settings);
        env.reset(&[true], &[1], &[0]);
        stepped(&mut env, Decision::Orders(&[0.25, 0.5]));
        let resting: Vec<_> = env.engines()[0].resting().collect();
        // a quarter of 100 coins at 100 is 250 contracts, bought ten ticks below at 99
        assert_eq!(resting[0].side, Side::Sell);
        assert!((resting[0].price.unwrap().get() - 1.0 / 99.0).abs() < 1e-15);
        assert!((resting[0].size.get() - 2_500.0).abs() < 1e-9);
    }

    #[test]
    fn a_reset_opens_only_the_environments_its_mask_names() {
        let batch = EnvBatch::new(2, Candles::new(vec![flat(100.0); 5]), perp());
        let mut env = Environment::new(batch, settings(1));
        env.reset(&[true, true], &[1, 1], &[0, 0]);
        let (mut rewards, mut ended, mut cut) = ([0.0; 2], [false; 2], [false; 2]);
        env.step(
            Decision::Weights(&[1.0, 1.0]),
            &[true, true],
            &mut rewards,
            &mut ended,
            &mut cut,
        )
        .unwrap();
        env.reset(&[false, true], &[3, 3], &[0, 0]);
        assert_eq!(env.progress(), vec![(1, 1), (3, 0)]);
        let (first, second) = (&env.engines()[0], &env.engines()[1]);
        assert_eq!((first.position(0), second.position(0)), (100.0, 0.0));
    }

    #[test]
    fn a_slot_reports_its_resting_order_and_a_size_of_zero_takes_it_back() {
        let settings = Settings {
            orders: true,
            ..settings(1)
        };
        let mut env = opened(vec![flat(50_000.0); 4], settings, 1);
        stepped(&mut env, Decision::Orders(&[0.25, 0.5]));
        // flat bars never reach a limit ten ticks below them
        assert_eq!(&observed(&env)[5..11], &[0.0, 0.0, 0.0, 1.0, 1.0, 10.0]);
        stepped(&mut env, Decision::Orders(&[0.0, 0.5]));
        assert_eq!(&observed(&env)[5..11], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_window_that_traded_nothing_reads_a_volume_of_zero() {
        let quiet = bar(100.0, 100.0, 100.0, 100.0, 0.0);
        let env = opened(vec![quiet; 3], settings(2), 2);
        let row = observed(&env);
        assert_eq!((row[4], row[9]), (0.0, 0.0));
    }

    #[test]
    fn a_step_gives_the_same_results_on_any_number_of_threads() {
        let rows: Vec<Vec<Candle>> = (0..12)
            .map(|t| {
                let (a, b) = (100.0 + t as f64, 50.0 - 0.5 * t as f64);
                vec![
                    bar(a, a + 2.0, a - 3.0, a + 1.0, 1e6),
                    bar(b, b + 1.0, b - 1.0, b, 1e6),
                ]
            })
            .collect();
        let run = |threads: usize| {
            let settings = Settings {
                threads: Some(threads),
                ..settings(3)
            };
            let mut env = pair(rows.clone(), 5, settings);
            assert_eq!(env.threads(), threads);
            env.reset(&[true; 5], &[2, 3, 2, 4, 2], &[0, 1].repeat(5));
            let mut seen = Vec::new();
            for step in 0..6 {
                let weights: Vec<f64> = (0..10)
                    .map(|j| ((step * 7 + j) % 5) as f64 / 2.0 - 1.0)
                    .collect();
                let (mut rewards, mut ended, mut cut) = ([0.0; 5], [false; 5], [false; 5]);
                env.step(
                    Decision::Weights(&weights),
                    &[true; 5],
                    &mut rewards,
                    &mut ended,
                    &mut cut,
                )
                .unwrap();
                seen.push((rewards, ended, cut, observed(&env)));
            }
            seen
        };
        assert_eq!(run(1), run(3));
    }

    #[test]
    fn a_batch_takes_a_thread_for_each_2_17_floats_it_observes_up_to_half_the_cpus() {
        let half = (rayon::current_num_threads() / 2).max(1);
        assert_eq!(super::threads_for(0, 328), 1);
        assert_eq!(super::threads_for(399, 328), 1);
        // 400 environments of 328 floats are 131,200, past 2^17
        assert_eq!(super::threads_for(400, 328), 2.min(half));
        assert_eq!(super::threads_for(4_096, 328), 11.min(half));
        assert_eq!(super::threads_for(1 << 20, 328), half);
        let env = pair(vec![vec![flat(100.0), flat(50.0)]; 4], 2, settings(1));
        assert_eq!(env.threads(), 1);
    }

    #[test]
    fn the_account_block_alone_is_the_observation_after_its_market_block() {
        let rows: Vec<Vec<Candle>> = (0..6)
            .map(|t| {
                let p = 100.0 + t as f64;
                vec![
                    bar(p, p + 2.0, p - 1.0, p + 1.0, 500.0),
                    flat(50.0 - t as f64),
                ]
            })
            .collect();
        let ordering = Settings {
            orders: true,
            ..settings(2)
        };
        for settings in [settings(2), ordering] {
            let mut env = pair(rows.clone(), 2, settings);
            env.reset(&[true, true], &[2, 3], &[0, 1, 1, 0]);
            let decision = [0.5, 0.25, -0.5, 0.75];
            let (mut rewards, mut ended, mut cut) = ([0.0; 2], [false; 2], [false; 2]);
            let step = if env.settings().orders {
                Decision::Orders(&[0.5, 0.0, -0.25, 1.0, 0.1, 0.5, 0.0, 0.0])
            } else {
                Decision::Weights(&decision)
            };
            env.step(step, &[true, true], &mut rewards, &mut ended, &mut cut)
                .unwrap();
            let (size, account) = (env.size(), env.account_size());
            let mut alone = vec![1.0f32; 2 * account];
            env.observe_account(&mut alone);
            let whole = observed(&env);
            for e in 0..2 {
                assert_eq!(
                    &alone[e * account..(e + 1) * account],
                    &whole[(e + 1) * size - account..(e + 1) * size]
                );
            }
            assert_eq!(account, if env.settings().orders { 16 } else { 12 });
            assert_eq!(env.ticks(), vec![3, 4]);
        }
    }

    #[test]
    fn a_bar_s_cells_and_window_mean_are_what_the_observation_writes_from() {
        // the second instrument prints nothing on row 1, and trades nothing on rows 2
        // and 3, so its window over rows 2 and 3 has a mean volume of 0
        let rows = vec![
            vec![
                bar(100.0, 104.0, 98.0, 102.0, 300.0),
                bar(50.0, 51.0, 49.0, 50.5, 70.0),
            ],
            vec![bar(102.0, 103.0, 99.0, 100.0, 100.0), ABSENT],
            vec![
                bar(100.0, 101.0, 97.0, 98.0, 200.0),
                bar(50.5, 52.0, 50.0, 51.0, 0.0),
            ],
            vec![
                bar(98.0, 99.0, 95.0, 96.0, 400.0),
                bar(51.0, 51.5, 50.5, 51.0, 0.0),
            ],
        ];
        let candles = pair(rows.clone(), 1, settings(2));
        // the printed bars, row-major over the rows and the instruments
        let bars = candles.candles();
        assert_eq!(bars.len(), 8);
        assert_eq!((bars[2].close, bars[6].close), (100.0, 96.0));
        assert!(bars[3].close.is_nan());
        let means = candles.window_means();
        assert!(means[0].is_nan() && means[1].is_nan());
        assert_eq!(&means[2..], &[200.0, 70.0, 150.0, 0.0, 300.0, 0.0]);
        // a window of the one bar the second instrument did not print has no mean
        let single = pair(rows.clone(), 1, settings(1)).window_means();
        assert_eq!(&single[..3], &[300.0, 70.0, 100.0]);
        assert!(single[3].is_nan());
        assert_eq!(&single[4..], &[200.0, 0.0, 400.0, 0.0]);
        assert_eq!(candles.cells(), None);
        let values: Vec<f64> = (0..4 * 2 * 2).map(|v| v as f64).collect();
        let featured = Settings {
            features: Some(values.into()),
            width: 2,
            scale: Some(vec![(1.0, 2.0), (0.0, 4.0)]),
            ..settings(2)
        };
        let mut env = pair(rows, 1, featured);
        let cells = env.cells().unwrap();
        assert_eq!(&cells[..8], &[-0.5, 0.25, 0.5, 0.75, 1.5, 1.25, 0.0, 0.0]);
        // the observation's market block is the cells of the window's rows
        env.reset(&[true], &[3], &[0, 1]);
        let row = observed(&env);
        let window = [&cells[8..12], &cells[12..16]].concat();
        assert_eq!(&row[..8], &window[..]);
    }

    #[test]
    fn a_clone_or_a_fork_of_environments_steps_on_as_they_do() {
        let rows: Vec<Vec<Candle>> = (0..12)
            .map(|t| {
                let p = 100.0 + 5.0 * (t as f64 * 0.9).sin();
                vec![
                    bar(p, p + 2.0, p - 1.0, p + 1.0, 500.0),
                    flat(50.0 + t as f64),
                ]
            })
            .collect();
        let mut env = pair(rows, 3, settings(2));
        env.reset(&[true; 3], &[2, 3, 4], &[0, 1, 1, 0, 0, 1]);
        // each environment's weights at step k, so a fork's copies take environment 1's
        let weights = |e: usize, k: usize| {
            [
                ((e + k) % 5) as f64 / 5.0 - 0.4,
                ((2 * e + k) % 3) as f64 / 3.0 - 0.3,
            ]
        };
        let step = |env: &mut Environment, of: &[usize], k: usize| {
            let n = env.len();
            let decision: Vec<f64> = of.iter().flat_map(|&e| weights(e, k)).collect();
            let (mut rewards, mut ended, mut cut) = (vec![0.0; n], vec![false; n], vec![false; n]);
            env.step(
                Decision::Weights(&decision),
                &vec![true; n],
                &mut rewards,
                &mut ended,
                &mut cut,
            )
            .unwrap();
            (rewards, ended, cut, observed(env))
        };
        step(&mut env, &[0, 1, 2], 0);
        let (mut twin, mut fork) = (env.clone(), env.fork(1, 2));
        assert_eq!((fork.len(), fork.ticks()), (2, vec![4, 4]));
        let size = env.size();
        for k in 1..6 {
            let ahead = step(&mut env, &[0, 1, 2], k);
            assert_eq!(step(&mut twin, &[0, 1, 2], k), ahead);
            let (rewards, ended, cut, rows) = step(&mut fork, &[1, 1], k);
            for copy in 0..2 {
                assert_eq!(
                    (rewards[copy], ended[copy], cut[copy]),
                    (ahead.0[1], ahead.1[1], ahead.2[1])
                );
                assert_eq!(
                    &rows[copy * size..(copy + 1) * size],
                    &ahead.3[size..2 * size]
                );
            }
        }
    }

    #[test]
    fn a_clone_or_a_fork_of_environments_shares_their_features_as_it_shares_the_bars() {
        let featured = Settings {
            features: Some((0..4 * 2 * 2).map(|v| v as f64).collect()),
            width: 2,
            ..settings(2)
        };
        let env = pair(vec![vec![flat(100.0), flat(50.0)]; 4], 3, featured);
        let held = |e: &Environment| e.settings().features.as_deref().map(<[f64]>::as_ptr);
        assert_eq!(held(&env.clone()), held(&env));
        assert_eq!(held(&env.fork(1, 2)), held(&env));
    }
}
