//! Perp funding, charged on the bars of the series that fall on an interval
//! boundary, on the position held then: a positive rate is paid by a long and
//! received by a short, and what was paid is kept for the result (ADR 0020).

use probatum_core::{Candle, Price};

use super::Engine;

impl Engine {
    /// Charge funding on the position held now, marked at `bar`'s close, when the
    /// current bar falls on an interval boundary. The boundary is read off the
    /// bar's index in the series rather than the bars since a reset, so every env
    /// of a batch is charged on the same bars however its episode was offset
    /// (ADR 0020).
    pub(super) fn charge_funding(&mut self, bar: &Candle) {
        if self.config.funding_interval > 0 && self.tick % self.config.funding_interval == 0 {
            self.funding_paid += self
                .account
                .apply_funding(self.config.funding_rate, Price(bar.close));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::candles::Candles;
    use crate::engine::fixtures::{cfg, flat_series, levered, ohlc, perp_cfg, series};
    use crate::engine::{Engine, EngineConfig, CLOSE_EPS};
    use probatum_core::Market;

    #[test]
    fn perp_funding_charges_a_held_long_each_interval() {
        // funding every bar at 0.001 of notional; a long pays, a short receives
        let mut config = perp_cfg();
        config.quote = 10_000.0;
        config.funding_rate = 0.001;
        config.funding_interval = 1;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        let s1 = e.step(); // long 1 @ 200; funding at bar 1 close 250: pay 0.25
        let q1 = s1.quote;
        assert!((10_000.0 - q1 - 0.25).abs() < 1e-9); // 1 * 250 * 0.001
        let s2 = e.step(); // funding at bar 2 close 350: pay another 0.35
        assert!((q1 - s2.quote - 0.35).abs() < 1e-9);
    }

    #[test]
    fn funding_is_off_by_default_and_on_spot() {
        // default config funds nothing (interval 0); spot never funds either
        let mut e = Engine::new(series(), cfg()); // spot, no funding
        e.reset();
        e.market_buy(1.0);
        let s = e.step();
        assert_eq!(s.quote, 10_000.0 - 200.0); // only the buy, no funding debit
    }

    #[test]
    fn funding_can_bust_the_account_on_the_same_bar() {
        // a punitive funding rate on a leveraged long drains equity below zero within
        // the step, before the liquidation check, so the account busts (ADR 0020)
        let mut config = perp_cfg(); // quote 100
        config.funding_rate = 1.0; // 100% of notional per event, for the test
        config.funding_interval = 1;
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(2.0); // long 2 @ 100, notional 200 on 100 equity
        let s = e.step(); // funding 2*100*1.0 = 200 debited -> equity -100
        assert!(e.is_bust());
        assert!(s.equity <= 0.0);
    }

    #[test]
    fn funding_is_charged_only_on_interval_bars() {
        // every other funding test uses interval 1, where the modulo is trivially
        // true on every bar, so the schedule itself is never exercised (ADR 0020)
        let mut config = perp_cfg();
        config.quote = 10_000.0;
        config.funding_rate = 0.001;
        config.funding_interval = 3;
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 10_000.0);
        let mut e = Engine::new(Candles::new(vec![flat; 8]), config);
        e.reset();
        e.market_buy(1.0);
        let mut charged = Vec::new();
        let mut prev = 10_000.0;
        for _ in 0..7 {
            let s = e.step();
            charged.push((prev - s.quote).abs() > 1e-12);
            prev = s.quote;
        }
        // ticks 1..7; funding fires where tick % 3 == 0, so on ticks 3 and 6 only
        assert_eq!(
            charged,
            vec![false, false, true, false, false, true, false],
            "charged on {charged:?}"
        );
    }

    #[test]
    fn funding_is_kept_where_the_result_can_read_it() {
        // the account returned every payment and nothing kept it, so the one cost
        // unique to a perp was unmeasurable from a finished run, and a carry could
        // not be told apart from a direction (ADR 0020)
        let config = EngineConfig {
            market: Market::Perp,
            funding_rate: 0.001,
            funding_interval: 1,
            report: true,
            ..cfg()
        };
        let mut long = Engine::new(series(), config);
        long.reset();
        long.market_buy(1.0);
        let s1 = long.step(); // fills at the open of 200, funded at the close of 250
        assert!((s1.funding_paid - 0.25).abs() < CLOSE_EPS);
        let s2 = long.step(); // funded again at the close of 350
        assert!((s2.funding_paid - 0.60).abs() < CLOSE_EPS);
        let stats = long.stats(365.0, 0.0).expect("reporting is on");
        assert!((stats.funding_paid - 0.60).abs() < CLOSE_EPS);

        // a positive rate is paid BY the long and TO the short, so the sign flips
        let mut short = Engine::new(series(), config);
        short.reset();
        short.market_sell(1.0);
        let s = short.step();
        assert!((s.funding_paid + 0.25).abs() < CLOSE_EPS);
    }

    #[test]
    fn funding_on_a_liquidating_bar_marks_at_the_fence_and_not_at_the_raw_close() {
        // ADR 0020. Every order on a fenced bar resolves against the clipped
        // candle, which `fenced` calls the honest picture of a market the account
        // was not in, and funding reached past it to the raw close. This short
        // dies at 110 on a bar that closes at 128, so the two readings are 11.0
        // and 12.8: the second credits the account for 18 points of a rise it had
        // already been closed out of, and funding_paid is reported
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 130.0, 100.0, 128.0, 1e6),
        ]);
        let config = EngineConfig {
            funding_rate: 0.01,
            funding_interval: 2, // tick 1 is not a boundary, tick 2 is
            ..levered()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_sell(10.0); // short 10 at the open of 100, on 100 of margin
        let state = e.step();
        assert_eq!(state.position, -10.0);
        assert_eq!(state.funding_paid, 0.0, "tick 1 is not a funding boundary");

        // the margin runs out at 110, so the bar the account was in closes there
        let state = e.step();
        assert!(
            (state.funding_paid + 11.0).abs() < 1e-9,
            "funding was marked at {}, not at the fence",
            state.funding_paid / -0.1
        );
        assert!(
            (state.funding_paid + 12.8).abs() > 1e-9,
            "funding was marked at the raw close of a bar the account never saw"
        );
        // and the credit lands before the liquidation, so the exit moves with it:
        // 100 of margin plus 11 of funding runs out at 111.1 rather than 110
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        // ADR 0018's nothing left, to the last bit the funding arithmetic allows:
        // 100 of margin, 11 of credit and an exit at 111.1 land 5.7e-14 off zero
        assert!(state.equity.abs() < 1e-9);
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert!((trades[0].exit_price - 111.1).abs() < 1e-9);
        assert!(trades[0].liquidated);
    }

    #[test]
    fn funding_fires_on_the_absolute_bar_index_not_on_bars_since_the_reset() {
        // ADR 0020: the funding schedule belongs to the series, so every
        // env in a vectorized run charges the same bars however its episode was
        // offset. Only an offset start separates the two counters, and nothing did
        // that: starting at bar 2 with an interval of 3 funds ticks 3 and 6, while
        // a per-episode counter funds the third bar of the episode, tick 5, and
        // nothing else in this window
        let config = EngineConfig {
            quote: 10_000.0,
            funding_rate: 0.0625, // one sixteenth, so each charge is exactly 6.25
            funding_interval: 3,
            ..perp_cfg()
        };
        let mut e = Engine::new(flat_series(), config); // eight flat bars at 100
        e.reset_at(2);
        e.market_buy(1.0); // fills at the open of bar 3, ahead of that bar's funding
        let mut charged = Vec::new();
        for _ in 0..5 {
            let state = e.step();
            charged.push((state.tick_index, state.funding_paid));
        }
        // 1 unit at 100 pays 6.25 a time, on tick 3 and on tick 6 and on no other
        assert_eq!(
            charged,
            vec![(3, 6.25), (4, 6.25), (5, 6.25), (6, 12.5), (7, 12.5)]
        );
        assert_eq!(e.current_state().quote, 9_987.5);
    }
}
