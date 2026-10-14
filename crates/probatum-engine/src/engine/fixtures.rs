//! The bars and configurations the engine's tests share.

use probatum_core::{Candle, Market, Tiers};

use super::{EngineConfig, Intrabar, Liquidation};
use crate::candles::{Candles, Instrument};

pub(super) fn ohlc(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
    Candle {
        open,
        high,
        low,
        close,
        volume,
    }
}

pub(super) fn series() -> Candles {
    Candles::new(vec![
        ohlc(100.0, 160.0, 90.0, 150.0, 1000.0),
        ohlc(200.0, 260.0, 190.0, 250.0, 1000.0),
        ohlc(300.0, 360.0, 290.0, 350.0, 1000.0),
    ])
}

// A first bar at 100, then a bar that dips to a low of 94, then a calm bar.
pub(super) fn dip_series() -> Candles {
    Candles::new(vec![
        ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
        ohlc(100.0, 101.0, 94.0, 100.0, 1000.0),
        ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
    ])
}

// A first bar at 100, then a crash to a low of 80.
pub(super) fn crash_series() -> Candles {
    Candles::new(vec![
        ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
        ohlc(100.0, 101.0, 80.0, 85.0, 1000.0),
        ohlc(85.0, 86.0, 84.0, 85.0, 1000.0),
    ])
}

/// Spot, no fees, no slippage, so fills land on round numbers.
pub(super) fn cfg() -> EngineConfig {
    EngineConfig {
        market: Market::Spot,
        quote: 10_000.0,
        idle_cash: 0.0,
        fee_taker: 0.0,
        fee_maker: 0.0,
        slippage_bps: 0.0,
        max_fill_fraction: 1.0,
        max_open_orders: 8,
        report: false,
        max_leverage: 0.0,
        impact: 0.0,
        funding_rate: 0.0,
        funding_interval: 0,
        intrabar: Intrabar::Pessimistic,
        liquidation: Liquidation::Takeover,
    }
}

pub(super) fn perp_cfg() -> EngineConfig {
    EngineConfig {
        market: Market::Perp,
        quote: 100.0,
        ..cfg()
    }
}

pub(super) fn report_cfg() -> EngineConfig {
    EngineConfig {
        report: true,
        ..cfg()
    }
}

pub(super) fn levered() -> EngineConfig {
    EngineConfig {
        max_leverage: 10.0,
        report: true,
        ..perp_cfg()
    }
}

// eight flat bars, so nothing moves but the position
pub(super) fn flat_series() -> Candles {
    Candles::new(
        (0..8)
            .map(|_| ohlc(100.0, 100.0, 100.0, 100.0, 1e6))
            .collect::<Vec<_>>(),
    )
}

/// What a row holds for an instrument that printed no bar on it.
pub(super) const ABSENT: Candle = Candle {
    open: f64::NAN,
    high: f64::NAN,
    low: f64::NAN,
    close: f64::NAN,
    volume: f64::NAN,
};

/// A universe of instruments trading on `markets`, each row of `rows` a bar per
/// instrument, with no tier tables.
pub(super) fn universe(markets: &[Market], rows: Vec<Vec<Candle>>) -> Candles {
    let bars: Vec<Candle> = rows.into_iter().flatten().collect();
    let specs = markets
        .iter()
        .map(|&market| Instrument {
            market: Some(market),
            tiers: Tiers::default(),
        })
        .collect();
    Candles::universe(bars, markets.len(), None).with_instruments(specs)
}
