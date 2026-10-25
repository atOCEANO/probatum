//! The window a fill reads (Specification, The window and The spread estimate): for
//! each row, an instrument's last bars before it, their mean volume, the sample
//! deviation of their log returns close to close, and the spread Ardia, Guidotti and
//! Kroencke's estimator reads from them. Read once for every env that shares the
//! series, since no fill's size or price may depend on a bar printed after its
//! order was decided.

use std::collections::VecDeque;
use std::sync::Arc;

use probatum_core::Candle;
use serde::{Deserialize, Serialize};

use crate::candles::printed;

/// What the window before a row shows of one instrument.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Trailing {
    /// The mean volume of its bars, zero with none.
    pub volume: f64,
    /// The sample standard deviation of their log returns close to close, each from
    /// the bar before it, zero with fewer than two.
    pub sigma: f64,
    /// The spread as a fraction of the price where it is estimated, and zero where
    /// the estimate is undefined or not asked for.
    pub spread: f64,
}

/// Each row's window of the last `bars` bars for every instrument of a series held
/// row by row: the mean volume, and the deviation and the spread where a fill model
/// reads them, none where no fill model does.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Windows {
    pub(crate) bars: usize,
    pub(crate) volume: Arc<[f64]>,
    pub(crate) sigma: Option<Arc<[f64]>>,
    pub(crate) spread: Option<Arc<[f64]>>,
}

/// A bar a window holds: its volume, its log return from the close its instrument
/// printed before it, and the logs the spread estimate reads, each taken once as it
/// enters and only where it is read.
#[derive(Clone, Copy)]
struct Seen {
    volume: f64,
    ret: f64,
    logs: [f64; 5],
}

impl Windows {
    /// Read the windows of `series`, `rows` rows of `width` bars, an instrument that
    /// printed no bar on a row adding nothing to its window there: the deviation
    /// where `sigma` and the spread where `estimated`.
    pub(crate) fn read(
        series: &[Candle],
        rows: usize,
        width: usize,
        bars: usize,
        sigma: bool,
        estimated: bool,
    ) -> Windows {
        let cells = rows * width;
        let mut volume = vec![0.0; cells];
        let mut deviations = vec![0.0; if sigma { cells } else { 0 }];
        let mut spreads = vec![0.0; if estimated { cells } else { 0 }];
        // each instrument's window, its own bars and the bar before them, whose close
        // the first of their returns is measured from, a window longer than the series
        // holding all of it; read row by row as the series lies, since a column at a
        // time reads a wide universe a cache line a bar
        let mut windows: Vec<VecDeque<Seen>> = (0..width)
            .map(|_| VecDeque::with_capacity(bars.min(rows) + 1))
            .collect();
        let mut closes = vec![f64::NAN; width];
        let mut logs: Vec<[f64; 5]> = Vec::new();
        for t in 0..rows {
            for i in 0..width {
                let (recent, close) = (&mut windows[i], &mut closes[i]);
                let k = t * width + i;
                let held = recent.len().min(bars);
                let window = recent.iter().skip(recent.len() - held);
                if held > 0 {
                    volume[k] = window.clone().map(|seen| seen.volume).sum::<f64>() / held as f64;
                }
                if sigma {
                    deviations[k] = deviation(recent);
                }
                if estimated {
                    logs.clear();
                    logs.extend(window.map(|seen| seen.logs));
                    spreads[k] = estimate(&logs);
                }
                let bar = series[k];
                if printed(&bar) {
                    if recent.len() == bars.saturating_add(1) {
                        recent.pop_front();
                    }
                    recent.push_back(Seen {
                        volume: bar.volume,
                        ret: if sigma {
                            libm::log(bar.close / *close)
                        } else {
                            0.0
                        },
                        logs: if estimated { logged(&bar) } else { [0.0; 5] },
                    });
                    *close = bar.close;
                }
            }
        }
        Windows {
            bars,
            volume: volume.into(),
            sigma: sigma.then(|| deviations.into()),
            spread: estimated.then(|| spreads.into()),
        }
    }

    /// What the window of cell `k` shows.
    #[inline]
    pub(crate) fn at(&self, k: usize) -> Trailing {
        Trailing {
            volume: self.volume[k],
            sigma: self.sigma.as_ref().map_or(0.0, |sigma| sigma[k]),
            spread: self.spread.as_ref().map_or(0.0, |spread| spread[k]),
        }
    }
}

/// The sample standard deviation of the log returns of the bars `recent` holds after
/// its first, each from the close before it, zero with fewer than two.
fn deviation(recent: &VecDeque<Seen>) -> f64 {
    if recent.len() < 3 {
        return 0.0;
    }
    let returns = recent.iter().skip(1).map(|seen| seen.ret);
    let n = (recent.len() - 1) as f64;
    let mean = returns.clone().sum::<f64>() / n;
    let squares = returns.map(|r| (r - mean) * (r - mean)).sum::<f64>();
    (squares / (n - 1.0)).sqrt()
}

/// Ardia, Guidotti and Kroencke's estimate of the spread over the consecutive pairs
/// of `bars`, as a fraction of the price, floored at zero, and zero where it is
/// undefined: with fewer than three bars, fewer than two pairs across which the
/// price moved, or none where the open or the close left an extreme (Specification,
/// The spread estimate).
pub fn edge(bars: &[Candle]) -> f64 {
    estimate(&bars.iter().map(logged).collect::<Vec<_>>())
}

/// The logs of a bar's open, high, low and close, and the mid of its log high and
/// log low, which the estimate reads.
fn logged(bar: &Candle) -> [f64; 5] {
    let (o, h, l, c) = (
        libm::log(bar.open),
        libm::log(bar.high),
        libm::log(bar.low),
        libm::log(bar.close),
    );
    [o, h, l, c, (h + l) / 2.0]
}

/// The estimate over bars whose logs `logged` took.
fn estimate(logs: &[[f64; 5]]) -> f64 {
    if logs.len() < 3 {
        return 0.0;
    }
    let over = |f: fn(&[f64; 5], &[f64; 5]) -> f64| -> Vec<f64> {
        logs.windows(2).map(|pair| f(&pair[0], &pair[1])).collect()
    };
    let r1 = over(|_, b| b[4] - b[0]);
    let r2 = over(|a, b| b[0] - a[4]);
    let r3 = over(|a, b| b[4] - a[3]);
    let r4 = over(|a, _| a[3] - a[4]);
    let r5 = over(|a, b| b[0] - a[3]);
    let tau = over(|a, b| indicator(b[1] != b[2] || b[2] != a[3]));
    let moved = |flags: Vec<f64>| -> f64 {
        mean(
            &tau.iter()
                .zip(&flags)
                .map(|(w, f)| w * f)
                .collect::<Vec<_>>(),
        )
    };
    let po =
        moved(over(|_, b| indicator(b[0] != b[1]))) + moved(over(|_, b| indicator(b[0] != b[2])));
    let pc =
        moved(over(|a, _| indicator(a[3] != a[1]))) + moved(over(|a, _| indicator(a[3] != a[2])));
    let pt = mean(&tau);
    if tau.iter().sum::<f64>() < 2.0 || po == 0.0 || pc == 0.0 {
        return 0.0;
    }
    let centred = |r: &[f64]| -> Vec<f64> {
        let shift = mean(r) / pt;
        r.iter().zip(&tau).map(|(r, w)| r - shift * w).collect()
    };
    let (d1, d3, d5) = (centred(&r1), centred(&r3), centred(&r5));
    let x1: Vec<f64> = (0..r1.len())
        .map(|k| -4.0 / po * d1[k] * r2[k] - 4.0 / pc * d3[k] * r4[k])
        .collect();
    let x2: Vec<f64> = (0..r1.len())
        .map(|k| -4.0 / po * d1[k] * r5[k] - 4.0 / pc * d5[k] * r4[k])
        .collect();
    let (e1, e2) = (mean(&x1), mean(&x2));
    let v1 = mean(&x1.iter().map(|x| x * x).collect::<Vec<_>>()) - e1 * e1;
    let v2 = mean(&x2.iter().map(|x| x * x).collect::<Vec<_>>()) - e2 * e2;
    let s2 = if v1 + v2 > 0.0 {
        (v2 * e1 + v1 * e2) / (v1 + v2)
    } else {
        (e1 + e2) / 2.0
    };
    s2.max(0.0).sqrt()
}

/// One where `holds`, zero otherwise.
fn indicator(holds: bool) -> f64 {
    f64::from(u8::from(holds))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::{edge, Windows};
    use probatum_core::Candle;

    fn bar(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
        Candle {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn the_specifications_six_bars_estimate_a_spread_of_0_1027_percent() {
        let bars = [
            bar(100.0, 100.5, 99.8, 100.2, 1.0),
            bar(100.3, 100.6, 100.0, 100.0, 1.0),
            bar(100.1, 100.6, 99.9, 100.4, 1.0),
            bar(100.5, 100.8, 100.1, 100.1, 1.0),
            bar(100.2, 100.7, 100.0, 100.5, 1.0),
            bar(100.6, 100.9, 100.3, 100.3, 1.0),
        ];
        let s = edge(&bars);
        assert!((s * s - 1.0553e-6).abs() < 5e-11, "s2 {}", s * s);
        // half of it moves a buy whose reference is 100.40 up by 0.05
        assert!((s * 100.40 / 2.0 - 0.05).abs() < 0.005);
    }

    #[test]
    fn an_estimate_without_enough_movement_is_a_spread_of_zero() {
        let moving = bar(100.0, 101.0, 99.0, 100.5, 1.0);
        assert_eq!(edge(&[moving, moving]), 0.0);
        let flat = bar(100.0, 100.0, 100.0, 100.0, 1.0);
        assert_eq!(edge(&[flat, flat, flat, flat]), 0.0);
        // the price moves across every pair, but no open leaves an extreme
        let level = |price: f64| bar(price, price, price, price, 1.0);
        assert_eq!(edge(&[level(100.0), level(101.0), level(102.0)]), 0.0);
    }

    #[test]
    fn a_row_reads_the_last_bars_its_instrument_printed_before_it() {
        // closes of 100, 110 and 99 with volumes of 10, 20 and 30, a window of two
        // bars, and a row the instrument printed no bar on, which adds nothing
        let nan = f64::NAN;
        let series = [
            bar(100.0, 100.0, 100.0, 100.0, 10.0),
            bar(110.0, 110.0, 110.0, 110.0, 20.0),
            bar(nan, nan, nan, nan, nan),
            bar(99.0, 99.0, 99.0, 99.0, 30.0),
            bar(99.0, 99.0, 99.0, 99.0, 40.0),
        ];
        let windows = Windows::read(&series, 5, 1, 2, true, false);
        let read: Vec<(f64, f64)> = (0..5)
            .map(|k| (windows.at(k).volume, windows.at(k).sigma))
            .collect();
        assert_eq!(read[0], (0.0, 0.0));
        assert_eq!(read[1], (10.0, 0.0));
        assert_eq!(read[2], (15.0, 0.0));
        assert_eq!(read[3], (15.0, 0.0));
        // the window of the fifth row is the second and fourth bars, and its returns
        // run from the first bar's close
        assert_eq!(read[4].0, 25.0);
        assert!(
            (read[4].1 - 0.14189560954670769).abs() < 1e-15,
            "{}",
            read[4].1
        );
    }

    #[test]
    fn a_window_longer_than_the_series_reads_all_of_it() {
        let series = [
            bar(100.0, 100.0, 100.0, 100.0, 10.0),
            bar(110.0, 110.0, 110.0, 110.0, 20.0),
            bar(99.0, 99.0, 99.0, 99.0, 30.0),
        ];
        let windows = Windows::read(&series, 3, 1, usize::MAX, false, false);
        assert_eq!(windows.at(2).volume, 15.0);
        // the deviation and the spread, not asked for, read as zero
        assert_eq!((windows.at(2).sigma, windows.at(2).spread), (0.0, 0.0));
    }
}
