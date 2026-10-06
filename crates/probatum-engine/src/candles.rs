//! The shared candle series. One immutable copy backs every env through an
//! `Arc`, so thousands of envs share it with no duplication, and a window is a
//! zero-copy borrow into it.

use std::sync::Arc;

use probatum_core::Candle;

use crate::clock::Clock;

/// An immutable OHLCV series shared cheaply across envs.
#[derive(Clone, Debug)]
pub struct Candles {
    bars: Arc<[Candle]>,
    clock: Option<Clock>,
}

impl Candles {
    /// Build a series from anything that becomes an `Arc<[Candle]>`, such as a
    /// `Vec<Candle>`. The bars are stored once and shared by clone. A series built
    /// here has no clock, so nothing that reads time can run on it.
    pub fn new(bars: impl Into<Arc<[Candle]>>) -> Candles {
        Candles {
            bars: bars.into(),
            clock: None,
        }
    }

    /// The same, with each bar's open time on `clock`. Panics unless the clock has
    /// one row per bar.
    pub fn on_clock(bars: impl Into<Arc<[Candle]>>, clock: Clock) -> Candles {
        let bars = bars.into();
        assert_eq!(clock.len(), bars.len(), "the clock needs one time per bar");
        Candles {
            bars,
            clock: Some(clock),
        }
    }

    /// The series' clock, or `None` when its bars came without times.
    pub fn clock(&self) -> Option<&Clock> {
        self.clock.as_ref()
    }

    /// Number of bars, fixed for the life of the series and shared by every clone.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// True when the series has no bars, which is what `Engine::new` refuses.
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// The bar at `i`, copied out of the shared buffer, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<Candle> {
        self.bars.get(i).copied()
    }

    /// The whole series as a slice, a zero-copy borrow.
    pub fn as_slice(&self) -> &[Candle] {
        &self.bars
    }

    /// The `len` bars ending just before `end`, as a zero-copy borrow. Clamped
    /// to what exists: the start saturates at zero and `end` saturates at the
    /// series length, so out-of-range requests return the available bars.
    pub fn window(&self, end: usize, len: usize) -> &[Candle] {
        let end = end.min(self.bars.len());
        let start = end.saturating_sub(len);
        &self.bars[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::Candles;
    use crate::clock::Clock;
    use probatum_core::Candle;

    fn series(n: usize) -> Candles {
        let bars: Vec<Candle> = (0..n)
            .map(|i| Candle {
                open: i as f64,
                high: i as f64 + 1.0,
                low: i as f64 - 1.0,
                close: i as f64 + 0.5,
                volume: 100.0,
            })
            .collect();
        Candles::new(bars)
    }

    #[test]
    fn len_and_get() {
        let c = series(5);
        assert_eq!(c.len(), 5);
        assert!(!c.is_empty());
        assert_eq!(c.get(2).unwrap().open, 2.0);
        assert!(c.get(5).is_none());
    }

    #[test]
    fn window_is_the_preceding_bars() {
        let c = series(10);
        let w = c.window(5, 3); // bars 2, 3, 4
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].open, 2.0);
        assert_eq!(w[2].open, 4.0);
    }

    #[test]
    fn window_saturates_at_the_start() {
        let c = series(10);
        let w = c.window(2, 5); // only bars 0, 1 exist before index 2
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].open, 0.0);
    }

    #[test]
    fn window_clamps_past_the_end() {
        let c = series(4);
        let w = c.window(99, 2); // end clamps to 4, so bars 2, 3
        assert_eq!(w.len(), 2);
        assert_eq!(w[1].open, 3.0);
    }

    #[test]
    fn clone_shares_the_same_buffer_zero_copy() {
        let a = series(8);
        let b = a.clone();
        // Same backing allocation: the clone is a refcount bump, not a copy.
        assert!(std::ptr::eq(a.as_slice().as_ptr(), b.as_slice().as_ptr()));
    }

    #[test]
    fn a_series_carries_its_clock_and_one_without_times_has_none() {
        let bars = series(3).as_slice().to_vec();
        let timed = Candles::on_clock(bars.clone(), Clock::new(vec![0, 10, 20]));
        assert_eq!(timed.clock().and_then(|clock| clock.time(2)), Some(20));
        assert!(Candles::new(bars).clock().is_none());
    }

    #[test]
    #[should_panic(expected = "one time per bar")]
    fn a_clock_of_another_length_is_refused() {
        Candles::on_clock(series(3).as_slice().to_vec(), Clock::new(vec![0, 10]));
    }
}
