//! The clock: each bar's open time in UTC nanoseconds, for the bars that exist and
//! no others. A market that closes on Friday opens its next bar on Monday, and the
//! rows run straight across the closure with nothing invented between them
//! (ADR 0004).

use std::sync::Arc;

/// A day in nanoseconds.
const DAY: i64 = 86_400 * 1_000_000_000;

/// The open times of a series' bars, strictly increasing, and the interval a bar
/// spans.
#[derive(Clone, Debug, PartialEq)]
pub struct Clock {
    times: Arc<[i64]>,
    interval: i64,
}

impl Clock {
    /// A clock over `times`, each a bar's open in UTC nanoseconds, whose bars span
    /// the median gap between them, the mean of the middle two when there is no
    /// single middle. Panics unless every time comes after the one before it; the
    /// Python boundary refuses such a series with the row it found (ADR 0007).
    pub fn new(times: impl Into<Arc<[i64]>>) -> Clock {
        let times = times.into();
        assert!(
            times.windows(2).all(|pair| pair[0] < pair[1]),
            "times must increase"
        );
        let mut gaps: Vec<i64> = times.windows(2).map(|pair| pair[1] - pair[0]).collect();
        gaps.sort_unstable();
        let interval = match gaps.len() {
            0 => 0,
            n if n % 2 == 1 => gaps[n / 2],
            n => (gaps[n / 2 - 1] + gaps[n / 2]) / 2,
        };
        Clock { times, interval }
    }

    /// The span of one bar in nanoseconds: a row ends at its open plus this, which
    /// is where the funding it owns is stamped. Zero on a clock of one row.
    pub fn interval(&self) -> i64 {
        self.interval
    }

    /// Every open time, in order.
    pub fn times(&self) -> &[i64] {
        &self.times
    }

    /// The number of rows, one per bar.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// True when the clock has no rows.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The open time of row `row`, or `None` past the end.
    pub fn time(&self, row: usize) -> Option<i64> {
        self.times.get(row).copied()
    }

    /// Bars a year: the rows less one, per 365 days of the span from the first open
    /// to the last. A market that never closes gets its interval's count and one
    /// that closes gets the bars it actually trades, a stock's daily bars about 252
    /// where a median gap reads 365. `None` with fewer than two rows, which have no
    /// span to count over (ADR 0004).
    pub fn periods_per_year(&self) -> Option<f64> {
        let rows = self.times.len();
        if rows < 2 {
            return None;
        }
        let span = self.times[rows - 1] - self.times[0];
        // multiplied out before the one division, so a regular series, whose bars
        // and days are both whole, lands on its interval's count exactly
        Some((rows - 1) as f64 * 365.0 * DAY as f64 / span as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{Clock, DAY};

    const HOUR: i64 = 3_600 * 1_000_000_000;

    fn regular(rows: usize, step: i64) -> Clock {
        Clock::new((0..rows as i64).map(|row| row * step).collect::<Vec<_>>())
    }

    #[test]
    fn a_year_of_hourly_bars_annualises_at_eight_thousand_seven_hundred_and_sixty() {
        // the Specification's example: 00:00 on 1 January to 00:00 on the next is
        // 8,761 rows over 365 days
        assert_eq!(regular(8_761, HOUR).periods_per_year(), Some(8_760.0));
    }

    #[test]
    fn a_stocks_daily_bars_annualise_at_the_days_it_trades() {
        // the Specification's other example: 253 daily bars from one 2 January to
        // the next, 365 days apart, are 252 a year, where a median gap of a day
        // reads 365
        let mut times = vec![0_i64];
        for day in 1..365 {
            if times.len() < 252 && day % 7 != 5 && day % 7 != 6 {
                times.push(day * DAY);
            }
        }
        times.push(365 * DAY);
        assert_eq!(times.len(), 253);
        assert_eq!(Clock::new(times).periods_per_year(), Some(252.0));
    }

    #[test]
    fn a_missing_day_counts_against_the_year_rather_than_being_assumed_away() {
        // the median gap called a feed missing a day still hourly; the clock counts
        // the bars that exist, so the year holds 24 fewer of them
        let times: Vec<i64> = (0..8_761_i64)
            .filter(|row| !(24..48).contains(row))
            .map(|row| row * HOUR)
            .collect();
        assert_eq!(Clock::new(times).periods_per_year(), Some(8_736.0));
    }

    #[test]
    fn one_row_has_no_span_to_annualise_over() {
        assert_eq!(regular(1, HOUR).periods_per_year(), None);
        assert_eq!(regular(2, HOUR).periods_per_year(), Some(8_760.0));
    }

    #[test]
    fn the_span_runs_from_the_first_open_wherever_the_clock_starts() {
        // two hourly rows opening on 1 January 2024 annualise as two from 0 do
        let start = 1_704_067_200 * 1_000_000_000;
        assert_eq!(
            Clock::new(vec![start, start + HOUR]).periods_per_year(),
            Some(8_760.0)
        );
    }

    #[test]
    fn a_clock_is_empty_only_without_rows() {
        assert!(Clock::new(Vec::<i64>::new()).is_empty());
        assert!(!regular(1, HOUR).is_empty());
    }

    #[test]
    #[should_panic(expected = "times must increase")]
    fn a_clock_that_runs_backwards_is_refused() {
        Clock::new(vec![0, 2 * HOUR, HOUR]);
    }

    #[test]
    #[should_panic(expected = "times must increase")]
    fn two_bars_at_one_time_are_refused() {
        Clock::new(vec![0, HOUR, HOUR]);
    }
}
