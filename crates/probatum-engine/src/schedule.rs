//! Funding events placed on the clock. A row owns the events stamped after its open
//! and up to and including its end, its open plus the interval, and charges them at
//! its close. An event stamped after one row's end and up to the next row's open,
//! across a closure or a hole in the data, is charged at the start of that next row.
//! A stamp within a second of a row boundary is taken as that boundary (ADR 0020).

use crate::clock::Clock;

/// A stamp within this many nanoseconds of a row boundary is on it.
const SNAP: i64 = 1_000_000_000;

/// Each row's funding events, placed once and shared by every env on the series.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Schedule {
    // per row, the rate of each event it owns and whether its stamp falls inside the
    // row rather than at its end
    owned: Vec<Vec<(f64, bool)>>,
    // per row, the rate of each event stamped between the row before and this one
    between: Vec<Vec<f64>>,
}

impl Schedule {
    /// Place `events`, each a stamp in UTC nanoseconds and a rate, on `clock`, in
    /// the order given. An event at or before the first open, or after the last
    /// row's end, falls outside the run and charges nothing.
    pub fn new(clock: &Clock, events: &[(i64, f64)]) -> Schedule {
        let times = clock.times();
        let interval = clock.interval();
        let mut owned = vec![Vec::new(); times.len()];
        let mut between = vec![Vec::new(); times.len()];
        let mut boundaries: Vec<i64> = times.iter().flat_map(|&t| [t, t + interval]).collect();
        boundaries.sort_unstable();
        boundaries.dedup();
        for &(stamp, rate) in events {
            let stamp = snapped(&boundaries, stamp);
            // the rows that open before the stamp, and of those the first that runs
            // to it: on a clock with a gap shorter than the interval two rows can
            // both reach a stamp, and the earlier one owns it
            let opened = times.partition_point(|&t| t < stamp);
            if opened == 0 {
                continue;
            }
            let first = times.partition_point(|&t| t < stamp - interval);
            if first < opened {
                owned[first].push((rate, stamp < times[first] + interval));
            } else if opened < times.len() {
                between[opened].push(rate);
            }
        }
        Schedule { owned, between }
    }

    /// The rate of each event row `row` owns, and whether it was stamped inside the
    /// row rather than at its end.
    pub fn owned(&self, row: usize) -> &[(f64, bool)] {
        self.owned.get(row).map_or(&[], Vec::as_slice)
    }

    /// The rate of each event stamped between the row before `row` and `row`.
    pub fn between(&self, row: usize) -> &[f64] {
        self.between.get(row).map_or(&[], Vec::as_slice)
    }
}

/// The events of a venue that charges `rate` every `hours` from 00:00 UTC, over
/// the span of `clock`: each multiple after its first open, up to the end of its
/// last row.
pub fn every(clock: &Clock, hours: f64, rate: f64) -> Vec<(i64, f64)> {
    let every = (hours * 3_600.0 * 1e9) as i64;
    let times = clock.times();
    if every <= 0 || times.is_empty() {
        return Vec::new();
    }
    let last = times[times.len() - 1] + clock.interval();
    let mut stamp = (times[0].div_euclid(every) + 1) * every;
    let mut out = Vec::new();
    while stamp <= last {
        out.push((stamp, rate));
        stamp += every;
    }
    out
}

/// `stamp`, moved onto the nearest boundary when one lies within a second of it,
/// the earlier of two at the same distance.
fn snapped(boundaries: &[i64], stamp: i64) -> i64 {
    let at = boundaries.partition_point(|&b| b < stamp);
    let below = at.checked_sub(1).map(|i| boundaries[i]);
    let above = boundaries.get(at).copied();
    let nearest = match (below, above) {
        (Some(b), Some(a)) if stamp - b <= a - stamp => b,
        (_, Some(a)) => a,
        (Some(b), None) => b,
        (None, None) => return stamp,
    };
    if (nearest - stamp).abs() <= SNAP {
        nearest
    } else {
        stamp
    }
}

#[cfg(test)]
mod tests {
    use super::{every, Schedule};
    use crate::clock::Clock;

    const HOUR: i64 = 3_600 * 1_000_000_000;

    fn hourly(rows: i64) -> Clock {
        Clock::new((0..rows).map(|row| row * HOUR).collect::<Vec<_>>())
    }

    #[test]
    fn an_event_at_a_rows_end_belongs_to_that_row() {
        // the Specification's example of the sources: on hourly bars from 00:00,
        // events every 8 hours land at 08:00, 16:00 and the next 00:00, each owned by
        // the bar that ends there, the ones opening at 07:00, 15:00 and 23:00
        let clock = hourly(25);
        let events = every(&clock, 8.0, 0.0001);
        assert_eq!(
            events
                .iter()
                .map(|&(stamp, _)| stamp / HOUR)
                .collect::<Vec<_>>(),
            vec![8, 16, 24]
        );
        let schedule = Schedule::new(&clock, &events);
        for row in [7, 15, 23] {
            assert_eq!(schedule.owned(row), &[(0.0001, false)], "row {row}");
        }
        assert_eq!(
            (0..25).map(|row| schedule.owned(row).len()).sum::<usize>(),
            3
        );
    }

    #[test]
    fn a_stamp_within_a_second_of_a_boundary_is_on_it() {
        // the Specification's example: 08:00:00.003 is taken as 08:00, and so
        // belongs to the bar that opened at 07:00 rather than the one at 08:00
        let schedule = Schedule::new(&hourly(10), &[(8 * HOUR + 3_000_000, 0.0001)]);
        assert_eq!(schedule.owned(7), &[(0.0001, false)]);
        assert!(schedule.owned(8).is_empty());
    }

    #[test]
    fn an_event_inside_a_bar_is_owned_by_it_and_marked_inside() {
        // daily bars against 8-hour funding: 08:00 and 16:00 fall inside the day,
        // and the next 00:00 is its end
        let day = 24 * HOUR;
        let clock = Clock::new(vec![0, day, 2 * day]);
        let schedule = Schedule::new(&clock, &every(&clock, 8.0, 0.0001));
        assert_eq!(
            schedule.owned(0),
            &[(0.0001, true), (0.0001, true), (0.0001, false)]
        );
    }

    #[test]
    fn an_event_across_a_closure_is_charged_at_the_next_rows_start() {
        // daily bars stop at Friday's close and resume on Monday, and a Saturday
        // event belongs to no bar, so it is charged at the start of Monday's
        let day = 24 * HOUR;
        let clock = Clock::new(vec![0, day, 2 * day, 3 * day, 4 * day, 7 * day]);
        let schedule = Schedule::new(&clock, &[(5 * day + 8 * HOUR, 0.0005)]);
        assert_eq!(schedule.between(5), &[0.0005]);
        assert!((0..6).all(|row| schedule.owned(row).is_empty()));
    }

    #[test]
    fn an_event_outside_the_run_charges_nothing() {
        let schedule = Schedule::new(&hourly(3), &[(0, 0.1), (3 * HOUR + 2_000_000_000, 0.1)]);
        assert!((0..3).all(|row| schedule.owned(row).is_empty()));
        assert!((0..3).all(|row| schedule.between(row).is_empty()));
    }

    #[test]
    fn an_irregular_clock_gives_a_stamp_two_rows_reach_to_the_earlier() {
        // the bars are an hour long and one opens half an hour after the last, so a
        // stamp a quarter past lies in both spans, and the first owns it
        let clock = Clock::new(vec![0, HOUR, 3 * HOUR / 2, 3 * HOUR, 4 * HOUR]);
        assert_eq!(clock.interval(), HOUR);
        let schedule = Schedule::new(&clock, &[(7 * HOUR / 4, 0.1)]);
        assert_eq!(schedule.owned(1), &[(0.1, true)]);
        assert!(schedule.owned(2).is_empty());
    }
}
