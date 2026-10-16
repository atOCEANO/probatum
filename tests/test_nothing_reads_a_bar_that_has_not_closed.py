"""Nothing reads a bar that has not closed. What is known at the end of bar t is the
same whether the bars after t are there or not, and whatever they hold: the coarse bars
`resample` builds from these bars, and the values `asof` puts on their clock.
"""

import numpy as np
import pytest

import probatum

HOUR  = 3_600_000_000_000
START = 1_704_067_200_000_000_000
STEPS = {"2h": 2 * HOUR, "4h": 4 * HOUR, "1d": 24 * HOUR}


def bars(draw, rows):
    closes = 100.0 * np.exp(np.cumsum(draw.normal(0.0, 0.01, rows)))
    opens  = closes * np.exp(draw.normal(0.0, 0.005, rows))
    highs  = np.maximum(opens, closes) * (1.0 + 0.01 * draw.random(rows))
    lows   = np.minimum(opens, closes) * (1.0 - 0.01 * draw.random(rows))

    return np.column_stack([opens, highs, lows, closes, 100.0 * draw.random(rows)])


def universe(seed, n=200, width=3):
    """Hourly bars of ``width`` instruments over ``n`` hours, each listing late,
    missing a tenth of its bars and delisting early, at random."""
    draw = np.random.default_rng(seed)
    out  = np.full((n, width, 5), np.nan)

    for k in range(width):
        rows = np.arange(draw.integers(0, n // 4), draw.integers(3 * n // 4, n + 1))
        rows = rows[draw.random(len(rows)) > 0.1]
        out[rows, k] = bars(draw, len(rows))

    return out, START + HOUR * np.arange(n, dtype=np.int64)


def data(out, times):
    return probatum.Data(out, time=times, symbol=["A", "B", "C"], interval="1h")


def cuts(seed, data):
    return np.random.default_rng(seed + 1_000).integers(10, len(data) - 1, 6)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("interval", list(STEPS))
def test_a_coarse_bar_is_the_same_whether_the_bars_after_it_are_there_or_not(seed,
                                                                             interval):
    whole  = data(*universe(seed))
    coarse = whole.resample(interval)

    for t in cuts(seed, whole):
        end    = whole.time[t] + HOUR
        closed = coarse.time + STEPS[interval] <= end
        try:
            cut = whole[:t + 1].resample(interval)
        except ValueError:
            assert not closed.any()
            continue

        assert cut.time.tolist() == coarse.time[closed].tolist()
        np.testing.assert_array_equal(cut.bars, coarse.bars[closed])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("interval", list(STEPS))
def test_a_slice_or_a_selection_reads_nothing_after_its_own_end(seed, interval):
    whole = data(*universe(seed))
    start = int(np.random.default_rng(seed + 4_000).integers(5, 60))

    for view in (whole[start:], whole.select(["C", "A"])):
        coarse = view.resample(interval)
        for t in cuts(seed, view):
            end    = view.time[t] + HOUR
            closed = coarse.time + STEPS[interval] <= end
            try:
                cut = view[:t + 1].resample(interval)
            except ValueError:
                assert not closed.any()
                continue

            assert cut.time.tolist() == coarse.time[closed].tolist()
            np.testing.assert_array_equal(cut.bars, coarse.bars[closed])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("interval", list(STEPS))
def test_a_coarse_bar_is_the_same_whatever_the_bars_after_it_hold(seed, interval):
    out, times = universe(seed)
    whole  = data(out, times)
    coarse = whole.resample(interval)
    draw   = np.random.default_rng(seed + 2_000)

    for t in cuts(seed, whole):
        later = times > whole.time[t]
        moved = out.copy()
        moved[later] = np.nan
        for k in range(moved.shape[1]):
            rows = np.flatnonzero(later & (draw.random(len(times)) > 0.3))
            moved[rows, k] = bars(draw, len(rows))

        changed = data(moved, times).resample(interval)
        end     = whole.time[t] + HOUR
        mine    = coarse.time + STEPS[interval] <= end
        theirs  = changed.time + STEPS[interval] <= end

        assert coarse.time[mine].tolist() == changed.time[theirs].tolist()
        np.testing.assert_array_equal(coarse.bars[mine], changed.bars[theirs])


@pytest.mark.parametrize("seed", range(8))
def test_a_value_put_on_the_clock_is_the_same_whatever_comes_after_it(seed):
    whole   = data(*universe(seed))
    coarse  = whole.resample("4h")
    draw    = np.random.default_rng(seed + 3_000)
    values  = draw.random((len(coarse), 3))
    known   = np.sort(draw.integers(whole.time[0], whole.time[-1] + HOUR, 40))
    stamped = draw.random(40)

    on, at = whole.asof(values, coarse), whole.asof(stamped, known)

    for t in cuts(seed, whole):
        end   = whole.time[t] + HOUR
        moved = values.copy()
        moved[coarse.time + 4 * HOUR > end] = draw.random(3)

        np.testing.assert_array_equal(whole.asof(moved, coarse)[:t + 1], on[:t + 1])
        np.testing.assert_array_equal(whole[:t + 1].asof(values, coarse), on[:t + 1])

        shifted = stamped.copy()
        shifted[known > end] = -1.0

        np.testing.assert_array_equal(whole.asof(shifted, known)[:t + 1], at[:t + 1])
