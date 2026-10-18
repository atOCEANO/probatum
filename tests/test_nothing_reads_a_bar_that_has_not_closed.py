"""Nothing reads a bar that has not closed. What is known at the end of bar t is the
same whether the bars after t are there or not, and whatever they hold: the coarse bars
`resample` builds from these bars, the values `asof` puts on their clock, and every
line of every indicator in `probatum.ta`.
"""

import numpy as np
import pytest

import probatum
from test_every_indicator_gives_one_value_a_bar import LINES

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


def test_no_indicator_can_see_the_future():
    """The invariant everything else rests on.

    These are computed once over the whole series in ``init`` and then read bar by
    bar inside the loop, so a value at bar i that depended on bar i+1 would be
    silent lookahead in every strategy that used it, and would flatter every one
    of them. Truncating the series must therefore change nothing about the bars
    that remain.
    """
    rng    = np.random.default_rng(4)
    close  = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 90))
    high   = close + rng.uniform(0.1, 1.0, 90)
    low    = close - rng.uniform(0.1, 1.0, 90)
    volume = rng.uniform(500.0, 1500.0, 90)

    for name, call in LINES.items():
        whole = call(high, low, close, volume)
        for cut in (20, 45, 89, 90):
            part = call(high[:cut], low[:cut], close[:cut], volume[:cut])
            for line, (full, upto) in enumerate(zip(whole, part)):
                assert len(upto) == cut, f"{name} line {line} at cut {cut}"
                for i in range(cut):
                    a, b = full[i], upto[i]
                    same = (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b)
                    assert same, (
                        f"{name} line {line} bar {i} moved from {a} to {b} when "
                        f"the series was cut at {cut}: it can see the future"
                    )


def test_removing_the_future_outright_changes_no_past_value():
    # the same invariant from the other side: rewrite the tail and the head must
    # not move by a single ulp
    rng   = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 90))
    high, low, volume = close + 1.0, close - 1.0, np.full(90, 1000.0)

    moved = close.copy()
    moved[60:] += 500.0

    for name, call in LINES.items():
        before = call(high, low, close, volume)
        after  = call(high, low, moved, volume)
        for line, (a, b) in enumerate(zip(before, after)):
            assert np.allclose(a[:60], b[:60], equal_nan=True), (
                f"{name} line {line} changed a bar before 60 when only bars after "
                f"60 were moved"
            )
