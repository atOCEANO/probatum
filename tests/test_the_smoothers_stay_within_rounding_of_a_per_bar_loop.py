"""The recursive smoothers are one vectorized pass, taken in blocks short enough that no
weight overflows, and stay within rounding of the recurrence taken one bar at a time:
across every block's seam, after a gap, over a hundred thousand bars, and at a short
length and a long one over the same series.
"""

import math

import numpy as np
import pytest

from probatum import ta


def block_size(alpha):
    # where the seams fall, so a test crosses the block hand-off on purpose rather
    # than by being long enough by luck. Every test below is worthless the moment
    # its series stops crossing one. Read from the module rather than recomputed
    # here: a copy of the formula beside the tests agrees with itself, so replacing
    # the module's with a picked constant used to leave all three of these green
    return ta._block(1.0 - alpha)


def rolled_ema(values, length):
    # the recurrence written the slow, obvious way, one bar at a time: the mean of
    # the first clean window seeds it, one multiply-add carries it, and a gap ends
    # the run so the next clean window seeds a fresh one. Written out here rather
    # than taken from the module, since a reference computed by the code under test
    # agrees with it however wrong both are (ADR 0043)
    alpha   = 2.0 / (length + 1.0)
    keep    = 1.0 - alpha
    plain   = values.tolist()
    out     = [float("nan")] * len(plain)
    running = None
    clean   = 0

    for i, value in enumerate(plain):
        if not math.isfinite(value):
            running, clean = None, 0
            continue
        clean += 1
        if running is None:
            if clean == length:
                # the same numpy call the module makes, so the seed is bit for bit
                running = float(values[i - length + 1:i + 1].mean())
                out[i] = running
        else:
            running = alpha * value + keep * running
            out[i] = running

    return np.array(out)


def test_the_exponential_pass_matches_the_plain_recurrence_across_many_blocks():
    # the hand-off between `_decay`'s blocks, `carried`, is what this holds: drop
    # it, seed every block from zero, or from the original seed, and every bar after
    # the first seam is wrong, worst at the seam and fading over the next hundred
    # bars. The block is 851 bars for `ema(5)`, so the series crosses four seams
    length = 5
    alpha  = 2.0 / (length + 1.0)
    keep   = 1.0 - alpha
    block  = block_size(alpha)

    assert block == 851, "the seam moved; this series may no longer cross one"

    size     = 4 * block + length + 7
    step     = np.arange(float(size))
    close    = 100.0 + 10.0 * np.sin(step / 17.0) + 0.5 * np.cos(step / 2.0)
    out      = ta.ema(close, length)
    expected = rolled_ema(close, length)

    # ADR 0037 holds the smoothers within rounding of a per-bar loop rather than to
    # its bits, so the tolerance sits far above what a reassociation moves and far
    # below anything a mishandled carry produces
    assert np.isfinite(out[length - 1:]).all()
    assert np.allclose(out[length - 1:], expected[length - 1:],
                       rtol=1e-12, atol=0.0), "the blocked pass and the loop parted"

    # and the seams one at a time, where a lost carry shows before it decays away.
    # out[seam - 1] is the module's own, so this asks only that the one step across
    # the block boundary is the step the recurrence says it is
    seams = [length + m * block for m in range(1, 5)]
    assert seams[-1] < size
    for seam in seams:
        assert out[seam] == pytest.approx(
            alpha * close[seam] + keep * out[seam - 1], rel=1e-12
        ), f"the carry into the block starting at bar {seam} was dropped"


def test_a_gap_between_two_multi_block_stretches_reseeds_and_carries_correctly():
    # the blocking is per clean stretch, so the second stretch counts its blocks
    # from the bar the gap ended on and its seams land at a different phase from the
    # first stretch's. No test has ever run a stretch past one block, let alone two
    # stretches at two phases, so a `_decay` blocking the whole array rather than
    # each stretch, or carrying the value from before the gap across it, passes
    # everything else in this file
    length  = 5
    alpha   = 2.0 / (length + 1.0)
    keep    = 1.0 - alpha
    block   = block_size(alpha)
    head    = 2 * block + length + 20
    restart = head + 3
    size    = restart + 2 * block + length + 40
    step    = np.arange(float(size))
    close   = 100.0 + 10.0 * np.sin(step / 23.0) + 0.5 * np.cos(step / 3.0)
    close[head:restart] = np.nan

    out      = ta.ema(close, length)
    expected = rolled_ema(close, length)

    assert np.allclose(out, expected, rtol=1e-12, atol=0.0, equal_nan=True)
    # the hole, the warm-up behind it, and the fresh seed from the next clean window
    assert np.isfinite(out[head - 1])
    assert np.isnan(out[head:restart + length - 1]).all()
    assert out[restart + length - 1] == pytest.approx(
        float(close[restart:restart + length].mean())
    )
    assert np.isfinite(out[-1]), "one hole killed the rest of the series"

    # both stretches cross two seams, and the second stretch's sit at a phase the
    # first's do not, so blocking from bar zero would put them somewhere else
    first  = [length + block, length + 2 * block]
    second = [restart + length + block, restart + length + 2 * block]

    assert first[-1] < head and second[-1] < size
    assert (second[0] - first[0]) % block != 0
    for seam in first + second:
        assert out[seam] == pytest.approx(
            alpha * close[seam] + keep * out[seam - 1], rel=1e-12
        ), f"the carry into the block starting at bar {seam} was dropped"


def test_a_short_length_over_a_hundred_thousand_bars_stays_finite_throughout():
    # the whole reason `_decay` is not a one-liner. Written as a single cumulative
    # sum the recurrence needs (1 - alpha)**-k, which passes 1e308 within a few
    # thousand bars and takes the rest of the output with it: `_decay` records
    # length 20 over 100k bars returning 7,045 finite values out of 99,981, in
    # silence
    length = 20
    size   = 100000
    alpha  = 2.0 / (length + 1.0)
    block  = block_size(alpha)

    assert size > 25 * block, "tens of seams is the point; one is not"

    step  = np.arange(float(size))
    close = 100.0 + 10.0 * np.sin(step / 500.0) + 0.5 * np.cos(step / 7.0)
    out   = ta.ema(close, length)

    assert np.isnan(out[:length - 1]).all()
    assert np.isfinite(out[length - 1:]).all(), "the weights overflowed"
    # size less the warm-up, and the number `_decay` quotes against the one-liner's
    assert int(np.isfinite(out).sum()) == 99981
    # an exponential average of a bounded series is a convex combination of it, its
    # weights summing to one, so a weight multiplied back wrong leaves the series
    # rather than landing somewhere plausible inside it
    assert out[length - 1:].min() >= close.min() - 1e-9
    assert out[length - 1:].max() <= close.max() + 1e-9

    # and the numbers themselves, against the recurrence taken one bar at a time
    expected = rolled_ema(close, length)
    assert np.allclose(out[length - 1:], expected[length - 1:],
                       rtol=1e-12, atol=0.0)
    assert out[-1] == pytest.approx(expected[-1], rel=1e-12)


def test_no_single_block_size_suits_both_a_short_length_and_a_long_one():
    # the block has to be DERIVED, and one length cannot show that. ema(20) keeps
    # 0.905 a bar and affords 3,450; ema(2) keeps a third and affords 314, because
    # a third to the minus thousandth is 1e477. So a block of a thousand bars, which
    # suits the test above, overflows here inside its first block, silently: the
    # output goes infinite and then NaN while the shape, the dtype and the warm-up
    # are all still right. Both lengths run over the same series, so no constant
    # can pass this
    size  = 100000
    step  = np.arange(float(size))
    close = 100.0 + 10.0 * np.sin(step / 500.0) + 0.5 * np.cos(step / 7.0)

    short, long_ = 2, 20
    lo, hi = block_size(2.0 / (short + 1.0)), block_size(2.0 / (long_ + 1.0))
    assert lo < 1000 < hi, "the two lengths no longer straddle a picked block"

    for length in (short, long_):
        out = ta.ema(close, length)
        assert np.isnan(out[:length - 1]).all()
        assert np.isfinite(out[length - 1:]).all(), f"ema({length}) overflowed"
        assert out[length - 1:].min() >= close.min() - 1e-9
        assert out[length - 1:].max() <= close.max() + 1e-9

    assert np.allclose(ta.ema(close, short)[short - 1:],
                       rolled_ema(close, short)[short - 1:], rtol=1e-12, atol=0.0)
