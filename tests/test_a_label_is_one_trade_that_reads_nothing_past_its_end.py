"""A label is one trade of a fresh account, recorded as `Labels`: refused where its
horizon, notional, side, barriers or venue cannot trade, read-only and kept with what
it was made with. It reads nothing past its end: it is the same on data cut once its
horizon has closed, and on data whose bars after its end move, while moving the bar
that closed it changes it. A selection labels each instrument as the whole universe
does.
"""

import math
import pickle
import random

import numpy as np
import pytest

import probatum
from probatum import (ProbatumAttributeError, ProbatumTypeError, ProbatumValueError,
                      VenueError)
from reference import differential

START = 1_704_067_200 * 10**9   # 2024-01-01 00:00 UTC
VENUE = probatum.Venue(
    fees  = {"maker": 0.0002, "taker": 0.0006},
    fills = {"slippage": {"fixed": 5.0}, "spread": "estimated"},
)


def walked(count=60, width=2, seed=3):
    """``width`` linear perpetuals of the differential's walks on an hourly clock, the
    second printing from row 5 with a gap of three rows at row 30, as their bars."""
    rng  = random.Random(seed)
    bars = np.full((count, width, 5), np.nan)
    for k in range(width):
        bars[:, k] = np.array(differential.walk(rng, count), dtype=np.float64)

    if width > 1:
        bars[:5, 1] = bars[30:33, 1] = np.nan

    return bars


def data_of(bars):
    names = [f"C{k}USDT" for k in range(bars.shape[1])]
    times = START + np.arange(len(bars)) * differential.HOUR
    return probatum.Data(bars, symbol=names, market="linear", time=times, interval="1h")


def labelled(bars, **given):
    return probatum.labels(data_of(bars), VENUE, given.pop("horizon", 4), 10_000.0,
                           **{"stop": 0.03, "take_profit": 0.05, **given})


def same(mine, theirs, t, k):
    return (mine.exit[t, k], mine.end[t, k], mine.bars[t, k]) == (
        theirs.exit[t, k], theirs.end[t, k], theirs.bars[t, k]) and (
        mine.ret[t, k] == theirs.ret[t, k] or (math.isnan(mine.ret[t, k])
                                              and math.isnan(theirs.ret[t, k])))


@pytest.mark.parametrize("given, error, message", [
    (dict(horizon=0), ProbatumValueError, "horizon: 0; give a whole number of bars from 1"),
    (dict(horizon=2.0), ProbatumTypeError,
     "horizon: 2.0; give a whole number of bars from 1"),
    (dict(horizon=True), ProbatumTypeError,
     "horizon: True; give a whole number of bars from 1"),
    (dict(notional=0.0), ProbatumValueError,
     "notional: 0.0; give a positive amount, in the currency the account settles in"),
    (dict(notional=math.nan), ProbatumValueError, "notional: nan; give a positive amount"),
    (dict(notional="10000"), ProbatumTypeError,
     "notional: '10000'; give the amount each label trades, a positive number"),
    (dict(side="flat"), ProbatumValueError, "side: 'flat'; use \"long\" or \"short\""),
    (dict(stop=0.0), ProbatumValueError, "stop: 0.0; give a return from the entry above 0"),
    (dict(take_profit=np.full((3, 2), 0.01)), ProbatumValueError,
     "take_profit: an array of shape (3, 2) for a Data of 60 rows and 2 instruments"),
    (dict(venue=VENUE.replace(intrabar="random")), VenueError,
     "venue: intrabar is \"random\", which labels refuses so that a label repeats"),
])
def test_labels_refuse_what_they_cannot_trade(given, error, message):
    data = data_of(walked())
    args = dict(venue=VENUE, horizon=4, notional=10_000.0)
    args.update(given)

    with pytest.raises(error) as raised:
        probatum.labels(data, args.pop("venue"), args.pop("horizon"), args.pop("notional"),
                        **args)

    assert str(raised.value).startswith(message)


def test_a_short_label_is_refused_on_spot():
    bars  = walked(width=1)
    times = START + np.arange(len(bars)) * differential.HOUR
    spot  = probatum.Data(
        bars,
        symbol   = ["C0USDT"],
        market   = "spot",
        time     = times,
        interval = "1h",
    )

    with pytest.raises(ProbatumValueError, match="side: \"short\", and C0USDT is spot"):
        probatum.labels(spot, VENUE, 4, 10_000.0, side="short")


def test_labels_are_read_only_and_keep_what_they_were_made_with():
    data   = data_of(walked())
    stop   = np.full((len(data), 2), 0.03)
    made   = probatum.labels(data, VENUE, 4, 10_000.0, stop=stop, side="short")
    fields = (made.ret, made.exit, made.bars, made.end, made.weight, made.stop)

    assert all(not field.flags.writeable for field in fields)
    assert (made.horizon, made.notional, made.side, made.take_profit, made.venue,
            made.names, made.fingerprint) == (4, 10_000.0, "short", None, VENUE,
                                              data.names, data.fingerprint)
    assert (made.ret.shape, made.exit.dtype.kind, made.end.dtype) == (
        (60, 2), "U", np.dtype(np.int64))

    with pytest.raises(ProbatumAttributeError, match="Labels.ret: labels do not change"):
        made.ret = made.ret

    back = pickle.loads(pickle.dumps(made))
    assert not back.ret.flags.writeable and np.array_equal(back.end, made.end)
    assert repr(made) == "Labels(2 instruments over 60 bars, short 10,000.0 at a horizon of 4)"


def test_a_label_is_the_same_on_data_cut_once_its_horizon_has_closed():
    bars, horizon = walked(), 4
    whole = labelled(bars, horizon=horizon)
    cuts  = {}
    for t, k in zip(*np.nonzero(np.isin(whole.exit, ["stop", "take_profit", "horizon"]))):
        cuts.setdefault(max(int(whole.end[t, k]), t + horizon + 1), []).append((t, k))

    assert len(cuts) > 20

    for cut, held in cuts.items():
        part = labelled(bars[:cut + 1], horizon=horizon)
        assert all(same(part, whole, t, k) for t, k in held), cut


def test_a_label_is_the_same_when_the_bars_after_its_end_move_and_moves_with_its_last():
    bars  = walked()
    whole = labelled(bars)
    rng   = random.Random(11)
    held  = list(zip(*np.nonzero(np.isin(whole.exit, ["stop", "take_profit", "horizon"]))))

    for t, k in rng.sample(held, 12):
        end   = int(whole.end[t, k])
        later = bars.copy()
        later[end + 1:, k, :4] *= 1.1
        assert same(labelled(later), whole, t, k)

        # the bar that closed it: a horizon's close fills at its open, which moves,
        # and a barrier that filled there is not reached on a bar at its entry
        moved = bars.copy()
        if whole.exit[t, k] == "horizon":
            moved[end, k, :4] *= 1.02
        else:
            moved[end, k, :4] = bars[end - whole.bars[t, k], k, 0]

        assert not same(labelled(moved), whole, t, k)


def test_a_barrier_of_rows_holds_a_nan_as_none_where_one_label_runs():
    # ten bars at a horizon of 8 run the label at row 0 alone, its barrier an array of
    # one value
    data = data_of(walked(count=10, width=1))
    held = probatum.labels(data, VENUE, 8, 10_000.0, stop=np.full((10, 1), np.nan))

    assert same(held, probatum.labels(data, VENUE, 8, 10_000.0), 0, 0)


def test_a_slice_labels_its_rows_as_the_whole_data_does():
    # the slice's engine starts on the bars before it, which its fills' windows read
    data  = data_of(walked())
    whole = probatum.labels(data, VENUE, 4, 10_000.0, stop=0.03)
    part  = probatum.labels(data[20:], VENUE, 4, 10_000.0, stop=0.03)

    for field in ("exit", "bars"):
        assert np.array_equal(getattr(part, field), getattr(whole, field)[20:])

    np.testing.assert_array_equal(part.ret, whole.ret[20:])

    least = np.iinfo(np.int64).min
    later = whole.end[20:]
    assert np.array_equal(part.end, np.where(later == least, least, later - 20))


def test_a_selection_labels_its_instrument_as_the_whole_universe_does():
    data  = data_of(walked())
    whole = probatum.labels(data, VENUE, 4, 10_000.0, stop=0.03)
    alone = probatum.labels(data.select(["C1USDT"]), VENUE, 4, 10_000.0, stop=0.03)

    for field in ("exit", "end", "bars"):
        assert np.array_equal(getattr(alone, field)[:, 0], getattr(whole, field)[:, 1])

    for field in ("ret", "weight"):
        assert np.array_equal(getattr(alone, field)[:, 0], getattr(whole, field)[:, 1],
                              equal_nan=True)

    assert (alone.exit[:5, 0] == "").all() and (alone.exit[30:33, 0] == "").all()
