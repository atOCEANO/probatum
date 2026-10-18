"""A search checks its policy before it trusts a trial: the first trial that decided
something is run twice on the in-sample bars and on the bars up to each cut, placed
among the rows it decides on, the targets taking each instrument in turn, moved to
changes that alternate and cut with the row before, each keeping the rows its last close
plays, and a policy whose decisions do not repeat, or change on a cut, is refused, a
reader's or a verb's refusal on a cut among them, while an error of its own stays its
own; the winner is checked too. Floats are held to bounds that follow their precision,
whatever the policy returns them in, and the check reads the in-sample bars alone.
"""

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import probatum
from probatum import (Categorical, DataError, Float, Int, LookaheadError,
                      ProbatumValueError, Strategy)
from probatum._causality import _cuts, _Weights
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (VENUE, bars,
                                                                        indexed, one,
                                                                        pair)
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross

AHEAD = ("policy: its decision for BTCUSDT at (.+) is (.+) on the in-sample bars and "
         "(.+) on the bars up to \\1; decide from the bars up to each close")


def searched(policy, space, data=None, **given):
    return probatum.search(policy, space, one() if data is None else data, VENUE, **{
        **dict(sampler="grid", holdout=0.25), **given})


def rising(rows=40):
    """Hourly bars that rise with a wiggle, every close after the first above it."""
    t     = np.arange(rows, dtype=np.float64)
    close = 100.0 + 0.5 * t + 3.0 * np.sin(t / 2.0)
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")

    return probatum.Data(
        pd.DataFrame({"open": close, "high": close * 1.01,
                      "low": close * 0.99, "close": close,
                      "volume": 1e6}, index=index),
        symbol = "BTCUSDT",
        market = "linear",
    )


def rolled(data, k):
    # the Specification's worked example: each close decided from the next
    close = data.close
    return np.sign(np.roll(close, -1) - close)


def test_a_policy_that_decides_from_the_next_close_is_refused_on_a_cut():
    # rolled round, a cut's last row reads its first close, which is below every
    # other, so a cut at a rise says the bars fall
    with pytest.raises(LookaheadError) as caught:
        searched(rolled, {"k": Int(1, 1)}, rising())

    said = re.fullmatch(AHEAD, str(caught.value))
    assert said is not None, str(caught.value)

    data  = rising().split(30)[0]
    c     = int(np.flatnonzero(data.time == pd.Timestamp(said[1], tz="UTC").value)[0])
    close = data.close

    assert float(said[2]) == np.sign(close[c + 1] - close[c])
    assert float(said[3]) == np.sign(close[0] - close[c]) != float(said[2])


def crossing(data, fast, slow):
    up, down = probatum.ta.ema(data.close, fast), probatum.ta.ema(data.close, slow)
    return np.where(np.isnan(down), np.nan, (up > down).astype(np.float64))


def strategic(data, fast, slow):
    return Cross(fast, slow)


def both(data, size):
    return np.full((len(data), 2), size)


def framed(data, fast, slow):
    return indexed(data, crossing(data, fast, slow)[:, None], ("BTCUSDT",))


def serial(data, fast, slow):
    return framed(data, fast, slow)["BTCUSDT"]


@pytest.mark.parametrize("policy, space, data", [
    (Cross, {"fast": Int(2, 4), "slow": Int(8, 10)}, one()),
    (crossing, {"fast": Int(2, 4), "slow": Int(8, 10)}, one()),
    (strategic, {"fast": Int(2, 4), "slow": Int(8, 10)}, one()),
    (both, {"size": Float(-1.0, 1.0, step=1.0)}, pair()),
    (framed, {"fast": Int(2, 4), "slow": Int(8, 10)}, one()),
    (serial, {"fast": Int(2, 4), "slow": Int(8, 10)}, one()),
])
def test_a_causal_policy_passes_and_spends_no_look(policy, space, data, caplog):
    with caplog.at_level("INFO", logger="probatum"):
        study = probatum.search(policy, space, data, VENUE, sampler="grid", holdout=0.25)

    assert study.looks == 0 and len(study.trials)
    assert "search: trial 0's decisions are causal" in caplog.messages


class Leading(Strategy):
    """Cross reading each line one bar ahead, whose pad on a cut's last row is the exit
    a dense two-state policy takes, so only a cut that ends on an entry shows it."""


    def __init__(self, fast=3, slow=8):
        self.fast, self.slow = fast, slow


    def init(self, data):
        self.warmup = self.slow
        self.up     = np.roll(probatum.ta.ema(data.close, self.fast), -1)
        self.down   = np.roll(probatum.ta.ema(data.close, self.slow), -1)


    def next(self, bar, account):
        account.target(1.0 if self.up[bar.i] > self.down[bar.i] else 0.0)


def test_a_lead_whose_pad_is_the_exit_is_refused_at_a_cut_on_an_entry():
    with pytest.raises(LookaheadError, match=r"^policy: at .+ it calls target\(1\.0\) on "
                                             r"the in-sample bars and target\(0\.0\) on "
                                             r"the bars up to .+; decide from the bars "
                                             r"up to each close$"):
        searched(Leading, {"fast": Int(3, 3), "slow": Int(8, 8)})


def pivots(data, left):
    """The Specification's second worked example: weight 0 on a close no lower than the
    ``left`` before it and the next one, 1 elsewhere, the next close read only to
    confirm a top."""
    close, out = data.close, np.full(len(data), np.nan)
    for t in range(left, len(close)):
        out[t] = 0.0 if close[t] >= close[t - left:t + 2].max() else 1.0

    return out


def test_a_lead_that_shows_only_where_a_decision_holds_is_cut_on_the_row_before():
    # every change is a top or the bar after it, where a cut agrees with the whole run,
    # and the rising close before a top is a five-bar high the next close passes
    with pytest.raises(LookaheadError, match=r"^policy: its decision for BTCUSDT at .+ "
                                             r"is 1\.0 on the in-sample bars and 0\.0 on "
                                             r"the bars up to "):
        searched(pivots, {"left": Int(4, 4)}, rising(400))


def sparse(data, k):
    """BTCUSDT long and flat by turns of three bars, and ETHUSDT half long on the bar
    before its one jump, which it reads from the next close."""
    busy = np.where((np.arange(len(data)) // 3) % 2 == 0, 1.0, 0.0)
    jump = np.where(np.diff(data.closes[:, 1], append=np.nan) > 0.5, 0.5, 0.0)
    return np.stack([busy, jump], axis=1)


def test_a_lead_on_one_instrument_beside_a_busier_one_meets_a_cut_of_its_own():
    eth = bars(level=10.0)
    eth[["open", "high", "low", "close"]] = 10.0
    eth.loc[eth.index[11]:, ["open", "high", "low", "close"]] = 11.0

    data = probatum.Data({"BTCUSDT": bars(), "ETHUSDT": eth}, market="linear")

    with pytest.raises(LookaheadError, match=r"^policy: its decision for ETHUSDT at "
                                             r"2024-01-01 10:00 is 0\.5 on the in-sample "
                                             r"bars and 0\.0 on the bars up to "
                                             r"2024-01-01 10:00;"):
        searched(sparse, {"k": Int(1, 1)}, data)


def interpolated(data, k):
    close = pd.Series(data.close).interpolate().to_numpy()
    return np.where(np.diff(close, prepend=np.nan) > 0.0, 1.0, 0.0)


def test_a_value_on_a_row_nothing_printed_is_cut_with_the_close_that_plays_it():
    # ETHUSDT's close rises to row 19, falls at 20, prints nothing at 21, where only
    # BTCUSDT does, and rises at 22, so row 21's interpolated close reads row 22's, and
    # the account plays it at 20's close
    t     = np.arange(40, dtype=np.float64)
    close = np.where(t < 20, 100.0 + t, np.where(t < 22, 110.0, 130.0 - t))
    close[22] = 125.0

    frame = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                          "close": close, "volume": 1e6},
                         index=pd.date_range("2024-01-01", periods=40, freq="h",
                                             tz="UTC"))
    universe = probatum.Data({"BTCUSDT": bars(), "ETHUSDT": frame.drop(frame.index[21])},
                             market="linear")

    with pytest.raises(LookaheadError, match=r"^policy: its decision for ETHUSDT at "
                                             r"2024-01-01 21:00 is 1\.0 on the in-sample "
                                             r"bars and 0\.0 on the bars up to "
                                             r"2024-01-01 20:00;"):
        searched(interpolated, {"k": Int(1, 1)}, universe["ETHUSDT"])


class Shifted(Strategy):
    """The next bar's move as its weight, NaN on the last row, which a whole run never
    decides on and a cut's run does."""


    def __init__(self, size=1.0):
        self.size = size


    def init(self, data):
        self.warmup = 2
        self.moves  = np.sign(np.diff(data.close, append=np.nan)) * self.size


    def next(self, bar, account):
        account.target(max(float(self.moves[bar.i]), 0.0)
                       if not np.isnan(self.moves[bar.i]) else float("nan"))


def test_a_verb_s_refusal_on_a_cut_is_a_lookahead_chained_from_it():
    with pytest.raises(LookaheadError) as caught:
        searched(Shifted, {"size": Categorical([1.0])})

    assert re.fullmatch(r"policy: on the bars up to .+ its decisions are refused "
                        r"\(weights: nan for BTCUSDT; give a finite number\), and on the "
                        r"in-sample bars they are not; decide from the bars up to each "
                        r"close", str(caught.value))
    assert type(caught.value.__cause__) is ProbatumValueError


class Noisy(Strategy):

    def __init__(self, seeded=False):
        self.seeded = seeded
        self.rng    = np.random.default_rng(7) if seeded else None


    def next(self, bar, account):
        draw = self.rng.random() if self.seeded else np.random.random()
        account.target(0.5 if draw > 0.5 else 0.0)


def noisy(data, k):
    return np.random.random(len(data))


def test_decisions_that_do_not_repeat_are_refused_unless_every_draw_is_seeded():
    fix = "; seed every random draw the policy makes, so the check can compare its runs"

    with pytest.raises(ProbatumValueError) as caught:
        searched(Noisy, {"seeded": Categorical([False])})

    assert type(caught.value) is ProbatumValueError
    assert re.fullmatch(r"policy: at .+ it calls target\(\d\.\d\) in one run on the "
                        r"in-sample bars and target\(\d\.\d\) in the next" + re.escape(fix),
                        str(caught.value))

    with pytest.raises(ProbatumValueError) as caught:
        searched(noisy, {"k": Int(1, 1)})

    said = re.fullmatch(r"policy: its decision for BTCUSDT at .+ is (\S+) in one run on "
                        r"the in-sample bars and (\S+) in the next" + re.escape(fix),
                        str(caught.value))
    assert said and float(said[1]) != float(said[2])
    # each run builds its own instance, whose stream is seeded as the trial's was
    assert searched(Noisy, {"seeded": Categorical([True])}).looks == 0


EVERY = np.full(30, 0.5)


def every(data, k):
    return EVERY


def test_a_reader_s_refusal_of_a_cut_s_decisions_is_a_lookahead_chained_from_it():
    with pytest.raises(LookaheadError) as caught:
        searched(every, {"k": Int(1, 1)})

    assert str(caught.value).startswith("policy: on the bars up to ")
    assert "its decisions are refused (policy: an array of shape (30,) for a Data of " in (
        str(caught.value))
    assert type(caught.value.__cause__) is ProbatumValueError


def lengthwise(dtype, by):
    """Weights in ``dtype`` that move by ``by`` of themselves on every cut."""
    def weights(data, k):
        values = (0.5 + 0.4 * np.sin(np.arange(len(data)) / 3.0)).astype(dtype)
        if len(data) < 30:
            values = (values * (1.0 + by)).astype(dtype) if by else np.nextafter(
                values, np.array(np.inf, dtype=dtype))

        return values

    return weights


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_a_narrow_float_s_last_bit_moving_with_the_bars_is_no_lookahead(dtype):
    assert searched(lengthwise(dtype, 0.0), {"k": Int(1, 1)}).looks == 0


def test_a_float64_moving_by_a_ten_millionth_with_the_bars_is_refused():
    with pytest.raises(LookaheadError, match="^policy: its decision for BTCUSDT at "):
        searched(lengthwise(np.float64, 1e-7), {"k": Int(1, 1)})


class Single(Strategy):
    """Weights a float32 model gives, whose last bit, or a ten-millionth in float64,
    moves on every cut."""


    def __init__(self, wide=False):
        self.wide = wide


    def init(self, data):
        values = 0.5 + 0.4 * np.sin(np.arange(len(data)) / 3.0)

        if self.wide:
            self.weights = values * (1.0 + 1e-7 * (len(data) < 30))
        else:
            narrow       = values.astype(np.float32)
            self.weights = (np.nextafter(narrow, np.float32(np.inf)) if len(data) < 30
                            else narrow)


    def next(self, bar, account):
        account.target(float(self.weights[bar.i]))


def test_a_strategy_s_float32_argument_is_held_to_float32_s_bound():
    assert searched(Single, {"wide": Categorical([False])}).looks == 0

    with pytest.raises(LookaheadError, match=r"^policy: at .+ it calls target\("):
        searched(Single, {"wide": Categorical([True])})


def barred(data, k):
    stops = np.full(len(data), 0.02, dtype=np.float32)
    if len(data) < 30:
        stops = np.nextafter(stops, np.float32(np.inf))

    return {"policy": np.full(len(data), 0.5), "stop": stops}


def tabled(data, k):
    values = lengthwise(np.float32, 0.0)(data, k)
    return pa.table({
        "time":    pa.array(data.time, pa.timestamp("ns", tz="UTC")),
        "BTCUSDT": pa.array(values, pa.float32()),
    })


def masked(data, k):
    values = lengthwise(np.float32, 0.0)(data, k)
    return indexed(data, values[:, None], ("BTCUSDT",)).astype("Float32")


class Once:
    """An Arrow stream a policy hands over once, as a reader that moves it does."""


    def __init__(self, table):
        self.capsule = table.__arrow_c_stream__()


    def __arrow_c_stream__(self, requested_schema=None):
        capsule, self.capsule = self.capsule, None
        return capsule


def streamed(data, k):
    return Once(tabled(data, k))


@pytest.mark.parametrize("policy", [barred, tabled, masked, streamed])
def test_a_narrow_float_barrier_or_table_is_held_to_its_own_bound(policy):
    # each read once, an Arrow table's floats judged by the values the reader read
    assert searched(policy, {"k": Int(1, 1)}).looks == 0


class Late(Strategy):
    """A one-bar lead from ``start`` on, so a trial that starts past the bars decides
    nothing to check."""


    def __init__(self, start=0):
        self.start = start


    def init(self, data):
        self.warmup = self.start
        self.moves  = np.sign(np.diff(data.close, append=data.close[-1]))


    def next(self, bar, account):
        account.target(max(float(self.moves[bar.i]), 0.0))


def test_a_trial_that_decides_nothing_leaves_the_check_to_the_next(caplog):
    with caplog.at_level("INFO", logger="probatum"), pytest.raises(LookaheadError):
        searched(Late, {"start": Categorical([1_000, 0])})

    # the second trial's own check refused it, so no trial was passed before it
    assert not [said for said in caplog.messages if said.endswith("decisions are causal")]


def peeking(data, ahead):
    moves = np.sign(np.diff(data.close, append=np.nan))
    return np.maximum(moves, 0.0) if ahead else np.zeros(len(data))


def test_the_winner_is_checked_where_its_parameters_differ_from_the_trial_checked():
    with pytest.raises(LookaheadError, match="^policy: its decision for BTCUSDT at "):
        searched(peeking, {"ahead": Categorical([False, True])})


def test_a_warm_up_that_leaves_two_rows_cuts_on_neither_the_last():
    # Cross decides on rows 27 and 28 of 30, and no cut ends on row 29, where it never
    # decides
    assert searched(Cross, {"fast": Int(2, 2), "slow": Int(27, 27)}).looks == 0


def fragile(data, k):
    if len(data) < 30:
        raise RuntimeError("needs the whole sample")

    return np.full(len(data), 0.5)


def test_an_error_the_policy_s_own_code_raises_on_a_cut_stays_its_own():
    with pytest.raises(RuntimeError, match="needs the whole sample") as caught:
        searched(fragile, {"k": Int(1, 1)})

    assert type(caught.value) is RuntimeError

    [note] = caught.value.__notes__
    assert re.fullmatch(r"raised by the policy on the bars up to .+, in search's causality "
                        r"check", note)


def weekly(data, k):
    data.resample("1w")
    return np.full(len(data), 0.5)


def test_a_refusal_of_the_policy_s_own_by_probatum_on_a_cut_stays_its_own():
    # the in-sample bars hold a week and the first cuts do not
    data = probatum.Data(bars(400), symbol="BTCUSDT", market="linear")

    with pytest.raises(DataError, match="^interval: '1w', and these bars cover no 1w "
                                        "bar to its end") as caught:
        searched(weekly, {"k": Int(1, 1)}, data)

    assert type(caught.value) is DataError

    [note] = caught.value.__notes__
    assert note.startswith("raised by the policy on the bars up to ")


class Peeking(Strategy):
    """The next close's move as its weight, which a cut's last close has no bar for."""


    def __init__(self, size=1.0):
        self.size = size


    def init(self, data):
        self.warmup, self.close = 2, data.close


    def next(self, bar, account):
        rise = self.close[bar.i + 1] > self.close[bar.i]
        account.target(self.size if rise else 0.0)


class Moving(Strategy):
    """Cash moved by the next close, read one bar ahead, which a cut's last close has
    none of."""


    def __init__(self, size=1.0):
        self.size = size


    def init(self, data):
        self.warmup, self.plan = 2, dict(enumerate(data.close.tolist()))


    def next(self, bar, account):
        ahead = self.plan.get(bar.i + 1)
        account.transfer(None if ahead is None else self.size, "spot")


def test_a_transfer_s_refusal_on_a_cut_is_a_lookahead_as_every_verb_s_is():
    with pytest.raises(LookaheadError, match=r"^policy: on the bars up to .+ its "
                                             r"decisions are refused \(amount: None; give "
                                             r"the cash to move\), and on the in-sample "
                                             r"bars they are not; "):
        searched(Moving, {"size": Categorical([1.0])})


def test_an_error_at_a_cut_s_last_close_says_that_no_bar_follows():
    with pytest.raises(IndexError) as caught:
        searched(Peeking, {"size": Categorical([1.0])})

    first, second = caught.value.__notes__
    assert first == ("raised by next at the last close, where the check has a strategy "
                     "decide though no bar follows")
    assert second.startswith("raised by the policy on the bars up to ")


SEEN = []


class Seen(Cross):

    def init(self, data):
        SEEN.append(data.time[-1])
        super().init(data)


def seen(data, fast, slow):
    SEEN.append(data.time[-1])
    return crossing(data, fast, slow)


@pytest.mark.parametrize("policy", [Seen, seen])
def test_the_check_hands_the_policy_the_in_sample_bars_alone(policy):
    SEEN.clear()

    searched(policy, {"fast": Int(2, 2), "slow": Int(8, 8)})

    # the trial, the two whole runs and the cuts, none past the in-sample bars
    assert len(SEEN) > 3 and max(SEEN) == one().split(30)[0].time[-1]


def scaled(data, k):
    return {
        "policy": np.full(len(data), 0.5),
        "stop":   float(np.std(data.close)) / 100.0,
    }


def test_a_setting_computed_from_every_bar_is_refused_as_one():
    with pytest.raises(LookaheadError, match=r"^policy: its stop is 0\.\d+ on the "
                                             r"in-sample bars and 0\.\d+ on the bars up "
                                             r"to .+; decide from the bars up to each "
                                             r"close$"):
        searched(scaled, {"k": Int(1, 1)})


def two_states(rows, period=3):
    """One instrument's decisions over ``rows`` rows, 1.0 and 0.0 by turns of
    ``period``."""
    values = np.where((np.arange(rows) // period) % 2 == 0, 1.0, 0.0)[:, None]
    return _Weights(values, {"stop": None, "take_profit": None}, "weight", 0.0, 1e-9)


def test_cuts_move_to_changes_that_alternate_and_run_once_with_the_row_before():
    whole   = two_states(25)
    printed = np.arange(25)
    rows    = list(range(24))

    # targets 4, 8, 12, 16 and 20: 6 an entry, 9 an exit, 12 an entry, 21 the exit
    # after the entry at 18, and 21 again, the only change left, each with the row
    # before it
    assert _cuts(whole, rows, printed, 1) == [5, 6, 8, 9, 11, 12, 20, 21]

    constant = _Weights(np.full((25, 1), 0.5), {"stop": None, "take_profit": None},
                        "weight", 0.0, 1e-9)

    assert _cuts(constant, rows, printed, 1) == [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
    # a cut that holds one printed row is dropped
    assert _cuts(constant, [0, 1], np.arange(3), 1) == [1]

    # with two instruments the second's targets, 8 and 16, move along its own changes:
    # to 10, where it is 0.5 for one row, and, with none after 16, nowhere
    second = np.zeros((25, 1))
    second[10] = 0.5
    pair = _Weights(np.hstack([two_states(25).rows, second]),
                    {"stop": None, "take_profit": None}, "weight", 0.0, 1e-9)

    assert _cuts(pair, rows, printed, 2) == [5, 6, 9, 10, 14, 15, 16, 20, 21]
