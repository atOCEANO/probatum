"""A validation searches the bars before its holdout first, its winner the Study's, then
each split's training bars, and trades each split's winner on its test span: a fit of
one piece is a search over its training bars, a piece after a span opens where the
Specification works it, every path is one run whose windows own the rows after their
decisions and are scored on them, a fit fails where no trial of its search completes
and a path where its run raises or a fit it trades failed, each winner the fits chose
that the first search did not check is checked for causality, the same seed gives the
same Study in this process and in workers, and every refusal names what to give
instead.
"""

import math
import multiprocessing

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import (Categorical, DataError, Int, LookaheadError, PolicyError,
                      ProbatumAttributeError, ProbatumNotImplementedError,
                      ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError,
                      ProbatumWarning, VenueError)
from probatum._backtest import _annualised, _markets
from probatum._data import _before
from probatum._probatum import Engine
from probatum._result import COSTS
from probatum._search import _pieces, _Trial
from probatum._strategy import _play, _Run
from probatum._streams import stream
from probatum._validate import _Fit, _laid, _owned, _Walk
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross
from test_a_training_piece_reads_no_bar_of_the_span_before_it import Reader

SPACE  = {"fast": Int(3, 8), "slow": Int(10, 30)}
SEARCH = {"trials": 6, "sampler": "random", "objective": "total_return_pct"}
# one instrument that printed nothing before the in-sample bars
FRESH = np.zeros(1, dtype=bool)


def swung(n=600):
    """Hourly bars that swing on two periods, so a crossing of averages trades both
    ways."""
    t      = np.arange(n, dtype=np.float64)
    close  = 100.0 * (1.0 + 0.1 * np.sin(t / 3.0)) * (1.0 + 0.05 * np.cos(t / 17.0))
    opened = np.concatenate([[close[0]], close[:-1]])
    index  = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    return pd.DataFrame({
        "open":   opened,
        "high":   np.maximum(opened, close) * 1.01,
        "low":    np.minimum(opened, close) * 0.99,
        "close":  close,
        "volume": 1e6,
    }, index=index)


def swinging(n=600):
    """``swung`` bars of one instrument."""
    return probatum.Data(swung(n), symbol="BTCUSDT", market="linear")


# the first in-sample bar, and the first of the last of four groups of them
TIMES = swinging().time
FIRST, LATE = TIMES[0], TIMES[360]


def validated(cv, policy=Cross, data=None, **given):
    """A validation of ``policy`` on 600 hourly bars, the last 120 held out."""
    return probatum.validate(policy, swinging() if data is None else data, VENUE, cv,
                             **{**dict(space=SPACE, search=SEARCH, holdout=0.2, seed=7),
                                **given})


def seeded(name, seed=7):
    """The seed of a validation's search ``name``, as its stream gives it."""
    return int(stream(seed, name).integers(0, 2**63))


# each splitter beside the first of the bars it gives the holdout to train on
SPLITTERS = [(probatum.cv.PurgedKFold(4), 0), (probatum.cv.Holdout(0.25), 0),
             (probatum.cv.WalkForward(150, 100, anchored=True), 0),
             (probatum.cv.WalkForward(150, 100), 330)]


@pytest.mark.parametrize("cv, first", SPLITTERS)
def test_the_winner_is_a_plain_search_s_on_the_holdout_s_training_bars(cv, first):
    data  = swinging()
    study = validated(cv)

    plain = probatum.search(Cross, SPACE, data[first:], VENUE, holdout=120,
                            seed=seeded("validate final"), keep="none", **SEARCH)

    assert study.best_params == plain.best_params
    assert study.best_score  == plain.best_score
    assert study.provenance["final"]["seed"] == seeded("validate final")
    assert study.best.stats == plain.best.stats
    assert study.evaluate_holdout().stats == plain.evaluate_holdout().stats
    assert study.evaluate_holdout() is study.holdout and study.looks == 1


def test_a_walk_forward_window_s_fit_is_a_search_over_its_training_bars():
    data = swinging()
    fits = validated(probatum.cv.WalkForward(150, 100)).fits

    assert list(fits["test"]) == ["150 to 249", "250 to 349", "350 to 479"]
    assert list(fits["train"]) == [150] * 3 and list(fits["pieces"]) == [1] * 3

    for row in fits:
        first = 150 + 100 * row["number"]
        assert row["seed"] == seeded(f"validate fit {row['number']}")

        plain = probatum.search(Cross, SPACE, data[first - 150:first], VENUE, holdout=0,
                                seed=row["seed"], keep="none", **SEARCH)
        assert {"fast": row["fast"], "slow": row["slow"]} == plain.best_params
        assert row["fitted"] == plain.best_score
        assert (row["trials"], row["complete"], row["state"]) == (6, 6, "complete")


def test_a_piece_after_a_span_opens_where_the_page_works_it():
    # the Specification's worked example, Fits: PurgedKFold(5) on 1,000 hourly bars of
    # a full clock, split 2 testing 400 to 599, at a venue whose fills read 24 bars
    full = np.ones((1_000, 1), dtype=bool)
    fit  = _laid(probatum.cv.PurgedKFold(5), 1_000, full, FRESH, 0, 24)[2]

    assert fit.specs == [(0, 0, 400, None), (601, 601, 1_000, 24)]
    assert (fit.train, fit.purged, fit.embargoed) == (799, 1, 0)

    data = swinging(1_000)
    piece, run, later = _pieces(data, VENUE, fit.specs)[1]

    for warmup, opens in ((20, 625), (200, 801)):
        engine       = Engine(run.candles, report=True, **run.settings)
        played       = _Run(engine, piece, run.start, _markets(run))
        played.later = later
        reader       = Reader(warmup)
        _play(reader, played)
        assert 601 + reader.rows[0] == opens and reader.first == data.time[601]

    embargoed = _laid(probatum.cv.PurgedKFold(5, embargo=30), 1_000, full, FRESH, 30,
                      24)[2]

    assert embargoed.specs == [(0, 0, 400, None), (601, 601, 1_000, 30)]
    assert (embargoed.train, embargoed.purged, embargoed.embargoed) == (769, 1, 30)

    row = validated(probatum.cv.PurgedKFold(5), data=data, holdout=0).fits[2]

    assert (row["test"], row["train"], row["purged"], row["embargoed"],
            row["pieces"]) == ("400 to 599", 799, 1, 0, 2)


class Opening(probatum.Strategy):
    """A strategy that records, for each run, the first bar of the Data it was handed
    and the first row it decided at."""

    opened = []


    def __init__(self, k=0):
        self.k = k


    def init(self, data):
        self.warmup, self.first, self.decided = 20, data.time[0], False


    def next(self, bar, account):
        if not self.decided:
            self.decided = True
            Opening.opened.append((self.first, bar.i))


def test_a_piece_after_a_span_too_short_to_open_is_left_out():
    # a piece after a span opens at its 25th printed row, the fill window's 24 and the
    # one before them, so it holds two printed rows from there at 26 rows: two groups
    # of 27 bars leave split 0 such a piece, and two of 26 leave it nothing to train on
    full = np.ones((54, 1), dtype=bool)
    fits = _laid(probatum.cv.PurgedKFold(2), 54, full, FRESH, 0, 24)

    assert [fit.specs for fit in fits] == [[(28, 28, 54, 24)], [(0, 0, 27, None)]]

    with pytest.raises(ProbatumValueError) as caught:
        _laid(probatum.cv.PurgedKFold(2), 52, full[:52], FRESH, 0, 24)

    assert str(caught.value) == ("cv: split 0, testing group 0, whose training bars "
                                 "hold no piece two printed rows long once its account "
                                 "opens; give it more bars to train on")


def test_a_piece_in_which_no_instrument_fills_a_window_still_runs():
    # A prints rows 0 to 39 and B rows 45 to 50, so the piece after group 0, from row
    # 31, holds 9 of A's bars and opens at the last of them, row 39, while B, which
    # printed nothing before it, holds no opening, leaving it 7 printed rows
    available = np.zeros((60, 2), dtype=bool)
    available[:40, 0]   = True
    available[45:51, 1] = True

    fits = _laid(probatum.cv.PurgedKFold(2), 60, available, np.zeros(2, dtype=bool), 0,
                 24)

    assert [fit.specs for fit in fits] == [[(31, 31, 60, 8)], [(0, 0, 30, None)]]


def test_a_piece_after_two_spans_runs_from_the_later_of_their_reaches():
    # bars 100 to 309 print nothing, so of split 1's spans group 0 reaches bar 310 and
    # group 2, printing no row, its own last bar, 299: the piece runs from bar 311
    available = np.ones((600, 1), dtype=bool)
    available[100:310] = False

    fit = _laid(probatum.cv.CombinatorialPurged(6, 2), 600, available, FRESH, 0,
                24)[1]

    assert fit.specs == [(311, 311, 600, 24)]


@pytest.mark.parametrize("window, embargo, opens", [(24, None, 24), (48, None, 48),
                                                    (24, 30, 30)])
def test_a_piece_opens_past_its_embargo_and_its_venue_s_window(window, embargo, opens):
    # split 2 of PurgedKFold(5) on 1,000 bars tests 400 to 599, so the piece after it
    # runs on a Data from bar 601 and opens at its printed row window + 1, window rows
    # in, or past the embargo where that is later, past a warm-up of 20 either way
    data           = swinging(1_000)
    Opening.opened = []
    study          = probatum.validate(Opening, data, VENUE.replace(fills={"window": window}),
                                       probatum.cv.PurgedKFold(5, embargo=embargo),
                                       space={"k": Categorical([0])}, search=SEARCH, holdout=0,
                                       seed=7)

    assert (data.time[601], opens) in Opening.opened
    assert all(row == opens for first, row in Opening.opened if first != data.time[0])
    assert study.fits["embargoed"][2] == (embargo or 0)


def test_on_a_sparse_clock_a_span_reaches_and_a_window_owns_by_printed_rows():
    frame = swung()

    # an instrument that lists at row 300 and prints no bar at rows 360 to 364
    late           = frame.iloc[list(range(300, 360)) + list(range(365, 600))]
    universe       = probatum.Data({"BTCUSDT": frame, "NEWUSDT": late}, market="linear")
    data           = universe["NEWUSDT"]
    Opening.opened = []
    study          = probatum.validate(Opening, data, VENUE, probatum.cv.PurgedKFold(4),
                                       space={"k": Categorical([0])}, search=SEARCH, holdout=0.2,
                                       seed=7)
    fits, windows = study.fits, study.windows

    # groups 0 and 1 print nothing and reach their own last bars; group 2's last printed
    # row is 359 and the first after it 365, so 360 to 365 are purged
    assert list(fits["purged"]) == [0, 0, 6, 0]

    # the pieces from the first row in splits 1 and 2 hold no printed row and are left
    # out, rather than run on no bars
    assert list(fits["pieces"]) == [1, 1, 1, 1]

    # the piece after group 0, on a Data from bar 120, decides first at its first
    # printed row, bar 300, its warm-up of 20 rows past, as the instrument printed
    # nothing before it and so has the window the in-sample bars give it; the piece
    # after group 2, on a Data from bar 366, opens at its 25th printed row, 24 rows in
    assert (data.time[120], 300 - 120) in Opening.opened
    assert (data.time[366], 24) in Opening.opened

    printed = np.flatnonzero(data.available[:480].any(axis=1))

    assert list(windows["fit"]) == [2, 3]
    assert (list(windows["first"]), list(windows["last"])) == ([1, 61], [60, 174])
    np.testing.assert_array_equal(windows["first_time"], data.time[printed[[1, 61]]])


def test_on_a_universe_a_piece_waits_for_each_instrument_the_cut_took_bars_from():
    frame = swung()
    late  = frame.iloc[list(range(300, 360)) + list(range(365, 600))]

    # an instrument that prints bars 150 to 160 after a halt and then never again
    gone     = frame.iloc[list(range(120)) + list(range(150, 161))]
    universe = probatum.Data({"BTCUSDT": frame, "NEWUSDT": late, "OLDUSDT": gone},
                             market="linear")

    Opening.opened = []
    probatum.validate(Opening, universe, VENUE, probatum.cv.PurgedKFold(4),
                      space={"k": Categorical([0])}, search=SEARCH, holdout=0.2, seed=7)

    # NEWUSDT lists at bar 300, so the pieces from bars 121 and 241, which start before
    # it, have its window as the in-sample bars do and never wait for it: the piece
    # after group 1 opens at BTCUSDT's 25th bar in it; the piece after group 0 opens at
    # OLDUSDT's last bar, 160, as its 11 bars there never fill a window; and the piece
    # after group 2, from bar 361, waits for NEWUSDT's 25th bar in it, bar 389, four
    # rows after BTCUSDT's, as it prints nothing at 361 to 364
    pieces = {(first, row) for first, row in Opening.opened
              if first != universe.time[0]}

    assert pieces == {(universe.time[121], 39), (universe.time[241], 24),
                      (universe.time[361], 28)}


def test_a_piece_waits_for_an_instrument_that_printed_before_the_data_given():
    frame          = swung(700)
    eth            = frame.iloc[list(range(95)) + list(range(450, 700))]
    universe       = probatum.Data({"BTCUSDT": frame, "ETHUSDT": eth}, market="linear")
    data           = universe[100:]
    Opening.opened = []

    probatum.validate(Opening, data, VENUE, probatum.cv.PurgedKFold(4),
                      space={"k": Categorical([0])}, search=SEARCH, holdout=0, seed=7)

    # the in-sample run's window still holds ETHUSDT's bars before bar 95, though the
    # Data given starts at bar 100, so the pieces after groups 0 and 1, from rows 151
    # and 301, wait for ETHUSDT's 25th bar in them, row 374, where it prints from row
    # 350; the piece after group 2, from row 451, opens at the 25th bar of both
    pieces = {(first, row) for first, row in Opening.opened if first != data.time[0]}

    assert pieces == {(data.time[151], 223), (data.time[301], 73),
                      (data.time[451], 24)}


def test_a_path_is_one_run_whose_windows_own_the_rows_after_their_decisions():
    data = swinging()

    # funding on every bar, so every row a window holds a position on pays some
    funded = VENUE.replace(funding={"every_bars": 1, "rate": 0.0001})
    study  = probatum.validate(Cross, data, funded,
                               probatum.cv.CombinatorialPurged(4, 2), space=SPACE,
                               search=SEARCH, holdout=0.2, seed=7)
    paths, fits, windows = study.paths, study.fits, study.windows
    returns = study.returns

    assert returns.shape == (3, 480) and not returns.flags.writeable

    # path p takes each group from the p-th split that tests it, and adjacent groups
    # one split trades are one window
    assert [list(windows["fit"][windows["path"] == p]) for p in range(3)] == [
        [0, 1, 2], [1, 3, 4], [2, 4, 5]]

    for p in range(3):
        mine = windows[windows["path"] == p]
        first, last = mine["first"], mine["last"]

        assert first[0] == 1 and last[-1] == 479
        assert list(first[1:]) == list(last[:-1] + 1)
        np.testing.assert_array_equal(mine["first_time"], data.time[first])
        np.testing.assert_array_equal(mine["last_time"], data.time[last])
        assert np.isnan(returns[p, 0]) and not np.isnan(returns[p, 1:]).any()

        for row in mine:
            grown = np.prod(1.0 + returns[p, row["first"]:row["last"] + 1]) - 1.0
            assert row["total_return_pct"] == pytest.approx(100.0 * grown, abs=1e-9)
            assert row["score"] == row["total_return_pct"]

            # each window decides at its rows from its winner's warm-up, the closes
            # before the rows it owns
            warmup = fits["slow"][row["fit"]]
            assert row["decided"] == (row["last"] - row["first"] + 1
                                      - max(0, warmup - row["first"] + 1))

        whole = paths[p]
        grown = np.prod(1.0 + returns[p, 1:]) - 1.0

        assert whole["state"] == "complete" and whole["error"] is None
        assert whole["total_return_pct"] == pytest.approx(100.0 * grown, abs=1e-9)
        assert whole["score"] == whole["total_return_pct"]

        for name in ("num_trades", "num_fills"):
            assert mine[name].sum() == whole[name]
        for name in COSTS + ("funding_paid",):
            assert mine[name].sum() == pytest.approx(whole[name], abs=1e-9)
        assert whole["funding_paid"] != 0.0

        # the steps each window held a position in add up to the path's
        held = (mine["exposure_pct"] * (last - first + 1)).sum()
        assert held == pytest.approx(whole["exposure_pct"] * 479)


def test_a_window_s_statistics_are_the_engine_s_over_the_same_rows():
    # over every row after the opening a window's arithmetic gives the engine's own
    # statistics and costs, so it reads each record on the rows it names
    data  = swinging()
    venue = VENUE.replace(fees={"maker": 0.0002, "taker": 0.0005},
                          funding={"every_bars": 1, "rate": 0.0001})

    result  = probatum.backtest(Cross(5, 20), data, venue)
    holding = np.abs(result.holdings["position"]) > 0.0

    stats, costs = _owned(result, holding, 1, len(result.time) - 1, _annualised(data),
                          0.0)

    assert stats == pytest.approx(result.stats, nan_ok=True)
    assert costs == pytest.approx(result.costs)
    assert result.stats["funding_paid"] != 0.0 and costs["fees"] != 0.0


class Trip(probatum.Strategy):
    """A long bought at row 10's close and sold at row 20's, its fills on rows 11 and
    21."""


    def next(self, bar, account):
        if bar.i == 10:
            account.buy(1.0)
        elif bar.i == 20:
            account.sell(1.0)


def test_a_window_owns_the_fills_and_the_round_trip_its_decisions_made():
    # a window deciding at rows 10 to 20 owns rows 11 to 21, the trip's two fills and
    # its exit among them, and the windows either side of it none of the trip
    data    = swinging()
    result  = probatum.backtest(Trip(), data, VENUE)
    holding = np.abs(result.holdings["position"]) > 0.0
    rate    = _annualised(data)
    owned   = [_owned(result, holding, first, last, rate, 0.0)[0]
               for first, last in ((1, 10), (11, 21), (22, len(result.time) - 1))]

    assert [(stats["num_trades"], stats["num_fills"]) for stats in owned] == [
        (0, 0), (1, 2), (0, 0)]

    # rows 11 to 20 end holding the long, ten of the window's eleven
    assert owned[1]["exposure_pct"] == pytest.approx(100.0 * 10 / 11)
    assert owned[0]["exposure_pct"] == owned[2]["exposure_pct"] == 0.0


class Waiting(probatum.Strategy):
    """A strategy whose warm-up is its parameter, so a run's decisions say whose
    parameters it ran, holding ``weight`` from there."""


    def __init__(self, wait=0, weight=0.5):
        self.wait, self.weight = wait, weight


    def init(self, data):
        self.warmup = self.wait


    def next(self, bar, account):
        account.target(self.weight)


@pytest.mark.parametrize("waits, decided", [((0, 10_000), [240, 0]),
                                            ((10_000, 0), [0, 239])])
def test_each_window_of_a_path_trades_its_own_fit_s_winner(waits, decided):
    # a warm-up past every bar never decides
    inside = _before(swinging(), 480)
    fits   = [_Fit(number, [], 0, 0, 0, []) for number in range(2)]
    for fit, wait in zip(fits, waits):
        fit.winner = _Trial(fit.number, {"wait": wait}, "complete", 0.0, {})

    walk = _Walk(0, [(0, 239, 0), (240, 478, 1)], np.arange(480))

    walk.ran(Waiting, "class", "strategy", fits, "cancel", inside, VENUE,
             _annualised(inside), 0.0, "total_return_pct")

    assert walk.decided == decided


def test_a_path_counts_the_rows_it_ends_holding_a_short_on():
    # a short from row 101's open, so 379 of the 479 rows after the opening end
    # holding it
    inside     = _before(swinging(), 480)
    fit        = _Fit(0, [], 0, 0, 0, [])
    fit.winner = _Trial(0, {"wait": 100, "weight": -0.5}, "complete", 0.0, {})
    walk       = _Walk(0, [(0, 478, 0)], np.arange(480))

    walk.ran(Waiting, "class", "strategy", [fit], "cancel", inside, VENUE,
             _annualised(inside), 0.0, "total_return_pct")

    assert walk.stats["exposure_pct"]       == pytest.approx(100.0 * 379 / 479)
    assert walk.parts[0][0]["exposure_pct"] == walk.stats["exposure_pct"]


class Late(Cross):
    """A crossing that raises once a run that did not open at the first in-sample bar
    reaches the last of their four groups, as a fit's piece after a span does."""


    def __init__(self, fast=5, slow=20, flag=False):
        super().__init__(fast, slow)
        self.flag = flag


    def init(self, data):
        super().init(data)
        self.opened = data.time[0]


    def next(self, bar, account):
        if self.opened != FIRST and bar.ns >= LATE:
            raise ValueError("a piece reached the last group")

        super().next(bar, account)


def test_a_fit_with_no_complete_trial_fails_and_so_does_each_path_it_trades():
    study = validated(probatum.cv.CombinatorialPurged(4, 2), policy=Late,
                      space={**SPACE, "flag": Categorical([False, True])})
    fits, paths = study.fits, study.paths

    # the splits whose pieces after a span reach the last group: 0, 1 and 3
    assert list(fits["state"]) == ["failed", "failed", "complete", "failed", "complete",
                                   "complete"]
    assert list(fits["trials"])   == [6] * 6
    assert list(fits["complete"]) == [0, 0, 6, 0, 6, 6]
    assert fits["error"][0] == ("ProbatumRuntimeError: policy: all 6 trials failed, "
                                "the last, trial 5, with ValueError: a piece reached "
                                "the last group; make it run on each of the fit's "
                                "training pieces, each a Data of its own")

    # a failed fit has no winner, its parameters missing whatever their kind
    for name in ("fast", "flag"):
        assert list(fits[name].mask) == [True, True, False, True, False, False]
        assert fits[0][name] is None
    assert math.isnan(fits["fitted"][0])

    framed = fits.to_pandas()

    assert framed["flag"].dtype        == "boolean"
    assert list(framed["flag"].isna()) == list(fits["flag"].mask)
    assert list(paths["state"])        == ["failed", "failed", "complete"]
    assert paths["error"][0].startswith("fit 0 failed (ProbatumRuntimeError: policy: "
                                        "all 6 trials failed")
    assert paths["error"][1].startswith("fit 1 failed (")
    assert np.isnan(study.returns[:2]).all()
    assert not np.isnan(study.returns[2, 1:]).any()

    windows = study.windows
    lost    = windows[windows["path"] < 2]

    assert lost["decided"].mask.all() and np.isnan(lost["score"]).all()

    median = paths["score"][2]

    assert repr(study) == (f"Study(3 paths, 2 failed, of Late from 6 fits by "
                           f"CombinatorialPurged(4, 2), stability evidence, median "
                           f"total_return_pct {median:,.2f} over 480 bars; 120 held "
                           f"out, not yet read)")


class Fragile(Cross):
    """A crossing whose on_inherit raises at a seam in the last group of the bars."""


    def on_inherit(self, bar, account):
        if bar.i >= 360:
            raise ValueError("no protection after row 360")


def test_a_path_whose_run_raises_fails_alone():
    study = validated(probatum.cv.CombinatorialPurged(4, 2), policy=Fragile)

    assert list(study.fits["state"]) == ["complete"] * 6

    paths = study.paths

    # paths 0 and 1 hand over at row 360, path 2 at rows 120 and 240
    assert list(paths["state"]) == ["failed", "failed", "complete"]

    lost = "ValueError: no protection after row 360"
    assert list(paths["error"]) == [lost, lost, None]


class Early(Cross):
    """A crossing that raises in any run that does not open at the first in-sample
    bar."""


    def init(self, data):
        if data.time[0] != FIRST:
            raise ValueError("a piece after a span")

        super().init(data)


def test_a_validation_with_no_path_complete_raises_from_the_last_failure():
    with pytest.raises(ProbatumRuntimeError) as raised:
        validated(probatum.cv.CombinatorialPurged(4, 2), policy=Early)

    assert str(raised.value).startswith("policy: all 3 paths failed, the last, path 2, "
                                        "with fit 2 failed (ProbatumRuntimeError: "
                                        "policy: all 6 trials failed")
    assert str(raised.value.__cause__).startswith("policy: all 6 trials failed")
    assert str(raised.value.__cause__.__cause__) == "a piece after a span"


class Peeking(Cross):
    """A crossing that, with ``peek`` and on fewer than 400 bars, reads the next close,
    so on the 480 bars before the holdout it decides as the crossing does."""


    def __init__(self, fast=5, slow=20, peek=False):
        super().__init__(fast, slow)
        self.peek = peek


    def init(self, data):
        super().init(data)
        self.ahead = data.close if self.peek and len(data) < 400 else None


    def next(self, bar, account):
        if self.ahead is None or bar.i + 1 == len(self.ahead):
            return super().next(bar, account)

        account.target(1.0 if self.ahead[bar.i + 1] > bar.close else -1.0)


class Ahead(probatum.Strategy):
    """A strategy that waits 200 bars and then reads the next close."""


    def __init__(self, k=1):
        self.k = k


    def init(self, data):
        self.warmup, self.close = 200, data.close


    def next(self, bar, account):
        if bar.i + 1 < len(self.close):
            account.target(1.0 if self.close[bar.i + 1] > bar.close else -1.0)


def test_a_winner_the_first_search_could_not_check_is_checked_on_the_bars():
    # the first search of a rolling WalkForward runs on the last 150 of the 480 bars,
    # where a warm-up of 200 decides nothing, so it checks nothing; the one fit chooses
    # the first point of the grid, which the 480 bars find reading ahead, and its one
    # window leaves no seam to check
    with pytest.raises(LookaheadError):
        validated(probatum.cv.WalkForward(150, 330), policy=Ahead,
                  space={"k": Categorical([1, 2])},
                  search={"sampler": "grid", "objective": "total_return_pct"})


def test_a_winner_that_decides_nothing_is_not_recorded_as_checked():
    # Opening places nothing, so no search and no fit's winner has a decision to check
    study = validated(probatum.cv.Holdout(0.25), policy=Opening,
                      space={"k": Categorical([0, 1])},
                      search={"sampler": "grid", "objective": "total_return_pct"})

    assert study.provenance["checked"] == []


def test_a_fit_its_floor_leaves_no_complete_trial_fails_naming_its_pieces():
    # the first search's 480 bars close some 48 round trips and fit 0's 150 some 14
    with pytest.raises(ProbatumRuntimeError) as raised:
        validated(probatum.cv.WalkForward(150, 100, anchored=True),
                  search={**SEARCH, "min_trades": 30})

    said = str(raised.value)

    assert said.startswith("policy: its one path failed, with fit 0 failed "
                           "(ProbatumValueError: min_trades: 30, and no trial of 6 "
                           "closed that many round trips on the fit's training pieces, "
                           "the most ")
    assert said.endswith("; the error it was raised from shows where")
    assert type(raised.value.__cause__) is ProbatumValueError


def test_each_winner_the_fits_chose_is_checked_for_causality():
    # the first search finds both alike and checks the first; every fit, on fewer bars,
    # chooses the one that reads ahead, which the check refuses on the in-sample bars
    space  = {"peek": Categorical([False, True])}
    search = {"sampler": "grid", "objective": "total_return_pct"}
    first  = probatum.search(Peeking, space, swinging(), VENUE, holdout=120, **search)

    assert first.best_params == {"peek": False}

    with pytest.raises(LookaheadError):
        validated(probatum.cv.PurgedKFold(4), policy=Peeking, space=space,
                  search=search)


class Peeks(Cross):
    """A crossing whose on_inherit reads the next close."""


    def init(self, data):
        super().init(data)
        self.close = data.close


    def on_inherit(self, bar, account):
        if bar.i + 1 < len(self.close):
            account.target(0.5 if self.close[bar.i + 1] > bar.close else 0.25)
        else:
            account.target(0.1)


def test_what_on_inherit_decides_at_a_seam_is_checked_for_causality():
    # next is causal, and the fits never seam, so only a path cut at a seam shows it
    with pytest.raises(LookaheadError):
        validated(probatum.cv.CombinatorialPurged(4, 2), policy=Peeks)


def test_a_rolling_walk_forward_whose_first_search_holds_no_bars_is_refused():
    frame = swung(1_000)

    # an instrument that prints no bar at rows 551 to 749, which the 200 bars before a
    # holdout from row 750 cover but one of
    gappy    = frame.drop(frame.index[551:750])
    universe = probatum.Data({"BTCUSDT": frame, "ETHUSDT": gappy}, market="linear")

    with pytest.raises(ProbatumValueError) as caught:
        validated(probatum.cv.WalkForward(200, 100), data=universe["ETHUSDT"],
                  holdout=0.25)

    assert str(caught.value) == ("cv: WalkForward(200, 100), whose first search runs "
                                 "on the last 200 bars before the holdout, 1 of them "
                                 "printed; give a train that holds 2")


def test_a_seeded_validation_repeats_and_its_workers_give_what_this_process_does():
    pytest.importorskip("cloudpickle")

    cv = probatum.cv.CombinatorialPurged(4, 2)
    here, again = validated(cv), validated(cv)
    far = validated(cv, search={**SEARCH, "n_jobs": 2})

    for name in ("paths", "fits", "windows"):
        assert getattr(here, name) == getattr(again, name) == getattr(far, name)

    np.testing.assert_array_equal(here.returns, far.returns)
    assert here.best_params == far.best_params


def test_a_study_of_paths_reads_as_one_and_refuses_what_a_search_s_holds():
    study = validated(probatum.cv.WalkForward(150, 100))

    assert study.kind == "paths" and not hasattr(study, "trials")
    with pytest.raises(ProbatumAttributeError) as caught:
        study.trials

    assert str(caught.value) == ("trials: a Study of paths, which keeps each fit's "
                                 "winner and no trials; read paths, fits or windows")

    searched = probatum.search(Cross, SPACE, swinging(), VENUE, holdout=0.2, seed=7,
                               **SEARCH)

    assert searched.kind == "trials" and searched.provenance["kind"] == "trials"

    for name in ("paths", "fits", "windows"):
        assert not hasattr(searched, name)
        with pytest.raises(ProbatumAttributeError) as caught:
            getattr(searched, name)
        assert str(caught.value) == (f"{name}: a Study of trials, which search made; "
                                     f"validate makes a Study of paths")

    score = study.paths["score"][0]

    assert repr(study) == (f"Study(1 path of Cross from 3 fits by WalkForward(150, "
                           f"100), stability evidence, median total_return_pct "
                           f"{score:,.2f} over 480 bars; 120 held out, not yet read)")

    given = study.provenance

    assert given["kind"] == "paths" and given["seam"] == "cancel"
    assert given["cv"] == dict(given="WalkForward(150, 100)", bars=480, splits=3,
                               paths=1, embargo=0)
    assert given["fits"] == list(study.fits["seed"]) and given["seed"] == 7
    assert given["holdout"]         == dict(given=0.2, rows=120, first="2024-01-21 00:00")
    assert given["final"]["params"] == study.best_params
    assert given["final"]["bars"] == 150 and study.best_params in given["checked"]

    for params in given["checked"]:
        params["fast"] = 99

    assert study.best_params == given["final"]["params"]


def test_keep_none_keeps_no_returns_and_every_table():
    study = validated(probatum.cv.Holdout(0.25), keep="none")

    assert study.returns is None
    assert (len(study.paths), len(study.fits), len(study.windows)) == (1, 1, 1)


def test_a_validation_that_held_nothing_out_says_so():
    study = validated(probatum.cv.Holdout(0.25), holdout=0)

    with pytest.raises(ProbatumValueError) as caught:
        study.evaluate_holdout()

    assert str(caught.value) == ("evaluate_holdout: the validation held nothing out; "
                                 "validate with a holdout, such as holdout=0.3")
    assert repr(study).endswith("over 600 bars; nothing held out)")

    with pytest.warns(ProbatumWarning, match="holdout: none, so nothing is left to "
                                             "judge the winner on"):
        validated(probatum.cv.Holdout(0.25), holdout=None)


def test_validate_refuses_to_run_in_a_worker_as_it_starts(monkeypatch):
    monkeypatch.setattr(multiprocessing.current_process(), "_inheriting", True,
                        raising=False)

    with pytest.raises(ProbatumRuntimeError) as caught:
        validated(probatum.cv.Holdout(0.25))

    assert str(caught.value) == ("validate: called by a worker of a search as it "
                                 "starts, as spawn runs the script's top level again "
                                 "in each; put the call under if __name__ == "
                                 "\"__main__\":")


class Model:

    def fit(self, x, y):
        return self


class Clashing(probatum.Strategy):

    def __init__(self, test=1):
        self.test = test


    def next(self, bar, account):
        pass


REFUSED = [
    (dict(policy=Cross()), PolicyError,
     "policy: an instance of Cross; pass the class, Cross, which each fit builds with "
     "its parameters"),
    (dict(policy=dict), PolicyError,
     "policy: the class dict, which is no Strategy; pass a Strategy subclass"),
    (dict(policy=np.zeros(3)), PolicyError,
     "policy: an object of type ndarray, which is no Strategy; pass a Strategy "
     "subclass"),
    (dict(policy=Model()), ProbatumNotImplementedError,
     "policy: an object of type Model, a model that fits, which validate does not fit "
     "yet; pass a Strategy subclass"),
    (dict(policy=Model), ProbatumNotImplementedError,
     "policy: the class Model, a model that fits, which validate does not fit yet; "
     "pass a Strategy subclass"),
    (dict(features=np.zeros((480, 1, 3))), ProbatumValueError,
     "features: an object of type ndarray beside a Strategy, which reads its own Data; "
     "leave it out"),
    (dict(fill="ffill"), ProbatumValueError,
     "fill: an object of type str beside a Strategy, which reads its own Data; leave "
     "it out"),
    (dict(space=None), ProbatumValueError,
     "space: none, and a fixed parameter set fits nothing, so a Strategy is validated "
     "by a search; give each parameter a range, such as {\"fast\": "
     "probatum.Int(5, 40)}"),
    (dict(policy=Clashing, space={"test": Int(1, 2)}), ProbatumValueError,
     "space: test, a column the fits table has of its own; name the parameter "
     "otherwise"),
    (dict(cv="kfold"), ProbatumTypeError,
     "cv: 'kfold'; give a probatum.cv splitter, such as probatum.cv.PurgedKFold(5)"),
    (dict(cv=probatum.cv.PurgedKFold(4, end=np.arange(1, 481))), ProbatumValueError,
     "cv: PurgedKFold(4, end of shape (480,)), built with the labels' end, which a "
     "Strategy has none of; build it without end"),
    (dict(search=[("trials", 3)]), ProbatumTypeError,
     "search: [('trials', 3)]; give a dict of probatum.search's keywords, such as "
     "{\"trials\": 200}"),
    (dict(search={"holdout": 0.3}), ProbatumValueError,
     "search: holdout, which validate takes as its own; pass holdout= to validate"),
    (dict(search={"storage": "studies"}), ProbatumValueError,
     "search: storage, which a validation's searches do not save to; leave it out"),
    (dict(search={"trails": 3}), ProbatumTypeError,
     "search: 'trails', which probatum.search does not take; give objective, "
     "direction, trials, sampler, min_trades, n_jobs"),
    (dict(search={"trials": 0}), ProbatumValueError,
     "trials: 0; give a whole number from 1"),
    (dict(search={"objective": "calmar"}), ProbatumValueError,
     "objective: 'calmar' with PurgedKFold, whose fits join pieces in an order it "
     "reads; use one the order does not change, such as sharpe"),
    (dict(keep="holdout"), ProbatumValueError,
     "keep: 'holdout'; use \"all\" or \"none\""),
    (dict(seam="close"), ProbatumValueError,
     "seam: 'close'; use \"cancel\" or \"flatten\""),
    (dict(seam=True), ProbatumTypeError, "seam: True; use \"cancel\" or \"flatten\""),
    (dict(seed=-1), ProbatumValueError, "seed: -1; give a whole number from 0"),
    (dict(data=swinging()[:1], holdout=0), DataError,
     "data: one printed row to validate; give a Data of at least 2"),
    (dict(risk_free=math.inf), ProbatumValueError,
     "risk_free: inf; give the annual rate as a finite number, such as 0.04"),
    (dict(venue=VENUE.replace(intrabar="random")), VenueError,
     "venue: intrabar is \"random\", which validate refuses so that a validation "
     "repeats without a seed; replace(intrabar=\"pessimistic\") or \"optimistic\""),
    (dict(cv=probatum.cv.PurgedKFold(500)), ProbatumValueError,
     "k: 500 for 480 bars, which leaves a group with none; give k up to 480"),
    (dict(cv=probatum.cv.WalkForward(1, 100)), ProbatumValueError,
     "cv: window 0, testing bars 1 to 100, whose training bars hold no piece two "
     "printed rows long once its account opens; give it more bars to train on"),
    (dict(cv=probatum.cv.PurgedKFold(480), seam="flatten"), ProbatumValueError,
     "seam: 'flatten', and path 0's window at bar 0, which fit 0 trades, decides at "
     "one row, which the seam takes, so it never decides; use \"cancel\", or give "
     "the groups more bars"),
    (dict(cv=probatum.cv.WalkForward(150, 1), seam="flatten"), ProbatumValueError,
     "seam: 'flatten', and path 0's window at bar 150, which fit 0 trades, decides at "
     "one row, which the seam takes, so it never decides; use \"cancel\", or give "
     "its windows more bars to test"),
    (dict(cv=probatum.cv.CombinatorialPurged(4, 2),
          search={"objective": "max_drawdown_pct"}), ProbatumValueError,
     "objective: 'max_drawdown_pct' with CombinatorialPurged, whose fits join pieces "
     "in an order it reads; use one the order does not change, such as sharpe"),
    (dict(cv=probatum.cv.Holdout(1)), ProbatumValueError,
     "cv: Holdout(1), whose path tests no row a run decides at, a printed row before "
     "the in-sample bars' last; give a longer tail"),
    (dict(cv=probatum.cv.WalkForward(479, 1)), ProbatumValueError,
     "cv: WalkForward(479, 1), whose path tests no row a run decides at, a printed row "
     "before the in-sample bars' last; give a shorter train"),
]


@pytest.mark.parametrize("given, kind, said", REFUSED)
def test_a_validation_refuses_what_it_cannot_run_with_the_fix(given, kind, said):
    given = {**dict(policy=Cross, data=swinging(), venue=VENUE,
                    cv=probatum.cv.PurgedKFold(4), space=SPACE, search=SEARCH,
                    holdout=0.2, seed=7), **given}

    with pytest.raises(kind) as raised:
        probatum.validate(**given)

    assert type(raised.value) is kind and str(raised.value).startswith(said)
