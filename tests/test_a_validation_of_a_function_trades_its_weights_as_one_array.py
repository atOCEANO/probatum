"""A function validates as a Strategy class does: each fit is a search over its training
pieces, the function called with each piece's Data, and a path of its weights is one
array run of each window's rows of its winner's weights on the bars before the holdout,
whose seams cancel what is not yet filled and place each inherited bracket again. A
path whose windows return another sizing or tolerance fails, a function that returns a
Strategy is validated as that Strategy's class is, and its workers give what this
process does.
"""

import numpy as np
import pytest

import probatum
from probatum import ProbatumValueError
from probatum._data import _before, _piece
from probatum._validate import _laid, _weighed
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross
from test_a_validation_trades_each_split_s_winner_on_its_test_span import (SEARCH, SPACE,
                                                                           seeded,
                                                                           swinging)


def crossing(data, fast, slow):
    """Long while the fast average is above the slow one and flat otherwise, a stop 2%
    below each entry."""
    up, down = probatum.ta.ema(data.closes, fast), probatum.ta.ema(data.closes, slow)
    weights  = np.where(np.isnan(down), np.nan, (up > down).astype(np.float64))

    return {"policy": weights, "stop": 0.02}


def validated(cv, policy=crossing, **given):
    """A validation of ``policy`` on 600 hourly bars, the last 120 held out."""
    return probatum.validate(policy, swinging(), VENUE, cv,
                             **{**dict(space=SPACE, search=SEARCH, holdout=0.2, seed=7),
                                **given})


def composed(study, inside):
    """The rows a path of ``crossing``'s weights runs, from the windows table and each
    fit's winner: each window's from the decision row before the first account row it
    owns, NaN before the first window."""
    rows    = np.full((len(inside), 1), np.nan)
    winners = {row["number"]: (row["fast"], row["slow"]) for row in study.fits}
    windows = sorted((row["first"] - 1, row["fit"]) for row in study.windows
                     if row["path"] == 0)
    ends    = [first for first, _ in windows[1:]] + [len(inside)]

    for (first, fit), end in zip(windows, ends):
        fast, slow = winners[fit]
        rows[first:end] = crossing(inside, fast, slow)["policy"][first:end]

    return rows


class Fitted:
    """What `_weighed` reads of a fit, its winner's parameters."""


    def __init__(self, fast, slow):
        self.winner = type("Winner", (), {"params": {"fast": fast, "slow": slow}})


def test_a_walk_forward_window_s_fit_is_a_search_of_the_function_over_its_bars():
    data = swinging()
    fits = validated(probatum.cv.WalkForward(150, 100)).fits

    for row in fits:
        first = 150 + 100 * row["number"]
        plain = probatum.search(crossing, SPACE, data[first - 150:first], VENUE,
                                holdout=0, seed=seeded(f"validate fit {row['number']}"),
                                keep="none", **SEARCH)
        assert {"fast": row["fast"], "slow": row["slow"]} == plain.best_params
        assert row["fitted"] == plain.best_score


def test_a_piece_after_a_span_runs_the_function_s_weights_from_where_it_opens():
    # long on every bar, so the fit testing the first group scores the return of
    # holding from where its one piece's account opens, past the venue's fill window
    def held(data, fast, slow):
        return np.ones((len(data), 1))

    study  = validated(probatum.cv.PurgedKFold(4), policy=held,
                       search={**SEARCH, "trials": 1})
    inside = _before(swinging(), 480)
    fit    = _laid(probatum.cv.PurgedKFold(4), 480, inside.available,
                   np.zeros(1, dtype=bool), 0, 24)[0]

    [(first, _, stop, later)] = fit.specs
    opened = probatum.backtest(np.ones((stop - first - later, 1)),
                               _piece(inside, first + later, stop), VENUE)

    assert later > 0
    assert study.fits["fitted"][0] == opened.stats["total_return_pct"]


def test_a_path_of_weights_trades_as_one_array_of_its_windows_rows():
    # a seam cancels the stop an inherited long holds and places it again at the same
    # level, so the path's returns are those of one run of its rows
    study  = validated(probatum.cv.PurgedKFold(4))
    inside = _before(swinging(), 480)
    held   = probatum.backtest(composed(study, inside), inside, VENUE, stop=0.02)
    start  = int(np.flatnonzero(~np.isnan(study.returns[0]))[0])

    np.testing.assert_array_equal(study.returns[0, start:], held.returns[start:])
    assert study.paths["state"].tolist() == ["complete"]


def test_each_seam_places_the_inherited_stop_again_and_a_flattening_one_closes():
    study   = validated(probatum.cv.PurgedKFold(4))
    inside  = _before(swinging(), 480)
    fitted  = {row["number"]: Fitted(row["fast"], row["slow"]) for row in study.fits}
    windows = sorted((row["first"] - 1, row["last"] - 1, row["fit"]) for row in
                     study.windows if row["path"] == 0)

    cancel, decided = _weighed(crossing, windows, fitted, "cancel", inside, VENUE, 0.0)
    seams = [event for event in cancel.orders if event["reason"] == "seam"]

    assert seams and {event["kind"] for event in seams} == {"cancelled"}

    # every stop the seams cancelled rests again, priced as the one it replaced
    cancelled = {event["order"] for event in seams}
    stops     = [event for event in cancel.orders
                 if event["kind"] == "placed" and event["reason"] == "stop"]

    assert len(stops) > len(cancelled)
    assert decided == [int(np.isfinite(composed(study, inside)[first:last + 1]).any(
        axis=1).sum()) for first, last, _ in windows]

    flat, _ = _weighed(crossing, windows, fitted, "flatten", inside, VENUE, 0.0)
    closed = [fill for fill in flat.fills if fill["reason"] == "seam"]

    assert closed and {fill["bar"] for fill in closed} <= {last + 1 for _, last, _ in
                                                          windows[:-1]}


def test_a_path_whose_windows_return_another_tolerance_is_refused_naming_them():
    def tolerant(data, fast, slow):
        return {**crossing(data, fast, slow), "tolerance": fast / 1_000.0}

    inside  = _before(swinging(), 480)
    windows = [(30, 199, 0), (200, 478, 1)]

    with pytest.raises(ProbatumValueError, match="policy: the windows from bars 30 and "
                                                 "200 return sizing 'weight' at a "
                                                 "tolerance of 0.003 and sizing 'weight' "
                                                 "at 0.004; return one sizing and one "
                                                 "tolerance for every parameter"):
        _weighed(tolerant, windows, {0: Fitted(3, 10), 1: Fitted(4, 10)}, "cancel",
                 inside, VENUE, 0.0)


def test_a_function_that_returns_a_strategy_validates_as_its_class_does():
    def built(data, fast, slow):
        return Cross(fast=fast, slow=slow)

    cv = probatum.cv.CombinatorialPurged(4, 2)
    by_function, by_class = validated(cv, policy=built), validated(cv, policy=Cross)

    for name in ("paths", "windows"):
        assert getattr(by_function, name) == getattr(by_class, name)
    np.testing.assert_array_equal(by_function.returns, by_class.returns)


def test_every_fit_s_trials_decide_in_the_kind_the_first_search_s_winner_did():
    # a Strategy for an odd fast and weights for an even one, so a fit's trial of the
    # other kind than the first search's winner fails
    def mixed(data, fast, slow):
        return Cross(fast=fast, slow=slow) if fast % 2 else crossing(data, fast, slow)

    study = validated(probatum.cv.PurgedKFold(4), policy=mixed)
    kinds = {fast % 2 for fast in study.fits["fast"].tolist()}

    assert kinds == {study.best_params["fast"] % 2}


def test_a_seeded_validation_of_a_function_gives_in_workers_what_this_process_does():
    pytest.importorskip("cloudpickle")

    cv   = probatum.cv.CombinatorialPurged(4, 2)
    here = validated(cv)
    far  = validated(cv, search={**SEARCH, "n_jobs": 2})

    for name in ("paths", "fits", "windows"):
        assert getattr(here, name) == getattr(far, name)
    np.testing.assert_array_equal(here.returns, far.returns)


def test_a_function_is_refused_a_model_s_arguments_and_labels_end():
    with pytest.raises(ProbatumValueError, match="features: an object of type ndarray "
                                                 "beside a function, which reads its own "
                                                 "Data; leave it out"):
        validated(probatum.cv.PurgedKFold(4), features=np.zeros((600, 1, 2)))

    with pytest.raises(ProbatumValueError, match="which a function has none of; build "
                                                 "it without end"):
        validated(probatum.cv.PurgedKFold(4, end=np.arange(1, 481)))

    with pytest.raises(ProbatumValueError, match="so a function is validated by a "
                                                 "search"):
        validated(probatum.cv.PurgedKFold(4), space=None)
