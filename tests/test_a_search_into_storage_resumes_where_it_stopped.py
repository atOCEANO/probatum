"""A search into storage writes each batch of trials as it arrives and its Study when
it ends, and the same call made again resumes it: every sampler, stopped part way, in
the workers or in Rust, gives the Study one search that never stopped gives, a larger
``trials`` extends a random or TPE search, and its looks carry through. A resume is
refused where a setting differs, where the policy now runs its first trial otherwise,
where the sampler would not ask a stored trial, and where the directory holds no
search.
"""

import json
import math
import multiprocessing
import os
import pathlib
import shutil
import sys

import numpy as np
import pytest

import probatum
from probatum import (ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError,
                      ProbatumWarning, Study)
from reference import statistics
from test_a_search_is_judged_across_its_trials import HOUR, SPACE, alternating, bars
from test_a_search_runs_its_other_trials_in_workers_spawn_starts import Fragile
from test_a_search_runs_its_other_trials_in_workers_spawn_starts import (
    searched as spawned)
from test_a_study_summarises_what_it_found_and_how_it_held_up import (
    raising_on_the_tail)

FLIPS = {"fast": probatum.Int(2, 20), "lag": probatum.Int(0, 4)}
# the trial a search of Flipping stops at, raising as Ctrl-C does, and the weight it
# holds, which a test changes to change the policy's code
STOPS = {"at": None, "made": 0, "weight": 1.0}


class Flipping(probatum.Strategy):
    """Long for ``fast`` bars and flat for as many, from ``lag`` bars in."""


    def __init__(self, fast, lag):
        STOPS["made"] += 1
        if STOPS["made"] == STOPS["at"]:
            raise KeyboardInterrupt

        self.fast, self.lag = fast, lag


    def next(self, bar, account):
        held = (bar.i + self.lag) % (2 * self.fast) < self.fast
        account.target(STOPS["weight"] if held else 0.0)


class Exiting(Fragile):
    """Fragile, its worker giving back a fast of 7 as a trial its policy's exit failed,
    rather than dying."""


    def init(self, data):
        if self.fast == 7 and multiprocessing.current_process().name != "MainProcess":
            sys.exit(3)

        super().init(data)


@pytest.fixture(autouse=True)
def unstopped():
    yield
    STOPS.update(at=None, made=0, weight=1.0)


def searched(sampler="random", **given):
    settings = dict(trials=12, sampler=sampler, holdout=0.3, seed=5)
    settings.update(given)

    return probatum.search(Flipping, FLIPS, bars(), "frictionless", **settings)


def same(found, held):
    """Assert that ``found`` holds the trials, returns and winner ``held`` holds."""
    assert found.trials == held.trials and found.best_params == held.best_params
    assert np.array_equal(found.returns, held.returns, equal_nan=True)


@pytest.mark.parametrize("sampler", ["random", "grid", "tpe"])
def test_a_search_stopped_part_way_resumes_to_the_study_one_run_gives(sampler,
                                                                        tmp_path):
    whole = searched(sampler)
    STOPS.update(made=0, at=9)

    with pytest.raises(KeyboardInterrupt):
        searched(sampler, storage=tmp_path / "study")

    STOPS.update(at=None)

    stored = sorted((tmp_path / "study" / "trials").glob("*.arrow"))
    assert 0 < len(stored) < 12 and not (tmp_path / "study" / "study.json").exists()

    same(searched(sampler, storage=tmp_path / "study"), whole)
    assert not (tmp_path / "study" / "trials").exists()
    same(Study.load(tmp_path / "study"), whole)


def test_a_larger_trials_extends_a_search_in_workers_or_in_rust(tmp_path):
    # random trials in two workers, as at one, and TPE's at the two it ran at
    for sampler in ("random", "tpe"):
        searched(sampler, trials=6, n_jobs=2, storage=tmp_path / sampler)
        found = searched(sampler, trials=10, n_jobs=2, storage=tmp_path / sampler)
        alone = 1 if sampler == "random" else 2
        same(found, searched(sampler, trials=10, n_jobs=alone))

    # weights a random search runs together in Rust
    def run(**given):
        return probatum.search(alternating, SPACE, bars(), "frictionless",
                               sampler="random", holdout=0.3, seed=1, **given)

    run(trials=6, storage=tmp_path / "rust")
    same(run(trials=10, storage=tmp_path / "rust"), run(trials=10))


def test_a_search_in_workers_stopped_part_way_keeps_each_trial_that_came_back(
        tmp_path, monkeypatch):
    whole = searched(trials=10, n_jobs=2)

    wrote = probatum._storage.Storage.wrote

    def stopping(self, trial):
        wrote(self, trial)
        if len(list((self.folder / "trials").glob("*.arrow"))) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(probatum._storage.Storage, "wrote", stopping)

    with pytest.raises(KeyboardInterrupt):
        searched(trials=10, n_jobs=2, storage=tmp_path / "study")

    monkeypatch.undo()

    same(searched(trials=10, n_jobs=2, storage=tmp_path / "study"), whole)


def test_a_seed_of_none_resumes_on_the_seed_the_search_drew(tmp_path):
    first = searched(trials=4, seed=None, storage=tmp_path / "study")
    found = searched(trials=6, seed=None, storage=tmp_path / "study")

    same(found, searched(trials=6, seed=first.provenance["seed"]))


def test_a_resume_runs_only_the_trials_it_does_not_hold(tmp_path, monkeypatch):
    searched(storage=tmp_path / "study")

    ran, trial = [], probatum._search._Runner.trial

    def counted(self, number, params):
        ran.append(number)
        return trial(self, number, params)

    monkeypatch.setattr(probatum._search._Runner, "trial", counted)

    searched(storage=tmp_path / "study")
    searched(trials=14, storage=tmp_path / "study")

    # the first trial each time, run again to compare, and the two the second asked
    assert ran == [0, 0, 12, 13]


def test_a_study_whose_workers_broke_is_kept_out_and_its_failures_run_again(tmp_path):
    space = {"fast": probatum.Int(2, 7), "slow": probatum.Int(12, 12)}

    with pytest.warns(ProbatumWarning, match="the workers broke"):
        broke = spawned(Fragile, space, sampler="grid", n_jobs=2,
                        storage=tmp_path / "study")

    assert broke.trials["state"].tolist()[-1] == "failed"
    assert not (tmp_path / "study" / "study.json").exists()

    # saved anywhere, as loaded or into its storage, it names the trials that failed
    shutil.copytree(tmp_path / "study", tmp_path / "kept")
    broke.save(tmp_path / "kept")
    broke.save(tmp_path / "saved")
    Study.load(tmp_path / "saved").save(tmp_path / "copied")

    for name in ("study", "kept", "saved", "copied"):
        again = spawned(Fragile, space, sampler="grid", storage=tmp_path / name)
        assert again.trials["state"].tolist() == ["complete"] * 6


def test_a_trial_the_workers_breaking_failed_is_no_neighbour_on_the_plateau():
    study = searched(trials=40)
    level = float(np.nanmedian(study.trials["score"]))

    # every odd trial but the winner, as the Study names those a broken pool failed
    study._broken = [number for number in range(1, 40, 2)
                     if number != study._winner.number]

    found  = study.plateau(level)
    names  = list(study.provenance["space"])
    rows   = [row for row in study.trials if row["number"] not in study._broken]
    trials = [(tuple(row[name] for name in names), row["state"], row["score"])
              for row in rows]

    best = [row["number"] for row in rows].index(study._winner.number)
    held = statistics.plateau(trials, [(2, 20, False), (0, 4, False)], best, level)

    assert [None if math.isnan(share) else share for share in found.share] == held


def test_a_stored_study_keeps_its_looks_through_a_resume(tmp_path, monkeypatch):
    folder = tmp_path / "study"
    first  = searched(trials=6, storage=folder)
    first.evaluate_holdout()

    assert Study.load(folder).looks == 1

    again = searched(trials=6, storage=folder)
    assert again.looks == 1 and again.holdout.stats == first.holdout.stats

    # forty trials find another winner, whose holdout is not yet run
    more = searched(trials=40, storage=folder)
    assert more.best_params != first.best_params
    assert (more.looks, more.holdout) == (1, None)

    said = ("read 1 time before, with no result kept for this winner; "
            "evaluate_holdout() spends another look")
    assert more.summary().to_pandas().loc["holdout", "why_none"] == said
    with pytest.raises(ProbatumValueError, match=r"^decay: the holdout was read 1 time "
                       r"before, and no result is kept for this winner; "
                       r"evaluate_holdout\(\) runs it, and spends a look$"):
        more.decay()

    def broken(self, params, data):
        raise RuntimeError("the held-out run broke")

    # the look is written before the winner is handed the held-out bars
    monkeypatch.setattr(probatum._search._Ran, "__call__", broken)

    with pytest.raises(RuntimeError, match="the held-out run broke"):
        more.evaluate_holdout()

    monkeypatch.undo()

    loaded = Study.load(folder)
    assert (loaded.looks, loaded.holdout) == (2, None)


def test_a_holdout_run_on_bars_since_revised_is_not_carried(tmp_path):
    first = searched(trials=6, storage=tmp_path / "study")
    first.evaluate_holdout()

    # the last bar's volume revised, which the bars before the holdout do not hold
    shaped = bars().bars[:, 0].copy()
    shaped[-1, 4] += 1.0
    revised = probatum.Data(
        shaped,
        symbol = "BTCUSDT",
        market = "linear",
        time   = bars().time,
    )

    again = probatum.search(Flipping, FLIPS, revised, "frictionless", trials=6,
                            sampler="random", holdout=0.3, seed=5,
                            storage=tmp_path / "study")

    assert again.best_params == first.best_params
    assert (again.looks, again.holdout) == (1, None)


def test_a_holdout_is_written_only_where_its_own_study_is(tmp_path):
    first = searched(trials=6)
    first.save(tmp_path / "study")

    # the same search of bars whose held-out tail was revised, saved over it
    shaped = bars().bars[:, 0].copy()
    shaped[-1, 4] += 1.0
    revised = probatum.Data(shaped, symbol="BTCUSDT", market="linear", time=bars().time)

    probatum.search(Flipping, FLIPS, revised, "frictionless", trials=6,
                    sampler="random", holdout=0.3, seed=5).save(tmp_path / "study")

    first.evaluate_holdout()

    loaded = Study.load(tmp_path / "study")
    assert (loaded.looks, loaded.holdout) == (1, None)


def test_a_look_outlasts_a_final_save_stopped_before_its_manifest(tmp_path,
                                                                 monkeypatch):
    folder = tmp_path / "study"
    searched(trials=6, storage=folder).evaluate_holdout()

    replace = os.replace

    def stopped(source, target):
        if pathlib.Path(target).name == "study.json":
            raise OSError("the disk is full")
        replace(source, target)

    monkeypatch.setattr(os, "replace", stopped)

    with pytest.raises(OSError, match="the disk is full"):
        searched(trials=8, storage=folder)

    monkeypatch.undo()

    assert not (folder / "study.json").exists()
    assert searched(trials=8, storage=folder).looks == 1


def test_a_study_whose_workers_broke_still_counts_its_looks_there(tmp_path):
    space = {"fast": probatum.Int(2, 7), "slow": probatum.Int(12, 12)}
    spawned(Fragile, space, trials=3, storage=tmp_path / "study").evaluate_holdout()

    with pytest.warns(ProbatumWarning, match="the workers broke"):
        broke = spawned(Fragile, space, trials=8, n_jobs=2, storage=tmp_path / "study")

    # the winner is the first search's, whose holdout it holds and spends no look on
    assert broke.looks == 1 and broke.holdout is not None
    assert broke.evaluate_holdout() is broke.holdout and broke.looks == 1


def test_each_study_of_one_search_counts_its_look_where_the_search_is_kept(tmp_path):
    folder = tmp_path / "study"
    first  = searched(trials=6, storage=folder)
    second = searched(trials=6, storage=folder)

    first.evaluate_holdout()
    second.evaluate_holdout()

    assert (first.looks, second.looks, Study.load(folder).looks) == (1, 2, 2)

    # a look by a Study of fewer trials than the directory's leaves those files whole
    small = searched(trials=6, storage=tmp_path / "other")
    searched(trials=9, storage=tmp_path / "other")

    small.evaluate_holdout()

    loaded = Study.load(tmp_path / "other")
    assert (len(loaded.trials), loaded.looks, loaded.holdout) == (9, 1, None)


@pytest.mark.parametrize("error", [
    ProbatumRuntimeError("best: trial 0's sharpe was 1.0 on the in-sample bars, and "
                         "run again it is 2.0; set PYTHONHASHSEED"),
    ValueError("best: trial 0's sharpe was 1.0, and its policy raised run again"),
])
def test_a_winner_that_runs_otherwise_is_returned_and_not_kept(tmp_path, monkeypatch,
                                                              error):
    def otherwise(self):
        raise error

    monkeypatch.setattr(probatum._search.Study, "best", property(otherwise))

    said = (r"^storage: '[^']+' keeps this search's trials but not its Study, since "
            r"best: trial 0's sharpe was 1.0")
    with pytest.warns(ProbatumWarning, match=said):
        study = searched(storage=tmp_path / "study")

    assert len(study.trials) == 12 and not (tmp_path / "study" / "study.json").exists()


def test_a_holdout_a_write_stopped_part_way_leaves_no_manifest_naming_it(tmp_path,
                                                                         monkeypatch):
    folder = tmp_path / "study"
    first  = searched(trials=6, storage=folder)
    second = searched(trials=6, storage=folder)

    first.evaluate_holdout()

    replace = os.replace

    def stopped(source, target):
        if pathlib.Path(target).name == "holdout":
            raise OSError("the disk is full")
        replace(source, target)

    monkeypatch.setattr(os, "replace", stopped)

    with pytest.raises(OSError, match="the disk is full"):
        second.evaluate_holdout()

    monkeypatch.undo()

    loaded = Study.load(folder)
    assert (loaded.looks, loaded.holdout) == (2, None)
    # the run kept no result there, and did not raise
    assert "raised" not in loaded.summary().to_pandas().loc["holdout", "why_none"]
    assert searched(trials=6, storage=folder).looks == 2


def test_a_loaded_study_says_its_run_raised_only_where_it_saw_it_raise(tmp_path):
    def raised(**given):
        return probatum.search(raising_on_the_tail, {"fast": probatum.Int(2, 20)},
                               bars(), "frictionless", trials=4, sampler="random",
                               holdout=0.3, seed=1, **given)

    study = raised()

    with pytest.raises(RuntimeError, match="the tail"):
        study.evaluate_holdout()

    study.save(tmp_path / "saved")
    loaded = Study.load(tmp_path / "saved")
    said   = "read 1 time; its run raised, so no result is kept"

    assert loaded.summary() == study.summary()
    assert loaded.summary().to_pandas().loc["holdout", "why_none"] == said

    # a look taken once the Study was kept is counted there with no result
    kept = raised(storage=tmp_path / "kept")

    with pytest.raises(RuntimeError, match="the tail"):
        kept.evaluate_holdout()

    said = ("read 1 time before, with no result kept for this winner; "
            "evaluate_holdout() spends another look")
    assert Study.load(tmp_path / "kept").summary().to_pandas().loc[
        "holdout", "why_none"] == said


def test_a_trial_its_policy_exits_in_a_worker_is_written_as_it_comes(tmp_path,
                                                                     monkeypatch):
    written, wrote = [], probatum._storage.Storage.wrote

    def kept(self, trial):
        written.append((trial.state, trial.error))
        wrote(self, trial)

    monkeypatch.setattr(probatum._storage.Storage, "wrote", kept)

    spawned(Exiting, {"fast": probatum.Int(2, 7), "slow": probatum.Int(12, 12)},
            sampler="grid", n_jobs=2, storage=tmp_path / "study")

    assert ("failed", "SystemExit: 3") in written


def test_settings_a_kill_left_part_written_are_no_bar_to_the_next_search(tmp_path):
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "search.json.partial").write_text("{", encoding="utf-8")

    same(searched(storage=tmp_path / "study"), searched())


def test_a_resume_on_other_bars_is_refused_naming_the_data(tmp_path):
    searched(trials=4, storage=tmp_path / "study")

    # the first close revised, so the same rows are held out
    revised = bars().bars[:, 0].copy()
    revised[0, 3] += 0.25
    data = probatum.Data(revised, symbol="BTCUSDT", market="linear", time=bars().time)

    with pytest.raises(ProbatumValueError, match=r"^storage: '[^']+' holds a search on "
                       r"other bars than this one's data holds before its holdout; "
                       r"resume it on the data it ran on, or search into another "
                       r"directory$"):
        probatum.search(Flipping, FLIPS, data, "frictionless", trials=4,
                        sampler="random", holdout=0.3, seed=5,
                        storage=tmp_path / "study")


def edited(folder, change):
    """Make ``change`` to the settings the search kept in ``folder`` wrote, as a hand
    would, and take its Study's manifest away, which is read before them."""
    started = json.loads((folder / "search.json").read_text("utf-8"))
    change(started)
    (folder / "search.json").write_text(json.dumps(started), "utf-8")
    (folder / "study.json").unlink()


def test_held_out_rows_and_starts_and_an_optuna_that_differ_are_named_as_such(
        tmp_path):
    # a holdout given as the time of row 560 holds out more rows of longer bars, the
    # bars before it the same
    start = str(np.datetime64(int(bars().time[560]), "ns"))
    probatum.search(Flipping, FLIPS, bars(), "frictionless", trials=4, sampler="random",
                    holdout=start, seed=5, storage=tmp_path / "study")

    with pytest.raises(ProbatumValueError, match=r"^storage: '[^']+' holds a search "
                       r"that held out 240 rows, and this one's data holds 340 there; "
                       r"resume it on the data it ran on, or search into another "
                       r"directory$"):
        probatum.search(Flipping, FLIPS, bars(rows=900), "frictionless", trials=4,
                        sampler="random", holdout=start, seed=5,
                        storage=tmp_path / "study")

    # the same 300 rows held out of bars an hour later
    searched(trials=4, holdout=300, storage=tmp_path / "moved")
    later = probatum.Data(
        bars().bars[:, 0],
        symbol = "BTCUSDT",
        market = "linear",
        time   = bars().time + HOUR,
    )

    with pytest.raises(ProbatumValueError, match=r"^storage: '[^']+' holds a search "
                       r"whose held-out bars start at 1970-01-21 20:00, and this one's "
                       r"start at 1970-01-21 21:00; resume it on the data it ran on, "
                       r"or search into another directory$"):
        probatum.search(Flipping, FLIPS, later, "frictionless", trials=4,
                        sampler="random", holdout=300, seed=5,
                        storage=tmp_path / "moved")

    searched("tpe", trials=4, storage=tmp_path / "tpe")
    edited(tmp_path / "tpe",
           lambda started: started["settings"]["sampler"].update(optuna="4.9.0"))

    with pytest.raises(ProbatumValueError, match=r"^storage: '[^']+' holds a TPE "
                       r"search Optuna 4.9.0 ran, and this is Optuna \S+; resume it "
                       r"with that Optuna, or search into another directory$"):
        searched("tpe", trials=4, storage=tmp_path / "tpe")


def test_another_version_is_named_before_the_settings_it_may_name_otherwise(tmp_path):
    said = (r"^storage: '[^']+' holds a search probatum 0.0.1 ran, and this is "
            r"probatum \S+; resume it with that version, or search into another "
            r"directory$")

    def older(held):
        held.update(version="0.0.1")
        del held["venue"]

    searched(trials=4, storage=tmp_path / "settings")
    edited(tmp_path / "settings", lambda started: older(started["settings"]))

    with pytest.raises(ProbatumValueError, match=said):
        searched(trials=4, storage=tmp_path / "settings")

    # a Study's provenance, read before the settings beside it
    searched(trials=4, storage=tmp_path / "study")
    kept = json.loads((tmp_path / "study" / "study.json").read_text("utf-8"))
    older(kept["provenance"])
    (tmp_path / "study" / "study.json").write_text(json.dumps(kept), "utf-8")

    with pytest.raises(ProbatumValueError, match=said):
        searched(trials=4, storage=tmp_path / "study")


@pytest.mark.parametrize("sampler, change", [
    ("random", lambda started: started["settings"].update(holdout=0.3)),
    ("random", lambda started: started["settings"].update(sampler="random")),
    ("random", lambda started: started["settings"].update(seed="5")),
    ("random", lambda started: started.pop("workers")),
    ("tpe", lambda started: started.update(workers=None)),
])
def test_settings_edited_by_hand_are_refused_as_no_search_s(tmp_path, sampler, change):
    searched(sampler, trials=4, storage=tmp_path / "study")
    edited(tmp_path / "study", change)

    with pytest.raises(ProbatumValueError, match=r"^storage: '[^']+''s search.json is "
                       r"not the settings a search writes; give the directory a search "
                       r"kept its Study in, or an empty one$"):
        searched(sampler, trials=4, storage=tmp_path / "study")


def test_a_study_s_provenance_edited_by_hand_is_refused_as_no_manifest(tmp_path):
    searched(trials=4, storage=tmp_path / "study")

    kept = json.loads((tmp_path / "study" / "study.json").read_text("utf-8"))
    del kept["provenance"]["venue"]
    (tmp_path / "study" / "study.json").write_text(json.dumps(kept), "utf-8")

    with pytest.raises(ProbatumValueError, match=r"^storage: '[^']+''s study.json is "
                       r"not a manifest save writes; give the directory a search kept "
                       r"its Study in, or an empty one$"):
        searched(trials=4, storage=tmp_path / "study")


def test_a_directory_a_save_stopped_in_is_refused_naming_it(tmp_path, monkeypatch):
    study, replace = searched(trials=4), os.replace

    def stopped(source, target):
        if pathlib.Path(target).name == "study.json":
            raise OSError("the disk is full")
        replace(source, target)

    monkeypatch.setattr(os, "replace", stopped)

    with pytest.raises(OSError, match="the disk is full"):
        study.save(tmp_path / "study")

    monkeypatch.undo()

    with pytest.raises(ProbatumValueError, match=r"^storage: '[^']+' holds a Study "
                       r"whose save stopped part way; save it there again, or search "
                       r"into another directory$"):
        searched(trials=4, storage=tmp_path / "study")


def test_the_study_of_one_search_beside_the_settings_of_another_is_refused(tmp_path):
    STOPS.update(at=9)

    with pytest.raises(KeyboardInterrupt):
        searched(storage=tmp_path / "study")

    STOPS.update(at=None)

    searched(objective="sortino").save(tmp_path / "other")
    for name in ("study.json", "trials.arrow", "returns.npy"):
        shutil.copy(tmp_path / "other" / name, tmp_path / "study" / name)
    shutil.copytree(tmp_path / "other" / "best", tmp_path / "study" / "best")

    said = (r"^storage: '[^']+' holds the Study of one search and the settings of "
            r"another; give another directory$")
    with pytest.raises(ProbatumValueError, match=said):
        searched(objective="sortino", storage=tmp_path / "study")


def test_a_changed_policy_is_refused_before_a_trial_runs(tmp_path):
    searched(trials=4, storage=tmp_path / "study")

    STOPS.update(weight=0.5)

    said = (r"^storage: trial 0's \w+ was [^,]+ in '[^']+', and run again it is [^;]+; "
            r"the policy's code or the globals it reads changed since it ran, so "
            r"search into another directory$")
    with pytest.raises(ProbatumRuntimeError, match=said):
        searched(trials=8, storage=tmp_path / "study")


def test_a_trial_the_sampler_would_not_ask_is_refused(tmp_path):
    # a grid of 95 points with 12 trials draws them, and with 20 draws others
    STOPS.update(at=9)

    with pytest.raises(KeyboardInterrupt):
        searched("grid", storage=tmp_path / "study")

    STOPS.update(at=None)

    said = (r"^storage: trial \d+ in '[^']+' ran with \{[^}]+\}, and this search asks "
            r"\{[^}]+\} for it, so it is not the search that ran there; search into "
            r"another directory$")
    with pytest.raises(ProbatumValueError, match=said):
        searched("grid", trials=20, storage=tmp_path / "study")


@pytest.mark.parametrize("given, error, said", [
    (dict(objective="sortino"), ProbatumValueError,
     r"^storage: '[^']+' holds a search whose objective is 'sharpe', and this one's is "
     r"'sortino'; search with the objective it ran with to resume it, or into another "
     r"directory$"),
    (dict(seed=6), ProbatumValueError,
     r"^storage: '[^']+' holds a search whose seed is 5, and this one's is 6; search "
     r"with the seed it ran with to resume it, or into another directory$"),
    (dict(trials=3), ProbatumValueError,
     r"^trials: 3, and '[^']+' holds this search's trials up to trial 3; ask for at "
     r"least 4$"),
])
def test_a_resume_of_another_search_is_refused(tmp_path, given, error, said):
    searched(trials=4, storage=tmp_path / "study")

    with pytest.raises(error, match=said):
        searched(storage=tmp_path / "study", **{"trials": 4, **given})


def test_a_tpe_search_resumes_only_at_the_workers_it_ran_at(tmp_path):
    searched("tpe", trials=4, n_jobs=2, storage=tmp_path / "study")

    with pytest.raises(ProbatumValueError, match=r"^n_jobs: 1, which runs 1 at a time, "
                       r"and the TPE search in '[^']+' asked its trials 2 at a time; "
                       r"resume it at that many, as TPE repeats only at the number of "
                       r"workers it ran at, or search into another directory$"):
        searched("tpe", trials=6, storage=tmp_path / "study")


def test_a_directory_that_holds_no_search_is_refused(tmp_path):
    (tmp_path / "file").write_text("", encoding="utf-8")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.txt").write_text("", encoding="utf-8")

    paths = probatum.validate(alternating, bars(), "frictionless",
                              probatum.cv.WalkForward(200, 100), space=SPACE,
                              search=dict(trials=4, sampler="random"), holdout=0.3,
                              seed=1)
    paths.save(tmp_path / "paths")

    for folder, error, said in [
            (5, ProbatumTypeError,
             r"^storage: 5; give the directory a search keeps its Study in, as a "
             r"path$"),
            (tmp_path / "file", ProbatumValueError,
             r"^storage: '[^']+' is a file; name a directory$"),
            (tmp_path / "other", ProbatumValueError,
             r"^storage: '[^']+' holds files that are no search's; give an empty "
             r"directory, or the one a search kept its Study in$"),
            (tmp_path / "paths", ProbatumValueError,
             r"^storage: '[^']+' holds a validation's Study, which a search does not "
             r"resume; give another directory$")]:
        with pytest.raises(error, match=said):
            searched(storage=folder)


def test_a_search_writes_its_settings_before_any_trial_runs(tmp_path):
    STOPS.update(at=1)

    with pytest.raises(KeyboardInterrupt):
        searched(storage=tmp_path / "study")

    folder  = tmp_path / "study"
    started = json.loads((folder / "search.json").read_text(encoding="utf-8"))

    assert started["settings"]["seed"] == 5 and not (folder / "trials").exists()
