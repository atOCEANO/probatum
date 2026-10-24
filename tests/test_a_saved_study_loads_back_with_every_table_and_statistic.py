"""A Study is saved into a directory of open files, as a Result is, and loads back with
every table, statistic and summary it had: a search's trials, each trial's choices and
their returns, a validation's paths, fits, windows and forecast, the winner's Results
and the looks. A loaded Study holds no policy and no data, so it refuses a holdout it
never ran before any look, and load refuses what save did not write.
"""

import json
import math
import os
import pathlib
import shutil

import numpy as np
import pytest

import probatum
from probatum import ProbatumError, ProbatumValueError, Study
from test_a_search_is_judged_across_its_trials import SPACE, alternating, bars
from test_a_validation_of_a_model_fits_each_split_afresh import (CUT, CV, DATA, LABELS,
                                                                  VENUE, features,
                                                                  weights)

# a None among floats, which the trials table holds as NaN
STOPPED = {
    "lag":  probatum.Int(0, 9),
    "stop": probatum.Categorical([None, 0.02, 0.05]),
}


def stopped(data, lag, stop):
    return {"policy": alternating(data, 5, lag), "stop": stop}


def later():
    """The search tests' bars on a clock that starts in 2024, so a span read back as
    starting in 1970 shows in a summary."""
    data = bars()
    return probatum.Data(
        data.bars[:, 0],
        symbol = "BTCUSDT",
        market = "linear",
        time   = data.time + 1_704_067_200 * 10**9,
    )


def searched(**given):
    settings = dict(trials=12, sampler="random", holdout=0.3, seed=3)
    settings.update(given)
    return probatum.search(stopped, STOPPED, later(), "frictionless", **settings)


def read(how, study):
    """What ``how`` reads of ``study``, or the refusal it gives."""
    try:
        return how(study)
    except ProbatumError as error:
        return type(error), str(error)


class Plain:
    """A model that predicts each row's first feature."""


    def get_params(self, deep=True):
        return {}


    def set_params(self, **params):
        return self


    def fit(self, x, y, sample_weight=None):
        return self


    def predict(self, x):
        return x[:, 0].copy()


def test_a_search_loads_back_with_its_trials_choices_returns_and_statistics(tmp_path):
    study = searched()
    study.evaluate_holdout()
    study.save(tmp_path / "study")
    loaded = Study.load(tmp_path / "study")

    assert loaded.trials == study.trials and loaded._params == study._params
    assert np.array_equal(loaded.returns, study.returns, equal_nan=True)
    assert not loaded.returns.flags.writeable
    assert loaded.provenance == json.loads(json.dumps(study.provenance))
    assert (loaded.best_params, loaded.best_score, loaded.looks) == (
        study.best_params, study.best_score, 1)
    assert loaded.best.stats == study.best.stats
    assert loaded.evaluate_holdout() is loaded.holdout and loaded.looks == 1
    assert loaded.holdout.stats == study.holdout.stats

    null = searched(seed=9)

    for how in (repr, lambda s: s.summary(null=null, plateau=0.0),
                lambda s: s.deflated_sharpe(null), lambda s: s.effective_trials(),
                lambda s: s.overfitting(),
                lambda s: s.reality_check(resamples=50, seed=1),
                lambda s: s.min_backtest_length(), lambda s: s.plateau(0.0),
                lambda s: s.consistency(), lambda s: s.decay()):
        assert read(how, loaded) == read(how, study)

    kept = searched(keep="none")
    kept.save(tmp_path / "kept")

    assert Study.load(tmp_path / "kept").returns is None


def test_a_validation_loads_back_with_its_paths_fits_windows_and_forecast(tmp_path):
    paths = probatum.validate(alternating, bars(), "frictionless",
                              probatum.cv.WalkForward(200, 100), space=SPACE,
                              search=dict(trials=4, sampler="random"), holdout=0.3,
                              seed=1)
    model = probatum.validate(Plain(), DATA, VENUE, cv=CV, features=features,
                              labels=LABELS, to_weights=weights,
                              holdout=len(DATA) - CUT)

    searched().save(tmp_path / "study")
    for study in (paths, model):
        # into a directory a search was saved in, which keeps none of its files
        study.save(tmp_path / "study")
        loaded = Study.load(tmp_path / "study")

        assert not (tmp_path / "study" / "trials.arrow").exists()
        assert (loaded.paths, loaded.fits, loaded.windows) == (study.paths, study.fits,
                                                               study.windows)
        assert np.array_equal(loaded.returns, study.returns, equal_nan=True)
        assert (loaded.forecast, loaded.best_params, loaded.best_score) == (
            study.forecast, study.best_params, study.best_score)
        assert (repr(loaded), loaded.summary()) == (repr(study), study.summary())

    assert loaded.forecast is not None


def test_a_model_s_fill_given_as_a_numpy_number_is_saved_as_a_number(tmp_path):
    model = probatum.validate(Plain(), DATA, VENUE, cv=CV, features=features,
                              labels=LABELS, to_weights=weights,
                              holdout=len(DATA) - CUT, fill=np.float32(0.25))
    model.save(tmp_path / "study")

    assert Study.load(tmp_path / "study").provenance["model"]["fill"] == 0.25


def test_a_loaded_study_refuses_a_holdout_it_never_ran_before_any_look(tmp_path):
    searched().save(tmp_path / "study")
    loaded = Study.load(tmp_path / "study")

    with pytest.raises(ProbatumValueError, match=r"^evaluate_holdout: a Study loaded "
                       r"with no holdout run, and no policy or data to run its winner "
                       r"on the held-out bars; search again with storage= the "
                       r"directory it was saved in, which runs only what is not "
                       r"there$"):
        loaded.evaluate_holdout()

    assert (loaded.looks, loaded.holdout) == (0, None)

    with pytest.raises(ProbatumValueError, match=r"^decay: a Study loaded with no "
                       r"holdout run, and no policy or data to run its winner on the "
                       r"held-out bars; search again with storage= the directory it "
                       r"was saved in, which runs only what is not there, then "
                       r"evaluate its holdout$"):
        loaded.decay()


def test_a_provenance_holding_a_float_json_has_no_number_for_loads_back(tmp_path):
    # the checked trial's parameters hold the infinite cap, as the provenance does
    def capped(data, lag, cap):
        return np.minimum(alternating(data, 5, lag), cap)

    study = probatum.search(capped, {"lag": probatum.Int(0, 9),
                                     "cap": probatum.Categorical([math.inf])},
                            later(), "frictionless", trials=4, sampler="random",
                            holdout=0.3, seed=3)
    study.save(tmp_path / "study")
    loaded = Study.load(tmp_path / "study")

    assert loaded.provenance["checked"][0]["cap"] == math.inf
    assert loaded.trials == study.trials


def test_a_save_into_a_directory_that_holds_no_study_is_refused(tmp_path):
    (tmp_path / "work" / "holdout").mkdir(parents=True)
    (tmp_path / "work" / "notes.txt").write_text("", encoding="utf-8")

    with pytest.raises(ProbatumValueError, match=r"^path: '[^']+' holds files that are "
                       r"no Study's; give an empty directory, or one a Study was saved "
                       r"into$"):
        searched().save(tmp_path / "work")

    assert (tmp_path / "work" / "holdout").is_dir()

    # another program's study.json, or study.json.old beside a file of its own
    for name in ("study.json", "study.json.old"):
        folder = tmp_path / "other" / name.replace(".", "-")
        folder.mkdir(parents=True)
        (folder / name).write_text("{}", encoding="utf-8")
        (folder / "returns.npy").write_bytes(b"its own")

        with pytest.raises(ProbatumValueError, match=rf"^path: '[^']+' holds a {name} "
                           rf"save did not write; give another directory$"):
            searched().save(folder)

        assert (folder / "returns.npy").read_bytes() == b"its own"


def test_a_save_that_stops_leaves_the_study_that_was_there(tmp_path, monkeypatch):
    first = searched()
    first.save(tmp_path / "study")

    def full(path, columns, metadata):
        raise OSError("the disk is full")

    monkeypatch.setattr(probatum._ipc, "write", full)

    with pytest.raises(OSError, match="the disk is full"):
        searched(seed=4).save(tmp_path / "study")
    monkeypatch.undo()

    assert Study.load(tmp_path / "study").trials == first.trials
    assert not list((tmp_path / "study").glob("*.partial"))


@pytest.mark.parametrize("first", [True, False])
def test_a_save_stopped_part_way_is_saved_again_where_it_stopped(tmp_path, monkeypatch,
                                                                 first):
    # a first save into an empty directory, or one into a Study's
    study = searched(trials=6)
    if not first:
        study.save(tmp_path / "study")

    replace = os.replace

    def stopped(source, target):
        if pathlib.Path(target).name == "study.json":
            raise OSError("the disk is full")
        replace(source, target)

    monkeypatch.setattr(os, "replace", stopped)

    with pytest.raises(OSError, match="the disk is full"):
        study.save(tmp_path / "study")
    monkeypatch.undo()

    with pytest.raises(ProbatumValueError, match=r"^path: '[^']+' holds a Study whose "
                       r"save stopped part way; save it there again, or give another "
                       r"directory$"):
        Study.load(tmp_path / "study")

    study.save(tmp_path / "study")

    assert Study.load(tmp_path / "study").trials == study.trials
    assert not (tmp_path / "study" / "study.json.old").exists()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    folder = tmp_path_factory.mktemp("saved") / "study"
    searched().save(folder)
    return folder


def edited(saved, tmp_path, change):
    """A copy of the Study ``saved`` with ``change`` made to its directory."""
    folder = tmp_path / "study"
    shutil.copytree(saved, folder)
    change(folder)
    return folder


def manifest(change):
    def changed(folder):
        held = json.loads((folder / "study.json").read_text(encoding="utf-8"))
        change(held)
        (folder / "study.json").write_text(json.dumps(held), encoding="utf-8")

    return changed


def chosen(folder):
    probatum._ipc.write(folder / "choices.arrow",
                        [("stop", "int64", np.full(12, 3), None, False)], {})


def short(folder):
    probatum._ipc.write(folder / "choices.arrow",
                        [("stop", "int64", np.zeros(5, dtype=np.int64), None, False)],
                        {})


@pytest.mark.parametrize("change, said", [
    (lambda folder: (folder / "study.json").unlink(),
     r" holds no study.json; give a directory a Study was saved into$"),
    (manifest(lambda held: held.update(format=2)),
     r" holds a Study saved in format 2 by probatum \S+, and this version, \S+, reads "
     r"format 1; load it with the version that saved it, or a later one$"),
    (manifest(lambda held: held.pop("bars")),
     r"'s study.json is not a manifest save writes; give a directory a Study was saved "
     r"into$"),
    (manifest(lambda held: held["provenance"].pop("keep")),
     r"'s study.json is not a manifest save writes; give a directory a Study was saved "
     r"into$"),
    (manifest(lambda held: held.update(winner=held["winner"] + 1)),
     r"'s study.json names trial \d+ its winner, and its trials make trial \d+ the "
     r"winner; give a directory a Study was saved into$"),
    (lambda folder: np.save(folder / "returns.npy", np.zeros((11, 560))),
     r"'s returns.npy does not hold a row of returns for each row of its table; give a "
     r"directory a Study was saved into$"),
    (chosen,
     r"'s choices.arrow does not hold an index among stop's choices for each trial; "
     r"give a directory a Study was saved into$"),
    (short,
     r"'s choices.arrow does not hold an index among stop's choices for each trial; "
     r"give a directory a Study was saved into$"),
    (lambda folder: shutil.rmtree(folder / "best"),
     r" holds no best, the winner's Result; give a directory a Study was saved "
     r"into$"),
])
def test_load_refuses_what_save_did_not_write(saved, tmp_path, change, said):
    with pytest.raises(ProbatumValueError, match=r"^path: '[^']+'" + said):
        Study.load(edited(saved, tmp_path, change))


def test_a_save_into_a_file_is_refused(tmp_path):
    (tmp_path / "file").write_text("", encoding="utf-8")

    with pytest.raises(ProbatumValueError, match=r"^path: '[^']+' is a file; name a "
                       r"directory$"):
        searched().save(tmp_path / "file")
