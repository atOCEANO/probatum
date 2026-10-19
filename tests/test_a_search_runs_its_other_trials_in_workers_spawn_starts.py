"""A search runs the trials that do not run together in Rust, a Strategy's or TPE's, in
workers spawn starts from a folder it writes, passing them its path alone: a random
search gives the trials it gives in one process, a seeded TPE search repeats at the
same workers, a worker's warnings come back once a place under their categories and
from their modules, a pool that breaks fails only the trials in flight, a script with no
main guard is told to add one, and a winner a worker scored that runs otherwise here
says where it ran.
"""

import multiprocessing
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import threading
import warnings

import numpy as np
import pytest

import probatum
import probatum._search
import probatum._workers
from probatum import (Float, Int, MissingDependency, PolicyError, ProbatumRuntimeError,
                      ProbatumWarning, Strategy)
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE, one
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross

SPACE = {"fast": Int(2, 8), "slow": Int(10, 20)}
TESTS = pathlib.Path(__file__).resolve().parent


def searched(policy=Cross, space=SPACE, **given):
    return probatum.search(policy, space, one(), VENUE, **{
        **dict(sampler="random", trials=8, holdout=0.25, seed=3), **given})


def in_a_worker():
    return multiprocessing.current_process().name != "MainProcess"


def test_workers_give_a_random_search_the_trials_one_process_gives():
    alone, pooled = searched(), searched(n_jobs=2)

    assert pooled.trials == alone.trials
    np.testing.assert_array_equal(pooled.returns, alone.returns)
    assert pooled.provenance["n_jobs"] == {"given": 2, "resolved": 2}
    assert alone.provenance["n_jobs"]  == {"given": 1, "resolved": 1}


def test_a_seeded_tpe_search_repeats_at_the_same_workers():
    first, second = (searched(sampler="tpe", trials=14, n_jobs=2) for _ in range(2))

    assert first.trials == second.trials


class Told(UserWarning):
    pass


class Drift(UserWarning):
    """A warning that cannot be built from its text alone."""


    def __init__(self, text, size):
        super().__init__(text)
        self.size = size


class Warned(Strategy):
    """Cross that warns in a worker, once a run under ``kind``: its module's class, one
    of its own nobody can import, or one that needs more than its text; and with
    "every", at every bar."""


    def __init__(self, fast=3, slow=12, kind="told"):
        self.fast, self.slow, self.kind = fast, slow, kind


    def init(self, data):
        self.cross = Cross(self.fast, self.slow)
        self.cross.init(data)
        self.warmup = self.cross.warmup

        if in_a_worker() and self.kind != "every":
            class Own(UserWarning):
                pass

            if self.kind == "drift":
                warnings.warn(Drift(f"fast {self.fast}", 3))
            else:
                warnings.warn(f"fast {self.fast}", Told if self.kind == "told" else Own)


    def next(self, bar, account):
        if in_a_worker() and self.kind == "every":
            warnings.warn("every bar", Told)

        self.cross.next(bar, account)


def test_a_worker_s_warnings_come_back_under_their_category_or_named():
    with pytest.warns(Told, match="^fast "):
        searched(Warned, dict(SPACE, kind=probatum.Categorical(["told"])), n_jobs=2)

    with pytest.warns(ProbatumWarning, match=r"^test_a_search_runs_its_other_trials_in_"
                                             r"workers_spawn_starts\.Warned\.init\."
                                             r"<locals>\.Own: fast "):
        searched(Warned, dict(SPACE, kind=probatum.Categorical(["own"])), n_jobs=2)


def test_a_worker_s_warning_the_caller_made_an_error_fails_its_trial_alone():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study = searched(Warned, dict(SPACE, kind=probatum.Categorical(["told"])),
                         n_jobs=2)

    states = study.trials.state.tolist()
    assert states[0] == "complete" and set(states[1:]) == {"failed"}
    assert set(study.trials.error.tolist()[1:]) == {
        f"Told: fast {fast}" for fast in study.trials.fast.tolist()[1:]}


def test_a_worker_s_warning_meets_the_filters_set_for_the_module_it_came_from():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=Told, module=__name__)
        study = searched(Warned, dict(SPACE, kind=probatum.Categorical(["told"])),
                         n_jobs=2)

    states = study.trials.state.tolist()
    assert states[0] == "complete" and set(states[1:]) == {"failed"}


def test_a_worker_s_warnings_come_back_once_a_place_built_or_named():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study = searched(Warned, dict(SPACE, kind=probatum.Categorical(["every"])),
                         n_jobs=2)

    # every bar of seven trials warned from one place, which comes back once a trial
    assert len([given for given in caught if given.category is Told]) == len(
        study.trials) - 1

    with pytest.warns(ProbatumWarning, match=rf"^{__name__}\.Drift: fast "):
        study = searched(Warned, dict(SPACE, kind=probatum.Categorical(["drift"])),
                         n_jobs=2)

    assert set(study.trials.state.tolist()) == {"complete"}


class Fragile(Strategy):
    """Cross that kills the worker it runs in for a ``fast`` of 7."""


    def __init__(self, fast=3, slow=12):
        self.fast, self.slow = fast, slow


    def init(self, data):
        if self.fast == 7 and in_a_worker():
            os._exit(1)

        self.cross = Cross(self.fast, self.slow)
        self.cross.init(data)
        self.warmup = self.cross.warmup


    def next(self, bar, account):
        self.cross.next(bar, account)


def test_a_pool_that_breaks_after_a_trial_came_back_fails_the_trials_in_flight():
    space = {"fast": Int(2, 7), "slow": Int(12, 12)}

    with pytest.warns(ProbatumWarning, match=r"^n_jobs: the workers broke after \d+ "
                                             r"trials came back \(BrokenProcessPool: "
                      ) as caught:
        study = searched(Fragile, space, sampler="grid", n_jobs=2)

    rows = list(study.trials)
    assert rows[-1]["state"] == "failed" and rows[-1]["error"].startswith(
        "BrokenProcessPool: ")
    assert rows[0]["state"] == "complete" and len(rows) == 6

    [said] = [str(given.message) for given in caught
              if "workers broke" in str(given.message)]
    came = int(re.search(r"after (\d+) trials", said)[1])
    assert came >= 1 and came == sum(row["state"] == "complete" for row in rows[1:5])


def test_a_pool_that_breaks_with_no_trial_complete_says_the_workers_died(monkeypatch):
    # where spawn runs the script again, as under pytest's own script and not under
    # python -m pytest, the workers came up all the same
    monkeypatch.setattr(probatum._workers, "_reruns_main", lambda: True)
    space = {"fast": Int(2, 7), "slow": Int(12, 12)}

    with pytest.raises(ProbatumRuntimeError) as caught:
        searched(Fragile, space, sampler="grid", n_jobs=2, min_trades=10**9)

    assert str(caught.value).startswith("n_jobs: 2, and the workers died with no trial "
                                        "complete (BrokenProcessPool: ")


def test_a_broken_pool_s_refusal_names_the_n_jobs_given(tmp_path):
    pool = probatum._workers._Pool(tmp_path, 3, -1)

    try:
        pool.broken, pool.returned = RuntimeError("gone"), 1
        assert str(pool.refused()) == ("n_jobs: -1 (3 workers), and the workers died "
                                       "with no trial complete (RuntimeError: gone); "
                                       "their output says why")
    finally:
        pool.close()


def test_a_script_that_calls_search_outside_a_main_guard_is_told_to_add_one(tmp_path):
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(TESTS)!r})
        import probatum
        from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
            VENUE, one)
        from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import (
            Cross)
        probatum.search(Cross, {{"fast": probatum.Int(2, 8),
                                 "slow": probatum.Int(10, 20)}},
                        one(), VENUE, sampler="random", trials=6, holdout=0.25, seed=1,
                        n_jobs=2)
    """))

    ran = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=300, cwd=tmp_path)

    assert ran.returncode != 0
    assert ("probatum._errors.ProbatumRuntimeError: n_jobs: 2, and every worker died as "
            "it started, as they do where the script calls search or validate "
            "outside a guard, since spawn runs its top level again in each; put the "
            "call under if __name__ == \"__main__\":, or where it is under one, read "
            "what the workers printed") in ran.stderr
    assert "search: called by a worker of a search as it starts" in ran.stderr


def test_search_refuses_to_run_in_a_worker_as_it_starts(monkeypatch):
    monkeypatch.setattr(multiprocessing.current_process(), "_inheriting", True,
                        raising=False)

    with pytest.raises(ProbatumRuntimeError) as caught:
        searched()

    assert str(caught.value) == ("search: called by a worker of a search as it starts, as "
                                 "spawn runs the script's top level again in each; put "
                                 "the call under if __name__ == \"__main__\":")


def split_minded(data, size):
    # nothing held in this process, and the size asked in a worker
    return np.full(len(data), size if in_a_worker() else 0.0)


def test_a_winner_a_worker_scored_that_runs_otherwise_here_says_where_it_ran():
    study = searched(split_minded, {"size": Float(-1.0, 1.0, step=0.5)}, sampler="tpe",
                     trials=6, n_jobs=2)

    assert study.best_score > 0.0

    with pytest.raises(ProbatumRuntimeError, match=r"; it ran in a worker, which imported "
                                                   r"the policy afresh under a hash seed "
                                                   r"of its own, so give the policy what "
                                                   r"it reads as its arguments, or set "
                                                   r"PYTHONHASHSEED$"):
        study.best


def test_weights_a_random_search_draws_run_together_whatever_the_workers(monkeypatch):
    def refused(*given):
        raise AssertionError("a pool started")

    monkeypatch.setattr(probatum._search, "_pooled", refused)

    def held(data, size):
        return np.full(len(data), size)

    study = searched(held, {"size": Float(0.0, 1.0, step=0.25)}, sampler="grid",
                     n_jobs=2)

    assert len(study.trials) == 5


def test_workers_need_cloudpickle(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cloudpickle", None)
    monkeypatch.setattr(probatum._search.tempfile, "tempdir", str(tmp_path))

    with pytest.raises(MissingDependency) as caught:
        searched(n_jobs=2)

    assert str(caught.value) == ("n_jobs: workers need cloudpickle to carry the policy, "
                                 "which is not installed; pip install cloudpickle")
    assert not list(tmp_path.iterdir())


def test_a_policy_the_workers_cannot_carry_is_refused_naming_it():
    lock = threading.Lock()

    def locked(data, size):
        with lock:
            return np.full(len(data), size)

    with pytest.raises(PolicyError, match=r"^policy: locked cannot be carried to the "
                                          r"workers \(TypeError: cannot pickle "
                                          r"'_thread.lock' object\); keep what it holds "
                                          r"picklable, or pass n_jobs=1$"):
        searched(locked, {"size": Float(0.0, 1.0, step=0.25)}, sampler="tpe", n_jobs=2)


def test_a_worker_sends_no_returns_where_the_search_keeps_none(tmp_path):
    settings = dict(kind="class", made="strategy", objective="sharpe", min_trades=0,
                    rate=8_760.0, risk_free=0.0, keep="none")
    probatum._workers._wrote(tmp_path, Cross, VENUE, one().split(30)[0], settings)

    given = probatum._workers._trial(str(tmp_path), 1, {"fast": 3, "slow": 12})

    assert given["state"] == "complete" and given["returns"] is None


def test_a_worker_that_cannot_load_its_search_fails_the_trial_with_why(tmp_path):
    given = probatum._workers._trial(str(tmp_path), 3, {"fast": 2})

    assert (given["number"], given["state"]) == (3, "failed")
    assert given["error"].startswith("FileNotFoundError: ")
    assert given["trace"].startswith("Traceback (most recent call last):")


def test_a_warning_s_category_is_found_where_this_process_can_import_it():
    category = probatum._workers._category

    assert category("probatum._errors", "ProbatumWarning") is ProbatumWarning
    assert category(__name__, "Told") is Told
    assert category(__name__, "Warned.init.<locals>.Own") is None
    assert category("no.such.module", "Told") is None
    assert category("builtins", "int") is None
