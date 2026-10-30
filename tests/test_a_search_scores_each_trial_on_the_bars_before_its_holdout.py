"""A search scores each trial on one run over the bars before its holdout, a Data of
their own that holds no later bar, and keeps every trial as a row of its Study: complete,
floored or failed, its parameters, statistics and returns, and the winner run again as a
Result that equals its trial, then once on the held-out bars, counted as a look. The
random and grid samplers draw from named streams as the Specification states, TPE from
Optuna seeded from one, and every refusal names what to give instead.
"""

import json
import logging
import math
import sys
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

import probatum
import probatum._data
import probatum._samplers
import probatum._venue
from probatum import (Categorical, DataError, Float, Int, MissingDependency, PolicyError,
                      ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError,
                      ProbatumWarning, VenueError)
from probatum._result import STATS
from probatum._search import _Runner, _scored, _Trial, _winner
from probatum._streams import stream
from test_a_data_from_the_router_is_every_bar_it_asked_for import (HOUR, START, Router,
                                                                   bars, fetch,
                                                                   settlements)
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE, one, pair
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross

SPACE = {"fast": Int(2, 8), "slow": Int(10, 20)}


def swing(data, fast, slow, stop=None):
    """Long while the fast average is above the slow one, flat below, with a stop."""
    up, down = probatum.ta.ema(data.close, fast), probatum.ta.ema(data.close, slow)
    weights = np.where(np.isnan(down), np.nan, (up > down).astype(np.float64))
    return {"policy": weights, "stop": stop}


def held(**given):
    """A search of Cross on one instrument with a quarter held out."""
    return probatum.search(Cross, SPACE, one(), VENUE, **{
        **dict(sampler="random", trials=6, holdout=0.25, seed=3), **given})


def hourly(n):
    t     = np.arange(n, dtype=np.float64)
    close = 100.0 * (1.0 + 0.05 * np.sin(t / 5.0)) + t * 0.01
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    return probatum.Data(
        pd.DataFrame({"open": close, "high": close * 1.01,
                      "low": close * 0.99, "close": close,
                      "volume": 1e6}, index=index),
        symbol = "BTCUSDT",
        market = "linear",
    )


@pytest.mark.parametrize("holdout", [0.3, 300, "time of row 700"])
def test_a_holdout_of_0_3_of_1000_rows_scores_rows_0_to_699(holdout):
    # the Specification's worked example: holdout=300 and the time of row 700 hold out
    # the same rows
    data  = hourly(1_000)
    given = data.time[700].astype("datetime64[ns]") if isinstance(holdout, str) else holdout
    seen  = []

    def kept(data, fast):
        seen.append(data)
        return np.full(len(data), 0.5)

    study = probatum.search(kept, {"fast": Int(1, 2)}, data, VENUE, sampler="grid",
                            holdout=given)

    # the trials, and the check's runs, which cut it shorter, see no row after 699
    assert len(seen[0]) == 700 and max(map(len, seen)) == 700
    np.testing.assert_array_equal(study.best.time, data.time[:700])
    assert study.provenance["holdout"]["rows"]  == 300
    assert study.provenance["holdout"]["first"] == "2024-01-30 04:00"


def test_the_bars_a_trial_sees_are_a_data_of_their_own_under_the_slice_s_fingerprint():
    data, seen = hourly(100), []

    def kept(data, fast):
        seen.append(data)
        return np.full(len(data), 0.5)

    probatum.search(kept, {"fast": Int(1, 1)}, data, VENUE, sampler="grid", holdout=25)

    inside = seen[0]
    assert len(inside._root.time) == 75 and inside._root is not data._root
    assert inside.fingerprint == data.split(75)[0].fingerprint
    assert inside.names == data.names and inside.interval == data.interval
    assert inside.provenance == data.split(75)[0].provenance


def test_a_perpetual_first_funded_after_the_cut_runs_as_it_runs_in_that_slice():
    # ETHUSDT lists at hour 32 and is first charged at hour 40, after the cut at 36: the
    # bars before the cut hold none of its funding, and it runs as in data[:36]
    router = Router({"BTCUSDT": bars(48), "ETHUSDT": bars(16, start=START + 32 * HOUR,
                                                          close=10.0)},
                    {"BTCUSDT": settlements(),
                     "ETHUSDT": settlements(first=START + 40 * HOUR, rate=-0.0002)})

    with pytest.warns(ProbatumWarning, match="ETHUSDT's bars begin at 2024-01-02 08:00"):
        data = fetch(router, ("BTCUSDT", "ETHUSDT"), funding=True)

    venue = probatum.venues.binance_perp

    def pair_of(data, size):
        return np.full((len(data), 2), size)

    study = probatum.search(pair_of, {"size": Float(0.1, 0.3, step=0.1)}, data, venue,
                            sampler="grid", holdout=12)
    alone = probatum.backtest(np.full((36, 2), study.best_params["size"]),
                              data.split(36)[0], venue)

    assert study.best.stats == alone.stats

    # what lets it run is the set of instruments the whole Data funds
    inside = probatum._data._before(data, 36)
    inside._root.funded = None

    with pytest.raises(DataError, match="holds none for ETHUSDT"):
        probatum._venue._run(venue, inside)


def test_a_floor_a_failure_and_ties_pick_the_winner_by_score_then_number():
    # the Specification's worked example: round trips of 12, 41 and 41 at Sharpe ratios
    # of 2.1, 1.4 and 1.4 are floored, complete and complete, and trial 1 wins
    made = [_scored(k, {}, dict.fromkeys(STATS, 0.0) | {"num_trades": trades,
                                                         "sharpe": sharpe},
                    np.zeros(3), "sharpe", 30)
            for k, (trades, sharpe) in enumerate([(12, 2.1), (41, 1.4), (41, 1.4)])]

    assert [trial.state for trial in made]      == ["floored", "complete", "complete"]
    assert _winner(made, "maximize", 30).number == 1
    assert _winner(made, "minimize", 30).number == 1
    assert made[0].score is None and made[0].returns is not None

    made.append(_Trial(3, {}, "failed", 9.0, made[1].stats, error="x"))
    made[2].score = 1.2

    assert _winner(made, "maximize", 30).number == 1
    assert _winner(made, "minimize", 30).number == 2


def test_a_minimizing_search_wins_on_its_lowest_score_at_its_risk_free_rate():
    study = probatum.search(swing, SPACE, one(), VENUE, objective="max_drawdown_pct",
                            direction="minimize", sampler="tpe", trials=12,
                            holdout=0.25, seed=6, risk_free=0.04)

    rows   = [row for row in study.trials if row["state"] == "complete"]
    winner = min(rows, key=lambda row: (row["score"], row["number"]))

    assert study.best_params == {name: winner[name] for name in SPACE}

    inside = one().split(30)[0]
    again  = probatum.backtest(swing(inside, winner["fast"], winner["slow"])["policy"],
                               inside, VENUE, risk_free=0.04)

    assert again.stats._values == {key: winner[key] for key in STATS} == study.best.stats._values


def test_no_complete_trial_raises_for_its_floor_or_its_last_failure():
    with pytest.raises(ProbatumValueError) as caught:
        held(min_trades=1_000)

    assert str(caught.value) == ("min_trades: 1000, and no trial of 6 closed that many "
                                 "round trips on the in-sample bars, the most 2; "
                                 "lower it")

    def broken(data, fast, slow):
        return 1 / 0

    with pytest.raises(ProbatumRuntimeError) as caught:
        probatum.search(broken, SPACE, one(), VENUE, sampler="random", trials=3,
                        holdout=0.25, seed=1)

    assert str(caught.value).startswith(
        "policy: all 3 trials failed, the last, trial 2, with ZeroDivisionError: "
        "division by zero; backtest it with {'fast': ")
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


def test_a_trial_that_raises_or_returns_another_kind_fails_alone():
    def sometimes(data, fast, slow):
        if fast == 3:
            raise ArithmeticError("no three")
        if fast == 4:
            return Cross(fast, slow)
        return swing(data, fast, slow)

    study = probatum.search(sometimes, SPACE, one(), VENUE, sampler="grid", trials=200,
                            holdout=0.25)

    trials  = study.trials
    by_fast = {row["fast"]: (row["state"], row["error"]) for row in trials}

    assert by_fast[3] == ("failed", "ArithmeticError: no three")
    assert by_fast[4] == ("failed", "PolicyError: policy: returned a Strategy, where "
                                    "the trials before it returned weights; return one "
                                    "kind for every parameter")
    assert by_fast[2][0] == "complete" and by_fast[2][1] is None

    failed = trials.state == "failed"
    assert np.isnan(trials.sharpe[failed]).all() and trials.num_trades.mask[failed].all()
    assert np.isnan(study.returns[failed]).all()
    assert not np.isnan(study.returns[~failed]).any()


def test_the_winner_runs_again_as_the_result_its_trial_was_scored_on():
    for policy, space in [(Cross, SPACE), (swing, dict(SPACE, stop=Float(0.01, 0.05,
                                                                         step=0.02)))]:
        data  = one()
        study = probatum.search(policy, space, data, VENUE, sampler="random", trials=5,
                                holdout=0.25, seed=11)

        trials = study.trials
        # the first of the best scores, the lowest-numbered, row k being trial k
        complete = np.flatnonzero(trials.state == "complete")
        winner   = trials[int(complete[np.argmax(trials.score[complete])])]

        assert {key: winner[key] for key in STATS} == study.best.stats._values
        assert study.best_score == winner["score"] == study.best.stats._values["sharpe"]

        given = (Cross(**study.best_params) if policy is Cross
                 else policy(data.split(30)[0], **study.best_params))
        if isinstance(given, dict):
            again = probatum.backtest(given["policy"], data.split(30)[0], VENUE,
                                      stop=given["stop"])
        else:
            again = probatum.backtest(given, data.split(30)[0], VENUE)

        np.testing.assert_array_equal(study.best.equity, again.equity)
        assert study.best is study.best
        np.testing.assert_array_equal(study.returns[winner["number"]],
                                      study.best.returns)


SHIFT = [0.0]


def shifted(data, fast, slow):
    return swing(data, fast + SHIFT[0], slow)


def test_a_winner_whose_code_changed_since_the_search_is_refused_before_a_look():
    SHIFT[0] = 0.0
    study = probatum.search(shifted, SPACE, one(), VENUE, sampler="random", trials=4,
                            holdout=0.25, seed=2)

    SHIFT[0] = 3.0
    for read in (lambda: study.best, study.evaluate_holdout):
        with pytest.raises(ProbatumRuntimeError,
                           match=r"^best: trial \d+'s \w+ was .* on the in-sample bars, "
                                 r"and run again it is .*; the policy's code or the "
                                 r"globals it reads changed since the search, so search "
                                 r"again$"):
            read()

    assert study.looks == 0 and study.holdout is None


def test_the_holdout_is_run_once_counted_as_a_look_and_kept():
    data  = one()
    study = held()

    assert repr(study).endswith("; 10 held out, not yet read)")

    result = study.evaluate_holdout()

    assert study.evaluate_holdout() is result is study.holdout and study.looks == 1

    again = probatum.backtest(Cross(**study.best_params), data.split(30)[1], VENUE)

    np.testing.assert_array_equal(result.equity, again.equity)
    assert repr(study).endswith("; 10 held out, read once)")

    nothing = held(holdout=0)

    with pytest.raises(ProbatumValueError) as caught:
        nothing.evaluate_holdout()

    assert str(caught.value) == ("evaluate_holdout: the search held nothing out; search "
                                 "with a holdout, such as holdout=0.3")
    assert repr(nothing).endswith("over 40 bars; nothing held out)")


def test_a_holdout_run_that_raises_has_spent_its_look():
    def short(data, fast, slow):
        # the held-out bars by their first time, as no cut of the check starts there
        if data.time[0] == one().time[30]:
            raise RuntimeError("too few bars")
        return swing(data, fast, slow)

    study = probatum.search(short, SPACE, one(), VENUE, sampler="random", trials=3,
                            holdout=0.25, seed=5)

    for looks in (1, 2):
        with pytest.raises(RuntimeError, match="too few bars"):
            study.evaluate_holdout()
        assert study.looks == looks and study.holdout is None

    assert repr(study).endswith("; 10 held out, read 2 times)")


def test_the_random_sampler_draws_each_parameter_as_the_specification_states():
    space = {
        "n":       Int(3, 30, step=3),
        "log_n":   Int(1, 1_000, log=True),
        "x":       Float(-1.0, 2.0),
        "log_x":   Float(0.001, 10.0, log=True),
        "stepped": Float(0.01, 0.1, step=0.01),
        "kind":    Categorical(["ema", "sma", None]),
    }

    def any_of(data, **given):
        return np.full(len(data), 0.5)

    study = probatum.search(any_of, space, one(), VENUE, sampler="random", trials=20,
                            holdout=0.25, seed=42)

    rng, ln, exp = stream(42, "search random"), probatum._probatum.ln, probatum._probatum.exp
    for row in study.trials:
        n = 3 + 3 * int(rng.integers(0, 10))
        low, high = ln(0.5), ln(1_000.5)
        log_n = min(max(round(exp(low + rng.random() * (high - low))), 1), 1_000)
        x     = min(-1.0 + rng.random() * 3.0, math.nextafter(2.0, -math.inf))
        low, high = ln(0.001), ln(10.0)
        log_x = min(max(exp(low + rng.random() * (high - low)), 0.001),
                    math.nextafter(10.0, -math.inf))
        k       = int(rng.integers(0, 10))
        stepped = 0.1 if k == 9 else 0.01 + k * 0.01
        kind    = ["ema", "sma", None][int(rng.integers(0, 3))]

        assert (row["n"], row["log_n"], row["x"], row["log_x"], row["stepped"],
                row["kind"]) == (n, log_n, x, log_x, stepped, kind)

    assert study.provenance["sampler"] == {"name": "random", "stream": "search random"}


def test_a_grid_runs_every_point_in_order_or_a_drawn_sample_of_them():
    # the Specification's worked example: ten points of 0.01 to 0.1 beside two choices
    # are 20, run in order with the last parameter fastest, the last stop 0.1 exactly
    def any_of(data, stop, kind):
        return np.full(len(data), 0.5)

    space = {"stop": Float(0.01, 0.1, step=0.01), "kind": Categorical(["a", "b"])}
    study = probatum.search(any_of, space, one(), VENUE, sampler="grid", trials=100,
                            holdout=0.25)

    points = [(row["stop"], row["kind"]) for row in study.trials]
    stops  = [0.01 + k * 0.01 for k in range(9)] + [0.1]

    assert points == [(stop, kind) for stop in stops for kind in "ab"]
    assert points[-1] == (0.1, "b") and study.provenance["ran"] == 20

    sample = probatum.search(any_of, space, one(), VENUE, sampler="grid", trials=7,
                             holdout=0.25, seed=9)
    drawn = np.sort(stream(9, "search grid").choice(20, 7, replace=False))

    assert [(row["stop"], row["kind"]) for row in sample.trials] == [
        points[k] for k in drawn]


def test_tpe_repeats_from_its_seed_and_searches_probatum_s_lattice():
    space = {
        "fast": Int(2, 8),
        "slow": Int(10, 20),
        "stop": Float(0.01, 0.1, step=0.01),
    }

    first, second = (probatum.search(swing, space, one(), VENUE, sampler="tpe",
                                     trials=14, holdout=0.25, seed=4)
                     for _ in range(2))

    assert first.trials == second.trials

    lattice = {0.01 + k * 0.01 for k in range(9)} | {0.1}
    assert set(first.trials.stop.tolist()) <= lattice
    # Optuna's top point is an ulp short of the lattice's
    assert probatum._samplers._onto(Float(0.01, 0.1, step=0.01), 0.01 + 9 * 0.01) == 0.1
    assert first.provenance["sampler"] == dict(
        name="tpe", n_startup_trials=10, n_ei_candidates=24, multivariate=True,
        group=False, constant_liar=True,
        seed=int(stream(4, "search tpe").integers(0, 2**32)), optuna="5.0.0")


def test_tpe_is_told_a_floored_or_failed_trial_as_failed():
    optuna = pytest.importorskip("optuna")
    draws  = probatum._samplers._Tpe({"n": Int(1, 5)}, 3, 0, "maximize")

    for number, state in enumerate(["complete", "floored", "failed"]):
        draws.ask(number)
        draws.tell(number, _Trial(number, {}, state, 1.0))

    assert [trial.state for trial in draws.study.trials] == [
        optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.FAIL]


def test_tpe_needs_optuna_5_and_puts_its_logger_back():
    optuna = pytest.importorskip("optuna")
    level  = optuna.logging.get_verbosity()

    held(sampler="tpe", trials=2)

    assert optuna.logging.get_verbosity() == level


@pytest.mark.parametrize("installed, said", [
    (None, "sampler: \"tpe\" needs optuna, which is not installed; pip install "
           "\"optuna>=5.0\""),
    ("4.5.0", "sampler: \"tpe\" needs optuna 5.0 or later, and 4.5.0 is installed; pip "
              "install \"optuna>=5.0\""),
])
def test_tpe_without_optuna_5_names_the_install(installed, said, monkeypatch):
    if installed is None:
        monkeypatch.setitem(sys.modules, "optuna", None)
    else:
        optuna = pytest.importorskip("optuna")
        monkeypatch.setattr(optuna, "__version__", installed)

    with pytest.raises(MissingDependency) as caught:
        held(sampler="tpe")

    assert str(caught.value) == said

    if installed is None:
        assert held(sampler="random").trials


def test_the_trials_table_gives_each_column_the_kind_its_values_share():
    space = {
        "n":     Int(1, 2),
        "x":     Float(0.5, 1.0),
        "mixed": Categorical([1, 2.5]),
        "named": Categorical(["a", 1]),
        "flag":  Categorical([True, False]),
        "maybe": Categorical([True, None]),
    }

    def any_of(data, **given):
        return np.full(len(data), 0.5)

    trials = probatum.search(any_of, space, one(), VENUE, sampler="random", trials=8,
                             holdout=0.25, seed=1).trials

    assert trials.columns == ("number", "state", "score", *space, *STATS, "error")

    kinds = {name: kind for name, kind, _ in trials._held.kinds}
    assert [kinds[name] for name in ("number", "state", "score", *space)] == [
        "int", "text", "float", "int", "float", "float", "text", "bool", "text"]
    assert set(trials.mixed.tolist()) <= {1.0, 2.5}
    assert set(trials.named.tolist()) <= {"a", "1"}
    assert set(trials.maybe.tolist()) <= {"True", None}
    assert [kinds[name] for name in ("num_trades", "num_fills", "sharpe", "error")] == [
        "int", "int", "float", "text"]


def test_keep_none_keeps_no_returns_and_keep_all_every_trial_s():
    assert held(keep="none").returns is None

    study = held()

    assert study.returns.shape == (6, 30) and not study.returns.flags.writeable
    assert (study.returns[:, 0] == 0.0).all()


def test_a_seed_left_out_is_drawn_and_recorded_so_the_search_repeats():
    study = held(seed=None)
    seed  = study.provenance["seed"]

    assert isinstance(seed, int) and seed >= 0
    assert held(seed=seed).trials == study.trials

    json.dumps(study.provenance)


def test_numpy_s_integers_are_taken_as_whole_numbers_and_recorded_as_python_s():
    given = held(trials=np.int64(6), min_trades=np.int64(0), n_jobs=np.int64(1))

    assert given.trials == held().trials

    json.dumps(given.provenance)

    assert [type(given.provenance[key]) for key in ("trials", "min_trades")] == [int, int]


def test_the_winner_s_parameters_are_its_own_whatever_a_caller_does_with_them():
    study  = held()
    winner = study.best_params

    study.best_params["fast"] = 99

    assert study.best_params == winner
    assert study.best.stats == probatum.backtest(Cross(**winner), one().split(30)[0],
                                                 VENUE).stats

    study.evaluate_holdout()
    again = probatum.backtest(Cross(**winner), one().split(30)[1], VENUE)

    np.testing.assert_array_equal(study.holdout.equity, again.equity)

    for name in ("looks", "best_score"):
        with pytest.raises(AttributeError):
            setattr(study, name, 0)

    assert study.looks == 1


class Locked(Cross):
    """Cross holding a lock, which no copy can take."""


    def __init__(self, fast=3, slow=12):
        super().__init__(fast, slow)
        self.lock = threading.Lock()


def test_a_strategy_backtest_cannot_copy_fails_its_trials_as_backtest_does():
    with pytest.raises(ProbatumRuntimeError, match=r"^policy: all 6 trials failed, the "
                                                   r"last, trial 5, with TypeError: "
                                                   r"cannot pickle '_thread.lock'"):
        probatum.search(Locked, SPACE, one(), VENUE, sampler="random", trials=6,
                        holdout=0.25, seed=3)


def test_a_search_raises_from_its_last_failure_alone():
    def broken(data, fast, slow):
        return 1 / 0

    # each failure's traceback holds the frames that ran it, so only the last keeps one
    runner = _Runner(broken, "function", one(), probatum._venue._run(VENUE, one()),
                     8_760.0, 0.0, "sharpe", 0)

    first  = runner.trial(0, {"fast": 2, "slow": 10})
    second = runner.trial(1, {"fast": 3, "slow": 11})

    assert first.raised is None and isinstance(second.raised, ZeroDivisionError)


def test_a_search_logs_its_start_and_end_and_each_trial(caplog):
    with caplog.at_level(logging.DEBUG, logger="probatum"):
        held()

    said = [(record.levelname, record.getMessage()) for record in caplog.records
            if record.getMessage().startswith("search:")]

    assert said[0] == ("INFO", "search: 6 trials of Cross by random on 30 rows, 10 held "
                               "out")

    trials = [message for level, message in said if level == "DEBUG"]
    assert len(trials) == 6 and trials[0].startswith("search: trial 0 complete, sharpe ")
    assert ("INFO", "search: trial 0's decisions are causal") in said
    assert said[-1][0] == "INFO" and said[-1][1].startswith("search: 6 complete of 6; "
                                                            "the best, trial ")


def test_holdout_none_warns_at_the_caller_s_line_and_0_quietly():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held(holdout=None)
        held(holdout=0)

    assert [type(warning.message) for warning in caught] == [ProbatumWarning]
    assert caught[0].filename == __file__
    assert str(caught[0].message).startswith("holdout: none, so every trial is scored on "
                                             "every bar")


def test_a_function_s_strategy_is_a_prototype_each_trial_copies():
    returned = Cross(fast=3, slow=12)

    def the_same(data, fast):
        return returned

    probatum.search(the_same, {"fast": Int(1, 2)}, one(), VENUE, sampler="grid",
                    holdout=0.25)

    assert not hasattr(returned, "calls")


REFUSED = [
    (dict(policy=Cross(), space=SPACE), PolicyError,
     "policy: an instance of Cross; pass the class, Cross, which each trial builds with "
     "its parameters"),
    (dict(policy=dict, space=SPACE), PolicyError,
     "policy: the class dict, which is no Strategy; pass a Strategy subclass, or a "
     "function of the data that returns a policy"),
    (dict(policy=np.zeros(3), space=SPACE), PolicyError,
     "policy: an object of type ndarray; pass a Strategy subclass, or a function of the "
     "data that returns a policy"),
    (dict(policy=lambda: None, space=SPACE), PolicyError,
     "policy: <lambda> takes no data; give it the data as its first argument"),
    (dict(space=[1, 2]), ProbatumTypeError,
     "space: [1, 2]; give a dict of each parameter's range, such as {\"fast\": "
     "probatum.Int(5, 40)}"),
    (dict(space={}), ProbatumValueError,
     "space: no parameters; give each a range, such as {\"fast\": probatum.Int(5, 40)}"),
    (dict(space={"fast": (2, 8)}), ProbatumTypeError,
     "space: fast is (2, 8); give a probatum.Int, probatum.Float or "
     "probatum.Categorical"),
    (dict(space=None), ProbatumTypeError, "space: None; give a dict"),
    (dict(space={"sharpe": Int(1, 2)}), ProbatumValueError,
     "space: sharpe, a column the trials table has of its own; name the parameter "
     "otherwise"),
    (dict(space={"fast": Int(1, 2), "speed": Int(1, 2)}), ProbatumTypeError,
     "space: speed, which Cross does not take; name its parameters, fast, slow"),
    (dict(policy=lambda data, fast, slow: None, space={"fast": Int(1, 2)}),
     ProbatumTypeError, "space: no range for slow, which <lambda> needs; give it one"),
    (dict(objective="sharp"), ProbatumValueError, "objective: 'sharp'; use one of "
                                                  "total_return_pct, "),
    (dict(direction="max"), ProbatumValueError,
     "direction: 'max'; use \"maximize\" or \"minimize\""),
    (dict(trials=0), ProbatumValueError, "trials: 0; give a whole number from 1"),
    (dict(trials=2.0), ProbatumTypeError, "trials: 2.0; give a whole number from 1"),
    (dict(sampler="cmaes"), ProbatumValueError,
     "sampler: 'cmaes'; use \"tpe\", \"random\" or \"grid\""),
    (dict(min_trades=-1), ProbatumValueError,
     "min_trades: -1; give a whole number of round trips from 0"),
    (dict(keep="holdout"), ProbatumValueError,
     "keep: 'holdout'; use \"all\" or \"none\""),
    (dict(keep="some"), ProbatumValueError, "keep: 'some'; use \"all\" or \"none\""),
    (dict(n_jobs=0), ProbatumValueError,
     "n_jobs: 0; give a number of workers from 1, or -1 for one a core"),
    (dict(n_jobs=-2), ProbatumValueError,
     "n_jobs: -2; give a number of workers from 1, or -1 for one a core"),
    (dict(n_jobs=2.5), ProbatumTypeError, "n_jobs: 2.5; give a whole number"),
    (dict(risk_free=math.nan), ProbatumValueError,
     "risk_free: nan; give the annual rate as a finite number, such as 0.04"),
    (dict(seed=-1), ProbatumValueError, "seed: -1; give a whole number from 0"),
    (dict(seed="7"), ProbatumTypeError, "seed: '7'; give a whole number from 0"),
    (dict(holdout=True), ProbatumTypeError,
     "holdout: True; give a share of the rows, a number of them, or the time the "
     "holdout starts at"),
    (dict(holdout=1.5), ProbatumValueError,
     "holdout: 1.5; give a share of the rows above 0 and below 1, a number of them, or "
     "the time the holdout starts at"),
    (dict(holdout=math.nan), ProbatumValueError,
     "holdout: nan; give a share of the rows above 0 and below 1, a number of them, or "
     "the time the holdout starts at"),
    (dict(holdout=-math.inf), ProbatumValueError,
     "holdout: -inf; give a share of the rows above 0 and below 1, a number of them, or "
     "the time the holdout starts at"),
    (dict(holdout=-3), ProbatumValueError, "holdout: -3; give a number of rows from 1"),
    (dict(data=one()[:1], holdout=0), DataError,
     "data: one printed row to search; give a Data of at least 2"),
    (dict(holdout=39), ProbatumValueError,
     "holdout: 39 leaves 1 of the printed rows to search and 39 to hold out; leave at "
     "least 2 on each side"),
    (dict(holdout="soon"), ProbatumValueError,
     "holdout: 'soon' is not a time; give one such as \"2024-06\" or \"2024-06-01 "
     "08:00\""),
    (dict(venue=VENUE.replace(intrabar="random")), VenueError,
     "venue: intrabar is \"random\", which search refuses so that a search repeats "
     "without a seed; replace(intrabar=\"pessimistic\") or \"optimistic\""),
    (dict(sampler="grid", space={"fast": Int(2, 8), "slow": Int(10, 20),
                                 "stop": Float(0.0, 1.0)}, policy=swing),
     ProbatumValueError,
     "space: stop is Float(0.0, 1.0), which has no step to list its values; give it "
     "one, or search it with sampler=\"random\""),
    (dict(sampler="random", space={"fast": Int(-2**63, 2**63 - 1), "slow": Int(1, 2)}),
     ProbatumValueError,
     "space: fast holds 18,446,744,073,709,551,616 points, more than sampler=\"random\" "
     "draws among, 2**63 - 1; give it a narrower range"),
    (dict(sampler="grid", trials=10, space={"fast": Int(0, 2**40),
                                            "slow": Int(0, 2**40)}),
     ProbatumValueError,
     f"space: a grid of {(2**40 + 1)**2:,} points, more than sampler=\"grid\" draws "
     f"among, 2**63 - 1; give it fewer, or search it with sampler=\"random\""),
]


@pytest.mark.parametrize("given, kind, said", REFUSED)
def test_a_search_refuses_what_it_cannot_run_with_the_fix(given, kind, said):
    given = {**dict(policy=Cross, space=SPACE, data=one(), venue=VENUE,
                    sampler="random", trials=2, holdout=0.25, seed=1), **given}

    with pytest.raises(kind) as caught:
        probatum.search(**given)

    assert type(caught.value) is kind and str(caught.value).startswith(said)


def test_a_universe_of_weights_searches_as_one_instrument_does():
    def both(data, size):
        return np.full((len(data), 2), size)

    study = probatum.search(both, {"size": Float(-1.0, 1.0, step=0.5)}, pair(), VENUE,
                            sampler="grid", holdout=0.25)

    assert len(study.trials) == 5 and set(study.trials.state.tolist()) == {"complete"}
    assert study.best.stats == probatum.backtest(
        np.full((30, 2), study.best_params["size"]), pair().split(30)[0], VENUE).stats
