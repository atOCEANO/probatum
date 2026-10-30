"""A random or grid search over weights runs its trials, once the first is checked,
together in Rust, each on an engine of its own over the shared bars that keeps only what
its statistics read: every trial's statistics and returns are those backtest gives the
same weights, whatever the chunks, a trial that raises or returns a Strategy fails
alone, and TPE and a Strategy run one trial at a time.
"""

import numpy as np
import pytest

import probatum
import probatum._search
from probatum import Float, Int, ProbatumValueError
from probatum._probatum import Engine, run_array, run_arrays
from probatum._result import STATS
from probatum._venue import _run
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE, one, pair
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross

SPACE = {"fast": Int(2, 8), "slow": Int(10, 20), "stop": Float(0.01, 0.05, step=0.01)}


def swing(data, fast, slow, stop):
    up, down = probatum.ta.ema(data.close, fast), probatum.ta.ema(data.close, slow)
    weights = np.where(np.isnan(down), np.nan, (up > down).astype(np.float64))
    return {"policy": weights, "stop": stop}


def searched(policy=swing, space=SPACE, **given):
    return probatum.search(policy, space, one(), VENUE, **{
        **dict(sampler="random", trials=12, holdout=0.25, seed=5), **given})


@pytest.fixture
def chunks(monkeypatch):
    """The number of trials each call of run_arrays ran, three threads to a pool."""
    calls, real = [], probatum._search.run_arrays

    def counted(engine, trials, *rest):
        calls.append(len(trials))
        return real(engine, trials, *rest)

    monkeypatch.setattr(probatum._search, "run_arrays", counted)
    monkeypatch.setattr(probatum._search, "threads", lambda: 3)

    return calls


def test_every_trial_run_together_is_the_run_backtest_makes_of_its_weights(chunks):
    study = searched()

    assert chunks == [11]

    inside = one().split(30)[0]
    for row, returns in zip(study.trials, study.returns):
        given  = swing(inside, row["fast"], row["slow"], row["stop"])
        result = probatum.backtest(given["policy"], inside, VENUE, stop=given["stop"])
        assert {key: row[key] for key in STATS} == result.stats._values
        np.testing.assert_array_equal(returns, result.returns)


def test_chunks_hold_a_trial_a_thread_and_then_trials_until_their_bytes(chunks,
                                                                        monkeypatch):
    whole = searched(trials=12)

    monkeypatch.setattr(probatum._search, "CHUNK", 1)
    chunks.clear()
    parted = searched(trials=12)

    assert chunks        == [3, 3, 3, 2]
    assert parted.trials == whole.trials
    np.testing.assert_array_equal(parted.returns, whole.returns)

    # each trial's 30 rows count 8 bytes of curve and twice their 8 bytes of weights
    monkeypatch.setattr(probatum._search, "CHUNK", 5 * (8 * 30 + 2 * 30 * 8))
    chunks.clear()

    assert searched(trials=12).trials == whole.trials and chunks == [5, 5, 1]


def test_trials_run_together_score_at_the_search_s_objective_direction_and_rate(chunks):
    study = searched(objective="max_drawdown_pct", direction="minimize", risk_free=0.04)

    assert chunks == [11]

    inside = one().split(30)[0]
    for row in study.trials:
        given  = swing(inside, row["fast"], row["slow"], row["stop"])
        result = probatum.backtest(given["policy"], inside, VENUE, stop=given["stop"],
                                   risk_free=0.04)
        assert {key: row[key] for key in STATS} == result.stats._values

    winner = min(study.trials, key=lambda row: (row["score"], row["number"]))
    assert study.best_params == {name: winner[name] for name in SPACE}


def test_a_universe_s_grid_runs_together_as_backtest_runs_each_point(chunks):
    def both(data, size):
        return np.full((len(data), 2), size)

    study = probatum.search(both, {"size": Float(-1.0, 1.0, step=0.25)}, pair(), VENUE,
                            sampler="grid", holdout=0.25)

    assert chunks == [8]

    for row in study.trials:
        result = probatum.backtest(np.full((30, 2), row["size"]), pair().split(30)[0],
                                   VENUE)
        assert {key: row[key] for key in STATS} == result.stats._values


def test_a_trial_that_raises_or_returns_a_strategy_in_a_chunk_fails_alone(chunks):
    def sometimes(data, fast, slow, stop):
        if fast == 3:
            raise ArithmeticError("no three")
        if fast == 4:
            return Cross(fast, slow)
        return swing(data, fast, slow, stop)

    # in the grid's order fast=2 comes first, so the first trial checked is weights,
    # and of the 23 after it the twelve of fast 3 and 4 fail before the chunk runs
    space = {"fast": Int(2, 5), "slow": Int(10, 12), "stop": Float(0.01, 0.02, step=0.01)}
    study = searched(sometimes, space, sampler="grid", trials=100)

    assert chunks == [11]

    errors = {row["fast"]: row["error"] for row in study.trials if row["state"] == "failed"}
    assert errors.get(3) == "ArithmeticError: no three"
    assert errors.get(4) == ("PolicyError: policy: returned a Strategy, where the trials "
                             "before it returned weights; return one kind for every "
                             "parameter")
    assert {row["state"] for row in study.trials if row["fast"] not in (3, 4)} == {
        "complete"}


def test_tpe_and_a_strategy_run_one_trial_at_a_time(chunks):
    searched(sampler="tpe")
    searched(Cross, {"fast": Int(2, 8), "slow": Int(10, 20)})

    assert chunks == []


def test_run_arrays_reads_each_trial_as_run_array_does_and_needs_a_reporting_engine():
    run    = _run(VENUE, one())
    engine = Engine(run.candles, report=True, record="account", **run.settings)
    good   = (np.full((40, 1), 0.5), "weight", 0.0, 0.02, None)

    [(stats, curve)] = run_arrays(engine, [good], 0, 8_760.0)
    run_array(engine, *good, start=0)

    assert stats == engine.stats(8_760.0) and np.array_equal(curve, engine.equity_curve())

    for given, said in [
        (dict(trials=[good, (np.full((39, 1), 0.5), "weight", 0.0, None, None)]),
         "rows: an array of shape (39, 1) from bar 0 of 40 over 1 instruments; give one "
         "row for each bar from the first"),
        (dict(trials=[(np.full((40, 1), 0.5), "units", 0.0, None, None)]),
         "sizing: \"units\"; use \"weight\" or \"quantity\""),
        (dict(periods_per_year=0.0),
         "periods_per_year: 0; give the bars a year as a finite number above 0"),
        (dict(engine=Engine(run.candles, **run.settings)),
         "engine: built with report=False, which keeps no statistics; build it with "
         "report=True"),
    ]:
        called = {**dict(engine=engine, trials=[good], start=0,
                         periods_per_year=8_760.0), **given}
        with pytest.raises(ProbatumValueError) as caught:
            run_arrays(**called)
        assert str(caught.value) == said
