"""A Study's summary is a frame of what it found and how the winner held up: a search's
best score, its deflated Sharpe beside a random null's mean, its overfitting
probability and its holdout as it stands; a validation's paths, their median score
beside the median path's interval, its windows, its decay and its costs; and its
context, what it searched or validated, at which venue, on which bars and what it held
out.
Consistency scores the winner in stretches, or a validation's windows, against the
objective's break-even, and decay sets what each fit scored against what its bars then
earned (Specification, Summaries; Consistency and decay).
"""

import math
import pickle

import numpy as np
import pytest

import probatum
from probatum import ProbatumTypeError, ProbatumValueError, stats

from reference import statistics

HOUR  = 3_600_000_000_000
SPACE = {"fast": probatum.Int(2, 20)}


def bars(rows=800):
    close  = 100.0 + np.random.default_rng(9).normal(0.0, 1.0, rows).cumsum()
    shaped = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(rows, 1e3)])

    return probatum.Data(
        shaped,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * HOUR,
    )


def alternating(data, fast=5):
    return np.where(np.arange(len(data.close)) % (2 * fast) < fast, 1.0, 0.0)


def raising_on_the_tail(data, fast=5):
    # the held-out bars are the only ones that do not open at the clock's first time
    if data.time[0] > 0:
        raise RuntimeError("the tail")

    return alternating(data, fast)


class Flip(probatum.Strategy):

    def __init__(self, fast=5):
        self.fast = fast


    def next(self, bar, account):
        account.target(1.0 if (bar.i // self.fast) % 2 == 0 else 0.0)


def searched(policy=alternating, seed=1, **given):
    settings = dict(trials=12, sampler="random", holdout=0.3, seed=seed)
    settings.update(given)

    return probatum.search(policy, SPACE, bars(), "frictionless", **settings)


@pytest.fixture(scope="module")
def study():
    return searched()


@pytest.fixture(scope="module")
def validated():
    return probatum.validate(Flip, bars(), "frictionless", probatum.cv.PurgedKFold(4),
                             space=SPACE, search=dict(trials=4, sampler="random"),
                             holdout=0.2, seed=1)


def frame(study, **given):
    return study.summary(**given).to_pandas()


def test_a_search_s_summary_names_its_winner_its_null_and_its_holdout(study):
    null     = searched(seed=2)
    found    = study.summary(null=null)
    shown    = found.to_pandas()
    trials   = null.trials
    done     = np.asarray(trials["state"]) == "complete"
    deflated = stats.deflated_sharpe(study, null)
    halved   = stats.overfitting(study)
    distinct = len(set(np.asarray(study.trials["fast"]).tolist()))

    assert list(shown.index) == ["best", "deflated_sharpe", "deflated_cluster", "null_mean",
                                 "overfitting", "overfitting_loss", "holdout",
                                 "buy_and_hold"]
    assert shown["value"].tolist()[:6] == pytest.approx(
        [study.best_score, deflated.value, deflated.effective_value,
         np.asarray(trials["score"], dtype=np.float64)[done].mean(), halved.probability,
         halved.loss])
    assert shown.loc["holdout", "why_none"] == "not yet run; evaluate_holdout() spends a look"
    assert list(found.context.items()) == [
        ("policy", "alternating"), ("venue", "frictionless"), ("data", "BTCUSDT 1h"),
        ("objective", "sharpe"), ("direction", "maximize"), ("sampler", "random"),
        ("trials", 12), ("distinct", distinct), ("first", "1970-01-01"),
        ("last", "1970-01-24"), ("held out", "the last 30% (240 bars)"), ("looks", 0),
        ("best fast", study.best_params["fast"])]
    # a grid draws each configuration once, so none repeats
    assert "distinct" not in searched(sampler="grid").summary().context


def test_a_search_says_why_it_has_no_deflated_sharpe_or_overfitting_probability():
    kept = frame(searched(keep="none"))

    assert kept.loc["overfitting", "why_none"] == "the search kept no returns"

    ordered = frame(searched(objective="calmar"), null=searched(seed=2, objective="calmar"))

    assert ordered.loc["deflated_sharpe", "why_none"] == (
        "the search ranked on calmar, and a deflation is of a Sharpe a search maximised")
    assert ordered.loc["overfitting", "why_none"] == (
        "calmar is not a statistic the halves give whatever the order of their returns")

    lowest = frame(searched(direction="minimize"), null=searched(seed=2, direction="minimize"))

    assert lowest.loc["deflated_sharpe", "why_none"] == (
        "the search minimised sharpe, and a deflation is of a Sharpe a search maximised")

    alone = frame(searched(trials=1), null=searched(seed=2))

    assert alone.loc["deflated_sharpe", "why_none"] == (
        "study: 1 trial; a deflation corrects for more than one, so search with trials=2 "
        "or more")
    assert alone.loc["overfitting", "why_none"] == "1 complete trial"

    short = frame(searched(trials=2, holdout=800 - 16))

    assert short.loc["overfitting", "why_none"] == "15 returns, fewer than its 16 blocks"


LAG = [0]


def lagging(data, fast=5):
    return np.where((np.arange(len(data.close)) + LAG[0]) % (2 * fast) < fast, 1.0, 0.0)


def test_a_winner_that_runs_otherwise_now_leaves_the_summary_saying_why():
    # the deflation reruns the winner, and a policy whose global changed since its
    # search is refused there; the summary names that refusal and prints the rest
    moved = searched(policy=lagging)
    null  = searched(policy=lagging, seed=2)

    LAG[0] = 3
    try:
        shown = frame(moved, null=null)
    finally:
        LAG[0] = 0

    assert shown.loc["deflated_sharpe", "why_none"].startswith("best: trial ")
    assert shown.loc["holdout", "why_none"].startswith("not yet run")


def test_a_search_s_shortcuts_are_the_statistics_of_its_trials(study):
    null  = searched(seed=2)
    given = dict(benchmark="buy_and_hold", resamples=40, seed=3)

    assert study.deflated_sharpe(null)  == stats.deflated_sharpe(study, null)
    assert study.effective_trials()     == stats.effective_trials(study)
    assert study.overfitting()          == stats.overfitting(study)
    assert study.reality_check(**given) == stats.reality_check(study, **given)
    assert study.min_backtest_length()  == stats.min_backtest_length(study)


def test_a_null_that_is_not_a_random_search_of_the_same_ground_is_refused(study,
                                                                         validated):
    with pytest.raises(ProbatumValueError, match="null: a search by grid; give a random "
                       "search"):
        study.summary(null=searched(sampler="grid"))
    with pytest.raises(ProbatumValueError, match="null: its objective, 'calmar', is not "
                       "the search's, 'sharpe'"):
        study.summary(null=searched(objective="calmar"))
    with pytest.raises(ProbatumValueError, match="null: its in-sample bars, 'b2:"):
        study.summary(null=searched(holdout=0.2))
    with pytest.raises(ProbatumValueError, match="null: a validation's Study; give the "
                       "Study of a random search"):
        study.summary(null=validated)
    with pytest.raises(ProbatumTypeError, match="null: int; give the Study of a random "
                       "search"):
        study.summary(null=3)
    with pytest.raises(ProbatumValueError, match="null: given to a validation's summary"):
        validated.summary(null=study)


def test_the_holdout_reads_as_it_stands_and_decay_follows_it():
    fresh = searched()

    with pytest.raises(ProbatumValueError, match="decay: the holdout is not yet run; "
                       "evaluate_holdout\\(\\) runs it, and spends a look"):
        fresh.decay()

    held   = fresh.evaluate_holdout()
    looked = held.stats.interval("sharpe")
    shown  = frame(fresh)

    assert shown.loc["holdout", ["value", "low", "high", "level"]].tolist() == [
        held.stats["sharpe"], looked.low, looked.high, 0.95]
    assert shown.loc["buy_and_hold", "value"] == held.summary().to_pandas().loc[
        "sharpe", "buy_and_hold"]

    gap = fresh.best_score - held.stats["sharpe"]
    assert fresh.decay() == stats.Decay(["holdout"], [held.time[0]], [held.time[-1]],
                                        [fresh.best_score], [held.stats["sharpe"]], [gap],
                                        gap, None, "sharpe", "maximize")

    raised = searched(policy=raising_on_the_tail)

    with pytest.raises(RuntimeError, match="the tail"):
        raised.evaluate_holdout()

    assert frame(raised).loc["holdout", "why_none"] == ("read 1 time; its run raised, so no "
                                                        "result is kept")

    with pytest.raises(ProbatumValueError, match="decay: the holdout was read 1 time and "
                       "its run raised, so no result is kept; evaluate_holdout\\(\\) runs it "
                       "again, and spends a look"):
        raised.decay()

    whole = searched(holdout=0)

    assert [whole.summary().context[name] for name in ("held out", "looks")] == ["none", 0]
    assert frame(whole).loc["holdout", "why_none"] == "the search held every bar"

    with pytest.raises(ProbatumValueError, match="decay: the search held nothing out"):
        whole.decay()


def test_a_search_s_consistency_scores_its_winner_in_stretches_of_its_run(study):
    found   = study.consistency()
    best    = study.best
    returns = best.returns
    rate    = best.provenance["periods_per_year"]
    rows    = len(returns)

    size, extra = divmod(rows - 1, 4)
    expected, first = [], 1
    for k in range(4):
        last = first + size + (k < extra) - 1
        expected.append(stats.sharpe(returns[first:last + 1], periods_per_year=rate))
        first = last + 1

    assert found.scores == pytest.approx(expected, rel=1e-9)

    past = sum(score > 0.0 for score in found.scores)
    assert (found.unit, found.break_even, found.share) == ("stretches", 0.0, past / 4)
    assert len(study.consistency(stretches=6).scores) == 6

    with pytest.raises(ProbatumValueError, match="stretches: 1; give a whole number from "
                       "2"):
        study.consistency(stretches=1)


def test_a_drawdown_objective_keeps_the_record_s_sign():
    deep = searched(objective="max_drawdown_pct", direction="minimize")

    assert frame(deep).loc["best", "value"] == deep.best_score > 0.0
    assert f"best max_drawdown_pct {deep.best_score:,.2f} over" in repr(deep)

    held = deep.evaluate_holdout()

    # the holdout's 239 returns are too few to resample, so it has no range
    assert held.stats.interval("max_drawdown_pct") is None
    assert frame(deep).loc["holdout", ["value", "why_none"]].tolist() == [
        held.stats["max_drawdown_pct"], "returns: 239 of them; a bootstrap of returns needs "
                                        "at least 500"]


def test_a_venue_built_by_hand_is_named_as_one():
    built = probatum.search(alternating, SPACE, bars(), probatum.Venue(), trials=2,
                            sampler="random", holdout=0.3, seed=1)

    assert list(built.summary().context.items())[:2] == [("policy", "alternating"),
                                                         ("venue", "built by hand")]


def picking(data, pick=None):
    return np.full(len(data.close), 1.0 if pick is None else pick)


def test_a_choice_of_none_or_nan_is_one_configuration_however_often_drawn():
    # the trials table holds a None among floats as NaN, and NaN equals nothing, not
    # even another trial's once a pickle has carried them
    for choices in ([None, 0.5, 1.5], [math.nan, 0.5, 1.5]):
        found = probatum.search(picking, {"pick": probatum.Categorical(choices)}, bars(),
                                "frictionless", trials=12, sampler="random", holdout=0.3,
                                seed=1)
        again = pickle.loads(pickle.dumps(found))

        for held in (found, again):
            shown = held.summary()._shown()[1]["value"]
            assert [shown[name] for name in ("trials", "distinct", "best pick")] == [
                "12", "3", str(held.best_params["pick"])]


def holding(data, pick=None):
    return np.full(len(data.close), 0.0 if pick == 0.0 else 1.0)


def test_a_winner_that_chose_none_or_nan_names_its_choice_and_keeps_it():
    # holding these bars beats holding nothing, so the winner is the choice other than 0
    for choice in (None, math.nan):
        found = probatum.search(holding, {"pick": probatum.Categorical([choice, 0.0])},
                                bars(), "frictionless", trials=12, sampler="random",
                                holdout=0.3, seed=1)
        given = found.summary()

        assert given._shown()[1]["value"]["best pick"] == str(choice)
        assert pickle.loads(pickle.dumps(found)).summary() == given


def test_a_break_even_and_a_decay_read_as_the_reference_reads_them(validated):
    assert stats.BREAK_EVEN == statistics.BREAK_EVEN and stats.GROWS == statistics.GROWS

    found = searched(objective="profit_factor").consistency()

    assert found.break_even == statistics.BREAK_EVEN["profit_factor"] == 1.0
    assert round(found.share * len(found.scores)) == statistics.gained(
        list(found.scores), "profit_factor", "maximize")

    windows, fits = validated.windows, validated.fits
    fitted = dict(zip(np.asarray(fits["number"]).tolist(),
                      np.asarray(fits["fitted"], dtype=np.float64).tolist()))
    gaps, mean = statistics.decay([fitted[fit] for fit in np.asarray(windows["fit"]).tolist()],
                                  np.asarray(windows["score"], dtype=np.float64).tolist(),
                                  "maximize")

    decayed = validated.decay()

    assert list(decayed.gaps) == pytest.approx(gaps, rel=1e-12)
    assert decayed.mean       == pytest.approx(mean, rel=1e-12)


def test_an_objective_with_no_break_even_gives_its_scores_and_no_share():
    found = searched(objective="win_rate").consistency()

    assert (found.break_even, found.share) == (None, None)
    assert frame(searched(objective="win_rate")).loc["overfitting", "why_none"] == (
        "win_rate is not a statistic the halves give whatever the order of their returns")


def test_a_validation_reads_its_windows_and_its_fits(validated):
    windows, fits = validated.windows, validated.fits
    scores = np.asarray(windows["score"], dtype=np.float64)
    found  = validated.consistency()

    assert found.scores == tuple(scores) and found.unit == "windows"

    fitted = dict(zip(np.asarray(fits["number"]).tolist(),
                      np.asarray(fits["fitted"], dtype=np.float64).tolist()))
    made = [fitted[fit] for fit in np.asarray(windows["fit"]).tolist()]
    gaps = [fit - score for fit, score in zip(made, scores.tolist())]
    decayed = validated.decay()

    assert decayed.window == tuple(range(len(windows)))
    assert decayed.first  == tuple(np.asarray(windows["first_time"]).tolist())
    assert decayed.last   == tuple(np.asarray(windows["last_time"]).tolist())
    assert decayed.fitted == tuple(made) and decayed.earned == tuple(scores.tolist())
    assert decayed.gaps   == tuple(gaps) and decayed.mean == sum(gaps) / len(gaps)
    assert decayed.error  == pytest.approx(np.std(gaps, ddof=1) / math.sqrt(len(gaps)),
                                           rel=1e-12)

    with pytest.raises(ProbatumValueError, match="stretches: given to a validation"):
        validated.consistency(stretches=4)
    with pytest.raises(ProbatumValueError, match="decay: total_return_pct grows with "
                       "the span"):
        searched(objective="total_return_pct").decay()


def test_an_even_count_of_paths_reads_the_lower_middle_one_and_its_costs():
    done = probatum.validate(Flip, bars(), probatum.venues.binance_perp.replace(
        funding="none"), probatum.cv.CombinatorialPurged(5, 2), space=SPACE,
        search=dict(trials=4, sampler="random"), holdout=0.2, seed=1)

    paths  = done.paths
    scores = np.asarray(paths["score"], dtype=np.float64)

    assert len(scores) == 4 and not np.isnan(scores).any()

    lower   = int(np.argsort(scores, kind="stable")[1])
    returns = done.returns[lower][~np.isnan(done.returns[lower])]
    held    = stats.sharpe_interval(returns,
                                    periods_per_year=done.provenance["periods_per_year"])

    shown = frame(done)

    assert shown.loc["score", ["value", "low", "high"]].tolist() == [
        float(np.median(scores)), held.low, held.high]

    paid = sum(float(np.asarray(paths[name], dtype=np.float64).sum())
               for name in ("fees", "funding", "spread", "slippage", "impact",
                            "liquidation", "dust"))
    net = float((np.asarray(paths["net_profit_pct"], dtype=np.float64) * 100.0).sum())

    assert paid > 0.0
    assert shown.loc["costs", "value"] == pytest.approx(paid, rel=1e-12)
    assert shown.loc["price_pnl", "value"] == pytest.approx(net + paid, rel=1e-12)


def test_a_validation_s_summary_names_its_paths_and_its_median_path(validated):
    found = validated.summary()
    shown = found.to_pandas()
    paths = validated.paths
    held  = stats.sharpe_interval(validated.returns[0][~np.isnan(validated.returns[0])],
                                  periods_per_year=validated.provenance["periods_per_year"])
    decay = validated.decay()

    assert list(shown.index) == ["paths", "score", "windows", "windows_gained", "decay",
                                 "decay_error", "costs", "price_pnl", "holdout",
                                 "buy_and_hold"]
    assert shown.loc["score", ["value", "low", "high"]].tolist() == [
        float(paths["score"][0]), held.low, held.high]
    assert shown.loc[["decay", "decay_error"], "value"].tolist() == [decay.mean, decay.error]
    assert list(found.context.items()) == [
        ("policy", "Flip"), ("venue", "frictionless"), ("data", "BTCUSDT 1h"),
        ("objective", "sharpe"), ("direction", "maximize"), ("splitter", "PurgedKFold(4)"),
        ("fits", 4), ("embargo", 0), ("first", "1970-01-01"), ("last", "1970-01-27"),
        ("held out", "the last 20% (160 bars)"), ("looks", 0)]
