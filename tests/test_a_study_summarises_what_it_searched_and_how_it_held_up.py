"""A Study says in a few lines what it searched and how the winner held up: a search's
summary names its winner, its deflated Sharpe beside a random null's mean, its
overfitting probability and its holdout as it stands, and a validation's its paths,
their median path's interval and the share of the gross edge its costs took;
consistency scores the winner in stretches, or a validation's windows, against the
objective's break-even, and decay sets what each fit scored against what its bars then
earned (Specification, Summaries; Consistency and decay).
"""

import math
import textwrap

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


def line(label, said):
    """A summary's line as a Study lays it out, wrapped at 88 columns beneath itself."""
    return textwrap.fill(f"  {label:<18}{said}", width=88, subsequent_indent=" " * 20,
                         break_long_words=False, break_on_hyphens=False).split("\n")


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


def test_a_search_s_summary_names_its_winner_its_null_and_its_holdout(study):
    null        = searched(seed=2)
    lines       = study.summary(null=null).split("\n")
    trials      = null.trials
    done        = np.asarray(trials["state"]) == "complete"
    mean        = np.asarray(trials["score"], dtype=np.float64)[done].mean()
    deflated    = str(stats.deflated_sharpe(study, null)).removeprefix("deflated sharpe ")
    overfitting = str(stats.overfitting(study)).removeprefix("overfitting ")

    assert lines == [
        "search: alternating, frictionless, BTCUSDT 1h, 12 random trials",
        "in sample 1970-01-01 to 1970-01-24, holdout the last 30% (240 bars)",
        f"  best              fast {study.best_params['fast']}: sharpe "
        f"{study.best_score:.2f}",
        *line("deflated sharpe", deflated),
        f"  null              mean sharpe {mean:.2f} over {done.sum()} trials",
        *line("overfitting", overfitting),
        "  holdout           not yet run; evaluate_holdout() spends a look"]


def test_a_search_says_why_it_has_no_deflated_sharpe_or_overfitting_probability():
    kept = searched(keep="none")

    assert line("overfitting", "none: the search kept no returns")[0] in kept.summary()

    ordered = searched(objective="calmar")
    shown   = ordered.summary(null=searched(seed=2, objective="calmar"))

    assert line("deflated sharpe", "none: the search ranked on calmar, and a deflation "
                "is of a Sharpe a search maximised")[0] in shown
    assert line("overfitting", "none: calmar is not a statistic the halves give whatever "
                "the order of their returns")[0] in shown

    lowest = searched(direction="minimize").summary(
        null=searched(seed=2, direction="minimize"))

    assert line("deflated sharpe", "none: the search minimised sharpe, and a deflation is "
                "of a Sharpe a search maximised")[0] in lowest

    alone = searched(trials=1).summary(null=searched(seed=2))

    assert line("deflated sharpe", "none: study: 1 trial; a deflation corrects for more "
                "than one, so search with trials=2 or more")[0] in alone
    assert line("overfitting", "none: 1 complete trial")[0] in alone

    short = searched(trials=2, holdout=800 - 16)

    assert line("overfitting", "none: 15 returns, fewer than its 16 blocks")[0] in (
        short.summary())


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
        shown = moved.summary(null=null)
    finally:
        LAG[0] = 0

    assert "  deflated sharpe   none: best: trial " in shown
    assert shown.split("\n")[-1].startswith("  holdout           not yet run")


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

    assert fresh.summary().split("\n")[-1] == (
        f"  holdout           sharpe {held.stats['sharpe']:.2f}, 95% interval "
        f"{looked.low:.2f} to {looked.high:.2f}, look 1 of 1")

    gap = fresh.best_score - held.stats["sharpe"]
    assert fresh.decay() == stats.Decay([gap], gap, "sharpe", "maximize")

    raised = searched(policy=raising_on_the_tail)

    with pytest.raises(RuntimeError, match="the tail"):
        raised.evaluate_holdout()

    assert raised.summary().endswith("holdout           read 1 time; its run raised, so "
                                     "no result is kept")

    with pytest.raises(ProbatumValueError, match="decay: the holdout was read 1 time and "
                       "its run raised, so no result is kept; evaluate_holdout\\(\\) runs it "
                       "again, and spends a look"):
        raised.decay()

    whole = searched(holdout=0)

    assert whole.summary().split("\n")[1].endswith(", every bar searched, none held out")
    assert whole.summary().endswith("holdout           none: the search held every bar")

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
    assert str(found).startswith(f"{past} of 4 stretches scored above 0: ")
    assert len(study.consistency(stretches=6).scores) == 6

    with pytest.raises(ProbatumValueError, match="stretches: 1; give a whole number from "
                       "2"):
        study.consistency(stretches=1)


def test_a_drawdown_objective_reads_as_the_loss_it_is():
    deep  = searched(objective="max_drawdown_pct", direction="minimize")
    lines = deep.summary().split("\n")

    assert lines[2].endswith(f"max_drawdown_pct {-deep.best_score:.2f}")
    assert f"best max_drawdown_pct {-deep.best_score:,.2f} over" in repr(deep)

    found = deep.consistency()

    assert str(found).endswith(", ".join(f"{-score:.2f}" for score in found.scores))

    held = deep.evaluate_holdout()

    # the holdout's 239 returns are too few to resample, so its line has no range
    assert held.stats.interval("max_drawdown_pct") is None
    assert deep.summary().split("\n")[-1] == (
        f"  holdout           max_drawdown_pct {-held.stats['max_drawdown_pct']:.2f}, look "
        f"1 of 1")

    fall = stats.Interval("max_drawdown_pct", 10.0, 5.0, 20.0, 0.95,
                          "stationary block bootstrap", 1_000, kind="range")
    assert probatum._search._bounds(fall, -1.0) == "-20.00 to -5.00"


def test_a_venue_built_by_hand_is_named_as_one():
    built = probatum.search(alternating, SPACE, bars(), probatum.Venue(), trials=2,
                            sampler="random", holdout=0.3, seed=1)

    assert built.summary().startswith("search: alternating, a venue built by hand, ")


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
    assert str(found).startswith("win_rate has no break-even, so no part gains or loses")


def test_a_validation_reads_its_windows_and_its_fits(validated):
    windows, fits = validated.windows, validated.fits
    scores = np.asarray(windows["score"], dtype=np.float64)
    found  = validated.consistency()

    assert found.scores == tuple(scores) and found.unit == "windows"

    fitted = dict(zip(np.asarray(fits["number"]).tolist(),
                      np.asarray(fits["fitted"], dtype=np.float64).tolist()))
    gaps = [fitted[fit] - score
            for fit, score in zip(np.asarray(windows["fit"]).tolist(), scores.tolist())]
    assert validated.decay() == stats.Decay(gaps, sum(gaps) / len(gaps), "sharpe",
                                            "maximize")

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

    shown = " ".join(done.summary().split())
    assert (f"the median path's 95% interval {held.low:.2f} to {held.high:.2f}, and the "
            f"paths share bars") in shown

    paid = sum(float(np.asarray(paths[name], dtype=np.float64).sum())
               for name in ("fees", "funding", "spread", "slippage", "impact",
                            "liquidation", "dust"))
    net = float((np.asarray(paths["net_profit_pct"], dtype=np.float64) * 100.0).sum())
    assert paid > 0.0
    assert (f"costs took {paid / (net + paid):.0%} of the gross edge" if net + paid > 0.0
            else f"costs costs {paid:,.0f}, with no gross edge") in shown


def test_a_validation_s_summary_names_its_paths_and_its_median_path(validated):
    lines = validated.summary().split("\n")
    paths = validated.paths
    score = float(paths["score"][0])
    held  = stats.sharpe_interval(validated.returns[0][~np.isnan(validated.returns[0])],
                                  periods_per_year=validated.provenance["periods_per_year"])

    assert lines[0] == ("validate: Flip, frictionless, BTCUSDT 1h, holdout the last 20% "
                        "(160 bars)")
    assert lines[3] == "  embargo           0 bars after each test span"
    assert lines[4] == f"  sharpe            {score:.2f}"
    assert " ".join(lines[5:7]).split() == (
        f"median {score:.2f}; the median path's 95% interval {held.low:.2f} to "
        f"{held.high:.2f}, and the paths share bars").split()
    assert "  evidence          the paths are stability evidence; the verdict comes from "\
           "the holdout" in lines
    assert lines[-1] == "  holdout           not yet run; evaluate_holdout() spends a look"
