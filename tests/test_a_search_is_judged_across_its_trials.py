"""A search is judged across all of its trials: its winner's Sharpe deflated by what the
best of every trial it asked reaches by luck, against every trial and against the
effective number its clusters make, the backtest length that many trials need, and the
share of the trials near its winner that score past a level, so a spike reads as one.
Each reads a search's Study or a matrix of trials' returns, the plateau a Study alone,
and refuses what it cannot read (ADR 0033; Specification, Across a search).
"""

import math
import warnings

import numpy as np
import pytest

import probatum
from probatum import ProbatumTypeError, ProbatumValueError, ProbatumWarning, stats

from reference import statistics

HOUR  = 3_600_000_000_000
SPACE = {"fast": probatum.Int(2, 30), "lag": probatum.Int(0, 9)}


def bars(rows=800):
    close  = 100.0 + np.random.default_rng(9).normal(0.0, 1.0, rows).cumsum()
    shaped = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(rows, 1e3)])
    return probatum.Data(
        shaped,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * HOUR,
    )


def alternating(data, fast=5, lag=0):
    return np.where((np.arange(len(data.close)) + lag) % (2 * fast) < fast, 1.0, 0.0)


def searched(seed=1, **given):
    settings = dict(trials=24, sampler="random", holdout=0.3, seed=seed)
    settings.update(given)
    return probatum.search(alternating, SPACE, bars(), "frictionless", **settings)


def blocks(seed=3, trials=12, bars=200):
    """Three blocks of trials, each sharing a factor of its own."""
    rng    = np.random.default_rng(seed)
    factor = rng.normal(size=(3, bars))
    return np.array([factor[i * 3 // trials] + rng.normal(scale=0.3, size=bars)
                     for i in range(trials)]) * 0.01


@pytest.fixture(scope="module")
def study():
    return searched()


@pytest.fixture(scope="module")
def null():
    return searched(seed=2)


def test_the_threshold_is_the_spread_times_the_expected_best_of_that_many_trials():
    assert stats.deflation_threshold(0.5, 100) == pytest.approx(
        0.5 * statistics.expected_maximum(100), rel=1e-12)
    assert stats.deflation_threshold(0.0, 7) == 0.0


def test_a_threshold_needs_two_trials_and_a_spread_of_0_or_more():
    with pytest.raises(ProbatumValueError, match="trials: 1; a deflation corrects for more "
                       "than one trial, so give 2 or more"):
        stats.deflation_threshold(0.5, 1)
    with pytest.raises(ProbatumTypeError, match="trials: 2.5; give a whole number"):
        stats.deflation_threshold(0.5, 2.5)
    with pytest.raises(ProbatumValueError, match="spread: -0.1; give the annual Sharpes' "
                       "deviation, 0 or more"):
        stats.deflation_threshold(-0.1, 10)


def test_a_backtest_length_is_the_years_the_expected_best_takes_to_fall_to_a_sharpe():
    held = stats.min_backtest_length(46, 1.0)

    assert held.years == pytest.approx(statistics.min_backtest_length(46, 1.0), rel=1e-12)
    assert (held.trials, held.sharpe, held.series_years) == (46, 1.0, None)
    assert str(held) == ("minimum backtest length 5.0 years for the best of 46 trials to "
                         "fall to a sharpe of 1.00")

    with pytest.raises(ProbatumTypeError, match="sharpe: missing"):
        stats.min_backtest_length(46)
    with pytest.raises(ProbatumValueError, match="sharpe: 0.0; give an annual Sharpe above 0"):
        stats.min_backtest_length(46, 0.0)


def test_a_search_s_backtest_length_reads_its_trials_and_its_winner(study):
    held   = stats.min_backtest_length(study)
    trials = study.trials
    winner = np.asarray(trials["number"]) == study._winner.number

    assert held.trials == len(trials)
    assert held.sharpe == float(np.asarray(trials["sharpe"])[winner][0])
    assert held.series_years == pytest.approx(
        (len(study.best.time) - 1) / study.provenance["periods_per_year"])
    assert str(held).endswith(f"; this series has {held.series_years:.2f}")

    with pytest.raises(ProbatumTypeError, match="sharpe: 1.0 beside a Study"):
        stats.min_backtest_length(study, 1.0)


def test_the_effective_number_clusters_the_trials_that_share_their_returns():
    rows = blocks()
    held = stats.effective_trials(rows, periods_per_year=252)

    assert (held.count, held.trials, held.dropped) == (3, 12, 0)
    assert held.clusters == (0,) * 4 + (1,) * 4 + (2,) * 4
    assert held.numbers  == tuple(range(12))

    series = [statistics.combined(rows[np.asarray(held.clusters) == label].tolist())
              for label in range(3)]

    assert held.spread == pytest.approx(
        statistics.deviation([statistics.sharpe(s, 252) for s in series], 1), rel=1e-9)
    assert str(held) == (f"3 effective trials of 12, quality {held.quality:.2f}, their "
                         f"clusters' sharpes spread {held.spread:.2f} a year")


def test_a_trial_that_never_varies_is_left_out_and_two_are_their_own_clusters():
    rows = np.vstack([blocks(), np.full(200, 0.001)])
    held = stats.effective_trials(rows, periods_per_year=252)

    assert (held.trials, held.dropped, held.numbers) == (13, 1, tuple(range(12)))

    pair = stats.effective_trials(blocks()[:2], periods_per_year=252)

    assert (pair.count, pair.clusters, pair.quality) == (2, (0, 1), None)

    one = stats.effective_trials(blocks()[:1], periods_per_year=252)

    assert (one.count, one.spread) == (1, None)
    assert str(one) == "1 effective trial of 1"


def test_a_study_s_effective_number_reads_its_trials_that_ran_and_is_kept(study):
    held = stats.effective_trials(study)

    assert stats.effective_trials(study) is held

    ran = np.asarray(study.trials["state"]) != "failed"

    assert held.trials == int(ran.sum())
    assert set(held.numbers) <= set(np.asarray(study.trials["number"])[ran].tolist())

    with pytest.raises(ProbatumTypeError, match="periods_per_year: given beside a Study"):
        stats.effective_trials(study, periods_per_year=252)


def test_the_deflated_sharpe_is_the_winner_s_probabilistic_sharpe_against_luck(study,
                                                                               null):
    found   = stats.deflated_sharpe(study, null)
    table   = null.trials
    ran     = np.asarray(table["state"]) != "failed"
    sharpes = np.asarray(table["sharpe"], dtype=np.float64)[ran]
    spread  = float(np.std(sharpes[np.isfinite(sharpes)], ddof=1))

    assert (found.trials, found.null_trials) == (len(study.trials), int(ran.sum()))
    assert found.spread == pytest.approx(spread, rel=1e-12)
    assert found.threshold == pytest.approx(stats.deflation_threshold(spread, 24),
                                            rel=1e-12)

    every = stats.probabilistic_sharpe(study.best, benchmark=found.threshold)

    assert (found.value, found.lag, found.floored) == (every.value, every.lag,
                                                       every.floored)

    held = stats.effective_trials(study)
    bar  = 0.0 if held.spread is None else stats.deflation_threshold(held.spread,
                                                                     held.count)

    assert (found.effective, found.effective_threshold) == (held.count,
                                                            pytest.approx(bar))

    # the winner's cluster is judged, its members combined, where the winner's own
    # Sharpe beside the clusters' spread would leave out the choice inside its cluster
    rate    = study.provenance["periods_per_year"]
    excess  = np.asarray(study.returns)[:, 1:] - study.provenance["risk_free"] / rate
    labels  = dict(zip(held.numbers, held.clusters))
    done    = np.asarray(study.trials["state"]) == "complete"
    scores  = np.asarray(study.trials["score"], dtype=np.float64)
    winner  = int(np.asarray(study.trials["number"])[done][np.argmax(scores[done])])
    members = [list(excess[n]) for n in held.numbers if labels[n] == labels[winner]]

    assert len(members) > 1
    assert found.effective_value == pytest.approx(statistics.deflated_sharpe(
        statistics.combined(members), rate, held.spread, held.count), rel=1e-9)
    assert str(found) == (f"deflated sharpe {found.value:.2f} against all 24 trials; "
                          f"{found.effective_value:.2f} for the winner's cluster against "
                          f"{held.count} effective trial{'s' * (held.count != 1)}")


def test_a_floor_keeps_no_trial_out_of_the_count_or_out_of_the_null_s_spread():
    # a slow alternation closes few round trips, so a floor of 20 binds over much of
    # the space, in the search and in its null alike
    floored = searched(min_trades=20)
    null    = searched(seed=2, min_trades=20)
    states  = np.asarray(null.trials["state"])

    assert "floored" in set(states) and "floored" in set(np.asarray(floored.trials["state"]))

    found   = stats.deflated_sharpe(floored, null)
    sharpes = np.asarray(null.trials["sharpe"], dtype=np.float64)[states != "failed"]

    assert (found.trials, found.null_trials) == (24, int((states != "failed").sum()))
    assert found.spread == pytest.approx(float(np.std(sharpes, ddof=1)), rel=1e-12)

    complete = int((np.asarray(floored.trials["state"]) == "complete").sum())

    assert stats.overfitting(floored).trials == complete < 24


def test_a_search_that_kept_no_returns_is_deflated_against_every_trial_alone(null):
    kept  = searched(keep="none")
    found = stats.deflated_sharpe(kept, null)

    assert (found.effective, found.effective_value, found.effective_threshold) == (
        None, None, None)
    assert str(found) == f"deflated sharpe {found.value:.2f} against all 24 trials"

    with pytest.raises(ProbatumValueError, match="trials: a Study kept with keep=\"none\""):
        stats.effective_trials(kept)


def test_the_deflated_sharpe_refuses_what_it_cannot_deflate(study, null):
    with pytest.raises(ProbatumValueError, match="study: a search that ranked on sortino "
                       "to maximize it; the deflation is of a Sharpe a search maximised"):
        stats.deflated_sharpe(searched(objective="sortino"), null)
    with pytest.raises(ProbatumValueError, match="null: its objective"):
        stats.deflated_sharpe(study, searched(seed=2, objective="sortino"))
    with pytest.raises(ProbatumTypeError, match="null: list; give the Study of a random "
                       "search"):
        stats.deflated_sharpe(study, [])
    with pytest.raises(ProbatumTypeError, match="study: list; give the Study of a search"):
        stats.deflated_sharpe([], null)
    with pytest.raises(ProbatumValueError, match="study: a search that ranked on sharpe "
                       "to minimize it; the deflation is of a Sharpe a search maximised, "
                       "so search with direction=\"maximize\""):
        stats.deflated_sharpe(searched(direction="minimize"), null)
    with pytest.raises(ProbatumValueError, match="study: a search that ranked on "
                       "max_drawdown_pct to minimize it; the deflation is of a Sharpe a "
                       "search maximised, so search with objective=\"sharpe\" and "
                       "direction=\"maximize\""):
        stats.deflated_sharpe(searched(objective="max_drawdown_pct", direction="minimize"),
                              null)


def test_a_null_smaller_than_the_search_warns_that_its_spread_rests_on_fewer_draws(
        study, null):
    with pytest.warns(ProbatumWarning, match="^null: a search of 12 trials beside the 24 "
                      "this one asked, so the spread that deflates it rests on fewer draws "
                      "than the trials it corrects for; search the null with trials=24$"):
        stats.deflated_sharpe(study, searched(seed=2, trials=12))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats.deflated_sharpe(study, null)


def test_an_adaptive_search_is_deflated_and_called_descriptive(null):
    pytest.importorskip("optuna")

    adaptive = searched(sampler="tpe", trials=12)
    found    = stats.deflated_sharpe(adaptive, null)

    assert found.descriptive
    assert str(found).endswith(", descriptive")
    assert stats.overfitting(adaptive).descriptive
    assert stats.reality_check(adaptive, resamples=20).descriptive


def test_the_overfitting_probability_ranks_a_search_s_complete_trials(study):
    found    = stats.overfitting(study)
    count    = len(study.best.time) - 1
    complete = int((np.asarray(study.trials["state"]) == "complete").sum())

    assert (found.trials, found.blocks, found.dropped) == (complete, 16, count % 16)
    assert (found.objective, found.direction, found.descriptive) == ("sharpe", "maximize",
                                                                     False)
    assert 0.0 <= found.probability <= 1.0 and 0.0 <= found.loss <= 1.0
    assert str(found) == (f"overfitting {found.probability:.2f} probability the in-sample "
                          f"best ranks below the median on the held-back blocks (16 "
                          f"blocks, {complete} trials); it loses on {found.loss:.0%} of "
                          f"them")


def test_trials_with_no_skill_overfit_about_half_the_halvings_and_skill_none():
    # one set of trials varies widely about a half, its halvings sharing every bar, so
    # ten are averaged
    found = [stats.overfitting(np.random.default_rng(seed).normal(0.0, 0.01, (20, 320)),
                               periods_per_year=252).probability for seed in range(10)]

    assert 0.35 < np.mean(found) < 0.65

    noise = np.random.default_rng(10).normal(0.0, 0.01, (20, 320))
    noise[7] += 0.004
    skilled = stats.overfitting(noise, periods_per_year=252)

    assert (skilled.probability, skilled.loss) == (0.0, 0.0)

    calm = stats.overfitting(noise, "volatility_pct", periods_per_year=252)

    assert (calm.direction, calm.loss) == ("minimize", None)
    assert str(calm).endswith("(16 blocks, 20 trials)")


def test_a_winner_that_breaks_even_out_of_sample_neither_loses_nor_overfits():
    # a trial that never trades beside three that lose: it wins every half at a Sharpe
    # of 0 and ranks first on the other
    rows    = np.random.default_rng(2).normal(-0.01, 0.01, (4, 64))
    rows[0] = 0.0
    found   = stats.overfitting(rows, periods_per_year=252)

    assert (found.probability, found.loss) == (0.0, 0.0)


def test_the_halvings_read_a_chunk_at_a_time_give_what_one_pass_gives(monkeypatch):
    # 20 trials read their 6,435 halvings and complements in one chunk; a cap of 100
    # statistics reads them 5 at a time, in 1,287 chunks
    rows  = np.random.default_rng(11).normal(0.0002, 0.01, (20, 160))
    whole = stats.overfitting(rows, periods_per_year=252)

    monkeypatch.setattr(stats, "AT_ONCE", 100)
    parts = stats.overfitting(rows, periods_per_year=252)

    assert (parts.probability, parts.loss) == (whole.probability, whole.loss)


def test_the_overfitting_probability_refuses_order_and_what_halves_cannot_read(study):
    rows = np.random.default_rng(1).normal(0.0, 0.01, (4, 64))

    for objective in ("max_drawdown_pct", "calmar", "value_at_risk", "win_rate"):
        with pytest.raises(ProbatumValueError, match=f"objective: '{objective}'; the "
                           f"overfitting probability ranks on a statistic the halves "
                           f"give whatever the order of their returns"):
            stats.overfitting(rows, objective, periods_per_year=252)

    with pytest.raises(ProbatumValueError, match="trials: 15 returns; the overfitting "
                       "probability cuts them into 16 blocks, so give at least 16"):
        stats.overfitting(rows[:, :15], periods_per_year=252)
    with pytest.raises(ProbatumValueError, match="trials: 1 complete; the overfitting "
                       "probability ranks a winner among 2 or more"):
        stats.overfitting(rows[:1], periods_per_year=252)
    with pytest.raises(ProbatumTypeError, match="objective: 'sortino' beside a Study"):
        stats.overfitting(study, "sortino")
    with pytest.raises(ProbatumValueError, match="trials: a Study kept with keep=\"none\""):
        stats.overfitting(searched(keep="none"))
    with pytest.raises(ProbatumTypeError, match="periods_per_year: missing for a matrix "
                       "of returns"):
        stats.overfitting(rows)
    with pytest.raises(ProbatumValueError, match="trials: trial 2 holds nan at position 5"):
        broken = rows.copy()
        broken[2, 5] = np.nan
        stats.overfitting(broken, periods_per_year=252)


def test_the_reality_check_tests_every_trial_that_ran_against_cash(study):
    found = stats.reality_check(study, resamples=200)
    ran   = int((np.asarray(study.trials["state"]) != "failed").sum())

    assert (found.trials, found.excluded, found.benchmark) == (ran, 0, "cash")
    assert 0.0 <= found.p <= 1.0 and 0.0 <= found.spa <= 1.0
    assert stats.reality_check(study, resamples=200) == found
    assert str(found) == (f"reality check p {found.p:.2f} and SPA p {found.spa:.2f} that no "
                          f"trial beats cash, 200 stationary resamples in blocks of "
                          f"{found.block:.1f}, {ran} trials")

    held = stats.reality_check(study, "buy_and_hold", resamples=50)

    assert held.benchmark == "buying and holding"


def test_spa_is_0_for_a_trial_ahead_on_every_bar_and_1_for_none_ahead_at_all():
    rows = np.random.default_rng(4).normal(-0.002, 0.01, (6, 300))

    assert stats.reality_check(rows, periods_per_year=252, resamples=50).spa == 1.0

    rows[2] = 0.001
    found = stats.reality_check(rows, periods_per_year=252, resamples=50)

    assert (found.spa, found.spa_statistic, found.best) == (0.0, math.inf, 2)


def test_a_trial_that_lost_a_whole_bar_is_left_out_and_the_benchmark_is_checked():
    rows = np.random.default_rng(6).normal(0.0, 0.01, (4, 120))
    rows[1, 30] = -1.0
    given = np.random.default_rng(7).normal(0.0, 0.01, 120)
    found = stats.reality_check(rows, given, periods_per_year=252, resamples=50)

    assert (found.trials, found.excluded, found.benchmark) == (3, 1, "the benchmark")

    with pytest.raises(ProbatumValueError, match="benchmark: \"buy_and_hold\" beside a "
                       "matrix, which holds no instruments to buy"):
        stats.reality_check(rows, "buy_and_hold", periods_per_year=252)
    with pytest.raises(ProbatumValueError, match="benchmark: 'cash'; give "
                       "\"buy_and_hold\", a series of returns, or None for cash"):
        stats.reality_check(rows, "cash", periods_per_year=252)
    with pytest.raises(ProbatumValueError, match="benchmark: 119 returns beside 120"):
        stats.reality_check(rows, given[1:], periods_per_year=252)
    with pytest.raises(ProbatumTypeError, match="^benchmark: float; give \"buy_and_hold\", "
                       "a series of returns, or None for cash$"):
        stats.reality_check(rows, 0.0, periods_per_year=252)

    ruined = given.copy()
    ruined[90] = -1.0

    with pytest.raises(ProbatumValueError, match="^benchmark: position 90 holds -1.0, all "
                       "of a bar lost, which leaves no log growth to measure the trials "
                       "against$"):
        stats.reality_check(rows, ruined, periods_per_year=252)

    with pytest.raises(ProbatumValueError, match="trials: 2 returns; the Reality Check "
                       "needs 3"):
        stats.reality_check(rows[:, :2], periods_per_year=252)


def test_cash_earns_the_risk_free_rate_a_bar():
    rows   = np.random.default_rng(8).normal(0.0005, 0.01, (4, 120))
    cash   = stats.reality_check(rows, periods_per_year=252, risk_free=0.05, resamples=50)
    series = stats.reality_check(rows, np.full(120, 0.05 / 252), periods_per_year=252,
                                 risk_free=0.05, resamples=50)

    assert (cash.p, cash.spa, cash.statistic) == (series.p, series.spa, series.statistic)


def plateaued(study, ranges, level, maximize=True):
    """The reference's plateau of ``study``'s trials, each parameter's range in
    ``ranges``."""
    names  = [name for name in study.provenance["space"]]
    trials = [(tuple(row[name] for name in names), row["state"], row["score"])
              for row in study.trials]
    best   = [row["number"] for row in study.trials].index(study._winner.number)

    return statistics.plateau(trials, ranges, best, level, maximize)


def test_the_plateau_is_the_share_of_the_trials_near_the_winner_past_a_level(study):
    level = float(np.nanmedian(study.trials["score"]))
    found = stats.plateau(study, level)
    held  = plateaued(study, [(2, 30, False), (0, 9, False)], level)

    assert [None if math.isnan(share) else share for share in found.share] == held
    assert found.radii == tuple(statistics.RADII) and found.level == level
    assert sum(found.trials) > 0 and study.plateau(level) == found


def test_a_plateau_reads_a_log_scale_and_a_choice_and_a_search_that_minimises():
    def shaped(data, fast, kind, scale):
        weights = alternating(data, fast)
        return weights * scale if kind == "long" else -weights * scale

    space = {
        "fast":  probatum.Int(2, 64, log=True),
        "kind":  probatum.Categorical(["long", "short"]),
        "scale": probatum.Float(0.5, 2.0, log=True),
    }

    study = probatum.search(shaped, space, bars(), "frictionless", trials=40,
                            sampler="random", holdout=0.3, seed=4,
                            objective="max_drawdown_pct", direction="minimize")
    level = float(np.nanmedian(study.trials["score"]))
    found = stats.plateau(study, level)
    held  = plateaued(study, [(2, 64, True), ("categorical",), (0.5, 2.0, True)], level,
                      maximize=False)

    assert [None if math.isnan(share) else share for share in found.share] == held
    assert sum(found.trials) > 0
    # a drawdown prints as the loss it is, and below a level becomes above its loss
    assert f"score above {-level + 0.0:g}" in str(found)


def test_a_trial_at_a_radius_s_edge_lies_within_it():
    # a grid of Int(0, 4) puts its values a quarter apart, so the winner's neighbours
    # lie at 0.25 exactly, the fifth radius
    study = probatum.search(alternating, {"lag": probatum.Int(0, 4)}, bars(),
                            "frictionless", trials=5, sampler="grid", holdout=0.3,
                            seed=1)
    found = stats.plateau(study, -1e9)
    held  = plateaued(study, [(0, 4, False)], -1e9)
    assert [None if math.isnan(share) else share for share in found.share] == held
    assert found.trials[3] == 0 and found.trials[4] >= 1


def test_a_choice_of_none_among_numbers_is_the_winner_s_where_the_winner_chose_it():
    # the trials table holds a None among floats as NaN, which is still one choice
    def stopped(data, lag, stop):
        weights = alternating(data, 5, lag)
        return {"policy": weights if stop is None else 0.0 * weights, "stop": stop}

    space = {
        "lag":  probatum.Int(0, 9),
        "stop": probatum.Categorical([None, 0.02, 0.05]),
    }

    study = probatum.search(stopped, space, bars(), "frictionless", trials=30,
                            sampler="grid", holdout=0.3, seed=1)
    found = stats.plateau(study, -1e9)
    held  = plateaued(study, [(0, 9, False), ("categorical",)], -1e9)

    assert study.best_params == {"lag": 5, "stop": None}
    assert found.trials      == (0, 0, 2, 2, 4, 4, 6, 6, 8, 8)
    assert [None if math.isnan(share) else share for share in found.share] == held


def test_a_configuration_run_again_is_one_neighbour_and_the_winner_s_is_none():
    study = probatum.search(alternating, {"lag": probatum.Int(0, 4)}, bars(),
                            "frictionless", trials=20, sampler="random", holdout=0.3,
                            seed=2)
    lags  = study.trials["lag"].tolist()
    won   = study.best_params["lag"]
    found = stats.plateau(study, -1e9)
    held  = plateaued(study, [(0, 4, False)], -1e9)
    assert lags.count(won) > 1 and found.trials[0] == 0
    assert found.trials[-1] == len({lag for lag in lags if 0 < abs(lag - won) <= 2})
    assert [None if math.isnan(share) else share for share in found.share] == held


def test_a_whole_number_past_a_float_s_precision_keeps_its_own_place():
    # 2**60 and the four after it are one float64, and five configurations
    low   = 2**60
    study = probatum.search(alternating, {"lag": probatum.Int(low, low + 4)}, bars(),
                            "frictionless", trials=5, sampler="grid", holdout=0.3,
                            seed=1)
    found = stats.plateau(study, -1e9)
    held  = plateaued(study, [(low, low + 4, False)], -1e9)
    assert [None if math.isnan(share) else share for share in found.share] == held
    assert found.trials[3] == 0 and found.trials[4] >= 1


def test_a_plateau_reads_as_the_share_at_0_1_and_then_at_every_radius():
    shares = [math.nan, 0.5] + [0.25] * 8
    found  = stats.Plateau(stats.RADII, [0, 2] + [4] * 8, [0, 1] + [1] * 8, shares, 1.0,
                           "sharpe", "maximize")

    assert str(found) == ("50% of the 2 trials within distance 0.1 of the best, in "
                          "normalised parameters, score above 1; from 0.05 to 0.5: none, "
                          "50%, 25%, 25%, 25%, 25%, 25%, 25%, 25%, 25%")

    alone = stats.Plateau(stats.RADII, [0] * 10, [0] * 10, [math.nan] * 10, 1.0,
                          "sharpe", "maximize")

    assert str(alone).startswith("no trial within distance 0.1 of the best, in "
                                 "normalised parameters, to score above 1; from 0.05 to "
                                 "0.5: none, none")


def test_a_search_s_summary_prints_its_plateau_and_a_validation_s_refuses_one(study):
    # its line wrapped beneath itself, after the overfitting probability's and before
    # the holdout's
    level = float(np.nanmedian(study.trials["score"]))
    lines = [line.strip() for line in study.summary(plateau=level).splitlines()]
    first = next(k for k, line in enumerate(lines) if line.startswith("plateau "))
    last  = next(k for k, line in enumerate(lines) if line.startswith("holdout "))

    assert " ".join(" ".join(lines[first:last]).split()) == (
        f"plateau {study.plateau(level)}")
    assert any(line.startswith("overfitting ") for line in lines[:first])

    with pytest.raises(ProbatumTypeError, match=r"^plateau: '1'; give a score in the "
                       r"objective's units, a finite number$"):
        study.summary(plateau="1")

    paths = probatum.validate(alternating, bars(), "frictionless",
                              probatum.cv.WalkForward(200, 100), space=SPACE,
                              search=dict(trials=4, sampler="random"), holdout=0.3,
                              seed=1)

    with pytest.raises(ProbatumValueError, match=r"^plateau: 1.0 given to a validation's "
                       r"summary, whose paths are no competing configurations; give it "
                       r"to its search's summary$"):
        paths.summary(plateau=1.0)
    with pytest.raises(ProbatumValueError, match="study: a validation's Study, whose "
                       "paths are no competing configurations"):
        stats.plateau(paths, 1.0)


@pytest.mark.parametrize("given, level, error, said", [
    (np.zeros((3, 10)), 1.0, ProbatumTypeError, "study: ndarray; give the Study of a "
                                                "search"),
    (None, math.nan, ProbatumValueError, "level: nan; give a score in the objective's "
                                         "units, a finite number"),
    (None, "high", ProbatumTypeError, "level: 'high'; give a score in the objective's "
                                      "units, a finite number"),
])
def test_a_plateau_refuses_what_is_no_search_and_a_level_that_is_no_score(study, given,
                                                                           level, error,
                                                                           said):
    with pytest.raises(error) as raised:
        stats.plateau(study if given is None else given, level)

    assert str(raised.value) == said
