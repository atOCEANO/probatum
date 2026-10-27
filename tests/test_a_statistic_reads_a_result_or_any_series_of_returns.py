"""A statistic reads a Result, at the annualisation and the risk-free rate it ran at, or
any series of returns by position, at the bars a year it is told, and refuses what it
cannot read or cannot support by name.

What it computes is held to the reference by the agreement test, and the numbers of
the Specification's worked example by the page's own test; these hold what it reads,
what it refuses and the shape of what it gives back.
"""

import math
import pickle

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import (ProbatumAttributeError, ProbatumTypeError, ProbatumValueError,
                      stats)

MONTHLY = [0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01]


@pytest.fixture(scope="module")
def result():
    close = 100.0 + np.random.default_rng(5).normal(0.0, 1.0, 400).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(400, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(400, dtype=np.int64) * 3_600_000_000_000,
    )

    weights = np.where(np.arange(400) % 40 < 20, 1.0, 0.0)
    return probatum.backtest(weights, data, "frictionless")


def test_a_plain_series_without_its_bars_a_year_is_refused_in_the_plan_s_words():
    for call in (stats.sharpe, stats.sharpe_interval, stats.probabilistic_sharpe,
                 stats.min_track_record_length):
        with pytest.raises(ProbatumTypeError, match="periods_per_year: missing for a plain "
                           "return series; pass 365 for daily crypto returns"):
            call(MONTHLY)


def test_a_result_refuses_an_annualisation_or_a_rate_it_names_itself(result):
    with pytest.raises(ProbatumTypeError, match="periods_per_year: given beside a "
                       "Result, which runs at 8760.0; leave it out"):
        stats.sharpe(result, periods_per_year=365)
    with pytest.raises(ProbatumTypeError, match="risk_free: given beside a Result, which "
                       "ran at 0.0; leave it out"):
        stats.sharpe_interval(result, risk_free=0.04)


def test_a_result_s_sharpe_is_its_record_s(result):
    assert stats.sharpe(result) == pytest.approx(result.stats["sharpe"], rel=1e-12)


def test_a_pandas_series_is_read_by_position_and_never_by_its_index():
    shuffled = pd.Series(MONTHLY, index=[7, 3, 5, 1, 0, 2, 6, 4])

    assert stats.sharpe_interval(shuffled, periods_per_year=12) == stats.sharpe_interval(
        np.array(MONTHLY), periods_per_year=12)


def test_a_nan_or_an_infinity_is_refused_at_its_first_position():
    with pytest.raises(ProbatumValueError, match="returns: position 2 holds nan; a "
                       "return series holds finite numbers"):
        stats.sharpe([0.01, 0.02, math.nan, math.inf], periods_per_year=12)


@pytest.mark.parametrize("given, said", [
    ({"key": 0.01}, "returns: dict; pass a Result, or a 1-D array"),
    ("0.01", "returns: str; pass a Result"),
    (np.zeros((3, 2)), r"returns: an array of shape \(3, 2\); pass one series, 1-D")])
def test_what_is_no_series_of_returns_is_refused_by_what_it_is(given, said):
    with pytest.raises((ProbatumTypeError, ProbatumValueError), match=said):
        stats.sharpe(given, periods_per_year=12)


@pytest.mark.parametrize("name", ["periods_per_year", "risk_free", "level", "benchmark",
                                  "confidence"])
def test_a_setting_that_is_no_finite_number_is_refused_by_name(name):
    call = (stats.min_track_record_length if name in ("benchmark", "confidence")
            else stats.sharpe_interval)
    given = {"periods_per_year": 12, name: math.nan}

    with pytest.raises(ProbatumValueError, match=f"{name}: nan; give"):
        call(MONTHLY, **given)

    given[name] = "0.5"

    with pytest.raises(ProbatumTypeError, match=f"{name}: '0.5'; give"):
        call(MONTHLY, **given)


def test_a_level_or_a_confidence_outside_0_and_1_is_refused():
    with pytest.raises(ProbatumValueError, match="level: 95.0; give a fraction between 0 "
                       "and 1, such as 0.95"):
        stats.sharpe_interval(MONTHLY, periods_per_year=12, level=95)
    with pytest.raises(ProbatumValueError, match="confidence: 1.0; give a fraction"):
        stats.min_track_record_length(MONTHLY, periods_per_year=12, confidence=1.0)
    with pytest.raises(ProbatumValueError, match="periods_per_year: 0; give the bars a "
                       "year, such as 365"):
        stats.sharpe(MONTHLY, periods_per_year=0)


def test_three_returns_or_returns_that_never_vary_support_no_interval_but_a_sharpe():
    with pytest.raises(ProbatumValueError, match="returns: 3 of them; an interval needs "
                       "at least 4"):
        stats.sharpe_interval(MONTHLY[:3], periods_per_year=12)
    for call in (stats.sharpe_interval, stats.probabilistic_sharpe,
                 stats.min_track_record_length):
        with pytest.raises(ProbatumValueError, match="returns: all 5 are the same; an "
                           "interval needs returns that vary"):
            call([0.0] * 5, periods_per_year=12)
        # two values in the one proportion whose Sharpe's influence is zero on each bar,
        # which rounding leaves a hair above zero, so only the threshold refuses them
        with pytest.raises(ProbatumValueError, match="returns: two values whose Sharpe's "
                           "influence is zero on every bar, so it has no standard error; "
                           "an interval needs returns that take a third"):
            call([0.3, 0.075, 0.075, 0.075, 0.075], periods_per_year=12)

    assert stats.sharpe(MONTHLY[:3], periods_per_year=12) == pytest.approx(
        np.mean(MONTHLY[:3]) / np.std(MONTHLY[:3], ddof=1) * math.sqrt(12.0))
    assert stats.sharpe([0.0] * 5, periods_per_year=12) == 0.0
    assert stats.sharpe([0.01], periods_per_year=12)    == 0.0


def test_a_track_record_has_no_years_where_the_sharpe_is_not_above_its_benchmark():
    record = stats.min_track_record_length(MONTHLY, benchmark=3.0, periods_per_year=12)

    assert (record.observations, record.years) == (None, None)
    assert str(record)         == "minimum track record none: the sharpe 1.73 is not above 3.00"
    assert record.series_years == pytest.approx(8 / 12)


def test_each_result_says_its_numbers_never_changes_and_pickles_as_it_was():
    interval    = stats.sharpe_interval(MONTHLY, periods_per_year=12)
    probability = stats.probabilistic_sharpe(MONTHLY, benchmark=1.0, periods_per_year=12)
    record      = stats.min_track_record_length(MONTHLY, periods_per_year=12)

    assert (interval.statistic, interval.method, interval.count, interval.lag,
            interval.floored, interval.level) == ("sharpe", "delta method", 8, 2, True, 0.95)
    assert str(probability) == "probabilistic sharpe 0.705 against 1.00, one-sided"
    assert repr(record).startswith("TrackRecord(minimum track record 1.0 years for 95% ")

    for made in (interval, probability, record):
        assert pickle.loads(pickle.dumps(made))       == made
        assert hash(pickle.loads(pickle.dumps(made))) == hash(made)
        with pytest.raises(ProbatumAttributeError, match="a result does not change; "
                           "compute another"):
            made.count = 9


@pytest.fixture(scope="module")
def traded():
    """A run of 251 round trips, a long held on the bars a coin flip picks, so its
    exposure leaves its returns no memory for the bootstrap's blocks to span."""
    close = 100.0 + np.random.default_rng(9).normal(0.0, 1.0, 1_000).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(1_000, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(1_000, dtype=np.int64) * 3_600_000_000_000,
    )

    weights = np.where(np.random.default_rng(9).random(1_000) < 0.5, 1.0, 0.0)
    return probatum.backtest(weights, data, "frictionless")


def test_a_bootstrap_repeats_on_every_read_for_a_result_and_its_series(traded):
    rate = traded.provenance["periods_per_year"]
    once = stats.bootstrap(traded, "cagr_pct")

    assert once == stats.bootstrap(traded, "cagr_pct")
    assert once == stats.bootstrap(traded.returns[1:], "cagr_pct", periods_per_year=rate)
    assert once != stats.bootstrap(traded, "cagr_pct", seed=3)
    assert once.estimate == pytest.approx(traded.stats["cagr_pct"], rel=1e-9)
    assert (once.method, once.kind, once.count) == (
        "studentized stationary block bootstrap", "interval", 999)
    assert once.block >= 1.0 and once.lag is None and once.floored is None


def test_a_drawdown_is_a_range_of_histories_and_a_loss_prints_as_a_negative_return(traded):
    fall = stats.bootstrap(traded, "max_drawdown_pct")

    assert fall.kind == "range" and 0.0 < fall.low <= fall.estimate <= fall.high
    assert str(fall).startswith(f"max drawdown -{fall.estimate:.2f}%; middle 95% of "
                                f"resampled histories -{fall.high:.2f}% to "
                                f"-{fall.low:.2f}%, stationary block bootstrap, block ")

    worst = stats.bootstrap(traded, "value_at_risk")
    assert worst.kind == "interval" and str(worst).startswith(
        f"value at risk -{worst.estimate:.2f}%, two-sided 95% interval")
    assert worst.method == "percentile stationary block bootstrap"


def test_a_result_s_trade_statistics_take_a_bootstrap_of_its_round_trips(traded):
    assert traded.stats["num_trades"] == 251

    for name, method in (("win_rate", "percentile bootstrap of round trips"),
                         ("profit_factor", "bias-corrected and accelerated bootstrap of "
                                           "round trips"),
                         ("avg_trade_pct", "studentized bootstrap of round trips")):
        interval = stats.bootstrap(traded, name)
        assert interval.estimate == pytest.approx(traded.stats[name], rel=1e-12)
        assert interval.low <= interval.estimate <= interval.high
        assert (interval.method, interval.count, interval.block) == (method, 251, None)
        assert str(interval).endswith(f"{method}, 251 round trips")


def test_a_bootstrap_refuses_what_its_sample_cannot_support_by_name(result, traded):
    with pytest.raises(ProbatumValueError, match="statistic: 'max_drawdown'; give one of "
                       "total_return_pct, net_profit_pct, cagr_pct"):
        stats.bootstrap(result, "max_drawdown")
    with pytest.raises(ProbatumValueError, match="statistic: win_rate reads round trips, "
                       "which a plain series does not hold; pass the Result"):
        stats.bootstrap(MONTHLY, "win_rate", periods_per_year=12)
    with pytest.raises(ProbatumValueError, match=f"round trips: "
                       f"{result.stats['num_trades']} closed; an interval of win_rate "
                       f"needs at least 60"):
        stats.bootstrap(result, "win_rate")
    with pytest.raises(ProbatumValueError, match="sortino: inf, a reward over no measured "
                       "risk; an estimate ranked at infinity has no interval"):
        stats.bootstrap([0.01, 0.02, 0.0, 0.03], "sortino", periods_per_year=12)
    with pytest.raises(ProbatumValueError, match="returns: 399 of them; a bootstrap of "
                       "returns needs at least 500"):
        stats.bootstrap(result, "cagr_pct")

    rng       = np.random.default_rng(6)
    stretched = np.where(np.repeat(rng.random(20) < 0.5, 30), rng.normal(0.0, 0.01, 600),
                         0.0)
    with pytest.raises(ProbatumValueError, match=r"returns: 600 in blocks of 29\.7, 20\.2 "
                       r"of them; a bootstrap of returns needs at least 100 blocks"):
        stats.bootstrap(stretched, "volatility_pct", periods_per_year=365)

    gone = np.random.default_rng(4).normal(0.0, 0.01, 1_000)
    gone[300] = -1.0
    with pytest.raises(ProbatumValueError, match="total_return_pct: a return of -100.00%, "
                       "which leaves no equity to grow; growth has no interval past a "
                       "loss of all of it"):
        stats.bootstrap(gone, "total_return_pct", periods_per_year=365)
    assert stats.bootstrap(gone, "volatility_pct", periods_per_year=365).high > 0.0

    tied = np.random.default_rng(2).normal(0.001, 0.01, 1_000)
    tied[np.argsort(tied)[:60]] = -0.02
    with pytest.raises(ProbatumValueError, match="conditional_value_at_risk: the 50 "
                       "lowest returns are all -2.00%; an interval needs a tail that "
                       "varies"):
        stats.bootstrap(tied, "conditional_value_at_risk", periods_per_year=365)
    assert stats.bootstrap(tied, "value_at_risk", periods_per_year=365).low < 2.0

    # ten losses beside 490 flat bars: every resample's 50th lowest return is flat
    flat = np.random.default_rng(8).permutation(np.concatenate(
        [np.full(10, -0.01), np.zeros(490), np.random.default_rng(7).uniform(0.0, 0.01,
                                                                          500)]))
    with pytest.raises(ProbatumValueError, match="value_at_risk: its resamples put both "
                       "bounds on 0.00%; an interval needs resamples that differ "
                       "between them"):
        stats.bootstrap(flat, "value_at_risk", periods_per_year=365)

    deep = tied.copy()
    deep[np.argsort(deep)[:300]] = -0.02
    with pytest.raises(ProbatumValueError, match="value_at_risk: its resamples put both "
                       "bounds on -2.00%"):
        stats.bootstrap(deep, "value_at_risk", periods_per_year=365)
    with pytest.raises(ProbatumValueError, match="win_rate: its resamples put both bounds "
                       "on 0.47; an interval needs resamples that differ between them"):
        stats.bootstrap(traded, "win_rate", level=0.01)
    with pytest.raises(ProbatumValueError, match="profit_factor: its resamples put both "
                       "bounds on 1.07"):
        stats.bootstrap(traded, "profit_factor", level=0.2, resamples=4, seed=0)
    with pytest.raises(ProbatumValueError, match="resamples: 3 at a level of 0.2, which "
                       "puts both bounds on one resample; give more"):
        stats.bootstrap(tied, "volatility_pct", periods_per_year=365, level=0.2,
                        resamples=3)
    with pytest.raises(ProbatumValueError, match="resamples: 1; give a whole number "
                       "from 2"):
        stats.bootstrap(traded, "win_rate", resamples=1)
    with pytest.raises(ProbatumValueError, match="resamples: 2, every one of them on one "
                       "side of the profit factor's estimate; give more"):
        stats.bootstrap(traded, "profit_factor", resamples=2, seed=0)
    with pytest.raises(ProbatumTypeError, match=r"statistic: \['sortino'\]; give one of"):
        stats.bootstrap(traded, ["sortino"])
    with pytest.raises(ProbatumTypeError, match="seed: 1.5; give a whole number from 0"):
        stats.bootstrap(traded, "win_rate", seed=1.5)


def held(result):
    """The run's buy-and-hold baseline as returns a row, the first nothing."""
    equity = np.asarray(result.baselines["buy_and_hold"])
    return np.concatenate([[0.0], equity[1:] / equity[:-1] - 1.0])


def test_a_run_alone_is_tested_against_buying_and_holding_its_own_bars(result):
    found = stats.sharpe_difference(result)
    rate  = result.provenance["periods_per_year"]
    plain = stats.sharpe_difference(result.returns[1:], held(result)[1:],
                                    periods_per_year=rate)

    assert found.delta == plain.delta and found.bootstrap.low == pytest.approx(
        plain.bootstrap.low, rel=1e-12)
    assert found.estimate == pytest.approx(
        stats.sharpe(result) - stats.sharpe(held(result)[1:], periods_per_year=rate))
    assert (found.delta.method, found.bootstrap.method) == (
        "delta method", "studentized circular block bootstrap")
    assert 0.0 <= found.delta.p <= 1.0 and 0.0 < found.bootstrap.p <= 1.0

    lines = str(found).split("\n")
    assert len(lines) == 2 and lines[0].startswith("sharpe difference ")
    assert "p-value" in lines[0] and "block" in lines[1]


def test_two_sharpes_are_refused_unless_they_read_the_same_bars(result, traded):
    with pytest.raises(probatum.DataError, match="b: a Result on other rows; compare "
                       "runs on the same bars"):
        stats.sharpe_difference(result, traded)
    with pytest.raises(ProbatumTypeError, match="b: list beside a Result; pass two "
                       "Results, or two plain series"):
        stats.sharpe_difference(result, MONTHLY)
    with pytest.raises(ProbatumTypeError, match="b: NoneType beside a plain series"):
        stats.sharpe_difference(MONTHLY, periods_per_year=12)
    with pytest.raises(ProbatumValueError, match="b: 6 returns beside 8; give two series "
                       "of one length"):
        stats.sharpe_difference(MONTHLY, MONTHLY[:6], periods_per_year=12)
    with pytest.raises(ProbatumValueError, match="b: returns whose Sharpe moves as a's "
                       "does on every bar, so the difference has no standard error; "
                       "compare series that differ"):
        stats.sharpe_difference(MONTHLY, [3.0 * r for r in MONTHLY], periods_per_year=12)
    with pytest.raises(ProbatumValueError, match="b: two values whose Sharpe's "
                       "influence is zero on every bar, as a's is, so the difference "
                       "has no standard error; compare series that take a third"):
        stats.sharpe_difference([0.3] * 3 + [0.075] * 12, [0.02] * 5 + [0.01] * 10,
                                periods_per_year=12)
    with pytest.raises(ProbatumValueError, match="b: position 2 holds nan"):
        stats.sharpe_difference(MONTHLY, [0.01, 0.0, np.nan, 0.02, 0.0, 0.01, 0.0, 0.03],
                                periods_per_year=12)


def test_a_run_kept_at_the_account_level_has_no_baseline_to_be_measured_against():
    close = 100.0 + np.random.default_rng(5).normal(0.0, 1.0, 400).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(400, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(400, dtype=np.int64) * 3_600_000_000_000,
    )

    kept = probatum.backtest(np.where(np.arange(400) % 40 < 20, 1.0, 0.0), data,
                             "frictionless", record="account")

    with pytest.raises(ProbatumValueError, match="b: none given, and a run recorded at "
                       "'account' has no baselines; pass one, or run at the full level"):
        stats.sharpe_difference(kept)
    with pytest.raises(ProbatumValueError, match="benchmark: none given"):
        stats.alpha_beta(kept)


def test_alpha_and_beta_read_buying_and_holding_unless_a_benchmark_is_given(result):
    found = stats.alpha_beta(result)
    times = pd.to_datetime(result.time, utc=True)

    assert stats.alpha_beta(result, benchmark=held(result)) == found
    assert stats.alpha_beta(result, benchmark=pd.Series(held(result), index=times)) == found

    # the first row is ignored, so a benchmark read as pct_change() is taken as it is
    unread = held(result)
    unread[0] = np.nan
    assert stats.alpha_beta(result, benchmark=unread) == found

    unread[3] = np.nan
    with pytest.raises(ProbatumValueError, match="benchmark: position 3 holds nan"):
        stats.alpha_beta(result, benchmark=unread)

    assert found.count == 399 and found.alpha_low <= found.alpha <= found.alpha_high
    assert str(found).startswith(f"alpha {found.alpha * 100:.2f}% a year, standard error")
    with pytest.raises(probatum.DataError, match="benchmark: a Series whose index is not "
                       "the run's row times"):
        stats.alpha_beta(result, benchmark=pd.Series(held(result)))
    with pytest.raises(ProbatumValueError, match="benchmark: 8 values beside a run of 400 "
                       "rows; give one a row"):
        stats.alpha_beta(result, benchmark=MONTHLY)
    with pytest.raises(ProbatumTypeError, match="result: list; alpha_beta reads a Result"):
        stats.alpha_beta(MONTHLY)
    with pytest.raises(ProbatumValueError, match="lag: -1; give a whole number of bars"):
        stats.alpha_beta(result, lag=-1)


def test_the_order_of_trades_permutes_a_run_s_closed_round_trips(traded):
    found = stats.trade_order(traded)

    assert (found.count, found.n) == (251, 2_000)
    assert found.drawdown_low <= found.drawdown_high and 0.0 <= found.deeper <= 1.0
    assert found.losing_run_low <= found.losing_run_high
    assert found == stats.trade_order(traded) and found != stats.trade_order(traded, seed=4)

    said = str(found)
    assert said.startswith(f"drawdown at trade closes -{found.drawdown:.2f}% in the order "
                           f"the 251 round trips closed; middle 95% of 2,000 other orders")
    assert said.endswith("what the account marked inside a trade is left out")

    close = 100.0 + np.random.default_rng(5).normal(0.0, 1.0, 50).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(50, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(50, dtype=np.int64) * 3_600_000_000_000,
    )
    with pytest.raises(ProbatumValueError, match="round trips: 0 closed; permuting needs "
                       "at least 2"):
        stats.trade_order(probatum.backtest(np.ones(50), data, "frictionless"))


def test_the_autocorrelation_is_1_at_lag_0_and_0_where_nothing_varies():
    assert stats.autocorrelation(MONTHLY, 0)     == 1.0
    assert stats.autocorrelation([0.01] * 6, 1)  == 0.0
    assert stats.autocorrelation(MONTHLY[:2], 1) == 0.0
    assert stats.effective_sample(MONTHLY[:3])   == 3.0
    with pytest.raises(ProbatumValueError, match="lag: -1; give a whole number of bars, "
                       "from 0"):
        stats.autocorrelation(MONTHLY, -1)


def test_what_reads_a_run_s_time_trips_costs_or_rows_refuses_a_plain_series():
    calls = [(stats.period_returns, ()), (stats.segment, ()), (stats.trade_stats, ()),
             (stats.long_short_split, ()), (stats.session_buckets, ()),
             (stats.benchmark, ([1.0] * 8,)), (stats.decompose, ())]

    for call, given in calls:
        with pytest.raises(ProbatumTypeError, match=f"result: list; {call.__name__} reads "
                           f"a Result"):
            call(MONTHLY, *given)


def test_a_plain_series_rows_are_its_path_s_and_a_result_s_its_own(result):
    plain = stats.drawdown_table(result.returns[1:])
    own   = stats.drawdown_table(result)

    assert stats.drawdown(result).size == 400 and stats.drawdown(MONTHLY).size == 8
    assert stats.drawdown(result)[0] == 0.0
    for column in ("peak_bar", "trough_bar", "bars_to_trough", "bars_under"):
        assert plain[column].tolist() == own[column].tolist()
    assert np.ma.getmaskarray(plain["peak_time"]).all()
    assert own["peak_time"].tolist() == result.time[own["peak_bar"]].tolist()

    under = stats.time_under_water(result)
    assert under == stats.time_under_water(result)
    assert stats.time_under_water(MONTHLY).longest_days is None
    assert str(under).startswith(f"longest under water {under.longest:,} rows, "
                                 f"{under.longest_days:,.0f} days")


def test_a_series_that_loses_everything_falls_100_percent_and_no_further():
    # the path reaches -0.55 on the second return and moves no further
    values = [0.1, -1.5, 0.2, 0.3]

    assert stats.drawdown(values).tolist() == [0.0, 100.0, 100.0, 100.0]
    assert stats.time_under_water(values).longest == 3


def test_too_few_returns_or_returns_that_never_vary_have_a_normal_shape():
    assert stats.skew([0.01, 0.02, 0.06]) > 0.0 and stats.skew([0.01, 0.02]) == 0.0
    assert stats.kurtosis([0.01, 0.02, 0.06]) == 3.0
    assert stats.kurtosis([0.01, 0.02, 0.06, -0.03]) != 3.0
    assert stats.skew([0.01] * 6) == 0.0 and stats.kurtosis([0.01] * 6, excess=True) == 0.0


def test_a_pandas_series_of_returns_is_read_by_position():
    given = pd.Series(MONTHLY, index=list("abcdefgh"))

    assert stats.drawdown(given).tolist() == stats.drawdown(MONTHLY).tolist()
    assert stats.skew(given) == stats.skew(MONTHLY)
    assert stats.rolling_sharpe(given, 4, periods_per_year=12).tolist() == pytest.approx(
        stats.rolling_sharpe(MONTHLY, 4, periods_per_year=12).tolist(), nan_ok=True)


def test_a_window_a_count_and_a_choice_are_refused_by_name(result):
    with pytest.raises(ProbatumValueError, match="window: 1; give a whole number of "
                       "returns from 2"):
        stats.rolling_sharpe(MONTHLY, 1, periods_per_year=12)
    with pytest.raises(ProbatumTypeError, match="window: 4.0; give a whole number"):
        stats.rolling_volatility(MONTHLY, 4.0, periods_per_year=12)
    with pytest.raises(ProbatumTypeError, match="periods_per_year: missing"):
        stats.rolling_sharpe(MONTHLY, 4)
    with pytest.raises(ProbatumValueError, match="top: 0; give a whole number of "
                       "drawdowns from 1"):
        stats.drawdown_table(MONTHLY, top=0)
    with pytest.raises(ProbatumValueError, match="by: 'week'; give \"month\", "
                       "\"quarter\" or \"year\""):
        stats.period_returns(result, by="week")
    with pytest.raises(ProbatumTypeError, match="by: 1; give \"hour\" or \"weekday\""):
        stats.session_buckets(result, by=1)
    with pytest.raises(ProbatumValueError, match="alpha: 1.0; give a fraction"):
        stats.value_at_risk(MONTHLY, alpha=1.0)
    with pytest.raises(ProbatumValueError, match="returns: none; a tail needs at least "
                       "one return"):
        stats.tail_ratio([])
    with pytest.raises(ProbatumTypeError, match="periods_per_year: missing"):
        stats.omega(MONTHLY, threshold=0.1)

    assert stats.omega(MONTHLY) == stats.omega(MONTHLY, periods_per_year=12)


def test_a_segment_holds_a_return_and_reads_the_record_s_rows(result):
    with pytest.raises(ProbatumValueError, match="start: 400; give a row from 0 to 399"):
        stats.segment(result, 400)
    with pytest.raises(ProbatumValueError, match="stop: 1; give a row from 2 to 400, so "
                       "the segment holds a return"):
        stats.segment(result, 0, 1)
    with pytest.raises(ProbatumValueError, match="stop: 10; give a row from 11 to 400"):
        stats.segment(result, 10, 10)
    with pytest.raises(ProbatumTypeError, match="stop: 2.5; give a whole number of rows"):
        stats.segment(result, 0, 2.5)

    part = stats.segment(result, 200)
    assert part.interval("sharpe") == stats.sharpe_interval(
        result.returns[200:], periods_per_year=result.provenance["periods_per_year"])


def test_prices_held_beside_a_run_are_one_a_row_finite_and_above_zero(result):
    prices = 100.0 + np.arange(400.0)
    found  = stats.benchmark(result, prices)
    times  = pd.to_datetime(result.time, utc=True)

    assert found == stats.benchmark(result, pd.Series(prices, index=times))
    assert found.hold_return_pct == pytest.approx(399.0)
    with pytest.raises(ProbatumValueError, match="prices: 8 values beside a run of 400 "
                       "rows; give one price a row"):
        stats.benchmark(result, MONTHLY)

    prices[7] = 0.0
    with pytest.raises(ProbatumValueError, match="prices: position 7 holds 0.0; a price "
                       "is a finite number above zero"):
        stats.benchmark(result, prices)
    with pytest.raises(probatum.DataError, match="prices: a Series whose index is not "
                       "the run's row times"):
        stats.benchmark(result, pd.Series(100.0 + np.arange(400.0)))
    with pytest.raises(ProbatumTypeError, match="prices: str; pass an array"):
        stats.benchmark(result, "close")


def test_a_run_kept_at_the_account_level_counts_no_fills_and_no_turnover():
    close = 100.0 + np.random.default_rng(5).normal(0.0, 1.0, 400).cumsum()
    bars  = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(400, 1e3)])
    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(400, dtype=np.int64) * 3_600_000_000_000,
    )

    weights = np.where(np.arange(400) % 40 < 20, 1.0, 0.0)
    venue   = probatum.venues.binance_perp.replace(funding="none")
    full    = probatum.backtest(weights, data, venue)
    kept    = probatum.backtest(weights, data, venue, record="account")

    assert math.isnan(stats.decompose(kept).turnover) and stats.decompose(full).turnover > 0
    assert stats.segment(kept, 100)["num_fills"] is None
    for name in kept.stats:
        if name != "num_fills":
            assert stats.segment(kept, 100)[name] == stats.segment(full, 100)[name]
    assert "turnover none" in str(stats.decompose(kept))


def test_each_new_result_never_changes_and_pickles_as_what_it_holds(traded):
    found = [stats.time_under_water(traded), stats.trade_stats(traded),
             stats.benchmark(traded, 100.0 + np.arange(1_000.0)), stats.decompose(traded)]

    for held in found:
        assert pickle.loads(pickle.dumps(held)) == held
        assert hash(pickle.loads(pickle.dumps(held))) == hash(held)
        with pytest.raises(ProbatumAttributeError, match="a result does not change"):
            held.mean = 0.0

    split = stats.long_short_split(traded)
    assert list(split) == ["long", "short"] and split["short"].trades == 0
    assert str(split["short"]) == "no closed round trips"
    assert split["long"] == stats.trade_stats(traded)


def test_runs_set_side_by_side_are_refused_by_name():
    with pytest.raises(ProbatumTypeError, match="results: str; pass a list of Results"):
        probatum.compare("run")
    with pytest.raises(ProbatumValueError, match="results: none given"):
        probatum.compare([])
    with pytest.raises(ProbatumTypeError, match="results: a list among them; pass "
                       "Results"):
        probatum.compare([MONTHLY])


def test_a_key_no_run_reports_is_refused(result):
    with pytest.raises(probatum.ProbatumKeyError, match="keys: 'alpha', which no run "
                       "reports"):
        probatum.compare([result], keys=["alpha"])
