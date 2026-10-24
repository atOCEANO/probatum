"""``probatum.stats``: how far a run's statistics can be trusted, read from a Result or
from any series of returns: the Sharpe's interval by the delta method over a long-run
variance with Newey and West's weights, its probability of beating a benchmark and the
track record that would show it, the long-run Sharpe, and every other statistic's
interval by a bootstrap of the returns or of the round trips; and, across all of a
search's trials, how far its winner can be trusted and whether it stands among
neighbours that score as it does (ADRs 0031, 0033; Specification, Inference; Across a
search)."""

import hashlib
import itertools
import math
import os
import statistics
import warnings
from collections.abc import Mapping

import numpy as np

from ._errors import (DataError, ProbatumAttributeError, ProbatumKeyError,
                      ProbatumTypeError, ProbatumValueError, ProbatumWarning)
from ._space import _flag, _number, _whole
from ._streams import stream

__all__ = ["AlphaBeta", "BacktestLength", "Consistency", "Decay", "Deflated", "Difference",
           "EffectiveTrials", "Forecast", "Interval", "Overfitting", "Permutations",
           "Plateau", "Probability", "RealityCheck", "Statistics", "TrackRecord",
           "alpha_beta",
           "autocorrelation",
           "bootstrap", "deflated_sharpe", "deflation_threshold", "effective_sample",
           "effective_trials", "min_backtest_length", "min_track_record_length",
           "overfitting", "plateau", "probabilistic_sharpe", "reality_check", "sharpe",
           "sharpe_difference", "sharpe_interval", "trade_order"]

# the package's files, which a warning skips to name the caller's line
PACKAGE = (os.path.dirname(__file__),)

NORMAL   = statistics.NormalDist()
ROUNDING = 1e-20     # a standardised influence's variance below this is rounding
NUDGE    = 1e-9      # a rank this near a whole number is that number
CEILING  = 1e12      # what a CAGR saturates at, as the record's does
FLOOR    = 60        # the closed round trips a trade statistic's interval needs
SPAN     = 500       # the returns a bootstrap of them needs
BLOCKS   = 100       # the mean blocks those returns need to hold
STARTS   = 10        # the clustering of trials' starts
MOST     = 100       # the most clusters a start tries
LLOYD    = 300       # Lloyd's iterations at most
PARTS    = 16        # the blocks the overfitting probability cuts a run into
AT_ONCE  = 1 << 21   # the halves' statistics a chunk reads, beside its complements'
RADII    = tuple(0.05 * k for k in range(1, 11))   # the plateau's, in normalised units
# the objectives the overfitting probability reads from halves joined out of order
HALVED = ("sharpe", "sortino", "total_return_pct", "net_profit_pct", "cagr_pct",
          "volatility_pct")
# the Euler-Mascheroni constant of the expected maximum of normal draws
EULER = 0.5772156649015329

# the statistics a stationary bootstrap of the returns serves: each one's words in a
# text, its unit, whether a text prints it as a loss, and how the resamples are read,
# studentized on a pivot, at their percentiles, or as the middle of the resampled
# histories, which is no interval
RETURNS = {
    "total_return_pct":          ("total return", "%", False, "studentized"),
    "net_profit_pct":            ("net profit", "%", False, "studentized"),
    "cagr_pct":                  ("cagr", "%", False, "studentized"),
    "volatility_pct":            ("volatility", "%", False, "studentized"),
    "sortino":                   ("sortino", "", False, "studentized"),
    "value_at_risk":             ("value at risk", "%", True, "percentile"),
    "conditional_value_at_risk": ("conditional value at risk", "%", True, "studentized"),
    "max_drawdown_pct":          ("max drawdown", "%", True, "range"),
    "calmar":                    ("calmar", "", False, "range"),
}
# the statistics read on the growth a bar, which a return of -100% leaves undefined
GROWTH = ("total_return_pct", "net_profit_pct", "cagr_pct")
# the score past which a stretch of a run gains, for the objectives that have one
BREAK_EVEN = {
    "sharpe":           0.0,
    "sortino":          0.0,
    "calmar":           0.0,
    "total_return_pct": 0.0,
    "net_profit_pct":   0.0,
    "cagr_pct":         0.0,
    "avg_trade_pct":    0.0,
    "profit_factor":    1.0,
}
# the objectives that grow with the span they are scored on, whose decay would measure
# the spans' lengths
GROWS = ("total_return_pct", "net_profit_pct", "max_drawdown_pct", "num_trades",
         "num_fills", "funding_paid")
# the statistics a bootstrap of the closed round trips serves: words, unit and method
TRADES = {
    "win_rate":      ("win rate", "", "percentile bootstrap of round trips"),
    "profit_factor": ("profit factor", "",
                      "bias-corrected and accelerated bootstrap of round trips"),
    "avg_trade_pct": ("average trade", "%", "studentized bootstrap of round trips"),
}


class _Frozen:
    """What the module's results share: set once, compared and hashed by what they
    hold, and pickled as the values they were made from."""

    __slots__ = ()


    def __setattr__(self, name, value):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a result does not "
                                     f"change; compute another")


    def __delattr__(self, name):
        raise ProbatumAttributeError(f"{type(self).__name__}.{name}: a result does not "
                                     f"change; compute another")


    def __eq__(self, other):
        return type(other) is type(self) and self._args() == other._args()


    def __hash__(self):
        return hash((type(self).__name__, self._args()))


    def __reduce__(self):
        return type(self), self._args()


    def __repr__(self):
        return f"{type(self).__name__}({self})"


    def _args(self):
        return tuple(getattr(self, name) for name in self.__slots__)


    def _set(self, **given):
        for name, value in given.items():
            object.__setattr__(self, name, value)


class Interval(_Frozen):
    """A statistic's interval: ``statistic``, its name; ``estimate``, ``low`` and
    ``high``, in its units and with its sign, a drawdown and a value at risk positive
    losses; ``level``; ``method``, how it was found; ``count``, the returns or round
    trips it read; ``lag`` and ``floored``, the delta method's lag and whether its
    floor at lag zero bound; ``block``, a block bootstrap's block; ``kind``,
    ``"interval"``, or ``"range"`` for the middle of the resampled histories, which is
    no confidence interval; and ``p``, the two-sided p-value of an estimate of zero,
    where the interval is a test's. Its text says each, a loss as the negative return
    it is."""

    __slots__ = ("statistic", "estimate", "low", "high", "level", "method", "count", "lag",
                 "floored", "block", "kind", "p")


    def __init__(self, statistic, estimate, low, high, level, method, count, lag=None,
                 floored=None, block=None, kind="interval", p=None):
        self._set(
            statistic = statistic,
            estimate  = float(estimate),
            low       = float(low),
            high      = float(high),
            level     = float(level),
            method    = method,
            count     = int(count),
            lag       = None if lag is None else int(lag),
            floored   = None if floored is None else bool(floored),
            block     = None if block is None else float(block),
            kind      = kind,
            p         = None if p is None else float(p),
        )


    def __str__(self):
        words, unit, loss = _shown(self.statistic)
        flip = -1.0 if loss else 1.0
        low, high = sorted((flip * self.low + 0.0, flip * self.high + 0.0))
        what = (f"; middle {_percent(self.level)} of resampled histories"
                if self.kind == "range" else f", two-sided {_percent(self.level)} interval")

        how = self.method
        if self.lag is not None:
            how += f", Newey-West lag {self.lag}"
            how += " floored at lag zero" if self.floored else ""
        if self.block is not None:
            how += f", block {self.block:.1f}"
        if self.p is not None:
            how += f", p-value {self.p:.3f}"

        noun = "round trips" if self.statistic in TRADES else "returns"
        return (f"{words} {flip * self.estimate + 0.0:.2f}{unit}{what} {low:.2f}{unit} to "
                f"{high:.2f}{unit}, {how}, {self.count:,} {noun}")


class Probability(_Frozen):
    """The probabilistic Sharpe: ``value``, the one-sided probability that the Sharpe is
    above ``benchmark``, an annual Sharpe; ``lag`` and ``floored``, its variance's; and
    ``count``, the returns it read."""

    __slots__ = ("value", "benchmark", "lag", "floored", "count")


    def __init__(self, value, benchmark, lag, floored, count):
        self._set(
            value     = float(value),
            benchmark = float(benchmark),
            lag       = int(lag),
            floored   = bool(floored),
            count     = int(count),
        )


    def __str__(self):
        return (f"probabilistic sharpe {self.value:.3f} against "
                f"{_against(self.benchmark)}, one-sided")


class TrackRecord(_Frozen):
    """The minimum track record: ``observations``, the returns a Sharpe of ``sharpe``
    needs for a one-sided ``confidence`` that it is above ``benchmark``, an annual
    Sharpe, and ``years``, those over the annualisation, both None where the Sharpe is
    not above it; ``series_years``, the years the series holds; and ``lag``,
    ``floored`` and ``count`` as a Probability's."""

    __slots__ = ("observations", "years", "series_years", "sharpe", "confidence",
                 "benchmark", "lag", "floored", "count")


    def __init__(self, observations, years, series_years, sharpe, confidence, benchmark,
                 lag, floored, count):
        self._set(
            observations = None if observations is None else float(observations),
            years        = None if years is None else float(years),
            series_years = float(series_years),
            sharpe       = float(sharpe),
            confidence   = float(confidence),
            benchmark    = float(benchmark),
            lag          = int(lag),
            floored      = bool(floored),
            count        = int(count),
        )


    def __str__(self):
        if self.years is None:
            return (f"minimum track record none: the sharpe {self.sharpe:.2f} is not "
                    f"above {_against(self.benchmark)}")

        return (f"minimum track record {_years(self.years)} years for "
                f"{_percent(self.confidence)} one-sided confidence against "
                f"{_against(self.benchmark)}; this series has {_years(self.series_years)}")


class Difference(_Frozen):
    """Two Sharpes on the same bars: ``estimate``, the first's less the second's a year,
    and its two tests, ``delta``, the delta method's, and ``bootstrap``, Ledoit and
    Wolf's studentized circular block bootstrap, each an Interval with its p-value of no
    difference."""

    __slots__ = ("estimate", "delta", "bootstrap")


    def __init__(self, estimate, delta, bootstrap):
        self._set(estimate=float(estimate), delta=delta, bootstrap=bootstrap)


    def __str__(self):
        return f"{self.delta}\n{self.bootstrap}"


class AlphaBeta(_Frozen):
    """A regression of a run's excess returns on a benchmark's: ``alpha`` a year,
    with its ``alpha_error``, its two-sided interval ``alpha_low`` to ``alpha_high``
    and ``alpha_t``; ``beta``, with ``beta_error``, ``beta_low`` and ``beta_high``; the
    ``tracking_error`` and the ``information_ratio`` a year; and ``lag``, ``level`` and
    ``count``, the returns it read. Alpha and the tracking error are fractions."""

    __slots__ = ("alpha", "alpha_error", "alpha_low", "alpha_high", "alpha_t", "beta",
                 "beta_error", "beta_low", "beta_high", "tracking_error",
                 "information_ratio", "lag", "level", "count")


    def __init__(self, alpha, alpha_error, alpha_low, alpha_high, alpha_t, beta, beta_error,
                 beta_low, beta_high, tracking_error, information_ratio, lag, level, count):
        self._set(
            alpha             = float(alpha),
            alpha_error       = float(alpha_error),
            alpha_low         = float(alpha_low),
            alpha_high        = float(alpha_high),
            alpha_t           = float(alpha_t),
            beta              = float(beta),
            beta_error        = float(beta_error),
            beta_low          = float(beta_low),
            beta_high         = float(beta_high),
            tracking_error    = float(tracking_error),
            information_ratio = float(information_ratio),
            lag               = int(lag),
            level             = float(level),
            count             = int(count),
        )


    def __str__(self):
        return (f"alpha {self.alpha * 100.0:.2f}% a year, standard error "
                f"{self.alpha_error * 100.0:.2f}%, two-sided {_percent(self.level)} "
                f"interval {self.alpha_low * 100.0:.2f}% to {self.alpha_high * 100.0:.2f}%, "
                f"t {self.alpha_t:.2f}; beta {self.beta:.3g}, standard error "
                f"{self.beta_error:.3g}, interval {self.beta_low:.3g} to "
                f"{self.beta_high:.3g}; tracking error {self.tracking_error * 100.0:.2f}%, "
                f"information ratio {self.information_ratio:.2f}; Newey-West lag "
                f"{self.lag}, {self.count:,} returns")


class Permutations(_Frozen):
    """The order of a run's round trips: ``drawdown``, the drawdown at trade closes in
    the order they closed, in percent and a positive loss, with ``drawdown_low`` and
    ``drawdown_high``, the middle 95% of ``n`` permuted orders', and ``deeper``, the
    share of them that drew down further; ``losing_run``, the longest run of losing
    trips in the run's order, with ``losing_run_low`` and ``losing_run_high``; and
    ``count``, the round trips. Each trip compounds its return on the equity it opened
    on, and what the account marked inside a trade is left out."""

    __slots__ = ("drawdown", "drawdown_low", "drawdown_high", "deeper", "losing_run",
                 "losing_run_low", "losing_run_high", "count", "n")


    def __init__(self, drawdown, drawdown_low, drawdown_high, deeper, losing_run,
                 losing_run_low, losing_run_high, count, n):
        self._set(
            drawdown        = float(drawdown),
            drawdown_low    = float(drawdown_low),
            drawdown_high   = float(drawdown_high),
            deeper          = float(deeper),
            losing_run      = int(losing_run),
            losing_run_low  = int(losing_run_low),
            losing_run_high = int(losing_run_high),
            count           = int(count),
            n               = int(n),
        )


    def __str__(self):
        return (f"drawdown at trade closes {-self.drawdown + 0.0:.2f}% in the order the "
                f"{self.count:,} round trips closed; middle 95% of {self.n:,} other orders "
                f"{-self.drawdown_high + 0.0:.2f}% to {-self.drawdown_low + 0.0:.2f}%, "
                f"{self.deeper:.1%} of them deeper; longest losing run {self.losing_run}, "
                f"middle 95% {self.losing_run_low} to {self.losing_run_high}; each trip "
                f"compounds its return on the equity it opened on, and what the account "
                f"marked inside a trade is left out")


class Consistency(_Frozen):
    """How a fit's score holds up across a run: ``scores``, each part's in order, the
    ``unit`` its parts are, stretches of a search's winner or a validation's windows;
    ``break_even``, the score past which a part gains, None for an objective that has
    none; ``share``, the share of parts past it in the ``direction`` the search went,
    None without one; and the ``objective``."""

    __slots__ = ("scores", "unit", "break_even", "share", "objective", "direction")


    def __init__(self, scores, unit, break_even, share, objective, direction):
        self._set(
            scores     = tuple(float(score) for score in scores),
            unit       = unit,
            break_even = None if break_even is None else float(break_even),
            share      = None if share is None else float(share),
            objective  = objective,
            direction  = direction,
        )


    def __str__(self):
        flip   = -1.0 if _shown(self.objective)[2] else 1.0
        listed = ", ".join(f"{flip * score + 0.0:.2f}" for score in self.scores)
        if self.share is None:
            return (f"{self.objective} has no break-even, so no part gains or loses: "
                    f"{listed}")

        past = round(self.share * len(self.scores))
        side = "above" if self.direction == "maximize" else "below"
        return (f"{past} of {len(self.scores)} {self.unit} scored {side} "
                f"{self.break_even:g}: {listed}")


class Decay(_Frozen):
    """How far fits fell on the bars they had not seen: ``gaps``, each fit's score less
    what its bars then earned, signed so that a positive gap is worse out of sample in
    either ``direction``, and ``mean``, their mean, in the ``objective``."""

    __slots__ = ("gaps", "mean", "objective", "direction")


    def __init__(self, gaps, mean, objective, direction):
        self._set(
            gaps      = tuple(float(gap) for gap in gaps),
            mean      = float(mean),
            objective = objective,
            direction = direction,
        )


    def __str__(self):
        listed  = ", ".join(f"{gap:.2f}" for gap in self.gaps)
        counted = f"{len(self.gaps)} gap{'s' * (len(self.gaps) != 1)}"
        return (f"decay {self.mean:.2f} in {self.objective}, the mean of {counted} "
                f"between what a fit scored and what its bars then earned: {listed}")


class Deflated(_Frozen):
    """The deflated Sharpe of a search's winner: ``value``, the probabilistic Sharpe of
    its annual ``sharpe`` against ``threshold``, the annual Sharpe the best of
    ``trials`` trials reaches by luck when their annual Sharpes spread ``spread``, the
    deviation across the ``null_trials`` of the null that ran; ``effective``,
    ``effective_value`` and ``effective_threshold``, the same of the winner's cluster at
    the effective number of trials, None where the search kept no returns; ``lag``, ``floored`` and ``count`` as
    a Probability's; and ``descriptive``, True where an adaptive search chose its trials
    on the bars they are judged on."""

    __slots__ = ("value", "threshold", "trials", "spread", "null_trials", "sharpe",
                 "effective", "effective_value", "effective_threshold", "lag", "floored",
                 "count", "descriptive")


    def __init__(self, value, threshold, trials, spread, null_trials, sharpe, effective,
                 effective_value, effective_threshold, lag, floored, count, descriptive):
        self._set(
            value               = float(value),
            threshold           = float(threshold),
            trials              = int(trials),
            spread              = float(spread),
            null_trials         = int(null_trials),
            sharpe              = float(sharpe),
            effective           = None if effective is None else int(effective),
            effective_value     = _maybe(effective_value),
            effective_threshold = _maybe(effective_threshold),
            lag                 = int(lag),
            floored             = bool(floored),
            count               = int(count),
            descriptive         = bool(descriptive),
        )


    def __str__(self):
        said = f"deflated sharpe {self.value:.2f} against all {self.trials:,} trials"
        if self.effective is not None:
            said += (f"; {self.effective_value:.2f} for the winner's cluster against "
                     f"{self.effective:,} effective trial{'s' * (self.effective != 1)}")

        return said + (", descriptive" if self.descriptive else "")


class EffectiveTrials(_Frozen):
    """The effective number of trials: ``count``, the clusters Lopez de Prado and
    Lewis's clustering finds among the ``trials`` it read, ``numbers``, those whose
    returns vary, which it clustered, and ``clusters``, each one's cluster, numbered by
    its first trial; ``quality``, the clustering's, None below 3 trials; ``spread``, the
    annual Sharpes' sample deviation across the clusters' series, None at one cluster;
    and ``dropped``, the trials left out because their returns never vary."""

    __slots__ = ("count", "trials", "numbers", "clusters", "quality", "spread", "dropped")


    def __init__(self, count, trials, numbers, clusters, quality, spread, dropped):
        self._set(
            count    = int(count),
            trials   = int(trials),
            numbers  = tuple(int(number) for number in numbers),
            clusters = tuple(int(cluster) for cluster in clusters),
            quality  = _maybe(quality),
            spread   = _maybe(spread),
            dropped  = int(dropped),
        )


    def __str__(self):
        said = (f"{self.count:,} effective trial{'s' * (self.count != 1)} of "
                f"{self.trials:,}")
        if self.quality is not None:
            said += f", quality {self.quality:.2f}"
        if self.spread is not None:
            said += f", their clusters' sharpes spread {self.spread:.2f} a year"

        return said


class Overfitting(_Frozen):
    """The probability of backtest overfitting: ``probability``, the share of the
    halvings of ``blocks`` blocks whose in-sample winner ranks below the median out of
    sample among ``trials`` trials, and ``loss``, the share whose winner's statistic is
    below 0 out of sample, None for volatility; ``dropped``, the oldest returns left
    out so the blocks are even; ``objective`` and ``direction``, what was ranked; and
    ``descriptive``, True where an adaptive search chose its trials on the bars they are
    judged on."""

    __slots__ = ("probability", "loss", "trials", "blocks", "dropped", "objective",
                 "direction", "descriptive")


    def __init__(self, probability, loss, trials, blocks, dropped, objective, direction,
                 descriptive):
        self._set(
            probability = float(probability),
            loss        = _maybe(loss),
            trials      = int(trials),
            blocks      = int(blocks),
            dropped     = int(dropped),
            objective   = objective,
            direction   = direction,
            descriptive = bool(descriptive),
        )


    def __str__(self):
        said = (f"overfitting {self.probability:.2f} probability the in-sample best ranks "
                f"below the median on the held-back blocks ({self.blocks} blocks, "
                f"{self.trials:,} trials)")
        if self.loss is not None:
            said += f"; it loses on {self.loss:.0%} of them"

        return said + (", descriptive" if self.descriptive else "")


class RealityCheck(_Frozen):
    """White's Reality Check and Hansen's SPA of whether any trial beats a benchmark:
    ``p``, White's p-value, and ``spa``, Hansen's consistent one with each resample
    studentized on its own variance; ``statistic`` and ``spa_statistic``, the two tests'
    maxima; ``best``, the trial of the highest log growth over the ``benchmark``;
    ``block``, the stationary bootstrap's mean block, over ``resamples``; ``trials``,
    those read, and ``excluded``, those left out for a return at or below -100%; and
    ``descriptive``, True where an adaptive search chose its trials on the bars they are
    judged on."""

    __slots__ = ("p", "spa", "statistic", "spa_statistic", "best", "block", "resamples",
                 "trials", "excluded", "benchmark", "descriptive")


    def __init__(self, p, spa, statistic, spa_statistic, best, block, resamples, trials,
                 excluded, benchmark, descriptive):
        self._set(
            p             = float(p),
            spa           = float(spa),
            statistic     = float(statistic),
            spa_statistic = float(spa_statistic),
            best          = int(best),
            block         = float(block),
            resamples     = int(resamples),
            trials        = int(trials),
            excluded      = int(excluded),
            benchmark     = benchmark,
            descriptive   = bool(descriptive),
        )


    def __str__(self):
        return (f"reality check p {self.p:.2f} and SPA p {self.spa:.2f} that no trial "
                f"beats {self.benchmark}, {self.resamples:,} stationary resamples in "
                f"blocks of {self.block:.1f}, {self.trials:,} trials"
                + (", descriptive" if self.descriptive else ""))


class BacktestLength(_Frozen):
    """The minimum backtest length: the ``years`` over which the expected best of
    ``trials`` trials with no skill falls to the annual ``sharpe``, and
    ``series_years``, the years a Study's in-sample bars hold, None for numbers given."""

    __slots__ = ("years", "trials", "sharpe", "series_years")


    def __init__(self, years, trials, sharpe, series_years):
        self._set(
            years        = float(years),
            trials       = int(trials),
            sharpe       = float(sharpe),
            series_years = _maybe(series_years),
        )


    def __str__(self):
        said = (f"minimum backtest length {_years(self.years)} years for the best of "
                f"{self.trials:,} trials to fall to a sharpe of {self.sharpe:.2f}")
        if self.series_years is not None:
            said += f"; this series has {_years(self.series_years)}"

        return said


class Plateau(_Frozen):
    """Whether a search's winner stands among neighbours that score as it does:
    ``radii``, from 0.05 to 0.5 in normalised parameters; ``trials``, how many trials
    lie within each, a configuration run more than once counted once and the winner's
    not at all; ``passed``, how many of those completed with a score past ``level`` in
    the search's ``direction``; ``share``, the one over the other, NaN where none lies
    within; and the ``objective``."""

    __slots__ = ("radii", "trials", "passed", "share", "level", "objective", "direction")


    def __init__(self, radii, trials, passed, share, level, objective, direction):
        self._set(
            radii     = tuple(float(radius) for radius in radii),
            trials    = tuple(int(count) for count in trials),
            passed    = tuple(int(count) for count in passed),
            share     = tuple(float(value) for value in share),
            level     = float(level),
            objective = objective,
            direction = direction,
        )


    def __str__(self):
        # a loss prints as the negative return it is, its side turned with it
        flip  = -1.0 if _shown(self.objective)[2] else 1.0
        side  = "above" if (self.direction == "maximize") == (flip > 0.0) else "below"
        bound = f"score {side} {flip * self.level + 0.0:g}"
        count = self.trials[1]
        head  = (f"{self.share[1]:.0%} of the {count:,} trial{'s' * (count != 1)} within "
                 f"distance 0.1 of the best, in normalised parameters, {bound}" if count
                 else f"no trial within distance 0.1 of the best, in normalised "
                      f"parameters, to {bound}")

        listed = ", ".join("none" if math.isnan(share) else f"{share:.0%}"
                           for share in self.share)
        return f"{head}; from 0.05 to 0.5: {listed}"


class Forecast(_Frozen):
    """A model's skill out of sample, apart from its weights and its costs:
    ``rank_ic``, Spearman's correlation of its predictions with its labels' returns
    over ``rows`` rows of ``paths`` paths; ``error``, its standard error, Newey-West
    over the bars at ``lag``, the longest label; ``t``, the correlation over its error;
    and ``r_squared``, against a forecast of zero (Specification, A model)."""

    __slots__ = ("rank_ic", "error", "t", "lag", "r_squared", "rows", "paths")


    def __init__(self, rank_ic, error, t, lag, r_squared, rows, paths):
        self._set(
            rank_ic   = float(rank_ic),
            error     = float(error),
            t         = float(t),
            lag       = int(lag),
            r_squared = float(r_squared),
            rows      = int(rows),
            paths     = int(paths),
        )


    def __str__(self):
        return (f"forecast rank IC {self.rank_ic:.3f}, t = {self.t:.1f} (Newey-West, lag "
                f"{self.lag}); R squared {self.r_squared:.1%}")


class Statistics(Mapping):
    """A run's record of its fifteen statistics: a Mapping of each name to its value,
    each name an attribute too, that holds what its intervals read, the returns after
    the first row, the closed round trips' pnl and the opening balance, at the
    annualisation and the risk-free rate the run had. ``interval(name, level=0.95)``
    is a statistic's Interval, computed when first read and kept, or None where it has
    none: exposure, the counts and funding paid, and a sample that cannot support one.
    Pickling leaves out what it has computed, which is the same when computed again
    (Specification, Inference)."""

    __slots__ = ("_values", "_returns", "_pnl", "_opening", "_rate", "_risk_free", "_kept")


    def __init__(self, values, returns, pnl, opening, rate, risk_free):
        for name, value in (("_values", dict(values)), ("_returns", returns), ("_pnl", pnl),
                            ("_opening", float(opening)), ("_rate", float(rate)),
                            ("_risk_free", float(risk_free)), ("_kept", {})):
            object.__setattr__(self, name, value)


    def __getitem__(self, name):
        return self._values[name]


    def __iter__(self):
        return iter(self._values)


    def __len__(self):
        return len(self._values)


    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise ProbatumAttributeError(f"Statistics.{name}: no such statistic; give one of "
                                     f"{', '.join(values)}")


    def __setattr__(self, name, value):
        raise ProbatumAttributeError(f"Statistics.{name}: a run's record does not change; "
                                     f"run again for another")


    def __delattr__(self, name):
        raise ProbatumAttributeError(f"Statistics.{name}: a run's record does not change; "
                                     f"run again for another")


    def __reduce__(self):
        return Statistics, (self._values, self._returns, self._pnl, self._opening,
                            self._rate, self._risk_free)


    def __repr__(self):
        return f"Statistics({self._values!r})"


    def interval(self, name, level=0.95):
        """Statistic ``name``'s Interval at ``level``, or None where it has none."""
        if not isinstance(name, str):
            raise ProbatumTypeError(f"name: {name!r}; give one of {', '.join(self._values)}")
        if name not in self._values:
            raise ProbatumKeyError(f"name: {name!r}; give one of {', '.join(self._values)}")

        level = _fraction("level", level)
        if (name, level) not in self._kept:
            self._kept[(name, level)] = self._made(name, level)

        return self._kept[(name, level)]


    def _made(self, name, level):
        apart = _rank(2000, (1.0 - level) / 2.0) != _rank(2000, (1.0 + level) / 2.0)
        if (name in TRADES or name in RETURNS) and not apart:
            return None

        if name in TRADES:
            return (None if _thin_trips(self._pnl, name)
                    else _trips_core(self._pnl, self._opening, name, level, 2000,
                                     None)[0])

        if name != "sharpe" and name not in RETURNS:
            return None

        x = self._returns - self._risk_free / self._rate
        if _thin(x):
            return None

        if name == "sharpe":
            return _sharpe_core(x, self._rate, level)[0]
        return _return_core(np.asarray(self._returns), x, self._rate, (name,), level, 2000,
                            None)[name][0]


def _percent(level):
    return f"{level * 100.0:.10g}%"


def _shown(statistic):
    """A statistic's words in a text, its unit and whether the text prints it as a
    loss."""
    if statistic in RETURNS:
        return RETURNS[statistic][:3]
    if statistic in TRADES:
        return TRADES[statistic][0], TRADES[statistic][1], False
    return statistic, "", False


def _against(benchmark):
    return "zero" if benchmark == 0.0 else f"{benchmark:.2f}"


def _years(years):
    return f"{years:.1f}" if years >= 1.0 else f"{years:.2f}"


def sharpe(returns, periods_per_year=None, risk_free=None, long_run=False):
    """The Sharpe of ``returns``, a Result's, which is its record's, or a plain series'
    at ``periods_per_year`` against an annual ``risk_free``: the excess returns' mean
    over their deviation, annualised by the square root of the bars a year. With
    ``long_run`` the deviation is the long-run one, the Bartlett sum of the returns'
    autocovariances at Newey and West's 1994 lag, not floored, which tells what the
    returns' autocorrelation does to a Sharpe the square root of time scales
    (Specification, What a statistic reads; The long-run Sharpe)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    long_run = _flag("long_run", long_run)

    if x.size < 2:
        return 0.0

    if not long_run:
        return _ranked(x.mean(), x.std(ddof=1), math.sqrt(rate))

    centred = x - x.mean()
    m       = _lag(centred)
    spread  = _bartlett(_autocovariances(centred, m), m)
    return _ranked(x.mean(), math.sqrt(max(spread, 0.0)), math.sqrt(rate))


def sharpe_interval(returns, periods_per_year=None, risk_free=None, level=0.95):
    """The Sharpe's two-sided interval at ``level``, as an Interval, read as
    ``sharpe`` reads ``returns``: its standard error is the long-run variance of the
    Sharpe's influence series at Newey and West's 1994 lag, floored at its value at lag
    zero, over the returns less one, so autocorrelation widens it and never narrows it.
    Fewer than 4 returns, returns that never vary and returns whose Sharpe's influence
    is zero on every bar have none (Specification, The Sharpe's variance; Samples with
    no interval)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    level = _fraction("level", level)
    _supports(x)

    made, said = _sharpe_core(x, rate, level)
    if made is None:
        raise ProbatumValueError(said)

    return made


def probabilistic_sharpe(returns, benchmark=0.0, periods_per_year=None, risk_free=None):
    """The one-sided probability that the Sharpe of ``returns`` is above
    ``benchmark``, an annual Sharpe, as a Probability, from the variance
    ``sharpe_interval`` reads (Specification, The probabilistic Sharpe and the minimum
    track record)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    benchmark = _finite("benchmark", benchmark, "an annual Sharpe, such as 0.0")

    variance, lag, floored = _variance(x)
    gap   = _per_bar(x) - benchmark / math.sqrt(rate)
    value = NORMAL.cdf(gap * math.sqrt(x.size - 1) / math.sqrt(variance))

    return Probability(value, benchmark, lag, floored, x.size)


def min_track_record_length(returns, benchmark=0.0, confidence=0.95, periods_per_year=None,
                            risk_free=None):
    """The returns at which the Sharpe of ``returns`` and its variance would give a
    one-sided ``confidence`` that it is above ``benchmark``, an annual Sharpe, as a
    TrackRecord in returns and in years, None where the Sharpe is not above the
    benchmark (Specification, The probabilistic Sharpe and the minimum track
    record)."""
    x, rate = _read(returns, periods_per_year, risk_free)
    benchmark  = _finite("benchmark", benchmark, "an annual Sharpe, such as 0.0")
    confidence = _fraction("confidence", confidence)

    variance, lag, floored = _variance(x)
    gap    = _per_bar(x) - benchmark / math.sqrt(rate)
    needed = (None if gap <= 0.0
              else 1.0 + variance * (NORMAL.inv_cdf(confidence) / gap) ** 2)

    return TrackRecord(needed, None if needed is None else needed / rate, x.size / rate,
                       _per_bar(x) * math.sqrt(rate), confidence, benchmark, lag, floored,
                       x.size)


def bootstrap(returns, statistic, periods_per_year=None, risk_free=None, level=0.95,
              resamples=2000, seed=None):
    """``statistic``'s interval at ``level``, as an Interval, by a bootstrap of
    ``returns``, read as ``sharpe`` reads them. Total return, net profit, CAGR,
    volatility, Sortino, value at risk and conditional value at risk take a stationary
    block bootstrap of the returns, its mean block the longest Politis and White's rule
    finds on the returns, their squared deviations or their tail, each studentized on
    its own standard error but value at risk, whose interval is percentile, and maximum
    drawdown and Calmar the middle of those resampled histories, which is no confidence
    interval. A Result's win rate, profit factor and average trade take a bootstrap of
    its closed round trips. A bootstrap of the returns needs 500 of them in 100 of its
    blocks, and one of the round trips 60 of them. The ``resamples`` come from a stream
    seeded by the values they resample, or by ``seed``, so an interval repeats on every
    read with no seed (Specification, Intervals of the returns; Intervals of the round
    trips; Samples with no interval)."""
    if not isinstance(statistic, str):
        raise ProbatumTypeError(f"statistic: {statistic!r}; give one of "
                                f"{', '.join([*RETURNS, *TRADES])}")
    if statistic not in RETURNS and statistic not in TRADES:
        raise ProbatumValueError(f"statistic: {statistic!r}; give one of "
                                 f"{', '.join([*RETURNS, *TRADES])}")

    level = _fraction("level", level)
    resamples, seed = _resamples(resamples), _seed(seed)
    if _rank(resamples, (1.0 - level) / 2.0) == _rank(resamples, (1.0 + level) / 2.0):
        raise ProbatumValueError(f"resamples: {resamples} at a level of {level:g}, which "
                                 f"puts both bounds on one resample; give more")

    if statistic in TRADES:
        return _trade_interval(returns, statistic, periods_per_year, risk_free, level,
                               resamples, seed)

    x, rate = _read(returns, periods_per_year, risk_free)
    _supports(x)

    made, said = _return_core(_values(returns), x, rate, (statistic,), level, resamples,
                              seed)[statistic]
    if made is None:
        raise ProbatumValueError(said)

    return made


def sharpe_difference(a, b=None, periods_per_year=None, risk_free=None, level=0.95,
                      resamples=2000, seed=None):
    """The difference of two Sharpes on the same bars, ``a``'s less ``b``'s, as a
    Difference that tests it twice: by the delta method, on the long-run variance of
    the two influence series' difference at the 1994 lag floored at lag zero, and by
    Ledoit and Wolf's studentized circular block bootstrap. ``a`` and ``b`` are two
    Results on the same rows or two plain series of one length, read as ``sharpe``
    reads them, or ``a`` is a Result alone, measured against its own buy-and-hold
    baseline at its own risk-free rate (Specification, Two Sharpes on the same bars)."""
    level = _fraction("level", level)
    resamples, seed = _resamples(resamples), _seed(seed)
    first, second, rate, raw = _paired(a, b, periods_per_year, risk_free, "a", "b")
    _supports(first, "a")
    _supports(second, "b")

    count = first.size
    z     = _influence(first) - _influence(second)
    variance, lag, floored = _floored(z)

    if variance < ROUNDING and _floored(_influence(first))[0] < ROUNDING:
        raise ProbatumValueError("b: two values whose Sharpe's influence is zero on "
                                 "every bar, as a's is, so the difference has no "
                                 "standard error; compare series that take a third")
    if variance < ROUNDING:
        raise ProbatumValueError("b: returns whose Sharpe moves as a's does on every bar, "
                                 "so the difference has no standard error; compare series "
                                 "that differ")

    d     = _per_bar(first) - _per_bar(second)
    error = math.sqrt(variance / (count - 1))
    root  = math.sqrt(rate)
    q     = NORMAL.inv_cdf((1.0 + level) / 2.0)
    delta = Interval("sharpe difference", d * root, (d - q * error) * root,
                     (d + q * error) * root, level, "delta method", count, lag, floored,
                     p=2.0 * NORMAL.cdf(-abs(d) / error))

    length = _block_lengths(z)[1]
    rng    = _drawn(np.concatenate(raw), "stats sharpe difference", seed)
    found  = _studentized(first, second, d, length,
                          _block_starts(count, length, resamples, rng))

    half      = _ordered(found, level) * error
    p         = (np.count_nonzero(found >= abs(d) / error) + 1) / (resamples + 1)
    bootstrap = Interval(
        "sharpe difference",
        d * root,
        (d - half) * root,
        (d + half) * root,
        level,
        "studentized circular block bootstrap",
        count,
        block = length,
        p     = p,
    )

    return Difference(d * root, delta, bootstrap)


def alpha_beta(result, benchmark=None, lag=None, level=0.95):
    """Alpha and beta of a Result against a benchmark, as an AlphaBeta: the least
    squares of the run's excess returns on the benchmark's with an intercept, their
    covariance Newey and West's at ``lag`` or at the 1994 rule's on the slope's score,
    with the tracking error and the information ratio. ``benchmark`` is the run's
    buy-and-hold baseline unless given: a Result on the same rows, an array shaped as
    the run's returns, its row 0 ignored, or a pandas Series indexed by the run's row
    times (Specification, Alpha and beta)."""
    from ._result import Result
    if not isinstance(result, Result):
        raise ProbatumTypeError(f"result: {type(result).__name__}; alpha_beta reads a "
                                f"Result")

    level = _fraction("level", level)
    if lag is not None and not _whole(lag):
        raise ProbatumTypeError(f"lag: {lag!r}; give a whole number of bars from 0, or "
                                f"None for the 1994 rule's")
    if lag is not None and lag < 0:
        raise ProbatumValueError(f"lag: {lag}; give a whole number of bars from 0, or "
                                 f"None for the 1994 rule's")

    x, y, rate, _ = _paired(result, benchmark, None, None, "result", "benchmark")
    _supports(x)
    _supports(y, "benchmark")

    return _regression(x, y, rate, lag, level)


def _regression(x, y, rate, lag, level):
    """Alpha and beta of the excess returns ``x`` on a benchmark's ``y``, at ``rate``
    bars a year (Specification, Alpha and beta)."""
    rows = np.column_stack([np.ones(x.size), y])
    coef, *_ = np.linalg.lstsq(rows, x, rcond=None)

    e          = x - rows @ coef
    m          = lag if lag is not None else _lag(y * e)
    covariance = _newey_west(rows, e, m)
    q          = NORMAL.inv_cdf((1.0 + level) / 2.0)

    alpha, alpha_error = coef[0] * rate, math.sqrt(covariance[0, 0]) * rate
    beta, beta_error = coef[1], math.sqrt(covariance[1, 1])

    # a run that is exactly a line in its benchmark leaves no error to scale alpha by
    t = (alpha / alpha_error if alpha_error > 0.0
         else math.copysign(math.inf, alpha) if alpha != 0.0 else 0.0)

    active = x - y
    spread = active.std(ddof=1)

    return AlphaBeta(alpha, alpha_error, alpha - q * alpha_error, alpha + q * alpha_error,
                     t, beta, beta_error, beta - q * beta_error, beta + q * beta_error,
                     spread * math.sqrt(rate), _ranked(active.mean(), spread,
                                                       math.sqrt(rate)), m, level, x.size)


def trade_order(result, n=2000, seed=None):
    """The drawdown and the longest losing run of a Result's closed round trips in the
    order they closed, beside ``n`` permutations of that order, as Permutations: each
    trip returns its pnl on the account's equity at the close before its opening fill,
    and an order compounds those returns from the opening equity, the model of a policy
    sized on its equity, so the order is all that moves. The drawdown is at trade closes
    and leaves out what the account marked inside a trade (Specification, The order of
    trades)."""
    from ._result import Result
    if not isinstance(result, Result):
        raise ProbatumTypeError(f"result: {type(result).__name__}; trade_order reads a "
                                f"Result")

    n, seed = _resamples(n, "n"), _seed(seed)

    trades = result.trades
    closed = ~np.asarray(trades["open"])
    pnl    = np.asarray(trades["pnl"][closed], dtype=np.float64)
    if pnl.size < 2:
        raise ProbatumValueError(f"round trips: {pnl.size} closed; permuting needs at "
                                 f"least 2")

    order    = np.argsort(np.asarray(trades["close_bar"][closed]), kind="stable")
    opened   = np.asarray(trades["open_bar"][closed])[order]
    pnl      = pnl[order]
    equity   = np.asarray(result.equity)
    returned = pnl / equity[np.maximum(opened - 1, 0)]

    fall, run = _paths(returned, pnl, np.arange(pnl.size)[None, :])
    rng    = _drawn(returned, "stats trade order", seed)
    orders = rng.permuted(np.broadcast_to(np.arange(pnl.size), (n, pnl.size)), axis=1)
    falls, runs = _paths(returned, pnl, orders)

    return Permutations(fall[0], _ordered(falls, 0.025), _ordered(falls, 0.975),
                        _deeper(falls, fall[0]), run[0],
                        _ordered(runs, 0.025), _ordered(runs, 0.975), pnl.size, n)


def deflation_threshold(spread, trials):
    """The annual Sharpe the best of ``trials`` independent trials reaches by luck when
    their annual Sharpes spread ``spread`` about zero: ``spread`` times Bailey and Lopez
    de Prado's approximation of the expected maximum of that many standard normal
    draws (Specification, The deflated Sharpe)."""
    spread = _finite("spread", spread, "the annual Sharpes' deviation, such as 0.5")
    if spread < 0.0:
        raise ProbatumValueError(f"spread: {spread!r}; give the annual Sharpes' deviation, "
                                 f"0 or more")

    return spread * _most(_looks("trials", trials))


def deflated_sharpe(study, null):
    """The deflated Sharpe of the winner of ``study``, a search that maximised the
    Sharpe, as a Deflated: its probabilistic Sharpe, on the variance
    ``sharpe_interval`` reads, against what the best of every trial the search asked
    reaches by luck, their Sharpes spread as those of every trial that ran of ``null``,
    a random search of the same policy over the same space, venue and in-sample bars,
    scored alike; and the same of the winner's cluster, its members combined, against
    the effective number of trials, the search's own clustered, where it kept their
    returns (ADR 0033; Specification, The deflated Sharpe)."""
    study = _searched(study, "study")
    given = study.provenance

    if (given["objective"], given["direction"]) != ("sharpe", "maximize"):
        fix = ("direction=\"maximize\"" if given["objective"] == "sharpe" else
               "objective=\"sharpe\"" if given["direction"] == "maximize" else
               "objective=\"sharpe\" and direction=\"maximize\"")
        raise ProbatumValueError(f"study: a search that ranked on {given['objective']} "
                                 f"to {given['direction']} it; the deflation is of a "
                                 f"Sharpe a search maximised, so search with {fix}")

    study._vetted(null)
    trials = len(study.trials)
    if trials < 2:
        raise ProbatumValueError("study: 1 trial; a deflation corrects for more than one, "
                                 "so search with trials=2 or more")

    if len(null.trials) < trials:
        warnings.warn(f"null: a search of {len(null.trials):,} trials beside the "
                      f"{trials:,} this one asked, so the spread that deflates it rests on "
                      f"fewer draws than the trials it corrects for; search the null with "
                      f"trials={trials}", ProbatumWarning, skip_file_prefixes=PACKAGE)

    table   = null.trials
    ran     = np.isin(np.asarray(table["state"]), ("complete", "floored"))
    sharpes = np.asarray(table["sharpe"], dtype=np.float64)[ran]
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size < 2:
        raise ProbatumValueError(f"null: {sharpes.size} of its trials ran to a finite "
                                 f"sharpe; their spread needs 2")

    spread    = float(sharpes.std(ddof=1))
    threshold = spread * _most(trials)
    best      = study.best
    found     = probabilistic_sharpe(best, benchmark=threshold)
    effective = value = bar = None
    if study.returns is not None:
        held      = effective_trials(study)
        effective = held.count
        bar       = 0.0 if held.spread is None else held.spread * _most(held.count)
        value     = _family(study, held, bar).value

    return Deflated(found.value, threshold, trials, spread, sharpes.size,
                    best.stats["sharpe"], effective, value, bar, found.lag, found.floored,
                    found.count, given["sampler"]["name"] == "tpe")


def effective_trials(trials, periods_per_year=None, risk_free=None, seed=None):
    """The effective number of ``trials``, as an EffectiveTrials: the clusters Lopez de
    Prado and Lewis's clustering finds among the trials whose returns vary, by k-means
    on the distances of their correlations, keeping the clustering whose silhouettes
    are most consistent and refining its weakest clusters, and the annual Sharpes'
    spread across the clusters' series, each its members combined at minimum variance.
    ``trials`` is a search's Study, whose trials that ran it reads, or a matrix of a row
    a trial's returns on the same bars. The starts draw from a stream seeded by the
    returns, or by ``seed`` (Specification, The effective number of trials)."""
    x, rate, numbers = _trials(trials, periods_per_year, risk_free,
                               ("complete", "floored"))
    seed = _seed(seed)

    from ._search import Study
    kept = isinstance(trials, Study) and seed is None
    if kept and trials._effective is not None:
        return trials._effective

    varies = np.ptp(x, axis=1) > 0.0
    rows   = x[varies]

    if rows.shape[0] < 3:
        labels, quality = np.arange(rows.shape[0]), None
    else:
        labels, quality = _onc(rows, _drawn(rows, "stats onc", seed))

    count  = int(labels.max()) + 1 if labels.size else 0
    series = [_combined(rows[labels == label]) for label in range(count)]

    if len(series) > 1:
        shapes = [_ranked(s.mean(), s.std(ddof=1), math.sqrt(rate)) for s in series]
        spread = float(np.std(shapes, ddof=1))
    else:
        spread = None

    held = EffectiveTrials(len(series), x.shape[0], np.asarray(numbers)[varies], labels,
                           quality, spread, int((~varies).sum()))
    if kept:
        trials._effective = held

    return held


def overfitting(trials, objective=None, periods_per_year=None, risk_free=None):
    """The probability of backtest overfitting of ``trials``, as an Overfitting, by
    combinatorially symmetric cross-validation: the returns cut into 16 contiguous
    blocks, and in each of the 12,870 ways of holding half of them out, the trial best
    on the other half ranked on the held-out one, overfit where it ranks below the
    median (Bailey, Borwein, Lopez de Prado and Zhu 2017). It ranks on a statistic the
    halves give whatever the order of their returns: a search's objective, in its
    direction, reading its complete trials, or ``objective`` for a matrix of a row a
    trial's returns, the Sharpe unless given, volatility minimised and the rest
    maximised (ADR 0033; Specification, The probability of backtest overfitting)."""
    from ._search import Study
    if isinstance(trials, Study):
        if objective is not None:
            raise ProbatumTypeError(f"objective: {objective!r} beside a Study, which "
                                    f"ranks on its own; leave it out")

        given     = _searched(trials, "trials").provenance
        objective = given["objective"]
        direction = given["direction"]
        adaptive  = given["sampler"]["name"] == "tpe"
        free      = given["risk_free"]
    else:
        objective = "sharpe" if objective is None else objective
        direction = "minimize" if objective == "volatility_pct" else "maximize"
        adaptive  = False
        free      = risk_free

    if objective not in HALVED:
        raise ProbatumValueError(f"objective: {objective!r}; the overfitting probability "
                                 f"ranks on a statistic the halves give whatever the "
                                 f"order of their returns: {', '.join(HALVED)}")

    x, rate, _ = _trials(trials, periods_per_year, risk_free, ("complete",))
    count = x.shape[0]
    if count < 2:
        raise ProbatumValueError(f"trials: {count} complete; the overfitting probability "
                                 f"ranks a winner among 2 or more")
    if x.shape[1] < PARTS:
        raise ProbatumValueError(f"trials: {x.shape[1]} returns; the overfitting "
                                 f"probability cuts them into {PARTS} blocks, so give at "
                                 f"least {PARTS}")

    raw = x + (free or 0.0) / rate
    probability, loss = _overfit(x, raw, rate, objective, direction == "maximize", PARTS)
    return Overfitting(probability, None if objective == "volatility_pct" else loss, count,
                       PARTS, x.shape[1] % PARTS, objective, direction, adaptive)


def reality_check(trials, benchmark=None, periods_per_year=None, risk_free=None,
                  resamples=1000, seed=None):
    """White's Reality Check and Hansen's SPA of whether any of ``trials`` beats
    ``benchmark``, as a RealityCheck: each trial's log growth over the benchmark's a bar,
    resampled by one stationary bootstrap every trial shares, its mean block the median
    of Politis and White's lengths on the trials; White's maximum mean, and Hansen's
    maximum studentized mean, recentred by his consistent rule, each resample
    studentized on its own stationary-bootstrap variance. ``trials`` is a search's
    Study, whose trials that ran it reads, or a matrix of a row a trial's returns; the
    benchmark is cash at the risk-free rate unless ``"buy_and_hold"``, a Study's
    baseline, or a series of returns on the same bars. The resamples draw from a stream
    seeded by what they resample, or by ``seed`` (ADR 0033; Specification, The Reality
    Check and SPA)."""
    from ._result import _returns
    from ._search import Study

    resamples, seed = _resamples(resamples), _seed(seed)
    x, rate, numbers = _trials(trials, periods_per_year, risk_free, ("complete", "floored"))

    if isinstance(trials, Study):
        free     = trials.provenance["risk_free"]
        adaptive = trials.provenance["sampler"]["name"] == "tpe"
    else:
        free, adaptive = risk_free or 0.0, False

    r     = x + free / rate
    count = r.shape[1]
    if count < 3:
        raise ProbatumValueError(f"trials: {count} returns; the Reality Check needs 3")

    if isinstance(benchmark, str) and benchmark != "buy_and_hold":
        raise ProbatumValueError(f"benchmark: {benchmark!r}; give \"buy_and_hold\", a "
                                 f"series of returns, or None for cash")

    if benchmark is None:
        said, b = "cash", np.full(count, free / rate)
    elif isinstance(benchmark, str):
        if not isinstance(trials, Study):
            raise ProbatumValueError("benchmark: \"buy_and_hold\" beside a matrix, which "
                                     "holds no instruments to buy; pass a series of "
                                     "returns")

        held    = np.asarray(trials.best.baselines["buy_and_hold"], dtype=np.float64)
        said, b = "buying and holding", _returns(held)[1:]
    else:
        start  = 1 if isinstance(trials, Study) else 0
        values = _series(benchmark, "benchmark", first=start,
                         fix="give \"buy_and_hold\", a series of returns, or None for cash")
        ruined = np.flatnonzero(values[start:] <= -1.0) + start
        if ruined.size:
            raise ProbatumValueError(f"benchmark: position {ruined[0]} holds "
                                     f"{values[ruined[0]]}, all of a bar lost, which leaves "
                                     f"no log growth to measure the trials against")

        values = values[start:]
        if values.size != count:
            raise ProbatumValueError(f"benchmark: {values.size} returns beside {count}; "
                                     f"give a return a bar the trials cover")

        said, b = "the benchmark", values

    lost = (r <= -1.0).any(axis=1)
    if lost.all():
        raise ProbatumValueError("trials: each lost all of a bar, which leaves no log "
                                 "growth to test")

    f       = np.log1p(r[~lost]) - np.log1p(b)
    fbar    = f.mean(axis=1)
    root    = math.sqrt(count)
    big_v   = root * float(fbar.max())
    varies  = np.ptp(f, axis=1) > 0.0
    lengths = [_block_lengths(row)[0] for row in f[varies]]
    block   = float(np.median(lengths)) if lengths else 1.0
    weights = _window(count, 1.0 / block)
    omega   = np.sqrt(np.maximum(_spectral(f[varies], weights), 0.0))
    live    = omega > 0.0
    t       = root * fbar[varies][live] / omega[live]

    if (~varies & (fbar > 0.0)).any():
        big_t = math.inf
    else:
        big_t = max(0.0, float(t.max())) if t.size else 0.0

    shift = np.where(t >= -math.sqrt(2.0 * math.log(math.log(count))), fbar[varies][live],
                     0.0)
    rows  = f[varies][live]
    white = above = 0
    rng   = _drawn(np.vstack([r[~lost], b]), "stats reality check", seed)

    for chunk in _stationary(count, block, resamples, rng):
        for drawn in chunk:
            star   = f[:, drawn]
            white += root * float((star.mean(axis=1) - fbar).max()) > big_v
            if rows.size and 0.0 < big_t < math.inf:
                taken = rows[:, drawn]
                again = np.sqrt(np.maximum(_spectral(taken, weights), 0.0))
                again = np.where((np.ptp(taken, axis=1) > 0.0) & (again > 0.0), again,
                                 omega[live])
                above += max(0.0, float((root * (taken.mean(axis=1) - shift)
                                         / again).max())) > big_t

    spa  = 0.0 if big_t == math.inf else 1.0 if big_t == 0.0 else above / resamples
    best = int(np.asarray(numbers)[~lost][int(np.argmax(fbar))])
    return RealityCheck(white / resamples, spa, big_v, big_t, best, block, resamples,
                        int((~lost).sum()), int(lost.sum()), said, adaptive)


def min_backtest_length(trials, sharpe=None):
    """The minimum backtest length, as a BacktestLength: the years over which the
    expected best of ``trials`` independent trials with no skill falls to the annual
    ``sharpe`` (Bailey, Borwein, Lopez de Prado and Zhu 2014). A search's Study gives its
    trials' count and its winner's Sharpe on the in-sample bars, beside the years those
    bars hold (Specification, The minimum backtest length)."""
    from ._search import Study
    years = None

    if isinstance(trials, Study):
        study = _searched(trials, "trials")
        if sharpe is not None:
            raise ProbatumTypeError(f"sharpe: {sharpe!r} beside a Study, whose winner's "
                                    f"it reads; leave it out")

        count  = len(study.trials)
        sharpe = float(study._winner.stats["sharpe"])
        years  = (study._bars.cut - 1) / study.provenance["periods_per_year"]
        if count < 2:
            raise ProbatumValueError("trials: a Study of 1 trial; a backtest length is "
                                     "for the best of more than one, so search with "
                                     "trials=2 or more")
        if not sharpe > 0.0:
            raise ProbatumValueError(f"trials: a Study whose winner's sharpe is "
                                     f"{sharpe:.2f}; a backtest length is for a Sharpe "
                                     f"above 0")
    else:
        count = _looks("trials", trials)
        if sharpe is None:
            raise ProbatumTypeError("sharpe: missing; give the annual Sharpe the best "
                                    "trial is to fall to, such as 1.0")

        sharpe = _finite("sharpe", sharpe, "an annual Sharpe above 0, such as 1.0")
        if sharpe <= 0.0:
            raise ProbatumValueError(f"sharpe: {sharpe!r}; give an annual Sharpe above 0, "
                                     f"such as 1.0")

    return BacktestLength((_most(count) / sharpe) ** 2, count, sharpe, years)


def plateau(study, level):
    """Whether a search's winner stands among neighbours that score as it does, as a
    Plateau: each parameter put on [0, 1] across its range on the scale it was drawn
    on, a categorical one 0 from the winner's where a trial's choice is the winner's
    and 1 where it is not, each trial's distance from the winner the Euclidean one
    over its parameters, and at each radius from 0.05 to 0.5 the share of the
    configurations but the winner's within it, each counted once by the first trial
    that ran it, that completed with a score past ``level``, a finite number in the
    objective's units, in the search's direction, a floored or failed trial counting
    against it and one the workers' breaking failed left out (ADR 0033; Specification,
    The plateau)."""
    from ._space import Categorical, _one
    study = _searched(study, "study")
    level = _finite("level", level, "a score in the objective's units, a finite number")

    given = study.provenance
    table = study.trials
    best  = int(np.flatnonzero(np.asarray(table["number"]) == study._winner.number)[0])
    total = np.zeros(len(table))
    axes  = []

    for name, drawn in study._space.items():
        if isinstance(drawn, Categorical):
            # a choice's place among the choices, since the trials table holds a None
            # among floats as NaN
            place = [next(k for k, choice in enumerate(drawn.choices)
                          if _one(choice, params[name])) for params in study._params]
            part  = np.array([0.0 if k == place[best] else 1.0 for k in place])
        else:
            # as each trial was given it, since a whole number past 2**53 has no float64
            # of its own
            place = [params[name] for params in study._params]
            part  = _placed(place, drawn)
            part  = part - part[best]

        total = total + part * part
        axes.append(place)

    distance = np.sqrt(total)
    score    = np.asarray(table["score"], dtype=np.float64)

    # a floored or failed trial has no score, so it passes no level
    past = (score > level) if given["direction"] == "maximize" else (score < level)

    # a configuration run again is no new neighbour, the winner's is none, and nor is a
    # trial the workers' breaking failed, which never ran
    keys    = list(zip(*axes))
    seen    = {keys[best]}
    broken  = set(study._broken)
    numbers = np.asarray(table["number"]).tolist()
    others  = np.zeros(len(table), dtype=bool)

    for k, key in enumerate(keys):
        if numbers[k] in broken:
            continue
        others[k] = key not in seen
        seen.add(key)

    trials, passed = [], []
    for radius in RADII:
        near = others & (distance <= radius)
        trials.append(int(near.sum()))
        passed.append(int((near & past).sum()))

    share = [count / within if within else math.nan
             for count, within in zip(passed, trials)]

    return Plateau(RADII, trials, passed, share, level, given["objective"],
                   given["direction"])


def _placed(values, drawn):
    """An Int's or a Float's ``values`` on [0, 1] across its range, on the scale it was
    drawn on, 0 where the range is one value (Specification, The plateau)."""
    low, high = drawn.low, drawn.high
    if low == high:
        return np.zeros(len(values))

    if drawn.log:
        # Python's math.log, since numpy's own can part from it in a last bit at an edge
        span = math.log(high / low)
        return np.array([math.log(value / low) / span for value in values])

    # in Python's numbers, which take a whole number's difference exactly
    return np.array([(value - low) / (high - low) for value in values])


def autocorrelation(returns, lag=1):
    """The autocorrelation of ``returns``, a Result's or a plain series, at ``lag``: 1
    at lag 0, and 0 where the returns never vary or number no more than ``lag + 1``. No
    interval reads it (Specification, What a statistic reads)."""
    if not _whole(lag):
        raise ProbatumTypeError(f"lag: {lag!r}; give a whole number of bars, from 0")

    if lag < 0:
        raise ProbatumValueError(f"lag: {lag}; give a whole number of bars, from 0")

    x = _values(returns)
    if lag == 0:
        return 1.0

    centred = x - x.mean() if x.size else x
    below   = float(centred @ centred)
    if x.size <= lag + 1 or below == 0.0:
        return 0.0

    return float(centred[lag:] @ centred[:-lag]) / below


def effective_sample(returns):
    """emsl's count of independent returns in ``returns``, ``n * (1 - a) / (1 + a)`` at
    the first autocorrelation held to [0, 0.99], at least 2, and the count itself below
    4 returns. No interval reads it (Specification, What a statistic reads)."""
    x = _values(returns)
    if x.size < 4:
        return float(x.size)

    a = min(max(autocorrelation(x, 1), 0.0), 0.99)
    return max(2.0, x.size * (1.0 - a) / (1.0 + a))


def _read(returns, periods_per_year, risk_free, name="returns"):
    """The excess returns a statistic reads and the bars a year it annualises by: a
    Result's after its first row, at its own annualisation and risk-free rate, or a
    plain series by position at the rate it is given, each refused by the ``name`` it
    was passed as (Specification, What a statistic reads)."""
    # _result reads this module's types, so it is imported here and not at the top
    from ._result import Result
    if isinstance(returns, Result):
        given = returns.provenance
        if periods_per_year is not None:
            raise ProbatumTypeError(f"periods_per_year: given beside a Result, which runs "
                                    f"at {given['periods_per_year']}; leave it out")
        if risk_free is not None:
            raise ProbatumTypeError(f"risk_free: given beside a Result, which ran at "
                                    f"{given['risk_free']}; leave it out")

        rate = given["periods_per_year"]
        if rate is None:
            raise ProbatumValueError(f"{name}: a Result whose clock is too short to "
                                     f"annualise; run it over more than one row")
        return np.asarray(returns.returns[1:]) - given["risk_free"] / rate, rate

    rate, free = _annual(periods_per_year, risk_free, "a plain return series")
    return _series(returns, name) - free / rate, rate


def _values(returns):
    """The returns a Result's record reads, or a plain series', at no annualisation."""
    from ._result import Result
    return (np.asarray(returns.returns[1:]) if isinstance(returns, Result)
            else _series(returns))


def _series(returns, name="returns", first=0, fix=None):
    """A plain return series passed as ``name``: a 1-D array, a list or a pandas Series
    read by position, finite from position ``first``; ``fix`` is what its refusal of
    another type asks for."""
    if isinstance(returns, (str, bytes, dict)) or not hasattr(returns, "__len__"):
        raise ProbatumTypeError(f"{name}: {type(returns).__name__}; " + (
            fix or "pass a Result, or a 1-D array, a list or a pandas Series of returns"))

    try:
        values = np.asarray(returns, dtype=np.float64)
    except (TypeError, ValueError):
        raise ProbatumTypeError(f"{name}: a {type(returns).__name__} that holds no "
                                f"numbers; pass a 1-D series of returns") from None

    if values.ndim != 1:
        raise ProbatumValueError(f"{name}: an array of shape {values.shape}; pass one "
                                 f"series, 1-D")

    bad = np.flatnonzero(~np.isfinite(values[first:])) + first
    if bad.size:
        raise ProbatumValueError(f"{name}: position {bad[0]} holds {values[bad[0]]}; a "
                                 f"return series holds finite numbers")

    return values


def _finite(name, value, fix):
    if not _number(value):
        raise ProbatumTypeError(f"{name}: {value!r}; give {fix}")
    if not math.isfinite(value):
        raise ProbatumValueError(f"{name}: {value!r}; give {fix}")
    return float(value)


def _fraction(name, value):
    value = _finite(name, value, "a fraction between 0 and 1, such as 0.95")
    if not 0.0 < value < 1.0:
        raise ProbatumValueError(f"{name}: {value!r}; give a fraction between 0 and 1, "
                                 f"such as 0.95")

    return value


def _ranked(reward, risk, annualizer):
    """A reward over a risk that can be zero, ranked as the record ranks it: infinite
    for a positive reward over no risk, and zero for any other (ADR 0030)."""
    if risk > 0.0:
        return float(reward / risk * annualizer)
    return math.inf if reward > 0.0 else 0.0


def _per_bar(x):
    return float(x.mean() / x.std(ddof=1))


def _thin(x, name="returns"):
    """Why returns, given as ``name``, cannot support an interval, fewer than 4 or
    returns that never vary, or None where they can (Specification, Samples with no
    interval)."""
    if x.size < 4:
        counted = "of them" if name == "returns" else "returns"
        return f"{name}: {x.size} {counted}; an interval needs at least 4"

    if np.all(x == x[0]):
        same = "" if name == "returns" else " returns"
        return (f"{name}: all {x.size}{same} are the same; an interval needs returns "
                f"that vary")

    return None


def _supports(x, name="returns"):
    said = _thin(x, name)
    if said:
        raise ProbatumValueError(said)


def _resamples(value, name="resamples"):
    # two at least, the fewest that can spread
    if not _whole(value):
        raise ProbatumTypeError(f"{name}: {value!r}; give a whole number from 2")
    if value < 2:
        raise ProbatumValueError(f"{name}: {value}; give a whole number from 2")
    return int(value)


def _seed(seed):
    if seed is not None and not _whole(seed):
        raise ProbatumTypeError(f"seed: {seed!r}; give a whole number from 0, or None")
    if seed is not None and seed < 0:
        raise ProbatumValueError(f"seed: {seed}; give a whole number from 0, or None")
    return seed


def _paired(a, b, periods_per_year, risk_free, head, name):
    """The excess returns of two series on the same bars, passed as ``head`` and
    ``name``, the bars a year they share and their returns as given: two Results on the
    same rows at one annualisation and one risk-free rate, two plain series of one
    length, or a Result and, as ``b`` is not given, its buy-and-hold baseline, or as
    ``name`` is the benchmark, an array or a pandas Series of a value a row of the
    Result, its first ignored."""
    from ._result import Result, _recorded, _returns
    if not isinstance(a, Result):
        if isinstance(b, Result) or b is None:
            raise ProbatumTypeError(f"{name}: {type(b).__name__} beside a plain series; "
                                    f"pass two Results, or two plain series")

        x, rate = _read(a, periods_per_year, risk_free, head)
        y, _ = _read(b, periods_per_year, risk_free, name)
        if y.size != x.size:
            raise ProbatumValueError(f"{name}: {y.size} returns beside {x.size}; give two "
                                     f"series of one length")
        return x, y, rate, (_series(a, head), _series(b, name))

    x, rate = _read(a, periods_per_year, risk_free, head)
    free = a.provenance["risk_free"]

    if b is None:
        if _recorded(a.provenance) != "full":
            raise ProbatumValueError(f"{name}: none given, and a run recorded at "
                                     f"{_recorded(a.provenance)!r} has no baselines; pass "
                                     f"one, or run at the full level")
        held = np.asarray(a.baselines["buy_and_hold"], dtype=np.float64)
        if not np.all(np.isfinite(held)):
            raise ProbatumValueError(f"{name}: none given, and the run's buy-and-hold "
                                     f"baseline is missing on some rows; pass one")
        other = _returns(held)[1:]
    elif isinstance(b, Result):
        if not np.array_equal(a.time, b.time):
            raise DataError(f"{name}: a Result on other rows; compare runs on the same "
                            f"bars")
        given = b.provenance
        if (given["periods_per_year"], given["risk_free"]) != (rate, free):
            raise DataError(f"{name}: a Result at {given['periods_per_year']} bars a year "
                            f"and a risk-free rate of {given['risk_free']}, beside "
                            f"{rate} and {free}; compare runs at one of each")
        other = np.asarray(b.returns[1:])
    elif name != "benchmark":
        raise ProbatumTypeError(f"{name}: {type(b).__name__} beside a Result; pass two "
                                f"Results, or two plain series")
    else:
        if hasattr(b, "index") and hasattr(b, "to_numpy"):
            try:
                times = np.asarray(b.index.values).astype("datetime64[ns]").view(np.int64)
            except (TypeError, ValueError):
                times = None
            if times is None or not np.array_equal(times, a.time):
                raise DataError(f"{name}: a Series whose index is not the run's row "
                                f"times; index it by them, as result.time holds them")
            b = b.to_numpy()

        values = _series(b, name, first=1)
        if values.size != a.time.size:
            raise ProbatumValueError(f"{name}: {values.size} values beside a run of "
                                     f"{a.time.size} rows; give one a row, the first "
                                     f"ignored, as result.returns holds them")
        other = values[1:]

    return x, other - free / rate, rate, (np.asarray(a.returns[1:]), other)


def _variance(x):
    """The Sharpe's variance, its lag and whether its floor bound, refused where it is
    zero."""
    _supports(x)

    variance, lag, floored = _floored(_influence(x))
    if variance < ROUNDING:
        raise ProbatumValueError(_still())

    return variance, lag, floored


def _still():
    """Why returns whose Sharpe's influence is zero on every bar have no interval: the
    influence is a quadratic in each standardised return, so they hold two values."""
    return ("returns: two values whose Sharpe's influence is zero on every bar, so it "
            "has no standard error; an interval needs returns that take a third")


def _influence(x):
    """The Sharpe's influence series on the moments with ``n`` in every divisor."""
    mu = x.mean()
    s  = math.sqrt(float(((x - mu) ** 2).mean()))
    u  = (x - mu) / s
    return u - (mu / s) / 2.0 * (u * u - 1.0)


def _autocovariances(z, upto):
    n = z.size
    return np.array([float(z[j:] @ z[:n - j]) / n if j < n else 0.0
                     for j in range(upto + 1)])


def _lag(z):
    """Newey and West's 1994 lag for the Bartlett kernel, on a series whose mean is
    zero."""
    n  = z.size
    p  = math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))
    g  = _autocovariances(z, p)
    s0 = g[0] + 2.0 * g[1:].sum()
    s1 = 2.0 * (np.arange(1, p + 1) * g[1:]).sum()
    if s0 <= 0.0:
        return 0

    return min(math.floor(1.1447 * ((s1 / s0) ** 2 * n) ** (1.0 / 3.0)), n - 1)


def _bartlett(g, m):
    weights = 1.0 - np.arange(1, m + 1) / (m + 1.0)
    return float(g[0] + 2.0 * (weights * g[1:m + 1]).sum())


def _floored(z):
    """The Bartlett sum at the 1994 lag on ``z``, floored at its value at lag zero,
    with that lag and whether the floor bound."""
    m      = _lag(z)
    g      = _autocovariances(z, m)
    summed = _bartlett(g, m)
    return max(summed, float(g[0])), m, summed < g[0]


def _trade_interval(returns, statistic, periods_per_year, risk_free, level, resamples,
                    seed):
    """A trade statistic's interval from a bootstrap of a Result's closed round trips:
    the win rate's percentile, the profit factor's bias-corrected and accelerated and
    the average trade's studentized (Specification, Intervals of the round trips)."""
    from ._result import Result
    if not isinstance(returns, Result):
        raise ProbatumValueError(f"statistic: {statistic} reads round trips, which a plain "
                                 f"series does not hold; pass the Result")
    _read(returns, periods_per_year, risk_free)

    pnl  = _closed(returns.trades)
    said = _thin_trips(pnl, statistic)
    if said:
        raise ProbatumValueError(said)

    made, said = _trips_core(pnl, float(returns.equity[0]), statistic, level, resamples,
                             seed)
    if made is None:
        raise ProbatumValueError(said)

    return made


def _closed(trades):
    """The closed round trips' pnl, in the order the table holds them."""
    pnl = np.array(trades["pnl"][~np.asarray(trades["open"])], dtype=np.float64)
    pnl.setflags(write=False)
    return pnl


def _thin_trips(pnl, statistic):
    """Why the closed round trips' ``pnl`` cannot support ``statistic``'s interval, or
    None where they can (Specification, Samples with no interval)."""
    count = pnl.size
    won, lost = np.count_nonzero(pnl > 0.0), np.count_nonzero(pnl < 0.0)

    if count < FLOOR:
        return (f"round trips: {count} closed; an interval of {statistic} needs at least "
                f"{FLOOR}")

    if statistic == "win_rate" and not 0 < won < count:
        return (f"win_rate: {won} of {count} round trips won; an interval needs one that "
                f"won and one that did not")

    if statistic == "profit_factor" and (won == 0 or lost < 2):
        return (f"profit_factor: {won} round trips won and {lost} lost; an interval needs "
                f"one that won and two that lost")

    if statistic == "avg_trade_pct" and np.all(pnl == pnl[0]):
        return ("avg_trade_pct: every round trip made the same; an interval needs pnl "
                "that varies")

    return None


def _sharpe_core(x, rate, level):
    """The Sharpe's interval of excess returns ``x`` that vary beside None, or None
    beside why its influence gives it none."""
    variance, lag, floored = _floored(_influence(x))
    if variance < ROUNDING:
        return None, _still()

    error = math.sqrt(variance / (x.size - 1))
    z     = NORMAL.inv_cdf((1.0 + level) / 2.0)
    estimate, root = _per_bar(x), math.sqrt(rate)
    return Interval("sharpe", estimate * root, (estimate - z * error) * root,
                    (estimate + z * error) * root, level, "delta method", x.size, lag,
                    floored), None


def _return_core(r, x, rate, names, level, resamples, seed):
    """Each of the return statistics ``names``' interval, from one stationary block
    bootstrap of returns ``r`` that vary, whose excess returns are ``x``: its Interval
    beside None, or None beside why the sample cannot support one (Specification,
    Intervals of the returns; Samples with no interval)."""
    whole  = _resampled(r, x, rate, np.arange(r.size)[None, :])
    block  = _return_block(r)
    said   = {name: _unsupported(r, name, float(whole[name][0]), block) for name in names}
    pivots = {}

    for name in names:
        if said[name] is None and RETURNS[name][3] == "studentized":
            theta, psi = _pivot(name, r[None, :], x[None, :])
            m     = _lag(psi[0])
            error = math.sqrt(float(_long_run(psi, m)[0]) / (r.size - 1))
            if error > 0.0:
                pivots[name] = float(theta[0]), error, m
            else:
                said[name] = (f"{name}: no standard error, as the returns never vary in "
                              f"what it reads; an interval needs returns that do")

    held  = [name for name in names if said[name] is None]
    found = {name: [] for name in held}

    if held:
        for part in _stationary(r.size, block, resamples, _drawn(r, "stats returns", seed)):
            taken, excess = r[part], x[part]
            plain = (None if all(name in pivots for name in held)
                     else _resampled(r, x, rate, part))
            for name in held:
                found[name].append(_ratios(name, taken, excess, pivots[name][0],
                                           pivots[name][2])
                                   if name in pivots else plain[name])

    low_q, high_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    made = {name: (None, said[name]) for name in names}
    for name in held:
        values, how = np.concatenate(found[name]), RETURNS[name][3]
        if name in pivots:
            theta, error, _ = pivots[name]
            low, high = sorted(_unpivoted(name, theta - _ordered(values, q) * error,
                                          r.size, rate) for q in (high_q, low_q))
        else:
            low, high = _ordered(values, low_q), _ordered(values, high_q)

        if how != "range" and low == high:
            made[name] = None, _met(name, low)
            continue

        method = ("stationary block bootstrap" if how == "range"
                  else f"{how} stationary block bootstrap")
        made[name] = Interval(
            name,
            float(whole[name][0]),
            low,
            high,
            level,
            method,
            r.size,
            block = block,
            kind  = "range" if how == "range" else "interval",
        ), None

    return made


def _unsupported(r, statistic, estimate, block):
    """Why the returns ``r``, resampled in a mean block of ``block``, cannot support an
    interval of ``statistic``, whose estimate on them is ``estimate``, or None where
    they can."""
    if math.isinf(estimate):
        return (f"{statistic}: inf, a reward over no measured risk; an estimate ranked "
                f"at infinity has no interval")

    if statistic in GROWTH and r.min() <= -1.0:
        return (f"{statistic}: a return of {r.min() * 100.0:.2f}%, which leaves no "
                f"equity to grow; growth has no interval past a loss of all of it")

    if r.size < SPAN:
        return (f"returns: {r.size:,} of them; a bootstrap of returns needs at least "
                f"{SPAN}")

    if r.size < BLOCKS * block:
        # the block rounded up and the count down, so neither prints as meeting the
        # floor it misses
        shown = math.ceil(block * 10.0) / 10.0
        held  = math.floor(r.size / block * 10.0) / 10.0
        return (f"returns: {r.size:,} in blocks of {shown:.1f}, {held:.1f} of them; a "
                f"bootstrap of returns needs at least {BLOCKS} blocks")

    if statistic == "conditional_value_at_risk":
        k      = (r.size + 19) // 20
        lowest = np.partition(r, k - 1)[:k]
        if np.all(lowest == lowest[0]):
            return (f"{statistic}: the {k} lowest returns are all "
                    f"{lowest[0] * 100.0:.2f}%; an interval needs a tail that varies")

    return None


def _met(statistic, value):
    """Why an interval of ``statistic`` whose resamples put both its bounds on ``value``
    has none (Specification, Samples with no interval)."""
    _, unit, loss = _shown(statistic)
    shown = -value + 0.0 if loss else value + 0.0
    return (f"{statistic}: its resamples put both bounds on {shown:.2f}{unit}; an "
            f"interval needs resamples that differ between them")


def _pivot(statistic, r, x):
    """What a studentized interval of ``statistic`` is read on, along each row of the
    returns ``r`` whose excess returns are ``x``, and its influence series, with ``n``
    in every divisor: the growth a bar for total return, net profit and CAGR, the log
    of the deviation for volatility, the Sortino a bar, and the mean of the ``k``
    lowest returns for conditional value at risk."""
    count = r.shape[1]

    # a resample whose returns never vary, or never fall below the rate, has a pivot
    # or an influence its ratios read as a move with no error to measure it
    with np.errstate(divide="ignore", invalid="ignore"):
        if statistic in GROWTH:
            grown = np.log1p(r)
            g     = grown.mean(axis=1, keepdims=True)
            return g[:, 0], grown - g

        if statistic == "volatility_pct":
            centred = r - r.mean(axis=1, keepdims=True)
            s2      = (centred * centred).mean(axis=1, keepdims=True)
            return (0.5 * np.log(s2[:, 0] * count / (count - 1.0)),
                    (centred * centred - s2) / (2.0 * s2))

        if statistic == "sortino":
            mu  = x.mean(axis=1, keepdims=True)
            low = np.minimum(x, 0.0) ** 2
            d2  = low.mean(axis=1, keepdims=True)
            d   = np.sqrt(d2)
            return (_ranked_all(mu[:, 0], d[:, 0], 1.0),
                    (x - mu) / d - mu / (2.0 * d * d2) * (low - d2))

    k      = (count + 19) // 20
    lowest = np.partition(r, k - 1, axis=1)
    q      = lowest[:, k - 1:k]
    tail   = lowest[:, :k].mean(axis=1, keepdims=True)
    return tail[:, 0], q + (r - q) * (r <= q) * (count / k) - tail


def _ratios(statistic, r, x, estimate, m):
    """Each resample's studentized statistic: its pivot's move from the sample's
    ``estimate`` over its own standard error, at the sample's lag ``m``, infinite in
    the sign of a move it has no error to measure, and 0 where it did not move."""
    theta, psi = _pivot(statistic, r, x)

    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.sqrt(_long_run(psi, m) / (r.shape[1] - 1))
        t     = np.where(np.isfinite(error) & (error > 0.0), (theta - estimate) / error,
                         np.sign(theta - estimate) * math.inf)

    return np.where(np.isnan(t), 0.0, t)


def _unpivoted(statistic, theta, count, rate):
    """A bound of ``statistic`` from its pivot's bound ``theta``, on ``count`` returns at
    ``rate`` bars a year."""
    with np.errstate(over="ignore"):
        if statistic in ("total_return_pct", "net_profit_pct"):
            return float(np.expm1(count * theta)) * 100.0
        if statistic == "cagr_pct":
            return min(float(np.expm1(rate * theta)), CEILING) * 100.0
        if statistic == "volatility_pct":
            return float(np.exp(theta)) * math.sqrt(rate) * 100.0

    if statistic == "sortino":
        return theta * math.sqrt(rate)
    return -theta * 100.0


def _long_run(psi, m):
    """Each row's Bartlett sum of the influence series ``psi`` at lag ``m``, floored at
    its value at lag zero. ``n * (m + 1)`` times the sum is the sum of the squared
    sums of every window of ``m + 1`` rows, the rows outside the series zero, so a
    resample costs its length at any lag."""
    rows, count = psi.shape
    prefix = np.concatenate([np.zeros((rows, 1)), np.cumsum(psi, axis=1)], axis=1)
    starts = np.arange(-m, count)
    sums   = prefix[:, np.minimum(starts + m + 1, count)] - prefix[:, np.maximum(starts, 0)]
    summed = (sums * sums).sum(axis=1) / (count * (m + 1.0))
    return np.maximum(summed, (psi * psi).sum(axis=1) / count)


def _mid_ranks(values):
    """Each value's mid-rank, 1 for the least, equal values sharing the mean of the
    ranks they span."""
    order  = np.argsort(values, kind="stable")
    held   = values[order]
    starts = np.flatnonzero(np.concatenate([[True], held[1:] != held[:-1]]))
    ends   = np.append(starts[1:], len(values))
    out    = np.empty(len(values))
    out[order] = np.repeat((starts + ends - 1) / 2.0 + 1.0, ends - starts)
    return out


def _forecast(predicted, returned, bars, lag, paths):
    """The Forecast of ``predicted`` against ``returned``, ``bars`` each row's bar, at
    ``lag``: the influence of each row on the correlation, summed over its bar, the
    bars between holding none adding 0, gives the Newey-West variance over the rows'
    count squared; or why there is none, fewer than 3 rows or a side that never varies
    (Specification, A model)."""
    count = len(predicted)
    if count < 3:
        return f"{count} row{'s' * (count != 1)} with a return, fewer than 3"

    for values, said in ((predicted, "predictions"), (returned, "returns")):
        if np.all(values == values[0]):
            return f"the {said} never vary"

    u, v = ((ranks - ranks.mean()) / ranks.std()
            for ranks in (_mid_ranks(predicted), _mid_ranks(returned)))
    rho     = float(u @ v) / count
    summed  = np.bincount(bars - bars.min(), weights=u * v - rho * (u * u + v * v) / 2.0)
    covered = np.array([float(summed[j:] @ summed[:len(summed) - j])
                        if j < len(summed) else 0.0 for j in range(lag + 1)])
    error   = math.sqrt(max(_bartlett(covered, lag), 0.0)) / count
    squared = 1.0 - float(((returned - predicted) ** 2).sum()) / float(
        (returned * returned).sum())
    return Forecast(rho, error, rho / error if error > 0.0 else math.nan, lag, squared,
                    count, paths)


def _trips_core(pnl, opening, statistic, level, resamples, seed):
    """A trade statistic's interval by a bootstrap of the closed round trips' ``pnl``
    that can support one, beside None, or None beside why its resamples give none."""
    try:
        estimate, low, high = _trade_bounds(pnl, opening, statistic, level, resamples,
                                            _drawn(pnl, "stats trades", seed))
    except ProbatumValueError as error:
        return None, str(error)

    if low == high:
        return None, _met(statistic, low)

    return Interval(statistic, estimate, low, high, level, TRADES[statistic][2],
                    pnl.size), None


def _trade_bounds(pnl, opening, statistic, level, resamples, rng):
    """A trade statistic's estimate and bounds from ``resamples`` draws of the round
    trips' ``pnl``, each as many trips drawn evenly with replacement."""
    count = pnl.size
    taken = pnl[rng.integers(0, count, (resamples, count))]
    low_q, high_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0

    if statistic == "win_rate":
        values = (taken > 0.0).mean(axis=1)
        return (np.count_nonzero(pnl > 0.0) / count, _ordered(values, low_q),
                _ordered(values, high_q))

    if statistic == "profit_factor":
        gains, losses = np.where(pnl > 0.0, pnl, 0.0), np.where(pnl < 0.0, -pnl, 0.0)
        estimate = gains.sum() / losses.sum()
        values   = _ranked_all(np.where(taken > 0.0, taken, 0.0).sum(axis=1),
                               np.where(taken < 0.0, -taken, 0.0).sum(axis=1), 1.0)
        return (estimate, *_corrected(values, estimate,
                                      (gains.sum() - gains) / (losses.sum() - losses),
                                      low_q, high_q))

    m, s, root = pnl.mean(), pnl.std(ddof=1), math.sqrt(count)
    means, spreads = taken.mean(axis=1), taken.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(spreads > 0.0, (means - m) / (spreads / root),
                     np.sign(means - m) * math.inf)
    t = np.where(np.isnan(t), 0.0, t)
    return (m / opening * 100.0, (m - _ordered(t, high_q) * s / root) / opening * 100.0,
            (m - _ordered(t, low_q) * s / root) / opening * 100.0)


def _corrected(values, estimate, jackknife, low_q, high_q):
    """The bias-corrected and accelerated bounds of resampled ``values`` around
    ``estimate``, its acceleration read from the ``jackknife``, each value with one
    observation left out (Efron 1987)."""
    below = (np.count_nonzero(values < estimate)
             + 0.5 * np.count_nonzero(values == estimate)) / values.size
    if not 0.0 < below < 1.0:
        raise ProbatumValueError(f"resamples: {values.size:,}, every one of them on one "
                                 f"side of the profit factor's estimate; give more, since "
                                 f"its bias correction needs some on each side")

    z0     = NORMAL.inv_cdf(below)
    away   = jackknife.mean() - jackknife
    a      = float((away ** 3).sum() / (6.0 * float((away ** 2).sum()) ** 1.5))
    bounds = []
    for q in (low_q, high_q):
        z = NORMAL.inv_cdf(q)
        bounds.append(_ordered(values, NORMAL.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))))

    return tuple(bounds)


def _drawn(values, name, seed):
    """The stream ``name`` seeded by a 64-bit BLAKE2b digest of the float64 ``values``
    it resamples, or by ``seed`` where one is given (ADR 0021)."""
    if seed is None:
        blob = np.ascontiguousarray(values, dtype="<f8").tobytes()
        seed = int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "little")

    return stream(seed, name)


def _block_lengths(y):
    """The stationary and the circular block lengths that Politis and White's rule gives
    ``y``, as Patton's opt_block_length_REV_dec07 computes them, a length held to at
    least 1, and 1 for a series too short or too flat for the rule to read."""
    count = y.size
    m_max = math.ceil(math.sqrt(count)) + 5
    b_max = math.ceil(min(3.0 * math.sqrt(count), count / 3.0))
    if count < m_max + 3 or np.all(y == y[0]):
        return 1.0, 1

    h     = 2.0 * math.sqrt(math.log10(count) / count)
    later = y[m_max:] - y[m_max:].mean()
    rho   = np.empty(m_max)

    for j in range(1, m_max + 1):
        side   = y[m_max - j:count - j]
        side   = side - side.mean()
        bottom = float(later @ later) * float(side @ side)
        if bottom == 0.0:
            return 1.0, 1
        rho[j - 1] = float(later @ side) / math.sqrt(bottom)

    small = np.abs(rho) < h
    m_hat = next((j for j in range(1, m_max - 3) if small[j - 1:j + 4].all()), None)
    if m_hat is None:
        m_hat = int(np.flatnonzero(np.abs(rho) > h)[-1]) + 1

    top  = min(2 * m_hat, m_max)
    head = y[top:] - y[top:].mean()
    r    = np.empty(top + 1)
    for j in range(top + 1):
        side = y[top - j:count - j]
        r[j] = float(head @ (side - side.mean())) / (head.size - 1)

    v        = np.arange(1, top + 1) / top
    w        = np.where(v < 0.5, 1.0, 2.0 * (1.0 - v))
    spectral = float(r[0] + 2.0 * (w * r[1:]).sum())

    if spectral == 0.0:
        raw = math.inf
    else:
        raw = (abs(2.0 * float((w * np.arange(1, top + 1) * r[1:]).sum()) / spectral)
               ** (2.0 / 3.0) * count ** (1.0 / 3.0))

    return (max(1.0, min(raw, b_max)),
            max(1, math.ceil(min(1.5 ** (1.0 / 3.0) * raw, b_max))))


def _return_block(r):
    """The mean block of a stationary bootstrap of the returns ``r``: the longest
    stationary length of the returns, their squared deviations from their mean and the
    indicator of a return at or below the value at risk's."""
    k      = (r.size + 19) // 20
    cutoff = np.partition(r, k - 1)[k - 1]
    return max(_block_lengths(series)[0] for series in (
        r, (r - r.mean()) ** 2, (r <= cutoff).astype(np.float64)))


def _stationary(count, block, resamples, rng):
    """The rows of ``resamples`` stationary resamples of ``count`` returns, some at a
    time: each first row drawn evenly, and each next the one after the last, wrapping
    past the last return, with probability ``1 - 1/block``, and drawn evenly again
    otherwise (Politis and Romano 1994)."""
    chunk = max(1, 2_000_000 // count)
    at    = np.arange(count)

    for first in range(0, resamples, chunk):
        rows  = min(chunk, resamples - first)
        fresh = rng.random((rows, count)) < 1.0 / block
        fresh[:, 0] = True
        starts = rng.integers(0, count, (rows, count))
        began  = np.maximum.accumulate(np.where(fresh, at, 0), axis=1)
        yield (np.take_along_axis(starts, began, axis=1) + (at - began)) % count


def _resampled(r, x, rate, rows):
    """Each return statistic on the resamples ``rows`` of the returns ``r``, whose
    excess returns are ``x``, computed as the record computes it, a resampled equity
    that reaches zero moving no further."""
    taken, excess = r[rows], x[rows]
    grown = np.cumprod(1.0 + taken, axis=1)
    gone  = grown <= 0.0
    for row in np.flatnonzero(gone.any(axis=1)):
        at = int(gone[row].argmax())
        grown[row, at + 1:] = grown[row, at]

    final = grown[:, -1]
    with np.errstate(over="ignore", invalid="ignore"):
        growth = np.where(final > 0.0, np.minimum(
            np.power(np.where(final > 0.0, final, 1.0), rate / r.size) - 1.0, CEILING),
            -1.0)

    path   = np.concatenate([np.ones((len(grown), 1)), grown], axis=1)
    peak   = np.maximum.accumulate(path, axis=1)
    fall   = np.minimum((peak - path) / peak, 1.0).max(axis=1)
    k      = (r.size + 19) // 20
    lowest = np.partition(taken, k - 1, axis=1)[:, :k]
    total  = (final - 1.0) * 100.0

    return {"total_return_pct": total, "net_profit_pct": total, "cagr_pct": growth * 100.0,
            "volatility_pct": taken.std(axis=1, ddof=1) * math.sqrt(rate) * 100.0,
            "sortino": _ranked_all(excess.mean(axis=1),
                                   np.sqrt((np.minimum(excess, 0.0) ** 2).mean(axis=1)),
                                   math.sqrt(rate)),
            "max_drawdown_pct": fall * 100.0,
            "calmar": np.where(growth < 0.0, growth * fall, _ranked_all(growth, fall, 1.0)),
            "value_at_risk": -lowest.max(axis=1) * 100.0,
            "conditional_value_at_risk": -lowest.mean(axis=1) * 100.0}


def _ranked_all(reward, risk, annualizer):
    """``_ranked`` over arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(risk > 0.0, reward / np.where(risk > 0.0, risk, 1.0) * annualizer,
                        np.where(reward > 0.0, math.inf, 0.0))


def _ordered(values, q):
    """The value of rank ``_rank(B, q)`` among the ``B`` values."""
    rank = _rank(values.size, q)
    return float(np.partition(values, rank - 1)[rank - 1])


def _rank(count, q):
    """The rank ``ceil(count * q)``, a rank within 1e-9 of a whole number being that
    number, the first at least (Specification, Intervals of the returns)."""
    return max(1, math.ceil(count * q - NUDGE))


def _block_starts(count, length, resamples, rng):
    """The blocks' starting rows of ``resamples`` circular block resamples of ``count``
    rows, some resamples at a time, each joining ``ceil(count / length)`` blocks."""
    blocks = -(-count // length)
    chunk  = max(1, 2_000_000 // count)
    for done in range(0, resamples, chunk):
        yield rng.integers(0, count, (min(chunk, resamples - done), blocks))


def _studentized(first, second, d, length, starts):
    """Ledoit and Wolf's studentized statistics ``abs(D* - D) / s*`` of the circular
    block resamples whose blocks start at ``starts``, each block ``length`` rows, the
    same rows of both excess series, a resample's natural standard error read from its
    own influence series over its whole blocks."""
    count   = first.size
    whole   = count // length
    offsets = np.arange(length)
    found   = []

    for chunk in starts:
        rows  = len(chunk)
        taken = ((chunk[:, :, None] + offsets) % count).reshape(rows, -1)[:, :count]
        a, b = first[taken], second[taken]

        # a resample whose returns never vary, or whose standard error is zero, has
        # no ratio to take, and counts as infinite
        with np.errstate(divide="ignore", invalid="ignore"):
            z      = _influences(a) - _influences(b)
            sums   = z[:, :whole * length].reshape(rows, whole, length).sum(axis=2)
            spread = np.sqrt((sums * sums / length).mean(axis=1) / (count - 1))
            moved  = (a.mean(axis=1) / a.std(axis=1, ddof=1)
                      - b.mean(axis=1) / b.std(axis=1, ddof=1))
            ratio = np.abs(moved - d) / spread

        found.append(np.where(np.isnan(ratio), math.inf, ratio))

    return np.concatenate(found)


def _influences(x):
    """``_influence`` along each row of ``x``."""
    mu = x.mean(axis=1, keepdims=True)
    s  = np.sqrt(((x - mu) ** 2).mean(axis=1, keepdims=True))
    u  = (x - mu) / s
    return u - (mu / s) / 2.0 * (u * u - 1.0)


def _newey_west(rows, e, m):
    """``(X'X)^-1 * W * (X'X)^-1`` for the regressors ``rows`` and residuals ``e``, with
    Bartlett weights at lag ``m`` and no small-sample factor."""
    scored = rows * e[:, None]
    w      = scored.T @ scored
    for j in range(1, m + 1):
        joint = scored[j:].T @ scored[:-j]
        w += (1.0 - j / (m + 1.0)) * (joint + joint.T)

    bread = np.linalg.inv(rows.T @ rows)
    return bread @ w @ bread


def _deeper(falls, fall):
    """The share of the drawdowns ``falls`` deeper than ``fall`` by more than a part in a
    trillion, so orders that differ only by rounding count as level (Specification, The
    order of trades)."""
    return np.count_nonzero(falls > fall * (1.0 + 1e-12)) / falls.size


def _paths(returned, pnl, orders):
    """The drawdown at trade closes, in percent, and the longest losing run of each
    order of round trips in ``orders``, compounding each trip's return from 1 and
    staying at zero once there."""
    grown  = np.cumprod(np.maximum(1.0 + returned[orders], 0.0), axis=1)
    path   = np.concatenate([np.ones((len(grown), 1)), grown], axis=1)
    peak   = np.maximum.accumulate(path, axis=1)
    falls  = np.minimum((peak - path) / peak, 1.0).max(axis=1) * 100.0
    losing = pnl[orders] < 0.0
    run = longest = np.zeros(len(orders), dtype=np.int64)
    for column in losing.T:
        run     = np.where(column, run + 1, 0)
        longest = np.maximum(longest, run)

    return falls, longest


def _maybe(value):
    return None if value is None else float(value)


def _most(trials):
    """Bailey and Lopez de Prado's approximation of the expected maximum of ``trials``
    standard normal draws (Specification, The deflated Sharpe)."""
    return ((1.0 - EULER) * NORMAL.inv_cdf(1.0 - 1.0 / trials)
            + EULER * NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e)))


def _looks(name, value):
    """A count of trials passed as ``name``, a whole number from 2."""
    if not _whole(value):
        raise ProbatumTypeError(f"{name}: {value!r}; give a whole number of trials from 2")
    if value < 2:
        raise ProbatumValueError(f"{name}: {value}; a deflation corrects for more than "
                                 f"one trial, so give 2 or more")

    return int(value)


def _searched(study, name):
    """``study``, passed as ``name``, refused unless it is a search's Study."""
    from ._search import Study
    if not isinstance(study, Study):
        raise ProbatumTypeError(f"{name}: {type(study).__name__}; give the Study of a "
                                f"search")

    if study.kind != "trials":
        raise ProbatumValueError(f"{name}: a validation's Study, whose paths are no "
                                 f"competing configurations; give the Study of a search")

    return study


def _trials(trials, periods_per_year, risk_free, states):
    """The excess returns a statistic of trials reads, a row a trial, the bars a year it
    annualises by and each row's trial: a search's trials in ``states`` after the
    in-sample bars' first row, at the Study's annualisation and risk-free rate, or a
    matrix's rows at the rate it is given (Specification, What a statistic of trials
    reads)."""
    from ._search import Study
    if isinstance(trials, Study):
        study = _searched(trials, "trials")
        given = study.provenance
        if periods_per_year is not None:
            raise ProbatumTypeError(f"periods_per_year: given beside a Study, which ran "
                                    f"at {given['periods_per_year']}; leave it out")
        if risk_free is not None:
            raise ProbatumTypeError(f"risk_free: given beside a Study, which ran at "
                                    f"{given['risk_free']}; leave it out")
        if study.returns is None:
            raise ProbatumValueError("trials: a Study kept with keep=\"none\", which "
                                     "holds no returns; search with keep=\"all\"")

        table = study.trials
        rows  = np.isin(np.asarray(table["state"]), states)
        rate  = given["periods_per_year"]
        return (np.asarray(study.returns)[rows, 1:] - given["risk_free"] / rate, rate,
                np.asarray(table["number"])[rows])

    rate, free = _annual(periods_per_year, risk_free, "a matrix of returns")
    values = _matrix(trials)
    return values - free / rate, rate, np.arange(values.shape[0])


def _annual(periods_per_year, risk_free, what):
    """The bars a year and the annual risk-free rate given beside ``what``."""
    if periods_per_year is None:
        raise ProbatumTypeError(f"periods_per_year: missing for {what}; pass 365 for "
                                f"daily crypto returns")

    rate = _finite("periods_per_year", periods_per_year, "the bars a year, such as 365")
    if rate <= 0.0:
        raise ProbatumValueError(f"periods_per_year: {periods_per_year!r}; give the bars "
                                 f"a year, such as 365")

    free = 0.0 if risk_free is None else _finite(
        "risk_free", risk_free, "the annual rate as a finite number, such as 0.04")
    return rate, free


def _matrix(trials):
    """A matrix of trials' returns, a 2-D array of a row a trial, finite."""
    if isinstance(trials, (str, bytes, dict)) or not hasattr(trials, "__len__"):
        raise ProbatumTypeError(f"trials: {type(trials).__name__}; pass a search's Study, "
                                f"or a 2-D array of a row a trial's returns")

    try:
        values = np.asarray(trials, dtype=np.float64)
    except (TypeError, ValueError):
        raise ProbatumTypeError(f"trials: a {type(trials).__name__} that holds no "
                                f"numbers; pass a 2-D array of a row a trial's "
                                f"returns") from None

    if values.ndim != 2 or not values.size:
        raise ProbatumValueError(f"trials: an array of shape {values.shape}; pass a row "
                                 f"a trial's returns, 2-D")

    bad = np.argwhere(~np.isfinite(values))
    if bad.size:
        trial, at = bad[0]
        raise ProbatumValueError(f"trials: trial {trial} holds {values[trial, at]} at "
                                 f"position {at}; a matrix of returns holds finite "
                                 f"numbers")

    return values


def _onc(rows, rng):
    """Lopez de Prado and Lewis's clustering of ``rows``, the returns of three or more
    trials that vary, its starts drawn from ``rng``: each trial's cluster, numbered by
    its first trial, and the clustering's quality (Specification, The effective number
    of trials)."""
    labels = _clustered(rows, rng)
    return labels, _quality(_silhouettes(_distances(rows)[1], labels))


def _clustered(rows, rng):
    """The base clustering of ``rows`` and its refinement: where more than 2 clusters
    fall below the clusters' mean quality, their trials clustered again on their own,
    the new clusters kept where their mean quality is above that of those they
    replace."""
    count = rows.shape[0]
    if count < 3:
        return np.arange(count)

    points, far = _distances(rows)
    labels = _base(points, far, rng)
    scores = _silhouettes(far, labels)
    each   = {label: _quality(scores[labels == label])
              for label in range(int(labels.max()) + 1) if (labels == label).sum() > 1}
    if not each:
        return labels

    middle = np.mean(list(each.values()))
    redo   = sorted(label for label, value in each.items() if value < middle)
    if len(redo) <= 2:
        return labels

    items  = np.flatnonzero(np.isin(labels, redo))
    merged = labels.copy()
    merged[items] = labels.max() + 1 + _clustered(rows[items], rng)

    merged  = _numbered(merged)
    renewed = _silhouettes(far, merged)
    after   = [_quality(renewed[merged == label]) for label in np.unique(merged[items])
               if (merged == label).sum() > 1]
    if after and np.mean(after) > np.mean([each[label] for label in redo]):
        return merged
    return labels


def _distances(rows):
    """Each trial's row of ``D = sqrt((1 - rho) / 2)`` and the Euclidean distances
    between those rows."""
    d = np.sqrt(np.clip((1.0 - np.corrcoef(rows)) / 2.0, 0.0, None))
    np.fill_diagonal(d, 0.0)

    sq  = (d * d).sum(axis=1)
    far = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * d @ d.T, 0.0))
    np.fill_diagonal(far, 0.0)
    return d, far


def _base(points, far, rng):
    """The clustering of the highest quality over the starts and each ``k``, the first
    on ties."""
    squared = far * far
    most    = min(points.shape[0] - 1, MOST)
    best    = None

    for _ in range(STARTS):
        order = _sequence(squared, most, rng)
        for k in range(2, most + 1):
            labels = _numbered(_lloyd(points, order[:k]))
            score  = _quality(_silhouettes(far, labels))
            if best is None or score > best[0]:
                best = (score, labels)

    return best[1]


def _sequence(squared, most, rng):
    """One greedy k-means++ sequence of ``most`` trials, at ``squared`` distances: the
    first drawn evenly, and each next the best of ``2 + floor(ln most)`` drawn in
    proportion to their squared distance from the sequence so far."""
    count = squared.shape[0]
    tries = 2 + int(math.log(most))
    order = [int(rng.integers(count))]
    near  = squared[order[0]].copy()

    for _ in range(1, most):
        cumulative = np.cumsum(near)
        if cumulative[-1] > 0.0:
            picks = np.minimum(np.searchsorted(cumulative,
                                               rng.random(tries) * cumulative[-1],
                                               side="right"), count - 1)
        else:
            picks = rng.integers(count, size=tries)

        found = squared[picks]
        best  = int(np.argmin(np.minimum(near, found).sum(axis=1)))
        order.append(int(picks[best]))
        near = np.minimum(near, found[best])

    return order


def _lloyd(points, centres):
    """Lloyd's iterations from the trials ``centres``: each trial to its nearest centre,
    the lowest-numbered on ties, each cluster left empty taking in turn the trial
    farthest from its own centre, and one emptied by that keeping its centre, until no
    trial moves."""
    count, k = points.shape[0], len(centres)
    rows   = np.arange(count)
    norms  = (points * points).sum(axis=1)
    middle = points[centres]
    labels = None

    for _ in range(LLOYD):
        near = norms[:, None] + (middle * middle).sum(axis=1) - 2.0 * points @ middle.T
        new  = np.argmin(near, axis=1)
        if labels is not None and np.array_equal(new, labels):
            break

        labels = new
        moved  = np.zeros(count, dtype=bool)
        for j in range(k):
            if (labels == j).any():
                continue
            worst = int(np.argmax(np.where(moved, -np.inf, near[rows, labels])))
            labels[worst] = j
            moved[worst]  = True

        onehot = np.zeros((count, k))
        onehot[rows, labels] = 1.0
        sizes  = onehot.sum(axis=0)
        middle = np.where(sizes[:, None] > 0.0,
                          (onehot.T @ points) / np.maximum(sizes, 1.0)[:, None], middle)

    return labels


def _numbered(labels):
    """``labels`` with each cluster numbered by its first trial."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(first.size)
    return rank[inverse]


def _silhouettes(far, labels):
    """Each trial's silhouette, ``(b - a) / max(a, b)``, 0 for a trial alone in its
    cluster."""
    count  = labels.size
    rows   = np.arange(count)
    onehot = np.zeros((count, int(labels.max()) + 1))
    onehot[rows, labels] = 1.0

    sizes = onehot.sum(axis=0)
    sums  = far @ onehot
    mates = sizes[labels] - 1.0
    a     = np.where(mates > 0.0, sums[rows, labels] / np.maximum(mates, 1.0), 0.0)
    means = sums / sizes
    means[rows, labels] = np.inf

    b   = means.min(axis=1)
    top = np.maximum(a, b)
    return np.where((mates > 0.0) & (top > 0.0), (b - a) / np.where(top > 0.0, top, 1.0),
                    0.0)


def _quality(scores):
    """Silhouettes' mean over their deviation, ``n`` in its divisor, infinite at no
    deviation and a mean above 0, and 0 at no deviation otherwise."""
    spread = float(scores.std())
    if spread == 0.0:
        return math.inf if scores.mean() > 0.0 else 0.0
    return float(scores.mean()) / spread


def _overfit(x, raw, rate, objective, maximize, blocks):
    """The share of the halvings of ``blocks`` contiguous blocks whose in-sample winner
    ranks below the median out of sample, tied winners weighted alike, and the share
    whose winner's statistic is below 0 there, each half's statistic merged from its
    blocks' sums as the record computes it: the Sharpe the excess returns' ``x`` mean
    over the returns' ``raw`` deviation, Sortino of the excess returns, and the rest of
    the returns. A halving is read with its complement, a chunk of them at a time
    (Specification, The probability of backtest overfitting)."""
    trials, count = x.shape
    size   = count // blocks
    first  = count - size * blocks
    excess = x[:, first:].reshape(trials, blocks, size)
    plain  = raw[:, first:].reshape(trials, blocks, size)
    picks  = np.array(list(itertools.combinations(range(blocks), blocks // 2)))
    mask   = np.zeros((len(picks), blocks))
    mask[np.arange(len(picks))[:, None], picks] = 1.0

    n      = size * (blocks // 2)
    sums   = plain.sum(axis=2)
    means  = sums / size
    within = ((plain - means[:, :, None]) ** 2).sum(axis=2)
    ahead  = excess.sum(axis=2)
    below  = (np.minimum(excess, 0.0) ** 2).sum(axis=2)
    grown  = np.prod(1.0 + plain, axis=2)

    def halves(chosen):
        if objective in ("sharpe", "sortino", "volatility_pct"):
            mean    = (chosen @ sums.T) / n
            between = chosen @ (size * means.T ** 2) - n * mean ** 2
            sd      = np.sqrt(np.maximum(chosen @ within.T + between, 0.0) / (n - 1))
            if objective == "volatility_pct":
                return sd * math.sqrt(rate) * 100.0
            risk = sd if objective == "sharpe" else np.sqrt((chosen @ below.T) / n)
            return _ranked_all((chosen @ ahead.T) / n, risk, math.sqrt(rate))

        growth = np.ones((len(chosen), trials))
        for b in range(blocks):
            growth *= np.where(chosen[:, b:b + 1] > 0.0, grown[:, b], 1.0)

        if objective != "cagr_pct":
            return (growth - 1.0) * 100.0

        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(growth > 0.0, np.minimum(
                np.power(np.where(growth > 0.0, growth, 1.0), rate / n) - 1.0, CEILING),
                -1.0) * 100.0

    halvings = len(picks)
    step     = max(1, AT_ONCE // trials)
    over = loss = 0.0
    for start in range(0, halvings // 2, step):
        these = np.arange(start, min(start + step, halvings // 2))
        value = halves(mask[np.concatenate([these, halvings - 1 - these])])
        for inside, outside in ((value[:these.size], value[these.size:]),
                                (value[these.size:], value[:these.size])):
            for held, other in zip(inside, outside):
                winners = np.flatnonzero(held == (held.max() if maximize else held.min()))
                for k in winners:
                    worse = np.count_nonzero(other < other[k] if maximize else
                                             other > other[k])
                    rank  = worse + (np.count_nonzero(other == other[k]) + 1) / 2.0
                    over += (rank / (trials + 1) < 0.5) / winners.size
                    loss += (other[k] < 0.0) / winners.size

    return over / halvings, loss / halvings


def _window(count, q):
    """Hansen's kernel as a spectral window: weights on a series' periodogram that give
    its stationary-bootstrap variance of ``sqrt(n)`` times a resample's mean at a
    restart probability ``q``, the circular form of ``g_0 + 2 * sum_i kappa(n, i) *
    g_i``."""
    k    = np.arange(count)
    w    = np.where(k == 0, 1.0, 2.0 * (1.0 - k / count) * (1.0 - q) ** k)
    full = np.fft.fft(w).real
    bins = full[:count // 2 + 1].copy()
    bins[1:(count + 1) // 2] *= 2.0
    return bins / count ** 2


def _spectral(rows, weights):
    """Each row's stationary-bootstrap variance through the window ``weights``."""
    centred = rows - rows.mean(axis=1, keepdims=True)
    return (np.abs(np.fft.rfft(centred, axis=1)) ** 2) @ weights


def _combined(rows):
    """A cluster's series: its members' returns at minimum variance under one
    correlation, inverse variance where that correlation is out of its range, or a lone
    member's own."""
    if rows.shape[0] == 1:
        return rows[0]

    m    = rows.shape[0]
    sd   = rows.std(axis=1, ddof=1)
    rbar = float(np.corrcoef(rows)[np.triu_indices(m, 1)].mean())

    if -1.0 / (m - 1) < rbar < 1.0:
        raw = 1.0 / sd ** 2 - rbar * (1.0 / sd).sum() / ((1.0 + (m - 1) * rbar) * sd)
    else:
        raw = 1.0 / sd ** 2

    return (raw / raw.sum()) @ rows


def _family(study, held, bar):
    """The probabilistic Sharpe against ``bar`` of the series the winner's cluster
    combines into, or of the winner's run where its returns never vary, so that it is
    in no cluster."""
    winner = study._winner.number
    if winner not in held.numbers:
        return probabilistic_sharpe(study.best, benchmark=bar)

    x, rate, _ = _trials(study, None, None, ("complete", "floored"))
    rows   = x[np.ptp(x, axis=1) > 0.0]
    labels = np.asarray(held.clusters)
    member = labels == labels[held.numbers.index(winner)]
    return probabilistic_sharpe(_combined(rows[member]), benchmark=bar,
                                periods_per_year=rate)
