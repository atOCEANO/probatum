"""A model is validated by fitting, not searching: before every fit a fresh estimator
is made, so a fold predicts alike whether or not the model given had been fitted on all
the data; each fit reads its split's training rows that are available, have finite
features and a return and are left by the purge and the embargo, in time-major order
with their uniqueness as sample weights; each path is one array run of its windows'
predictions through to_weights with the labels' barriers, scored against the labels as
a forecast; and the final fit is the Study's best and its holdout. Features that read
ahead are refused before any fit and predictions or weights that do before any path,
as is any of them that differs from one run to the next; a NaN or an infinite feature
leaves its row out unless filled; and every refusal names what to give instead.
"""

import math
import warnings

import numpy as np
import pytest

import probatum
from probatum import (DataError, LookaheadError, PolicyError, ProbatumRuntimeError,
                      ProbatumTypeError, ProbatumValueError)
from probatum._data import _before
from reference import differential, statistics

VENUE = probatum.Venue(fees={"maker": 0.0002, "taker": 0.0006})
CV    = probatum.cv.CombinatorialPurged(4, 2)
SIZE  = 0.03


def walked(count=1_200, width=3, seed=11):
    """``width`` linear perpetuals of random walks on an hourly clock, with volumes that
    vary, as a Data."""
    rng    = np.random.default_rng(seed)
    close  = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (count, width)), axis=0))
    opened = np.vstack([close[:1], close[:-1]])
    bars   = np.stack([opened, np.maximum(opened, close) * 1.003,
                       np.minimum(opened, close) * 0.997, close,
                       rng.uniform(5e5, 2e6, (count, width))], axis=-1)
    times  = 1_704_067_200 * 10**9 + np.arange(count) * differential.HOUR
    return probatum.Data(
        bars,
        symbol   = [f"C{k}USDT" for k in range(width)],
        market   = "linear",
        time     = times,
        interval = "1h",
    )


DATA   = walked()
LABELS = probatum.labels(DATA, VENUE, 8, 10_000.0, stop=SIZE, take_profit=SIZE)
# a cut some labels end on, which the final fit leaves out
CUT    = 964
# a value as a refusal shows it, a plain float
NUMBER = r"-?\d[\d.e+-]*"


def features(data):
    """Each instrument's return over 6 bars and its RSI over 14, ``(T, N, 2)``."""
    return np.stack([probatum.ta.roc(data.closes, 6), probatum.ta.rsi(data.closes, 14)],
                    axis=-1)


def weights(predicted):
    return np.clip(predicted * 10.0, 0.0, 0.2)


class Resuming:
    """A model whose fit resumes from what it saw before, predicting each row's first
    feature times the mean of every return it has been fitted on."""


    def get_params(self, deep=True):
        return {}


    def set_params(self, **params):
        return self


    def fit(self, x, y, sample_weight=None):
        self.seen = [*getattr(self, "seen", []), *y.tolist()]
        return self


    def predict(self, x):
        return x[:, 0] * float(np.mean(self.seen))


class Counted:
    """A model that predicts each row's first feature, and records with each prediction
    how many rows its fit read."""

    predicted = []


    def get_params(self, deep=True):
        return {}


    def set_params(self, **params):
        return self


    def fit(self, x, y, sample_weight=None):
        self.rows = len(x)
        return self


    def predict(self, x):
        Counted.predicted.append(self.rows)
        return x[:, 0].copy()


class Leading:
    """A model that predicts each row's first feature whatever it fitted, and records
    what each fit read."""

    fitted = []


    def get_params(self, deep=True):
        return {}


    def set_params(self, **params):
        return self


    def fit(self, x, y, sample_weight=None):
        Leading.fitted.append((x.copy(), y.copy(), sample_weight.copy()))
        return self


    def predict(self, x):
        return x[:, 0].copy()


def validated(model=None, **given):
    """A validation of ``model`` on ``DATA``, the bars from ``CUT`` held out."""
    Leading.fitted.clear()
    return probatum.validate(Leading() if model is None else model, DATA, VENUE,
                             **{**dict(cv=CV, features=features, labels=LABELS,
                                       to_weights=weights, holdout=len(DATA) - CUT),
                                **given})


def usable():
    """The rows a fit or a window can read, available with every feature finite."""
    return DATA.available & np.isfinite(features(DATA)).all(axis=-1)


def test_each_fit_reads_its_split_s_purged_rows_in_time_major_order_with_their_weight():
    validated()

    grid  = features(DATA)[:CUT].reshape(CUT * 3, 2)
    ret   = LABELS.ret[:CUT].reshape(-1)
    held  = LABELS.weight[:CUT].reshape(-1)
    read  = usable()[:CUT].reshape(-1) & np.isfinite(ret)
    final = read & (LABELS.end[:CUT].reshape(-1) < CUT)

    assert (read & (LABELS.end[:CUT].reshape(-1) == CUT)).any()

    split = probatum.cv.CombinatorialPurged(4, 2, end=LABELS.end[:CUT])
    rows  = [np.flatnonzero(final)]
    rows += [train[read[train]] for train, _ in split.split(grid)]

    assert len(Leading.fitted) == len(rows) == 7
    for (x, y, weight), at in zip(Leading.fitted, rows):
        np.testing.assert_array_equal(x, grid[at])
        np.testing.assert_array_equal(y, ret[at])
        np.testing.assert_array_equal(weight, held[at])


def test_each_window_predicts_with_its_own_split_s_fit():
    # the final fit predicts first, as validate checks it, and last, as best
    Counted.predicted.clear()

    study  = validated(Counted())
    read   = dict(zip(study.fits["number"].tolist(), study.fits["rows"].tolist()))
    final  = study.provenance["model"]["final"]
    wanted = [read[fit] for fit in study.windows["fit"].tolist()]

    assert Counted.predicted[0] == Counted.predicted[-1] == final
    assert [rows for rows in Counted.predicted if rows != final] == wanted


def test_a_fit_counts_the_rows_the_purge_and_the_embargo_took():
    study    = validated()
    ends     = LABELS.end[:CUT]
    labelled = (ends != np.iinfo(np.int64).min) & (ends < CUT)
    embargo  = study.provenance["cv"]["embargo"]
    rows     = np.zeros(CUT * 3)
    clean    = probatum.cv.CombinatorialPurged(4, 2, embargo=0, end=ends).split(rows)
    full     = probatum.cv.CombinatorialPurged(4, 2, embargo=embargo, end=ends).split(rows)

    for fit, (kept, _), (train, _) in zip(study.fits, clean, full):
        outside = np.ones(CUT, dtype=bool)
        for span in fit["test"].split(", "):
            a, b = map(int, span.split(" to "))
            outside[a:b + 1] = False
        given = int((labelled & outside[:, None]).sum())
        assert (fit["train"], fit["purged"], fit["embargoed"]) == (
            len(train), given - len(kept), len(kept) - len(train))

    assert embargo > 0 and any(fit["embargoed"] for fit in study.fits)


def test_a_fold_predicts_alike_when_the_model_given_was_fitted_on_all_the_data():
    # a model whose fit resumes, so one fit that saw another's rows predicts otherwise
    grid   = features(DATA).reshape(-1, 2)
    ret    = LABELS.ret.reshape(-1)
    keep   = np.isfinite(grid).all(axis=1) & np.isfinite(ret)
    fresh  = validated(Resuming())
    fitted = validated(Resuming().fit(grid[keep], ret[keep]))

    for name in ("paths", "fits", "windows"):
        assert getattr(fresh, name) == getattr(fitted, name)

    np.testing.assert_array_equal(fresh.returns, fitted.returns)
    np.testing.assert_array_equal(fresh.best.returns, fitted.best.returns)
    assert fresh.forecast == fitted.forecast


def test_a_path_trades_its_windows_predictions_through_to_weights_with_the_barriers():
    study  = validated()
    inside = _before(DATA, CUT)
    seen   = np.where(usable()[:CUT], features(DATA)[:CUT, :, 0], np.nan)
    held   = probatum.backtest(weights(seen), inside, VENUE, stop=SIZE, take_profit=SIZE)

    assert study.paths["state"].tolist() == ["complete"] * 3
    for path in range(3):
        np.testing.assert_array_equal(study.returns[path, 1:], held.returns[1:])


def test_the_forecast_is_spearman_s_against_the_labels_with_a_newey_west_error():
    # a label that ends in the holdout is priced on its bars, so the forecast leaves it
    study = validated()
    rows  = usable()[:CUT] & np.isfinite(LABELS.ret[:CUT]) & (LABELS.end[:CUT] < CUT)
    t, i  = np.nonzero(rows)
    p, y  = features(DATA)[t, i, 0].tolist(), LABELS.ret[t, i].tolist()
    lag   = int((LABELS.end[t, i] - t).max())
    rho, error, squared = statistics.forecast(p, y, t.tolist(), lag)
    paths = study.paths

    assert paths["rank_ic"].tolist()   == pytest.approx([rho] * 3, rel=1e-12)
    assert paths["rank_ic_t"].tolist() == pytest.approx([rho / error] * 3, rel=1e-9)
    assert paths["r_squared"].tolist() == pytest.approx([squared] * 3, rel=1e-12)
    assert paths["predicted"].tolist() == [len(p)] * 3

    pooled = statistics.forecast(p * 3, y * 3, t.tolist() * 3, lag)
    found  = study.forecast

    assert (found.rows, found.paths, found.lag) == (3 * len(p), 3, lag)
    assert (found.rank_ic, found.error, found.r_squared) == pytest.approx(pooled,
                                                                          rel=1e-9)
    assert study.summary().to_pandas().loc[["rank_ic", "rank_ic_t"], "value"].tolist() == [
        found.rank_ic, found.t]


def test_best_is_the_final_fit_on_the_in_sample_bars_and_the_holdout_its_tail():
    study = validated()
    seen  = np.where(usable(), features(DATA)[:, :, 0], np.nan)
    whole = weights(seen)
    best  = probatum.backtest(whole[:CUT], _before(DATA, CUT), VENUE, stop=SIZE,
                              take_profit=SIZE)
    tail  = probatum.backtest(whole[CUT:], DATA.split(CUT)[1], VENUE, stop=SIZE,
                              take_profit=SIZE)

    assert (study.best_params, study.best_score) == ({}, best.stats["sharpe"])
    np.testing.assert_array_equal(study.best.returns, best.returns)
    np.testing.assert_array_equal(study.evaluate_holdout().returns, tail.returns)
    assert study.looks == 1

    with pytest.raises(ProbatumValueError, match="decay: a model's fits score no bars"):
        study.decay()


def test_features_that_read_ahead_are_refused_before_any_fit():
    def normalised(data):
        closes = data.closes
        return ((closes - closes.mean(axis=0)) / closes.std(axis=0))[:, :, None]

    with pytest.raises(LookaheadError, match=rf"features: feature 0 of C0USDT at row 0 "
                                             rf"\(2024-01-01 00:00\) is {NUMBER} on the "
                                             rf"whole Data and {NUMBER} on the Data cut "
                                             rf"after row 95; compute each row's "
                                             rf"features from the bars up to it"):
        validated(features=normalised)

    assert Leading.fitted == []


def test_features_that_raise_on_a_cut_alone_are_named_with_the_cut():
    def long(data):
        if len(data) < 1_000:
            raise ArithmeticError("too few bars")
        return features(data)

    with pytest.raises(ArithmeticError, match="too few bars") as raised:
        validated(features=long)

    assert raised.value.__notes__ == ["raised by the features on the Data cut after row "
                                      "95, in validate's causality check"]


def test_a_prediction_or_a_weight_that_reads_ahead_is_refused():
    class Peeking(Leading):
        """A model that predicts each row against the mean of every row it is given."""


        def predict(self, x):
            return x[:, 0] - x[:, 0].mean()

    with pytest.raises(LookaheadError, match=rf"policy: its prediction at row \d+ \(.*\) "
                                             rf"for C\dUSDT is {NUMBER} on the in-sample "
                                             rf"rows and {NUMBER} on those up to row 95; "
                                             rf"predict each row from its own features"):
        validated(Peeking())

    with pytest.raises(LookaheadError, match=rf"to_weights: its weight at row \d+ .* is "
                                             rf"{NUMBER} on the in-sample rows and "
                                             rf"{NUMBER} on those up to row 95; compute "
                                             rf"each row's weight from the predictions "
                                             rf"up to it"):
        validated(to_weights=lambda predicted: 0.2 * predicted / np.nanmax(
            np.abs(predicted)))


def test_a_prediction_moving_in_its_last_digits_with_the_rows_given_is_not_refused():
    class Rounded(Leading):
        """A model whose predictions move by parts in 10^12 with the rows it is given,
        as a sum taken in another order moves them, which the check bounds as the
        search's does, at 1e-9 of an instrument's largest."""


        def predict(self, x):
            return x[:, 0] * (1.0 + 1e-15 * len(x))

    assert validated(Rounded()).paths["state"].tolist() == ["complete"] * 3


class Drawing(Leading):
    """A model that draws noise into each prediction, unseeded."""


    def predict(self, x):
        return x[:, 0] + np.random.default_rng().normal(0.0, 1e-3, len(x))


@pytest.mark.parametrize("given, said", [
    (dict(features=lambda data: features(data) + np.random.default_rng().normal(
        0.0, 1e-3, (len(data), 3, 2))),
     rf"features: feature \d of C\dUSDT at row \d+ \(.*\) is {NUMBER} in one computation "
     rf"on the whole Data and {NUMBER} in the next"),
    (dict(model=Drawing()),
     rf"policy: its prediction at row \d+ \(.*\) for C\dUSDT is {NUMBER} in one run on "
     rf"the in-sample rows and {NUMBER} in the next"),
    (dict(to_weights=lambda predicted: weights(predicted) + np.random.default_rng(
        ).uniform(0.0, 1e-3, predicted.shape)),
     rf"to_weights: its weight at row \d+ \(.*\) for C\dUSDT is {NUMBER} in one run on "
     rf"the in-sample rows and {NUMBER} in the next"),
])
def test_what_differs_from_one_run_to_the_next_on_the_same_rows_is_refused(given, said):
    with pytest.raises(ProbatumValueError, match=said) as raised:
        validated(**given)

    assert type(raised.value) is ProbatumValueError
    assert str(raised.value).endswith("; seed every random draw it makes, so the check "
                                      "can compare its runs")


def test_a_prediction_refused_or_raising_on_a_cut_alone_is_named_with_the_cut():
    class Whole(Leading):
        """A model that predicts NaN, or raises, on fewer than 2,000 rows, as a cut's
        hold and the in-sample rows do not."""

        raises = False


        def predict(self, x):
            if len(x) < 2_000 and self.raises:
                raise ArithmeticError("too few rows")
            return x[:, 0] * (1.0 if len(x) >= 2_000 else math.nan)

    with pytest.raises(LookaheadError, match=r"policy: on the rows up to row 95 \(.*\) "
                                             r"its predictions are refused \(policy: "
                                             r"predicted nan for a row; predict a finite "
                                             r"number a row\), and on the in-sample rows "
                                             r"they are not; predict each row from its "
                                             r"own features") as raised:
        validated(Whole())

    assert isinstance(raised.value.__cause__, PolicyError)

    Whole.raises = True
    with pytest.raises(ArithmeticError, match="too few rows") as raised:
        validated(Whole())

    assert raised.value.__notes__ == ["raised on the in-sample rows up to row 95, in "
                                      "validate's causality check"]


def test_a_short_weight_on_spot_is_refused_naming_to_weights():
    spot   = probatum.Data(
        np.array(DATA.bars),
        symbol   = list(DATA.names),
        market   = "spot",
        time     = DATA.time,
        interval = "1h",
    )
    labels = probatum.labels(spot, VENUE, 8, 10_000.0, stop=SIZE, take_profit=SIZE)

    with pytest.raises(PolicyError, match=r"to_weights: -0\.\d+ at row \d+ \(.*\) for "
                                          r"C\dUSDT, which is spot and cannot be short; "
                                          r"return weights from 0 for it"):
        probatum.validate(Leading(), spot, VENUE, cv=CV, features=features,
                          labels=labels, holdout=len(spot) - CUT,
                          to_weights=lambda predicted: np.clip(predicted, -0.2, 0.2))


def test_a_rolling_walk_forward_s_final_fit_reads_and_trades_its_last_train_bars():
    study = validated(cv=probatum.cv.WalkForward(400, 150))
    read  = usable()[:CUT] & np.isfinite(LABELS.ret[:CUT]) & (LABELS.end[:CUT] < CUT)
    read[:CUT - 400] = False
    x, _, _ = Leading.fitted[0]

    np.testing.assert_array_equal(
        x, features(DATA)[:CUT].reshape(CUT * 3, 2)[np.flatnonzero(read.reshape(-1))])

    # as a searched policy's first search's winner trades them
    seen = np.where(usable()[:CUT], features(DATA)[:CUT, :, 0], np.nan)
    best = probatum.backtest(weights(seen)[CUT - 400:], _before(DATA, CUT)[CUT - 400:],
                             VENUE, stop=SIZE, take_profit=SIZE)

    np.testing.assert_array_equal(study.best.returns, best.returns)
    assert study.best_score == best.stats["sharpe"]


def test_a_nan_or_an_infinite_feature_leaves_its_row_out_unless_it_is_filled():
    def holed(data):
        given = features(data)
        given[500:509, 1, 0] = np.nan
        given[509:510, 1, 0] = np.inf
        return given

    # the holes fall in the third of four groups, which three of the six splits train
    # on, beside the features' warm-up, which no fill fills; the check compares an
    # infinity with nothing a float warns of
    plain = np.asarray(validated().fits["dropped"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        left = validated(features=holed).fits

    filled = validated(features=holed, fill=0.0).fits
    holes  = np.asarray(left["dropped"]) - plain

    assert sorted(holes.tolist()) == [0, 0, 0] + [holes.max()] * 3 and holes.max() > 0
    assert filled["dropped"].tolist() == plain.tolist()
    assert (np.asarray(filled["rows"]) - np.asarray(left["rows"])).tolist() == (
        holes.tolist())


def test_the_labels_end_is_the_splitter_s_and_its_embargo_the_longest_label_held():
    study = validated()
    ends  = LABELS.end[:CUT]
    held  = (ends != np.iinfo(np.int64).min) & (ends < CUT)
    t     = np.arange(CUT)[:, None]

    assert study.provenance["cv"]["embargo"] == int((ends - t - 1)[held].max())

    split = probatum.cv.CombinatorialPurged(4, 2, end=ends)
    train = [len(rows) for rows, _ in split.split(np.zeros(CUT * 3))]

    assert study.fits["train"].tolist() == train


def test_example_30_validates_a_ridge_on_a_universe_end_to_end():
    sklearn  = pytest.importorskip("sklearn.linear_model")
    universe = walked(2_400, 4, seed=3)

    def features(data):                                          # (T, N, 3)
        return np.stack([
            probatum.ta.roc(data.closes, 24),
            probatum.ta.rsi(data.closes, 14),
            probatum.ta.zscore(data.volumes, 168),
        ], axis=-1)

    labels = probatum.labels(universe, VENUE, horizon=24, notional=10_000.0,
                             stop=0.03, take_profit=0.03)

    study = probatum.validate(
        sklearn.Ridge(alpha=1.0), universe, VENUE,
        cv=probatum.cv.CombinatorialPurged(groups=8, test_groups=2, embargo=24),
        features=features, labels=labels,
        to_weights=lambda predicted: np.clip(predicted * 10.0, 0.0, 0.2),
        holdout=0.2,
    )
    found = study.summary()
    shown = found.to_pandas()

    assert found.lines[0].startswith("policy Ridge(); venue built by hand; data 4 "
                                     "instruments 1h; objective sharpe")
    assert list(shown.index) == ["paths", "score", "windows", "windows_gained", "decay",
                                 "decay_error", "rank_ic", "rank_ic_t", "costs",
                                 "price_pnl", "holdout", "buy_and_hold"]
    assert shown.loc["paths", "value"] == 7 and "; fits 28; " in found.lines[0]
    assert study.forecast is not None and len(study.paths) == 7


REFUSED = [
    (dict(space={"alpha": probatum.Float(0.1, 1.0)}), ProbatumValueError,
     "space: {'alpha': Float(0.1, 1.0)} beside a model, which is fitted rather than "
     "searched; leave it out"),
    (dict(search={"trials": 3}), ProbatumValueError,
     "search: {'trials': 3} beside a model, which is fitted rather than searched; leave "
     "it out"),
    (dict(seed=7), ProbatumValueError,
     "seed: 7 beside a model, which validate fits drawing nothing; give the model a "
     "seed of its own"),
    (dict(features=None), ProbatumValueError,
     "features: none beside a model; give its (T, N, F) features, or a function of the "
     "Data that returns them"),
    (dict(to_weights=None), ProbatumValueError,
     "to_weights: none beside a model; give a function of its predictions that returns "
     "weights"),
    (dict(labels=LABELS.ret), ProbatumTypeError,
     "labels: an object of type ndarray; give the Labels probatum.labels made of the "
     "data"),
    (dict(labels=probatum.labels(walked(seed=12), VENUE, 8, 10_000.0)), DataError,
     "labels: of bars fingerprinted"),
    (dict(to_weights=0.1), ProbatumTypeError,
     "to_weights: an object of type float; give a function of its predictions that "
     "returns weights"),
    (dict(cv=probatum.cv.PurgedKFold(4, end=LABELS.end[:960])), ProbatumValueError,
     "cv: PurgedKFold(4, end of shape (960, 3)), built with end, which validate passes "
     "from the labels; build it without end"),
    (dict(model=type("Bare", (), {"fit": lambda self, x, y: self,
                                  "predict": lambda self, x: x[:, 0]})()), PolicyError,
     "policy: an object of type Bare, a model with no get_params, which validate cannot "
     "build afresh for each fit; give a function of no arguments that builds it"),
    (dict(model=lambda alpha: Leading()), PolicyError,
     "policy: <lambda>, which needs alpha; give a model, or a function of no arguments "
     "that builds one"),
    (dict(model=lambda: 3.0), PolicyError,
     "policy: built an object of type float, which has no fit and predict; build a "
     "model"),
]


@pytest.mark.parametrize("given, kind, said", REFUSED)
def test_a_model_is_refused_what_it_cannot_fit_with_the_fix(given, kind, said):
    with pytest.raises(kind) as raised:
        validated(**given)

    assert type(raised.value) is kind and str(raised.value).startswith(said)


@pytest.mark.parametrize("given, said", [
    (dict(model=type("Lost", (Leading,), {"predict": lambda self, x: x[:, 0] * math.nan})()),
     "policy: predicted nan for a row; predict a finite number a row"),
    (dict(to_weights=lambda predicted: predicted[:, :1]),
     "to_weights: returned shape (964, 1) for predictions of shape (964, 3); return "
     "weights of the shape it was given"),
])
def test_a_model_or_weights_the_final_fit_cannot_trade_are_refused_before_a_path(given,
                                                                                  said):
    with pytest.raises(PolicyError) as raised:
        validated(**given)

    assert str(raised.value).startswith(said)


class Unsure(Counted):
    """A model that predicts NaN where its fit read fewer than 2,000 rows, as each
    split's does and the final fit's does not."""


    def predict(self, x):
        return x[:, 0] * (math.nan if self.rows < 2_000 else 1.0)


def test_a_window_its_fit_cannot_trade_fails_its_path():
    with pytest.raises(ProbatumRuntimeError) as raised:
        validated(Unsure())

    assert "all 3 paths failed" in str(raised.value)
    assert str(raised.value.__cause__) == ("policy: predicted nan for a row; predict a "
                                           "finite number a row")
