"""``validate``: a search over a Strategy's or a function's parameters on the training
bars of each split a splitter gives, or a model fitted to its labels on them, each
split's winner traded on its test span and the test spans joined into out-of-sample
paths, each one run of the engine across explicit seams, beside a search or a fit over
every bar before the holdout, whose winner the holdout runs (ADRs 0034, 0035;
Specification, Validation)."""

import collections.abc
import copy
import inspect
import math
import multiprocessing
import shutil

import numpy as np

from . import _search, venues
from ._backtest import (_agent, _annualised, _arrayed, _decided, _kind, _rate,
                        _repeatable, _seen, backtest)
from ._causality import WIDE, _check, _read
from ._data import (Data, _before, _chosen, _computed, _frozen, _when,
                    features as _features)
from ._errors import (DataError, LookaheadError, MissingDependency, PolicyError,
                      ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError)
from ._labels import Labels
from ._probatum import Table as _Held, stats_of
from ._result import COSTS, COUNTS, STATS, Table, _environment, _version
from ._search import (GUARD, KEEPS, LOGGER, Study, _asked, _controls, _counted, _cut,
                      _draws, _floats, _ints, _named, _param, _pieces, _pooled, _Runner,
                      _shown, _spaced, _texts, _told, _Trial, _winner)
from ._strategy import Strategy, _warmup
from ._streams import stream
from ._venue import _market, _run
from .cv import CombinatorialPurged, PurgedKFold, WalkForward, _built, _rows, _Splitter
from .stats import Forecast, _forecast

SEAMS = ("cancel", "flatten")
# the keywords of search that search= gives every search of a validation, and those
# validate takes as its own
SEARCHED = ("objective", "direction", "trials", "sampler", "min_trades", "n_jobs")
OWNED    = ("policy", "space", "data", "venue", "holdout", "keep", "seed", "risk_free")
# the statistics the order of a fit's joined pieces changes (Specification, Fits)
ORDERED = ("max_drawdown_pct", "calmar")
# the columns the fits table has of its own, which no parameter is named
FITTED = ("number", "test", "train", "purged", "embargoed", "pieces", "fitted",
          "trials", "complete", "seed", "state", "error")
# what a model's causality check compares, by the name its refusal opens with: one of
# them, all of them and the fix (Specification, A model)
CHECKED = dict(policy=("prediction", "predictions", "predict each row from its own "
                                                     "features"),
               to_weights=("weight", "weights", "compute each row's weight from the "
                                                "predictions up to it"))
REPEAT = "seed every random draw it makes, so the check can compare its runs"


def validate(policy, data, venue, cv, space=None, search=None, features=None,
             labels=None, to_weights=None, holdout=None, fill=None, keep="all",
             seam="cancel", seed=None, risk_free=0.0):
    """Validate ``policy``, a Strategy subclass or a function ``policy(data, **params)``
    searched over ``space``, on ``data``'s bars before a holdout at ``venue``, a Venue
    or a preset's name, across the splits of ``cv``, a `probatum.cv` splitter, and
    return the `Study` of its paths (ADRs 0034, 0035; Specification, Validation).

    ``search`` is a dict of `probatum.search`'s ``objective``, ``direction``,
    ``trials``, ``sampler``, ``min_trades`` and ``n_jobs``, which every search of the
    validation runs by. One searches first over the bars the splitter gives the
    holdout as its test span, and its winner is the Study's, which
    ``evaluate_holdout`` runs. Then each split's search runs on its training bars,
    each run of them a piece of its own that holds no test bar, and its winner trades
    the split's test span, a function called with each piece's Data. A path is one run
    over the bars before the holdout of its windows' winners in turn, the account
    carried across each seam, where every order not yet filled is cancelled, the
    targets are forgotten and the incoming window's ``on_inherit(bar, account)`` is
    called before its first decision; a path of a function's weights is one array run
    of each window's rows of its winner's weights, called with those bars, whose seams
    place again the legs of each inherited bracket still resting. With
    ``seam="flatten"`` the outgoing window's last close closes every position instead
    of deciding. ``holdout`` holds out the last rows as search's does, ``keep="all"``
    keeps every path's returns and ``"none"`` none, ``seed`` seeds each search from a
    stream of its own and is drawn from the operating system's entropy and recorded
    where it is None, and ``risk_free`` is the annual rate the statistics are computed
    against.

    A model, which has ``fit`` and ``predict``, or a function of no arguments that
    builds one in its place, is fitted rather than searched: before every fit a fresh
    estimator is made, scikit-learn's clone of the model or a call of the function, and
    it fits ``labels``, a `Labels` of the data, on ``features``, a ``(T, N, F)`` array
    or a function of a Data that returns one, refused where cutting away the bars after
    a row changes a function's values up to it, a NaN past a feature's warm-up filled
    with ``fill`` or leaving its row out. Each fit reads its split's training rows
    that are available, have finite features and a return and are left by the purge
    and the embargo, the labels' end passed to ``cv``, with their uniqueness as
    ``sample_weight`` where ``fit`` takes it; each window predicts on its rows, and
    ``to_weights`` of those predictions gives its weights, traded with the labels'
    barriers. A final fit on every in-sample row whose label ends before the holdout,
    with a rolling WalkForward on its last ``train`` bars alone, traded on those bars,
    is the Study's ``best``, and the paths' predictions are scored against the labels
    as its ``forecast``."""
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        raise ProbatumRuntimeError(f"validate: called by a worker of a search as it "
                                   f"starts, as spawn runs the script's top level "
                                   f"again in each; {GUARD}")

    kind = _validated(policy, labels)
    if kind == "model":
        return _modelled(policy, data, venue, cv, space, search, features, labels,
                         to_weights, holdout, fill, keep, seam, seed, risk_free)

    noun = "a Strategy" if kind == "class" else "a function"
    for name, given in (("features", features), ("labels", labels),
                        ("to_weights", to_weights), ("fill", fill)):
        if given is not None:
            raise ProbatumValueError(f"{name}: {_kind(given)} beside {noun}, which "
                                     f"reads its own Data; leave it out")

    if space is None:
        raise ProbatumValueError(f"space: none, and a fixed parameter set fits nothing, "
                                 f"so {noun} is validated by a search; give each "
                                 f"parameter a range, such as {{\"fast\": "
                                 f"probatum.Int(5, 40)}}")

    space = _spaced(space, policy, kind)
    for name in space:
        if name in FITTED:
            raise ProbatumValueError(f"space: {name}, a column the fits table has of "
                                     f"its own; name the parameter otherwise")

    if not isinstance(cv, _Splitter):
        raise ProbatumTypeError(f"cv: {cv!r}; give a probatum.cv splitter, such as "
                                f"probatum.cv.PurgedKFold(5)")

    if cv.end is not None:
        raise ProbatumValueError(f"cv: {cv!r}, built with the labels' end, which "
                                 f"{noun} has none of; build it without end")

    controls = _searched(search)
    objective, direction = controls["objective"], controls["direction"]
    sampler = controls["sampler"]
    trials, min_trades, n_jobs, resolved = _controls(**controls)

    if isinstance(cv, (PurgedKFold, CombinatorialPurged)) and objective in ORDERED:
        raise ProbatumValueError(f"objective: {objective!r} with {type(cv).__name__}, "
                                 f"whose fits join pieces in an order it reads; use "
                                 f"one the order does not change, such as sharpe")

    _chosen("keep", keep, KEEPS, "use \"all\" or \"none\"")
    _chosen("seam", seam, SEAMS, "use \"cancel\" or \"flatten\"")
    _rate(risk_free)
    if seed is not None:
        _counted("seed", seed, 0)

    seed = np.random.SeedSequence().entropy if seed is None else int(seed)
    data = data if isinstance(data, Data) else Data(data)
    if not len(data) or not data.available.any():
        raise DataError("data: no bars to validate; give a Data that has some")

    held    = _repeatable(venue, "validate", "validation")
    cut     = _cut(holdout, data, "validate")
    inside  = _before(data, cut)
    run     = _run(held, inside)
    rate    = _annualised(inside)
    bars    = len(inside)
    printed = np.flatnonzero(inside.available.any(axis=1))
    embargo = cv._embargo(None, None, bars)
    window  = venues._venue_settings(held._preset)["window"]
    # the instruments that printed before the Data's first row, whose bars there the
    # in-sample run's windows still read
    earlier = inside._root.available[:inside._rows[0]].any(axis=0)
    fits    = _laid(cv, bars, inside.available, earlier, embargo, window)

    # the rows a run decides at, every printed one but the last
    decisions = printed[:-1]
    paths     = [_traded(segments, decisions, cv) for segments in cv._paths(bars)]
    if seam == "flatten":
        _flattened(paths, cv)

    rolling = isinstance(cv, WalkForward) and not cv.anchored
    if rolling:
        count = int(inside.available[cut - cv.train:].any(axis=1).sum())
        if count < 2:
            raise ProbatumValueError(f"cv: {cv!r}, whose first search runs on the last "
                                     f"{cv.train:,} bars before the holdout, {count:,} "
                                     f"of them printed; give a train that holds 2")

    named = _named(policy)
    LOGGER.info("validate: %s fits and %s paths of %s by %r on %s rows, %s held out",
                len(fits), len(paths), named, cv, f"{cut:,}", f"{len(data) - cut:,}")

    final_seed = _seeded(seed, "validate final")
    final      = _search.search(
        policy, space, data[cut - cv.train:] if rolling else data, held,
        objective=objective, direction=direction, trials=trials, sampler=sampler,
        min_trades=min_trades, holdout=len(data) - cut, keep="none", n_jobs=n_jobs,
        seed=final_seed, risk_free=risk_free)

    # a first search the workers' breaking cut short has no winner to judge, as a fit
    # it cuts short has none
    if final._broken:
        raise ProbatumRuntimeError(f"n_jobs: the workers broke in the first search, "
                                   f"failing {len(final._broken):,} of its trials; "
                                   f"their output says why")

    # what every trial decides, as the first search's winner decides on these bars
    made = ("strategy" if kind == "class"
            or isinstance(policy(inside, **final.best_params), Strategy) else "weights")
    settings = dict(kind=kind, made=made, objective=objective, min_trades=min_trades,
                    rate=rate, risk_free=float(risk_free), keep="none",
                    fits=[fit.specs for fit in fits])
    shared = _Shared(policy, held, inside, settings, n_jobs)

    try:
        for fit in fits:
            fit.seed = _seeded(seed, f"validate fit {fit.number}")
            runner   = _Runner(policy, kind, inside, run, rate, float(risk_free),
                               objective, min_trades,
                               pieces=_pieces(inside, held, fit.specs))
            # what the first search's winner decides, which every fit's trials decide
            runner.made = made
            shared.fit  = fit.number

            # every winner is checked once the fits are done (Specification, Fits)
            fit.made, _, pool = _asked(runner, _draws(sampler, space, trials, fit.seed,
                                                      direction),
                                       sampler, "none", resolved, {}, shared, None)
            if pool is not None and pool.broken is not None:
                raise ProbatumRuntimeError(f"n_jobs: the workers broke in fit "
                                           f"{fit.number} after {pool.returned:,} "
                                           f"trials came back ({_told(pool.broken)}); "
                                           f"their output says why") from pool.broken

            try:
                fit.winner = _winner(fit.made, direction, min_trades, pieces=True)
            except (ProbatumValueError, ProbatumRuntimeError) as error:
                fit.error, fit.raised = _told(error), error

            LOGGER.info("validate: fit %s %s", fit.number, fit.state)
    finally:
        shared.close()

    # what the first search checked, which leaves out a winner that decided nothing
    # on its bars
    checked = list(final.provenance["checked"])
    # and every winner a check ran on, so none runs twice
    seen = list(checked)
    for fit in fits:
        if fit.winner is not None and fit.winner.params not in seen:
            if _check(policy, kind, fit.winner.params, inside, held, run):
                LOGGER.info("validate: fit %s's winner's decisions are causal",
                            fit.number)
                checked.append(fit.winner.params)
            seen.append(fit.winner.params)

    walks, raised = [], None
    for number, windows in enumerate(paths):
        walk = _Walk(number, windows, printed)
        lost = [fits[split] for _, _, split in windows if fits[split].winner is None]
        if lost:
            walk.error = f"fit {lost[0].number} failed ({lost[0].error})"
            raised     = lost[0].raised
        else:
            try:
                walk.ran(policy, kind, made, fits, seam, inside, held, rate,
                         float(risk_free), objective)
            except Exception as error:
                walk.error, raised = _told(error), error

        walks.append(walk)
        LOGGER.info("validate: path %s %s", number, walk.state)

    # each path of Strategies that ran cut at its seams, so what on_inherit decides
    # there is checked as next's decisions are (Specification, Fits)
    for walk in walks:
        if made == "strategy" and walk.state == "complete" and len(walk.windows) > 1:
            composed = dict(policy=policy, kind=kind,
                            windows=_composed(walk.windows, fits), seam=seam)
            if _check(_Path, "class", composed, inside, held, run,
                      cuts=[first for first, _, _ in walk.windows[1:]]):
                LOGGER.info("validate: path %s's seams are causal", walk.number)

    if all(walk.state == "failed" for walk in walks):
        last   = walks[-1]
        failed = ("its one path failed," if len(walks) == 1 else
                  f"all {len(walks):,} paths failed, the last, path {last.number},")
        raise ProbatumRuntimeError(f"policy: {failed} with {last.error}; the error it "
                                   f"was raised from shows where") from raised

    returns = None
    if keep == "all":
        returns = np.full((len(walks), len(printed)), np.nan)
        for walk in walks:
            if walk.returns is not None:
                returns[walk.number] = walk.returns
        returns.flags.writeable = False

    provenance = dict(
        kind="paths", policy=named,
        space={name: repr(given) for name, given in space.items()},
        cv=dict(given=repr(cv), bars=bars, splits=len(fits), paths=len(walks),
                embargo=embargo),
        seam=seam, objective=objective, direction=direction,
        sampler=final.provenance["sampler"], seed=seed, trials=trials,
        min_trades=min_trades,
        final=dict(seed=final_seed, ran=final.provenance["ran"],
                   winner=final._winner.number, params=final.best_params,
                   score=final.best_score, bars=final._ran.cut),
        fits=[fit.seed for fit in fits], checked=checked,
        holdout=dict(given=_shown(holdout), rows=len(data) - cut,
                     first=None if cut == len(data) else _when(data.time[cut])),
        keep=keep, n_jobs=dict(given=n_jobs, resolved=resolved),
        risk_free=float(risk_free), periods_per_year=rate,
        fingerprint=inside.fingerprint, venue=repr(held), venue_name=held.name,
        data=data.provenance, version=_version(), **_environment())

    LOGGER.info("validate: %s of %s paths complete",
                sum(walk.state == "complete" for walk in walks), len(walks))
    tables = dict(paths=_paths(walks), fits=_fits(fits, space),
                  windows=_windows(walks, inside.time[printed]))
    return Study(None, returns, final._winner, final._ran, provenance, tables)


def _validated(policy, labels):
    """Whether ``policy`` is a Strategy subclass each fit builds, a function of the
    data, or a model fitted to ``labels``: one with ``fit``, or a function or a class
    that builds one given beside its labels; anything else is refused as no policy
    validate takes."""
    if isinstance(policy, type):
        if issubclass(policy, Strategy):
            return "class"
        if labels is not None:
            return "model"
        named = f"the class {policy.__name__}"
    elif isinstance(policy, Strategy):
        name = type(policy).__name__
        raise PolicyError(f"policy: an instance of {name}; pass the class, {name}, "
                          f"which each fit builds with its parameters")
    elif hasattr(policy, "fit") or labels is not None and callable(policy):
        return "model"
    else:
        named = _kind(policy)

    if hasattr(policy, "fit"):
        raise PolicyError(f"policy: {named}, a model's class; give a model, or the "
                          f"class beside its features, labels and to_weights, which "
                          f"builds one for each fit")

    if callable(policy) and not isinstance(policy, type) and not _agent(policy):
        return "function"

    raise PolicyError(f"policy: {named}, which is no Strategy; pass a Strategy "
                      f"subclass, or a function of the data that returns a policy")


def _searched(given):
    """``search=`` as each keyword of search it gives, search's default where it gives
    none, refusing what is no dict of them, a keyword validate takes as its own and
    one a validation's searches do not take."""
    if given is None:
        given = {}

    if not isinstance(given, collections.abc.Mapping):
        raise ProbatumTypeError(f"search: {given!r}; give a dict of probatum.search's "
                                f"keywords, such as {{\"trials\": 200}}")

    for name in given:
        if name in OWNED:
            raise ProbatumValueError(f"search: {name}, which validate takes as its "
                                     f"own; pass {name}= to validate")
        if name == "storage":
            raise ProbatumValueError("search: storage, which a validation's searches "
                                     "do not save to; leave it out")
        if name not in SEARCHED:
            raise ProbatumTypeError(f"search: {name!r}, which probatum.search does not "
                                    f"take; give {', '.join(SEARCHED)}")

    defaults = inspect.signature(_search.search).parameters
    return {name: given.get(name, defaults[name].default) for name in SEARCHED}


def _seeded(seed, name):
    """The seed of the validation's search ``name``, drawn from the stream of that name
    of ``seed``."""
    return int(stream(seed, name).integers(0, 2**63))


class _Fit:
    """One split of a validation of a Strategy: its ``number``, its ``tests`` spans,
    its training bars, those the purge and the embargo took and its pieces as
    `_pieces` reads them, and once its search has run, its seed, its trials and its
    winner, or why it failed."""

    __slots__ = ("number", "tests", "train", "purged", "embargoed", "specs", "seed",
                 "made", "winner", "error", "raised")


    def __init__(self, number, tests, train, purged, embargoed, specs):
        self.number, self.tests, self.train = number, tests, train
        self.purged, self.embargoed, self.specs = purged, embargoed, specs

        self.seed = self.winner = self.error = self.raised = None
        self.made = []


    @property
    def state(self):
        return "failed" if self.winner is None else "complete"


def _laid(cv, bars, available, earlier, embargo, window):
    """The fits of ``cv``'s splits of ``bars`` bars for a Strategy, ``available`` saying
    which instruments printed each row and ``earlier`` which printed before the first:
    a span from ``a`` to ``b`` reaches the first printed row after its last one, or
    ``b`` where it prints none or none follows, and a split trains on its bars outside
    its spans but those after each span up to its reach and the ``embargo`` after it,
    each run of them a piece, one after a span on a Data of its own from the bar after
    the latest reach before it, opening past the embargo and at the earliest where each
    instrument that printed before that Data has printed ``window + 1`` bars in it, or
    its last bar there where it prints fewer, ``window`` the venue's fill window, whose
    first return is measured from the bar before it, so each fill there reads the
    window the in-sample bars give it. A piece that holds no two printed rows from
    where it opens is left out, and a split left with none is refused (Specification,
    Fits)."""
    printed = np.flatnonzero(available.any(axis=1))
    fits    = []

    for number, (tests, trains) in enumerate(cv._layout(bars)):
        kept = np.zeros(bars, dtype=bool)
        for lo, hi in trains:
            kept[lo:hi] = True

        purged = embargoed = 0
        reaches = []
        for a, b in tests:
            after   = int(np.searchsorted(printed, b, side="right"))
            spanned = after > 0 and printed[after - 1] >= a
            reach   = int(printed[after]) if spanned and after < len(printed) else b
            purged += int(kept[b + 1:reach + 1].sum())
            embargoed += int(kept[reach + 1:reach + embargo + 1].sum())
            kept[b + 1:reach + embargo + 1] = False
            reaches.append((b, reach))

        specs = []
        for lo, hi in _runs(kept):
            before = [reach for b, reach in reaches if b < lo]
            if not before:
                opens, spec = lo, (0, lo, hi, None)
            else:
                # a span that prints no row reaches its own last bar, which can fall
                # before an earlier span's reach
                first = max(before) + 1
                own   = available[first:hi]

                # an instrument that printed nothing before the Data has the same window
                # in it as in the in-sample bars
                waits = []
                for k in np.flatnonzero(earlier | available[:first].any(axis=0)):
                    rows = np.flatnonzero(own[:, k])
                    if len(rows):
                        waits.append(first + int(rows[min(window, len(rows) - 1)]))

                opens = max([lo] + waits)
                spec  = (first, first, hi, opens - first)

            if np.searchsorted(printed, hi) - np.searchsorted(printed, opens) >= 2:
                specs.append(spec)

        if not specs:
            raise ProbatumValueError(f"cv: {cv._named(number, tests, bars)}, whose "
                                     f"training bars hold no piece two printed rows "
                                     f"long once its account opens; give it more bars "
                                     f"to train on")

        fits.append(_Fit(number, tests, int(kept.sum()), purged, embargoed, specs))

    return fits


def _runs(mask):
    """The runs of ``mask``'s true bars, as half-open ``(lo, hi)``."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges  = np.flatnonzero(np.diff(padded))
    return [(int(lo), int(hi)) for lo, hi in zip(edges[::2], edges[1::2])]


def _traded(segments, decisions, cv):
    """A path's windows from its ``segments`` of ``cv``'s, each ``(a, b, split)`` in
    bars: each run of segments one split's winner trades, as the first and last of
    ``decisions``, the rows a run decides at, that it holds, and the split, a segment
    holding none making no window."""
    out = []
    for a, b, split in segments:
        lo, hi = np.searchsorted(decisions, (a, b + 1))
        if lo == hi:
            continue

        first, last = int(decisions[lo]), int(decisions[hi - 1])
        if out and out[-1][2] == split:
            out[-1] = (out[-1][0], last, split)
        else:
            out.append((first, last, split))

    if not out:
        # the other splitters' paths cover every bar, and a WalkForward's runs from its
        # train to the last bar whatever its test
        fix = "a shorter train" if isinstance(cv, WalkForward) else "a longer tail"
        raise ProbatumValueError(f"cv: {cv!r}, whose path tests no row a run decides "
                                 f"at, a printed row before the in-sample bars' last; "
                                 f"give {fix}")

    return out


def _composed(windows, fits):
    """Each window's first and last row with its fit's winner's parameters, as a path's
    composite takes them."""
    return [(first, last, fits[split].winner.params) for first, last, split in windows]


def _flattened(paths, cv):
    """Refuse a flattening seam at the end of a window that decides at one row alone,
    which the seam would take from it."""
    more = ("its windows more bars to test" if isinstance(cv, WalkForward)
            else "the groups more bars")

    for number, windows in enumerate(paths):
        for first, last, split in windows[:-1]:
            if first == last:
                raise ProbatumValueError(f"seam: 'flatten', and path {number}'s window "
                                         f"at bar {first:,}, which fit {split} trades, "
                                         f"decides at one row, which the seam takes, "
                                         f"so it never decides; use \"cancel\", or "
                                         f"give {more}")


class _Shared:
    """The workers every fit of a validation runs its trials in, started the first
    time a fit asks for them, the trials naming the fit whose they are."""


    def __init__(self, policy, venue, data, settings, n_jobs):
        self.policy, self.venue, self.data = policy, venue, data
        self.settings, self.n_jobs = settings, n_jobs

        self.pool = self.fit = None


    def __call__(self, workers):
        if self.pool is None:
            self.pool = _pooled(self.policy, self.venue, self.data, self.settings,
                                workers, self.n_jobs)

        self.pool.fit = self.fit
        return self.pool


    def close(self):
        if self.pool is not None:
            try:
                self.pool.close()
            finally:
                shutil.rmtree(self.pool.folder, ignore_errors=True)


class _Path(Strategy):
    """A validation's path as one Strategy: each window's own Strategy, built with its
    fit's winner or, for ``kind`` "function", the one the function returns on the
    Data, deciding at its window's closes from its own warm-up, and a seam where one
    window hands the account to the next (Specification, Paths and seams).
    ``windows`` holds each window's first and last row and its parameters."""


    def __init__(self, policy, kind, windows, seam):
        self.policy, self.kind, self.windows, self.seam = policy, kind, windows, seam


    def init(self, data):
        # what a function returns is a prototype, copied once before it runs (ADR 0024)
        self.parts = [self.policy(**params) if self.kind == "class"
                      else copy.deepcopy(self.policy(data, **params))
                      for _, _, params in self.windows]
        self.warmups = []
        for part in self.parts:
            part.init(data)
            self.warmups.append(_warmup(part))

        self.decided = [0] * len(self.parts)
        self.at      = 0
        self.warmup  = self.windows[0][0]


    def next(self, bar, account):
        row, windows = bar.i, self.windows
        while row > windows[self.at][1]:
            self.at += 1

        at = self.at
        first, last, _ = windows[at]
        part = self.parts[at]
        if row == first and at:
            account._run.seam(False)
            part.on_inherit(bar, account)

        if self.seam == "flatten" and row == last and at < len(windows) - 1:
            account._run.seam(True)
        elif row >= self.warmups[at]:
            part.next(bar, account)
            self.decided[at] += 1


class _Walk:
    """One path of a validation: its ``number``, its ``windows`` and the account rows
    each owns, from the one after its first decision to the one after its last, the
    last window's to the end, and once it has run, the statistics and the costs of its
    rows and of each window's, the rows each window decided on and its returns, or why
    it failed (Specification, Paths and seams)."""


    def __init__(self, number, windows, printed):
        self.number, self.windows = number, windows

        starts     = [int(np.searchsorted(printed, first)) + 1 for first, _, _ in windows]
        self.owned = list(zip(starts, [start - 1 for start in starts[1:]]
                              + [len(printed) - 1]))

        self.stats = self.costs = self.score = self.returns = self.error = None
        self.parts   = [None] * len(windows)
        self.decided = [None] * len(windows)


    @property
    def state(self):
        return "failed" if self.stats is None else "complete"


    def ran(self, policy, kind, made, fits, seam, inside, venue, rate, risk_free,
            objective):
        """The path run once over ``inside``, a Strategy's or, where the policy ``made``
        weights, one array run, and what its records give."""
        if made == "strategy":
            path   = _Path(policy, kind, _composed(self.windows, fits), seam)
            result = _decided(path, inside, venue, "weight", None, None, 0.0, risk_free,
                              "full")
            decided = list(result.policy.decided)
        else:
            result, decided = _weighed(policy, self.windows, fits, seam, inside, venue,
                                       risk_free)

        self.scored(result, decided, inside, rate, risk_free, objective)


    def scored(self, result, decided, inside, rate, risk_free, objective):
        """What the path's run, ``result``, gives: the statistics and the costs of its
        rows and of each window's, scored by ``objective``, the rows each window
        ``decided`` at and its returns."""
        rows      = len(result.time)
        positions = result.holdings["position"].reshape(rows, len(inside.names))
        holding   = (np.abs(positions) > 0.0).any(axis=1)

        for k, (first, last) in enumerate(self.owned):
            stats, costs = _owned(result, holding, first, last, rate, risk_free)
            self.parts[k] = (stats, costs, float(stats[objective]))

        start = self.owned[0][0]
        self.stats, self.costs = _owned(result, holding, start, rows - 1, rate,
                                        risk_free)
        self.score   = float(self.stats[objective])
        self.decided = decided
        self.returns = np.full(rows, np.nan)
        self.returns[start:] = result.returns[start:]


def _weighed(policy, windows, fits, seam, inside, venue, risk_free):
    """A path of a function's weights run over ``inside`` as one array, and the rows
    each window decided at: each window's rows, from its first decision row to the
    next window's, and their barriers are its fit's winner's, the function called with
    ``inside``, and the rows before the first window hold; a seam opens each window
    but the first, and with ``seam="flatten"`` each window's last decision row but the
    last's closes every position instead. Refused where two windows' winners return
    another sizing or tolerance (Specification, A function; Paths and seams)."""
    def decide(first, end, split):
        return _read(policy(inside, **fits[split].winner.params), inside)

    return _arrayed_path(windows, decide, seam, inside, venue, risk_free)


def _arrayed_path(windows, decide, seam, inside, venue, risk_free):
    """A path of weights run over ``inside`` as one array, and the rows each window
    decided at: ``decide(first, end, split)`` gives each window's rows over ``inside``,
    their barriers, sizing and tolerance, read from its first decision row to before
    ``end``, and the rows before the first window hold; a seam opens each window but
    the first, and with ``seam="flatten"`` each window's last decision row but the
    last's closes every position instead. Refused where two windows give another sizing
    or tolerance (Specification, Paths and seams)."""
    shape    = (len(inside), len(inside.names))
    rows     = np.full(shape, np.nan)
    barriers = {"stop": np.full(shape, np.nan), "take_profit": np.full(shape, np.nan)}
    given, sizing, tolerance, opening = set(), None, None, None
    ends = [first for first, _, _ in windows[1:]] + [shape[0]]

    for (first, _, split), end in zip(windows, ends):
        read, held, own_sizing, own_tolerance = decide(first, end, split)
        if sizing is None:
            sizing, tolerance, opening = own_sizing, own_tolerance, first
        elif (own_sizing, own_tolerance) != (sizing, tolerance):
            raise ProbatumValueError(
                f"policy: the windows from bars {opening:,} and {first:,} return "
                f"sizing {sizing!r} at a tolerance of {tolerance!r} and sizing "
                f"{own_sizing!r} at {own_tolerance!r}; return one sizing and one "
                f"tolerance for every parameter")

        rows[first:end] = read[first:end]
        for name, barrier in held.items():
            if barrier is not None:
                given.add(name)
                barriers[name][first:end] = (barrier if isinstance(barrier, float)
                                             else barrier[first:end])

    printed  = inside.available.any(axis=1)
    kept     = np.flatnonzero(printed)
    seams    = [int(np.searchsorted(kept, first)) for first, _, _ in windows[1:]]
    flattens = ([int(np.searchsorted(kept, last)) for _, last, _ in windows[:-1]]
                if seam == "flatten" else [])
    seen, _  = _seen(rows, printed)
    decided  = []

    for k, (first, last, _) in enumerate(windows):
        lo, hi = np.searchsorted(kept, (first, last + 1))
        hi    -= seam == "flatten" and k < len(windows) - 1
        decided.append(int((~np.isnan(seen[lo:hi])).any(axis=1).sum()))

    result = _arrayed(rows, {name: barriers[name] if name in given else None
                             for name in barriers}, sizing, tolerance, inside, venue,
                      risk_free, "full", seams=seams, flattens=flattens)
    return result, decided


def _modelled(model, data, venue, cv, space, search, features, labels, to_weights,
              holdout, fill, keep, seam, seed, risk_free):
    """``validate`` of ``model`` fitted to ``labels`` on ``features``: a final fit on
    every in-sample row whose label ends before the holdout, with a rolling WalkForward
    on its last ``train`` bars alone, the Study's winner, and each split's fit on its
    training rows, its predictions traded on its test span through ``to_weights`` with
    the labels' barriers, every path one array run and its predictions scored against
    the labels (Specification, A model)."""
    for name, given, fix in (
            ("space", space, "is fitted rather than searched; leave it out"),
            ("search", search, "is fitted rather than searched; leave it out"),
            ("seed", seed, "validate fits drawing nothing; give the model a seed of its "
                           "own")):
        if given is not None:
            raise ProbatumValueError(f"{name}: {given!r} beside a model, which {fix}")

    wanted = dict(features="its (T, N, F) features, or a function of the Data that "
                           "returns them",
                  labels="the Labels probatum.labels made of the data",
                  to_weights="a function of its predictions that returns weights")

    for name, given in (("features", features), ("labels", labels),
                        ("to_weights", to_weights)):
        if given is None:
            raise ProbatumValueError(f"{name}: none beside a model; give {wanted[name]}")

    if not isinstance(labels, Labels):
        raise ProbatumTypeError(f"labels: {_kind(labels)}; give {wanted['labels']}")

    if not callable(to_weights):
        raise ProbatumTypeError(f"to_weights: {_kind(to_weights)}; give "
                                f"{wanted['to_weights']}")

    fresh = _fresh(model)

    if not isinstance(cv, _Splitter):
        raise ProbatumTypeError(f"cv: {cv!r}; give a probatum.cv splitter, such as "
                                f"probatum.cv.PurgedKFold(5)")

    if cv.end is not None:
        raise ProbatumValueError(f"cv: {cv!r}, built with end, which validate passes "
                                 f"from the labels; build it without end")

    _chosen("keep", keep, KEEPS, "use \"all\" or \"none\"")
    _chosen("seam", seam, SEAMS, "use \"cancel\" or \"flatten\"")
    _rate(risk_free)

    data = data if isinstance(data, Data) else Data(data)
    if not len(data) or not data.available.any():
        raise DataError("data: no bars to validate; give a Data that has some")

    _labelled(labels, data)

    held    = _repeatable(venue, "validate", "validation")
    cut     = _cut(holdout, data, "validate")
    inside  = _before(data, cut)
    rate    = _annualised(inside)
    width   = len(data.names)
    printed = np.flatnonzero(inside.available.any(axis=1))
    grid    = _featured(data, features, fill, cut)
    origin  = held.name or "venue"
    spot    = np.array([_market(instrument, name, held, origin) == "spot"
                        for instrument, name in zip(data.instruments, data.names)])

    # the rows a fit or a window can read, available with every feature finite
    usable = data.available & np.isfinite(grid).all(axis=-1)
    ret    = labels.ret[:cut]
    split  = _built(type(cv), cv._args(), cv._keywords(),
                    np.ascontiguousarray(labels.end[:cut]))

    # the splitter's refusals, before any fit
    list(split.split(np.zeros((cut * width, 0))))
    ends, labelled = split._ends(cut, width)
    embargo  = split._embargo(ends, labelled, cut)
    returned = np.isfinite(ret)
    fits     = []

    for number, (tests, trains) in enumerate(split._layout(cut)):
        given = np.zeros((cut, width), dtype=bool)
        for lo, hi in trains:
            given[lo:hi] = True

        clean, _ = _rows(tests, trains, ends, labelled, 0)
        train, _ = _rows(tests, trains, ends, labelled, embargo)
        read     = train & usable[:cut] & returned
        if not read.any():
            raise ProbatumValueError(f"cv: {split._named(number, tests, cut)}, whose "
                                     f"training rows hold none available with finite "
                                     f"features and a return; give it more rows to "
                                     f"train on, or fill=")

        fits.append(_Fitting(number, tests, int(train.sum()),
                             int((given & labelled & ~clean).sum()),
                             int((clean & ~train).sum()), read,
                             int((train & data.available[:cut] & returned
                                  & ~usable[:cut]).sum())))

    decisions = printed[:-1]
    paths     = [_traded(segments, decisions, cv) for segments in cv._paths(cut)]
    if seam == "flatten":
        _flattened(paths, cv)

    named = (repr(model) if hasattr(model, "fit") and not isinstance(model, type)
             else _named(model))
    LOGGER.info("validate: %s fits and %s paths of %s by %r on %s rows, %s held out",
                len(fits), len(paths), named, cv, f"{cut:,}", f"{len(data) - cut:,}")

    x, y = grid[:cut].reshape(cut * width, -1), ret.reshape(-1)
    weight  = labels.weight[:cut].reshape(-1)
    final   = usable[:cut] & returned & (labels.end[:cut] < cut)
    rolling = isinstance(cv, WalkForward) and not cv.anchored
    if rolling:
        # as a searched policy's first search reads and trades the last train bars alone
        final[:cut - cv.train] = False

    if not final.any():
        raise ProbatumValueError("labels: no in-sample row available with finite "
                                 "features and a return that ends before the holdout, "
                                 "which the final fit reads; give more bars, or fill=")

    ran = _Fitted(_fit(fresh, x, y, weight, final.reshape(-1)), grid, usable,
                  to_weights, labels, held, float(risk_free), data, cut,
                  inside[cut - cv.train:] if rolling else inside, spot)
    LOGGER.info("validate: the final fit read %s rows", f"{int(final.sum()):,}")
    _causal(ran.model, grid, usable, to_weights, cut, spot, data)

    for fit in fits:
        try:
            fit.model = _fit(fresh, x, y, weight, fit.read.reshape(-1))
        except Exception as error:
            fit.error, fit.raised = _told(error), error
        LOGGER.info("validate: fit %s %s", fit.number, fit.state)

    barriers = {name: _rowed(getattr(labels, name), 0, cut)
                for name in ("stop", "take_profit")}
    walks, raised, foreseen = [], None, []
    for number, windows in enumerate(paths):
        walk, kept = _Walk(number, windows, printed), []
        lost = [fits[at] for _, _, at in windows if fits[at].model is None]
        if lost:
            walk.error = f"fit {lost[0].number} failed ({lost[0].error})"
            raised     = lost[0].raised
        else:
            def decide(first, end, at, kept=kept):
                rows = np.full((cut, width), np.nan)
                rows[first:end] = _foreseen(fits[at].model, grid, usable, first, end,
                                            to_weights, spot, data, kept)
                return rows, barriers, "weight", 0.0

            try:
                result, decided = _arrayed_path(windows, decide, seam, inside, held,
                                                float(risk_free))
                walk.scored(result, decided, inside, rate, float(risk_free), "sharpe")
            except Exception as error:
                walk.error, raised = _told(error), error

        foreseen.append(kept if walk.state == "complete" else None)
        walks.append(walk)
        LOGGER.info("validate: path %s %s", number, walk.state)

    if all(walk.state == "failed" for walk in walks):
        last   = walks[-1]
        failed = ("its one path failed," if len(walks) == 1 else
                  f"all {len(walks):,} paths failed, the last, path {last.number},")
        raise ProbatumRuntimeError(f"policy: {failed} with {last.error}; the error it "
                                   f"was raised from shows where") from raised

    forecasts = [None if kept is None else _scored_rows(kept, ret, labels.end[:cut], 1)
                 for kept in foreseen]
    complete = [kept for kept in foreseen if kept is not None]
    pooled   = _scored_rows([part for kept in complete for part in kept], ret,
                            labels.end[:cut], len(complete))

    returns = None
    if keep == "all":
        returns = np.full((len(walks), len(printed)), np.nan)
        for walk in walks:
            if walk.returns is not None:
                returns[walk.number] = walk.returns
        returns.flags.writeable = False

    best       = ran({}, ran.inside)
    provenance = dict(
        kind="paths", policy=named,
        model=dict(features=_named(features) if callable(features)
                   else f"a {np.shape(features)} array", to_weights=_named(to_weights),
                   fill=fill, final=int(final.sum())),
        labels=dict(horizon=labels.horizon, notional=labels.notional, side=labels.side,
                    **{name: (given if given is None or isinstance(given, float)
                              else f"a {given.shape} array")
                       for name, given in (("stop", labels.stop),
                                           ("take_profit", labels.take_profit))}),
        cv=dict(given=repr(cv), bars=cut, splits=len(fits), paths=len(walks),
                embargo=embargo),
        seam=seam, objective="sharpe", direction="maximize",
        holdout=dict(given=_shown(holdout), rows=len(data) - cut,
                     first=None if cut == len(data) else _when(data.time[cut])),
        keep=keep, risk_free=float(risk_free), periods_per_year=rate,
        fingerprint=inside.fingerprint, venue=repr(held), venue_name=held.name,
        data=data.provenance, version=_version(), **_environment())

    LOGGER.info("validate: %s of %s paths complete",
                sum(walk.state == "complete" for walk in walks), len(walks))
    tables = dict(paths=_paths(walks, forecasts), fits=_model_fits(fits),
                  windows=_windows(walks, inside.time[printed]))
    winner      = _Trial(0, {}, "complete", float(best.stats["sharpe"]), best.stats)
    study       = Study(None, returns, winner, ran, provenance, tables, forecast=pooled)
    study._best = best
    return study


def _fresh(model):
    """How each fit makes its estimator from ``model``: scikit-learn's clone of a model
    with get_params, or a call with no arguments of the function or the class given in
    a model's place, so no fit starts from another's weights (Specification, A
    model)."""
    if hasattr(model, "fit") and not isinstance(model, type):
        if not callable(getattr(model, "get_params", None)):
            raise PolicyError(f"policy: {_kind(model)}, a model with no get_params, "
                              f"which validate cannot build afresh for each fit; give a "
                              f"function of no arguments that builds it")
        try:
            from sklearn.base import clone
        except ImportError:
            raise MissingDependency("policy: a model validate builds afresh for each fit "
                                    "with scikit-learn's clone, which is not installed; "
                                    "pip install scikit-learn, or give a function of no "
                                    "arguments that builds the model") from None
        return lambda: clone(model)

    try:
        parameters = list(inspect.signature(model).parameters.values())
    except (TypeError, ValueError):
        return model

    needed = [parameter.name for parameter in parameters
              if parameter.default is parameter.empty
              and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)]
    if needed:
        raise PolicyError(f"policy: {_named(model)}, which needs {needed[0]}; give a "
                          f"model, or a function of no arguments that builds one")

    return model


def _labelled(labels, data):
    """Refuse ``labels`` unless they are of ``data``: its rows, its instruments and its
    fingerprint."""
    fix = "give the labels of the data validate is given"
    if labels.ret.shape[0] != len(data):
        raise DataError(f"labels: of {labels.ret.shape[0]:,} rows, and the data holds "
                        f"{len(data):,}; {fix}")

    if tuple(labels.names) != tuple(data.names):
        raise DataError(f"labels: of {', '.join(labels.names)}, and the data holds "
                        f"{', '.join(data.names)}; {fix}")

    if labels.fingerprint != data.fingerprint:
        raise DataError(f"labels: of bars fingerprinted {labels.fingerprint}, and the "
                        f"data's are {data.fingerprint}; {fix}")


def _featured(data, given, fill, cut):
    """A model's features over ``data``'s rows, ``(T, N, F)`` and read-only: the array
    given, or what the function given returns on the whole Data, filled where ``fill``
    is given as `probatum._data.features` fills them, each NaN otherwise left to keep
    its row out. A function's are computed again on the whole Data and refused as a
    ProbatumValueError where its ``cut`` in-sample rows differ, and then on the Data cut
    after each of up to 10 rows spread evenly over them, and refused as a LookaheadError
    where a value on a row up to the cut differs from the whole Data's, as `_changed`
    compares them (Specification, A model)."""
    def computed(over):
        if fill is not None:
            return _features(over, given, fill)
        values, _, start = _computed(over, given)
        return values[start:]

    grid = _frozen(np.array(computed(data)))
    if not callable(given):
        return grid

    again   = np.asarray(computed(data))
    changed = _changed(again[:cut], grid[:cut])
    if changed is not None:
        t, i, f = changed
        raise ProbatumValueError(f"features: feature {f} of {data.names[i]} at row {t:,} "
                                 f"({_when(data.time[t])}) is {float(grid[t, i, f])!r} in "
                                 f"one computation on the whole Data and "
                                 f"{float(again[t, i, f])!r} in the next; {REPEAT}")

    count = min(10, cut)
    for last in sorted({cut * (k + 1) // count - 1 for k in range(count)}):
        try:
            part = computed(_before(data, last + 1))
        except Exception as error:
            error.add_note(f"raised by the features on the Data cut after row "
                           f"{last:,}, in validate's causality check")
            raise

        whole   = grid[:last + 1]
        changed = _changed(part, whole)
        if changed is not None:
            t, i, f = changed
            raise LookaheadError(f"features: feature {f} of {data.names[i]} at row {t:,} "
                                 f"({_when(data.time[t])}) is {float(whole[t, i, f])!r} on "
                                 f"the whole Data and {float(part[t, i, f])!r} on the Data "
                                 f"cut after row {last:,}; compute each row's features "
                                 f"from the bars up to it")

    return grid


class _Fitting:
    """One split of a model's validation: its ``number``, its ``tests`` spans, its
    training rows, those the purge and the embargo took, the rows its fit ``read`` as a
    mask and those a NaN feature left out, and once it has run, its model, or why it
    failed."""

    __slots__ = ("number", "tests", "train", "purged", "embargoed", "read", "dropped",
                 "model", "error", "raised")


    def __init__(self, number, tests, train, purged, embargoed, read, dropped):
        self.number, self.tests, self.train = number, tests, train
        self.purged, self.embargoed, self.read = purged, embargoed, read
        self.dropped = dropped

        self.model = self.error = self.raised = None


    @property
    def state(self):
        return "failed" if self.model is None else "complete"


def _fit(fresh, x, y, weight, read):
    """A fresh estimator from ``fresh`` fitted on the rows ``read`` of the long rows
    ``x`` and ``y``, their ``weight`` as its sample_weight where its fit takes that
    keyword."""
    model = fresh()
    if not (callable(getattr(model, "fit", None))
            and callable(getattr(model, "predict", None))):
        raise PolicyError(f"policy: built {_kind(model)}, which has no fit and predict; "
                          f"build a model")

    try:
        taken = "sample_weight" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        taken = False

    model.fit(x[read], y[read], **({"sample_weight": weight[read]} if taken else {}))
    return model


def _changed(part, whole):
    """Where ``part`` first differs from ``whole``, the rows of it ``part`` covers, as
    the causality check compares a float64 array, or None: a NaN or an infinity of one
    meeting another value of the other, or two finite values further apart than 1e-9
    of the largest finite magnitude in ``whole``'s column, and never 1e-12."""
    finite  = np.isfinite(whole)
    largest = np.where(finite, np.abs(whole), 0.0).max(axis=0) if len(whole) else 0.0
    bound   = np.maximum(WIDE * largest, 1e-12)
    both    = finite & np.isfinite(part)
    apart   = np.abs(np.where(both, part, 0.0) - np.where(both, whole, 0.0)) > bound
    same    = np.where(both, ~apart, (part == whole) | (np.isnan(part) & np.isnan(whole)))
    return None if same.all() else tuple(map(int, np.argwhere(~same)[0]))


def _causal(model, grid, usable, to_weights, cut, spot, data):
    """Refuse a final fit ``model`` whose predictions on the ``cut`` in-sample rows, or
    ``to_weights`` of them, differ when computed again, as a ProbatumValueError, or
    change on a row when the rows after a cut are cut away, as a LookaheadError, cut
    after each of up to 10 rows spread evenly over them as a feature function is; the
    predictions on a cut are compared before ``to_weights`` reads the whole's up to it
    (Specification, A model)."""
    whole   = _predicted(model, grid, usable, 0, cut)
    weights = _weighted(to_weights, whole, 0, spot, data)

    _differ("policy", _predicted(model, grid, usable, 0, cut), whole, None, data)
    _differ("to_weights", _weighted(to_weights, whole, 0, spot, data), weights, None,
            data)

    count = min(10, cut)
    for last in sorted({cut * (k + 1) // count - 1 for k in range(count)}):
        part = _on_cut("policy", lambda: _predicted(model, grid, usable, 0, last + 1),
                       last, data)
        _differ("policy", part, whole, last, data)
        given = _on_cut("to_weights",
                        lambda: _weighted(to_weights, whole[:last + 1], 0, spot, data),
                        last, data)
        _differ("to_weights", given, weights, last, data)


def _on_cut(name, compute, last, data):
    """``compute()``, the predictions or the weights ``name`` says, on the rows up to
    ``last``: a refusal of them, where the in-sample rows' were not refused, refused as
    a LookaheadError chained from it, and an error the model's or ``to_weights``' own
    code raises carrying a note naming the cut."""
    _, nouns, fix = CHECKED[name]

    try:
        return compute()
    except PolicyError as error:
        raise LookaheadError(f"{name}: on the rows up to row {last:,} "
                             f"({_when(data.time[last])}) its {nouns} are refused "
                             f"({error}), and on the in-sample rows they are not; "
                             f"{fix}") from error
    except Exception as error:
        error.add_note(f"raised on the in-sample rows up to row {last:,}, in validate's "
                       f"causality check")
        raise


def _differ(name, mine, theirs, last, data):
    """Refuse ``mine``, the predictions or the weights ``name`` says, where they differ
    from the in-sample rows' ``theirs`` on a row they share: computed again on those
    rows where ``last`` is None, as a ProbatumValueError, and on the rows up to
    ``last``, as a LookaheadError."""
    changed = _changed(mine, theirs[:len(mine)])
    if changed is None:
        return

    t, i = changed
    noun, _, fix = CHECKED[name]
    shown = (f"{name}: its {noun} at row {t:,} ({_when(data.time[t])}) for "
             f"{data.names[i]} is {float(theirs[t, i])!r}")

    if last is None:
        raise ProbatumValueError(f"{shown} in one run on the in-sample rows and "
                                 f"{float(mine[t, i])!r} in the next; {REPEAT}")

    raise LookaheadError(f"{shown} on the in-sample rows and {float(mine[t, i])!r} on "
                         f"those up to row {last:,}; {fix}")


def _foreseen(model, grid, usable, first, end, to_weights, spot, data, kept=None):
    """``model``'s weights on the rows ``first`` to before ``end`` of ``grid``, rows of
    ``data``: ``to_weights`` of its predictions there (Specification, A model)."""
    return _weighted(to_weights, _predicted(model, grid, usable, first, end, kept), first,
                     spot, data)


def _predicted(model, grid, usable, first, end, kept=None):
    """``model``'s predictions on the rows ``first`` to before ``end`` of ``grid`` that are
    ``usable``, in time-major order, one finite number a row, placed on a grid of those
    rows, NaN elsewhere; ``kept`` collects each predicted row's bar, instrument and
    prediction."""
    width     = grid.shape[1]
    at        = np.flatnonzero(usable[first:end].reshape(-1))
    predicted = np.full((end - first, width), np.nan)

    if len(at):
        rows   = grid[first:end].reshape(-1, grid.shape[2])[at]
        values = np.asarray(model.predict(rows), dtype=np.float64)
        if values.size != len(at):
            raise PolicyError(f"policy: predicted {values.size:,} values for "
                              f"{len(at):,} rows; predict one number a row")
        values = values.reshape(-1)
        if not np.isfinite(values).all():
            bad = values[~np.isfinite(values)][0]
            raise PolicyError(f"policy: predicted {bad} for a row; predict a finite "
                              f"number a row")

        predicted.reshape(-1)[at] = values
        if kept is not None:
            kept.append((first + at // width, at % width, values))

    return predicted


def _weighted(to_weights, predicted, first, spot, data):
    """``to_weights`` of ``predicted``, the grid of the rows from ``first`` of ``data``,
    refused where it is not of their shape, holds an infinity, or shorts an instrument
    ``spot`` says is spot."""
    weights = np.asarray(to_weights(predicted.copy()), dtype=np.float64)
    if weights.shape != predicted.shape:
        raise PolicyError(f"to_weights: returned shape {weights.shape} for predictions "
                          f"of shape {predicted.shape}; return weights of the shape it "
                          f"was given")

    if np.isinf(weights).any():
        raise PolicyError(f"to_weights: returned {weights[np.isinf(weights)][0]}; "
                          f"return finite weights, or NaN to hold")

    short = (weights < 0.0) & spot
    if short.any():
        t, i = map(int, np.argwhere(short)[0])
        raise PolicyError(f"to_weights: {weights[t, i]} at row {first + t:,} "
                          f"({_when(data.time[first + t])}) for {data.names[i]}, which is "
                          f"spot and cannot be short; return weights from 0 for it")

    return weights


def _rowed(barrier, first, end):
    """A barrier of the labels, None, a number or a ``(T, N)`` array, on the rows
    ``first`` to before ``end``."""
    if barrier is None or isinstance(barrier, float):
        return barrier
    return np.ascontiguousarray(barrier[first:end])


def _scored_rows(kept, ret, ends, paths):
    """The Forecast of the predictions ``kept`` on the rows they were made on whose label
    has a return and ends before the holdout, ``ret`` and ``ends`` the in-sample rows',
    at the lag of their longest label, or why there is none."""
    if not kept:
        return _forecast(np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64), 0,
                         paths)

    bars, columns, values = (np.concatenate(part) for part in zip(*kept))
    returned = ret[bars, columns]
    has      = np.isfinite(returned) & (ends[bars, columns] < len(ret))
    bars     = bars[has]
    lag      = int((ends[bars, columns[has]] - bars).max()) if has.any() else 0

    return _forecast(values[has], returned[has], bars, lag, paths)


class _Fitted:
    """How a model's Study trades its final fit: its predictions on a Data's rows,
    read as a window's, through ``to_weights`` and traded with the labels' barriers
    at the venue as backtest trades them, beside the venue, the risk-free rate, the
    whole Data, the row its holdout starts at and the in-sample Data its ``best``
    trades, with a rolling WalkForward its last ``train`` bars, which a Study reads as
    a search's runner holds them."""

    __slots__ = ("model", "grid", "usable", "to_weights", "labels", "venue",
                 "risk_free", "data", "cut", "inside", "spot")


    def __init__(self, model, grid, usable, to_weights, labels, venue, risk_free, data,
                 cut, inside, spot):
        self.model, self.grid, self.usable = model, grid, usable
        self.to_weights, self.labels, self.venue = to_weights, labels, venue
        self.risk_free, self.data, self.cut, self.inside = risk_free, data, cut, inside
        self.spot = spot


    def __call__(self, params, data):
        """The final fit traded on ``data``, the in-sample Data or the held-out one."""
        first = data._rows[0] - self.data._rows[0]
        end   = first + len(data)

        return backtest(_foreseen(self.model, self.grid, self.usable, first, end,
                                  self.to_weights, self.spot, self.data), data, self.venue,
                        stop=_rowed(self.labels.stop, first, end),
                        take_profit=_rowed(self.labels.take_profit, first, end),
                        risk_free=self.risk_free)


def _model_fits(fits):
    """The fits table of a model's validation, a row a split."""
    parts = [_whole("number", [fit.number for fit in fits]),
             _texts("test", [", ".join(f"{a} to {b}" for a, b in fit.tests)
                             for fit in fits], nullable=False),
             _whole("train", [fit.train for fit in fits]),
             _whole("purged", [fit.purged for fit in fits]),
             _whole("embargoed", [fit.embargoed for fit in fits]),
             _whole("rows", [int(fit.read.sum()) for fit in fits]),
             _whole("dropped", [fit.dropped for fit in fits]),
             _texts("state", [fit.state for fit in fits], nullable=False),
             _texts("error", [fit.error for fit in fits])]
    return Table(_Held(parts))


def _owned(result, holding, first, last, rate, risk_free):
    """The statistics and the costs of ``result``'s account rows ``first`` to ``last``,
    opening at the equity of the row before them, ``holding`` saying which rows ended
    holding a position: the round trips that closed on them, their fills and their
    ledger's costs."""
    trades, ledger = result.trades, result.ledger
    closed = np.ma.filled(trades["close_bar"], -1)
    exits  = ~trades["open"] & (closed >= first) & (closed <= last)
    bars   = result.fills["bar"]
    stats  = stats_of(float(result.equity[first - 1]),
                      np.ascontiguousarray(result.equity[first:last + 1]),
                      np.ascontiguousarray(trades["pnl"][exits]), rate, risk_free,
                      int(holding[first:last + 1].sum()),
                      int(((bars >= first) & (bars <= last)).sum()),
                      math.fsum(ledger["funding"][first:last + 1]))

    return stats, {name: math.fsum(ledger[name][first:last + 1]) for name in COSTS}


def _measured(stats, costs):
    """The columns of the 15 statistics and the seven costs, a row with none of them
    missing each."""
    parts = []
    for key in STATS:
        values = [None if given is None else given[key] for given in stats]
        parts.append(_ints(key, values) if key in COUNTS
                     else (key, "float", True, _floats(values), None, None))

    for key in COSTS:
        values = [None if given is None else given[key] for given in costs]
        parts.append((key, "float", True, _floats(values), None, None))

    return parts


def _paths(walks, forecasts=None):
    """The paths table, a row a path, with a model's ``forecasts`` its forecast's rank
    correlation, its t, its R squared and the rows it read, missing where it has
    none."""
    parts = [_whole("number", [walk.number for walk in walks]),
             _texts("state", [walk.state for walk in walks], nullable=False),
             ("score", "float", True, _floats([walk.score for walk in walks]), None,
              None)]
    parts += _measured([walk.stats for walk in walks], [walk.costs for walk in walks])

    if forecasts is not None:
        held = [given if isinstance(given, Forecast) else None for given in forecasts]
        for name, read in (("rank_ic", "rank_ic"), ("rank_ic_t", "t"),
                           ("r_squared", "r_squared")):
            parts.append((name, "float", True, _floats(
                [None if given is None else getattr(given, read) for given in held]),
                None, None))
        parts.append(_ints("predicted", [None if given is None else given.rows
                                         for given in held]))

    parts.append(_texts("error", [walk.error for walk in walks]))
    return Table(_Held(parts))


def _fits(fits, space):
    """The fits table, a row a split, a failed fit's winner missing."""
    winners = [fit.winner for fit in fits]
    parts   = [_whole("number", [fit.number for fit in fits]),
               _texts("test", [", ".join(f"{a} to {b}" for a, b in fit.tests)
                               for fit in fits], nullable=False),
               _whole("train", [fit.train for fit in fits]),
               _whole("purged", [fit.purged for fit in fits]),
               _whole("embargoed", [fit.embargoed for fit in fits]),
               _whole("pieces", [len(fit.specs) for fit in fits])]
    parts += [_won(name, given, [None if winner is None else winner.params[name]
                                 for winner in winners])
              for name, given in space.items()]

    scores = [None if winner is None else winner.score for winner in winners]
    parts += [("fitted", "float", True, _floats(scores), None, None),
              _whole("trials", [len(fit.made) for fit in fits]),
              _whole("complete", [sum(trial.state == "complete" for trial in fit.made)
                                  for fit in fits]),
              _whole("seed", [fit.seed for fit in fits]),
              _texts("state", [fit.state for fit in fits], nullable=False),
              _texts("error", [fit.error for fit in fits])]
    return Table(_Held(parts))


def _whole(name, values):
    """A column of integers none of which is missing."""
    return name, "int", False, np.array(values, dtype=np.int64), None, None


def _won(name, given, values):
    """The fits table's column of the parameter ``name`` of the range ``given``, of
    the kind the trials table gives it, a failed fit's missing, a bool's among them."""
    part = _param(name, given, values)
    if part[1] != "bool":
        return part

    present = np.array([value is not None for value in values])
    return (name, "bool", True, np.array([bool(value) for value in values]),
            None if present.all() else present, None)


def _windows(walks, times):
    """The windows table, a row a path's window, ``times`` the open of each account
    row."""
    rows  = [(walk, k) for walk in walks for k in range(len(walk.windows))]
    parts = [_whole("path", [walk.number for walk, _ in rows]),
             _whole("fit", [walk.windows[k][2] for walk, k in rows]),
             _whole("first", [walk.owned[k][0] for walk, k in rows]),
             _whole("last", [walk.owned[k][1] for walk, k in rows])]
    for name, end in (("first_time", 0), ("last_time", 1)):
        at = np.array([walk.owned[k][end] for walk, k in rows], dtype=np.int64)
        parts.append((name, "time", False, times[at], None, None))

    parts.append(_ints("decided", [walk.decided[k] for walk, k in rows]))
    ran = [walk.parts[k] for walk, k in rows]
    parts.append(("score", "float", True,
                  _floats([None if part is None else part[2] for part in ran]), None,
                  None))
    parts += _measured([None if part is None else part[0] for part in ran],
                       [None if part is None else part[1] for part in ran])
    return Table(_Held(parts))
