"""``validate``: a search over a Strategy's parameters on the training bars of each
split a splitter gives, each split's winner traded on its test span and the test spans
joined into out-of-sample paths, each one run of the engine across explicit seams,
beside a search over every bar before the holdout, whose winner the holdout runs (ADRs
0034, 0035; Specification, Validation)."""

import collections.abc
import inspect
import math
import multiprocessing
import shutil

import numpy as np

from . import _search, venues
from ._backtest import _agent, _annualised, _decided, _kind, _rate, _repeatable
from ._causality import _check
from ._data import Data, _before, _chosen, _when
from ._errors import (DataError, PolicyError, ProbatumNotImplementedError,
                      ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError)
from ._probatum import Table as _Held, stats_of
from ._result import COSTS, COUNTS, STATS, Table, _environment, _version
from ._search import (GUARD, KEEPS, LOGGER, Study, _asked, _controls, _counted, _cut,
                      _draws, _floats, _ints, _named, _param, _pieces, _pooled, _Runner,
                      _shown, _spaced, _texts, _told, _winner)
from ._strategy import Strategy, _warmup
from ._streams import stream
from ._venue import _run
from .cv import CombinatorialPurged, PurgedKFold, WalkForward, _Splitter

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


def validate(policy, data, venue, cv, space=None, search=None, features=None,
             labels=None, to_weights=None, holdout=None, fill=None, keep="all",
             seam="cancel", seed=None, risk_free=0.0):
    """Validate ``policy``, a Strategy subclass searched over ``space``, on ``data``'s
    bars before a holdout at ``venue``, a Venue or a preset's name, across the splits
    of ``cv``, a `probatum.cv` splitter, and return the `Study` of its paths (ADRs
    0034, 0035; Specification, Validation).

    ``search`` is a dict of `probatum.search`'s ``objective``, ``direction``,
    ``trials``, ``sampler``, ``min_trades`` and ``n_jobs``, which every search of the
    validation runs by. One searches first over the bars the splitter gives the
    holdout as its test span, and its winner is the Study's, which
    ``evaluate_holdout`` runs. Then each split's search runs on its training bars,
    each run of them a piece of its own that holds no test bar, and its winner trades
    the split's test span. A path is one run over the bars before the holdout of its
    windows' winners in turn, the account carried across each seam, where every order
    not yet filled is cancelled, the targets are forgotten and the incoming window's
    ``on_inherit(bar, account)`` is called before its first decision; with
    ``seam="flatten"`` the outgoing window's last close closes every position instead
    of deciding. ``holdout`` holds out the last rows as search's does, ``keep="all"``
    keeps every path's returns and ``"none"`` none, ``seed`` seeds each search from a
    stream of its own and is drawn from the operating system's entropy and recorded
    where it is None, and ``risk_free`` is the annual rate the statistics are computed
    against. ``features``, ``labels``, ``to_weights`` and ``fill`` are a model's,
    which validate does not fit yet."""
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        raise ProbatumRuntimeError(f"validate: called by a worker of a search as it "
                                   f"starts, as spawn runs the script's top level "
                                   f"again in each; {GUARD}")

    _strategy(policy)

    for name, given in (("features", features), ("labels", labels),
                        ("to_weights", to_weights), ("fill", fill)):
        if given is not None:
            raise ProbatumValueError(f"{name}: {_kind(given)} beside a Strategy, which "
                                     f"reads its own Data; leave it out")

    if space is None:
        raise ProbatumValueError("space: none, and a fixed parameter set fits nothing, "
                                 "so a Strategy is validated by a search; give each "
                                 "parameter a range, such as {\"fast\": "
                                 "probatum.Int(5, 40)}")

    space = _spaced(space, policy, "class")
    for name in space:
        if name in FITTED:
            raise ProbatumValueError(f"space: {name}, a column the fits table has of "
                                     f"its own; name the parameter otherwise")

    if not isinstance(cv, _Splitter):
        raise ProbatumTypeError(f"cv: {cv!r}; give a probatum.cv splitter, such as "
                                f"probatum.cv.PurgedKFold(5)")

    if cv.end is not None:
        raise ProbatumValueError(f"cv: {cv!r}, built with the labels' end, which a "
                                 f"Strategy has none of; build it without end")

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

    settings = dict(kind="class", made="strategy", objective=objective,
                    min_trades=min_trades, rate=rate, risk_free=float(risk_free),
                    keep="none", fits=[fit.specs for fit in fits])
    shared = _Shared(policy, held, inside, settings, n_jobs)

    try:
        for fit in fits:
            fit.seed = _seeded(seed, f"validate fit {fit.number}")
            runner   = _Runner(policy, "class", inside, run, rate, float(risk_free),
                               objective, min_trades,
                               pieces=_pieces(inside, held, fit.specs))
            shared.fit = fit.number

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
            if _check(policy, "class", fit.winner.params, inside, held, run):
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
                walk.ran(policy, fits, seam, inside, held, rate, float(risk_free),
                         objective)
            except Exception as error:
                walk.error, raised = _told(error), error

        walks.append(walk)
        LOGGER.info("validate: path %s %s", number, walk.state)

    # each path that ran cut at its seams, so what on_inherit decides there is checked
    # as next's decisions are (Specification, Fits)
    for walk in walks:
        if walk.state == "complete" and len(walk.windows) > 1:
            composed = dict(policy=policy, windows=_composed(walk.windows, fits),
                            seam=seam)
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


def _strategy(policy):
    """Refuse ``policy`` unless it is a Strategy subclass: a model that fits and a
    function, which validate does not run yet, and anything else as no policy it
    takes."""
    if isinstance(policy, type):
        if issubclass(policy, Strategy):
            return
        named = f"the class {policy.__name__}"
    elif isinstance(policy, Strategy):
        name = type(policy).__name__
        raise PolicyError(f"policy: an instance of {name}; pass the class, {name}, "
                          f"which each fit builds with its parameters")
    else:
        named = _kind(policy)

    if hasattr(policy, "fit"):
        raise ProbatumNotImplementedError(f"policy: {named}, a model that fits, which "
                                          f"validate does not fit yet; pass a Strategy "
                                          f"subclass")

    if callable(policy) and not isinstance(policy, type) and not _agent(policy):
        raise ProbatumNotImplementedError(f"policy: {named}, which validate does not "
                                          f"run yet; pass a Strategy subclass")

    raise PolicyError(f"policy: {named}, which is no Strategy; pass a Strategy "
                      f"subclass")


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
    fit's winner, deciding at its window's closes from its own warm-up, and a seam
    where one window hands the account to the next (Specification, Paths and seams).
    ``windows`` holds each window's first and last row and its parameters."""


    def __init__(self, policy, windows, seam):
        self.policy, self.windows, self.seam = policy, windows, seam


    def init(self, data):
        self.parts   = [self.policy(**params) for _, _, params in self.windows]
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


    def ran(self, policy, fits, seam, inside, venue, rate, risk_free, objective):
        """The path run once over ``inside``, and what its records give."""
        result = _decided(_Path(policy, _composed(self.windows, fits), seam), inside,
                          venue, "weight", None, None, 0.0, risk_free, "full")

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
        self.decided = list(result.policy.decided)
        self.returns = np.full(rows, np.nan)
        self.returns[start:] = result.returns[start:]


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


def _paths(walks):
    """The paths table, a row a path."""
    parts = [_whole("number", [walk.number for walk in walks]),
             _texts("state", [walk.state for walk in walks], nullable=False),
             ("score", "float", True, _floats([walk.score for walk in walks]), None,
              None)]
    parts += _measured([walk.stats for walk in walks], [walk.costs for walk in walks])
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
