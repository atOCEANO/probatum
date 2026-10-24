"""``search``: a policy's parameters searched over a space on the bars before a holdout,
each trial scored on one run there and kept as a row of a Study beside every trial's
returns, the winner run again as a Result when first read and once on the held-out bars
(ADR 0032; Specification, Search)."""

import collections.abc
import copy
import inspect
import logging
import math
import multiprocessing
import os
import pathlib
import shutil
import sys
import tempfile
import textwrap
import warnings

import numpy as np

from ._backtest import (_agent, _annualised, _kind, _markets, _number, _played,
                        _rate, _repeatable, backtest)
from ._causality import _check, _read
from ._data import Data, _before, _chosen, _piece, _row, _when
from ._errors import (DataError, PolicyError, ProbatumAttributeError,
                      ProbatumNotImplementedError, ProbatumRuntimeError,
                      ProbatumTypeError, ProbatumValueError, ProbatumWarning)
from ._probatum import Engine, Table as _Held, run_array, run_arrays, stats_of, threads
from ._result import (COSTS, COUNTS, STATS, Table, _environment, _money, _returns,
                      _text, _version)
from ._samplers import _Grid, _Random, _Tpe
from ._space import Categorical, Float, Int, _whole
from ._strategy import Strategy, _Run, _play
from ._venue import _run
from .stats import (BREAK_EVEN, GROWS, HALVED, PARTS, RETURNS, TRADES, Consistency, Decay,
                    Forecast, _finite, _percent, bootstrap, deflated_sharpe,
                    effective_trials, min_backtest_length, overfitting, plateau,
                    reality_check, sharpe_interval)

LOGGER     = logging.getLogger("probatum")
SAMPLERS   = ("tpe", "random", "grid")
DIRECTIONS = ("maximize", "minimize")
KEEPS      = ("all", "none")
# the columns the trials table has of its own, which no parameter is named
OWN = ("number", "state", "score", "error") + STATS
# what the trials run together in Rust may hold, each one's arrays twice, Python's and
# Rust's, and its equity curve, beyond one a thread
CHUNK = 256 * 2**20
# the package's files, which a warning skips to name the caller's line
PACKAGE = (os.path.dirname(__file__),)
GUARD   = 'put the call under if __name__ == "__main__":'
UNHELD  = {
    "search":   ("holdout: none, so every trial is scored on every bar and nothing is "
                 "left to judge the winner on; hold out the tail, such as holdout=0.3, "
                 "or pass holdout=0 to search every bar"),
    "validate": ("holdout: none, so nothing is left to judge the winner on, and the "
                 "paths show how the search holds up and no more; hold out the tail, "
                 "such as holdout=0.3, or pass holdout=0 to validate on every bar"),
}


def search(policy, space, data, venue, objective="sharpe", direction="maximize",
           trials=100, sampler="tpe", min_trades=0, holdout=None, keep="all", n_jobs=1,
           seed=None, storage=None, risk_free=0.0):
    """Search ``policy``'s parameters over ``space`` on ``data``'s bars before a holdout
    at ``venue``, a Venue or a preset's name, and return the `Study` of every trial
    (ADR 0032; Specification, Search).

    ``policy`` is a Strategy subclass, each trial built as ``Cross(**params)``, or a
    function ``policy(data, **params)`` of the bars a trial runs on that returns
    weights, as an array, a table or a dict of backtest's arguments ``{"policy": ...,
    "stop": ..., "take_profit": ..., "tolerance": ..., "sizing": ...}``, or a Strategy,
    copied before it runs; ``space`` maps each of its parameters to a `probatum.Int`,
    `probatum.Float` or `probatum.Categorical`. A trial is scored on one run over the
    bars before the holdout by ``objective``, a statistic's name, which ``direction``
    maximizes or minimizes; it is complete, floored where it closed fewer round trips
    than ``min_trades``, or failed where its policy or its run raised or its score was
    NaN. ``sampler`` is ``"tpe"``, Optuna's, ``"random"`` or ``"grid"``, each seeded
    from ``seed``, which is drawn from the operating system's entropy and recorded where
    it is None. ``holdout`` holds out the last rows: a share of them in (0, 1), a number
    of them, or the time it starts at; 0 holds out nothing, and None nothing with a
    warning. ``keep="all"`` keeps every trial's returns and ``"none"`` none.
    ``n_jobs`` above 1 runs the trials that do not run together in Rust in that many
    workers spawn starts, -1 one a core, ``risk_free`` is the annual rate the statistics
    are computed against, and ``storage`` is not built yet."""
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        raise ProbatumRuntimeError(f"search: called by a worker of a search as it starts, "
                                   f"as spawn runs the script's top level again in each; "
                                   f"{GUARD}")

    kind  = _policy(policy)
    space = _spaced(space, policy, kind)
    trials, min_trades, n_jobs, resolved = _controls(objective, direction, trials,
                                                     sampler, min_trades, n_jobs)
    _chosen("keep", keep, KEEPS, "use \"all\" or \"none\"")
    if storage is not None:
        raise ProbatumNotImplementedError(f"storage: {storage!r}, which search does not "
                                          f"save to yet; leave it out")

    _rate(risk_free)
    if seed is not None:
        _counted("seed", seed, 0)
    seed = np.random.SeedSequence().entropy if seed is None else int(seed)

    data = data if isinstance(data, Data) else Data(data)
    if not len(data) or not data.available.any():
        raise DataError("data: no bars to search; give a Data that has some")

    held   = _repeatable(venue, "search")
    cut    = _cut(holdout, data)
    inside = _before(data, cut)
    run    = _run(held, inside)
    rate   = _annualised(inside)
    draws  = _draws(sampler, space, trials, seed, direction)
    runner = _Runner(policy, kind, inside, run, rate, float(risk_free), objective,
                     min_trades)

    named = _named(policy)
    LOGGER.info("search: %s trials of %s by %s on %s rows, %s held out", draws.count,
                named, sampler, f"{cut:,}", f"{len(data) - cut:,}")

    pools = []

    def pooled(workers):
        settings = dict(kind=kind, made=runner.made, objective=objective,
                        min_trades=min_trades, rate=rate, risk_free=float(risk_free),
                        keep=keep)
        pools.append(_pooled(policy, held, inside, settings, workers, n_jobs))
        return pools[-1]

    try:
        made, checked, pool = _asked(
            runner, draws, sampler, keep, resolved, None, pooled,
            lambda params: _check(policy, kind, params, inside, held, run))
    finally:
        for started in pools:
            try:
                started.close()
            finally:
                shutil.rmtree(started.folder, ignore_errors=True)

    if pool is not None and pool.broken is not None:
        if not any(trial.state == "complete" for trial in made):
            raise pool.refused() from pool.broken
        warnings.warn(f"n_jobs: the workers broke after {pool.returned:,} trials came "
                      f"back ({_told(pool.broken)}), so the trials in flight failed and "
                      f"no more were asked; the Study holds {len(made):,} of "
                      f"{draws.count:,}", ProbatumWarning, skip_file_prefixes=PACKAGE)

    winner = _winner(made, direction, min_trades)
    # copies, as the winner's own parameters are what best and evaluate_holdout run
    passed = [] if checked is None else [dict(checked)]
    if checked is not None and winner.params != checked:
        if _check(policy, kind, winner.params, inside, held, run):
            passed.append(dict(winner.params))
            LOGGER.info("search: the winner's decisions are causal")

    LOGGER.info("search: %s complete of %s; the best, trial %s, %s %.6g",
                sum(trial.state == "complete" for trial in made), len(made),
                winner.number, objective, _signed(objective) * winner.score + 0.0)

    provenance = dict(
        kind="trials", policy=named,
        space={name: repr(given) for name, given in space.items()},
        objective=objective, direction=direction,
        sampler=dict(name=sampler, **draws.arguments()), seed=seed, trials=trials,
        ran=len(made), min_trades=min_trades, checked=passed,
        holdout=dict(given=_shown(holdout), rows=len(data) - cut,
                     first=None if cut == len(data) else _when(data.time[cut])),
        keep=keep, n_jobs=dict(given=n_jobs, resolved=resolved),
        risk_free=float(risk_free),
        periods_per_year=rate, fingerprint=inside.fingerprint, venue=repr(held),
        venue_name=held.name, data=data.provenance, version=_version(),
        **_environment())

    return Study(
        _trials(made, space),
        _kept(made, keep),
        winner,
        _Ran(policy, kind, held, float(risk_free), data, cut, inside),
        provenance,
        space  = space,
        params = [trial.params for trial in made],
    )


def _asked(runner, draws, sampler, keep, resolved, checked, pooled, check):
    """The trials ``draws`` asks, in their order: once one is checked, weights a random
    or grid search draws run together in Rust, and every other trial in the workers
    ``pooled`` starts where there are more than one, TPE asking a round of them at a
    time, since it chooses each trial from those before; and the first trial that ran
    and decided something checked by ``check`` before any other runs, where
    ``checked`` holds no parameters checked already (Specification, The causality
    check). Returns the trials, the parameters checked and the pool."""
    made, pool = [], None
    while len(made) < draws.count and (pool is None or pool.broken is None):
        number = len(made)
        # a validation's fit runs its trials on pieces of its own
        if (checked is not None and sampler != "tpe" and runner.made == "weights"
                and runner.pieces is None):
            arrived = runner.together(number, draws)
        elif checked is not None and resolved > 1:
            if pool is None:
                pool = pooled(min(resolved, draws.count - number))
            last = min(number + resolved if sampler == "tpe" else draws.count,
                       draws.count)
            arrived = pool.run([(k, draws.ask(k)) for k in range(number, last)])
        else:
            arrived = [runner.trial(number, draws.ask(number))]

        for trial in arrived:
            draws.tell(trial.number, trial)
            if keep == "none":
                trial.returns = None
            made.append(trial)
            LOGGER.debug("search: trial %s %s%s", trial.number, trial.state,
                         "" if trial.score is None
                         else f", {runner.objective} "
                              f"{_signed(runner.objective) * trial.score + 0.0:.6g}")
            if checked is None and trial.state != "failed" and check(trial.params):
                checked = trial.params
                LOGGER.info("search: trial %s's decisions are causal", trial.number)

    return made, checked, pool


def _pooled(policy, venue, data, settings, workers, n_jobs):
    """The workers a search's trials run in, ``workers`` of them for the ``n_jobs``
    given, and the folder they run from, written first."""
    from . import _workers
    folder = pathlib.Path(tempfile.mkdtemp(prefix="probatum-search-"))

    try:
        _workers._wrote(folder, policy, venue, data, settings)
        return _workers._Pool(folder, workers, n_jobs)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _controls(objective, direction, trials, sampler, min_trades, n_jobs):
    """The controls a search's trials run by, refused where one names no statistic,
    direction or sampler or is no whole number it takes: the trials, the round trips a
    trial closes to be scored and the workers asked for, as ints, and the workers that
    gives."""
    _chosen("objective", objective, STATS, f"use one of {', '.join(STATS)}")
    _chosen("direction", direction, DIRECTIONS, "use \"maximize\" or \"minimize\"")
    _counted("trials", trials, 1)
    _chosen("sampler", sampler, SAMPLERS, "use \"tpe\", \"random\" or \"grid\"")
    _counted("min_trades", min_trades, 0, "round trips")
    _counted("n_jobs", n_jobs, None)
    if n_jobs == 0 or n_jobs < -1:
        raise ProbatumValueError(f"n_jobs: {n_jobs}; give a number of workers from 1, "
                                 f"or -1 for one a core")

    resolved = (os.process_cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    if sys.platform == "win32":
        # the most workers a process pool takes on that platform
        resolved = min(resolved, 61)

    return int(trials), int(min_trades), int(n_jobs), resolved


def _draws(sampler, space, trials, seed, direction):
    """The trials ``sampler`` draws from ``space``, ``trials`` of them at most, seeded
    with ``seed``."""
    if sampler == "tpe":
        return _Tpe(space, trials, seed, direction)
    return (_Random if sampler == "random" else _Grid)(space, trials, seed)


def _counted(name, value, least, unit=None):
    """Refuse ``value``, given as ``name``, where it is no whole number of ``unit`` from
    ``least``."""
    fix = ("a whole number" + (f" of {unit}" if unit else "")
           + ("" if least is None else f" from {least}"))

    if not _whole(value):
        raise ProbatumTypeError(f"{name}: {value!r}; give {fix}")

    if least is not None and value < least:
        raise ProbatumValueError(f"{name}: {value}; give {fix}")


def _shown(holdout):
    """What a provenance records of ``holdout`` as it was given."""
    if holdout is None or isinstance(holdout, (bool, np.bool_)):
        return holdout
    if _whole(holdout):
        return int(holdout)
    return float(holdout) if _number(holdout) else str(holdout)


def _policy(policy):
    """Whether ``policy`` is a Strategy subclass each trial builds or a function of the
    data, refusing anything else."""
    if isinstance(policy, type):
        if issubclass(policy, Strategy):
            return "class"
        raise PolicyError(f"policy: the class {policy.__name__}, which is no Strategy; "
                          f"pass a Strategy subclass, or a function of the data that "
                          f"returns a policy")

    if isinstance(policy, Strategy):
        name = type(policy).__name__
        raise PolicyError(f"policy: an instance of {name}; pass the class, {name}, "
                          f"which each trial builds with its parameters")

    if callable(policy) and not _agent(policy):
        return "function"

    raise PolicyError(f"policy: {_kind(policy)}; pass a Strategy subclass, or a "
                      f"function of the data that returns a policy")


def _named(policy):
    """What the search calls ``policy``."""
    return getattr(policy, "__name__", type(policy).__name__)


def _spaced(space, policy, kind):
    """``space`` as a dict of each parameter's range, refusing a name ``policy`` does
    not take, a parameter it needs that the space leaves out, and a name the trials
    table has a column of its own for."""
    if not isinstance(space, collections.abc.Mapping):
        raise ProbatumTypeError(f"space: {space!r}; give a dict of each parameter's "
                                f"range, such as {{\"fast\": probatum.Int(5, 40)}}")

    if not space:
        raise ProbatumValueError("space: no parameters; give each a range, such as "
                                 "{\"fast\": probatum.Int(5, 40)}")

    out = {}
    for name, given in space.items():
        if not isinstance(name, str):
            raise ProbatumTypeError(f"space: a name {name!r}; name each parameter by its "
                                    f"argument, as text")
        if not isinstance(given, (Int, Float, Categorical)):
            raise ProbatumTypeError(f"space: {name} is {given!r}; give a probatum.Int, "
                                    f"probatum.Float or probatum.Categorical")
        if name in OWN:
            raise ProbatumValueError(f"space: {name}, a column the trials table has of "
                                     f"its own; name the parameter otherwise")
        out[name] = given

    label = _named(policy)
    try:
        parameters = list(inspect.signature(policy).parameters.values())
    except (TypeError, ValueError):
        # a callable whose signature cannot be read takes whatever it is given
        return out

    if kind == "function":
        first = next((parameter for parameter in parameters if parameter.kind in (
            parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD,
            parameter.VAR_POSITIONAL)), None)
        if first is None:
            raise PolicyError(f"policy: {label} takes no data; give it the data as its "
                              f"first argument")
        if first.kind != first.VAR_POSITIONAL:
            parameters.remove(first)

    named = {parameter.name: parameter for parameter in parameters
             if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD,
                                   parameter.KEYWORD_ONLY)}

    if not any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters):
        for name in out:
            if name not in named:
                taken = ", ".join(named) or "of which it has none"
                raise ProbatumTypeError(f"space: {name}, which {label} does not take; "
                                        f"name its parameters, {taken}")

    for name, parameter in named.items():
        if parameter.default is parameter.empty and name not in out:
            raise ProbatumTypeError(f"space: no range for {name}, which {label} needs; "
                                    f"give it one")

    return out


def _cut(holdout, data, called="search"):
    """The rows of ``data`` before ``holdout``, which holds out its last rows: a share
    in (0, 1) of them, rounded half to even, a whole number of them, or the time it
    starts at, its first row the first at or after it; 0 holds out nothing, and None
    nothing with the warning of ``called``, search or validate. Each side needs two
    rows an instrument printed."""
    rows    = len(data)
    printed = data.available.any(axis=1)

    if holdout is None or _number(holdout) and holdout == 0:
        if printed.sum() < 2:
            raise DataError(f"data: one printed row to {called}; give a Data of at "
                            f"least 2")
        if holdout is None:
            warnings.warn(UNHELD[called], ProbatumWarning, skip_file_prefixes=PACKAGE)
        return rows

    if isinstance(holdout, (bool, np.bool_)):
        raise ProbatumTypeError(f"holdout: {holdout!r}; give a share of the rows, a "
                                f"number of them, or the time the holdout starts at")

    if _whole(holdout):
        if holdout < 0:
            raise ProbatumValueError(f"holdout: {holdout}; give a number of rows from 1")
        cut = max(rows - int(holdout), 0)
    elif isinstance(holdout, (float, np.floating)):
        if not 0.0 < holdout < 1.0:
            raise ProbatumValueError(f"holdout: {holdout!r}; give a share of the rows "
                                     f"above 0 and below 1, a number of them, or the "
                                     f"time the holdout starts at")
        cut = rows - round(rows * float(holdout))
    else:
        try:
            cut = _row(data, holdout, True, "holdout")
        except DataError as error:
            raise ProbatumValueError(str(error)) from None

    before, after = int(printed[:cut].sum()), int(printed[cut:].sum())
    if before < 2 or after < 2:
        raise ProbatumValueError(f"holdout: {holdout!r} leaves {before:,} of the printed "
                                 f"rows to search and {after:,} to hold out; leave at "
                                 f"least 2 on each side")

    return cut


class _Trial:
    """One trial: its ``number``, its ``params``, its ``state`` and ``score``, the
    ``stats`` and ``returns`` of a trial that ran, and a failure's ``error`` as text,
    the exception it ``raised``, kept for the last failure in this process alone, and its
    ``trace`` where it was in a ``worker``."""

    __slots__ = ("number", "params", "state", "score", "stats", "returns", "error",
                 "raised", "trace", "worker")


    def __init__(self, number, params, state, score=None, stats=None, returns=None,
                 error=None, raised=None, trace=None, worker=False):
        self.number, self.params, self.state, self.score = number, params, state, score
        self.stats, self.returns, self.error, self.raised = stats, returns, error, raised
        self.trace, self.worker = trace, worker


def _scored(number, params, stats, returns, objective, min_trades):
    """The trial that ran with ``stats`` and ``returns``: floored where it closed fewer
    round trips than ``min_trades``, failed where its score is NaN, else complete."""
    if stats["num_trades"] < min_trades:
        return _Trial(number, params, "floored", stats=stats, returns=returns)

    score = float(stats[objective])
    if math.isnan(score):
        return _Trial(
            number,
            params,
            "failed",
            stats = stats,
            error = f"objective: {objective} is NaN",
        )

    return _Trial(number, params, "complete", score, stats, returns)


def _told(error):
    """A trial's error as text."""
    return f"{type(error).__name__}: {error}"


class _Runner:
    """What runs a search's trials in this process: the policy and its kind, the
    in-sample Data and the run `_run` gives it, the annualisation and the risk-free
    rate, an engine of each level built once and reset for every trial, and the kind of
    decisions the first trial that ran made, which every later one makes too."""


    def __init__(self, policy, kind, data, run, rate, risk_free, objective, min_trades,
                 pieces=None):
        self.policy, self.kind, self.data, self.run = policy, kind, data, run
        # a validation's fit: each piece's Data, its run and the rows it opens past
        self.pieces = pieces
        self.rate, self.risk_free = rate, risk_free
        self.objective, self.min_trades = objective, min_trades

        self.markets = _markets(run)
        quote        = run.settings["quote"]
        # what the account opens with, as the statistics count it from
        self.opening = (quote["spot"] + quote["margin"] if isinstance(quote, dict)
                        else float(quote))

        self.engines = {}
        self.made = self.failed = None


    def engine(self, level):
        """The engine at ``level``, built the first time it is asked for."""
        engine = self.engines.get(level)
        if engine is None:
            engine = self.engines[level] = Engine(
                self.run.candles,
                report = True,
                record = level,
                **self.run.settings,
            )

        return engine


    def decided(self, params, data=None, run=None):
        """What the policy decides with ``params``: a Strategy to play, or the arguments
        of an array run, on ``data`` and its ``run``, the in-sample bars where they are
        None."""
        if self.kind == "class":
            # copied as backtest copies it, so an instance that cannot be copied fails
            # its trial rather than the winner's run (ADR 0024)
            return "strategy", copy.deepcopy(self.policy(**params))

        data, run = (self.data, self.run) if data is None else (data, run)
        given = self.policy(data, **params)
        if isinstance(given, Strategy):
            # what a function returns is a prototype, copied once before it runs
            # (ADR 0024)
            return "strategy", copy.deepcopy(given)

        return "weights", _given(given, data, run)


    def played(self, made, decided):
        """The statistics and the returns of one run of ``decided``, a Strategy on an
        engine of the full level, whose account reads the order log, and weights on
        one of the account level."""
        if made == "strategy":
            engine = self.engine("full")
            _play(decided, _Run(engine, self.data, self.run.start, self.markets))
        else:
            engine = self.engine("account")
            run_array(engine, *decided, start=self.run.start)

        stats  = engine.stats(self.rate, self.risk_free)
        equity = np.concatenate([[self.opening], engine.equity_curve()])
        return stats, _returns(equity)


    def kind_of(self, params, data=None, run=None):
        """What the policy decides with ``params``, refused where it is of another kind
        than the trials before decided."""
        made, decided = self.decided(params, data, run)
        if self.made is not None and made != self.made:
            given  = "a Strategy" if made == "strategy" else "weights"
            before = "a Strategy" if self.made == "strategy" else "weights"
            # a validation's fit decides in the kind its first search's winner did
            those  = ("the trials before it" if self.pieces is None
                      else "the first search's winner")
            raise PolicyError(f"policy: returned {given}, where {those} returned "
                              f"{before}; return one kind for every parameter")

        return made, decided


    def trial(self, number, params):
        """Trial ``number``, run with ``params``."""
        try:
            if self.pieces is None:
                made, decided = self.kind_of(params)
                stats, returns = self.played(made, decided)
            else:
                made, (stats, returns) = self.pieced(params)
        except Exception as error:
            # a search raises from its last failure alone, and each one's traceback
            # holds the frames that ran it
            if self.failed is not None:
                self.failed.raised = None
            self.failed = _Trial(
                number,
                params,
                "failed",
                error  = _told(error),
                raised = error,
            )
            return self.failed

        self.made = made
        return _scored(number, params, stats, returns, self.objective, self.min_trades)


    def pieced(self, params):
        """A trial on each piece of a validation's fit, a Strategy's own copy or a
        function called with the piece's Data on the piece's own engine, weights run as
        an array that opens where the piece's account opens; scored as a search's trial
        where one piece ran and otherwise on the pieces' returns joined end to end
        (Specification, Fits; A function). Returns what the trial decided, and its
        statistics and returns."""
        parts, made = [], None
        for k, (data, run, later) in enumerate(self.pieces):
            engine = self.engines.get(k)
            if engine is None:
                engine = self.engines[k] = Engine(
                    run.candles,
                    report = True,
                    record = "full",
                    **run.settings,
                )

            made, decided = self.kind_of(params, data, run)
            if made == "strategy":
                played       = _Run(engine, data, run.start, self.markets)
                played.later = later
                _play(decided, played)
                if len(played.rows) >= 2:
                    parts.append(engine)
                continue

            # the rows the account sees from the first printed row at or after later
            skip = int(data.available[:later].any(axis=1).sum()) if later else 0
            if len(decided[0]) - skip >= 2:
                run_array(engine, *_skipped(decided, skip), start=run.start + skip)
                parts.append(engine)

        if not parts:
            raise ProbatumValueError(f"policy: no piece of the fit holds two rows past "
                                     f"its warm-up with {params}; give the training "
                                     f"spans more bars")

        if len(parts) == 1:
            stats = parts[0].stats(self.rate, self.risk_free)
            return made, (stats, _returns(np.concatenate([[self.opening],
                                                          parts[0].equity_curve()])))

        return made, (_joined(parts, self.opening, self.rate, self.risk_free), None)


    def together(self, first, draws):
        """The trials from ``first`` on that a chunk holds, as many as there are threads
        and then until their arrays reach ``CHUNK``, each decided in turn, one that
        raises or decides otherwise than weights failing alone, and every one that
        decided run together on an engine of its own over the shared bars."""
        least = threads()
        made, played, size = [], [], 0
        while first + len(made) < draws.count and (len(played) < least or size < CHUNK):
            number = first + len(made)
            params = draws.ask(number)
            try:
                decided = self.kind_of(params)[1]
            except Exception as error:
                # no search raises from a failure here, as one trial was checked, and
                # its traceback would hold the chunk
                made.append(_Trial(number, params, "failed", error=_told(error)))
                continue

            made.append(_Trial(number, params, None))
            played.append(decided)
            size += 8 * len(decided[0]) + 2 * sum(
                given.nbytes for given in decided if isinstance(given, np.ndarray))

        engine = self.engine("account")
        # what the trials run alone recorded there, which nothing reads any more
        engine.reset()
        ran = iter(run_arrays(engine, played, self.run.start, self.rate,
                              self.risk_free))

        for k, trial in enumerate(made):
            if trial.state is None:
                stats, curve = next(ran)
                returns = _returns(np.concatenate([[self.opening], curve]))
                made[k] = _scored(trial.number, trial.params, stats, returns,
                                  self.objective, self.min_trades)

        return made


def _joined(engines, opening, rate, risk_free):
    """The statistics of the runs on ``engines``, each opened at ``opening``, joined end
    to end: each one's equity scaled to continue from the end of the one before, every
    round trip's pnl as its run made it, and the fills, the steps held in a position and
    the funding summed, the join stopping at the first run that ends at or below zero
    equity, the later ones adding their steps at that equity and nothing else."""
    curves, pnl, held, fills, funding, scale = [], [], 0, 0, 0.0, 1.0
    for engine in engines:
        curve = engine.equity_curve()
        if scale <= 0.0:
            curves.append(np.full(len(curve), curves[-1][-1]))
            continue

        curves.append(curve * scale)
        pnl.extend(trip["pnl"] for trip in engine.trades_since(0))
        stats = engine.stats(rate, risk_free)
        held, fills = held + engine.held_steps(), fills + stats["num_fills"]
        funding += stats["funding_paid"]
        scale = curves[-1][-1] / opening

    return stats_of(float(opening), np.concatenate(curves), np.array(pnl, dtype=float),
                    rate, risk_free, held, fills, funding)


def _pieces(data, venue, specs):
    """A fit's pieces from ``specs``, each ``(first, start, stop, later)`` in the rows
    of ``data``, the bars before a validation's holdout: a piece from the first row on
    is those bars cut after its last, from ``start``, and one after a test span a Data
    of its own from ``first``, its account opening ``later`` rows in at the
    earliest."""
    out = []
    for first, start, stop, later in specs:
        piece = (_before(data, stop)[start:] if first == 0
                 else _piece(data, first, stop))
        out.append((piece, _run(venue, piece), later))

    return out


def _given(given, data, run):
    """What a function returned, weights, as the arguments of an array run on ``run``,
    read as backtest reads them."""
    return _played(*_read(given, data), data, run)


def _skipped(decided, skip):
    """An array run's arguments ``decided`` without its first ``skip`` rows."""
    values, sizing, tolerance, *barriers = decided
    return (values[skip:], sizing, tolerance,
            *[barrier if barrier is None or isinstance(barrier, float) else barrier[skip:]
              for barrier in barriers])


def _winner(made, direction, min_trades, pieces=False):
    """The complete trial of ``made`` with the best score, the lowest-numbered of equal
    ones, refusing a search with none; ``pieces`` where its trials ran on a
    validation's training pieces, which a backtest of the in-sample bars does not
    repeat."""
    complete = [trial for trial in made if trial.state == "complete"]
    if complete:
        pick = max if direction == "maximize" else min
        return pick(complete, key=lambda trial: trial.score)

    floored = [trial for trial in made if trial.state == "floored"]
    if floored:
        most  = max(trial.stats["num_trades"] for trial in floored)
        where = "the fit's training pieces" if pieces else "the in-sample bars"
        raise ProbatumValueError(f"min_trades: {min_trades}, and no trial of {len(made):,} "
                                 f"closed that many round trips on {where}, the most "
                                 f"{most:,}; lower it")

    last = made[-1]
    fix  = ("make it run on each of the fit's training pieces, each a Data of its own"
            if pieces else f"backtest it with {last.params} to see where")

    error = ProbatumRuntimeError(f"policy: all {len(made):,} trials failed, the last, "
                                 f"trial {last.number}, with {last.error}; {fix}")
    if last.trace is not None:
        error.add_note(f"in the worker that ran it:\n{last.trace}")

    raise error from last.raised


def _trials(made, space):
    """The trials table of ``made``, a row a trial: its number, state and score, each
    parameter, the statistics and the error, a value a trial lacks missing."""
    parts = [("number", "int", False, np.array([trial.number for trial in made],
                                               dtype=np.int64), None, None),
             _texts("state", [trial.state for trial in made], nullable=False),
             ("score", "float", True, _floats([trial.score for trial in made]), None,
              None)]
    parts += [_param(name, given, [trial.params[name] for trial in made])
              for name, given in space.items()]

    for key in STATS:
        values = [None if trial.stats is None else trial.stats[key] for trial in made]
        if key in COUNTS:
            parts.append(_ints(key, values))
        else:
            parts.append((key, "float", True, _floats(values), None, None))

    parts.append(_texts("error", [trial.error for trial in made]))
    return Table(_Held(parts))


def _floats(values):
    return np.array([np.nan if value is None else value for value in values],
                    dtype=np.float64)


def _ints(name, values):
    given   = np.array([value is not None for value in values])
    numbers = np.array([0 if value is None else value for value in values],
                       dtype=np.int64)
    return name, "int", True, numbers, None if given.all() else given, None


def _texts(name, values, nullable=True):
    ends, valid, data = _text([None if value is None else str(value)
                               for value in values])
    return name, "text", nullable, ends, valid, data


def _param(name, given, values):
    """The column of a parameter of the range ``given``, of the kind its values can
    share: an Int's int and a Float's float, and a Categorical's int where every choice
    is an int, float where ints and floats meet, bool where every choice is a bool, and
    text where a string meets any other kind or a bool meets None, None a missing value
    of the others' kind."""
    choices = given.choices if isinstance(given, Categorical) else (
        [0] if isinstance(given, Int) else [0.0])
    kinds = {type(choice) for choice in choices if choice is not None}

    if kinds == {bool} and None not in choices:
        return name, "bool", False, np.array(values, dtype=bool), None, None
    if kinds and kinds <= {int}:
        return _ints(name, values)
    if kinds and kinds <= {int, float}:
        return name, "float", True, _floats(values), None, None
    return _texts(name, values)


def _kept(made, keep):
    """Every trial's returns on the in-sample bars, a failed trial's row NaN, or None
    where ``keep`` keeps none."""
    if keep == "none":
        return None

    width = next(len(trial.returns) for trial in made if trial.returns is not None)
    out   = np.full((len(made), width), np.nan)

    for k, trial in enumerate(made):
        if trial.returns is not None:
            out[k] = trial.returns

    out.flags.writeable = False
    return out


def _line(label, said):
    """One line of a Study's summary, its words wrapped at 88 columns beneath
    themselves."""
    return textwrap.fill(f"  {label:<18}{said}", width=88, subsequent_indent=" " * 20,
                         break_long_words=False, break_on_hyphens=False)


def _day(time):
    """The UTC date of ``time``, int64 nanoseconds since 1970."""
    return str(np.datetime64(int(time), "ns").astype("datetime64[D]"))


def _venue_named(given):
    """The venue a summary names: its preset's name, or what it was built as."""
    return given["venue_name"] or "a venue built by hand"


def _signed(objective):
    """-1 for an objective a text prints as a loss, the negative return it is, else 1."""
    return -1.0 if objective in RETURNS and RETURNS[objective][2] else 1.0


def _bounds(held, flip):
    """An interval's bounds as a summary prints them, a loss as the negative return it
    is."""
    low, high = sorted((flip * held.low + 0.0, flip * held.high + 0.0))
    return f"{low:.2f} to {high:.2f}"


def _consistent(scores, unit, objective, direction):
    """A Consistency of ``scores``, the share of them past the objective's break-even
    in ``direction``, where it has one."""
    if not len(scores):
        raise ProbatumValueError(f"consistency: no {unit[:-1]} completed, so there is no "
                                 f"score to read")

    even = BREAK_EVEN.get(objective)
    past = (None if even is None else
            sum(1 for score in scores
                if (score > even if direction == "maximize" else score < even)))

    return Consistency(scores, unit, even, None if even is None else past / len(scores),
                       objective, direction)


class _Ran:
    """How a Study runs its winner again: the policy and its kind, the venue and the
    risk-free rate, the whole Data, the row its holdout starts at and the in-sample
    Data."""

    __slots__ = ("policy", "kind", "venue", "risk_free", "data", "cut", "inside")


    def __init__(self, policy, kind, venue, risk_free, data, cut, inside):
        self.policy, self.kind, self.venue, self.risk_free = policy, kind, venue, risk_free
        self.data, self.cut, self.inside = data, cut, inside


    def __call__(self, params, data):
        """The winner, ``params``, run on ``data`` at the full level, as backtest runs
        it."""
        if self.kind == "class":
            return backtest(self.policy(**params), data, self.venue,
                            risk_free=self.risk_free)

        given = self.policy(data, **params)
        if isinstance(given, dict):
            return backtest(given.get("policy"), data, self.venue,
                            sizing=given.get("sizing", "weight"), stop=given.get("stop"),
                            take_profit=given.get("take_profit"),
                            tolerance=given.get("tolerance", 0.0),
                            risk_free=self.risk_free)

        return backtest(given, data, self.venue, risk_free=self.risk_free)


class Study:
    """What a search or a validation found (ADR 0026), ``kind`` saying which.

    A search's, of kind "trials", keeps every trial it asked: ``trials``, a table of a
    row each, its ``number``, ``state``, ``score``, each parameter, the 15 statistics
    and its ``error``, and ``returns``, every trial's returns on the in-sample bars,
    ``(R, T)`` and read-only, counted as a Result counts them, a failed trial's row
    NaN. A validation's, of kind "paths", keeps ``paths``, a table of a row a path, its
    ``number``, ``state`` and ``score``, the 15 statistics and the seven costs of the
    account rows it owns, and its ``error``; ``fits``, a row a split, with what it
    tests, its training, purged and embargoed bars, its pieces, its winner's
    parameters and fitted score, its trials run and complete, its seed, state and
    error; ``windows``, a row a stretch of a path one fit traded, with the first and
    last account rows it owns and their times, the rows it decided on, its score, its
    statistics and its costs; and ``returns``, each path's returns on the in-sample
    account rows, ``(paths, T)`` and read-only, NaN where it owns no row and on a
    failed path. ``returns`` is None where the call kept none. A model's validation
    scores its paths and windows on the Sharpe, adds each path's forecast to its row,
    counts each fit's rows and those a NaN feature left out in place of its search, and
    keeps the pooled ``forecast``, None for a searched policy.

    Both hold a winner, a validation's the one of the search it runs over the bars
    before the holdout first, or a model's final fit: ``best``, its Result on the
    in-sample bars, run again when first read; ``best_params`` and ``best_score``;
    ``evaluate_holdout()``, the winner run once on the held-out bars, and ``holdout``,
    the Result it gave;
    ``looks``, how many times the winner has been handed the held-out bars; and
    ``provenance``, what the call was given and where it ran. A search's is judged
    across its trials by ``deflated_sharpe(null)``, ``effective_trials()``,
    ``overfitting()``, ``reality_check()``, ``min_backtest_length()`` and
    ``plateau(level)``, as ``probatum.stats`` computes them (ADR 0033)."""


    def __init__(self, trials, returns, winner, ran, provenance, tables=None,
                 forecast=None, space=None, params=None):
        self._trials, self.returns, self.provenance = trials, returns, provenance
        # a validation's paths, fits and windows, None for a search's Study
        self._tables = tables
        # a model's pooled Forecast, or why it has none
        self._forecast = forecast
        self._looks    = 0
        self._winner, self._ran = winner, ran
        # a search's ranges and each trial's parameters, which its plateau reads
        self._space, self._params = space, params
        self._best = self._holdout = self._effective = None


    @property
    def kind(self):
        """What the Study holds: "trials", a search's, or "paths", a validation's."""
        return "trials" if self._tables is None else "paths"


    @property
    def trials(self):
        """A search's trials, a row each."""
        if self._tables is not None:
            raise ProbatumAttributeError("trials: a Study of paths, which keeps each "
                                         "fit's winner and no trials; read paths, fits "
                                         "or windows")

        return self._trials


    @property
    def paths(self):
        """A validation's paths, a row each."""
        return self._table("paths")


    @property
    def fits(self):
        """A validation's fits, a row a split."""
        return self._table("fits")


    @property
    def windows(self):
        """A validation's windows, a row each stretch of a path one fit traded."""
        return self._table("windows")


    def _table(self, name):
        if self._tables is None:
            raise ProbatumAttributeError(f"{name}: a Study of trials, which search "
                                         f"made; validate makes a Study of paths")

        return self._tables[name]


    def _called(self):
        """What a refusal calls the call that made the Study, and the call to make
        again."""
        if self._tables is None:
            return "search", "search"
        return "validation", "validate"


    @property
    def forecast(self):
        """A model's pooled Forecast over every complete path's rows, None where it has
        none and for a searched policy."""
        return self._forecast if isinstance(self._forecast, Forecast) else None


    @property
    def best_score(self):
        """The winner's score."""
        return self._winner.score


    @property
    def looks(self):
        """How many times the winner has been handed the held-out bars, which nothing
        but evaluate_holdout counts."""
        return self._looks


    @property
    def best_params(self):
        """The winner's parameters, a copy each time, so what ``best`` and
        ``evaluate_holdout`` run is the winner whatever a caller does with one."""
        return dict(self._winner.params)


    @property
    def best(self):
        """The winner's Result on the in-sample bars, at the full level, run again when
        first read and kept. Its statistics are the trial's to the bit, so one that
        differs is refused: the policy's code, or the globals it reads, changed since
        the search."""
        if self._best is None:
            ran    = self._ran
            result = ran(self.best_params, ran.inside)

            row, again = self._winner.stats, result.stats
            changed = [key for key in STATS if again[key] != row[key]]
            if changed:
                key = changed[0]
                noun, verb = self._called()
                why = ("it ran in a worker, which imported the policy afresh under a hash "
                       "seed of its own, so give the policy what it reads as its "
                       "arguments, or set PYTHONHASHSEED" if self._winner.worker else
                       f"the policy's code or the globals it reads changed since the "
                       f"{noun}, so {verb} again")
                raise ProbatumRuntimeError(f"best: trial {self._winner.number}'s {key} was "
                                           f"{row[key]!r} on the in-sample bars, and run "
                                           f"again it is {again[key]!r}; {why}")

            self._best = result

        return self._best


    @property
    def holdout(self):
        """The Result of the winner on the held-out bars once ``evaluate_holdout`` has
        run it, None until then."""
        return self._holdout


    def evaluate_holdout(self):
        """The winner run once on the held-out bars, ``data[cut:]``, at the full level,
        its warm-up falling inside them, and the Result kept as ``holdout``; a second
        call returns it. ``best`` is read first, so a winner that no longer gives its
        trial's statistics is refused before the held-out bars are touched, and the look
        counts from the moment the winner is handed them, so a run that raises there
        spends it. A search or a validation that held nothing out refuses."""
        ran = self._ran
        if ran.cut == len(ran.data):
            noun, verb = self._called()
            raise ProbatumValueError(f"evaluate_holdout: the {noun} held nothing out; "
                                     f"{verb} with a holdout, such as holdout=0.3")

        if self._holdout is None:
            self.best
            held = ran.data.split(ran.cut)[1]
            self._looks += 1
            self._holdout = ran(self.best_params, held)

        return self._holdout


    def summary(self, null=None, plateau=None):
        """The Study in a few lines of text. A search's: what it searched, at which venue
        and on what bars, its in-sample span and holdout, the winner's parameters and
        score, with ``null``, a random search of the same policy over the same space on
        the same venue and in-sample bars scored alike, the winner's deflated Sharpe and
        the mean of the null's complete trials, the overfitting probability, with
        ``plateau``, a level in the objective's units, the share of the trials near the
        winner that scored past it, and the holdout as it stands, each line that cannot
        be read saying why. A validation's: its paths and how they were cut, each path's
        score, their median with the median path's interval, its costs as a share of the
        gross edge, the reminder that paths are stability evidence, and the holdout
        (Specification, Summaries)."""
        given = self.provenance
        data  = given["data"]
        names = [name.split(":", 1)[-1] for name in data["instruments"]]
        where = (f"{names[0] if len(names) == 1 else f'{len(names)} instruments'} "
                 f"{data['interval']}")

        if self._tables is None:
            if plateau is not None:
                plateau = _finite("plateau", plateau, "a score in the objective's "
                                  "units, a finite number")

            lines = [f"search: {given['policy']}, {_venue_named(given)}, {where}, "
                     f"{given['ran']:,} {given['sampler']['name']} trials",
                     f"in sample {_day(self._ran.inside.time[0])} to "
                     f"{_day(self._ran.inside.time[-1])}, {self._held()}"]
            chosen = ", ".join(f"{key} {value}" for key, value in self.best_params.items())
            best   = _signed(given["objective"]) * self.best_score + 0.0
            lines.append(_line("best", f"{chosen}: {given['objective']} {best:.2f}"))
            if null is not None:
                lines.append(_line("deflated sharpe", self._deflated(null)))
                lines.append(_line("null", self._null(null)))

            lines.append(_line("overfitting", self._overfit()))
            if plateau is not None:
                lines.append(_line("plateau", str(self.plateau(plateau))))

            lines.append(_line("holdout", self._held_out()))
            return "\n".join(lines)

        if null is not None:
            raise ProbatumValueError("null: given to a validation's summary, which reads "
                                     "its paths; give the null to its search's summary")

        if plateau is not None:
            raise ProbatumValueError(f"plateau: {plateau!r} given to a validation's "
                                     f"summary, whose paths are no competing "
                                     f"configurations; give it to its search's summary")

        paths, fits = self._tables["paths"], self._tables["fits"]
        cut   = given["cv"]
        lines = [f"validate: {given['policy']}, {_venue_named(given)}, {where}, "
                 f"{self._held()}",
                 _line("paths", f"{len(paths):,} out-of-sample path"
                                f"{'s' * (len(paths) != 1)} from {len(fits):,} fits by "
                                f"{cut['given']}, over the bars before the holdout"),
                 _line("embargo", f"{cut['embargo']:,} bars after each test span")]

        objective = given["objective"]
        flip      = _signed(objective)
        done      = np.asarray(paths["state"]) == "complete"
        scores    = np.asarray(paths["score"], dtype=np.float64)

        lines.append(_line(objective, "  ".join(f"{flip * score + 0.0:.2f}"
                                                for score in scores[done])
                           or "no path completed"))

        if done.any():
            median = flip * float(np.median(scores[done])) + 0.0
            lines.append(_line("median", f"{median:.2f}; "
                                         f"{self._median_path(scores, done)}"))
            if self._forecast is not None:
                said = (f"none: {self._forecast}" if isinstance(self._forecast, str)
                        else str(self._forecast).removeprefix("forecast "))
                lines.append(_line("forecast", said))
            lines.append(_line("costs", self._costs(paths, done)))

        lines.append(_line("evidence", "the paths are stability evidence; the verdict "
                                       "comes from the holdout"))
        lines.append(_line("holdout", self._held_out()))
        return "\n".join(lines)


    def consistency(self, stretches=None):
        """How the winner's score holds up across its run, as a Consistency: a search's
        winner scored on ``stretches`` stretches of its run over the in-sample bars, 4
        unless given, its rows after the first divided as PurgedKFold divides bars and
        each scored by the objective as a validation's window is, or a validation's
        windows; and the share of them past the objective's break-even in the search's
        direction, where the objective has one (Specification, Consistency and decay)."""
        given = self.provenance
        objective, direction = given["objective"], given["direction"]

        if self._tables is not None:
            if stretches is not None:
                raise ProbatumValueError("stretches: given to a validation, whose windows "
                                         "are its stretches; leave it out")
            scores = np.asarray(self._tables["windows"]["score"], dtype=np.float64)
            return _consistent(scores[~np.isnan(scores)], "windows", objective, direction)

        stretches = 4 if stretches is None else stretches
        if not _whole(stretches):
            raise ProbatumTypeError(f"stretches: {stretches!r}; give a whole number from 2")

        result = self.best
        rows   = len(result.time)
        if not 2 <= stretches <= rows - 1:
            raise ProbatumValueError(f"stretches: {stretches}; give a whole number from 2 "
                                     f"to {rows - 1:,}, the in-sample rows after the first")

        from ._validate import _owned
        positions = result.holdings["position"].reshape(rows, -1)
        holding   = (np.abs(positions) > 0.0).any(axis=1)
        rate, risk_free = result.provenance["periods_per_year"], given["risk_free"]

        size, extra = divmod(rows - 1, stretches)
        scores, first = [], 1
        for k in range(stretches):
            last = first + size + (k < extra) - 1
            scores.append(float(_owned(result, holding, first, last, rate,
                                       risk_free)[0][objective]))
            first = last + 1

        return _consistent(scores, "stretches", objective, direction)


    def decay(self):
        """The mean gap between what a fit scored and what its bars then earned, as a
        Decay, signed so that a positive decay is worse out of sample: each window of a
        validation against its fit's in-sample score, or a search's winner against its
        holdout, which needs a look whose run kept its result. An objective that grows
        with the span it is scored on is refused, since spans of other lengths would
        make the gap measure their lengths (Specification, Consistency and decay)."""
        given = self.provenance
        objective, direction = given["objective"], given["direction"]

        if objective in GROWS:
            raise ProbatumValueError(f"decay: {objective} grows with the span it is scored "
                                     f"on, so a gap would measure the spans' lengths; "
                                     f"decay reads a rate or a ratio, such as sharpe")

        if self._forecast is not None:
            raise ProbatumValueError("decay: a model's fits score no bars of their own, "
                                     "so no window has a fitted score to fall from; read "
                                     "its forecast")

        if self._tables is not None:
            windows, fits = self._tables["windows"], self._tables["fits"]
            fitted = dict(zip(np.asarray(fits["number"]).tolist(),
                              np.asarray(fits["fitted"], dtype=np.float64).tolist()))
            pairs = [(fitted[fit], score) for fit, score in zip(
                np.asarray(windows["fit"]).tolist(),
                np.asarray(windows["score"], dtype=np.float64).tolist())]
            pairs = [(a, b) for a, b in pairs if not (math.isnan(a) or math.isnan(b))]
        else:
            noun = "the search"
            if self._ran.cut == len(self._ran.data):
                raise ProbatumValueError(f"decay: {noun} held nothing out, so no bars "
                                         f"earned after the fit; search with a holdout, "
                                         f"such as holdout=0.3")
            if self._holdout is None and not self._looks:
                raise ProbatumValueError("decay: the holdout is not yet run; "
                                         "evaluate_holdout() runs it, and spends a look")
            if self._holdout is None:
                raise ProbatumValueError(f"decay: the holdout was read {self._looks} "
                                         f"time{'s' * (self._looks != 1)} and its run "
                                         f"raised, so no result is kept; "
                                         f"evaluate_holdout() runs it again, and spends a "
                                         f"look")

            pairs = [(self.best_score, float(self._holdout.stats[objective]))]

        gaps = [a - b if direction == "maximize" else b - a for a, b in pairs]
        if not gaps:
            raise ProbatumValueError("decay: no window completed, so no fit has bars it "
                                     "earned on")

        return Decay(gaps, sum(gaps) / len(gaps), objective, direction)


    def _held(self):
        """What the Study held out, as its summary says it."""
        held = self.provenance["holdout"]
        if not held["rows"]:
            return "every bar searched, none held out"

        given = held["given"]
        if isinstance(given, float) and 0.0 < given < 1.0:
            return f"holdout the last {given:.0%} ({held['rows']:,} bars)"
        if isinstance(given, int) and not isinstance(given, bool):
            return f"holdout the last {held['rows']:,} bars"
        return f"holdout from {held['first']} ({held['rows']:,} bars)"


    def _held_out(self):
        """The holdout as it stands: none, not yet run, run, or run and raised."""
        noun, verb = self._called()
        if self._ran.cut == len(self._ran.data):
            return f"none: the {noun} held every bar"

        if self._holdout is None:
            if not self._looks:
                return "not yet run; evaluate_holdout() spends a look"
            return (f"read {self._looks} time{'s' * (self._looks != 1)}; its run raised, so "
                    f"no result is kept")

        objective = self.provenance["objective"]
        record    = self._holdout.stats
        flip      = _signed(objective)
        said      = f"{objective} {flip * record[objective] + 0.0:.2f}"
        held      = record.interval(objective)

        if held is not None:
            what = ("middle {} of resampled histories" if held.kind == "range"
                    else "{} interval").format(_percent(held.level))
            said += f", {what} {_bounds(held, flip)}"

        said += f", look {self._looks} of {self._looks}"
        if held is not None and self._looks > 1:
            wider = record.interval(objective, level=1.0 - 0.05 / self._looks)
            said += f"; at {_percent(wider.level)}, {_bounds(wider, flip)}"

        return said


    def deflated_sharpe(self, null):
        """The winner's deflated Sharpe against every trial, and its cluster's against
        the effective number, ``null`` a random search of the same policy over the same
        space, venue and in-sample bars, scored alike
        (``probatum.stats.deflated_sharpe``)."""
        return deflated_sharpe(self, null)


    def effective_trials(self):
        """The effective number of the trials that ran
        (``probatum.stats.effective_trials``)."""
        return effective_trials(self)


    def overfitting(self):
        """The probability of backtest overfitting of the complete trials
        (``probatum.stats.overfitting``)."""
        return overfitting(self)


    def reality_check(self, benchmark=None, resamples=1000, seed=None):
        """White's Reality Check and Hansen's SPA of whether any trial that ran beats
        ``benchmark``, cash unless given (``probatum.stats.reality_check``)."""
        return reality_check(self, benchmark, resamples=resamples, seed=seed)


    def min_backtest_length(self):
        """The minimum backtest length of the trials at the winner's Sharpe
        (``probatum.stats.min_backtest_length``)."""
        return min_backtest_length(self)


    def plateau(self, level):
        """The share of the trials near the winner, in normalised parameters, that
        scored past ``level`` (``probatum.stats.plateau``)."""
        return plateau(self, level)


    def _deflated(self, null):
        """The deflated Sharpe's line, or why it has none."""
        self._vetted(null)

        given = self.provenance
        if (given["objective"], given["direction"]) != ("sharpe", "maximize"):
            ranked = ("minimised sharpe" if given["objective"] == "sharpe" else
                      f"ranked on {given['objective']}")
            return (f"none: the search {ranked}, and a deflation is of a Sharpe a search "
                    f"maximised")

        try:
            found = deflated_sharpe(self, null)
        except (ProbatumRuntimeError, ProbatumValueError) as error:
            return f"none: {error}"

        return str(found).removeprefix("deflated sharpe ")


    def _overfit(self):
        """The overfitting probability's line, or why it has none."""
        objective = self.provenance["objective"]
        complete  = int((np.asarray(self._trials["state"]) == "complete").sum())

        if self.returns is None:
            return "none: the search kept no returns"
        if objective not in HALVED:
            return (f"none: {objective} is not a statistic the halves give whatever the "
                    f"order of their returns")
        if complete < 2:
            return "none: 1 complete trial"
        if self.returns.shape[1] - 1 < PARTS:
            return (f"none: {self.returns.shape[1] - 1} returns, fewer than its {PARTS} "
                    f"blocks")

        return str(overfitting(self)).removeprefix("overfitting ")


    def _null(self, null):
        """The line a null gives: the mean score of its complete trials."""
        self._vetted(null)

        mine   = self.provenance
        trials = null.trials
        scores = np.asarray(trials["score"], dtype=np.float64)[
            np.asarray(trials["state"]) == "complete"]
        if not scores.size:
            raise ProbatumValueError("null: none of its trials completed, so it has no "
                                     "mean to give")

        mean = _signed(mine["objective"]) * scores.mean() + 0.0
        return f"mean {mine['objective']} {mean:.2f} over {scores.size:,} trials"


    def _vetted(self, null):
        """``null`` refused unless it is a random search of the same policy over the same
        space on the same venue and in-sample bars, scored alike."""
        if not isinstance(null, Study):
            raise ProbatumTypeError(f"null: {type(null).__name__}; give the Study of a "
                                    f"random search")

        if null.kind != "trials":
            raise ProbatumValueError("null: a validation's Study; give the Study of a "
                                     "random search")

        mine, theirs = self.provenance, null.provenance
        if theirs["sampler"]["name"] != "random":
            raise ProbatumValueError(f"null: a search by {theirs['sampler']['name']}; give "
                                     f"a random search, whose trials are independent "
                                     f"draws of the space")

        for key, said in (("policy", "policy"), ("space", "space"), ("venue", "venue"),
                          ("fingerprint", "in-sample bars"), ("objective", "objective"),
                          ("direction", "direction"), ("min_trades", "min_trades"),
                          ("risk_free", "risk-free rate")):
            if theirs[key] != mine[key]:
                raise ProbatumValueError(
                    f"null: its {said}, {theirs[key]!r}, is not the search's, "
                    f"{mine[key]!r}; give a random search of the same policy over the "
                    f"same space, venue and in-sample bars, scored alike")


    def _median_path(self, scores, done):
        """The median path's interval, the lower of the two middle paths for an even
        count, on its returns, with the reason where it has none."""
        objective, given = self.provenance["objective"], self.provenance
        ordered = np.flatnonzero(done)[np.argsort(scores[done], kind="stable")]
        path    = int(ordered[(len(ordered) - 1) // 2])

        if self.returns is None:
            return "no path's interval: the returns were not kept"
        if objective in TRADES:
            return (f"no path's interval: {objective} reads round trips, which a path's "
                    f"returns do not hold")
        if objective != "sharpe" and objective not in RETURNS:
            return f"no path's interval: {objective} has none"

        returns = self.returns[path][~np.isnan(self.returns[path])]
        rate, risk_free = given["periods_per_year"], given["risk_free"]

        try:
            held = (sharpe_interval(returns, periods_per_year=rate, risk_free=risk_free)
                    if objective == "sharpe" else
                    bootstrap(returns, objective, periods_per_year=rate,
                              risk_free=risk_free))
        except ProbatumValueError as error:
            return f"no path's interval: {error}"

        what = ("middle {} of resampled histories" if held.kind == "range"
                else "{} interval").format(_percent(held.level))
        return (f"the median path's {what} {_bounds(held, _signed(objective))}, and the "
                f"paths share bars")


    def _costs(self, paths, done):
        """The complete paths' costs as a share of their gross edge, the net profit
        and the costs summed."""
        balance = self._ran.venue._balance
        opening = sum(balance.values()) if isinstance(balance, dict) else balance
        paid    = sum(float(np.asarray(paths[name], dtype=np.float64)[done].sum())
                      for name in COSTS)

        net = float((np.asarray(paths["net_profit_pct"], dtype=np.float64)[done]
                     * opening / 100.0).sum())
        if net + paid <= 0.0:
            return f"costs {_money(paid)}, with no gross edge"
        return f"took {paid / (net + paid):.0%} of the gross edge"


    def __repr__(self):
        given, ran = self.provenance, self._ran

        if self._tables is None:
            best = _signed(given["objective"]) * self.best_score + 0.0
            what = (f"{given['ran']:,} trials of {given['policy']} by "
                    f"{given['sampler']['name']}, best {given['objective']} "
                    f"{best:,.2f} over {ran.cut:,} bars")
        else:
            paths, fits = self._tables["paths"], len(self._tables["fits"])
            failed = int((paths["state"] == "failed").sum())
            scores = paths["score"][~np.isnan(paths["score"])]
            median = (_signed(given["objective"]) * float(np.median(scores)) + 0.0
                      if len(scores) else math.nan)
            counted = (f"{len(paths):,} path{'s' * (len(paths) != 1)}"
                       + (f", {failed:,} failed," if failed else ""))
            what = (f"{counted} of {given['policy']} from {fits:,} "
                    f"fit{'s' * (fits != 1)} by {given['cv']['given']}, stability "
                    f"evidence, median {given['objective']} {median:,.2f} over "
                    f"{given['cv']['bars']:,} bars")

        held = len(ran.data) - ran.cut
        if not held:
            return f"Study({what}; nothing held out)"

        read = ("not yet read" if not self.looks else "read once" if self.looks == 1
                else f"read {self.looks} times")
        return f"Study({what}; {held:,} held out, {read})"
