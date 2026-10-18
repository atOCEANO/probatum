"""A search's trials in workers that ``spawn`` starts (ADR 0032): the search writes what
they run from into a folder, the in-sample Data, which of its instruments are funded,
the policy, the venue and the settings, and passes them its path alone, so each worker
rebuilds the engine rather than receives one, and every result crosses back as plain
data, its warnings and errors as text."""

import concurrent.futures
import importlib
import json
import multiprocessing
import os
import pathlib
import pickle
import sys
import traceback
import warnings

from ._data import Data
from ._errors import (MissingDependency, PolicyError, ProbatumRuntimeError,
                      ProbatumWarning)
from ._search import GUARD, _named, _Runner, _Trial, _told
from ._venue import _run

# the runner each worker builds from its search's folder the first time a trial asks
LOADED = {}


def _cloudpickle():
    try:
        import cloudpickle
    except ImportError:
        raise MissingDependency("n_jobs: workers need cloudpickle to carry the policy, "
                                "which is not installed; pip install cloudpickle") from None

    return cloudpickle


def _wrote(folder, policy, venue, data, settings):
    """What the workers run from, written into ``folder``."""
    cloudpickle = _cloudpickle()

    try:
        carried = cloudpickle.dumps(policy)
    except Exception as error:
        raise PolicyError(f"policy: {_named(policy)} cannot be carried to the workers "
                          f"({_told(error)}); keep what it holds picklable, or pass "
                          f"n_jobs=1") from error

    data.save(folder / "data")
    (folder / "policy.pkl").write_bytes(carried)
    (folder / "venue.pkl").write_bytes(pickle.dumps(venue))

    funded = sorted(data._root.funded or ())
    (folder / "settings.json").write_text(json.dumps(dict(settings, funded=funded)),
                                          encoding="utf-8")


def _started(path):
    """A worker's start, marked in the folder by its process, so the search can tell a
    worker that died starting from one that died later."""
    pathlib.Path(path, f"started-{os.getpid()}").touch()


def _loaded(path):
    """The runner of the search whose folder is ``path``, built once a worker."""
    runner = LOADED.get(path)
    if runner is None:
        folder            = pathlib.Path(path)
        settings          = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        data              = Data.load(folder / "data")
        data._root.funded = frozenset(settings["funded"])
        policy            = pickle.loads((folder / "policy.pkl").read_bytes())
        venue             = pickle.loads((folder / "venue.pkl").read_bytes())
        runner = LOADED[path] = _Runner(policy, settings["kind"], data, _run(venue, data),
                                        settings["rate"], settings["risk_free"],
                                        settings["objective"], settings["min_trades"])
        runner.made, runner.keep = settings["made"], settings["keep"]

    return runner


def _trial(path, number, params):
    """Trial ``number`` with ``params`` in a worker, as plain data the search reads: its
    state, score, statistics, error, and returns where the search keeps them, the
    traceback of a failure, and the warnings it gave, each place once, as its
    category's module and name, its text, file and line and the module it came from. A
    worker that cannot load what it runs from fails the trial with why, and what the
    loading warns of stays in the worker."""
    caught = []

    try:
        runner = _loaded(path)
    except Exception as error:
        trial = _Trial(number, params, "failed", error=_told(error), raised=error)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trial = runner.trial(number, params)
        if runner.keep == "none":
            trial.returns = None

    trace = (None if trial.raised is None
             else "".join(traceback.format_exception(trial.raised)))

    # a place a run warns from on every bar is sent once, as this process shows it
    places = dict.fromkeys((given.category.__module__, given.category.__qualname__,
                            str(given.message), given.filename, given.lineno)
                           for given in caught)

    return dict(number=number, state=trial.state, score=trial.score, stats=trial.stats,
                returns=trial.returns, error=trial.error, trace=trace,
                warnings=[(*place, _module(place[3])) for place in places])


def _module(filename):
    """The name of the module whose file is ``filename``, which a filter by module
    matches as it would in the search's own process, or None."""
    for name, module in list(sys.modules.items()):
        if getattr(module, "__file__", None) == filename:
            # spawn imports the script it runs again under this name
            return "__main__" if name == "__mp_main__" else name

    return None


def _category(module, name):
    """The warning class ``name`` of ``module``, None where this process cannot import
    it."""
    try:
        found = importlib.import_module(module)
        for part in name.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError):
        return None

    return found if isinstance(found, type) and issubclass(found, Warning) else None


def _built(category, message):
    """A warning of ``category`` with ``message``, None where there is no category or
    its class cannot be built from the text alone."""
    if category is None:
        return None

    try:
        return category(message)
    except Exception:
        return None


def _reruns_main():
    """Whether spawn runs the main module again in each worker, as it does for a script
    and a module run with -m."""
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and spec.name:
        return not (spec.name == "__main__" or spec.name.endswith(".__main__"))

    return getattr(main, "__file__", None) is not None


class _Pool:
    """The workers a search's trials run in, ``workers`` of them for the ``n_jobs`` the
    caller gave, which spawn starts, each loading what the search wrote into ``folder``
    when its first trial arrives."""


    def __init__(self, folder, workers, n_jobs):
        self.folder, self.workers, self.n_jobs = folder, workers, n_jobs

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_started, initargs=(str(folder),))

        self.returned, self.broken = 0, None
        # each warning a worker gave, shown once a place, as this process shows its own
        self.registry = {}


    def run(self, asked):
        """``asked``, a number and its parameters for each trial, run in the workers and
        given back as trials in that order. Where the pool breaks before any worker
        gave back a trial, the search ends, naming the main guard where no worker came
        up and spawn runs the script again; where it breaks later, the trials it had not
        given back fail with its error."""
        broken  = (concurrent.futures.BrokenExecutor, OSError)
        futures = {}
        for number, params in asked:
            try:
                futures[number] = self.executor.submit(_trial, str(self.folder), number,
                                                       params)
            except broken as error:
                self.broken = self.broken or error
                break

        concurrent.futures.wait(futures.values())
        for future in futures.values():
            if isinstance(future.exception(), broken):
                self.broken = self.broken or future.exception()

        came = sum(future.exception() is None for future in futures.values())
        if self.broken is not None and not self.returned and not came:
            raise self.refused() from self.broken

        out = []
        for number, params in asked:
            future = futures.get(number)
            error  = self.broken if future is None else future.exception()
            if error is not None:
                out.append(_Trial(
                    number,
                    params,
                    "failed",
                    error  = _told(error),
                    worker = True,
                ))
            else:
                self.returned += 1
                out.append(self.issued(future.result(), params))

        return out


    def refused(self):
        """What a search whose workers broke raises, naming the main guard only where
        no worker came up and spawn runs the script again."""
        said = (f"n_jobs: {self.n_jobs}" if self.n_jobs == self.workers
                else f"n_jobs: {self.n_jobs} ({self.workers:,} workers)")
        came = self.returned or any(self.folder.glob("started-*"))
        if not came and _reruns_main():
            return ProbatumRuntimeError(f"{said}, and every worker died as it started, "
                                        f"as they do where the script calls search "
                                        f"outside a guard, since spawn runs its top level "
                                        f"again in each; {GUARD}, or where it is under "
                                        f"one, read what the workers printed")

        return ProbatumRuntimeError(f"{said}, and the workers died with no trial complete "
                                    f"({_told(self.broken)}); their output says why")


    def issued(self, given, params):
        """A worker's trial as the search's, its warnings given again here from the
        module they came from, under their own categories where this process can import
        them and build one from its text and as a ProbatumWarning naming them where
        not, a warning the caller's filters make an error failing the trial with it."""
        try:
            for module, name, message, filename, lineno, origin in given["warnings"]:
                warning = _built(_category(module, name), message)
                if warning is None:
                    warning = ProbatumWarning(f"{module}.{name}: {message}")
                warnings.warn_explicit(warning, type(warning), filename, lineno,
                                       module=origin, registry=self.registry)
        except Exception as error:
            return _Trial(
                given["number"],
                params,
                "failed",
                error  = _told(error),
                worker = True,
            )

        return _Trial(
            given["number"],
            params,
            given["state"],
            given["score"],
            given["stats"],
            given["returns"],
            given["error"],
            trace  = given["trace"],
            worker = True,
        )


    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
