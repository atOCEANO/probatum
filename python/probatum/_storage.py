"""A Study on disk: written into a directory of open files as a Result is, read back by
its format, and kept by a search as it runs, so the same call resumes it there (ADR
0046; Specification, Saving and resuming)."""

import json
import math
import os
import pathlib
import shutil
import warnings

import numpy as np

from . import _ipc
from ._errors import (ProbatumError, ProbatumRuntimeError, ProbatumTypeError,
                      ProbatumValueError, ProbatumWarning)
from ._probatum import Table as _Held
from ._result import (COUNTS, STATS, STORED, Result, Table, _array, _constant, _read,
                      _stored, _version)
from ._search import PACKAGE, Bars, Study, _Trial, _winner
from ._space import Categorical, Float, Int, _one
from .stats import Forecast

# the format a Study is written in, which load reads by
FORMAT = 1
# what a resumed search must be given as the search before it was, by the provenance's
# names, in the order a refusal looks for the first that differs, the seed before the
# sampler's arguments it moves; the version is held as the settings are read, before
# any of the rest, which another version may name otherwise
SETTINGS = ("version", "policy", "space", "objective", "direction", "seed", "sampler",
            "min_trades", "holdout", "keep", "risk_free", "periods_per_year",
            "fingerprint", "venue")
# what a provenance says of where the call ran, which two runs of one call can differ in
WHERE = ("packages", "python", "platform", "wheel", "n_jobs")
# the ranges a space is written with, each by its name
RANGES = {"Int": Int, "Float": Float, "Categorical": Categorical}
# the tables of each kind of Study
TABLES = {"trials": ("trials",), "paths": ("paths", "fits", "windows")}
# what load reads of the provenance of each kind of Study
READ = {
    "trials": ("keep", "space", "ran", "direction", "min_trades"),
    "paths":  ("keep",),
}
# the files a Study of either kind may write, beside its manifest and its Results
FILES = ("trials.arrow", "choices.arrow", "paths.arrow", "fits.arrow", "windows.arrow",
         "returns.npy")
# the manifests a directory may hold: a Study's, the one a save set aside, and the
# settings of the search that keeps it
MANIFESTS = ("study.json", "study.json.old", "search.json")


def save(study, path):
    """Write ``study`` into the directory ``path``, made where it is missing, each file
    under another name first and the manifest last, its looks the most of its own and
    of those counted there of its search; and count each look it takes from then on
    there too (Specification, Saving and resuming)."""
    folder = pathlib.Path(path)
    if folder.exists() and not folder.is_dir():
        raise ProbatumValueError(f"path: {str(path)!r} is a file; name a directory")

    _ours(path, folder, study)

    best   = study.best
    tables = {"trials": study._trials} if study._tables is None else study._tables
    files  = {f"{name}.arrow": _stored(table) for name, table in tables.items()}
    chosen = _chosen(study)
    if chosen:
        files["choices.arrow"] = chosen

    results = dict(best=best) | ({} if study.holdout is None
                                 else dict(holdout=study.holdout))
    written      = [*files, *(["returns.npy"] if study.returns is not None else [])]
    study._looks = max(study._looks, _counted(folder, study))
    manifest     = _manifested(study)

    folder.mkdir(parents=True, exist_ok=True)
    try:
        for name, columns in files.items():
            _ipc.write(folder / f"{name}.partial", columns, {})
        if study.returns is not None:
            with open(folder / "returns.npy.partial", "wb") as file:
                np.save(file, np.ascontiguousarray(study.returns), allow_pickle=False)
        for name, result in results.items():
            shutil.rmtree(folder / f"{name}.partial", ignore_errors=True)
            result.save(folder / f"{name}.partial")
        with open(folder / "study.json.partial", "w", encoding="utf-8") as file:
            json.dump(manifest, file, allow_nan=False, indent=1)
    except BaseException:
        for name in [*written, "study.json"]:
            (folder / f"{name}.partial").unlink(missing_ok=True)
        for name in results:
            shutil.rmtree(folder / f"{name}.partial", ignore_errors=True)
        raise

    # the manifest there before, or this one where there is none, is set aside until
    # this one is in place, so a save stopped between leaves a directory load refuses
    # that still names a Study's, and its looks
    if (folder / "study.json").is_file():
        os.replace(folder / "study.json", folder / "study.json.old")
    else:
        _wrote(folder / "study.json.old", manifest)

    for name in FILES:
        if name not in written:
            (folder / name).unlink(missing_ok=True)
    for name in ("best", "holdout"):
        shutil.rmtree(folder / name, ignore_errors=True)

    for name in [*written, *results]:
        os.replace(folder / f"{name}.partial", folder / name)
    os.replace(folder / "study.json.partial", folder / "study.json")
    (folder / "study.json.old").unlink(missing_ok=True)

    _mirrored(folder, study)
    if folder.resolve() not in study._bound:
        study._bound.append(folder.resolve())


def load(path):
    """The Study ``save`` wrote into the directory ``path``, read by its format,
    refused where it holds no manifest, one of a format this version does not read, or
    files that do not agree with one another (Specification, Saving and resuming)."""
    folder   = pathlib.Path(path)
    fix      = "give a directory a Study was saved into"
    manifest = _opened(path, folder, fix)
    kind, provenance = manifest["kind"], manifest["provenance"]

    tables  = {name: _table(path, folder, name, fix) for name in TABLES[kind]}
    returns = (_array(path, folder, "returns", fix) if provenance["keep"] == "all"
               else None)
    rows    = len(tables["trials" if kind == "trials" else "paths"])

    if returns is not None:
        if returns.dtype != np.float64 or returns.ndim != 2 or len(returns) != rows:
            raise ProbatumValueError(f"path: {str(path)!r}'s returns.npy does not hold "
                                     f"a row of returns for each row of its table; "
                                     f"{fix}")
        returns.flags.writeable = False

    bars    = Bars(**manifest["bars"])
    results = {name: _result(path, folder, name, fix)
               for name in ("best", "holdout") if name == "best" or manifest["holdout"]}

    if kind == "trials":
        study = _searched(path, folder, manifest, tables["trials"], returns, bars, fix)
    else:
        study = Study(
            None,
            returns,
            _validated(provenance, results["best"]),
            None,
            provenance,
            tables,
            forecast = _forecast(manifest["forecast"]),
            bars     = bars,
        )

    if kind == "trials":
        # the engine's values, which its trials hold, where the record withholds some
        ranked = results["best"].stats._values
        if any(ranked[key] != study._winner.stats[key] for key in STATS):
            raise ProbatumValueError(f"path: {str(path)!r}'s best is not the run of the "
                                     f"winner its trials hold; {fix}")

    study._looks, study._best = manifest["looks"], results["best"]
    study._holdout = results.get("holdout")
    study._spent, study._broken = manifest["spent"], manifest["broken"]
    study._id = manifest["id"]

    return study


class Storage:
    """The directory a search keeps its Study in as it runs: what a search before this
    one kept there, each of its trials taken in its turn, and each trial this one runs,
    written there as it arrives (Specification, Saving and resuming)."""


    def __init__(self, path):
        if not isinstance(path, (str, os.PathLike)):
            raise ProbatumTypeError(f"storage: {path!r}; give the directory a search "
                                    f"keeps its Study in, as a path")

        self.path, self.folder = str(path), pathlib.Path(path)
        if self.folder.exists() and not self.folder.is_dir():
            raise ProbatumValueError(f"storage: {self.path!r} is a file; name a "
                                     f"directory")

        # what a search before this one kept: its settings, the workers a TPE search
        # asked its rounds of, its space, its Study's manifest, the looks counted at
        # its held-out bars, and each trial by number
        self.settings = self.workers = self.space = self.manifest = None
        self.looks, self.trials = 0, {}

        if (self.folder / "study.json").is_file():
            try:
                self._studied()
            except ProbatumValueError as error:
                said = str(error)
                # a saved Study's refusal, which names the directory as its path
                if said.startswith("path:"):
                    said = said.replace("path:", "storage:", 1)
                raise ProbatumValueError(said) from None

        if (self.folder / "search.json").is_file():
            self._started()
        elif self.settings is None and (self.folder / "study.json.old").is_file():
            raise ProbatumValueError(f"storage: {self.path!r} holds a Study whose save "
                                     f"stopped part way; save it there again, or "
                                     f"search into another directory")
        elif self.settings is None and _files(self.folder):
            raise ProbatumValueError(f"storage: {self.path!r} holds files that are no "
                                     f"search's; give an empty directory, or the one a "
                                     f"search kept its Study in")

        self.seed = None if self.settings is None else self.settings["seed"]


    def _studied(self):
        """Read the Study a search kept here as it ended."""
        fix      = "give the directory a search kept its Study in, or an empty one"
        manifest = _opened(self.path, self.folder, fix)
        if manifest["kind"] != "trials":
            raise ProbatumValueError(f"path: {self.path!r} holds a validation's Study, "
                                     f"which a search does not resume; give another "
                                     f"directory")

        provenance = manifest["provenance"]
        if "version" in provenance:
            self._versioned(provenance["version"])

        try:
            held = _search(provenance)
        except (KeyError, TypeError, ValueError):
            held = None

        if held is None or not _settled(held):
            raise ProbatumValueError(f"path: {self.path!r}'s study.json is not a "
                                     f"manifest save writes; {fix}")

        table = _table(self.path, self.folder, "trials", fix)
        space, made, _ = _made(self.path, self.folder, manifest, table, fix)
        returns = (_array(self.path, self.folder, "returns", fix)
                   if provenance["keep"] == "all" else None)

        if returns is not None and (returns.ndim != 2 or len(returns) != len(made)):
            raise ProbatumValueError(f"path: {self.path!r}'s returns.npy does not hold "
                                     f"a row of returns for each row of its table; "
                                     f"{fix}")

        broken = set(manifest["broken"])
        for k, trial in enumerate(made):
            if returns is not None and trial.state in ("complete", "floored"):
                trial.returns = returns[k]
            # a trial the workers' breaking failed is run again
            if trial.number not in broken:
                self.trials[trial.number] = trial

        self.settings = {key: held[key] for key in SETTINGS}
        self.workers, self.space, self.manifest = held["workers"], space, manifest
        self.looks = manifest["looks"]


    def _started(self):
        """Read the settings a search wrote here as it started, held to the Study's
        where both are written, and each batch of trials it wrote as they arrived."""
        fix      = "give the directory a search kept its Study in, or an empty one"
        started  = _json(self.folder / "search.json")
        settings = started.get("settings") if isinstance(started, dict) else None
        if isinstance(settings, dict) and "version" in settings:
            self._versioned(settings["version"])

        if not _begun(started):
            raise ProbatumValueError(f"storage: {self.path!r}'s search.json is not the "
                                     f"settings a search writes; {fix}")

        if self.settings is None:
            self.settings, self.workers = started["settings"], started["workers"]
            self.space = _restored(started["space"])
        elif {**self.settings, "workers": self.workers} != _begun(started):
            raise ProbatumValueError(f"storage: {self.path!r} holds the Study of one "
                                     f"search and the settings of another; give "
                                     f"another directory")

        self.looks = max(self.looks, started["looks"])

        for batch in sorted((self.folder / "trials").glob("*.arrow")):
            try:
                trials = _batched(batch, self.space)
            except (_ipc.Refused, KeyError, IndexError, TypeError, ValueError):
                raise ProbatumValueError(f"storage: {self.path!r}'s "
                                         f"trials/{batch.name} is not a batch of "
                                         f"trials a search writes; {fix}") from None

            for trial in trials:
                self.trials.setdefault(trial.number, trial)


    def _versioned(self, version):
        """Refuse the search kept here where another version of probatum ran it, whose
        settings may hold other names as well as other values."""
        if version != _version():
            raise ProbatumValueError(f"storage: {self.path!r} holds a search probatum "
                                     f"{version} ran, and this is probatum "
                                     f"{_version()}; resume it with that version, or "
                                     f"search into another directory")


    def start(self, provenance, space, count):
        """Hold this search, of ``provenance`` over ``space`` asking ``count`` trials,
        to the one before it where there was one, refusing one that differs, and write
        its settings where none are written."""
        held    = _search(provenance)
        given   = {key: held[key] for key in SETTINGS}
        workers = held["workers"]

        if self.settings is not None:
            for key in SETTINGS:
                if self.settings[key] != given[key]:
                    raise ProbatumValueError(self._differs(key, given))

            if self.workers != workers:
                raise ProbatumValueError(f"n_jobs: {provenance['n_jobs']['given']!r}, "
                                         f"which runs {workers:,} at a time, and the "
                                         f"TPE search in {self.path!r} asked its "
                                         f"trials {self.workers:,} at a time; resume "
                                         f"it at that many, as TPE repeats only at the "
                                         f"number of workers it ran at, or search into "
                                         f"another directory")

            last = max(self.trials, default=-1)
            if last >= count:
                raise ProbatumValueError(f"trials: {provenance['trials']:,}, and "
                                         f"{self.path!r} holds this search's trials up "
                                         f"to trial {last:,}; ask for at least "
                                         f"{last + 1:,}")

        self.settings, self.workers, self.space = given, workers, space
        if not (self.folder / "search.json").is_file():
            self.folder.mkdir(parents=True, exist_ok=True)
            _wrote(self.folder / "search.json",
                   dict(format=FORMAT, version=_version(), settings=given,
                        workers=workers, space=_spaced(space), looks=self.looks))


    def _differs(self, key, given):
        """The refusal of a search whose ``key`` differs from the stored one, named by
        what the caller gave."""
        stored, now = self.settings[key], given[key]

        if key in ("fingerprint", "periods_per_year"):
            return (f"storage: {self.path!r} holds a search on other bars than this "
                    f"one's data holds before its holdout; resume it on the data it "
                    f"ran on, or search into another directory")

        if key == "holdout" and stored["given"] == now["given"]:
            if stored["rows"] != now["rows"]:
                return (f"storage: {self.path!r} holds a search that held out "
                        f"{stored['rows']:,} rows, and this one's data holds "
                        f"{now['rows']:,} there; resume it on the data it ran on, or "
                        f"search into another directory")
            return (f"storage: {self.path!r} holds a search whose held-out bars start "
                    f"at {stored['first']}, and this one's start at {now['first']}; "
                    f"resume it on the data it ran on, or search into another "
                    f"directory")

        # the seed and the version are held first, so TPE's arguments differ only in
        # the Optuna that runs them
        if (key == "sampler" and stored["name"] == now["name"] == "tpe"
                and stored.get("optuna") != now.get("optuna")):
            return (f"storage: {self.path!r} holds a TPE search Optuna "
                    f"{stored.get('optuna')} ran, and this is Optuna {now['optuna']}; "
                    f"resume it with that Optuna, or search into another directory")

        if key == "holdout":
            stored, now = stored["given"], now["given"]
        elif key == "sampler" and stored["name"] != now["name"]:
            stored, now = stored["name"], now["name"]

        return (f"storage: {self.path!r} holds a search whose {key} is {stored!r}, and "
                f"this one's is {now!r}; search with the {key} it ran with to resume "
                f"it, or into another directory")


    def again(self, runner):
        """Run the lowest-numbered trial kept here that ran again on ``runner``,
        refusing a policy that runs it otherwise now."""
        ran = [trial for trial in self.trials.values() if trial.stats is not None]
        if not ran:
            return

        kept    = min(ran, key=lambda trial: trial.number)
        again   = runner.trial(kept.number, dict(kept.params))
        changed = (["state"] if again.stats is None
                   else [key for key in STATS if again.stats[key] != kept.stats[key]])
        if not changed:
            return

        key = changed[0]
        was, now = ((kept.state, f"failed, {again.error}") if again.stats is None
                    else (kept.stats[key], again.stats[key]))
        why = ("it ran in a worker, which imported the policy afresh under a hash seed "
               "of its own, so give the policy what it reads as its arguments, or set "
               "PYTHONHASHSEED" if kept.worker else
               "the policy's code or the globals it reads changed since it ran, so "
               "search into another directory")

        raise ProbatumRuntimeError(f"storage: trial {kept.number}'s {key} was {was!r} "
                                   f"in {self.path!r}, and run again it is {now!r}; "
                                   f"{why}")


    def taken(self, asked):
        """The trials kept here of ``asked``, a number and its parameters each, by
        number, refused where one ran with parameters other than this search asks."""
        kept = {}
        for number, params in asked:
            trial = self.trials.get(number)
            if trial is None:
                continue

            if any(not _one(trial.params[name], value)
                   for name, value in params.items()):
                raise ProbatumValueError(f"storage: trial {number} in {self.path!r} "
                                         f"ran with {trial.params}, and this search "
                                         f"asks {params} for it, so it is not the "
                                         f"search that ran there; search into another "
                                         f"directory")

            kept[number] = trial

        return kept


    def write(self, trials):
        """Write those of ``trials`` not kept here as a batch, their returns where the
        search keeps them and then their rows, each under another name first, so a
        batch whose rows are written is whole."""
        new = [trial for trial in trials if trial.number not in self.trials]
        if not new:
            return

        folder = self.folder / "trials"
        folder.mkdir(parents=True, exist_ok=True)

        name = f"{new[0].number:08d}-{new[-1].number:08d}"
        kept = [trial.returns for trial in new if trial.returns is not None]

        if kept and self.settings["keep"] == "all":
            rows = np.full((len(new), len(kept[0])), np.nan)
            for k, trial in enumerate(new):
                if trial.returns is not None:
                    rows[k] = trial.returns
            with open(folder / f"{name}.npy.partial", "wb") as file:
                np.save(file, rows, allow_pickle=False)
            os.replace(folder / f"{name}.npy.partial", folder / f"{name}.npy")

        _ipc.write(folder / f"{name}.arrow.partial", _batch(new, self.space), {})
        os.replace(folder / f"{name}.arrow.partial", folder / f"{name}.arrow")


    def wrote(self, trial):
        """Write ``trial``, which a worker gave back, as a batch of its own."""
        self.write([trial])


    def carry(self, study):
        """Hold ``study`` to what was kept here: the looks counted at its held-out bars,
        its holdout where its winner and those bars are the ones it ran on, and every
        look it takes counted here; and let go of the trials kept, which ``study`` now
        holds."""
        study._looks = max(study._looks, self.looks)
        manifest     = self.manifest

        if (manifest is not None and manifest["holdout"]
                and manifest["winner"] == study._winner.number):
            held = study._ran.data.split(study._ran.cut)[1]
            try:
                holdout = Result.load(self.folder / "holdout")
            except ProbatumValueError:
                # a holdout a write stopped part way left unreadable, which the looks
                # still count
                holdout = None

            # a holdout run on bars since revised is no run on these
            if (holdout is not None
                    and holdout.provenance["data"]["fingerprint"] == held.fingerprint):
                study._holdout = holdout

        study._bound.append(self.folder.resolve())
        self.trials, self.manifest = {}, None


    def finish(self, study):
        """Keep ``study`` here as ``save`` writes it, and remove the batches it holds;
        a Study whose winner runs otherwise than its trial is not kept, with a warning,
        so its trials are kept as they ran."""
        try:
            study.best
        except Exception as error:
            # the Study is returned as a search with no storage returns it, its best
            # raising when read
            warnings.warn(f"storage: {self.path!r} keeps this search's trials but not "
                          f"its Study, since {error}", ProbatumWarning,
                          skip_file_prefixes=PACKAGE)
            return

        save(study, self.folder)
        shutil.rmtree(self.folder / "trials", ignore_errors=True)


def looked(folder, study):
    """Count a look at ``study``'s held-out bars in the directory ``folder``: one more
    than the most counted there, in each manifest there of its search, written under
    another name first, the count given; 0 where it holds no manifest of its search."""
    held, found = _search(study.provenance), []
    for name in MANIFESTS:
        kept = _json(folder / name)
        if kept is not None and _of(name, kept) == held:
            found.append((name, kept))

    if not found:
        return 0

    looks = 1 + max(kept["looks"] for _, kept in found)
    for name, kept in found:
        _wrote(folder / name, dict(kept, looks=looks))

    return looks


def held(folder, study):
    """Keep ``study``'s holdout in the directory ``folder`` once it ran, where the
    directory still holds this Study as it was saved, by the id its manifest names: the
    Result, then the manifest that names it."""
    kept = _json(folder / "study.json")
    if not _manifest(kept) or kept["id"] != study._id:
        return

    shutil.rmtree(folder / "holdout.partial", ignore_errors=True)
    study.holdout.save(folder / "holdout.partial")

    # the manifest stops naming the holdout there before it is taken away, so a write
    # stopped between leaves no manifest that names one it does not hold
    if kept["holdout"]:
        kept = dict(kept, holdout=False)
        _wrote(folder / "study.json", kept)

    shutil.rmtree(folder / "holdout", ignore_errors=True)
    os.replace(folder / "holdout.partial", folder / "holdout")

    _wrote(folder / "study.json", dict(kept, holdout=True,
                                       looks=max(kept["looks"], study.looks)))


def _searched(path, folder, manifest, table, returns, bars, fix):
    """A search's Study from its trials table, each trial's choices and the space its
    manifest holds, refused where they do not agree with one another."""
    space, made, winner = _made(path, folder, manifest, table, fix)
    return Study(
        table,
        returns,
        winner,
        None,
        manifest["provenance"],
        space  = space,
        params = [trial.params for trial in made],
        bars   = bars,
    )


def _made(path, folder, manifest, table, fix):
    """The space a search's manifest holds, each trial its trials table and their
    choices hold, and its winner, refused where they do not agree with one another."""
    provenance = manifest["provenance"]
    space      = _restored(manifest["space"])
    shown      = {name: repr(given) for name, given in space.items()}

    if shown != provenance["space"] or not {"number", "state", "score", "error", *STATS,
                                           *space} <= set(table.columns):
        raise ProbatumValueError(f"path: {str(path)!r}'s trials.arrow does not hold "
                                 f"the columns of the space its study.json holds; "
                                 f"{fix}")

    named  = [name for name, given in space.items() if isinstance(given, Categorical)]
    chosen = _read(path, folder, "choices.arrow", fix) if named else {}

    values = {}
    for name in space:
        if name not in named:
            values[name] = table[name].tolist()
        else:
            values[name] = (chosen[name][1].tolist() if name in chosen
                            and len(chosen[name][1]) == len(table)
                            else [None] * len(table))

    states, scores = table["state"].tolist(), table["score"].tolist()
    errors = table["error"].tolist()
    stats  = {key: table[key].tolist() for key in STATS}
    made   = []

    for k, number in enumerate(table["number"].tolist()):
        params = {}
        for name, given in space.items():
            value = values[name][k]
            if isinstance(given, Categorical):
                if not (type(value) is int and 0 <= value < len(given.choices)):
                    raise ProbatumValueError(f"path: {str(path)!r}'s choices.arrow "
                                             f"does not hold an index among {name}'s "
                                             f"choices for each trial; {fix}")
                value = given.choices[value]
            params[name] = value

        ran = stats[COUNTS[0]][k] is not None
        made.append(_Trial(
            number,
            params,
            states[k],
            scores[k] if states[k] == "complete" else None,
            {key: stats[key][k] for key in STATS} if ran else None,
            error = errors[k],
        ))

    numbers = [trial.number for trial in made]
    if (numbers != list(range(len(made))) or provenance["ran"] != len(made)
            or any(trial.state not in ("complete", "floored", "failed")
                   for trial in made)):
        raise ProbatumValueError(f"path: {str(path)!r}'s trials.arrow is not the "
                                 f"trials of the search its study.json holds; {fix}")

    winner = _winner(made, provenance["direction"], provenance["min_trades"])
    if winner.number != manifest["winner"]:
        raise ProbatumValueError(f"path: {str(path)!r}'s study.json names trial "
                                 f"{manifest['winner']} its winner, and its trials "
                                 f"make trial {winner.number} the winner; {fix}")

    winner.worker = manifest["worker"]

    return space, made, winner


def _validated(provenance, best):
    """A validation's winner, as its Study holds it: its first search's, or a model's
    final fit with no parameters and its best's Sharpe as its score."""
    if "model" in provenance:
        return _Trial(0, {}, "complete", float(best.stats._values["sharpe"]), best.stats)

    final = provenance["final"]
    return _Trial(final["winner"], dict(final["params"]), "complete", final["score"])


def _opened(path, folder, fix):
    """The manifest of the Study ``save`` wrote into ``folder``, the directory ``path``
    names, its provenance and its space read back as they were, refused where there is
    none, it is of another format or it is not one save writes."""
    if not (folder / "study.json").is_file() and (folder / "study.json.old").is_file():
        raise ProbatumValueError(f"path: {str(path)!r} holds a Study whose save "
                                 f"stopped part way; save it there again, or give "
                                 f"another directory")

    if not (folder / "study.json").is_file():
        raise ProbatumValueError(f"path: {str(path)!r} holds no study.json; {fix}")

    manifest = _json(folder / "study.json")
    given    = manifest.get("format") if isinstance(manifest, dict) else None
    if type(given) is int and given != FORMAT:
        raise ProbatumValueError(f"path: {str(path)!r} holds a Study saved in format "
                                 f"{given} by probatum {manifest.get('version')}, and "
                                 f"this version, {_version()}, reads format {FORMAT}; "
                                 f"load it with the version that saved it, or a later "
                                 f"one")

    try:
        if _manifest(manifest):
            manifest["provenance"] = _given(manifest["provenance"])
            if manifest["kind"] == "trials":
                _restored(manifest["space"])
            return manifest
    except (ProbatumError, KeyError, TypeError, ValueError):
        pass

    raise ProbatumValueError(f"path: {str(path)!r}'s study.json is not a manifest save "
                             f"writes; {fix}")


def _manifested(study):
    """What ``study``'s manifest holds."""
    return dict(
        format     = FORMAT,
        version    = _version(),
        kind       = study.kind,
        id         = study._id,
        provenance = _plain(study.provenance),
        looks      = study.looks,
        winner     = study._winner.number,
        worker     = study._winner.worker,
        holdout    = study.holdout is not None,
        spent      = study._spent,
        broken     = list(study._broken),
        bars       = study._bars._asdict(),
        space      = _spaced(study._space),
        forecast   = _told(study._forecast),
    )


def _manifest(manifest):
    """Whether ``manifest`` holds what a Study's manifest holds."""
    if not isinstance(manifest, dict) or manifest.get("kind") not in TABLES:
        return False

    provenance, bars = manifest.get("provenance"), manifest.get("bars")
    space, looks     = manifest.get("space"), manifest.get("looks")

    return (isinstance(provenance, dict) and provenance.get("kind") == manifest["kind"]
            and set(READ[manifest["kind"]]) <= set(provenance)
            and (manifest["kind"] == "trials" or "model" in provenance
                 or isinstance(provenance.get("final"), dict)
                 and {"winner", "params", "score"} <= set(provenance["final"]))
            and isinstance(bars, dict) and set(bars) == set(Bars._fields)
            and all(type(bars[name]) is int
                    for name in ("rows", "cut", "first", "last"))
            and type(looks) is int and looks >= 0
            and type(manifest.get("winner")) is int
            and isinstance(manifest.get("worker"), bool)
            and isinstance(manifest.get("holdout"), bool)
            and isinstance(manifest.get("spent"), bool)
            and (looks or not (manifest["holdout"] or manifest["spent"]))
            and isinstance(manifest.get("broken"), list)
            and all(type(number) is int for number in manifest["broken"])
            and isinstance(manifest.get("id"), str)
            and (manifest["kind"] == "paths" or isinstance(space, dict)))


def _search(provenance):
    """What makes the search a Study of ``provenance`` read the one it is, as JSON
    holds it: a search's settings, with the workers where TPE asked its trials in
    rounds of them, and a validation's provenance but where it ran."""
    if provenance["kind"] == "trials":
        held = {key: provenance[key] for key in SETTINGS}
        held["workers"] = (provenance["n_jobs"]["resolved"]
                           if provenance["sampler"]["name"] == "tpe" else None)
    else:
        held = {key: value for key, value in provenance.items() if key not in WHERE}

    return json.loads(json.dumps(_plain(held), allow_nan=False))


def _of(name, kept):
    """The search the manifest ``kept``, a study.json or a search.json by ``name``,
    holds, as `_search` gives it, None where it holds none."""
    try:
        if name == "search.json":
            return _begun(kept)
        if _manifest(kept):
            return _search(_given(kept["provenance"]))
    except (KeyError, TypeError, ValueError):
        pass

    return None


def _begun(started):
    """The search the settings a search wrote as it started hold, with its workers,
    None where they are not the settings a search writes."""
    if not (isinstance(started, dict) and started.get("format") == FORMAT
            and isinstance(started.get("settings"), dict)
            and set(started["settings"]) == set(SETTINGS)
            and _settled(started["settings"])
            and isinstance(started.get("space"), dict)
            and type(started.get("looks")) is int and started["looks"] >= 0
            and (type(started.get("workers")) is int
                 if started["settings"]["sampler"]["name"] == "tpe"
                 else started.get("workers", "") is None)):
        return None

    try:
        _restored(started["space"])
    except (ProbatumError, KeyError, TypeError, ValueError):
        return None

    return {**started["settings"], "workers": started["workers"]}


def _settled(settings):
    """Whether a search's ``settings`` hold its holdout, its sampler and its seed in
    the form a search writes them."""
    holdout, sampler = settings.get("holdout"), settings.get("sampler")
    return (isinstance(holdout, dict) and {"given", "rows", "first"} <= set(holdout)
            and type(holdout["rows"]) is int
            and isinstance(sampler, dict) and isinstance(sampler.get("name"), str)
            and type(settings.get("seed")) is int)


def _counted(folder, study):
    """The most looks the manifests in ``folder`` of ``study``'s search have counted,
    0 where it holds none."""
    held  = _search(study.provenance)
    found = [(name, _json(folder / name)) for name in MANIFESTS]

    return max([kept["looks"] for name, kept in found
                if kept is not None and _of(name, kept) == held] + [0])


def _mirrored(folder, study):
    """Write ``study``'s looks into the settings of its search in ``folder``, where a
    search keeps it, so they outlast a save that is stopped before its manifest is in
    place."""
    started = _json(folder / "search.json")
    if started is not None and _begun(started) == _search(study.provenance):
        _wrote(folder / "search.json",
               dict(started, looks=max(started["looks"], study.looks)))


def _ours(path, folder, study):
    """Refuse to save ``study`` into ``folder``, the directory ``path`` names, where it
    holds files and no manifest, any of which save may delete or overwrite; a Study's
    manifest save did not write; or the storage of another search, which a Study saved
    there would break."""
    names = _files(folder)
    if names and not names & set(MANIFESTS):
        raise ProbatumValueError(f"path: {str(path)!r} holds files that are no "
                                 f"Study's; give an empty directory, or one a Study "
                                 f"was saved into")

    for name in ("study.json", "study.json.old"):
        if name in names and not _manifest(_json(folder / name)):
            raise ProbatumValueError(f"path: {str(path)!r} holds a {name} save did not "
                                     f"write; give another directory")

    if "search.json" in names:
        kept = _of("search.json", _json(folder / "search.json"))
        if kept != _search(study.provenance):
            raise ProbatumValueError(f"path: {str(path)!r} holds the storage of "
                                     f"another search, which this Study saved there "
                                     f"would break; give another directory")


def _files(folder):
    """The names ``folder`` holds, but what a write stopped part way left under
    another name."""
    if not folder.is_dir():
        return set()
    return {child.name for child in folder.iterdir()
            if not child.name.endswith(".partial")}


def _json(file):
    """The JSON ``file`` holds, None where it is missing or holds none."""
    try:
        with open(file, encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_constant)
    except (OSError, ValueError):
        return None


def _wrote(file, held):
    """Write ``held`` as the JSON of ``file``, under another name first."""
    written = file.with_name(f"{file.name}.partial")
    with open(written, "w", encoding="utf-8") as handle:
        json.dump(held, handle, allow_nan=False, indent=1)

    os.replace(written, file)


def _table(path, folder, name, fix):
    """The table ``name``.arrow of ``folder``, the directory ``path`` names, its
    columns as they were written."""
    columns = _read(path, folder, f"{name}.arrow", fix)
    kinds   = {stored: kind for kind, stored in STORED.items()}
    parts   = []

    for column, (stored, values, present, nullable) in columns.items():
        if stored not in kinds:
            raise ProbatumValueError(f"path: {str(path)!r}'s {name}.arrow holds a "
                                     f"column {column} of a kind save does not write; "
                                     f"{fix}")

        data = None
        if stored == "large_text":
            values, data = values
        parts.append((column, kinds[stored], nullable, values,
                      None if present.all() else present, data))

    try:
        return Table(_Held(parts))
    except ValueError as error:
        raise ProbatumValueError(f"path: {str(path)!r}'s {name}.arrow is not a table "
                                 f"save writes; {fix}") from error


def _result(path, folder, name, fix):
    """The Result ``save`` wrote into ``name`` of ``folder``."""
    if not (folder / name / "result.json").is_file():
        raise ProbatumValueError(f"path: {str(path)!r} holds no {name}, the winner's "
                                 f"Result; {fix}")

    return Result.load(folder / name)


def _chosen(study):
    """Each trial's choice of each Categorical of a search's space, as its index among
    the choices, as `_ipc.write` takes columns."""
    if study._space is None:
        return []
    return [(name, "int64", np.array([next(k for k, choice in enumerate(given.choices)
                                           if _one(choice, params[name]))
                                      for params in study._params], dtype=np.int64),
             None, False)
            for name, given in study._space.items() if isinstance(given, Categorical)]


def _told(forecast):
    """A model's pooled Forecast as the manifest holds it: its fields, or why it has
    none."""
    if isinstance(forecast, Forecast):
        return dict(zip(Forecast.__slots__, _plain(list(forecast._args()))))
    return forecast


def _forecast(told):
    """The pooled Forecast the manifest holds, or why it has none."""
    if isinstance(told, dict):
        return Forecast(*_given([told[name] for name in Forecast.__slots__]))
    return told


def _spaced(space):
    """A search's ``space`` as a manifest holds it, each range by its name and what it
    was built with, None for a validation's Study, which holds none."""
    if space is None:
        return None
    return {name: [type(given).__name__, _plain(list(given._args()))]
            for name, given in space.items()}


def _restored(spaced):
    """The space `_spaced` wrote."""
    return {name: RANGES[form](*_given(args)) for name, (form, args) in spaced.items()}


def _batch(trials, space):
    """``trials`` as the columns of a batch, as `_ipc.write` takes them: each one's
    number, state, score, error and trace, whether it ran in a worker, each parameter
    of ``space`` as it ran, a Categorical's as its index among the choices, and the
    statistics of each one that ran."""
    ran     = np.array([trial.stats is not None for trial in trials])
    errors  = [trial.error for trial in trials]
    traces  = [trial.trace for trial in trials]
    columns = [
        ("number", "int64",
         np.array([trial.number for trial in trials], dtype=np.int64), None, False),
        ("state", "text", [trial.state for trial in trials], None, False),
        ("score", "float", np.array([math.nan if trial.score is None else trial.score
                                     for trial in trials]), None, False),
        ("error", "text", errors, np.array([error is not None for error in errors]),
         True),
        ("trace", "text", traces, np.array([trace is not None for trace in traces]),
         True),
        ("worker", "bool", np.array([trial.worker for trial in trials]), None, False),
    ]

    # a parameter's column is named apart from the batch's own, which no name it is
    # given can be
    for name, given in space.items():
        values = [trial.params[name] for trial in trials]
        if isinstance(given, Categorical):
            values = [next(k for k, choice in enumerate(given.choices)
                           if _one(choice, value)) for value in values]
        kind = "float" if isinstance(given, Float) else "int64"
        columns.append((f"param {name}", kind,
                        np.array(values, dtype=np.float64 if kind == "float"
                                 else np.int64), None, False))

    for key in STATS:
        counted = key in COUNTS
        values  = [trial.stats[key] if trial.stats is not None else 0 if counted
                   else math.nan for trial in trials]
        columns.append((key, "int64" if counted else "float",
                        np.array(values, dtype=np.int64 if counted else np.float64),
                        ran if counted else None, counted))

    return columns


def _batched(batch, space):
    """The trials the batch file ``batch`` holds over ``space``, each one's returns
    from the array beside it where the search kept them."""
    columns, _ = _ipc.read(batch)
    numbers = columns["number"][1].tolist()
    states  = columns["state"][1]
    scores  = columns["score"][1].tolist()
    errors, traces = columns["error"][1], columns["trace"][1]
    workers = columns["worker"][1].tolist()
    ran     = columns[COUNTS[0]][2].tolist()
    stats   = {key: columns[key][1].tolist() for key in STATS}
    values  = {name: columns[f"param {name}"][1].tolist() for name in space}
    rows    = (np.load(batch.with_suffix(".npy"), allow_pickle=False)
               if batch.with_suffix(".npy").is_file() else None)

    trials = []
    for k, number in enumerate(numbers):
        params = {name: given.choices[values[name][k]] if isinstance(given, Categorical)
                  else values[name][k] for name, given in space.items()}
        kept = rows is not None and states[k] in ("complete", "floored")
        trials.append(_Trial(
            number,
            params,
            states[k],
            scores[k] if states[k] == "complete" else None,
            {key: stats[key][k] for key in STATS} if ran[k] else None,
            rows[k] if kept else None,
            errors[k],
            trace  = traces[k],
            worker = workers[k],
        ))

    return trials


def _plain(value):
    """``value`` as JSON holds it, a numpy number as Python's, each float that is not
    finite as its text in a dict of its own, a tuple as a list."""
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _given(value):
    """What `_plain` wrote."""
    if isinstance(value, dict):
        if set(value) == {"float"} and value["float"] in ("nan", "inf", "-inf"):
            return float(value["float"])
        return {key: _given(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_given(item) for item in value]
    return value
