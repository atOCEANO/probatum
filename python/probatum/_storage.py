"""A Study on disk: written into a directory of open files as a Result is, and read
back by its format (ADR 0046; Specification, Saving and resuming)."""

import json
import math
import os
import pathlib
import shutil

import numpy as np

from . import _ipc
from ._errors import ProbatumError, ProbatumValueError
from ._probatum import Table as _Held
from ._result import (COUNTS, STATS, STORED, Result, Table, _array, _constant, _read,
                      _stored, _version)
from ._search import Bars, Study, _Trial, _winner
from ._space import Categorical, Float, Int, _one
from .stats import Forecast

# the format a Study is written in, which load reads by
FORMAT = 1
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
# the manifests a directory may hold: a Study's and the one a save set aside
MANIFESTS = ("study.json", "study.json.old")


def save(study, path):
    """Write ``study`` into the directory ``path``, made where it is missing, each file
    under another name first and the manifest last (Specification, Saving and
    resuming)."""
    folder = pathlib.Path(path)
    if folder.exists() and not folder.is_dir():
        raise ProbatumValueError(f"path: {str(path)!r} is a file; name a directory")
    _ours(path, folder)

    best   = study.best
    tables = {"trials": study._trials} if study._tables is None else study._tables
    files  = {f"{name}.arrow": _stored(table) for name, table in tables.items()}
    chosen = _chosen(study)
    if chosen:
        files["choices.arrow"] = chosen

    results = dict(best=best) | ({} if study.holdout is None
                                 else dict(holdout=study.holdout))
    written  = [*files, *(["returns.npy"] if study.returns is not None else [])]
    manifest = _manifested(study)

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
    # that still names a Study's
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

    if kind == "trials" and any(results["best"].stats[key] != study._winner.stats[key]
                                for key in STATS):
        raise ProbatumValueError(f"path: {str(path)!r}'s best is not the run of the "
                                 f"winner its trials hold; {fix}")

    study._looks, study._best = manifest["looks"], results["best"]
    study._holdout = results.get("holdout")
    return study


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
        return _Trial(0, {}, "complete", float(best.stats["sharpe"]), best.stats)

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
        provenance = _plain(study.provenance),
        looks      = study.looks,
        winner     = study._winner.number,
        worker     = study._winner.worker,
        holdout    = study.holdout is not None,
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
            and (looks or not manifest["holdout"])
            and (manifest["kind"] == "paths" or isinstance(space, dict)))


def _ours(path, folder):
    """Refuse to save into ``folder``, the directory ``path`` names, where it holds
    files and no manifest, any of which save may delete or overwrite, or a Study's
    manifest save did not write."""
    names = _files(folder)
    if names and not names & set(MANIFESTS):
        raise ProbatumValueError(f"path: {str(path)!r} holds files that are no "
                                 f"Study's; give an empty directory, or one a Study "
                                 f"was saved into")

    for name in ("study.json", "study.json.old"):
        if name in names and not _manifest(_json(folder / name)):
            raise ProbatumValueError(f"path: {str(path)!r} holds a {name} save did not "
                                     f"write; give another directory")


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
