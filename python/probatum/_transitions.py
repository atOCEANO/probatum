"""Transitions: every step an agent took, as offline reinforcement learning reads them,
one row a step of each environment, kept by an environment or an agent's backtest
recorded at ``"transitions"`` and written as one Arrow IPC file (Specification,
Transitions; ADR 0040)."""

import copy
import json

import numpy as np

from . import _ipc
from ._data import _frozen
from ._errors import ProbatumKeyError, ProbatumValueError

# each column and its kind in the file, the action's kind its values'
KINDS = {
    "env":              "int64",
    "episode":          "int64",
    "step":             "int64",
    "observation":      "float32_list",
    "action":           None,
    "reward":           "float",
    "terminated":       "bool",
    "truncated":        "bool",
    "next_observation": "float32_list",
}
COLUMNS = tuple(KINDS)
# what the metadata holds, each under probatum. on the file's schema
META = ("spaces", "reward", "spec", "version")


class Transitions:
    """The rows, ``transitions[name]`` each a read-only numpy array: ``env``,
    ``episode`` and ``step``, int64; ``observation`` and ``next_observation``,
    ``(R, D)`` float32, flat whatever the layout; ``action``, ``(R, A)``, float32 or
    int64 for a discrete action, as the agent gave it before it was decoded;
    ``reward``, float64; and ``terminated`` and ``truncated``, booleans; with
    ``metadata``, the ``spaces``, the ``reward`` and the ``spec`` they were recorded
    under and probatum's ``version``. ``export("arrow", path)`` writes them as one
    Arrow IPC file."""

    __slots__ = ("_columns", "metadata")


    def __init__(self, columns, metadata):
        self._columns = {name: _frozen(np.asarray(columns[name])) for name in COLUMNS}
        self.metadata = metadata


    def __len__(self):
        return len(self._columns["reward"])


    def __getitem__(self, name):
        if name not in self._columns:
            raise ProbatumKeyError(f"transitions: no column {name!r}; use one of "
                                   f"{', '.join(COLUMNS)}")

        return self._columns[name]


    def __repr__(self):
        return (f"Transitions({len(self):,} rows, observations of "
                f"{self._columns['observation'].shape[1]:,} values, actions of "
                f"{self._columns['action'].shape[1]:,})")


    def __reduce__(self):
        return Transitions, (self._columns, self.metadata)


    def __deepcopy__(self, memo):
        # the columns never change, so a copy shares them
        return Transitions(self._columns, copy.deepcopy(self.metadata, memo))


    @property
    def columns(self):
        """The columns' names, in their order."""
        return list(COLUMNS)


    def export(self, format, path):
        """Write the rows as one Arrow IPC file at ``path``, which any Arrow reader
        opens: ``env``, ``episode`` and ``step`` as int64, the observations and the
        action as fixed-size lists, ``reward`` as float64 and the flags as booleans,
        with the metadata on its schema as JSON text under ``probatum.spaces``,
        ``probatum.reward``, ``probatum.spec`` and ``probatum.version``."""
        if format != "arrow":
            raise ProbatumValueError(f"format: {format!r}; give \"arrow\", the one "
                                     f"format export writes")

        _ipc.write(path, _stored(self._columns), {
            f"probatum.{key}": json.dumps(self.metadata[key], allow_nan=False)
            for key in META})


    @classmethod
    def _read(cls, path):
        """The transitions `export` wrote into the file ``path``, or the reason it holds
        none as a `ProbatumValueError`'s text."""
        try:
            columns, metadata = _ipc.read(path)
        except _ipc.Refused as error:
            raise ProbatumValueError(str(error)) from None

        rows  = len(columns["reward"][1]) if "reward" in columns else -1
        kinds = {name: kind for name, (kind, *_) in columns.items()}
        if (list(columns) != list(COLUMNS)
                or any(kinds[name] != kind for name, kind in KINDS.items() if kind)
                or kinds["action"] not in ("float32_list", "int64_list")
                or any(len(columns[name][1]) != rows or not columns[name][2].all()
                       for name in COLUMNS)
                or columns["observation"][1].shape != columns["next_observation"][1].shape
                or sorted(metadata) != sorted(f"probatum.{key}" for key in META)):
            raise ProbatumValueError("holds no transitions export wrote")

        try:
            held = {key: json.loads(metadata[f"probatum.{key}"]) for key in META}
        except ValueError:
            raise ProbatumValueError("holds metadata that is not JSON") from None

        return cls({name: columns[name][1] for name in COLUMNS}, held)


class _Recorder:
    """The rows an environment's steps leave as they come, with each environment's last
    observation, which the next row it takes opens on, and what each row's action is
    kept as, ``width`` values of ``dtype``."""

    __slots__ = ("last", "parts", "width", "dtype", "made")


    def __init__(self, n, size, width, dtype):
        self.last  = np.zeros((n, size), dtype=np.float32)
        self.parts = []
        self.width = width
        self.dtype = dtype
        self.made  = None


    def opened(self, mask, observed):
        """Each environment ``mask`` names starting an episode on its row of
        ``observed``, the flat observations."""
        self.last[mask] = observed[mask]


    def stepped(self, active, episode, step, actions, rewards, terminated, truncated,
                observed):
        """A row for each environment ``active`` names: its ``episode`` and its
        ``step``, the steps its episode had taken before this one, its last observation,
        its row of ``actions``, its reward and flags, and its row of ``observed``, the
        flat observations the step returned, which every environment opens on next."""
        rows  = np.flatnonzero(active)
        acted = np.asarray(actions).reshape(len(active), -1)
        given = np.asarray(acted, dtype=self.dtype)[rows]

        self.parts.append(dict(
            env              = rows.astype(np.int64),
            episode          = np.asarray(episode, dtype=np.int64)[rows],
            step             = np.asarray(step, dtype=np.int64)[rows],
            observation      = self.last[rows],
            action           = given,
            reward           = np.asarray(rewards, dtype=np.float64)[rows],
            terminated       = np.asarray(terminated, dtype=bool)[rows],
            truncated        = np.asarray(truncated, dtype=bool)[rows],
            next_observation = np.array(observed[rows], dtype=np.float32),
        ))

        self.last[:] = observed
        self.made    = None


    def transitions(self, metadata):
        """Every row so far as `Transitions` with ``metadata``, the same object until a
        later step adds rows or the metadata changes, as the spec's does when SB3's
        statistics are copied into it."""
        if self.made is not None and self.made.metadata == metadata:
            return self.made

        if self.made is None:
            columns = {name: (np.concatenate([part[name] for part in self.parts])
                              if self.parts else empty)
                       for name, empty in _empty(self.last.shape[1], self.width,
                                                 self.dtype).items()}
        else:
            columns = self.made._columns

        self.made = Transitions(columns, metadata)
        return self.made


    def cloned(self, n=None, env=None):
        """A recorder of the same rows and last observations, or with ``env`` one of
        ``n`` environments that each open on that environment's and keep no rows."""
        twin = _Recorder.__new__(_Recorder)
        twin.width, twin.dtype, twin.made = self.width, self.dtype, None

        if env is None:
            twin.last, twin.parts = self.last.copy(), list(self.parts)
        else:
            twin.last, twin.parts = np.repeat(self.last[env:env + 1], n, axis=0), []

        return twin


def _metadata(spec, version):
    """What transitions recorded under the agent spec ``spec`` hold beside their rows,
    with probatum's ``version``."""
    return dict(spaces={"observation": spec["observation"], "action": spec["actions"]},
                reward=spec["reward"], spec=spec, version=version)


def _empty(size, width, dtype):
    """Every column with no rows, the observations ``size`` values a row and the action
    ``width`` of ``dtype``."""
    return dict(
        env              = np.zeros(0, dtype=np.int64),
        episode          = np.zeros(0, dtype=np.int64),
        step             = np.zeros(0, dtype=np.int64),
        observation      = np.zeros((0, size), dtype=np.float32),
        action           = np.zeros((0, width), dtype=dtype),
        reward           = np.zeros(0, dtype=np.float64),
        terminated       = np.zeros(0, dtype=bool),
        truncated        = np.zeros(0, dtype=bool),
        next_observation = np.zeros((0, size), dtype=np.float32),
    )


def _stored(columns):
    """``columns`` as `_ipc.write` takes them, none of them holding a null."""
    action = "int64_list" if columns["action"].dtype == np.int64 else "float32_list"
    return [(name, kind or action, columns[name], None, False)
            for name, kind in KINDS.items()]
