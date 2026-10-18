"""What probatum raises and warns with: one root, ``ProbatumError``, a class beneath it
for each built-in a refusal raises, and the kinds a refusal is where both its topic and
its built-in are the kind's (ADR 0029). It imports nothing from the package, so the
binding can read its classes as it loads."""


class ProbatumError(Exception):
    """What every error probatum raises derives from, beside the built-in it means."""


    def __str__(self):
        # a KeyError quotes its message, and probatum's read as written
        return Exception.__str__(self)


class ProbatumValueError(ProbatumError, ValueError):
    """A wrong value that is no kind's, a policy's among them, since a policy's kind is
    a wrong type."""


class ProbatumTypeError(ProbatumError, TypeError):
    """A wrong type that is not a policy's, a Data's among them, since a Data's kind is
    a wrong value."""


class ProbatumKeyError(ProbatumError, KeyError):
    """A name none of a set's holds."""


class ProbatumIndexError(ProbatumError, IndexError):
    """A position past the end."""


class ProbatumAttributeError(ProbatumError, AttributeError):
    """An attribute an object does not have, such as one instrument's field on a
    universe."""


class ProbatumRuntimeError(ProbatumError, RuntimeError):
    """A search that cannot go on: its workers lost, every trial of it failed, or its
    winner run again not matching its trial."""


class ProbatumNotImplementedError(ProbatumRuntimeError, NotImplementedError):
    """What is not done yet: a Strategy that writes no ``next``, an agent, which
    ``backtest`` does not run yet, the transitions level, which it does not record yet,
    and a search's ``storage``, which it does not save to yet. It is a
    ProbatumRuntimeError as NotImplementedError is a RuntimeError."""


class DataError(ProbatumValueError):
    """A wrong value in bars, frames, the clock, features, funding or a series an
    indicator reads."""


class VenueError(ProbatumValueError):
    """A wrong value in a venue's settings or a preset."""


class SpecError(ProbatumValueError):
    """An agent whose spec does not match the run it is given."""


class LookaheadError(ProbatumValueError):
    """A policy or a feature the causality check finds reading a bar that has not
    closed."""


class PolicyError(ProbatumTypeError):
    """A policy ``backtest`` or ``search`` cannot run as given: a class for an instance,
    and for a search an instance for its class, a class that is no Strategy or a
    function that takes no data, a model that fits, an object of no kind it reads, an
    array of values it cannot read as decisions, and a trial that decides in another
    kind than the trials before it."""


class MissingDependency(ProbatumError, ImportError):
    """A package a call needs that is not installed, named with the line that installs
    it."""


class ProbatumWarning(UserWarning):
    """Every warning probatum gives, so a caller can silence them, or make them errors
    in a test, without touching anyone else's."""
