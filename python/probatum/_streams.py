"""Named random streams, derived from a run's seed (ADR 0021)."""

import numpy as np


def stream(seed, name):
    """The random stream ``name`` of a run seeded with ``seed``: PCG64 from a NumPy
    SeedSequence of the seed, keyed by the name. A stream's draws depend on the seed
    and its name alone, never on which other streams a run makes or in what order,
    and they are the same on every machine. A ``seed`` of None draws fresh entropy.
    """
    sequence = np.random.SeedSequence(seed, spawn_key=tuple(name.encode("utf-8")))
    return np.random.Generator(np.random.PCG64(sequence))
