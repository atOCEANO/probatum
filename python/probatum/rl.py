"""Reinforcement learning on the engine: ``VectorEnv``, the Gymnasium vector environment
``probatum.env`` returns, stepped in Rust; ``Env``, one environment of it, which
Gymnasium's ``make`` builds as ``probatum/Trading-v0``, and ``make_vec`` as the batch,
once this module or ``probatum.env`` has loaded; and ``Batch``, the raw batch of
accounts, for a loop of the caller's own. It imports Gymnasium, which ``import
probatum`` does not (ADR 0044)."""

from ._env import Env, VectorEnv
from ._probatum import Batch

__all__ = ["Batch", "Env", "VectorEnv"]
