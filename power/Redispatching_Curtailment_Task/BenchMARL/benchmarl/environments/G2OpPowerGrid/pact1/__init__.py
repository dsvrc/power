"""PACT-1 for the Grid2Op redispatching / curtailment task.

PACT1Env is imported lazily: it pulls in grid2op through the env base class,
while basis / rls / compensator / diagnostics are pure numpy.  Keeping the
import lazy is what lets the self-check gate (IV.4) run before -- and without --
the simulator.
"""
from .basis import CHANNEL_NAMES, CouplingBasis, gram_cond
from .compensator import apply_inverse, compensation_delta
from .rls import RLSEstimator

__all__ = ["CouplingBasis", "CHANNEL_NAMES", "gram_cond", "RLSEstimator",
           "compensation_delta", "apply_inverse", "PACT1Env"]


def __getattr__(name):
    if name == "PACT1Env":
        from .env import PACT1Env as _E
        return _E
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
