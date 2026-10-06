"""Class-Optimized Robust Evaluation (CORE).

CORE is a verifier-agnostic controller.  It certifies each target-class margin
with an ordered sequence of increasingly expensive verifiers and escalates only
the targets that remain unresolved.
"""

from .cascade import COREConfig, run_core
from .result import COREOutcome, COREStatus, MarginResult, StageRecord
from .verifier import CORECache, COREProblem, CallableMarginVerifier, MarginVerifier

__all__ = [
    "CORECache",
    "COREConfig",
    "COREOutcome",
    "COREProblem",
    "COREStatus",
    "CallableMarginVerifier",
    "MarginResult",
    "MarginVerifier",
    "StageRecord",
    "run_core",
]
