"""Result types shared by the CORE controller and verifier adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Tuple


class COREStatus(str, Enum):
    """The two sound outcomes produced by an incomplete verifier cascade."""

    ROBUST = "robust"
    UNCERTIFIED = "uncertified"


@dataclass(frozen=True)
class MarginResult:
    """One verifier's upper bound for one target-vs-reference margin.

    ``upper_bound`` always uses the convention

        f_target(x') - f_reference(x').

    A negative, valid upper bound certifies the target class.  ``valid`` is
    deliberately separate from ``status`` so each adapter can conservatively
    decide which native solver statuses are acceptable.
    """

    verifier: str
    target_class: int
    upper_bound: float
    valid: bool
    status: str = "computed"
    runtime_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def certifies(self, tolerance: float = 0.0) -> bool:
        """Return whether this result is a valid robustness certificate."""
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        return (
            self.valid
            and math.isfinite(self.upper_bound)
            and self.upper_bound < -tolerance
        )


@dataclass(frozen=True)
class StageRecord:
    """One target's result at one cascade stage."""

    target_class: int
    stage_index: int
    result: MarginResult


@dataclass(frozen=True)
class COREOutcome:
    """Final result and complete execution trace for one input."""

    status: COREStatus
    reference_class: int
    target_classes: Tuple[int, ...]
    records: Tuple[StageRecord, ...]
    certified_by: Tuple[Tuple[int, str], ...]
    failing_target: Optional[int] = None

    @property
    def robust(self) -> bool:
        return self.status is COREStatus.ROBUST

    def stage_for_target(self, target_class: int) -> Optional[str]:
        """Return the verifier that certified ``target_class``, if any."""
        for target, verifier in self.certified_by:
            if target == target_class:
                return verifier
        return None
