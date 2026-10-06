"""Verifier-neutral problem, cache, and adapter interfaces for CORE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Hashable, Mapping, Tuple

from .result import MarginResult


@dataclass(frozen=True)
class COREProblem:
    """One clean input and the class-wise margins that must be certified."""

    model: Any
    x: Any
    reference_class: int
    epsilon: float
    target_classes: Tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        targets = tuple(int(target) for target in self.target_classes)
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if len(set(targets)) != len(targets):
            raise ValueError("target_classes must not contain duplicates")
        if int(self.reference_class) in targets:
            raise ValueError("target_classes must exclude the reference class")
        object.__setattr__(self, "reference_class", int(self.reference_class))
        object.__setattr__(self, "target_classes", targets)


@dataclass
class CORECache:
    """Per-input cache shared by all stages in a CORE cascade."""

    values: Dict[Hashable, Any] = field(default_factory=dict)

    def get_or_create(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        if key not in self.values:
            self.values[key] = factory()
        return self.values[key]


class MarginVerifier(ABC):
    """Common interface implemented by every verifier used in CORE."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("verifier name must be nonempty")
        self.name = name

    def prepare(self, problem: COREProblem, cache: CORECache) -> None:
        """Optionally populate reusable per-input state before target solves."""

    @abstractmethod
    def verify_target(
        self,
        problem: COREProblem,
        target_class: int,
        cache: CORECache,
    ) -> MarginResult:
        """Upper-bound one target-vs-reference margin."""


class CallableMarginVerifier(MarginVerifier):
    """Small adapter for a function that computes one margin upper bound."""

    def __init__(
        self,
        name: str,
        bound_fn: Callable[[COREProblem, int, CORECache], float | MarginResult],
    ) -> None:
        super().__init__(name)
        self._bound_fn = bound_fn

    def verify_target(
        self,
        problem: COREProblem,
        target_class: int,
        cache: CORECache,
    ) -> MarginResult:
        start = perf_counter()
        value = self._bound_fn(problem, target_class, cache)
        elapsed = perf_counter() - start
        if isinstance(value, MarginResult):
            return value
        return MarginResult(
            verifier=self.name,
            target_class=int(target_class),
            upper_bound=float(value),
            valid=True,
            runtime_seconds=elapsed,
        )
