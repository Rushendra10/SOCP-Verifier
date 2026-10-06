"""Paper-facing implementation of the CORE class-wise cascade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .result import COREOutcome, COREStatus, MarginResult, StageRecord
from .verifier import CORECache, COREProblem, MarginVerifier


@dataclass(frozen=True)
class COREConfig:
    """Numerical and failure-handling policy for a CORE run."""

    certificate_tolerance: float = 0.0
    raise_on_verifier_error: bool = False

    def __post_init__(self) -> None:
        if self.certificate_tolerance < 0.0:
            raise ValueError("certificate_tolerance must be nonnegative")


def _error_result(verifier: MarginVerifier, target: int, phase: str, exc: Exception) -> MarginResult:
    return MarginResult(
        verifier=verifier.name,
        target_class=target,
        upper_bound=float("inf"),
        valid=False,
        status=f"{phase}_error",
        metadata={"error_type": type(exc).__name__, "error": str(exc)},
    )


def run_core(
    problem: COREProblem,
    verifiers: Iterable[MarginVerifier],
    *,
    config: COREConfig | None = None,
    cache: CORECache | None = None,
) -> COREOutcome:
    """Run CORE for one input.

    Targets are processed in ``problem.target_classes`` order.  A target stops
    at the first verifier that returns a valid negative upper bound.  If no
    verifier certifies a target, CORE immediately returns ``uncertified`` and
    does not evaluate the remaining targets.  It returns ``robust`` only after
    every target class has been certified.
    """
    stages = tuple(verifiers)
    if not stages:
        raise ValueError("CORE requires at least one verifier")
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("CORE verifier names must be unique")

    cfg = config or COREConfig()
    shared_cache = cache or CORECache()
    prepare_errors: Dict[int, Exception] = {}
    prepared_stages = set()

    records: List[StageRecord] = []
    certified_by: List[Tuple[int, str]] = []

    for target in problem.target_classes:
        target_certified = False
        for stage_index, verifier in enumerate(stages):
            if stage_index not in prepared_stages and stage_index not in prepare_errors:
                try:
                    verifier.prepare(problem, shared_cache)
                    prepared_stages.add(stage_index)
                except Exception as exc:
                    if cfg.raise_on_verifier_error:
                        raise
                    prepare_errors[stage_index] = exc

            if stage_index in prepare_errors:
                result = _error_result(
                    verifier,
                    target,
                    "prepare",
                    prepare_errors[stage_index],
                )
            else:
                try:
                    result = verifier.verify_target(problem, target, shared_cache)
                    if int(result.target_class) != int(target):
                        raise ValueError(
                            f"{verifier.name} returned target {result.target_class} "
                            f"while verifying target {target}"
                        )
                except Exception as exc:
                    if cfg.raise_on_verifier_error:
                        raise
                    result = _error_result(verifier, target, "verify", exc)

            records.append(StageRecord(target, stage_index, result))
            if result.certifies(cfg.certificate_tolerance):
                certified_by.append((target, verifier.name))
                target_certified = True
                break

        if not target_certified:
            return COREOutcome(
                status=COREStatus.UNCERTIFIED,
                reference_class=problem.reference_class,
                target_classes=problem.target_classes,
                records=tuple(records),
                certified_by=tuple(certified_by),
                failing_target=target,
            )

    return COREOutcome(
        status=COREStatus.ROBUST,
        reference_class=problem.reference_class,
        target_classes=problem.target_classes,
        records=tuple(records),
        certified_by=tuple(certified_by),
    )
