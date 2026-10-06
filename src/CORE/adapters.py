"""Adapters from the repository's bound routines to the CORE interface.

Imports of PyTorch, CVXPY, and the existing verifier modules are intentionally
lazy.  This keeps the verifier-neutral CORE controller independently testable
and allows reviewers to inspect its cascade logic without solver dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, FrozenSet

from .result import MarginResult
from .verifier import CORECache, COREProblem, MarginVerifier


def _ensure_repository_imports() -> None:
    """Expose the repository's current mixed package layout consistently.

    Existing modules use both ``src.dual_bounds``-style relative imports and
    ``SOCP.*`` top-level imports.  The aliases below keep that legacy layout
    contained at the adapter boundary without modifying existing files.
    """
    src_dir = Path(__file__).resolve().parents[1]
    repo_root = src_dir.parent
    for path in (str(repo_root), str(src_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    dual_bounds = importlib.import_module("src.dual_bounds")
    sys.modules.setdefault("dual_bounds", dual_bounds)
    socp_package = importlib.import_module("src.SOCP")
    sys.modules.setdefault("SOCP", socp_package)


def _shared_module(name: str):
    _ensure_repository_imports()
    return importlib.import_module(f"src.{name}")


def _socp_module(name: str):
    _ensure_repository_imports()
    return importlib.import_module(f"SOCP.{name}")


def _torch():
    return importlib.import_module("torch")


def _clean_logits(problem: COREProblem, cache: CORECache):
    def compute():
        torch = _torch()
        with torch.no_grad():
            return problem.model(problem.x).detach()

    return cache.get_or_create(("core", "clean_logits"), compute)


def _objective(problem: COREProblem, target_class: int, cache: CORECache, *, target_minus_reference: bool):
    torch = _torch()
    logits = _clean_logits(problem, cache)
    if int(problem.x.size(0)) != 1:
        raise ValueError("CORE verifier adapters currently expect one input at a time")
    num_classes = int(logits.shape[-1])
    if not 0 <= target_class < num_classes:
        raise ValueError(f"target class {target_class} is outside [0, {num_classes})")
    c = torch.zeros(
        int(problem.x.size(0)),
        num_classes,
        device=problem.x.device,
        dtype=problem.x.dtype,
    )
    if target_minus_reference:
        c[:, target_class] = 1.0
        c[:, problem.reference_class] -= 1.0
    else:
        c[:, problem.reference_class] = 1.0
        c[:, target_class] -= 1.0
    return c


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)
        if int(value.numel()) != 1:
            raise ValueError("a CORE target verifier must return exactly one bound")
        return float(value.item())
    return float(value)


class CROWNMarginVerifier(MarginVerifier):
    """CROWN-style LP upper bound for one target margin."""

    def __init__(self, name: str = "crown-lp") -> None:
        super().__init__(name)

    def verify_target(self, problem: COREProblem, target_class: int, cache: CORECache) -> MarginResult:
        start = perf_counter()
        dual_bounds = _shared_module("dual_bounds")
        c = _objective(problem, target_class, cache, target_minus_reference=True)
        value = _scalar(
            dual_bounds.dual_upper_bound_for_objective(
                problem.model,
                problem.x,
                problem.epsilon,
                c,
            )
        )
        return MarginResult(
            verifier=self.name,
            target_class=target_class,
            upper_bound=value,
            valid=math.isfinite(value),
            runtime_seconds=perf_counter() - start,
        )


class WongKolterMarginVerifier(MarginVerifier):
    """Wong-Kolter LP-dual upper bound for one target margin."""

    def __init__(self, name: str = "wong-kolter-lp") -> None:
        super().__init__(name)

    def verify_target(self, problem: COREProblem, target_class: int, cache: CORECache) -> MarginResult:
        start = perf_counter()
        dual_bounds = _shared_module("dual_bounds")
        c = _objective(problem, target_class, cache, target_minus_reference=False)
        lower_bound = _scalar(
            dual_bounds.dual_wk_lower_bound_for_objective(
                problem.model,
                problem.x,
                problem.epsilon,
                c,
            )
        )
        value = -lower_bound
        return MarginResult(
            verifier=self.name,
            target_class=target_class,
            upper_bound=value,
            valid=math.isfinite(value),
            runtime_seconds=perf_counter() - start,
        )


class DeepPolyMarginVerifier(MarginVerifier):
    """DeepPoly upper bound for one target margin."""

    def __init__(self, name: str = "deeppoly") -> None:
        super().__init__(name)

    def verify_target(self, problem: COREProblem, target_class: int, cache: CORECache) -> MarginResult:
        start = perf_counter()
        deeppoly_bounds = _shared_module("deeppoly_bounds")
        c = _objective(problem, target_class, cache, target_minus_reference=True)
        value = _scalar(
            deeppoly_bounds.deeppoly_upper_bound_for_objective(
                problem.model,
                problem.x,
                problem.epsilon,
                c,
            )
        )
        return MarginResult(
            verifier=self.name,
            target_class=target_class,
            upper_bound=value,
            valid=math.isfinite(value),
            runtime_seconds=perf_counter() - start,
        )


@dataclass(frozen=True)
class SparseSOCPStageConfig:
    """One level of the sparse SOCP hierarchy."""

    nodes_per_layer: int = 8
    max_E: int = 8
    max_S: int = 0
    max_T: int = 0
    prev_candidate_limit: int = 16
    gamma_backend: str = "dual"
    solver: str = "SCS"
    solver_max_iters: int = 200
    solver_verbose: bool = False
    accepted_statuses: FrozenSet[str] = frozenset({"optimal"})


class SparseSOCPMarginVerifier(MarginVerifier):
    """Adapter for one configured level of the sparse SOCP hierarchy."""

    def __init__(self, name: str, config: SparseSOCPStageConfig) -> None:
        super().__init__(name)
        self.config = config

    def prepare(self, problem: COREProblem, cache: CORECache) -> None:
        def build_shared():
            utils = _socp_module("utils")
            bounds = _socp_module("socp_bounds")
            input_shape = tuple(problem.x.shape[1:])
            affines = utils.sequential_to_dense_affines(
                problem.model,
                input_shape=input_shape,
            )
            input_lower, input_upper, hidden_bounds, _, _ = bounds.collect_socp_bounds(
                problem.model,
                problem.x,
                problem.epsilon,
            )
            return affines, input_lower, input_upper, hidden_bounds

        cache.get_or_create(("socp", "prepared"), build_shared)

    def verify_target(self, problem: COREProblem, target_class: int, cache: CORECache) -> MarginResult:
        start = perf_counter()
        self.prepare(problem, cache)
        affines, input_lower, input_upper, hidden_bounds = cache.values[("socp", "prepared")]

        influence = _socp_module("socp_influence")
        relaxation = _socp_module("socp_relaxation")
        solver_module = _socp_module("socp_solver")

        gamma_key = ("socp", "gammas", self.config.gamma_backend, int(target_class))
        gammas = cache.get_or_create(
            gamma_key,
            lambda: influence.compute_margin_gammas(
                problem.model,
                problem.x,
                eps=problem.epsilon,
                true_class=problem.reference_class,
                target_class=target_class,
                gamma_backend=self.config.gamma_backend,
            ),
        )

        pattern_key = (
            "socp",
            "pattern",
            int(target_class),
            self.config.gamma_backend,
            self.config.nodes_per_layer,
            self.config.max_E,
            self.config.max_S,
            self.config.max_T,
            self.config.prev_candidate_limit,
        )
        pattern = cache.get_or_create(
            pattern_key,
            lambda: relaxation.select_sparse_couplings(
                affines=affines,
                hidden_bounds=hidden_bounds,
                input_lower=input_lower,
                input_upper=input_upper,
                gammas=gammas,
                nodes_per_layer=self.config.nodes_per_layer,
                max_E=self.config.max_E,
                max_S=self.config.max_S,
                max_T=self.config.max_T,
                prev_candidate_limit=self.config.prev_candidate_limit,
            ),
        )

        solver_key = (
            "socp",
            "solver",
            self.name,
            self.config.solver,
            self.config.solver_max_iters,
            self.config.solver_verbose,
        )
        solver = cache.get_or_create(
            solver_key,
            lambda: solver_module.SparseSOCPVerifier(
                solver=self.config.solver,
                verbose=self.config.solver_verbose,
                max_iters=self.config.solver_max_iters,
            ),
        )
        raw = solver.solve_margin(
            affines=affines,
            input_lower=input_lower,
            input_upper=input_upper,
            hidden_bounds=hidden_bounds,
            pattern=pattern,
            true_class=problem.reference_class,
            target_class=target_class,
        )
        value = float(raw.value)
        status = str(raw.status)
        return MarginResult(
            verifier=self.name,
            target_class=target_class,
            upper_bound=value,
            valid=status in self.config.accepted_statuses and math.isfinite(value),
            status=status,
            runtime_seconds=perf_counter() - start,
            metadata={
                "solver": str(raw.solver),
                "num_variables": int(raw.num_variables),
                "num_constraints": int(raw.num_constraints),
                **dict(raw.metadata),
            },
        )
