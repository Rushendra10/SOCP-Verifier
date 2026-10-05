"""
High-level certificate API.

This module maps one PyTorch model/sample into the exact sparse SOCP problem:

    phi_SOCP(x, y_true, y_target; E,S,T)

and optionally computes all target-class margins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from SOCP.socp_bounds import collect_socp_bounds
from SOCP.socp_influence import compute_margin_gammas
from SOCP.socp_relaxation import select_sparse_couplings
from SOCP.socp_solver import SOCPResult, SparseSOCPVerifier
from SOCP.utils import sequential_to_dense_affines


@dataclass
class CertificateConfig:
    """Configuration for one SOCP certificate call."""

    epsilon: float = 0.3
    nodes_per_layer: int = 16
    max_E: int = 64
    max_S: int = 32
    max_T: int = 32
    prev_candidate_limit: int = 128
    solver: str = "CLARABEL"
    solver_max_iters: int = 100
    solver_verbose: bool = False
    all_classes: bool = True
    max_targets: int = 4

    # Option-B source for gamma in Equations (26)--(29).
    # Recommended for the paper: "dual_WK".
    # Ablations: "dual" for CROWN, "clean" for clean-gradient gamma.
    gamma_backend: str = "dual"


@dataclass
class AllTargetCertificate:
    """All target margins for one sample."""

    true_class: int
    margins: Dict[int, SOCPResult]

    @property
    def worst_margin(self) -> float:
        if not self.margins:
            return float("inf")
        return max(r.value for r in self.margins.values())

    @property
    def certified(self) -> bool:
        return self.worst_margin < 0.0
    
    # @property
    # def certified(self) -> bool:
    #     return bool(self.margins) and all(
    #         r.status == "optimal" and r.value < 0.0
    #         for r in self.margins.values()
    #     )


def choose_target_classes(
    model: nn.Module,
    x: torch.Tensor,
    true_class: int,
    cfg: CertificateConfig,
) -> List[int]:
    """
    Choose which target classes to certify.

    If cfg.all_classes is True:
        return all classes except the true class.

    If cfg.all_classes is False:
        return the cfg.max_targets most dangerous classes, where "dangerous"
        means the non-true classes with the largest clean logits.

    This reduces the number of SOCP solves during debugging/training.
    For final full multiclass certification, use cfg.all_classes=True.
    """
    with torch.no_grad():
        logits = model(x).view(-1)

    num_classes = int(logits.numel())
    targets = [c for c in range(num_classes) if c != int(true_class)]

    if cfg.all_classes:
        return targets

    max_targets = max(1, min(int(cfg.max_targets), len(targets)))

    ranked_targets = sorted(
        targets,
        key=lambda c: float(logits[c].item()),
        reverse=True,
    )

    return ranked_targets[:max_targets]


def certify_sample(
    model: nn.Sequential,
    x: torch.Tensor,
    y: int,
    cfg: CertificateConfig,
) -> AllTargetCertificate:
    """
    Compute sparse SOCP class-wise certificates for one sample.

    Full certification:
        cfg.all_classes = True

    Debugging / faster training:
        cfg.all_classes = False
        cfg.max_targets = k

    When cfg.all_classes is False, this function certifies only the k most
    dangerous non-true classes according to the clean logits.
    """
    model.eval()

    if x.size(0) != 1:
        raise ValueError("certify_sample expects a single sample with batch dimension 1.")

    true_class = int(y)

    # affines = sequential_to_dense_affines(model)
    # input_l, input_u, hidden_bounds, _, _ = collect_socp_bounds(model, x, cfg.epsilon)
    input_shape = tuple(x.shape[1:])

    # print(f"[SOCP] Input shape: {input_shape}")

    affines = sequential_to_dense_affines(
        model,
        input_shape=input_shape,
    )

    input_l, input_u, hidden_bounds, _, _ = collect_socp_bounds(
        model,
        x,
        cfg.epsilon,
    )
    
    
    verifier = SparseSOCPVerifier(
        solver=cfg.solver,
        verbose=cfg.solver_verbose,
        max_iters=cfg.solver_max_iters,
    )

    targets = choose_target_classes(model, x, true_class, cfg)

    if cfg.solver_verbose:
        print(f"[SOCP] true_class={true_class}, targets={targets}")

    results: Dict[int, SOCPResult] = {}

    for target in targets:
        gammas = compute_margin_gammas(
            model,
            x,
            eps=cfg.epsilon,
            true_class=true_class,
            target_class=target,
            gamma_backend=cfg.gamma_backend,
        )

        pattern = select_sparse_couplings(
            affines=affines,
            hidden_bounds=hidden_bounds,
            input_lower=input_l,
            input_upper=input_u,
            gammas=gammas,
            nodes_per_layer=cfg.nodes_per_layer,
            max_E=cfg.max_E,
            max_S=cfg.max_S,
            max_T=cfg.max_T,
            prev_candidate_limit=cfg.prev_candidate_limit,
        )

        results[target] = verifier.solve_margin(
            affines=affines,
            input_lower=input_l,
            input_upper=input_u,
            hidden_bounds=hidden_bounds,
            pattern=pattern,
            true_class=true_class,
            target_class=target,
        )

    return AllTargetCertificate(true_class=true_class, margins=results)