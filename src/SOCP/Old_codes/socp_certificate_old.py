"""
High-level certificate API.

This module maps one PyTorch model/sample into the exact sparse SOCP problem:

    phi_SOCP(x, y_true, y_target; E,S,T)

and optionally computes all target-class margins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
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
    solver_max_iters: int = 500
    solver_verbose: bool = False
    all_classes: bool = True
    max_targets: int = 9
    # Option-B source for gamma in Equations (26)--(29).
    # Recommended for the paper: "dual_WK".
    # Ablations: "dual" for CROWN, "clean" for clean-gradient gamma.
    gamma_backend: str = "dual_WK"


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


def certify_sample(model: nn.Sequential, x: torch.Tensor, y: int, cfg: CertificateConfig) -> AllTargetCertificate:
    """
    Compute sparse SOCP class-wise certificates for one MNIST sample.

    By default this uses all 9 wrong classes, which is required for the full
    multiclass certificate.  For debugging, cfg.max_targets can restrict the
    number of classes solved.
    """
    model.eval()
    if x.size(0) != 1:
        raise ValueError("certify_sample expects a single sample with batch dimension 1.")

    affines = sequential_to_dense_affines(model)
    input_l, input_u, hidden_bounds, _, _ = collect_socp_bounds(model, x, cfg.epsilon)
    verifier = SparseSOCPVerifier(solver=cfg.solver, verbose=cfg.solver_verbose, max_iters=cfg.solver_max_iters)

    targets = [c for c in range(10) if c != int(y)]
    if not cfg.all_classes:
        targets = targets[: cfg.max_targets]

    results: Dict[int, SOCPResult] = {}
    for target in targets:
        gammas = compute_margin_gammas(
            model,
            x,
            eps=cfg.epsilon,
            true_class=int(y),
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
            true_class=int(y),
            target_class=target,
        )

    return AllTargetCertificate(true_class=int(y), margins=results)
