"""
Sparse coupling selection and lifted-relaxation bookkeeping.

This implements the Option-B coupling-selection equations from the draft paper.
Gamma is supplied by socp_influence.py and can come from Wong--Kolter dual_WK
or CROWN/dual backward relaxation:

    node_score_j^(k) = |gamma_j^(k)| (u_s,j^(k) - l_s,j^(k))       Eq. (26)
    score_E(i,j)     = |gamma_j| |W_ji| Delta z_i^(k-1) u_z,j     Eq. (27)
    score_S(j,j')    = |gamma_j gamma_j'| u_z,j u_z,j'             Eq. (28)
    score_T(p,q)     = A_p A_q Delta z_p Delta z_q                Eq. (29)

The selected sets E, S, and T are then consumed by socp_solver.py to create
McCormick envelopes and SOC-representable 2x2 PSD minors.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from SOCP.socp_bounds import LayerBounds, unstable_mask
from SOCP.utils import DenseAffineLayer


@dataclass
class CouplingSets:
    """Sparse lifted coupling sets for one hidden layer k."""

    layer_index: int
    retained_nodes: np.ndarray
    E: List[Tuple[int, int]]  # cross-layer pairs (previous activation i, current activation j)
    S: List[Tuple[int, int]]  # current activation pairs (j, j')
    T: List[Tuple[int, int]]  # previous activation pairs (p, q)


@dataclass
class SOCPPattern:
    """The complete sparse coupling pattern for one sample/target."""

    per_layer: List[CouplingSets]


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores in descending order."""
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.size)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def select_sparse_couplings(
    affines: Sequence[DenseAffineLayer],
    hidden_bounds: Sequence[LayerBounds],
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    gammas: Sequence[np.ndarray],
    nodes_per_layer: int,
    max_E: int,
    max_S: int,
    max_T: int,
    prev_candidate_limit: int = 128,
) -> SOCPPattern:
    """
    Select sparse SOCP coupling sets E, S, and T layer-by-layer.

    The first hidden layer couples input variables x_i to first-layer ReLU
    activations z_j.  Later layers couple previous hidden activations z^(k-1)
    to current hidden activations z^(k).
    """
    per_layer: List[CouplingSets] = []

    # Previous activation bounds start with the perturbed input box.
    prev_l = input_lower
    prev_u = input_upper

    for k, hb in enumerate(hidden_bounds):
        W = affines[k].weight  # affine map feeding the current ReLU layer
        gamma = np.abs(gammas[k]) if k < len(gammas) else np.ones_like(hb.s_lower)
        u_width = hb.s_upper - hb.s_lower
        unstable = unstable_mask(hb.s_lower, hb.s_upper)

        node_scores = gamma * np.maximum(u_width, 0.0)
        node_scores = np.where(unstable, node_scores, -np.inf)
        retained = _topk_indices(node_scores, nodes_per_layer)

        prev_width = np.maximum(prev_u - prev_l, 0.0)
        current_u = np.maximum(hb.z_upper, 0.0)

        # To avoid scoring millions of previous-layer variables, restrict to the
        # largest-width previous variables first.
        prev_candidates = _topk_indices(prev_width, min(prev_candidate_limit, prev_width.size))

        # E^(k): cross-layer products z_prev_i * z_current_j.
        E_scores: List[Tuple[float, int, int]] = []
        for j in retained:
            row = np.abs(W[j, prev_candidates])
            scores = gamma[j] * row * prev_width[prev_candidates] * max(current_u[j], 0.0)
            for local_idx, score in enumerate(scores):
                if np.isfinite(score) and score > 0:
                    E_scores.append((float(score), int(prev_candidates[local_idx]), int(j)))
        E_scores.sort(reverse=True, key=lambda t: t[0])
        E = [(i, j) for _, i, j in E_scores[:max_E]]

        # S^(k): products among retained current activations.
        S_scores: List[Tuple[float, int, int]] = []
        for a, b in combinations(retained.tolist(), 2):
            score = gamma[a] * gamma[b] * max(current_u[a], 0.0) * max(current_u[b], 0.0)
            if np.isfinite(score) and score > 0:
                S_scores.append((float(score), int(a), int(b)))
        S_scores.sort(reverse=True, key=lambda t: t[0])
        S = [(a, b) for _, a, b in S_scores[:max_S]]

        # T^(k): products among previous activations, weighted by influence on
        # the retained current nodes.
        influence = np.zeros(prev_u.shape[0], dtype=np.float64)
        if len(retained) > 0:
            influence[prev_candidates] = np.sum(
                gamma[retained, None] * np.abs(W[np.ix_(retained, prev_candidates)]), axis=0
            )
        T_scores: List[Tuple[float, int, int]] = []
        for p, q in combinations(prev_candidates.tolist(), 2):
            score = influence[p] * influence[q] * prev_width[p] * prev_width[q]
            if np.isfinite(score) and score > 0:
                T_scores.append((float(score), int(p), int(q)))
        T_scores.sort(reverse=True, key=lambda t: t[0])
        T = [(p, q) for _, p, q in T_scores[:max_T]]

        per_layer.append(CouplingSets(k, retained_nodes=retained, E=E, S=S, T=T))

        # Current ReLU activation becomes the previous activation for next layer.
        prev_l = hb.z_lower
        prev_u = hb.z_upper

    return SOCPPattern(per_layer=per_layer)
