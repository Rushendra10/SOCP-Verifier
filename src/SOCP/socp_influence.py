"""
Backward influence signals gamma for SOCP sparse coupling selection.

The AAAI SOCP draft does not select lifted pairs using interval width alone.
It uses a reverse-mode importance vector gamma and scores unstable neurons by

    node_score_j^(k) = |gamma_j^(k)| (u_s,j^(k) - l_s,j^(k)).

This file computes gamma on top of the existing LP/CROWN or Wong--Kolter
backward bound machinery WITHOUT modifying the original baseline files.

Supported gamma backends
------------------------
1) gamma_backend="dual_WK"
   Uses the Wong--Kolter fixed-alpha backward recursion.  This is the default
   because it is closest to the LP baseline in Wong & Kolter.

2) gamma_backend="dual"
   Uses the existing CROWN-style sign-dependent backward relaxation.

3) gamma_backend="clean"
   Uses the plain clean-margin gradient.  This is kept only for debugging and
   ablations; it is not the recommended paper-faithful option.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from dual_bounds import (
        collect_bound_records,
        _backward_conv2d,
        _backward_linear,
        _backward_relu_upper,
        _wk_backward_conv2d,
        _wk_backward_linear,
        _wk_backward_relu_fixed_alpha,
    )
except ImportError:  # pragma: no cover
    from dual_bounds import (  # type: ignore
        collect_bound_records,
        _backward_conv2d,
        _backward_linear,
        _backward_relu_upper,
        _wk_backward_conv2d,
        _wk_backward_linear,
        _wk_backward_relu_fixed_alpha,
    )


def _margin_vector(batch_size: int, true_class: int, target_class: int, device, dtype) -> torch.Tensor:
    """
    Build c for the target-vs-true margin:

        m(x) = z_target(x) - z_true(x) = c^T f(x).

    Shape is (1, 10) because the SOCP verifier currently certifies one sample
    at a time.
    """
    c = torch.zeros(batch_size, 10, device=device, dtype=dtype)
    c[:, target_class] = 1.0
    c[:, true_class] -= 1.0
    return c


def _clean_margin_gammas(model: nn.Sequential, x: torch.Tensor, true_class: int, target_class: int) -> List[np.ndarray]:
    """
    Debug/ablation gamma: gradient of clean margin wrt each ReLU preactivation.

    This does NOT use the robust LP relaxation.  It is useful as an ablation,
    but the recommended paper setting is gamma_backend="dual_WK".
    """
    model.zero_grad(set_to_none=True)
    preacts: List[torch.Tensor] = []

    h = x.detach().clone().requires_grad_(True)
    affine_outputs: List[torch.Tensor] = []
    for layer in model:
        h = layer(h)
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            h.retain_grad()
            affine_outputs.append(h)

    logits = h
    margin = logits[0, target_class] - logits[0, true_class]
    margin.backward()

    # Last affine output is logits, not a hidden ReLU preactivation.
    hidden_preacts = affine_outputs[:-1]
    gammas: List[np.ndarray] = []
    for s in hidden_preacts:
        grad = s.grad
        if grad is None:
            gammas.append(np.zeros(s.numel(), dtype=np.float64))
        else:
            gammas.append(grad.detach().cpu().view(-1).numpy().astype(np.float64))
    return gammas


def _crown_margin_gammas(model: nn.Sequential, x: torch.Tensor, eps: float, true_class: int, target_class: int) -> List[np.ndarray]:
    """
    Gamma from the existing CROWN-style LP relaxation.

    We start from the target-vs-true objective c^T f(x).  During the backward
    relaxation, whenever we cross a ReLU, the returned coefficient is the
    relaxed coefficient on the preactivation s^(k).  That coefficient is used as
    gamma^(k), because the paper's scores measure how much preactivation j
    influences the relaxed target margin.
    """
    (_, _), records = collect_bound_records(model, x, eps)
    c = _margin_vector(x.size(0), true_class, target_class, x.device, x.dtype)

    lambda_cur = c
    const = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    gammas_reversed: List[np.ndarray] = []

    for record in reversed(records):
        layer = record.layer
        if isinstance(layer, nn.Linear):
            lambda_cur, const = _backward_linear(lambda_cur, layer, const)
        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == "Flatten":
            lambda_cur = lambda_cur.view_as(record.lower_in)
        elif isinstance(layer, nn.ReLU):
            lambda_cur, const = _backward_relu_upper(lambda_cur, record.lower_in, record.upper_in, const)
            gammas_reversed.append(lambda_cur.detach().cpu().view(-1).numpy().astype(np.float64))
        elif isinstance(layer, nn.Conv2d):
            lambda_cur, const = _backward_conv2d(lambda_cur, layer, record.lower_in.shape, const)
        else:
            raise TypeError(f"Unsupported layer for CROWN gamma: {layer}")

    return list(reversed(gammas_reversed))


def _wk_margin_gammas(model: nn.Sequential, x: torch.Tensor, eps: float, true_class: int, target_class: int) -> List[np.ndarray]:
    """
    Gamma from the Wong--Kolter fixed-alpha LP-dual recursion.

    The WK routine in the baseline computes a lower bound on

        min (z_true - z_target).

    For coupling selection we need an influence signal for the opposite margin

        z_target - z_true.

    We therefore initialize the same target-vs-true c used by CROWN and then set
    nu_K = -c, matching the WK dual convention.  After each ReLU step, nu_cur is
    the fixed-alpha relaxed coefficient on the preactivation s^(k).  That vector
    is gamma^(k).  The sign is not important for scoring because the paper uses
    absolute values |gamma|.
    """
    (_, _), records = collect_bound_records(model, x, eps)
    c = _margin_vector(x.size(0), true_class, target_class, x.device, x.dtype)

    # Existing WK code initializes nu_k = -c.
    nu_cur = -c
    j_value = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    gammas_reversed: List[np.ndarray] = []

    for record in reversed(records):
        layer = record.layer
        if isinstance(layer, nn.Linear):
            nu_cur, j_value = _wk_backward_linear(nu_cur, layer, j_value)
        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == "Flatten":
            nu_cur = nu_cur.view_as(record.lower_in)
        elif isinstance(layer, nn.ReLU):
            nu_cur, j_value = _wk_backward_relu_fixed_alpha(nu_cur, record.lower_in, record.upper_in, j_value)
            gammas_reversed.append(nu_cur.detach().cpu().view(-1).numpy().astype(np.float64))
        elif isinstance(layer, nn.Conv2d):
            nu_cur, j_value = _wk_backward_conv2d(nu_cur, layer, record.lower_in.shape, j_value)
        else:
            raise TypeError(f"Unsupported layer for WK gamma: {layer}")

    return list(reversed(gammas_reversed))


def compute_margin_gammas(
    model: nn.Sequential,
    x: torch.Tensor,
    eps: float,
    true_class: int,
    target_class: int,
    gamma_backend: str = "dual_WK",
) -> List[np.ndarray]:
    """
    Public gamma API used by socp_certificate.py.

    Parameters
    ----------
    gamma_backend:
        "dual_WK"  -> Wong--Kolter fixed-alpha LP-dual influence. Recommended.
        "dual"     -> CROWN-style LP influence.
        "clean"    -> plain clean-margin gradient, for ablations only.
    """
    backend = gamma_backend.lower()
    if backend in {"dual_wk", "wk", "wong_kolter", "wong-kolter"}:
        return _wk_margin_gammas(model, x, eps, true_class, target_class)
    if backend in {"dual", "crown", "alpha_crown", "alpha-crown"}:
        return _crown_margin_gammas(model, x, eps, true_class, target_class)
    if backend == "clean":
        return _clean_margin_gammas(model, x, true_class, target_class)
    raise ValueError("Unknown gamma_backend={!r}. Use 'dual_WK', 'dual', or 'clean'.".format(gamma_backend))
