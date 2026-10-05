"""
Bound collection for the SOCP verifier/trainer.

The SOCP paper starts from certified scalar bounds on every pre-activation and
activation variable.  This file reuses the existing interval-bound logic from
src.dual_bounds / src.bound_layers, but does not edit those files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from .dual_bounds import collect_bound_records
except ImportError:  # Allows running from inside src/SOCP during debugging.
    from dual_bounds import collect_bound_records  # type: ignore


@dataclass
class LayerBounds:
    """
    Flat bounds for one affine/ReLU stage.

    s_lower/s_upper are pre-activation bounds for s^(k).
    z_lower/z_upper are post-ReLU activation bounds for z^(k), for hidden ReLU
    layers only.  The final affine layer has logits and no ReLU, so it is not
    stored here as a hidden layer.
    """

    s_lower: np.ndarray
    s_upper: np.ndarray
    z_lower: np.ndarray
    z_upper: np.ndarray
    layer_name: str


def collect_socp_bounds(model: nn.Sequential, x: torch.Tensor, eps: float) -> Tuple[np.ndarray, np.ndarray, List[LayerBounds], np.ndarray, np.ndarray]:
    """
    Collect input, hidden, and final-logit interval bounds for one sample.

    Parameters
    ----------
    model:
        Existing MNIST model from src/model.py.
    x:
        A single image tensor with shape (1, 1, 28, 28).  The current code is
        deliberately single-sample because exact SOCP certificates are expensive.
    eps:
        L_infinity perturbation radius.

    Returns
    -------
    input_lower, input_upper:
        Flattened input-box bounds.
    hidden_bounds:
        List of bounds for each hidden ReLU layer.
    final_lower, final_upper:
        Flattened interval bounds for the final logits.
    """
    if x.size(0) != 1:
        raise ValueError("SOCP certificate construction currently expects batch size 1 per solver call.")

    (final_l, final_u), records = collect_bound_records(model, x, eps)

    input_lower = torch.clamp(x - eps, 0.0, 1.0).detach().cpu().view(-1).numpy().astype(np.float64)
    input_upper = torch.clamp(x + eps, 0.0, 1.0).detach().cpu().view(-1).numpy().astype(np.float64)

    hidden_bounds: List[LayerBounds] = []
    relu_count = 0
    for rec in records:
        if isinstance(rec.layer, nn.ReLU):
            relu_count += 1
            s_l = rec.lower_in.detach().cpu().view(-1).numpy().astype(np.float64)
            s_u = rec.upper_in.detach().cpu().view(-1).numpy().astype(np.float64)
            z_l = rec.lower_out.detach().cpu().view(-1).numpy().astype(np.float64)
            z_u = rec.upper_out.detach().cpu().view(-1).numpy().astype(np.float64)
            hidden_bounds.append(LayerBounds(s_l, s_u, z_l, z_u, layer_name=f"relu{relu_count}"))

    return (
        input_lower,
        input_upper,
        hidden_bounds,
        final_l.detach().cpu().view(-1).numpy().astype(np.float64),
        final_u.detach().cpu().view(-1).numpy().astype(np.float64),
    )


def unstable_mask(s_lower: np.ndarray, s_upper: np.ndarray) -> np.ndarray:
    """Return a boolean mask for unstable ReLUs: l < 0 < u."""
    return (s_lower < 0.0) & (s_upper > 0.0)
