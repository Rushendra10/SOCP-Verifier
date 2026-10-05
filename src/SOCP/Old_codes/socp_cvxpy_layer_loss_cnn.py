# socp_cvxpy_layer_loss_cnn.py

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer

from SOCP.conv_to_dense_affines import conv2d_to_dense, linear_bounds


def relu_upper_coeffs(l, u):
    eps = 1e-6

    active = l >= 0
    inactive = u <= 0
    unstable = (~active) & (~inactive)

    a = torch.zeros_like(l)
    b = torch.zeros_like(l)

    a = torch.where(active, torch.ones_like(a), a)
    b = torch.where(active, torch.zeros_like(b), b)

    a = torch.where(inactive, torch.zeros_like(a), a)
    b = torch.where(inactive, torch.zeros_like(b), b)

    a_unstable = u / (u - l + eps)
    b_unstable = -a_unstable * l

    a = torch.where(unstable, a_unstable, a)
    b = torch.where(unstable, b_unstable, b)

    return a, b


@lru_cache(maxsize=8)
def build_cnn_head_socp_layer(feature_dim: int, hidden_dim: int, nout: int):
    h = cp.Variable(feature_dim)
    s = cp.Variable(hidden_dim)
    z = cp.Variable(hidden_dim)
    out = cp.Variable(nout)

    W1 = cp.Parameter((hidden_dim, feature_dim))
    b1 = cp.Parameter(hidden_dim)
    W2 = cp.Parameter((nout, hidden_dim))
    b2 = cp.Parameter(nout)

    h_l = cp.Parameter(feature_dim)
    h_u = cp.Parameter(feature_dim)

    a = cp.Parameter(hidden_dim)
    be = cp.Parameter(hidden_dim)

    c = cp.Parameter(nout)

    constraints = [
        h >= h_l,
        h <= h_u,

        s == W1 @ h + b1,

        z >= 0,
        z >= s,
        z <= cp.multiply(a, s) + be,

        out == W2 @ z + b2,
    ]

    problem = cp.Problem(cp.Minimize(-(c @ out)), constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[W1, b1, W2, b2, h_l, h_u, a, be, c],
        variables=[out],
    )

    return layer


def cnn_feature_bounds_as_affine_box(
    model: nn.Sequential,
    x: torch.Tensor,
    epsilon: float,
    input_shape: Tuple[int, int, int] = (1, 28, 28),
):
    """
    Assumes model structure:

        Conv2d
        ReLU
        Conv2d
        ReLU
        Flatten
        Linear
        ReLU
        Linear
    """

    conv1 = model[0]
    conv2 = model[2]

    assert isinstance(conv1, nn.Conv2d)
    assert isinstance(conv2, nn.Conv2d)

    x0 = x.view(1, -1).squeeze(0)

    x_l = torch.clamp(x0 - epsilon, 0.0, 1.0)
    x_u = torch.clamp(x0 + epsilon, 0.0, 1.0)

    Wc1, bc1, shape1 = conv2d_to_dense(conv1, input_shape)
    s1_l, s1_u = linear_bounds(Wc1, bc1, x_l, x_u)

    z1_l = torch.clamp(s1_l, min=0)
    z1_u = torch.clamp(s1_u, min=0)

    Wc2, bc2, shape2 = conv2d_to_dense(conv2, shape1)
    s2_l, s2_u = linear_bounds(Wc2, bc2, z1_l, z1_u)

    z2_l = torch.clamp(s2_l, min=0)
    z2_u = torch.clamp(s2_u, min=0)

    return z2_l, z2_u


def socp_cvxpy_layer_loss_cnn(
    model: nn.Sequential,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    max_targets: int = 1,
    max_E: int = 8,
    max_S: int = 4,
    max_T: int = 4,
    solver_iters: int = 50,
    clean_ac: int = 0.0,
):
    """
    Differentiable SOCP robust CE for small CNN:

        Conv-ReLU-Conv-ReLU = converted to affine bounds
        Linear-ReLU-Linear = certified with CVXPYLayer
    """

    if x.size(0) != 1:
        raise ValueError("This SOCP loss currently expects batch size 1.")

    fc1 = model[5]
    fc2 = model[7]

    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    h_l, h_u = cnn_feature_bounds_as_affine_box(model, x, epsilon)

    W1, b1 = fc1.weight, fc1.bias
    W2, b2 = fc2.weight, fc2.bias

    s_l, s_u = linear_bounds(W1, b1, h_l, h_u)

    eps_bound = 1e-6
    same = (s_u - s_l).abs() < eps_bound
    s_l = torch.where(same, s_l - eps_bound, s_l)
    s_u = torch.where(same, s_u + eps_bound, s_u)

    a, be = relu_upper_coeffs(s_l, s_u)

    logits = model(x)
    clean_ce = F.cross_entropy(logits, y)

    true_class = int(y.item())
    nout = logits.size(1)

    with torch.no_grad():
        ranked = [
            c for c in torch.argsort(logits[0], descending=True).tolist()
            if c != true_class
        ][:max_targets]

    layer = build_cnn_head_socp_layer(
        feature_dim=h_l.numel(),
        hidden_dim=W1.shape[0],
        nout=nout,
    )

    robust_logits = torch.zeros_like(logits)

    for target in ranked:
        c_vec = torch.zeros(nout, device=x.device, dtype=x.dtype)
        c_vec[target] = 1.0
        c_vec[true_class] = -1.0

        out_star, = layer(
            W1,
            b1,
            W2,
            b2,
            h_l,
            h_u,
            a,
            be,
            c_vec,
            solver_args={
                "solve_method": "SCS",
                "max_iters": solver_iters,
                "eps": 1e-3,
                "verbose": False,
            },
        )

        margin_ub = out_star[target] - out_star[true_class]

        # Important:
        # positive margin means unsafe, so CE should penalize it.
        robust_logits[0, target] = margin_ub

    robust_logits[0, true_class] = 0.0

    robust_ce = F.cross_entropy(robust_logits, y)
    if clean_ac > 0.6:
        alpha = 0.5
        print(f"alpha = 0.5 ce: {clean_ac}")
    else:
        alpha = 0.2
    total_loss = (1.0 - alpha) * clean_ce + alpha * robust_ce

    worst_margin = (
        robust_logits[0, ranked].max()
        if ranked
        else torch.tensor(0.0, device=x.device, dtype=x.dtype)
    )

    return total_loss, clean_ce, robust_ce, worst_margin