# conv_to_dense_affines.py

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_to_dense(
    conv: nn.Conv2d,
    input_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """
    Converts Conv2d into dense affine form:

        y_flat = W_dense @ x_flat + b_dense

    Only recommended for small CNNs.
    """
    C, H, W = input_shape
    device = conv.weight.device
    dtype = conv.weight.dtype

    input_dim = C * H * W

    eye = torch.eye(input_dim, device=device, dtype=dtype)
    basis = eye.view(input_dim, C, H, W)

    y = F.conv2d(
        basis,
        conv.weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )

    out_dim = y[0].numel()
    W_dense = y.reshape(input_dim, out_dim).T

    if conv.bias is None:
        b_dense = torch.zeros(out_dim, device=device, dtype=dtype)
    else:
        _, C_out, H_out, W_out = y.shape
        b_dense = conv.bias.view(-1, 1, 1).expand(C_out, H_out, W_out).reshape(-1)

    out_shape = y.shape[1:]

    return W_dense, b_dense, out_shape


def linear_bounds(W, b, l, u):
    Wp = torch.clamp(W, min=0)
    Wn = torch.clamp(W, max=0)

    lo = Wp @ l + Wn @ u + b
    hi = Wp @ u + Wn @ l + b

    return lo, hi