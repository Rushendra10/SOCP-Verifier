from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


TensorPair = Tuple[torch.Tensor, torch.Tensor]


def initial_linf_bounds(x: torch.Tensor, eps: float) -> TensorPair:
    """
    Input adversarial set:
        S(x, eps) = {x' : ||x' - x||_inf <= eps, 0 <= x' <= 1}

    Interval lower/upper bounds:
        l_0 = clamp(x - eps, 0, 1)
        u_0 = clamp(x + eps, 0, 1)
    """
    lower = torch.clamp(x - eps, 0.0, 1.0)
    upper = torch.clamp(x + eps, 0.0, 1.0)
    return lower, upper


def linear_interval(weight: torch.Tensor, bias: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> TensorPair:
    """
    Interval bound propagation for affine layer z = W x + b.

    Split W into positive and negative parts:
        W+ = max(W, 0), W- = min(W, 0)

    Then:
        lower_z = W+ lower_x + W- upper_x + b
        upper_z = W+ upper_x + W- lower_x + b

    This is the same basic interval idea used before constructing the LP-style
    convex outer ReLU relaxation.
    """
    w_pos = torch.clamp(weight, min=0)
    w_neg = torch.clamp(weight, max=0)
    lower_z = F.linear(lower, w_pos, bias) + F.linear(upper, w_neg, None)
    upper_z = F.linear(upper, w_pos, bias) + F.linear(lower, w_neg, None)
    return lower_z, upper_z


def conv2d_interval(layer: nn.Conv2d, lower: torch.Tensor, upper: torch.Tensor) -> TensorPair:
    """
    Same affine interval rule as linear_interval, but for convolution:
        z = W * x + b
    where * is convolution.
    """
    w_pos = torch.clamp(layer.weight, min=0)
    w_neg = torch.clamp(layer.weight, max=0)
    lower_z = F.conv2d(lower, w_pos, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups) \
            + F.conv2d(upper, w_neg, None, layer.stride, layer.padding, layer.dilation, layer.groups)
    upper_z = F.conv2d(upper, w_pos, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups) \
            + F.conv2d(lower, w_neg, None, layer.stride, layer.padding, layer.dilation, layer.groups)
    return lower_z, upper_z


def relu_interval_relaxation(lower: torch.Tensor, upper: torch.Tensor) -> TensorPair:
    """
    Convex outer relaxation of y = ReLU(x) over x in [l, u].

    Exact cases:
        if u <= 0: y = 0
        if l >= 0: y = x

    Unstable case l < 0 < u:
        ReLU is replaced by the convex triangle relaxation:
            y >= 0
            y >= x
            y <= (u / (u - l)) (x - l)

    In full Wong-Kolter LP training, these inequalities define a convex outer
    adversarial polytope. Here we do not solve the LP/dual exactly. We use the
    induced interval upper/lower bounds as a lightweight training surrogate.

    Interval output is:
        lower_y = max(lower_x, 0)
        upper_y = max(upper_x, 0)
    """
    return torch.clamp(lower, min=0), torch.clamp(upper, min=0)


def propagate_bounds(model: nn.Sequential, x: torch.Tensor, eps: float) -> TensorPair:
    """
    Propagate [lower, upper] bounds through the network.

    This gives a cheap LP-relaxation-inspired bound on possible logits under
    all perturbations ||delta||_inf <= eps.
    """
    lower, upper = initial_linf_bounds(x, eps)

    for layer in model:
        if isinstance(layer, nn.Conv2d):
            lower, upper = conv2d_interval(layer, lower, upper)
        elif isinstance(layer, nn.ReLU):
            lower, upper = relu_interval_relaxation(lower, upper)
        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == 'Flatten':
            lower = lower.view(lower.size(0), -1)
            upper = upper.view(upper.size(0), -1)
        elif isinstance(layer, nn.Linear):
            lower, upper = linear_interval(layer.weight, layer.bias, lower, upper)
        else:
            raise TypeError(f'Unsupported layer for bounds: {layer}')

    return lower, upper
