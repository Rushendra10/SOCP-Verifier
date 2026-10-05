from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .bound_layers import TensorPair, initial_linf_bounds, conv2d_interval, linear_interval, relu_interval_relaxation
except ImportError:
    from bound_layers import TensorPair, initial_linf_bounds, conv2d_interval, linear_interval, relu_interval_relaxation


@dataclass
class DeepPolyRecord:
    layer: nn.Module
    lower_in: torch.Tensor
    upper_in: torch.Tensor
    lower_out: torch.Tensor
    upper_out: torch.Tensor


def collect_deeppoly_records(model: nn.Sequential, x: torch.Tensor, eps: float) -> Tuple[TensorPair, List[DeepPolyRecord]]:
    lower, upper = initial_linf_bounds(x, eps)
    records: List[DeepPolyRecord] = []
    for layer in model:
        lower_in, upper_in = lower, upper
        if isinstance(layer, nn.Conv2d):
            lower, upper = conv2d_interval(layer, lower, upper)
        elif isinstance(layer, nn.ReLU):
            lower, upper = relu_interval_relaxation(lower, upper)
        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == "Flatten":
            lower = lower.view(lower.size(0), -1)
            upper = upper.view(upper.size(0), -1)
        elif isinstance(layer, nn.Linear):
            lower, upper = linear_interval(layer.weight, layer.bias, lower, upper)
        else:
            raise TypeError(f"Unsupported layer for DeepPoly bounds: {layer}")
        records.append(DeepPolyRecord(layer, lower_in, upper_in, lower, upper))
    return (lower, upper), records


def _backward_linear(lambda_out: torch.Tensor, layer: nn.Linear, const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer.bias is not None:
        const = const + torch.matmul(lambda_out, layer.bias)
    return torch.matmul(lambda_out, layer.weight), const


def _backward_conv2d(lambda_out: torch.Tensor, layer: nn.Conv2d, input_shape: torch.Size, const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer.bias is not None:
        const = const + (lambda_out.sum(dim=(2, 3)) * layer.bias).sum(dim=1)
    lambda_in = F.conv_transpose2d(lambda_out, layer.weight, None, layer.stride, layer.padding, 0, layer.groups, layer.dilation)
    return lambda_in[..., : input_shape[-2], : input_shape[-1]], const


def _backward_deeppoly_relu(lambda_out: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-12
    active = lower >= 0
    inactive = upper <= 0
    unstable = (~active) & (~inactive)
    lambda_in = torch.zeros_like(lambda_out)
    lambda_in = torch.where(active, lambda_out, lambda_in)
    lambda_in = torch.where(inactive, torch.zeros_like(lambda_in), lambda_in)
    alpha_u = upper / (upper - lower + eps)
    beta_u = -upper * lower / (upper - lower + eps)
    alpha_l = torch.where(upper > -lower, torch.ones_like(lower), torch.zeros_like(lower))
    use_upper = unstable & (lambda_out >= 0)
    use_lower = unstable & (lambda_out < 0)
    lambda_in = torch.where(use_upper, lambda_out * alpha_u, lambda_in)
    lambda_in = torch.where(use_lower, lambda_out * alpha_l, lambda_in)
    const = const + torch.where(use_upper, lambda_out * beta_u, torch.zeros_like(lambda_out)).view(lambda_out.size(0), -1).sum(dim=1)
    return lambda_in, const


def deeppoly_upper_bound_for_objective(model: nn.Sequential, x: torch.Tensor, eps: float, c: torch.Tensor) -> torch.Tensor:
    input_lower, input_upper = initial_linf_bounds(x, eps)
    (_, _), records = collect_deeppoly_records(model, x, eps)
    lambda_cur = c
    const = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    for record in reversed(records):
        layer = record.layer
        if isinstance(layer, nn.Linear):
            lambda_cur, const = _backward_linear(lambda_cur, layer, const)
        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == "Flatten":
            lambda_cur = lambda_cur.view_as(record.lower_in)
        elif isinstance(layer, nn.ReLU):
            lambda_cur, const = _backward_deeppoly_relu(lambda_cur, record.lower_in, record.upper_in, const)
        elif isinstance(layer, nn.Conv2d):
            lambda_cur, const = _backward_conv2d(lambda_cur, layer, record.lower_in.shape, const)
        else:
            raise TypeError(f"Unsupported layer for DeepPoly bounds: {layer}")
    input_max = torch.maximum(lambda_cur * input_lower, lambda_cur * input_upper).view(x.size(0), -1).sum(dim=1)
    return input_max + const


def deeppoly_margin_bounds(model: nn.Sequential, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    batch_size = x.size(0)
    num_classes = 10
    batch_idx = torch.arange(batch_size, device=x.device)
    bounds = []
    for j in range(num_classes):
        c = torch.zeros(batch_size, num_classes, device=x.device, dtype=x.dtype)
        c[:, j] = 1.0
        c[batch_idx, y] -= 1.0
        bounds.append(deeppoly_upper_bound_for_objective(model, x, eps, c))
    margin_upper = torch.stack(bounds, dim=1).clone()
    margin_upper[batch_idx, y] = 0.0
    return margin_upper
