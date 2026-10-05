"""
Dual/backward convex-relaxation bounds for the small MNIST network.

This file contains two related backward bound backends:

1) dual / CROWN-style upper bounds with sign-dependent ReLU lines.
2) dual_WK / Wong--Kolter LP-dual bounds with fixed alpha and the J_epsilon objective.

The original implementation added the missing Wong--Kolter-style idea:

    Forward pass:  compute pre-activation bounds [l_i, u_i].
    Backward pass: propagate a linear objective c^T f(x) backward through
                   the convex ReLU relaxation to upper-bound the worst-case
                   adversarial margin.

We use the bound to train against the margin

    max_{||delta||_inf <= eps} ( z_j(x + delta) - z_y(x + delta) )

for every wrong class j. If every margin upper bound is < 0, then the model is
robust for that training example under this relaxation.

This is not a full verifier script: it is the dual bound used inside training.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bound_layers import (
    TensorPair,
    initial_linf_bounds,
    conv2d_interval,
    linear_interval,
    relu_interval_relaxation,
)

# from bound_layers import (
#     TensorPair,
#     initial_linf_bounds,
#     conv2d_interval,
#     linear_interval,
#     relu_interval_relaxation,
# )


@dataclass
class BoundRecord:
    """Stores the layer and the interval bounds before/after that layer."""

    layer: nn.Module
    lower_in: torch.Tensor
    upper_in: torch.Tensor
    lower_out: torch.Tensor
    upper_out: torch.Tensor


def collect_bound_records(model: nn.Sequential, x: torch.Tensor, eps: float) -> Tuple[TensorPair, List[BoundRecord]]:
    """
    Compute and store L/U values for every layer.

    Initial adversarial input box:
        X = {x' : ||x' - x||_inf <= eps, 0 <= x' <= 1}
        l_0 = clamp(x - eps, 0, 1)
        u_0 = clamp(x + eps, 0, 1)

    For affine layers z = W h + b:
        W+ = max(W, 0), W- = min(W, 0)
        l_z = W+ l_h + W- u_h + b
        u_z = W+ u_h + W- l_h + b

    For ReLU layers h = max(z, 0):
        l_h = max(l_z, 0)
        u_h = max(u_z, 0)

    The returned records are needed by the backward dual pass because each ReLU
    relaxation depends on its own pre-activation bounds [l_z, u_z].
    """
    lower, upper = initial_linf_bounds(x, eps)
    records: List[BoundRecord] = []

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
            raise TypeError(f"Unsupported layer for dual bounds: {layer}")

        records.append(BoundRecord(layer, lower_in, upper_in, lower, upper))

    return (lower, upper), records


def _backward_linear(lambda_out: torch.Tensor, layer: nn.Linear, const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward objective through z = W h + b.

    Suppose the current objective is:
        lambda_z^T z + const

    Substitute z = W h + b:
        lambda_z^T (W h + b) + const
        = (W^T lambda_z)^T h + lambda_z^T b + const

    Therefore:
        lambda_h = W^T lambda_z
        const <- const + lambda_z^T b
    """
    if layer.bias is not None:
        const = const + torch.matmul(lambda_out, layer.bias)
    lambda_in = torch.matmul(lambda_out, layer.weight)
    return lambda_in, const


def _backward_conv2d(
    lambda_out: torch.Tensor,
    layer: nn.Conv2d,
    input_shape: torch.Size,
    const: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward objective through convolution z = W * h + b.

    For an objective <lambda_z, z>, the coefficient on h is the transposed
    convolution:
        lambda_h = ConvTranspose2d(lambda_z, W)

    The bias contribution is:
        const <- const + sum_{batch,channel,height,width}(lambda_z * b)
    """
    if layer.bias is not None:
        # Sum lambda over spatial dimensions, then dot with the conv bias.
        const = const + (lambda_out.sum(dim=(2, 3)) * layer.bias).sum(dim=1)

    lambda_in = F.conv_transpose2d(
        lambda_out,
        layer.weight,
        bias=None,
        stride=layer.stride,
        padding=layer.padding,
        output_padding=0,
        dilation=layer.dilation,
        groups=layer.groups,
    )

    # With the requested MNIST architecture, output_padding=0 gives the exact
    # inverse spatial size. This crop is a safety guard if the architecture is
    # later changed.
    lambda_in = lambda_in[..., : input_shape[-2], : input_shape[-1]]
    return lambda_in, const


def _wk_backward_linear(nu_next: torch.Tensor, layer: nn.Linear, j_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wong--Kolter affine dual step for z_{i+1} = W_i z_i + b_i.

    Theorem 1 uses nu_k = -c and accumulates the specific dual objective

        J = - sum_i nu_{i+1}^T b_i
            - x^T nuhat_1
            - eps ||nuhat_1||_1
            + sum_{unstable ReLUs} l_i [nu_i]_+.

    This helper performs:
        nuhat_i = W_i^T nu_{i+1}
        J <- J - nu_{i+1}^T b_i
    """
    if layer.bias is not None:
        j_value = j_value - torch.matmul(nu_next, layer.bias)
    nuhat_prev = torch.matmul(nu_next, layer.weight)
    return nuhat_prev, j_value


def _wk_backward_conv2d(
    nu_next: torch.Tensor,
    layer: nn.Conv2d,
    input_shape: torch.Size,
    j_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wong--Kolter affine dual step for convolutional layers.

    This is the convolutional analogue of:
        nuhat_i = W_i^T nu_{i+1}
        J <- J - nu_{i+1}^T b_i
    """
    if layer.bias is not None:
        j_value = j_value - (nu_next.sum(dim=(2, 3)) * layer.bias).sum(dim=1)

    nuhat_prev = F.conv_transpose2d(
        nu_next,
        layer.weight,
        bias=None,
        stride=layer.stride,
        padding=layer.padding,
        output_padding=0,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    nuhat_prev = nuhat_prev[..., : input_shape[-2], : input_shape[-1]]
    return nuhat_prev, j_value


def _wk_backward_relu_fixed_alpha(
    nuhat: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    j_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact Wong--Kolter fixed-alpha ReLU dual step.

    For each pre-ReLU activation with bounds [l, u], the paper partitions nodes:

        I- : u <= 0       -> always inactive -> nu_i = 0
        I+ : l >= 0       -> always active   -> nu_i = nuhat_i
        I  : l < 0 < u    -> unstable

    For unstable nodes, Wong--Kolter choose the fixed dual-feasible value

        alpha = u / (u - l),

    and Theorem 1 gives

        nu_i = alpha [nuhat_i]_+ - alpha [nuhat_i]_-
             = alpha * nuhat_i.

    The specific dual objective also includes

        sum_{j in I_i} l_{i,j} [nu_{i,j}]_+.

    This differs from the CROWN-style backend above, which chooses upper or
    lower ReLU lines based on the sign of the backward coefficient.
    """
    eps = 1e-12
    active = lower >= 0
    inactive = upper <= 0
    unstable = (~active) & (~inactive)

    slope = upper / (upper - lower + eps)

    nu = torch.zeros_like(nuhat)

    # Stable active ReLU: z = s, so nu = nuhat
    nu = torch.where(active, nuhat, nu)

    # Stable inactive ReLU: z = 0, so nu = 0
    nu = torch.where(inactive, torch.zeros_like(nu), nu)

    # Unstable ReLU Wong-Kolter rule:
    # nu = alpha [nuhat]_+ + [nuhat]_-
    # where [a]_+ = max(a,0), [a]_- = min(a,0)
    positive_part = torch.clamp(nuhat, min=0)
    negative_part = torch.clamp(nuhat, max=0)
    unstable_nu = slope * positive_part + negative_part

    nu = torch.where(unstable, unstable_nu, nu)

    # WK objective term:
    # sum_{unstable} l * [nu]_+
    unstable_j_term = torch.where(
        unstable,
        lower * torch.clamp(nu, min=0),
        torch.zeros_like(nu),
    )

    j_value = j_value + unstable_j_term.view(unstable_j_term.size(0), -1).sum(dim=1)

    return nu, j_value


def _backward_relu_upper(
    lambda_out: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    const: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward dual step through h = ReLU(z), using the convex triangle relaxation.

    For each pre-activation z in [l, u]:

    Stable inactive neuron, u <= 0:
        h = 0

    Stable active neuron, l >= 0:
        h = z

    Unstable neuron, l < 0 < u:
        ReLU is relaxed by the triangle constraints:
            h >= 0
            h >= z
            h <= alpha z + beta
        where:
            alpha = u / (u - l)
            beta  = -u l / (u - l)

    We need an UPPER bound on a linear objective lambda_h * h.

    If lambda_h >= 0, use the upper ReLU line:
        lambda_h h <= lambda_h (alpha z + beta)

    If lambda_h < 0, use a valid lower ReLU line h >= alpha_l z, because
    multiplying by a negative number flips the inequality:
        lambda_h h <= lambda_h (alpha_l z)

    A common CROWN/Wong--Kolter choice is:
        alpha_l = 1 if u > |l| else 0

    The result is another linear objective in z:
        lambda_z z + const
    """
    eps = 1e-12
    active = lower >= 0
    inactive = upper <= 0
    unstable = (~active) & (~inactive)

    lambda_in = torch.zeros_like(lambda_out)

    # Stable active: h=z, so lambda_z=lambda_h.
    lambda_in = torch.where(active, lambda_out, lambda_in)

    # Stable inactive: h=0, so lambda_z=0 and no constant change.
    lambda_in = torch.where(inactive, torch.zeros_like(lambda_in), lambda_in)

    # Unstable triangle relaxation.
    alpha_u = upper / (upper - lower + eps)
    beta_u = -upper * lower / (upper - lower + eps)
    alpha_l = torch.where(upper > -lower, torch.ones_like(lower), torch.zeros_like(lower))

    use_upper_line = unstable & (lambda_out >= 0)
    use_lower_line = unstable & (lambda_out < 0)

    lambda_in = torch.where(use_upper_line, lambda_out * alpha_u, lambda_in)
    lambda_in = torch.where(use_lower_line, lambda_out * alpha_l, lambda_in)

    # Constant only comes from the upper line beta term.
    beta_term = torch.where(use_upper_line, lambda_out * beta_u, torch.zeros_like(lambda_out))
    const = const + beta_term.view(beta_term.size(0), -1).sum(dim=1)

    return lambda_in, const


def dual_upper_bound_for_objective(
    model: nn.Sequential,
    x: torch.Tensor,
    eps: float,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Upper-bound max_{x' in Linf box} c^T f(x') using a dual/backward relaxation.

    The final objective is initialized as:
        c^T z_K

    We propagate the objective backward through the relaxed network until it is
    linear in the input image x':
        c^T f(x') <= lambda_0^T x' + const

    Since x' is inside an interval box [l_0, u_0], the maximum of the final
    linear expression is available in closed form:
        max_{l_0 <= x' <= u_0} lambda_0^T x' + const
        = sum_i max(lambda_0_i l_0_i, lambda_0_i u_0_i) + const

    This is the dual-style bound used for robust training.
    """
    (input_lower, input_upper) = initial_linf_bounds(x, eps)
    (_, _), records = collect_bound_records(model, x, eps)

    lambda_cur = c
    const = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    for record in reversed(records):
        layer = record.layer

        if isinstance(layer, nn.Linear):
            lambda_cur, const = _backward_linear(lambda_cur, layer, const)

        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == "Flatten":
            lambda_cur = lambda_cur.view_as(record.lower_in)

        elif isinstance(layer, nn.ReLU):
            # ReLU bounds are on the pre-activation input to this ReLU.
            lambda_cur, const = _backward_relu_upper(lambda_cur, record.lower_in, record.upper_in, const)

        elif isinstance(layer, nn.Conv2d):
            lambda_cur, const = _backward_conv2d(lambda_cur, layer, record.lower_in.shape, const)

        else:
            raise TypeError(f"Unsupported layer for dual bounds: {layer}")

    input_linear_max = torch.maximum(lambda_cur * input_lower, lambda_cur * input_upper)
    input_linear_max = input_linear_max.view(input_linear_max.size(0), -1).sum(dim=1)
    return input_linear_max + const


def dual_wk_lower_bound_for_objective(
    model: nn.Sequential,
    x: torch.Tensor,
    eps: float,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Wong--Kolter LP-dual lower bound for min_{||delta||_inf <= eps} c^T f(x+delta).

    This follows Theorem 1 from Wong & Kolter more directly than the default
    CROWN-style backend:

        nu_k = -c
        nuhat_i = W_i^T nu_{i+1}

    Stable ReLUs are exact. Unstable ReLUs use the paper's fixed dual-feasible
    value

        alpha = u / (u - l),

    rather than the sign-dependent CROWN lower/upper line choice.

    The returned value is the paper's J_epsilon objective:

        J = - sum_i nu_{i+1}^T b_i
            - x^T nuhat_1
            - eps ||nuhat_1||_1
            + sum_{unstable hidden ReLUs} l_i [nu_i]_+.

    Therefore:

        J <= min_{||delta||_inf <= eps} c^T f(x+delta).

    Note: this uses the exact L_inf-ball support term from the paper,
    -x^T nuhat_1 - eps ||nuhat_1||_1. The interval bounds used to form ReLU
    states are still produced by collect_bound_records(...), which clips image
    bounds to [0, 1]. This is conservative for MNIST images.
    """
    (_, _), records = collect_bound_records(model, x, eps)

    # Theorem 1 initializes nu_k = -c for a minimization objective c^T z_k.
    nu_cur = -c
    j_value = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    for record in reversed(records):
        layer = record.layer

        if isinstance(layer, nn.Linear):
            nu_cur, j_value = _wk_backward_linear(nu_cur, layer, j_value)

        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == "Flatten":
            nu_cur = nu_cur.view_as(record.lower_in)

        elif isinstance(layer, nn.ReLU):
            # ReLU bounds are pre-activation bounds, i.e., the input to ReLU.
            nu_cur, j_value = _wk_backward_relu_fixed_alpha(
                nu_cur,
                record.lower_in,
                record.upper_in,
                j_value,
            )

        elif isinstance(layer, nn.Conv2d):
            nu_cur, j_value = _wk_backward_conv2d(nu_cur, layer, record.lower_in.shape, j_value)

        else:
            raise TypeError(f"Unsupported layer for Wong--Kolter dual bounds: {layer}")

    # At this point nu_cur is nuhat_1, the coefficient on the input layer.
    input_dot = (x * nu_cur).view(x.size(0), -1).sum(dim=1)
    input_l1 = nu_cur.view(x.size(0), -1).abs().sum(dim=1)
    j_value = j_value - input_dot - eps * input_l1
    return j_value


def dual_wk_margin_bounds(model: nn.Sequential, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute Wong--Kolter robust margin scores for all classes.

    For each target class j, define the true-vs-target margin:

        c_j = e_y - e_j
        margin_j(x') = z_y(x') - z_j(x') = c_j^T f(x').

    dual_wk_lower_bound_for_objective returns

        J_j <= min_{||delta||_inf <= eps} z_y(x+delta) - z_j(x+delta).

    Robust training needs an upper bound on the opposite margin z_j - z_y, so
    this function returns

        M_j = -J_j >= max_{||delta||_inf <= eps} z_j(x+delta) - z_y(x+delta).

    The true-class score is set to zero, matching the robust CE construction in
    Wong--Kolter Theorem 2: CE(-J, y).
    """
    batch_size = x.size(0)
    num_classes = 10
    scores = []
    batch_idx = torch.arange(batch_size, device=x.device)

    for j in range(num_classes):
        c = torch.zeros(batch_size, num_classes, device=x.device, dtype=x.dtype)
        c[batch_idx, y] = 1.0
        c[:, j] -= 1.0
        j_lower = dual_wk_lower_bound_for_objective(model, x, eps, c)
        scores.append(-j_lower)

    margin_upper = torch.stack(scores, dim=1)
    margin_upper = margin_upper.clone()
    margin_upper[batch_idx, y] = 0.0
    return margin_upper


def dual_margin_bounds(model: nn.Sequential, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute dual upper bounds for all class margins.

    For each class j, define:
        c_j = e_j - e_y
        margin_j(x') = z_j(x') - z_y(x') = c_j^T f(x')

    This function returns:
        M[b, j] >= max_{||delta||_inf <= eps} margin_j(x_b + delta)

    The true-class margin is forced to zero:
        M[b, y_b] = 0

    Robust training minimizes CE(M, y), where M[b, y] = 0 and
    M[b, j] is an upper bound on z_j - z_y. This encourages all
    wrong-class margin upper bounds to become negative.
    """
    batch_size = x.size(0)
    num_classes = 10
    bounds = []

    for j in range(num_classes):
        c = torch.zeros(batch_size, num_classes, device=x.device, dtype=x.dtype)
        c[:, j] = 1.0
        c[torch.arange(batch_size, device=x.device), y] -= 1.0
        bounds.append(dual_upper_bound_for_objective(model, x, eps, c))

    margin_upper = torch.stack(bounds, dim=1)
    # margin_upper[torch.arange(batch_size, device=x.device), y] = 0.0
    batch_idx = torch.arange(batch_size, device=x.device)
    margin_upper = margin_upper.clone()
    margin_upper[batch_idx, y] = 0.0
    return margin_upper
