from __future__ import annotations

import torch
import torch.nn.functional as F


def _choose_targets_from_logits(logits: torch.Tensor, y: torch.Tensor, max_targets: int):
    """
    Choose the most dangerous target classes using clean logits.
    """
    true_class = int(y.item())
    logits_1d = logits.detach().view(-1)

    targets = [c for c in range(logits_1d.numel()) if c != true_class]
    targets = sorted(targets, key=lambda c: float(logits_1d[c].item()), reverse=True)

    return targets[:max_targets]


def differentiable_socp_surrogate_loss(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
    step_size: float = 0.01,
    max_targets: int = 1,
):
    """
    Differentiable unrolled robust-margin surrogate.

    This replaces the non-differentiable CVXPY certificate during training.

    Important:
        This is differentiable and can train the network.
        It is not yet a mathematically certified SOCP upper bound.
        Use the CVXPY SOCP verifier only for post-hoc certification/evaluation.
    """
    model.train()

    with torch.no_grad():
        clean_logits = model(x)
        targets = _choose_targets_from_logits(clean_logits, y, max_targets)

    delta = torch.zeros_like(x, requires_grad=True)

    for _ in range(steps):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        logits_adv = model(x_adv)

        true_logit = logits_adv[:, y.item()]
        margins = []

        for target in targets:
            target_logit = logits_adv[:, target]
            margins.append(target_logit - true_logit)

        margin_tensor = torch.stack(margins, dim=1)
        inner_loss = margin_tensor.max(dim=1).values.mean()

        grad = torch.autograd.grad(
            inner_loss,
            delta,
            retain_graph=False,
            create_graph=False,
        )[0]

        with torch.no_grad():
            delta += step_size * grad.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.data = torch.clamp(x + delta.data, 0.0, 1.0) - x

        delta.requires_grad_(True)

    x_adv = torch.clamp(x + delta, 0.0, 1.0)
    logits_adv = model(x_adv)

    true_logit = logits_adv[:, y.item()]
    final_margins = []

    for target in targets:
        target_logit = logits_adv[:, target]
        final_margins.append(target_logit - true_logit)

    final_margins = torch.stack(final_margins, dim=1)
    worst_margin = final_margins.max(dim=1).values.mean()

    cert_loss = F.softplus(worst_margin)

    return cert_loss, worst_margin