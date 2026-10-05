"""
SOCP-certified loss helpers.

The paper's loss is

    L(theta) = CE(f_theta(x), y) + lambda * ell_cert(theta; x, y)

where ell_cert is formed from SOCP class-wise upper bounds.

Current implementation note:
    The exact CVXPY SOCP solve is not differentiable in this code.  Therefore
    the returned certificate loss is detached from PyTorch autograd.  This file
    is still useful for logging, post-hoc evaluation, and for the training-loop
    interface that will later be connected to an unrolled/implicit differentiable
    conic solver.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from SOCP.socp_certificate import CertificateConfig, certify_sample


def socp_margin_ce_loss_single(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: CertificateConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wong-Kolter-style SOCP robust cross entropy.

    Builds J_SOCP where:
        J[y] = 0
        J[t] = phi_SOCP(x, y, t)

    phi_SOCP is computed by the sparse SOCP verifier using:
        - LP ReLU convex hulls
        - lifted variables eta/sigma/tau/zeta/xi
        - McCormick envelopes
        - SOC 2x2 PSD-minor constraints

    Important:
        This uses CVXPY, so the SOCP certificate value is detached from
        PyTorch autograd. It is the correct loss value structurally, but not
        yet differentiable through the SOCP solve.
    """
    if x.size(0) != 1:
        raise ValueError("socp_margin_ce_loss_single expects batch size 1.")

    true_class = int(y.item())

    cert = certify_sample(model, x, true_class, cfg)

    with torch.no_grad():
        logits = model(x)
        num_classes = logits.size(1)

    J = torch.zeros((1, num_classes), device=x.device, dtype=logits.dtype)

    for target_class, result in cert.margins.items():
        J[0, int(target_class)] = float(result.value)

    J[0, true_class] = 0.0

    robust_ce = F.cross_entropy(J, y)

    worst_margin = torch.tensor(
        cert.worst_margin,
        device=x.device,
        dtype=logits.dtype,
    )

    return robust_ce, worst_margin, J


def socp_total_loss_single(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: CertificateConfig,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total AAAI/WK-style objective:

        L = (1 - alpha) * CE(clean logits, y)
            + alpha * CE(J_SOCP, y)

    alpha controls the clean/robust tradeoff.
    """
    logits = model(x)
    clean_ce = F.cross_entropy(logits, y)

    robust_ce, worst_margin, J = socp_margin_ce_loss_single(
        model=model,
        x=x,
        y=y,
        cfg=cfg,
    )
    #Start wwith a low alpha and gradually increase as clean acc gets better
    if(clean_ce < 1.0):
        alpha = 0.5

    total_loss = (1.0 - alpha) * clean_ce + alpha * robust_ce

    return total_loss, clean_ce, robust_ce, worst_margin