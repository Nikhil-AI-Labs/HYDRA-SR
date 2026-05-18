"""
Charbonnier loss — smooth L1 approximation for image restoration.

L_charb(x, y) = sqrt((x - y)^2 + ε^2)

Properties vs L1/L2:
  - More robust to outliers than L2 (like L1)
  - Smooth gradient everywhere unlike L1 (no discontinuity at 0)
  - ε controls the smoothness: small ε → L1-like, large ε → L2-like

ε = 1e-3 is the standard for SR (from NAFNet, SwinIR).

Used in Stage 1 (geometry lock) training.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier (pseudo-Huber) loss.

    Args:
        eps: Smoothness parameter (default 1e-3).
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()
