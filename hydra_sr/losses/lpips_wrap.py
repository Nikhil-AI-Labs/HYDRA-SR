"""
LPIPS perceptual loss wrapper.

Thin wrapper around the richzhang/PerceptualSimilarity LPIPS implementation
with proper normalization (HYDRA-SR uses [0,1] images, LPIPS expects [-1,1]).

Used in Stage 3 perceptual training with weight 1.0.
"""

import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss.

    Args:
        net:    Feature network backbone. 'alex' (faster) or 'vgg' (stronger).
                Use 'alex' for training speed, 'vgg' for final refinement.
        weight: Loss weight multiplier (default 1.0).
    """

    def __init__(self, net: str = 'alex', weight: float = 1.0):
        super().__init__()
        self.weight = weight
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net, verbose=False)
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
            self._available = True
        except ImportError:
            self._available = False
            import torch.nn.functional as F
            self._fallback_fn = lambda x, y: F.l1_loss(x, y)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr: (B, 3, H, W) in [0, 1]
            hr: (B, 3, H, W) in [0, 1]
        """
        if self._available:
            sr_norm = sr * 2.0 - 1.0   # [0,1] → [-1,1]
            hr_norm = hr * 2.0 - 1.0
            return self.lpips_fn(sr_norm, hr_norm).mean() * self.weight
        else:
            return self._fallback_fn(sr, hr) * self.weight
