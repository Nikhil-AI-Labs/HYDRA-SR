"""
Degradation Predictor Network for blind super-resolution.

A lightweight 3-layer CNN + Global Average Pooling network (~0.2M params)
that estimates degradation parameters from the LR input image.

Outputs:
  1. d_hat ∈ ℝ⁴: degradation vector [σ_blur, σ_noise, q_JPEG, s_downsample]
     Used for degradation-aware loss supervision during training.
  2. p_d ∈ ℝ^prompt_dim: 128-dimensional degradation prompt
     Injected via FiLM modulation into every processing block.

This makes HYDRA-SR a *blind* SR model that handles in one weight set:
  - Bicubic (clean) degradation
  - Real-world composite degradations (from Real-ESRGAN pipeline)
  - JPEG compression artifacts
  - Unknown degradations (NTIRE 2026 perceptual track)

Critical training note (from implementation plan §6, Pitfall #7):
  The predictor MUST be trained jointly from Stage 1 epoch 0 with a
  synthetic Real-ESRGAN pipeline mixed into training data (10% real
  degradation in Stage 1, 100% in Stage 2+).
  Bolting it on later does NOT help — it needs co-adaptation.

Shape contract:
  lr:    (B, 3, H, W)   — LR input image, values in [0, 1]
  d_hat: (B, 4)          — predicted degradation parameters
  p_d:   (B, prompt_dim) — degradation prompt for FiLM injection
"""

import torch
import torch.nn as nn


class DegradationPredictor(nn.Module):
    """
    3-layer strided CNN + Global Average Pooling → dual heads.

    Architecture:
        3→32, stride 2, GELU
        32→64, stride 2, GELU
        64→128, stride 2, GELU
        AdaptiveAvgPool2d(1)
        Flatten
        → head_d: Linear(128, 4)      — degradation parameter regression
        → head_p: Linear(128, 128)    — degradation prompt for FiLM

    Total parameters: ≈ 0.19 M
    """

    def __init__(self, prompt_dim: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Head 1: predict degradation parameters (regression)
        # Output interpretation:
        #   d[0] = σ_blur    in [0, 5]   (blur sigma)
        #   d[1] = σ_noise   in [0, 50]  (noise std, pixel scale 0..255)
        #   d[2] = q_JPEG    in [0, 100] (JPEG quality)
        #   d[3] = s_ds      in [0, 1]   (additional downsample factor)
        self.head_d = nn.Linear(128, 4, bias=True)

        # Head 2: produce degradation prompt for FiLM injection
        self.head_p = nn.Linear(128, prompt_dim, bias=True)

        # Initialize heads with small weights for training stability
        nn.init.xavier_uniform_(self.head_d.weight, gain=0.1)
        nn.init.zeros_(self.head_d.bias)
        nn.init.xavier_uniform_(self.head_p.weight, gain=0.1)
        nn.init.zeros_(self.head_p.bias)

    def forward(
        self,
        lr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lr: (B, 3, H, W) LR input image.

        Returns:
            d_hat: (B, 4)          — predicted degradation parameters.
            p_d:   (B, prompt_dim) — degradation prompt for FiLM.
        """
        f = self.backbone(lr)       # (B, 128)
        d_hat = self.head_d(f)      # (B, 4)
        p_d   = self.head_p(f)      # (B, prompt_dim)
        return d_hat, p_d
