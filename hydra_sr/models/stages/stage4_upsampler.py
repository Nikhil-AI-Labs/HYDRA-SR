"""
Stage 4: Frequency-Gated Upsampler.

Converts processed features (B, C, H, W) → HR output (B, 3, 4H, 4W).

Components:
1. PixelShuffle ×4 — upsampling from LR to HR resolution
   Uses sub-pixel convolution (Shi et al., 2016) which is the standard
   for SR. The conv before PixelShuffle learns which HR pixels to assemble.

2. Laplacian Pyramid Sharpening — lightweight detail recovery after PixelShuffle
   Recovers ~0.3 dB on textured images at negligible parameter cost.

3. Difficulty-aware blend (from MELD-SR) — routes hard regions (high-frequency)
   to the pixel-shuffle path and easy regions (smooth) to the bicubic baseline.
   This is MELD-SR's "difficulty-aware residual" idea, kept because it works.

4. Bicubic global residual — add bicubic(LR) to final output.
   Prevents color drift (Failure Mode #8 in implementation plan).
   Ensures the network only needs to learn the residual correction, not
   the entire image content.

Shape contract:
  f_in: (B, C,    H,   W)   — final processed features
  lr:   (B, 3, H,   W)   — original LR input (for bicubic residual)
  out:  (B, 3, 4H, 4W)   — SR output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.laplacian_pyramid import LaplacianPyramidSharpening


class FreqGatedUpsampler(nn.Module):
    """
    Frequency-gated ×4 upsampler.

    Args:
        in_dim:  Number of input feature channels.
        scale:   Upsampling factor (default 4).
        mid_dim: Intermediate channels before PixelShuffle.
    """

    def __init__(self, in_dim: int, scale: int = 4, mid_dim: int = None):
        super().__init__()
        self.scale = scale
        mid_dim = mid_dim or in_dim

        # PixelShuffle path: conv → PixelShuffle
        self.conv_before_ps = nn.Conv2d(in_dim, mid_dim * scale * scale, 3, 1, 1)
        self.pixel_shuffle   = nn.PixelShuffle(scale)           # (B, mid_dim, 4H, 4W)
        self.conv_after_ps   = nn.Conv2d(mid_dim, 3, 3, 1, 1)  # → (B, 3, 4H, 4W)

        # Laplacian sharpening on the pixel-shuffle output
        self.laplacian = LaplacianPyramidSharpening(channels=3, levels=2)

        # Difficulty-aware frequency gating (from MELD-SR)
        # Estimates per-pixel "difficulty" (high-freq content) from LR image
        # and routes hard pixels to pixel-shuffle path, easy to bicubic.
        self.difficulty_estimator = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),      # → (B, 1, H, W) difficulty map at LR scale
            nn.Sigmoid(),
        )

        # Bicubic residual blend factor (learnable, initialized to 0.5)
        self.blend_alpha = nn.Parameter(torch.tensor(0.5))

        # Initialize pixel-shuffle conv with orthogonal init for stability
        nn.init.orthogonal_(self.conv_before_ps.weight)
        nn.init.zeros_(self.conv_before_ps.bias)
        nn.init.orthogonal_(self.conv_after_ps.weight)
        nn.init.zeros_(self.conv_after_ps.bias)

    def forward(
        self,
        f_in: torch.Tensor,
        lr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f_in: (B, C, H, W)  — processed features from Stage 3
            lr:   (B, 3, H, W)  — original LR image

        Returns:
            sr: (B, 3, 4H, 4W) — super-resolved output
        """
        H, W = f_in.shape[-2], f_in.shape[-1]
        scale = self.scale

        # ── Bicubic baseline ──────────────────────────────────────────────
        bicubic_hr = F.interpolate(
            lr, scale_factor=scale, mode='bicubic', align_corners=False
        )  # (B, 3, 4H, 4W)

        # ── PixelShuffle path ─────────────────────────────────────────────
        ps_out = self.pixel_shuffle(self.conv_before_ps(f_in))  # (B, mid_dim, 4H, 4W)
        ps_out = self.conv_after_ps(ps_out)                      # (B, 3, 4H, 4W)

        # ── Laplacian sharpening ──────────────────────────────────────────
        ps_out = self.laplacian(ps_out)

        # ── Difficulty-aware blend ────────────────────────────────────────
        # difficulty_map: (B, 1, H, W) at LR scale → upsample to HR scale
        diff_map = self.difficulty_estimator(lr)
        diff_map_hr = F.interpolate(
            diff_map, scale_factor=scale, mode='bilinear', align_corners=False
        )  # (B, 1, 4H, 4W)

        # Hard regions (high difficulty) → pixel-shuffle output (more detail)
        # Easy regions (low difficulty) → bicubic (color-preserving)
        #
        # CORRECT formulation: sr = bicubic + correction
        #   where correction = alpha * diff_map * (ps_out - bicubic)
        # This guarantees:
        #   - smooth regions (diff_map ≈ 0): sr = bicubic  (exactly, no drift)
        #   - hard regions (diff_map ≈ 1):   sr = bicubic + alpha*(ps_out - bicubic)
        #
        # WRONG old code: sr = (alpha*diff)*ps_out + (1-alpha*diff)*bicubic + 0.1*bicubic
        # In smooth regions: sr = 0 + 1*bicubic + 0.1*bicubic = 1.1*bicubic  ← 10% drift!
        alpha = self.blend_alpha.clamp(0.0, 1.0)
        correction = alpha * diff_map_hr * (ps_out - bicubic_hr)
        sr = bicubic_hr + correction

        return sr
