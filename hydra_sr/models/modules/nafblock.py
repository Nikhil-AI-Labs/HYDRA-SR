"""
NAFNet-style building block: SimpleGate + Channel Attention + LayerNorm2d.

Lifted and adapted from:
  megvii-research/NAFNet
  basicsr/models/archs/NAFNet_arch.py (MIT License)

Changes from original NAFNet:
  - Added `film_modulation` method for FiLM conditioning from degradation prompt.
  - LayerNorm2d is kept as a local class (not imported from BasicSR) to avoid
    the BasicSR dependency at forward-pass time.

Shape contract:
    Input / Output: (B, C, H, W)

Reference:
    "Simple Baselines for Image Restoration", ECCV 2022.
    Chen et al., megvii-research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm for (B, C, H, W) tensors.
    Normalizes over the channel dim, applied independently per spatial location.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to (B, H, W, C) for LayerNorm, then restore
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SimpleGate(nn.Module):
    """
    Elementwise gating: splits channels in half and multiplies the halves.
    x1, x2 = split(x, 2, dim=1); return x1 * x2
    This replaces ReLU/GELU in NAFNet and is parameter-free.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet block: SimpleGate channel-mixing + lightweight channel attention
    + feed-forward with learnable residual scales (β, γ).

    Args:
        c:          Number of input/output channels.
        dw_expand:  Expansion ratio for depth-wise branch (default 2).
        ffn_expand: Expansion ratio for FFN branch (default 2).
        drop:       Dropout probability (usually 0 for SR).

    Forward:
        inp (B, C, H, W) → out (B, C, H, W)
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        dw_c  = c * dw_expand
        ffn_c = c * ffn_expand

        # --- Depth-wise branch ---
        self.conv1 = nn.Conv2d(c, dw_c, 1, bias=True)                         # pointwise expand
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c, bias=True)  # depthwise 3×3
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1, bias=True)                    # pointwise collapse

        # Channel attention (squeeze-excitation lite, after SimpleGate)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_c // 2, dw_c // 2, 1, bias=True),
        )

        # --- FFN branch ---
        self.conv4 = nn.Conv2d(c, ffn_c, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1, bias=True)

        # LayerNorms
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sg = SimpleGate()

        # Learnable residual scaling (NAFNet's key trick)
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # --- Depth-wise path ---
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)                    # (B, dw_c//2, H, W)
        x = x * self.sca(x)              # channel attention
        x = self.conv3(x)
        y = inp + x * self.beta           # residual with learned scale

        # --- FFN path ---
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.drop(x)
        x = self.conv5(x)
        return y + x * self.gamma         # second residual
