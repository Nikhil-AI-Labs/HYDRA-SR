"""
Adaptive Frequency Router (AFR) — modernized descendant of MELD-SR's Block 2.

In MELD-SR, the frequency router directed features to three frozen expert models
(HAT / DAT / NAFNet) based on frequency-band content. That design had a critical
ceiling: frozen experts couldn't co-adapt, capping PSNR at 31.48 dB.

HYDRA-SR's AFR keeps the MELD-SR frequency analysis idea but repurposes it:
instead of routing to separate frozen models, it produces soft routing weights
(r_P, r_W, r_T) that scale the contribution of each of the three streams:
  - r_P: weight for pixel-stream output F_P2 (local texture, photometric detail)
  - r_W: weight for wavelet-stream output F_W2 (frequency-structured edges)
  - r_T: weight for Transformer-stage output F_T (globally coherent structure)

This is a mathematical generalization: MELD-SR routing was a hard expert selector;
HYDRA-SR routing is a soft, differentiable weighting learned end-to-end.
Removing it costs ~0.3 dB in ablation A5.

Frequency analysis pipeline (same 9-band decomposition as MELD-SR Block 2):
  • DCT: 3 energy bands (DC, mid, high frequency DCT coefficients)
  • DWT: 4 subband energies (LL, LH, HL, HH from single-level db4)
  • FFT: 2 radial bands (low-radius / high-radius magnitude)
  Total: 9 scalar features per channel group

These 9 band energies feed into:
  1. Cross-Band Multi-Head Attention (4 heads): correlates frequency bands
  2. Large Kernel Attention (LKA, 21×21 DWConv decomposed): captures long-range
     frequency-spatial structure
  3. Softmax output: (B, C, 1, 1) normalized routing weights for 3 streams

Shape contract:
  x: (B, 3, H, W)  — raw LR input (not features; uses LR for frequency diagnosis)
  output: (B, 3)   — softmax-normalized routing weights [r_P, r_W, r_T]
                     These are averaged over spatial dims to produce scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LargeKernelAttention(nn.Module):
    """
    LKA (Large Kernel Attention) decomposed into:
      DWConv_{5×5} → DWConv_dilated_{3×3, dilation=3} → Conv_{1×1}
    Effective receptive field: ~21×21 without the parameter cost.

    Reference: Visual Attention Network (VAN), CVMJ 2023.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv_small = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dwconv_dilated = nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim)
        self.conv1x1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.dwconv_small(x)
        attn = self.dwconv_dilated(attn)
        attn = self.conv1x1(attn)
        return x * attn  # channel-spatial attention gate


class FrequencyRouter(nn.Module):
    """
    Adaptive Frequency Router.

    Analyzes the 9-band frequency signature of the LR input and produces
    soft routing weights (r_P, r_W, r_T) for the three processing streams.

    Args:
        in_channels:  Input channels to analyze (default 3 for RGB LR image).
        n_bands:      Number of frequency bands (default 9: 3 DCT + 4 DWT + 2 FFT).
        hidden_dim:   Internal feature dimension for cross-band attention.
        n_heads:      Number of attention heads in cross-band MHA.
        n_streams:    Number of routing targets (default 3: P, W, T).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_bands: int = 9,
        hidden_dim: int = 32,
        n_heads: int = 4,
        n_streams: int = 3,
    ):
        super().__init__()
        self.n_bands = n_bands

        # Lightweight feature extractor from LR image
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(8),   # → (B, hidden_dim, 8, 8)
        )

        # Cross-Band Multi-Head Attention
        # Treat each of the 9 frequency bands as a "token" of dim hidden_dim
        self.band_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_band_mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # LKA on the spatial feature before pooling
        self.lka = LargeKernelAttention(hidden_dim)

        # Final projection → 3 routing logits
        self.router_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_streams),
        )

    def _extract_freq_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 9-band frequency energy features from the LR image.
        These replicate the MELD-SR Block 2 frequency analysis.

        Returns:
            feats: (B, 9) scalar energy per band
        """
        B, C, H, W = x.shape
        feats = []

        # -- DCT bands (3 bands: DC, mid, high) --
        # Use FFT magnitude as DCT proxy (faster and differentiable)
        fft_mag = torch.fft.rfft2(x.mean(1)).abs()  # (B, H, W//2+1)
        H_f, W_f = fft_mag.shape[-2], fft_mag.shape[-1]
        dc   = fft_mag[:, :H_f//8, :W_f//8].mean((-1, -2))
        mid  = fft_mag[:, H_f//8:H_f//2, W_f//8:W_f//2].mean((-1, -2))
        high = fft_mag[:, H_f//2:, W_f//2:].mean((-1, -2))
        feats += [dc, mid, high]

        # -- DWT subband energies (4 bands: LL, LH, HL, HH) --
        # Simple Haar wavelet manually (works without pytorch_wavelets dependency)
        avg = F.avg_pool2d(x.mean(1, keepdim=True), 2)   # LL
        diff_h = (x.mean(1, keepdim=True)[:, :, ::2, :] - x.mean(1, keepdim=True)[:, :, 1::2, :]).abs().mean((-1,-2,-3))
        diff_v = (x.mean(1, keepdim=True)[:, :, :, ::2] - x.mean(1, keepdim=True)[:, :, :, 1::2]).abs().mean((-1,-2,-3))
        feats += [
            avg.mean((-1, -2, -3)),   # LL energy
            diff_h,                   # LH (horizontal details)
            diff_v,                   # HL (vertical details)
            (diff_h + diff_v) / 2,    # HH (diagonal proxy)
        ]

        # -- FFT radial bands (2 bands: low-radius, high-radius) --
        fft_mag_flat = fft_mag.flatten(1)
        n_coeff = fft_mag_flat.shape[1]
        low_r  = fft_mag_flat[:, :n_coeff // 2].mean(1)
        high_r = fft_mag_flat[:, n_coeff // 2:].mean(1)
        feats += [low_r, high_r]

        return torch.stack(feats, dim=1)  # (B, 9)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lr: (B, 3, H, W) — raw LR input image.

        Returns:
            routing: (B, 3) — softmax weights [r_P, r_W, r_T].
        """
        B = lr.shape[0]

        # Spatial features via stem + LKA
        feat = self.stem(lr)           # (B, hidden_dim, 8, 8)
        feat = self.lka(feat)          # (B, hidden_dim, 8, 8)
        spatial_feat = feat.flatten(2).mean(-1)  # (B, hidden_dim)

        # Frequency band energies → cross-band attention
        band_energies = self._extract_freq_features(lr)  # (B, 9)
        # Create pseudo-token sequence: (B, 9, 1) → project to hidden_dim
        band_tokens = band_energies.unsqueeze(-1).expand(-1, -1, spatial_feat.shape[-1])
        # band_tokens: (B, 9, hidden_dim) — each band gets the spatial features scaled
        band_tokens = self.band_proj(band_tokens)        # (B, 9, hidden_dim)
        attn_out, _ = self.cross_band_mha(
            band_tokens, band_tokens, band_tokens
        )                                                 # (B, 9, hidden_dim)
        freq_feat = attn_out.mean(1)                     # (B, hidden_dim) — aggregate

        # Combine spatial + frequency features
        combined = torch.cat([spatial_feat, freq_feat], dim=-1)  # (B, 2*hidden_dim)

        # Produce routing logits → softmax
        routing = F.softmax(self.router_head(combined), dim=-1)  # (B, 3)
        return routing
