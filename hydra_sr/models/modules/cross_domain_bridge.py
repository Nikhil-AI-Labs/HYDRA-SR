"""
Cross-Domain Bridge (CDB) — bidirectional information exchange
between the pixel stream (P) and the wavelet stream (W).

Architecture:
    CDB operates at every "stage boundary" (between Stage 1 and Stage 2,
    between Stage 2 and Stage 3). Two bridges: CDB-1, CDB-2.

Forward pass (per CDB):
    P → W:  DWT(F_P) → 1×1 Conv → add to F_W (weighted by λ_W)
    W → P:  Project F_W channels → bilinear upsample 4× → add to F_P (weighted by λ_P)

Mathematical form (from HYDRA-SR architecture spec):
    F_P_new = F_P + λ_P · clamp · Conv_{1×1}(Bilinear4×(F_W_proj))
    F_W_new = F_W + λ_W · clamp · Conv_{1×1}(DWT(F_P)_stacked)

Critical initialization (Pitfall #4 in implementation plan):
    λ_P = λ_W = 0 at init. Clamped to [0, 0.5] for first 5K iters.
    Without this: NaN gradients within 200 iterations (confirmed in MELD-SR).
    The clamp is applied INSIDE forward() and is always active (not just early).

Why DWT for P→W instead of straight pooling:
    We want to inject STRUCTURAL frequency information (edges, textures)
    from the pixel domain into the wavelet domain, not spatial statistics.
    DWT decomposes F_P into the same frequency representation as F_W,
    allowing channel-aligned fusion via 1×1 conv.

Why bilinear upsample for W→P instead of iDWT:
    F_W has custom channel layout (not the original 3-channel image).
    Performing iDWT requires knowing the original LL/LH/HL/HH partition,
    which is lost after the Stage 1-W processing. Bilinear upsample is
    semantically equivalent for feature-level fusion.

Shape contract:
    F_P: (B, C_P, H,   W)    — pixel stream at full spatial resolution
    F_W: (B, C_W, H/4, W/4)  — wavelet stream at 2-level DWT resolution

    Output:
    F_P_new: (B, C_P, H,   W)    — augmented pixel stream
    F_W_new: (B, C_W, H/4, W/4)  — augmented wavelet stream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward
    _WAVELETS_AVAILABLE = True
except ImportError:
    _WAVELETS_AVAILABLE = False


class CrossDomainBridge(nn.Module):
    """
    Cross-Domain Bridge: fuses pixel stream ↔ wavelet stream.

    Args:
        C_P:  Number of channels in the pixel stream.
        C_W:  Number of channels in the wavelet stream.
        J:    DWT decomposition levels (must match the HYDRA-SR global setting).
              J=2 → 2-level Daubechies db4 → F_W is at H/4, W/4.
        wave: Wavelet family (default 'db4', as in HYDRA-SR architecture).
    """

    def __init__(
        self,
        C_P: int,
        C_W: int,
        J: int = 2,
        wave: str = 'db4',
    ):
        super().__init__()
        self.J = J

        # For J=2 db4 DWT of F_P (3-channel input becomes stacked subbands):
        #   LL level-2:    C_P channels
        #   LH, HL, HH at level-2:  3 × C_P channels
        #   LH, HL, HH at level-1:  3 × C_P channels
        # Total: C_P × (1 + 3×J) channels
        dwt_out_channels = C_P * (1 + 3 * J)

        if _WAVELETS_AVAILABLE:
            self.dwt  = DWTForward(J=J, wave=wave, mode='periodization')
        else:
            self.dwt = None   # fallback: use average pooling

        # P → W: 1×1 conv to project DWT(F_P) channels to C_W
        self.p2w_conv = nn.Conv2d(dwt_out_channels, C_W, kernel_size=1, bias=True)

        # W → P: 1×1 conv to project F_W channels to C_P (before upsample)
        self.w2p_conv = nn.Conv2d(C_W, C_P, kernel_size=1, bias=True)

        # Learnable coupling strengths — initialized to 0 (identity at start)
        # Clamped to [0, 0.5] during forward to prevent explosive fusion
        self.lam_p = nn.Parameter(torch.zeros(1))
        self.lam_w = nn.Parameter(torch.zeros(1))

    def _stack_dwt_subbands(
        self,
        yl: torch.Tensor,
        yh: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Stack all DWT subbands into a single tensor along the channel dim.

        pytorch_wavelets returns a spatial pyramid, not same-size tensors:
            yl:    (B, C, H/4, W/4) — deepest LL
            yh[0]: (B, C, 3, H/2, W/2) — level-1 details (FINER)
            yh[1]: (B, C, 3, H/4, W/4) — level-2 details (COARSER)

        We canonicalise to H/4 (the yl resolution) by avg-pooling any
        finer-level subbands. This matches the wavelet stream spatial size
        and avoids RuntimeError on torch.cat.

        Args:
            yl: (B, C, H/4, W/4)
            yh: list of (B, C, 3, H', W') — H' may differ across levels

        Returns:
            stacked: (B, C*(1+3J), H/4, W/4)
        """
        target_h, target_w = yl.shape[-2], yl.shape[-1]
        parts = [yl]
        for yh_level in yh:
            for k in range(3):
                subband = yh_level[:, :, k, :, :]  # (B, C, H', W')
                if subband.shape[-2] != target_h or subband.shape[-1] != target_w:
                    stride = subband.shape[-1] // target_w
                    subband = F.avg_pool2d(subband, kernel_size=stride)
                parts.append(subband)
        return torch.cat(parts, dim=1)

    def _dwt_or_pool(self, F_P: torch.Tensor) -> torch.Tensor:
        """DWT F_P into subband stack. Falls back to avg-pool if pytorch_wavelets missing."""
        if self.dwt is not None:
            yl, yh = self.dwt(F_P)
            return self._stack_dwt_subbands(yl, yh)
        else:
            # Fallback: simple spatial downsampling (loses frequency structure)
            H_out = F_P.shape[2] // (2 ** self.J)
            W_out = F_P.shape[3] // (2 ** self.J)
            pooled = F.adaptive_avg_pool2d(F_P, (H_out, W_out))
            # Repeat channels to match expected DWT output channel count
            n_reps = 1 + 3 * self.J
            return pooled.repeat(1, n_reps, 1, 1)

    def forward(
        self,
        F_P: torch.Tensor,
        F_W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-domain fusion.

        Args:
            F_P: (B, C_P, H,   W)    pixel-domain feature map
            F_W: (B, C_W, H/4, W/4)  wavelet-domain feature map

        Returns:
            F_P_new: (B, C_P, H,   W)
            F_W_new: (B, C_W, H/4, W/4)
        """
        # Clamp coupling strengths to prevent explosive fusion
        lam_p = self.lam_p.clamp(0.0, 0.5)
        lam_w = self.lam_w.clamp(0.0, 0.5)

        # ── P → W: inject pixel-domain structure into wavelet stream ───
        stacked = self._dwt_or_pool(F_P)          # (B, C_P*(1+3J), H/4, W/4)
        # DWT of F_P may round differently from the original wavelet stream.
        # e.g. F_P=510px → DWT(F_P) LL = 128px, but F_W = 127px.
        # Clamp to exactly match F_W spatial size so cat/add doesn't crash.
        if stacked.shape[-2:] != F_W.shape[-2:]:
            stacked = F.adaptive_avg_pool2d(stacked, F_W.shape[-2:])
        F_W_new = F_W + lam_w * self.p2w_conv(stacked)

        # ── W → P: inject wavelet-domain features into pixel stream ────
        # Project channels then upsample to EXACTLY F_P's spatial size.
        # Using scale_factor=4 causes ±2px errors when H is not divisible by 4
        # (e.g. LR=510 → W-stream H/4=127 → 127*4=508 ≠ 510).
        f_w_proj = self.w2p_conv(F_W)             # (B, C_P, H/4, W/4)
        f_w_up = F.interpolate(
            f_w_proj,
            size=(F_P.shape[-2], F_P.shape[-1]),  # exact target — no rounding error
            mode='bilinear',
            align_corners=False,
        )                                          # (B, C_P, H, W) — guaranteed match
        F_P_new = F_P + lam_p * f_w_up

        return F_P_new, F_W_new
