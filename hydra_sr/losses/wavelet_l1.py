"""
Wavelet L1 loss — supervises the SR output in the wavelet domain.

Applied on DWT subbands of SR vs HR, with band-specific weights:
  LL:  0.5  (smooth regions — already well-supervised by pixel-space L1)
  LH:  1.5  (horizontal edges — harder to recover, upweight)
  HL:  1.5  (vertical edges)
  HH:  2.0  (diagonal/high-freq — hardest, highest weight)

This forces the model to recover high-frequency wavelet coefficients
explicitly, complementing pixel-space losses which are dominated by
low-frequency (smooth) content.

Used in Stage 2 (frequency lock) training.

Reference:
    Adapts the frequency-specific loss weighting from DTWSR (ICCV 2025)
    which demonstrated that band-weighted wavelet supervision is crucial
    for recovering fine-grained textures in real-world SR.
"""

import torch
import torch.nn as nn

try:
    from pytorch_wavelets import DWTForward
    _WAVELETS_AVAILABLE = True
except ImportError:
    _WAVELETS_AVAILABLE = False


class WaveletL1Loss(nn.Module):
    """
    L1 loss computed on DWT subbands with frequency-specific weights.

    Args:
        wave:         Wavelet family (default 'db4' matching HYDRA-SR backbone).
        J:            Decomposition levels (default 2).
        band_weights: Dict of per-band loss weights.
                      Keys: 'LL', 'LH', 'HL', 'HH'
                      Values: scalar weights (should sum roughly to 1).
    """

    DEFAULT_WEIGHTS = {'LL': 0.5, 'LH': 1.5, 'HL': 1.5, 'HH': 2.0}

    def __init__(
        self,
        wave: str = 'db4',
        J: int = 2,
        band_weights: dict = None,
    ):
        super().__init__()
        self.J = J
        self.band_weights = band_weights or self.DEFAULT_WEIGHTS

        if _WAVELETS_AVAILABLE:
            self.dwt = DWTForward(J=J, wave=wave, mode='reflect')
        else:
            self.dwt = None

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr: (B, 3, H, W) — super-resolved output
            hr: (B, 3, H, W) — ground truth HR

        Returns:
            loss: scalar wavelet L1 loss
        """
        if self.dwt is None:
            # Fallback: simple frequency-weighted L1 via 2D difference
            # (Used on CPU/without pytorch_wavelets)
            diff = (sr - hr).abs()
            # Approximate HF content via Laplacian
            laplac = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],
                                   dtype=sr.dtype, device=sr.device)
            laplac = laplac.view(1,1,3,3).expand(3,-1,-1,-1)
            sr_hf = torch.nn.functional.conv2d(sr, laplac, padding=1, groups=3).abs()
            return (diff * (1 + sr_hf.detach())).mean()

        # Compute DWT of both SR and HR
        sr_yl, sr_yh = self.dwt(sr)   # sr_yl: (B,3,H/4,W/4), sr_yh: list of (B,3,3,H',W')
        hr_yl, hr_yh = self.dwt(hr)

        # LL subband loss
        ll_w = self.band_weights.get('LL', 0.5)
        loss = ll_w * (sr_yl - hr_yl).abs().mean()

        # High-freq subband losses
        band_names = ['LH', 'HL', 'HH']
        for level_idx in range(self.J):
            sr_hf = sr_yh[level_idx]   # (B, 3, 3, H', W') — 3 orientations
            hr_hf = hr_yh[level_idx]
            for band_idx, band_name in enumerate(band_names):
                w = self.band_weights.get(band_name, 1.0)
                loss = loss + w * (sr_hf[:, :, band_idx] - hr_hf[:, :, band_idx]).abs().mean()

        # Normalize by number of bands
        n_bands = 1 + 3 * self.J
        return loss / n_bands
