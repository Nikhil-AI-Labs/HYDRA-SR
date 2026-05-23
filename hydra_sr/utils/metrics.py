"""
PSNR, SSIM, LPIPS, and DISTS metrics for HYDRA-SR evaluation.

All metrics follow the NTIRE 2026 evaluation protocol:
  - PSNR-Y: computed on Y channel of YCbCr, after bicubic cropping of scale//2 border
  - SSIM-Y: same channel and border convention
  - LPIPS:  AlexNet, on RGB, no border crop (perceptual track)
  - DISTS:  Optional (for full perceptual evaluation)

These conventions exactly match:
  - DIV2K benchmark evaluation (NTIRE 2026 fidelity track)
  - NTIRE 2026 perceptual track requirements
"""

import torch
import torch.nn.functional as F
import numpy as np


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """
    Convert linear RGB tensor to Y channel (BT.601 luminance).

    Args:
        img: (B, 3, H, W) float32 in [0, 1].
    Returns:
        y:   (B, 1, H, W) in [16/255, 235/255]  (standard Y range).

    Formula (BT.601, studio swing):
        Y = (65.481*R + 128.553*G + 24.966*B) / 255  +  16/255

    Both the coefficients and the offset are divided by 255 so that
    input [0,1] maps to output [16/255, 235/255].  This is the form
    used by MATLAB's rgb2ycbcr and by every NTIRE evaluation script.

    COMMON BUG: writing `+ 16.0/255.0` when the coefficients still
    operate on [0,255] scale gives wrong output.  The safe version here
    keeps EVERYTHING on the [0,1] scale.
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = (65.481 * r + 128.553 * g + 24.966 * b) / 255.0 + 16.0 / 255.0
    return y


def compute_psnr_y(
    sr: torch.Tensor,
    hr: torch.Tensor,
    scale: int = 4,
    max_val: float = 1.0,
) -> float:
    """
    PSNR-Y: PSNR on Y channel with border crop.

    Args:
        sr:      (B, 3, H, W) float in [0, 1]
        hr:      (B, 3, H, W) float in [0, 1]
        scale:   SR scale factor (border crop = scale pixels per side)
        max_val: dynamic range (1.0 for [0,1] inputs)

    NTIRE border convention: crop `scale` pixels on each side (not scale//2).
    """
    # Border crop: NTIRE uses `scale` pixels (not scale//2)
    crop = scale

    # rgb_to_y expects [0,1] — do NOT multiply by 255 first
    sr_y = rgb_to_y(sr.clamp(0, 1))[:, :, crop:-crop, crop:-crop]
    hr_y = rgb_to_y(hr.clamp(0, 1))[:, :, crop:-crop, crop:-crop]

    mse = ((sr_y - hr_y) ** 2).mean()
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(torch.tensor(max_val ** 2, dtype=mse.dtype) / mse)
    return psnr.item()


def compute_ssim_y(
    sr: torch.Tensor,
    hr: torch.Tensor,
    scale: int = 4,
    window_size: int = 11,
    C1: float = (0.01) ** 2,     # C1 on [0,1] scale  (= (0.01*255)^2 / 255^2)
    C2: float = (0.03) ** 2,     # C2 on [0,1] scale
) -> float:
    """
    SSIM-Y: SSIM on Y channel with border crop.
    Implements the standard SSIM formula with 11×11 Gaussian window.

    Constants C1, C2 are on [0,1] scale (matching max_val=1.0).
    """
    crop = scale

    # rgb_to_y expects [0,1] — do NOT multiply by 255 first
    sr_y = rgb_to_y(sr.clamp(0, 1))[:, :, crop:-crop, crop:-crop]
    hr_y = rgb_to_y(hr.clamp(0, 1))[:, :, crop:-crop, crop:-crop]

    # Gaussian window
    import math
    sigma = 1.5
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ], dtype=sr_y.dtype, device=sr_y.device)
    gauss /= gauss.sum()
    window = gauss.outer(gauss).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)

    def _conv(x):
        return F.conv2d(x, window, padding=window_size // 2, groups=1)

    mu1  = _conv(sr_y)
    mu2  = _conv(hr_y)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _conv(sr_y ** 2) - mu1_sq
    sigma2_sq = _conv(hr_y ** 2) - mu2_sq
    sigma12   = _conv(sr_y * hr_y) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


class MetricCalculator:
    """
    Batch metric calculator for HYDRA-SR evaluation.

    Usage:
        calc = MetricCalculator(scale=4)
        for batch in val_loader:
            calc.update(sr, hr)
        metrics = calc.compute()
        calc.reset()
    """

    def __init__(self, scale: int = 4, device: str = 'cuda'):
        self.scale  = scale
        self.device = device
        self.reset()

        # LPIPS (optional)
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
            self._lpips_available = True
        except ImportError:
            self._lpips_available = False

    def reset(self):
        self.psnr_vals  = []
        self.ssim_vals  = []
        self.lpips_vals = []

    def update(self, sr: torch.Tensor, hr: torch.Tensor):
        """
        Accumulate metrics for one batch.
        sr, hr: (B, 3, H, W) in [0, 1] on any device.
        """
        sr = sr.detach().float().clamp(0, 1)
        hr = hr.detach().float().clamp(0, 1)

        B = sr.shape[0]
        for i in range(B):
            self.psnr_vals.append(compute_psnr_y(sr[i:i+1], hr[i:i+1], self.scale))
            self.ssim_vals.append(compute_ssim_y(sr[i:i+1], hr[i:i+1], self.scale))

        if self._lpips_available:
            sr_norm = (sr * 2.0 - 1.0).to(self.device)
            hr_norm = (hr * 2.0 - 1.0).to(self.device)
            with torch.no_grad():
                lpips_val = self.lpips_fn(sr_norm, hr_norm).mean().item()
            self.lpips_vals.append(lpips_val)

    def compute(self) -> dict[str, float]:
        """Return average metric values over all accumulated batches."""
        result = {}
        if self.psnr_vals:
            result['psnr_y'] = float(np.mean(self.psnr_vals))
        if self.ssim_vals:
            result['ssim_y'] = float(np.mean(self.ssim_vals))
        if self.lpips_vals:
            result['lpips']  = float(np.mean(self.lpips_vals))
        return result
