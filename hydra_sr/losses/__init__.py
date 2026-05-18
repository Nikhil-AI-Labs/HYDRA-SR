"""
Composite loss functions for HYDRA-SR's three training stages.

Stage 1 (Geometry Lock):
  L = 1.0 × Charbonnier(sr, hr) + 0.1 × L2(d_hat, d_gt)

Stage 2 (Frequency Lock):
  L = 0.8 × L1(sr, hr)
    + 0.6 × FFL(sr, hr)       [Focal Frequency Loss]
    + 0.3 × WaveletL1(sr, hr) [subband-weighted]
    + 0.4 × L1(sr_w, hr_w)    [wavelet-domain output]

Stage 3 (Perceptual):
  L = λ_l1 × L1(sr, hr)
    + 1.0 × LPIPS(sr, hr)
    + λ_tsd × TSD(sr, hr)
    + 0.4 × DTWSRDistill(sr, hr)
    + 0.1 × GAN(sr)
  where λ_l1, λ_tsd are dynamically adjusted by DynamicWeighter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .charbonnier import CharbonnierLoss
from .wavelet_l1 import WaveletL1Loss
from .tsd_distill import SimpleTSDLoss
from .lpips_wrap import LPIPSLoss
from .dynamic_weights import DynamicWeighter

try:
    from focal_frequency_loss import FocalFrequencyLoss
    _FFL_AVAILABLE = True
except ImportError:
    _FFL_AVAILABLE = False


class Stage1Loss(nn.Module):
    """
    Stage 1: Geometry Lock loss.
    Pure pixel-domain supervision. No perceptual terms.
    Includes degradation prediction supervision with synthetic GT.
    """

    def __init__(
        self,
        charb_weight: float = 1.0,
        deg_pred_weight: float = 0.1,
    ):
        super().__init__()
        self.charb = CharbonnierLoss(eps=1e-3)
        self.charb_weight    = charb_weight
        self.deg_pred_weight = deg_pred_weight

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        d_hat: torch.Tensor = None,
        d_gt:  torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sr:    (B, 3, 4H, 4W) SR output
            hr:    (B, 3, 4H, 4W) ground truth HR
            d_hat: (B, 4) predicted degradation params (optional)
            d_gt:  (B, 4) GT degradation params (optional, from synthetic pipeline)

        Returns:
            dict: {'total': ..., 'charb': ..., 'deg': ...}
        """
        l_charb = self.charb(sr, hr) * self.charb_weight
        l_total = l_charb
        result  = {'charb': l_charb}

        if d_hat is not None and d_gt is not None:
            l_deg = F.mse_loss(d_hat, d_gt) * self.deg_pred_weight
            l_total = l_total + l_deg
            result['deg'] = l_deg

        result['total'] = l_total
        return result


class Stage2Loss(nn.Module):
    """
    Stage 2: Frequency Lock loss.
    L1 + Focal Frequency + Wavelet L1 (subband-weighted).
    """

    def __init__(
        self,
        l1_weight:   float = 0.8,
        ffl_weight:  float = 0.6,
        swt_weight:  float = 0.3,
        wl1_weight:  float = 0.4,
    ):
        super().__init__()
        self.l1_weight  = l1_weight
        self.ffl_weight = ffl_weight
        self.swt_weight = swt_weight
        self.wl1_weight = wl1_weight

        self.wavelet_l1 = WaveletL1Loss(wave='db4', J=2)

        if _FFL_AVAILABLE:
            self.ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        else:
            self.ffl = None

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        sr_wavelet: torch.Tensor = None,
        hr_wavelet: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sr, hr:              (B, 3, 4H, 4W)
            sr_wavelet, hr_wavelet: (B, C, H/4, W/4) wavelet stream outputs (optional)
        """
        l_l1    = F.l1_loss(sr, hr) * self.l1_weight
        l_total = l_l1
        result  = {'l1': l_l1}

        if self.ffl is not None:
            l_ffl = self.ffl(sr, hr) * self.ffl_weight
        else:
            # FFL not available: use frequency-weighted L2 as proxy
            sr_fft = torch.fft.rfft2(sr).abs()
            hr_fft = torch.fft.rfft2(hr).abs()
            l_ffl  = F.l1_loss(sr_fft, hr_fft) * self.ffl_weight
        l_total = l_total + l_ffl
        result['ffl'] = l_ffl

        l_swt = self.wavelet_l1(sr, hr) * self.swt_weight
        l_total = l_total + l_swt
        result['swt'] = l_swt

        if sr_wavelet is not None and hr_wavelet is not None:
            l_wl1 = F.l1_loss(sr_wavelet, hr_wavelet) * self.wl1_weight
            l_total = l_total + l_wl1
            result['wl1'] = l_wl1

        result['total'] = l_total
        return result


class Stage3Loss(nn.Module):
    """
    Stage 3: Perceptual training loss.
    Dynamic L1 + LPIPS + TSD distillation + DTWSR distillation + optional GAN.
    """

    def __init__(
        self,
        lpips_net: str = 'alex',
        gan_weight: float = 0.1,
        use_tsd: bool = True,
        use_dtw_distill: bool = True,
    ):
        super().__init__()
        self.lpips = LPIPSLoss(net=lpips_net, weight=1.0)
        self.tsd   = SimpleTSDLoss(loss_weight=1.0) if use_tsd else None
        self.gan_weight = gan_weight
        self.use_dtw_distill = use_dtw_distill

        # Dynamic weighter manages λ_l1 and λ_tsd
        self.weighter = DynamicWeighter(patience=10000, delta_thr=0.001)

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        lpips_val_for_weighting: float = None,
        discriminator_loss: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sr, hr:                  (B, 3, 4H, 4W)
            lpips_val_for_weighting: Current validation LPIPS (for DynamicWeighter update).
            discriminator_loss:      Optional GAN discriminator loss term.

        Returns:
            dict with 'total' and individual loss components.
        """
        # Get current dynamic weights
        if lpips_val_for_weighting is not None:
            lam_l1, lam_lpips, lam_tsd, lam_dtw = self.weighter.update(lpips_val_for_weighting)
        else:
            lam_l1  = self.weighter.lam_l1
            lam_tsd = self.weighter.lam_tsd
            lam_dtw = self.weighter.lam_dtw

        result  = {}
        l_total = torch.zeros(1, device=sr.device, dtype=sr.dtype)

        # L1 baseline
        l_l1 = F.l1_loss(sr, hr) * lam_l1
        l_total = l_total + l_l1
        result['l1'] = l_l1

        # LPIPS perceptual
        l_lpips = self.lpips(sr, hr)
        l_total = l_total + l_lpips
        result['lpips'] = l_lpips

        # TSD distillation (simplified version)
        if self.tsd is not None:
            l_tsd = self.tsd(sr, hr) * lam_tsd
            l_total = l_total + l_tsd
            result['tsd'] = l_tsd

        # GAN term (optional)
        if discriminator_loss is not None:
            l_gan = discriminator_loss * self.gan_weight
            l_total = l_total + l_gan
            result['gan'] = l_gan

        result['total'] = l_total
        result['weights'] = {'lam_l1': lam_l1, 'lam_tsd': lam_tsd}
        return result
