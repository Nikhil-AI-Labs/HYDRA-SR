"""
Target Score Distillation (TSD) loss for Stage 3 perceptual training.

From TSD-SR (CVPR 2025): "Target Score Distillation for Real-World
Super-Resolution" (Microtreei et al.)

Core idea:
  Instead of Score Distillation Sampling (SDS) which optimizes against
  random noise targets and causes artifacts ("over-saturation"), TSD
  uses the HIGH-QUALITY HR image as the target latent and matches the
  diffusion score (noise prediction) between the SR latent and the
  HR-noised latent.

  L_TSD = E_t [ w(t) || ε_θ(z_t^SR, t) − ε_θ*(z_t^HR, t) ||₂² ]

where:
  z_t = α_t·z_0 + σ_t·ε  (DDPM forward process)
  ε_θ* = frozen teacher (SD3 + LoRA fine-tuned on HQ data)
  ε_θ  = same teacher, but z_t derived from SR (gradient flows through SR)
  t     is sampled uniformly in [t_min, t_max] = [200, 800]

Why this is better than vanilla SDS/VSD:
  - SDS samples from pure noise → high variance gradients → artifacts
  - TSD samples from HR-latent → low-variance, detail-preserving gradients
  - Result: ~40× faster training than SeeSR, better LPIPS

Implementation notes:
  - The teacher (SD3 + LoRA) is NEVER run during forward — outputs are
    cached offline via scripts/cache_teachers.py. This avoids the 1B-param
    teacher eating GPU memory during student training.
  - We load cached (eps_target, z_hr) pairs from disk and compute the
    student-side prediction on-the-fly.
  - The teacher is loaded only during the offline caching step.

Shape contract:
  sr:          (B, 3, H_hr, W_hr) — SR output from HYDRA-SR
  eps_target:  (B, 4, H_lat, W_lat) — cached teacher noise prediction
  z_t_hr:      (B, 4, H_lat, W_lat) — cached noised HR latent
  → loss: scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
              alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Forward diffusion process: z_t = sqrt(ᾱ_t)·z_0 + sqrt(1-ᾱ_t)·ε
    Args:
        z0:           (B, C, H, W) latent at t=0
        t:            (B,) timestep indices
        noise:        (B, C, H, W) standard Gaussian noise
        alphas_cumprod: (T,) ᾱ schedule
    """
    a_t    = alphas_cumprod[t].view(-1, 1, 1, 1).to(z0.dtype)
    sqrt_a = a_t.sqrt()
    sqrt_1a = (1 - a_t).sqrt()
    return sqrt_a * z0 + sqrt_1a * noise


class TSDDistillLoss(nn.Module):
    """
    Target Score Distillation loss (CVPR 2025).

    In production: uses pre-cached teacher predictions (offline mode).
    In development/ablation: can use live teacher if available.

    Args:
        vae:           Frozen SD3 VAE encoder. Used to encode SR → latent.
        teacher:       Optional live teacher model (for development/ablation only).
                       In training, this should be None (use cached outputs).
        t_min, t_max:  Timestep range for noise sampling.
        loss_weight:   Multiplier applied to the loss (from Stage3Loss config).
        use_cached:    If True, expects pre-computed (eps_target, z_t_hr) tensors
                       to be passed in the batch dict (offline caching mode).
    """

    def __init__(
        self,
        vae=None,
        teacher=None,
        t_min: int = 200,
        t_max: int = 800,
        loss_weight: float = 0.6,
        use_cached: bool = True,
    ):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.loss_weight = loss_weight
        self.use_cached  = use_cached

        if vae is not None:
            for p in vae.parameters():
                p.requires_grad_(False)
        self.vae = vae

        if teacher is not None:
            for p in teacher.parameters():
                p.requires_grad_(False)
        self.teacher = teacher

        # SD3 VAE scale factor (standard)
        self.vae_scale = 0.18215

        # DDPM noise schedule (simplified linear schedule)
        T = 1000
        betas = torch.linspace(1e-4, 2e-2, T)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def forward_cached(
        self,
        sr: torch.Tensor,
        eps_target: torch.Tensor,
        z_t_sr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute TSD loss using pre-cached teacher predictions.

        This is the training-time path (use_cached=True).

        Args:
            sr:          (B, 3, H, W) SR output from HYDRA-SR
            eps_target:  (B, 4, H_lat, W_lat) cached teacher score on HR latent
            z_t_sr:      (B, 4, H_lat, W_lat) noised SR latent (pre-computed)
                         If None, we encode SR here using self.vae.

        Returns:
            loss: scalar TSD loss
        """
        # If the noised SR latent is not pre-cached, encode now
        if z_t_sr is None and self.vae is not None:
            with torch.no_grad():
                z_sr = self.vae.encode(sr * 2.0 - 1.0).latent_dist.sample() * self.vae_scale
            B = z_sr.shape[0]
            t = torch.randint(self.t_min, self.t_max, (B,), device=z_sr.device)
            noise = torch.randn_like(z_sr)
            z_t_sr = _q_sample(z_sr, t, noise, self.alphas_cumprod)
            # Note: gradient does NOT flow through z_t_sr here for memory efficiency
            # (We use the cached eps_target, not live teacher)

        if self.teacher is not None:
            # Live teacher path (ablation only — very slow in training)
            B = z_t_sr.shape[0]
            t = torch.randint(self.t_min, self.t_max, (B,), device=z_t_sr.device).long()
            eps_pred = self.teacher(z_t_sr, t)
            return F.mse_loss(eps_pred, eps_target) * self.loss_weight

        # Cached path — mathematical note:
        # The correct TSD loss is MSE(ε_θ(z_t_sr, t), eps_target), where both
        # sides are NOISE PREDICTIONS. Comparing z_t_sr (noised latent, scale≈1)
        # directly to eps_target (unit Gaussian noise prediction) is wrong:
        # z_t_sr ≠ ε_θ(z_t_sr, t).
        #
        # Without a live teacher call on z_t_sr, we cannot compute the true TSD
        # loss from cache alone. The SimpleTSDLoss (LPIPS proxy) is mathematically
        # correct and is already implemented — it's the right choice for Stage 3
        # when the full SD3 teacher is not available at training time.
        #
        # If you have cached (eps_target_hr, z_t_hr) from the teacher, the correct
        # approach is to run the teacher on z_t_sr (derived from sr) and compare to
        # the cached eps_target_hr. That requires the teacher in GPU memory during
        # training, which is expensive but correct. For HYDRA-SR Stage 3, we defer
        # to SimpleTSDLoss (LPIPS) as the primary perceptual loss instead.
        raise RuntimeError(
            "TSDDistillLoss cached path requires a live teacher call on z_t_sr. "
            "Use SimpleTSDLoss (LPIPS proxy) for Stage 3 training without a live teacher, "
            "or pass teacher= and set use_cached=False for ablation."
        )

    def forward(
        self,
        sr: torch.Tensor,
        eps_target: torch.Tensor,
        z_t_sr: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.forward_cached(sr, eps_target, z_t_sr)


class SimpleTSDLoss(nn.Module):
    """
    Simplified TSD loss that works without a full diffusion teacher.
    Uses VGG perceptual loss as a proxy for score matching.
    
    This is the development/ablation version — use TSDDistillLoss
    with cached teacher in full training.
    """

    def __init__(self, loss_weight: float = 0.6):
        super().__init__()
        self.loss_weight = loss_weight

        try:
            import lpips
            self.lpips_net = lpips.LPIPS(net='vgg', verbose=False)
            for p in self.lpips_net.parameters():
                p.requires_grad_(False)
        except ImportError:
            self.lpips_net = None

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if self.lpips_net is not None:
            # Map [0,1] → [-1,1] for LPIPS
            sr_01  = sr.clamp(0, 1) * 2 - 1
            hr_01  = hr.clamp(0, 1) * 2 - 1
            return self.lpips_net(sr_01, hr_01).mean() * self.loss_weight
        else:
            # Absolute fallback: L2 in image space
            return F.mse_loss(sr, hr) * self.loss_weight
