"""
HYDRA-SR: Hierarchical Yoked Dual-domain Restoration Architecture
          for Super-Resolution

Top-level model class assembling all five innovations:
  1. Dual-Domain Yoked Backbone (Pixel ⇌ Wavelet)
  2. Attentive Hilbert-Nested Mamba (AHN-Mamba) — Stage 2
  3. Degradation Prompt Conditioning (DPC) — Blind SR
  4. Adaptive Frequency Router (AFR) — MELD-SR descendant
  5. Stage 3 Deformable Windowed Attention

Target performance:
  DIV2K-Val PSNR: 33.8–34.3 dB (vs MELD-SR 31.48 dB)
  SSIM:           0.920+         (vs MELD-SR 0.8815)
  LPIPS:          0.115–0.135    (vs MELD-SR 0.2180)
  Parameters:     ~16.9 M        (vs MELD-SR 172.59 M)
  Inference 256²: ~75 ms         (vs MELD-SR 10,911 ms)

The model supports:
  - return_aux=True for training (returns routing weights + degradation params)
  - gradient_checkpointing on Stage 2-W (prevents OOM at batch=8 on 16GB cards)
  - 4K inference via tile_runner (see hydra_sr/inference/tile_runner.py)

Parameter budget:
  Degradation Predictor     0.2 M
  Shallow + Stage 1         3.0 M  (both streams)
  Stage 2-P (AHN×6)         6.5 M
  Stage 2-W (AHN×4)         2.5 M
  CDB-1, CDB-2              0.6 M
  Frequency Router          0.8 M
  Stage 3 Deform-Attn       1.8 M
  Stage 4 Upsampler         1.5 M
  Total                    ~16.9 M (trainable)

Shape contract:
  lr:  (B, 3, H, W)   — LR input, values in [0, 1]
  out: (B, 3, 4H, 4W) — SR output, values approximately in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .modules.degradation_predictor import DegradationPredictor
from .modules.nafblock import NAFBlock, LayerNorm2d
from .modules.attentive_ssm import AHNMambaBlock
from .modules.cross_domain_bridge import CrossDomainBridge
from .modules.freq_router import FrequencyRouter
from .modules.deformable_window_attn import DeformableWindowAttnStack
from .stages.stage4_upsampler import FreqGatedUpsampler

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    _WAVELETS_AVAILABLE = True
except ImportError:
    _WAVELETS_AVAILABLE = False


def _stack_wavelet_subbands(yl: torch.Tensor, yh: list) -> torch.Tensor:
    """
    Stack all DWT subbands into channel dimension at the deepest-level resolution.

    pytorch_wavelets DWTForward(J=2) returns a PYRAMID — NOT same-size tensors:
        yl:    (B, C, H/4, W/4)  — deepest LL approximation
        yh[0]: (B, C, 3, H/2, W/2) — level-1 LH/HL/HH details (FINEST)
        yh[1]: (B, C, 3, H/4, W/4) — level-2 LH/HL/HH details (COARSEST)

    We adopt H/4 as the canonical spatial size (matches the wavelet stream
    resolution in HYDRA-SR). Level-1 subbands are downsampled via avg-pool
    before stacking — this preserves their energy without introducing learned
    parameters, and is the standard approach in multi-scale wavelet CNNs.

    Returns: (B, C*(1+3J), H/4, W/4)
    """
    # Target spatial size = deepest level (yl)
    target_h, target_w = yl.shape[-2], yl.shape[-1]

    parts = [yl]
    for yh_level in yh:
        # yh_level: (B, C, 3, H', W')
        for k in range(3):
            subband = yh_level[:, :, k, :, :]   # (B, C, H', W')
            if subband.shape[-2] != target_h or subband.shape[-1] != target_w:
                # Downsample finer-level subbands to the canonical H/4 size
                subband = F.avg_pool2d(
                    subband,
                    kernel_size=subband.shape[-1] // target_w,  # integer stride
                )
            parts.append(subband)
    return torch.cat(parts, dim=1)


class HYDRASR(nn.Module):
    """
    HYDRA-SR full model.

    Args:
        scale:          SR upsampling factor (default 4).
        dim_p:          Pixel-stream channel dimension (default 96).
        dim_w:          Wavelet-stream channel dimension (default 64).
        n_mamba_p:      Number of AHN-Mamba blocks in pixel stream Stage 2 (default 6).
        n_mamba_w:      Number of AHN-Mamba blocks in wavelet stream Stage 2 (default 4).
        n_nafblocks_s1: Number of NAFBlocks in Stage 1 per stream (default 4).
        n_transformer:  Number of deformable attention blocks in Stage 3 (default 2).
        prompt_dim:     Degradation prompt dimension (default 128).
        J:              DWT decomposition levels (default 2 → H/4 wavelet resolution).
        wave:           Wavelet family (default 'db4').
        use_checkpoint: If True, use gradient checkpointing on Stage 2-W blocks.
                        CRITICAL for batch=8 on 16GB VRAM cards (Pitfall #4).
    """

    def __init__(
        self,
        scale: int = 4,
        dim_p: int = 192,          # pixel-stream width  (→ 16.9M target)
        dim_w: int = 160,          # wavelet-stream width
        n_mamba_p: int = 14,       # P-stream AHN-Mamba blocks
        n_mamba_w: int = 9,        # W-stream AHN-Mamba blocks
        n_nafblocks_s1: int = 4,
        n_transformer: int = 2,
        prompt_dim: int = 128,
        J: int = 2,
        wave: str = 'db4',
        use_checkpoint: bool = False,
        upsampler_mid_dim: int = 64,  # PixelShuffle neck  (SwinIR/HAT standard = 64)
    ):
        super().__init__()

        self.scale = scale
        self.J = J
        self.use_checkpoint = use_checkpoint

        # ── Degradation Predictor (Innovation #3) ───────────────────────
        self.deg_pred = DegradationPredictor(prompt_dim=prompt_dim)

        # ── Stream P: Pixel Domain ───────────────────────────────────────
        # Shallow feature extraction: 3 → dim_p
        self.conv_in_p = nn.Conv2d(3, dim_p, kernel_size=3, stride=1, padding=1, bias=True)

        # Stage 1-P: NAFNet local denoising/color blocks
        self.stage1_p = nn.Sequential(
            *[NAFBlock(dim_p) for _ in range(n_nafblocks_s1)]
        )

        # Stage 2-P: AHN-Mamba global structure (pixel stream)
        # tile=16 for full-resolution pixel stream
        # expand=4, d_state=32 — 4× channel expansion + 2× state for ~6.5M budget
        self.stage2_p = nn.ModuleList([
            AHNMambaBlock(
                dim=dim_p,
                d_state=32,
                expand=4,
                n_prompts=8,
                tile=16,
                prompt_dim=prompt_dim,
                wavelet_delta_bias=False,
            )
            for _ in range(n_mamba_p)
        ])

        # ── Stream W: Wavelet Domain ─────────────────────────────────────
        # 2-level Daubechies db4 DWT: LR (B,3,H,W) → (B,3*(1+3J), H/4, W/4)
        dwt_in_channels = 3 * (1 + 3 * J)   # = 3*(1+6) = 21 for J=2

        if _WAVELETS_AVAILABLE:
            self.dwt2  = DWTForward(J=J, wave=wave, mode='periodization')
            self.idwt2 = DWTInverse(wave=wave, mode='periodization')
        else:
            self.dwt2  = None
            self.idwt2 = None

        # Shallow conv for wavelet subbands: dwt_in_channels → dim_w
        self.conv_in_w = nn.Conv2d(dwt_in_channels, dim_w, kernel_size=3, stride=1, padding=1, bias=True)

        # Stage 1-W: NAFNet on wavelet coefficients
        self.stage1_w = nn.Sequential(
            *[NAFBlock(dim_w) for _ in range(n_nafblocks_s1)]
        )

        # Stage 2-W: AHN-Mamba on wavelet stream
        # tile=8: W-stream is already at H/4 spatial resolution
        # expand=4, d_state=32 — same hyperparams as P-stream for budgetary consistency
        # wavelet_delta_bias=True: Δ biased larger for high-freq subband channels
        self.stage2_w = nn.ModuleList([
            AHNMambaBlock(
                dim=dim_w,
                d_state=32,
                expand=4,
                n_prompts=8,
                tile=8,
                prompt_dim=prompt_dim,
                wavelet_delta_bias=True,  # HF-biased Δ initialization
            )
            for _ in range(n_mamba_w)
        ])

        # ── Cross-Domain Bridges (Innovation #1 fusion mechanism) ────────
        self.cdb1 = CrossDomainBridge(C_P=dim_p, C_W=dim_w, J=J, wave=wave)
        self.cdb2 = CrossDomainBridge(C_P=dim_p, C_W=dim_w, J=J, wave=wave)

        # W→P channel projection for final merge (reuse CDB2's w2p_conv)
        # We need to project dim_w → dim_p for the frequency-weighted merge
        self.w_to_p_proj = nn.Conv2d(dim_w, dim_p, 1, bias=True)

        # ── Adaptive Frequency Router (Innovation #4) ────────────────────
        self.router = FrequencyRouter(
            in_channels=3,
            n_bands=9,
            hidden_dim=32,
            n_heads=4,
            n_streams=3,
        )

        # ── Stage 3: Deformable Windowed Attention ───────────────────────
        self.stage3 = DeformableWindowAttnStack(
            dim=dim_p,
            depth=n_transformer,
            window_size=8,
            num_heads=max(1, dim_p // 16),  # 160//16=10, 128//16=8
            ffn_expand=4,
            prompt_dim=prompt_dim,
        )

        # ── Stage 4: Frequency-Gated Upsampler ───────────────────────────
        # mid_dim is pinned to upsampler_mid_dim (default 64, SwinIR/HAT standard).
        # Without this, conv_before_ps = Conv2d(dim_p, dim_p*16, 3) which is
        # 160*2560*9 = 3.69M for ONE layer — destroying the parameter budget.
        self.upsampler = FreqGatedUpsampler(
            in_dim=dim_p, scale=scale, mid_dim=upsampler_mid_dim
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Weight initialization following NAFNet/SwinIR convention.

        IMPORTANT — init order:
        The generic trunc_normal_(std=0.02) loop below would overwrite the
        carefully zeroed FiLM/offset/head_d weights that are needed for
        training stability (identity start, no FiLM shift at iter 0).
        We therefore re-apply those targeted zero-inits AFTER the generic loop.
        """
        # ── Generic init ────────────────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # ── Stability priming — re-apply AFTER generic loop ─────────────────
        # FiLM proj: must be zero so FiLM(x, p) = x at iter 0 (no shift/scale)
        for name, m in self.named_modules():
            if 'film' in name.lower() and hasattr(m, 'proj'):
                if isinstance(m.proj, nn.Linear):
                    nn.init.zeros_(m.proj.weight)
                    if m.proj.bias is not None:
                        nn.init.zeros_(m.proj.bias)
            # AHN-Mamba FiLM conditioning projection
            if name.endswith('film_proj') and isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # Offset predictor last layer: offsets must be zero at iter 0
            if 'offset_predictor' in name and isinstance(m, nn.Sequential):
                last = m[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
        # DegradationPredictor heads: small xavier (not std=0.02 which is too large)
        nn.init.xavier_uniform_(self.deg_pred.head_d.weight, gain=0.1)
        nn.init.zeros_(self.deg_pred.head_d.bias)
        nn.init.xavier_uniform_(self.deg_pred.head_p.weight, gain=0.1)
        nn.init.zeros_(self.deg_pred.head_p.bias)

    def _dwt_input(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Apply 2-level DWT to LR image and stack subbands.
        Falls back to strided conv + channel repeat if pytorch_wavelets unavailable.
        """
        if self.dwt2 is not None:
            yl, yh = self.dwt2(lr)           # yl: (B,3,H/4,W/4), yh: 2×(B,3,3,H',W')
            return _stack_wavelet_subbands(yl, yh)  # (B, 21, H/4, W/4)
        else:
            # Fallback: downsample + repeat (loses freq info, for CPU unit testing)
            H4, W4 = lr.shape[-2] // 4, lr.shape[-1] // 4
            pooled = F.adaptive_avg_pool2d(lr, (H4, W4))  # (B, 3, H/4, W/4)
            n_reps = 1 + 3 * self.J
            return pooled.repeat(1, n_reps, 1, 1)          # (B, 21, H/4, W/4)

    def _run_stage2_w(self, f_w: torch.Tensor, p_d: torch.Tensor) -> torch.Tensor:
        """
        Run Stage 2-W blocks with optional gradient checkpointing.
        Gradient checkpointing is CRITICAL on 16GB cards at batch=8.
        """
        for blk in self.stage2_w:
            if self.use_checkpoint and self.training:
                # Wrap in checkpoint to trade compute for memory
                f_w = checkpoint.checkpoint(blk, f_w, p_d, use_reentrant=False)
            else:
                f_w = blk(f_w, p_d)
        return f_w

    def forward(
        self,
        lr: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Full HYDRA-SR forward pass.

        Args:
            lr:         (B, 3, H, W) LR input image, values in [0, 1].
            return_aux: If True, return auxiliary outputs for loss computation.
                        Always True during training. False for inference.

        Returns:
            sr:          (B, 3, 4H, 4W) SR output.
            aux (dict):  Only if return_aux=True:
                         {
                           'd_hat':  (B, 4)  degradation parameters,
                           'p_d':    (B, 128) degradation prompt,
                           'r':      (r_P, r_W, r_T) routing weights,
                           'f_p2':   (B, dim_p, H, W) pixel stream after Stage 2,
                           'f_w2':   (B, dim_w, H/4, W/4) wavelet stream after Stage 2,
                         }
        """
        B, _, H, W = lr.shape

        # ── Step 1: Degradation Prediction ──────────────────────────────
        # d_hat: (B, 4)  — [σ_blur, σ_noise, q_JPEG, s_ds]
        # p_d:   (B, 128) — FiLM conditioning prompt
        d_hat, p_d = self.deg_pred(lr)

        # ── Step 2: Stream P — Pixel Domain Extraction ──────────────────
        f_p = self.conv_in_p(lr)        # (B, dim_p, H, W)
        f_p = self.stage1_p(f_p)        # (B, dim_p, H, W) — local denoising

        # ── Step 3: Stream W — Wavelet Domain Extraction ────────────────
        stacked_w = self._dwt_input(lr)        # (B, 21, H/4, W/4)
        f_w = self.conv_in_w(stacked_w)        # (B, dim_w, H/4, W/4)
        f_w = self.stage1_w(f_w)               # (B, dim_w, H/4, W/4) — wavelet denoising

        # ── Step 4: CDB-1 — Cross-Domain Bridge (post-Stage-1) ──────────
        f_p, f_w = self.cdb1(f_p, f_w)

        # ── Step 5: Stage 2 — AHN-Mamba (both streams in parallel) ──────
        for blk in self.stage2_p:
            f_p = blk(f_p, p_d)       # (B, dim_p, H, W)

        f_w = self._run_stage2_w(f_w, p_d)    # (B, dim_w, H/4, W/4)

        # ── Step 6: CDB-2 — Cross-Domain Bridge (post-Stage-2) ──────────
        f_p, f_w = self.cdb2(f_p, f_w)

        # ── Step 7: Adaptive Frequency Router ───────────────────────────
        routing = self.router(lr)        # (B, 3) — softmax weights
        r_p = routing[:, 0].view(B, 1, 1, 1)
        r_w = routing[:, 1].view(B, 1, 1, 1)
        r_t = routing[:, 2].view(B, 1, 1, 1)

        # Project W-stream to pixel resolution for merging.
        # Use size= (not scale_factor=4) to avoid ±2px rounding on odd-sized inputs.
        f_w_up = F.interpolate(
            self.w_to_p_proj(f_w),
            size=(f_p.shape[-2], f_p.shape[-1]),
            mode='bilinear',
            align_corners=False,
        )  # (B, dim_p, H, W) — exactly aligned with f_p


        # Frequency-weighted merge of P and W streams
        f_merged = r_p * f_p + r_w * f_w_up   # (B, dim_p, H, W)

        # ── Step 8: Stage 3 — Deformable Windowed Attention ─────────────
        f_t = self.stage3(f_merged, p_d)       # (B, dim_p, H, W)

        # Residual: routing weight r_T controls how much Transformer corrects
        f_final = f_merged + r_t * (f_t - f_merged)  # (B, dim_p, H, W)

        # ── Step 9: Stage 4 — Upsampler ──────────────────────────────────
        sr = self.upsampler(f_final, lr)       # (B, 3, 4H, 4W)

        if return_aux:
            aux = {
                'd_hat': d_hat,
                'p_d':   p_d,
                'r':     (r_p, r_w, r_t),
                'f_p2':  f_p,
                'f_w2':  f_w,
            }
            return sr, aux

        return sr

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
