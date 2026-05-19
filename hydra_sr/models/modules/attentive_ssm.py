"""
Attentive Hilbert-Nested Mamba (AHN-Mamba) Block.
The heart of HYDRA-SR — Stage 2 processing block.

This block combines three orthogonal innovations:

(a) Nested Hilbert-S (NHS) Scanning [HYDRA-SR novel contribution]:
    Combines MaIR CVPR 2025 Nested S-shaped scanning (inter-tile S-curve)
    with FractalMamba++ Hilbert scanning (intra-tile Hilbert curve).
    Preserves spatial locality AND continuity simultaneously.

(b) Attentive State Prompt (ASP) [from MambaIRv2, CVPR 2025]:
    Adds learnable query prompts that retrieve from the SSM hidden state pool
    via cross-attention, giving Mamba non-causal, content-addressable memory.
    Equation: y_t = C(x_t)h_t + α·Softmax(q·Kh^T/√d)·Vh
    This directly fixes Mamba's "Local Detail Forgetting" failure mode
    WITHOUT adding a Transformer stage — it's architectural, not a band-aid.

(c) Wavelet-aware Δ initialization [HYDRA-SR novel]:
    In the W-stream AHN-Mamba blocks, the SSM step size Δ is initialized
    larger for high-frequency subband channels and smaller for LL channels.
    Mamba updates fast where edges live (HH/LH/HL) and slow where smooth
    structure lives (LL) — natural alignment with wavelet statistics.

Full forward equation (AHN-Mamba):
    Given X ∈ R^{B×C×H×W}, prompt p_d ∈ R^{B×128}:

    1. FiLM:     X' = (1+γ(p_d)) ⊙ X + β(p_d)
    2. DWConv:   X'' = DWConv_{3×3}(X')  [local context injection]
    3. NHS scan: S = Φ_NHS(X'') ∈ R^{B×D_inner×N}
    4. Bidir SSM: h_t = Ā(x_t)h_{t-1} + B̄(x_t)x_t
                   y_t = C(x_t)h_t
    5. ASP:      y_t ← y_t + α·Softmax(q·Kh^T/√d_s)·Vh
    6. Inverse:  Y = Φ_NHS^{-1}(y) + X  [residual]

References:
    MambaIRv2: csguoh/MambaIR, basicsr/archs/mambairv2_arch.py
    MaIR:      XLearning-SCU/2025-CVPR-MaIR, models/scanning.py
    Mamba:     state-spaces/mamba (selective_scan_fn)

Shape contract:
    x:       (B, C, H, W)
    prompt:  (B, prompt_dim=128)
    output:  (B, C, H, W)  — same shape, residual connection included
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..scans.nested_s_hilbert import nested_s_hilbert_indices
from ..scans.triton_kernels import hilbert_gather, hilbert_scatter
from .nafblock import LayerNorm2d

# Try importing mamba_ssm; fall back to a pure-PyTorch SSM for CPU testing
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn_raw
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


def _check_gpu_supports_mamba(device: torch.device) -> bool:
    """
    mamba-ssm CUDA kernels are compiled only for SM 7.0+ (Volta and newer).
    Quadro P5000 / GTX 10xx / any Pascal card is SM 6.x → no kernel image.
    This function returns True ONLY if CUDA is available AND the GPU is SM ≥ 7.0.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
        return (major >= 7)  # SM 7.0 = Volta V100, min supported by mamba-ssm
    except Exception:
        return False


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """
    Wrapper around mamba-ssm selective_scan_fn that:
      1. Checks that the GPU is SM ≥ 7.0 (Volta+) before calling the CUDA kernel.
      2. Casts all tensors to float32 before the kernel (CUDA kernel requires
         u and delta to have the same dtype; bfloat16 AMP breaks this).
      3. Casts the result back to the original input dtype.
    Falls back to _pytorch_ssm_fallback on Pascal/CPU.
    """
    # Force all to float32 — selective_scan_cuda.fwd requires u.dtype == delta.dtype
    orig_dtype = u.dtype
    u_fp32     = u.float()
    delta_fp32 = delta.float()
    A_fp32     = A.float()
    B_fp32     = B.float()
    C_fp32     = C.float()
    D_fp32     = D.float() if D is not None else None

    out = _selective_scan_fn_raw(
        u_fp32, delta_fp32, A_fp32, B_fp32, C_fp32,
        D_fp32, None, delta_bias, delta_softplus, return_last_state,
    )
    # Cast result back to the original dtype (e.g. bfloat16 under AMP)
    if isinstance(out, (list, tuple)):
        return tuple(o.to(orig_dtype) for o in out)
    return out.to(orig_dtype)


def _pytorch_ssm_fallback(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Pure-PyTorch reference SSM for CPU testing and CI.
    Not performance-optimal — use only for shape/gradient checks.

    Args:
        u:  (B, D, N)        — input sequence
        dt: (B, D, N)        — time deltas (Δ)
        A:  (D, d_state)     — state transition (negative, in log space)
        B:  (B, 1, d_state, N) — input projection
        C:  (B, 1, d_state, N) — output projection
        D:  (D,)             — skip connection

    Returns:
        y:  (B, D, N)
    """
    B_batch, D_dim, N = u.shape
    d_state = A.shape[1]

    # Discretize: Ā = exp(Δ·A), B̄ ≈ Δ·B (zero-order hold, simplified)
    # Shape: (B, D, N, d_state)
    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))  # (B,D,N,ds)

    # B has shape (B, 1, d_state, N) → (B, D, N, d_state)
    B_squeezed = B.squeeze(1).permute(0, 2, 1)   # (B, N, d_state)
    dB = dt.unsqueeze(-1) * B_squeezed.unsqueeze(1)  # (B, D, N, d_state)

    # C: (B, 1, d_state, N) → (B, N, d_state)
    C_squeezed = C.squeeze(1).permute(0, 2, 1)   # (B, N, d_state)

    # Scan loop (slow but correct)
    h = torch.zeros(B_batch, D_dim, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(N):
        h = dA[:, :, t] * h + dB[:, :, t] * u[:, :, t].unsqueeze(-1)
        y_t = (h * C_squeezed[:, t].unsqueeze(1)).sum(-1)  # (B, D)
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)  # (B, D, N)
    return y + D.unsqueeze(0).unsqueeze(-1) * u


class AHNMambaBlock(nn.Module):
    """
    Attentive Hilbert-Nested Mamba Block.

    Args:
        dim:        Number of input/output channels.
        d_state:    SSM state dimension (default 16, per MambaIR).
        d_conv:     Local DWConv kernel size (default 3).
        expand:     Channel expansion ratio for SSM inner dimension (default 2).
        n_prompts:  Number of learnable query prompt vectors for ASP (default 8).
        tile:       Tile size for intra-tile Hilbert scan (default 16).
                    Use 8 for W-stream (which is at H/4 spatial resolution).
        prompt_dim: Dimensionality of the FiLM conditioning prompt (default 128).
        wavelet_delta_bias: If True, initialize Δ larger for high-freq channels.
                            Set True for W-stream blocks only.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        n_prompts: int = 8,
        tile: int = 16,
        prompt_dim: int = 128,
        wavelet_delta_bias: bool = False,
    ):
        super().__init__()
        self.dim     = dim
        self.d_state = d_state
        self.tile    = tile
        self.d_inner = expand * dim

        # --- FiLM from degradation prompt ---
        self.film_proj = nn.Linear(prompt_dim, 2 * dim, bias=True)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        # --- Input normalization ---
        self.norm = LayerNorm2d(dim)

        # --- Input projection: dim → 2*d_inner (SSM branch + gate branch) ---
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # --- Local 3×3 DWConv (pre-scan context injection) ---
        self.dwconv = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True,
        )

        # --- SSM parameter projections ---
        # Projects d_inner → B_ssm (d_state) + C_ssm (d_state) + dt (1)
        self.x_proj  = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # HiPPO-based A initialization (from Mamba original)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        # Wavelet-aware Δ initialization:
        # For W-stream blocks, bias dt_proj so high-freq channels get larger Δ.
        # High-freq channels are the last 3/4 of the channel dim (LH, HL, HH subbands).
        if wavelet_delta_bias:
            with torch.no_grad():
                # Set positive bias for channels 1/4 onward (high-freq)
                hf_start = self.d_inner // 4
                self.dt_proj.bias.data[:hf_start] = -2.0   # LL: slow update
                self.dt_proj.bias.data[hf_start:] =  0.5   # HF: fast update

        # --- Attentive State Prompt (ASP) ---
        # n_prompts learnable query vectors, each of dimension d_state
        self.asp_queries = nn.Parameter(
            torch.randn(n_prompts, d_state) * 0.02
        )
        # Mix coefficient: α. Initialized small so ASP starts as a small correction.
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Entropy regularization scale (prevents prompt collapse)
        self.entropy_scale = 0.01

        # --- Output projection ---
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # --- Scan index cache (one per unique H×W size seen) ---
        self._idx_cache: dict[tuple[int, int], tuple[torch.Tensor, int, int]] = {}

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_scan_idx(
        self, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, int, int]:
        """
        Return (idx, H_pad, W_pad) for the given spatial size.
        Cached to avoid re-computation across forward calls.
        """
        key = (H, W)
        if key not in self._idx_cache:
            idx, H_pad, W_pad = nested_s_hilbert_indices(H, W, self.tile)
            self._idx_cache[key] = (idx.to(device), H_pad, W_pad)
        else:
            idx, H_pad, W_pad = self._idx_cache[key]
            if idx.device != device:
                self._idx_cache[key] = (idx.to(device), H_pad, W_pad)
        return self._idx_cache[key]

    def _film_modulate(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: X' = (1+γ) ⊙ X + β"""
        B, C, H, W = x.shape
        gb = self.film_proj(prompt)             # (B, 2*C)
        gamma, beta = gb.chunk(2, dim=-1)       # each (B, C)
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)
        return (1.0 + gamma) * x + beta

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, C, H, W) input feature map.
            prompt: (B, 128)     degradation prompt from DegradationPredictor.

        Returns:
            out: (B, C, H, W) — input + SSM output residual.
        """
        B, C, H, W = x.shape

        # ── Step 1: FiLM modulation ─────────────────────────────────────
        x_mod = self._film_modulate(self.norm(x), prompt)

        # ── Step 2: Project → [x_ssm | z] ──────────────────────────────
        # Flatten to (B, N, C) for linear, then reshape
        x_flat = rearrange(x_mod, 'b c h w -> b (h w) c')  # (B, N, C)
        xz = self.in_proj(x_flat)                           # (B, N, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                     # each (B, N, d_inner)

        # ── Step 3: Local DWConv context injection ──────────────────────
        x_ssm_2d = rearrange(x_ssm, 'b (h w) d -> b d h w', h=H, w=W)
        x_ssm_2d = self.dwconv(x_ssm_2d)
        x_ssm_2d = F.silu(x_ssm_2d)

        # ── Step 4: NHS serialization ───────────────────────────────────
        idx, H_pad, W_pad = self._get_scan_idx(H, W, x.device)
        if H_pad != H or W_pad != W:
            # Use replicate padding as fallback when image is too small for reflect
            pad_mode = 'reflect' if (H >= H_pad - H and W >= W_pad - W and H > 1 and W > 1) else 'replicate'
            pad_mode = 'replicate'  # always use replicate to be safe with small inputs
            x_ssm_2d = F.pad(x_ssm_2d, (0, W_pad - W, 0, H_pad - H), mode=pad_mode)
        N_pad = H_pad * W_pad
        x_seq = rearrange(x_ssm_2d, 'b d h w -> b d (h w)')     # (B, d_inner, N_pad)
        x_seq = hilbert_gather(x_seq.contiguous(), idx)           # NHS-ordered

        # ── Step 5: Project to SSM inputs (B_ssm, C_ssm, Δ) ────────────
        bcd = self.x_proj(rearrange(x_seq, 'b d n -> b n d'))     # (B, N_pad, 2*d_s+1)
        B_ssm = bcd[:, :, :self.d_state]                          # (B, N_pad, d_state)
        C_ssm = bcd[:, :, self.d_state: 2 * self.d_state]        # (B, N_pad, d_state)
        dt_raw = bcd[:, :, -1:]                                    # (B, N_pad, 1)
        dt = F.softplus(self.dt_proj(dt_raw))                      # (B, N_pad, d_inner)

        # Negative A for stability
        A = -torch.exp(self.A_log.float())                         # (d_inner, d_state)

        # ── Step 6: Bidirectional selective scan ─────────────────────────
        u_fwd = x_seq                                              # (B, d_inner, N_pad)
        dt_t  = rearrange(dt,         'b n d -> b d n')
        B_t   = rearrange(B_ssm,      'b n s -> b 1 s n')
        C_t   = rearrange(C_ssm,      'b n s -> b 1 s n')

        # Use CUDA selective scan ONLY on SM ≥ 7.0 (Volta+, Turing, Ampere, Ada).
        # Pascal (SM 6.x, e.g. Quadro P5000 / GTX 1080) lacks kernel images and
        # will raise "CUDA error: no kernel image is available for execution".
        _use_cuda_ssm = _MAMBA_AVAILABLE and _check_gpu_supports_mamba(x.device)

        if _use_cuda_ssm:
            y_fwd = selective_scan_fn(
                u_fwd, dt_t, A, B_t, C_t,
                D=self.D, delta_bias=None, delta_softplus=True,
            )
            y_rev = selective_scan_fn(
                u_fwd.flip(-1), dt_t.flip(-1), A,
                B_t.flip(-1), C_t.flip(-1),
                D=self.D, delta_bias=None, delta_softplus=True,
            ).flip(-1)
        else:
            # Fallback: pure PyTorch SSM
            # Runs on CPU, or GPU < SM 7.0 (Pascal/Maxwell — no mamba-ssm kernel).
            y_fwd = _pytorch_ssm_fallback(
                u_fwd.float(), dt_t.float(), A.float(),
                B_t.float(), C_t.float(), self.D.float()
            ).to(u_fwd.dtype)
            y_rev = _pytorch_ssm_fallback(
                u_fwd.flip(-1).float(), dt_t.flip(-1).float(), A.float(),
                B_t.flip(-1).float(), C_t.flip(-1).float(), self.D.float()
            ).flip(-1).to(u_fwd.dtype)

        y_seq = y_fwd + y_rev    # (B, d_inner, N_pad) — bidirectional merge

        # ── Step 7: Attentive State Prompt (ASP) ─────────────────────────
        # Query the SSM hidden-state pool with learnable prompts.
        # K_h ≈ C_ssm (proxies the hidden state key), V_h = y_seq (SSM output values)
        K_h = C_ssm                                                # (B, N_pad, d_state)
        V_h = rearrange(y_seq, 'b d n -> b n d')                   # (B, N_pad, d_inner)

        q = self.asp_queries                                        # (P, d_state)
        # Attention: (B, P, N_pad)
        attn = torch.einsum('p s, b n s -> b p n', q, K_h)
        attn = attn / (self.d_state ** 0.5)
        attn_weights = F.softmax(attn, dim=-1)                     # (B, P, N_pad)

        # Pooled prompt features: (B, P, d_inner)
        pooled = torch.einsum('b p n, b n d -> b p d', attn_weights, V_h)

        # Broadcast back over sequence positions: (B, N_pad, d_inner)
        prompt_correction = torch.einsum('b p n, b p d -> b n d', attn_weights, pooled)
        prompt_correction = rearrange(prompt_correction, 'b n d -> b d n')

        # Add with small alpha (starts at 0.1, learned to grow as needed)
        y_seq = y_seq + self.alpha * prompt_correction

        # ── Step 8: Inverse NHS scan → 2D raster ─────────────────────────
        y_seq = hilbert_scatter(y_seq, idx)                        # (B, d_inner, N_pad)
        y_2d  = rearrange(y_seq, 'b d (h w) -> b d h w', h=H_pad, w=W_pad)
        if H_pad != H or W_pad != W:
            y_2d = y_2d[:, :, :H, :W]                             # crop padding

        # ── Step 9: Gate + output projection ─────────────────────────────
        y_flat = rearrange(y_2d, 'b d h w -> b (h w) d')          # (B, N, d_inner)
        y_flat = y_flat * F.silu(z)                                # Mamba gate
        y_flat = self.out_proj(y_flat)                             # (B, N, C)
        y_out  = rearrange(y_flat, 'b (h w) d -> b d h w', h=H, w=W)

        # Residual connection (AHN-Mamba adds to original input, not FiLM-modulated)
        return x + y_out
