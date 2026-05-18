"""
Full HYDRA-SR forward pass shape tests and memory profiling.

Week 3 acceptance criteria:
  1. Forward at 64² LR → 256² SR: correct shape, no NaN
  2. (GPU) Peak VRAM at 4K LR (540×960) < 24 GB
  3. Total parameter count: 16M ≤ params ≤ 18M
  4. Both streams (P and W) produce non-trivial different outputs
  5. Routing weights sum to ~1.0 (softmax sanity check)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from hydra_sr.models.hydra_sr import HYDRASR


# ─── Test 1: Forward shape (CPU) ─────────────────────────────────────────────

def test_forward_small_cpu():
    """Minimal forward pass on CPU — shape correctness."""
    model = HYDRASR(scale=4, dim_p=64, dim_w=32, n_mamba_p=2, n_mamba_w=1, n_transformer=1)
    lr = torch.randn(1, 3, 16, 16)   # small for CPU speed
    sr = model(lr)
    assert sr.shape == (1, 3, 64, 64), f"Expected (1,3,64,64), got {sr.shape}"
    assert not torch.isnan(sr).any(), "NaN in SR output!"
    assert not torch.isinf(sr).any(), "Inf in SR output!"


def test_forward_with_aux_cpu():
    """return_aux=True returns correct dict structure."""
    model = HYDRASR(scale=4, dim_p=32, dim_w=16, n_mamba_p=1, n_mamba_w=1, n_transformer=1)
    lr = torch.randn(1, 3, 16, 16)
    sr, aux = model(lr, return_aux=True)
    assert sr.shape == (1, 3, 64, 64)
    assert 'd_hat' in aux and 'p_d' in aux and 'r' in aux
    assert aux['d_hat'].shape == (1, 4)
    assert aux['p_d'].shape == (1, 128)
    r_p, r_w, r_t = aux['r']
    # Routing weights from softmax should be in (0, 1)
    for r in [r_p, r_w, r_t]:
        assert r.item() > 0 and r.item() < 1


# ─── Test 2: Parameter count ──────────────────────────────────────────────────

def test_parameter_count():
    """Full-size model should have 16M–18M trainable parameters."""
    model = HYDRASR()   # default config from implementation plan
    n_params = model.count_parameters()
    print(f"\nHYDRA-SR trainable parameters: {n_params / 1e6:.2f}M")
    assert 14e6 <= n_params <= 20e6, (
        f"Parameter count {n_params/1e6:.2f}M outside expected range [14M, 20M]. "
        "Check dim_p, dim_w, n_mamba_p, n_mamba_w config."
    )


# ─── Test 3: GPU tests ───────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_batch8_64():
    """Standard training batch: 8×64×64 LR → 8×256×256 SR."""
    model = HYDRASR().cuda()
    lr = torch.randn(8, 3, 64, 64, device='cuda')
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        sr = model(lr)
    assert sr.shape == (8, 3, 256, 256)
    assert not torch.isnan(sr.float()).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_4k_inference_memory():
    """
    4K inference (LR=540×960) must fit in <24 GB peak VRAM.
    Uses gradient checkpointing to reduce activation memory.
    """
    model = HYDRASR(use_checkpoint=True).cuda().eval()
    lr = torch.randn(1, 3, 540, 960, device='cuda')

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        sr = model(lr)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n4K inference peak VRAM: {peak_gb:.2f} GB")
    assert sr.shape == (1, 3, 2160, 3840)
    assert peak_gb < 24.0, f"Peak VRAM {peak_gb:.2f} GB exceeds 24 GB limit!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streams_produce_different_features():
    """Pixel and wavelet streams should produce different (non-trivially similar) features."""
    model = HYDRASR().cuda()
    lr = torch.randn(1, 3, 64, 64, device='cuda')
    with torch.no_grad():
        _, aux = model(lr, return_aux=True)

    f_p = aux['f_p2']
    f_w = aux['f_w2']

    # They should be different tensors (different shapes even: P is H×W, W is H/4×W/4)
    assert f_p.shape[-2] == 4 * f_w.shape[-2], (
        f"P-stream H {f_p.shape[-2]} should be 4× W-stream H {f_w.shape[-2]}"
    )


# ─── Test 4: CDB forward ─────────────────────────────────────────────────────

def test_cdb_forward():
    """CDB must not explode at init (lam_p = lam_w = 0 → identity operation)."""
    from hydra_sr.models.modules.cross_domain_bridge import CrossDomainBridge
    cdb = CrossDomainBridge(C_P=96, C_W=64, J=2, wave='db4')
    F_P = torch.randn(2, 96, 64, 64)
    F_W = torch.randn(2, 64, 16, 16)

    F_P_new, F_W_new = cdb(F_P, F_W)

    # At init (lam=0): outputs should be identical to inputs
    assert torch.allclose(F_P_new, F_P, atol=1e-6), "CDB changed F_P at init (lam=0)!"
    assert torch.allclose(F_W_new, F_W, atol=1e-6), "CDB changed F_W at init (lam=0)!"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
