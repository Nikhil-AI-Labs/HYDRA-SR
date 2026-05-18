"""
CRITICAL sanity test: Overfit one image.

RUN THIS BEFORE EVERY MULTI-DAY TRAINING RUN.

If HYDRA-SR cannot overfit ONE image to 40+ dB PSNR in 500 iterations,
there is a fundamental bug. This test catches 90% of architectural bugs
(wrong scan/inverse-scan, shape mismatch, broken residual, etc.)
in < 5 minutes, saving days of wasted training.

Expected results:
  - Iteration 0:   ~20 dB PSNR (random weights)
  - Iteration 100: ~30+ dB PSNR
  - Iteration 500: ≥ 40 dB PSNR  ← acceptance criterion

Uses a synthetic LR/HR pair (no real data needed).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import math
import pytest


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. Inputs in [0, 1]."""
    mse = ((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10 * math.log10(mse)


def make_synthetic_lr_hr(H: int = 32, W: int = 32, scale: int = 4, device: str = 'cpu'):
    """
    Create a synthetic HR image and its bicubic-downsampled LR.
    Uses a structured test image (not pure noise) to make overfitting meaningful.
    """
    # Structured test image: checkerboard + gradient
    y_grid = torch.linspace(0, 1, H * scale).view(-1, 1).expand(H * scale, W * scale)
    x_grid = torch.linspace(0, 1, W * scale).view(1, -1).expand(H * scale, W * scale)
    checker = ((torch.arange(H * scale).view(-1, 1) + torch.arange(W * scale).view(1, -1)) % 2).float()
    hr = (0.5 * y_grid + 0.3 * x_grid + 0.2 * checker).clamp(0, 1)
    hr = hr.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).to(device)  # (1, 3, 4H, 4W)

    lr = F.interpolate(hr, scale_factor=1/scale, mode='bicubic', align_corners=False)  # (1, 3, H, W)
    return lr.clamp(0, 1), hr.clamp(0, 1)


def test_overfit_one_image_cpu():
    """
    CRITICAL: Overfit one image to ≥ 40 dB PSNR in 500 iters on CPU.
    Uses a small model config for speed.
    """
    from hydra_sr.models.hydra_sr import HYDRASR

    torch.manual_seed(42)
    device = 'cpu'
    model  = HYDRASR(
        scale=4,
        dim_p=32,      # small for CPU speed
        dim_w=16,
        n_mamba_p=1,
        n_mamba_w=1,
        n_transformer=1,
        n_nafblocks_s1=2,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    lr, hr = make_synthetic_lr_hr(H=16, W=16, scale=4, device=device)

    psnr_vals = []
    for i in range(500):
        opt.zero_grad()
        sr   = model(lr)
        loss = F.l1_loss(sr, hr)
        loss.backward()
        opt.step()

        if i % 100 == 0:
            with torch.no_grad():
                psnr = compute_psnr(model(lr), hr)
            psnr_vals.append((i, psnr, loss.item()))
            print(f"  iter {i:4d}: loss={loss.item():.4f}, PSNR={psnr:.2f} dB")

    final_psnr = compute_psnr(model(lr).detach(), hr)
    print(f"\n  Final PSNR after 500 iters: {final_psnr:.2f} dB")

    assert final_psnr >= 40.0, (
        f"HYDRA-SR failed to overfit one image: final PSNR {final_psnr:.2f} dB < 40 dB. "
        "STOP! Debug these areas:\n"
        "  1. Run test_hilbert_kernel.py → check gather/scatter roundtrip\n"
        "  2. Check AHNMambaBlock residual connection (x + y_out)\n"
        "  3. Check CDB lam_p, lam_w initialization (should be 0)\n"
        "  4. Check upsampler bicubic residual connection\n"
        "  5. Print forward outputs at each stage for shape/NaN inspection"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overfit_one_image_gpu_full_size():
    """
    GPU version with full model size. Should be much faster than CPU.
    """
    from hydra_sr.models.hydra_sr import HYDRASR

    torch.manual_seed(42)
    model = HYDRASR().cuda()
    model.train()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    lr, hr = make_synthetic_lr_hr(H=32, W=32, scale=4, device='cuda')

    for i in range(500):
        opt.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            sr   = model(lr)
            loss = F.l1_loss(sr.float(), hr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_psnr = compute_psnr(model(lr).float(), hr)

    print(f"\n  GPU Full-size final PSNR: {final_psnr:.2f} dB")
    assert final_psnr >= 40.0, (
        f"HYDRA-SR GPU overfit failed: {final_psnr:.2f} dB < 40 dB"
    )


if __name__ == '__main__':
    # Run directly for quick sanity check before training
    test_overfit_one_image_cpu()
    print("\n✅ Overfit test PASSED — safe to start training!")
