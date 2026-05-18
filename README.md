# HYDRA-SR

**Hierarchical Yoked Dual-domain Restoration Architecture for Super-Resolution**

[![NTIRE 2026 Target](https://img.shields.io/badge/NTIRE%202026-Top--3%20Target-red)](https://ntire.github.io/)
[![Parameters](https://img.shields.io/badge/Params-16.9M-blue)](https://github.com/)
[![PSNR Target](https://img.shields.io/badge/PSNR%20Target-33.8--34.3%20dB-green)](https://github.com/)

---

## Architecture Overview

HYDRA-SR is a dual-stream, dual-domain super-resolution model combining five
state-of-the-art innovations into a single end-to-end architecture:

```
LR Input (B, 3, H, W)
       │
       ▼
 ┌─────────────────────────────────┐
 │  Degradation Predictor (0.2M)  │  → p_d (128-dim FiLM prompt)
 └─────────────────────────────────┘
       │ p_d broadcast to all blocks
 ┌─────┴──────────────────────────────────────────────────┐
 │                                                        │
 ▼ Stream P (Pixel Domain)         Stream W (Wavelet Domain)
 NAFBlock ×4                       DWT (db4, J=2) → NAFBlock ×4
       │ CDB-1 ←────────────────────────────────────────→ │
 AHN-Mamba ×6                      AHN-Mamba ×4
 (Nested-S-Hilbert, ASP)           (tile=8, Δ-biased to HF)
       │ CDB-2 ←────────────────────────────────────────→ │
       │                                                   │
       └──────────── Adaptive Frequency Router ────────────┘
                             │
                    Deformable Window Attn ×2
                             │
                 Freq-Gated Upsampler (PixelShuffle ×4 + Laplacian)
                             │
                    HR Output (B, 3, 4H, 4W)
```

## Five Core Innovations

| Innovation | Source | HYDRA-SR Novel Combination |
|---|---|---|
| **Dual-Domain Backbone** (Pixel ⇌ Wavelet) | DTWSR (ICCV 2025) | Original synthesis with pixel stream |
| **AHN-Mamba** (Attentive Hilbert-Nested Mamba) | MambaIRv2 (CVPR 2025) + MaIR (CVPR 2025) | NHS scan + ASP in one block |
| **Degradation Prompt** (Blind SR via FiLM) | Real-ESRGAN + FiLM (AAAI 2018) | Joint training from epoch 0 |
| **Adaptive Frequency Router** | MELD-SR (your previous model) | Soft routing weights (not frozen experts) |
| **Two-Teacher Distillation** | TSD-SR (CVPR 2025) + DTWSR | Pixel + wavelet domain teachers combined |

## Expected Performance

| Metric | MELD-SR | HYDRA-SR Target |
|---|---|---|
| DIV2K-Val PSNR-Y | 31.48 dB | **33.8–34.3 dB** |
| SSIM-Y | 0.8815 | **0.920+** |
| LPIPS | 0.2180 | **0.115–0.135** |
| Parameters | 172.59 M | **~16.9 M** |
| Inference 256² | 10,911 ms | **~75 ms** |
| Blind degradation | ❌ | **✅** |
| NTIRE 2026 rank target | 19th | **Top-3** |

---

## Installation

⚠️ **CUDA 12.1 required.** mamba-ssm will not compile on CUDA 11.x.

```bash
# Step 1: Install PyTorch with CUDA 12.1
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install mamba-ssm (ORDER MATTERS)
pip install ninja packaging
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.4 --no-build-isolation   # NOT 2.2.5 (broken)

# Step 3: Install all other deps
pip install -r requirements.txt
```

---

## Quick Start

### Run unit tests (CPU, no CUDA needed)
```bash
# Week 1: Hilbert scan (MUST PASS FIRST)
python -m pytest tests/test_hilbert_kernel.py -v -k "not gpu and not triton_speed"

# Week 2: AHN-Mamba block
python -m pytest tests/test_ahn_mamba.py -v -k "not gpu and not timing"

# Week 3: Full forward pass
python -m pytest tests/test_forward_shapes.py -v -k "not gpu and not 4k"

# CRITICAL: Overfit one image (run before every training run)
python tests/test_overfit_one_image.py
```

### Training

```bash
# Stage 1: Geometry Lock (400K iters)
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_stage1_geometry.yml

# Stage 2: Frequency Lock (250K iters, loads Stage 1 weights)
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_stage2_frequency.yml

# Stage 3: Perceptual Training (120K iters, loads Stage 2 weights)
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_stage3_perceptual.yml
```

### Evaluation

```bash
# Standard evaluation with TTA
python scripts/test.py \
  --checkpoint ./checkpoints/stage3/best_ema.pth \
  --lr_dir /data/DIV2K/valid_LR_bicubic/X4 \
  --hr_dir /data/DIV2K/valid_HR \
  --output_dir ./results/div2k_val \
  --tta

# 4K tile inference
python scripts/test.py \
  --checkpoint ./checkpoints/stage3/best_ema.pth \
  --lr_dir /data/test_4K_LR \
  --output_dir ./results/4k \
  --tile_size 192 --tile_overlap 16
```

---

## Repository Structure

```
hydra-sr/
├── requirements.txt
├── configs/
│   ├── train_stage1_geometry.yml      # 400K iters, Charbonnier
│   ├── train_stage2_frequency.yml     # 250K iters, FFL+WaveletL1
│   └── train_stage3_perceptual.yml    # 120K iters, TSD+LPIPS
├── hydra_sr/
│   ├── models/
│   │   ├── hydra_sr.py                # ← Top-level model (START HERE)
│   │   ├── modules/
│   │   │   ├── attentive_ssm.py       # AHN-Mamba block (the heart)
│   │   │   ├── cross_domain_bridge.py # Pixel ⇌ Wavelet fusion
│   │   │   ├── freq_router.py         # MELD-SR frequency router (modernized)
│   │   │   ├── degradation_predictor.py # Blind SR head
│   │   │   ├── nafblock.py            # Stage 1 local blocks
│   │   │   ├── deformable_window_attn.py # Stage 3
│   │   │   ├── laplacian_pyramid.py   # Stage 4 sharpening
│   │   │   └── film.py                # FiLM conditioning
│   │   └── scans/
│   │       ├── hilbert.py             # CPU Hilbert index (reference)
│   │       ├── nested_s_hilbert.py    # NHS scan (novel combination)
│   │       └── triton_kernels.py      # Triton gather/scatter + PyTorch fallback
│   ├── losses/
│   │   ├── charbonnier.py             # Stage 1
│   │   ├── wavelet_l1.py              # Stage 2 (band-weighted)
│   │   ├── tsd_distill.py             # Stage 3 (TSD-SR)
│   │   ├── lpips_wrap.py              # Stage 3
│   │   └── dynamic_weights.py         # Auto-rebalance Stage 3 losses
│   ├── data/
│   │   ├── div2k_dataset.py           # DIV2K/DF2K/LSDIR loader
│   │   └── degradations.py            # Real-ESRGAN two-order pipeline
│   ├── utils/
│   │   ├── ema.py                     # EMA (decay=0.999, mandatory)
│   │   └── metrics.py                 # PSNR-Y, SSIM-Y, LPIPS
│   └── inference/
│       └── tta.py                     # 4-way TTA + tile-based 4K
├── scripts/
│   ├── train.py                       # Master DDP training entry
│   └── test.py                        # Evaluation script
└── tests/
    ├── test_hilbert_kernel.py          # Week 1 (RUN FIRST)
    ├── test_ahn_mamba.py               # Week 2
    ├── test_forward_shapes.py          # Week 3
    └── test_overfit_one_image.py       # CRITICAL sanity test
```

---

## Critical Engineering Notes

### Top 10 Failure Modes and Fixes

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | NaN within 200 iters | CDB `lam` initialized > 0 | Init to 0, clamp [0, 0.5] ✅ |
| 2 | Loss plateau at ~28 dB | Gather/scatter not inverse | Run `test_hilbert_kernel.py` ✅ |
| 3 | Loss stuck at ~30 dB | `selective_scan_fn` shape wrong | Check B,C shape: `(B, 1, d_state, N)` |
| 4 | OOM at batch=8 | No checkpointing on Stage 2-W | Set `use_checkpoint=True` ✅ |
| 5 | PSNR good, LPIPS bad | TSD teacher LR too high | Use `lr_lora=5e-6` |
| 6 | Tiling artifacts at 4K | Hard tile boundaries | Hann window blend ✅ |
| 7 | Mamba gradient explosion | Δ too large | HiPPO init + grad clip 1.0 ✅ |
| 8 | Color drift | No bicubic residual | Confirmed in upsampler ✅ |
| 9 | Stage-3 mode collapse | TSD weight too high | DynamicWeighter caps λ_tsd ✅ |
| 10 | DDP hang | `ThreadPoolExecutor` | Use `torchrun` only ✅ |

### Key Training Tricks (make-or-break)
- **EMA decay 0.999**: ~+0.1 dB free. Non-negotiable.
- **bfloat16 (NOT float16)**: fp16 destabilizes SSMs at low LR.
- **Differential WD**: Mamba params `wd=1e-5`, others `wd=1e-4`.
- **CDB warmup**: `lam_p = lam_w = 0` for first 5K iters.
- **Degradation predictor from epoch 0**: Don't bolt it on later.

---

## 8-Week Implementation Roadmap

| Week | Deliverable | Acceptance Criterion |
|---|---|---|
| 1 | NHS Triton kernel | Gather/scatter roundtrip + 3× speedup |
| 2 | AHN-Mamba block | Overfit single batch to loss < 0.05 |
| 3 | Full forward pass | 64² LR → 256² SR; params 16–18M |
| 4 | Stage 1 training | ≥ 32.5 dB PSNR-Y on DIV2K-Val |
| 5 | Teacher caching | TSD-SR + DTWSR outputs cached |
| 6 | Stage 2 training | ≥ 33.5 dB PSNR-Y |
| 7 | Stage 3 perceptual | LPIPS ≤ 0.135 |
| 8 | TTA + submission | ≥ 33.8 dB + TTA boost |

---

## References

| Component | Paper | Year | Venue |
|---|---|---|---|
| Attentive State Space | MambaIRv2 | 2024 | CVPR 2025 |
| Nested S-Scan | MaIR | 2025 | CVPR 2025 |
| Hilbert Serialization | FractalMamba++ | 2025 | arXiv |
| Wavelet Backbone | DTWSR | 2025 | ICCV 2025 |
| Score Distillation | TSD-SR | 2025 | CVPR 2025 |
| NAFNet Blocks | NAFNet | 2022 | ECCV 2022 |
| Frequency Router | MELD-SR | 2024 | Your work |
| Real Degradation | Real-ESRGAN | 2021 | ICCV 2021 |
