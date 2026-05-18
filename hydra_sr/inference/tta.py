"""
Test-Time Augmentation (TTA) and tile-based 4K inference for HYDRA-SR.

TTA Strategy (4-way rotation):
  Run model on 4 orientations (0°, 90°, 180°, 270°), rotate outputs back,
  and average. Simple but effective: ~+0.05 dB PSNR for free at 4× compute cost.

Tile-based 4K inference:
  LR 4K inputs (960×540) cannot fit in most GPUs as a single batch.
  We split into overlapping tiles of size 192×192 (LR), process each,
  then blend with a Hann window to avoid hard tile boundary artifacts.

  Overlap = 16 pixels (LR space). Hann window makes the blending smooth.

Engineering note (Failure Mode #6):
  Hard tile boundaries → periodic artifacts at 4K. The Hann window is
  MANDATORY. Without it, visible grid patterns appear in the output.
"""

import torch
import torch.nn.functional as F
import math


def _hann_window_2d(size: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Hann window of given size for smooth tile blending."""
    w_1d = torch.hann_window(size, periodic=False, device=device)
    w_2d = w_1d.unsqueeze(0) * w_1d.unsqueeze(1)  # (size, size)
    return w_2d.unsqueeze(0).unsqueeze(0)           # (1, 1, size, size)


def tta_4_rotation(
    model: torch.nn.Module,
    lr: torch.Tensor,
    use_flip: bool = False,
) -> torch.Tensor:
    """
    4-way rotation TTA.

    Args:
        model:    HYDRA-SR model (in eval mode).
        lr:       (B, 3, H, W) LR input.
        use_flip: If True, also include 4 flipped versions (8-way TTA).

    Returns:
        sr_avg: (B, 3, 4H, 4W) averaged SR output.
    """
    outs = []
    augmentations = list(range(4))
    if use_flip:
        augmentations = augmentations + [f'flip_{k}' for k in range(4)]

    for k in augmentations:
        if isinstance(k, str):
            rot_k = int(k.split('_')[1])
            x = torch.flip(torch.rot90(lr, rot_k, dims=(-2, -1)), dims=[-1])
        else:
            rot_k = k
            x = torch.rot90(lr, rot_k, dims=(-2, -1))

        with torch.no_grad():
            y = model(x)

        # Inverse transform
        if isinstance(k, str):
            y = torch.flip(y, dims=[-1])
            y = torch.rot90(y, -rot_k, dims=(-2, -1))
        else:
            y = torch.rot90(y, -rot_k, dims=(-2, -1))

        outs.append(y)

    return torch.stack(outs).mean(0)


def tile_inference(
    model: torch.nn.Module,
    lr: torch.Tensor,
    tile: int = 192,
    overlap: int = 16,
    scale: int = 4,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Tile-based inference for 4K images.

    Args:
        model:    HYDRA-SR model (eval mode, no_grad context outside).
        lr:       (1, 3, H, W) LR input — single image (B=1 for 4K).
        tile:     Tile size in LR pixels.
        overlap:  Overlap in LR pixels between adjacent tiles.
        scale:    SR scale factor.
        use_amp:  If True, run tiles in bfloat16 for memory savings.

    Returns:
        sr: (1, 3, H*scale, W*scale) full SR output.
    """
    B, C, H, W = lr.shape
    assert B == 1, "tile_inference expects batch size 1"

    H_hr, W_hr = H * scale, W * scale
    sr      = torch.zeros(B, C, H_hr, W_hr, device=lr.device, dtype=lr.dtype)
    weight  = torch.zeros(B, 1, H_hr, W_hr, device=lr.device, dtype=lr.dtype)

    stride = tile - overlap

    # Hann window for blending (in HR space)
    tile_hr = tile * scale
    hann = _hann_window_2d(tile_hr, device=lr.device).to(lr.dtype)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile, H)
            x2 = min(x + tile, W)
            y1_actual = y2 - tile  # allow negative → use 0 clamped
            x1_actual = x2 - tile

            # Clamp to valid range
            y_start = max(y1_actual, 0)
            x_start = max(x1_actual, 0)

            patch_lr = lr[:, :, y_start:y2, x_start:x2]  # (1, 3, ph, pw)

            # Pad if needed (edge tiles may be smaller)
            ph, pw = patch_lr.shape[-2], patch_lr.shape[-1]
            pad_h = tile - ph
            pad_w = tile - pw
            if pad_h > 0 or pad_w > 0:
                patch_lr = F.pad(patch_lr, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                if use_amp and lr.is_cuda:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        patch_sr = model(patch_lr).float()
                else:
                    patch_sr = model(patch_lr)

            # Crop back if padded
            if pad_h > 0 or pad_w > 0:
                patch_sr = patch_sr[:, :, :ph * scale, :pw * scale]

            # Accumulate with Hann weight
            y_hr_s = y_start * scale
            x_hr_s = x_start * scale
            h_hr   = ph * scale
            w_hr   = pw * scale

            hann_patch = hann[:, :, :h_hr, :w_hr]
            sr    [:, :, y_hr_s:y_hr_s+h_hr, x_hr_s:x_hr_s+w_hr] += patch_sr * hann_patch
            weight[:, :, y_hr_s:y_hr_s+h_hr, x_hr_s:x_hr_s+w_hr] += hann_patch

    # Normalize by accumulated weights
    sr = sr / weight.clamp_min(1e-6)
    return sr.clamp(0.0, 1.0)
