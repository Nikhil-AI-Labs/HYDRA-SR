"""
Test/evaluation script for HYDRA-SR.

Runs inference on a test dataset and computes PSNR-Y, SSIM-Y, LPIPS metrics.
Supports both single-image and batch inference with optional TTA.

Usage:
  # Standard evaluation on DIV2K-Val:
  python scripts/test.py \\
    --checkpoint ./checkpoints/stage3/best_ema.pth \\
    --lr_dir /data/DIV2K/valid_LR_bicubic/X4 \\
    --hr_dir /data/DIV2K/valid_HR \\
    --output_dir ./results/div2k_val

  # With TTA:
  python scripts/test.py \\
    --checkpoint ./checkpoints/stage3/best_ema.pth \\
    --lr_dir /data/DIV2K/valid_LR_bicubic/X4 \\
    --hr_dir /data/DIV2K/valid_HR \\
    --tta

  # 4K tile inference:
  python scripts/test.py \\
    --checkpoint ./checkpoints/stage3/best_ema.pth \\
    --lr_dir /data/test_4K_LR \\
    --output_dir ./results/4k \\
    --tile_size 192 --tile_overlap 16
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra_sr.models.hydra_sr import HYDRASR
from hydra_sr.utils.metrics import MetricCalculator
from hydra_sr.inference.tta import tta_4_rotation, tile_inference

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('HYDRA-SR-Test')


def load_model(checkpoint_path: str, device: torch.device) -> HYDRASR:
    """Load HYDRA-SR model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try to get model config from checkpoint
    cfg = ckpt.get('cfg', {})
    model_cfg = cfg.get('model', {})

    model = HYDRASR(
        scale=model_cfg.get('scale', 4),
        dim_p=model_cfg.get('dim_p', 96),
        dim_w=model_cfg.get('dim_w', 64),
        n_mamba_p=model_cfg.get('n_mamba_p', 6),
        n_mamba_w=model_cfg.get('n_mamba_w', 4),
        n_nafblocks_s1=model_cfg.get('n_nafblocks_s1', 4),
        n_transformer=model_cfg.get('n_transformer', 2),
    ).to(device)

    # Try EMA model first, fall back to regular
    state_dict = ckpt.get('ema', {}).get('ema_model', ckpt.get('model', ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path} ({model.count_parameters()/1e6:.2f}M params)")
    return model


def read_img_as_tensor(path: str) -> torch.Tensor:
    """Read image → (1, 3, H, W) float32 tensor in [0, 1]."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def save_img(tensor: torch.Tensor, path: str):
    """Save (1, 3, H, W) tensor → PNG."""
    import cv2
    img = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def main():
    parser = argparse.ArgumentParser(description='HYDRA-SR Test/Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--lr_dir',     type=str, required=True, help='LR images directory')
    parser.add_argument('--hr_dir',     type=str, default=None,  help='HR ground truth (optional for PSNR)')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--tta',        action='store_true', help='Enable 4-way TTA')
    parser.add_argument('--tile_size',  type=int, default=None, help='Tile size for 4K (e.g. 192)')
    parser.add_argument('--tile_overlap', type=int, default=16)
    parser.add_argument('--scale',      type=int, default=4)
    parser.add_argument('--device',     type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.checkpoint, device)

    # Setup output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get LR files
    lr_dir = Path(args.lr_dir)
    lr_files = sorted(lr_dir.glob('*.png')) + sorted(lr_dir.glob('*.jpg'))
    logger.info(f"Found {len(lr_files)} LR images")

    # Setup metrics
    calc_metrics = args.hr_dir is not None
    if calc_metrics:
        metric_calc = MetricCalculator(scale=args.scale, device=str(device))

    all_psnr, all_ssim, all_lpips = [], [], []

    for lr_path in lr_files:
        lr = read_img_as_tensor(lr_path).to(device)

        # Inference
        with torch.no_grad():
            if args.tile_size is not None:
                sr = tile_inference(model, lr, tile=args.tile_size,
                                     overlap=args.tile_overlap, scale=args.scale)
            elif args.tta:
                sr = tta_4_rotation(model, lr)
            else:
                sr = model(lr)

        # Save output
        out_path = output_dir / lr_path.name.replace('x4', '').replace('_LR', '')
        save_img(sr, str(out_path))

        # Compute metrics if GT available
        if calc_metrics:
            hr_path = Path(args.hr_dir) / lr_path.name.replace('x4', '').replace('_LR', '')
            if not hr_path.exists():
                # Try without suffix manipulation
                hr_path = Path(args.hr_dir) / lr_path.stem.split('x')[0]
                hr_candidates = list(Path(args.hr_dir).glob(f'{lr_path.stem.split("x")[0]}*'))
                if hr_candidates:
                    hr_path = hr_candidates[0]

            if hr_path.exists():
                hr = read_img_as_tensor(str(hr_path)).to(device)
                metric_calc.update(sr, hr)

        logger.info(f"  Processed: {lr_path.name}")

    # Print final metrics
    if calc_metrics:
        metrics = metric_calc.compute()
        logger.info("\n" + "="*50)
        logger.info("HYDRA-SR Evaluation Results:")
        for k, v in metrics.items():
            logger.info(f"  {k.upper()}: {v:.4f}")
        logger.info("="*50)


if __name__ == '__main__':
    main()
