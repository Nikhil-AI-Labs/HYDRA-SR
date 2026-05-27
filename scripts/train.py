"""
Master training script for HYDRA-SR.

Supports all three training stages (Stage 1/2/3) via config YAML.
Uses PyTorch DDP (torchrun) for multi-GPU training.

Usage:
  # Single GPU:
  python scripts/train.py --config configs/train_stage1_geometry.yml

  # Multi-GPU (DDP) — MANDATORY for ≥ Stage 2 (use torchrun, NOT ThreadPoolExecutor):
  torchrun --nproc_per_node=2 scripts/train.py --config configs/train_stage1_geometry.yml

  # Resume from checkpoint:
  python scripts/train.py --config configs/train_stage1_geometry.yml --resume ./checkpoints/stage1/latest.pth

Engineering notes (Failure Mode #10):
  - NEVER use ThreadPoolExecutor with PyTorch — DDP hang.
  - Always use torchrun (or torch.distributed.launch) for multi-GPU.
  - Differential weight decay: Mamba params get wd=1e-5, others 1e-4.
  - bfloat16 (NOT float16) — fp16 destabilizes SSMs at low learning rates.
"""

import os
import sys
import argparse
import contextlib
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra_sr.models.hydra_sr import HYDRASR
from hydra_sr.losses import Stage1Loss, Stage2Loss, Stage3Loss
from hydra_sr.utils.ema import EMA
from hydra_sr.utils.metrics import MetricCalculator
from hydra_sr.data.div2k_dataset import DIV2KDataset


logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(asctime)s] %(message)s')
logger = logging.getLogger('HYDRA-SR')


def setup_ddp():
    """Initialize DDP from environment variables set by torchrun."""
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def build_optimizer(model: HYDRASR, cfg):
    """
    Differential weight decay:
      Mamba SSM parameters: wd = mamba_weight_decay (1e-5)
      All other parameters: wd = weight_decay (1e-4)
    """
    mamba_params = []
    other_params = []
    mamba_keywords = ['A_log', 'D', 'dt_proj', 'x_proj', 'in_proj', 'out_proj',
                      'dwconv', 'asp_queries', 'alpha']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_mamba = any(kw in name for kw in mamba_keywords)
        if is_mamba:
            mamba_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': mamba_params, 'weight_decay': cfg.train.optimizer.mamba_weight_decay},
        {'params': other_params, 'weight_decay': cfg.train.optimizer.weight_decay},
    ], lr=cfg.train.optimizer.lr, betas=cfg.train.optimizer.get('betas', [0.9, 0.99]))

    return optimizer, mamba_params, other_params


def build_scheduler(optimizer, cfg, total_iter: int):
    """Cosine annealing with linear warmup."""
    warmup_iters = cfg.train.scheduler.warmup_iter
    eta_min      = cfg.train.scheduler.eta_min

    def lr_lambda(step):
        if step < warmup_iters:
            return step / max(warmup_iters, 1)
        progress = (step - warmup_iters) / max(total_iter - warmup_iters, 1)
        import math
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        # Scale to [eta_min/lr, 1.0]
        base_lr = cfg.train.optimizer.lr
        return eta_min / base_lr + (1 - eta_min / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def determine_stage(cfg) -> int:
    """Determine training stage from config name."""
    name = cfg.name.lower()
    if 'stage1' in name or 'geometry' in name:
        return 1
    elif 'stage2' in name or 'frequency' in name:
        return 2
    elif 'stage3' in name or 'perceptual' in name:
        return 3
    return 1


def train(cfg, args):
    # ── DDP setup ───────────────────────────────────────────────────────
    use_ddp = 'LOCAL_RANK' in os.environ
    if use_ddp:
        rank, world_size = setup_ddp()
        device = torch.device(f'cuda:{rank}')
    else:
        rank, world_size = 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main = (rank == 0)

    # ── Model ────────────────────────────────────────────────────────────
    model_cfg = cfg.model
    model = HYDRASR(
        scale=model_cfg.scale,
        dim_p=model_cfg.dim_p,
        dim_w=model_cfg.dim_w,
        n_mamba_p=model_cfg.n_mamba_p,
        n_mamba_w=model_cfg.n_mamba_w,
        n_nafblocks_s1=model_cfg.n_nafblocks_s1,
        n_transformer=model_cfg.n_transformer,
        prompt_dim=model_cfg.get('prompt_dim', 128),
        J=model_cfg.get('J', 2),
        wave=model_cfg.get('wave', 'db4'),
        use_checkpoint=model_cfg.get('use_checkpoint', True),
        upsampler_mid_dim=model_cfg.get('upsampler_mid_dim', 64),
    ).to(device)

    if is_main:
        n_params = model.count_parameters()
        logger.info(f"HYDRA-SR trainable parameters: {n_params/1e6:.2f}M")

    # Load pretrained weights if resuming a stage
    resume_path = getattr(model_cfg, 'resume_from', None)
    if resume_path and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state.get('model', state), strict=False)
        if is_main:
            logger.info(f"Loaded weights from {resume_path}")

    # Wrap in DDP
    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Optimizer + Scheduler ────────────────────────────────────────────
    raw_model = model.module if use_ddp else model
    optimizer, mamba_params, other_params = build_optimizer(raw_model, cfg)
    scheduler = build_scheduler(optimizer, cfg, cfg.train.total_iter)

    # AMP: bfloat16 does NOT need GradScaler (only fp16 does)
    use_amp   = cfg.train.get('amp', {}).get('enabled', True)
    amp_dtype = (torch.bfloat16
                 if cfg.train.get('amp', {}).get('dtype', 'bfloat16') == 'bfloat16'
                 else torch.float16)
    scaler    = (torch.amp.GradScaler('cuda')
                 if (use_amp and amp_dtype == torch.float16)
                 else None)

    # ── EMA ─────────────────────────────────────────────────────────────
    ema = EMA(raw_model, decay=cfg.train.ema.decay, update_every=cfg.train.ema.update_every)

    # Resume training state if resuming
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if use_ddp:
            model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        ema.load_state_dict(ckpt['ema'])
        start_step = ckpt.get('step', 0)
        if is_main:
            logger.info(f"Resumed from step {start_step}")

    # ── Dataset + DataLoader ─────────────────────────────────────────────
    train_cfg = cfg.train
    ds_cfg    = train_cfg.dataset
    aug_cfg   = ds_cfg.get('augmentation', {})

    # Build patch curriculum from config  [(iter_threshold, patch_size), ...]
    raw_curriculum = train_cfg.get('patch_curriculum', [])
    patch_curriculum = [(entry['iter'], entry['patch_size'])
                        for entry in raw_curriculum]

    train_dataset = DIV2KDataset(
        hr_root=ds_cfg.hr_root,
        lr_root=ds_cfg.get('lr_root', None),
        patch_size=ds_cfg.patch_size,
        scale=model_cfg.scale,
        train=True,
        use_real_degradation=ds_cfg.get('use_real_degradation', False),
        real_deg_weight=ds_cfg.get('real_deg_weight', 0.0),
        # Augmentation flags from config (all default True if not specified)
        use_hflip=aug_cfg.get('use_hflip', True),
        use_vflip=aug_cfg.get('use_vflip', True),
        use_rotation=aug_cfg.get('use_rotation', True),
        use_channel_shuffle=aug_cfg.get('use_channel_shuffle', True),
        use_cutblur=aug_cfg.get('use_cutblur', True),
        cutblur_prob=aug_cfg.get('cutblur_prob', 0.3),
        patch_curriculum=patch_curriculum,
    )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.dataset.batch_size // world_size,
        sampler=sampler,
        num_workers=train_cfg.dataset.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    # ── Loss function ────────────────────────────────────────────────────
    stage = determine_stage(cfg)
    if stage == 1:
        loss_fn = Stage1Loss(
            charb_weight=cfg.train.loss.get('charbonnier', 1.0),
            deg_pred_weight=cfg.train.loss.get('degradation_pred', 0.1),
        )
    elif stage == 2:
        loss_fn = Stage2Loss(
            l1_weight=cfg.train.loss.get('l1_weight', 0.8),
            ffl_weight=cfg.train.loss.get('ffl_weight', 0.6),
            swt_weight=cfg.train.loss.get('swt_weight', 0.3),
            wl1_weight=cfg.train.loss.get('wl1_weight', 0.4),
        )
    else:
        loss_fn = Stage3Loss(
            use_tsd=True,
            gan_weight=cfg.train.loss.get('gan_weight', 0.1),
        )

    val_metric = MetricCalculator(scale=model_cfg.scale, device=str(device))
    checkpoint_dir = Path(cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ────────────────────────────────────────────────────
    total_iter  = cfg.train.total_iter
    val_every   = cfg.val.every
    save_every  = cfg.logging.save_every
    log_every   = cfg.logging.get('log_every', 100)

    iter_loader = iter(train_loader)
    model.train()

    for step in range(start_step, total_iter):
        # ── Progressive patch curriculum ─────────────────────────────────
        # update the dataset's active patch size from the curriculum schedule
        prev_patch = train_dataset.patch_size
        train_dataset.set_iter(step)
        if train_dataset.patch_size != prev_patch:
            # Patch size changed — rebuild DataLoader (batch shape changes)
            if is_main:
                logger.info(
                    f"Step {step}: patch_size {prev_patch} → {train_dataset.patch_size} "
                    f"(LR: {train_dataset.patch_size}×{train_dataset.patch_size}, "
                    f"HR: {train_dataset.patch_size*model_cfg.scale}×"
                    f"{train_dataset.patch_size*model_cfg.scale})"
                )
            if sampler is not None:
                sampler.set_epoch(step)
            iter_loader = iter(DataLoader(
                train_dataset,
                batch_size=ds_cfg.batch_size // world_size,
                sampler=sampler,
                num_workers=ds_cfg.get('num_workers', 4),
                pin_memory=True,
                drop_last=True,
            ))

        # Get next batch (cycle through dataloader)
        try:
            batch = next(iter_loader)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        lr_img = batch['lr'].to(device, non_blocking=True)
        hr_img = batch['hr'].to(device, non_blocking=True)
        deg_gt = batch.get('deg_vec', None)
        if deg_gt is not None:
            deg_gt = deg_gt.to(device, non_blocking=True)

        # Forward pass with AMP
        optimizer.zero_grad(set_to_none=True)
        amp_ctx = (torch.amp.autocast('cuda', dtype=amp_dtype)
                   if use_amp and device.type == 'cuda'
                   else contextlib.nullcontext())
        with amp_ctx:
            sr, aux = model(lr_img, return_aux=True)
            loss_dict = loss_fn(sr.float(), hr_img.float(),
                                d_hat=aux.get('d_hat'), d_gt=deg_gt)

        loss = loss_dict['total']

        # Backward + differential gradient clipping
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Clip mamba params separately
        if mamba_params:
            nn.utils.clip_grad_norm_(mamba_params, cfg.train.grad_clip.get('mamba_max_norm', 1.0))
        nn.utils.clip_grad_norm_(other_params, cfg.train.grad_clip.max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        # EMA update
        ema.update(raw_model)

        # ── Logging ──────────────────────────────────────────────────
        if is_main and step % log_every == 0:
            lr_now = optimizer.param_groups[0]['lr']
            loss_str = ' '.join(f"{k}={v.item():.4f}" for k, v in loss_dict.items()
                                if isinstance(v, torch.Tensor))
            logger.info(f"Step {step:6d}/{total_iter} | lr={lr_now:.2e} | {loss_str}")

        # ── Validation ───────────────────────────────────────────────
        if is_main and step % val_every == 0 and step > 0:
            import time
            t0 = time.time()
            ema.ema_model.eval()
            val_metric.reset()

            # Build val loader (once per validation call — cheap, ~100 images)
            val_ds_cfg = cfg.val.dataset
            val_dataset = DIV2KDataset(
                hr_root=val_ds_cfg.hr_root,
                lr_root=val_ds_cfg.get('lr_root', None),
                patch_size=None,          # full-image validation
                scale=model_cfg.scale,
                train=False,
                use_real_degradation=False,
                real_deg_weight=0.0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,             # full-res images vary in size — batch=1
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )

            with torch.no_grad():
                for val_batch in val_loader:
                    val_lr = val_batch['lr'].to(device, non_blocking=True)
                    val_hr = val_batch['hr'].to(device, non_blocking=True)
                    with (torch.amp.autocast('cuda', dtype=amp_dtype)
                          if use_amp and device.type == 'cuda'
                          else contextlib.nullcontext()):
                        val_sr, _ = ema.ema_model(val_lr, return_aux=True)
                    val_metric.update(
                        val_sr.float().clamp(0, 1),
                        val_hr.float().clamp(0, 1),
                    )

            metrics = val_metric.compute()
            elapsed = time.time() - t0

            # Format and log metrics
            psnr  = metrics.get('psnr_y',  float('nan'))
            ssim  = metrics.get('ssim_y',  float('nan'))
            lpips = metrics.get('lpips',   float('nan'))

            logger.info(
                f"\n{'='*60}\n"
                f"  VALIDATION @ step {step:,}/{total_iter:,}  ({elapsed:.1f}s)\n"
                f"  PSNR-Y : {psnr:.4f} dB\n"
                f"  SSIM-Y : {ssim:.6f}\n"
                f"  LPIPS  : {lpips:.6f}\n"
                f"{'='*60}"
            )

            # Track best and save best_ema checkpoint
            if not hasattr(train, '_best_psnr'):
                train._best_psnr = 0.0
            if psnr > train._best_psnr:
                train._best_psnr = psnr
                best_path = checkpoint_dir / 'best_ema.pth'
                torch.save({
                    'step':       step,
                    'model':      ema.ema_model.state_dict(),
                    'metrics':    metrics,
                    'cfg':        OmegaConf.to_container(cfg),
                }, best_path)
                logger.info(f"  ★ New best PSNR-Y {psnr:.4f} dB — saved {best_path}")

            ema.ema_model.train()

        # ── Save checkpoint ──────────────────────────────────────────
        if is_main and step % save_every == 0 and step > 0:
            ckpt_path = checkpoint_dir / f'step_{step:06d}.pth'
            ckpt = {
                'step':      step,
                'model':     raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema':       ema.state_dict(),
                'cfg':       OmegaConf.to_container(cfg),
            }
            torch.save(ckpt, ckpt_path)
            # Also save latest symlink
            latest_path = checkpoint_dir / 'latest.pth'
            torch.save(ckpt, latest_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    if use_ddp:
        cleanup_ddp()

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='HYDRA-SR Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML (e.g., configs/train_stage1_geometry.yml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg, args)


if __name__ == '__main__':
    main()
