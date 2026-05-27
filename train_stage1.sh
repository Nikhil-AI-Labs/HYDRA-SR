#!/bin/bash
#SBATCH --job-name=hydra_stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=shard:30
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# =============================================================================
# HYDRA-SR STAGE 1 TRAINING — SVNIT H100 HPC
# =============================================================================
# Architecture : 17.13M params, dim_p=192, dim_w=160
# Target PSNR  : ≥32.5 dB on DIV2K-Val (Week 4 acceptance criterion)
# Val every    : 5000 steps  → PSNR-Y / SSIM-Y printed to this log
# =============================================================================

set -euo pipefail

mkdir -p logs checkpoints/stage1

echo "======================================================="
echo " HYDRA-SR STAGE 1 TRAINING"
echo "======================================================="
echo " HOSTNAME   : $(hostname)"
echo " DATE       : $(date)"
echo " SLURM JOB  : ${SLURM_JOB_ID:-local}"
echo " SLURM NODE : ${SLURMD_NODENAME:-$(hostname)}"
echo "======================================================="

# -----------------------------------------------------------------------------
# CONDA ENVIRONMENT
# -----------------------------------------------------------------------------

source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate /home/akumar/hydra_env

# -----------------------------------------------------------------------------
# CUDA / GPU ENVIRONMENT
# -----------------------------------------------------------------------------

export CUDA_HOME=/apps/codes/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0"          # H100 = SM 9.0

# -----------------------------------------------------------------------------
# PERFORMANCE FLAGS
# -----------------------------------------------------------------------------

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# Triton kernel cache — keeps compilation fast on subsequent runs
export TRITON_CACHE_DIR=$HOME/.triton_cache
mkdir -p $TRITON_CACHE_DIR

# -----------------------------------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------------------------------

REPO_ROOT="$HOME/HYDRA-SR"
cd "$REPO_ROOT"

echo ""
echo "======================================================="
echo " GPU + ENVIRONMENT CHECK"
echo "======================================================="

python - <<EOF
import os, torch, platform

print(f"Python     : {platform.python_version()}")
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA       : {torch.version.cuda}")
print(f"CUDA avail : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU        : {props.name}  (SM {props.major}.{props.minor})")
    print(f"GPU count  : {torch.cuda.device_count()}")
    print(f"GPU memory : {props.total_memory / 1024**3:.1f} GB")
    print(f"BF16       : {torch.cuda.is_bf16_supported()}")

print("")
for d in ["data/train_HR", "data/train_LR", "data/val_HR", "data/val_LR"]:
    n = len([f for f in __import__('pathlib').Path(d).rglob('*') if f.is_file()]) if os.path.exists(d) else 0
    print(f"  {d:20s}  {'EXISTS' if os.path.exists(d) else 'MISSING':8s}  {n:6d} files")
EOF

echo ""
echo "======================================================="
echo " TRAINING CONFIGURATION"
echo "======================================================="
echo "  Config       : configs/train_stage1_geometry.yml"
echo "  Model params : 17.13M  (dim_p=192, dim_w=160)"
echo "  Total iters  : 400,000"
echo "  Val every    : 5,000 steps  →  PSNR-Y / SSIM-Y shown below"
echo "  Batch size   : 16  (from config; 1 GPU → all 16 per step)"
echo "  AMP          : bfloat16 (no GradScaler needed)"
echo "  Checkpoints  : checkpoints/stage1/"
echo ""

# -----------------------------------------------------------------------------
# DETECT GPU COUNT FOR torchrun
# -----------------------------------------------------------------------------

# Single-node, single-GPU on SVNIT HPC (gres=shard:30 = 1 H100)
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "  GPUs visible: $NUM_GPUS"
echo ""

# -----------------------------------------------------------------------------
# LAUNCH TRAINING
# -----------------------------------------------------------------------------

echo "======================================================="
echo " STARTING STAGE 1 TRAINING"
echo "======================================================="
echo ""

if [ "$NUM_GPUS" -gt "1" ]; then
    # Multi-GPU: use torchrun for DDP
    echo "Launching with torchrun (DDP, $NUM_GPUS GPUs)"
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        scripts/train.py \
        --config configs/train_stage1_geometry.yml
else
    # Single GPU: plain python (no DDP overhead)
    echo "Launching single-GPU training (1 H100)"
    python scripts/train.py \
        --config configs/train_stage1_geometry.yml
fi

# -----------------------------------------------------------------------------
# FINISHED
# -----------------------------------------------------------------------------

echo ""
echo "======================================================="
echo " TRAINING FINISHED"
echo "======================================================="
echo " $(date)"
echo ""

# Print final best checkpoint info
if [ -f "checkpoints/stage1/best_ema.pth" ]; then
    echo " Best checkpoint saved at: checkpoints/stage1/best_ema.pth"
    python - <<EOF
import torch
ckpt = torch.load("checkpoints/stage1/best_ema.pth", map_location="cpu")
step    = ckpt.get("step", "?")
metrics = ckpt.get("metrics", {})
print(f" Best @ step {step:,}")
print(f"   PSNR-Y : {metrics.get('psnr_y', float('nan')):.4f} dB")
print(f"   SSIM-Y : {metrics.get('ssim_y', float('nan')):.6f}")
print(f"   LPIPS  : {metrics.get('lpips',  float('nan')):.6f}")
EOF
fi