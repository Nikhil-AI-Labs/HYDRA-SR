"""
DF2K + LSDIR Dataset Loader for HYDRA-SR — with Full SOTA Augmentation Pipeline.

Supports the exact directory layout used for Stage 1 training:

    /data/DF2K_LSDIR/
    ├── HR/
    │   ├── DIV2K_train_HR/          (800 images,  2040×1356 avg)
    │   ├── Flickr2K_HR/             (2650 images, 2048×1365 avg)
    │   └── LSDIR_HR/                (84991 images, 3375×2250 avg)
    └── LR_bicubic/X4/
        ├── DIV2K_train_LR_bicubic_X4/
        ├── Flickr2K_LR_bicubic_X4/
        └── LSDIR_LR_bicubic_X4/

The class is deliberately named DIV2KDataset for backward-compatibility with
train.py and configs — you can pass any HR/LR root(s) and it handles them.

AUGMENTATION PIPELINE (applied in this order, all paired):
──────────────────────────────────────────────────────────
 1. Paired Random Crop              (always, first to cut compute)
 2. RGB Channel Shuffle (50%)       HYDRA-SR addition — prevents color overfitting
    → forces Mamba/Transformer to learn structure, not specific RGB mappings
 3. D4 Dihedral Geometry:
      a. Horizontal Flip (50%)
      b. Vertical Flip   (50%)
      c. 90° Rotation    (k ∈ {0,1,2,3}, uniform)
    → 8 equiprobable geometric orientations covering all dihedral symmetries
 4. CutBlur (optional, default ON for Stage 1, 30% prob)
    → Randomises SR difficulty in the patch; teaches focus on hard regions

Augmentation counts:
  Without channel shuffle: 8 orientations (D4 group)
  With channel shuffle:    8 × 6 = 48 effective distinct orientations per image
  With CutBlur:            continuous variation within each orientation

Progressive Patch Curriculum (from implementation plan):
  Stage 1 iters 0–149K:   patch = 64  (LR) / 256  (HR)
  Stage 1 iters 150K–300K: patch = 96  (LR) / 384  (HR)
  Stage 1 iters 300K–400K: patch = 128 (LR) / 512  (HR)

Reference augmentations inspired by:
  MELD-SR:  src/data/augmentations.py  (SRTrainAugmentation, CutBlur, ColorJitter)
  EDSR:     random 90° rotation + H-flip (original NTIRE winning recipe)
  MambaIR:  adds V-flip to prevent SSM directional bias
  NAFNet:   RGB channel shuffle for color generalization
"""

import cv2
import random
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level image utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_img(path: str) -> np.ndarray:
    """Read image from path → float32 numpy [H,W,3] in [0,1] RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC float32 numpy → CHW float32 tensor (contiguous, no copy if possible)."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def bicubic_downsample(hr: np.ndarray, scale: int) -> np.ndarray:
    """Downsample HR→LR using cv2 bicubic (approximates Matlab bicubic)."""
    h, w = hr.shape[:2]
    lr = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return np.clip(lr, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Paired crop
# ─────────────────────────────────────────────────────────────────────────────

def paired_random_crop(
    hr: np.ndarray,
    lr: np.ndarray,
    hr_patch_size: int,
    scale: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random crop that maintains perfect LR↔HR pixel alignment.

    Args:
        hr:            (H, W, 3) float32
        lr:            (H//scale, W//scale, 3) float32
        hr_patch_size: HR patch size in pixels (LR patch = hr_patch_size // scale)
        scale:         SR scale factor

    Returns:
        (hr_patch, lr_patch): cropped and aligned patches
    """
    lr_patch_size = hr_patch_size // scale

    lr_h, lr_w = lr.shape[:2]

    # Guard: if image is smaller than the requested patch, resize both
    if lr_h < lr_patch_size or lr_w < lr_patch_size:
        lr = cv2.resize(lr, (max(lr_w, lr_patch_size), max(lr_h, lr_patch_size)),
                        interpolation=cv2.INTER_CUBIC)
        hr = cv2.resize(hr, (lr.shape[1] * scale, lr.shape[0] * scale),
                        interpolation=cv2.INTER_CUBIC)
        lr_h, lr_w = lr.shape[:2]

    rnd_h = random.randint(0, lr_h - lr_patch_size)
    rnd_w = random.randint(0, lr_w - lr_patch_size)

    lr_patch = lr[rnd_h: rnd_h + lr_patch_size,
                  rnd_w: rnd_w + lr_patch_size]
    hr_patch = hr[rnd_h * scale: (rnd_h + lr_patch_size) * scale,
                  rnd_w * scale: (rnd_w + lr_patch_size) * scale]
    return hr_patch, lr_patch


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation primitives
# ─────────────────────────────────────────────────────────────────────────────

def _channel_shuffle(imgs: List[np.ndarray]) -> List[np.ndarray]:
    """
    RGB Channel Permutation — applied identically to every image in the list.

    Why: Forces Mamba/Transformer to learn spatial structure rather than
    memorising colour-domain statistics (e.g. "blue always means sky").
    Without this, models trained only on RGB data learn RGB-specific priors
    that hurt generalisation to different cameras/colour profiles.

    The SAME random permutation is applied to both LR and HR so that the
    pixel alignment contract is never violated.
    """
    perm = np.random.permutation(3).tolist()
    return [img[:, :, perm] for img in imgs]


def augment(
    imgs: List[np.ndarray],
    hflip: bool = True,
    vflip: bool = True,
    rotation: bool = True,
    channel_shuffle: bool = True,
) -> List[np.ndarray]:
    """
    Full D4 Dihedral augmentation + RGB channel shuffle for SR pairs.

    Application order:
        1. Channel shuffle  (must be before spatial ops for contiguity)
        2. Horizontal flip
        3. Vertical flip
        4. 90° rotation

    This order means the total set of possible transforms is:
        6 channel perms × 2 (H-flip) × 2 (V-flip) × 4 (rotation) = 96
    but since channel shuffle is 50% probable:
        50% × 6 perms × 8 geometries = 24 effective augmentations on average.

    Args:
        imgs:             List of images, all [H, W, 3] float32 in [0,1].
                          MUST include the same-length spatial pair [hr, lr].
        hflip:            Enable horizontal flip (50% prob).
        vflip:            Enable vertical flip (50% prob).
        rotation:         Enable random 90° rotation (uniform over {0°,90°,180°,270°}).
        channel_shuffle:  Enable RGB channel permutation (50% prob).

    Returns:
        Augmented images in the same order, all contiguous.
    """
    # ── 1. Channel shuffle ────────────────────────────────────────────────
    if channel_shuffle and random.random() < 0.5:
        imgs = _channel_shuffle(imgs)

    # ── 2–4. Geometry — one random decision applied identically ──────────
    hflip_flag = hflip and random.random() < 0.5
    vflip_flag = vflip and random.random() < 0.5
    rot_k      = random.randint(0, 3) if rotation else 0

    result = []
    for img in imgs:
        if hflip_flag:
            img = img[:, ::-1, :]
        if vflip_flag:
            img = img[::-1, :, :]
        if rot_k > 0:
            img = np.rot90(img, k=rot_k, axes=(0, 1))
        result.append(np.ascontiguousarray(img))

    return result


def cutblur(
    hr: np.ndarray,
    lr: np.ndarray,
    scale: int,
    prob: float = 0.3,
    alpha: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CutBlur augmentation (ICCV 2020, Yoo et al.).

    Replaces a random rectangular region of the HR patch with its bicubic
    downsampled-then-upsampled version — i.e. makes a local region 'blurry'.
    The LR patch is unchanged.

    This prevents the model from ignoring difficult low-frequency regions and
    forces attention to all spatial locations (including easy flat areas).

    The probability and region size are controlled by `alpha` (Beta distribution
    parameter). Smaller alpha → larger, more uniform cuts. alpha=0.7 follows
    the original paper's recommendation.

    Args:
        hr:    HR patch  (H, W, 3) float32
        lr:    LR patch  (H//scale, W//scale, 3) float32
        scale: SR scale factor
        prob:  Probability of applying CutBlur (0.3 = 30%)
        alpha: Beta distribution parameter for cut size

    Returns:
        (hr_out, lr_out) — lr is always unchanged; hr has cut region blurred.
    """
    if random.random() >= prob:
        return hr, lr

    H, W = hr.shape[:2]

    # Sample cut size from Beta distribution (same as original paper)
    cut_ratio = float(np.random.beta(alpha, alpha))
    cut_h = max(scale, int(H * cut_ratio))
    cut_w = max(scale, int(W * cut_ratio))

    # Align cut boundaries to scale (so LR pixel grid aligns exactly)
    cut_h = (cut_h // scale) * scale
    cut_w = (cut_w // scale) * scale
    cut_h = min(cut_h, H)
    cut_w = min(cut_w, W)

    # Random top-left corner (aligned to scale grid)
    cy = random.randint(0, (H - cut_h) // scale) * scale
    cx = random.randint(0, (W - cut_w) // scale) * scale

    # Create blurred version: HR region → downsample → upsample (bicubic)
    region_hr = hr[cy: cy + cut_h, cx: cx + cut_w]
    region_lr = cv2.resize(region_hr,
                           (cut_w // scale, cut_h // scale),
                           interpolation=cv2.INTER_CUBIC)
    region_blurred = cv2.resize(region_lr, (cut_w, cut_h),
                                interpolation=cv2.INTER_CUBIC)

    hr_out = hr.copy()
    hr_out[cy: cy + cut_h, cx: cx + cut_w] = region_blurred
    return hr_out, lr


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-source file scanner
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
             '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF', '.TIFF'}


def _scan_sources(roots: List[Path]) -> List[Path]:
    """
    Scan one or more root directories recursively for image files.
    Returns sorted list of Paths (stable across runs for reproducibility).
    """
    files = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {root}\n"
                f"Expected layout:\n"
                f"  <root>/HR/DIV2K_train_HR/\n"
                f"  <root>/HR/Flickr2K_HR/\n"
                f"  <root>/HR/LSDIR_HR/\n"
                f"  <root>/LR_bicubic/X4/DIV2K_train_LR_bicubic_X4/\n  …"
            )
        found = sorted(p for p in root.rglob('*')
                       if p.suffix in _IMG_EXTS and p.is_file())
        files.extend(found)
        print(f"    {root}: {len(found):,} images found")
    return sorted(files)  # final global sort for reproducibility


def _pair_by_stem(
    hr_files: List[Path],
    lr_files: List[Path],
) -> Tuple[List[Path], List[Path]]:
    """
    Match HR and LR files by stem, handling common LR naming conventions:
        0001.png   ↔  0001x4.png
        img001.png ↔  img001_LR.png
        0001.png   ↔  0001.png      (identical stem)

    Returns two aligned lists: (hr_paths, lr_paths).
    Raises ValueError if counts don't match after matching.
    """
    # Build a stem→path dict for LR, stripping common suffixes
    def clean_lr_stem(stem: str) -> str:
        for suffix in ('x4', 'x2', 'x3', 'x8', '_LR', '_lr', 'LR',
                       '_bicubic', '_BICUBIC', 'lr'):
            stem = stem.replace(suffix, '')
        return stem.rstrip('_')

    lr_by_stem: dict[str, Path] = {}
    for p in lr_files:
        lr_by_stem[clean_lr_stem(p.stem)] = p
        lr_by_stem[p.stem] = p  # also allow exact match (higher priority)

    paired_hr, paired_lr = [], []
    missing = []
    for hr_p in hr_files:
        # Try exact stem first, then cleaned stem
        lr_p = lr_by_stem.get(hr_p.stem) or lr_by_stem.get(clean_lr_stem(hr_p.stem))
        if lr_p is not None:
            paired_hr.append(hr_p)
            paired_lr.append(lr_p)
        else:
            missing.append(hr_p.name)

    if missing:
        sample = missing[:5]
        raise ValueError(
            f"Could not find LR counterparts for {len(missing)} HR files.\n"
            f"  Examples: {sample}\n"
            f"  Make sure LR filenames match HR stems (with optional 'x4' suffix)."
        )

    return paired_hr, paired_lr


# ─────────────────────────────────────────────────────────────────────────────
#  Main Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class DIV2KDataset(Dataset):
    """
    Multi-source SR dataset supporting DIV2K, Flickr2K, and LSDIR.

    Named 'DIV2KDataset' for backward-compatibility with train.py; internally
    it handles any combination of source directories.

    DATASET STRUCTURE (Stage 1 DF2K+LSDIR):
    ─────────────────────────────────────────
    /data/DF2K_LSDIR/
    ├── HR/
    │   ├── DIV2K_train_HR/           ← subdirectory auto-detected
    │   ├── Flickr2K_HR/
    │   └── LSDIR_HR/
    └── LR_bicubic/X4/
        ├── DIV2K_train_LR_bicubic_X4/
        ├── Flickr2K_LR_bicubic_X4/
        └── LSDIR_LR_bicubic_X4/

    Pass hr_root=/data/DF2K_LSDIR/HR and lr_root=/data/DF2K_LSDIR/LR_bicubic/X4
    and the loader will recursively find all images across all subdirectories.

    AUGMENTATION PIPELINE (training mode):
    ────────────────────────────────────────
    1. Paired random crop to (hr_patch_size × hr_patch_size)
    2. RGB channel shuffle (50%)        ← prevents color memorisation
    3. Horizontal flip (50%)            ┐
    4. Vertical flip (50%)              ├─ D4 Dihedral Group
    5. Random 90° rotation (25% each)  ┘
    6. CutBlur (30%)                    ← difficulty randomisation

    PROGRESSIVE PATCH CURRICULUM:
    Set current_iter dynamically via set_iter(step) during training to
    automatically increase patch_size at the configured milestones.

    Args:
        hr_root:              Path to HR directory (single or parent of subdirs).
        lr_root:              Path to LR directory (pre-downsampled). Pass None
                              to use on-the-fly bicubic downsampling.
        patch_size:           Initial LR patch size (default 64).
        scale:                SR scale factor (default 4).
        train:                If True, apply augmentation and random crop.
        use_real_degradation: If True, generate LR on-the-fly via Real-ESRGAN.
        real_deg_weight:      Fraction of samples to use real degradation [0,1].
        use_hflip:            Enable horizontal flip augmentation.
        use_vflip:            Enable vertical flip augmentation.
        use_rotation:         Enable 90° rotation augmentation.
        use_channel_shuffle:  Enable RGB channel permutation (recommended: True).
        use_cutblur:          Enable CutBlur augmentation.
        cutblur_prob:         Probability of applying CutBlur (default 0.3).
        patch_curriculum:     List of (iter_threshold, patch_size) milestones.
                              e.g. [(0,64),(150000,96),(300000,128)].
                              Call set_iter(step) each epoch to activate.
    """

    def __init__(
        self,
        hr_root: str,
        lr_root: Optional[str] = None,
        patch_size: int = 64,
        scale: int = 4,
        train: bool = True,
        use_real_degradation: bool = False,
        real_deg_weight: float = 0.0,
        # Augmentation flags
        use_hflip: bool = True,
        use_vflip: bool = True,
        use_rotation: bool = True,
        use_channel_shuffle: bool = True,
        use_cutblur: bool = True,
        cutblur_prob: float = 0.3,
        # Patch curriculum
        patch_curriculum: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.scale              = scale
        self.train              = train
        self.use_real           = use_real_degradation
        self.real_weight        = real_deg_weight
        self.use_hflip          = use_hflip
        self.use_vflip          = use_vflip
        self.use_rotation       = use_rotation
        self.use_channel_shuffle = use_channel_shuffle
        self.use_cutblur        = use_cutblur
        self.cutblur_prob       = cutblur_prob

        # Patch curriculum — sorted ascending by iter threshold
        self._patch_curriculum  = sorted(patch_curriculum or [], key=lambda x: x[0])
        self._base_patch_size   = patch_size
        self._current_iter      = 0
        self.patch_size         = patch_size  # active patch size (updated by set_iter)

        # ── Scan image files (multi-source, recursive) ─────────────────────
        print(f"\nScanning HR source: {hr_root}")
        self.hr_files = _scan_sources([Path(hr_root)])

        if lr_root is not None:
            print(f"Scanning LR source: {lr_root}")
            self.lr_files = _scan_sources([Path(lr_root)])
            print("Pairing HR ↔ LR by filename stem …")
            self.hr_files, self.lr_files = _pair_by_stem(self.hr_files, self.lr_files)
        else:
            self.lr_files = None  # on-the-fly bicubic

        print(f"Dataset ready: {len(self.hr_files):,} image pairs "
              f"({'multi-source' if lr_root else 'on-the-fly LR'})\n")

        # ── Optional real-degradation pipeline ──────────────────────────────
        self.real_deg = None
        if use_real_degradation:
            RealESRGANDegradation = None
            # Try relative import first (normal package usage)
            try:
                from .degradations import RealESRGANDegradation
            except ImportError:
                pass
            # Fallback: absolute import (standalone scripts / pytest from repo root)
            if RealESRGANDegradation is None:
                try:
                    from hydra_sr.data.degradations import RealESRGANDegradation
                except ImportError:
                    pass
            if RealESRGANDegradation is not None:
                self.real_deg = RealESRGANDegradation(scale=scale)
            else:
                print(
                    "WARNING: Could not import RealESRGANDegradation "
                    "(tried .degradations and hydra_sr.data.degradations). "
                    "Real degradation is disabled for this dataset instance."
                )

    # ── Public API ──────────────────────────────────────────────────────────

    def set_iter(self, step: int) -> None:
        """
        Update active patch size from the progressive curriculum.

        Call this at the start of each training step (or epoch):
            dataset.set_iter(global_step)

        The patch_size is set to the largest milestone threshold ≤ step.
        If no milestone is reached yet, falls back to _base_patch_size.

        Example curriculum  [(0,64), (150000,96), (300000,128)]:
            step=0       → patch=64
            step=149999  → patch=64
            step=150000  → patch=96
            step=300000  → patch=128
        """
        self._current_iter = step
        new_size = self._base_patch_size
        for threshold, size in self._patch_curriculum:
            if step >= threshold:
                new_size = size
        if new_size != self.patch_size:
            print(f"[Dataset] patch_size: {self.patch_size} → {new_size} "
                  f"(iter={step:,})")
        self.patch_size = new_size

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'lr':      (3, lr_h, lr_w) float32 tensor in [0,1]
                'hr':      (3, hr_h, hr_w) float32 tensor in [0,1]
                'deg_vec': (4,) float32 — normalised degradation vector
                'hr_path': str — path to original HR image
        """
        hr = read_img(self.hr_files[idx])

        # ── Degradation ──────────────────────────────────────────────────
        use_real_this = (
            self.real_deg is not None
            and random.random() < self.real_weight
        )

        if use_real_this:
            # Real-ESRGAN two-order degradation (Stage 2+)
            lr, deg_vec = self.real_deg(hr)
        elif self.lr_files is not None:
            # Pre-computed bicubic LR (Stage 1 standard path)
            lr = read_img(self.lr_files[idx])
            # Degradation vector — normalised to [0,1] matching RealESRGANDegradation:
            #   sigma_blur  / 5.0   → 0.0  (no blur)
            #   sigma_noise / 50.0  → 0.0  (no noise)
            #   q_jpeg      / 100.0 → 0.95 (95% quality ≈ lossless)
            #   s_ds                → 1.0
            # See: hydra_sr/data/degradations.py  RealESRGANDegradation.__call__
            deg_vec = np.array([0.0, 0.0, 0.95, 1.0], dtype=np.float32)
        else:
            # On-the-fly bicubic (Stage 1 when no LR dir is provided)
            lr = bicubic_downsample(hr, self.scale)
            deg_vec = np.array([0.0, 0.0, 0.95, 1.0], dtype=np.float32)

        # ── Training augmentations ───────────────────────────────────────
        if self.train:
            # 1. Paired crop (active patch size from curriculum)
            hr_patch_size = self.patch_size * self.scale
            hr, lr = paired_random_crop(hr, lr, hr_patch_size, self.scale)

            # 2–5. D4 geometry + RGB channel shuffle
            # Explicit unpack prevents silent bugs if augment() changes its return length.
            aug_results = augment(
                [hr, lr],
                hflip=self.use_hflip,
                vflip=self.use_vflip,
                rotation=self.use_rotation,
                channel_shuffle=self.use_channel_shuffle,
            )
            hr, lr = aug_results[0], aug_results[1]

            # 6. CutBlur — teaches difficulty-awareness in SR
            if self.use_cutblur:
                hr, lr = cutblur(hr, lr, scale=self.scale,
                                 prob=self.cutblur_prob)

        return {
            'lr':      to_tensor(lr),
            'hr':      to_tensor(hr),
            'deg_vec': torch.from_numpy(deg_vec),
            'hr_path': str(self.hr_files[idx]),
        }
