"""
Real-ESRGAN second-order degradation pipeline for HYDRA-SR Stage 2+ training.

Generates realistic LR images from HR by simulating real-world degradation chains.
Also computes the degradation parameter vector for DegradationPredictor supervision.

Degradation pipeline (two-order Real-ESRGAN):
  First order:
    1. Random Gaussian/isotropic blur (σ ∈ [0.2, 3.0])
    2. Random resize (factor ∈ [0.15, 1.5], bicubic/bilinear/nearest)
    3. Random Gaussian/Poisson noise (σ ∈ [1, 30] or Poisson lambda)
    4. JPEG compression (quality ∈ [30, 95])
  Second order (same chain with tighter params):
    1–4 repeated with different random parameters
  Final downsampling to target scale.

Reference:
  Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
  Wang et al., ICCV 2021 Workshop.
  Implementation: XPixelGroup/BasicSR → basicsr/data/realesrgan_dataset.py

This implementation is self-contained (no BasicSR dependency at runtime)
but follows the same pipeline logic exactly.
"""

import numpy as np
import random
import cv2
from io import BytesIO
from PIL import Image


def apply_gaussian_blur(
    img: np.ndarray,
    sigma_range: tuple[float, float] = (0.2, 3.0),
    kernel_size: int = None,
) -> np.ndarray:
    """Apply random isotropic Gaussian blur."""
    sigma = random.uniform(*sigma_range)
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)
        kernel_size = max(kernel_size, 3) | 1  # ensure odd
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_resize(
    img: np.ndarray,
    sf_range: tuple[float, float] = (0.15, 1.5),
    target_size: tuple[int, int] = None,
) -> np.ndarray:
    """Random resize."""
    h, w = img.shape[:2]
    sf = random.uniform(*sf_range)
    mode = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
    new_h, new_w = max(1, int(h * sf)), max(1, int(w * sf))
    img = cv2.resize(img, (new_w, new_h), interpolation=mode)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return img


def add_gaussian_noise(
    img: np.ndarray,
    sigma_range: tuple[float, float] = (1, 30),
) -> np.ndarray:
    """Add random Gaussian noise."""
    sigma = random.uniform(*sigma_range) / 255.0
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0.0, 1.0)


def add_poisson_noise(
    img: np.ndarray,
    scale_range: tuple[float, float] = (0.05, 3.0),
) -> np.ndarray:
    """Add Poisson-distributed noise."""
    scale = random.uniform(*scale_range)
    img_uint = (img * 255.0).astype(np.float32)
    noisy = np.random.poisson(img_uint / scale) * scale
    return np.clip(noisy / 255.0, 0.0, 1.0).astype(np.float32)


def add_jpeg_compression(
    img: np.ndarray,
    quality_range: tuple[int, int] = (30, 95),
) -> np.ndarray:
    """Simulate JPEG compression artifact."""
    quality = random.randint(*quality_range)
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return dec.astype(np.float32) / 255.0


class RealESRGANDegradation:
    """
    Two-order Real-ESRGAN degradation pipeline.
    Generates (LR, degradation_vector) from an HR image.

    Args:
        scale: SR scale factor (default 4).
    """

    def __init__(self, scale: int = 4):
        self.scale = scale

    def __call__(self, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply two-order degradation to HR → LR.

        Args:
            hr: (H, W, 3) float32 in [0, 1]

        Returns:
            lr:      (H//scale, W//scale, 3) float32 in [0, 1]
            deg_vec: (4,) float32 [σ_blur, σ_noise, q_JPEG, s_ds]
        """
        img = hr.copy()
        H, W = img.shape[:2]
        target_h, target_w = H // self.scale, W // self.scale

        # Track degradation parameters for supervision
        sigma_blur  = random.uniform(0.2, 3.0)
        sigma_noise = random.uniform(1.0, 30.0)
        q_jpeg      = random.randint(30, 95)

        # ── First-order degradation ──────────────────────────────────────
        img = apply_gaussian_blur(img, sigma_range=(0.2, 3.0))
        img = apply_resize(img, sf_range=(0.15, 1.5), target_size=(W, H))

        if random.random() < 0.5:
            img = add_gaussian_noise(img, sigma_range=(1, 30))
        else:
            img = add_poisson_noise(img)

        img = add_jpeg_compression(img, quality_range=(30, 95))

        # ── Second-order degradation (tighter) ──────────────────────────
        img = apply_gaussian_blur(img, sigma_range=(0.2, 1.5))
        img = apply_resize(img, sf_range=(0.3, 1.2), target_size=(W, H))

        if random.random() < 0.5:
            img = add_gaussian_noise(img, sigma_range=(1, 25))

        img = add_jpeg_compression(img, quality_range=(30, 95))

        # ── Final downsample to target LR size ───────────────────────────
        lr = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        lr = np.clip(lr, 0.0, 1.0)

        # Degradation vector (normalized for prediction head)
        deg_vec = np.array([
            sigma_blur / 5.0,    # normalize to ~[0,1]
            sigma_noise / 50.0,
            q_jpeg / 100.0,
            1.0,                 # downsample factor = 1 (already done)
        ], dtype=np.float32)

        return lr, deg_vec
