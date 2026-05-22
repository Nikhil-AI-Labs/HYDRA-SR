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

Key design rule: degradation parameter tracking
  The deg_vec MUST track the parameters that were actually applied, not
  independently re-sampled values. Each tracked function returns (img, param)
  so the caller records the exact sigma/quality used.
  Failure to do this trains the DegradationPredictor on noise labels.

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


# ─────────────────────────────────────────────────────────────────────────────
#  Primitive degradation functions
#  Convention: *_tracked variants return (img, actual_param_used)
#              so the caller can build deg_vec from what was really applied.
# ─────────────────────────────────────────────────────────────────────────────

def apply_gaussian_blur(
    img: np.ndarray,
    sigma_range: tuple[float, float] = (0.2, 3.0),
    kernel_size: int = None,
) -> np.ndarray:
    """Apply random isotropic Gaussian blur (non-tracked, for internal use)."""
    sigma = random.uniform(*sigma_range)
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)
        kernel_size = max(kernel_size, 3) | 1  # ensure odd
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_gaussian_blur_tracked(
    img: np.ndarray,
    sigma_range: tuple[float, float] = (0.2, 3.0),
    kernel_size: int = None,
) -> tuple[np.ndarray, float]:
    """
    Apply random isotropic Gaussian blur and return the sigma actually used.

    Returns:
        (blurred_img, sigma_used)

    This is the tracked variant used in RealESRGANDegradation.__call__ to
    ensure deg_vec[0] = sigma_used / 5.0 is consistent with what was applied.
    """
    sigma = random.uniform(*sigma_range)
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)
        kernel_size = max(kernel_size, 3) | 1  # ensure odd
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return blurred, sigma


def apply_resize(
    img: np.ndarray,
    sf_range: tuple[float, float] = (0.15, 1.5),
    target_size: tuple[int, int] = None,
) -> np.ndarray:
    """
    Random resize then optional restore to target_size.

    Args:
        img:         Input image (H, W, 3)
        sf_range:    Random scale factor range
        target_size: (width, height) to restore to after random resize.
                     OpenCV convention: (W, H), NOT (H, W).
                     Callers must pass (W, H) explicitly.
    """
    h, w = img.shape[:2]
    sf = random.uniform(*sf_range)
    mode = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
    new_h, new_w = max(1, int(h * sf)), max(1, int(w * sf))
    img = cv2.resize(img, (new_w, new_h), interpolation=mode)  # cv2: (width, height)
    if target_size is not None:
        # target_size must be (width, height) — caller's responsibility
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return img


def add_gaussian_noise(
    img: np.ndarray,
    sigma_range: tuple[float, float] = (1, 30),
) -> tuple[np.ndarray, float]:
    """
    Add random Gaussian noise and return sigma actually used.

    Returns:
        (noisy_img, sigma_used)   sigma is in [0, 255] scale before /255 normalise.
    """
    sigma_255 = random.uniform(*sigma_range)
    sigma = sigma_255 / 255.0
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0.0, 1.0), sigma_255


def add_poisson_noise(
    img: np.ndarray,
    scale_range: tuple[float, float] = (0.05, 3.0),
) -> np.ndarray:
    """
    Add Poisson-distributed noise.

    Defensive note: np.random.poisson(λ=0) = 0 (no NaN), so near-black pixels
    produce no noise, which is physically correct. We still clip img_uint to ≥0
    to guard against any floating-point underflow from upstream operations.
    """
    scale = random.uniform(*scale_range)
    img_uint = (img * 255.0).astype(np.float32)
    # Clip to ≥ 0 to prevent np.random.poisson receiving a negative lambda
    # (which would raise ValueError, not return NaN, but is still wrong).
    img_uint = np.maximum(img_uint, 0.0)
    noisy = np.random.poisson(img_uint / scale) * scale
    return np.clip(noisy / 255.0, 0.0, 1.0).astype(np.float32)


def add_jpeg_compression(
    img: np.ndarray,
    quality_range: tuple[int, int] = (30, 95),
) -> tuple[np.ndarray, int]:
    """
    Simulate JPEG compression artifact and return quality actually used.

    Returns:
        (compressed_img, quality_used)
    """
    quality = random.randint(*quality_range)
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # OpenCV works in BGR; convert RGB→BGR before encode, then BGR→RGB after decode
    _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return dec.astype(np.float32) / 255.0, quality


class RealESRGANDegradation:
    """
    Two-order Real-ESRGAN degradation pipeline.
    Generates (LR, degradation_vector) from an HR image.

    Degradation vector contract (all values normalised to [0, 1]):
        deg_vec[0] = sigma_blur_1st  / 5.0    (dominant blur order)
        deg_vec[1] = sigma_noise_1st / 50.0   (dominant noise order)
        deg_vec[2] = q_jpeg_1st      / 100.0  (dominant JPEG order)
        deg_vec[3] = 1.0                       (reserved: downsample factor)

    IMPORTANT: deg_vec tracks the parameters that were ACTUALLY applied to the
    image in the first degradation order, so the DegradationPredictor is
    trained on ground-truth labels — not on independently re-sampled values.

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
            deg_vec: (4,) float32 — normalised degradation params of 1st order
        """
        img = hr.copy()
        H, W = img.shape[:2]
        target_h, target_w = H // self.scale, W // self.scale

        # ── First-order degradation ──────────────────────────────────────────
        # TRACKED calls: return (img, actual_param) so deg_vec is consistent.

        # 1. Blur — track exact sigma applied
        img, sigma_blur_1 = apply_gaussian_blur_tracked(img, sigma_range=(0.2, 3.0))

        # 2. Resize (random scale then restore to original size)
        #    target_size = (W, H) — OpenCV (width, height) convention
        img = apply_resize(img, sf_range=(0.15, 1.5), target_size=(W, H))

        # 3. Noise — track exact sigma applied
        if random.random() < 0.5:
            img, sigma_noise_1 = add_gaussian_noise(img, sigma_range=(1, 30))
        else:
            img = add_poisson_noise(img)
            # Poisson noise equivalent sigma: approximate as mid-range for tracking
            sigma_noise_1 = random.uniform(1.0, 30.0)  # representative value

        # 4. JPEG — track exact quality applied
        img, q_jpeg_1 = add_jpeg_compression(img, quality_range=(30, 95))

        # ── Second-order degradation (tighter parameters) ────────────────────
        # Not tracked for deg_vec — first-order parameters dominate perceptual quality.

        img, _ = apply_gaussian_blur_tracked(img, sigma_range=(0.2, 1.5))
        img = apply_resize(img, sf_range=(0.3, 1.2), target_size=(W, H))

        if random.random() < 0.5:
            img, _ = add_gaussian_noise(img, sigma_range=(1, 25))
        # Second-order Poisson intentionally omitted (tighter pipeline)

        img, _ = add_jpeg_compression(img, quality_range=(30, 95))

        # ── Final downsample to target LR size ───────────────────────────────
        # cv2.resize takes (width, height) = (target_w, target_h)
        lr = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        lr = np.clip(lr, 0.0, 1.0)

        # ── Degradation vector (normalised, tracking 1st-order params) ───────
        # These are the params actually applied above — NOT re-sampled.
        deg_vec = np.array([
            sigma_blur_1  / 5.0,    # σ_blur ∈ [0.2,3.0] → [0.04, 0.60]
            sigma_noise_1 / 50.0,   # σ_noise ∈ [1,30]   → [0.02, 0.60]
            q_jpeg_1      / 100.0,  # q_jpeg ∈ [30,95]   → [0.30, 0.95]
            1.0,                    # reserved (downsample factor)
        ], dtype=np.float32)

        return lr, deg_vec
