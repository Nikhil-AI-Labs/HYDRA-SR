"""
Dynamic loss re-weighting for Stage 3 perceptual training.

The challenge in Stage 3: LPIPS and TSD-distill can plateau at different rates.
When LPIPS stagnates → push TSD distillation harder.
When wavelet edges blur → push DTWSR distillation harder.
When TSD overshoots → cap and rebalance toward L1.

This class tracks training metrics and auto-adjusts loss weights,
preventing both:
  1. Mode collapse (blurry outputs from over-weighting L1)
  2. Hallucination (from over-weighting TSD distillation)

Reference:
  Inspired by adaptive loss scheduling in EDSR and dynamic loss weighting
  in diffusion SR literature (e.g., ResShift, SeeSR).
"""

import torch
from collections import deque


class DynamicWeighter:
    """
    Adaptive loss weight manager for Stage 3 perceptual training.

    Tracks validation LPIPS and adjusts:
      λ_tsd: weight for TSD distillation loss
      λ_l1:  weight for L1/Charbonnier baseline loss

    The wavelet distillation weight (λ_dtw) is held at 0.4 (stable).

    Args:
        patience:  How many validation steps to consider for plateau detection.
        delta_thr: LPIPS improvement below this → plateau declared.
    """

    def __init__(self, patience: int = 10000, delta_thr: float = 0.001):
        self.patience   = patience
        self.delta_thr  = delta_thr
        self.history    = deque(maxlen=patience * 2)   # ring buffer

        # Initial weights (from implementation plan Stage 3 config)
        self.lam_tsd = 0.6
        self.lam_l1  = 0.1
        self.lam_dtw = 0.4     # fixed
        self.lam_lpips = 1.0   # fixed

        # Bounds
        self._tsd_max = 2.0
        self._tsd_min = 0.1
        self._l1_min  = 0.05
        self._l1_max  = 0.5

        self.step_count = 0

    def update(self, lpips_val: float) -> tuple[float, float, float, float]:
        """
        Update history and recompute weights.

        Args:
            lpips_val: Current validation LPIPS (lower = better).

        Returns:
            (lam_l1, lam_lpips, lam_tsd, lam_dtw) — current weights.
        """
        self.history.append(lpips_val)
        self.step_count += 1

        if len(self.history) >= self.patience:
            recent = list(self.history)[-self.patience:]
            improvement = max(recent) - min(recent)

            if improvement < self.delta_thr:
                # Plateau detected → boost distillation
                self.lam_tsd = min(self.lam_tsd * 1.1, self._tsd_max)
                self.lam_l1  = max(self.lam_l1 * 0.95, self._l1_min)
            else:
                # Improvement happening → slight decay of distillation weight
                # to prevent hallucination
                self.lam_tsd = max(self.lam_tsd * 0.99, self._tsd_min)
                self.lam_l1  = min(self.lam_l1 * 1.005, self._l1_max)

        return self.lam_l1, self.lam_lpips, self.lam_tsd, self.lam_dtw

    def state_dict(self) -> dict:
        return {
            'lam_tsd': self.lam_tsd,
            'lam_l1': self.lam_l1,
            'lam_dtw': self.lam_dtw,
            'lam_lpips': self.lam_lpips,
            'step_count': self.step_count,
        }

    def load_state_dict(self, state: dict):
        self.lam_tsd   = state['lam_tsd']
        self.lam_l1    = state['lam_l1']
        self.lam_dtw   = state.get('lam_dtw', 0.4)
        self.lam_lpips = state.get('lam_lpips', 1.0)
        self.step_count = state.get('step_count', 0)
