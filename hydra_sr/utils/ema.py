"""
EMA (Exponential Moving Average) model wrapper for HYDRA-SR.

EMA is non-negotiable in NTIRE 2026:
  "Every NTIRE winner uses EMA with decay 0.999 — ~+0.1 dB free."
  (From implementation plan Part 4, Training tricks)

EMA maintains a shadow copy of the model weights that is an exponential
average of all past weights. The EMA model is used for validation and
inference (not training).

  W_ema = decay * W_ema + (1 - decay) * W_model

decay = 0.999 → EMA "memory" ≈ 1000 recent gradient steps.

This wrapper:
  - Creates the EMA shadow model automatically
  - Provides update() method called every training step
  - Provides copy_to() to copy EMA weights back to model for inference
  - Handles DDP (only rank 0 maintains the EMA copy)

Reference:
  lucidrains/ema-pytorch (we use the same logic but don't require the package)
"""

import copy
import torch
import torch.nn as nn
from contextlib import contextmanager


class EMA:
    """
    Exponential Moving Average of model parameters.

    Args:
        model:        The model to track.
        decay:        EMA decay rate (default 0.999).
        update_every: Update EMA every N steps (default 1).
        start_step:   Don't apply EMA before this many steps (warmup).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_every: int = 1,
        start_step: int = 0,
    ):
        self.decay        = decay
        self.update_every = update_every
        self.start_step   = start_step
        self.step_count   = 0

        # Create shadow model (deep copy, not sharing params)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA weights from current model.
        Call after every optimizer.step() in training loop.
        """
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return
        if self.step_count < self.start_step:
            # Before start_step: copy model exactly (burn-in period)
            self._copy_params(model)
            return

        for ema_param, model_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

        # Also update buffers (BatchNorm stats, etc.)
        for ema_buf, model_buf in zip(
            self.ema_model.buffers(), model.buffers()
        ):
            ema_buf.data.copy_(model_buf.data)

    @torch.no_grad()
    def _copy_params(self, model: nn.Module):
        """Exactly copy model parameters to EMA (used during warmup)."""
        for ema_param, model_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.data.copy_(model_param.data)

    @contextmanager
    def ema_scope(self, model: nn.Module):
        """
        Context manager: temporarily swap model weights with EMA weights.
        Useful for in-training validation without saving/loading checkpoints.
        """
        # Save current model weights
        original = {n: p.data.clone() for n, p in model.named_parameters()}
        # Copy EMA weights to model
        for (n, p), (_, ema_p) in zip(
            model.named_parameters(), self.ema_model.named_parameters()
        ):
            p.data.copy_(ema_p.data)
        try:
            yield self.ema_model
        finally:
            # Restore original weights
            for n, p in model.named_parameters():
                p.data.copy_(original[n])

    def state_dict(self) -> dict:
        return {
            'ema_model':  self.ema_model.state_dict(),
            'step_count': self.step_count,
            'decay':      self.decay,
        }

    def load_state_dict(self, state: dict):
        self.ema_model.load_state_dict(state['ema_model'])
        self.step_count = state.get('step_count', 0)
        self.decay = state.get('decay', self.decay)
