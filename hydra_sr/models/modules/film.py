"""
FiLM (Feature-wise Linear Modulation) conditioning layer.

Injects a conditioning vector (e.g., degradation prompt p_d) into
spatial feature maps via learned affine transformation:
    x' = γ(p) ⊙ x + β(p)

where γ and β are predicted by a small MLP from the prompt vector p.

Reference:
    "FiLM: Visual Reasoning with a General Conditioning Layer"
    Perez et al., AAAI 2018.

Usage in HYDRA-SR:
    Every AHN-Mamba block and NAFBlock (Stage 1) is conditioned via FiLM
    using the 128-dimensional degradation prompt p_d from DegradationPredictor.
    This is what makes HYDRA-SR a blind SR model — one weight set handles
    bicubic, real, JPEG-corrupted, and unknown degradation types.

Shape contract:
    x:      (B, C, H, W)   — feature map to modulate
    prompt: (B, prompt_dim) — conditioning vector from DegradationPredictor
    output: (B, C, H, W)   — modulated feature map
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    FiLM modulation module.

    Predicts (γ, β) from a prompt vector, then applies:
        x' = (1 + γ) ⊙ x + β

    Note: we use ``1 + γ`` (initialized at γ=0) so that at training start
    the module is an identity — this is critical for training stability.

    Args:
        prompt_dim: Dimensionality of the input conditioning vector.
        feat_dim:   Number of channels in the feature map to be modulated.
    """

    def __init__(self, prompt_dim: int, feat_dim: int):
        super().__init__()
        # Single linear layer: prompt → (2 × feat_dim) for γ and β
        self.proj = nn.Linear(prompt_dim, 2 * feat_dim)

        # Initialize to identity: proj outputs ≈ 0 at start
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, C, H, W) feature map.
            prompt: (B, prompt_dim) conditioning vector.

        Returns:
            x_modulated: (B, C, H, W) — same shape as x.
        """
        # Predict affine parameters
        gamma_beta = self.proj(prompt)                  # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)       # each (B, C)

        # Reshape for broadcast over spatial dims
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta  = beta.view( beta.shape[0],  beta.shape[1], 1, 1)

        # Affine modulation (identity-initialized: γ starts at 0 → 1+γ = 1)
        return (1.0 + gamma) * x + beta
