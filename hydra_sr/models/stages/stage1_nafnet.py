"""
Stage 1 NAFNet — local denoising module wrapper.
(The stage1_p and stage1_w in HYDRA-SR are just nn.Sequential of NAFBlocks;
this file is a thin alias for clarity in the stages/ package.)
"""
from ..modules.nafblock import NAFBlock, LayerNorm2d

__all__ = ["NAFBlock", "LayerNorm2d"]
