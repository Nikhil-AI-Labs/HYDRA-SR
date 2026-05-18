"""
Nested Hilbert-S (NHS) scan index builder.

Combines two complementary strategies from two CVPR 2025 papers:
  - MaIR (XLearning-SCU/2025-CVPR-MaIR): Nested S-shaped scanning for
    spatial continuity preservation at stripe/tile boundaries.
  - FractalMamba++ (2025): Hilbert curves for locality preservation within tiles.

The combined NHS strategy:
  1. Divide the H×W feature into non-overlapping 'tile × tile' tiles.
  2. Within each tile: traverse pixels in Hilbert-curve order
     (preserves locality — nearby pixels in 2D stay close in 1D).
  3. Between tiles: traverse in S-shape (zig-zag row order),
     alternating left→right and right→left rows of tiles
     (preserves continuity at tile boundaries — avoids long jumps).

This is the HYDRA-SR "Nested Hilbert-S" (NHS) scan, a novel combination
not present in any single published paper.

Engineering notes:
  - Returns a precomputed int64 gather index. Cache as a register_buffer.
  - Handles non-power-of-2 H, W by padding to multiples of `tile`.
  - Tiles are padded with reflect mode; padded columns/rows are cropped after scan.
"""

import torch
from .hilbert import hilbert_indices


def nested_s_hilbert_indices(
    H: int,
    W: int,
    tile: int = 16,
) -> tuple[torch.Tensor, int, int]:
    """
    Build the full NHS scan gather index for an H×W feature map.

    Args:
        H:    Feature map height.
        W:    Feature map width.
        tile: Tile size for intra-tile Hilbert scan. Must be a power of 2.
              Typically 16 for pixel stream, 8 for wavelet stream
              (which is already 4× smaller spatially).

    Returns:
        full_idx: int64 tensor of length H_pad × W_pad.
                  ``x_hilbert = x_raster[:, :, full_idx]``
        H_pad:    Padded height (≥ H, multiple of tile).
        W_pad:    Padded width  (≥ W, multiple of tile).

    Shape contract:
        Input  feature: (B, D, H,    W   )
        After pad+flat: (B, D, H_pad*W_pad)
        After gather:   (B, D, H_pad*W_pad)  — in NHS order
        After scatter:  (B, D, H_pad*W_pad)  — back to raster
        Crop:           (B, D, H,    W   )
    """
    assert (tile & (tile - 1)) == 0, f"tile must be power of 2, got {tile}"

    # Pad to multiples of tile
    H_pad = ((H + tile - 1) // tile) * tile
    W_pad = ((W + tile - 1) // tile) * tile
    nH = H_pad // tile  # number of tile rows
    nW = W_pad // tile  # number of tile columns

    # Pre-compute intra-tile Hilbert order for a single tile × tile block
    # intra[k] = raster_position (within tile) of the k-th Hilbert position
    intra = hilbert_indices(tile, tile)  # length: tile*tile

    full_idx = torch.empty(H_pad * W_pad, dtype=torch.long)
    out_pos = 0

    for ty in range(nH):
        # S-shape (zig-zag): even rows go left→right, odd rows go right→left
        x_range = range(nW) if ty % 2 == 0 else range(nW - 1, -1, -1)
        for tx in x_range:
            # Top-left absolute coordinate of this tile
            base_y = ty * tile
            base_x = tx * tile
            # Walk through the intra-tile Hilbert order
            for k in intra:
                k_item = k.item()
                # Local (row, col) within tile
                local_y = k_item // tile
                local_x = k_item % tile
                # Absolute position in the padded H_pad × W_pad raster
                abs_pos = (base_y + local_y) * W_pad + (base_x + local_x)
                full_idx[out_pos] = abs_pos
                out_pos += 1

    assert out_pos == H_pad * W_pad, (
        f"Index mismatch: wrote {out_pos} positions, expected {H_pad * W_pad}"
    )
    return full_idx, H_pad, W_pad
