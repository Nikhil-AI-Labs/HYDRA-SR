"""
Hilbert curve index computation — reference implementation.

Maps a 2D spatial grid (H×W) to a 1D Hilbert curve order.
This is CPU-only and pre-computed once per unique spatial size.
The result is used as a gather index to reorder flattened 2D features
into Hilbert order for Mamba SSM scanning.

Reference:
    Hilbert curve space-filling curve algorithm (Peano / Hilbert, 1891).
    Implementation adapted from classic algorithms literature.
"""

import numpy as np
import torch


def _xy2d(n: int, x: int, y: int) -> int:
    """
    Map a (x, y) coordinate on an n×n grid to its Hilbert index d.

    Args:
        n: Grid size. Must be a power of 2.
        x: Column index [0, n).
        y: Row index    [0, n).

    Returns:
        d: Hilbert curve distance (1D index), in [0, n*n).
    """
    rx, ry, d = 0, 0, 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate / reflect the quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def hilbert_indices(H: int, W: int) -> torch.Tensor:
    """
    Build the gather index that converts raster-order to Hilbert-order.

    For pixel at raster position `i`, `inv[i]` gives the position in the
    Hilbert-ordered sequence where pixel `i` should be placed.

    Args:
        H: Feature map height. Must equal W and be a power of 2.
        W: Feature map width.

    Returns:
        inv: int64 tensor of length H*W.
             ``x_hilbert = x_raster[:, :, inv]``  reorders to Hilbert order.
             ``x_raster  = x_hilbert[:, :, inv_inv]`` can be obtained via argsort.

    Raises:
        AssertionError: If H ≠ W or H is not a power of 2.
    """
    assert H == W and (H & (H - 1)) == 0, (
        f"hilbert_indices requires H == W and H power-of-2, got H={H}, W={W}"
    )
    n = H
    # idx[raster_pos] = hilbert_distance for that (x, y)
    idx = np.zeros(n * n, dtype=np.int64)
    for y in range(n):
        for x in range(n):
            idx[y * n + x] = _xy2d(n, x, y)

    # We want inv[raster_pos] = position-in-hilbert-sequence
    # i.e.  inv = argsort(idx) because idx maps raster → hilbert-distance
    # and argsort gives: for each hilbert position p, which raster pos is there
    # Actually we want: gather index so that out[i] = x[inv[i]] gives hilbert order
    # argsort(idx)[h] = raster_pos that has hilbert_distance h
    # So inv = argsort(idx):  x_hilbert[h] = x_raster[inv[h]]  ✓
    inv = np.argsort(idx)
    return torch.from_numpy(inv.astype(np.int64))
