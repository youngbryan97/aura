"""Check a static exported frame against its independent full raster.

This tests the declared identity map between equal pixel grids. It estimates
no live pose. Transparent RGB is immaterial; alpha and premultiplied colour
must agree throughout, and translations on both axes must be distinguishable.
"""
from __future__ import annotations

import numpy as np

from core.verify import invariant


def same_declared_pixel_frame(first: np.ndarray, second: np.ndarray) -> bool:
    if (not all(isinstance(p, np.ndarray) and p.dtype == np.uint8 and p.ndim == 3 and p.shape[2] == 4
                and 16 <= min(p.shape[:2]) <= max(p.shape[:2]) <= 512 for p in (first, second))
            or first.shape != second.shape):
        return False
    a, b = first.astype(float)/255., second.astype(float)/255.
    a[..., :3] *= a[..., 3:]
    b[..., :3] *= b[..., 3:]
    error = abs(a-b)
    mean = float(error.mean())
    # Two 8-bit quantization steps cover independent premultiplication and
    # unpremultiplication rounding. This is checked at every pixel/channel.
    if error.max() > 2/255.+1e-12 or mean > .25/255.:
        return False
    support = np.argwhere((first[..., 3] >= 250) & (second[..., 3] >= 250))
    if len(support) < 64 or np.linalg.matrix_rank(support-support.mean(axis=0), tol=.05) < 2:
        return False
    for dx, dy in ((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1) if x or y):
        height, width = first.shape[:2]
        y0, y1, x0, x1 = max(0, -dy), min(height, height-dy), max(0, -dx), min(width, width-dx)
        shifted = abs(a[y0:y1, x0:x1]-b[y0+dy:y1+dy, x0+dx:x1+dx])
        if float(shifted.mean()) <= max(.01, 8*mean):
            return False
    return True


@invariant('perception.dense_static_frame_alignment_needs_visible_two_axis_evidence', scope='perception',
           owner='core/perception/two_rasters_in_one_declared_pixel_frame.py', observational=False)
def _dense_static_frame_invariant() -> tuple:
    pixels = np.zeros((32, 48, 4), np.uint8)
    pixels[5:7, 3:40] = (200, 180, 60, 255)
    pixels[6:25, 10:12] = (30, 70, 90, 255)
    pixels[15:17, 10:32] = (210, 40, 80, 255)
    assert same_declared_pixel_frame(pixels, pixels.copy())
    assert not same_declared_pixel_frame(pixels, np.roll(pixels, 1, axis=0))
    assert not same_declared_pixel_frame(np.full_like(pixels, 255), np.full_like(pixels, 255))
    return ()
