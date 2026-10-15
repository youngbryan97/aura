"""Locate supplied visual material through scale search and independent patch fits.

A manual illustration, toolbar icon or exported asset can supply reference
pixels. Correspondence supplies a current appearance pose, without deciding
its task role, continuing a temporal identity or identifying a controlled body.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage, signal

from core.perception.affine_motion_from_named_points import AffineMotion, affine_motion
from core.perception.points_before_the_picture_is_reduced import _colours, _refine
from core.perception.points_that_keep_their_motion import _grey, _new_points
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


@dataclass(frozen=True)
class ReferenceAppearance:
    """A correspondence in original picture pixels, with both byte identities."""

    reference_sha256: str
    picture_sha256: str
    linear: tuple[tuple[float, float], tuple[float, float]]
    offset: tuple[float, float]
    box: tuple[float, float, float, float]
    source_points: tuple[tuple[float, float], ...]
    observed_points: tuple[tuple[float, float], ...]
    error: float

    def project(self, point: tuple[float, float]) -> tuple[float, float]:
        """Project a reference coordinate; this does not certify unseen material."""
        return tuple(np.asarray(self.linear) @ np.asarray(point) + self.offset)


def _digest(pixels: np.ndarray) -> str:
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def _overlap(a: tuple, b: tuple) -> float:
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    common = max(0., right-x)*max(0., bottom-y)
    return common/max(1., a[2]*a[3]+b[2]*b[3]-common)


def _proposals(scene: np.ndarray, reference: np.ndarray, scales: tuple) -> list[tuple]:
    """Masked squared error proposes places; it cannot qualify correspondence."""
    height, width = scene.shape[:2]
    source = Image.fromarray(reference)
    proposed, sizes = [], set()
    for scale in scales:
        size = tuple(round(v*scale) for v in source.size)
        if size in sizes or min(size) < 12 or size[0] > width or size[1] > height:
            continue
        sizes.add(size)
        template = np.asarray(source.resize(size, Image.Resampling.BOX))
        mask = ndimage.binary_erosion(template[..., 3] >= 250).astype(np.float32)
        if mask.sum() < 36:
            continue
        rgb = template[..., :3].astype(np.float32)/255
        supported = rgb[mask.astype(bool)]
        if float(np.std(supported, axis=0).max()) < .03:
            continue
        cost = sum(signal.fftconvolve(scene[..., c]**2, mask[::-1, ::-1], mode='valid')
                   - 2*signal.fftconvolve(scene[..., c], (rgb[..., c]*mask)[::-1, ::-1], mode='valid')
                   + np.sum(rgb[..., c]**2*mask) for c in range(3))/(3*mask.sum())
        for _ in range(3):
            row, col = np.unravel_index(cost.argmin(), cost.shape)
            value = float(cost[row, col])
            if not math.isfinite(value):
                break
            proposed.append((value, int(col), int(row), template))
            # Keep separate copies eligible. One best peak cannot prove that
            # a reference has only one visible instance.
            dx, dy = max(4, size[0]//2), max(4, size[1]//2)
            cost[max(0, row-dy):row+dy+1, max(0, col-dx):col+dx+1] = math.inf
    return sorted(proposed, key=lambda candidate: candidate[0])


def _supported_fit(scene: np.ndarray, template: np.ndarray, x: int, y: int) -> tuple | None:
    """Patch journeys in both directions precede a redundant geometric fit."""
    mask = ndimage.binary_erosion(template[..., 3] >= 250, iterations=4)
    points = [point for point in _new_points(_grey(template[..., :3]), [], 128)
              if mask[round(point[1]), round(point[0])]]
    if len(points) < 6:
        return None
    starts = np.asarray(points)
    before, after = _colours(template), _colours(scene)
    ends, error, _angles, valid = _refine(before, after, starts, starts+(x, y))
    back, reverse_error, _angles, reverse_valid = _refine(after, before, ends, starts)
    valid &= reverse_valid & (error <= .055) & (reverse_error <= .055) & (np.linalg.norm(back-starts, axis=1) <= 1.)
    if int(valid.sum()) < max(6, math.ceil(.5*len(starts))):
        return None
    starts, ends = starts[valid], ends[valid]
    fitted = rigid_motion(starts, ends) or affine_motion(starts, ends)
    if fitted is None or len(fitted.inliers) < 6:
        return None
    indices = list(fitted.inliers)
    return fitted, starts[indices], ends[indices]


def reference_appearances(picture: np.ndarray, reference: np.ndarray, *, scales: tuple[float, ...],
                          edge: int = 480) -> tuple[ReferenceAppearance, ...]:
    """Return bounded appearance poses; competing instances remain separate.

    ``picture`` is RGB, ``reference`` is RGBA. Supplied scales map reference
    pixels to original picture pixels. Search is bounded to 32 scales and
    24 distinct proposals. Neither a returned pose nor an empty result names
    the player, a goal, a collision boundary or a successful action.
    """
    if (not isinstance(picture, np.ndarray) or not isinstance(reference, np.ndarray)
            or picture.dtype != np.uint8 or reference.dtype != np.uint8
            or picture.ndim != 3 or reference.ndim != 3 or picture.shape[2] != 3 or reference.shape[2] != 4
            or min(picture.shape[:2]) < 16 or max(picture.shape[:2]) > 4096 or picture.size > 24_000_000
            or min(reference.shape[:2]) < 16 or max(reference.shape[:2]) > 512
            or type(edge) is not int or not 64 <= edge <= 512
            or not isinstance(scales, tuple) or not 1 <= len(scales) <= 32
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                   or not .05 <= v <= 8 for v in scales)):
        return ()
    ratio = min(1., edge/max(picture.shape[:2]))
    size = (round(picture.shape[1]*ratio), round(picture.shape[0]*ratio))
    view_scale = np.array((size[0]/picture.shape[1], size[1]/picture.shape[0]))
    pixels = np.asarray(Image.fromarray(picture).resize(size, Image.Resampling.BOX))
    scene = pixels.astype(np.float32)/255
    proposals = _proposals(scene, reference, tuple(v*ratio for v in scales))
    tried, found = [], []
    reference_hash, picture_hash = _digest(reference), _digest(picture)
    for _cost, x, y, template in proposals:
        height, width = template.shape[:2]
        box = (x, y, width, height)
        if any(_overlap(box, previous) > .6 for previous in tried):
            continue
        tried.append(box)
        fit = _supported_fit(pixels, template, x, y)
        if fit is not None:
            fitted, starts, ends = fit
            sx, sy = width/reference.shape[1], height/reference.shape[0]
            if isinstance(fitted, AffineMotion):
                fitted_linear = np.asarray(fitted.linear)
            else:
                c, s = math.cos(fitted.angle), math.sin(fitted.angle)
                fitted_linear = np.array(((c, -s), (s, c)))
            linear = (fitted_linear @ np.diag((sx, sy)))/view_scale[:, None]
            offset = np.asarray(fitted.offset)/view_scale
            corners = np.array(((0., 0.), (reference.shape[1], 0.),
                                (0., reference.shape[0]), (reference.shape[1], reference.shape[0]))) @ linear.T+offset
            low, high = corners.min(axis=0), corners.max(axis=0)
            found.append(ReferenceAppearance(reference_hash, picture_hash,
                tuple(tuple(float(v) for v in row) for row in linear), tuple(float(v) for v in offset),
                (*map(float, low), *map(float, high-low)),
                tuple(tuple(float(v) for v in point/(sx, sy)) for point in starts),
                tuple(tuple(float(v) for v in point/view_scale) for point in ends), fitted.error/float(view_scale.min())))
        if len(tried) >= 24 or len(found) >= 8:
            break
    return tuple(found)


@invariant('perception.reference_appearance_is_not_temporal_identity', scope='perception',
           owner='core/perception/a_visual_reference_in_the_picture.py', observational=False)
def _reference_appearance_invariant() -> tuple:
    blank = np.zeros((32, 32, 4), np.uint8)
    blank[..., 3] = 255
    assert reference_appearances(blank[..., :3], blank, scales=(1.,)) == ()
    assert reference_appearances(blank[..., :3], blank, scales=(math.nan,)) == ()
    reference = np.random.default_rng(47).integers(0, 255, (48, 48, 4), dtype=np.uint8)
    reference[..., 3] = 255
    picture = np.zeros((96, 144, 3), np.uint8)
    picture[20:68, 30:78] = reference[..., :3]
    seen = reference_appearances(picture, reference, scales=(1.,))
    assert len(seen) == 1 and math.dist(seen[0].project((24., 24.)), (54., 44.)) < 1.
    assert 'owner' not in ReferenceAppearance.__dataclass_fields__
    assert 'controlled' not in ReferenceAppearance.__dataclass_fields__
    return ()
