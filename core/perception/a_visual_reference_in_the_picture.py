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
from scipy import fft, ndimage

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


def _oriented_template(reference: Image.Image, size: tuple[int, int], angle: float) -> tuple:
    """Return raster samples and their reference-to-template pixel-centre map."""
    sized = reference.resize(size, Image.Resampling.BOX)
    template = sized.rotate(-math.degrees(angle), resample=Image.Resampling.BICUBIC, expand=True) if angle else sized
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s), (s, c)))
    scale = np.array(size)/reference.size
    linear = rotation @ np.diag(scale)
    offset = (np.array(template.size)-1)/2-linear @ ((np.array(reference.size)-1)/2)
    return np.asarray(template), linear, offset


class _SceneCorrelations:
    """Reuse one captured image's spectra for bounded masked colour costs."""

    def __init__(self, scene: np.ndarray, largest: tuple[int, int] | None = None) -> None:
        self.shape = scene.shape[:2]
        self.largest = self.shape if largest is None else largest
        if any(not 1 <= k <= n for k, n in zip(self.largest, self.shape, strict=True)):
            raise ValueError('a colour-cost bank needs bounded kernels inside its scene')
        self.padded = tuple(fft.next_fast_len(n+k-1, real=True) for n, k in zip(self.shape, self.largest, strict=True))
        self.squared = fft.rfftn(np.sum(scene**2, axis=2), self.padded)
        self.colours = fft.rfftn(scene.transpose(2, 0, 1), self.padded, axes=(-2, -1))

    def cost(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        support = float(mask.sum())
        if (rgb.shape != (height, width, 3) or height > self.largest[0] or width > self.largest[1]
                or support <= 0):
            raise ValueError('a colour-cost kernel needs positive support inside its scene')
        mask_spectrum = fft.rfftn(mask[::-1, ::-1], self.padded)
        weighted = (rgb*mask[..., None])[::-1, ::-1].transpose(2, 0, 1)
        colour_spectrum = fft.rfftn(weighted, self.padded, axes=(-2, -1))
        combined = self.squared*mask_spectrum-2*np.sum(self.colours*colour_spectrum, axis=0)
        result = fft.irfftn(combined, self.padded)
        constant = float(np.sum(rgb**2*mask[..., None]))
        return (result[height-1:self.shape[0], width-1:self.shape[1]]+constant)/(3*support)


def _proposals(scene: np.ndarray, reference: np.ndarray, scales: tuple, angles: tuple) -> list[tuple]:
    """Masked squared error proposes places; it cannot qualify correspondence."""
    height, width = scene.shape[:2]
    source = Image.fromarray(reference)
    proposed, sizes, kernels = [], set(), []
    for scale, angle in ((scale, angle) for scale in scales for angle in angles):
        size = tuple(round(v*scale) for v in source.size)
        key = size, round(angle, 12)
        if key in sizes or min(size) < 12 or size[0] > width or size[1] > height:
            continue
        sizes.add(key)
        template, linear, offset = _oriented_template(source, size, angle)
        if template.shape[1] > width or template.shape[0] > height:
            continue
        mask = ndimage.binary_erosion(template[..., 3] >= 250)
        if mask.sum() < 36:
            continue
        supported = template[..., :3][mask].astype(np.float32)/255
        if float(np.std(supported, axis=0).max()) < .03:
            continue
        kernels.append((template, linear, offset, angle, mask))
    if not kernels:
        return []
    largest = tuple(max(item[0].shape[axis] for item in kernels) for axis in (0, 1))
    correlations = _SceneCorrelations(scene, largest)
    for template, linear, offset, angle, mask in kernels:
        # Keep one floating-point colour kernel at a time beside the bank's
        # byte rasters and boolean masks.
        rgb = template[..., :3].astype(np.float32)/255
        cost = correlations.cost(rgb, mask.astype(np.float32))
        for _ in range(3):
            row, col = np.unravel_index(cost.argmin(), cost.shape)
            value = float(cost[row, col])
            if not math.isfinite(value):
                break
            proposed.append((value, int(col), int(row), template, linear, offset, angle))
            # Keep separate copies eligible. One best peak cannot prove that
            # a reference has only one visible instance.
            dx, dy = max(4, template.shape[1]//2), max(4, template.shape[0]//2)
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
                          angles: tuple[float, ...] = (0.,),
                          edge: int = 480) -> tuple[ReferenceAppearance, ...]:
    """Return bounded appearance poses; competing instances remain separate.

    ``picture`` is RGB, ``reference`` is RGBA. Supplied scales map reference
    pixels to original picture pixels; angles are clockwise radians. Search
    is bounded to 32 scale/angle pairs and 24 distinct proposals. Neither a returned pose nor an empty result names
    the player, a goal, a collision boundary or a successful action.
    """
    if (not isinstance(picture, np.ndarray) or not isinstance(reference, np.ndarray)
            or picture.dtype != np.uint8 or reference.dtype != np.uint8
            or picture.ndim != 3 or reference.ndim != 3 or picture.shape[2] != 3 or reference.shape[2] != 4
            or min(picture.shape[:2]) < 16 or max(picture.shape[:2]) > 4096 or picture.size > 24_000_000
            or min(reference.shape[:2]) < 16 or max(reference.shape[:2]) > 512
            or type(edge) is not int or not 64 <= edge <= 512
            or not isinstance(scales, tuple) or not 1 <= len(scales) <= 32
            or not isinstance(angles, tuple) or not 1 <= len(angles) <= 32 or len(angles)*len(scales) > 32
            or any(type(v) not in (int, float) or not math.isfinite(v) or not -math.pi <= v <= math.pi for v in angles)
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                   or not .05 <= v <= 8 for v in scales)):
        return ()
    ratio = min(1., edge/max(picture.shape[:2]))
    size = (round(picture.shape[1]*ratio), round(picture.shape[0]*ratio))
    view_scale = np.array((size[0]/picture.shape[1], size[1]/picture.shape[0]))
    # BOX resampling locates a reduced sample at the centre of its source
    # footprint. Division alone returns its edge and adds a shared bias.
    view_origin = (1/view_scale-1)/2
    pixels = np.asarray(Image.fromarray(picture).resize(size, Image.Resampling.BOX))
    scene = pixels.astype(np.float32)/255
    angles = tuple({round((a+math.pi) % (2*math.pi)-math.pi, 12): (a+math.pi) % (2*math.pi)-math.pi
                    for a in angles}.values())
    proposals = _proposals(scene, reference, tuple(v*ratio for v in scales), angles)
    tried, found = [], []
    reference_hash, picture_hash = _digest(reference), _digest(picture)
    for _cost, x, y, template, template_linear, template_offset, angle in proposals:
        height, width = template.shape[:2]
        box = (x, y, width, height)
        if any(previous_angle == angle and _overlap(box, previous) > .6 for previous, previous_angle in tried):
            continue
        tried.append((box, angle))
        fit = _supported_fit(pixels, template, x, y)
        if fit is not None:
            fitted, starts, ends = fit
            if isinstance(fitted, AffineMotion):
                fitted_linear = np.asarray(fitted.linear)
            else:
                c, s = math.cos(fitted.angle), math.sin(fitted.angle)
                fitted_linear = np.array(((c, -s), (s, c)))
            linear = (fitted_linear @ template_linear)/view_scale[:, None]
            offset = (fitted_linear @ template_offset+np.asarray(fitted.offset))/view_scale+view_origin
            corners = np.array(((0., 0.), (reference.shape[1], 0.),
                                (0., reference.shape[0]), (reference.shape[1], reference.shape[0]))) @ linear.T+offset
            low, high = corners.min(axis=0), corners.max(axis=0)
            candidate = ReferenceAppearance(reference_hash, picture_hash,
                tuple(tuple(float(v) for v in row) for row in linear), tuple(float(v) for v in offset),
                (*map(float, low), *map(float, high-low)),
                tuple(tuple(float(v) for v in point) for point in
                      (starts-template_offset) @ np.linalg.inv(template_linear).T),
                tuple(tuple(float(v) for v in point/view_scale+view_origin) for point in ends), fitted.error/float(view_scale.min()))
            probes = ((0., 0.), (reference.shape[1]-1., 0.), (0., reference.shape[0]-1.))
            if not any(max(math.dist(candidate.project(point), old.project(point)) for point in probes) < 1.
                       for old in found):
                found.append(candidate)
        if len(tried) >= 24 or len(found) >= 8:
            break
    return tuple(found)


@invariant('perception.reference_orientation_keeps_source_coordinates', scope='perception',
           owner='core/perception/a_visual_reference_in_the_picture.py', observational=False)
def _oriented_reference_invariant() -> tuple:
    reference = np.random.default_rng(53).integers(0, 255, (48, 48, 4), dtype=np.uint8)
    reference[..., 3] = 255
    picture = np.zeros((96, 144, 3), np.uint8)
    picture[20:68, 30:78] = np.rot90(reference[..., :3], k=3)
    seen = reference_appearances(picture, reference, scales=(1.,), angles=(0., math.pi/2))
    assert len(seen) == 1
    assert math.dist(seen[0].project((11., 13.)), (64., 31.)) < .5
    return ()


@invariant('perception.shared_scene_cost_matches_direct_pixels', scope='perception',
           owner='core/perception/a_visual_reference_in_the_picture.py', observational=False)
def _scene_correlation_invariant() -> tuple:
    rng = np.random.default_rng(59)
    scene = rng.uniform(0., 1., (24, 32, 3)).astype(np.float32)
    rgb = rng.uniform(0., 1., (7, 9, 3)).astype(np.float32)
    mask = (rng.uniform(0., 1., (7, 9)) > .3).astype(np.float32)
    measured = _SceneCorrelations(scene).cost(rgb, mask)
    for y, x in ((0, 0), (11, 17), (17, 23)):
        direct = np.sum((scene[y:y+7, x:x+9]-rgb)**2*mask[..., None])/(3*mask.sum())
        assert abs(float(measured[y, x])-float(direct)) < 2e-6
    return ()


@invariant('perception.reduced_reference_matches_keep_pixel_centres', scope='perception',
           owner='core/perception/a_visual_reference_in_the_picture.py', observational=False)
def _reduced_reference_centres_invariant() -> tuple:
    reference = np.random.default_rng(53).integers(0, 255, (48, 48, 4), dtype=np.uint8)
    reference[..., 3] = 255
    raster = np.asarray(Image.fromarray(reference).resize((192, 192), Image.Resampling.BOX))
    picture = np.zeros((256, 256, 3), np.uint8)
    picture[32:224, 32:224] = raster[..., :3]
    pose, = reference_appearances(picture, reference, scales=(4.,), edge=64)
    assert math.dist(pose.project((13., 17.)), (85.5, 101.5)) < .25
    return ()


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
