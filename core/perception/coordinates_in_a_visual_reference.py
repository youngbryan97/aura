"""Carry source coordinates through independently matched reference pictures.

An exported image can crop or translate its source's coordinate origin. Its
visible box is not that origin, and neither one certifies a collision surface.
"""
from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance, _digest, reference_appearances
from core.verify import invariant

MAX_SVG_BYTES = 2*1024*1024
_SVG = '{http://www.w3.org/2000/svg}'
_FFDEC = '{https://www.free-decompiler.com/flash}'
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_MATRIX = re.compile(r'matrix\(\s*('+_NUMBER+r'(?:\s*,\s*'+_NUMBER+r'){5})\s*\)')


def exported_frame_coordinates(svg: bytes) -> tuple[tuple[int, int], tuple, tuple]:
    """Read a bounded frame export's declared source-to-SVG image transform.

    This adapter accepts the named renderer's frame convention. A different
    SVG structure needs its own coordinate contract, rather than an assumed
    origin. No SVG is executed or fetched here.
    """
    if (not isinstance(svg, bytes) or not 1 <= len(svg) <= MAX_SVG_BYTES
            or re.search(br'<!\s*(?:DOCTYPE|ENTITY)', svg, re.I)):
        raise ValueError('a bounded SVG without declarations is required')
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise ValueError('the coordinate document is not XML') from exc
    if (root.tag != _SVG+'svg' or root.get(_FFDEC+'objectType') != 'frame'
            or set(root.attrib)-{_FFDEC+'objectType', 'width', 'height'}):
        raise ValueError('the SVG has no supported exported-frame coordinate contract')
    size = []
    for axis in ('width', 'height'):
        raw = root.get(axis, '')
        if not re.fullmatch(_NUMBER+r'px', raw):
            raise ValueError('the exported frame needs explicit pixel dimensions')
        value = float(raw[:-2])
        if not math.isfinite(value) or not 16 <= value <= 512:
            raise ValueError('the exported frame dimensions are unsupported')
        size.append(math.ceil(value))
    children = list(root)
    groups = [child for child in children if child.tag == _SVG+'g']
    if len(groups) != 1 or any(child.tag not in (_SVG+'g', _SVG+'defs') for child in children):
        raise ValueError('the exported frame has no single source transform')
    group = groups[0]
    matrix = _MATRIX.fullmatch(group.get('transform', ''))
    if set(group.attrib) != {'transform'} or matrix is None:
        raise ValueError('the exported frame source transform is unsupported')
    for node in root.iter():
        if node.tag in tuple(_SVG+tag for tag in ('script', 'foreignObject', 'animate', 'animateTransform', 'set', 'style')):
            raise ValueError('active SVG content cannot supply a static registration')
        for key, value in node.attrib.items():
            urls = re.findall(r'url\(\s*([^)]*)\)', value, re.I)
            if (key.lower().startswith('on') or any(not url.strip(' \x22\x27').startswith('#') for url in urls)
                    or (key.endswith('href') and not value.startswith(('#', 'data:image/PNG;base64,', 'data:image/png;base64,')))):
                raise ValueError('external or active SVG content is unsupported')
    a, b, c, d, x, y = map(float, matrix[1].split(','))
    linear = np.array(((a, c), (b, d)))
    if not np.isfinite(linear).all() or not np.isfinite((x, y)).all() or abs(np.linalg.det(linear)) < 1e-8 or np.linalg.cond(linear) > 100:
        raise ValueError('the exported source transform is not resolved')
    return tuple(size), tuple(map(tuple, linear)), (x, y)


@dataclass(frozen=True)
class SourceImageCoordinates:
    """A declared source frame aligned with exact reference-image bytes."""

    source: str
    reference_sha256: str
    svg_sha256: str
    size: tuple[int, int]
    linear: tuple[tuple[float, float], tuple[float, float]]
    offset: tuple[float, float]
    calibration_error: float

    def __post_init__(self) -> None:
        try:
            linear, offset = np.asarray(self.linear, float), np.asarray(self.offset, float)
            valid = (isinstance(self.source, str) and bool(self.source)
                and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
                        for v in (self.reference_sha256, self.svg_sha256))
                and isinstance(self.size, tuple) and len(self.size) == 2
                and all(type(v) is int and 16 <= v <= 512 for v in self.size)
                and linear.shape == (2, 2) and offset.shape == (2,)
                and np.isfinite(linear).all() and np.isfinite(offset).all()
                and abs(np.linalg.det(linear)) >= 1e-8 and np.linalg.cond(linear) <= 100
                and type(self.calibration_error) in (int, float) and math.isfinite(self.calibration_error)
                and self.calibration_error >= 0)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            valid = False
        if not valid:
            raise ValueError('a reference coordinate frame needs finite measured geometry and byte identities')
        object.__setattr__(self, 'linear', tuple(tuple(float(v) for v in row) for row in linear))
        object.__setattr__(self, 'offset', tuple(float(v) for v in offset))

    def project(self, source_point: tuple, appearance: ReferenceAppearance, picture: np.ndarray,
                reference: np.ndarray, *, source: str) -> tuple[float, float] | None:
        """A source point in these exact matched pixels, without input authority."""
        if (source != self.source or not isinstance(appearance, ReferenceAppearance) or not _rgba(reference)
                or not isinstance(picture, np.ndarray) or picture.dtype != np.uint8
                or picture.ndim != 3 or picture.shape[2] != 3
                or (reference.shape[1], reference.shape[0]) != self.size
                or _digest(reference) != self.reference_sha256 or appearance.reference_sha256 != self.reference_sha256
                or _digest(picture) != appearance.picture_sha256
                or not isinstance(source_point, (tuple, list)) or len(source_point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in source_point)):
            return None
        point = np.asarray(self.linear) @ source_point + self.offset
        result = appearance.project(tuple(point))
        return tuple(float(v) for v in result) if all(math.isfinite(v) for v in result) else None


def _rgba(picture: Any) -> bool:
    return (isinstance(picture, np.ndarray) and picture.dtype == np.uint8 and picture.ndim == 3
            and picture.shape[2] == 4 and min(picture.shape[:2]) >= 16 and max(picture.shape[:2]) <= 512)


def calibrated_source_coordinates(source: str, svg: bytes, svg_pixels: np.ndarray,
                                  reference: np.ndarray) -> SourceImageCoordinates | None:
    """Require an independent raster correspondence before transferring coordinates."""
    size, linear, offset = exported_frame_coordinates(svg)
    if (not isinstance(source, str) or not source or not _rgba(svg_pixels) or not _rgba(reference)
            or (svg_pixels.shape[1], svg_pixels.shape[0]) != size):
        return None
    poses = reference_appearances(reference[..., :3], svg_pixels, scales=(1.,), edge=512)
    if len(poses) != 1:
        return None
    pose = poses[0]
    # These are exports of one whole frame. A common subimage cannot align
    # their coordinate frames when the outer image extent contradicts it.
    corners = ((0., 0.), (size[0], 0.), (0., size[1]), tuple(size))
    expected = ((0., 0.), (reference.shape[1], 0.), (0., reference.shape[0]),
                (reference.shape[1], reference.shape[0]))
    if any(math.dist(pose.project(corner), wanted) > 2. for corner, wanted in zip(corners, expected, strict=True)):
        return None
    fitted = np.asarray(pose.linear)
    transform, origin = fitted @ np.asarray(linear), fitted @ np.asarray(offset) + pose.offset
    return SourceImageCoordinates(source, _digest(reference), hashlib.sha256(svg).hexdigest(),
        (reference.shape[1], reference.shape[0]), tuple(tuple(float(v) for v in row) for row in transform),
        tuple(float(v) for v in origin), float(pose.error))


@invariant('perception.reference_coordinates_require_raster_alignment', scope='perception',
           owner='core/perception/coordinates_in_a_visual_reference.py', observational=False)
def _source_image_coordinates_invariant() -> tuple:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:f="https://www.free-decompiler.com/flash" f:objectType="frame" width="32px" height="32px"><g transform="matrix(1,0,0,1,8,-4)"/></svg>'
    assert exported_frame_coordinates(svg) == ((32, 32), ((1., 0.), (0., 1.)), (8., -4.))
    flat = np.zeros((32, 32, 4), np.uint8)
    flat[..., 3] = 255
    assert calibrated_source_coordinates('source', svg, flat, flat) is None
    return ()
