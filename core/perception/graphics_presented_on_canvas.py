"""Map geometric search regions through recorded framebuffer copies.

Coordinates follow homogeneous clipping and public copy rectangles. A mapped
polygon remains a search hint: fragment coverage, pixels and object identity
must be measured separately. Missing presentation provenance returns unknown.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.verify import invariant

Point = tuple[float, float]
Polygon = tuple[Point, ...]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageVersion:
    """An attachment image and its content within one context generation."""

    context: int
    epoch: int
    image: int
    content: int
    width: int
    height: int

    def valid(self) -> bool:
        identifiers = (self.context, self.epoch, self.image, self.content)
        return (all(type(v) is int and v >= 0 for v in identifiers)
                and all(type(v) is int and 0 < v <= 16384 for v in (self.width, self.height)))


@dataclass(frozen=True)
class FramebufferCopy:
    """One admitted nearest color blit, in bottom-left framebuffer units.

    Rectangles retain endpoint order, including flips. The collector must
    bind versions to successful public operations; framebuffer names alone
    cannot identify an image after attachments or contents change.
    """

    source: ImageVersion
    destination: ImageVersion
    source_rect: Rect
    destination_rect: Rect
    sequence: int
    color_only: bool = True
    nearest: bool = True
    scissor: Rect | None = None

    def valid(self) -> bool:
        return (self.source.valid() and self.destination.valid()
                and self.source.context == self.destination.context
                and self.source.epoch == self.destination.epoch
                and self.source.image != self.destination.image
                and type(self.sequence) is int and self.sequence >= 0
                and self.color_only is True and self.nearest is True
                and _rect(self.source_rect) and _rect(self.destination_rect)
                and (self.scissor is None or _rect(self.scissor)))


@dataclass(frozen=True)
class PresentedCanvas:
    """The image version captured with a frame's measured clock receipt."""

    image: ImageVersion
    sequence: int
    frame: MeasuredFrameTime

    def valid(self) -> bool:
        f = self.frame
        return (self.image.valid() and type(self.sequence) is int and self.sequence >= 0
                and isinstance(f, MeasuredFrameTime) and f.lower is not None
                and f.epoch == self.image.epoch and f.source_at is not None
                and all(_finite(v) for v in (float(f), f.lower, f.upper, f.arrived, f.source_at))
                and 0 <= f.lower <= float(f) <= f.upper <= f.arrived)


@dataclass(frozen=True)
class GeometricHints:
    """Current canvas polygons whose visible material is still unverified."""

    polygons: tuple[Polygon, ...]
    image: ImageVersion
    frame: MeasuredFrameTime
    copies: tuple[int, ...]
    coverage: str = "unverified"


@dataclass(frozen=True)
class GeometryResult:
    hints: GeometricHints | None
    reason: str | None = None


def _finite(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and abs(value) <= 1e12


def _rect(value) -> bool:
    return (type(value) is tuple and len(value) == 4 and all(_finite(v) for v in value)
            and value[0] != value[2] and value[1] != value[3])


def _clip(points: tuple, plane) -> tuple:
    """Sutherland-Hodgman clipping works before perspective division too."""
    if not points:
        return ()
    out, previous = [], points[-1]
    before = plane(previous)
    for current in points:
        after = plane(current)
        if (before >= 0) != (after >= 0):
            fraction = before / (before - after)
            out.append(tuple(a + fraction * (b - a) for a, b in zip(previous, current, strict=True)))
        if after >= 0:
            out.append(current)
        previous, before = current, after
    return tuple(out)


def _in_rect(polygon: Polygon, rect: Rect) -> Polygon:
    x0, y0, x1, y1 = rect
    for axis, bound, sign in ((0, min(x0, x1), 1), (0, max(x0, x1), -1),
                              (1, min(y0, y1), 1), (1, max(y0, y1), -1)):
        polygon = _clip(polygon, lambda p, a=axis, b=bound, s=sign: s * (p[a] - b))
    return polygon


def project_triangles(vertices: tuple, indices: tuple, viewport: Rect) -> tuple[Polygon, ...] | None:
    """Clip admitted triangles in homogeneous units, then apply the viewport.

    `viewport` contains x, y, width and height. Returned coordinates have a
    bottom-left origin. None means unsupported; an empty tuple means clipped.
    """
    if (type(vertices) is not tuple or not 0 < len(vertices) <= 4096
            or type(indices) is not tuple or len(indices) > 12288 or len(indices) % 3
            or type(viewport) is not tuple or len(viewport) != 4
            or not all(_finite(v) for v in viewport) or min(viewport[2:]) <= 0):
        return None
    if any(type(p) is not tuple or len(p) != 4 or not all(_finite(v) for v in p) for p in vertices):
        return None
    if any(type(i) is not int or not 0 <= i < len(vertices) for i in indices):
        return None
    polygons = []
    for offset in range(0, len(indices), 3):
        polygon = tuple(vertices[i] for i in indices[offset:offset + 3])
        for axis in range(3):
            for sign in (-1, 1):
                polygon = _clip(polygon, lambda p, a=axis, s=sign: p[3] + s * p[a])
        if not polygon:
            continue
        if any(p[3] <= 1e-12 for p in polygon):
            return None
        x, y, w, h = viewport
        result = tuple((x + (p[0] / p[3] + 1) * w / 2,
                        y + (p[1] / p[3] + 1) * h / 2) for p in polygon)
        if _area(result) > 1e-12:
            polygons.append(result)
    return tuple(polygons)


def _area(polygon: Polygon) -> float:
    return abs(sum(a[0] * b[1] - a[1] * b[0]
                   for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True))) / 2


def _copied(polygons: tuple[Polygon, ...], copy: FramebufferCopy) -> tuple[Polygon, ...]:
    sx0, sy0, sx1, sy1 = copy.source_rect
    dx0, dy0, dx1, dy1 = copy.destination_rect
    source_bounds = (0., 0., float(copy.source.width), float(copy.source.height))
    target_bounds = (0., 0., float(copy.destination.width), float(copy.destination.height))
    result = []
    for polygon in polygons:
        clipped = _in_rect(_in_rect(polygon, source_bounds), copy.source_rect)
        moved = tuple((dx0 + (x - sx0) * (dx1 - dx0) / (sx1 - sx0),
                       dy0 + (y - sy0) * (dy1 - dy0) / (sy1 - sy0)) for x, y in clipped)
        moved = _in_rect(moved, target_bounds)
        if copy.scissor is not None:
            moved = _in_rect(moved, copy.scissor)
        if _area(moved) > 1e-12:
            result.append(moved)
    return tuple(result)


def presented_hints(polygons: tuple[Polygon, ...], source: ImageVersion, draw_sequence: int,
                    copies: Iterable[FramebufferCopy], canvas: PresentedCanvas) -> GeometryResult:
    """Follow one current, ordered, unambiguous copy route to captured pixels."""
    if (not source.valid() or not canvas.valid() or source.context != canvas.image.context
            or source.epoch != canvas.image.epoch or type(draw_sequence) is not int
            or not 0 <= draw_sequence <= canvas.sequence):
        return GeometryResult(None, "capture_custody")
    if (type(polygons) is not tuple or len(polygons) > 4096
            or any(type(p) is not tuple or not 3 <= len(p) <= 16
                   or any(type(v) is not tuple or len(v) != 2 or not all(_finite(x) for x in v) for v in p)
                   for p in polygons)):
        return GeometryResult(None, "geometry_budget_or_values")
    edges = []
    for edge in copies:
        if len(edges) >= 64:
            return GeometryResult(None, "copy_budget")
        if not isinstance(edge, FramebufferCopy) or not edge.valid():
            return GeometryResult(None, "unsupported_copy")
        edges.append(edge)
    at, image, route = draw_sequence, source, []
    for _ in range(17):
        if image == canvas.image:
            bounded = tuple(_in_rect(p, (0., 0., float(image.width), float(image.height))) for p in polygons)
            top_left = tuple(tuple((x, image.height - y) for x, y in p) for p in bounded if _area(p) > 1e-12)
            return GeometryResult(GeometricHints(top_left, image, canvas.frame, tuple(route)))
        options = [e for e in edges if e.source == image and at < e.sequence <= canvas.sequence]
        if len(options) != 1:
            return GeometryResult(None, "missing_or_ambiguous_presentation")
        if len(route) >= 16:
            return GeometryResult(None, "presentation_hop_budget")
        edge = options[0]
        polygons = _copied(polygons, edge)
        image, at = edge.destination, edge.sequence
        route.append(at)
    return GeometryResult(None, "presentation_hop_budget")


@invariant("perception.graphics_hints_require_current_presentation", scope="perception",
           owner="core/perception/graphics_presented_on_canvas.py", observational=False)
def _presentation_custody_invariant() -> tuple:
    frame = MeasuredFrameTime(1., lower=1., upper=1., arrived=1.01,
                              source="browser-monotonic", source_at=10., epoch=2)
    source = ImageVersion(1, 2, 7, 4, 100, 80)
    target = ImageVersion(1, 2, 0, 9, 100, 80)
    canvas = PresentedCanvas(target, 10, frame)
    polygon = (((10., 20.), (20., 20.), (10., 30.)),)
    copy = FramebufferCopy(source, target, (0., 80., 100., 0.), (0., 0., 100., 80.), 9)
    result = presented_hints(polygon, source, 8, (copy,), canvas)
    assert result.hints is not None and result.hints.polygons == polygon
    assert result.hints.coverage == "unverified"
    assert presented_hints(polygon, source, 8, (), canvas).hints is None
    changed = ImageVersion(1, 2, 0, 10, 100, 80)
    assert presented_hints(polygon, source, 8, (copy,), PresentedCanvas(changed, 10, frame)).hints is None
    return ()
