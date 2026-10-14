"""Project texture regions through measured smooth triangle coordinates.

Source barycentric weights combine homogeneous destination positions. The
division then follows the same perspective interpolation as the rendering API.
Coverage and object identity remain unverified.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.perception.graphics_presented_on_canvas import (
    ImageVersion,
    Polygon,
    Rect,
    _area,
    _clip,
    _finite,
    _in_rect,
    _rect,
    project_polygon,
)
from core.verify import invariant


@dataclass(frozen=True)
class TextureProjection:
    source: ImageVersion
    destination: ImageVersion
    positions: tuple[tuple[float, float, float, float], ...]
    coordinates: tuple[tuple[float, float], ...]
    indices: tuple[int, ...]
    viewport: Rect
    sequence: int
    scissor: Rect | None = None
    smooth: bool = False
    base_level_nearest: bool = False

    def valid(self) -> bool:
        return (self.source.valid() and self.destination.valid()
                and self.source.context == self.destination.context and self.source.epoch == self.destination.epoch
                and self.source.image != self.destination.image and type(self.sequence) is int and self.sequence >= 0
                and self.smooth is True and self.base_level_nearest is True
                and type(self.positions) is tuple and 3 <= len(self.positions) <= 64
                and type(self.coordinates) is tuple and len(self.coordinates) == len(self.positions)
                and all(type(p) is tuple and len(p) == 4 and all(_finite(v) for v in p) for p in self.positions)
                and all(type(p) is tuple and len(p) == 2 and all(_finite(v) and 0 <= v <= 1 for v in p) for p in self.coordinates)
                and type(self.indices) is tuple and 0 < len(self.indices) <= 192 and len(self.indices) % 3 == 0
                and all(type(i) is int and 0 <= i < len(self.positions) for i in self.indices)
                and type(self.viewport) is tuple and len(self.viewport) == 4
                and all(_finite(v) for v in self.viewport) and min(self.viewport[2:]) > 0
                and (self.scissor is None or _rect(self.scissor)))

    def mapped(self, polygons: tuple[Polygon, ...]) -> tuple[Polygon, ...] | None:
        if not self.valid() or type(polygons) is not tuple or len(polygons) > 4096:
            return None
        result, work = [], 0
        source_bounds = (0., 0., float(self.source.width), float(self.source.height))
        target_bounds = (0., 0., float(self.destination.width), float(self.destination.height))
        for offset in range(0, len(self.indices), 3):
            indices = self.indices[offset:offset + 3]
            triangle = tuple(self.coordinates[i] for i in indices)
            weights = _weights(triangle)
            if weights is None:
                return None
            for polygon in polygons:
                if (type(polygon) is not tuple or not 3 <= len(polygon) <= 16
                        or any(type(p) is not tuple or len(p) != 2 or not all(_finite(v) for v in p) for p in polygon)):
                    return None
                work += len(polygon)
                if work > 65536:
                    return None
                clipped = _in_rect(polygon, source_bounds)
                def barycentric(p, current_weights=weights):
                    return current_weights(p[0] / self.source.width, p[1] / self.source.height)
                for axis in range(3):
                    clipped = _clip(clipped, lambda p, a=axis: barycentric(p)[a])
                if len(clipped) < 3 or _area(clipped) <= 1e-12:
                    continue
                homogeneous = tuple(tuple(sum(value * self.positions[i][component]
                                               for value, i in zip(barycentric(p), indices, strict=True))
                                          for component in range(4)) for p in clipped)
                mapped = project_polygon(homogeneous, self.viewport)
                if mapped is None:
                    return None
                mapped = _in_rect(mapped, target_bounds)
                if self.scissor is not None:
                    mapped = _in_rect(mapped, self.scissor)
                if len(mapped) > 16:
                    return None
                if _area(mapped) > 1e-12:
                    result.append(mapped)
                if len(result) > 4096:
                    return None
        return tuple(result)


def _weights(triangle):
    (u0, v0), (u1, v1), (u2, v2) = triangle
    det = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
    if abs(det) <= 1e-10:
        return None
    def weights(u, v):
        a = ((u - u0) * (v2 - v0) - (v - v0) * (u2 - u0)) / det
        b = ((u1 - u0) * (v - v0) - (v1 - v0) * (u - u0)) / det
        return 1 - a - b, a, b
    return weights


@invariant("perception.texture_regions_use_measured_projective_coordinates", scope="perception",
           owner="core/perception/texture_coordinates_to_a_surface.py", observational=False)
def _texture_projection_invariant() -> tuple:
    a, b = ImageVersion(1, 2, 3, 4, 100, 80), ImageVersion(1, 2, 5, 6, 100, 80)
    projection = TextureProjection(a, b, ((-1., -1., 0., 1.), (1., -1., 0., 1.), (-1., 1., 0., 1.)),
                                   ((0., 0.), (1., 0.), (0., 1.)), (0, 1, 2), (0., 0., 100., 80.), 7,
                                   smooth=True, base_level_nearest=True)
    polygon = (((10., 10.), (20., 10.), (10., 20.)),)
    mapped = projection.mapped(polygon)
    assert mapped is not None and len(mapped) == 1 and len(mapped[0]) == 3
    assert max(abs(a - b) for p, q in zip(mapped[0], polygon[0], strict=True)
               for a, b in zip(p, q, strict=True)) < 1e-9
    return ()
