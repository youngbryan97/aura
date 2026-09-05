"""Turning a solid model into the lines a drawing is made of.

Three things have to happen between a mesh and a technical illustration.
The geometry is projected onto the sheet, which for engineering work means
a parallel projection rather than a perspective one, because a parallel
projection keeps parallel edges parallel and lets a reader measure off the
drawing. Faces pointing away from the viewer are dropped. And edges hidden
behind material are found, so they can be drawn dashed or left out, which is
the difference between a technical illustration and a bird's nest of lines.

Isometric is the default because it shows three faces at once with all three
axes equally foreshortened, which is why every exploded parts diagram ever
printed uses it. The standard orthographic views are here too, in both
first-angle and third-angle arrangement, since a drawing that does not say
which one it uses is ambiguous by exactly one mirror image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "View",
    "VIEWS",
    "view_named",
    "Projected",
    "project_mesh",
    "shade",
    "PROJECTION_SYMBOL_FIRST",
    "PROJECTION_SYMBOL_THIRD",
]

#: The truncated cone that says which angle of projection a sheet uses.
#: ISO 128-30 makes the symbol mandatory; without it the side view could be
#: on either side and mean opposite things.
PROJECTION_SYMBOL_FIRST = "first"
PROJECTION_SYMBOL_THIRD = "third"


@dataclass(frozen=True, slots=True)
class View:
    """A camera: where it looks from, which way is up, and how it projects."""

    key: str
    name: str
    #: Direction the camera looks along, in model space.
    direction: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Parallel keeps measurements true; perspective looks natural and lies.
    perspective: float = 0.0
    description: str = ""

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Right, up and forward unit vectors for this view."""
        forward = np.asarray(self.direction, dtype=float)
        norm = np.linalg.norm(forward)
        if norm == 0:
            raise ValueError("a view direction cannot be zero")
        forward = forward / norm
        up = np.asarray(self.up, dtype=float)
        if abs(float(np.dot(up, forward))) > 0.999:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm == 0:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        true_up = np.cross(right, forward)
        return (right, true_up, forward)


_ISO = (1.0, 1.0, -1.0)

VIEWS: dict[str, View] = {
    entry.key: entry
    for entry in (
        View("iso", "Isometric", _ISO, (0.0, 0.0, 1.0),
             description="Three faces at once, all three directions equally scaled."),
        View("iso_left", "Isometric from the left", (-1.0, 1.0, -1.0), (0.0, 0.0, 1.0),
             description="The same view from the other side."),
        View("iso_rear", "Isometric from behind", (-1.0, -1.0, -1.0), (0.0, 0.0, 1.0),
             description="What the back looks like."),
        View("iso_below", "Isometric from below", (1.0, 1.0, 1.0), (0.0, 0.0, 1.0),
             description="Underside, for mounting faces and drains."),
        View("front", "Front", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
             description="Straight on. Widths and heights measure true here."),
        View("rear", "Rear", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        View("top", "Top", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0),
             description="Looking down. Widths and depths measure true here."),
        View("bottom", "Bottom", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        View("right", "Right", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
             description="From the right. Depths and heights measure true here."),
        View("left", "Left", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        View("dimetric", "Dimetric", (1.0, 0.55, -0.75), (0.0, 0.0, 1.0),
             description="Front face favoured, depth foreshortened."),
    )
}


def view_named(key: str) -> View:
    text = str(key or "iso").strip().lower().replace(" ", "_")
    if text in VIEWS:
        return VIEWS[text]
    raise KeyError(f"no view named {key!r}; the views are {', '.join(sorted(VIEWS))}")


@dataclass(frozen=True, slots=True)
class Projected:
    """One mesh flattened onto the sheet, with what is in front of what."""

    #: Nx2, in the projection's own units before any sheet scaling.
    points: np.ndarray
    #: N, distance along the view direction. Smaller is nearer the viewer.
    depth: np.ndarray
    #: Mx3 triangle indices, front-facing only, sorted back to front.
    faces: np.ndarray
    #: M, the depth of each face's centroid.
    face_depth: np.ndarray
    #: M, how square-on each face is to the light, 0 to 1.
    face_light: np.ndarray
    #: Edges that can be seen, and edges hidden behind material.
    visible_edges: tuple[tuple[int, int], ...]
    hidden_edges: tuple[tuple[int, int], ...]
    #: The outline of the whole silhouette, for masking parts behind it.
    silhouette: tuple[tuple[float, float], ...]

    def bounds(self) -> tuple[float, float, float, float]:
        if len(self.points) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
        )


#: Where the light comes from, for the wash that gives a wireframe its form.
#: Over the viewer's left shoulder, which is the convention in every
#: illustration manual and the direction that reads as "lit" rather than
#: "strange".
_LIGHT = np.array([-0.4, -0.5, 0.76])


def project_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    view: View,
    *,
    hidden_lines: bool = True,
    hidden_samples: int = 5,
) -> Projected:
    """Flatten a mesh for one view, sorting faces and classifying edges."""
    right, up, forward = view.basis()
    if len(vertices) == 0:
        empty2 = np.zeros((0, 2))
        return Projected(empty2, np.zeros(0), np.zeros((0, 3), dtype=int),
                         np.zeros(0), np.zeros(0), (), (), ())
    points = np.stack(
        [vertices @ right, vertices @ up], axis=1
    )
    depth = vertices @ forward
    if view.perspective > 0:
        scale = 1.0 / (1.0 + view.perspective * (depth - depth.min()))
        points = points * scale[:, None]

    if len(faces) == 0:
        return Projected(points, depth, np.zeros((0, 3), dtype=int), np.zeros(0),
                         np.zeros(0), tuple(edges), (), _hull(points))

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths[:, None]
    facing = normals @ forward
    front = facing < 0.0
    if not front.any():
        # An open surface with the winding the other way round.
        front = facing > 0.0
        normals = -normals
    kept = faces[front]
    kept_normals = normals[front]
    centroid_depth = (depth[kept[:, 0]] + depth[kept[:, 1]] + depth[kept[:, 2]]) / 3.0
    order = np.argsort(-centroid_depth)
    kept = kept[order]
    kept_normals = kept_normals[order]
    centroid_depth = centroid_depth[order]
    light = np.clip(kept_normals @ _LIGHT, 0.0, 1.0)

    visible: list[tuple[int, int]] = []
    hidden: list[tuple[int, int]] = []
    if hidden_lines and len(kept) > 0 and edges:
        occluded = _occlusion_mask(points, depth, kept, edges, hidden_samples)
        for edge, is_hidden in zip(edges, occluded):
            (hidden if is_hidden else visible).append(edge)
    else:
        visible = list(edges)

    return Projected(
        points,
        depth,
        kept,
        centroid_depth,
        light,
        tuple(visible),
        tuple(hidden),
        _hull(points),
    )


def _occlusion_mask(
    points: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    samples: int,
) -> list[bool]:
    """Which edges have material in front of them.

    Each edge is sampled along its length and each sample is tested against
    every front-facing triangle by barycentric coordinates. An edge counts
    as hidden when most of it is covered, which keeps a line that grazes a
    silhouette from flickering between states.
    """
    if len(faces) == 0 or not edges:
        return [False] * len(edges)
    tri = points[faces]
    tri_depth = depth[faces]
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    denominator = d00 * d11 - d01 * d01
    live = np.abs(denominator) > 1e-18
    denominator = np.where(live, denominator, 1.0)

    # A sample must be this much further back than a face to count as hidden,
    # so a face does not occlude its own boundary edges.
    span = float(depth.max() - depth.min()) or 1.0
    bias = span * 1e-3

    result: list[bool] = []
    fractions = np.linspace(0.12, 0.88, max(int(samples), 1))
    for start, end in edges:
        p0 = points[start]
        p1 = points[end]
        z0 = depth[start]
        z1 = depth[end]
        covered = 0
        for t in fractions:
            point = p0 + (p1 - p0) * t
            sample_depth = z0 + (z1 - z0) * t
            v2 = point - tri[:, 0]
            d20 = np.einsum("ij,ij->i", v2, v0)
            d21 = np.einsum("ij,ij->i", v2, v1)
            u = (d11 * d20 - d01 * d21) / denominator
            v = (d00 * d21 - d01 * d20) / denominator
            inside = live & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
            if not inside.any():
                continue
            w = 1.0 - u - v
            face_depth = (
                w * tri_depth[:, 0] + u * tri_depth[:, 1] + v * tri_depth[:, 2]
            )
            if bool((inside & (face_depth < sample_depth - bias)).any()):
                covered += 1
        result.append(covered > len(fractions) // 2)
    return result


def _hull(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    """The convex outline of a projected part, for masking what is behind it."""
    if len(points) < 3:
        return tuple((float(x), float(y)) for x, y in points)
    ordered = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(ordered) < 3:
        return tuple(ordered)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def shade(base: str, light: float, *, floor: float = 0.55, ceiling: float = 1.0) -> str:
    """Lighten or darken a hex colour by how square-on a face is to the light.

    A flat wireframe reads as a tangle. A wash that changes across a curved
    surface reads as a shape, and it costs one multiply per face.
    """
    text = base.lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except (ValueError, IndexError):
        return base
    factor = floor + (ceiling - floor) * max(0.0, min(1.0, float(light)))
    return "#%02x%02x%02x" % (
        max(0, min(255, int(r * factor))),
        max(0, min(255, int(g * factor))),
        max(0, min(255, int(b * factor))),
    )
