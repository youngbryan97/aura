"""core/worlds/obb.py
──────────────────
Oriented-bounding-box collision geometry (SAT + face clipping).

Pure functions over numpy arrays — no engine state — so every piece is
independently testable:

- quaternion → rotation matrix
- OBB vs plane: support-vertex contacts (up to 4-point manifold)
- OBB vs sphere: closest-point contact
- OBB vs OBB: separating-axis test over the 15 canonical axes, with a
  reference-face / incident-face clip (Sutherland–Hodgman) producing a
  stable contact manifold.

Convention matches core/worlds/physics.py: contact normal points from
body A toward body B; penetration is positive overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3×3 rotation matrix. Normalizes first so
    integration drift can never de-orthogonalize the frame."""
    q = np.asarray(q, dtype=np.float64)
    q = q / float(np.linalg.norm(q))
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass
class ContactPoint:
    point: np.ndarray        # world-space contact location
    penetration: float


@dataclass
class Manifold:
    normal: np.ndarray       # from A toward B, unit length
    points: list[ContactPoint]

    @property
    def max_penetration(self) -> float:
        return max((p.penetration for p in self.points), default=0.0)


def obb_vertices(center: np.ndarray, rotation: np.ndarray,
                 half_extents: np.ndarray) -> np.ndarray:
    """The 8 world-space corners, deterministic order."""
    signs = np.array([
        [sx, sy, sz]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ])
    return center + (signs * half_extents) @ rotation.T


def obb_vs_plane(center: np.ndarray, rotation: np.ndarray,
                 half_extents: np.ndarray, plane_height: float) -> Manifold | None:
    """Contacts where box corners dip below z = plane_height.
    Normal is (0, 0, -1): from the box (A) toward the plane (B)."""
    vertices = obb_vertices(center, rotation, half_extents)
    below = vertices[vertices[:, 2] < plane_height]
    if below.size == 0:
        return None
    # Deepest four corners give a stable manifold.
    order = np.argsort(below[:, 2])
    points = [
        ContactPoint(point=vertex.copy(),
                     penetration=float(plane_height - vertex[2]))
        for vertex in below[order][:4]
    ]
    return Manifold(normal=np.array([0.0, 0.0, -1.0]), points=points)


def obb_vs_sphere(center: np.ndarray, rotation: np.ndarray,
                  half_extents: np.ndarray, sphere_center: np.ndarray,
                  radius: float) -> Manifold | None:
    """Closest-point test in the box frame. Normal from box (A) toward
    sphere (B)."""
    local = rotation.T @ (sphere_center - center)
    clamped = np.clip(local, -half_extents, half_extents)
    delta = local - clamped
    distance_sq = float(delta @ delta)
    if distance_sq > radius * radius:
        return None
    if distance_sq > 1e-18:
        distance = float(np.sqrt(distance_sq))
        normal = rotation @ (delta / distance)
        penetration = radius - distance
        point = center + rotation @ clamped
    else:
        # Center inside the box: exit along the axis of least depth.
        gaps = half_extents - np.abs(local)
        axis = int(np.argmin(gaps))
        direction = np.zeros(3)
        direction[axis] = 1.0 if local[axis] >= 0.0 else -1.0
        normal = rotation @ direction
        penetration = float(gaps[axis]) + radius
        point = sphere_center - normal * radius
    return Manifold(normal=normal,
                    points=[ContactPoint(point=point, penetration=float(penetration))])


def _sat_axes(rot_a: np.ndarray, rot_b: np.ndarray) -> list[np.ndarray]:
    axes: list[np.ndarray] = [rot_a[:, i].copy() for i in range(3)]
    axes += [rot_b[:, i].copy() for i in range(3)]
    for i in range(3):
        for j in range(3):
            cross = np.cross(rot_a[:, i], rot_b[:, j])
            norm = float(np.linalg.norm(cross))
            if norm > 1e-9:
                axes.append(cross / norm)
    return axes


def _project_radius(rotation: np.ndarray, half_extents: np.ndarray,
                    axis: np.ndarray) -> float:
    return float(np.sum(np.abs((rotation.T @ axis)) * half_extents))


def obb_vs_obb(center_a: np.ndarray, rot_a: np.ndarray, half_a: np.ndarray,
               center_b: np.ndarray, rot_b: np.ndarray, half_b: np.ndarray,
               ) -> Manifold | None:
    """SAT over 15 axes; on overlap, clip the incident face of B against
    the reference face of A (or vice versa) for a stable manifold."""
    delta = center_b - center_a
    best_axis: np.ndarray | None = None
    best_overlap = np.inf
    for axis in _sat_axes(rot_a, rot_b):
        projection = float(delta @ axis)
        overlap = (
            _project_radius(rot_a, half_a, axis)
            + _project_radius(rot_b, half_b, axis)
            - abs(projection)
        )
        if overlap <= 0.0:
            return None
        if overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis if projection >= 0.0 else -axis
    assert best_axis is not None

    # Reference face: the face of A most anti-parallel... most aligned
    # with the contact normal; incident face: face of B most opposed.
    normal = best_axis / float(np.linalg.norm(best_axis))
    points = _clip_manifold(center_a, rot_a, half_a, center_b, rot_b, half_b, normal)
    if not points:
        # Degenerate clip (edge-edge): fall back to midpoint contact.
        points = [ContactPoint(
            point=(center_a + center_b) / 2.0, penetration=float(best_overlap))]
    return Manifold(normal=normal, points=points)


def _face_vertices(center: np.ndarray, rotation: np.ndarray,
                   half_extents: np.ndarray, axis_index: int,
                   sign: float) -> np.ndarray:
    """The 4 vertices of one box face, ordered around the perimeter."""
    u_index, v_index = (axis_index + 1) % 3, (axis_index + 2) % 3
    normal_offset = sign * half_extents[axis_index] * rotation[:, axis_index]
    u = half_extents[u_index] * rotation[:, u_index]
    v = half_extents[v_index] * rotation[:, v_index]
    face_center = center + normal_offset
    return np.array([
        face_center + u + v, face_center - u + v,
        face_center - u - v, face_center + u - v,
    ])


def _clip_polygon(polygon: np.ndarray, plane_normal: np.ndarray,
                  plane_offset: float) -> np.ndarray:
    """Keep the polygon region with plane_normal·p ≤ plane_offset."""
    output: list[np.ndarray] = []
    count = len(polygon)
    for index in range(count):
        current, following = polygon[index], polygon[(index + 1) % count]
        d_current = float(plane_normal @ current) - plane_offset
        d_next = float(plane_normal @ following) - plane_offset
        if d_current <= 0.0:
            output.append(current)
        if (d_current < 0.0) != (d_next < 0.0) and abs(d_next - d_current) > 1e-12:
            t = d_current / (d_current - d_next)
            output.append(current + t * (following - current))
    return np.array(output) if output else np.empty((0, 3))


def _clip_manifold(center_a, rot_a, half_a, center_b, rot_b, half_b,
                   normal: np.ndarray) -> list[ContactPoint]:
    # Reference face on A: face whose outward normal best matches +normal.
    alignments_a = [float(normal @ rot_a[:, i]) for i in range(3)]
    ref_axis = int(np.argmax(np.abs(alignments_a)))
    ref_sign = 1.0 if alignments_a[ref_axis] >= 0.0 else -1.0
    ref_normal = ref_sign * rot_a[:, ref_axis]
    ref_offset = float(ref_normal @ (center_a + ref_normal * half_a[ref_axis]))

    # Incident face on B: face most anti-parallel to the reference normal.
    alignments_b = [float(ref_normal @ rot_b[:, i]) for i in range(3)]
    inc_axis = int(np.argmax(np.abs(alignments_b)))
    inc_sign = -1.0 if alignments_b[inc_axis] >= 0.0 else 1.0
    polygon = _face_vertices(center_b, rot_b, half_b, inc_axis, inc_sign)

    # Clip against the four side planes of the reference face.
    for side in range(3):
        if side == ref_axis:
            continue
        axis = rot_a[:, side]
        for sign in (1.0, -1.0):
            plane_normal = sign * axis
            offset = float(plane_normal @ center_a) + half_a[side]
            polygon = _clip_polygon(polygon, plane_normal, offset)
            if polygon.size == 0:
                return []

    # Keep points at or below the reference face (penetrating).
    points: list[ContactPoint] = []
    for vertex in polygon:
        depth = ref_offset - float(ref_normal @ vertex)
        if depth >= -1e-9:
            points.append(ContactPoint(point=vertex.copy(),
                                       penetration=max(0.0, depth)))
    points.sort(key=lambda cp: -cp.penetration)
    return points[:4]
