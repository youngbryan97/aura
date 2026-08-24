"""Parametric solids that know their own volume, mass and shape.

Every part in a design is one of a small number of shapes with a handful of
dimensions. Given those dimensions the volume is exact arithmetic, the mass
follows from the material, and the surface a drawing needs is a mesh
generated from the same parameters. Nothing is estimated and nothing is
described in words that a renderer then has to interpret.

The mesh and the wireframe are separate on purpose. The mesh is triangles,
for section cuts, interference checks and STL export. The wireframe is the
parametric grid — the rings and rails of a surface of revolution, the
twelve edges of a box — which is what a technical illustration draws, and
what makes a curved surface read as curved instead of as a field of
triangles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.engineering.units import Q, Quantity

__all__ = [
    "Solid",
    "Box",
    "Cylinder",
    "Tube",
    "Sphere",
    "Dome",
    "Cone",
    "Frustum",
    "Torus",
    "Plate",
    "Ellipsoid",
    "Prism",
    "Capsule",
    "Mesh",
    "Placement",
    "solid_from_spec",
    "SOLID_KINDS",
]


@dataclass(frozen=True, slots=True)
class Mesh:
    """Triangles in metres, plus the edges worth drawing."""

    vertices: np.ndarray
    faces: np.ndarray
    edges: tuple[tuple[int, int], ...] = ()

    def transformed(self, placement: Placement) -> Mesh:
        return Mesh(placement.apply(self.vertices), self.faces, self.edges)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            zero = np.zeros(3)
            return zero, zero
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


@dataclass(frozen=True, slots=True)
class Placement:
    """Where a part sits: a rotation about each axis, then a translation."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> np.ndarray:
        rx, ry, rz = self.rotation
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
        my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
        mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
        return mz @ my @ mx

    def apply(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        return points @ self.matrix().T + np.asarray(self.position, dtype=float)

    def translated(self, offset: tuple[float, float, float]) -> Placement:
        return Placement(
            (
                self.position[0] + offset[0],
                self.position[1] + offset[1],
                self.position[2] + offset[2],
            ),
            self.rotation,
        )


def _length(value: Any, name: str) -> float:
    """Read one dimension in metres, refusing anything that is not a length."""
    quantity = value if isinstance(value, Quantity) else Q(value, "m")
    if quantity.dimension != Q(1, "m").dimension:
        raise ValueError(f"{name} must be a length, not {quantity.unit}")
    if quantity.value <= 0:
        raise ValueError(f"{name} must be positive, got {quantity.text()}")
    return float(quantity.value)


class Solid:
    """A shape with exact mass properties and a mesh."""

    kind = "solid"

    #: How many segments a full revolution is drawn with. Enough that a
    #: 100 mm cylinder is smooth on screen and cheap enough to section.
    segments = 32

    def volume(self) -> Quantity:  # pragma: no cover - overridden
        raise NotImplementedError

    def surface_area(self) -> Quantity:  # pragma: no cover - overridden
        raise NotImplementedError

    def mesh(self) -> Mesh:  # pragma: no cover - overridden
        raise NotImplementedError

    def parameters(self) -> dict[str, Quantity]:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- derived ---------------------------------------------------------
    def mass(self, density: Quantity) -> Quantity:
        return self.volume() * density

    def bounding_box(self) -> tuple[Quantity, Quantity, Quantity]:
        low, high = self.mesh().bounds()
        span = high - low
        return (Q(float(span[0]), "m"), Q(float(span[1]), "m"), Q(float(span[2]), "m"))

    def centroid(self) -> tuple[float, float, float]:
        """Centroid in local coordinates; every solid here is built centred."""
        return (0.0, 0.0, 0.0)

    def inertia(self, density: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        """Principal moments about the centroid, in ascending order.

        Shapes with a closed form use it, because a 32-sided polygon is about
        1% light on a circle's second moment and a rotating part sized off
        that error is sized wrong. Everything else is integrated over the
        triangles by the divergence theorem, which is exact for the mesh it
        is given.
        """
        analytic = self.analytic_inertia()
        if analytic is not None:
            rho = float(density.value) * float(self.volume().value)
            return tuple(  # type: ignore[return-value]
                Q(value * rho, "kg m^2") for value in sorted(analytic)
            )
        mesh = self.mesh()
        moments = _inertia_tensor(mesh.vertices, mesh.faces)
        rho = float(density.value)
        return (
            Q(moments[0] * rho, "kg m^2"),
            Q(moments[1] * rho, "kg m^2"),
            Q(moments[2] * rho, "kg m^2"),
        )

    def analytic_inertia(self) -> tuple[float, float, float] | None:
        """Principal moments per unit MASS, when the shape has a closed form."""
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": {k: v.to_dict() for k, v in self.parameters().items()},
            "volume": self.volume().to_dict(),
            "surface_area": self.surface_area().to_dict(),
        }

    def describe(self) -> str:  # pragma: no cover - overridden
        return self.kind


def _inertia_tensor(vertices: np.ndarray, faces: np.ndarray) -> tuple[float, float, float]:
    """Principal second moments per unit density for a closed triangle mesh."""
    if len(faces) == 0:
        return (0.0, 0.0, 0.0)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    # Signed volume of each tetrahedron from the origin.
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    total = det.sum() / 6.0
    if abs(total) < 1e-18:
        return (0.0, 0.0, 0.0)
    if total < 0.0:
        # The mesh is wound inside out. Flipping the sign here is cheaper
        # than rewinding every face, and leaving it produced a sphere with
        # negative moments of inertia.
        det = -det
        total = -total
    centre = (det[:, None] * (a + b + c) / 4.0).sum(axis=0) / (6.0 * total)
    products = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            term = (
                2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                + a[:, i] * b[:, j] + b[:, i] * a[:, j]
                + a[:, i] * c[:, j] + c[:, i] * a[:, j]
                + b[:, i] * c[:, j] + c[:, i] * b[:, j]
            )
            products[i, j] = (det * term).sum() / 120.0
    volume = abs(total)
    # Shift the second-moment matrix to the centroid.
    products = products - volume * np.outer(centre, centre)
    trace = np.trace(products)
    tensor = np.eye(3) * trace - products
    eigenvalues = np.linalg.eigvalsh(tensor)
    return (float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2]))


def _ring(radius: float, z: float, count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(count, z)], axis=1)


def _band_faces(lower: int, upper: int, count: int) -> list[tuple[int, int, int]]:
    faces: list[tuple[int, int, int]] = []
    for i in range(count):
        j = (i + 1) % count
        faces.append((lower + i, lower + j, upper + j))
        faces.append((lower + i, upper + j, upper + i))
    return faces


def _fan_faces(centre: int, ring: int, count: int, *, flip: bool) -> list[tuple[int, int, int]]:
    faces: list[tuple[int, int, int]] = []
    for i in range(count):
        j = (i + 1) % count
        faces.append((centre, ring + j, ring + i) if flip else (centre, ring + i, ring + j))
    return faces


def _ring_edges(start: int, count: int) -> list[tuple[int, int]]:
    return [(start + i, start + (i + 1) % count) for i in range(count)]


def _rail_edges(lower: int, upper: int, count: int, *, every: int = 4) -> list[tuple[int, int]]:
    return [(lower + i, upper + i) for i in range(0, count, max(1, every))]


@dataclass(frozen=True, slots=True)
class Box(Solid):
    """A rectangular block."""

    width: float
    height: float
    depth: float
    kind: str = field(default="box", init=False)

    @staticmethod
    def of(width: Any, height: Any, depth: Any) -> Box:
        return Box(_length(width, "width"), _length(height, "height"), _length(depth, "depth"))

    def volume(self) -> Quantity:
        return Q(self.width * self.height * self.depth, "m^3")

    def surface_area(self) -> Quantity:
        w, h, d = self.width, self.height, self.depth
        return Q(2.0 * (w * h + w * d + h * d), "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {
            "width": Q(self.width, "m"),
            "height": Q(self.height, "m"),
            "depth": Q(self.depth, "m"),
        }

    def mesh(self) -> Mesh:
        w, h, d = self.width / 2.0, self.height / 2.0, self.depth / 2.0
        vertices = np.array(
            [
                [-w, -d, -h], [w, -d, -h], [w, d, -h], [-w, d, -h],
                [-w, -d, h], [w, -d, h], [w, d, h], [-w, d, h],
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=int,
        )
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        return Mesh(vertices, faces, edges)

    def analytic_inertia(self) -> tuple[float, float, float]:
        w, h, d = self.width, self.height, self.depth
        return (
            (d * d + h * h) / 12.0,
            (w * w + h * h) / 12.0,
            (w * w + d * d) / 12.0,
        )

    def describe(self) -> str:
        return (
            f"a block {Q(self.width, 'm').text()} wide, "
            f"{Q(self.depth, 'm').text()} deep and {Q(self.height, 'm').text()} tall"
        )


@dataclass(frozen=True, slots=True)
class Plate(Solid):
    """A thin flat panel; a box whose thinness is the point."""

    width: float
    depth: float
    thickness: float
    kind: str = field(default="plate", init=False)

    @staticmethod
    def of(width: Any, depth: Any, thickness: Any) -> Plate:
        return Plate(
            _length(width, "width"), _length(depth, "depth"), _length(thickness, "thickness")
        )

    def volume(self) -> Quantity:
        return Q(self.width * self.depth * self.thickness, "m^3")

    def surface_area(self) -> Quantity:
        return Box(self.width, self.thickness, self.depth).surface_area()

    def parameters(self) -> dict[str, Quantity]:
        return {
            "width": Q(self.width, "m"),
            "depth": Q(self.depth, "m"),
            "thickness": Q(self.thickness, "m"),
        }

    def mesh(self) -> Mesh:
        return Box(self.width, self.thickness, self.depth).mesh()

    def analytic_inertia(self) -> tuple[float, float, float]:
        return Box(self.width, self.thickness, self.depth).analytic_inertia()

    def describe(self) -> str:
        return (
            f"a plate {Q(self.width, 'm').text()} by {Q(self.depth, 'm').text()}, "
            f"{Q(self.thickness, 'm').text()} thick"
        )


@dataclass(frozen=True, slots=True)
class Cylinder(Solid):
    """A solid round bar or disc, axis along z."""

    radius: float
    height: float
    kind: str = field(default="cylinder", init=False)

    @staticmethod
    def of(radius: Any, height: Any) -> Cylinder:
        return Cylinder(_length(radius, "radius"), _length(height, "height"))

    @staticmethod
    def by_diameter(diameter: Any, height: Any) -> Cylinder:
        return Cylinder(_length(diameter, "diameter") / 2.0, _length(height, "height"))

    def volume(self) -> Quantity:
        return Q(math.pi * self.radius**2 * self.height, "m^3")

    def surface_area(self) -> Quantity:
        return Q(
            2.0 * math.pi * self.radius * (self.radius + self.height), "m^2"
        )

    def parameters(self) -> dict[str, Quantity]:
        return {
            "diameter": Q(2.0 * self.radius, "m"),
            "radius": Q(self.radius, "m"),
            "height": Q(self.height, "m"),
        }

    def mesh(self) -> Mesh:
        n = self.segments
        half = self.height / 2.0
        bottom = _ring(self.radius, -half, n)
        top = _ring(self.radius, half, n)
        centres = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]], dtype=float)
        vertices = np.vstack([bottom, top, centres])
        faces = _band_faces(0, n, n)
        faces += _fan_faces(2 * n, 0, n, flip=True)
        faces += _fan_faces(2 * n + 1, n, n, flip=False)
        edges = tuple(_ring_edges(0, n) + _ring_edges(n, n) + _rail_edges(0, n, n))
        return Mesh(vertices, np.array(faces, dtype=int), edges)

    def analytic_inertia(self) -> tuple[float, float, float]:
        transverse = (3.0 * self.radius**2 + self.height**2) / 12.0
        return (transverse, transverse, self.radius**2 / 2.0)

    def describe(self) -> str:
        return (
            f"a round bar {Q(2 * self.radius, 'm').text()} across "
            f"and {Q(self.height, 'm').text()} long"
        )


@dataclass(frozen=True, slots=True)
class Tube(Solid):
    """A pipe or hollow shell of constant wall thickness, axis along z."""

    outer_radius: float
    wall: float
    height: float
    kind: str = field(default="tube", init=False)

    @staticmethod
    def of(outer_diameter: Any, wall: Any, height: Any) -> Tube:
        outer = _length(outer_diameter, "outer diameter") / 2.0
        thickness = _length(wall, "wall")
        if thickness >= outer:
            raise ValueError("wall thickness must be less than the outer radius")
        return Tube(outer, thickness, _length(height, "height"))

    @property
    def inner_radius(self) -> float:
        return self.outer_radius - self.wall

    def volume(self) -> Quantity:
        return Q(
            math.pi * (self.outer_radius**2 - self.inner_radius**2) * self.height, "m^3"
        )

    def surface_area(self) -> Quantity:
        rings = 2.0 * math.pi * (self.outer_radius**2 - self.inner_radius**2)
        walls = 2.0 * math.pi * (self.outer_radius + self.inner_radius) * self.height
        return Q(rings + walls, "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {
            "outer_diameter": Q(2.0 * self.outer_radius, "m"),
            "inner_diameter": Q(2.0 * self.inner_radius, "m"),
            "wall": Q(self.wall, "m"),
            "height": Q(self.height, "m"),
        }

    def mesh(self) -> Mesh:
        n = self.segments
        half = self.height / 2.0
        outer_bottom = _ring(self.outer_radius, -half, n)
        outer_top = _ring(self.outer_radius, half, n)
        inner_bottom = _ring(self.inner_radius, -half, n)
        inner_top = _ring(self.inner_radius, half, n)
        vertices = np.vstack([outer_bottom, outer_top, inner_bottom, inner_top])
        faces = _band_faces(0, n, n)
        faces += [(c, b, a) for a, b, c in _band_faces(2 * n, 3 * n, n)]
        faces += _band_faces(2 * n, 0, n)
        faces += _band_faces(n, 3 * n, n)
        edges = tuple(
            _ring_edges(0, n)
            + _ring_edges(n, n)
            + _ring_edges(2 * n, n)
            + _ring_edges(3 * n, n)
            + _rail_edges(0, n, n)
        )
        return Mesh(vertices, np.array(faces, dtype=int), edges)

    def analytic_inertia(self) -> tuple[float, float, float]:
        ro, ri = self.outer_radius, self.inner_radius
        axial = (ro * ro + ri * ri) / 2.0
        transverse = (3.0 * (ro * ro + ri * ri) + self.height**2) / 12.0
        return (transverse, transverse, axial)

    def describe(self) -> str:
        return (
            f"a tube {Q(2 * self.outer_radius, 'm').text()} across with a "
            f"{Q(self.wall, 'm').text()} wall, {Q(self.height, 'm').text()} long"
        )


@dataclass(frozen=True, slots=True)
class Sphere(Solid):
    """A ball."""

    radius: float
    kind: str = field(default="sphere", init=False)

    @staticmethod
    def of(radius: Any) -> Sphere:
        return Sphere(_length(radius, "radius"))

    def volume(self) -> Quantity:
        return Q(4.0 / 3.0 * math.pi * self.radius**3, "m^3")

    def surface_area(self) -> Quantity:
        return Q(4.0 * math.pi * self.radius**2, "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {"diameter": Q(2.0 * self.radius, "m"), "radius": Q(self.radius, "m")}

    def mesh(self) -> Mesh:
        return _revolved_mesh(
            [
                (self.radius * math.sin(t), self.radius * math.cos(t))
                for t in np.linspace(0.0, math.pi, self.segments // 2 + 1)
            ],
            self.segments,
        )

    def analytic_inertia(self) -> tuple[float, float, float]:
        value = 0.4 * self.radius**2
        return (value, value, value)

    def describe(self) -> str:
        return f"a ball {Q(2 * self.radius, 'm').text()} across"


@dataclass(frozen=True, slots=True)
class Ellipsoid(Solid):
    """A squashed or stretched ball, useful for hulls and bells."""

    radius_x: float
    radius_y: float
    radius_z: float
    kind: str = field(default="ellipsoid", init=False)

    @staticmethod
    def of(radius_x: Any, radius_y: Any, radius_z: Any) -> Ellipsoid:
        return Ellipsoid(
            _length(radius_x, "radius x"),
            _length(radius_y, "radius y"),
            _length(radius_z, "radius z"),
        )

    def volume(self) -> Quantity:
        return Q(
            4.0 / 3.0 * math.pi * self.radius_x * self.radius_y * self.radius_z, "m^3"
        )

    def surface_area(self) -> Quantity:
        # Knud Thomsen's approximation, within about 1% for any axis ratio.
        p = 1.6075
        a, b, c = self.radius_x, self.radius_y, self.radius_z
        term = ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
        return Q(4.0 * math.pi * term ** (1.0 / p), "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {
            "length_x": Q(2.0 * self.radius_x, "m"),
            "length_y": Q(2.0 * self.radius_y, "m"),
            "length_z": Q(2.0 * self.radius_z, "m"),
        }

    def mesh(self) -> Mesh:
        base = Sphere(1.0).mesh()
        scale = np.array([self.radius_x, self.radius_y, self.radius_z], dtype=float)
        return Mesh(base.vertices * scale, base.faces, base.edges)

    def analytic_inertia(self) -> tuple[float, float, float]:
        a, b, c = self.radius_x, self.radius_y, self.radius_z
        return ((b * b + c * c) / 5.0, (a * a + c * c) / 5.0, (a * a + b * b) / 5.0)

    def describe(self) -> str:
        return (
            f"a domed shell {Q(2 * self.radius_x, 'm').text()} across "
            f"and {Q(2 * self.radius_z, 'm').text()} tall"
        )


@dataclass(frozen=True, slots=True)
class Dome(Solid):
    """A hollow spherical cap of constant wall thickness: a bell or canopy."""

    radius: float
    wall: float
    sweep: float = math.pi / 2.0
    kind: str = field(default="dome", init=False)

    @staticmethod
    def of(radius: Any, wall: Any, sweep_degrees: float = 90.0) -> Dome:
        return Dome(
            _length(radius, "radius"),
            _length(wall, "wall"),
            math.radians(float(sweep_degrees)),
        )

    def volume(self) -> Quantity:
        outer = self.radius
        inner = max(self.radius - self.wall, 1e-9)
        height_outer = outer * (1.0 - math.cos(self.sweep))
        height_inner = inner * (1.0 - math.cos(self.sweep))
        cap_outer = math.pi * height_outer**2 * (outer - height_outer / 3.0)
        cap_inner = math.pi * height_inner**2 * (inner - height_inner / 3.0)
        return Q(cap_outer - cap_inner, "m^3")

    def surface_area(self) -> Quantity:
        outer = 2.0 * math.pi * self.radius**2 * (1.0 - math.cos(self.sweep))
        inner_radius = max(self.radius - self.wall, 1e-9)
        inner = 2.0 * math.pi * inner_radius**2 * (1.0 - math.cos(self.sweep))
        return Q(outer + inner, "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {
            "diameter": Q(2.0 * self.radius, "m"),
            "wall": Q(self.wall, "m"),
            "sweep": Q(math.degrees(self.sweep), "deg"),
        }

    def mesh(self) -> Mesh:
        steps = max(self.segments // 2, 6)
        angles = np.linspace(0.0, self.sweep, steps + 1)
        inner_radius = max(self.radius - self.wall, 1e-9)
        outer = [
            (self.radius * math.sin(t), self.radius * math.cos(t)) for t in angles
        ]
        inner = [
            (inner_radius * math.sin(t), inner_radius * math.cos(t)) for t in reversed(angles)
        ]
        return _revolved_mesh(outer + inner, self.segments, closed=True)

    def describe(self) -> str:
        return (
            f"a domed shell {Q(2 * self.radius, 'm').text()} across "
            f"with a {Q(self.wall, 'm').text()} wall"
        )


@dataclass(frozen=True, slots=True)
class Cone(Solid):
    """A cone standing on its base, axis along z."""

    radius: float
    height: float
    kind: str = field(default="cone", init=False)

    @staticmethod
    def of(radius: Any, height: Any) -> Cone:
        return Cone(_length(radius, "radius"), _length(height, "height"))

    def volume(self) -> Quantity:
        return Q(math.pi * self.radius**2 * self.height / 3.0, "m^3")

    def surface_area(self) -> Quantity:
        slant = math.hypot(self.radius, self.height)
        return Q(math.pi * self.radius * (self.radius + slant), "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {"diameter": Q(2.0 * self.radius, "m"), "height": Q(self.height, "m")}

    def mesh(self) -> Mesh:
        return Frustum(self.radius, 1e-6, self.height).mesh()

    def describe(self) -> str:
        return (
            f"a cone {Q(2 * self.radius, 'm').text()} across the base "
            f"and {Q(self.height, 'm').text()} tall"
        )


@dataclass(frozen=True, slots=True)
class Frustum(Solid):
    """A cone with its tip cut off; a taper or a nozzle."""

    base_radius: float
    top_radius: float
    height: float
    kind: str = field(default="frustum", init=False)

    @staticmethod
    def of(base_diameter: Any, top_diameter: Any, height: Any) -> Frustum:
        return Frustum(
            _length(base_diameter, "base diameter") / 2.0,
            _length(top_diameter, "top diameter") / 2.0,
            _length(height, "height"),
        )

    def volume(self) -> Quantity:
        r, t, h = self.base_radius, self.top_radius, self.height
        return Q(math.pi * h * (r * r + r * t + t * t) / 3.0, "m^3")

    def surface_area(self) -> Quantity:
        r, t, h = self.base_radius, self.top_radius, self.height
        slant = math.hypot(r - t, h)
        return Q(math.pi * (r * r + t * t + (r + t) * slant), "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {
            "base_diameter": Q(2.0 * self.base_radius, "m"),
            "top_diameter": Q(2.0 * self.top_radius, "m"),
            "height": Q(self.height, "m"),
        }

    def mesh(self) -> Mesh:
        n = self.segments
        half = self.height / 2.0
        bottom = _ring(self.base_radius, -half, n)
        top = _ring(self.top_radius, half, n)
        centres = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]], dtype=float)
        vertices = np.vstack([bottom, top, centres])
        faces = _band_faces(0, n, n)
        faces += _fan_faces(2 * n, 0, n, flip=True)
        faces += _fan_faces(2 * n + 1, n, n, flip=False)
        edges = tuple(_ring_edges(0, n) + _ring_edges(n, n) + _rail_edges(0, n, n))
        return Mesh(vertices, np.array(faces, dtype=int), edges)

    def describe(self) -> str:
        return (
            f"a taper from {Q(2 * self.base_radius, 'm').text()} "
            f"to {Q(2 * self.top_radius, 'm').text()} over {Q(self.height, 'm').text()}"
        )


@dataclass(frozen=True, slots=True)
class Torus(Solid):
    """A ring of round section: an O-ring, a coil, a hoop."""

    ring_radius: float
    section_radius: float
    kind: str = field(default="torus", init=False)

    @staticmethod
    def of(ring_diameter: Any, section_diameter: Any) -> Torus:
        return Torus(
            _length(ring_diameter, "ring diameter") / 2.0,
            _length(section_diameter, "section diameter") / 2.0,
        )

    def volume(self) -> Quantity:
        return Q(
            2.0 * math.pi**2 * self.ring_radius * self.section_radius**2, "m^3"
        )

    def surface_area(self) -> Quantity:
        return Q(4.0 * math.pi**2 * self.ring_radius * self.section_radius, "m^2")

    def parameters(self) -> dict[str, Quantity]:
        return {
            "ring_diameter": Q(2.0 * self.ring_radius, "m"),
            "section_diameter": Q(2.0 * self.section_radius, "m"),
        }

    def mesh(self) -> Mesh:
        major = self.segments
        minor = max(self.segments // 2, 8)
        profile = [
            (
                self.ring_radius + self.section_radius * math.cos(t),
                self.section_radius * math.sin(t),
            )
            for t in np.linspace(0.0, 2.0 * math.pi, minor, endpoint=False)
        ]
        return _revolved_mesh(profile, major, closed=True)

    def analytic_inertia(self) -> tuple[float, float, float]:
        R, r = self.ring_radius, self.section_radius
        axial = R * R + 0.75 * r * r
        transverse = 0.5 * R * R + 0.625 * r * r
        return (transverse, transverse, axial)

    def describe(self) -> str:
        return (
            f"a ring {Q(2 * self.ring_radius, 'm').text()} across, "
            f"made of {Q(2 * self.section_radius, 'm').text()} round section"
        )


@dataclass(frozen=True, slots=True)
class Capsule(Solid):
    """A cylinder with hemispherical ends: a pressure hull, a tank, a limb."""

    radius: float
    body_length: float
    kind: str = field(default="capsule", init=False)

    @staticmethod
    def of(diameter: Any, body_length: Any) -> Capsule:
        return Capsule(_length(diameter, "diameter") / 2.0, _length(body_length, "body length"))

    def volume(self) -> Quantity:
        cylinder = math.pi * self.radius**2 * self.body_length
        caps = 4.0 / 3.0 * math.pi * self.radius**3
        return Q(cylinder + caps, "m^3")

    def surface_area(self) -> Quantity:
        return Q(
            2.0 * math.pi * self.radius * self.body_length + 4.0 * math.pi * self.radius**2,
            "m^2",
        )

    def parameters(self) -> dict[str, Quantity]:
        return {
            "diameter": Q(2.0 * self.radius, "m"),
            "body_length": Q(self.body_length, "m"),
            "overall_length": Q(self.body_length + 2.0 * self.radius, "m"),
        }

    def mesh(self) -> Mesh:
        half = self.body_length / 2.0
        steps = max(self.segments // 4, 4)
        top = [
            (
                self.radius * math.sin(t),
                half + self.radius * math.cos(t),
            )
            for t in np.linspace(0.0, math.pi / 2.0, steps + 1)
        ]
        bottom = [
            (
                self.radius * math.cos(t),
                -half - self.radius * math.sin(t),
            )
            for t in np.linspace(0.0, math.pi / 2.0, steps + 1)
        ]
        profile = list(reversed(top)) + bottom
        return _revolved_mesh(profile, self.segments)

    def analytic_inertia(self) -> tuple[float, float, float]:
        """Cylinder plus two hemispheres, each shifted to the shared centroid."""
        r, h = self.radius, self.body_length
        cylinder_volume = math.pi * r * r * h
        cap_volume = 2.0 / 3.0 * math.pi * r**3
        total = cylinder_volume + 2.0 * cap_volume
        # Fractions of the whole mass.
        fc = cylinder_volume / total
        fh = cap_volume / total
        # A hemisphere's centroid sits 3r/8 from its flat face.
        offset = h / 2.0 + 3.0 * r / 8.0
        # Second moment of a hemisphere about its own centroid.
        hemi_axial = 0.4 * r * r
        hemi_transverse = 0.4 * r * r - (3.0 * r / 8.0) ** 2
        axial = fc * (r * r / 2.0) + 2.0 * fh * hemi_axial
        transverse = fc * ((3.0 * r * r + h * h) / 12.0) + 2.0 * fh * (
            hemi_transverse + offset * offset
        )
        return (transverse, transverse, axial)

    def describe(self) -> str:
        overall = self.body_length + 2.0 * self.radius
        return (
            f"a capsule {Q(2 * self.radius, 'm').text()} across "
            f"and {Q(overall, 'm').text()} long overall"
        )


@dataclass(frozen=True, slots=True)
class Prism(Solid):
    """A shape extruded from a flat outline: a bracket, a rail, a beam."""

    outline: tuple[tuple[float, float], ...]
    height: float
    kind: str = field(default="prism", init=False)

    @staticmethod
    def of(outline: Any, height: Any) -> Prism:
        points = tuple(
            (float(x), float(y))
            for x, y in (
                (p if isinstance(p, (tuple, list)) else (p, p)) for p in outline
            )
        )
        if len(points) < 3:
            raise ValueError("an outline needs at least three points")
        return Prism(points, _length(height, "height"))

    def _area(self) -> float:
        points = self.outline
        total = 0.0
        for i, (x0, y0) in enumerate(points):
            x1, y1 = points[(i + 1) % len(points)]
            total += x0 * y1 - x1 * y0
        return abs(total) / 2.0

    def _perimeter(self) -> float:
        points = self.outline
        return sum(
            math.dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))
        )

    def volume(self) -> Quantity:
        return Q(self._area() * self.height, "m^3")

    def surface_area(self) -> Quantity:
        return Q(2.0 * self._area() + self._perimeter() * self.height, "m^2")

    def parameters(self) -> dict[str, Quantity]:
        xs = [p[0] for p in self.outline]
        ys = [p[1] for p in self.outline]
        return {
            "section_area": Q(self._area(), "m^2"),
            "width": Q(max(xs) - min(xs), "m"),
            "depth": Q(max(ys) - min(ys), "m"),
            "height": Q(self.height, "m"),
        }

    def mesh(self) -> Mesh:
        n = len(self.outline)
        half = self.height / 2.0
        bottom = np.array([[x, y, -half] for x, y in self.outline], dtype=float)
        top = np.array([[x, y, half] for x, y in self.outline], dtype=float)
        centres = np.array(
            [
                [sum(p[0] for p in self.outline) / n, sum(p[1] for p in self.outline) / n, -half],
                [sum(p[0] for p in self.outline) / n, sum(p[1] for p in self.outline) / n, half],
            ],
            dtype=float,
        )
        vertices = np.vstack([bottom, top, centres])
        faces = _band_faces(0, n, n)
        faces += _fan_faces(2 * n, 0, n, flip=True)
        faces += _fan_faces(2 * n + 1, n, n, flip=False)
        edges = tuple(
            _ring_edges(0, n) + _ring_edges(n, n) + [(i, n + i) for i in range(n)]
        )
        return Mesh(vertices, np.array(faces, dtype=int), edges)

    def section_centroid(self) -> tuple[float, float]:
        points = self.outline
        area = 0.0
        cx = 0.0
        cy = 0.0
        for i, (x0, y0) in enumerate(points):
            x1, y1 = points[(i + 1) % len(points)]
            cross = x0 * y1 - x1 * y0
            area += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        if abs(area) < 1e-18:
            return (0.0, 0.0)
        return (cx / (3.0 * area), cy / (3.0 * area))

    def section_moments(self) -> tuple[Quantity, Quantity]:
        """Second moments of the outline about its own centroid.

        These are what a beam-bending calculation needs, and the exact
        polygon formula gives them without meshing anything.
        """
        cx, cy = self.section_centroid()
        points = [(x - cx, y - cy) for x, y in self.outline]
        ixx = 0.0
        iyy = 0.0
        for i, (x0, y0) in enumerate(points):
            x1, y1 = points[(i + 1) % len(points)]
            cross = x0 * y1 - x1 * y0
            ixx += (y0 * y0 + y0 * y1 + y1 * y1) * cross
            iyy += (x0 * x0 + x0 * x1 + x1 * x1) * cross
        return (Q(abs(ixx) / 12.0, "m^4"), Q(abs(iyy) / 12.0, "m^4"))

    def analytic_inertia(self) -> tuple[float, float, float]:
        area = self._area()
        ixx, iyy = self.section_moments()
        rx = float(ixx.value) / area
        ry = float(iyy.value) / area
        h2 = self.height**2 / 12.0
        return (rx + h2, ry + h2, rx + ry)

    def describe(self) -> str:
        return f"an extruded profile {Q(self.height, 'm').text()} long"


def _revolved_mesh(
    profile: list[tuple[float, float]], segments: int, *, closed: bool = False
) -> Mesh:
    """Sweep a (radius, z) profile around the z axis into a closed surface."""
    steps = max(int(segments), 6)
    angles = np.linspace(0.0, 2.0 * math.pi, steps, endpoint=False)
    rows = len(profile)
    vertices = np.zeros((rows * steps, 3), dtype=float)
    for r, (radius, z) in enumerate(profile):
        vertices[r * steps : (r + 1) * steps, 0] = radius * np.cos(angles)
        vertices[r * steps : (r + 1) * steps, 1] = radius * np.sin(angles)
        vertices[r * steps : (r + 1) * steps, 2] = z
    faces: list[tuple[int, int, int]] = []
    last = rows if closed else rows - 1
    for r in range(last):
        lower = (r % rows) * steps
        upper = ((r + 1) % rows) * steps
        faces += _band_faces(lower, upper, steps)
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        edges += _ring_edges(r * steps, steps)
    for i in range(0, steps, max(1, steps // 8)):
        for r in range(last):
            edges.append(((r % rows) * steps + i, ((r + 1) % rows) * steps + i))
    return Mesh(vertices, np.array(faces, dtype=int), tuple(edges))


#: The shapes a design brief may name, and how to build one from a mapping
#: of dimensions. The keys are what the parameters are called in a brief.
SOLID_KINDS: dict[str, tuple[type[Solid], tuple[str, ...]]] = {
    "box": (Box, ("width", "height", "depth")),
    "plate": (Plate, ("width", "depth", "thickness")),
    "cylinder": (Cylinder, ("diameter", "height")),
    "tube": (Tube, ("outer_diameter", "wall", "height")),
    "sphere": (Sphere, ("radius",)),
    "ellipsoid": (Ellipsoid, ("radius_x", "radius_y", "radius_z")),
    "dome": (Dome, ("radius", "wall")),
    "cone": (Cone, ("radius", "height")),
    "frustum": (Frustum, ("base_diameter", "top_diameter", "height")),
    "torus": (Torus, ("ring_diameter", "section_diameter")),
    "capsule": (Capsule, ("diameter", "body_length")),
    "prism": (Prism, ("outline", "height")),
}

#: The parameter names a brief is likely to use for each shape, mapped onto
#: the name the constructor takes. A brief that says a cylinder is 40 mm
#: across should not fail for saying "across" instead of "diameter".
_PARAMETER_ALIASES: dict[str, str] = {
    "d": "diameter",
    "dia": "diameter",
    "across": "diameter",
    "od": "outer_diameter",
    "id": "inner_diameter",
    "outer": "outer_diameter",
    "wall_thickness": "wall",
    "thick": "thickness",
    "t": "thickness",
    "len": "height",
    "length": "height",
    "long": "height",
    "tall": "height",
    "w": "width",
    "wide": "width",
    "deep": "depth",
    "r": "radius",
}


def solid_from_spec(spec: dict[str, Any]) -> Solid:
    """Build a solid from ``{"kind": "tube", "outer_diameter": "0.5 m", ...}``.

    Every dimension is read through :func:`core.engineering.units.Q`, so a
    brief that gives a diameter in inches and a wall in millimetres produces
    a tube in metres and a mass in kilograms.
    """
    kind = str(spec.get("kind") or spec.get("shape") or "").strip().lower()
    if kind not in SOLID_KINDS:
        raise KeyError(
            f"{kind!r} is not a shape this builds; the shapes are "
            f"{', '.join(sorted(SOLID_KINDS))}"
        )
    cls, names = SOLID_KINDS[kind]
    supplied: dict[str, Any] = {}
    for key, value in spec.items():
        if key in {"kind", "shape"}:
            continue
        supplied[_PARAMETER_ALIASES.get(str(key).lower(), str(key).lower())] = value
    if kind == "cylinder" and "diameter" not in supplied and "radius" in supplied:
        supplied["diameter"] = Q(supplied["radius"]) * 2
    if kind == "tube" and "outer_diameter" not in supplied and "diameter" in supplied:
        supplied["outer_diameter"] = supplied["diameter"]
    if kind == "tube" and "wall" not in supplied and {"outer_diameter", "inner_diameter"} <= supplied.keys():
        supplied["wall"] = (Q(supplied["outer_diameter"]) - Q(supplied["inner_diameter"])) / 2
    missing = [name for name in names if name not in supplied]
    if missing:
        raise KeyError(f"a {kind} needs {', '.join(missing)}")
    arguments = [supplied[name] for name in names]
    if kind == "sphere":
        return Sphere.of(*arguments)
    if kind == "dome":
        sweep = supplied.get("sweep", 90.0)
        sweep_value = float(Q(sweep).to("deg")) if not isinstance(sweep, (int, float)) else float(sweep)
        return Dome.of(arguments[0], arguments[1], sweep_value)
    if kind == "ellipsoid":
        return Ellipsoid.of(*arguments)
    if kind == "prism":
        return Prism.of(*arguments)
    if kind == "cylinder":
        return Cylinder.by_diameter(*arguments)
    if kind == "cone":
        return Cone.of(*arguments)
    return cls.of(*arguments)  # type: ignore[attr-defined]
