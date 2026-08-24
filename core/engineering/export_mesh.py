"""Geometry out of here and into whatever the next tool is.

A drawing that cannot leave is half a deliverable. These are the formats a
design actually has to arrive in: STL and 3MF for anything that gets
printed, OBJ for anything that gets rendered, DXF for anything that gets cut
or opened in CAD, and OpenSCAD source so the geometry can be edited as
parameters rather than as a frozen mesh.

3MF is worth preferring over STL where the receiving tool accepts it. STL
has no units, no part names and no colours, so a printer has to be told what
scale the file is in; 3MF carries all three, and every current slicer reads
it.

Every writer takes the same placed meshes, so a design exported to two
formats is the same object twice rather than two nearly-identical objects.
"""

from __future__ import annotations

import math
import struct
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

__all__ = [
    "placed_meshes",
    "stl_binary",
    "stl_text",
    "obj_text",
    "three_mf_bytes",
    "dxf_text",
    "openscad_text",
    "MESH_FORMATS",
]

#: What each geometry format is for, so a download list can say.
MESH_FORMATS: dict[str, tuple[str, str, str]] = {
    "stl": (".stl", "model/stl",
            "Triangles only, no units and no part names. Every printer and CAD tool "
            "reads it. Written in millimetres, which is what slicers assume."),
    "3mf": (".3mf", "model/3mf",
            "The modern replacement for STL: carries real units, part names and "
            "colours. Use this where the receiving tool accepts it."),
    "obj": (".obj", "model/obj",
            "Triangles with named groups per part. What rendering and animation "
            "tools want."),
    "dxf": (".dxf", "image/vnd.dxf",
            "Two-dimensional lines on layers, for laser cutters, plotters and any "
            "CAD package."),
    "scad": (".scad", "text/plain",
            "OpenSCAD source. The geometry as editable parameters rather than as a "
            "frozen mesh, so dimensions can be changed and the model rebuilt."),
}


def placed_meshes(design, *, explode: float = 0.0) -> list[tuple[Any, np.ndarray, np.ndarray]]:
    """Every part's triangles in world metres, with its explode offset."""
    out: list[tuple[Any, np.ndarray, np.ndarray]] = []
    for part in design.parts:
        if part.solid is None:
            continue
        mesh = part.solid.mesh().transformed(part.placement)
        vertices = mesh.vertices
        if explode:
            vertices = vertices + np.asarray(part.explode, dtype=float) * explode
        out.append((part, vertices, mesh.faces))
    return out


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    return normals / lengths[:, None]


def stl_binary(design, *, scale: float = 1000.0) -> bytes:
    """Binary STL in millimetres.

    STL carries no unit, and every slicer assumes millimetres. A model
    written in metres arrives a thousand times too small, which is the most
    common way a printed part comes out wrong.
    """
    entries = placed_meshes(design)
    triangles: list[bytes] = []
    for _part, vertices, faces in entries:
        if len(faces) == 0:
            continue
        scaled = vertices * scale
        normals = _normals(scaled, faces)
        for normal, face in zip(normals, faces):
            triangles.append(
                struct.pack(
                    "<12fH",
                    *normal,
                    *scaled[face[0]],
                    *scaled[face[1]],
                    *scaled[face[2]],
                    0,
                )
            )
    header = f"{design.name[:60]} - millimetres - Aura".encode("ascii", "replace")
    return header.ljust(80, b" ") + struct.pack("<I", len(triangles)) + b"".join(triangles)


def stl_text(design, *, scale: float = 1000.0) -> str:
    """ASCII STL, for when a file has to be readable."""
    lines = [f"solid {design.name.replace(' ', '_')}"]
    for _part, vertices, faces in placed_meshes(design):
        if len(faces) == 0:
            continue
        scaled = vertices * scale
        for normal, face in zip(_normals(scaled, faces), faces):
            lines.append(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}")
            lines.append("    outer loop")
            for index in face:
                point = scaled[index]
                lines.append(f"      vertex {point[0]:.6e} {point[1]:.6e} {point[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
    lines.append(f"endsolid {design.name.replace(' ', '_')}")
    return "\n".join(lines) + "\n"


def obj_text(design, *, scale: float = 1000.0) -> str:
    """Wavefront OBJ with one named group per part."""
    lines = [
        f"# {design.name}",
        f"# {design.purpose}",
        "# millimetres; one group per part",
    ]
    offset = 1
    for part, vertices, faces in placed_meshes(design):
        lines.append(f"g {part.id}")
        lines.append(f"o {part.name}")
        scaled = vertices * scale
        for point in scaled:
            lines.append(f"v {point[0]:.5f} {point[1]:.5f} {point[2]:.5f}")
        for face in faces:
            lines.append(
                f"f {face[0] + offset} {face[1] + offset} {face[2] + offset}"
            )
        offset += len(scaled)
    return "\n".join(lines) + "\n"


def three_mf_bytes(design, *, colours: dict[str, str] | None = None) -> bytes:
    """3MF: the same triangles, with units, part names and colours attached."""
    entries = placed_meshes(design)
    objects: list[str] = []
    items: list[str] = []
    for index, (part, vertices, faces) in enumerate(entries, start=1):
        scaled = vertices * 1000.0
        vertex_xml = "".join(
            f'<vertex x="{p[0]:.4f}" y="{p[1]:.4f}" z="{p[2]:.4f}"/>' for p in scaled
        )
        triangle_xml = "".join(
            f'<triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>' for f in faces
        )
        objects.append(
            f'<object id="{index}" type="model" name="{escape(part.name)}">'
            f"<mesh><vertices>{vertex_xml}</vertices>"
            f"<triangles>{triangle_xml}</triangles></mesh></object>"
        )
        items.append(f'<item objectid="{index}"/>')
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        "<metadata name=\"Title\">" + escape(design.name) + "</metadata>"
        "<metadata name=\"Description\">" + escape(design.purpose) + "</metadata>"
        "<metadata name=\"Designer\">" + escape(design.author) + "</metadata>"
        "<resources>" + "".join(objects) + "</resources>"
        "<build>" + "".join(items) + "</build></model>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", model)
    return buffer.getvalue()


#: DXF layer names, and the AutoCAD colour index each takes. Layer names
#: match the drawing's own so a CAD user sees the same structure.
_DXF_LAYERS: tuple[tuple[str, int], ...] = (
    ("VISIBLE", 7),
    ("HIDDEN", 8),
    ("CENTRE", 4),
    ("DIMENSIONS", 3),
    ("TEXT", 2),
)


def dxf_text(design, *, view: str = "front", scale: float = 1000.0) -> str:
    """A DXF R12 of one projected view, on proper layers.

    R12 is deliberate: it is the version every tool still reads without
    argument, and it needs no object handles or class tables.
    """
    from core.engineering.draw.project import project_mesh, view_named

    camera = view_named(view)
    right, up, _forward = camera.basis()
    lines: list[str] = []

    def pair(code: int, value: Any) -> None:
        lines.append(str(code))
        lines.append(str(value))

    pair(0, "SECTION")
    pair(2, "HEADER")
    pair(9, "$INSUNITS")
    pair(70, 4)  # millimetres
    pair(0, "ENDSEC")

    pair(0, "SECTION")
    pair(2, "TABLES")
    pair(0, "TABLE")
    pair(2, "LAYER")
    pair(70, len(_DXF_LAYERS))
    for name, colour in _DXF_LAYERS:
        pair(0, "LAYER")
        pair(2, name)
        pair(70, 0)
        pair(62, colour)
        pair(6, "CONTINUOUS")
    pair(0, "ENDTAB")
    pair(0, "ENDSEC")

    pair(0, "SECTION")
    pair(2, "ENTITIES")
    for part, vertices, faces in placed_meshes(design):
        mesh = part.solid.mesh()
        projected = project_mesh(vertices, faces, mesh.edges, camera, hidden_lines=True)
        flat = np.stack([vertices @ right, vertices @ up], axis=1) * scale
        for layer, edges in (
            ("VISIBLE", projected.visible_edges),
            ("HIDDEN", projected.hidden_edges),
        ):
            for start, end in edges:
                pair(0, "LINE")
                pair(8, layer)
                pair(10, f"{flat[start][0]:.4f}")
                pair(20, f"{flat[start][1]:.4f}")
                pair(30, "0.0")
                pair(11, f"{flat[end][0]:.4f}")
                pair(21, f"{flat[end][1]:.4f}")
                pair(31, "0.0")
        centre = flat.mean(axis=0)
        pair(0, "TEXT")
        pair(8, "TEXT")
        pair(10, f"{centre[0]:.4f}")
        pair(20, f"{centre[1]:.4f}")
        pair(30, "0.0")
        pair(40, "4.0")
        pair(1, (part.lay_name or part.name)[:60])
    pair(0, "ENDSEC")
    pair(0, "EOF")
    return "\n".join(lines) + "\n"


def openscad_text(design) -> str:
    """OpenSCAD source: the geometry as parameters that can still be changed.

    A mesh is a decision already made. This is the decisions, so a reader can
    open it, change a wall thickness, and rebuild.
    """
    lines = [
        f"// {design.name}",
        f"// {design.purpose}",
        "// Generated from the design model. Dimensions in millimetres.",
        f"// Model fingerprint {design.fingerprint()}",
        "",
        "$fn = 64;",
        "",
    ]
    for part in design.parts:
        if part.solid is None:
            continue
        params = {k: float(v.value) * 1000.0 for k, v in part.solid.parameters().items()}
        name = part.id
        lines.append(f"// {part.name}: {part.function or 'no stated function'}")
        if part.material is not None:
            lines.append(f"//   material: {part.material.name}")
        mass = part.mass()
        if mass is not None:
            lines.append(f"//   mass: {mass.text()}")
        for key, value in params.items():
            lines.append(f"{name}_{key} = {value:.4f};")
        lines.append(f"module {name}() {{")
        lines.append("  " + _scad_body(part, name))
        lines.append("}")
        position = [v * 1000.0 for v in part.placement.position]
        rotation = [math.degrees(v) for v in part.placement.rotation]
        lines.append(
            f"translate([{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]) "
            f"rotate([{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}]) "
            f"{name}();"
        )
        lines.append("")
    return "\n".join(lines)


def _scad_body(part, name: str) -> str:
    """The OpenSCAD call that builds one solid from its own parameters."""
    kind = part.solid.kind
    if kind == "box":
        return (
            f"cube([{name}_width, {name}_depth, {name}_height], center = true);"
        )
    if kind == "plate":
        return f"cube([{name}_width, {name}_depth, {name}_thickness], center = true);"
    if kind == "cylinder":
        return f"cylinder(h = {name}_height, d = {name}_diameter, center = true);"
    if kind == "tube":
        return (
            f"difference() {{ cylinder(h = {name}_height, d = {name}_outer_diameter, "
            f"center = true); cylinder(h = {name}_height + 1, d = {name}_inner_diameter, "
            "center = true); }"
        )
    if kind == "sphere":
        return f"sphere(d = {name}_diameter);"
    if kind == "ellipsoid":
        return (
            f"scale([{name}_length_x / 2, {name}_length_y / 2, {name}_length_z / 2]) "
            "sphere(r = 1);"
        )
    if kind == "dome":
        return (
            f"difference() {{ sphere(d = {name}_diameter); "
            f"sphere(d = {name}_diameter - 2 * {name}_wall); "
            f"translate([0, 0, -{name}_diameter]) "
            f"cube({name}_diameter * 2, center = true); }}"
        )
    if kind == "cone":
        return f"cylinder(h = {name}_height, d1 = {name}_diameter, d2 = 0.01, center = true);"
    if kind == "frustum":
        return (
            f"cylinder(h = {name}_height, d1 = {name}_base_diameter, "
            f"d2 = {name}_top_diameter, center = true);"
        )
    if kind == "torus":
        return (
            f"rotate_extrude() translate([{name}_ring_diameter / 2, 0]) "
            f"circle(d = {name}_section_diameter);"
        )
    if kind == "capsule":
        return (
            f"union() {{ cylinder(h = {name}_body_length, d = {name}_diameter, "
            f"center = true); translate([0, 0, {name}_body_length / 2]) "
            f"sphere(d = {name}_diameter); translate([0, 0, -{name}_body_length / 2]) "
            f"sphere(d = {name}_diameter); }}"
        )
    if kind == "prism":
        outline = ", ".join(
            f"[{x * 1000.0:.3f}, {y * 1000.0:.3f}]" for x, y in part.solid.outline
        )
        return (
            f"linear_extrude(height = {name}_height, center = true) "
            f"polygon(points = [{outline}]);"
        )
    return f"// {kind} has no OpenSCAD equivalent yet"
