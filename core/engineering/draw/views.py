"""The pictorial views: assembled, pulled apart, sectioned, and squared on.

Each view takes the design, a region of the sheet, and the analysis findings
that are allowed to be quoted, and returns what it drew. A callout may only
carry a number that came from a finding, so a view cannot state anything the
model did not compute.

Callout placement is the part that decides whether a drawing is readable.
Labels are pushed out to the left and right margins, ordered by the height
of the feature they point at, and separated so no two overlap. That is what
a draughtsman does by eye, and it is worth doing properly because a leader
crossing another leader is how a reader loses their place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.engineering.draw.canvas import Canvas, Point
from core.engineering.draw.project import View, project_mesh, shade, view_named
from core.engineering.units import Q

__all__ = [
    "Region",
    "DrawnView",
    "draw_assembly",
    "draw_exploded",
    "draw_section",
    "draw_orthographic",
    "Callout",
]


@dataclass(frozen=True, slots=True)
class Region:
    """A rectangle of the sheet, in millimetres."""

    x: float
    y: float
    width: float
    height: float

    @property
    def centre(self) -> Point:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def inset(self, margin: float) -> Region:
        return Region(
            self.x + margin,
            self.y + margin,
            max(self.width - 2 * margin, 1.0),
            max(self.height - 2 * margin, 1.0),
        )


@dataclass(frozen=True, slots=True)
class Callout:
    """One label pointing at one feature."""

    target: Point
    label: str
    sublabel: str = ""
    colour: str = ""
    part_id: str = ""


@dataclass(frozen=True, slots=True)
class DrawnView:
    """What a view actually put on the sheet."""

    scale: float
    parts_drawn: tuple[str, ...]
    callouts: tuple[str, ...]
    view_key: str
    bounds_mm: tuple[float, float, float, float]

    def scale_text(self) -> str:
        if self.scale >= 1.0:
            return f"{self.scale:.0f}:1" if self.scale >= 2 else "1:1"
        return f"1:{1.0 / self.scale:.0f}"


def _placed_meshes(design, *, explode: float = 0.0):
    """Every part's mesh in world millimetres, with its explode offset."""
    entries = []
    for part in design.parts:
        if part.solid is None:
            continue
        mesh = part.solid.mesh().transformed(part.placement)
        vertices = mesh.vertices
        if explode:
            offset = np.asarray(part.explode, dtype=float) * explode
            vertices = vertices + offset
        entries.append((part, vertices, mesh.faces, mesh.edges))
    return entries


def _subsystem_colour(canvas: Canvas, design, part) -> str:
    """One colour per subsystem, distinct and stable across every view.

    Taken from the subsystem's position in the design rather than from a
    hash of its name, so two subsystems never collide on the same ink and a
    part keeps its colour between the assembly, the exploded view and the
    schematic.
    """
    subsystem = design.subsystem(part.subsystem) if part.subsystem else None
    if subsystem is not None and subsystem.colour:
        return subsystem.colour
    if not part.subsystem:
        return canvas.theme.ink
    palette = [
        canvas.theme.domain_colours.get(key, canvas.theme.ink)
        for key in ("fluid", "electrical", "chemical", "thermal", "magnetic", "data", "optical")
    ]
    order = [entry.id for entry in design.subsystems]
    index = order.index(part.subsystem) if part.subsystem in order else len(order)
    return palette[index % len(palette)]


def _fit(
    entries, region: Region, view: View, *, margin: float = 0.08
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale and offset that put the whole model inside the region."""
    right, up, _forward = view.basis()
    xs: list[float] = []
    ys: list[float] = []
    for _part, vertices, _faces, _edges in entries:
        if len(vertices) == 0:
            continue
        xs.extend(np.atleast_1d(vertices @ right).tolist())
        ys.extend(np.atleast_1d(vertices @ up).tolist())
    if not xs:
        return (1.0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    low = np.array([min(xs), min(ys)])
    high = np.array([max(xs), max(ys)])
    span = np.maximum(high - low, 1e-9)
    usable = np.array(
        [region.width * (1.0 - 2 * margin), region.height * (1.0 - 2 * margin)]
    )
    scale = float(min(usable / span))
    centre_model = (low + high) / 2.0
    centre_sheet = np.array(region.centre)
    return (scale, centre_model, centre_sheet)


def _to_sheet(
    points: np.ndarray, scale: float, centre_model: np.ndarray, centre_sheet: np.ndarray
) -> np.ndarray:
    """Model-plane coordinates to sheet millimetres, y flipped for SVG."""
    shifted = (points - centre_model) * scale
    return np.stack(
        [centre_sheet[0] + shifted[:, 0], centre_sheet[1] - shifted[:, 1]], axis=1
    )


#: How much of the view's width each label gutter takes. The model is fitted
#: into what is left, so a leader never crosses the thing it points at and a
#: label never runs off the edge of the sheet.
CALLOUT_GUTTER = 0.22


def _place_callouts(
    canvas: Canvas,
    callouts: list[Callout],
    region: Region,
    *,
    text_height: float,
    layer: str = "callouts",
) -> int:
    """Push labels into the side gutters and separate them so none overlap."""
    if not callouts:
        return 0
    gutter = region.width * CALLOUT_GUTTER
    mid_x = region.x + region.width / 2.0
    left = sorted([c for c in callouts if c.target[0] < mid_x], key=lambda c: c.target[1])
    right = sorted([c for c in callouts if c.target[0] >= mid_x], key=lambda c: c.target[1])
    # Keep the two columns even; a drawing with six labels on one side and
    # none on the other wastes half the sheet and crowds the other half.
    while len(left) - len(right) > 1:
        right.insert(0, left.pop())
    while len(right) - len(left) > 1:
        left.append(right.pop(0))
    left.sort(key=lambda c: c.target[1])
    right.sort(key=lambda c: c.target[1])

    drawn = 0
    for side, group in (("left", left), ("right", right)):
        if not group:
            continue
        line_height = text_height * (2.6 if any(c.sublabel for c in group) else 2.0)
        available = region.height - text_height * 3
        step = min(line_height, available / max(len(group), 1))
        block = step * (len(group) - 1)
        wanted = sum(c.target[1] for c in group) / len(group)
        top = region.y + text_height * 2.0
        start = max(top, min(wanted - block / 2.0, region.y + region.height - block - text_height))
        for index, callout in enumerate(group):
            y = start + index * step
            if side == "left":
                landing_tip = region.x + gutter
                elbow_x = landing_tip + text_height * 2.2
                landing = landing_tip - elbow_x
            else:
                landing_tip = region.x + region.width - gutter
                elbow_x = landing_tip - text_height * 2.2
                landing = landing_tip - elbow_x
            canvas.leader(
                callout.target,
                (elbow_x, y),
                landing,
                callout.label,
                sublabel=callout.sublabel,
                colour=callout.colour or canvas.theme.ink,
                layer=layer,
                size=text_height,
            )
            drawn += 1
    return drawn


def _feature_point(
    part, sheet_points: np.ndarray, index_range: tuple[int, int]
) -> Point:
    """A visible attachment point for a callout leader.

    The outermost projected vertex reads best: a leader that lands in the
    middle of a wireframe is ambiguous about which part it means.
    """
    start, end = index_range
    block = sheet_points[start:end]
    if len(block) == 0:
        return (0.0, 0.0)
    centre = block.mean(axis=0)
    distances = np.linalg.norm(block - centre, axis=1)
    outer = block[int(np.argmax(distances))]
    # Pull the anchor a third of the way back toward the centre so the dot
    # sits on the part rather than on its outline.
    anchor = outer + (centre - outer) * 0.3
    return (float(anchor[0]), float(anchor[1]))


def _draw_parts(
    canvas: Canvas,
    design,
    entries,
    view: View,
    region: Region,
    *,
    scale: float,
    centre_model: np.ndarray,
    centre_sheet: np.ndarray,
    hidden_lines: bool,
    wireframe_opacity: float,
    highlight: str = "",
    ghost: tuple[str, ...] = (),
    layer_prefix: str = "",
    xray: bool = False,
) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    """Draw every part back to front, and report where each one landed."""
    right, up, forward = view.basis()
    ordered = sorted(
        entries,
        key=lambda entry: -float(np.mean(entry[1] @ forward)) if len(entry[1]) else 0.0,
    )
    placement: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
    for part, vertices, faces, edges in ordered:
        projected = project_mesh(
            vertices, faces, edges, view, hidden_lines=hidden_lines
        )
        sheet = _to_sheet(projected.points, scale, centre_model, centre_sheet)
        placement[part.id] = (sheet, (0, len(sheet)))
        colour = _subsystem_colour(canvas, design, part)
        faded = part.id in ghost
        opacity = 0.28 if faded else 1.0
        lifted = highlight and part.id == highlight

        # A wash of filled faces, back to front. This is what makes a
        # curved surface read as curved and what hides the parts behind.
        # In x-ray the wash is thin, so a hull shows what is inside it,
        # which is what a cutaway illustration is for; solid is the classic
        # drawing where a part in front covers the one behind.
        fill_layer = f"{layer_prefix}fill"
        wash = 0.14 if xray else 0.96
        for face, light in zip(
            projected.faces, projected.face_light, strict=True
        ):
            polygon = [(float(sheet[i][0]), float(sheet[i][1])) for i in face]
            canvas.polygon(
                polygon,
                fill=shade(
                    colour if xray else canvas.theme.paper,
                    float(light),
                    floor=0.55 if xray else 0.82,
                    ceiling=1.15 if xray else 1.12,
                ),
                layer=fill_layer,
                opacity=opacity * (wash if not faded else wash * 0.35),
            )

        edge_layer = f"{layer_prefix}edges"
        for start, end in projected.hidden_edges:
            canvas.line(
                (float(sheet[start][0]), float(sheet[start][1])),
                (float(sheet[end][0]), float(sheet[end][1])),
                kind="hidden",
                colour=colour,
                layer=edge_layer,
                opacity=opacity * 0.35,
            )
        for start, end in projected.visible_edges:
            canvas.line(
                (float(sheet[start][0]), float(sheet[start][1])),
                (float(sheet[end][0]), float(sheet[end][1])),
                kind="visible" if lifted else "thin",
                colour=canvas.theme.accent if lifted else colour,
                layer=edge_layer,
                opacity=opacity * wireframe_opacity,
                width=0.9 if lifted else None,
            )
    return placement


def _findings_for(findings, part_id: str) -> list:
    return [f for f in findings if f.subject == part_id]


def _callout_for_part(canvas: Canvas, design, part, findings, anchor: Point) -> Callout:
    """One label per part: what it is, and the number that matters about it."""
    headline = ""
    relevant = _findings_for(findings, part.id)
    ranked = sorted(
        relevant,
        key=lambda f: (
            0 if f.verdict == "fail" else 1 if f.verdict == "watch" else 2,
            0 if f.id.startswith(("safety", "buckle", "assurance")) else 1,
        ),
    )
    if ranked:
        # The finding's name already says which part it is about, and the
        # label above says it again. Strip the repetition so the sublabel is
        # the measurement rather than the part name twice.
        title = ranked[0].name
        for tail in (f", {part.name}", f" of {part.name}", f" in {part.name}",
                     f": {part.id}", f" {part.id}"):
            title = title.replace(tail, "")
        headline = f"{title.strip().rstrip(',')} {ranked[0].value.text()}"
    else:
        mass = part.mass()
        if mass is not None:
            headline = mass.text()
    label = (part.lay_name or part.name).upper()
    if part.quantity > 1:
        label = f"{label} x{part.quantity}"
    return Callout(
        target=anchor,
        label=label,
        sublabel=headline,
        colour=_subsystem_colour(canvas, design, part),
        part_id=part.id,
    )


def _scale_bar(canvas: Canvas, region: Region, scale: float, *, layer: str) -> None:
    """A bar with a real length on it, so a printed drawing stays measurable.

    ``scale`` is sheet millimetres per model metre, which is the unit every
    fit in this module works in. Treating it as millimetres per millimetre
    put a bar labelled one millimetre across a third of the sheet.
    """
    if scale <= 0:
        return
    #: Pick a round model length whose drawn size lands near this.
    wanted_mm = region.width * 0.18
    candidates = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
    ]
    best = min(candidates, key=lambda metres: abs(metres * scale - wanted_mm))
    length = best * scale
    x = region.x + region.width - length - 4.0
    y = region.y + region.height - 5.0
    canvas.line((x, y), (x + length, y), kind="visible", colour=canvas.theme.ink_soft,
                layer=layer, width=0.8)
    for tick in (x, x + length / 2.0, x + length):
        canvas.line((tick, y - 1.2), (tick, y + 1.2), kind="thin",
                    colour=canvas.theme.ink_soft, layer=layer)
    canvas.text(
        (x + length / 2.0, y - 2.2),
        Q(best, "m").text(),
        size=canvas.text_height * 0.7,
        anchor="middle",
        colour=canvas.theme.ink_soft,
        layer=layer,
    )


def _orientation_marker(canvas: Canvas, region: Region, view: View, *, layer: str) -> None:
    """A little axis tripod so the reader knows which way they are looking."""
    origin = (region.x + region.width - 12.0, region.y + 12.0)
    right, up, _forward = view.basis()
    axes = (
        (np.array([1.0, 0.0, 0.0]), "X"),
        (np.array([0.0, 1.0, 0.0]), "Y"),
        (np.array([0.0, 0.0, 1.0]), "Z"),
    )
    reach = 7.0
    for vector, name in axes:
        dx = float(vector @ right) * reach
        dy = -float(vector @ up) * reach
        if abs(dx) < 0.15 and abs(dy) < 0.15:
            continue
        end = (origin[0] + dx, origin[1] + dy)
        canvas.line(origin, end, kind="thin", colour=canvas.theme.ink_soft,
                    layer=layer, marker_end="arrow")
        canvas.text(
            (origin[0] + dx * 1.28, origin[1] + dy * 1.28),
            name,
            size=canvas.text_height * 0.62,
            anchor="middle",
            colour=canvas.theme.ink_soft,
            layer=layer,
            baseline="central",
        )


def draw_assembly(
    canvas: Canvas,
    design,
    region: Region,
    *,
    findings: tuple = (),
    view: str | View = "iso",
    hidden_lines: bool = True,
    callouts: bool = True,
    highlight: str = "",
    only: tuple[str, ...] = (),
    scale_bar: bool = True,
    layer_prefix: str = "",
    xray: bool | None = None,
) -> DrawnView:
    """The hero view: the whole thing assembled, labelled where it matters."""
    camera = view if isinstance(view, View) else view_named(view)
    entries = _placed_meshes(design)
    if only:
        entries = [e for e in entries if e[0].id in only]
    if not entries:
        return DrawnView(1.0, (), (), camera.key, (region.x, region.y, region.width, region.height))
    # Leave the side gutters clear for labels, so a leader never has to
    # cross the model to reach its own text.
    inner = (
        Region(
            region.x + region.width * CALLOUT_GUTTER,
            region.y,
            region.width * (1.0 - 2 * CALLOUT_GUTTER),
            region.height,
        )
        if callouts
        else region
    )
    see_through = xray if xray is not None else canvas.theme.key != "drafting"
    scale, centre_model, centre_sheet = _fit(entries, inner, camera)
    placement = _draw_parts(
        canvas, design, entries, camera, inner,
        scale=scale, centre_model=centre_model, centre_sheet=centre_sheet,
        hidden_lines=hidden_lines, wireframe_opacity=0.9, highlight=highlight,
        layer_prefix=layer_prefix, xray=see_through,
    )
    drawn_callouts: list[str] = []
    if callouts:
        entries_by_id = {part.id: part for part, *_rest in entries}
        labels: list[Callout] = []
        for part_id, (sheet, index_range) in placement.items():
            part = entries_by_id.get(part_id)
            if part is None:
                continue
            anchor = _feature_point(part, sheet, index_range)
            labels.append(_callout_for_part(canvas, design, part, findings, anchor))
        _place_callouts(canvas, labels, region, text_height=canvas.text_height * 0.78,
                        layer=f"{layer_prefix}callouts")
        drawn_callouts = [c.part_id for c in labels]
    if scale_bar:
        _scale_bar(canvas, region, scale, layer=f"{layer_prefix}annotation")
    _orientation_marker(canvas, region, camera, layer=f"{layer_prefix}annotation")
    return DrawnView(
        scale,
        tuple(placement),
        tuple(drawn_callouts),
        camera.key,
        (region.x, region.y, region.width, region.height),
    )


def draw_exploded(
    canvas: Canvas,
    design,
    region: Region,
    *,
    findings: tuple = (),
    view: str | View = "iso",
    spread: float = 1.0,
    balloons: bool = True,
    trails: bool = True,
    layer_prefix: str = "",
) -> DrawnView:
    """The same assembly pulled apart along each part's own explode vector.

    Trails show where each part came from and balloons key it to the parts
    list, which is what makes an exploded view a set of instructions rather
    than a decoration.
    """
    camera = view if isinstance(view, View) else view_named(view)
    reach = float(design.characteristic_length().value) * 0.75 * spread
    assembled = _placed_meshes(design)
    exploded = _placed_meshes(design, explode=reach)
    if not exploded:
        return DrawnView(1.0, (), (), camera.key, (region.x, region.y, region.width, region.height))
    scale, centre_model, centre_sheet = _fit(exploded, region, camera)
    right, up, _forward = camera.basis()

    if trails:
        for (_part, home, _f, _e), (_p, away, _f2, _e2) in zip(
            assembled, exploded, strict=True
        ):
            if len(home) == 0:
                continue
            start = np.array([[float(np.mean(home @ right)), float(np.mean(home @ up))]])
            end = np.array([[float(np.mean(away @ right)), float(np.mean(away @ up))]])
            a = _to_sheet(start, scale, centre_model, centre_sheet)[0]
            b = _to_sheet(end, scale, centre_model, centre_sheet)[0]
            if math.dist(a, b) < 1.0:
                continue
            canvas.line(
                (float(a[0]), float(a[1])),
                (float(b[0]), float(b[1])),
                kind="trail",
                colour=canvas.theme.ink_soft,
                layer=f"{layer_prefix}trails",
                opacity=0.65,
            )

    placement = _draw_parts(
        canvas, design, exploded, camera, region,
        scale=scale, centre_model=centre_model, centre_sheet=centre_sheet,
        hidden_lines=True, wireframe_opacity=0.9, layer_prefix=layer_prefix,
        xray=canvas.theme.key != "drafting",
    )
    keyed: list[str] = []
    if balloons:
        by_id = {part.id: part for part, *_r in exploded}
        for part_id, (sheet, index_range) in placement.items():
            part = by_id.get(part_id)
            if part is None or len(sheet) == 0:
                continue
            block = sheet[index_range[0] : index_range[1]]
            top = block[int(np.argmin(block[:, 1]))]
            canvas.balloon(
                (float(top[0]), float(top[1]) - 5.0),
                part.balloon or 0,
                layer=f"{layer_prefix}callouts",
                colour=_subsystem_colour(canvas, design, part),
            )
            canvas.line(
                (float(top[0]), float(top[1]) - 1.8),
                (float(top[0]), float(top[1]) - 3.2),
                kind="thin",
                colour=canvas.theme.ink_soft,
                layer=f"{layer_prefix}callouts",
            )
            keyed.append(part_id)
    _scale_bar(canvas, region, scale, layer=f"{layer_prefix}annotation")
    _orientation_marker(canvas, region, camera, layer=f"{layer_prefix}annotation")
    return DrawnView(
        scale, tuple(placement), tuple(keyed), camera.key,
        (region.x, region.y, region.width, region.height),
    )


def draw_section(
    canvas: Canvas,
    design,
    region: Region,
    *,
    axis: str = "y",
    offset: float = 0.0,
    findings: tuple = (),
    layer_prefix: str = "",
) -> DrawnView:
    """A cut through the model, hatched, so the inside can be seen.

    The cutting plane is axis-aligned. Every triangle crossing it produces a
    segment; the segments are chained into loops and each loop is hatched at
    forty-five degrees, turned ninety for the next part so two adjacent
    parts do not merge into one field of lines.
    """
    index = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower(), 1)
    camera = view_named({0: "right", 1: "front", 2: "top"}[index])
    entries = _placed_meshes(design)
    if not entries:
        return DrawnView(1.0, (), (), camera.key, (region.x, region.y, region.width, region.height))
    scale, centre_model, centre_sheet = _fit(entries, region, camera)
    right, up, _forward = camera.basis()
    drawn: list[str] = []
    for order, (part, vertices, faces, _edges) in enumerate(entries):
        loops = _cut_loops(vertices, faces, index, offset)
        if not loops:
            continue
        colour = _subsystem_colour(canvas, design, part)
        for loop in loops:
            plane = np.stack([loop @ right, loop @ up], axis=1)
            sheet = _to_sheet(plane, scale, centre_model, centre_sheet)
            polygon = [(float(x), float(y)) for x, y in sheet]
            if len(polygon) < 3:
                continue
            canvas.polyline(polygon, kind="visible", colour=colour, close=True,
                            layer=f"{layer_prefix}section", fill=canvas.theme.paper)
            canvas.hatch(
                polygon,
                angle=45.0 if order % 2 == 0 else 135.0,
                spacing=max(region.width / 90.0, 0.9),
                colour=colour,
                layer=f"{layer_prefix}hatch",
                opacity=0.7,
            )
        drawn.append(part.id)
    # The cutting-plane line and its viewing arrows, ISO 128-40.
    y = region.y + region.height - 3.0
    canvas.line((region.x + 4.0, y), (region.x + region.width - 4.0, y),
                kind="cutting", colour=canvas.theme.ink, layer=f"{layer_prefix}annotation")
    canvas.text((region.x + 4.0, y - 2.0), f"SECTION ON {axis.upper()}",
                size=canvas.text_height * 0.7, colour=canvas.theme.ink_soft,
                layer=f"{layer_prefix}annotation")
    _scale_bar(canvas, region, scale, layer=f"{layer_prefix}annotation")
    return DrawnView(
        scale, tuple(drawn), (), camera.key,
        (region.x, region.y, region.width, region.height),
    )


def _cut_loops(
    vertices: np.ndarray, faces: np.ndarray, axis: int, offset: float
) -> list[np.ndarray]:
    """Chain the triangle-plane intersections into closed outlines."""
    if len(faces) == 0:
        return []
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for face in faces:
        points = vertices[face]
        side = points[:, axis] - offset
        crossing = []
        for i in range(3):
            j = (i + 1) % 3
            if (side[i] > 0) == (side[j] > 0):
                continue
            denominator = side[i] - side[j]
            if abs(denominator) < 1e-15:
                continue
            t = side[i] / denominator
            crossing.append(points[i] + (points[j] - points[i]) * t)
        if len(crossing) == 2:
            segments.append((crossing[0], crossing[1]))
    if not segments:
        return []

    #: Two ends within this fraction of the model span are the same point.
    span = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
    tolerance = span * 1e-4
    loops: list[np.ndarray] = []
    remaining = list(segments)
    while remaining:
        start, end = remaining.pop()
        chain = [start, end]
        extended = True
        while extended and remaining:
            extended = False
            for index, (a, b) in enumerate(remaining):
                if np.linalg.norm(a - chain[-1]) < tolerance:
                    chain.append(b)
                elif np.linalg.norm(b - chain[-1]) < tolerance:
                    chain.append(a)
                elif np.linalg.norm(a - chain[0]) < tolerance:
                    chain.insert(0, b)
                elif np.linalg.norm(b - chain[0]) < tolerance:
                    chain.insert(0, a)
                else:
                    continue
                remaining.pop(index)
                extended = True
                break
        if len(chain) >= 3:
            loops.append(np.array(chain))
    return loops


def draw_orthographic(
    canvas: Canvas,
    design,
    region: Region,
    *,
    angle: str = "third",
    dimensions: bool = True,
    layer_prefix: str = "",
) -> DrawnView:
    """The three squared-on views, arranged and dimensioned.

    Third angle puts the top view above the front and the right view to the
    right, which is North American practice. First angle puts them the other
    way round, which is European practice, and the symbol in the corner is
    what tells the two apart.
    """
    entries = _placed_meshes(design)
    if not entries:
        return DrawnView(1.0, (), (), "ortho", (region.x, region.y, region.width, region.height))
    cell_w = region.width / 2.0
    cell_h = region.height / 2.0
    third = str(angle).lower().startswith("third")
    layout = (
        {"front": (0, 1), "top": (0, 0), "right": (1, 1)}
        if third
        else {"front": (0, 0), "top": (0, 1), "right": (1, 0)}
    )
    # One scale for all three, so a feature measures the same in each.
    scales = []
    for key in layout:
        camera = view_named(key)
        scale, _cm, _cs = _fit(entries, Region(0, 0, cell_w, cell_h), camera)
        scales.append(scale)
    scale = min(scales)
    drawn: list[str] = []
    for key, (col, row) in layout.items():
        camera = view_named(key)
        cell = Region(region.x + col * cell_w, region.y + row * cell_h, cell_w, cell_h)
        _s, centre_model, centre_sheet = _fit(entries, cell, camera)
        placement = _draw_parts(
            canvas, design, entries, camera, cell,
            scale=scale, centre_model=centre_model, centre_sheet=centre_sheet,
            hidden_lines=True, wireframe_opacity=1.0,
            layer_prefix=f"{layer_prefix}{key}_",
        )
        drawn.extend(placement)
        canvas.text(
            (cell.x + 3.0, cell.y + cell.height - 2.0),
            camera.name.upper(),
            size=canvas.text_height * 0.7,
            colour=canvas.theme.ink_soft,
            layer=f"{layer_prefix}annotation",
            letter_spacing=0.4,
        )
        if dimensions and placement:
            all_points = np.vstack([sheet for sheet, _r in placement.values()])
            x0, y0 = float(all_points[:, 0].min()), float(all_points[:, 1].min())
            x1, y1 = float(all_points[:, 0].max()), float(all_points[:, 1].max())
            width_m = (x1 - x0) / scale
            height_m = (y1 - y0) / scale
            canvas.dimension((x0, y1), (x1, y1), 5.0, Q(width_m, "m").text(),
                             layer=f"{layer_prefix}dimensions")
            canvas.dimension((x1, y0), (x1, y1), 5.0, Q(height_m, "m").text(),
                             layer=f"{layer_prefix}dimensions", vertical=True)
    _projection_symbol(canvas, region, third, layer=f"{layer_prefix}annotation")
    return DrawnView(
        scale, tuple(dict.fromkeys(drawn)), (), "ortho",
        (region.x, region.y, region.width, region.height),
    )


def _projection_symbol(canvas: Canvas, region: Region, third: bool, *, layer: str) -> None:
    """The truncated cone that says which projection arrangement is in use."""
    x = region.x + region.width - 26.0
    y = region.y + region.height - 12.0
    big, small = 4.2, 2.2
    left_r, right_r = (small, big) if third else (big, small)
    canvas.circle((x, y), left_r, kind="thin", colour=canvas.theme.ink_soft, layer=layer)
    canvas.circle((x + 11.0, y), right_r, kind="thin", colour=canvas.theme.ink_soft, layer=layer)
    canvas.line((x, y - left_r), (x + 11.0, y - right_r), kind="thin",
                colour=canvas.theme.ink_soft, layer=layer)
    canvas.line((x, y + left_r), (x + 11.0, y + right_r), kind="thin",
                colour=canvas.theme.ink_soft, layer=layer)
    canvas.text(
        (x + 5.5, y + big + 4.0),
        f"{'THIRD' if third else 'FIRST'} ANGLE",
        size=canvas.text_height * 0.6,
        anchor="middle",
        colour=canvas.theme.ink_soft,
        layer=layer,
    )
