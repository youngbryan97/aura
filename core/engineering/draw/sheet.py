"""The whole drawing sheet: views, tables, title block, and the plain reading.

A drawing is not one picture. It is a picture with the evidence around it —
what the thing weighs, what it is made of, what was checked, what failed,
and what it would take to build one. Putting those on the same sheet as the
view is what makes a drawing something a person can act on rather than
something they have to be talked through.

Sheet sizes follow ISO 216 and the title block follows ASME Y14.1: the
identity of the drawing sits bottom right, where every draughtsman since
1930 has looked for it. Everything else is arranged around the view.

Every panel reads from the design and the findings. No panel invents a
number, and a panel with nothing to say does not appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.engineering.draw.canvas import LINE_TYPES, Canvas
from core.engineering.draw.views import (
    Region,
    draw_assembly,
    draw_exploded,
    draw_orthographic,
    draw_section,
)

__all__ = ["SHEET_SIZES", "Sheet", "compose_sheet", "SHEET_KINDS"]

#: ISO 216 sheet sizes in millimetres, landscape.
SHEET_SIZES: dict[str, tuple[float, float]] = {
    "A5": (210.0, 148.0),
    "A4": (297.0, 210.0),
    "A3": (420.0, 297.0),
    "A2": (594.0, 420.0),
    "A1": (841.0, 594.0),
    "A0": (1189.0, 841.0),
}

#: What each kind of sheet puts in the middle.
SHEET_KINDS: dict[str, str] = {
    "assembly": "The thing as it goes together, labelled part by part.",
    "exploded": "The same thing pulled apart in assembly order, with find numbers.",
    "section": "A cut through it, so the inside can be seen.",
    "orthographic": "The three squared-on views, dimensioned.",
    "schematic": "How it is connected, rather than what it looks like.",
}


@dataclass(frozen=True, slots=True)
class Sheet:
    """One finished drawing, and what went onto it."""

    svg: str
    kind: str
    size: str
    width: float
    height: float
    title: str
    scale_text: str
    parts_shown: tuple[str, ...]
    findings_quoted: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "size": self.size,
            "width_mm": self.width,
            "height_mm": self.height,
            "title": self.title,
            "scale": self.scale_text,
            "parts_shown": list(self.parts_shown),
            "findings_quoted": list(self.findings_quoted),
            "warnings": list(self.warnings),
            "bytes": len(self.svg),
        }


def _share(
    wants: list[tuple[str, float, float]], available: float, gap: float
) -> list[tuple[str, float]]:
    """Divide a column between panels by how much each one has to say.

    Each entry is a name, the height it would like, and the least it can
    usefully take. Everybody gets at least their minimum; what is left is
    shared in proportion to what they asked for, and any surplus goes to
    whichever panel wanted the most, so a short sheet fills rather than
    leaving a band of empty paper.
    """
    if not wants:
        return []
    usable = available - gap * (len(wants) - 1)
    minimum = sum(entry[2] for entry in wants)
    if usable <= minimum:
        scale = usable / minimum if minimum > 0 else 0.0
        return [(name, floor * scale) for name, _want, floor in wants]
    spare = usable - minimum
    demand = sum(max(want - floor, 0.0) for _n, want, floor in wants)
    if demand <= 0:
        share = spare / len(wants)
        return [(name, floor + share) for name, _w, floor in wants]
    heights = [
        (name, floor + spare * max(want - floor, 0.0) / demand)
        for name, want, floor in wants
    ]
    used = sum(height for _n, height in heights)
    if used < usable - 0.5:
        greediest = max(range(len(wants)), key=lambda i: wants[i][1])
        heights[greediest] = (heights[greediest][0], heights[greediest][1] + usable - used)
    return heights


def _verdict_colour(canvas: Canvas, verdict: str) -> str:
    return {
        "pass": canvas.theme.pass_colour,
        "fail": canvas.theme.fail,
        "watch": canvas.theme.warn,
    }.get(verdict, canvas.theme.ink_soft)


def _header(canvas: Canvas, design, findings, region: Region) -> None:
    """The title strip, and the four numbers somebody wants at a glance."""
    canvas.rect(
        (region.x, region.y), region.width, region.height,
        kind="thin", colour=canvas.theme.ink_soft, fill=canvas.theme.paper,
        radius=1.0, layer="frame",
    )
    canvas.text(
        (region.x + 4.0, region.y + region.height * 0.52),
        design.name.upper(),
        size=canvas.text_height * 1.55,
        colour=canvas.theme.accent,
        weight="700",
        letter_spacing=0.6,
        layer="frame",
        baseline="central",
    )
    if design.purpose:
        canvas.text(
            (region.x + 4.0, region.y + region.height - 2.6),
            design.purpose,
            size=canvas.text_height * 0.72,
            colour=canvas.theme.ink_soft,
            layer="frame",
        )

    tiles = _headline_numbers(design, findings)
    if not tiles:
        return
    tile_width = min(38.0, (region.width * 0.55) / max(len(tiles), 1))
    x = region.x + region.width - tile_width * len(tiles) - 4.0
    for label, value, verdict in tiles:
        canvas.text(
            (x, region.y + 5.6),
            label.upper(),
            size=canvas.text_height * 0.62,
            colour=canvas.theme.ink_soft,
            layer="frame",
            letter_spacing=0.4,
        )
        canvas.text(
            (x, region.y + region.height - 3.4),
            value,
            size=canvas.text_height * 1.05,
            colour=_verdict_colour(canvas, verdict) if verdict else canvas.theme.ink,
            weight="600",
            layer="frame",
            mono=True,
        )
        x += tile_width


def _headline_numbers(design, findings) -> list[tuple[str, str, str]]:
    """The handful of figures that belong at the top of the sheet."""
    by_id = {f.id: f for f in findings}
    tiles: list[tuple[str, str, str]] = []
    mass = by_id.get("assurance.mass_growth") or by_id.get("mass.total")
    if mass is not None:
        tiles.append(("Mass", mass.value.text(), mass.verdict))
    power = by_id.get("electrical.total_draw")
    if power is not None:
        tiles.append(("Power", power.value.text(), power.verdict))
    runtime = next((f for f in findings if f.id.startswith("electrical.runtime")), None)
    if runtime is not None:
        tiles.append(("Runtime", runtime.value.text(), runtime.verdict))
    cost = design.total_cost()
    if cost is not None and float(cost.value) > 0:
        tiles.append(("Parts cost", f"{float(cost.value):,.0f}", ""))
    failures = [f for f in findings if f.verdict == "fail"]
    tiles.append((
        "Checks failed",
        str(len(failures)),
        "fail" if failures else "pass",
    ))
    return tiles[:5]


def _compare(finding, requirement) -> str:
    """Does this finding satisfy this requirement, in the requirement's units?"""
    if requirement.target is None:
        return "unverified"
    try:
        actual = float(finding.value.to(requirement.target.unit))
        target = float(requirement.target.to(requirement.target.unit))
    except (ValueError, KeyError, TypeError):
        actual = float(finding.value.value)
        target = float(requirement.target.value)
    comparison = requirement.comparison
    if comparison in {">=", "at least", "min"}:
        return "pass" if actual >= target else "fail"
    if comparison in {"<=", "at most", "max"}:
        return "pass" if actual <= target else "fail"
    if comparison == ">":
        return "pass" if actual > target else "fail"
    if comparison == "<":
        return "pass" if actual < target else "fail"
    return "pass" if abs(actual - target) <= abs(target) * 0.02 else "fail"


def _requirements_panel(canvas: Canvas, design, findings, region: Region) -> None:
    if not design.requirements:
        return
    panel = canvas.bounded_panel(
        (region.x, region.y), region.width, region.height,
        "Requirements", subtitle="What it has to do, and whether it does it",
    )
    by_id = {f.id: f for f in findings}
    size = canvas.text_height * 0.72
    shown = 0
    for requirement in design.requirements:
        finding = by_id.get(requirement.check)
        verdict = requirement.verdict
        actual = ""
        if finding is not None:
            actual = finding.value.text()
            verdict = _compare(finding, requirement)
        colour = _verdict_colour(canvas, verdict)
        if not panel.room_for(size * 3.0):
            break
        panel.marker(colour)
        panel.row(
            requirement.id, actual or verdict.upper(),
            size=size * 0.92, colour=canvas.theme.ink_soft,
            right_colour=colour, weight="600", indent=4.4,
        )
        panel.note(
            requirement.plain or requirement.statement,
            size=size * 0.88, colour=canvas.theme.ink, indent=4.4, gap=0.8,
        )
        shown += 1
    panel.truncation_note(len(design.requirements) - shown)


def _findings_panel(canvas: Canvas, findings, region: Region, *, limit: int = 8) -> tuple[str, ...]:
    """The results worth reading first: failures, then warnings, then the rest."""
    ranked = sorted(
        findings,
        key=lambda f: (
            0 if f.verdict == "fail" else 1 if f.verdict == "watch" else 2,
            f.id,
        ),
    )
    if not ranked:
        return ()
    panel = canvas.bounded_panel(
        (region.x, region.y), region.width, region.height,
        "What the checks found", subtitle="Every number computed, not written",
    )
    size = canvas.text_height * 0.72
    quoted: list[str] = []
    for finding in ranked[:limit]:
        if not panel.room_for(size * 3.2):
            break
        colour = _verdict_colour(canvas, finding.verdict)
        panel.row(
            finding.name, finding.value.text(),
            size=size, colour=canvas.theme.ink, right_colour=colour, weight="600",
        )
        panel.note(finding.plain, size=size * 0.86, colour=canvas.theme.ink_soft, gap=0.2)
        if finding.advice and panel.remaining > size * 2:
            panel.note(
                f"To fix: {finding.advice}", size=size * 0.84,
                colour=canvas.theme.warn, gap=0.8,
            )
        else:
            panel.gap(size * 0.5)
        quoted.append(finding.id)
    panel.truncation_note(len(ranked) - len(quoted))
    return tuple(quoted)


def _bom_panel(canvas: Canvas, design, region: Region) -> None:
    """The parts list, keyed to the balloons on the view."""
    panel = canvas.bounded_panel(
        (region.x, region.y), region.width, region.height,
        "Parts list", subtitle="Numbered to match the drawing",
    )
    size = canvas.text_height * 0.68
    columns = (
        (region.x + 3.0, "NO", "start"),
        (region.x + 9.0, "PART", "start"),
        (region.x + region.width * 0.60, "QTY", "end"),
        (region.x + region.width * 0.79, "MASS", "end"),
        (region.x + region.width - 3.0, "SOURCE", "end"),
    )
    for x, label, anchor in columns:
        canvas.text((x, panel.cursor), label, size=size * 0.85, anchor=anchor,
                    colour=canvas.theme.ink_soft, layer="tables", letter_spacing=0.3)
    panel.gap(size * 0.7)
    panel.rule()
    shown = 0
    for part in design.parts:
        if not panel.room_for(size * 1.6):
            break
        mass = part.mass()
        y = panel.cursor
        canvas.text((region.x + 3.0, y), str(part.balloon), size=size,
                    colour=canvas.theme.ink_soft, layer="tables", mono=True)
        for line in canvas.wrap_lines(part.lay_name or part.name,
                                      region.width * 0.48, size)[:1]:
            canvas.text((region.x + 9.0, y), line, size=size,
                        colour=canvas.theme.ink, layer="tables")
        canvas.text((region.x + region.width * 0.60, y), str(part.quantity),
                    size=size, anchor="end", colour=canvas.theme.ink_soft,
                    layer="tables", mono=True)
        canvas.text((region.x + region.width * 0.79, y),
                    mass.text() if mass else "-", size=size, anchor="end",
                    colour=canvas.theme.ink_soft, layer="tables", mono=True)
        canvas.text((region.x + region.width - 3.0, y),
                    part.sourcing.method.replace("_", " "), size=size * 0.88,
                    anchor="end", colour=canvas.theme.ink_soft, layer="tables")
        panel.gap(size * 1.45)
        shown += 1
    panel.truncation_note(len(design.parts) - shown)


def _how_it_works_panel(canvas: Canvas, design, region: Region, narrative: str) -> None:
    if not narrative:
        return
    panel = canvas.bounded_panel(
        (region.x, region.y), region.width, region.height,
        "How it works", subtitle="Traced through the connections, in order",
    )
    panel.note(narrative, size=canvas.text_height * 0.7, colour=canvas.theme.ink)


def _legend(canvas: Canvas, region: Region) -> None:
    """A key to the line types the drawing used, for a reader new to them."""
    panel = canvas.bounded_panel(
        (region.x, region.y), region.width, region.height, "Key",
    )
    size = canvas.text_height * 0.64
    for key in ("visible", "hidden", "centre", "trail"):
        style = LINE_TYPES[key]
        if not panel.room_for(size * 1.6):
            break
        canvas.line((region.x + 3.0, panel.cursor - size * 0.3),
                    (region.x + 14.0, panel.cursor - size * 0.3),
                    kind=key, layer="tables")
        canvas.text((region.x + 16.5, panel.cursor), style.meaning, size=size,
                    colour=canvas.theme.ink_soft, layer="tables")
        panel.gap(size * 1.55)


def _title_block(canvas: Canvas, design, region: Region, scale_text: str) -> None:
    """Bottom right, ASME Y14.1: who, what, when, which revision, what scale."""
    canvas.rect((region.x, region.y), region.width, region.height,
                kind="visible", colour=canvas.theme.ink_soft,
                fill=canvas.theme.paper, layer="frame")
    rows = [
        ("TITLE", design.name),
        ("DRAWN BY", design.author),
        ("STANDARD", design.standard),
        ("MATERIAL", _dominant_material(design)),
    ]
    cells = (
        ("SCALE", scale_text),
        ("REV", design.revision),
        ("ID", design.fingerprint()[:8]),
    )
    # Size the type to the block rather than the block to the type: a title
    # block is the one part of a sheet that must never overflow, because it
    # is what identifies the drawing.
    usable = region.height - 3.0
    while rows:
        size = min(canvas.text_height * 0.62, usable / (len(rows) * 1.85 + 3.4))
        if size >= canvas.text_height * 0.34:
            break
        rows.pop()
    else:
        size = canvas.text_height * 0.4
    y = region.y + size * 1.9
    for label, value in rows:
        canvas.text((region.x + 2.5, y), label, size=size * 0.85,
                    colour=canvas.theme.ink_soft, layer="frame", letter_spacing=0.3)
        for line in canvas.wrap_lines(str(value), region.width * 0.62, size)[:1]:
            canvas.text((region.x + region.width * 0.36, y), line, size=size,
                        colour=canvas.theme.ink, layer="frame", weight="600")
        y += size * 1.85
    canvas.line((region.x, y - size * 1.2), (region.x + region.width, y - size * 1.2),
                kind="thin", colour=canvas.theme.ink_soft, layer="frame", opacity=0.6)
    x = region.x + 2.5
    cell_width = (region.width - 5.0) / len(cells)
    for label, value in cells:
        canvas.text((x, y + size * 0.4), label, size=size * 0.85,
                    colour=canvas.theme.ink_soft, layer="frame", letter_spacing=0.3)
        canvas.text((x, y + size * 1.95), str(value), size=size * 1.05,
                    colour=canvas.theme.ink, layer="frame", mono=True, weight="600")
        x += cell_width


def _dominant_material(design) -> str:
    counts: dict[str, float] = {}
    for part in design.parts:
        if part.material is None:
            continue
        mass = part.mass()
        counts[part.material.name] = counts.get(part.material.name, 0.0) + (
            float(mass.value) if mass else 1.0
        )
    if not counts:
        return "as noted"
    return max(counts.items(), key=lambda pair: pair[1])[0]


def compose_sheet(
    design,
    findings: tuple = (),
    *,
    kind: str = "assembly",
    size: str = "A3",
    theme: str = "drafting",
    view: str = "iso",
    narrative: str = "",
    section_axis: str = "y",
    schematic_drawer: Any = None,
) -> Sheet:
    """Lay out one finished drawing sheet and return the SVG.

    ``schematic_drawer`` is called as ``drawer(canvas, design, region)`` when
    the sheet kind is a schematic, which keeps the symbol libraries out of
    this module's imports.
    """
    width, height = SHEET_SIZES.get(size.upper(), SHEET_SIZES["A3"])
    canvas = Canvas(width, height, theme=theme, base_width=0.42,
                    text_height=max(2.4, width / 105.0))
    canvas.layer("grid")
    canvas.layer("frame")
    canvas.layer("fill")
    canvas.layer("hatch")
    canvas.layer("section")
    canvas.layer("trails")
    canvas.layer("edges")
    canvas.layer("dimensions")
    canvas.layer("callouts")
    canvas.layer("tables")
    canvas.layer("annotation")
    canvas.layer("text")
    canvas.grid(10.0, opacity=0.4)

    margin = width * 0.014
    header_h = height * 0.085
    footer_h = height * 0.005
    column_w = width * 0.215

    _header(canvas, design, findings, Region(margin, margin, width - 2 * margin, header_h))

    body_y = margin + header_h + margin * 0.6
    body_h = height - body_y - margin
    left = Region(margin, body_y, column_w, body_h)
    right = Region(width - margin - column_w, body_y, column_w, body_h)
    centre = Region(
        left.x + left.width + margin * 0.8,
        body_y,
        width - 2 * margin - 2 * column_w - margin * 1.6,
        body_h,
    )

    # Panels are sized to what they hold rather than to a fixed fraction,
    # so a design with three parts does not leave a third of the sheet
    # empty and one with forty does not truncate at the same place.
    gap = margin * 0.6
    line = canvas.text_height * 1.0

    left_wants: list[tuple[str, float, float]] = []
    if design.requirements:
        left_wants.append(
            ("requirements", 12.0 + len(design.requirements) * line * 3.2, 26.0)
        )
    ranked_findings = sorted(
        findings,
        key=lambda f: (0 if f.verdict == "fail" else 1 if f.verdict == "watch" else 2, f.id),
    )
    left_wants.append(
        ("findings", 12.0 + min(len(ranked_findings), 8) * line * 4.0, 40.0)
    )
    left_heights = _share(left_wants, body_h, gap)

    y = left.y
    for name, panel_h in left_heights:
        if name == "requirements":
            _requirements_panel(canvas, design, findings,
                                Region(left.x, y, left.width, panel_h))
        else:
            quoted = _findings_panel(
                canvas, findings, Region(left.x, y, left.width, panel_h)
            )
        y += panel_h + gap

    right_wants = [
        ("bom", 16.0 + len(design.parts) * line * 1.5, 26.0),
        ("narrative", 14.0 + len(canvas.wrap_lines(
            narrative, right.width - 6.0, canvas.text_height * 0.7)) * canvas.text_height * 0.95, 22.0),
        ("legend", 12.0 + 4 * line * 1.1, 20.0),
        ("title", 34.0, 26.0),
    ]
    if not narrative:
        right_wants = [entry for entry in right_wants if entry[0] != "narrative"]
    right_heights = _share(right_wants, body_h, gap)

    y = right.y
    title_region = None
    title_h = 0.0
    for name, panel_h in right_heights:
        panel_region = Region(right.x, y, right.width, panel_h)
        if name == "bom":
            _bom_panel(canvas, design, panel_region)
        elif name == "narrative":
            _how_it_works_panel(canvas, design, panel_region, narrative)
        elif name == "legend":
            _legend(canvas, panel_region)
        else:
            title_region, title_h = panel_region, panel_h
        y += panel_h + gap

    # The middle: the view itself.
    canvas.rect((centre.x, centre.y), centre.width, centre.height,
                kind="thin", colour=canvas.theme.ink_soft, fill=canvas.theme.paper,
                radius=1.0, layer="frame")
    view_region = centre.inset(2.5)
    warnings: list[str] = []
    if kind == "exploded":
        drawn = draw_exploded(canvas, design, view_region, findings=findings, view=view)
    elif kind == "section":
        drawn = draw_section(canvas, design, view_region, axis=section_axis, findings=findings)
    elif kind == "orthographic":
        drawn = draw_orthographic(canvas, design, view_region)
    elif kind == "schematic" and schematic_drawer is not None:
        drawn = schematic_drawer(canvas, design, view_region)
    else:
        drawn = draw_assembly(canvas, design, view_region, findings=findings, view=view)
        kind = "assembly"

    canvas.text(
        (centre.x + 3.0, centre.y + 4.6),
        f"{kind.upper()} — {SHEET_KINDS.get(kind, '')}",
        size=canvas.text_height * 0.66,
        colour=canvas.theme.ink_soft,
        layer="annotation",
        letter_spacing=0.3,
    )

    if title_region is not None and title_h > 8:
        _title_block(canvas, design, title_region, drawn.scale_text())

    failures = [f.id for f in findings if f.verdict == "fail"]
    if failures:
        warnings.append(
            f"{len(failures)} checks failed on this design; see the panel on the left."
        )
    return Sheet(
        svg=canvas.to_svg(title=f"{design.name} — {kind}", description=design.purpose),
        kind=kind,
        size=size.upper(),
        width=width,
        height=height,
        title=design.name,
        scale_text=drawn.scale_text(),
        parts_shown=drawn.parts_drawn,
        findings_quoted=quoted,
        warnings=tuple(warnings),
    )
