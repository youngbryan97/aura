"""Laying out a schematic: what connects to what, rather than what it looks like.

A schematic answers a different question from a picture of the object, and
it answers it better. Where the power comes from, what it passes through,
what it ends up doing — those are graph facts, and a graph drawn well is
read in seconds while the same information written as prose is read in
minutes and remembered for none of them.

The layout follows the convention every discipline shares: sources on the
left, what they feed to the right of them, sinks at the far right. Depth in
the graph sets the column, so the drawing's left-to-right order is the order
energy actually travels. Wires run in right angles through channels between
the columns, because a diagonal wire on a schematic reads as a mistake.

Line style and colour come from the domain, so an electrical run and a
coolant line cannot be confused even in one ink.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from core.engineering.domains import domain as get_domain
from core.engineering.draw.canvas import Canvas, Point
from core.engineering.draw.symbols import Symbol, draw_symbol, symbol_for
from core.engineering.draw.views import Region

__all__ = ["draw_schematic", "SchematicResult", "schematic_drawer"]

#: Which drafting line type each domain's connections are drawn with, so a
#: reader can tell a signal from a coolant line with the colour turned off.
_DOMAIN_LINE: dict[str, str] = {
    "electrical": "visible",
    "signal": "hidden",
    "data": "hidden",
    "fluid": "flow",
    "hydraulic": "flow",
    "pneumatic": "electric",
    "thermal": "centre",
    "chemical": "flow",
    "biological": "visible",
    "structural": "visible",
    "mechanical_linear": "visible",
    "mechanical_rotary": "visible",
    "magnetic": "phantom",
    "optical": "hidden",
    "acoustic": "centre",
}


@dataclass(frozen=True, slots=True)
class SchematicResult:
    """What the schematic drew, so a caller can say what is on it."""

    scale: float
    parts_drawn: tuple[str, ...]
    callouts: tuple[str, ...]
    view_key: str
    bounds_mm: tuple[float, float, float, float]
    symbols_used: tuple[str, ...] = ()
    standards_used: tuple[str, ...] = ()
    links_drawn: int = 0

    def scale_text(self) -> str:
        return "not to scale"


def _depths(design) -> dict[str, int]:
    """How far each part is from a source, along the connection graph."""
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    ids = {p.id for p in design.parts}
    for link in design.connections:
        a = link.source.split(".")[0]
        b = link.target.split(".")[0]
        if a in ids and b in ids and a != b:
            outgoing[a].append(b)
            incoming[b].append(a)

    sources = [
        part.id
        for part in design.parts
        if not incoming[part.id]
        or any(k in part.ratings for k in ("supply_power", "capacity", "output_power"))
    ]
    if not sources:
        sources = [design.parts[0].id] if design.parts else []
    depth: dict[str, int] = {pid: 0 for pid in sources}
    frontier = list(sources)
    guard = 0
    while frontier and guard < 4000:
        guard += 1
        current = frontier.pop(0)
        for nxt in outgoing[current]:
            candidate = depth[current] + 1
            if candidate > depth.get(nxt, -1):
                depth[nxt] = candidate
                frontier.append(nxt)
    for part in design.parts:
        depth.setdefault(part.id, max(depth.values(), default=0) + 1)
    return depth


def _symbol_options(part, symbol: Symbol) -> dict[str, Any]:
    """Per-symbol drawing options taken from what the part declares."""
    options: dict[str, Any] = {}
    if symbol.key in {"block", "macromolecule", "controller", "compartment", "gain"}:
        options["label"] = part.lay_name or part.name
    if symbol.key == "instrument":
        options["tag"] = (part.designator or "PI")[:4]
        options["loop"] = str(part.balloon or 101)
    if symbol.key == "motor":
        options["letter"] = "M"
    if symbol.key == "sensor":
        options["letter"] = (part.designator or "S")[:1].upper()
    if symbol.key == "valve" and "control" in part.tags:
        options["kind"] = "control"
    if symbol.key == "capacitor" and "polarised" in part.tags:
        options["polarised"] = True
    if symbol.key == "battery":
        options["cells"] = 2
    return options


def _pin_for(
    pins: dict[str, Point], port_name: str, symbol: Symbol, *, outgoing: bool
) -> Point:
    """Which drawn pin a named port should attach to."""
    lowered = str(port_name or "").lower()
    for name, point in pins.items():
        if name.lower() == lowered:
            return point
    wanted = "output" if outgoing else "input"
    for pin in symbol.pins:
        if pin.role in {wanted, "source" if outgoing else "sink"} and pin.name in pins:
            return pins[pin.name]
    if pins:
        # Rightmost for a departure, leftmost for an arrival.
        ordered = sorted(pins.items(), key=lambda item: item[1][0])
        return ordered[-1][1] if outgoing else ordered[0][1]
    return (0.0, 0.0)


def _route(
    canvas: Canvas,
    start: Point,
    end: Point,
    channel_x: float,
    *,
    kind: str,
    colour: str,
    layer: str,
) -> list[Point]:
    """A right-angled path from one pin to another through a channel.

    Three segments when the two pins are at different heights and one when
    they are not, which is what a hand-drawn schematic does and what keeps
    the drawing legible when there are twenty of them.
    """
    if abs(start[1] - end[1]) < 0.4:
        path = [start, end]
    else:
        path = [start, (channel_x, start[1]), (channel_x, end[1]), end]
    canvas.polyline(path, kind=kind, colour=colour, layer=layer, marker_end="arrow")
    return path


def _junction_dots(canvas: Canvas, paths: list[list[Point]], colour: str, layer: str) -> int:
    """A filled dot wherever wires meet, so a crossing is not read as a join.

    This is the single most important convention on an electrical schematic
    and the one most often left out. A crossing with no dot passes over; a
    dot means connected.
    """
    counts: dict[tuple[float, float], int] = defaultdict(int)
    for path in paths:
        for point in (path[0], path[-1]):
            counts[(round(point[0], 1), round(point[1], 1))] += 1
    drawn = 0
    for (x, y), count in counts.items():
        if count >= 2:
            canvas.circle((x, y), canvas.base_width * 2.0, kind="thin",
                          colour=colour, fill=colour, layer=layer)
            drawn += 1
    return drawn


def draw_schematic(
    canvas: Canvas,
    design,
    region: Region,
    *,
    findings: tuple = (),
    show_values: bool = True,
    show_legend: bool = True,
    layer_prefix: str = "",
) -> SchematicResult:
    """Lay the design out as a connection diagram and wire it up."""
    if not design.parts:
        return SchematicResult(1.0, (), (), "schematic",
                               (region.x, region.y, region.width, region.height))

    depth = _depths(design)
    columns: dict[int, list] = defaultdict(list)
    for part in design.parts:
        columns[depth[part.id]].append(part)
    column_keys = sorted(columns)

    legend_height = canvas.text_height * 3.2 if show_legend else 0.0
    body = Region(region.x, region.y, region.width, region.height - legend_height)
    column_width = body.width / max(len(column_keys), 1)
    tallest = max(len(entries) for entries in columns.values())
    size = min(column_width * 0.20, body.height / (tallest * 3.4), 9.0)
    size = max(size, 2.4)

    placed: dict[str, tuple[Symbol, dict[str, Point], Point]] = {}
    symbols_used: set[str] = set()
    standards: set[str] = set()

    for index, key in enumerate(column_keys):
        entries = columns[key]
        cx = body.x + column_width * (index + 0.5)
        span = body.height / (len(entries) + 1)
        for row, part in enumerate(entries, start=1):
            cy = body.y + span * row
            domain = part.ports[0].domain if part.ports else design.discipline
            symbol = symbol_for(
                f"{part.name} {part.lay_name}",
                domain=domain,
                tags=tuple(part.tags),
                description=part.function,
            )
            colour = canvas.theme.domain_colours.get(symbol.domain, canvas.theme.ink)
            pins = draw_symbol(
                canvas, symbol, (cx, cy), size, colour=colour,
                layer=f"{layer_prefix}symbols", **_symbol_options(part, symbol),
            )
            placed[part.id] = (symbol, pins, (cx, cy))
            symbols_used.add(symbol.key)
            standards.add(symbol.standard)
            canvas.text(
                (cx, cy + size * 1.55),
                (part.lay_name or part.name).upper(),
                size=canvas.text_height * 0.62,
                anchor="middle",
                colour=canvas.theme.ink,
                layer=f"{layer_prefix}labels",
                weight="600",
            )
            if part.designator:
                canvas.text(
                    (cx, cy - size * 1.35), part.designator,
                    size=canvas.text_height * 0.6, anchor="middle",
                    colour=canvas.theme.ink_soft, layer=f"{layer_prefix}labels", mono=True,
                )
            canvas.text(
                (cx, cy + size * 1.55 + canvas.text_height * 0.9),
                symbol.lay_name,
                size=canvas.text_height * 0.55,
                anchor="middle",
                colour=canvas.theme.ink_soft,
                layer=f"{layer_prefix}labels",
            )

    paths_by_domain: dict[str, list[list[Point]]] = defaultdict(list)
    drawn_links = 0
    for link in design.connections:
        source_id = link.source.split(".")[0]
        target_id = link.target.split(".")[0]
        if source_id not in placed or target_id not in placed:
            continue
        source_symbol, source_pins, source_centre = placed[source_id]
        target_symbol, target_pins, target_centre = placed[target_id]
        start = _pin_for(source_pins, link.source.partition(".")[2], source_symbol,
                         outgoing=True)
        end = _pin_for(target_pins, link.target.partition(".")[2], target_symbol,
                       outgoing=False)
        spec = get_domain(link.domain)
        colour = canvas.theme.domain_colours.get(link.domain, canvas.theme.ink)
        kind = _DOMAIN_LINE.get(link.domain, "visible")
        channel = (source_centre[0] + target_centre[0]) / 2.0
        if abs(source_centre[0] - target_centre[0]) < 1.0:
            channel = source_centre[0] + column_width * 0.32
        path = _route(canvas, start, end, channel, kind=kind, colour=colour,
                      layer=f"{layer_prefix}wires")
        paths_by_domain[link.domain].append(path)
        drawn_links += 1
        if show_values:
            label = link.label or ""
            value = ""
            if link.through is not None and link.across is not None:
                value = f"{link.across.text()} / {link.through.text()}"
            elif link.through is not None:
                value = link.through.text()
            elif link.across is not None:
                value = link.across.text()
            if label or value:
                mid = path[len(path) // 2]
                canvas.text(
                    (mid[0], mid[1] - canvas.text_height * 0.45),
                    f"{label} {value}".strip(),
                    size=canvas.text_height * 0.56,
                    anchor="middle",
                    colour=colour,
                    layer=f"{layer_prefix}labels",
                    mono=True,
                )

    for domain_key, paths in paths_by_domain.items():
        _junction_dots(
            canvas, paths,
            canvas.theme.domain_colours.get(domain_key, canvas.theme.ink),
            f"{layer_prefix}wires",
        )

    if show_legend and paths_by_domain:
        _legend(canvas, design, region, paths_by_domain, standards,
                layer=f"{layer_prefix}annotation")

    return SchematicResult(
        1.0,
        tuple(placed),
        tuple(placed),
        "schematic",
        (region.x, region.y, region.width, region.height),
        symbols_used=tuple(sorted(symbols_used)),
        standards_used=tuple(sorted(standards)),
        links_drawn=drawn_links,
    )


def _legend(
    canvas: Canvas, design, region: Region, paths_by_domain, standards, *, layer: str
) -> None:
    """A key naming each line's domain, what it carries, and the standard used."""
    y = region.y + region.height - canvas.text_height * 2.2
    x = region.x + 2.0
    size = canvas.text_height * 0.6
    for domain_key in sorted(paths_by_domain):
        spec = get_domain(domain_key)
        colour = canvas.theme.domain_colours.get(domain_key, canvas.theme.ink)
        canvas.line((x, y - size * 0.3), (x + 9.0, y - size * 0.3),
                    kind=_DOMAIN_LINE.get(domain_key, "visible"),
                    colour=colour, layer=layer)
        canvas.text((x + 11.0, y), f"{spec.name.lower()} — {spec.carries}",
                    size=size, colour=canvas.theme.ink_soft, layer=layer)
        x += 11.0 + size * len(f"{spec.name} {spec.carries}") * 0.56 + 6.0
        if x > region.x + region.width - 40:
            x = region.x + 2.0
            y += size * 1.6
    if standards:
        canvas.text(
            (region.x + region.width - 2.0, region.y + region.height - canvas.text_height * 0.4),
            "Symbols: " + ", ".join(sorted(standards)),
            size=size * 0.92, anchor="end", colour=canvas.theme.ink_soft, layer=layer,
        )


def schematic_drawer(findings: tuple = ()):
    """A drawer for :func:`core.engineering.draw.sheet.compose_sheet`."""

    def drawer(canvas: Canvas, design, region: Region):
        return draw_schematic(canvas, design, region, findings=findings)

    return drawer
