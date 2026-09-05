"""The drafting primitives every view is drawn with.

Technical drawing has a hundred-year-old grammar and it is worth using
exactly. A thick continuous line is an edge you can see; a thin dashed one
is an edge behind something; a thin chain line is an axis and not a part.
A reader who knows the grammar reads a drawing in seconds, and a reader who
does not still gets a picture where the important lines are the heavy ones.

Line types follow ISO 128-2 and ASME Y14.2, with the 2:1 thick-to-thin ratio
those standards set. Dimensions follow ISO 129-1 placement rules: extension
lines with a visible gap from the feature, arrows inside where they fit and
outside where they do not, and the figure above the line reading from the
bottom of the sheet. Hatching follows ISO 128-50, forty-five degrees and
turned ninety degrees for the adjacent part so the two do not merge.

Everything is emitted as SVG with no external references, so a drawing is
one file that opens anywhere and prints the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

__all__ = [
    "Canvas",
    "LineType",
    "LINE_TYPES",
    "Theme",
    "THEMES",
    "Point",
    "arrow_marker_defs",
]

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class LineType:
    """One of the line types a technical drawing is allowed to use."""

    key: str
    name: str
    #: Multiplier on the sheet's base line width.
    weight: float
    dash: str
    meaning: str
    standard: str = "ISO 128-2"


LINE_TYPES: dict[str, LineType] = {
    entry.key: entry
    for entry in (
        LineType("visible", "Continuous thick", 1.0, "",
                 "An edge or outline you can see."),
        LineType("thin", "Continuous thin", 0.5, "",
                 "Dimension lines, extension lines, leaders and hatching."),
        LineType("hidden", "Dashed thin", 0.5, "4 2",
                 "An edge that is there but hidden behind something."),
        LineType("centre", "Long chain thin", 0.5, "12 2 2 2",
                 "The axis or centre of a feature. Not part of the object."),
        LineType("cutting", "Long chain thick at ends", 1.0, "12 2 2 2",
                 "Where the object has been cut to show the inside."),
        LineType("phantom", "Long chain thin double-dashed", 0.5, "12 2 2 2 2 2",
                 "Something adjacent, or the same part in another position."),
        LineType("break", "Continuous thin with zigzag", 0.5, "",
                 "The drawing stops here; the part carries on."),
        LineType("flow", "Continuous heavy", 1.6, "",
                 "A process line carrying material.", "ISA-5.1"),
        LineType("pneumatic", "Double-slashed", 0.6, "",
                 "A pneumatic signal line.", "ISA-5.1"),
        LineType("electric", "Dashed", 0.6, "5 3",
                 "An electrical signal line.", "ISA-5.1"),
        LineType("data", "Long dashed with dot", 0.6, "9 3 1 3",
                 "A data or software link.", "ISA-5.1"),
        LineType("capillary", "Crossed", 0.6, "3 3",
                 "A filled capillary tube.", "ISA-5.1"),
        LineType("trail", "Fine dashed", 0.4, "2 2",
                 "Where a part travels when the assembly is pulled apart."),
    )
}


@dataclass(frozen=True, slots=True)
class Theme:
    """A drawing's colours, sized for one background."""

    key: str
    name: str
    background: str
    paper: str
    ink: str
    ink_soft: str
    accent: str
    warn: str
    fail: str
    pass_colour: str
    grid: str
    hatch: str
    #: Colours for domain lines, keyed the way core.engineering.domains is.
    domain_colours: dict[str, str] = field(default_factory=dict)
    font: str = "'Helvetica Neue', Helvetica, Arial, sans-serif"
    mono: str = "'SF Mono', Menlo, Consolas, monospace"


_DOMAIN_INK = {
    "electrical": "#c8452f",
    "thermal": "#d97a1f",
    "fluid": "#2f6fb5",
    "hydraulic": "#2f6fb5",
    "pneumatic": "#4fa3c7",
    "mechanical_linear": "#4a5a4a",
    "mechanical_rotary": "#4a5a4a",
    "structural": "#4a5a4a",
    "magnetic": "#7a4f9c",
    "chemical": "#2f8f5b",
    "optical": "#c9a227",
    "acoustic": "#8a6f9c",
    "signal": "#6b7280",
    "data": "#3f7f8f",
    "biological": "#2f8f5b",
}

_DOMAIN_INK_DARK = {
    "electrical": "#ff7a5e",
    "thermal": "#ffab52",
    "fluid": "#6aa8e8",
    "hydraulic": "#6aa8e8",
    "pneumatic": "#7fd0ee",
    "mechanical_linear": "#b8c4b8",
    "mechanical_rotary": "#b8c4b8",
    "structural": "#b8c4b8",
    "magnetic": "#b98fe0",
    "chemical": "#5fd096",
    "optical": "#f0d060",
    "acoustic": "#c0a8d8",
    "signal": "#9aa4b2",
    "data": "#71c0d0",
    "biological": "#5fd096",
}

THEMES: dict[str, Theme] = {
    "drafting": Theme(
        "drafting", "Drafting paper",
        background="#f4f2ec", paper="#faf9f5", ink="#1f2320", ink_soft="#6b6f68",
        accent="#b3402a", warn="#b3762a", fail="#a32b1e", pass_colour="#2f6b45",
        grid="#d9d6cc", hatch="#8d908a", domain_colours=dict(_DOMAIN_INK),
    ),
    "instrument": Theme(
        "instrument", "Instrument console",
        background="#0e1512", paper="#121a17", ink="#d8e2da", ink_soft="#7f8f86",
        accent="#e0503a", warn="#e0a03a", fail="#e0503a", pass_colour="#5fd096",
        grid="#1e2a25", hatch="#4a5a52", domain_colours=dict(_DOMAIN_INK_DARK),
    ),
    "blueprint": Theme(
        "blueprint", "Blueprint",
        background="#0d2b4a", paper="#0f3157", ink="#e8f1fa", ink_soft="#9dbdd8",
        accent="#ffd166", warn="#ffd166", fail="#ff8a7a", pass_colour="#8ce0b0",
        grid="#1b4570", hatch="#5f8db5", domain_colours=dict(_DOMAIN_INK_DARK),
    ),
}


def _fmt(value: float) -> str:
    """Trim a coordinate so two renders of the same drawing are byte-equal."""
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def arrow_marker_defs(theme: Theme) -> str:
    """Arrowheads and dots, defined once and referenced by every line."""
    return (
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9.6" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,1.6 L10,5 L0,8.4 z" fill="{theme.ink}"/></marker>'
        f'<marker id="arrow-accent" viewBox="0 0 10 10" refX="9.6" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,1.6 L10,5 L0,8.4 z" fill="{theme.accent}"/></marker>'
        f'<marker id="arrow-open" viewBox="0 0 10 10" refX="9.6" refY="5" '
        f'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        f'<path d="M0.5,1.5 L9.5,5 L0.5,8.5" fill="none" stroke="{theme.ink}" '
        f'stroke-width="1.2"/></marker>'
        f'<marker id="dot" viewBox="0 0 6 6" refX="3" refY="3" '
        f'markerWidth="4" markerHeight="4">'
        f'<circle cx="3" cy="3" r="2.4" fill="{theme.ink}"/></marker>'
        f'<marker id="junction" viewBox="0 0 6 6" refX="3" refY="3" '
        f'markerWidth="5" markerHeight="5">'
        f'<circle cx="3" cy="3" r="2.6" fill="{theme.ink}"/></marker>'
    )


class Canvas:
    """An SVG drawing sheet in millimetres, drawn on with drafting lines.

    Coordinates are sheet millimetres with the origin at the top left, which
    is what SVG uses and what a plotter expects. A view puts its own model-to-
    sheet transform in front of that.
    """

    def __init__(
        self,
        width: float,
        height: float,
        *,
        theme: str | Theme = "drafting",
        base_width: float = 0.5,
        text_height: float = 3.5,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.theme = theme if isinstance(theme, Theme) else THEMES.get(str(theme), THEMES["drafting"])
        self.base_width = float(base_width)
        self.text_height = float(text_height)
        self._layers: dict[str, list[str]] = {}
        self._order: list[str] = []
        self._defs: list[str] = [arrow_marker_defs(self.theme)]
        self._clip_count = 0

    # -- layers ----------------------------------------------------------
    def layer(self, name: str) -> None:
        if name not in self._layers:
            self._layers[name] = []
            self._order.append(name)

    def _emit(self, layer: str, markup: str) -> None:
        self.layer(layer)
        self._layers[layer].append(markup)

    def define(self, markup: str) -> None:
        self._defs.append(markup)

    # -- geometry --------------------------------------------------------
    def line(
        self,
        start: Point,
        end: Point,
        *,
        kind: str = "visible",
        colour: str | None = None,
        layer: str = "geometry",
        opacity: float = 1.0,
        marker_start: str = "",
        marker_end: str = "",
        width: float | None = None,
    ) -> None:
        style = LINE_TYPES.get(kind, LINE_TYPES["visible"])
        stroke = colour or (self.theme.ink if style.weight >= 1.0 else self.theme.ink_soft)
        attributes = [
            f'x1="{_fmt(start[0])}"',
            f'y1="{_fmt(start[1])}"',
            f'x2="{_fmt(end[0])}"',
            f'y2="{_fmt(end[1])}"',
            f'stroke="{stroke}"',
            f'stroke-width="{_fmt((width if width is not None else style.weight) * self.base_width)}"',
        ]
        if style.dash:
            attributes.append(f'stroke-dasharray="{style.dash}"')
        if opacity < 1.0:
            attributes.append(f'opacity="{_fmt(opacity)}"')
        if marker_start:
            attributes.append(f'marker-start="url(#{marker_start})"')
        if marker_end:
            attributes.append(f'marker-end="url(#{marker_end})"')
        self._emit(layer, f"<line {' '.join(attributes)}/>")

    def polyline(
        self,
        points: list[Point],
        *,
        kind: str = "visible",
        colour: str | None = None,
        layer: str = "geometry",
        close: bool = False,
        fill: str = "none",
        opacity: float = 1.0,
        marker_end: str = "",
        width: float | None = None,
    ) -> None:
        if len(points) < 2:
            return
        style = LINE_TYPES.get(kind, LINE_TYPES["visible"])
        stroke = colour or (self.theme.ink if style.weight >= 1.0 else self.theme.ink_soft)
        path = " ".join(
            ("M" if index == 0 else "L") + f"{_fmt(x)},{_fmt(y)}"
            for index, (x, y) in enumerate(points)
        )
        if close:
            path += " Z"
        attributes = [
            f'd="{path}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{_fmt((width if width is not None else style.weight) * self.base_width)}"',
            'stroke-linejoin="round"',
        ]
        if style.dash:
            attributes.append(f'stroke-dasharray="{style.dash}"')
        if opacity < 1.0:
            attributes.append(f'opacity="{_fmt(opacity)}"')
        if marker_end:
            attributes.append(f'marker-end="url(#{marker_end})"')
        self._emit(layer, f"<path {' '.join(attributes)}/>")

    def circle(
        self,
        centre: Point,
        radius: float,
        *,
        kind: str = "visible",
        colour: str | None = None,
        fill: str = "none",
        layer: str = "geometry",
        opacity: float = 1.0,
    ) -> None:
        style = LINE_TYPES.get(kind, LINE_TYPES["visible"])
        stroke = colour or (self.theme.ink if style.weight >= 1.0 else self.theme.ink_soft)
        attributes = [
            f'cx="{_fmt(centre[0])}"',
            f'cy="{_fmt(centre[1])}"',
            f'r="{_fmt(radius)}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{_fmt(style.weight * self.base_width)}"',
        ]
        if style.dash:
            attributes.append(f'stroke-dasharray="{style.dash}"')
        if opacity < 1.0:
            attributes.append(f'opacity="{_fmt(opacity)}"')
        self._emit(layer, f"<circle {' '.join(attributes)}/>")

    def rect(
        self,
        origin: Point,
        width: float,
        height: float,
        *,
        kind: str = "visible",
        colour: str | None = None,
        fill: str = "none",
        radius: float = 0.0,
        layer: str = "geometry",
        opacity: float = 1.0,
    ) -> None:
        style = LINE_TYPES.get(kind, LINE_TYPES["visible"])
        stroke = colour or (self.theme.ink if style.weight >= 1.0 else self.theme.ink_soft)
        attributes = [
            f'x="{_fmt(origin[0])}"',
            f'y="{_fmt(origin[1])}"',
            f'width="{_fmt(width)}"',
            f'height="{_fmt(height)}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{_fmt(style.weight * self.base_width)}"',
        ]
        if radius:
            attributes.append(f'rx="{_fmt(radius)}"')
        if style.dash:
            attributes.append(f'stroke-dasharray="{style.dash}"')
        if opacity < 1.0:
            attributes.append(f'opacity="{_fmt(opacity)}"')
        self._emit(layer, f"<rect {' '.join(attributes)}/>")

    def polygon(
        self,
        points: list[Point],
        *,
        fill: str,
        stroke: str = "none",
        layer: str = "geometry",
        opacity: float = 1.0,
        width: float = 0.5,
    ) -> None:
        if len(points) < 3:
            return
        path = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in points)
        attributes = [f'points="{path}"', f'fill="{fill}"']
        if stroke != "none":
            attributes.append(f'stroke="{stroke}"')
            attributes.append(f'stroke-width="{_fmt(width * self.base_width)}"')
        if opacity < 1.0:
            attributes.append(f'opacity="{_fmt(opacity)}"')
        self._emit(layer, f"<polygon {' '.join(attributes)}/>")

    # -- text ------------------------------------------------------------
    def text(
        self,
        position: Point,
        content: str,
        *,
        size: float | None = None,
        anchor: str = "start",
        colour: str | None = None,
        layer: str = "text",
        weight: str = "normal",
        mono: bool = False,
        rotation: float = 0.0,
        baseline: str = "auto",
        letter_spacing: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        height = size if size is not None else self.text_height
        attributes = [
            f'x="{_fmt(position[0])}"',
            f'y="{_fmt(position[1])}"',
            f'font-size="{_fmt(height)}"',
            f'font-family="{self.theme.mono if mono else self.theme.font}"',
            f'fill="{colour or self.theme.ink}"',
            f'text-anchor="{anchor}"',
        ]
        if weight != "normal":
            attributes.append(f'font-weight="{weight}"')
        if baseline != "auto":
            attributes.append(f'dominant-baseline="{baseline}"')
        if letter_spacing:
            attributes.append(f'letter-spacing="{_fmt(letter_spacing)}"')
        if opacity < 1.0:
            attributes.append(f'opacity="{_fmt(opacity)}"')
        if rotation:
            attributes.append(
                f'transform="rotate({_fmt(rotation)} {_fmt(position[0])} {_fmt(position[1])})"'
            )
        self._emit(layer, f"<text {' '.join(attributes)}>{escape(str(content))}</text>")

    def wrap_lines(
        self, content: str, width: float, size: float, *, mono: bool = False
    ) -> list[str]:
        """Break a paragraph into lines that fit a column.

        Width is measured in characters estimated from the font size, which
        is close enough for a drawing note and needs no font metrics.
        """
        per_character = size * (0.6 if mono else 0.52)
        columns = max(int(width / per_character), 8)
        lines: list[str] = []
        current = ""
        for word in str(content or "").split():
            candidate = f"{current} {word}".strip()
            if len(candidate) <= columns:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word if len(word) <= columns else word[: columns - 1] + "-"
        if current:
            lines.append(current)
        return lines

    def wrapped_text(
        self,
        position: Point,
        content: str,
        *,
        width: float,
        size: float | None = None,
        line_spacing: float = 1.35,
        colour: str | None = None,
        layer: str = "text",
        anchor: str = "start",
        mono: bool = False,
        max_height: float | None = None,
    ) -> float:
        """Lay out a paragraph and return the height it used.

        With ``max_height`` the paragraph is cut to fit and the last line
        ends in an ellipsis. A drawing with a note running off the bottom of
        its own panel and over the title block is worse than a note that
        says it has more to say.
        """
        height = size if size is not None else self.text_height
        lines = self.wrap_lines(content, width, height, mono=mono)
        step = height * line_spacing
        if max_height is not None:
            allowed = max(int(max_height / step), 0)
            if allowed == 0:
                return 0.0
            if len(lines) > allowed:
                lines = lines[:allowed]
                lines[-1] = lines[-1].rstrip(",.;") + "..."
        for index, line in enumerate(lines):
            self.text(
                (position[0], position[1] + index * step),
                line,
                size=height,
                anchor=anchor,
                colour=colour,
                layer=layer,
                mono=mono,
            )
        return len(lines) * step

    # -- drafting annotation ---------------------------------------------
    def leader(
        self,
        target: Point,
        elbow: Point,
        landing_length: float,
        label: str,
        *,
        sublabel: str = "",
        colour: str | None = None,
        layer: str = "callouts",
        size: float | None = None,
    ) -> Point:
        """A leader line: arrow at the feature, a bend, a horizontal landing.

        The label sits above the landing and reads left to right, per ISO
        129-1. Returning the text anchor lets a caller stack more under it.
        """
        ink = colour or self.theme.ink
        direction = 1.0 if landing_length >= 0 else -1.0
        landing_end = (elbow[0] + landing_length, elbow[1])
        self.polyline(
            [target, elbow, landing_end],
            kind="thin",
            colour=ink,
            layer=layer,
            marker_end="",
        )
        # A dot at the feature reads better than an arrow on a wireframe,
        # where an arrowhead disappears into the mesh lines behind it.
        self.circle(target, self.base_width * 1.6, kind="thin", colour=ink,
                    fill=ink, layer=layer)
        height = size if size is not None else self.text_height
        anchor = "start" if direction > 0 else "end"
        text_x = landing_end[0] + direction * height * 0.4
        self.text(
            (text_x, landing_end[1] - height * 0.35),
            label,
            size=height,
            anchor=anchor,
            colour=ink,
            layer=layer,
            weight="600",
        )
        if sublabel:
            self.text(
                (text_x, landing_end[1] + height * 0.95),
                sublabel,
                size=height * 0.82,
                anchor=anchor,
                colour=self.theme.ink_soft,
                layer=layer,
            )
        return (text_x, landing_end[1])

    def balloon(
        self,
        centre: Point,
        number: int | str,
        *,
        radius: float = 3.2,
        layer: str = "callouts",
        colour: str | None = None,
    ) -> None:
        """A find number in a circle, ASME Y14.34 style."""
        ink = colour or self.theme.ink
        self.circle(centre, radius, kind="thin", colour=ink,
                    fill=self.theme.paper, layer=layer)
        self.text(
            (centre[0], centre[1]),
            str(number),
            size=radius * 1.15,
            anchor="middle",
            colour=ink,
            layer=layer,
            baseline="central",
            weight="600",
        )

    def dimension(
        self,
        start: Point,
        end: Point,
        offset: float,
        label: str,
        *,
        layer: str = "dimensions",
        colour: str | None = None,
        size: float | None = None,
        vertical: bool = False,
    ) -> None:
        """A linear dimension with extension lines, arrows and a figure.

        The extension lines start clear of the feature and run past the
        dimension line, both per ISO 129-1. When the space between the
        arrows is too small for the figure, the arrows go outside and the
        figure sits beside them, which is what a draughtsman does.
        """
        ink = colour or self.theme.ink_soft
        height = size if size is not None else self.text_height * 0.85
        gap = height * 0.35
        overrun = height * 0.5
        if vertical:
            line_x = max(start[0], end[0]) + offset
            self.line((start[0] + math.copysign(gap, offset), start[1]),
                      (line_x + math.copysign(overrun, offset), start[1]),
                      kind="thin", colour=ink, layer=layer)
            self.line((end[0] + math.copysign(gap, offset), end[1]),
                      (line_x + math.copysign(overrun, offset), end[1]),
                      kind="thin", colour=ink, layer=layer)
            span = abs(end[1] - start[1])
            inside = span > height * len(label) * 0.6
            self.line((line_x, start[1]), (line_x, end[1]), kind="thin", colour=ink,
                      layer=layer, marker_start="arrow", marker_end="arrow")
            self.text(
                (line_x - height * 0.4, (start[1] + end[1]) / 2.0),
                label,
                size=height,
                anchor="middle",
                colour=ink,
                layer=layer,
                rotation=-90,
                baseline="central",
            )
            return
        line_y = max(start[1], end[1]) + offset
        self.line((start[0], start[1] + math.copysign(gap, offset)),
                  (start[0], line_y + math.copysign(overrun, offset)),
                  kind="thin", colour=ink, layer=layer)
        self.line((end[0], end[1] + math.copysign(gap, offset)),
                  (end[0], line_y + math.copysign(overrun, offset)),
                  kind="thin", colour=ink, layer=layer)
        span = abs(end[0] - start[0])
        inside = span > height * len(label) * 0.62
        if inside:
            self.line((start[0], line_y), (end[0], line_y), kind="thin", colour=ink,
                      layer=layer, marker_start="arrow", marker_end="arrow")
            self.text(((start[0] + end[0]) / 2.0, line_y - height * 0.4), label,
                      size=height, anchor="middle", colour=ink, layer=layer)
        else:
            reach = height * 2.0
            self.line((start[0] - reach, line_y), (start[0], line_y), kind="thin",
                      colour=ink, layer=layer, marker_start="arrow")
            self.line((end[0], line_y), (end[0] + reach, line_y), kind="thin",
                      colour=ink, layer=layer, marker_end="arrow")
            self.text((end[0] + reach + height * 0.4, line_y - height * 0.35), label,
                      size=height, anchor="start", colour=ink, layer=layer)

    def centre_mark(self, centre: Point, radius: float, *, layer: str = "geometry") -> None:
        """The cross that says where a circular feature's axis is."""
        reach = radius * 1.25
        self.line((centre[0] - reach, centre[1]), (centre[0] + reach, centre[1]),
                  kind="centre", layer=layer)
        self.line((centre[0], centre[1] - reach), (centre[0], centre[1] + reach),
                  kind="centre", layer=layer)

    def hatch(
        self,
        polygon: list[Point],
        *,
        angle: float = 45.0,
        spacing: float = 2.0,
        colour: str | None = None,
        layer: str = "hatch",
        opacity: float = 0.85,
    ) -> None:
        """Section hatching inside a polygon, ISO 128-50.

        The lines are clipped to the polygon by an SVG clip path, so a
        concave section hatches correctly without any polygon arithmetic.
        """
        if len(polygon) < 3:
            return
        self._clip_count += 1
        clip_id = f"hatch{self._clip_count}"
        points = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in polygon)
        self.define(f'<clipPath id="{clip_id}"><polygon points="{points}"/></clipPath>')
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        reach = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        centre = ((max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0)
        radians = math.radians(angle)
        dx, dy = math.cos(radians), math.sin(radians)
        nx, ny = -dy, dx
        ink = colour or self.theme.hatch
        lines: list[str] = []
        steps = int(reach / spacing) + 2
        for index in range(-steps, steps + 1):
            offset = index * spacing
            base = (centre[0] + nx * offset, centre[1] + ny * offset)
            a = (base[0] - dx * reach, base[1] - dy * reach)
            b = (base[0] + dx * reach, base[1] + dy * reach)
            lines.append(
                f'<line x1="{_fmt(a[0])}" y1="{_fmt(a[1])}" '
                f'x2="{_fmt(b[0])}" y2="{_fmt(b[1])}"/>'
            )
        self._emit(
            layer,
            f'<g clip-path="url(#{clip_id})" stroke="{ink}" '
            f'stroke-width="{_fmt(0.35 * self.base_width)}" opacity="{_fmt(opacity)}">'
            + "".join(lines)
            + "</g>",
        )

    def grid(self, spacing: float, *, layer: str = "grid", opacity: float = 0.5) -> None:
        """A faint background grid, the way a drawing board has one."""
        lines: list[str] = []
        x = 0.0
        while x <= self.width:
            lines.append(f'<line x1="{_fmt(x)}" y1="0" x2="{_fmt(x)}" y2="{_fmt(self.height)}"/>')
            x += spacing
        y = 0.0
        while y <= self.height:
            lines.append(f'<line x1="0" y1="{_fmt(y)}" x2="{_fmt(self.width)}" y2="{_fmt(y)}"/>')
            y += spacing
        self._emit(
            layer,
            f'<g stroke="{self.theme.grid}" stroke-width="{_fmt(0.25 * self.base_width)}" '
            f'opacity="{_fmt(opacity)}">' + "".join(lines) + "</g>",
        )

    def panel(
        self,
        origin: Point,
        width: float,
        height: float,
        title: str = "",
        *,
        subtitle: str = "",
        layer: str = "frame",
    ) -> Point:
        """A bordered region with a heading, and the point content starts at."""
        self.rect(origin, width, height, kind="thin", colour=self.theme.ink_soft,
                  fill=self.theme.paper, radius=1.0, layer=layer)
        cursor_y = origin[1] + 4.4
        if title:
            self.text((origin[0] + 3.0, cursor_y), title.upper(),
                      size=self.text_height * 0.78, colour=self.theme.ink,
                      weight="700", letter_spacing=0.5, layer=layer)
            cursor_y += 2.6
            self.line((origin[0] + 3.0, cursor_y), (origin[0] + width - 3.0, cursor_y),
                      kind="thin", colour=self.theme.ink_soft, layer=layer, opacity=0.6)
            cursor_y += 3.4
        if subtitle:
            self.text((origin[0] + 3.0, cursor_y), subtitle,
                      size=self.text_height * 0.68, colour=self.theme.ink_soft, layer=layer)
            cursor_y += 3.6
        return (origin[0] + 3.0, cursor_y)

    def bounded_panel(
        self,
        origin: Point,
        width: float,
        height: float,
        title: str = "",
        *,
        subtitle: str = "",
        layer: str = "frame",
    ) -> BoundedPanel:
        """A panel that keeps its own cursor and refuses to write past itself."""
        cursor = self.panel(origin, width, height, title, subtitle=subtitle, layer=layer)
        return BoundedPanel(self, origin, width, height, cursor[1])

    # -- output ----------------------------------------------------------
    def to_svg(self, *, title: str = "", description: str = "") -> str:
        body: list[str] = []
        for name in self._order:
            markup = "".join(self._layers[name])
            if markup:
                body.append(f'<g id="{escape(name)}">{markup}</g>')
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {_fmt(self.width)} {_fmt(self.height)}" '
            f'width="{_fmt(self.width)}mm" height="{_fmt(self.height)}mm" '
            f'role="img" aria-label="{escape(title or "Engineering drawing")}">'
        )
        parts = [head]
        if title:
            parts.append(f"<title>{escape(title)}</title>")
        if description:
            parts.append(f"<desc>{escape(description)}</desc>")
        parts.append(f"<defs>{''.join(self._defs)}</defs>")
        parts.append(
            f'<rect x="0" y="0" width="{_fmt(self.width)}" height="{_fmt(self.height)}" '
            f'fill="{self.theme.background}"/>'
        )
        parts.extend(body)
        parts.append("</svg>")
        return "".join(parts)

    def legend_entries(self) -> list[tuple[str, LineType]]:
        """The line types this drawing actually used, for its own key."""
        used: list[tuple[str, LineType]] = []
        blob = "".join("".join(v) for v in self._layers.values())
        for key, style in LINE_TYPES.items():
            if style.dash and style.dash in blob:
                used.append((key, style))
        return used


class BoundedPanel:
    """A panel with a cursor that stops at its own bottom edge.

    Every table and note on a sheet is written through one of these, because
    the failure that ruins a drawing is not a wrong number: it is a note that
    runs off its panel and across the title block. Each write reports whether
    it fitted, so a caller can stop and say how much was left rather than
    scribbling over the next panel.
    """

    def __init__(
        self, canvas: Canvas, origin: Point, width: float, height: float, top: float
    ) -> None:
        self.canvas = canvas
        self.x = origin[0]
        self.y = origin[1]
        self.width = width
        self.height = height
        self.cursor = top
        self.padding = 3.0
        self.overflow = 0

    @property
    def bottom(self) -> float:
        return self.y + self.height - self.padding * 0.8

    @property
    def remaining(self) -> float:
        return max(self.bottom - self.cursor, 0.0)

    @property
    def inner_width(self) -> float:
        return self.width - self.padding * 2

    def room_for(self, height: float) -> bool:
        return self.cursor + height <= self.bottom

    def row(
        self,
        left: str,
        right: str = "",
        *,
        size: float | None = None,
        colour: str | None = None,
        right_colour: str | None = None,
        weight: str = "normal",
        mono_right: bool = True,
        layer: str = "tables",
        indent: float = 0.0,
    ) -> bool:
        """One line with a label on the left and a value on the right."""
        height = size if size is not None else self.canvas.text_height * 0.72
        if not self.room_for(height * 1.2):
            self.overflow += 1
            return False
        self.canvas.text(
            (self.x + self.padding + indent, self.cursor), left,
            size=height, colour=colour or self.canvas.theme.ink,
            weight=weight, layer=layer,
        )
        if right:
            self.canvas.text(
                (self.x + self.width - self.padding, self.cursor), right,
                size=height, anchor="end",
                colour=right_colour or colour or self.canvas.theme.ink_soft,
                layer=layer, mono=mono_right, weight=weight,
            )
        self.cursor += height * 1.35
        return True

    def note(
        self,
        text: str,
        *,
        size: float | None = None,
        colour: str | None = None,
        layer: str = "tables",
        indent: float = 0.0,
        gap: float = 0.45,
    ) -> bool:
        """A wrapped paragraph, cut to whatever room is left."""
        height = size if size is not None else self.canvas.text_height * 0.66
        if self.remaining < height:
            self.overflow += 1
            return False
        used = self.canvas.wrapped_text(
            (self.x + self.padding + indent, self.cursor), text,
            width=self.inner_width - indent, size=height,
            colour=colour or self.canvas.theme.ink_soft, layer=layer,
            max_height=self.remaining,
        )
        wanted = len(self.canvas.wrap_lines(text, self.inner_width - indent, height))
        if used < wanted * height * 1.35:
            self.overflow += 1
        self.cursor += used + height * gap
        return True

    def rule(self, *, layer: str = "tables", opacity: float = 0.5) -> None:
        if not self.room_for(1.5):
            return
        self.canvas.line(
            (self.x + self.padding, self.cursor),
            (self.x + self.width - self.padding, self.cursor),
            kind="thin", colour=self.canvas.theme.ink_soft, layer=layer, opacity=opacity,
        )
        self.cursor += self.canvas.text_height * 0.55

    def gap(self, amount: float) -> None:
        self.cursor += amount

    def marker(self, colour: str, *, radius: float = 1.3, layer: str = "tables") -> None:
        """A status dot beside the row about to be written."""
        self.canvas.circle(
            (self.x + self.padding + radius, self.cursor - radius * 0.3), radius,
            kind="thin", colour=colour, fill=colour, layer=layer,
        )

    def truncation_note(self, hidden: int, *, layer: str = "tables") -> None:
        """Say how much did not fit, rather than letting it vanish."""
        if hidden <= 0:
            return
        height = self.canvas.text_height * 0.62
        self.canvas.text(
            (self.x + self.width - self.padding, self.bottom + height * 0.2),
            f"+{hidden} more",
            size=height, anchor="end", colour=self.canvas.theme.ink_soft,
            layer=layer, mono=True,
        )
