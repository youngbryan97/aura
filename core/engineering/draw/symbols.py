"""The symbol libraries, drawn to the standards their disciplines use.

A schematic is a language, and every discipline settled its alphabet decades
ago. An electrician reads IEC 60617 or IEEE 315; a process engineer reads
ISA-5.1; a hydraulics engineer reads ISO 1219; a systems biologist reads
SBGN. Inventing new pictures for those meanings would make a drawing that
nobody can read and that no tool can import.

Each symbol declares where its pins are, in a unit box from minus one to
plus one, so a router can join two symbols without knowing what either of
them is. Each also carries the plain name and the sentence that says what
the thing does, because the whole point of drawing a standard symbol is
that a reader who knows the standard needs no caption, and a reader who
does not needs one badly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.engineering.draw.canvas import Canvas, Point

__all__ = ["Pin", "Symbol", "SYMBOLS", "symbol_for", "draw_symbol", "STANDARDS"]

STANDARDS: dict[str, str] = {
    "IEC 60617": "International graphical symbols for electrical diagrams.",
    "IEEE 315": "North American graphical symbols for electrical diagrams.",
    "ISA-5.1": "Instrumentation symbols for process and instrumentation diagrams.",
    "ISO 1219": "Graphical symbols for fluid power systems.",
    "ISO 10628": "Flow diagrams for process plants.",
    "SBGN PD": "Systems Biology Graphical Notation, process description.",
    "ISO 14617": "General graphical symbols for diagrams.",
}


@dataclass(frozen=True, slots=True)
class Pin:
    """One connection point on a symbol, in unit-box coordinates."""

    name: str
    x: float
    y: float
    domain: str = "electrical"
    role: str = "bidirectional"

    def at(self, centre: Point, size: float) -> Point:
        return (centre[0] + self.x * size, centre[1] + self.y * size)


@dataclass(frozen=True, slots=True)
class Symbol:
    """One standard symbol: how to draw it, where its pins are, what it means."""

    key: str
    name: str
    lay_name: str
    standard: str
    domain: str
    render: Callable[..., None]
    pins: tuple[Pin, ...] = ()
    description: str = ""
    #: How wide the symbol wants to be, as a multiple of its height.
    aspect: float = 1.0
    keywords: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "lay_name": self.lay_name,
            "standard": self.standard,
            "domain": self.domain,
            "description": self.description,
            "pins": [{"name": p.name, "domain": p.domain, "role": p.role} for p in self.pins],
        }


SYMBOLS: dict[str, Symbol] = {}


def _register(
    key: str,
    name: str,
    lay_name: str,
    standard: str,
    domain: str,
    pins: tuple[Pin, ...],
    description: str,
    *,
    aspect: float = 1.0,
    keywords: tuple[str, ...] = (),
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def decorate(function: Callable[..., None]) -> Callable[..., None]:
        SYMBOLS[key] = Symbol(
            key=key, name=name, lay_name=lay_name, standard=standard, domain=domain,
            render=function, pins=pins, description=description, aspect=aspect,
            keywords=keywords,
        )
        return function

    return decorate


def _p(centre: Point, size: float, x: float, y: float) -> Point:
    return (centre[0] + x * size, centre[1] + y * size)


def _lead(canvas: Canvas, centre: Point, size: float, pin: Pin, to: Point,
          colour: str, layer: str) -> None:
    canvas.line(pin.at(centre, size), to, kind="thin", colour=colour, layer=layer)


LEFT_RIGHT = (Pin("a", -1.0, 0.0), Pin("b", 1.0, 0.0))
TOP_BOTTOM = (Pin("a", 0.0, -1.0), Pin("b", 0.0, 1.0))


# ---------------------------------------------------------------------------
# Electrical, IEC 60617 with the IEEE 315 alternatives noted
# ---------------------------------------------------------------------------


@_register("resistor", "Resistor", "resistor", "IEC 60617", "electrical", LEFT_RIGHT,
           "Turns some of the electricity into heat, to limit current or set a voltage.",
           keywords=("resistor", "resistance"))
def _resistor(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
              variant: str = "iec", **_: Any) -> None:
    if variant == "ieee":
        # The zigzag North American form.
        points = [_p(centre, size, -1.0, 0.0), _p(centre, size, -0.55, 0.0)]
        for index in range(6):
            x = -0.55 + index * 0.183
            points.append(_p(centre, size, x + 0.09, 0.42 if index % 2 == 0 else -0.42))
        points += [_p(centre, size, 0.55, 0.0), _p(centre, size, 1.0, 0.0)]
        canvas.polyline(points, kind="visible", colour=colour, layer=layer)
        return
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.55, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.rect(_p(centre, size, -0.55, -0.35), size * 1.1, size * 0.7,
                kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.55, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("capacitor", "Capacitor", "capacitor", "IEC 60617", "electrical", LEFT_RIGHT,
           "Stores a little charge; passes changing signals and blocks steady ones.",
           keywords=("capacitor", "cap"))
def _capacitor(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
               polarised: bool = False, **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.16, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.16, -0.62), _p(centre, size, -0.16, 0.62),
                kind="visible", colour=colour, layer=layer)
    if polarised:
        canvas.polyline(
            [_p(centre, size, 0.16, -0.62), _p(centre, size, 0.30, -0.30),
             _p(centre, size, 0.30, 0.30), _p(centre, size, 0.16, 0.62)],
            kind="visible", colour=colour, layer=layer,
        )
        canvas.text(_p(centre, size, -0.45, -0.75), "+", size=size * 0.7,
                    anchor="middle", colour=colour, layer=layer)
    else:
        canvas.line(_p(centre, size, 0.16, -0.62), _p(centre, size, 0.16, 0.62),
                    kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.16, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("inductor", "Inductor", "coil", "IEC 60617", "electrical", LEFT_RIGHT,
           "Resists a change in current; stores energy in a magnetic field.",
           keywords=("inductor", "coil", "choke"))
def _inductor(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.6, 0.0),
                kind="thin", colour=colour, layer=layer)
    points = []
    for index in range(49):
        t = index / 48.0
        angle = t * 4.0 * math.pi
        points.append(_p(centre, size, -0.6 + t * 1.2, -abs(math.sin(angle)) * 0.42))
    canvas.polyline(points, kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.6, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("diode", "Diode", "one-way valve for electricity", "IEC 60617", "electrical",
           (Pin("anode", -1.0, 0.0, role="input"), Pin("cathode", 1.0, 0.0, role="output")),
           "Lets current through one way and blocks it the other.",
           keywords=("diode", "rectifier"))
def _diode(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
           kind: str = "plain", **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.4, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.polygon(
        [_p(centre, size, -0.4, -0.5), _p(centre, size, -0.4, 0.5), _p(centre, size, 0.3, 0.0)],
        fill=colour, layer=layer,
    )
    canvas.line(_p(centre, size, 0.3, -0.5), _p(centre, size, 0.3, 0.5),
                kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.3, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)
    if kind == "led":
        for offset in (-0.15, 0.15):
            canvas.line(_p(centre, size, 0.0 + offset, -0.6), _p(centre, size, 0.35 + offset, -1.0),
                        kind="thin", colour=colour, layer=layer, marker_end="arrow")
    elif kind == "zener":
        canvas.polyline(
            [_p(centre, size, 0.12, -0.5), _p(centre, size, 0.3, -0.5),
             _p(centre, size, 0.3, 0.5), _p(centre, size, 0.48, 0.5)],
            kind="visible", colour=colour, layer=layer,
        )


@_register("transistor_npn", "NPN transistor", "electronic switch", "IEC 60617", "electrical",
           (Pin("base", -1.0, 0.0, role="input"), Pin("collector", 0.55, -1.0),
            Pin("emitter", 0.55, 1.0, role="output")),
           "A small current at the base lets a much larger one flow through it.",
           keywords=("transistor", "npn", "bjt", "switch"))
def _npn(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
         pnp: bool = False, **_: Any) -> None:
    canvas.circle(centre, size * 0.85, kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.3, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.3, -0.55), _p(centre, size, -0.3, 0.55),
                kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.3, -0.35), _p(centre, size, 0.55, -0.8),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.3, 0.35), _p(centre, size, 0.55, 0.8),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.55, -0.8), _p(centre, size, 0.55, -1.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.55, 0.8), _p(centre, size, 0.55, 1.0),
                kind="thin", colour=colour, layer=layer)
    tip = _p(centre, size, 0.30, 0.62) if not pnp else _p(centre, size, 0.05, 0.44)
    back = _p(centre, size, 0.05, 0.44) if not pnp else _p(centre, size, 0.30, 0.62)
    canvas.polygon(
        [tip, (back[0] + (tip[1] - back[1]) * 0.28, back[1] - (tip[0] - back[0]) * 0.28),
         (back[0] - (tip[1] - back[1]) * 0.28, back[1] + (tip[0] - back[0]) * 0.28)],
        fill=colour, layer=layer,
    )


@_register("mosfet", "MOSFET", "electronic switch", "IEC 60617", "electrical",
           (Pin("gate", -1.0, 0.0, role="input"), Pin("drain", 0.55, -1.0),
            Pin("source", 0.55, 1.0, role="output")),
           "Switched by voltage rather than current; what most power switching uses now.",
           keywords=("mosfet", "fet", "switch"))
def _mosfet(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.circle(centre, size * 0.85, kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.45, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.45, -0.55), _p(centre, size, -0.45, 0.55),
                kind="visible", colour=colour, layer=layer)
    for y0, y1 in ((-0.62, -0.24), (-0.18, 0.18), (0.24, 0.62)):
        canvas.line(_p(centre, size, -0.2, y0), _p(centre, size, -0.2, y1),
                    kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.2, -0.43), _p(centre, size, 0.55, -0.43),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.2, 0.43), _p(centre, size, 0.55, 0.43),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.55, -0.43), _p(centre, size, 0.55, -1.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.55, 0.43), _p(centre, size, 0.55, 1.0),
                kind="thin", colour=colour, layer=layer)


@_register("battery", "Battery", "battery", "IEC 60617", "electrical",
           (Pin("positive", 1.0, 0.0, role="source"), Pin("negative", -1.0, 0.0, role="sink")),
           "Stores energy chemically and pushes current out of its positive end.",
           keywords=("battery", "cell", "pack", "accumulator"))
def _battery(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
             cells: int = 2, **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.5, 0.0),
                kind="thin", colour=colour, layer=layer)
    x = -0.5
    step = 1.0 / max(cells * 2 - 1, 1)
    for index in range(cells * 2):
        tall = index % 2 == 1
        canvas.line(_p(centre, size, x, -0.65 if tall else -0.32),
                    _p(centre, size, x, 0.65 if tall else 0.32),
                    kind="visible", colour=colour, layer=layer,
                    width=0.6 if tall else 1.4)
        x += step
    canvas.line(_p(centre, size, x - step, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.text(_p(centre, size, 0.75, -0.75), "+", size=size * 0.75,
                anchor="middle", colour=colour, layer=layer)


@_register("ground", "Ground", "earth", "IEC 60617", "electrical",
           (Pin("a", 0.0, -1.0, role="sink"),),
           "The common reference every other voltage is measured against.",
           keywords=("ground", "gnd", "earth", "return", "common"))
def _ground(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.line(_p(centre, size, 0.0, -1.0), _p(centre, size, 0.0, 0.0),
                kind="thin", colour=colour, layer=layer)
    for index, half in enumerate((0.7, 0.45, 0.2)):
        y = index * 0.3
        canvas.line(_p(centre, size, -half, y), _p(centre, size, half, y),
                    kind="visible", colour=colour, layer=layer)


@_register("switch", "Switch", "switch", "IEC 60617", "electrical", LEFT_RIGHT,
           "Makes or breaks the circuit.",
           keywords=("switch", "contact", "breaker", "isolator"))
def _switch(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.45, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.circle(_p(centre, size, -0.45, 0.0), size * 0.09, kind="thin",
                  colour=colour, fill=colour, layer=layer)
    canvas.line(_p(centre, size, -0.45, 0.0), _p(centre, size, 0.4, -0.55),
                kind="visible", colour=colour, layer=layer)
    canvas.circle(_p(centre, size, 0.45, 0.0), size * 0.09, kind="thin",
                  colour=colour, fill=colour, layer=layer)
    canvas.line(_p(centre, size, 0.45, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("fuse", "Fuse", "fuse", "IEC 60617", "electrical", LEFT_RIGHT,
           "Melts and breaks the circuit if too much current flows.",
           keywords=("fuse", "protection", "breaker"))
def _fuse(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.6, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.rect(_p(centre, size, -0.6, -0.32), size * 1.2, size * 0.64,
                kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.6, 0.0), _p(centre, size, 0.6, 0.0),
                kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.6, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("motor", "Motor", "motor", "IEC 60617", "electrical",
           (Pin("a", -1.0, 0.0, role="sink"), Pin("shaft", 1.0, 0.0, "mechanical_rotary", "source")),
           "Turns electricity into rotation.",
           keywords=("motor", "actuator", "thruster", "drive", "servo"))
def _motor(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
           letter: str = "M", **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.8, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.circle(centre, size * 0.8, kind="visible", colour=colour, layer=layer)
    canvas.text(centre, letter, size=size * 0.9, anchor="middle", colour=colour,
                layer=layer, baseline="central", weight="600")
    canvas.line(_p(centre, size, 0.8, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("lamp", "Lamp", "light", "IEC 60617", "electrical", LEFT_RIGHT,
           "Turns electricity into light.",
           keywords=("lamp", "light", "led", "indicator", "illumination"))
def _lamp(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.7, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.circle(centre, size * 0.7, kind="visible", colour=colour, layer=layer)
    reach = 0.7 / math.sqrt(2.0)
    canvas.line(_p(centre, size, -reach, -reach), _p(centre, size, reach, reach),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -reach, reach), _p(centre, size, reach, -reach),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.7, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("amplifier", "Amplifier", "amplifier", "IEC 60617", "electrical",
           (Pin("in_plus", -1.0, -0.4, role="input"), Pin("in_minus", -1.0, 0.4, role="input"),
            Pin("out", 1.0, 0.0, role="output")),
           "Makes a small signal into a larger copy of itself.",
           keywords=("amplifier", "op-amp", "opamp", "gain", "comparator"))
def _amplifier(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.polyline(
        [_p(centre, size, -0.7, -0.9), _p(centre, size, 0.8, 0.0), _p(centre, size, -0.7, 0.9)],
        kind="visible", colour=colour, layer=layer, close=True,
    )
    canvas.line(_p(centre, size, -1.0, -0.4), _p(centre, size, -0.7, -0.4),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -1.0, 0.4), _p(centre, size, -0.7, 0.4),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.8, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.text(_p(centre, size, -0.5, -0.28), "+", size=size * 0.55,
                anchor="middle", colour=colour, layer=layer, baseline="central")
    canvas.text(_p(centre, size, -0.5, 0.4), "-", size=size * 0.7,
                anchor="middle", colour=colour, layer=layer, baseline="central")


@_register("controller", "Controller", "computer", "IEC 60617", "data",
           (Pin("in", -1.0, 0.0, "data", "input"), Pin("out", 1.0, 0.0, "data", "output"),
            Pin("power", 0.0, -1.0, "electrical", "sink")),
           "Reads the sensors, decides, and drives the outputs.",
           keywords=("controller", "computer", "board", "processor", "mcu", "plc", "autopilot"))
def _controller(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
                label: str = "", **_: Any) -> None:
    canvas.rect(_p(centre, size, -0.9, -0.7), size * 1.8, size * 1.4,
                kind="visible", colour=colour, layer=layer, radius=size * 0.1)
    for offset in (-0.35, 0.0, 0.35):
        canvas.line(_p(centre, size, -1.0, offset), _p(centre, size, -0.9, offset),
                    kind="thin", colour=colour, layer=layer)
        canvas.line(_p(centre, size, 0.9, offset), _p(centre, size, 1.0, offset),
                    kind="thin", colour=colour, layer=layer)
    if label:
        canvas.text(centre, label[:6], size=size * 0.5, anchor="middle",
                    colour=colour, layer=layer, baseline="central", mono=True)


@_register("sensor", "Sensor", "sensor", "IEC 60617", "signal",
           (Pin("sense", 0.0, 1.0, "signal", "input"), Pin("out", 1.0, 0.0, "signal", "output")),
           "Measures something and reports it as a signal.",
           keywords=("sensor", "probe", "detector", "gauge", "transducer", "camera"))
def _sensor(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
            letter: str = "S", **_: Any) -> None:
    canvas.polyline(
        [_p(centre, size, -0.8, -0.6), _p(centre, size, 0.8, -0.6),
         _p(centre, size, 0.8, 0.35), _p(centre, size, 0.0, 0.85),
         _p(centre, size, -0.8, 0.35)],
        kind="visible", colour=colour, layer=layer, close=True,
    )
    canvas.text(_p(centre, size, 0.0, -0.1), letter, size=size * 0.6, anchor="middle",
                colour=colour, layer=layer, baseline="central", weight="600")
    canvas.line(_p(centre, size, 0.8, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


# ---------------------------------------------------------------------------
# Process and instrumentation, ISA-5.1 and ISO 10628
# ---------------------------------------------------------------------------


@_register("vessel", "Vessel", "tank", "ISO 10628", "fluid",
           (Pin("in", -1.0, -0.5, "fluid", "input"), Pin("out", 0.0, 1.0, "fluid", "output"),
            Pin("vent", 0.0, -1.0, "fluid", "output")),
           "Holds liquid or gas, usually with the inlet high and the outlet low.",
           keywords=("tank", "vessel", "drum", "reservoir", "sump", "accumulator"))
def _vessel(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
            level: float = 0.0, **_: Any) -> None:
    canvas.polyline(
        [_p(centre, size, -0.65, -0.75), _p(centre, size, -0.65, 0.6)],
        kind="visible", colour=colour, layer=layer,
    )
    canvas.polyline(
        [_p(centre, size, 0.65, -0.75), _p(centre, size, 0.65, 0.6)],
        kind="visible", colour=colour, layer=layer,
    )
    for y, flip in ((-0.75, -1.0), (0.6, 1.0)):
        canvas.polyline(
            [_p(centre, size, -0.65, y), _p(centre, size, -0.4, y + flip * 0.28),
             _p(centre, size, 0.4, y + flip * 0.28), _p(centre, size, 0.65, y)],
            kind="visible", colour=colour, layer=layer,
        )
    if level > 0:
        y = 0.6 - level * 1.35
        canvas.line(_p(centre, size, -0.65, y), _p(centre, size, 0.65, y),
                    kind="thin", colour=colour, layer=layer)


@_register("pump", "Centrifugal pump", "pump", "ISA-5.1", "fluid",
           (Pin("suction", -1.0, 0.0, "fluid", "input"), Pin("discharge", 0.0, -1.0, "fluid", "output")),
           "Pushes liquid along, raising its pressure.",
           keywords=("pump", "impeller", "circulator"))
def _pump(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.circle(centre, size * 0.72, kind="visible", colour=colour, layer=layer)
    canvas.polyline(
        [_p(centre, size, -1.0, 0.0), _p(centre, size, -0.72, 0.0)],
        kind="visible", colour=colour, layer=layer,
    )
    canvas.polyline(
        [_p(centre, size, 0.0, -0.72), _p(centre, size, 0.0, -1.0)],
        kind="visible", colour=colour, layer=layer,
    )
    canvas.polyline(
        [_p(centre, size, -0.72, 0.30), _p(centre, size, 0.55, 0.30),
         _p(centre, size, 0.0, -0.72)],
        kind="thin", colour=colour, layer=layer,
    )


@_register("compressor", "Compressor", "compressor", "ISA-5.1", "pneumatic",
           (Pin("suction", -1.0, 0.0, "pneumatic", "input"),
            Pin("discharge", 1.0, 0.0, "pneumatic", "output")),
           "Squeezes gas into a smaller space, raising its pressure.",
           keywords=("compressor", "blower", "fan"))
def _compressor(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.circle(centre, size * 0.72, kind="visible", colour=colour, layer=layer)
    canvas.polyline(
        [_p(centre, size, -0.5, -0.5), _p(centre, size, 0.5, -0.18),
         _p(centre, size, 0.5, 0.18), _p(centre, size, -0.5, 0.5)],
        kind="thin", colour=colour, layer=layer, close=True,
    )
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.72, 0.0),
                kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.72, 0.0), _p(centre, size, 1.0, 0.0),
                kind="visible", colour=colour, layer=layer)


@_register("valve", "Valve", "valve", "ISA-5.1", "fluid",
           (Pin("in", -1.0, 0.0, "fluid", "input"), Pin("out", 1.0, 0.0, "fluid", "output"),
            Pin("actuator", 0.0, -1.0, "signal", "input")),
           "Opens and closes to let flow through or stop it.",
           keywords=("valve", "cock", "tap", "throttle"))
def _valve(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
           kind: str = "gate", **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.6, 0.0),
                kind="visible", colour=colour, layer=layer)
    canvas.polygon(
        [_p(centre, size, -0.6, -0.5), _p(centre, size, -0.6, 0.5), _p(centre, size, 0.0, 0.0)],
        fill="none", stroke=colour, layer=layer, width=1.0,
    )
    canvas.polygon(
        [_p(centre, size, 0.6, -0.5), _p(centre, size, 0.6, 0.5), _p(centre, size, 0.0, 0.0)],
        fill="none", stroke=colour, layer=layer, width=1.0,
    )
    canvas.line(_p(centre, size, 0.6, 0.0), _p(centre, size, 1.0, 0.0),
                kind="visible", colour=colour, layer=layer)
    if kind == "control":
        canvas.line(_p(centre, size, 0.0, 0.0), _p(centre, size, 0.0, -0.75),
                    kind="thin", colour=colour, layer=layer)
        canvas.polyline(
            [_p(centre, size, -0.42, -0.75), _p(centre, size, 0.42, -0.75),
             _p(centre, size, 0.42, -1.0), _p(centre, size, -0.42, -1.0)],
            kind="visible", colour=colour, layer=layer, close=True,
        )
    elif kind == "check":
        canvas.line(_p(centre, size, 0.0, -0.5), _p(centre, size, 0.0, 0.5),
                    kind="visible", colour=colour, layer=layer)
    elif kind == "relief":
        canvas.polyline(
            [_p(centre, size, 0.0, 0.0), _p(centre, size, 0.0, -0.55),
             _p(centre, size, -0.3, -0.7), _p(centre, size, 0.3, -0.85),
             _p(centre, size, 0.0, -1.0)],
            kind="thin", colour=colour, layer=layer,
        )


@_register("heat_exchanger", "Heat exchanger", "heat exchanger", "ISO 10628", "thermal",
           (Pin("hot_in", -1.0, -0.45, "fluid", "input"), Pin("hot_out", 1.0, -0.45, "fluid", "output"),
            Pin("cold_in", 1.0, 0.45, "fluid", "input"), Pin("cold_out", -1.0, 0.45, "fluid", "output")),
           "Moves heat from one stream to another without mixing them.",
           keywords=("heat exchanger", "cooler", "radiator", "condenser", "chiller", "hx"))
def _heat_exchanger(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.circle(centre, size * 0.85, kind="visible", colour=colour, layer=layer)
    points = []
    for index in range(41):
        t = index / 40.0
        points.append(_p(centre, size, -0.85 + t * 1.7, math.sin(t * 3.0 * math.pi) * 0.34))
    canvas.polyline(points, kind="visible", colour=colour, layer=layer)
    for pin in SYMBOLS["heat_exchanger"].pins if "heat_exchanger" in SYMBOLS else ():
        canvas.line(pin.at(centre, size),
                    (centre[0] + pin.x * size * 0.85, centre[1] + pin.y * size * 0.85),
                    kind="thin", colour=colour, layer=layer)


@_register("filter", "Filter", "filter", "ISO 10628", "fluid",
           (Pin("in", -1.0, 0.0, "fluid", "input"), Pin("out", 1.0, 0.0, "fluid", "output")),
           "Catches what should not go downstream.",
           keywords=("filter", "strainer", "screen", "separator"))
def _filter(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.6, 0.0),
                kind="visible", colour=colour, layer=layer)
    canvas.polyline(
        [_p(centre, size, -0.6, -0.65), _p(centre, size, 0.6, -0.65),
         _p(centre, size, 0.6, 0.65), _p(centre, size, -0.6, 0.65)],
        kind="visible", colour=colour, layer=layer, close=True,
    )
    canvas.line(_p(centre, size, -0.6, 0.65), _p(centre, size, 0.6, -0.65),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.6, 0.0), _p(centre, size, 1.0, 0.0),
                kind="visible", colour=colour, layer=layer)


@_register("reactor", "Reactor", "reaction vessel", "ISO 10628", "chemical",
           (Pin("feed", -1.0, -0.5, "chemical", "input"), Pin("product", 1.0, 0.5, "chemical", "output"),
            Pin("jacket", 0.0, -1.0, "thermal", "input")),
           "Where the reaction happens, held long enough and stirred enough to finish.",
           keywords=("reactor", "bioreactor", "fermenter", "cstr", "vessel"))
def _reactor(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    _vessel(canvas, centre, size, colour, layer)
    canvas.line(_p(centre, size, 0.0, -0.75), _p(centre, size, 0.0, 0.25),
                kind="visible", colour=colour, layer=layer)
    for y in (0.05, 0.25):
        canvas.line(_p(centre, size, -0.32, y), _p(centre, size, 0.32, y),
                    kind="visible", colour=colour, layer=layer)


@_register("instrument", "Instrument", "instrument", "ISA-5.1", "signal",
           (Pin("process", 0.0, 1.0, "signal", "input"), Pin("signal", 1.0, 0.0, "signal", "output")),
           "A gauge, transmitter or controller. The letters say what it measures and does.",
           keywords=("instrument", "transmitter", "indicator", "gauge", "loop"))
def _instrument(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
                tag: str = "PI", loop: str = "101", mounting: str = "field", **_: Any) -> None:
    canvas.circle(centre, size * 0.85, kind="visible", colour=colour,
                  fill=canvas.theme.paper, layer=layer)
    if mounting in {"panel", "control_room"}:
        canvas.line(_p(centre, size, -0.85, 0.0), _p(centre, size, 0.85, 0.0),
                    kind="thin", colour=colour, layer=layer)
    elif mounting == "shared":
        canvas.circle(centre, size * 0.85, kind="thin", colour=colour, layer=layer)
        canvas.line(_p(centre, size, -0.85, 0.0), _p(centre, size, 0.85, 0.0),
                    kind="thin", colour=colour, layer=layer)
    canvas.text(_p(centre, size, 0.0, -0.32), tag[:4], size=size * 0.44, anchor="middle",
                colour=colour, layer=layer, baseline="central", weight="600", mono=True)
    canvas.text(_p(centre, size, 0.0, 0.34), loop[:5], size=size * 0.40, anchor="middle",
                colour=colour, layer=layer, baseline="central", mono=True)


# ---------------------------------------------------------------------------
# Fluid power, ISO 1219
# ---------------------------------------------------------------------------


@_register("cylinder", "Cylinder", "ram", "ISO 1219", "hydraulic",
           (Pin("extend", -1.0, -0.4, "hydraulic", "input"),
            Pin("retract", -1.0, 0.4, "hydraulic", "input"),
            Pin("rod", 1.0, 0.0, "mechanical_linear", "output")),
           "Pushes and pulls with fluid pressure. Feed one end to extend, the other to retract.",
           keywords=("cylinder", "ram", "piston", "jack", "linear actuator"),
           aspect=1.8)
def _cylinder(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.rect(_p(centre, size, -0.9, -0.6), size * 1.5, size * 1.2,
                kind="visible", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.05, -0.6), _p(centre, size, 0.05, 0.6),
                kind="visible", colour=colour, layer=layer, width=1.6)
    canvas.line(_p(centre, size, 0.05, 0.0), _p(centre, size, 1.0, 0.0),
                kind="visible", colour=colour, layer=layer, width=1.2)
    canvas.line(_p(centre, size, -1.0, -0.4), _p(centre, size, -0.9, -0.4),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -1.0, 0.4), _p(centre, size, -0.9, 0.4),
                kind="thin", colour=colour, layer=layer)


@_register("accumulator", "Accumulator", "pressure store", "ISO 1219", "hydraulic",
           (Pin("port", 0.0, 1.0, "hydraulic", "bidirectional"),),
           "Stores pressurised fluid to smooth out demand or hold pressure with the pump off.",
           keywords=("accumulator", "surge", "damper"))
def _accumulator(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.polyline(
        [_p(centre, size, -0.55, 0.7), _p(centre, size, -0.55, -0.2),
         _p(centre, size, -0.3, -0.65), _p(centre, size, 0.3, -0.65),
         _p(centre, size, 0.55, -0.2), _p(centre, size, 0.55, 0.7)],
        kind="visible", colour=colour, layer=layer, close=True,
    )
    canvas.line(_p(centre, size, -0.55, 0.05), _p(centre, size, 0.55, 0.05),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.0, 0.7), _p(centre, size, 0.0, 1.0),
                kind="visible", colour=colour, layer=layer)


# ---------------------------------------------------------------------------
# Systems biology, SBGN process description
# ---------------------------------------------------------------------------


@_register("macromolecule", "Macromolecule", "protein", "SBGN PD", "biological",
           (Pin("in", -1.0, 0.0, "biological", "input"), Pin("out", 1.0, 0.0, "biological", "output")),
           "A protein or other large molecule, drawn as a rounded box.",
           keywords=("protein", "enzyme", "macromolecule", "receptor", "antibody"),
           aspect=1.6)
def _macromolecule(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
                   label: str = "", **_: Any) -> None:
    canvas.rect(_p(centre, size, -0.95, -0.55), size * 1.9, size * 1.1,
                kind="visible", colour=colour, fill=canvas.theme.paper,
                radius=size * 0.3, layer=layer)
    if label:
        canvas.text(centre, label[:14], size=size * 0.42, anchor="middle",
                    colour=colour, layer=layer, baseline="central")


@_register("simple_chemical", "Simple chemical", "small molecule", "SBGN PD", "biological",
           (Pin("in", -1.0, 0.0, "biological", "input"), Pin("out", 1.0, 0.0, "biological", "output")),
           "A small molecule such as glucose or ATP, drawn as a circle.",
           keywords=("metabolite", "molecule", "substrate", "atp", "glucose", "chemical"))
def _simple_chemical(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
                     label: str = "", **_: Any) -> None:
    canvas.circle(centre, size * 0.8, kind="visible", colour=colour,
                  fill=canvas.theme.paper, layer=layer)
    if label:
        canvas.text(centre, label[:8], size=size * 0.42, anchor="middle",
                    colour=colour, layer=layer, baseline="central")


@_register("process_node", "Process", "reaction", "SBGN PD", "biological",
           (Pin("consume", -1.0, 0.0, "biological", "input"),
            Pin("produce", 1.0, 0.0, "biological", "output"),
            Pin("modulate", 0.0, -1.0, "signal", "input")),
           "Where one thing becomes another. The square is deliberately blank.",
           keywords=("reaction", "process", "conversion", "transformation"))
def _process_node(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.rect(_p(centre, size, -0.35, -0.35), size * 0.7, size * 0.7,
                kind="visible", colour=colour, fill=canvas.theme.paper, layer=layer)
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.35, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.35, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer, marker_end="arrow")


@_register("compartment", "Compartment", "compartment", "SBGN PD", "biological",
           (Pin("in", -1.0, 0.0, "biological", "input"), Pin("out", 1.0, 0.0, "biological", "output")),
           "A membrane-bounded space: a cell, a nucleus, a mitochondrion.",
           keywords=("cell", "compartment", "nucleus", "membrane", "organelle"),
           aspect=2.2)
def _compartment(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
                 label: str = "", **_: Any) -> None:
    canvas.rect(_p(centre, size, -1.1, -0.8), size * 2.2, size * 1.6,
                kind="visible", colour=colour, layer=layer, radius=size * 0.4)
    canvas.rect(_p(centre, size, -1.02, -0.72), size * 2.04, size * 1.44,
                kind="thin", colour=colour, layer=layer, radius=size * 0.34)
    if label:
        canvas.text(_p(centre, size, 0.0, -0.55), label[:18], size=size * 0.4,
                    anchor="middle", colour=colour, layer=layer)


# ---------------------------------------------------------------------------
# Control and general blocks, ISO 14617
# ---------------------------------------------------------------------------


@_register("block", "Block", "stage", "ISO 14617", "signal",
           (Pin("in", -1.0, 0.0, "signal", "input"), Pin("out", 1.0, 0.0, "signal", "output")),
           "One stage that takes something in and puts something out.",
           keywords=("block", "stage", "module", "function", "unit"),
           aspect=1.8)
def _block(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
           label: str = "", **_: Any) -> None:
    canvas.rect(_p(centre, size, -0.95, -0.6), size * 1.9, size * 1.2,
                kind="visible", colour=colour, fill=canvas.theme.paper, layer=layer)
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.95, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.95, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)
    if label:
        canvas.text(centre, label[:16], size=size * 0.4, anchor="middle",
                    colour=colour, layer=layer, baseline="central")


@_register("summing", "Summing junction", "adder", "ISO 14617", "signal",
           (Pin("a", -1.0, 0.0, "signal", "input"), Pin("b", 0.0, -1.0, "signal", "input"),
            Pin("out", 1.0, 0.0, "signal", "output")),
           "Adds signals together, or subtracts one from another.",
           keywords=("sum", "adder", "error", "difference", "comparator"))
def _summing(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.circle(centre, size * 0.55, kind="visible", colour=colour,
                  fill=canvas.theme.paper, layer=layer)
    canvas.line(_p(centre, size, -0.39, -0.39), _p(centre, size, 0.39, 0.39),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, -0.39, 0.39), _p(centre, size, 0.39, -0.39),
                kind="thin", colour=colour, layer=layer)


@_register("gain", "Gain", "multiplier", "ISO 14617", "signal",
           (Pin("in", -1.0, 0.0, "signal", "input"), Pin("out", 1.0, 0.0, "signal", "output")),
           "Multiplies the signal by a fixed amount.",
           keywords=("gain", "scale", "multiplier", "amplify"))
def _gain(canvas: Canvas, centre: Point, size: float, colour: str, layer: str,
          label: str = "K", **_: Any) -> None:
    canvas.polyline(
        [_p(centre, size, -0.6, -0.7), _p(centre, size, 0.7, 0.0), _p(centre, size, -0.6, 0.7)],
        kind="visible", colour=colour, layer=layer, close=True,
    )
    canvas.text(_p(centre, size, -0.18, 0.0), label[:4], size=size * 0.45, anchor="middle",
                colour=colour, layer=layer, baseline="central")
    canvas.line(_p(centre, size, -1.0, 0.0), _p(centre, size, -0.6, 0.0),
                kind="thin", colour=colour, layer=layer)
    canvas.line(_p(centre, size, 0.7, 0.0), _p(centre, size, 1.0, 0.0),
                kind="thin", colour=colour, layer=layer)


@_register("structure", "Structural member", "beam", "ISO 14617", "structural",
           (Pin("a", -1.0, 0.0, "structural", "bidirectional"),
            Pin("b", 1.0, 0.0, "structural", "bidirectional")),
           "A beam, strut or bracket that carries load between two points.",
           keywords=("beam", "strut", "bracket", "frame", "member", "hull", "chassis", "plate"),
           aspect=2.0)
def _structure(canvas: Canvas, centre: Point, size: float, colour: str, layer: str, **_: Any) -> None:
    canvas.rect(_p(centre, size, -1.0, -0.22), size * 2.0, size * 0.44,
                kind="visible", colour=colour, layer=layer)
    canvas.hatch(
        [_p(centre, size, -1.0, -0.22), _p(centre, size, 1.0, -0.22),
         _p(centre, size, 1.0, 0.22), _p(centre, size, -1.0, 0.22)],
        angle=45.0, spacing=size * 0.28, colour=colour, layer=layer, opacity=0.5,
    )


def symbol_for(text: str, *, domain: str = "") -> Symbol:
    """Pick the symbol a part's name, tags and domain point at.

    Falls back to a plain block, because an unlabelled box in the right place
    with the right connections is still a readable schematic, and a wrong
    symbol is worse than a neutral one.
    """
    haystack = str(text or "").lower()
    best: Symbol | None = None
    best_score = 0
    for symbol in SYMBOLS.values():
        score = 0
        for word in symbol.keywords:
            if word in haystack:
                score += len(word)
        if domain and symbol.domain == domain:
            score += 1
        if score > best_score:
            best = symbol
            best_score = score
    if best is not None:
        return best
    if domain in {"fluid", "hydraulic", "pneumatic"}:
        return SYMBOLS["vessel"]
    if domain == "biological":
        return SYMBOLS["macromolecule"]
    if domain == "structural":
        return SYMBOLS["structure"]
    return SYMBOLS["block"]


def draw_symbol(
    canvas: Canvas,
    symbol: Symbol,
    centre: Point,
    size: float,
    *,
    colour: str = "",
    layer: str = "symbols",
    **options: Any,
) -> dict[str, Point]:
    """Draw one symbol and report where its pins ended up on the sheet."""
    ink = colour or canvas.theme.ink
    symbol.render(canvas, centre, size, ink, layer, **options)
    return {pin.name: pin.at(centre, size) for pin in symbol.pins}
