"""The design itself: parts, the ports they meet at, and what runs between.

This is the single object every drawing, analysis, export and build sheet is
made from. A drawing is a projection of it, a bill of materials is a
grouping of it, and an assembly procedure is a walk over its connection
graph. Nothing downstream invents anything the model does not hold, which is
why a callout cannot say a number the model cannot produce.

The shape is deliberately close to what model-based systems engineering
settled on. Parts have PORTS rather than loose wires, because an interface
you can name is an interface you can check. Connections join ports in one
:mod:`~core.engineering.domains` domain, so the checker knows which
conservation law applies. Requirements point at the analysis that satisfies
them, so a design either meets a requirement, fails it, or is honest that
nothing has been run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

from core.engineering.domains import Domain
from core.engineering.domains import domain as get_domain
from core.engineering.geometry import Placement, Solid, solid_from_spec
from core.engineering.materials import Material
from core.engineering.materials import material as get_material
from core.engineering.units import Q, Quantity

__all__ = [
    "Port",
    "Sourcing",
    "Part",
    "Connection",
    "Requirement",
    "Subsystem",
    "Design",
    "Net",
    "design_from_brief",
    "slug",
]


def slug(text: str, *, limit: int = 48) -> str:
    """A stable identifier from a name, for ids and filenames."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return (cleaned or "item")[:limit]


def _quantity(value: Any, *, default_unit: str = "") -> Quantity | None:
    if value is None or value == "":
        return None
    if isinstance(value, Quantity):
        return value
    if isinstance(value, dict) and "value" in value:
        return Q(float(value["value"]), str(value.get("unit") or default_unit))
    return Q(value, default_unit) if not isinstance(value, str) else Q(value)


@dataclass(frozen=True, slots=True)
class Port:
    """A named place where one part meets another.

    ``across`` is the port's working potential — the voltage a terminal sits
    at, the pressure a nozzle sees. ``through`` is its rating — the current
    the terminal carries, the flow the nozzle passes. Both are optional,
    because a design in progress knows some of its numbers and not others,
    and the checker reports what is missing rather than filling it in.
    """

    id: str
    name: str
    domain: str
    role: str = "bidirectional"
    across: Quantity | None = None
    through: Quantity | None = None
    #: A thread, a connector series, a pipe size: what physically mates here.
    interface: str = ""
    lay_name: str = ""
    #: Where the port sits on the part, in part-local metres.
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def domain_spec(self) -> Domain:
        return get_domain(self.domain)

    def power(self) -> Quantity | None:
        """The power this port is rated to pass, when both variables are known."""
        if self.across is None or self.through is None:
            return None
        spec = self.domain_spec
        if not spec.conserved:
            return None
        return (self.across * self.through).as_("W")

    def describe(self) -> str:
        spec = self.domain_spec
        parts = [self.lay_name or self.name]
        if self.across is not None:
            parts.append(f"{spec.across_name} {self.across.text()}")
        if self.through is not None:
            parts.append(f"{spec.through_name} {self.through.text()}")
        if self.interface:
            parts.append(self.interface)
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "role": self.role,
            "interface": self.interface,
            "lay_name": self.lay_name or self.name,
            "position": list(self.position),
        }
        if self.across is not None:
            out["across"] = self.across.to_dict()
        if self.through is not None:
            out["through"] = self.through.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class Sourcing:
    """How this part comes to exist, so the design can be acted on.

    A schematic that cannot be turned into an order and a job is a picture.
    ``method`` says buy, print, machine, cut, fabricate or assemble;
    ``specification`` is what to write on the order or the job sheet.
    """

    method: str = "unspecified"
    specification: str = ""
    standard: str = ""
    supplier_class: str = ""
    unit_cost: Quantity | None = None
    lead_time: Quantity | None = None
    process_notes: str = ""
    tools_required: tuple[str, ...] = ()

    @property
    def buyable(self) -> bool:
        return self.method in {"buy", "off_the_shelf", "stock"}

    @property
    def makeable(self) -> bool:
        return self.method in {"print", "machine", "cut", "fabricate", "mould", "assemble"}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "specification": self.specification,
            "standard": self.standard,
            "supplier_class": self.supplier_class,
            "process_notes": self.process_notes,
            "tools_required": list(self.tools_required),
            "buyable": self.buyable,
            "makeable": self.makeable,
        }
        if self.unit_cost is not None:
            out["unit_cost"] = self.unit_cost.to_dict()
        if self.lead_time is not None:
            out["lead_time"] = self.lead_time.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class Part:
    """One physical thing in the design."""

    id: str
    name: str
    #: What it does, in a sentence a non-engineer reads without stopping.
    function: str = ""
    #: What to call it when the engineering name would not help.
    lay_name: str = ""
    solid: Solid | None = None
    material: Material | None = None
    quantity: int = 1
    ports: tuple[Port, ...] = ()
    placement: Placement = field(default_factory=Placement)
    #: Which way this part travels in an exploded view, in metres.
    explode: tuple[float, float, float] = (0.0, 0.0, 1.0)
    subsystem: str = ""
    #: The drawing's find number, ASME Y14.34 style. Assigned by the design.
    balloon: int = 0
    #: A reference designator where the discipline uses one: R1, C3, V-101.
    designator: str = ""
    sourcing: Sourcing = field(default_factory=Sourcing)
    #: Measured or specified figures that are properties of the part rather
    #: than of its shape: a motor's rated torque, a battery's capacity.
    ratings: dict[str, Quantity] = field(default_factory=dict)
    notes: str = ""
    tags: tuple[str, ...] = ()

    def mass(self) -> Quantity | None:
        if self.solid is None or self.material is None:
            declared = self.ratings.get("mass")
            return declared * self.quantity if declared is not None else None
        return self.solid.mass(self.material.density) * self.quantity

    def volume(self) -> Quantity | None:
        if self.solid is None:
            return None
        return self.solid.volume() * self.quantity

    def port(self, port_id: str) -> Port | None:
        for entry in self.ports:
            if entry.id == port_id or entry.name == port_id:
                return entry
        return None

    def describe(self) -> str:
        """One line about this part, written for a reader without the jargon."""
        head = self.lay_name or self.name
        pieces = [head]
        if self.function:
            pieces.append(self.function.rstrip("."))
        if self.solid is not None:
            pieces.append(self.solid.describe())
        if self.material is not None:
            pieces.append(f"made of {self.material.name.lower()}")
        mass = self.mass()
        if mass is not None:
            pieces.append(f"weighing {mass.text()}")
        return ", ".join(pieces) + "."

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "lay_name": self.lay_name or self.name,
            "function": self.function,
            "quantity": self.quantity,
            "subsystem": self.subsystem,
            "balloon": self.balloon,
            "designator": self.designator,
            "ports": [p.to_dict() for p in self.ports],
            "sourcing": self.sourcing.to_dict(),
            "notes": self.notes,
            "tags": list(self.tags),
            "placement": {
                "position": list(self.placement.position),
                "rotation": list(self.placement.rotation),
            },
            "explode": list(self.explode),
            "ratings": {k: v.to_dict() for k, v in self.ratings.items()},
        }
        if self.solid is not None:
            out["solid"] = self.solid.to_dict()
        if self.material is not None:
            out["material"] = {
                "key": self.material.key,
                "name": self.material.name,
                "feels_like": self.material.feels_like,
            }
        mass = self.mass()
        if mass is not None:
            out["mass"] = mass.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class Connection:
    """One link between two ports, in one domain."""

    id: str
    source: str
    target: str
    domain: str
    #: What the link is rated to carry, in the domain's through variable.
    through: Quantity | None = None
    #: The potential the link sits at, in the domain's across variable.
    across: Quantity | None = None
    label: str = ""
    #: A sentence explaining what travels along this link and why.
    explanation: str = ""
    medium: str = ""
    notes: str = ""

    @property
    def domain_spec(self) -> Domain:
        return get_domain(self.domain)

    def power(self) -> Quantity | None:
        if self.across is None or self.through is None:
            return None
        if not self.domain_spec.conserved:
            return None
        return (self.across * self.through).as_("W")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "from": self.source,
            "to": self.target,
            "domain": self.domain,
            "label": self.label,
            "explanation": self.explanation,
            "medium": self.medium,
            "notes": self.notes,
        }
        if self.through is not None:
            out["through"] = self.through.to_dict()
        if self.across is not None:
            out["across"] = self.across.to_dict()
        power = self.power()
        if power is not None:
            out["power"] = power.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class Requirement:
    """Something the design has to do, and how anyone would know it does.

    A requirement with no ``check`` is a wish. The verifier reports those as
    unverified rather than passing them, because a design that claims to meet
    a requirement nothing tested is the failure this package exists to stop.
    """

    id: str
    statement: str
    #: The name of a computed finding whose value settles this.
    check: str = ""
    target: Quantity | None = None
    comparison: str = ">="
    #: Filled in by the verifier: pass, fail or unverified.
    verdict: str = "unverified"
    actual: Quantity | None = None
    margin: float | None = None
    rationale: str = ""
    plain: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "statement": self.statement,
            "plain": self.plain or self.statement,
            "check": self.check,
            "comparison": self.comparison,
            "verdict": self.verdict,
            "rationale": self.rationale,
        }
        if self.target is not None:
            out["target"] = self.target.to_dict()
        if self.actual is not None:
            out["actual"] = self.actual.to_dict()
        if self.margin is not None:
            out["margin"] = self.margin
        return out


@dataclass(frozen=True, slots=True)
class Subsystem:
    """A named group of parts that does one job."""

    id: str
    name: str
    purpose: str = ""
    lay_name: str = ""
    colour: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "lay_name": self.lay_name or self.name,
            "purpose": self.purpose,
            "colour": self.colour,
        }


@dataclass(frozen=True, slots=True)
class Net:
    """Every port joined at one electrical, fluid or thermal node."""

    id: str
    domain: str
    ports: tuple[str, ...]
    across: Quantity | None = None
    through_sum: Quantity | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "domain": self.domain, "ports": list(self.ports)}
        if self.across is not None:
            out["across"] = self.across.to_dict()
        if self.through_sum is not None:
            out["through_sum"] = self.through_sum.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class Design:
    """A whole design: what it is for, what it is made of, and what holds."""

    name: str
    purpose: str = ""
    discipline: str = "mechanical"
    parts: tuple[Part, ...] = ()
    connections: tuple[Connection, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    subsystems: tuple[Subsystem, ...] = ()
    #: The conditions the design has to work in: depth, temperature, supply.
    #: Values are quantities where they are measurable and plain strings
    #: where they name something — the working fluid, the design maturity.
    environment: dict[str, Any] = field(default_factory=dict)
    #: Free text the cortex wrote about intent. Never a source of numbers.
    rationale: str = ""
    scale_note: str = ""
    revision: str = "A"
    author: str = "Aura"
    standard: str = "ISO 128 / ASME Y14"

    # -- lookups ---------------------------------------------------------
    def part(self, part_id: str) -> Part | None:
        for entry in self.parts:
            if entry.id == part_id:
                return entry
        return None

    def find_port(self, reference: str) -> tuple[Part, Port] | None:
        """Resolve ``"battery.positive"`` or a bare port id to its part."""
        text = str(reference or "")
        if "." in text:
            part_id, _, port_id = text.partition(".")
            found = self.part(part_id)
            if found is not None:
                port = found.port(port_id)
                if port is not None:
                    return (found, port)
        for entry in self.parts:
            port = entry.port(text)
            if port is not None:
                return (entry, port)
        return None

    def subsystem(self, subsystem_id: str) -> Subsystem | None:
        for entry in self.subsystems:
            if entry.id == subsystem_id:
                return entry
        return None

    def parts_in(self, subsystem_id: str) -> tuple[Part, ...]:
        return tuple(p for p in self.parts if p.subsystem == subsystem_id)

    # -- roll-ups --------------------------------------------------------
    def total_mass(self) -> Quantity | None:
        masses = [p.mass() for p in self.parts]
        known = [m for m in masses if m is not None]
        if not known:
            return None
        total = known[0]
        for entry in known[1:]:
            total = total + entry
        return total

    def total_cost(self) -> Quantity | None:
        """What the parts cost, when every part says what it costs.

        A partial total is worse than no total, so this returns nothing
        unless every part carries either a unit cost or a material price.
        """
        running = 0.0
        for part in self.parts:
            unit = part.sourcing.unit_cost
            if unit is not None:
                running += float(unit.value) * part.quantity
                continue
            if part.material is not None and part.material.cost_per_kg is not None:
                mass = part.mass()
                if mass is None:
                    return None
                running += float(mass.value) * float(part.material.cost_per_kg.value)
                continue
            return None
        return Q(running, "count")

    def nets(self) -> tuple[Net, ...]:
        """Group connected ports into nodes, one group per domain."""
        parent: dict[str, str] = {}

        def find(key: str) -> str:
            parent.setdefault(key, key)
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        by_domain: dict[str, list[Connection]] = defaultdict(list)
        for link in self.connections:
            by_domain[link.domain].append(link)

        nets: list[Net] = []
        for domain_key, links in by_domain.items():
            parent = {}
            for link in links:
                union(f"{domain_key}|{link.source}", f"{domain_key}|{link.target}")
            groups: dict[str, list[str]] = defaultdict(list)
            for link in links:
                for endpoint in (link.source, link.target):
                    key = f"{domain_key}|{endpoint}"
                    root = find(key)
                    if endpoint not in groups[root]:
                        groups[root].append(endpoint)
            for index, (_root, ports) in enumerate(sorted(groups.items()), start=1):
                nets.append(
                    Net(
                        id=f"{domain_key}_net_{index}",
                        domain=domain_key,
                        ports=tuple(sorted(ports)),
                    )
                )
        return tuple(nets)

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The whole assembly's extent in metres, placements applied."""
        lows: list[Any] = []
        highs: list[Any] = []
        for part in self.parts:
            if part.solid is None:
                continue
            mesh = part.solid.mesh().transformed(part.placement)
            low, high = mesh.bounds()
            lows.append(low)
            highs.append(high)
        if not lows:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        import numpy as np

        low = np.min(np.array(lows), axis=0)
        high = np.max(np.array(highs), axis=0)
        return (tuple(float(v) for v in low), tuple(float(v) for v in high))  # type: ignore[return-value]

    def characteristic_length(self) -> Quantity:
        low, high = self.bounds()
        span = max(high[i] - low[i] for i in range(3))
        return Q(span if span > 0 else 1.0, "m")

    # -- identity --------------------------------------------------------
    def fingerprint(self) -> str:
        """A stable hash of the model, so a drawing can name its source."""
        payload = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def with_requirements(self, requirements: tuple[Requirement, ...]) -> Design:
        return replace(self, requirements=requirements)

    def with_parts(self, parts: tuple[Part, ...]) -> Design:
        return replace(self, parts=parts)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "purpose": self.purpose,
            "discipline": self.discipline,
            "revision": self.revision,
            "author": self.author,
            "standard": self.standard,
            "rationale": self.rationale,
            "scale_note": self.scale_note,
            "parts": [p.to_dict() for p in self.parts],
            "connections": [c.to_dict() for c in self.connections],
            "requirements": [r.to_dict() for r in self.requirements],
            "subsystems": [s.to_dict() for s in self.subsystems],
            "environment": {
                k: (v.to_dict() if isinstance(v, Quantity) else v)
                for k, v in self.environment.items()
            },
            "nets": [n.to_dict() for n in self.nets()],
        }
        mass = self.total_mass()
        if mass is not None:
            out["total_mass"] = mass.to_dict()
        if include_fingerprint:
            out["fingerprint"] = self.fingerprint()
        return out


# ---------------------------------------------------------------------------
# Building a design from a brief
# ---------------------------------------------------------------------------

#: The keys a brief may use for each field, so a plan written in ordinary
#: words resolves without the cortex having to learn a schema by heart.
_PART_ALIASES: dict[str, str] = {
    "part": "name",
    "title": "name",
    "does": "function",
    "purpose": "function",
    "role": "function",
    "plain_name": "lay_name",
    "common_name": "lay_name",
    "shape": "solid",
    "geometry": "solid",
    "made_of": "material",
    "count": "quantity",
    "qty": "quantity",
    "group": "subsystem",
    "system": "subsystem",
    "ref": "designator",
    "reference": "designator",
    "connections": "ports",
    "interfaces": "ports",
}


def _read_ports(entries: Any, part_id: str) -> tuple[Port, ...]:
    ports: list[Port] = []
    for index, entry in enumerate(entries or (), start=1):
        if isinstance(entry, str):
            ports.append(
                Port(id=f"{part_id}.{slug(entry)}", name=entry, domain="signal")
            )
            continue
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("id") or f"port_{index}")
        port_id = str(entry.get("id") or f"{part_id}.{slug(name)}")
        if "." not in port_id:
            port_id = f"{part_id}.{port_id}"
        ports.append(
            Port(
                id=port_id,
                name=name,
                domain=get_domain(entry.get("domain") or "signal").key,
                role=str(entry.get("role") or "bidirectional"),
                across=_quantity(entry.get("across") or entry.get("voltage")
                                 or entry.get("pressure") or entry.get("temperature")),
                through=_quantity(entry.get("through") or entry.get("current")
                                  or entry.get("flow") or entry.get("force")),
                interface=str(entry.get("interface") or entry.get("fitting") or ""),
                lay_name=str(entry.get("lay_name") or entry.get("plain") or ""),
                position=tuple(
                    float(v) for v in (entry.get("position") or (0.0, 0.0, 0.0))
                ),  # type: ignore[arg-type]
            )
        )
    return tuple(ports)


def _read_sourcing(entry: Any) -> Sourcing:
    if not isinstance(entry, dict):
        return Sourcing()
    return Sourcing(
        method=str(entry.get("method") or entry.get("make_or_buy") or "unspecified").lower(),
        specification=str(entry.get("specification") or entry.get("spec") or ""),
        standard=str(entry.get("standard") or ""),
        supplier_class=str(entry.get("supplier_class") or entry.get("supplier") or ""),
        unit_cost=_quantity(entry.get("unit_cost") or entry.get("cost"), default_unit="count"),
        lead_time=_quantity(entry.get("lead_time"), default_unit="day"),
        process_notes=str(entry.get("process_notes") or entry.get("notes") or ""),
        tools_required=tuple(str(t) for t in (entry.get("tools_required") or ())),
    )


def _read_part(entry: dict[str, Any], index: int) -> Part:
    normalised: dict[str, Any] = {}
    for key, value in entry.items():
        normalised[_PART_ALIASES.get(str(key).lower(), str(key).lower())] = value
    name = str(normalised.get("name") or f"part {index}")
    part_id = slug(normalised.get("id") or name)
    solid_spec = normalised.get("solid")
    solid = None
    if isinstance(solid_spec, dict):
        solid = solid_from_spec(solid_spec)
    elif isinstance(solid_spec, Solid):
        solid = solid_spec
    material_name = normalised.get("material")
    material = None
    if isinstance(material_name, Material):
        material = material_name
    elif material_name:
        material = get_material(str(material_name))
    placement_spec = normalised.get("placement") or {}
    placement = Placement(
        position=tuple(float(v) for v in (placement_spec.get("position") or (0, 0, 0))),  # type: ignore[arg-type]
        rotation=tuple(float(v) for v in (placement_spec.get("rotation") or (0, 0, 0))),  # type: ignore[arg-type]
    )
    ratings = {}
    for key, value in (normalised.get("ratings") or {}).items():
        quantity = _quantity(value)
        if quantity is not None:
            ratings[str(key)] = quantity
    return Part(
        id=part_id,
        name=name,
        function=str(normalised.get("function") or ""),
        lay_name=str(normalised.get("lay_name") or ""),
        solid=solid,
        material=material,
        quantity=int(normalised.get("quantity") or 1),
        ports=_read_ports(normalised.get("ports"), part_id),
        placement=placement,
        explode=tuple(float(v) for v in (normalised.get("explode") or (0, 0, 1))),  # type: ignore[arg-type]
        subsystem=slug(normalised.get("subsystem") or "") if normalised.get("subsystem") else "",
        designator=str(normalised.get("designator") or ""),
        sourcing=_read_sourcing(normalised.get("sourcing")),
        ratings=ratings,
        notes=str(normalised.get("notes") or ""),
        tags=tuple(str(t) for t in (normalised.get("tags") or ())),
    )


def _read_connection(entry: dict[str, Any], index: int) -> Connection:
    source = str(entry.get("from") or entry.get("source") or "")
    target = str(entry.get("to") or entry.get("target") or "")
    domain_key = get_domain(entry.get("domain") or "signal").key
    return Connection(
        id=str(entry.get("id") or f"link_{index}"),
        source=source,
        target=target,
        domain=domain_key,
        through=_quantity(
            entry.get("through") or entry.get("current") or entry.get("flow")
            or entry.get("force") or entry.get("heat")
        ),
        across=_quantity(
            entry.get("across") or entry.get("voltage") or entry.get("pressure")
            or entry.get("temperature")
        ),
        label=str(entry.get("label") or ""),
        explanation=str(entry.get("explanation") or entry.get("plain") or ""),
        medium=str(entry.get("medium") or ""),
        notes=str(entry.get("notes") or ""),
    )


def _read_requirement(entry: dict[str, Any], index: int) -> Requirement:
    return Requirement(
        id=str(entry.get("id") or f"REQ-{index:03d}"),
        statement=str(entry.get("statement") or entry.get("requirement") or ""),
        check=str(entry.get("check") or entry.get("verified_by") or ""),
        target=_quantity(entry.get("target") or entry.get("value")),
        comparison=str(entry.get("comparison") or entry.get("op") or ">="),
        rationale=str(entry.get("rationale") or ""),
        plain=str(entry.get("plain") or ""),
    )


def design_from_brief(brief: dict[str, Any]) -> Design:
    """Turn a design brief into a model, resolving every name and number.

    The brief is the cortex's contribution: what to build, which parts, what
    they are for, what has to hold. Every dimension in it goes through the
    unit parser and every material through the materials table, so a brief
    that names a shape nothing can build or a material nothing knows fails
    here with a message that says which one.
    """
    if not isinstance(brief, dict):
        raise TypeError("a design brief must be a mapping")
    parts = tuple(
        _read_part(entry, index)
        for index, entry in enumerate(brief.get("parts") or (), start=1)
        if isinstance(entry, dict)
    )
    # Balloon numbers follow the part order, which is what an assembly
    # drawing's find numbers do.
    parts = tuple(replace(part, balloon=index) for index, part in enumerate(parts, start=1))
    connections = tuple(
        _read_connection(entry, index)
        for index, entry in enumerate(brief.get("connections") or (), start=1)
        if isinstance(entry, dict)
    )
    requirements = tuple(
        _read_requirement(entry, index)
        for index, entry in enumerate(brief.get("requirements") or (), start=1)
        if isinstance(entry, dict)
    )
    subsystems = tuple(
        Subsystem(
            id=slug(entry.get("id") or entry.get("name") or f"group_{index}"),
            name=str(entry.get("name") or f"group {index}"),
            purpose=str(entry.get("purpose") or ""),
            lay_name=str(entry.get("lay_name") or ""),
            colour=str(entry.get("colour") or entry.get("color") or ""),
        )
        for index, entry in enumerate(brief.get("subsystems") or (), start=1)
        if isinstance(entry, dict)
    )
    environment: dict[str, Any] = {}
    for key, value in (brief.get("environment") or {}).items():
        try:
            quantity = _quantity(value)
        except (ValueError, KeyError):
            # A named condition rather than a measured one: the working
            # fluid, the design maturity, the factor set. Kept as written.
            environment[str(key)] = value
            continue
        environment[str(key)] = quantity if quantity is not None else value
    return Design(
        name=str(brief.get("name") or brief.get("title") or "Untitled design"),
        purpose=str(brief.get("purpose") or brief.get("goal") or ""),
        discipline=str(brief.get("discipline") or "mechanical").lower(),
        parts=parts,
        connections=connections,
        requirements=requirements,
        subsystems=subsystems,
        environment=environment,
        rationale=str(brief.get("rationale") or ""),
        revision=str(brief.get("revision") or "A"),
        author=str(brief.get("author") or "Aura"),
    )
