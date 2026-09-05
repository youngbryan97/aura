"""Turning a design into an order and a job: what to buy, make, and do.

A drawing that cannot be acted on is a picture. What makes a design real is
that somebody can take it to a supplier and a bench and end up with the
thing. So this produces the three lists that turn a model into work: what to
buy and from what sort of supplier, what to make and by what process out of
what stock, and in what order to put it together.

The assembly order is derived rather than written. An exploded view already
encodes it: the part that travels furthest comes off last, so it goes on
last. Walking that order backwards gives a sequence where nothing has to be
installed through something already fitted, which is the mistake that makes
a set of instructions unusable halfway through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.engineering.units import Q, Quantity

__all__ = [
    "ShoppingItem",
    "MakeItem",
    "BuildStep",
    "BuildPlan",
    "build_plan",
    "PROCESSES",
]

#: What each making process needs and what it is good for, so a plan can say
#: whether the person reading it can actually do the job.
PROCESSES: dict[str, dict[str, Any]] = {
    "print": {
        "name": "3D printing",
        "tools": ("3D printer",),
        "tolerance": Q(0.2, "mm"),
        "note": "Cheap for one, slow for many. Weak across the layers.",
        "materials": ("pla", "abs", "nylon_66", "peek"),
    },
    "machine": {
        "name": "Machining",
        "tools": ("lathe", "milling machine", "measuring tools"),
        "tolerance": Q(0.02, "mm"),
        "note": "Accurate and strong. Needs a machinist and stock material.",
        "materials": ("al_6061_t6", "al_7075_t6", "steel_1018", "steel_4140",
                      "steel_304", "steel_316", "ti_6al4v", "brass_360", "peek"),
    },
    "cut": {
        "name": "Sheet cutting",
        "tools": ("laser cutter or waterjet", "deburring tools"),
        "tolerance": Q(0.1, "mm"),
        "note": "Fast and cheap for flat parts. Nothing three-dimensional.",
        "materials": ("al_6061_t6", "steel_304", "steel_316", "abs", "polycarbonate",
                      "gfrp", "wood_pine"),
    },
    "fabricate": {
        "name": "Fabrication",
        "tools": ("welder", "clamps", "grinder", "safety equipment"),
        "tolerance": Q(1.0, "mm"),
        "note": "For frames and vessels. Welding heat changes the material near the seam.",
        "materials": ("steel_1018", "steel_304", "steel_316", "al_6061_t6"),
    },
    "mould": {
        "name": "Moulding",
        "tools": ("mould", "press or vacuum pot"),
        "tolerance": Q(0.15, "mm"),
        "note": "Only worth the mould cost past a few hundred parts.",
        "materials": ("silicone_rubber", "nitrile_rubber", "pdms", "abs", "nylon_66"),
    },
    "assemble": {
        "name": "Assembly",
        "tools": ("hand tools",),
        "tolerance": Q(0.5, "mm"),
        "note": "Built up from parts that already exist.",
        "materials": (),
    },
}

#: Roughly how long each kind of step takes a competent person, for a plan
#: that has to be scheduled. These are working estimates, not measurements,
#: and every step that uses one says so.
_STEP_MINUTES: dict[str, float] = {
    "fit": 8.0,
    "fasten": 6.0,
    "connect_electrical": 12.0,
    "connect_fluid": 15.0,
    "connect_data": 5.0,
    "seal": 20.0,
    "test": 25.0,
}


@dataclass(frozen=True, slots=True)
class ShoppingItem:
    """One line on an order."""

    part_id: str
    name: str
    specification: str
    quantity: int
    supplier_class: str = ""
    unit_cost: Quantity | None = None
    lead_time: Quantity | None = None
    standard: str = ""

    def total_cost(self) -> Quantity | None:
        if self.unit_cost is None:
            return None
        return self.unit_cost * self.quantity

    def to_dict(self) -> dict[str, Any]:
        out = {
            "part": self.part_id,
            "name": self.name,
            "specification": self.specification,
            "quantity": self.quantity,
            "supplier_class": self.supplier_class,
            "standard": self.standard,
        }
        if self.unit_cost is not None:
            out["unit_cost"] = self.unit_cost.to_dict()
            out["total_cost"] = self.total_cost().to_dict()
        if self.lead_time is not None:
            out["lead_time"] = self.lead_time.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class MakeItem:
    """One part to be made, and what it takes to make it."""

    part_id: str
    name: str
    process: str
    material: str
    stock: str
    quantity: int
    tools: tuple[str, ...] = ()
    tolerance: Quantity | None = None
    note: str = ""
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {
            "part": self.part_id,
            "name": self.name,
            "process": self.process,
            "material": self.material,
            "stock": self.stock,
            "quantity": self.quantity,
            "tools": list(self.tools),
            "note": self.note,
        }
        if self.tolerance is not None:
            out["tolerance"] = self.tolerance.to_dict()
        if self.warning:
            out["warning"] = self.warning
        return out


@dataclass(frozen=True, slots=True)
class BuildStep:
    """One instruction, in order."""

    number: int
    action: str
    detail: str
    parts: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    check: str = ""
    minutes: float = 0.0
    kind: str = "fit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "action": self.action,
            "detail": self.detail,
            "parts": list(self.parts),
            "tools": list(self.tools),
            "check": self.check,
            "minutes": self.minutes,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Everything needed to go from this design to the thing."""

    buy: tuple[ShoppingItem, ...] = ()
    make: tuple[MakeItem, ...] = ()
    steps: tuple[BuildStep, ...] = ()
    tools: tuple[str, ...] = ()
    unsourced: tuple[str, ...] = ()
    total_cost: Quantity | None = None
    longest_lead: Quantity | None = None
    build_minutes: float = 0.0

    @property
    def actionable(self) -> bool:
        return not self.unsourced and bool(self.steps)

    def plain(self) -> str:
        if self.unsourced:
            return (
                f"{len(self.unsourced)} parts do not say how they would be obtained, so "
                "this cannot be ordered as it stands: "
                + ", ".join(self.unsourced[:4])
                + "."
            )
        pieces = []
        if self.buy:
            cost = f" for about {self.total_cost.text()}" if self.total_cost else ""
            lead = (
                f", the longest lead being {self.longest_lead.text()}"
                if self.longest_lead
                else ""
            )
            pieces.append(f"{len(self.buy)} items to buy{cost}{lead}")
        if self.make:
            processes = sorted({item.process for item in self.make})
            pieces.append(
                f"{len(self.make)} parts to make by "
                + " and ".join(PROCESSES[p]["name"].lower() for p in processes if p in PROCESSES)
            )
        if self.steps:
            hours = self.build_minutes / 60.0
            pieces.append(
                f"{len(self.steps)} assembly steps, roughly "
                + (f"{hours:.1f} hours" if hours >= 1 else f"{self.build_minutes:.0f} minutes")
                + " of work"
            )
        return "This can be built: " + "; ".join(pieces) + "."

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "actionable": self.actionable,
            "buy": [item.to_dict() for item in self.buy],
            "make": [item.to_dict() for item in self.make],
            "steps": [step.to_dict() for step in self.steps],
            "tools": list(self.tools),
            "unsourced": list(self.unsourced),
            "build_minutes": self.build_minutes,
            "plain": self.plain(),
        }
        if self.total_cost is not None:
            out["total_cost"] = self.total_cost.to_dict()
        if self.longest_lead is not None:
            out["longest_lead"] = self.longest_lead.to_dict()
        return out


def _stock_for(part) -> str:
    """The raw material to order, sized from the part's own bounding box."""
    if part.solid is None or part.material is None:
        return ""
    kind = part.solid.kind
    params = part.solid.parameters()
    material = part.material.name

    def mm(key: str, default: float = 0.0) -> float:
        value = params.get(key)
        return float(value.value) * 1000.0 if value is not None else default

    # A machining allowance: stock has to be bigger than the finished part.
    allowance = 3.0
    if kind in {"cylinder", "cone"}:
        return (
            f"{material} round bar, {mm('diameter') + allowance:.0f} mm diameter x "
            f"{mm('height') + allowance * 2:.0f} mm"
        )
    if kind == "tube":
        return (
            f"{material} tube, {mm('outer_diameter') + allowance:.0f} mm outside x "
            f"{mm('wall'):.1f} mm wall x {mm('height') + allowance * 2:.0f} mm"
        )
    if kind in {"box", "plate"}:
        thickness = mm("thickness") or mm("height")
        return (
            f"{material} plate, {mm('width') + allowance:.0f} x "
            f"{mm('depth') + allowance:.0f} x {thickness + allowance:.0f} mm"
        )
    if kind in {"sphere", "dome", "ellipsoid", "capsule"}:
        low, high = part.solid.mesh().bounds()
        span = (np.asarray(high) - np.asarray(low)) * 1000.0
        return (
            f"{material} billet, {span[0] + allowance:.0f} x {span[1] + allowance:.0f} x "
            f"{span[2] + allowance:.0f} mm"
        )
    low, high = part.solid.mesh().bounds()
    span = (np.asarray(high) - np.asarray(low)) * 1000.0
    return (
        f"{material} stock, {span[0] + allowance:.0f} x {span[1] + allowance:.0f} x "
        f"{span[2] + allowance:.0f} mm"
    )


def _assembly_order(design) -> list:
    """Parts in the order they go together: nearest the centre first.

    The explode vector already says which way a part comes off, so the part
    that travels furthest is the last one on. Installing in that order means
    nothing has to be fitted through something already in place.
    """
    scored = []
    for part in design.parts:
        if part.solid is None:
            scored.append((0.0, part))
            continue
        reach = float(np.linalg.norm(np.asarray(part.explode, dtype=float)))
        position = float(np.linalg.norm(np.asarray(part.placement.position, dtype=float)))
        scored.append((reach + position, part))
    scored.sort(key=lambda entry: entry[0])
    return [part for _score, part in scored]


def _connection_steps(design, number: int, placed: set[str]) -> list[BuildStep]:
    """One step per connection, once both of its parts are in place."""
    from core.engineering.domains import domain as get_domain

    steps: list[BuildStep] = []
    for link in design.connections:
        source_id = link.source.split(".")[0]
        target_id = link.target.split(".")[0]
        if source_id not in placed or target_id not in placed:
            continue
        if source_id == target_id:
            # A part joined to itself is a modelling error, not a job. The
            # verifier reports it; there is nothing to instruct anyone to do.
            continue
        source = design.part(source_id)
        target = design.part(target_id)
        if source is None or target is None:
            continue
        spec = get_domain(link.domain)
        kind = (
            "connect_electrical"
            if link.domain == "electrical"
            else "connect_fluid"
            if link.domain in {"fluid", "hydraulic", "pneumatic"}
            else "connect_data"
        )
        detail = (
            f"Run the {spec.carries or spec.name.lower()} line from the "
            f"{source.lay_name or source.name.lower()} to the "
            f"{target.lay_name or target.name.lower()}."
        )
        if link.through is not None:
            detail += f" It carries {link.through.text()}"
            if link.across is not None:
                detail += f" at {link.across.text()}"
            detail += "."
        found = design.find_port(link.source)
        if found is not None and found[1].interface:
            detail += f" The fitting is {found[1].interface}."
        check = ""
        if link.domain == "electrical":
            check = (
                "Before powering anything, check continuity end to end and check that "
                "nothing is shorted to the case."
            )
        elif link.domain in {"fluid", "hydraulic", "pneumatic"}:
            check = "Pressure-test the joint before the system runs for real."
        steps.append(
            BuildStep(
                number=number + len(steps),
                action=f"Connect {source.lay_name or source.name} to "
                f"{target.lay_name or target.name}",
                detail=detail,
                parts=(source_id, target_id),
                tools=("wire strippers", "crimp tool") if link.domain == "electrical"
                else ("spanners", "thread sealant") if link.domain in {"fluid", "hydraulic", "pneumatic"}
                else ("hand tools",),
                check=check,
                minutes=_STEP_MINUTES[kind],
                kind=kind,
            )
        )
    return steps


def build_plan(design, findings: tuple = ()) -> BuildPlan:
    """Work out what to buy, what to make, and what order to do it in."""
    buy: list[ShoppingItem] = []
    make: list[MakeItem] = []
    unsourced: list[str] = []
    tools: set[str] = set()

    for part in design.parts:
        sourcing = part.sourcing
        if sourcing.method == "unspecified" or not sourcing.specification:
            unsourced.append(part.lay_name or part.name)
            continue
        if sourcing.buyable:
            buy.append(ShoppingItem(
                part_id=part.id,
                name=part.lay_name or part.name,
                specification=sourcing.specification,
                quantity=part.quantity,
                supplier_class=sourcing.supplier_class,
                unit_cost=sourcing.unit_cost,
                lead_time=sourcing.lead_time,
                standard=sourcing.standard,
            ))
            continue
        process = PROCESSES.get(sourcing.method, PROCESSES["assemble"])
        warning = ""
        if part.material is not None and process["materials"]:
            if part.material.key not in process["materials"]:
                warning = (
                    f"{part.material.name} is not normally worked by "
                    f"{process['name'].lower()}. Either the process or the material has "
                    "to change."
                )
        make.append(MakeItem(
            part_id=part.id,
            name=part.lay_name or part.name,
            process=sourcing.method,
            material=part.material.name if part.material else "not stated",
            stock=_stock_for(part),
            quantity=part.quantity,
            tools=tuple(sourcing.tools_required) or tuple(process["tools"]),
            tolerance=process["tolerance"],
            note=sourcing.process_notes or process["note"],
            warning=warning,
        ))
        tools.update(sourcing.tools_required or process["tools"])

    total = 0.0
    priced = 0
    longest = 0.0
    for item in buy:
        if item.unit_cost is not None:
            total += float(item.unit_cost.value) * item.quantity
            priced += 1
        if item.lead_time is not None:
            longest = max(longest, float(item.lead_time.value))
    for item in make:
        part = design.part(item.part_id)
        if part is not None and part.material is not None and part.material.cost_per_kg:
            mass = part.mass()
            if mass is not None:
                total += float(mass.value) * float(part.material.cost_per_kg.value)
                priced += 1

    steps: list[BuildStep] = []
    placed: set[str] = set()
    for part in _assembly_order(design):
        number = len(steps) + 1
        detail = part.function or "Fit it in place."
        if part.solid is not None:
            detail = f"{detail.rstrip('.')}. It is {part.solid.describe()}"
            mass = part.mass()
            if mass is not None and float(mass.value) > 20.0:
                detail += f", and at {mass.text()} it needs two people or a hoist"
            detail += "."
        steps.append(BuildStep(
            number=number,
            action=f"Fit the {part.lay_name or part.name}",
            detail=detail,
            parts=(part.id,),
            tools=("hand tools",),
            check=(
                "Check it sits square and that nothing is trapped underneath it."
                if part.quantity == 1
                else f"All {part.quantity} go in at this stage; check they match each other."
            ),
            minutes=_STEP_MINUTES["fit"] * part.quantity,
            kind="fit",
        ))
        placed.add(part.id)
        steps.extend(_connection_steps_for(design, part.id, placed, len(steps) + 1))

    # A last step that says how to know it worked, taken from the
    # requirements rather than invented.
    for requirement in design.requirements:
        steps.append(BuildStep(
            number=len(steps) + 1,
            action=f"Verify {requirement.id}",
            detail=requirement.plain or requirement.statement,
            tools=("measuring equipment",),
            check=(
                f"The design predicts this passes; the build has to be measured to "
                f"confirm it. Analysis is not a test."
            ),
            minutes=_STEP_MINUTES["test"],
            kind="test",
        ))

    tools.update(tool for step in steps for tool in step.tools)
    return BuildPlan(
        buy=tuple(buy),
        make=tuple(make),
        steps=tuple(steps),
        tools=tuple(sorted(tools)),
        unsourced=tuple(unsourced),
        total_cost=Q(total, "count") if priced and priced == len(buy) + len(make) else None,
        longest_lead=Q(longest, "s").as_("day") if longest else None,
        build_minutes=sum(step.minutes for step in steps),
    )


def _connection_steps_for(design, part_id: str, placed: set[str], number: int) -> list[BuildStep]:
    """Connections that become possible now this part is in place."""
    from core.engineering.domains import domain as get_domain

    steps: list[BuildStep] = []
    for link in design.connections:
        source_id = link.source.split(".")[0]
        target_id = link.target.split(".")[0]
        if part_id not in {source_id, target_id}:
            continue
        if source_id not in placed or target_id not in placed:
            continue
        made = _connection_steps(design, number + len(steps), {source_id, target_id})
        for step in made:
            if step.parts == (source_id, target_id):
                steps.append(step)
                break
    return steps
