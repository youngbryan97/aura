"""Power in, power out, and whether the wire between them is big enough.

An electrical design fails in three ordinary ways, and all three are
arithmetic. The supply cannot deliver what the loads draw. The wire is thin
enough that the voltage at the far end is not the voltage at the near end.
The battery runs out sooner than anybody said.

Kirchhoff's current law is checked too, in
:mod:`core.engineering.analysis.conservation`, where it shares one
implementation with the mass balance and the heat balance because they are
the same law written in three domains.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.units import Q

#: Copper resistivity at 20 C, and its temperature coefficient. A hot wire
#: has more resistance than a cold one, by enough to matter over a run.
_COPPER_RESISTIVITY = 1.68e-8
_COPPER_ALPHA = 0.00393

#: American Wire Gauge: gauge number to conductor area in square millimetres
#: and a conservative free-air continuous current, per NEC 310.15 chassis
#: wiring practice. The current figure is the one a design should size to.
AWG_TABLE: tuple[tuple[int, float, float], ...] = (
    (0, 53.5, 150.0),
    (2, 33.6, 95.0),
    (4, 21.2, 60.0),
    (6, 13.3, 37.0),
    (8, 8.37, 24.0),
    (10, 5.26, 15.0),
    (12, 3.31, 9.3),
    (14, 2.08, 5.9),
    (16, 1.31, 3.7),
    (18, 0.823, 2.3),
    (20, 0.518, 1.5),
    (22, 0.326, 0.92),
    (24, 0.205, 0.577),
    (26, 0.129, 0.361),
)


def wire_for(current: float, *, headroom: float = 1.25) -> tuple[int, float, float]:
    """The smallest AWG that carries this current with headroom to spare."""
    wanted = current * headroom
    for gauge, area, rating in reversed(AWG_TABLE):
        if rating >= wanted:
            return (gauge, area, rating)
    return AWG_TABLE[0]


def _supplies_and_loads(design) -> tuple[list, list]:
    supplies = []
    loads = []
    for part in design.parts:
        for key in ("supply_power", "output_power", "capacity"):
            if key in part.ratings:
                supplies.append((part, part.ratings[key]))
                break
        else:
            for key in ("power", "power_draw", "load_power", "consumption"):
                if key in part.ratings:
                    loads.append((part, part.ratings[key]))
                    break
    return supplies, loads


@register(
    "power_budget",
    "Power budget",
    "Does the supply cover everything drawing from it?",
    domains=("electrical",),
    discipline="electrical",
)
def power_budget(design) -> Iterable[Finding]:
    supplies, loads = _supplies_and_loads(design)
    draw_terms = [value for _part, value in loads if value.dimension == Q(1, "W").dimension]
    if not draw_terms:
        return
    total_draw = draw_terms[0]
    for entry in draw_terms[1:]:
        total_draw = total_draw + entry
    biggest = max(loads, key=lambda pair: float(pair[1].value))
    share = float(biggest[1].value) / float(total_draw.value) if float(total_draw.value) else 0.0
    yield Finding(
        id="electrical.total_draw",
        name="Total power draw",
        value=total_draw.as_("W"),
        formula="P_total = sum of every load's rated power",
        inputs={part.name: value for part, value in loads},
        method="Declared part ratings",
        plain=(
            f"Everything running at once draws {total_draw.as_('W').text()}. The "
            f"{biggest[0].lay_name or biggest[0].name.lower()} is {share * 100:.0f}% of that."
        ),
        subject="electrical",
    )

    supply_terms = [
        value for _part, value in supplies if value.dimension == Q(1, "W").dimension
    ]
    if supply_terms:
        total_supply = supply_terms[0]
        for entry in supply_terms[1:]:
            total_supply = total_supply + entry
        margin = float(total_supply.value) / float(total_draw.value) if float(total_draw.value) else float("inf")
        yield Finding(
            id="electrical.supply_margin",
            name="Supply headroom",
            value=Q(margin, "count"),
            formula="headroom = P_supply / P_draw",
            inputs={"P_supply": total_supply.as_("W"), "P_draw": total_draw.as_("W")},
            method="Declared supply and load ratings",
            plain=(
                f"The supply delivers {total_supply.as_('W').text()} against a "
                f"{total_draw.as_('W').text()} draw, leaving {(margin - 1) * 100:.0f}% spare."
                if margin >= 1
                else f"The supply delivers {total_supply.as_('W').text()} and the loads want "
                f"{total_draw.as_('W').text()}. It is short by "
                f"{(total_draw - total_supply).as_('W').text()}, so something has to be "
                "turned off or the supply has to grow."
            ),
            subject="electrical",
            verdict="pass" if margin >= 1.2 else ("watch" if margin >= 1.0 else "fail"),
            margin=margin - 1.0,
            advice=(
                ""
                if margin >= 1.2
                else "Add supply capacity, or stagger the loads so they do not all run at once."
            ),
        )


@register(
    "battery_runtime",
    "How long it runs",
    "How long does a charge last?",
    domains=("electrical",),
    discipline="electrical",
)
def battery_runtime(design) -> Iterable[Finding]:
    _supplies, loads = _supplies_and_loads(design)
    draw_terms = [value for _p, value in loads if value.dimension == Q(1, "W").dimension]
    if not draw_terms:
        return
    total_draw = draw_terms[0]
    for entry in draw_terms[1:]:
        total_draw = total_draw + entry
    for part in design.parts:
        energy = part.ratings.get("energy") or part.ratings.get("capacity")
        if energy is None:
            continue
        if energy.dimension == Q(1, "Ah").dimension:
            voltage = part.ratings.get("voltage")
            if voltage is None:
                continue
            energy = (energy * voltage).as_("Wh")
        if energy.dimension != Q(1, "J").dimension:
            continue
        # A cell delivered down to empty is a cell destroyed. Eighty per cent
        # usable is the figure a lithium pack is designed around.
        usable = float(part.ratings.get("usable_fraction", Q(0.8, "count")).value)
        hours = float(energy.value) * usable / float(total_draw.value) / 3600.0
        yield Finding(
            id=f"electrical.runtime.{part.id}",
            name=f"Runtime on {part.name}",
            value=Q(hours, "h"),
            formula="t = E usable / P_draw",
            inputs={"E": energy.as_("Wh"), "usable": Q(usable, "count"), "P_draw": total_draw.as_("W")},
            method="Usable energy over average draw; excludes the reserve left in the pack",
            plain=(
                f"On a full {energy.as_('Wh').text()} charge it runs about "
                f"{Q(hours, 'h').text()} at the full {total_draw.as_('W').text()} draw, "
                f"keeping {(1 - usable) * 100:.0f}% in reserve so the cells are not "
                "damaged by being emptied."
            ),
            subject=part.id,
            assumptions=("every load on at once", "no temperature derating"),
        )


@register(
    "voltage_drop",
    "Voltage lost in the wiring",
    "Does the far end get the voltage the near end sent?",
    domains=("electrical",),
    discipline="electrical",
)
def voltage_drop(design) -> Iterable[Finding]:
    for link in design.connections:
        if link.domain != "electrical" or link.through is None:
            continue
        length = None
        for key in ("length", "run_length", "cable_length"):
            if key in (design.environment or {}):
                length = design.environment[key]
        declared = link.notes
        gauge_area = None
        if "mm2" in declared:
            try:
                gauge_area = float(declared.split("mm2")[0].split()[-1]) * 1e-6
            except (ValueError, IndexError):
                gauge_area = None
        current = float(link.through.as_("A").value)
        gauge, area_mm2, rating = wire_for(current)
        area = gauge_area or area_mm2 * 1e-6
        run = float(length.value) if length is not None else float(design.characteristic_length().value) * 2.0
        resistance = _COPPER_RESISTIVITY * run * 2.0 / area
        drop = current * resistance
        supply = link.across
        percent = (drop / float(supply.value) * 100.0) if supply is not None and float(supply.value) else 0.0
        yield Finding(
            id=f"electrical.drop.{link.id}",
            name=f"Voltage drop on {link.label or link.id}",
            value=Q(drop, "V"),
            formula="dV = I rho (2L) / A",
            inputs={
                "I": link.through.as_("A"),
                "L": Q(run, "m"),
                "A": Q(area, "m^2"),
                "rho": Q(_COPPER_RESISTIVITY, "ohm m"),
            },
            method=f"Copper at 20 C, out-and-back run, AWG {gauge} conductor",
            plain=(
                f"Carrying {link.through.as_('A').text()} down "
                f"{Q(run, 'm').text()} of AWG {gauge} copper and back loses "
                f"{Q(drop, 'V').text()}"
                + (f", which is {percent:.1f}% of the supply." if percent else ".")
                + (
                    " Under three per cent is normal practice."
                    if percent and percent < 3
                    else " Over three per cent, and the far end is running short."
                    if percent
                    else ""
                )
            ),
            subject=link.id,
            verdict="pass" if percent and percent < 3 else ("watch" if percent else ""),
            advice=(
                ""
                if not percent or percent < 3
                else f"Move to AWG {max(gauge - 4, 0)} or shorten the run."
            ),
        )
        yield Finding(
            id=f"electrical.wire.{link.id}",
            name=f"Wire size for {link.label or link.id}",
            value=Q(area_mm2, "mm^2"),
            formula="the smallest gauge whose rating covers 1.25 x the load current",
            inputs={"I": link.through.as_("A"), "rating": Q(rating, "A")},
            method="AWG table, free-air chassis-wiring current per NEC 310.15 practice",
            plain=(
                f"AWG {gauge} wire ({area_mm2:g} mm2 of copper) carries this "
                f"{link.through.as_('A').text()} with a quarter more in hand. Thinner wire "
                "would run hot."
            ),
            subject=link.id,
        )


@register(
    "fuse_sizing",
    "Protection",
    "What size fuse or breaker goes here?",
    domains=("electrical",),
    discipline="electrical",
)
def fuse_sizing(design) -> Iterable[Finding]:
    #: Standard fuse ratings, IEC 60269 / common automotive values.
    standard = (0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200)
    for link in design.connections:
        if link.domain != "electrical" or link.through is None:
            continue
        if "protected" not in link.notes and "supply" not in (link.label or "").lower():
            continue
        current = float(link.through.as_("A").value)
        wanted = current * 1.25
        chosen = next((value for value in standard if value >= wanted), standard[-1])
        _gauge, area_mm2, rating = wire_for(current)
        yield Finding(
            id=f"electrical.fuse.{link.id}",
            name=f"Fuse for {link.label or link.id}",
            value=Q(chosen, "A"),
            formula="the next standard rating above 1.25 x the working current",
            inputs={"I_working": link.through.as_("A"), "I_wire": Q(rating, "A")},
            method="Standard IEC 60269 rating series, sized to protect the conductor",
            plain=(
                f"A {chosen:g} A fuse suits this branch: above the "
                f"{link.through.as_('A').text()} it normally carries, and below the "
                f"{Q(rating, 'A').text()} the wire can take, so the fuse blows before the "
                "wire gets hot."
            ),
            subject=link.id,
            verdict="pass" if chosen <= rating else "fail",
            advice=(
                ""
                if chosen <= rating
                else "The wire is too thin for the fuse that would protect the load. Fatten the wire."
            ),
        )


@register(
    "resistive_heating",
    "Heat made by the electronics",
    "How much heat has to go somewhere?",
    domains=("electrical", "thermal"),
    discipline="electrical",
)
def resistive_heating(design) -> Iterable[Finding]:
    for part in design.parts:
        efficiency = part.ratings.get("efficiency")
        power = part.ratings.get("power") or part.ratings.get("power_draw")
        if efficiency is None or power is None:
            continue
        fraction = float(efficiency.value)
        if fraction <= 0 or fraction > 1:
            continue
        waste = float(power.as_("W").value) * (1.0 - fraction)
        yield Finding(
            id=f"thermal.waste.{part.id}",
            name=f"Waste heat from {part.name}",
            value=Q(waste, "W"),
            formula="P_heat = P_in (1 - efficiency)",
            inputs={"P_in": power.as_("W"), "efficiency": efficiency},
            method="Energy balance: what does not leave as useful work leaves as heat",
            plain=(
                f"The {part.lay_name or part.name.lower()} turns "
                f"{fraction * 100:.0f}% of its {power.as_('W').text()} into useful work. "
                f"The other {Q(waste, 'W').text()} comes out as heat and has to go somewhere."
            ),
            subject=part.id,
        )
