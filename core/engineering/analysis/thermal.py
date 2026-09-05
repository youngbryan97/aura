"""Where the heat goes, and how hot things get on the way.

Heat is the constraint that kills designs quietly. Everything works on the
bench and then the enclosure closes and the electronics sit twenty degrees
above what they were rated for. The way to see it coming is a resistance
network: every path heat can take has a thermal resistance in kelvin per
watt, and the temperature rise is the heat times the resistance, exactly as
voltage is current times resistance.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.units import Q, Quantity

#: Stefan-Boltzmann constant, CODATA.
_SIGMA = 5.670374419e-8

#: Natural convection in still air, a flat surface. The coefficient depends
#: on orientation and size; this is the mid-range figure used for a first
#: estimate, and it is named as an assumption on every finding that uses it.
_STILL_AIR_H = 8.0

#: Forced convection with a fan moving air over a surface.
_FORCED_AIR_H = 40.0

#: Water moving over a surface carries heat two orders of magnitude better
#: than air does, which is the whole reason liquid cooling exists.
_WATER_H = 1000.0


def _heat_sources(design, findings: tuple[Finding, ...] = ()) -> list[tuple[object, Quantity]]:
    sources = []
    for part in design.parts:
        for key in ("heat", "waste_heat", "dissipation"):
            if key in part.ratings:
                sources.append((part, part.ratings[key]))
                break
        else:
            efficiency = part.ratings.get("efficiency")
            power = part.ratings.get("power") or part.ratings.get("power_draw")
            if efficiency is not None and power is not None:
                waste = float(power.as_("W").value) * (1.0 - float(efficiency.value))
                sources.append((part, Q(waste, "W")))
    return sources


@register(
    "heat_load",
    "Total heat to get rid of",
    "How much heat is made, and can the case shed it?",
    domains=("thermal",),
    discipline="thermal",
)
def heat_load(design) -> Iterable[Finding]:
    sources = _heat_sources(design)
    if not sources:
        return
    total = sum(float(value.as_("W").value) for _part, value in sources)
    if total <= 0:
        return
    yield Finding(
        id="thermal.total_load",
        name="Heat to remove",
        value=Q(total, "W"),
        formula="Q = sum of every part's waste heat",
        inputs={part.name: value.as_("W") for part, value in sources},
        method="Declared dissipation, or rated power times one minus efficiency",
        plain=(
            f"{Q(total, 'W').text()} of heat is made inside. That is about the same as "
            + _heat_comparison(total)
            + ", and it all has to get out through the outside surface."
        ),
        subject="thermal",
    )

    # The surface that can actually shed it.
    area = 0.0
    for part in design.parts:
        if "enclosure" in part.tags or "housing" in part.tags or "case" in part.tags:
            if part.solid is not None:
                area += float(part.solid.surface_area().value)
    if area <= 0:
        low, high = design.bounds()
        span = [high[i] - low[i] for i in range(3)]
        area = 2.0 * (span[0] * span[1] + span[0] * span[2] + span[1] * span[2])
    if area <= 0:
        return
    ambient = design.environment.get("ambient_temperature") or Q(20, "degC")
    medium = str(design.environment.get("cooling_medium", "")) or "air"
    coefficient = (
        _WATER_H
        if "water" in medium or "sea" in medium
        else _FORCED_AIR_H
        if "fan" in medium or "forced" in medium
        else _STILL_AIR_H
    )
    rise = total / (coefficient * area)
    inside = float(ambient.value) + rise
    yield Finding(
        id="thermal.surface_rise",
        name="Temperature rise",
        value=Q(rise, "K"),
        formula="dT = Q / (h A)",
        inputs={
            "Q": Q(total, "W"),
            "h": Q(coefficient, "W/(m^2 K)"),
            "A": Q(area, "m^2"),
        },
        method=(
            f"Newton's law of cooling, {medium} at h = {coefficient:g} W/(m2 K)"
        ),
        plain=(
            f"Shedding {Q(total, 'W').text()} through {Q(area, 'm^2').text()} of surface "
            f"into {medium} raises the inside to about {Q(inside, 'K').as_('degC').text()}, "
            f"which is {Q(rise, 'K').text()} above the surroundings."
            + (
                " That is a comfortable margin for electronics."
                if rise < 25
                else " That is hot enough to shorten the life of anything electronic inside."
            )
        ),
        subject="thermal",
        verdict="pass" if rise < 25 else ("watch" if rise < 45 else "fail"),
        margin=(45.0 - rise) / 45.0,
        advice=(
            ""
            if rise < 25
            else "Add fins to increase the area, move air over it, or move the heat to a cold plate."
        ),
        assumptions=(
            f"still {medium} unless the model says otherwise",
            "uniform surface temperature",
        ),
    )


def _heat_comparison(watts: float) -> str:
    """Something a reader already has a feel for."""
    if watts < 3:
        return "a phone charging"
    if watts < 15:
        return "a night light"
    if watts < 60:
        return "a laptop under load"
    if watts < 200:
        return "a person sitting still"
    if watts < 1200:
        return "a small fan heater"
    return "a kettle"


@register(
    "conduction_path",
    "Heat path through the material",
    "Does the heat get from where it is made to where it leaves?",
    domains=("thermal",),
    discipline="thermal",
)
def conduction_path(design) -> Iterable[Finding]:
    for link in design.connections:
        if link.domain != "thermal" or link.through is None:
            continue
        found = design.find_port(link.source)
        if found is None:
            continue
        part, _port = found
        if part.material is None or part.solid is None:
            continue
        k = part.material.thermal_conductivity
        if k is None:
            continue
        params = part.solid.parameters()
        thickness = params.get("thickness") or params.get("wall") or params.get("height")
        area_quantity = params.get("section_area")
        if thickness is None:
            continue
        area = (
            float(area_quantity.value)
            if area_quantity is not None
            else float(part.solid.surface_area().value) / 6.0
        )
        resistance = float(thickness.value) / (float(k.value) * area)
        rise = float(link.through.as_("W").value) * resistance
        yield Finding(
            id=f"thermal.conduction.{link.id}",
            name=f"Heat path through {part.name}",
            value=Q(rise, "K"),
            formula="dT = Q L / (k A)",
            inputs={
                "Q": link.through.as_("W"),
                "L": thickness,
                "k": k,
                "A": Q(area, "m^2"),
            },
            method="Fourier conduction through a slab",
            plain=(
                f"Pushing {link.through.as_('W').text()} through "
                f"{thickness.text()} of {part.material.name.lower()} costs "
                f"{Q(rise, 'K').text()} of temperature. "
                + (
                    "That is a good heat path."
                    if rise < 5
                    else "That is a bottleneck; the source will sit that much hotter than the sink."
                )
            ),
            subject=link.id,
            verdict="pass" if rise < 5 else "watch",
            advice="" if rise < 5 else "Shorten the path, widen it, or use a better conductor.",
        )


@register(
    "radiation_balance",
    "Heat radiated away",
    "How much heat leaves as infrared, with no air to carry it?",
    domains=("thermal",),
    discipline="thermal",
)
def radiation_balance(design) -> Iterable[Finding]:
    if "vacuum" not in str(design.environment.get("medium", "")).lower():
        return
    sources = _heat_sources(design)
    total = sum(float(value.as_("W").value) for _p, value in sources)
    if total <= 0:
        return
    low, high = design.bounds()
    span = [high[i] - low[i] for i in range(3)]
    area = 2.0 * (span[0] * span[1] + span[0] * span[2] + span[1] * span[2])
    if area <= 0:
        return
    emissivity = float(design.environment.get("emissivity", Q(0.85, "count")).value)
    sink = float((design.environment.get("sink_temperature") or Q(2.7, "K")).value)
    # Solve for the equilibrium surface temperature.
    surface = (total / (emissivity * _SIGMA * area) + sink**4) ** 0.25
    yield Finding(
        id="thermal.radiation",
        name="Equilibrium temperature in vacuum",
        value=Q(surface, "K"),
        formula="Q = epsilon sigma A (T^4 - T_sink^4)",
        inputs={
            "Q": Q(total, "W"),
            "epsilon": Q(emissivity, "count"),
            "A": Q(area, "m^2"),
            "T_sink": Q(sink, "K"),
        },
        method="Stefan-Boltzmann radiation to a cold sink",
        plain=(
            f"With no air to carry heat away, the outside settles at "
            f"{Q(surface, 'K').as_('degC').text()}, radiating "
            f"{Q(total, 'W').text()} as infrared. Radiators are the only way to lose heat "
            "out there, and they work by area."
        ),
        subject="thermal",
        verdict="pass" if surface < 350 else "watch",
    )
