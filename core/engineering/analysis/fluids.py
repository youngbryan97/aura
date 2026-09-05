"""Pipes, pumps, pressure drop, and whether the thing floats.

Flow calculations reward being done properly, because the shortcuts are
wrong in the interesting cases. Friction factor is solved from Colebrook
rather than read off a single turbulent approximation, since the laminar and
transitional cases turn up constantly in small tubing. Buoyancy is exact
from the displaced volume, so a hull either floats or it does not, and the
answer is arithmetic rather than optimism.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.materials import STANDARD_GRAVITY
from core.engineering.materials import fluid as get_fluid
from core.engineering.units import Q

#: Wall roughness in metres for the pipe materials a design names.
ROUGHNESS: dict[str, float] = {
    "drawn": 1.5e-6,
    "copper": 1.5e-6,
    "plastic": 1.5e-6,
    "pvc": 1.5e-6,
    "hdpe": 1.5e-6,
    "stainless": 1.5e-5,
    "commercial steel": 4.6e-5,
    "steel": 4.6e-5,
    "galvanised": 1.5e-4,
    "cast iron": 2.6e-4,
    "concrete": 1.0e-3,
    "rubber": 2.5e-5,
}


def friction_factor(reynolds: float, relative_roughness: float) -> tuple[float, str]:
    """Darcy friction factor, with the regime it came from.

    Laminar flow has an exact answer. Turbulent flow needs Colebrook, which
    is implicit and is solved here by fixed-point iteration seeded with the
    Swamee-Jain explicit approximation. The transitional band has no
    reliable correlation and is interpolated, and says so.
    """
    if reynolds <= 0:
        return (0.0, "no flow")
    if reynolds < 2300:
        return (64.0 / reynolds, "laminar, smooth and orderly")
    if reynolds < 4000:
        laminar = 64.0 / 2300.0
        turbulent, _ = friction_factor(4000.0, relative_roughness)
        blend = (reynolds - 2300.0) / 1700.0
        return (
            laminar + blend * (turbulent - laminar),
            "transitional, where no correlation is reliable",
        )
    # Swamee-Jain seed, then Colebrook to convergence.
    guess = 0.25 / (
        math.log10(relative_roughness / 3.7 + 5.74 / reynolds**0.9) ** 2
    )
    value = guess
    for _ in range(40):
        root = 1.0 / math.sqrt(value)
        updated = -2.0 * math.log10(
            relative_roughness / 3.7 + 2.51 / (reynolds * math.sqrt(value))
        )
        new_value = 1.0 / updated**2
        if abs(new_value - value) < 1e-12:
            value = new_value
            break
        value = new_value
    return (value, "turbulent, well mixed")


def _working_fluid(design):
    name = design.environment.get("fluid") or design.environment.get("medium")
    if isinstance(name, str) and name:
        try:
            return get_fluid(name)
        except KeyError:
            pass
    for link in design.connections:
        if link.domain in {"fluid", "hydraulic", "pneumatic"} and link.medium:
            try:
                return get_fluid(link.medium)
            except KeyError:
                continue
    return None


@register(
    "pipe_pressure_drop",
    "Pressure lost in the pipework",
    "How hard does the pump have to push?",
    domains=("fluid", "hydraulic", "pneumatic"),
    discipline="fluid",
)
def pipe_pressure_drop(design) -> Iterable[Finding]:
    working = _working_fluid(design)
    if working is None:
        return
    for part in design.parts:
        if part.solid is None or part.solid.kind != "tube":
            continue
        if "pipe" not in part.tags and "duct" not in part.tags and "line" not in part.tags:
            continue
        flow = None
        for port in part.ports:
            if port.through is not None and port.domain in {"fluid", "hydraulic", "pneumatic"}:
                flow = port.through
                break
        if flow is None:
            flow = part.ratings.get("flow")
        if flow is None:
            continue
        diameter = 2.0 * part.solid.inner_radius
        length = part.solid.height
        area = math.pi * (diameter / 2.0) ** 2
        if flow.dimension == Q(1, "kg/s").dimension:
            volumetric = float(flow.value) / float(working.density.value)
        elif flow.dimension == Q(1, "m^3/s").dimension:
            volumetric = float(flow.value)
        else:
            continue
        velocity = volumetric / area
        reynolds = (
            float(working.density.value) * velocity * diameter
            / float(working.dynamic_viscosity.value)
        )
        roughness_key = (part.material.family if part.material else "steel").lower()
        roughness = ROUGHNESS.get(roughness_key, 4.6e-5)
        factor, regime = friction_factor(reynolds, roughness / diameter)
        drop = factor * (length / diameter) * float(working.density.value) * velocity**2 / 2.0
        yield Finding(
            id=f"fluid.reynolds.{part.id}",
            name=f"Flow regime in {part.name}",
            value=Q(reynolds, "count"),
            formula="Re = rho v D / mu",
            inputs={
                "rho": working.density,
                "v": Q(velocity, "m/s"),
                "D": Q(diameter, "m"),
                "mu": working.dynamic_viscosity,
            },
            method=f"Reynolds number for {working.name.lower()}",
            plain=(
                f"The {working.name.lower()} moves at {Q(velocity, 'm/s').text()} and the "
                f"flow is {regime}. Reynolds number {reynolds:,.0f}."
            ),
            subject=part.id,
        )
        yield Finding(
            id=f"fluid.drop.{part.id}",
            name=f"Pressure drop across {part.name}",
            value=Q(drop, "Pa"),
            formula="dp = f (L/D) rho v^2 / 2",
            inputs={
                "f": Q(factor, "count"),
                "L": Q(length, "m"),
                "D": Q(diameter, "m"),
                "rho": working.density,
                "v": Q(velocity, "m/s"),
            },
            method=(
                "Darcy-Weisbach with the friction factor from Colebrook"
                if reynolds >= 4000
                else "Darcy-Weisbach, laminar friction factor 64/Re"
            ),
            plain=(
                f"Pushing that flow through {Q(length, 'm').text()} of "
                f"{Q(diameter, 'm').text()} bore costs {Q(drop, 'Pa').as_('bar').text()} "
                f"of pressure, which is {Q(drop / float(working.density.value) / 9.80665, 'm').text()} "
                "of head the pump has to make up."
            ),
            subject=part.id,
            assumptions=("straight run", "bends and fittings not included"),
        )
        if velocity > 3.0:
            yield Finding(
                id=f"fluid.velocity_warning.{part.id}",
                name=f"Flow speed in {part.name}",
                value=Q(velocity, "m/s"),
                formula="v = Q / A",
                inputs={"Q": Q(volumetric, "m^3/s"), "A": Q(area, "m^2")},
                method="Continuity",
                plain=(
                    f"At {Q(velocity, 'm/s').text()} the flow is fast enough to erode the "
                    "pipe wall and make noise. Under 3 m/s is normal practice for liquid."
                ),
                subject=part.id,
                verdict="watch",
                advice="Use a larger bore, or accept the noise and the wear.",
            )


@register(
    "pump_duty",
    "What the pump has to do",
    "How much pump does this need?",
    domains=("fluid", "hydraulic"),
    discipline="fluid",
)
def pump_duty(design) -> Iterable[Finding]:
    working = _working_fluid(design)
    if working is None:
        return
    for part in design.parts:
        if "pump" not in part.tags:
            continue
        flow = part.ratings.get("flow")
        head = part.ratings.get("head") or part.ratings.get("pressure")
        if flow is None or head is None:
            continue
        if flow.dimension == Q(1, "kg/s").dimension:
            volumetric = float(flow.value) / float(working.density.value)
        else:
            volumetric = float(flow.value)
        if head.dimension == Q(1, "m").dimension:
            pressure = float(head.value) * float(working.density.value) * 9.80665
        else:
            pressure = float(head.value)
        hydraulic = volumetric * pressure
        efficiency = float(part.ratings.get("efficiency", Q(0.65, "count")).value)
        shaft = hydraulic / efficiency
        yield Finding(
            id=f"fluid.pump.{part.id}",
            name=f"Pump power for {part.name}",
            value=Q(shaft, "W"),
            formula="P_shaft = Q dp / efficiency",
            inputs={
                "Q": Q(volumetric, "m^3/s"),
                "dp": Q(pressure, "Pa"),
                "efficiency": Q(efficiency, "count"),
            },
            method="Hydraulic power over pump efficiency",
            plain=(
                f"Moving {Q(volumetric, 'm^3/s').as_('L/s').text()} against "
                f"{Q(pressure, 'Pa').as_('bar').text()} is "
                f"{Q(hydraulic, 'W').text()} of useful work, and at "
                f"{efficiency * 100:.0f}% efficiency the motor has to supply "
                f"{Q(shaft, 'W').text()}."
            ),
            subject=part.id,
        )


@register(
    "buoyancy",
    "Does it float",
    "Does it float, sink, or hang level?",
    domains=("fluid", "structural"),
    discipline="fluid",
)
def buoyancy(design) -> Iterable[Finding]:
    environment = design.environment
    if not any(
        key in environment for key in ("depth", "water_depth", "submerged", "fluid")
    ):
        return
    working = _working_fluid(design) or get_fluid("seawater")
    displaced = 0.0
    for part in design.parts:
        volume = part.volume()
        if volume is None:
            continue
        if "flooded" in part.tags:
            continue
        displaced += float(volume.value)
    # A hull encloses volume that the solid parts do not account for.
    for part in design.parts:
        enclosed = part.ratings.get("enclosed_volume")
        if enclosed is not None:
            displaced += float(enclosed.value)
    mass = design.total_mass()
    if mass is None or displaced <= 0:
        return
    lift = displaced * float(working.density.value)
    net = lift - float(mass.value)
    yield Finding(
        id="fluid.buoyancy",
        name="Buoyancy",
        value=Q(net * 9.80665, "N"),
        formula="F = (rho_fluid V_displaced - m) g",
        inputs={
            "rho_fluid": working.density,
            "V_displaced": Q(displaced, "m^3"),
            "m": mass,
        },
        method="Archimedes' principle over the displaced volume",
        plain=(
            f"It displaces {Q(displaced, 'm^3').as_('L').text()} of "
            f"{working.name.lower()}, which lifts {Q(lift, 'kg').text()}, against its own "
            f"{mass.text()}. "
            + (
                f"It floats, with {Q(net, 'kg').text()} of spare lift."
                if net > 0.02 * float(mass.value)
                else f"It sinks; it is {Q(-net, 'kg').text()} heavy."
                if net < -0.02 * float(mass.value)
                else "It hangs level in the water, neither rising nor sinking, which is what "
                "a vehicle meant to hold depth wants."
            )
        ),
        subject="assembly",
        verdict="pass",
        margin=net / float(mass.value) if float(mass.value) else 0.0,
    )


@register(
    "hydrostatic_pressure",
    "Pressure at depth",
    "How hard does the water squeeze at that depth?",
    domains=("fluid",),
    discipline="fluid",
)
def hydrostatic_pressure(design) -> Iterable[Finding]:
    depth = design.environment.get("depth") or design.environment.get("water_depth")
    if depth is None:
        return
    working = _working_fluid(design) or get_fluid("seawater")
    pressure = float(working.density.value) * 9.80665 * float(depth.value)
    absolute = pressure + 101325.0
    yield Finding(
        id="fluid.hydrostatic",
        name="Pressure at depth",
        value=Q(pressure, "Pa"),
        formula="p = rho g h",
        inputs={"rho": working.density, "g": STANDARD_GRAVITY, "h": depth},
        method="Hydrostatic column",
        plain=(
            f"At {depth.text()} down, the water presses at "
            f"{Q(pressure, 'Pa').as_('bar').text()} above the surface, "
            f"{Q(absolute, 'Pa').as_('bar').text()} in total. That is about "
            f"{Q(pressure, 'Pa').as_('Pa').value / 101325:.0f} atmospheres — the weight of "
            f"{Q(pressure * 1e-4, 'kg').text()} standing on every square centimetre."
        ),
        subject="environment",
    )


@register(
    "drag",
    "Drag through the water or air",
    "How much push does it take to move?",
    domains=("fluid",),
    discipline="fluid",
)
def drag(design) -> Iterable[Finding]:
    speed = design.environment.get("speed") or design.environment.get("velocity")
    if speed is None:
        return
    working = _working_fluid(design)
    if working is None:
        return
    low, high = design.bounds()
    span = [high[i] - low[i] for i in range(3)]
    frontal = span[0] * span[1]
    if frontal <= 0:
        return
    coefficient = float(design.environment.get("drag_coefficient", Q(0.5, "count")).value)
    velocity = float(speed.value)
    force = 0.5 * float(working.density.value) * velocity**2 * coefficient * frontal
    power = force * velocity
    yield Finding(
        id="fluid.drag",
        name="Drag force",
        value=Q(force, "N"),
        formula="F = 0.5 rho v^2 Cd A",
        inputs={
            "rho": working.density,
            "v": speed,
            "Cd": Q(coefficient, "count"),
            "A": Q(frontal, "m^2"),
        },
        method="Standard drag equation on the frontal area",
        plain=(
            f"Moving at {speed.text()} through {working.name.lower()} costs "
            f"{Q(force, 'N').text()} of drag, which takes {Q(power, 'W').text()} to hold. "
            "Drag goes as the square of speed, so going twice as fast takes eight times "
            "the power."
        ),
        subject="assembly",
        assumptions=(f"drag coefficient {coefficient:g}", "frontal area from the bounding box"),
    )
