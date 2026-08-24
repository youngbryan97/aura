"""What it weighs, where it balances, and how hard it is to turn.

The first three questions asked of any physical design, and the three most
often answered with a guess. Every number here comes from the part
geometry and the material density, both of which the model already holds.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from core.engineering.analysis import Finding, register
from core.engineering.materials import STANDARD_GRAVITY
from core.engineering.units import Q


@register(
    "mass_rollup",
    "Mass and weight",
    "How heavy is it, and which part dominates?",
    discipline="mechanical",
)
def mass_rollup(design) -> Iterable[Finding]:
    total = design.total_mass()
    if total is None:
        return
    weight = total * STANDARD_GRAVITY
    heaviest = max(
        (p for p in design.parts if p.mass() is not None),
        key=lambda p: float(p.mass().value),
        default=None,
    )
    share = ""
    if heaviest is not None and float(total.value) > 0:
        fraction = float(heaviest.mass().value) / float(total.value)
        share = (
            f" The {heaviest.lay_name or heaviest.name.lower()} is the heaviest single "
            f"item at {fraction * 100:.0f}% of the total."
        )
    yield Finding(
        id="mass.total",
        name="Total mass",
        value=total,
        formula="m = sum over parts of volume x density x quantity",
        inputs={"parts": Q(len(design.parts), "count")},
        method="Exact solid volumes times handbook densities",
        plain=(
            f"The whole thing weighs {total.text()}, which on Earth is a pull of "
            f"{weight.as_('N').text()}.{share}"
        ),
        subject="assembly",
    )
    yield Finding(
        id="mass.weight",
        name="Weight on Earth",
        value=weight.as_("N"),
        formula="W = m g",
        inputs={"m": total, "g": STANDARD_GRAVITY},
        method="Standard gravity, CGPM 1901",
        plain=f"On Earth it pushes down with {weight.as_('N').text()} of force.",
        subject="assembly",
    )

    for part in design.parts:
        part_mass = part.mass()
        if part_mass is None:
            continue
        yield Finding(
            id=f"mass.part.{part.id}",
            name=f"Mass of {part.name}",
            value=part_mass,
            formula="m = V rho n",
            inputs={
                "V": part.solid.volume() if part.solid else Q(0, "m^3"),
                "rho": part.material.density if part.material else Q(0, "kg/m^3"),
                "n": Q(part.quantity, "count"),
            },
            method=f"{part.material.source}" if part.material else "declared",
            plain=(
                f"{part.lay_name or part.name} weighs {part_mass.text()}"
                + (f" for all {part.quantity}." if part.quantity > 1 else ".")
            ),
            subject=part.id,
        )


@register(
    "centre_of_mass",
    "Balance point",
    "Where does it balance, and does that sit where it should?",
    discipline="mechanical",
)
def centre_of_mass(design) -> Iterable[Finding]:
    weighted = []
    total = 0.0
    for part in design.parts:
        part_mass = part.mass()
        if part_mass is None or part.solid is None:
            continue
        mesh = part.solid.mesh().transformed(part.placement)
        centre = mesh.vertices.mean(axis=0)
        weighted.append(np.asarray(centre) * float(part_mass.value))
        total += float(part_mass.value)
    if not weighted or total <= 0:
        return
    centre = np.sum(np.array(weighted), axis=0) / total
    low, high = design.bounds()
    span = max(high[i] - low[i] for i in range(3)) or 1.0
    geometric = [(low[i] + high[i]) / 2.0 for i in range(3)]
    offset = float(np.linalg.norm(centre - np.array(geometric)))
    yield Finding(
        id="mass.centre_of_mass",
        name="Balance point",
        value=Q(offset, "m"),
        formula="r_cm = sum(m_i r_i) / sum(m_i)",
        inputs={"total mass": Q(total, "kg")},
        method="Mass-weighted mean of the placed part centroids",
        plain=(
            f"It balances {Q(offset, 'm').text()} away from the middle of its own "
            f"outline, which is {offset / span * 100:.0f}% of its overall size. "
            + (
                "That is close enough to the middle that it will sit level."
                if offset / span < 0.05
                else "It will lean toward the heavy end unless something holds it."
            )
        ),
        subject="assembly",
        verdict="pass" if offset / span < 0.15 else "watch",
        margin=1.0 - offset / span,
        advice=(
            ""
            if offset / span < 0.15
            else "Move the heaviest part toward the centre, or add ballast opposite it."
        ),
    )


@register(
    "envelope",
    "Overall size",
    "How much room does it need?",
    discipline="mechanical",
)
def envelope(design) -> Iterable[Finding]:
    low, high = design.bounds()
    span = [high[i] - low[i] for i in range(3)]
    if max(span) <= 0:
        return
    volume = span[0] * span[1] * span[2]
    solid_volume = 0.0
    for part in design.parts:
        part_volume = part.volume()
        if part_volume is not None:
            solid_volume += float(part_volume.value)
    packing = solid_volume / volume if volume > 0 else 0.0
    yield Finding(
        id="envelope.size",
        name="Overall size",
        value=Q(max(span), "m"),
        formula="the bounding box of every placed part",
        inputs={
            "width": Q(span[0], "m"),
            "depth": Q(span[1], "m"),
            "height": Q(span[2], "m"),
        },
        method="Bounding box of the placed meshes",
        plain=(
            f"It occupies a box {Q(span[0], 'm').text()} by {Q(span[1], 'm').text()} "
            f"by {Q(span[2], 'm').text()}, and {packing * 100:.0f}% of that box is "
            "solid material."
        ),
        subject="assembly",
    )


@register(
    "rotational_inertia",
    "How hard it is to spin up",
    "How much torque does it take to get it moving?",
    domains=("mechanical_rotary",),
    discipline="mechanical",
)
def rotational_inertia(design) -> Iterable[Finding]:
    for part in design.parts:
        if part.solid is None or part.material is None:
            continue
        if "rotating" not in part.tags and "rotor" not in part.tags:
            continue
        moments = part.solid.inertia(part.material.density)
        spin = moments[-1]
        yield Finding(
            id=f"inertia.{part.id}",
            name=f"Rotational inertia of {part.name}",
            value=spin,
            formula="I = integral r^2 dm about the spin axis",
            inputs={"mass": part.solid.mass(part.material.density)},
            method="Closed-form second moment for the solid, about its centroid",
            plain=(
                f"Spinning the {part.lay_name or part.name.lower()} up to 1000 rpm in "
                f"one second needs about "
                f"{(spin * Q(104.7, '1/s') / Q(1, 's')).as_('N m').text()} of twist."
            ),
            subject=part.id,
        )
