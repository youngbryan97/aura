"""Gears, levers, motors and screws: what turns what, and how hard.

Every mechanism here trades force against distance, and the trade is exact.
A gear ratio that multiplies torque divides speed by the same number, and
a design that expects both is a design with an arithmetic error in it. These
checks find that error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.materials import STANDARD_GRAVITY
from core.engineering.units import Q


@register(
    "gear_train",
    "Gear ratios",
    "What comes out the other end of the gearbox?",
    domains=("mechanical_rotary",),
    discipline="mechanical",
)
def gear_train(design) -> Iterable[Finding]:
    gears = [p for p in design.parts if "gear" in p.tags or "pulley" in p.tags]
    if len(gears) < 2:
        return
    for driver, driven in zip(gears, gears[1:]):
        teeth_in = driver.ratings.get("teeth")
        teeth_out = driven.ratings.get("teeth")
        if teeth_in is None or teeth_out is None:
            diameter_in = driver.solid.parameters().get("diameter") if driver.solid else None
            diameter_out = driven.solid.parameters().get("diameter") if driven.solid else None
            if diameter_in is None or diameter_out is None:
                continue
            ratio = float(diameter_out.value) / float(diameter_in.value)
            basis = "pitch diameters"
            inputs = {"d_in": diameter_in, "d_out": diameter_out}
        else:
            ratio = float(teeth_out.value) / float(teeth_in.value)
            basis = "tooth counts"
            inputs = {"N_in": teeth_in, "N_out": teeth_out}
        torque_in = driver.ratings.get("torque")
        speed_in = driver.ratings.get("speed")
        detail = ""
        if torque_in is not None:
            detail += f" {torque_in.text()} in becomes {(torque_in * ratio).as_('N m').text()} out."
        if speed_in is not None:
            detail += f" {speed_in.text()} in becomes {(speed_in / ratio).text()} out."
        yield Finding(
            id=f"motion.gear.{driver.id}_{driven.id}",
            name=f"Ratio, {driver.name} to {driven.name}",
            value=Q(ratio, "count"),
            formula="ratio = N_out / N_in",
            inputs=inputs,
            method=f"Gear ratio from {basis}",
            plain=(
                f"The {driven.lay_name or driven.name.lower()} turns "
                f"{ratio:.2f} times "
                + ("slower" if ratio > 1 else "faster")
                + f" than the {driver.lay_name or driver.name.lower()}, and with "
                f"{ratio:.2f} times "
                + ("more" if ratio > 1 else "less")
                + " twist." + detail
            ),
            subject=driven.id,
        )


@register(
    "motor_sizing",
    "Motor size",
    "Is the motor big enough for the job?",
    domains=("mechanical_rotary", "electrical"),
    discipline="mechanical",
)
def motor_sizing(design) -> Iterable[Finding]:
    for part in design.parts:
        if "motor" not in part.tags and "actuator" not in part.tags:
            continue
        torque = part.ratings.get("torque")
        speed = part.ratings.get("speed")
        if torque is None or speed is None:
            continue
        omega = float(speed.as_("rad/s").value) if speed.unit != "rpm" else float(speed.value)
        mechanical = float(torque.value) * omega
        efficiency = float(part.ratings.get("efficiency", Q(0.8, "count")).value)
        electrical = mechanical / efficiency
        yield Finding(
            id=f"motion.motor.{part.id}",
            name=f"Shaft power of {part.name}",
            value=Q(mechanical, "W"),
            formula="P = T omega",
            inputs={"T": torque, "omega": Q(omega, "rad/s")},
            method="Rotational power from rated torque and speed",
            plain=(
                f"The {part.lay_name or part.name.lower()} delivers "
                f"{Q(mechanical, 'W').text()} at the shaft, and draws about "
                f"{Q(electrical, 'W').text()} from the supply to do it. The difference "
                f"is {Q(electrical - mechanical, 'W').text()} of heat."
            ),
            subject=part.id,
        )
        load = part.ratings.get("load_torque") or part.ratings.get("required_torque")
        if load is not None and float(load.value) > 0:
            factor = float(torque.value) / float(load.value)
            yield Finding(
                id=f"motion.motor_margin.{part.id}",
                name=f"Torque margin, {part.name}",
                value=Q(factor, "count"),
                formula="margin = T_rated / T_required",
                inputs={"T_rated": torque, "T_required": load},
                method="Rated against required torque",
                plain=(
                    f"It can push {factor:.1f} times harder than the job asks for."
                    + (
                        " Comfortable."
                        if factor >= 1.5
                        else " Thin: a stiff joint or a cold day will stall it."
                    )
                ),
                subject=part.id,
                verdict="pass" if factor >= 1.5 else ("watch" if factor >= 1.1 else "fail"),
                margin=factor - 1.0,
                advice="" if factor >= 1.5 else "Pick a larger motor or add gear reduction.",
            )


@register(
    "lever_and_linkage",
    "Leverage",
    "How much does the linkage multiply the force?",
    domains=("mechanical_linear",),
    discipline="mechanical",
)
def lever_and_linkage(design) -> Iterable[Finding]:
    for part in design.parts:
        if "lever" not in part.tags and "linkage" not in part.tags:
            continue
        effort_arm = part.ratings.get("effort_arm")
        load_arm = part.ratings.get("load_arm")
        if effort_arm is None or load_arm is None:
            continue
        advantage = float(effort_arm.value) / float(load_arm.value)
        effort = part.ratings.get("effort")
        detail = ""
        if effort is not None:
            detail = (
                f" Pushing with {effort.text()} lifts "
                f"{(effort * advantage).as_('N').text()}."
            )
        yield Finding(
            id=f"motion.lever.{part.id}",
            name=f"Mechanical advantage of {part.name}",
            value=Q(advantage, "count"),
            formula="MA = effort arm / load arm",
            inputs={"effort_arm": effort_arm, "load_arm": load_arm},
            method="Moment balance about the pivot",
            plain=(
                f"The {part.lay_name or part.name.lower()} multiplies force by "
                f"{advantage:.2f}, and divides movement by the same amount.{detail}"
            ),
            subject=part.id,
        )


@register(
    "leadscrew",
    "Screw drive",
    "How much force does the screw make, and does it hold when the power goes off?",
    domains=("mechanical_linear", "mechanical_rotary"),
    discipline="mechanical",
)
def leadscrew(design) -> Iterable[Finding]:
    for part in design.parts:
        if "leadscrew" not in part.tags and "screw_drive" not in part.tags:
            continue
        pitch = part.ratings.get("pitch") or part.ratings.get("lead")
        torque = part.ratings.get("torque")
        if pitch is None or torque is None:
            continue
        diameter = part.ratings.get("diameter")
        if diameter is None and part.solid is not None:
            diameter = part.solid.parameters().get("diameter")
        if diameter is None:
            continue
        mu = float(part.ratings.get("friction", Q(0.2, "count")).value)
        lead = float(pitch.value)
        mean_diameter = float(diameter.value) - lead / 2.0
        helix = math.atan(lead / (math.pi * mean_diameter))
        efficiency = math.tan(helix) / math.tan(helix + math.atan(mu))
        force = 2.0 * math.pi * float(torque.value) * efficiency / lead
        self_locking = mu > math.tan(helix)
        yield Finding(
            id=f"motion.leadscrew.{part.id}",
            name=f"Thrust from {part.name}",
            value=Q(force, "N"),
            formula="F = 2 pi T eta / lead, with eta = tan(lambda) / tan(lambda + phi)",
            inputs={"T": torque, "lead": pitch, "mu": Q(mu, "count")},
            method="Screw thread efficiency with Coulomb friction",
            plain=(
                f"Turning it with {torque.text()} pushes {Q(force, 'N').text()}, at "
                f"{efficiency * 100:.0f}% efficiency — the rest is friction in the thread. "
                + (
                    "It holds its position with the power off, because friction beats the "
                    "thread angle."
                    if self_locking
                    else "It will back-drive: with the power off the load pushes it back down, "
                    "so it needs a brake."
                )
            ),
            subject=part.id,
            verdict="pass" if self_locking else "watch",
            advice="" if self_locking else "Add a brake or a worm stage if the load must be held.",
        )


@register(
    "actuation_energy",
    "Energy per movement",
    "How much energy does one stroke cost?",
    domains=("mechanical_linear",),
    discipline="mechanical",
)
def actuation_energy(design) -> Iterable[Finding]:
    for part in design.parts:
        stroke = part.ratings.get("stroke")
        force = part.ratings.get("force")
        if stroke is None or force is None:
            continue
        work = float(force.value) * float(stroke.value)
        rate = part.ratings.get("cycles_per_second") or part.ratings.get("frequency")
        detail = ""
        if rate is not None:
            detail = (
                f" Running at {rate.text()} that is "
                f"{Q(work * float(rate.value), 'W').text()} of continuous draw."
            )
        yield Finding(
            id=f"motion.stroke_energy.{part.id}",
            name=f"Work per stroke, {part.name}",
            value=Q(work, "J"),
            formula="W = F d",
            inputs={"F": force, "d": stroke},
            method="Work done against a constant force",
            plain=(
                f"Each {stroke.text()} stroke against {force.text()} costs "
                f"{Q(work, 'J').text()}.{detail}"
            ),
            subject=part.id,
        )
