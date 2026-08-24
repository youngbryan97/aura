"""Chemical and biological process: what goes in, what comes out, what is left.

A process design lives or dies on its mass balance. What enters a vessel
leaves it or accumulates in it, and a flowsheet where those do not agree has
a stream nobody has drawn. Reaction rate and residence time settle whether
the vessel is big enough, and the enzyme and growth kinetics are the same
question asked of a bioreactor.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.units import Q

#: The gas constant, CODATA.
_R = 8.314462618


@register(
    "mass_balance",
    "Mass balance",
    "Does what goes in equal what comes out?",
    domains=("fluid", "chemical", "biological"),
    discipline="chemical",
)
def mass_balance(design) -> Iterable[Finding]:
    for part in design.parts:
        if not {"vessel", "reactor", "tank", "separator", "mixer"} & set(part.tags):
            continue
        inflow = 0.0
        outflow = 0.0
        counted = 0
        for link in design.connections:
            if link.through is None or link.domain not in {"fluid", "chemical", "biological"}:
                continue
            rate = link.through
            if rate.dimension != Q(1, "kg/s").dimension:
                continue
            if link.target.startswith(part.id):
                inflow += float(rate.value)
                counted += 1
            elif link.source.startswith(part.id):
                outflow += float(rate.value)
                counted += 1
        if counted < 2:
            continue
        total = max(inflow, outflow)
        imbalance = abs(inflow - outflow)
        error = imbalance / total if total > 0 else 0.0
        accumulating = part.ratings.get("accumulation")
        yield Finding(
            id=f"process.balance.{part.id}",
            name=f"Mass balance around {part.name}",
            value=Q(inflow - outflow, "kg/s"),
            formula="sum(in) - sum(out) = accumulation",
            inputs={"in": Q(inflow, "kg/s"), "out": Q(outflow, "kg/s")},
            method="Steady-state mass balance over the control volume",
            plain=(
                f"{Q(inflow, 'kg/s').text()} goes into the "
                f"{part.lay_name or part.name.lower()} and {Q(outflow, 'kg/s').text()} "
                "comes out. "
                + (
                    "They match, so nothing is building up or unaccounted for."
                    if error < 0.01
                    else f"They differ by {Q(imbalance, 'kg/s').text()}, which is "
                    f"{error * 100:.0f}%. Either a stream is missing from the drawing or "
                    "material is accumulating."
                )
            ),
            subject=part.id,
            verdict="pass" if error < 0.01 else ("watch" if accumulating is not None else "fail"),
            margin=1.0 - error,
            advice="" if error < 0.01 else "Find the missing stream, or declare the accumulation.",
        )


@register(
    "residence_time",
    "Time in the vessel",
    "How long does the material stay in there?",
    domains=("fluid", "chemical", "biological"),
    discipline="chemical",
)
def residence_time(design) -> Iterable[Finding]:
    for part in design.parts:
        if not {"vessel", "reactor", "tank"} & set(part.tags):
            continue
        volume = part.ratings.get("working_volume")
        if volume is None and part.solid is not None:
            volume = part.solid.volume()
        flow = part.ratings.get("flow")
        if volume is None or flow is None:
            continue
        if flow.dimension == Q(1, "kg/s").dimension:
            density = part.ratings.get("density") or Q(1000, "kg/m^3")
            volumetric = float(flow.value) / float(density.value)
        else:
            volumetric = float(flow.value)
        if volumetric <= 0:
            continue
        tau = float(volume.value) / volumetric
        yield Finding(
            id=f"process.residence.{part.id}",
            name=f"Residence time in {part.name}",
            value=Q(tau, "s"),
            formula="tau = V / Q",
            inputs={"V": volume, "Q": Q(volumetric, "m^3/s")},
            method="Mean residence time for a well-mixed vessel",
            plain=(
                f"Material spends about {Q(tau, 's').text()} in the "
                f"{part.lay_name or part.name.lower()} on average. Whatever has to happen "
                "in there has that long to happen."
            ),
            subject=part.id,
            assumptions=("perfectly mixed", "steady flow"),
        )


@register(
    "reaction_conversion",
    "How far the reaction goes",
    "How much of the feed is converted?",
    domains=("chemical",),
    discipline="chemical",
)
def reaction_conversion(design) -> Iterable[Finding]:
    for part in design.parts:
        if "reactor" not in part.tags:
            continue
        rate_constant = part.ratings.get("rate_constant")
        volume = part.ratings.get("working_volume") or (
            part.solid.volume() if part.solid else None
        )
        flow = part.ratings.get("flow")
        if rate_constant is None or volume is None or flow is None:
            continue
        if flow.dimension == Q(1, "kg/s").dimension:
            density = part.ratings.get("density") or Q(1000, "kg/m^3")
            volumetric = float(flow.value) / float(density.value)
        else:
            volumetric = float(flow.value)
        if volumetric <= 0:
            continue
        tau = float(volume.value) / volumetric
        k = float(rate_constant.value)
        mixed = "plug" not in part.tags
        conversion = (k * tau / (1.0 + k * tau)) if mixed else (1.0 - math.exp(-k * tau))
        yield Finding(
            id=f"process.conversion.{part.id}",
            name=f"Conversion in {part.name}",
            value=Q(conversion, "count"),
            formula=(
                "X = k tau / (1 + k tau)" if mixed else "X = 1 - exp(-k tau)"
            ),
            inputs={"k": rate_constant, "tau": Q(tau, "s")},
            method=(
                "First-order reaction in a continuous stirred tank"
                if mixed
                else "First-order reaction in a plug-flow reactor"
            ),
            plain=(
                f"About {conversion * 100:.0f}% of the feed reacts before it leaves. "
                + (
                    "A stirred tank mixes fresh feed with finished product, which is why it "
                    "converts less than a plug-flow tube of the same size."
                    if mixed
                    else "A plug-flow tube converts more than a stirred tank of the same size, "
                    "because nothing short-circuits to the outlet."
                )
            ),
            subject=part.id,
        )


@register(
    "enzyme_kinetics",
    "Enzyme rate",
    "How fast does the enzyme work at this concentration?",
    domains=("biological",),
    discipline="bio",
)
def enzyme_kinetics(design) -> Iterable[Finding]:
    for part in design.parts:
        vmax = part.ratings.get("vmax")
        km = part.ratings.get("km")
        substrate = part.ratings.get("substrate")
        if vmax is None or km is None or substrate is None:
            continue
        s = float(substrate.value)
        rate = float(vmax.value) * s / (float(km.value) + s)
        saturation = s / (float(km.value) + s)
        yield Finding(
            id=f"bio.kinetics.{part.id}",
            name=f"Reaction rate in {part.name}",
            value=Q(rate, "mol/s"),
            formula="v = Vmax [S] / (Km + [S])",
            inputs={"Vmax": vmax, "Km": km, "[S]": substrate},
            method="Michaelis-Menten kinetics",
            plain=(
                f"At this substrate level the enzyme runs at {saturation * 100:.0f}% of "
                f"its top speed, {Q(rate, 'mol/s').text()}. "
                + (
                    "It is nearly saturated, so adding more substrate buys almost nothing."
                    if saturation > 0.8
                    else "More substrate would still speed it up noticeably."
                )
            ),
            subject=part.id,
        )


@register(
    "cell_growth",
    "Culture growth",
    "How fast does the culture grow, and when does it fill the vessel?",
    domains=("biological",),
    discipline="bio",
)
def cell_growth(design) -> Iterable[Finding]:
    for part in design.parts:
        if "bioreactor" not in part.tags and "culture" not in part.tags:
            continue
        mu = part.ratings.get("growth_rate")
        if mu is None:
            continue
        rate = float(mu.value)
        if rate <= 0:
            continue
        doubling = math.log(2.0) / rate
        start = part.ratings.get("starting_density")
        target = part.ratings.get("target_density")
        detail = ""
        if start is not None and target is not None and float(start.value) > 0:
            elapsed = math.log(float(target.value) / float(start.value)) / rate
            detail = (
                f" Getting from {start.text()} to {target.text()} takes "
                f"{Q(elapsed, 's').as_('h').text()}."
            )
        yield Finding(
            id=f"bio.growth.{part.id}",
            name=f"Doubling time in {part.name}",
            value=Q(doubling, "s"),
            formula="t_double = ln(2) / mu",
            inputs={"mu": mu},
            method="Exponential growth from the specific growth rate",
            plain=(
                f"The culture doubles every {Q(doubling, 's').as_('h').text()} while "
                f"nothing is limiting it.{detail}"
            ),
            subject=part.id,
            assumptions=("nutrients not limiting", "no product inhibition"),
        )


@register(
    "oxygen_transfer",
    "Oxygen supply",
    "Can enough oxygen reach the cells?",
    domains=("biological",),
    discipline="bio",
)
def oxygen_transfer(design) -> Iterable[Finding]:
    for part in design.parts:
        kla = part.ratings.get("kla")
        demand = part.ratings.get("oxygen_demand")
        if kla is None or demand is None:
            continue
        saturation = float(part.ratings.get("oxygen_saturation", Q(0.21, "mol/m^3")).value)
        working = float(part.ratings.get("dissolved_oxygen", Q(0.05, "mol/m^3")).value)
        supply = float(kla.value) * (saturation - working)
        margin = supply / float(demand.value) if float(demand.value) > 0 else float("inf")
        yield Finding(
            id=f"bio.oxygen.{part.id}",
            name=f"Oxygen transfer in {part.name}",
            value=Q(supply, "mol/(m^3 s)"),
            formula="OTR = kLa (C* - C)",
            inputs={"kLa": kla, "C*": Q(saturation, "mol/m^3"), "C": Q(working, "mol/m^3")},
            method="Two-film oxygen transfer",
            plain=(
                f"The vessel delivers oxygen {margin:.1f} times as fast as the culture "
                "consumes it. "
                + (
                    "Oxygen will not be the limit."
                    if margin >= 1.5
                    else "The culture will go oxygen-limited and stop growing at its top rate."
                )
            ),
            subject=part.id,
            verdict="pass" if margin >= 1.5 else "fail",
            margin=margin - 1.0,
            advice="" if margin >= 1.5 else "Stir harder, sparge more air, or enrich with oxygen.",
        )


@register(
    "gas_state",
    "Gas volume and pressure",
    "How much gas is in there, and what happens when it warms?",
    domains=("pneumatic", "chemical"),
    discipline="chemical",
)
def gas_state(design) -> Iterable[Finding]:
    for part in design.parts:
        if "gas" not in part.tags and "cylinder" not in part.tags:
            continue
        pressure = part.ratings.get("pressure")
        temperature = part.ratings.get("temperature") or design.environment.get(
            "ambient_temperature"
        )
        volume = part.ratings.get("working_volume") or (
            part.solid.volume() if part.solid else None
        )
        if pressure is None or temperature is None or volume is None:
            continue
        moles = float(pressure.value) * float(volume.value) / (_R * float(temperature.value))
        warmed = float(temperature.value) + 30.0
        risen = float(pressure.value) * warmed / float(temperature.value)
        yield Finding(
            id=f"process.gas.{part.id}",
            name=f"Gas charge in {part.name}",
            value=Q(moles, "mol"),
            formula="n = p V / (R T)",
            inputs={"p": pressure, "V": volume, "T": temperature},
            method="Ideal gas law",
            plain=(
                f"The {part.lay_name or part.name.lower()} holds {Q(moles, 'mol').text()} "
                f"of gas at {pressure.as_('bar').text()}. Warm it by 30 degrees with the "
                f"valve shut and the pressure climbs to {Q(risen, 'Pa').as_('bar').text()}."
            ),
            subject=part.id,
            assumptions=("ideal gas", "constant volume when heated"),
        )
