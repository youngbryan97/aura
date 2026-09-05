"""Will it hold: stress, deflection, buckling and the margin left over.

The calculations here are the ones a design review asks for first, and each
one is a closed-form result an engineer would check by hand. That matters
more than sophistication: a hand-checkable number with its formula attached
can be argued with, and a finite-element colour plot with no formula cannot.

Sign conventions follow Roark's Formulas for Stress and Strain. Pressure
vessel work follows the thin-wall membrane equations, with Lame's thick-wall
solution used automatically once the wall passes a tenth of the radius,
because the thin-wall answer is wrong there and quietly so.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.geometry import Capsule, Cylinder, Dome, Prism, Sphere, Tube
from core.engineering.materials import STANDARD_GRAVITY
from core.engineering.units import Q, Quantity

#: Below this the thin-wall membrane equations are within a few per cent.
#: Above it they under-report the bore stress and Lame's solution is used.
_THIN_WALL_RATIO = 0.1

#: What a factor of safety means to somebody reading the drawing.
_SAFETY_WORDS: tuple[tuple[float, str, str], ...] = (
    (1.0, "fail", "It breaks at this load. The part has to get stronger or the load smaller."),
    (1.5, "watch", "It holds, with little to spare. Normal for a weight-critical part, thin for anything else."),
    (2.5, "pass", "It holds comfortably, which is the usual target for a part carrying a known load."),
    (6.0, "pass", "It holds with a lot to spare, which is right for anything a person stands under."),
    (float("inf"), "pass", "It is far stronger than it needs to be, and there is weight and cost to recover."),
)


def _safety_words(factor: float) -> tuple[str, str]:
    for ceiling, verdict, sentence in _SAFETY_WORDS:
        if factor < ceiling:
            return verdict, sentence
    return "pass", ""


def _external_pressure(design) -> Quantity | None:
    """The outside pressure the design works against, from its environment."""
    environment = design.environment
    for key in ("external_pressure", "pressure", "ambient_pressure"):
        if key in environment:
            return environment[key]
    depth = environment.get("depth") or environment.get("water_depth")
    if depth is not None:
        from core.engineering.materials import fluid

        water = fluid("seawater")
        return (water.density * STANDARD_GRAVITY * depth).as_("Pa")
    altitude = environment.get("altitude")
    if altitude is not None:
        # A vessel above sea level works against less, not more.
        return Q(101325.0 * math.exp(-float(altitude.value) / 8400.0), "Pa")
    return None


def _internal_pressure(part) -> Quantity | None:
    for key in ("internal_pressure", "working_pressure", "pressure"):
        if key in part.ratings:
            return part.ratings[key]
    for port in part.ports:
        if port.domain in {"fluid", "hydraulic", "pneumatic"} and port.across is not None:
            return port.across
    return None


@register(
    "pressure_vessel",
    "Pressure vessel stress",
    "Does the wall hold the pressure, inside or out?",
    discipline="mechanical",
)
def pressure_vessel(design) -> Iterable[Finding]:
    outside = _external_pressure(design)
    for part in design.parts:
        if part.solid is None or part.material is None:
            continue
        inside = _internal_pressure(part)
        if inside is None and outside is None:
            continue
        solid = part.solid
        if isinstance(solid, Tube):
            radius = solid.outer_radius - solid.wall / 2.0
            wall = solid.wall
            shape = "cylinder"
        elif isinstance(solid, Dome):
            radius = solid.radius - solid.wall / 2.0
            wall = solid.wall
            shape = "sphere"
        elif isinstance(solid, Capsule) and "wall" in part.ratings:
            radius = solid.radius
            wall = float(part.ratings["wall"].value)
            shape = "cylinder"
        else:
            continue
        if wall <= 0 or radius <= 0:
            continue

        differential = inside if inside is not None else outside
        net = float(differential.value)
        if inside is not None and outside is not None:
            net = abs(float(inside.value) - float(outside.value))
        ratio = wall / radius
        thick = ratio > _THIN_WALL_RATIO

        if shape == "cylinder":
            if thick:
                inner = radius - wall / 2.0
                outer = radius + wall / 2.0
                hoop = net * (outer**2 + inner**2) / (outer**2 - inner**2)
                formula = "sigma = p (ro^2 + ri^2) / (ro^2 - ri^2)"
                method = "Lame thick-wall solution, bore hoop stress"
            else:
                hoop = net * radius / wall
                formula = "sigma_hoop = p r / t"
                method = "Thin-wall membrane equation"
        else:
            hoop = net * radius / (2.0 * wall)
            formula = "sigma = p r / 2 t"
            method = "Thin-wall spherical membrane equation"

        stress = Q(hoop, "Pa")
        strength = part.material.yield_strength or part.material.ultimate_strength
        subject = part.id
        yield Finding(
            id=f"stress.pressure.{part.id}",
            name=f"Wall stress in {part.name}",
            value=stress,
            formula=formula,
            inputs={
                "p": Q(net, "Pa"),
                "r": Q(radius, "m"),
                "t": Q(wall, "m"),
            },
            method=method,
            plain=(
                f"The pressure stretches the {part.lay_name or part.name.lower()} wall "
                f"at {stress.as_('MPa').text()}."
                + (
                    " The wall is thick enough that the simple formula understates it, "
                    "so the thick-wall solution was used."
                    if thick
                    else ""
                )
            ),
            subject=subject,
            assumptions=("uniform wall", "no stress raisers at penetrations"),
        )

        if strength is not None and float(stress.value) > 0:
            factor = float(strength.value) / float(stress.value)
            verdict, sentence = _safety_words(factor)
            yield Finding(
                id=f"safety.pressure.{part.id}",
                name=f"Safety factor, {part.name}",
                value=Q(factor, "count"),
                formula="n = sigma_yield / sigma_working",
                inputs={"sigma_yield": strength, "sigma_working": stress},
                method=f"Yield strength from {part.material.source}",
                plain=(
                    f"The {part.lay_name or part.name.lower()} is {factor:.1f} times "
                    f"stronger than the pressure needs it to be. {sentence}"
                ),
                subject=subject,
                verdict=verdict,
                margin=factor - 1.0,
                advice=(
                    ""
                    if verdict == "pass"
                    else f"Thicken the wall to about {Q(wall * max(2.5 / factor, 1.0), 'm').text()} "
                    "or use a stronger material."
                ),
            )

        # A vessel loaded from outside fails by collapsing long before it
        # fails by yielding, which is the mistake a hoop-stress check alone
        # invites. Reported whenever the outside pressure is the higher one.
        if outside is not None and (inside is None or float(outside.value) > float(inside.value)):
            E = part.material.youngs_modulus
            nu = part.material.poisson_ratio or 0.3
            if E is not None and shape == "cylinder":
                length = getattr(solid, "height", None) or 4.0 * radius
                critical = _collapse_pressure(
                    float(E.value), nu, radius, wall, float(length)
                )
                collapse = Q(critical, "Pa")
                factor = critical / net if net > 0 else float("inf")
                verdict, _ = _safety_words(factor)
                yield Finding(
                    id=f"buckle.external.{part.id}",
                    name=f"Collapse pressure, {part.name}",
                    value=collapse,
                    formula="p_cr = 2.42 E (t/2r)^2.5 / [(1-nu^2)^0.75 (L/2r - 0.45 (t/2r)^0.5)]",
                    inputs={"E": E, "t": Q(wall, "m"), "r": Q(radius, "m"), "L": Q(float(length), "m")},
                    method="Windenburg and Trilling elastic instability of a cylinder under external pressure",
                    plain=(
                        f"Squeezed from outside, the {part.lay_name or part.name.lower()} "
                        f"crushes at {collapse.as_('bar').text()}, which is {factor:.1f} times "
                        f"the {Q(net, 'Pa').as_('bar').text()} it will actually see. A tube "
                        "under outside pressure buckles inward before it ever reaches its "
                        "strength limit."
                    ),
                    subject=subject,
                    verdict=verdict,
                    margin=factor - 1.0,
                    advice=(
                        ""
                        if factor >= 2.0
                        else "Add stiffening rings, shorten the unsupported span, or thicken the wall."
                    ),
                    assumptions=("perfectly round", "no out-of-roundness allowance"),
                )


def _collapse_pressure(E: float, nu: float, radius: float, wall: float, length: float) -> float:
    """Elastic collapse pressure of a cylinder squeezed from outside."""
    diameter = 2.0 * radius
    ratio = wall / diameter
    denominator = (1.0 - nu * nu) ** 0.75 * (length / diameter - 0.45 * math.sqrt(ratio))
    if denominator <= 0:
        # Short enough that the end closures carry it; fall back to the
        # long-cylinder limit, which is the conservative answer.
        return 2.0 * E / (1.0 - nu * nu) * ratio**3
    return 2.42 * E * ratio**2.5 / denominator


def _section_properties(part) -> tuple[float, float, float] | None:
    """Second moment, extreme fibre distance and area for a part's section."""
    solid = part.solid
    if isinstance(solid, Cylinder):
        r = solid.radius
        return (math.pi * r**4 / 4.0, r, math.pi * r * r)
    if isinstance(solid, Tube):
        ro, ri = solid.outer_radius, solid.inner_radius
        return (
            math.pi * (ro**4 - ri**4) / 4.0,
            ro,
            math.pi * (ro * ro - ri * ri),
        )
    if isinstance(solid, Prism):
        ixx, _iyy = solid.section_moments()
        cx, cy = solid.section_centroid()
        extreme = max(abs(y - cy) for _x, y in solid.outline)
        return (float(ixx.value), extreme, solid._area())
    if solid is not None and solid.kind in {"box", "plate"}:
        params = solid.parameters()
        width = float(params.get("width", Q(0, "m")).value)
        height = float(
            params.get("height", params.get("thickness", Q(0, "m"))).value
        )
        if width <= 0 or height <= 0:
            return None
        return (width * height**3 / 12.0, height / 2.0, width * height)
    return None


def _applied_load(design, part) -> tuple[Quantity, str] | None:
    """The force on a part, from its ratings or a structural connection."""
    for key in ("load", "applied_load", "force", "thrust"):
        if key in part.ratings:
            return (part.ratings[key], f"declared {key}")
    for link in design.connections:
        if link.domain != "structural" or link.through is None:
            continue
        if link.source.startswith(part.id) or link.target.startswith(part.id):
            return (link.through, f"load carried by {link.id}")
    return None


@register(
    "beam_bending",
    "Bending strength and stiffness",
    "How far does it bend, and does it break?",
    domains=("structural",),
    discipline="mechanical",
)
def beam_bending(design) -> Iterable[Finding]:
    for part in design.parts:
        if part.solid is None or part.material is None:
            continue
        if "beam" not in part.tags and "cantilever" not in part.tags and "arm" not in part.tags:
            continue
        section = _section_properties(part)
        if section is None:
            continue
        second_moment, extreme, area = section
        loaded = _applied_load(design, part)
        if loaded is None:
            continue
        force, origin = loaded
        span = float(getattr(part.solid, "height", 0.0) or 0.0)
        if span <= 0 or second_moment <= 0:
            continue
        cantilever = "cantilever" in part.tags or "arm" in part.tags
        E = part.material.youngs_modulus
        moment = float(force.value) * span * (1.0 if cantilever else 0.25)
        stress = moment * extreme / second_moment
        formula = (
            "sigma = F L c / I" if cantilever else "sigma = F L c / (4 I)"
        )
        yield Finding(
            id=f"stress.bending.{part.id}",
            name=f"Bending stress in {part.name}",
            value=Q(stress, "Pa"),
            formula=formula,
            inputs={
                "F": force,
                "L": Q(span, "m"),
                "I": Q(second_moment, "m^4"),
                "c": Q(extreme, "m"),
            },
            method=(
                "Roark, cantilever with an end load"
                if cantilever
                else "Roark, simply supported with a central load"
            ),
            plain=(
                f"Bending the {part.lay_name or part.name.lower()} with {force.text()} "
                f"({origin}) stretches its outer surface at {Q(stress, 'Pa').as_('MPa').text()}."
            ),
            subject=part.id,
        )
        strength = part.material.yield_strength or part.material.ultimate_strength
        if strength is not None and stress > 0:
            factor = float(strength.value) / stress
            verdict, sentence = _safety_words(factor)
            yield Finding(
                id=f"safety.bending.{part.id}",
                name=f"Bending safety factor, {part.name}",
                value=Q(factor, "count"),
                formula="n = sigma_yield / sigma_bending",
                inputs={"sigma_yield": strength, "sigma_bending": Q(stress, "Pa")},
                method=f"Yield strength from {part.material.source}",
                plain=(
                    f"It is {factor:.1f} times stronger than the bending load. {sentence}"
                ),
                subject=part.id,
                verdict=verdict,
                margin=factor - 1.0,
            )
        if E is not None:
            factor_d = 3.0 if cantilever else 48.0
            deflection = float(force.value) * span**3 / (factor_d * float(E.value) * second_moment)
            yield Finding(
                id=f"deflection.{part.id}",
                name=f"Deflection of {part.name}",
                value=Q(deflection, "m"),
                formula=(
                    "delta = F L^3 / (3 E I)" if cantilever else "delta = F L^3 / (48 E I)"
                ),
                inputs={"F": force, "L": Q(span, "m"), "E": E, "I": Q(second_moment, "m^4")},
                method="Roark, elastic beam deflection",
                plain=(
                    f"Under that load the end moves {Q(deflection, 'm').text()}, which is "
                    f"{deflection / span * 100:.2f}% of its length."
                    + (
                        " That is small enough to be invisible."
                        if deflection / span < 0.002
                        else " That is enough to see, and enough to matter for anything that has to line up."
                    )
                ),
                subject=part.id,
                verdict="pass" if deflection / span < 0.005 else "watch",
            )


@register(
    "column_buckling",
    "Buckling under compression",
    "Does it fold before it crushes?",
    domains=("structural",),
    discipline="mechanical",
)
def column_buckling(design) -> Iterable[Finding]:
    for part in design.parts:
        if part.solid is None or part.material is None:
            continue
        if "column" not in part.tags and "strut" not in part.tags and "leg" not in part.tags:
            continue
        section = _section_properties(part)
        E = part.material.youngs_modulus
        loaded = _applied_load(design, part)
        if section is None or E is None or loaded is None:
            continue
        second_moment, _extreme, area = section
        force, _origin = loaded
        length = float(getattr(part.solid, "height", 0.0) or 0.0)
        if length <= 0 or area <= 0:
            continue
        # Both ends pinned unless the model says the ends are held square.
        k = 0.5 if "fixed_ends" in part.tags else 1.0
        critical = math.pi**2 * float(E.value) * second_moment / (k * length) ** 2
        crush = (
            float((part.material.yield_strength or part.material.ultimate_strength).value)
            * area
        )
        governs = "folding" if critical < crush else "crushing"
        limit = min(critical, crush)
        factor = limit / float(force.value) if float(force.value) > 0 else float("inf")
        verdict, sentence = _safety_words(factor)
        yield Finding(
            id=f"buckle.column.{part.id}",
            name=f"Buckling load, {part.name}",
            value=Q(critical, "N"),
            formula="P_cr = pi^2 E I / (K L)^2",
            inputs={"E": E, "I": Q(second_moment, "m^4"), "L": Q(length, "m"), "K": Q(k, "count")},
            method="Euler column buckling, effective-length factor per end condition",
            plain=(
                f"Pushed end-on, the {part.lay_name or part.name.lower()} folds at "
                f"{Q(critical, 'N').text()} and crushes at {Q(crush, 'N').text()}, so "
                f"{governs} is what limits it. Against the {force.text()} it carries "
                f"that leaves a factor of {factor:.1f}. {sentence}"
            ),
            subject=part.id,
            verdict=verdict,
            margin=factor - 1.0,
            advice=(
                ""
                if verdict == "pass"
                else "Shorten the unsupported length, brace it at the middle, or use a fatter section."
            ),
            assumptions=("straight to start with", "load applied on the axis"),
        )


@register(
    "thermal_stress",
    "Stress from heating",
    "What happens when it warms up and cannot expand?",
    domains=("thermal", "structural"),
    discipline="mechanical",
)
def thermal_stress(design) -> Iterable[Finding]:
    swing = design.environment.get("temperature_swing")
    if swing is None:
        return
    for part in design.parts:
        if part.material is None or "constrained" not in part.tags:
            continue
        alpha = part.material.thermal_expansion
        E = part.material.youngs_modulus
        if alpha is None or E is None:
            continue
        stress = float(alpha.value) * float(swing.value) * float(E.value)
        strength = part.material.yield_strength or part.material.ultimate_strength
        factor = float(strength.value) / stress if strength and stress > 0 else float("inf")
        verdict, sentence = _safety_words(factor)
        yield Finding(
            id=f"stress.thermal.{part.id}",
            name=f"Thermal stress in {part.name}",
            value=Q(stress, "Pa"),
            formula="sigma = alpha dT E",
            inputs={"alpha": alpha, "dT": swing, "E": E},
            method="Fully restrained uniaxial expansion",
            plain=(
                f"Held so it cannot expand, a {swing.text()} temperature change loads the "
                f"{part.lay_name or part.name.lower()} at {Q(stress, 'Pa').as_('MPa').text()} "
                f"with nothing touching it. {sentence}"
            ),
            subject=part.id,
            verdict=verdict,
            margin=factor - 1.0,
            advice=(
                ""
                if verdict == "pass"
                else "Let one end slide, add a flexible joint, or match the two materials' expansion."
            ),
        )


@register(
    "natural_frequency",
    "Where it rings",
    "What shakes it apart, and does anything on board hit that note?",
    domains=("structural",),
    discipline="mechanical",
)
def natural_frequency(design) -> Iterable[Finding]:
    for part in design.parts:
        if part.solid is None or part.material is None:
            continue
        if "cantilever" not in part.tags and "arm" not in part.tags:
            continue
        section = _section_properties(part)
        E = part.material.youngs_modulus
        if section is None or E is None:
            continue
        second_moment, _extreme, area = section
        length = float(getattr(part.solid, "height", 0.0) or 0.0)
        if length <= 0 or area <= 0:
            continue
        mass_per_length = float(part.material.density.value) * area
        # First bending mode of a uniform cantilever.
        frequency = (
            (1.875104**2 / (2.0 * math.pi))
            * math.sqrt(float(E.value) * second_moment / (mass_per_length * length**4))
        )
        yield Finding(
            id=f"frequency.{part.id}",
            name=f"First bending mode, {part.name}",
            value=Q(frequency, "Hz"),
            formula="f1 = (1.8751^2 / 2 pi) sqrt(E I / (rho A L^4))",
            inputs={
                "E": E,
                "I": Q(second_moment, "m^4"),
                "rho": part.material.density,
                "A": Q(area, "m^2"),
                "L": Q(length, "m"),
            },
            method="Euler-Bernoulli cantilever, first mode",
            plain=(
                f"The {part.lay_name or part.name.lower()} rings at {Q(frequency, 'Hz').text()}. "
                "Anything driving it at that rate will shake it much harder than the force alone "
                "suggests, so keep motors and pumps away from it."
            ),
            subject=part.id,
        )
