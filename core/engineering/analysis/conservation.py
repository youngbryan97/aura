"""One conservation check for every domain that has one.

Kirchhoff's current law, a mass balance around a junction, and the
requirement that heat arriving at a node also leaves it are the same
statement. Modelica made that structural: a connection carries an across
variable that is equal everywhere at a node, and a through variable that
sums to zero there. So this is one function, and adding a domain to
:mod:`core.engineering.domains` extends it without a line being written
here.

The sign comes from the port's role. A port that sources into the node
counts positive and one that draws from it counts negative, and a node
where any port has not said which it is comes back unchecked rather than
passed. Reporting an unchecked node as balanced is the failure this whole
package is built to avoid.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.domains import domain as get_domain
from core.engineering.units import Q

#: Port roles that push into the node, and roles that draw from it.
_SOURCES = frozenset({"source", "output", "outlet", "supply", "producer", "out"})
_SINKS = frozenset({"sink", "input", "inlet", "load", "consumer", "draw", "in"})

#: Two figures that differ by less than this are the same figure written
#: with different rounding, not an imbalance.
_TOLERANCE = 0.02


@register(
    "conservation",
    "Does it balance",
    "At every junction, does what arrives equal what leaves?",
    discipline="general",
)
def conservation(design) -> Iterable[Finding]:
    for net in design.nets():
        spec = get_domain(net.domain)
        if not spec.conserved:
            continue
        ports = []
        for reference in net.ports:
            found = design.find_port(reference)
            if found is None:
                continue
            ports.append(found)
        if len(ports) < 2:
            continue

        signed: list[tuple[str, float]] = []
        unknown: list[str] = []
        magnitude = 0.0
        total = 0.0
        for part, port in ports:
            if port.through is None:
                unknown.append(f"{part.name} {port.name}")
                continue
            role = str(port.role or "").strip().lower()
            if role in _SOURCES:
                sign = 1.0
            elif role in _SINKS:
                sign = -1.0
            else:
                unknown.append(f"{part.name} {port.name}")
                continue
            value = float(port.through.value)
            signed.append((f"{part.name}.{port.name}", sign * value))
            total += sign * value
            magnitude += abs(value)

        label = f"{spec.through_name.capitalize()} balance"
        if unknown or not signed:
            yield Finding(
                id=f"conservation.{net.id}",
                name=f"{label} at {net.id}",
                value=Q(len(unknown), "count"),
                formula=f"sum of {spec.through_name} into a junction = 0",
                inputs={},
                method=f"{spec.name} conservation over the model graph",
                plain=(
                    f"This junction cannot be checked yet. "
                    + (
                        f"{len(unknown)} of its {len(ports)} connections do not say which "
                        f"way the {spec.carries or spec.through_name} goes or how much: "
                        + ", ".join(unknown[:4])
                        + "."
                        if unknown
                        else "Nothing here declares a rate."
                    )
                ),
                subject=net.id,
                verdict="",
                advice=(
                    f"Give each of those ports a role (source or sink) and a "
                    f"{spec.through_name} rating, and the balance checks itself."
                ),
            )
            continue

        error = abs(total) / magnitude if magnitude > 0 else 0.0
        balanced = error < _TOLERANCE
        yield Finding(
            id=f"conservation.{net.id}",
            name=f"{label} at {net.id}",
            value=Q(total, ports[0][1].through.unit if ports[0][1].through else ""),
            formula=f"sum of {spec.through_name} into a junction = 0",
            inputs={name: Q(abs(value), "") for name, value in signed},
            method=(
                f"{spec.name} conservation: {spec.across_name} is the same for everything "
                f"joined here, and {spec.through_name} sums to zero"
            ),
            plain=(
                f"{len(signed)} connections meet here. "
                + (
                    f"What arrives equals what leaves, to within {error * 100:.1f}%. "
                    f"The {spec.carries or 'flow'} adds up."
                    if balanced
                    else f"They are out by {error * 100:.0f}% — "
                    f"{Q(abs(total), '').text()} of {spec.through_name} unaccounted for. "
                    "Either a branch is missing from the model or one of the figures is wrong."
                )
            ),
            subject=net.id,
            verdict="pass" if balanced else "fail",
            margin=1.0 - error,
            advice=(
                ""
                if balanced
                else f"Find the missing branch, or correct the {spec.through_name} on one "
                "of the connections listed."
            ),
        )


@register(
    "energy_balance",
    "Energy in against energy out",
    "Does more come out than went in?",
    discipline="general",
)
def energy_balance(design) -> Iterable[Finding]:
    """No subsystem may deliver more power than reaches it.

    A design that does is either wrong or a perpetual motion machine, and
    both are worth catching before a drawing is made of it.
    """
    supplied = 0.0
    consumed = 0.0
    names_in: dict[str, object] = {}
    names_out: dict[str, object] = {}
    for part in design.parts:
        for key in ("supply_power", "output_power", "generated_power"):
            value = part.ratings.get(key)
            if value is not None and value.dimension == Q(1, "W").dimension:
                supplied += float(value.value)
                names_in[part.name] = value.as_("W")
                break
        for key in ("power", "power_draw", "load_power", "consumption"):
            value = part.ratings.get(key)
            if value is not None and value.dimension == Q(1, "W").dimension:
                consumed += float(value.value) * part.quantity
                names_out[part.name] = value.as_("W")
                break
    if supplied <= 0 or consumed <= 0:
        return
    ratio = consumed / supplied
    yield Finding(
        id="conservation.energy",
        name="Energy balance",
        value=Q(ratio, "count"),
        formula="sum(power drawn) <= sum(power supplied)",
        inputs={**{f"in: {k}": v for k, v in names_in.items()},
                **{f"out: {k}": v for k, v in names_out.items()}},
        method="First law of thermodynamics over the whole design",
        plain=(
            f"The sources provide {Q(supplied, 'W').text()} and the loads want "
            f"{Q(consumed, 'W').text()}, which is {ratio * 100:.0f}% of it. "
            + (
                "That is inside what the sources can give."
                if ratio <= 1.0
                else "More is being drawn than is being supplied, which cannot happen. "
                "Either a source is missing from the model or a load is overstated."
            )
        ),
        subject="assembly",
        verdict="pass" if ratio <= 1.0 else "fail",
        margin=1.0 - ratio,
        advice="" if ratio <= 1.0 else "Add the missing source, or correct the load figures.",
    )
