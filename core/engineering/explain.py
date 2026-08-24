"""Saying what the thing does, in words a reader already has.

Two rules, and they pull in opposite directions on purpose. The engineering
word stays, because a reader who is told "hoop stress" can look it up and a
reader who is told "the sideways squeeze" cannot. And the engineering word is
never left alone, because a term nobody has met is a wall.

So every term carries a gloss, and the narrative that walks a design is
generated from the connection graph rather than written. It follows the
energy from wherever it enters to wherever it leaves, naming each part it
passes through and what that part does to it. A design where the walk
cannot get from a source to a load has a gap in it, and the walk says so
instead of glossing over it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.engineering.domains import domain as get_domain

__all__ = [
    "GLOSSARY",
    "gloss",
    "annotate_terms",
    "narrate",
    "explain_part",
    "explain_finding",
    "reading_order",
]


#: Engineering terms and what they mean to somebody who has not met them.
#: The gloss says what the term is FOR, not what it translates to, because
#: a synonym for a word nobody knows is another word nobody knows.
GLOSSARY: dict[str, str] = {
    # Structures
    "yield strength": "the load at which a material stops springing back and stays bent",
    "ultimate strength": "the load at which a material tears apart",
    "factor of safety": "how many times stronger something is than it needs to be",
    "margin of safety": "what is left over after the required safety factor is applied; below zero fails",
    "hoop stress": "the sideways stretch in a pipe or tank wall, the way a barrel hoop is pulled",
    "buckling": "folding sideways under a squeeze, long before the material itself gives way",
    "second moment of area": "how much a cross-section resists bending, from its shape alone",
    "deflection": "how far something bends under load",
    "fatigue": "failure from being loaded and unloaded many times, well below the breaking load",
    "stress concentration": "a corner or hole where the load crowds together and cracks start",
    "preload": "the squeeze already in a bolted joint before any working load arrives",
    "modulus": "how stiff a material is, separate from how strong it is",
    "poisson ratio": "how much something gets thinner when it is stretched",
    "creep": "slowly stretching under a steady load, over months rather than seconds",
    "limit load": "the worst load the thing will actually see in service",
    "collapse depth": "the depth at which a hull crushes; kept well below where it works",
    # Electrical
    "voltage": "how hard the electricity is pushed",
    "current": "how much electricity is flowing",
    "resistance": "how much a conductor fights the current, turning some into heat",
    "impedance": "resistance for signals that change, which depends on how fast they change",
    "voltage drop": "the push lost along a wire, so the far end gets less than the near end",
    "derating": "running a part below its rated limit so it lasts",
    "ampacity": "how much current a wire can carry before it gets too hot",
    "duty cycle": "the fraction of the time something is actually switched on",
    "kirchhoff's current law": "everything flowing into a junction has to flow back out",
    "ground": "the common reference everything else's voltage is measured against",
    "brushless": "a motor switched electronically rather than by rubbing contacts",
    # Thermal
    "thermal resistance": "how much temperature it costs to push heat along a path",
    "convection": "heat carried away by moving air or liquid",
    "conduction": "heat travelling through solid material",
    "radiation": "heat leaving as infrared, which works even in a vacuum",
    "emissivity": "how well a surface radiates heat away; matt black is best",
    "heat sink": "a finned lump of metal that gives heat more surface to leave from",
    "thermal expansion": "how much something grows when it warms up",
    # Fluids
    "reynolds number": "whether a flow is smooth and orderly or churning and mixed",
    "laminar": "flowing in smooth layers that do not mix",
    "turbulent": "churning and mixing, which costs more pressure and carries more heat",
    "pressure drop": "the pressure it costs to push fluid through a pipe",
    "head": "pressure expressed as the height of a column of the fluid",
    "cavitation": "bubbles forming where the pressure drops, then collapsing and eating the metal",
    "buoyancy": "the upward push a fluid gives anything that displaces it",
    "displacement": "the volume of fluid something pushes out of the way",
    "viscosity": "how thick a fluid is; honey against water",
    "orifice": "a deliberate restriction, used to measure or limit flow",
    # Chemical and biological
    "stoichiometry": "the fixed ratios in which substances react",
    "residence time": "how long material stays in a vessel on average",
    "conversion": "the fraction of the feed that has actually reacted",
    "michaelis-menten": "the standard curve for how enzyme speed depends on how much there is to work on",
    "kla": "how fast a vessel can get oxygen into the liquid",
    "plug flow": "everything moving through together, nothing overtaking",
    "mass balance": "what goes in equals what comes out, plus what builds up",
    # Controls
    "damping ratio": "how much a system resists overshooting; too little rings, too much crawls",
    "natural frequency": "the rate at which something wants to vibrate on its own",
    "nyquist": "you have to look at least twice as often as the fastest thing you want to see",
    "setpoint": "the value the controller is trying to hold",
    "overshoot": "going past the target before settling back",
    "latency": "the delay between something happening and the response",
    "quantisation": "the step size of a digital measurement; nothing finer can be reported",
    # Practice
    "tolerance": "the allowed range a real part can be made to, since nothing is exact",
    "stack-up": "how several parts' tolerances add together across an assembly",
    "gd&t": "a symbolic language for saying exactly which features have to line up with which",
    "bill of materials": "the numbered list of every part it takes to build one",
    "single point of failure": "a part with nothing behind it, so the whole thing stops when it does",
    "fmea": "a table of every way it can fail, how bad that is, and whether anybody would notice",
    "trl": "how far a technology is from being proven in real use, on a scale of one to nine",
    "mass growth allowance": "the weight a design will gain before it is finished, added on purpose",
    "uncertainty": "how wrong a number could be; a figure without one is a guess in disguise",
}


def gloss(term: str) -> str:
    """The lay meaning of a term, or nothing if it is already plain."""
    return GLOSSARY.get(str(term or "").strip().lower(), "")


def annotate_terms(text: str, *, limit: int = 3) -> list[tuple[str, str]]:
    """Which glossary terms appear in a passage, so a panel can define them."""
    haystack = str(text or "").lower()
    found: list[tuple[str, str]] = []
    for term, meaning in GLOSSARY.items():
        if term in haystack:
            found.append((term, meaning))
        if len(found) >= limit:
            break
    return found


def explain_part(part) -> str:
    """One paragraph about a part: what it is, does, weighs and costs."""
    name = part.lay_name or part.name
    pieces = [f"{name.capitalize()}."]
    if part.function:
        pieces.append(part.function.rstrip(".") + ".")
    if part.solid is not None:
        pieces.append(f"It is {part.solid.describe()}.")
    if part.material is not None:
        pieces.append(
            f"Made of {part.material.name.lower()} — {part.material.feels_like.rstrip('.')}."
        )
    mass = part.mass()
    if mass is not None:
        pieces.append(
            f"It weighs {mass.text()}"
            + (f", for all {part.quantity} of them." if part.quantity > 1 else ".")
        )
    if part.sourcing.specification:
        verb = {
            "buy": "Bought in",
            "off_the_shelf": "Bought in",
            "stock": "Taken from stock",
            "print": "3D printed",
            "machine": "Machined",
            "cut": "Cut",
            "fabricate": "Fabricated",
            "mould": "Moulded",
            "assemble": "Assembled",
        }.get(part.sourcing.method, "Obtained")
        pieces.append(f"{verb} as {part.sourcing.specification}.")
    return " ".join(pieces)


def explain_finding(finding) -> str:
    """A finding written out with its arithmetic, for somebody checking it."""
    lines = [finding.plain]
    if finding.formula:
        lines.append(f"Worked out as {finding.substituted()}.")
    if finding.method:
        lines.append(f"Method: {finding.method}.")
    if finding.assumptions:
        lines.append("Assumes " + ", ".join(finding.assumptions) + ".")
    if finding.advice:
        lines.append(f"To fix it: {finding.advice}")
    return " ".join(lines)


def reading_order(design) -> list:
    """Parts in the order a person would meet them following the energy.

    Starts at whatever supplies power and walks outward through the
    connections. A part nothing connects to comes last, which is also a
    signal: it may have been forgotten.
    """
    by_id = {p.id: p for p in design.parts}
    neighbours: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for link in design.connections:
        a = link.source.split(".")[0]
        b = link.target.split(".")[0]
        if a in by_id and b in by_id:
            neighbours[a].append((b, link))
            neighbours[b].append((a, link))
    sources = [
        p.id
        for p in design.parts
        if any(k in p.ratings for k in ("supply_power", "capacity", "output_power"))
        or any(port.role in {"source", "output", "outlet", "supply"} for port in p.ports)
    ]
    if not sources:
        sources = sorted(neighbours, key=lambda pid: -len(neighbours[pid]))[:1]
    order: list = []
    seen: set[str] = set()
    frontier = list(sources)
    while frontier:
        current = frontier.pop(0)
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        order.append(by_id[current])
        for nxt, _link in neighbours[current]:
            if nxt not in seen:
                frontier.append(nxt)
    for part in design.parts:
        if part.id not in seen:
            order.append(part)
    return order


def narrate(design, findings: tuple = (), *, limit: int = 9) -> str:
    """Walk the design and say what happens, stage by stage.

    Generated from the graph, so it cannot describe a connection the model
    does not have and cannot omit one it does.
    """
    order = reading_order(design)
    if not order:
        return ""
    by_subject: dict[str, list] = defaultdict(list)
    for finding in findings:
        by_subject[finding.subject].append(finding)

    lines: list[str] = []
    if design.purpose:
        lines.append(design.purpose.rstrip(".") + ".")

    links_from: dict[str, list] = defaultdict(list)
    for link in design.connections:
        links_from[link.source.split(".")[0]].append(link)

    for part in order[:limit]:
        name = part.lay_name or part.name.lower()
        sentence = f"The {name}"
        if part.function:
            sentence += f" {part.function[0].lower()}{part.function[1:].rstrip('.')}"
        else:
            sentence += " is part of the assembly"
        outgoing = links_from.get(part.id, [])
        if outgoing:
            described = []
            for link in outgoing[:2]:
                target = design.part(link.target.split(".")[0])
                if target is None:
                    continue
                spec = get_domain(link.domain)
                carried = spec.carries or spec.through_name
                amount = f" ({link.through.text()})" if link.through is not None else ""
                described.append(
                    f"{carried}{amount} to the {target.lay_name or target.name.lower()}"
                )
            if described:
                sentence += ", passing " + " and ".join(described)
        sentence += "."
        headline = next(
            (f for f in by_subject.get(part.id, ()) if f.verdict in {"fail", "watch"}),
            None,
        )
        if headline is not None:
            # The first sentence of the finding, not the whole thing. A walk
            # through the design that pastes in three paragraphs of failure
            # analysis per part stops being a walk through the design.
            first = headline.plain.split(". ")[0].rstrip(".")
            sentence += f" {first}."
        lines.append(sentence)

    if len(order) > limit:
        lines.append(
            f"{len(order) - limit} further parts are listed in the parts table."
        )
    return " ".join(lines)
