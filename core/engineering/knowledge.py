"""What a design teaches that outlives the design.

One finished vehicle is a datum. What generalises is the relationship behind
it: that a tube squeezed from outside collapses long before it yields, that
a fuse must be smaller than the wire it protects, that drag goes as the
square of speed. Those are facts about the world, and they are worth
remembering separately from the drawing that happened to produce them.

So each analysis that has a transferable principle behind it says what that
principle is, and a finished design yields two kinds of record: the
principle, stated generally, and the worked case that instantiates it with
real numbers. The second is what makes the first recallable later — a rule
with no example attached is a rule nobody applies.

Nothing here writes a principle the analysis did not name. A lesson with no
finding behind it would be exactly the invented knowledge the rest of this
package exists to prevent, one level up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Lesson",
    "PRINCIPLES",
    "lessons_from",
    "record_design_knowledge",
    "record_design_knowledge_async",
]


@dataclass(frozen=True, slots=True)
class Lesson:
    """One thing now known, and the case that showed it."""

    #: The general statement, true beyond this design.
    principle: str
    #: The worked case, with the numbers that made it concrete.
    evidence: str
    #: Which finding produced it, so it can be traced back.
    finding_id: str
    discipline: str
    #: Where the principle comes from, when it comes from somewhere.
    source: str = ""
    kind: str = "principle"

    def statement(self) -> str:
        return f"{self.principle} {self.evidence}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "principle": self.principle,
            "evidence": self.evidence,
            "finding": self.finding_id,
            "discipline": self.discipline,
            "source": self.source,
            "kind": self.kind,
            "statement": self.statement(),
        }


#: The general rule behind each family of finding. Keyed by the prefix a
#: finding id carries, so an analysis that produces a new finding without a
#: principle here simply teaches nothing rather than teaching something
#: invented.
PRINCIPLES: dict[str, tuple[str, str, str]] = {
    "buckle.external": (
        "A tube squeezed from outside collapses by buckling inward long before it "
        "reaches the strength of its material, so a depth or vacuum rating is set by "
        "stiffness and roundness rather than by how strong the metal is.",
        "structures",
        "Windenburg and Trilling, elastic instability under external pressure",
    ),
    "buckle.column": (
        "A slender column folds sideways at a load far below what would crush it, and "
        "the load it folds at falls with the square of its unsupported length. Bracing "
        "the middle of a strut is worth more than thickening it.",
        "structures",
        "Euler column buckling",
    ),
    "stress.pressure": (
        "Hoop stress in a pressure wall rises with radius and falls with thickness, so "
        "a larger vessel at the same pressure needs a proportionally thicker wall. "
        "Doubling the diameter doubles the stress.",
        "structures",
        "Thin-wall membrane equation",
    ),
    "stress.thermal": (
        "A part held so it cannot expand builds up stress from temperature alone, with "
        "nothing touching it. Two materials with different expansion rates bolted "
        "together will tear at their joint if the temperature swings far enough.",
        "structures",
        "Restrained thermal expansion",
    ),
    "electrical.drop": (
        "Voltage lost along a cable is current times resistance, and resistance rises "
        "as the conductor gets thinner. Long runs at low voltage are where this bites: "
        "the same power at half the voltage loses four times as much in the wire.",
        "electrical",
        "Ohm's law over the conductor",
    ),
    "electrical.fuse": (
        "A fuse protects the wire, not the load. It has to be rated below what the "
        "conductor can carry, or the wire becomes the fuse and does so somewhere "
        "nobody can see.",
        "electrical",
        "Overcurrent protection practice",
    ),
    "electrical.runtime": (
        "Battery runtime is usable energy over average draw, and usable is not the same "
        "as rated: a lithium pack emptied completely is a pack damaged, so about a fifth "
        "of its capacity is not available for planning.",
        "electrical",
        "Cell life against depth of discharge",
    ),
    "thermal.surface_rise": (
        "Heat leaves a sealed box through its outside surface, so how hot the inside "
        "gets depends on surface area and the fluid outside it. Water carries heat away "
        "roughly a hundred times better than still air.",
        "thermal",
        "Newton's law of cooling",
    ),
    "thermal.conduction": (
        "Pushing heat along a path costs temperature in proportion to the path's length "
        "and inversely to its area and conductivity. A thermal bottleneck shows up as "
        "the source sitting hotter than the sink, not as anything failing.",
        "thermal",
        "Fourier conduction",
    ),
    "fluid.drag": (
        "Drag rises with the square of speed and the power to overcome it with the cube, "
        "so going twice as fast costs eight times the power. Range and speed trade "
        "against each other far more sharply than intuition suggests.",
        "fluids",
        "Standard drag equation",
    ),
    "fluid.buoyancy": (
        "Anything in a fluid is pushed up by the weight of fluid it displaces. Neutral "
        "buoyancy — floating neither up nor down — is what a vehicle that must hold "
        "depth is trimmed to, and it takes ballast to reach.",
        "fluids",
        "Archimedes' principle",
    ),
    "fluid.drop": (
        "Pressure lost in a pipe rises with the square of flow speed and falls sharply "
        "with bore, so a pipe one size larger costs far less pressure than a pump one "
        "size larger costs money.",
        "fluids",
        "Darcy-Weisbach",
    ),
    "fluid.hydrostatic": (
        "Pressure underwater rises about one atmosphere for every ten metres of depth, "
        "whatever the shape of the water above it.",
        "fluids",
        "Hydrostatic column",
    ),
    "controls.sampling": (
        "A signal has to be sampled at more than twice its highest frequency or fast "
        "changes come back as slow ones that were never there, and no amount of "
        "filtering afterwards can undo it.",
        "controls",
        "Nyquist criterion",
    ),
    "controls.response": (
        "A feedback loop with too little damping overshoots and rings; with too much it "
        "crawls. The useful band sits around a damping ratio of 0.7, which arrives "
        "quickly with a small, brief overshoot.",
        "controls",
        "Second-order step response",
    ),
    "assurance.margin": (
        "A margin of safety is what remains after the required design factor has already "
        "been applied to the load. A bare strength-to-load ratio is a different number "
        "and comparing the two is how two engineers agree on a figure and disagree "
        "about whether it passes.",
        "assurance",
        "NASA-STD-5001B",
    ),
    "assurance.mass_growth": (
        "A design gains weight between sketch and hardware, reliably enough that the "
        "gain is added on purpose and sized by how mature the design is. Planning "
        "against the drawn mass is planning against a number that will not be true.",
        "assurance",
        "ANSI/AIAA S-120A",
    ),
    "assurance.spf": (
        "A part with nothing in parallel stops the whole thing when it fails. Finding "
        "those while the design is still a drawing is the cheapest redundancy ever gets.",
        "assurance",
        "MIL-STD-1629A",
    ),
    "conservation": (
        "What arrives at a junction has to leave it. That one rule is Kirchhoff's "
        "current law, a mass balance and a heat balance at once, and a junction that "
        "does not balance has a branch nobody has drawn.",
        "general",
        "Conservation of the through variable at a node",
    ),
    "mass.centre_of_mass": (
        "Where something balances is the mass-weighted average of where its parts are, "
        "so one heavy part far from the middle moves the balance point more than "
        "several light ones near it.",
        "mechanics",
        "Definition of the centre of mass",
    ),
    "motion.leadscrew": (
        "A screw drive holds its load with the power off only when thread friction "
        "beats the thread angle. A fast lead back-drives and needs a brake; a fine one "
        "holds but wastes most of its input in friction.",
        "mechanics",
        "Power screw efficiency",
    ),
    "process.balance": (
        "What goes into a vessel leaves it or builds up in it. A flowsheet where the "
        "streams do not add up has a stream missing, not a rounding error.",
        "chemical",
        "Steady-state mass balance",
    ),
    "bio.oxygen": (
        "A culture stops growing at its own rate when oxygen cannot reach it fast "
        "enough, whatever the nutrients say, because oxygen is barely soluble in water.",
        "bio",
        "Two-film oxygen transfer",
    ),
}


def lessons_from(design, findings, verdict=None) -> tuple[Lesson, ...]:
    """The transferable knowledge one finished design produced.

    Only findings whose family has a stated principle teach anything, and
    only findings that passed the provenance gate are read at all.
    """
    seen: set[str] = set()
    lessons: list[Lesson] = []
    for finding in findings:
        match = None
        for prefix, entry in PRINCIPLES.items():
            if finding.id.startswith(prefix) and (match is None or len(prefix) > len(match[0])):
                match = (prefix, entry)
        if match is None:
            continue
        prefix, (principle, discipline, source) = match
        if prefix in seen:
            continue
        seen.add(prefix)
        lessons.append(Lesson(
            principle=principle,
            evidence=(
                f"Seen in {design.name}: {finding.plain} "
                f"Worked as {finding.substituted()}."
            ),
            finding_id=finding.id,
            discipline=discipline,
            source=source,
        ))

    # A design that failed teaches the failure, which is the more useful half.
    if verdict is not None:
        for problem in verdict.blocking[:3]:
            lessons.append(Lesson(
                principle=(
                    "A design can be arithmetically correct everywhere and still be "
                    "impossible, because the parts have to agree with each other as "
                    "well as with physics."
                ),
                evidence=(
                    f"In {design.name}, {problem.message} "
                    + (f"The fix: {problem.advice}" if problem.advice else "")
                ),
                finding_id=problem.code,
                discipline="assurance",
                source="Design verification",
                kind="failure",
            ))
    return tuple(lessons)


def _memory_facade():
    """The memory service, or nothing when the runtime is not up.

    A design produced from a script, a test or a cold start has no service
    container behind it. That is a normal condition, not a fault, so the
    absent-service error is caught here rather than reported as a failure of
    the design.
    """
    try:
        from core.container import get_container
        from core.exceptions import ServiceNotFoundError
        from core.service_names import ServiceNames
    except ImportError:
        return None
    try:
        return get_container().get(ServiceNames.MEMORY_FACADE)
    except (ServiceNotFoundError, AttributeError, KeyError, RuntimeError):
        return None


def _memory_payload(design, lesson: Lesson) -> tuple[str, dict[str, Any]]:
    return (
        lesson.statement(),
        {
            "kind": "engineering_principle",
            "discipline": lesson.discipline,
            "design": design.name,
            "design_fingerprint": design.fingerprint(),
            "finding": lesson.finding_id,
            "source": lesson.source,
            "transferable": lesson.kind == "principle",
            "confidence": 0.9 if lesson.kind == "principle" else 0.75,
        },
    )


def record_design_knowledge(design, findings, verdict=None) -> tuple[Lesson, ...]:
    """Write what this design taught into general memory, and return it.

    Failure to write is recorded as a degradation rather than raised: a
    design that was drawn correctly should not be reported as failed
    because the memory backend was busy.
    """
    from core.runtime.errors import record_degradation

    lessons = lessons_from(design, findings, verdict)
    if not lessons:
        return ()
    memory = _memory_facade()
    if memory is None:
        # Designing works with no memory service up; it just teaches nothing
        # that outlives the run, and says so rather than failing the design.
        return lessons
    import asyncio

    async def write() -> None:
        for lesson in lessons:
            content, metadata = _memory_payload(design, lesson)
            await memory.remember(content, metadata)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(write())
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("engineering.knowledge", exc, action="recording design lessons")
    return lessons


async def record_design_knowledge_async(design, findings, verdict=None) -> tuple[Lesson, ...]:
    """The same, from the running loop, which is where the live runtime is."""
    from core.runtime.errors import record_degradation

    lessons = lessons_from(design, findings, verdict)
    if not lessons:
        return ()
    try:
        memory = _memory_facade()
        if memory is None:
            return lessons
        for lesson in lessons:
            content, metadata = _memory_payload(design, lesson)
            await memory.remember(content, metadata)
    except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "engineering.knowledge", exc, action="recording what a design taught"
        )
    return lessons
