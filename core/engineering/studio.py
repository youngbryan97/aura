"""One call, from a design brief to a finished, checked, downloadable design.

Everything else in this package does one job. This puts them in order and is
the only entry point anything outside needs: resolve the brief into a model,
place the parts, run every analysis that has its inputs, verify the result,
draw it, write down how it works, work out what it would take to build, and
export the lot.

The order is not arbitrary. Verification happens before drawing, so a number
that cannot say where it came from never reaches a sheet. The build plan
happens after verification, so nobody is told to build something that failed
its own checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DesignStudioResult", "design_from", "SHEET_ORDER"]

#: Which sheets a design gets, and when each one is worth drawing.
SHEET_ORDER: tuple[tuple[str, str], ...] = (
    ("assembly", "always"),
    ("exploded", "more than one part with a shape"),
    ("schematic", "any connections at all"),
    ("section", "anything hollow"),
    ("orthographic", "anything that has to be made to size"),
)


@dataclass(frozen=True, slots=True)
class DesignStudioResult:
    """Everything one design produced."""

    design: Any
    findings: tuple
    verdict: Any
    sheets: tuple
    narrative: str
    plan: Any
    bundle: Any
    kinds_drawn: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.verdict.ok

    def summary(self) -> str:
        """What to say about this design, in the reply that produced it."""
        pieces = [self.verdict.plain()]
        if self.plan is not None:
            pieces.append(self.plan.plain())
        if self.bundle is not None and self.bundle.written:
            pieces.append(
                f"{len(self.bundle.written)} files written, starting with "
                f"{self.bundle.written[0]}."
            )
        return " ".join(pieces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "design": self.design.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "verification": self.verdict.to_dict(),
            "narrative": self.narrative,
            "sheets": [sheet.to_dict() for sheet in self.sheets],
            "kinds_drawn": list(self.kinds_drawn),
            "build": self.plan.to_dict() if self.plan is not None else None,
            "bundle": self.bundle.to_dict() if self.bundle is not None else None,
            "summary": self.summary(),
        }


def _kinds_worth_drawing(design, requested: tuple[str, ...]) -> tuple[str, ...]:
    """Only draw the views this design actually has something to show in.

    A section through a design with nothing hollow in it is an empty
    rectangle, and an exploded view of a single part is that part.
    """
    if requested:
        return requested
    shaped = [p for p in design.parts if p.solid is not None]
    kinds = ["assembly"]
    if len(shaped) > 1:
        kinds.append("exploded")
    if design.connections:
        kinds.append("schematic")
    if any(
        p.solid is not None and p.solid.kind in {"tube", "dome", "capsule", "ellipsoid"}
        for p in design.parts
    ):
        kinds.append("section")
    if shaped:
        kinds.append("orthographic")
    return tuple(kinds)


def design_from(
    brief: dict[str, Any],
    *,
    theme: str = "drafting",
    size: str = "A3",
    kinds: tuple[str, ...] = (),
    formats: tuple[str, ...] = (),
    out_dir: str | None = None,
    write: bool = True,
    check_validation: bool = True,
    learn: bool = True,
) -> DesignStudioResult:
    """Take a design brief all the way to checked drawings and files."""
    import time

    from core.engineering.analysis import run_analyses
    from core.engineering.build import build_plan
    from core.engineering.draw.schematic import schematic_drawer
    from core.engineering.draw.sheet import compose_sheet
    from core.engineering.explain import narrate
    from core.engineering.export import build_bundle, write_bundle
    from core.engineering.layout import arrange
    from core.engineering.model import design_from_brief
    from core.engineering.verify import verify_design

    started = time.perf_counter()
    design = arrange(design_from_brief(brief))
    findings = run_analyses(design)
    verdict = verify_design(design, findings, check_validation=check_validation)
    # Only findings that carry their working reach a drawing.
    shown = verdict.grounded
    narrative = narrate(design, shown)
    plan = build_plan(design, shown)

    drawn = _kinds_worth_drawing(design, kinds)
    sheets = []
    for kind in drawn:
        sheets.append(compose_sheet(
            design, shown, kind=kind, size=size, theme=theme,
            narrative=narrative, schematic_drawer=schematic_drawer(shown),
        ))

    bundle = build_bundle(
        design, shown, verdict,
        sheets=tuple(sheets), plan=plan, narrative=narrative, formats=formats,
    )
    if write:
        bundle = write_bundle(bundle, out_dir)

    result = DesignStudioResult(
        design=design,
        findings=shown,
        verdict=verdict,
        sheets=tuple(sheets),
        narrative=narrative,
        plan=plan,
        bundle=bundle,
        kinds_drawn=drawn,
    )

    # Two things happen after a design finishes, and neither may fail it.
    # Metacognition gets a record it can measure her own engineering from,
    # and what the design taught that generalises is written into memory.
    from core.engineering.faculty import record_design
    from core.engineering.knowledge import record_design_knowledge
    from core.runtime.errors import record_degradation

    try:
        record_design(result, seconds=time.perf_counter() - started)
    except (AttributeError, TypeError, ValueError) as exc:
        record_degradation("engineering.studio", exc, action="recording a design for metacognition")
    if learn:
        try:
            record_design_knowledge(design, shown, verdict)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            record_degradation("engineering.studio", exc, action="recording what a design taught")
    return result


async def design_from_async(
    brief: dict[str, Any], **options: Any
) -> DesignStudioResult:
    """The same, off the event loop, for the live runtime.

    Every part of this is CPU work and one part is a synchronous fsync, so
    running it inline on the loop would stall the runtime for as long as the
    drawing takes.
    """
    import asyncio
    import functools

    return await asyncio.to_thread(functools.partial(design_from, brief, **options))
