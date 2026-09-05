"""Design something and draw it, with every number computed rather than written.

The split is the whole point. The model decides WHAT to build: the parts,
what each one is for, what it is made of, roughly how big, what connects to
what, and what the thing has to achieve. It is never asked for a mass, a
stress, a pressure drop, a safety factor or a coordinate, because those are
the numbers a model gets confidently wrong and a drawing makes look
authoritative.

Everything downstream is arithmetic. Mass comes from the geometry and the
material density. Stress comes from the pressure and the wall. Where the
parts sit comes from which of them enclose the others. And a value that
cannot name the formula behind it is dropped before it reaches a sheet.

LIVE, 2026-08-24: asked to show how something works, the honest options were
prose or an image model. Prose cannot be dimensioned and an image model
draws a plausible-looking machine that does not work. This is the third
option.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


class DesignEngineeringInput(BaseModel):
    name: str = Field("", description="What the thing is called.")
    purpose: str = Field(
        "", description="What it has to do, in one sentence. Shown on the drawing."
    )
    discipline: str = Field(
        "mechanical",
        description=(
            "mechanical, electrical, fluid, thermal, chemical, bio, controls or "
            "structural. Sets which analyses are expected, not which are run."
        ),
    )
    parts: list[dict] = Field(
        default_factory=list,
        description=(
            "One entry per part. Each takes: name; function (one sentence on what it "
            "does); lay_name (what to call it for a non-engineer); solid (a shape and "
            "its dimensions WITH UNITS, e.g. {\"kind\": \"tube\", \"outer_diameter\": "
            "\"300 mm\", \"wall\": \"12 mm\", \"height\": \"700 mm\"}); material (a name "
            "such as aluminium, Ti-6Al-4V, ABS); quantity; subsystem; tags (motor, "
            "sensor, battery, enclosure, beam, pump, valve, vessel, rotating); "
            "ratings (declared figures with units: power, efficiency, torque, "
            "capacity); ports (each with name, domain, role of source or sink, and "
            "the voltage/pressure/flow it carries); and sourcing (method of buy, "
            "machine, print, cut or fabricate, plus a specification). Do not give "
            "positions or coordinates; the runtime works out where everything sits."
        ),
    )
    connections: list[dict] = Field(
        default_factory=list,
        description=(
            "What joins what: from and to as part.port, a domain (electrical, fluid, "
            "thermal, signal, data, structural, mechanical_rotary), and what it "
            "carries as across and through with units."
        ),
    )
    requirements: list[dict] = Field(
        default_factory=list,
        description=(
            "What it has to achieve: a statement, a target with units, a comparison, "
            "and the check that settles it. Leave check empty if unsure; the runtime "
            "reports it as unverified rather than passed."
        ),
    )
    subsystems: list[dict] = Field(
        default_factory=list,
        description="Named groups of parts: id, name, purpose.",
    )
    environment: dict = Field(
        default_factory=dict,
        description=(
            "The conditions it works in: depth, ambient_temperature, speed, fluid, "
            "mass_budget, maturity. Quantities take units; names stay as names."
        ),
    )
    # Names the model has reached for when there is no schema in front of it.
    components: list[dict] = Field(default_factory=list, description="Alias for parts.")
    items: list[dict] = Field(default_factory=list, description="Alias for parts.")
    links: list[dict] = Field(default_factory=list, description="Alias for connections.")
    request: str = Field("", description="What was asked for, used for the title.")
    theme: str = Field("", description="drafting, instrument or blueprint.")
    sheet_size: str = Field("A3", description="A4, A3, A2, A1 or A0.")
    views: list[str] = Field(
        default_factory=list,
        description=(
            "assembly, exploded, schematic, section, orthographic. Empty picks the "
            "ones this design has something to show in."
        ),
    )
    out_dir: str = Field("", description="Where to write it; empty uses the standard place.")


class DesignEngineeringSkill(BaseSkill):
    name = "design_engineering"
    description = (
        "Design a physical, electrical or process thing and produce checked engineering "
        "drawings of it: an assembly with labelled callouts, an exploded view with a "
        "parts list, a wiring or piping schematic, a section, and dimensioned views. "
        "Computes mass, stress, safety factors, power, heat, flow and buoyancy from the "
        "geometry and the materials rather than stating them, and writes downloadable "
        "SVG, printable HTML, STL, 3MF, DXF, OpenSCAD, CSV and JSON. Use when asked to "
        "design or engineer a device, machine, mechanism, vehicle, robot, drone, "
        "enclosure, frame, circuit or part, for a schematic, a blueprint, an assembly "
        "or a wiring diagram, to size a bracket or a beam, or to work out whether "
        "something will hold. Not for pictures or artwork."
    )
    input_model = DesignEngineeringInput

    timeout_seconds = 120.0
    metabolic_cost = 2
    effect_scope = "read_write_artifacts"
    requires_approval = False

    @staticmethod
    def available_here() -> bool:
        return True

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = DesignEngineeringInput(**params)
        elif not isinstance(params, DesignEngineeringInput):
            params = DesignEngineeringInput.model_validate(params)

        from core.engineering.studio import design_from_async

        parts = list(params.parts) or list(params.components) or list(params.items)
        if not parts:
            return {
                "ok": False,
                "skill": self.name,
                "error": "no parts were given",
                "summary": (
                    "I could not draw anything: the design had no parts in it. Each part "
                    "needs a name, what it does, a shape with dimensions, and what it is "
                    "made of."
                ),
            }

        from core.conversation.session_scope import the_persons_own_words

        asked = the_persons_own_words(params.request)
        brief = {
            "name": _title_worth_using(params.name, asked),
            "purpose": params.purpose or asked[:180],
            "discipline": params.discipline,
            "parts": parts,
            "connections": list(params.connections) or list(params.links),
            "requirements": list(params.requirements),
            "subsystems": list(params.subsystems),
            "environment": dict(params.environment),
        }

        try:
            result = await design_from_async(
                brief,
                theme=params.theme or "drafting",
                size=(params.sheet_size or "A3").upper(),
                kinds=tuple(params.views),
                out_dir=params.out_dir or None,
            )
        except (KeyError, ValueError, TypeError) as exc:
            # A brief that names a shape or a material nothing knows fails
            # here with the name in the message, which is repairable. A
            # drawing built on a guessed substitute would not be.
            return {
                "ok": False,
                "skill": self.name,
                "error": str(exc),
                "summary": f"I could not build that design: {exc}",
            }

        verdict = result.verdict
        payload: dict[str, Any] = {
            "ok": verdict.ok,
            "skill": self.name,
            "design": result.design.name,
            "fingerprint": result.design.fingerprint(),
            "views": list(result.kinds_drawn),
            "findings": len(result.findings),
            "dropped": len(verdict.dropped),
            "blocking": [p.to_dict() for p in verdict.blocking],
            "warnings": [p.to_dict() for p in verdict.warnings][:8],
            "buildable": verdict.buildable,
            "narrative": result.narrative,
            "checks_run": list(verdict.checks_run),
            "validation": verdict.validation_note,
            "files": [entry.to_dict() for entry in result.bundle.files],
            "paths": list(result.bundle.written),
            "headline": _headline(result),
            "summary": result.summary(),
        }
        if result.plan is not None:
            payload["build"] = result.plan.to_dict()

        # A thing that exists is the answer to a request for it. The same
        # rule build_document lives by: the file is the deliverable, and a
        # reply that re-narrates it and never says where it is has not
        # delivered anything.
        try:
            from core.conversation.session_scope import record_solved_answer

            record_solved_answer("built_artifact", payload["summary"])
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        return payload


def _headline(result) -> list[str]:
    """The handful of numbers worth putting in the reply itself."""
    lines: list[str] = []
    by_id = {f.id: f for f in result.findings}
    # Mass twice over would be noise, so the predicted figure wins and the
    # bare drawn one is only used when there is no growth allowance.
    mass = by_id.get("assurance.mass_growth") or by_id.get("mass.total")
    if mass is not None:
        lines.append(f"Mass: {mass.value.text()}")
    for key, label in (
        ("electrical.total_draw", "Power"),
        ("envelope.size", "Overall size"),
    ):
        finding = by_id.get(key)
        if finding is not None:
            lines.append(f"{label}: {finding.value.text()}")
    for finding in result.findings:
        if finding.verdict == "fail":
            lines.append(f"FAILS — {finding.name}: {finding.plain}")
    for finding in result.findings:
        if finding.verdict == "watch" and len(lines) < 8:
            lines.append(f"Watch — {finding.name}: {finding.plain}")
    return lines[:8]


def _title_worth_using(given: str, asked: str) -> str:
    """A title from what was asked for, when the model did not supply one."""
    import re

    text = str(given or "").strip()
    if text:
        return text
    words = re.findall(r"[A-Za-z0-9-]+", str(asked or ""))
    ignore = {
        "a", "an", "the", "me", "my", "please", "can", "you", "design", "draw",
        "make", "build", "show", "for", "of", "to", "with", "some", "that",
    }
    kept = [w for w in words if w.lower() not in ignore][:6]
    return " ".join(kept).capitalize() if kept else "Untitled design"
