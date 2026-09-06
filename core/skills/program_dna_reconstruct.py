"""Program DNA reconstruction skill.

Live capability surface for authorized clean-room reconstruction and mechanism
study of programs from observable behavior, open/user-owned source, metadata,
UI notes, host/Aura interaction traces, and research evidence.
"""
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.service_registry import get_runtime_service
from core.service_names import ServiceNames
from core.skills.base_skill import BaseSkill


class ProgramDNAInput(BaseModel):
    target: str = Field(..., description="Program/app/library name or target path label.")
    authorization: str = Field(
        "unspecified",
        description=(
            "open_source | owner_authorized | explicit_permission | internal | educational | "
            "user_owned | public_observation | external_observation | host_observation | "
            "defensive_analysis | security_research"
        ),
    )
    analysis_mode: str = Field(
        "reconstruct",
        description="reconstruct | reverse_engineer | study | observe | monitor | defensive_analysis",
    )
    source_paths: list[str] = Field(default_factory=list)
    observed_behaviors: list[str] = Field(default_factory=list)
    ui_notes: list[str] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)
    research_queries: list[str] = Field(default_factory=list)
    perform_research: bool = False
    max_research_results: int = Field(3, ge=1, le=8)
    similar_programs: list[str] = Field(default_factory=list)
    api_observations: list[str] = Field(default_factory=list)
    file_formats: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    workflows: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    study_questions: list[str] = Field(default_factory=list)
    interaction_observations: list[str] = Field(default_factory=list)
    aura_interactions: list[str] = Field(default_factory=list)
    host_interactions: list[str] = Field(default_factory=list)
    network_observations: list[str] = Field(default_factory=list)
    hardware_observations: list[str] = Field(default_factory=list)
    process_observations: list[str] = Field(default_factory=list)
    security_observations: list[str] = Field(default_factory=list)
    compatibility_targets: list[str] = Field(default_factory=list)
    target_stack: str = "python"
    enable_binary_static_analysis: bool = False
    capture_live_host_snapshot: bool = False
    emit_scaffold: bool = False
    output_dir: str | None = None


class ProgramDNAReconstructSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill here
    #: returns `ok`, and a schema claiming to be complete would be wrong
    #: for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "program_dna_reconstruct"
    description = (
        "Authorized clean-room reconstruction and mechanism study of a program's behavior DNA "
        "from source, metadata, UI/UX observations, Aura/host/network/hardware interactions, "
        "research notes, and similar-program hints."
    )
    input_model = ProgramDNAInput
    timeout_seconds = 120.0
    metabolic_cost = 2
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ProgramDNAInput(**params)
        elif not isinstance(params, ProgramDNAInput):
            params = ProgramDNAInput.model_validate(params)

        engine = get_runtime_service(ServiceNames.PROGRAM_DNA_RECONSTRUCTION, default=None)
        if engine is None:
            program_dna = importlib.import_module("core.self_improvement.program_dna")

            engine = program_dna.register_program_dna_reconstruction_engine(project_root=Path.cwd())

        # A named program with published rules and a place to put it is a
        # request for a program, not a blueprint. The scaffold lane answered it
        # with a ReconstructedProgram whose execute() returns status="planned" —
        # analysis where a playable file was asked for. This lane reconstructs
        # the behaviour, verifies it against held-out positions the synthesizer
        # never saw, plays the result headlessly, and only then writes it.
        materialized = await self._materialize_named_program(engine, params)
        if materialized is not None:
            return materialized

        # Runnable reverse-engineering: observe a REAL host binary, reconstruct
        # its behavior via cognition, and VERIFY against held-out real outputs.
        # Preferred whenever the target is a known safe host binary — that is
        # the strongest, verifiable answer — for both the explicit
        # reverse_engineer mode and the default reconstruct mode.
        if params.analysis_mode in {"reverse_engineer", "reconstruct"}:
            reverse = await self._reverse_engineer_host(engine, params.target)
            if reverse is not None:
                return reverse

        result = await engine.reconstruct(params.model_dump())
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        feature_names = [feature.get("name") for feature in payload.get("features", [])]
        return {
            "ok": bool(payload.get("ok")),
            "skill": self.name,
            "target": payload.get("target_name"),
            "features": feature_names,
            "research_plan": payload.get("research_plan", []),
            "implementation_plan": payload.get("implementation_plan", []),
            "standards_review": payload.get("standards_review", []),
            "result": payload,
            "summary": self._summary(payload, feature_names),
        }


    async def _materialize_named_program(
        self, engine: Any, params: ProgramDNAInput
    ) -> dict[str, Any] | None:
        """Build and prove a known program, or return None to fall through."""
        try:
            from core.self_improvement.program_materialization import (
                materialize_program,
                resolve_program_spec,
            )
        except ImportError:
            return None
        spec = resolve_program_spec(params.target)
        if spec is None or not str(params.output_dir or "").strip():
            return None

        # expanduser() touches the filesystem, so keep it off the event loop.
        destination = await asyncio.to_thread(
            lambda: Path(str(params.output_dir)).expanduser()
        )
        report = await materialize_program(engine, spec, destination)
        written = bool(report.get("written"))
        passed = report.get("held_out_passed")
        total = report.get("held_out_total")
        if written:
            summary = (
                f"Reconstructed {spec.name} from its published rules with no source: "
                f"{passed}/{total} held-out positions reproduced exactly, and the "
                f"program I wrote {report.get('play_evidence')}. It is at "
                f"{report.get('destination')}."
            )
        else:
            summary = (
                f"I did not finish {spec.name} and I am not claiming I did: "
                f"{passed}/{total} held-out positions reproduced. "
                f"{report.get('reason') or 'verification did not pass'}. "
                "Nothing was written to disk."
            )
        return {
            "ok": written,
            "skill": self.name,
            "target": spec.name,
            "result": report,
            "summary": summary,
        }

    async def _reverse_engineer_host(self, engine: Any, target_label: str) -> dict[str, Any] | None:
        """Runnable reverse-engineering of a real host binary, verified against
        held-out real outputs. Returns None if the target is not a known safe
        host binary (caller falls back to structural reconstruction)."""
        try:
            from core.self_improvement.host_reconstruction import (
                resolve_target,
                reverse_engineer_host_binary,
            )
        except ImportError:
            return None
        target = resolve_target(target_label)
        if target is None:
            return None
        report = await reverse_engineer_host_binary(engine, target)
        status = report.get("status")
        return {
            "ok": status == "supported",
            "skill": self.name,
            "target": report.get("target"),
            "result": report,
            "summary": (
                f"Reverse-engineered {report.get('target')} from behavior only "
                f"(no source): {report.get('held_out_passed')}/{report.get('held_out_total')} "
                f"held-out cases reproduced — epistemic status: {status}."
            ),
        }

    def _summary(self, payload: dict[str, Any], feature_names: list[str]) -> str:
        if not payload.get("ok"):
            reasons = ", ".join(payload.get("blocked_reasons") or ["blocked"])
            return f"Program DNA reconstruction blocked: {reasons}"
        scaffold = payload.get("scaffold_path")
        suffix = f"; scaffold emitted at {scaffold}" if scaffold else ""
        standards = payload.get("standards_review") or []
        standards_suffix = f"; standards reviewed={len(standards)}" if standards else ""
        return (
            f"Program DNA captured for {payload.get('target_name')}: "
            f"{len(payload.get('evidence', []))} evidence item(s), "
            f"{len(feature_names)} inferred feature(s){standards_suffix}{suffix}."
        )


__all__ = ["ProgramDNAInput", "ProgramDNAReconstructSkill"]
