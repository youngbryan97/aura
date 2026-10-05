"""Program DNA reconstruction skill.

Live capability surface for authorized clean-room reconstruction and mechanism
study of programs from observable behavior, open/user-owned source, metadata,
UI notes, host/Aura interaction traces, and research evidence.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.service_registry import get_runtime_service
from core.service_names import ServiceNames
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

logger = logging.getLogger(__name__)


class ProgramDNAInput(BaseModel):
    target: str = Field("", description="Program/app/library name or target path label. Read from the request when not given.")
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
        "research notes, and similar-program hints. Asked to reconstruct or rebuild a named "
        "application, builds a working one from what is written about it and checks every "
        "feature by using it."
    )
    input_model = ProgramDNAInput
    #: Rebuilding an application is written part by part by her own model and
    #: checked as it goes (core/rebuilding), which is measured in tens of
    #: minutes; the other lanes finish well inside it.
    timeout_seconds = 10800.0
    metabolic_cost = 2
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ProgramDNAInput(**params)
        elif not isinstance(params, ProgramDNAInput):
            params = ProgramDNAInput.model_validate(params)

        asked = " ".join(str((context or {}).get(key) or "") for key in ("objective", "message", "user_message", "goal")).strip()
        # The program the person named, in their words, over the caller's
        # restatement of it: LIVE 2026-10-05 the target arrived as
        # "reconstruction of microsoft word" and no article was found for it.
        named = await the_program_named_in(asked) if asked else ""
        params.target = named or params.target.strip()
        if not params.target:
            return {"ok": False, "skill": self.name, "error": "no program was named",
                    "summary": "Which program should I reconstruct? I need its name."}
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
        # Any other program asked to be reconstructed is rebuilt as a working
        # one: a blueprint is not what a person asking for a program can use.
        if params.analysis_mode == "reconstruct" and not engine._policy_blocks(
            str(params.authorization or "").strip().lower(), f"{asked} {params.target}".lower()
        ):
            return await _rebuild_it(params, self.name, asked, context)

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
        # not a failure: the docstring above says it: fall through to the caller.
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
        # not a failure: the docstring above says it: the caller falls back to structural
        # reconstruction.
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


async def the_program_named_in(asked: str) -> str:
    """The program a request asks to reconstruct, read by code from its words and her corpus (core/rebuilding/which_program_is_named.py)."""
    from core.rebuilding.which_program_is_named import the_program_named_in as named_in

    return await asyncio.to_thread(named_in, asked or "")


async def _rebuild_it(params: ProgramDNAInput, skill: str, asked: str = "", context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild the program clean-room as a working one, open it where the person can use it, and say what works."""
    from core.rebuilding.her_model import ask_her_model
    from core.rebuilding.rebuilding_a_program import rebuild
    from core.runtime.payload_values import payload_path
    from core.skills.screen_pursuit import _tell

    # Confined to the runtime's own place for built programs, as build_app is:
    # a folder the caller names is a folder under it, never anywhere on disk.
    root = await asyncio.to_thread(lambda: (Path(__file__).resolve().parents[2] / "artifacts" / "rebuilt_programs").resolve())
    where = payload_path({"out_dir": str(params.output_dir or "")}, "out_dir", root=root, default=root)
    rebuilt = await rebuild(params.target, ask_her_model, where, tell=_tell, asked=asked)
    opened = ""
    if rebuilt.built is not None and rebuilt.built.working():
        opened = await _open_for_the_person(rebuilt.built.path)
    summary = rebuilt.summary() + (f" {opened}" if opened else "") + await _used_as_asked(rebuilt.built, asked, context)
    working = rebuilt.built.working() if rebuilt.built is not None else []
    return {
        "ok": bool(working),
        "skill": skill,
        "target": params.target,
        "path": str(rebuilt.built.path) if rebuilt.built is not None else "",
        "features": [o.feature.name for o in working],
        "not_working": [o.feature.name for o in (rebuilt.built.outcomes if rebuilt.built is not None else []) if not o.kept],
        "sources": rebuilt.sources,
        "seconds": round(rebuilt.seconds, 1),
        "summary": summary,
    }


async def _used_as_asked(built: Any, asked: str, context: dict[str, Any] | None) -> str:
    """When the request also asks for the program to be used ("prove it works by..."), that, done with its own controls."""
    from core.skills.using_a_program import how_the_use_went, use_what_she_built, what_to_do_with_it

    task = what_to_do_with_it(asked)
    if built is None or not task or not built.working() or built.path.suffix.lower() != ".html":
        return ""
    return " " + how_the_use_went(await use_what_she_built(built.path, task, asked, context))


#: Windows left open for the person, held so nothing closes them behind their back.
_LEFT_OPEN: list[Any] = []


async def _open_for_the_person(path: Path) -> str:
    """The program installed as a Mac application and opened; else opened in a window of her browser."""
    if path.suffix.lower() not in (".html", ".htm"):
        return ""  # code is used by calling it, not by looking at it
    try:
        import json as _json
        import subprocess

        from core.rebuilding.as_a_mac_app import as_a_mac_app

        kept = path.parent / "program.json"
        made = _json.loads(kept.read_text("utf-8")) if kept.exists() else {}
        name = str(made.get("title") or path.parent.name.replace("-", " ").title())
        app = await asyncio.to_thread(as_a_mac_app, path, name, accent=str(made.get("accent") or ""))
        await asyncio.to_thread(subprocess.run, ["open", str(app)], check=False, timeout=30)
        return f"It is installed as an application, {app}, and I opened it for you."
    except Exception as why:  # noqa: BLE001 - no application here: the page still opens in her browser
        logger.info("the program could not be made an application: %s", why)
    try:
        from core.capabilities.phantom_browser import PhantomBrowser

        browser = PhantomBrowser(visible=True, browser_type="chromium", principal="owner")
        if not await browser.ensure_ready():
            return ""
        await browser.page.goto(path.as_uri(), wait_until="load")
        await browser.page.bring_to_front()
        await browser.come_forward()
        _LEFT_OPEN.append(browser)
        return "I opened it in a window for you."
    except Exception as why:  # noqa: BLE001 - not opening it leaves it on disk, where the summary says it is
        logger.info("the rebuilt program could not be opened: %s", why)
        return ""


__all__ = ["ProgramDNAInput", "ProgramDNAReconstructSkill", "the_program_named_in"]
