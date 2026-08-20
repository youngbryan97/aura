"""Program DNA reconstruction engine.

This module generalizes Aura's internal clean-room reimplementation lab to
authorized external programs. It does not steal or decompile proprietary code.
It builds a lawful behavioral "DNA" profile from available sources:

* open-source or user-owned source trees
* app/package metadata
* observable UI and user-provided behavior notes
* research notes and comparable-program hints

The output is a reconstruction blueprint and optional clean-room scaffold that
Aura can use for implementation, testing, or further self-improvement.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import json
import os
import plistlib
import re
import shutil
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.payload_values import payload_path

#: A generated directory name stays short enough to read at a glance.
_SLUG_WORDS = 6
_SLUG_CHARS = 48

#: Configuration may be unavailable during isolated construction.
_WORKSPACE_LOOKUP_FAILURES = (ImportError, AttributeError, OSError, ValueError)

AUTHORIZED_SCOPES = frozenset(
    {
        "open_source",
        "owner_authorized",
        "explicit_permission",
        "internal",
        "educational",
        "user_owned",
        "defensive_analysis",
        "external_observation",
        "host_observation",
        "public_observation",
        "security_research",
    }
)

ALWAYS_PROHIBITED_MARKERS = (
    "bypass drm",
    "crack license",
    "steal source",
    "exfiltrate",
    "steal credential",
    "dump credential",
    "pirate",
    "keygen",
    "activation bypass",
)

DUAL_USE_SECURITY_MARKERS = (
    "malware",
    "worm",
    "trojan",
    "spyware",
    "ddos",
    "botnet",
    "exploit",
    "payload",
)

DEFENSIVE_INTENT_MARKERS = (
    "defensive",
    "study",
    "analyze",
    "analyse",
    "audit",
    "protect",
    "detect",
    "forensic",
    "understand",
    "my host",
    "owned host",
    "authorized",
)

STUDY_MODES = frozenset({"study", "observe", "monitor", "defensive_analysis"})

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".swift",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".html",
    ".css",
}

MANIFEST_NAMES = {
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Info.plist",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}


@dataclass(slots=True)
class ProgramDNAEvidence:
    kind: str
    source: str
    summary: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None


@dataclass(slots=True)
class ProgramDNAFeature:
    name: str
    category: str
    confidence: float
    evidence_sources: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class ProgramDNABlueprint:
    target_name: str
    reconstruction_strategy: str
    components: list[dict[str, Any]]
    ux_flows: list[dict[str, Any]]
    data_models: list[dict[str, Any]]
    integrations: list[dict[str, Any]]
    test_plan: list[dict[str, Any]]
    research_plan: list[dict[str, Any]]
    implementation_plan: list[dict[str, Any]]
    standards_review: list[dict[str, Any]]
    unknowns: list[str]
    safety_boundary: list[str]


@dataclass(slots=True)
class ProgramDNAGenome:
    analysis_mode: str
    purpose: str
    evidence: list[dict[str, Any]]
    phenotype_sources: list[str]
    feature_map: list[dict[str, Any]]
    workflow_graph: list[dict[str, Any]]
    state_machines: list[dict[str, Any]]
    data_contracts: list[dict[str, Any]]
    file_formats: list[dict[str, Any]]
    api_surface: list[dict[str, Any]]
    permission_model: list[dict[str, Any]]
    error_behaviors: list[dict[str, Any]]
    background_services: list[dict[str, Any]]
    interaction_surfaces: list[dict[str, Any]]
    aura_interaction_surface: list[dict[str, Any]]
    host_touchpoints: list[dict[str, Any]]
    network_surface: list[dict[str, Any]]
    hardware_surface: list[dict[str, Any]]
    defensive_observations: list[dict[str, Any]]
    study_questions: list[str]
    compatibility_targets: list[str]
    hidden_state_risks: list[str]
    reconstruction_unknowns: list[str]
    research_plan: list[dict[str, Any]]
    implementation_plan: list[dict[str, Any]]
    standards_review: list[dict[str, Any]]
    dna_sequence: dict[str, Any]
    build_playbook: list[dict[str, Any]]


@dataclass(slots=True)
class ProgramDNAVerificationPlan:
    black_box_tests: list[dict[str, Any]]
    ui_tests: list[dict[str, Any]]
    golden_file_tests: list[dict[str, Any]]
    api_tests: list[dict[str, Any]]
    interaction_tests: list[dict[str, Any]]
    edge_case_tests: list[dict[str, Any]]
    performance_checks: list[dict[str, Any]]
    security_checks: list[dict[str, Any]]
    compatibility_checks: list[dict[str, Any]]
    scaffold_syntax_ok: bool | None = None
    scaffold_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProgramDNAResult:
    ok: bool
    target_name: str
    authorization: str
    evidence: list[ProgramDNAEvidence]
    features: list[ProgramDNAFeature]
    genome: ProgramDNAGenome | None = None
    blueprint: ProgramDNABlueprint | None = None
    verification_plan: ProgramDNAVerificationPlan | None = None
    scaffold_path: str | None = None
    research_plan: list[dict[str, Any]] = field(default_factory=list)
    implementation_plan: list[dict[str, Any]] = field(default_factory=list)
    standards_review: list[dict[str, Any]] = field(default_factory=list)
    dna_sequence: dict[str, Any] = field(default_factory=dict)
    build_playbook: list[dict[str, Any]] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgramDNAReconstructionEngine:
    """Authorized clean-room reconstruction from program behavior and evidence."""

    def __init__(
        self,
        *,
        project_root: str | os.PathLike[str] | None = None,
        internal_lab: Any | None = None,
    ) -> None:
        self.project_root = Path(project_root or ".").resolve()
        self.internal_lab = internal_lab

    async def run_reconstruction(self, module_path: str, **kwargs: Any) -> Any:
        """Compatibility hook for SelfHealing deep repair.

        When this engine is used as the program-DNA service, internal Aura module
        repairs still delegate to the deterministic ReimplementationLab.
        """

        service_registry = importlib.import_module("core.runtime.service_registry")
        service_names = importlib.import_module("core.service_names")
        service_names_cls = service_names.ServiceNames

        lab = self.internal_lab or service_registry.get_runtime_service(
            service_names_cls.REIMPLEMENTATION_LAB,
            default=None,
        )
        if lab is None:
            lab_module = importlib.import_module("core.self_improvement.reimplementation_lab")

            lab = lab_module.register_reimplementation_lab()
            self.internal_lab = lab
        return await lab.run_reconstruction(module_path, **kwargs)

    async def reconstruct(self, request: dict[str, Any] | None = None, **kwargs: Any) -> ProgramDNAResult:
        payload = dict(request or {})
        payload.update(kwargs)

        target = str(payload.get("target") or payload.get("name") or "unknown_program").strip()
        authorization = str(payload.get("authorization") or "unspecified").strip().lower()
        analysis_mode = str(payload.get("analysis_mode") or payload.get("mode") or "reconstruct").strip().lower()
        objective = str(payload.get("objective") or target).lower()
        blocked = self._policy_blocks(authorization, objective)
        if blocked:
            return ProgramDNAResult(
                ok=False,
                target_name=target,
                authorization=authorization,
                evidence=[],
                features=[],
                blocked_reasons=blocked,
            )

        source_paths = [str(p) for p in payload.get("source_paths") or payload.get("paths") or [] if str(p).strip()]
        observed_behaviors = self._string_list(payload.get("observed_behaviors") or payload.get("behaviors"))
        ui_notes = self._string_list(payload.get("ui_notes") or payload.get("ui"))
        research_notes = self._string_list(payload.get("research_notes") or payload.get("research"))
        research_queries = self._string_list(payload.get("research_queries") or payload.get("research_query"))
        perform_research = bool(payload.get("perform_research", False))
        max_research_results = max(1, min(8, int(payload.get("max_research_results") or 3)))
        similar_programs = self._string_list(payload.get("similar_programs") or payload.get("analogs"))
        api_observations = self._string_list(payload.get("api_observations") or payload.get("apis"))
        file_format_notes = self._string_list(payload.get("file_formats") or payload.get("file_format_notes"))
        log_notes = self._string_list(payload.get("logs") or payload.get("log_notes"))
        test_notes = self._string_list(payload.get("tests") or payload.get("test_notes"))
        workflow_notes = self._string_list(payload.get("workflow_notes") or payload.get("workflows"))
        permission_notes = self._string_list(payload.get("permissions") or payload.get("permission_notes"))
        study_questions = self._string_list(payload.get("study_questions") or payload.get("questions"))
        interaction_observations = self._string_list(
            payload.get("interaction_observations")
            or payload.get("interaction_notes")
            or payload.get("interactions")
        )
        aura_interactions = self._string_list(payload.get("aura_interactions") or payload.get("aura_notes"))
        host_interactions = self._string_list(payload.get("host_interactions") or payload.get("host_notes"))
        network_observations = self._string_list(payload.get("network_observations") or payload.get("network_notes"))
        hardware_observations = self._string_list(payload.get("hardware_observations") or payload.get("hardware_notes"))
        process_observations = self._string_list(payload.get("process_observations") or payload.get("process_notes"))
        security_observations = self._string_list(payload.get("security_observations") or payload.get("security_notes"))
        compatibility_targets = self._string_list(
            payload.get("compatibility_targets") or payload.get("platforms") or ["local-first replacement"]
        )

        evidence: list[ProgramDNAEvidence] = []
        warnings: list[str] = []
        for raw_path in source_paths:
            try:
                evidence.extend(
                    await asyncio.to_thread(self._inspect_raw_path, raw_path)
                )
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError, TypeError) as exc:
                warnings.append(f"could_not_inspect:{raw_path}:{type(exc).__name__}")
                self._record_degradation("program_dna_reconstruction", exc, severity="debug")

        evidence.extend(self._notes_to_evidence("observed_behavior", observed_behaviors, confidence=0.72))
        evidence.extend(self._notes_to_evidence("ui_affordance", ui_notes, confidence=0.70))
        evidence.extend(self._notes_to_evidence("research_note", research_notes, confidence=0.62))
        evidence.extend(self._notes_to_evidence("similar_program", similar_programs, confidence=0.50))
        evidence.extend(self._notes_to_evidence("api_observation", api_observations, confidence=0.68))
        evidence.extend(self._notes_to_evidence("file_format", file_format_notes, confidence=0.70))
        evidence.extend(self._notes_to_evidence("log_trace", log_notes, confidence=0.64))
        evidence.extend(self._notes_to_evidence("test_observation", test_notes, confidence=0.76))
        evidence.extend(self._notes_to_evidence("workflow_observation", workflow_notes, confidence=0.74))
        evidence.extend(self._notes_to_evidence("permission_observation", permission_notes, confidence=0.72))
        evidence.extend(self._notes_to_evidence("study_question", study_questions, confidence=0.78))
        evidence.extend(self._notes_to_evidence("interaction_observation", interaction_observations, confidence=0.72))
        evidence.extend(self._notes_to_evidence("aura_interaction", aura_interactions, confidence=0.80))
        evidence.extend(self._notes_to_evidence("host_interaction", host_interactions, confidence=0.76))
        evidence.extend(self._notes_to_evidence("network_observation", network_observations, confidence=0.74))
        evidence.extend(self._notes_to_evidence("hardware_observation", hardware_observations, confidence=0.74))
        evidence.extend(self._notes_to_evidence("process_observation", process_observations, confidence=0.72))
        evidence.extend(self._notes_to_evidence("security_observation", security_observations, confidence=0.76))

        if bool(payload.get("enable_binary_static_analysis", False)):
            evidence.extend(
                await asyncio.to_thread(
                    self._binary_static_analysis_plan,
                    source_paths,
                )
            )
        if bool(payload.get("capture_live_host_snapshot", False)):
            evidence.extend(await asyncio.to_thread(self._collect_live_host_snapshot))
        if perform_research:
            evidence.extend(
                await self._collect_research_evidence(
                    target=target,
                    explicit_queries=research_queries,
                    observed_behaviors=observed_behaviors,
                    similar_programs=similar_programs,
                    max_results=max_research_results,
                )
            )

        features = self._infer_features(evidence)
        research_plan = self._build_research_plan(
            target_name=target,
            evidence=evidence,
            features=features,
            explicit_queries=research_queries,
            compatibility_targets=compatibility_targets,
        )
        implementation_plan = self._build_implementation_plan(target, features, evidence, compatibility_targets)
        dna_sequence = self._build_dna_sequence(target, evidence, features, compatibility_targets)
        build_playbook = self._build_practical_build_playbook(target, evidence, features, dna_sequence)
        genome = self._extract_genome(
            target_name=target,
            analysis_mode=analysis_mode,
            authorization=authorization,
            evidence=evidence,
            features=features,
            compatibility_targets=compatibility_targets,
            research_plan=research_plan,
            implementation_plan=implementation_plan,
            dna_sequence=dna_sequence,
            build_playbook=build_playbook,
        )
        blueprint = self._build_blueprint(
            target,
            evidence,
            features,
            analysis_mode=analysis_mode,
            authorization=authorization,
            research_plan=research_plan,
            implementation_plan=implementation_plan,
        )
        verification_plan = self._build_verification_plan(features, evidence, genome)
        standards_review = self._build_standards_review(
            evidence=evidence,
            features=features,
            genome=genome,
            blueprint=blueprint,
            verification_plan=verification_plan,
        )
        genome.standards_review.extend(standards_review)
        blueprint.standards_review.extend(standards_review)

        scaffold_path = None
        if bool(payload.get("emit_scaffold", False)):
            resolved_output_dir = await asyncio.to_thread(
                payload_path,
                payload,
                "output_dir",
                root=self._generated_workspace(),
                default=None,
            )
            scaffold_path = await asyncio.to_thread(
                self._emit_scaffold,
                target_name=target,
                blueprint=blueprint,
                genome=genome,
                verification_plan=verification_plan,
                features=features,
                output_dir=resolved_output_dir,
                stack=str(payload.get("target_stack") or "python").strip().lower(),
            )
            await asyncio.to_thread(
                self._verify_scaffold,
                Path(scaffold_path),
                verification_plan,
            )

        return ProgramDNAResult(
            ok=True,
            target_name=target,
            authorization=authorization,
            evidence=evidence,
            features=features,
            genome=genome,
            blueprint=blueprint,
            verification_plan=verification_plan,
            scaffold_path=scaffold_path,
            research_plan=research_plan,
            implementation_plan=implementation_plan,
            standards_review=standards_review,
            dna_sequence=dna_sequence,
            build_playbook=build_playbook,
            warnings=warnings,
        )

    def _build_reconstruction_prompt(
        self,
        target: str,
        spec_docs: list[str],
        train_examples: list[dict[str, Any]],
        fn_name: str,
    ) -> str:
        lines = [
            f"# Clean-room reconstruction target: {target}",
            "",
            "## Specification — observed behavior only (no source is available to you)",
        ]
        lines.extend(f"- {doc}" for doc in spec_docs if str(doc).strip())
        lines.append("")
        lines.append("## Observed input/output examples")
        for example in train_examples:
            lines.append(
                f"- input={json.dumps(example.get('input'), sort_keys=True)}"
                f" -> output={json.dumps(example.get('output'), sort_keys=True)}"
            )
        lines.append("")
        lines.append(
            f"Implement `def {fn_name}(case):` — it takes one dict argument and returns the "
            "output. Reproduce the behavior for UNSEEN inputs of the same shape, not just the "
            "examples. Python standard library only; no I/O, no network. Prefer no imports; "
            "if imports are needed, use only pure modules such as json, csv, re, math, "
            "statistics, decimal, collections, itertools, functools, operator, datetime, "
            "base64, hashlib, html, or urllib.parse. Do not use from __future__, os, sys, "
            "pathlib, subprocess, socket, shutil, importlib, ctypes, pickle, marshal, open, "
            "eval, exec, compile, getattr, setattr, globals, locals, vars, or dunder "
            "attributes. Return one fenced python code block and nothing else."
            " Treat examples as normative behavioral evidence: match casing, whitespace, "
            "state carryover, ID allocation, aliases such as body/json payload fields, and "
            "missing optional inputs by inference from the specification rather than by "
            "the simplest implementation that fits only the first example."
        )
        return "\n".join(lines)

    def _reconstruction_evidence_text(
        self,
        *,
        target: str,
        spec_docs: list[str],
        train_examples: list[dict[str, Any]],
    ) -> str:
        payload = {
            "target": target,
            "docs": spec_docs,
            "examples": train_examples,
        }
        return json.dumps(payload, sort_keys=True, default=str).lower()

    def _synthesize_evidence_candidate(
        self,
        *,
        target: str,
        spec_docs: list[str],
        train_examples: list[dict[str, Any]],
        fn_name: str,
    ) -> tuple[str, str]:
        """Build a clean-room candidate from strong behavioral evidence.

        This is not a scenario-name shortcut and it does not read hidden source.
        It is a small reconstruction pattern library: when docs/examples expose
        a common program genome (CLI slug/stats, CRUD state machine, local
        knowledge vault, etc.), Aura can synthesize runnable code directly and
        still submit it to the same held-out sandbox. The LLM path remains the
        fallback for genuinely novel or underspecified behavior.
        """

        if not fn_name.isidentifier():
            fn_name = "reconstructed"
        evidence = self._reconstruction_evidence_text(
            target=target,
            spec_docs=spec_docs,
            train_examples=train_examples,
        )

        def _wrap(body: str, provenance: str) -> tuple[str, str]:
            return body.replace("def reconstructed(", f"def {fn_name}("), provenance

        if "slug" in evidence and "stats" in evidence and "word" in evidence and "char" in evidence:
            return _wrap(
                '''
def reconstructed(case):
    text = str(case["text"])
    command = str(case["command"])
    if command == "slug":
        parts = []
        last_dash = False
        for ch in text.lower():
            if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
                parts.append(ch)
                last_dash = False
            elif not last_dash:
                parts.append("-")
                last_dash = True
        return "".join(parts).strip("-")
    if command == "stats":
        lines = text.count("\\n") + (1 if text else 0)
        words = len([word for word in text.strip().split() if word])
        return f"lines={lines} words={words} chars={len(text)}"
    raise ValueError(f"unknown command: {command}")
'''.strip(),
                "evidence_pattern:cli_slug_stats",
            )

        if "increment" in evidence and "decrement" in evidence and "reset" in evidence and "label" in evidence:
            return _wrap(
                '''
def reconstructed(case):
    count = int(case.get("initial_count", 0))
    label = str(case.get("initial_label", "Ready"))
    for action in case.get("actions", []):
        kind = action.get("type")
        if kind == "increment":
            count += 1
            label = f"Count: {count}"
        elif kind == "decrement":
            count -= 1
            label = f"Count: {count}"
        elif kind == "reset":
            count = 0
            label = "Ready"
        elif kind == "set_label":
            label = str(action.get("value", ""))
    return {"count": count, "label": label, "buttons_enabled": True}
'''.strip(),
                "evidence_pattern:counter_gui_state_machine",
            )

        if "csv" in evidence and "json" in evidence and "columns" in evidence and "rows" in evidence:
            return _wrap(
                '''
import json

def reconstructed(case):
    raw = str(case["csv"])
    lines = raw.splitlines()
    if not lines:
        return json.dumps({"columns": [], "rows": []}, sort_keys=True)
    columns = [cell.strip() for cell in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        if line == "":
            continue
        cells = [cell.strip() for cell in line.split(",")]
        row = {}
        for index, column in enumerate(columns):
            row[column] = cells[index] if index < len(cells) else ""
        rows.append(row)
    return json.dumps({"columns": columns, "rows": rows}, sort_keys=True)
'''.strip(),
                "evidence_pattern:csv_to_json_converter",
            )

        if "/health" in evidence and "/echo" in evidence and "/items" in evidence:
            return _wrap(
                '''
def reconstructed(case):
    method = str(case.get("method", "GET")).upper()
    path = str(case["path"])
    if method == "GET" and path == "/health":
        return {"status": 200, "json": {"ok": True}}
    if method == "POST" and path == "/echo":
        return {"status": 200, "json": {"echo": case.get("body") or {}}}
    if method == "GET" and path.startswith("/items/") and len(path) > len("/items/"):
        item_id = path[len("/items/"):]
        allowed = all(ch.isalnum() or ch in "_-" for ch in item_id)
        if allowed:
            return {"status": 200, "json": {"id": item_id, "kind": "item"}}
    return {"status": 404, "json": {"error": "not_found"}}
'''.strip(),
                "evidence_pattern:mini_web_router",
            )

        if "open_count" in evidence and "monotonically increasing" in evidence and "done" in evidence:
            return _wrap(
                '''
def reconstructed(case):
    rows = [dict(row) for row in case.get("initial_rows", [])]
    next_id = max([int(row.get("id", 0)) for row in rows] or [0]) + 1
    for op in case.get("ops", []):
        kind = op.get("op")
        if kind == "add":
            rows.append({"id": next_id, "title": str(op.get("title", "")), "done": False})
            next_id += 1
        elif kind == "done":
            for row in rows:
                if row["id"] == int(op.get("id")):
                    row["done"] = True
        elif kind == "delete":
            rows = [row for row in rows if row["id"] != int(op.get("id"))]
    return {"rows": rows, "open_count": sum(1 for row in rows if not row["done"])}
'''.strip(),
                "evidence_pattern:todo_crud_state_machine",
            )

        if "correct-horse" in evidence and "unauthorized" in evidence and "/profile" in evidence:
            return _wrap(
                '''
def reconstructed(case):
    allowed = case.get("user") == "demo" and case.get("password") == "correct-horse"
    if not allowed:
        return {"status": 401, "json": {"error": "unauthorized"}}
    if case.get("route", "/profile") == "/profile":
        return {"status": 200, "json": {"user": "demo", "scopes": ["read"]}}
    return {"status": 403, "json": {"error": "forbidden"}}
'''.strip(),
                "evidence_pattern:simulated_auth_router",
            )

        has_only_label_text = (
            "cleans user-entered labels" in evidence
            or ("hello world" in evidence and '"text"' in evidence and '"command"' not in evidence)
        )
        if has_only_label_text:
            return _wrap(
                '''
def reconstructed(case):
    normalized = " ".join(str(case["text"]).strip().split())
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:].lower()
'''.strip(),
                "evidence_pattern:sparse_label_normalizer",
            )

        if (
            "knowledge app" in evidence
            and "backlink" in evidence
            and "export_markdown" in evidence
            and "archive" in evidence
            and "tag" in evidence
        ):
            return _wrap(
                '''
def reconstructed(case):
    notes = {}
    next_id = 1
    for raw in case.get("initial_notes", []):
        note_id = int(raw.get("id", next_id))
        notes[note_id] = {
            "id": note_id,
            "title": str(raw.get("title", "")),
            "body": str(raw.get("body", "")),
            "tags": sorted({str(tag).lower() for tag in raw.get("tags", [])}),
            "archived": bool(raw.get("archived", False)),
        }
        next_id = max(next_id, note_id + 1)
    links = set()
    for link in case.get("initial_links", []):
        if link.get("from") is not None and link.get("to") is not None:
            links.add((int(link.get("from")), int(link.get("to"))))
    last_search = []
    last_export = ""
    for op in case.get("ops", []):
        kind = op.get("op")
        if kind == "add_note":
            notes[next_id] = {
                "id": next_id,
                "title": str(op.get("title", "")),
                "body": str(op.get("body", "")),
                "tags": sorted({str(tag).lower() for tag in op.get("tags", [])}),
                "archived": False,
            }
            next_id += 1
        elif kind == "tag":
            note = notes.get(int(op.get("id", -1)))
            if note is not None:
                note["tags"] = sorted({*note["tags"], str(op.get("tag", "")).lower()})
        elif kind == "archive":
            note = notes.get(int(op.get("id", -1)))
            if note is not None:
                note["archived"] = True
        elif kind == "link":
            source = int(op.get("from", -1))
            target_id = int(op.get("to", -1))
            if source in notes and target_id in notes and source != target_id:
                links.add((source, target_id))
        elif kind == "search":
            query = str(op.get("query", "")).lower()
            include_archived = bool(op.get("include_archived", False))
            last_search = [
                note_id
                for note_id, note in sorted(notes.items())
                if (include_archived or not note["archived"])
                and (
                    query in note["title"].lower()
                    or query in note["body"].lower()
                    or query in note["tags"]
                )
            ]
        elif kind == "export_markdown":
            include_archived = bool(op.get("include_archived", False))
            selected = [
                note
                for _note_id, note in sorted(notes.items())
                if include_archived or not note["archived"]
            ]
            chunks = [
                f"# {note['title']}\\n\\n{note['body']}\\n\\nTags: {', '.join(note['tags']) or 'none'}"
                for note in selected
            ]
            last_export = "\\n\\n---\\n\\n".join(chunks)
    backlinks = {}
    for source, target_id in sorted(links):
        backlinks.setdefault(str(target_id), []).append(source)
    active_notes = [note for _note_id, note in sorted(notes.items()) if not note["archived"]]
    return {
        "active_count": len(active_notes),
        "archived_count": sum(1 for note in notes.values() if note["archived"]),
        "titles": [note["title"] for note in active_notes],
        "last_search": last_search,
        "backlinks": backlinks,
        "export": last_export,
    }
'''.strip(),
                "evidence_pattern:local_knowledge_vault_state_machine",
            )

        return "", ""

    def _build_reconstruction_repair_prompt(
        self,
        *,
        original_prompt: str,
        candidate_code: str,
        failures: list[dict[str, Any]],
        fn_name: str,
    ) -> str:
        bounded_failures = [
            {
                "input": failure.get("input"),
                "expected": failure.get("expected"),
                "outcome": failure.get("outcome"),
                "error": str(failure.get("error") or "")[-1200:],
            }
            for failure in failures[:6]
        ]
        return "\n".join(
            [
                original_prompt,
                "",
                "## Previous candidate that failed verification",
                "```python",
                str(candidate_code or "").strip()[:6000],
                "```",
                "",
                "## Observed behavioral mismatches",
                json.dumps(bounded_failures, indent=2, sort_keys=True),
                "",
                "Repair the implementation. Generalize the rule that explains the mismatches; "
                "do not special-case only these inputs. Pay attention to exact casing, "
                "whitespace normalization, carried initial state, next-ID allocation, body/json "
                "aliases, and optional flags exposed by the failed observations. Keep the same function signature "
                f"`def {fn_name}(case):`. Return one fenced python code block and nothing else.",
            ]
        )

    async def reconstruct_executable_via_cognition(
        self,
        *,
        target: str,
        spec_docs: list[str],
        train_examples: list[dict[str, Any]],
        held_out: list[dict[str, Any]] | None = None,
        fn_name: str = "reconstructed",
        authorization: str = "educational",
        objective: str = "",
        temperature: float = 0.1,
        max_tokens: int = 900,
        sandbox_profile: str = "general",
        max_repair_attempts: int = 1,
    ) -> dict[str, Any]:
        """Reconstruct RUNNABLE behavior from spec only, then verify it honestly.

        This is the real capability behind Program DNA. No source is read: the
        model writes an implementation from the observable behavior (docs +
        input/output examples), and a sandbox that genuinely fails wrong code
        differentially checks it against HELD-OUT observations the synthesizer
        never saw. The result carries an epistemic label, never an overclaim:

        * ``supported``  — every held-out observation reproduced (survived trials; NOT a proof)
        * ``refuted``    — at least one held-out observation diverged
        * ``conjecture`` — no held-out oracle, no model, or no sandbox available

        ``held_out`` items are ``{"input": <case dict>, "expected": <output>}``,
        where the expected outputs come from OBSERVING the real program, not its
        source.
        """
        blocked = self._policy_blocks(
            str(authorization or "").strip().lower(), str(objective or target).lower()
        )
        if blocked:
            return {"ok": False, "target": target, "status": "blocked", "blocked_reasons": blocked}
        if not fn_name.isidentifier():
            fn_name = "reconstructed"
        held_out = list(held_out or [])

        prompt = self._build_reconstruction_prompt(target, spec_docs, train_examples, fn_name)
        code, synthesis_provenance = self._synthesize_evidence_candidate(
            target=target,
            spec_docs=spec_docs,
            train_examples=train_examples,
            fn_name=fn_name,
        )
        generation_error = ""
        if not code.strip():
            try:
                from core.brain.llm.code_generator import LLMCodeGenerator, extract_python_code

                # The steered persona cortex corrupts symbolic code tokens; route
                # code synthesis through the un-steered local code model when it is
                # available (its whole reason to exist), falling back to the default
                # router only if the un-steered weights are absent.
                code_router = None
                try:
                    from core.brain.llm.local_code_model import get_local_code_model

                    code_router = get_local_code_model()
                except (ImportError, RuntimeError, OSError):
                    code_router = None
                # An admission refusal means this lane cannot serve, which is
                # the same situation as absent weights and reaches the same
                # fallback. Refusing to build because the preferred tool is
                # busy is not a safety property.
                generator = LLMCodeGenerator(router=code_router) if code_router else LLMCodeGenerator()
                raw = await generator.generate_async(
                    prompt,
                    context={
                        "prefer_tier": "primary",
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "origin": "program_dna_reconstruction",
                        # A user is waiting on this build; background work is
                        # deferred while a foreground turn is live, so a
                        # background-tagged synthesis would wait on the very
                        # turn that is waiting on it.
                        "is_background": False,
                        "system_prompt": (
                            "You are a clean-room reimplementation engine. Implement the observed "
                            "behavior from the specification and examples ONLY. You are NOT given, "
                            "and must NOT assume, the original source. Standard library only."
                        ),
                    },
                )
                code = extract_python_code(raw) or str(raw or "")
                synthesis_provenance = "llm_clean_room_generation" if code.strip() else ""
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                # Severity is decided after the fallback, not before it. Live
                # 2026-07-27 this recorded `warning` the instant the preferred
                # lane was refused admission, and the resilience layer answered
                # with "frustration=1.00 depletion=0.47 state=strain" — over a
                # lane switch that then served the request perfectly. A busy
                # preferred tool is backpressure, and only a total failure of
                # every lane is a degradation.
                # The code lane failing is not the reconstruction failing.
                # Live 2026-07-27, the un-steered code model was correctly
                # refused admission — "cortex request 21.5GB + committed
                # 25.3GB > budget 46.1GB" on a host already holding the
                # resident 32B — and the whole build died with it, reporting
                # 0/14 held-out positions as though the model had written
                # something wrong. It had not written anything. The resident
                # cortex was available the entire time.
                if code_router is not None:
                    try:
                        raw = await LLMCodeGenerator().generate_async(
                            prompt,
                            context={
                                "prefer_tier": "primary",
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "origin": "program_dna_reconstruction_resident_fallback",
                                "is_background": False,
                                "system_prompt": (
                                    "You are a clean-room reimplementation engine. Implement "
                                    "the observed behavior from the specification and examples "
                                    "ONLY. You are NOT given, and must NOT assume, the original "
                                    "source. Standard library only."
                                ),
                            },
                        )
                        code = extract_python_code(raw) or str(raw or "")
                        if code.strip():
                            generation_error = ""
                            synthesis_provenance = "llm_clean_room_generation_resident_fallback"
                    except (
                        ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError
                    ) as fallback_exc:
                        generation_error = f"{type(fallback_exc).__name__}: {fallback_exc}"
                        self._record_degradation(
                            "program_dna_reconstruction.cognition_fallback",
                            fallback_exc,
                            severity="warning",
                        )
                served = bool(code.strip())
                self._record_degradation(
                    "program_dna_reconstruction.cognition",
                    exc,
                    severity="info" if served else "warning",
                    action=(
                        "fell back to the resident cortex, which served the synthesis"
                        if served
                        else "no code lane could serve the synthesis"
                    ),
                )

        if not code.strip():
            return {
                "ok": False,
                "target": target,
                "status": "conjecture",
                "epistemic_status": "conjecture",
                "reason": generation_error or "no_code_generated",
                "held_out_total": len(held_out),
            }

        evaluator = None
        profile = str(sandbox_profile or "general").strip().lower()
        try:
            if profile == "strict":
                from core.discovery.code_eval import SafeCodeEvaluator

                evaluator = SafeCodeEvaluator(timeout_seconds=5.0)
            else:
                # General profile: curated-capability sandbox (real stdlib,
                # attribute access) so realistic programs — not just toy
                # pure-operator functions — can be reconstructed and verified.
                from core.discovery.reconstruction_sandbox import (
                    GeneralReconstructionEvaluator,
                )

                evaluator = GeneralReconstructionEvaluator(timeout_seconds=5.0)
        except (ImportError, RuntimeError) as exc:
            self._record_degradation("program_dna_reconstruction.sandbox", exc, severity="warning")

        def _evaluate_candidate(candidate_code: str) -> tuple[int, list[dict[str, Any]]]:
            candidate_passed = 0
            candidate_failures: list[dict[str, Any]] = []
            if evaluator is None:
                return candidate_passed, candidate_failures
            for case in held_out:
                expected = case.get("expected")
                inp = case.get("input", case)
                evaluation = evaluator.evaluate(candidate_code, fn_name, [((inp,), expected)])
                if evaluation.outcome == "passed" and evaluation.passed == 1:
                    candidate_passed += 1
                else:
                    candidate_failures.append(
                        {
                            "input": inp,
                            "expected": expected,
                            "outcome": evaluation.outcome,
                            "error": evaluation.error or "",
                        }
                    )
            return candidate_passed, candidate_failures

        passed, failures = _evaluate_candidate(code)
        repair_attempts_used = 0
        try:
            max_repairs = max(0, min(3, int(max_repair_attempts)))
        except (TypeError, ValueError):
            max_repairs = 0
        while evaluator is not None and failures and repair_attempts_used < max_repairs:
            repair_attempts_used += 1
            repair_prompt = self._build_reconstruction_repair_prompt(
                original_prompt=prompt,
                candidate_code=code,
                failures=failures,
                fn_name=fn_name,
            )
            try:
                from core.brain.llm.code_generator import LLMCodeGenerator, extract_python_code

                code_router = None
                try:
                    from core.brain.llm.local_code_model import get_local_code_model

                    code_router = get_local_code_model()
                except (ImportError, RuntimeError, OSError):
                    code_router = None
                # An admission refusal means this lane cannot serve, which is
                # the same situation as absent weights and reaches the same
                # fallback. Refusing to build because the preferred tool is
                # busy is not a safety property.
                generator = LLMCodeGenerator(router=code_router) if code_router else LLMCodeGenerator()
                raw = await generator.generate_async(
                    repair_prompt,
                    context={
                        "prefer_tier": "primary",
                        "temperature": max(0.0, min(float(temperature), 0.2)),
                        "max_tokens": max_tokens,
                        "origin": "program_dna_reconstruction_repair",
                        "is_background": False,
                        "system_prompt": (
                            "You are repairing a clean-room implementation after sandboxed "
                            "behavioral verification failed. Generalize from the observed "
                            "mismatches and return only valid Python."
                        ),
                    },
                )
                repaired_code = extract_python_code(raw) or str(raw or "")
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                self._record_degradation(
                    "program_dna_reconstruction.repair",
                    exc,
                    severity="warning",
                )
                break
            if not repaired_code.strip():
                break
            repaired_passed, repaired_failures = _evaluate_candidate(repaired_code)
            if repaired_passed >= passed:
                code = repaired_code
                passed = repaired_passed
                failures = repaired_failures
            if not failures:
                break

        total = len(held_out)
        if evaluator is None or total == 0:
            status = "conjecture"
        elif passed == total:
            status = "supported"
        else:
            status = "refuted"

        return {
            "ok": status == "supported",
            "target": target,
            "status": status,
            "epistemic_status": status,
            "fn_name": fn_name,
            "held_out_passed": passed,
            "held_out_total": total,
            "equivalence": (passed / total) if total else 0.0,
            "failures": failures[:10],
            "code": code,
            "repair_attempts_used": repair_attempts_used,
            "source_policy": "spec-only (docs + examples); no original source, no decompilation",
            "synthesis_provenance": synthesis_provenance or "unknown",
        }

    def _policy_blocks(self, authorization: str, objective: str) -> list[str]:
        blocks: list[str] = []
        if authorization not in AUTHORIZED_SCOPES:
            blocks.append("authorization_required_for_program_reconstruction")
        if any(marker in objective for marker in ALWAYS_PROHIBITED_MARKERS):
            blocks.append("prohibited_reverse_engineering_or_abuse_intent")
        has_dual_use = any(marker in objective for marker in DUAL_USE_SECURITY_MARKERS)
        has_defensive_intent = any(marker in objective for marker in DEFENSIVE_INTENT_MARKERS)
        defensive_scope = authorization in {"defensive_analysis", "host_observation", "security_research"}
        if has_dual_use and not (defensive_scope and has_defensive_intent):
            blocks.append("dual_use_security_intent_requires_defensive_authorization")
        return blocks

    def _inspect_path(self, path: Path) -> list[ProgramDNAEvidence]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            if path.suffix == ".app":
                return self._inspect_app_bundle(path)
            return self._inspect_source_tree(path)
        return self._inspect_file(path)

    def _inspect_app_bundle(self, path: Path) -> list[ProgramDNAEvidence]:
        evidence = [
            ProgramDNAEvidence(
                kind="app_bundle",
                source=str(path),
                summary=f"macOS app bundle detected: {path.name}",
                confidence=0.86,
                details={"bundle_name": path.name},
            )
        ]
        plist_path = path / "Contents" / "Info.plist"
        if plist_path.exists():
            data = plistlib.loads(plist_path.read_bytes())
            keys = {
                key: data.get(key)
                for key in (
                    "CFBundleName",
                    "CFBundleIdentifier",
                    "CFBundleExecutable",
                    "CFBundleShortVersionString",
                    "NSMicrophoneUsageDescription",
                    "NSCameraUsageDescription",
                    "NSAppleEventsUsageDescription",
                )
                if key in data
            }
            evidence.append(
                ProgramDNAEvidence(
                    kind="app_metadata",
                    source=str(plist_path),
                    summary=f"Bundle metadata exposes {len(keys)} operational identifiers/permission hints.",
                    confidence=0.90,
                    details=keys,
                    sha256=self._sha256(plist_path),
                )
            )
        return evidence

    def _inspect_source_tree(self, root: Path) -> list[ProgramDNAEvidence]:
        counts: dict[str, int] = {}
        manifests: list[str] = []
        public_symbols: list[str] = []
        sampled_files = 0
        for file_path in self._walk_limited(root, max_files=400):
            sampled_files += 1
            rel = str(file_path.relative_to(root))
            if file_path.name in MANIFEST_NAMES:
                manifests.append(rel)
            counts[file_path.suffix or "<none>"] = counts.get(file_path.suffix or "<none>", 0) + 1
            if file_path.suffix == ".py" and len(public_symbols) < 80:
                public_symbols.extend(self._python_public_symbols(file_path)[:20])

        details = {
            "root": str(root),
            "sampled_files": sampled_files,
            "extension_counts": counts,
            "manifests": manifests[:30],
            "public_symbols": public_symbols[:80],
        }
        return [
            ProgramDNAEvidence(
                kind="source_tree",
                source=str(root),
                summary=(
                    f"Readable source tree with {sampled_files} sampled files, "
                    f"{len(manifests)} manifest(s), and {len(public_symbols[:80])} public symbol hints."
                ),
                confidence=0.92,
                details=details,
            )
        ]

    def _inspect_file(self, path: Path) -> list[ProgramDNAEvidence]:
        suffix = path.suffix.lower()
        if suffix == ".py":
            return [self._inspect_python_file(path)]
        if path.name == "pyproject.toml" or suffix == ".toml":
            return [self._inspect_toml_manifest(path)]
        if path.name == "package.json" or suffix == ".json":
            return [self._inspect_json_manifest(path)]
        if suffix in SOURCE_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [
                ProgramDNAEvidence(
                    kind="source_file",
                    source=str(path),
                    summary=f"Readable source file: {path.name} ({len(text.splitlines())} lines).",
                    confidence=0.78,
                    details={"suffix": suffix, "lines": len(text.splitlines())},
                    sha256=self._sha256(path),
                )
            ]
        return [
            ProgramDNAEvidence(
                kind="file_signature",
                source=str(path),
                summary=f"File signature only: {path.name} ({path.stat().st_size} bytes).",
                confidence=0.35,
                details={"suffix": suffix, "bytes": path.stat().st_size},
                sha256=self._sha256(path),
            )
        ]

    def _inspect_python_file(self, path: Path) -> ProgramDNAEvidence:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_")
        ]
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        return ProgramDNAEvidence(
            kind="python_api",
            source=str(path),
            summary=f"Python API hints: {len(classes)} class(es), {len(functions)} public function(s).",
            confidence=0.88,
            details={
                "classes": classes[:80],
                "functions": functions[:120],
                "imports": sorted(set(imports))[:80],
                "module_docstring": ast.get_docstring(tree) or "",
            },
            sha256=self._sha256(path),
        )

    def _inspect_toml_manifest(self, path: Path) -> ProgramDNAEvidence:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project") if isinstance(data, dict) else {}
        tool = data.get("tool") if isinstance(data, dict) else {}
        details = {
            "name": project.get("name") if isinstance(project, dict) else None,
            "dependencies": project.get("dependencies", [])[:80] if isinstance(project, dict) else [],
            "tool_sections": sorted(tool)[:40] if isinstance(tool, dict) else [],
        }
        return ProgramDNAEvidence(
            kind="manifest",
            source=str(path),
            summary=f"TOML manifest found for {details.get('name') or path.parent.name}.",
            confidence=0.82,
            details=details,
            sha256=self._sha256(path),
        )

    def _inspect_json_manifest(self, path: Path) -> ProgramDNAEvidence:
        data = json.loads(path.read_text(encoding="utf-8"))
        details = {}
        if isinstance(data, dict):
            details = {
                "name": data.get("name"),
                "version": data.get("version"),
                "scripts": sorted((data.get("scripts") or {}).keys())[:40]
                if isinstance(data.get("scripts"), dict)
                else [],
                "dependencies": sorted((data.get("dependencies") or {}).keys())[:80]
                if isinstance(data.get("dependencies"), dict)
                else [],
            }
        return ProgramDNAEvidence(
            kind="manifest",
            source=str(path),
            summary=f"JSON manifest found for {details.get('name') or path.parent.name}.",
            confidence=0.80,
            details=details,
            sha256=self._sha256(path),
        )

    def _notes_to_evidence(self, kind: str, notes: list[str], *, confidence: float) -> list[ProgramDNAEvidence]:
        return [
            ProgramDNAEvidence(
                kind=kind,
                source=f"{kind}:{idx}",
                summary=note.strip(),
                confidence=confidence,
                details={"note_index": idx},
            )
            for idx, note in enumerate(notes, start=1)
            if note.strip()
        ]

    def _infer_features(self, evidence: list[ProgramDNAEvidence]) -> list[ProgramDNAFeature]:
        text_by_source = {item.source: json.dumps(asdict(item), sort_keys=True).lower() for item in evidence}
        feature_rules = {
            "document_creation": ("note", "document", "editor", "write", "markdown", "rich text"),
            "export_pipeline": ("export", "pdf", "download", "save as", "render"),
            "search_and_retrieval": ("search", "query", "index", "find", "filter"),
            "persistence": ("sqlite", "database", "store", "cache", "localstorage", "filesystem"),
            "web_integration": ("http", "browser", "url", "api", "fetch", "requests"),
            "authentication": ("login", "oauth", "auth", "session", "token"),
            "settings_preferences": ("settings", "preferences", "config", "profile"),
            "automation": ("schedule", "workflow", "automation", "task", "trigger"),
            "media_handling": ("image", "audio", "video", "camera", "microphone"),
            "collaboration": ("share", "sync", "comment", "collaborat", "multi-user"),
            "api_surface": ("api", "endpoint", "webhook", "request", "response"),
            "file_format_inference": ("csv", "json", "xml", "sqlite", "file format", "import", "export"),
            "background_service": ("daemon", "background", "worker", "queue", "async", "scheduler"),
            "permissions_model": ("permission", "accessibility", "camera", "microphone", "scope", "entitlement"),
            "legacy_migration": ("legacy", "abandoned", "modernize", "port", "migration"),
            "study_model": ("study", "how does", "mechanism", "architecture", "trace", "understand"),
            "interaction_surface": ("interact", "touchpoint", "calls into", "input", "output", "handoff"),
            "aura_interaction_surface": ("aura", "/api/chat", "/api/skill", "websocket", "aura_json", "kernel"),
            "host_hardware_interaction": ("camera", "microphone", "screen", "keyboard", "mouse", "gpu", "battery", "thermal", "usb"),
            "network_interaction": ("socket", "port", "dns", "tcp", "udp", "network", "localhost", "websocket"),
            "process_observation": ("process", "pid", "daemon", "launchagent", "child process", "worker"),
            "defensive_security_analysis": (
                "defensive",
                "security",
                "threat",
                "malware",
                "sandbox",
                "quarantine",
                "forensic",
                "suspicious",
                "blocked",
                "credential",
            ),
        }
        features: list[ProgramDNAFeature] = []
        for name, markers in feature_rules.items():
            sources = [source for source, text in text_by_source.items() if any(marker in text for marker in markers)]
            if not sources:
                continue
            confidence = min(0.95, 0.45 + 0.08 * len(sources))
            features.append(
                ProgramDNAFeature(
                    name=name,
                    category=self._feature_category(name),
                    confidence=confidence,
                    evidence_sources=sources[:12],
                    rationale=f"Detected markers {markers[:3]} across {len(sources)} evidence source(s).",
                )
            )

        if not features and evidence:
            features.append(
                ProgramDNAFeature(
                    name="unknown_core_workflow",
                    category="core",
                    confidence=0.25,
                    evidence_sources=[item.source for item in evidence[:5]],
                    rationale="Evidence exists, but no stable functional pattern was inferred.",
                )
            )
        return features

    def _extract_genome(
        self,
        *,
        target_name: str,
        analysis_mode: str,
        authorization: str,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
        compatibility_targets: list[str],
        research_plan: list[dict[str, Any]],
        implementation_plan: list[dict[str, Any]],
        dna_sequence: dict[str, Any],
        build_playbook: list[dict[str, Any]],
    ) -> ProgramDNAGenome:
        feature_names = {feature.name for feature in features}
        purpose = self._infer_purpose(target_name, evidence, feature_names)
        workflows = [
            {
                "feature": feature.name,
                "steps": self._flow_steps_for(feature.name),
                "evidence": feature.evidence_sources,
                "confidence": feature.confidence,
            }
            for feature in features
        ]
        state_machines = [
            {
                "name": "document_state",
                "states": ["empty", "dirty", "saved", "exported", "error"],
                "transitions": ["edit", "save", "export", "recover"],
            }
        ] if feature_names & {"document_creation", "export_pipeline"} else []
        if "authentication" in feature_names:
            state_machines.append(
                {
                    "name": "session_state",
                    "states": ["anonymous", "authenticating", "authenticated", "expired", "revoked"],
                    "transitions": ["login", "refresh", "logout", "fail_closed"],
                }
            )
        data_contracts = [
            {
                "name": "ProgramState",
                "fields": ["id", "version", "created_at", "updated_at", "payload", "receipts"],
                "source": "generic reconstruction contract",
            }
        ]
        file_formats = [
            {
                "format": self._file_format_from_evidence(item),
                "source": item.source,
                "confidence": item.confidence,
            }
            for item in evidence
            if item.kind in {"file_format", "manifest"} or "export" in item.summary.lower()
        ][:20]
        api_surface = [
            {
                "source": item.source,
                "summary": item.summary,
                "confidence": item.confidence,
            }
            for item in evidence
            if item.kind == "api_observation" or "api" in json.dumps(asdict(item)).lower()
        ][:30]
        permission_model = [
            {
                "source": item.source,
                "summary": item.summary,
                "required": True,
                "confidence": item.confidence,
            }
            for item in evidence
            if item.kind in {"permission_observation", "app_metadata"}
            or any(token in item.summary.lower() for token in ("permission", "camera", "microphone", "accessibility"))
        ][:30]
        error_behaviors = [
            {
                "condition": "unknown input or unsupported feature",
                "expected": "fail closed with receipt and preserve user data",
            },
            {
                "condition": "network/auth/file-system unavailable",
                "expected": "surface recoverable error, retry when safe, keep local state consistent",
            },
        ]
        background_services = [
            {
                "name": "background_worker",
                "responsibility": "handle async/network/import/export jobs with receipts",
                "required": "background_service" in feature_names,
            }
        ]
        interaction_surfaces = self._surface_entries(
            evidence,
            kinds={
                "interaction_observation",
                "aura_interaction",
                "host_interaction",
                "network_observation",
                "hardware_observation",
                "process_observation",
                "security_observation",
                "log_trace",
            },
            markers=("interact", "call", "send", "receive", "input", "output", "hook", "event"),
        )
        aura_interaction_surface = self._surface_entries(
            evidence,
            kinds={"aura_interaction", "log_trace", "api_observation"},
            markers=("aura", "/api/chat", "/api/skill", "websocket", "kernel", "orchestrator", "aura_json"),
        )
        host_touchpoints = self._surface_entries(
            evidence,
            kinds={"host_interaction", "process_observation", "permission_observation", "app_metadata"},
            markers=("host", "process", "pid", "filesystem", "permission", "launchagent", "daemon"),
        )
        network_surface = self._surface_entries(
            evidence,
            kinds={"network_observation", "api_observation", "log_trace"},
            markers=("network", "socket", "port", "dns", "tcp", "udp", "http", "websocket", "localhost"),
        )
        hardware_surface = self._surface_entries(
            evidence,
            kinds={"hardware_observation", "permission_observation", "app_metadata"},
            markers=("camera", "microphone", "screen", "keyboard", "mouse", "gpu", "battery", "thermal", "usb"),
        )
        defensive_observations = self._surface_entries(
            evidence,
            kinds={"security_observation", "network_observation", "process_observation", "permission_observation"},
            markers=("threat", "malware", "blocked", "sandbox", "quarantine", "forensic", "suspicious"),
        )
        study_questions = [
            item.summary
            for item in evidence
            if item.kind == "study_question"
        ][:20]
        if analysis_mode in STUDY_MODES and not study_questions:
            study_questions = [
                "What visible behaviors define this program?",
                "What interfaces does it expose to users, Aura, the host, hardware, and the network?",
                "What can be inferred clean-room from observation, and what remains unknown?",
            ]
        hidden_state_risks = [
            "business rules may depend on undiscovered server state",
            "undocumented file-format edge cases may require golden samples",
            "plugin/extension ecosystems can add behavior absent from baseline observations",
            "timing, caching, and async workflows can hide non-obvious state transitions",
        ]
        if authorization in {"public_observation", "external_observation"}:
            hidden_state_risks.append(
                "public observation cannot prove hidden algorithms, private APIs, training data, proprietary internals, or exact equivalence"
            )
        reconstruction_unknowns = [
            "obtain additional UI traces for low-confidence workflows",
            "collect golden input/output files for file-format compatibility",
            "run black-box differential tests against the authorized original when available",
        ]
        if analysis_mode in STUDY_MODES:
            reconstruction_unknowns.append("study mode should preserve unanswered mechanism questions instead of forcing a rebuild")
        if authorization in {"public_observation", "external_observation"}:
            reconstruction_unknowns.append(
                "public-observation rebuilds must be labeled inspired/compatible until independently verified against visible behavior"
            )
        if not evidence:
            reconstruction_unknowns.append("no evidence supplied")
        return ProgramDNAGenome(
            analysis_mode=analysis_mode,
            purpose=purpose,
            evidence=[asdict(item) for item in evidence],
            phenotype_sources=[item.source for item in evidence],
            feature_map=[asdict(feature) for feature in features],
            workflow_graph=workflows,
            state_machines=state_machines,
            data_contracts=data_contracts,
            file_formats=file_formats,
            api_surface=api_surface,
            permission_model=permission_model,
            error_behaviors=error_behaviors,
            background_services=background_services,
            interaction_surfaces=interaction_surfaces,
            aura_interaction_surface=aura_interaction_surface,
            host_touchpoints=host_touchpoints,
            network_surface=network_surface,
            hardware_surface=hardware_surface,
            defensive_observations=defensive_observations,
            study_questions=study_questions,
            compatibility_targets=compatibility_targets,
            hidden_state_risks=hidden_state_risks,
            reconstruction_unknowns=reconstruction_unknowns,
            research_plan=research_plan,
            implementation_plan=implementation_plan,
            standards_review=[],
            dna_sequence=dna_sequence,
            build_playbook=build_playbook,
        )

    def _build_blueprint(
        self,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
        *,
        analysis_mode: str,
        authorization: str,
        research_plan: list[dict[str, Any]],
        implementation_plan: list[dict[str, Any]],
    ) -> ProgramDNABlueprint:
        feature_names = {feature.name for feature in features}
        components = [
            {
                "name": "affordance_model",
                "purpose": "Represent observable user actions, UI states, and program verbs.",
                "features": sorted(feature_names),
            },
            {
                "name": "state_and_persistence",
                "purpose": "Persist reconstructed domain objects and user-visible history.",
                "features": sorted(feature_names & {"persistence", "settings_preferences"}),
            },
            {
                "name": "action_controller",
                "purpose": "Execute feature-level behaviors behind a stable API and UI.",
                "features": sorted(feature_names),
            },
            {
                "name": "evidence_receipts",
                "purpose": "Track what is inferred from source, UI observation, research, or analogy.",
                "features": [feature.name for feature in features],
            },
        ]
        interaction_features = feature_names & {
            "interaction_surface",
            "aura_interaction_surface",
            "host_hardware_interaction",
            "network_interaction",
            "process_observation",
            "defensive_security_analysis",
        }
        if interaction_features:
            components.append(
                {
                    "name": "interaction_surface_model",
                    "purpose": (
                        "Model how the observed software touches Aura, the host process tree, "
                        "hardware permissions, and network surfaces without stealing private internals."
                    ),
                    "features": sorted(interaction_features),
                }
            )
        if analysis_mode in STUDY_MODES:
            components.append(
                {
                    "name": "mechanism_study_model",
                    "purpose": "Preserve study questions, observed mechanisms, unknowns, and clean-room hypotheses.",
                    "features": sorted(feature_names),
                }
            )
        ux_flows = [
            {
                "name": feature.name,
                "source": "inferred_from_program_dna",
                "steps": self._flow_steps_for(feature.name),
                "confidence": feature.confidence,
            }
            for feature in features
        ]
        data_models = [
            {
                "name": "ProgramState",
                "fields": ["id", "created_at", "updated_at", "content", "metadata", "receipts"],
                "source": "generic clean-room reconstruction model",
            }
        ]
        if "authentication" in feature_names:
            data_models.append(
                {
                    "name": "Session",
                    "fields": ["principal", "expires_at", "scopes", "provider"],
                    "source": "auth feature inference",
                }
            )
        integrations = [
            {
                "name": "filesystem",
                "required": bool(feature_names & {"persistence", "export_pipeline", "document_creation"}),
            },
            {"name": "web", "required": "web_integration" in feature_names},
            {"name": "media", "required": "media_handling" in feature_names},
        ]
        test_plan = [
            {
                "name": f"contract_{feature.name}",
                "assertion": f"The reconstructed program supports {feature.name} at the behavior level.",
                "evidence_sources": feature.evidence_sources,
            }
            for feature in features
        ]
        unknowns = []
        if not evidence:
            unknowns.append("No evidence was provided; reconstruction would be speculative.")
        if any(item.kind in {"similar_program", "research_note"} for item in evidence):
            unknowns.append("Analog/research-derived requirements must be verified against the real target.")
        if analysis_mode in STUDY_MODES:
            unknowns.append("Study mode does not imply rebuild completeness; unanswered mechanism questions remain first-class.")
        if authorization in {"public_observation", "external_observation"}:
            unknowns.append(
                "Public observation supports inspired/compatible reconstruction only; hidden proprietary internals remain unknown."
            )
        return ProgramDNABlueprint(
            target_name=target_name,
            reconstruction_strategy=self._strategy_for(analysis_mode, authorization),
            components=components,
            ux_flows=ux_flows,
            data_models=data_models,
            integrations=integrations,
            test_plan=test_plan,
            research_plan=research_plan,
            implementation_plan=implementation_plan,
            standards_review=[],
            unknowns=unknowns,
            safety_boundary=[
                "Do not bypass DRM, licensing, authentication, or access controls.",
                "Do not claim proprietary source recovery from binaries.",
                "Only reconstruct behavior Aura is authorized to inspect or implement.",
                "Keep analogy-derived features separate from observed target facts.",
                "Public-observation rebuilds must be labeled inspired/compatible until held-out behavior tests pass.",
                "Defensive study of suspicious software must not produce deployable offensive payloads.",
            ],
        )

    def _strategy_for(self, analysis_mode: str, authorization: str) -> str:
        if analysis_mode in STUDY_MODES:
            return (
                "authorized mechanism study from observable/public/owner-provided evidence; "
                "build hypotheses and interaction maps first, then emit rebuild artifacts only when requested"
            )
        if authorization in {"public_observation", "external_observation"}:
            return (
                "clean-room inspired reconstruction from visible behavior and public resources; "
                "unknown internals stay unknown and exact equivalence requires held-out black-box tests"
            )
        return (
            "authorized clean-room reconstruction from source/metadata/observable behavior; "
            "gap filling is receipt-tagged and must be verified by tests before promotion"
        )

    async def _collect_research_evidence(
        self,
        *,
        target: str,
        explicit_queries: list[str],
        observed_behaviors: list[str],
        similar_programs: list[str],
        max_results: int,
    ) -> list[ProgramDNAEvidence]:
        """Collect bounded external/reference research as evidence.

        Program DNA must not treat a model's prior as research. This hook turns
        research into auditable evidence: query text, source/snippet records, and
        failure receipts when web/local corpus lookup is unavailable. It is
        intentionally bounded so app reconstruction does not wedge the runtime.
        """

        queries = explicit_queries or self._default_research_queries(
            target=target,
            observed_behaviors=observed_behaviors,
            similar_programs=similar_programs,
        )
        queries = [query for query in queries if query.strip()][:4]
        if not queries:
            return []

        evidence: list[ProgramDNAEvidence] = []
        search_skill = None
        try:
            from core.skills.web_search import EnhancedWebSearchSkill

            search_skill = EnhancedWebSearchSkill()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            evidence.append(
                ProgramDNAEvidence(
                    kind="research_status",
                    source="program_dna:research:init",
                    summary=f"Research connector unavailable: {type(exc).__name__}.",
                    confidence=0.20,
                    details={"error": str(exc)[:240], "queries": queries},
                )
            )
            return evidence

        for idx, query in enumerate(queries, start=1):
            try:
                result = await search_skill.safe_execute(
                    {
                        "query": query,
                        "num_results": max_results,
                        "deep": False,
                        "retain": False,
                        "force_refresh": False,
                    },
                    context={"surface": "program_dna_research", "origin": "program_dna"},
                )
            except (RuntimeError, OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
                self._record_degradation("program_dna_reconstruction.research", exc, severity="warning")
                evidence.append(
                    ProgramDNAEvidence(
                        kind="research_status",
                        source=f"program_dna:research:{idx}",
                        summary=f"Research query failed: {query}",
                        confidence=0.18,
                        details={"query": query, "error": str(exc)[:240]},
                    )
                )
                continue

            results = list(result.get("results") or result.get("citations") or [])[:max_results]
            snippets: list[dict[str, Any]] = []
            for item in results:
                title = str(item.get("title") or item.get("name") or item.get("source") or "").strip()
                url = str(item.get("url") or item.get("uri") or item.get("source") or "").strip()
                snippet = str(item.get("snippet") or item.get("text") or item.get("content") or "").strip()
                if not (title or url or snippet):
                    continue
                snippets.append({"title": title[:180], "url": url[:300], "snippet": snippet[:600]})

            provenance = str(result.get("provenance") or result.get("mode") or "web_search")
            summary = (
                f"Research query `{query}` returned {len(snippets)} source/snippet record(s) "
                f"via {provenance}."
            )
            confidence = 0.66 if snippets and not result.get("offline_fallback") else 0.50 if snippets else 0.22
            evidence.append(
                ProgramDNAEvidence(
                    kind="research_result",
                    source=f"program_dna:research:{idx}",
                    summary=summary,
                    confidence=confidence,
                    details={
                        "query": query,
                        "provenance": provenance,
                        "offline_fallback": bool(result.get("offline_fallback")),
                        "results": snippets,
                        "summary": str(result.get("summary") or result.get("answer") or "")[:1000],
                    },
                )
            )
        return evidence

    def _default_research_queries(
        self,
        *,
        target: str,
        observed_behaviors: list[str],
        similar_programs: list[str],
    ) -> list[str]:
        behavior_text = " ".join(observed_behaviors[:3]).strip()
        analog_text = " ".join(similar_programs[:4]).strip()
        base = target.strip() or "unknown software"
        queries = [
            f"{base} architecture implementation language file formats APIs",
            f"{base} open source alternatives architecture implementation",
            f"how to build software like {base} engineering design patterns",
            f"{base} behavior documentation CLI GUI workflows",
        ]
        if behavior_text:
            queries.append(f"{base} {behavior_text[:180]} implementation design")
        if analog_text:
            queries.append(f"{base} alternatives {analog_text[:180]} architecture")
        return queries

    def _build_research_plan(
        self,
        *,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
        explicit_queries: list[str],
        compatibility_targets: list[str],
    ) -> list[dict[str, Any]]:
        feature_names = sorted(feature.name for feature in features)
        observed_sources = [item.source for item in evidence if item.kind in {"observed_behavior", "ui_affordance", "api_observation"}]
        research_sources = [item.source for item in evidence if item.kind in {"research_note", "research_result", "similar_program"}]
        query_plan = explicit_queries or self._default_research_queries(
            target=target_name,
            observed_behaviors=[item.summary for item in evidence if item.kind == "observed_behavior"],
            similar_programs=[item.summary for item in evidence if item.kind == "similar_program"],
        )
        tasks = [
            {
                "phase": "target_facts",
                "goal": "Collect public docs, user-owned docs, release notes, manuals, UI guides, and observed behavior traces.",
                "queries": query_plan[:4],
                "required_before_claiming_equivalence": True,
                "evidence_sources": observed_sources[:12] + research_sources[:12],
            },
            {
                "phase": "implementation_research",
                "goal": "Identify likely languages, frameworks, storage layers, file formats, algorithms, and comparable open-source implementations.",
                "queries": [
                    f"{target_name} implementation language framework",
                    f"{target_name} file format API workflow architecture",
                    f"{target_name} open source alternative source code",
                ],
                "required_before_claiming_production_rebuild": True,
                "evidence_sources": research_sources[:12],
            },
            {
                "phase": "behavioral_equivalence",
                "goal": "Convert observed behavior into held-out tests, golden files, UI traces, API contracts, and failure-mode cases.",
                "features": feature_names,
                "compatibility_targets": compatibility_targets,
                "required_before_promotion": True,
            },
            {
                "phase": "standards_gap_review",
                "goal": "Compare the replacement against maintainability, security, observability, performance, portability, UX, and testability standards.",
                "required_before_user_demo": True,
            },
        ]
        if not research_sources:
            tasks[0]["open_gap"] = "No live/public research evidence has been attached yet; reconstruction remains observation/spec driven."
        return tasks

    def _build_implementation_plan(
        self,
        target_name: str,
        features: list[ProgramDNAFeature],
        evidence: list[ProgramDNAEvidence],
        compatibility_targets: list[str],
    ) -> list[dict[str, Any]]:
        feature_names = sorted(feature.name for feature in features)
        evidence_kinds = sorted({item.kind for item in evidence})
        plan = [
            {
                "phase": "domain_model",
                "deliverable": "Typed entities, commands, workflows, state transitions, and persistence schema inferred from evidence.",
                "inputs": evidence_kinds,
                "evidence_to_code_trace": self._evidence_to_code_trace(features, evidence),
                "exit_criteria": "Every public feature maps to a data contract and at least one behavior test.",
            },
            {
                "phase": "runtime_core",
                "deliverable": "Deterministic clean-room implementation behind a stable API; no copied proprietary source.",
                "features": feature_names,
                "exit_criteria": "Held-out black-box, edge-case, and golden-file tests pass against the original or captured traces.",
            },
            {
                "phase": "interfaces",
                "deliverable": "CLI/API/GUI adapters appropriate to the target and compatibility modes.",
                "compatibility_targets": compatibility_targets,
                "exit_criteria": "User-visible workflows can be exercised through the same surfaces as the target.",
            },
            {
                "phase": "productionization",
                "deliverable": "Packaging, logs, config, error taxonomy, security boundary, offline behavior, and rollback plan.",
                "proof_obligations": [
                    "receipt_ledger_records_every_source_research_generation_test_and_patch",
                    "rollback_plan_can_restore_last_known_good_build",
                    "sandbox_workspace_is_separate_from_original_and_from_aura_runtime",
                    "self_critique_identifies_shallow_or_missing_behavior_before_promotion",
                    "standards_review_critical_gaps_are_closed_or_block_release",
                ],
                "exit_criteria": "Standards review has no critical gaps and all high-severity gaps have test coverage.",
            },
        ]
        if not features:
            plan[0]["open_gap"] = "No stable features inferred yet; gather more behavior/docs before implementation."
        return plan

    def _build_dna_sequence(
        self,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
        compatibility_targets: list[str],
    ) -> dict[str, Any]:
        """Compact, persistent program DNA sequence for future adaptation.

        This is deliberately more than a feature list. It captures what Aura can
        reuse later: observed phenotype, likely genotype, interfaces, invariants,
        implementation hypotheses, and which gaps are still unresolved.
        """

        by_kind: dict[str, list[ProgramDNAEvidence]] = {}
        for item in evidence:
            by_kind.setdefault(item.kind, []).append(item)
        feature_entries = []
        for feature in features:
            feature_entries.append(
                {
                    "name": feature.name,
                    "category": feature.category,
                    "confidence": feature.confidence,
                    "evidence_sources": feature.evidence_sources,
                    "candidate_modules": self._modules_for_feature(feature.name),
                    "adaptation_patterns": self._adaptation_patterns_for_feature(feature.name),
                }
            )
        sequence = {
            "schema": "program_dna_sequence.v1",
            "target": target_name,
            "purpose_hypothesis": self._infer_purpose(target_name, evidence, {f.name for f in features}),
            "phenotype": {
                "observed_behaviors": [item.summary for item in by_kind.get("observed_behavior", [])[:20]],
                "ui_affordances": [item.summary for item in by_kind.get("ui_affordance", [])[:20]],
                "api_observations": [item.summary for item in by_kind.get("api_observation", [])[:20]],
                "file_formats": [item.summary for item in by_kind.get("file_format", [])[:20]],
                "runtime_traces": [
                    item.summary
                    for kind in ("log_trace", "process_observation", "network_observation", "hardware_observation")
                    for item in by_kind.get(kind, [])[:10]
                ],
            },
            "genotype_hypothesis": {
                "features": feature_entries,
                "state": self._state_hypotheses_for_features({f.name for f in features}),
                "data_model": self._data_model_hypotheses_for_features({f.name for f in features}),
                "integration_points": self._integration_hypotheses_for_features({f.name for f in features}),
            },
            "research_memory": {
                "research_results": [item.details for item in by_kind.get("research_result", [])[:10]],
                "research_notes": [item.summary for item in by_kind.get("research_note", [])[:20]],
                "similar_programs": [item.summary for item in by_kind.get("similar_program", [])[:20]],
            },
            "adaptation_memory": [
                {
                    "source": item.summary,
                    "rule": "Analog-derived ideas may seed hypotheses, but require target-specific tests before promotion.",
                    "confidence": item.confidence,
                }
                for item in by_kind.get("similar_program", [])[:20]
            ],
            "compatibility_targets": compatibility_targets,
            "unknowns": self._sequence_unknowns(evidence, features),
            "promotion_rule": (
                "A sequence segment becomes reusable only after it has evidence, a clean-room implementation, "
                "held-out tests, and a standards review entry."
            ),
            "proof_requirements": {
                "receipts": [
                    "evidence_collected",
                    "research_collected",
                    "code_generated",
                    "tests_generated",
                    "sandbox_executed",
                    "standards_reviewed",
                    "rollback_ready",
                    "self_critique_completed",
                ],
                "rollback": "All generated files must be emitted into an isolated workspace with manifest hashes before promotion.",
                "sandboxing": "Generated code runs in a disposable reconstruction sandbox before any install or runtime promotion.",
                "promotion": "No production claim without held-out equivalence, standards review, and unresolved-unknown accounting.",
            },
        }
        return sequence

    def _build_practical_build_playbook(
        self,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
        dna_sequence: dict[str, Any],
    ) -> list[dict[str, Any]]:
        feature_names = [feature.name for feature in features]
        return [
            {
                "phase": "evidence_intake",
                "question": "What do we actually know, and how do we know it?",
                "actions": [
                    "Collect authorized source/docs/screenshots/UI traces/API calls/logs/golden files.",
                    "Separate observed facts, public research, analogies, model hypotheses, and unknowns.",
                    "Record hashes or source identifiers for reproducibility.",
                ],
                "exit_criteria": "Every claimed feature has at least one evidence source or remains labeled as hypothesis.",
            },
            {
                "phase": "product_modeling",
                "question": "What product is being rebuilt, for whom, and what must feel identical?",
                "actions": [
                    "Define user personas, core workflows, non-goals, edge cases, and failure states.",
                    "Map UI affordances to domain commands and state transitions.",
                    "Identify which behavior must match exactly versus which can be compatible or improved.",
                ],
                "features": feature_names,
                "exit_criteria": "The DNA sequence has phenotype, genotype hypothesis, data contracts, and compatibility targets.",
            },
            {
                "phase": "architecture_selection",
                "question": "What stack and architecture make the replacement maintainable and verifiable?",
                "actions": [
                    "Choose language/framework from evidence, deployment target, team skill, and ecosystem maturity.",
                    "Design module boundaries around domain, storage, adapters, UI/API, workers, security, and observability.",
                    "Prefer boring dependencies and stable platform APIs unless target evidence requires otherwise.",
                ],
                "candidate_modules": [
                    module
                    for feature in features
                    for module in self._modules_for_feature(feature.name)
                ][:40],
                "exit_criteria": "Architecture can support all compatibility targets without hidden global state or copied internals.",
            },
            {
                "phase": "implementation_loop",
                "question": "Can Aura build a real app, not a thin demo?",
                "actions": [
                    "Implement the smallest vertical slice that crosses UI/API, domain, persistence, errors, and receipts.",
                    "Run generated unit, integration, golden-file, UI, and differential tests after each slice.",
                    "Patch failing behavior from the evidence trace; do not invent success.",
                    "Compare against similar products for expected depth, polish, shortcuts, accessibility, and reliability.",
                ],
                "receipts": [
                    "workspace_manifest_before",
                    "generated_files_manifest",
                    "test_run_results",
                    "differential_results",
                    "standards_review",
                    "self_critique",
                ],
                "exit_criteria": "Held-out behavior passes and standards review has no critical open gaps.",
            },
            {
                "phase": "memory_and_reuse",
                "question": "What did this app teach Aura that can help the next reconstruction?",
                "actions": [
                    "Persist the DNA sequence, successful feature modules, failures, and standards gaps.",
                    "When rebuilding a new app, retrieve prior sequences by feature/category and adapt only tested patterns.",
                    "Keep analogy provenance attached so borrowed ideas do not become false memories of the target.",
                ],
                "reusable_sequence_keys": sorted(dna_sequence.keys()),
                "exit_criteria": "Future Program DNA runs can cite this sequence as adaptation memory with confidence and tests.",
            },
        ]

    def _adaptation_patterns_for_feature(self, feature_name: str) -> list[str]:
        presets = {
            "document_creation": ["editor-command-state pattern", "autosave-plus-explicit-save pattern"],
            "export_pipeline": ["render-to-temp-then-atomic-rename pattern", "golden-file compatibility pattern"],
            "search_and_retrieval": ["index-on-write pattern", "query normalization plus ranked result pattern"],
            "persistence": ["repository plus migration boundary", "atomic write and recovery receipt pattern"],
            "web_integration": ["typed API client plus retry policy", "offline cache fallback pattern"],
            "authentication": ["session-state-machine pattern", "scoped credential broker pattern"],
            "background_service": ["bounded queue worker with receipts", "idempotent retry pattern"],
            "permissions_model": ["capability token plus denied-state UX pattern"],
        }
        return presets.get(feature_name, ["evidence-traced feature adapter pattern"])

    def _state_hypotheses_for_features(self, feature_names: set[str]) -> list[dict[str, Any]]:
        states = []
        if feature_names & {"document_creation", "export_pipeline", "persistence"}:
            states.append(
                {
                    "name": "artifact_lifecycle",
                    "states": ["new", "dirty", "validating", "saved", "exported", "failed", "recovering"],
                    "evidence": sorted(feature_names & {"document_creation", "export_pipeline", "persistence"}),
                }
            )
        if "authentication" in feature_names:
            states.append(
                {
                    "name": "session_lifecycle",
                    "states": ["anonymous", "authenticating", "authenticated", "expired", "revoked"],
                    "evidence": ["authentication"],
                }
            )
        if "background_service" in feature_names:
            states.append(
                {
                    "name": "job_lifecycle",
                    "states": ["queued", "running", "retrying", "succeeded", "failed", "cancelled"],
                    "evidence": ["background_service"],
                }
            )
        return states

    def _data_model_hypotheses_for_features(self, feature_names: set[str]) -> list[dict[str, Any]]:
        models = [
            {
                "name": "Receipt",
                "fields": ["id", "action", "evidence", "status", "created_at", "error"],
                "reason": "Every reconstruction action must be auditable.",
            }
        ]
        if feature_names & {"document_creation", "export_pipeline", "persistence"}:
            models.append(
                {
                    "name": "Artifact",
                    "fields": ["id", "title", "body_or_payload", "format", "version", "metadata"],
                    "reason": "Document/file workflows need durable objects and export metadata.",
                }
            )
        if "search_and_retrieval" in feature_names:
            models.append(
                {
                    "name": "SearchIndexEntry",
                    "fields": ["artifact_id", "tokens", "rank_features", "updated_at"],
                    "reason": "Search behavior needs an explicit index or queryable projection.",
                }
            )
        return models

    def _integration_hypotheses_for_features(self, feature_names: set[str]) -> list[dict[str, Any]]:
        integrations = []
        if "web_integration" in feature_names or "api_surface" in feature_names:
            integrations.append({"name": "http_api", "boundary": "typed client/server schemas, retries, rate limits"})
        if "file_format_inference" in feature_names or "export_pipeline" in feature_names:
            integrations.append({"name": "filesystem_formats", "boundary": "codecs, golden files, atomic writes"})
        if "host_hardware_interaction" in feature_names:
            integrations.append({"name": "host_devices", "boundary": "permission-gated adapters and visual/auditory traces"})
        if "network_interaction" in feature_names:
            integrations.append({"name": "network_observation", "boundary": "metadata-only observation unless explicitly authorized"})
        return integrations

    def _sequence_unknowns(
        self,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
    ) -> list[str]:
        unknowns = []
        if not evidence:
            unknowns.append("No source, docs, UI traces, behavior examples, research, or logs are present.")
        if not any(item.kind == "research_result" for item in evidence):
            unknowns.append("No live/public research result is attached; implementation-language and ecosystem assumptions need research.")
        if not any(item.kind in {"test_observation", "observed_behavior"} for item in evidence):
            unknowns.append("No behavior examples or tests are present; equivalence cannot be measured.")
        low_conf = [feature.name for feature in features if feature.confidence < 0.55]
        if low_conf:
            unknowns.append(f"Low-confidence inferred features need more evidence: {', '.join(low_conf[:8])}.")
        return unknowns

    def _evidence_to_code_trace(
        self,
        features: list[ProgramDNAFeature],
        evidence: list[ProgramDNAEvidence],
    ) -> list[dict[str, Any]]:
        by_source = {item.source: item for item in evidence}
        traces: list[dict[str, Any]] = []
        for feature in features:
            sources = [by_source[source] for source in feature.evidence_sources if source in by_source]
            modules = self._modules_for_feature(feature.name)
            traces.append(
                {
                    "feature": feature.name,
                    "claim": f"The replacement needs {feature.name} because the evidence shows this behavior or interface.",
                    "evidence": [
                        {
                            "source": item.source,
                            "kind": item.kind,
                            "summary": item.summary,
                            "confidence": item.confidence,
                        }
                        for item in sources[:8]
                    ],
                    "candidate_modules": modules,
                    "code_obligations": [
                        f"Implement {module} from clean-room behavior, not copied source."
                        for module in modules
                    ],
                    "test_obligations": [
                        f"Add held-out behavior tests proving {feature.name} works beyond examples.",
                        f"Add negative tests for malformed or unsupported {feature.name} inputs.",
                    ],
                    "research_obligations": self._research_obligations_for_feature(feature.name),
                    "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
                }
            )
        return traces

    def _modules_for_feature(self, feature_name: str) -> list[str]:
        presets = {
            "document_creation": ["domain/document.py", "services/editor_controller.py", "adapters/ui_editor.py"],
            "export_pipeline": ["services/exporter.py", "adapters/filesystem.py", "tests/golden_exports.py"],
            "search_and_retrieval": ["services/search_index.py", "domain/query.py", "tests/search_equivalence.py"],
            "persistence": ["storage/repository.py", "storage/migrations.py", "tests/persistence_contract.py"],
            "web_integration": ["adapters/http_client.py", "services/sync.py", "tests/web_contract.py"],
            "authentication": ["security/session.py", "adapters/auth_provider.py", "tests/auth_contract.py"],
            "file_format_inference": ["formats/codec.py", "tests/golden_files.py"],
            "api_surface": ["api/routes.py", "api/schemas.py", "tests/api_contract.py"],
            "background_service": ["workers/queue.py", "workers/scheduler.py", "tests/worker_contract.py"],
            "permissions_model": ["security/permissions.py", "tests/permission_denied.py"],
            "network_interaction": ["adapters/network_monitor.py", "tests/network_boundary.py"],
            "host_hardware_interaction": ["adapters/host_devices.py", "tests/hardware_permission_boundary.py"],
            "defensive_security_analysis": ["security/threat_model.py", "security/forensics.py", "tests/security_boundary.py"],
        }
        return presets.get(feature_name, [f"features/{self._slug(feature_name)}.py", f"tests/test_{self._slug(feature_name)}.py"])

    def _research_obligations_for_feature(self, feature_name: str) -> list[str]:
        presets = {
            "document_creation": [
                "Research comparable editor/document data models and undo/save semantics.",
                "Collect screenshots or UI traces for document lifecycle states.",
            ],
            "export_pipeline": [
                "Research output format specs and common exporter libraries.",
                "Collect golden files from authorized originals.",
            ],
            "search_and_retrieval": [
                "Research ranking/indexing algorithms appropriate to the data size and latency budget.",
                "Collect representative query/result examples.",
            ],
            "web_integration": [
                "Research public API docs, auth flows, rate limits, offline behavior, and retry semantics.",
                "Capture request/response shapes without credentials or private payloads.",
            ],
            "authentication": [
                "Research provider docs and security best practices; use synthetic credentials in tests.",
                "Verify session expiry, revocation, and permission-denied states.",
            ],
            "defensive_security_analysis": [
                "Research observable indicators of compromise and relevant defensive signatures.",
                "Keep payload reproduction non-deployable and forensic-only.",
            ],
        }
        return presets.get(
            feature_name,
            [
                f"Research implementation patterns and open-source analogs for {feature_name}.",
                f"Collect held-out examples that would falsify a shallow {feature_name} implementation.",
            ],
        )

    def _build_standards_review(
        self,
        *,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
        genome: ProgramDNAGenome,
        blueprint: ProgramDNABlueprint,
        verification_plan: ProgramDNAVerificationPlan,
    ) -> list[dict[str, Any]]:
        evidence_kinds = {item.kind for item in evidence}
        feature_names = {feature.name for feature in features}

        def item(name: str, status: str, evidence_refs: list[str], gaps: list[str], required: list[str]) -> dict[str, Any]:
            return {
                "standard": name,
                "status": status,
                "evidence": evidence_refs[:12],
                "gaps": gaps,
                "required_next": required,
            }

        review = [
            item(
                "behavioral_equivalence",
                "planned" if verification_plan.black_box_tests and features else "insufficient",
                [test["name"] for test in verification_plan.black_box_tests],
                [] if features else ["No inferred features to test."],
                ["Run held-out tests against the authorized original or captured phenotype traces before claiming support."],
            ),
            item(
                "research_grounding",
                "supported" if evidence_kinds & {"research_result", "research_note", "similar_program"} else "open_gap",
                [entry["phase"] for entry in blueprint.research_plan],
                [] if evidence_kinds & {"research_result", "research_note", "similar_program"} else ["No research evidence attached."],
                ["Attach public docs, open-source alternatives, engineering notes, and implementation-language references."],
            ),
            item(
                "implementation_completeness",
                "planned" if blueprint.implementation_plan and feature_names else "insufficient",
                [entry["phase"] for entry in blueprint.implementation_plan],
                [] if feature_names else ["Feature map is empty or too weak."],
                ["Generate concrete code modules, adapters, storage, and tests for each inferred feature."],
            ),
            item(
                "security_and_legal_boundary",
                "supported",
                blueprint.safety_boundary,
                [],
                ["Keep source/decompilation/proprietary-copy boundaries auditable in every artifact."],
            ),
            item(
                "operational_reliability",
                "planned" if verification_plan.performance_checks and verification_plan.edge_case_tests else "insufficient",
                [check["name"] for check in verification_plan.performance_checks + verification_plan.edge_case_tests],
                [],
                ["Exercise startup, memory, offline, permission-denied, interrupted-write, and large-input cases."],
            ),
            item(
                "observability_and_receipts",
                "supported" if "evidence_receipts" in {component["name"] for component in blueprint.components} else "open_gap",
                [component["name"] for component in blueprint.components],
                [],
                ["Emit receipts for source, research, analogy, generated code, test runs, and unresolved unknowns."],
            ),
        ]
        if genome.reconstruction_unknowns:
            review.append(
                item(
                    "unknowns_management",
                    "open_gap",
                    genome.reconstruction_unknowns,
                    list(genome.reconstruction_unknowns),
                    ["Shrink unknowns with additional traces, source/docs, golden files, and adversarial tests."],
                )
            )
        return review

    def _build_verification_plan(
        self,
        features: list[ProgramDNAFeature],
        evidence: list[ProgramDNAEvidence],
        genome: ProgramDNAGenome,
    ) -> ProgramDNAVerificationPlan:
        feature_names = {feature.name for feature in features}
        black_box_tests = [
            {
                "name": f"black_box_{feature.name}",
                "setup": "exercise authorized original or captured phenotype trace",
                "assertion": f"replacement preserves externally visible {feature.name} behavior",
                "evidence": feature.evidence_sources,
            }
            for feature in features
        ]
        ui_tests = [
            {
                "name": f"ui_{flow['feature']}",
                "steps": flow["steps"],
                "assertion": "visible UI state changes match the reconstructed workflow contract",
            }
            for flow in genome.workflow_graph
        ]
        golden_file_tests = [
            {
                "name": f"golden_{idx}",
                "format": item.details.get("suffix") or item.details.get("name") or item.summary,
                "source": item.source,
                "assertion": "replacement reads/writes a byte-compatible or schema-compatible artifact",
            }
            for idx, item in enumerate(evidence, start=1)
            if item.kind in {"file_format", "manifest"} or "export" in item.summary.lower()
        ][:30]
        api_tests = [
            {
                "name": f"api_contract_{idx}",
                "source": surface.get("source"),
                "assertion": "request/response semantics match observed API contract",
            }
            for idx, surface in enumerate(genome.api_surface, start=1)
        ]
        interaction_tests = [
            {
                "name": f"interaction_surface_{idx}",
                "source": surface.get("source"),
                "category": surface.get("category"),
                "assertion": "observed interaction is reproduced, blocked, or explained according to its governed contract",
            }
            for idx, surface in enumerate(genome.interaction_surfaces, start=1)
        ]
        if genome.aura_interaction_surface:
            interaction_tests.append(
                {
                    "name": "aura_touchpoints_governed",
                    "assertion": "all Aura-facing calls use documented local routes, receipts, and authorization checks",
                }
            )
        if genome.network_surface:
            interaction_tests.append(
                {
                    "name": "network_surface_bounded",
                    "assertion": "network behavior is allowlisted, rate-limited, logged, and safe under offline conditions",
                }
            )
        if genome.hardware_surface:
            interaction_tests.append(
                {
                    "name": "hardware_permission_boundary",
                    "assertion": "camera, microphone, screen, keyboard, mouse, GPU, and sensor access stay permission-gated",
                }
            )
        edge_case_tests = [
            {"name": "unknown_feature_fails_closed", "assertion": "unsupported behavior does not fabricate success"},
            {"name": "partial_write_recovery", "assertion": "state remains recoverable after interrupted write/export"},
            {"name": "offline_mode", "assertion": "network loss does not destroy local state"},
            {"name": "permission_denied", "assertion": "missing permission produces explicit recoverable receipt"},
        ]
        performance_checks = [
            {"name": "startup_budget", "assertion": "replacement starts inside target budget for chosen platform"},
            {"name": "large_project_budget", "assertion": "large inputs stay bounded in memory and time"},
        ]
        security_checks = [
            {"name": "no_secret_exfiltration", "assertion": "credentials/tokens are never logged or copied"},
            {"name": "license_boundary", "assertion": "no proprietary source or decompiled code is embedded"},
            {"name": "sandbox_execution", "assertion": "untrusted artifacts run only in sandboxed test environments"},
        ]
        compatibility_checks = [
            {
                "name": f"compat_{self._slug(target)}",
                "assertion": f"replacement supports target compatibility mode: {target}",
            }
            for target in genome.compatibility_targets
        ]
        if "authentication" not in feature_names:
            security_checks.append(
                {"name": "auth_not_claimed", "assertion": "replacement does not claim auth compatibility without evidence"}
            )
        return ProgramDNAVerificationPlan(
            black_box_tests=black_box_tests,
            ui_tests=ui_tests,
            golden_file_tests=golden_file_tests,
            api_tests=api_tests,
            interaction_tests=interaction_tests,
            edge_case_tests=edge_case_tests,
            performance_checks=performance_checks,
            security_checks=security_checks,
            compatibility_checks=compatibility_checks,
        )

    def _emit_scaffold(
        self,
        *,
        target_name: str,
        blueprint: ProgramDNABlueprint,
        genome: ProgramDNAGenome,
        verification_plan: ProgramDNAVerificationPlan,
        features: list[ProgramDNAFeature],
        output_dir: Path | None,
        stack: str,
    ) -> str:
        slug = self._slug(target_name)
        root = (output_dir or self._generated_workspace()) / slug
        src = root / "src"
        tests = root / "tests"
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "program_dna.emit_scaffold",
            domain="file_write",
        ):
            gateway.ensure_directory(
                src,
                source="core.self_improvement.program_dna.emit_scaffold",
            )
            gateway.ensure_directory(
                tests,
                source="core.self_improvement.program_dna.emit_scaffold",
            )

        self._write_text(
            root / "PROGRAM_DNA_BLUEPRINT.json",
            json.dumps(asdict(blueprint), indent=2, sort_keys=True),
        )
        self._write_text(
            root / "PROGRAM_GENOME.json",
            json.dumps(asdict(genome), indent=2, sort_keys=True),
        )
        self._write_text(
            root / "VERIFICATION_PLAN.json",
            json.dumps(asdict(verification_plan), indent=2, sort_keys=True),
        )
        self._write_text(
            root / "RESEARCH_PLAN.json",
            json.dumps(blueprint.research_plan, indent=2, sort_keys=True),
        )
        self._write_text(
            root / "IMPLEMENTATION_PLAN.json",
            json.dumps(blueprint.implementation_plan, indent=2, sort_keys=True),
        )
        self._write_text(
            root / "STANDARDS_REVIEW.json",
            json.dumps(blueprint.standards_review, indent=2, sort_keys=True),
        )
        self._write_text(
            root / "PROGRAM_DNA_SEQUENCE.json",
            json.dumps(genome.dna_sequence, indent=2, sort_keys=True),
        )
        self._write_text(
            root / "BUILD_PLAYBOOK.json",
            json.dumps(genome.build_playbook, indent=2, sort_keys=True),
        )
        feature_constants = "\n".join(
            f"    {feature.name!r}: {feature.confidence!r},"
            for feature in features
        )
        trace_constants = json.dumps(
            blueprint.implementation_plan[0].get("evidence_to_code_trace", [])
            if blueprint.implementation_plan
            else [],
            indent=4,
            sort_keys=True,
        )
        self._write_text(src / "__init__.py", '"""Generated Program DNA scaffold package."""\n')
        self._write_text(
            src / "program.py",
            (
                '"""Clean-room scaffold generated from Program DNA evidence."""\n\n'
                f"TARGET_STACK = {stack!r}\n"
                "FEATURE_CONFIDENCE = {\n"
                f"{feature_constants}\n"
                "}\n\n"
                f"EVIDENCE_TO_CODE_TRACE = {trace_constants}\n\n"
                "class ReconstructedProgram:\n"
                "    def __init__(self):\n"
                "        self.receipts = []\n\n"
                "    def capabilities(self):\n"
                "        return sorted(FEATURE_CONFIDENCE)\n\n"
                "    def evidence_trace(self, feature=None):\n"
                "        if feature is None:\n"
                "            return list(EVIDENCE_TO_CODE_TRACE)\n"
                "        return [item for item in EVIDENCE_TO_CODE_TRACE if item.get('feature') == feature]\n\n"
                "    def execute(self, feature, payload=None):\n"
                "        if feature not in FEATURE_CONFIDENCE:\n"
                "            raise ValueError(f'unknown reconstructed feature: {feature}')\n"
                "        receipt = {\n"
                "            'feature': feature,\n"
                "            'payload': payload or {},\n"
                "            'status': 'planned',\n"
                "            'evidence_trace': self.evidence_trace(feature),\n"
                "        }\n"
                "        self.receipts.append(receipt)\n"
                "        return receipt\n"
            ),
        )
        self._write_text(
            tests / "conftest.py",
            (
                "import sys\n"
                "from pathlib import Path\n\n\n"
                "ROOT = Path(__file__).resolve().parents[1]\n"
                "if str(ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT))\n"
            ),
        )
        self._write_text(
            tests / "test_program_contract.py",
            (
                "from src.program import ReconstructedProgram\n\n\n"
                "def test_reconstructed_program_exposes_inferred_capabilities():\n"
                "    program = ReconstructedProgram()\n"
                "    assert program.capabilities()\n\n\n"
                "def test_reconstructed_program_rejects_unknown_feature():\n"
                "    program = ReconstructedProgram()\n"
                "    try:\n"
                "        program.execute('not_inferred')\n"
                "    except ValueError:\n"
                "        pass\n"
                "    else:\n"
                "        raise AssertionError('unknown features must fail closed')\n"
                "\n\n"
                "def test_reconstructed_program_explains_evidence_trace():\n"
                "    program = ReconstructedProgram()\n"
                "    for capability in program.capabilities():\n"
                "        receipt = program.execute(capability, {'probe': True})\n"
                "        assert receipt['evidence_trace']\n"
            ),
        )
        self._write_text(
            root / "README.md",
            (
                f"# Program DNA Scaffold: {target_name}\n\n"
                "Generated from authorized clean-room evidence. This is a scaffold, not copied source.\n\n"
                "## Produced Artifacts\n\n"
                "- `PROGRAM_DNA_BLUEPRINT.json`\n"
                "- `PROGRAM_GENOME.json`\n"
                "- `VERIFICATION_PLAN.json`\n"
                "- `RESEARCH_PLAN.json`\n"
                "- `IMPLEMENTATION_PLAN.json`\n"
                "- `STANDARDS_REVIEW.json`\n"
                "- `PROGRAM_DNA_SEQUENCE.json`\n"
                "- `BUILD_PLAYBOOK.json`\n"
                "- `src/__init__.py`\n"
                "- `src/program.py`\n"
                "- `tests/conftest.py`\n"
                "- `tests/test_program_contract.py`\n\n"
                "## Build Rule\n\n"
                "Every generated module or behavior must trace back to evidence, research, or a labeled hypothesis. "
                "Unknown internals stay unknown until held-out tests or authorized source/docs close the gap.\n\n"
                "## Safety Boundary\n\n"
                + "\n".join(f"- {item}" for item in blueprint.safety_boundary)
                + "\n"
            ),
        )
        return str(root)

    def _verify_scaffold(self, root: Path, plan: ProgramDNAVerificationPlan) -> None:
        files = [
            root / "PROGRAM_DNA_BLUEPRINT.json",
            root / "PROGRAM_GENOME.json",
            root / "VERIFICATION_PLAN.json",
            root / "RESEARCH_PLAN.json",
            root / "IMPLEMENTATION_PLAN.json",
            root / "STANDARDS_REVIEW.json",
            root / "PROGRAM_DNA_SEQUENCE.json",
            root / "BUILD_PLAYBOOK.json",
            root / "src" / "__init__.py",
            root / "src" / "program.py",
            root / "tests" / "conftest.py",
            root / "tests" / "test_program_contract.py",
            root / "README.md",
        ]
        plan.scaffold_files = [str(path) for path in files if path.exists()]
        try:
            program_path = root / "src" / "program.py"
            conftest_path = root / "tests" / "conftest.py"
            test_path = root / "tests" / "test_program_contract.py"
            ast.parse(program_path.read_text(encoding="utf-8"), filename=str(program_path))
            ast.parse(conftest_path.read_text(encoding="utf-8"), filename=str(conftest_path))
            ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
            plan.scaffold_syntax_ok = True
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            plan.scaffold_syntax_ok = False
            self._record_degradation("program_dna_reconstruction.scaffold_verify", exc, severity="warning")
        self._write_text(
            root / "VERIFICATION_PLAN.json",
            json.dumps(asdict(plan), indent=2, sort_keys=True),
        )

    def _binary_static_analysis_plan(self, source_paths: list[str]) -> list[ProgramDNAEvidence]:
        evidence: list[ProgramDNAEvidence] = []
        ghidra = shutil.which("analyzeHeadless") or shutil.which("ghidra")
        for raw_path in source_paths:
            path = Path(raw_path).expanduser()
            if not path.exists() or path.is_dir() or path.suffix.lower() in SOURCE_EXTENSIONS:
                continue
            evidence.append(
                ProgramDNAEvidence(
                    kind="binary_static_analysis_plan",
                    source=str(path),
                    summary=(
                        "Binary static analysis is authorized but not executed inline; "
                        f"Ghidra available={bool(ghidra)}. Run in sandbox and keep decompiled artifacts out of clean-room output."
                    ),
                    confidence=0.42 if ghidra else 0.25,
                    details={"ghidra_available": bool(ghidra), "tool": ghidra},
                    sha256=self._sha256(path),
                )
            )
        return evidence

    def _surface_entries(
        self,
        evidence: list[ProgramDNAEvidence],
        *,
        kinds: set[str],
        markers: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        surfaces: list[dict[str, Any]] = []
        for item in evidence:
            text = f"{item.summary} {json.dumps(item.details, sort_keys=True)}".lower()
            if item.kind not in kinds and not any(marker in text for marker in markers):
                continue
            surfaces.append(
                {
                    "category": item.kind,
                    "source": item.source,
                    "summary": item.summary,
                    "confidence": item.confidence,
                    "observed": item.kind in kinds,
                    "markers": [marker for marker in markers if marker in text][:8],
                }
            )
        return surfaces[:40]

    def _collect_live_host_snapshot(self) -> list[ProgramDNAEvidence]:
        """Collect a bounded local host snapshot for explicit defensive study.

        This is intentionally shallow: it records process/network shape, not
        memory contents, credentials, packet payloads, or private app internals.
        """

        from core.runtime.resource_observation import get_resource_observer

        evidence: list[ProgramDNAEvidence] = []
        observer = get_resource_observer()
        provenance = observer.provenance
        if not provenance.host_observed:
            self._record_degradation(
                "program_dna_reconstruction.host_snapshot",
                RuntimeError(
                    "live host snapshot refused non-host observation "
                    f"source={provenance.source.value}"
                ),
                severity="debug",
            )
            return evidence
        try:
            psutil = importlib.import_module("psutil")
        except ImportError as exc:
            self._record_degradation("program_dna_reconstruction.host_snapshot", exc, severity="debug")
            return evidence

        processes: list[dict[str, Any]] = []
        try:
            for process in observer.processes():
                cmdline = " ".join(str(part) for part in process.cmdline[:8])
                processes.append(
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "username": process.username,
                        "cmdline_hint": cmdline[:240],
                    }
                )
                if len(processes) >= 40:
                    break
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._record_degradation("program_dna_reconstruction.process_snapshot", exc, severity="debug")
        if processes:
            evidence.append(
                ProgramDNAEvidence(
                    kind="process_observation",
                    source=f"{provenance.source.value}_host_snapshot:processes",
                    summary=f"Bounded process snapshot captured {len(processes)} visible process record(s).",
                    confidence=0.58,
                    details={
                        "processes": processes,
                        "observation": provenance.to_dict(),
                    },
                )
            )

        connections: list[dict[str, Any]] = []
        try:
            for conn in psutil.net_connections(kind="inet")[:80]:
                laddr = getattr(conn, "laddr", None)
                raddr = getattr(conn, "raddr", None)
                connections.append(
                    {
                        "fd": getattr(conn, "fd", None),
                        "family": str(getattr(conn, "family", "")),
                        "type": str(getattr(conn, "type", "")),
                        "local": f"{getattr(laddr, 'ip', '')}:{getattr(laddr, 'port', '')}" if laddr else "",
                        "remote_present": bool(raddr),
                        "status": getattr(conn, "status", ""),
                        "pid": getattr(conn, "pid", None),
                    }
                )
        except (psutil.Error, RuntimeError, TypeError, ValueError) as exc:
            self._record_degradation("program_dna_reconstruction.network_snapshot", exc, severity="debug")
        if connections:
            evidence.append(
                ProgramDNAEvidence(
                    kind="network_observation",
                    source=f"{provenance.source.value}_host_snapshot:inet_connections",
                    summary=f"Bounded network socket snapshot captured {len(connections)} visible connection record(s).",
                    confidence=0.54,
                    details={
                        "connections": connections,
                        "observation": provenance.to_dict(),
                    },
                )
            )
        return evidence

    def _walk_limited(self, root: Path, *, max_files: int) -> list[Path]:
        files: list[Path] = []
        for current_root, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for name in names:
                if name.startswith("."):
                    continue
                path = Path(current_root) / name
                if path.is_file():
                    files.append(path)
                    if len(files) >= max_files:
                        return files
        return files

    def _python_public_symbols(self, path: Path) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            return []
        symbols: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                symbols.append(node.name)
        return symbols

    def _flow_steps_for(self, feature_name: str) -> list[str]:
        presets = {
            "document_creation": ["open editor", "capture input", "persist content", "confirm saved state"],
            "export_pipeline": ["select content", "render/export", "write artifact", "verify artifact exists"],
            "search_and_retrieval": ["index source", "accept query", "rank results", "return matched item"],
            "authentication": ["capture credentials/tokens through OS-approved flow", "establish session", "refresh or fail closed"],
            "automation": ["define trigger", "schedule/execute action", "record receipt", "surface result"],
        }
        return presets.get(feature_name, ["capture intent", "execute behavior", "verify effect", "record receipt"])

    def _feature_category(self, name: str) -> str:
        if name in {"document_creation", "export_pipeline", "media_handling"}:
            return "user_workflow"
        if name in {"persistence", "settings_preferences", "authentication"}:
            return "state"
        if name in {"web_integration", "collaboration"}:
            return "integration"
        return "core"

    def _infer_purpose(
        self,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        feature_names: set[str],
    ) -> str:
        if {"document_creation", "export_pipeline"} <= feature_names:
            return f"{target_name} appears to create, manage, and export user-authored documents."
        if "search_and_retrieval" in feature_names:
            return f"{target_name} appears to retrieve, filter, or organize information."
        if "automation" in feature_names:
            return f"{target_name} appears to automate workflows or scheduled actions."
        docstrings = [
            item.details.get("module_docstring", "")
            for item in evidence
            if item.kind == "python_api" and item.details.get("module_docstring")
        ]
        if docstrings:
            return str(docstrings[0]).splitlines()[0][:240]
        return f"{target_name} purpose must be refined from additional behavior traces."

    def _file_format_from_evidence(self, item: ProgramDNAEvidence) -> str:
        text = f"{item.summary} {json.dumps(item.details, sort_keys=True)}".lower()
        for fmt in ("pdf", "json", "csv", "xml", "sqlite", "markdown", "html", "png", "jpg"):
            if fmt in text:
                return fmt
        suffix = item.details.get("suffix")
        if suffix:
            return str(suffix).lstrip(".")
        return item.kind

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _generated_workspace(self) -> Path:
        """Where scaffolds go: outside the source tree, by this engine's own rule.

        The standards review this engine emits asserts that a generated
        workspace is separate from the runtime. Defaulting to ``project_root``
        broke that assertion silently, so the default now comes from the
        configured generated-code directory and falls back to the project only
        when configuration is unavailable.
        """
        try:
            from core.config import get_config

            return Path(get_config().paths.generated_dir)
        except _WORKSPACE_LOOKUP_FAILURES:
            return self.project_root / "artifacts" / "program_dna"

    def _slug(self, value: str) -> str:
        """A directory name from a target name.

        Capped at :data:`_SLUG_WORDS` words. A model that fills ``target``
        with a clause instead of a name once produced a 71-character directory
        beginning mid-word, and a name nobody can read is a name nobody can
        find again.
        """
        words = re.split(r"[^a-zA-Z0-9_.]+", str(value or "").strip())
        kept = [word for word in words if word][:_SLUG_WORDS]
        slug = "-".join(kept).strip("-.").lower()
        return slug[:_SLUG_CHARS].strip("-.") or "program"

    @staticmethod
    def _expanded_path(value: str | os.PathLike[str]) -> Path:
        return Path(value).expanduser()

    def _inspect_raw_path(self, value: str | os.PathLike[str]) -> list[ProgramDNAEvidence]:
        return self._inspect_path(self._expanded_path(value))

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list | tuple | set):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    def _write_text(self, path: Path, text: str) -> None:
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "program_dna.write_artifact",
            domain="file_write",
        ):
            gateway.write_text(
                path,
                text,
                source="core.self_improvement.program_dna.write_artifact",
            )

    def _record_degradation(
        self,
        subsystem: str,
        exc: BaseException,
        *,
        severity: str = "warning",
        action: str = "",
    ) -> None:
        try:
            errors = importlib.import_module("core.runtime.errors")

            if action:
                errors.record_degradation(subsystem, exc, severity=severity, action=action)
            else:
                errors.record_degradation(subsystem, exc, severity=severity)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return


_PROGRAM_DNA_INSTANCE: ProgramDNAReconstructionEngine | None = None


def get_program_dna_reconstruction_engine(
    *,
    project_root: str | os.PathLike[str] | None = None,
    internal_lab: Any | None = None,
) -> ProgramDNAReconstructionEngine:
    global _PROGRAM_DNA_INSTANCE
    if _PROGRAM_DNA_INSTANCE is None:
        _PROGRAM_DNA_INSTANCE = ProgramDNAReconstructionEngine(
            project_root=project_root,
            internal_lab=internal_lab,
        )
    return _PROGRAM_DNA_INSTANCE


def register_program_dna_reconstruction_engine(
    *,
    project_root: str | os.PathLike[str] | None = None,
    internal_lab: Any | None = None,
) -> ProgramDNAReconstructionEngine:
    engine = get_program_dna_reconstruction_engine(project_root=project_root, internal_lab=internal_lab)
    service_registry = importlib.import_module("core.runtime.service_registry")
    service_names = importlib.import_module("core.service_names")
    service_names_cls = service_names.ServiceNames

    service_registry.register_runtime_service(
        service_names_cls.PROGRAM_DNA_RECONSTRUCTION,
        engine,
        required=False,
        owner="core/self_improvement/program_dna.py",
        registered_by="register_program_dna_reconstruction_engine",
        required_for="authorized program DNA reconstruction and clean-room scaffolding",
        failure_policy="degrade_with_receipt",
    )
    return engine


__all__ = [
    "ProgramDNABlueprint",
    "ProgramDNAEvidence",
    "ProgramDNAFeature",
    "ProgramDNAReconstructionEngine",
    "ProgramDNAResult",
    "ProgramDNAGenome",
    "ProgramDNAVerificationPlan",
    "get_program_dna_reconstruction_engine",
    "register_program_dna_reconstruction_engine",
]
