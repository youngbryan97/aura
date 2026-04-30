"""Staged refactor planning for the Autonomous Architecture Governor."""
from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.architect.config import ASAConfig
from core.architect.models import (
    ArchitecturalSmell,
    ArchitectureGraph,
    MutationProposal,
    MutationTier,
    RefactorPlan,
    RefactorStep,
    SemanticSurface,
)
from core.architect.mutation_classifier import MutationClassifier
from core.runtime.atomic_writer import atomic_write_text


class RefactorPlanner:
    """Generate staged plans with invariants and rollback descriptions."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.classifier = MutationClassifier(self.config)

    def plan_for_target(
        self,
        target: str,
        graph: ArchitectureGraph,
        smells: list[ArchitecturalSmell],
        *,
        persist: bool = True,
    ) -> RefactorPlan:
        smell = next((item for item in smells if item.id == target), None)
        if smell is not None:
            plan = self.plan_for_smell(smell, graph)
        else:
            rel = target.replace("\\", "/").lstrip("./")
            plan = self._proposal_plan_for_path(rel, graph)
        if persist:
            self.persist_plan(plan)
        return plan

    def plan_for_smell(self, smell: ArchitecturalSmell, graph: ArchitectureGraph) -> RefactorPlan:
        if smell.kind == "unused_import":
            return self._unused_import_plan(smell, graph)
        if smell.kind == "dead_symbol_candidate":
            return self._dead_code_quarantine_plan(smell, graph)
        if smell.kind == "broad_exception_cluster":
            return self._suggestion_plan(smell, graph, "Tighten broad exception handling with typed failures and explicit degradation decisions.")
        if smell.kind in {"god_file", "god_class"}:
            return self._suggestion_plan(smell, graph, "Split responsibilities through an adapter, migrate callers, then retire the compatibility layer after proof.")
        if smell.kind in {"duplicate_implementation", "duplicate_responsibility"}:
            return self._suggestion_plan(smell, graph, "Introduce a canonical implementation, migrate callers in stages, compare behavior, then quarantine duplicates.")
        if "bypass" in smell.kind:
            return self._suggestion_plan(smell, graph, "Route the effect through the canonical authority, capability, memory, or state gateway.")
        return self._suggestion_plan(smell, graph, "Generate a proposal artifact and require proof before any live mutation.")

    def find_auto_cleanup_plan(self, graph: ArchitectureGraph, smells: list[ArchitecturalSmell]) -> RefactorPlan | None:
        unused = self.find_unused_import_candidates(graph)
        if unused:
            smell = unused[0]
            plan = self._unused_import_plan(smell, graph)
            self.persist_plan(plan)
            return plan
        for smell in smells:
            if smell.kind == "dead_symbol_candidate" and not self.config.is_protected(smell.path):
                plan = self._dead_code_quarantine_plan(smell, graph)
                self.persist_plan(plan)
                return plan
        return None

    def find_unused_import_candidates(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        candidates: list[ArchitecturalSmell] = []
        for node in graph.nodes.values():
            if node.kind != "file" or not node.path.endswith(".py"):
                continue
            if node.path.startswith(("tests/", "docs/")) or self.config.is_protected(node.path) or self.config.is_sealed(node.path):
                continue
            path = self.config.repo_root / node.path
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            removals = _unused_import_rewrites(source)
            if not removals:
                continue
            names = tuple(sorted(item["name"] for item in removals))
            digest = hashlib.sha256((node.path + ",".join(names)).encode("utf-8")).hexdigest()[:12]
            candidates.append(
                ArchitecturalSmell(
                    id=f"unused-import-{digest}",
                    kind="unused_import",
                    severity=smell_severity_for_imports(len(removals)),
                    path=node.path,
                    evidence=(f"unused imports: {', '.join(names[:8])}",),
                    graph_refs=(node.id,),
                    suggested_tier=MutationTier.T1_CLEANUP,
                    proof_obligations=("unused_import_static_proof", "shadow_imports_pass", "rollback_packet"),
                    auto_fixable=True,
                )
            )
        return candidates

    def persist_plan(self, plan: RefactorPlan) -> Path:
        plan_dir = self.config.artifacts / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        path = plan_dir / f"{plan.id}.json"
        atomic_write_text(path, json.dumps(plan_to_dict(plan), indent=2, sort_keys=True, default=str))
        if plan.risk_tier >= MutationTier.T4_GOVERNANCE_SENSITIVE:
            proposals = self.config.artifacts / "proposals"
            proposals.mkdir(parents=True, exist_ok=True)
            atomic_write_text(proposals / f"{plan.id}.json", json.dumps(plan_to_dict(plan), indent=2, sort_keys=True, default=str))
        return path

    def load_plan(self, plan_id_or_path: str) -> RefactorPlan:
        candidate = Path(plan_id_or_path)
        if not candidate.exists():
            candidate = self.config.artifacts / "plans" / f"{plan_id_or_path}.json"
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return plan_from_dict(payload)

    def _unused_import_plan(self, smell: ArchitecturalSmell, graph: ArchitectureGraph) -> RefactorPlan:
        path = self.config.repo_root / smell.path
        source = path.read_text(encoding="utf-8")
        new_source = apply_unused_import_cleanup(source)
        surfaces = graph.semantic_surfaces.get(smell.path, (SemanticSurface.UTILITY_PERIPHERAL,))
        tier = self.classifier.classify((smell.path,), surfaces=surfaces)
        if tier > MutationTier.T1_CLEANUP:
            tier = MutationTier.T4_GOVERNANCE_SENSITIVE
        plan_id = _plan_id("unused-import", (smell.path,), new_source)
        step = RefactorStep(
            id=f"{plan_id}-step-1",
            description=f"Remove statically unused imports from {smell.path}",
            operation="replace_file",
            target_path=smell.path,
            new_content=new_source,
            invariants=("module parses", "public symbols preserved", "changed module imports"),
            rollback="restore original file from rollback packet",
            metadata={"smell_id": smell.id, "static_proof": tuple(smell.evidence)},
        )
        proposal = MutationProposal(
            id=f"{plan_id}-proposal",
            objective=step.description,
            tier=tier,
            affected_files=(smell.path,),
            semantic_surfaces=surfaces,
            smell_ids=(smell.id,),
        )
        return RefactorPlan(
            id=plan_id,
            objective=step.description,
            risk_tier=tier,
            affected_files=(smell.path,),
            affected_symbols=(),
            semantic_surfaces=surfaces,
            steps=(step,),
            proof_obligations=("syntax", "graph_rebuild", "changed_modules_import", "unused_import_static_proof", "rollback_dry_run"),
            expected_smell_reduction=(smell.id,),
            expected_behavior_delta="equivalent",
            promotion_eligible=tier <= MutationTier.T1_CLEANUP,
            proposal=proposal,
        )

    def _dead_code_quarantine_plan(self, smell: ArchitecturalSmell, graph: ArchitectureGraph) -> RefactorPlan:
        surfaces = graph.semantic_surfaces.get(smell.path, (SemanticSurface.UTILITY_PERIPHERAL,))
        tier = max(MutationTier.T1_CLEANUP, self.classifier.classify((smell.path,), surfaces=surfaces))
        if tier > MutationTier.T1_CLEANUP:
            tier = MutationTier.T4_GOVERNANCE_SENSITIVE
        plan_id = _plan_id("dead-code-quarantine", (smell.path, smell.symbol), ",".join(smell.graph_refs))
        step = RefactorStep(
            id=f"{plan_id}-step-1",
            description=f"Quarantine uncertain dead symbol {smell.symbol} from {smell.path}",
            operation="quarantine_symbol",
            target_path=smell.path,
            invariants=("static reachability remains zero", "changed module imports", "quarantine manifest written"),
            rollback="restore original file from rollback packet and quarantine manifest",
            metadata={"symbol": smell.symbol, "graph_refs": tuple(smell.graph_refs)},
        )
        return RefactorPlan(
            id=plan_id,
            objective=step.description,
            risk_tier=tier,
            affected_files=(smell.path,),
            affected_symbols=(smell.symbol,),
            semantic_surfaces=surfaces,
            steps=(step,),
            proof_obligations=("static_reachability_proof", "shadow_imports_pass", "quarantine_manifest", "rollback_dry_run"),
            expected_smell_reduction=(smell.id,),
            expected_behavior_delta="equivalent",
            promotion_eligible=False,
        )

    def _suggestion_plan(self, smell: ArchitecturalSmell, graph: ArchitectureGraph, objective: str) -> RefactorPlan:
        surfaces = graph.semantic_surfaces.get(smell.path, (SemanticSurface.UTILITY_PERIPHERAL,))
        tier = max(smell.suggested_tier, self.classifier.classify((smell.path,), surfaces=surfaces, symbols=(smell.symbol,) if smell.symbol else ()))
        plan_id = _plan_id(smell.kind, (smell.path, smell.symbol), objective)
        step = RefactorStep(
            id=f"{plan_id}-step-1",
            description=objective,
            operation="proposal",
            target_path=smell.path,
            invariants=("no live mutation before proof", "affected callers identified", "rollback plan generated"),
            rollback="proposal-only; no live rollback required",
            metadata={"smell_id": smell.id, "evidence": tuple(smell.evidence), "graph_refs": tuple(smell.graph_refs)},
        )
        return RefactorPlan(
            id=plan_id,
            objective=objective,
            risk_tier=tier,
            affected_files=(smell.path,),
            affected_symbols=(smell.symbol,) if smell.symbol else (),
            semantic_surfaces=surfaces,
            steps=(step,),
            proof_obligations=tuple(dict.fromkeys(smell.proof_obligations + ("shadow_workspace_required", "behavior_fingerprint_required"))),
            expected_smell_reduction=(smell.id,),
            expected_behavior_delta="equivalent or explicitly improved",
            promotion_eligible=tier.autonomous_allowed and not tier.proposal_only and smell.auto_fixable,
        )

    def _proposal_plan_for_path(self, path: str, graph: ArchitectureGraph) -> RefactorPlan:
        surfaces = graph.semantic_surfaces.get(path, (SemanticSurface.UTILITY_PERIPHERAL,))
        tier = self.classifier.classify((path,), surfaces=surfaces)
        plan_id = _plan_id("path-proposal", (path,), ",".join(surface.value for surface in surfaces))
        step = RefactorStep(
            id=f"{plan_id}-step-1",
            description=f"Analyze {path} and create staged refactor proposal",
            operation="proposal",
            target_path=path,
            invariants=("affected surfaces classified", "proof obligations selected", "no autonomous sealed edit"),
            rollback="proposal-only; live files unchanged",
        )
        return RefactorPlan(
            id=plan_id,
            objective=step.description,
            risk_tier=tier,
            affected_files=(path,),
            affected_symbols=(),
            semantic_surfaces=surfaces,
            steps=(step,),
            proof_obligations=("graph_rebuild", "changed_modules_import", "rollback_packet_before_promotion"),
            expected_smell_reduction=(),
            expected_behavior_delta="unknown until shadow proof",
            promotion_eligible=tier.autonomous_allowed and not tier.proposal_only,
        )


def _unused_import_rewrites(source: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    used: set[str] = set()
    imports: list[ast.Import | ast.ImportFrom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
            used.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    removals: list[dict[str, Any]] = []
    for node in imports:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            continue
        unused_aliases: list[str] = []
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            if local not in used:
                unused_aliases.append(alias.name)
        if unused_aliases and len(unused_aliases) == len(node.names):
            removals.append({"line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno) or node.lineno, "name": ",".join(unused_aliases)})
    return removals


def apply_unused_import_cleanup(source: str) -> str:
    removals = _unused_import_rewrites(source)
    if not removals:
        return source
    remove_lines: set[int] = set()
    for removal in removals:
        remove_lines.update(range(int(removal["line"]), int(removal["end_line"]) + 1))
    lines = source.splitlines(keepends=True)
    kept = [line for idx, line in enumerate(lines, start=1) if idx not in remove_lines]
    return "".join(kept)


def smell_severity_for_imports(count: int):
    from core.architect.models import SmellSeverity

    if count >= 5:
        return SmellSeverity.MEDIUM
    if count >= 2:
        return SmellSeverity.LOW
    return SmellSeverity.INFO


def _plan_id(prefix: str, paths: tuple[str, ...], payload: str) -> str:
    digest = hashlib.sha256((prefix + "|" + "|".join(paths) + "|" + payload).encode("utf-8")).hexdigest()[:12]
    return f"asa-{prefix}-{digest}"


def plan_to_dict(plan: RefactorPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["risk_tier"] = plan.risk_tier.name
    payload["semantic_surfaces"] = [surface.value for surface in plan.semantic_surfaces]
    if plan.proposal is not None:
        payload["proposal"]["tier"] = plan.proposal.tier.name
        payload["proposal"]["semantic_surfaces"] = [surface.value for surface in plan.proposal.semantic_surfaces]
    return payload


def plan_from_dict(payload: dict[str, Any]) -> RefactorPlan:
    steps = tuple(RefactorStep(**step) for step in payload.get("steps", ()))
    proposal_payload = payload.get("proposal")
    proposal = None
    if isinstance(proposal_payload, dict):
        proposal = MutationProposal(
            id=str(proposal_payload["id"]),
            objective=str(proposal_payload["objective"]),
            tier=MutationTier.parse(proposal_payload["tier"]),
            affected_files=tuple(proposal_payload.get("affected_files", ())),
            affected_symbols=tuple(proposal_payload.get("affected_symbols", ())),
            semantic_surfaces=tuple(SemanticSurface(surface) for surface in proposal_payload.get("semantic_surfaces", ())),
            expected_behavior_delta=str(proposal_payload.get("expected_behavior_delta", "equivalent")),
            smell_ids=tuple(proposal_payload.get("smell_ids", ())),
            created_at=float(proposal_payload.get("created_at", time.time())),
        )
    return RefactorPlan(
        id=str(payload["id"]),
        objective=str(payload["objective"]),
        risk_tier=MutationTier.parse(payload["risk_tier"]),
        affected_files=tuple(payload.get("affected_files", ())),
        affected_symbols=tuple(payload.get("affected_symbols", ())),
        semantic_surfaces=tuple(SemanticSurface(surface) for surface in payload.get("semantic_surfaces", ())),
        steps=steps,
        proof_obligations=tuple(payload.get("proof_obligations", ())),
        expected_smell_reduction=tuple(payload.get("expected_smell_reduction", ())),
        expected_behavior_delta=str(payload.get("expected_behavior_delta", "")),
        promotion_eligible=bool(payload.get("promotion_eligible", False)),
        proposal=proposal,
        created_at=float(payload.get("created_at", time.time())),
    )
