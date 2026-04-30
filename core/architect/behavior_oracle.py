"""Semantic behavior oracle for T2/T3 ASA promotion proof."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from core.architect.models import ArchitectureGraph, MutationTier, ProofResult, RefactorPlan, SemanticSurface


PROTECTED_EFFECTS = frozenset(
    {
        "authority_call",
        "capability_token",
        "memory_write",
        "state_write",
        "tool_execution",
        "subprocess",
        "database_write",
        "network",
        "llm_call",
    }
)

PROTECTED_SURFACES = frozenset(
    {
        SemanticSurface.AUTHORITY_GOVERNANCE,
        SemanticSurface.CAPABILITY_TOOL_EXECUTION,
        SemanticSurface.MEMORY_WRITE_READ,
        SemanticSurface.STATE_MUTATION,
        SemanticSurface.BOOT_RUNTIME_KERNEL,
        SemanticSurface.LLM_MODEL_ROUTING,
        SemanticSurface.IDENTITY_PERSONA,
        SemanticSurface.SELF_MODIFICATION,
        SemanticSurface.PROOF_TEST_EVALUATION,
    }
)


@dataclass(frozen=True)
class SemanticOracleVerdict:
    equivalent: bool
    regressions: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_proof_result(self) -> ProofResult:
        return ProofResult(
            obligation_id="semantic_behavior_equivalence",
            passed=self.equivalent,
            status="passed" if self.equivalent else "failed",
            evidence={
                "regressions": list(self.regressions),
                "improvements": list(self.improvements),
                **self.evidence,
            },
        )


class SemanticBehaviorOracle:
    """Compare semantic graph contracts, not only aggregate metrics."""

    def evaluate(
        self,
        plan: RefactorPlan,
        before: ArchitectureGraph,
        after: ArchitectureGraph,
        proof_statuses: dict[str, str],
    ) -> SemanticOracleVerdict:
        regressions: list[str] = []
        improvements: list[str] = []
        evidence: dict[str, Any] = {
            "changed_files": list(plan.changed_files),
            "tier": plan.risk_tier.name,
        }

        if plan.risk_tier <= MutationTier.T1_CLEANUP:
            return SemanticOracleVerdict(True, evidence={**evidence, "reason": "T0/T1 semantic oracle not required"})

        for rel in plan.changed_files:
            before_contract = _file_contract(before, rel)
            after_contract = _file_contract(after, rel)
            evidence[rel] = {"before": before_contract, "after": after_contract}
            regressions.extend(_compare_file_contract(rel, before_contract, after_contract, plan))

        if proof_statuses.get("safe_boot") != "passed":
            regressions.append("safe skeletal Aura boot did not pass")
        if proof_statuses.get("changed_modules_import") != "passed":
            regressions.append("changed module import proof did not pass")
        critical_status = proof_statuses.get("critical_tests")
        if critical_status not in {None, "passed", "not_available"}:
            regressions.append("critical test subset did not pass")

        before_receipts = int(before.metrics.get("runtime_receipts", 0) or 0)
        after_receipts = int(after.metrics.get("runtime_receipts", 0) or 0)
        evidence["runtime_receipts"] = {"before": before_receipts, "after": after_receipts}
        if after_receipts < before_receipts and plan.risk_tier <= MutationTier.T2_REFACTOR:
            regressions.append("runtime receipt coverage decreased")
        elif after_receipts > before_receipts:
            improvements.append("runtime receipt coverage increased")

        before_paths = int(before.metrics.get("runtime_receipt_paths", 0) or 0)
        after_paths = int(after.metrics.get("runtime_receipt_paths", 0) or 0)
        evidence["runtime_receipt_paths"] = {"before": before_paths, "after": after_paths}
        if after_paths < before_paths and plan.risk_tier <= MutationTier.T2_REFACTOR:
            regressions.append("runtime receipt path coverage decreased")

        return SemanticOracleVerdict(
            equivalent=not regressions,
            regressions=tuple(dict.fromkeys(regressions)),
            improvements=tuple(dict.fromkeys(improvements)),
            evidence=evidence,
        )


def _file_contract(graph: ArchitectureGraph, rel: str) -> dict[str, Any]:
    nodes = graph.nodes_for_path(rel)
    public_symbols = {
        node.qualified_name: {
            "kind": node.kind,
            "name": node.name,
            "args": tuple(node.metadata.get("args", ())),
            "line_count": node.metadata.get("line_count", 0),
        }
        for node in nodes
        if node.kind in {"class", "function", "async_function"} and not node.name.startswith("_")
    }
    effects = Counter(effect for node in nodes for effect in node.metadata.get("effects", ()))
    calls = sorted(
        edge.target
        for edge in graph.edges
        if edge.path == rel and edge.kind == "calls" and not str(edge.target).startswith("_")
    )
    service_regs = sorted(edge.target for edge in graph.edges if edge.path == rel and "register" in edge.target.lower())
    surfaces = tuple(sorted(surface.value for surface in graph.semantic_surfaces.get(rel, ())))
    return {
        "public_symbols": public_symbols,
        "effects": dict(effects),
        "protected_effects": {key: effects.get(key, 0) for key in PROTECTED_EFFECTS if effects.get(key, 0)},
        "calls_hash": _stable_list_hash(calls),
        "call_count": len(calls),
        "service_registrations": service_regs,
        "surfaces": surfaces,
    }


def _compare_file_contract(rel: str, before: dict[str, Any], after: dict[str, Any], plan: RefactorPlan) -> list[str]:
    regressions: list[str] = []
    before_public = set(before["public_symbols"])
    after_public = set(after["public_symbols"])
    removed = sorted(before_public - after_public)
    if removed and "caller migration" not in " ".join(plan.proof_obligations).lower():
        regressions.append(f"{rel}: public symbols removed without caller migration proof: {removed[:5]}")
    changed_signature = sorted(
        name
        for name in before_public & after_public
        if before["public_symbols"][name].get("args") != after["public_symbols"][name].get("args")
    )
    if changed_signature:
        regressions.append(f"{rel}: public signatures changed: {changed_signature[:5]}")

    before_surfaces = set(before["surfaces"])
    after_surfaces = set(after["surfaces"])
    dropped_protected = sorted(surface for surface in before_surfaces - after_surfaces if SemanticSurface(surface) in PROTECTED_SURFACES)
    if dropped_protected and plan.risk_tier <= MutationTier.T2_REFACTOR:
        regressions.append(f"{rel}: protected semantic surfaces disappeared: {dropped_protected}")

    for effect in PROTECTED_EFFECTS:
        before_count = int(before["effects"].get(effect, 0) or 0)
        after_count = int(after["effects"].get(effect, 0) or 0)
        if after_count > before_count:
            regressions.append(f"{rel}: protected effect increased: {effect} {before_count}->{after_count}")

    before_regs = set(before["service_registrations"])
    after_regs = set(after["service_registrations"])
    if before_regs != after_regs and plan.risk_tier <= MutationTier.T2_REFACTOR:
        regressions.append(f"{rel}: service registration contract changed")
    return regressions


def _stable_list_hash(values: list[str]) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:16]
