"""Semantic behavior oracle for ASA promotion proof.

The oracle is the thing that says "this change did not alter behaviour". Four
of the six CP126 findings here are the same failure in different places: a
check that was declared but not performed. T0/T1 plans were handed an
unconditional pass, a missing critical-test result counted as a non-regression,
the call graph was measured and never compared, and the protected-surface,
receipt-coverage and service-registration checks were all scoped to T2 — so a
T3 behavioural change could drop them silently. A removal could also be waived
by the words "caller migration" appearing anywhere in the plan's obligations.

CP126 169305bf / 04927872 / 1978de68 / dadd9e6d / 880aae0e / 62eb172e.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from core.architect.models import (
    ArchitectureGraph,
    MutationTier,
    ProofResult,
    RefactorPlan,
    SemanticSurface,
)

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

        # CP126 169305bf: T0/T1 used to return "equivalent" before examining
        # anything at all — and those are exactly the tiers autonomous cleanup
        # runs at. A cleanup that alters a public signature, a protected
        # effect, a call path or a service registration is not a cleanup, so
        # the low tiers are held to the STRICTEST contract, not exempted.
        strict = plan.risk_tier <= MutationTier.T1_CLEANUP
        evidence["contract"] = "strict_equivalence" if strict else "behavioral_review"

        for rel in plan.changed_files:
            before_contract = _file_contract(before, rel)
            after_contract = _file_contract(after, rel)
            evidence[rel] = {"before": before_contract, "after": after_contract}
            regressions.extend(
                _compare_file_contract(
                    rel, before_contract, after_contract, plan, after, strict=strict
                )
            )

        # CP126 04927872: a missing result and an explicit "not_available"
        # both counted as a non-regression, so the absence of the critical
        # suite could never block equivalence. Absence of evidence is not
        # evidence of equivalence — it is UNPROVEN, which the tiers this
        # oracle governs (T2/T3) must not promote on. At T0/T1 an unavailable
        # harness is tolerated, matching ProofReceipt.passed, but it is
        # recorded as unproven rather than silently counted as a pass.
        unproven: list[str] = []
        for obligation, label in (
            ("safe_boot", "safe skeletal Aura boot"),
            ("changed_modules_import", "changed module import proof"),
            ("critical_tests", "critical test subset"),
        ):
            status = proof_statuses.get(obligation)
            evidence[obligation] = status or "missing"
            regression = _proof_regression(label, status, plan.risk_tier, unproven)
            if regression:
                regressions.append(regression)
        if unproven:
            evidence["unproven"] = unproven

        before_receipts = int(before.metrics.get("runtime_receipts", 0) or 0)
        after_receipts = int(after.metrics.get("runtime_receipts", 0) or 0)
        evidence["runtime_receipts"] = {"before": before_receipts, "after": after_receipts}
        # CP126 62eb172e: these were gated on `<= T2`, so a T3 behavioural
        # change could delete receipt coverage and service registrations
        # without the oracle failing. Losing an observability or governance
        # contract is a regression at every tier.
        if after_receipts < before_receipts:
            regressions.append(
                f"runtime receipt coverage decreased ({before_receipts}->{after_receipts})"
            )
        elif after_receipts > before_receipts:
            improvements.append("runtime receipt coverage increased")

        before_paths = int(before.metrics.get("runtime_receipt_paths", 0) or 0)
        after_paths = int(after.metrics.get("runtime_receipt_paths", 0) or 0)
        evidence["runtime_receipt_paths"] = {"before": before_paths, "after": after_paths}
        if after_paths < before_paths:
            regressions.append(
                f"runtime receipt path coverage decreased ({before_paths}->{after_paths})"
            )

        return SemanticOracleVerdict(
            equivalent=not regressions,
            regressions=tuple(dict.fromkeys(regressions)),
            improvements=tuple(dict.fromkeys(improvements)),
            evidence=evidence,
        )


#: Statuses that mean "the check could not run", as distinct from "the check
#: ran and failed". Absence is UNPROVEN, never PASSED.
UNAVAILABLE_STATUSES = frozenset(
    {"not_available", "unavailable", "BOOT_HARNESS_UNAVAILABLE", "skipped", "not_run"}
)


def _proof_regression(
    label: str,
    status: str | None,
    tier: MutationTier,
    unproven: list[str],
) -> str:
    """The regression this proof status implies, or '' when it is acceptable."""
    if status == "passed":
        return ""
    if status is None or status in UNAVAILABLE_STATUSES:
        if tier <= MutationTier.T1_CLEANUP:
            unproven.append(f"{label}: {status or 'missing'}")
            return ""
        return (
            f"{label} evidence is unavailable ({status or 'missing'}); "
            f"{tier.name} promotion requires it"
        )
    return f"{label} did not pass (status: {status})"


def _is_public_qualified(qualified_name: str, name: str) -> bool:
    """Public means every component of the path is public."""
    if name.startswith("_"):
        return False
    parts = [part for part in str(qualified_name or "").split(".") if part]
    # Skip the module path; only the owning class / symbol components matter.
    return not any(part.startswith("_") for part in parts[-2:])


def _file_contract(graph: ArchitectureGraph, rel: str) -> dict[str, Any]:
    nodes = graph.nodes_for_path(rel)
    # CP126 1978de68: methods were excluded entirely and only positional arg
    # names were recorded, so a class's entire public surface — and every
    # default, keyword-only parameter, variadic and return type — was outside
    # the contract.
    public_symbols = {
        node.qualified_name: {
            "kind": node.kind,
            "name": node.name,
            "args": tuple(node.metadata.get("args", ())),
            "signature": _normalized_signature(node.metadata.get("signature")),
            "decorators": tuple(node.metadata.get("decorators", ())),
            "line_count": node.metadata.get("line_count", 0),
        }
        for node in nodes
        if node.kind in {"class", "function", "async_function", "method", "async_method"}
        and _is_public_qualified(node.qualified_name, node.name)
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
        "calls": calls,
        "calls_hash": _stable_list_hash(calls),
        "call_count": len(calls),
        "service_registrations": service_regs,
        "surfaces": surfaces,
    }


def _normalized_signature(signature: Any) -> tuple[tuple[str, Any], ...]:
    """A hashable, comparable form of the recorded signature."""
    if not isinstance(signature, dict):
        return ()
    return tuple(sorted((str(key), _hashable(value)) for key, value in signature.items()))


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    return value


def _remaining_callers(graph: ArchitectureGraph, rel: str, symbol: str) -> list[str]:
    """Call sites for ``symbol`` outside its own file, in the AFTER graph.

    Name-based: the graph records call targets as source text, so this is a
    sound *alarm* (a hit means something still names it) and a weak *proof*
    of absence. It is paired with an explicit obligation, not trusted alone.
    """
    short = symbol.rsplit(".", 1)[-1]
    hits: list[str] = []
    for edge in graph.edges:
        if edge.kind != "calls" or edge.path == rel:
            continue
        target = str(edge.target)
        if target == short or target.endswith("." + short):
            hits.append(f"{edge.path}:{edge.line}")
    return sorted(set(hits))[:10]


def _migration_declared(plan: RefactorPlan, symbol: str) -> bool:
    """Whether the plan names THIS symbol's migration, not just the words.

    CP126 880aae0e: removal was accepted whenever any obligation text
    contained "caller migration", with nothing binding that phrase to the
    symbol actually removed.
    """
    short = symbol.rsplit(".", 1)[-1].lower()
    for obligation in plan.proof_obligations:
        text = str(obligation).lower()
        if "caller migration" not in text and "caller_migration" not in text:
            continue
        if short in text or symbol.lower() in text:
            return True
    return False


def _compare_file_contract(
    rel: str,
    before: dict[str, Any],
    after: dict[str, Any],
    plan: RefactorPlan,
    after_graph: ArchitectureGraph,
    *,
    strict: bool = False,
) -> list[str]:
    regressions: list[str] = []
    before_public = set(before["public_symbols"])
    after_public = set(after["public_symbols"])

    for removed in sorted(before_public - after_public):
        callers = _remaining_callers(after_graph, rel, removed)
        declared = _migration_declared(plan, removed)
        if callers:
            regressions.append(
                f"{rel}: public symbol {removed} removed but still called from {callers[:3]}"
            )
        elif not declared:
            regressions.append(
                f"{rel}: public symbol {removed} removed without a caller-migration "
                "obligation naming it"
            )

    changed_signature = sorted(
        name
        for name in before_public & after_public
        if before["public_symbols"][name].get("signature")
        != after["public_symbols"][name].get("signature")
        or before["public_symbols"][name].get("args")
        != after["public_symbols"][name].get("args")
    )
    if changed_signature:
        regressions.append(f"{rel}: public signatures changed: {changed_signature[:5]}")

    changed_decorators = sorted(
        name
        for name in before_public & after_public
        if before["public_symbols"][name].get("decorators")
        != after["public_symbols"][name].get("decorators")
    )
    if changed_decorators:
        regressions.append(f"{rel}: public decorators changed: {changed_decorators[:5]}")

    before_surfaces = set(before["surfaces"])
    after_surfaces = set(after["surfaces"])
    dropped_protected = sorted(
        surface for surface in before_surfaces - after_surfaces
        if SemanticSurface(surface) in PROTECTED_SURFACES
    )
    if dropped_protected:
        # CP126 62eb172e: scoped to `<= T2` before, so T3 could drop them.
        regressions.append(f"{rel}: protected semantic surfaces disappeared: {dropped_protected}")

    for effect in PROTECTED_EFFECTS:
        before_count = int(before["effects"].get(effect, 0) or 0)
        after_count = int(after["effects"].get(effect, 0) or 0)
        if after_count > before_count:
            regressions.append(f"{rel}: protected effect increased: {effect} {before_count}->{after_count}")
        elif strict and after_count < before_count:
            regressions.append(
                f"{rel}: protected effect removed under a cleanup tier: "
                f"{effect} {before_count}->{after_count}"
            )

    before_regs = set(before["service_registrations"])
    after_regs = set(after["service_registrations"])
    if before_regs != after_regs:
        regressions.append(
            f"{rel}: service registration contract changed "
            f"(-{sorted(before_regs - after_regs)[:3]} +{sorted(after_regs - before_regs)[:3]})"
        )

    # CP126 dadd9e6d: calls_hash and call_count were computed on both sides and
    # never read, so behavioural call-path rewiring passed as equivalent.
    if before["calls_hash"] != after["calls_hash"]:
        gone = sorted(set(before["calls"]) - set(after["calls"]))
        added = sorted(set(after["calls"]) - set(before["calls"]))
        detail = f"(-{gone[:3]} +{added[:3]})"
        if strict:
            regressions.append(f"{rel}: call graph changed under a cleanup tier {detail}")
        elif gone:
            regressions.append(f"{rel}: call paths removed {detail}")

    return regressions


def _stable_list_hash(values: list[str]) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:16]
