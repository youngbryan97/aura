"""Risk-tier policy for Aura self-modification.

The self-repair loop is allowed to be useful, but it is not allowed to treat
every file as equally mutable.  This module is the single classifier used by
repair, safe modification, proof bundles, and tests.

Tier 0: free auto-fix after targeted tests.
Tier 1: auto-fix only after shadow validation and regression checks.
Tier 2: proposal-only; Aura may draft the patch but not apply it.
Tier 3: sealed; no runtime self-editing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence


class MutationTier(IntEnum):
    FREE_AUTO_FIX = 0
    SHADOW_VALIDATED_AUTO_FIX = 1
    PROPOSE_ONLY = 2
    SEALED = 3

    @property
    def label(self) -> str:
        return {
            MutationTier.FREE_AUTO_FIX: "tier0_free_auto_fix",
            MutationTier.SHADOW_VALIDATED_AUTO_FIX: "tier1_shadow_validated_auto_fix",
            MutationTier.PROPOSE_ONLY: "tier2_propose_only",
            MutationTier.SEALED: "tier3_sealed",
        }[self]


@dataclass(frozen=True)
class MutationTierDecision:
    path: str
    tier: MutationTier
    reason: str
    required_gates: tuple[str, ...]

    @property
    def auto_apply_allowed(self) -> bool:
        return self.tier in {
            MutationTier.FREE_AUTO_FIX,
            MutationTier.SHADOW_VALIDATED_AUTO_FIX,
        }

    @property
    def sealed(self) -> bool:
        return self.tier is MutationTier.SEALED

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "tier": int(self.tier),
            "tier_label": self.tier.label,
            "reason": self.reason,
            "required_gates": list(self.required_gates),
            "auto_apply_allowed": self.auto_apply_allowed,
            "sealed": self.sealed,
        }


_TIER3_PATTERNS: tuple[str, ...] = (
    "core/will.py",
    "core/constitution.py",
    "core/governance_context.py",
    "core/governance/constitutional_amendment.py",
    "core/executive/authority_gateway.py",
    "core/executive/executive_core.py",
    "core/runtime/gateways.py",
    "core/runtime/will_transaction.py",
    "core/runtime/effect_boundary.py",
    "core/runtime/security.py",
    "core/runtime/receipts.py",
    "core/runtime/atomic_writer.py",
    "core/runtime/conformance.py",
    "core/runtime/executors.py",
    "core/runtime/errors.py",
    "core/runtime/boot_safety.py",
    "core/runtime/shutdown_coordinator.py",
    "core/runtime/capability_tokens.py",
    "core/runtime/causal_trace.py",
    "core/runtime/consequential_primitives.py",
    "core/runtime/autonomy_conductor.py",
    "core/runtime/activation_audit.py",
    "core/self_modification/safe_modification.py",
    "core/self_modification/safe_pipeline.py",
    "core/self_modification/self_modification_engine.py",
    "core/self_modification/engine.py",
    "core/self_modification/boot_validator.py",
    "core/self_modification/formal_verifier.py",
    "core/self_modification/mutation_safety.py",
    "core/self_modification/mutation_tiers.py",
    "core/self_modification/fault_pipeline.py",
    "core/self_modification/repair_approval.py",
    "core/self_modification/repair_calibration.py",
    "core/self_modification/patch_genealogy.py",
    "core/bus/actor_bus.py",
    "core/bus/shared_mem_bus.py",
    "core/bus/local_pipe_bus.py",
    "core/consciousness/phi_core.py",
    "core/consciousness/hierarchical_phi.py",
    "core/memory/scar_formation.py",
    "core/memory/scar_court.py",
    "core/security/**",
    "core/guardians/**",
    "aura_main.py",
)

_TIER2_PATTERNS: tuple[str, ...] = (
    "core/agency/agency_orchestrator.py",
    "core/agency/capability_system.py",
    "core/agency/capability_token.py",
    "core/orchestrator/**",
    "core/memory/memory_write_gateway.py",
    "core/memory/**gateway*.py",
    "core/state/state_gateway.py",
    "core/state/state_repository.py",
    "core/state/vault.py",
    "core/consciousness/substrate_*.py",
    "core/consciousness/liquid_substrate*.py",
    "core/brain/llm/llm_router.py",
    "core/brain/llm/model_registry.py",
    "core/brain/llm/mlx_worker.py",
    "core/brain/inference_gate.py",
)

_TIER0_PATTERNS: tuple[str, ...] = (
    "docs/**",
    "tests/**",
    "skills/generated/**",
    "plugins/generated/**",
    "interface/static/**",
    "scratch/**",
    "patches/proposals/**",
)

_TIER1_PATTERNS: tuple[str, ...] = (
    "core/**",
    "skills/**",
    "plugins/**",
    "interface/**",
    "tools/**",
)

_GATES = {
    MutationTier.FREE_AUTO_FIX: ("targeted_test", "syntax", "import"),
    MutationTier.SHADOW_VALIDATED_AUTO_FIX: (
        "targeted_test",
        "behavioral_contracts",
        "shadow_runtime",
        "regression_suite",
        "rollback_snapshot",
    ),
    MutationTier.PROPOSE_ONLY: (
        "targeted_test",
        "behavioral_contracts",
        "shadow_runtime",
        "human_approval",
    ),
    MutationTier.SEALED: ("external_review", "manual_patch", "cold_restart"),
}


def normalize_path(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    parts = []
    for part in text.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def classify_mutation_path(path: str | Path) -> MutationTierDecision:
    normalized = normalize_path(path)
    if _matches(normalized, _TIER3_PATTERNS):
        return MutationTierDecision(
            normalized,
            MutationTier.SEALED,
            "sealed safety, governance, phi, scar, bus, or self-modification root",
            _GATES[MutationTier.SEALED],
        )
    if _matches(normalized, _TIER2_PATTERNS):
        return MutationTierDecision(
            normalized,
            MutationTier.PROPOSE_ONLY,
            "consequential agency, memory, substrate, model-routing, or orchestrator path",
            _GATES[MutationTier.PROPOSE_ONLY],
        )
    if _matches(normalized, _TIER0_PATTERNS):
        return MutationTierDecision(
            normalized,
            MutationTier.FREE_AUTO_FIX,
            "low-risk tests, generated tools, docs, UI static assets, or proposal workspace",
            _GATES[MutationTier.FREE_AUTO_FIX],
        )
    if _matches(normalized, _TIER1_PATTERNS):
        return MutationTierDecision(
            normalized,
            MutationTier.SHADOW_VALIDATED_AUTO_FIX,
            "runtime code outside sealed and proposal-only surfaces",
            _GATES[MutationTier.SHADOW_VALIDATED_AUTO_FIX],
        )
    return MutationTierDecision(
        normalized,
        MutationTier.PROPOSE_ONLY,
        "outside explicit self-modification allow surface",
        _GATES[MutationTier.PROPOSE_ONLY],
    )


def is_sealed_path(path: str | Path) -> bool:
    return classify_mutation_path(path).sealed


def required_gates_for_paths(paths: Iterable[str | Path]) -> tuple[str, ...]:
    gates: set[str] = set()
    for path in paths:
        gates.update(classify_mutation_path(path).required_gates)
    return tuple(sorted(gates))


def assert_runtime_mutation_allowed(path: str | Path, *, auto_apply: bool) -> MutationTierDecision:
    decision = classify_mutation_path(path)
    if decision.tier is MutationTier.SEALED:
        raise PermissionError(f"{decision.path} is sealed from runtime self-modification")
    if auto_apply and decision.tier is MutationTier.PROPOSE_ONLY:
        raise PermissionError(f"{decision.path} is proposal-only and requires explicit approval")
    return decision


__all__ = [
    "MutationTier",
    "MutationTierDecision",
    "classify_mutation_path",
    "is_sealed_path",
    "required_gates_for_paths",
    "assert_runtime_mutation_allowed",
    "normalize_path",
]
