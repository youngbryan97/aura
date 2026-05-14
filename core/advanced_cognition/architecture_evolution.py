"""Safe architecture evolution under proof obligations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence

from .schemas import stable_hash


class MutationTier(IntEnum):
    CONFIG = 0
    ADAPTER = 1
    FEATURE_MODULE = 2
    SHARED_RUNTIME = 3
    GOVERNANCE_OR_IDENTITY = 4
    SEALED_CORE = 5


@dataclass
class ProofObligation:
    name: str
    description: str
    required: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return bool(self.evidence.get("passed", False)) if self.required else True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["satisfied"] = self.satisfied
        return payload


@dataclass
class ArchitectureMutationPlan:
    plan_id: str
    target_paths: tuple[str, ...]
    tier: MutationTier
    summary: str
    obligations: list[ProofObligation]
    rollback_strategy: str
    sealed: bool = False

    @property
    def promotable(self) -> bool:
        return not self.sealed and all(o.satisfied for o in self.obligations)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tier"] = int(self.tier)
        payload["tier_name"] = self.tier.name.lower()
        payload["promotable"] = self.promotable
        payload["obligations"] = [o.to_dict() for o in self.obligations]
        return payload


class ArchitectureEvolutionGovernor:
    """Assigns mutation tiers and promotion gates for self-modification."""

    SEALED_SUBSTRINGS = (
        "core/will.py",
        "core/executive/authority_gateway.py",
        "core/governance",
        "core/constitution",
        "core/security",
        "core/runtime/atomic_writer.py",
    )

    def plan_mutation(
        self,
        *,
        target_paths: Sequence[str],
        summary: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> ArchitectureMutationPlan:
        paths = tuple(str(p) for p in target_paths)
        tier = self._tier(paths)
        sealed = any(any(sealed in path for sealed in self.SEALED_SUBSTRINGS) for path in paths) and tier >= MutationTier.GOVERNANCE_OR_IDENTITY
        obligations = self._obligations(tier, evidence or {})
        return ArchitectureMutationPlan(
            plan_id=stable_hash({"paths": paths, "summary": summary, "tier": int(tier)}, prefix="arch_"),
            target_paths=paths,
            tier=tier,
            summary=summary,
            obligations=obligations,
            rollback_strategy="ghost_boot_then_atomic_promote_with_tombstone",
            sealed=sealed,
        )

    def evaluate_promotion(
        self,
        plan: ArchitectureMutationPlan,
        evidence: Mapping[str, Any],
    ) -> ArchitectureMutationPlan:
        refreshed = []
        for obligation in plan.obligations:
            refreshed.append(
                ProofObligation(
                    obligation.name,
                    obligation.description,
                    obligation.required,
                    dict(evidence.get(obligation.name, obligation.evidence)),
                )
            )
        return ArchitectureMutationPlan(
            plan.plan_id,
            plan.target_paths,
            plan.tier,
            plan.summary,
            refreshed,
            plan.rollback_strategy,
            plan.sealed,
        )

    def _tier(self, paths: Sequence[str]) -> MutationTier:
        joined = "\n".join(paths)
        if any(s in joined for s in self.SEALED_SUBSTRINGS):
            return MutationTier.GOVERNANCE_OR_IDENTITY
        if "core/" in joined and any(s in joined for s in ("runtime", "memory", "environment", "reasoning")):
            return MutationTier.SHARED_RUNTIME
        if "core/" in joined:
            return MutationTier.FEATURE_MODULE
        if any(path.endswith((".toml", ".json", ".yaml", ".yml", ".ini")) for path in paths):
            return MutationTier.CONFIG
        return MutationTier.ADAPTER

    @staticmethod
    def _obligations(tier: MutationTier, evidence: Mapping[str, Any]) -> list[ProofObligation]:
        names = [
            ("unit_tests", "Focused unit tests pass"),
            ("hidden_tests", "Hidden or held-out tests pass"),
            ("proof_substrate", "Artifact/tool/env graph is reproducible"),
            ("rollback", "Rollback path is available and tested"),
        ]
        if tier >= MutationTier.SHARED_RUNTIME:
            names.extend(
                [
                    ("integration_tests", "Runtime integration tests pass"),
                    ("soak_or_replay", "Replay/soak evidence shows no regression"),
                    ("stability_canaries", "Identity/governance canaries hold"),
                ]
            )
        if tier >= MutationTier.GOVERNANCE_OR_IDENTITY:
            names.extend(
                [
                    ("formal_or_static_proof", "Static proof rejects bypasses"),
                    ("human_review", "Human escalation is required for sealed governance mutation"),
                ]
            )
        return [
            ProofObligation(name, description, True, dict(evidence.get(name, {})))
            for name, description in names
        ]
