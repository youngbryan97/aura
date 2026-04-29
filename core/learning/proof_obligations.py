"""Proof obligations for self-modification.

Aura may propose broad changes, but promotion requires machine-checkable proof
obligations. Arbitrary self-modifications are rejected unless all required
obligations are discharged by formal verifier output and locked policy checks.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.self_modification.formal_verifier import VerificationResult, verify_mutation


class ProofStatus(str, Enum):
    PROVED = "PROVED"
    NOT_PROVEN = "NOT_PROVEN"
    BLOCKED_UNSAFE = "BLOCKED_UNSAFE"


@dataclass(frozen=True)
class ProofObligationResult:
    status: ProofStatus
    obligations: List[str]
    discharged: List[str]
    violations: List[str]
    certificate: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ProofStatus.PROVED

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class ProofObligationEngine:
    """Discharge structural proof obligations for a proposed source change."""

    BASE_OBLIGATIONS = [
        "source_parses",
        "public_surface_preserved",
        "async_preserved",
        "governance_boundary_preserved",
        "governance_fences_not_weakened",
        "protected_symbols_preserved",
        "unsafe_import_boundary_preserved",
    ]

    def prove_source_mutation(
        self,
        *,
        file_path: str,
        before_source: str,
        after_source: str,
        arbitrary_scope: bool = False,
    ) -> ProofObligationResult:
        verifier = verify_mutation(
            file_path=file_path,
            before_source=before_source,
            after_source=after_source,
            touches_tick_loop=("mind_tick" in file_path or "orchestrator" in file_path),
        )
        discharged = list(verifier.invariants_satisfied)
        violations = list(verifier.invariants_violated)
        obligations = list(self.BASE_OBLIGATIONS)

        if arbitrary_scope:
            obligations.extend([
                "termination_proof",
                "semantic_equivalence_or_improvement_proof",
                "resource_bound_proof",
                "identity_invariant_proof",
            ])
            violations.append("arbitrary_scope_requires_external_theorem_prover")

        status = ProofStatus.PROVED if verifier.ok and not arbitrary_scope else ProofStatus.NOT_PROVEN
        if any(
            item.startswith("protected_symbol_removed:")
            or item.startswith("unsafe_new_import:")
            or item.startswith("governance:")
            for item in violations
        ):
            status = ProofStatus.BLOCKED_UNSAFE

        return ProofObligationResult(
            status=status,
            obligations=obligations,
            discharged=discharged,
            violations=violations,
            certificate={
                "backend": verifier.backend,
                "diagnostics": verifier.diagnostics,
                "file_path": file_path,
            },
        )


__all__ = ["ProofObligationEngine", "ProofObligationResult", "ProofStatus"]
