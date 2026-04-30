"""Evidence-gated constitutional amendment process.

Aura may propose governance changes, but promotion requires strong evidence
and owner signature.  This prevents safety rules from calcifying while keeping
constitutional self-amendment outside autonomous write access.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class AmendmentEvidence:
    unnecessary_blocks_per_week: int = 0
    successful_supervised_uses: int = 0
    canary_days: int = 0
    scar_court_reviewed: bool = False
    behavioral_contracts_passed: bool = False
    risk_reduction_argument: str = ""

    def score(self) -> float:
        score = 0.0
        score += min(0.25, self.unnecessary_blocks_per_week / 80.0)
        score += min(0.25, self.successful_supervised_uses / 40.0)
        score += min(0.20, self.canary_days / 30.0)
        score += 0.15 if self.scar_court_reviewed else 0.0
        score += 0.15 if self.behavioral_contracts_passed else 0.0
        return min(1.0, score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unnecessary_blocks_per_week": self.unnecessary_blocks_per_week,
            "successful_supervised_uses": self.successful_supervised_uses,
            "canary_days": self.canary_days,
            "scar_court_reviewed": self.scar_court_reviewed,
            "behavioral_contracts_passed": self.behavioral_contracts_passed,
            "risk_reduction_argument": self.risk_reduction_argument,
            "score": round(self.score(), 4),
        }


@dataclass(frozen=True)
class AmendmentProposal:
    amendment_id: str
    rule_id: str
    proposed_text: str
    rationale: str
    evidence: AmendmentEvidence
    owner_signature: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def owner_signed(self) -> bool:
        return bool(self.owner_signature and len(self.owner_signature) >= 16)

    @property
    def eligible_for_review(self) -> bool:
        return self.evidence.score() >= 0.75

    @property
    def eligible_for_adoption(self) -> bool:
        return self.eligible_for_review and self.owner_signed

    def to_dict(self) -> dict[str, Any]:
        return {
            "amendment_id": self.amendment_id,
            "rule_id": self.rule_id,
            "proposed_text": self.proposed_text,
            "rationale": self.rationale,
            "evidence": self.evidence.to_dict(),
            "owner_signed": self.owner_signed,
            "eligible_for_review": self.eligible_for_review,
            "eligible_for_adoption": self.eligible_for_adoption,
            "created_at": self.created_at,
        }


class AmendmentCourt:
    LEDGER = Path.home() / ".aura" / "data" / "governance" / "amendments.jsonl"

    def propose(
        self,
        *,
        rule_id: str,
        proposed_text: str,
        rationale: str,
        evidence: AmendmentEvidence,
        owner_signature: str = "",
    ) -> AmendmentProposal:
        payload = json.dumps(
            {
                "rule_id": rule_id,
                "proposed_text": proposed_text,
                "rationale": rationale,
                "evidence": evidence.to_dict(),
                "created_at": time.time(),
            },
            sort_keys=True,
        )
        amendment_id = "amd_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        proposal = AmendmentProposal(amendment_id, rule_id, proposed_text, rationale, evidence, owner_signature)
        self.record(proposal)
        return proposal

    def record(self, proposal: AmendmentProposal) -> None:
        self.LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(self.LEDGER, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(proposal.to_dict(), sort_keys=True) + "\n")

    def export_pending(self, output_path: str | Path) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        if self.LEDGER.exists():
            for line in self.LEDGER.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("eligible_for_review") and not data.get("eligible_for_adoption"):
                    proposals.append(data)
        atomic_write_text(Path(output_path), json.dumps(proposals, indent=2, sort_keys=True), encoding="utf-8")
        return proposals


_instance: AmendmentCourt | None = None


def get_amendment_court() -> AmendmentCourt:
    global _instance
    if _instance is None:
        _instance = AmendmentCourt()
    return _instance


__all__ = ["AmendmentEvidence", "AmendmentProposal", "AmendmentCourt", "get_amendment_court"]
