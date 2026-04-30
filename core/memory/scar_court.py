"""Immune tolerance review for behavioral scars.

The scar system is allowed to learn caution from recurring harm, but a single
ambiguous event must not become a permanent defensive attractor.  Scar Court is
the deterministic review layer that decides whether a scar remains provisional,
is reduced, or is mature enough to influence persistent learning.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

_COURT_DIR = Path.home() / ".aura" / "data" / "scars"
_COURT_LEDGER = _COURT_DIR / "scar_court.jsonl"


@dataclass(frozen=True)
class ScarCourtDecision:
    scar_id: str
    status: str
    confidence: float
    latent_influence_cap: float
    appeal_status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    reversible: bool = True

    @property
    def consolidated(self) -> bool:
        return self.status == "consolidated"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        data["consolidated"] = self.consolidated
        return data


class ScarCourt:
    """Deterministic tolerance gate for new and reinforced scars."""

    MIN_EVIDENCE = 3
    MIN_SOURCE_DIVERSITY = 2
    MIN_CONFIDENCE = 0.70
    PROVISIONAL_CAP = 0.15
    REDUCED_CAP = 0.07
    CONSOLIDATED_CAP = 0.45

    BENIGN_TERMS = (
        "permission",
        "accessibility",
        "timeout",
        "network",
        "transient",
        "screen",
        "sleep",
        "resource",
        "rate limit",
        "user cancelled",
        "not available",
    )

    def assess(
        self,
        *,
        scar_id: str,
        description: str,
        evidence_count: int,
        source_diversity: int,
        confidence: float,
        verified_threat: bool,
        benign_alternatives: Iterable[str] = (),
    ) -> ScarCourtDecision:
        reasons: list[str] = []
        alternatives = tuple(str(item) for item in benign_alternatives if str(item).strip())
        inferred_benign = self.search_benign_alternatives(description)
        if inferred_benign:
            alternatives = tuple(dict.fromkeys((*alternatives, *inferred_benign)))

        if evidence_count < self.MIN_EVIDENCE:
            reasons.append(f"evidence_count<{self.MIN_EVIDENCE}")
        if source_diversity < self.MIN_SOURCE_DIVERSITY:
            reasons.append(f"source_diversity<{self.MIN_SOURCE_DIVERSITY}")
        if confidence < self.MIN_CONFIDENCE:
            reasons.append(f"confidence<{self.MIN_CONFIDENCE:.2f}")
        if alternatives and not verified_threat:
            reasons.append("benign_alternative_present")

        if verified_threat and not reasons:
            return self._record(
                ScarCourtDecision(
                    scar_id=scar_id,
                    status="consolidated",
                    confidence=min(1.0, max(0.0, confidence)),
                    latent_influence_cap=self.CONSOLIDATED_CAP,
                    appeal_status="upheld",
                    reasons=("recurring_verified_threat",),
                    reversible=True,
                )
            )

        if evidence_count >= self.MIN_EVIDENCE and source_diversity >= self.MIN_SOURCE_DIVERSITY and confidence >= self.MIN_CONFIDENCE:
            return self._record(
                ScarCourtDecision(
                    scar_id=scar_id,
                    status="trial",
                    confidence=min(1.0, max(0.0, confidence)),
                    latent_influence_cap=0.25,
                    appeal_status="unreviewed",
                    reasons=tuple(reasons or ("awaiting_maturity_review",)),
                    reversible=True,
                )
            )

        cap = self.REDUCED_CAP if alternatives else self.PROVISIONAL_CAP
        status = "reduced" if alternatives else "provisional"
        return self._record(
            ScarCourtDecision(
                scar_id=scar_id,
                status=status,
                confidence=min(1.0, max(0.0, confidence)),
                latent_influence_cap=cap,
                appeal_status="unreviewed",
                reasons=tuple(reasons or ("single_event_guard",)),
                reversible=True,
            )
        )

    def search_benign_alternatives(self, description: str) -> tuple[str, ...]:
        text = str(description or "").lower()
        matches = [term for term in self.BENIGN_TERMS if term in text]
        return tuple(f"possible_{term.replace(' ', '_')}" for term in matches)

    def eligible_for_lora_consolidation(self, scar: Any) -> bool:
        return (
            getattr(scar, "maturity_status", "") == "consolidated"
            and getattr(scar, "appeal_status", "") == "upheld"
            and int(getattr(scar, "evidence_count", 0)) >= self.MIN_EVIDENCE
            and int(getattr(scar, "source_diversity", 0)) >= self.MIN_SOURCE_DIVERSITY
            and float(getattr(scar, "confidence", 0.0)) >= self.MIN_CONFIDENCE
        )

    def _record(self, decision: ScarCourtDecision) -> ScarCourtDecision:
        try:
            _COURT_DIR.mkdir(parents=True, exist_ok=True)
            entry = {"when": time.time(), **decision.to_dict()}
            with open(_COURT_LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        except Exception as exc:
            record_degradation("scar_court", exc)
        return decision

    def snapshot_policy(self) -> dict[str, Any]:
        return {
            "min_evidence": self.MIN_EVIDENCE,
            "min_source_diversity": self.MIN_SOURCE_DIVERSITY,
            "min_confidence": self.MIN_CONFIDENCE,
            "provisional_cap": self.PROVISIONAL_CAP,
            "reduced_cap": self.REDUCED_CAP,
            "consolidated_cap": self.CONSOLIDATED_CAP,
        }


_instance: ScarCourt | None = None


def get_scar_court() -> ScarCourt:
    global _instance
    if _instance is None:
        _instance = ScarCourt()
    return _instance


def write_scar_court_policy(path: str | Path) -> dict[str, Any]:
    policy = get_scar_court().snapshot_policy()
    atomic_write_text(Path(path), json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    return policy


__all__ = ["ScarCourt", "ScarCourtDecision", "get_scar_court", "write_scar_court_policy"]
