"""Deliberation receipts for external evidence Aura reads."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .schemas import stable_hash


@dataclass
class EvidenceDeliberation:
    receipt_id: str
    source_type: str
    source_ref: str
    summary: str
    claims: list[str]
    uncertainties: list[str]
    relevance: float
    recommended_memory_action: str
    next_questions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExternalEvidenceDeliberator:
    """Turns online/social/email artifacts into explicit reflection receipts."""

    def deliberate(
        self,
        *,
        source_type: str,
        source_ref: str,
        content: str,
        goal: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceDeliberation:
        text = re.sub(r"\s+", " ", str(content or "")).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 12]
        claims = self._claims(sentences)
        uncertainties = self._uncertainties(sentences, metadata or {})
        goal_terms = {tok for tok in re.findall(r"[a-zA-Z]{3,}", goal.lower())}
        text_terms = {tok for tok in re.findall(r"[a-zA-Z]{3,}", text.lower())}
        relevance = min(1.0, len(goal_terms & text_terms) / max(1, len(goal_terms)) + (0.25 if claims else 0.0))
        memory_action = "retain_with_provenance" if relevance >= 0.45 and claims else "do_not_retain_yet"
        next_questions = self._next_questions(goal, uncertainties)
        summary = " ".join(sentences[:2])[:600] if sentences else text[:600]
        receipt_id = stable_hash(
            {
                "source_type": source_type,
                "source_ref": source_ref,
                "summary": summary,
                "claims": claims[:6],
                "uncertainties": uncertainties[:6],
                "goal": goal,
            },
            prefix="delib_",
        )
        return EvidenceDeliberation(
            receipt_id=receipt_id,
            source_type=source_type,
            source_ref=source_ref,
            summary=summary,
            claims=claims[:8],
            uncertainties=uncertainties[:8],
            relevance=relevance,
            recommended_memory_action=memory_action,
            next_questions=next_questions,
        )

    @staticmethod
    def deliberate_many(
        artifacts: Sequence[Mapping[str, Any]],
        *,
        source_type: str,
        goal: str = "",
    ) -> list[dict[str, Any]]:
        engine = ExternalEvidenceDeliberator()
        receipts = []
        for idx, item in enumerate(artifacts):
            content = str(item.get("text") or item.get("content") or item.get("snippet") or item.get("body") or item)
            source_ref = str(item.get("url") or item.get("id") or item.get("uid") or idx)
            receipts.append(
                engine.deliberate(source_type=source_type, source_ref=source_ref, content=content, goal=goal, metadata=item).to_dict()
            )
        return receipts

    @staticmethod
    def _claims(sentences: list[str]) -> list[str]:
        hedges = ("might", "maybe", "possibly", "unclear", "unknown", "rumor")
        claims = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(h in lower for h in hedges):
                continue
            if any(v in lower for v in (" is ", " are ", " has ", " can ", " will ", " caused ", " shows ", " suggests ")):
                claims.append(sentence[:220])
        return claims

    @staticmethod
    def _uncertainties(sentences: list[str], metadata: Mapping[str, Any]) -> list[str]:
        uncertainties = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(h in lower for h in ("might", "maybe", "possibly", "unclear", "unknown", "unverified", "rumor", "claims")):
                uncertainties.append(sentence[:220])
        if not metadata.get("url") and not metadata.get("uid"):
            uncertainties.append("source identity/provenance is incomplete")
        return uncertainties

    @staticmethod
    def _next_questions(goal: str, uncertainties: list[str]) -> list[str]:
        if not uncertainties:
            return []
        prefix = "Verify"
        if goal:
            prefix = f"Verify for goal '{goal[:80]}'"
        return [f"{prefix}: {uncertainty[:120]}" for uncertainty in uncertainties[:3]]
