from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

from .unity_state import DraftBinding, ReconciledDraftSet


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _support_value(draft: Any) -> float:
    for key in ("coherence", "support", "confidence", "priority"):
        value = getattr(draft, key, None)
        if value is None and isinstance(draft, dict):
            value = draft.get(key)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except Exception:
                continue
    return 0.5


def _claim_value(draft: Any) -> str:
    for key in ("content", "claim", "text", "summary"):
        value = getattr(draft, key, None)
        if value is None and isinstance(draft, dict):
            value = draft.get(key)
        if value:
            return _normalize_text(value)
    return ""


def _draft_id(draft: Any, idx: int) -> str:
    value = getattr(draft, "draft_id", None)
    if value is None and isinstance(draft, dict):
        value = draft.get("draft_id")
    return str(value or f"draft_{idx}")


def _valence_value(draft: Any) -> float:
    value = getattr(draft, "valence", None)
    if value is None and isinstance(draft, dict):
        value = draft.get("valence")
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _text_distance(left: str, right: str) -> float:
    left_tokens = set(left.lower().split())
    right_tokens = set(right.lower().split())
    if not left_tokens or not right_tokens:
        return 0.0 if left == right else 1.0
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(0.0, min(1.0, 1.0 - overlap))


class DraftReconciliationEngine:
    """Preserve competing drafts instead of laundering them into one story."""

    def reconcile(
        self,
        drafts: Iterable[Any],
        *,
        fallback_claim: str = "",
    ) -> ReconciledDraftSet:
        raw_drafts = [item for item in list(drafts or []) if _claim_value(item)]
        if not raw_drafts:
            chosen = DraftBinding(
                draft_id="draft_default",
                claim=_normalize_text(fallback_claim) or "current interpretation",
                support=1.0,
                conflict=0.0,
                chosen=True,
            )
            return ReconciledDraftSet(chosen=chosen)

        extracted = []
        for idx, draft in enumerate(raw_drafts):
            extracted.append(
                {
                    "draft_id": _draft_id(draft, idx),
                    "claim": _claim_value(draft),
                    "support": _support_value(draft),
                    "valence": _valence_value(draft),
                }
            )

        contradiction_samples: list[float] = []
        for left, right in itertools.combinations(extracted, 2):
            text_distance = _text_distance(left["claim"], right["claim"])
            valence_delta = min(1.0, abs(float(left["valence"]) - float(right["valence"])) / 2.0)
            contradiction_samples.append((text_distance * 0.75) + (valence_delta * 0.25))
        contradiction_score = sum(contradiction_samples) / max(1, len(contradiction_samples))
        consensus_score = max(0.0, min(1.0, 1.0 - contradiction_score))

        scored = []
        for item in extracted:
            local_conflict = sum(
                _text_distance(item["claim"], other["claim"])
                for other in extracted
                if other["draft_id"] != item["draft_id"]
            ) / max(1, len(extracted) - 1)
            scored.append((item["support"] - (local_conflict * 0.35), local_conflict, item))
        scored.sort(key=lambda row: row[0], reverse=True)

        winner_local_conflict = float(scored[0][1])
        winner_item = scored[0][2]
        chosen = DraftBinding(
            draft_id=str(winner_item["draft_id"]),
            claim=str(winner_item["claim"]),
            support=round(float(winner_item["support"]), 4),
            conflict=round(winner_local_conflict, 4),
            chosen=True,
        )

        alternatives: list[DraftBinding] = []
        unresolved_residue: list[str] = []
        for _score, local_conflict, item in scored[1:]:
            suppressed_reason = "outcompeted by stronger support"
            if contradiction_score > 0.35:
                suppressed_reason = "preserved as conflicting alternative"
            alternatives.append(
                DraftBinding(
                    draft_id=str(item["draft_id"]),
                    claim=str(item["claim"]),
                    support=round(float(item["support"]), 4),
                    conflict=round(local_conflict, 4),
                    chosen=False,
                    suppressed_reason=suppressed_reason,
                )
            )
            if local_conflict > 0.35:
                unresolved_residue.append(str(item["claim"])[:160])

        if contradiction_score > 0.7:
            commit_mode = "defer"
        elif contradiction_score > 0.45:
            commit_mode = "conflicted"
        elif contradiction_score > 0.25:
            commit_mode = "qualified"
        else:
            commit_mode = "clean"

        return ReconciledDraftSet(
            chosen=chosen,
            alternatives=alternatives,
            consensus_score=round(consensus_score, 4),
            contradiction_score=round(contradiction_score, 4),
            unresolved_residue=unresolved_residue[:4],
            memory_commit_mode=commit_mode,
        )
