from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from typing import Any

from .unity_state import DraftBinding, ReconciledDraftSet

_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|without|refuse|refused|block|blocked|deny|denied|"
    r"avoid|stop|cannot|can't|won't|shouldn't|don't|do not)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PROPOSITION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "beneath",
    "but",
    "by",
    "can",
    "could",
    "for",
    "from",
    "has",
    "have",
    "in",
    "input",
    "interpretation",
    "intent",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "other",
    "response",
    "something",
    "surface",
    "that",
    "the",
    "this",
    "to",
    "under",
    "was",
    "were",
    "with",
}
_OPPOSITION_PAIRS = (
    ({"allow", "approve", "proceed", "publish", "push", "send"}, {"block", "deny", "refuse", "stop"}),
    ({"safe", "valid", "verified"}, {"danger", "dangerous", "invalid", "unsafe", "unverified"}),
    ({"increase", "raise"}, {"decrease", "lower", "reduce"}),
    ({"true", "yes"}, {"false", "no"}),
)


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
            except (TypeError, ValueError):
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
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0.0


def _stem_token(token: str) -> str:
    value = str(token or "").lower()
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) > len(suffix) + 3:
            if suffix == "ied":
                return value[:-3] + "y"
            return value[: -len(suffix)]
    return value


def _proposition_tokens(text: str) -> set[str]:
    return {
        stemmed
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if (stemmed := _stem_token(token))
        and stemmed not in _PROPOSITION_STOPWORDS
        and not stemmed.isdigit()
    }


def _contains_opposition(left_tokens: set[str], right_tokens: set[str]) -> bool:
    for positive, negative in _OPPOSITION_PAIRS:
        if (left_tokens & positive and right_tokens & negative) or (
            left_tokens & negative and right_tokens & positive
        ):
            return True
    return False


def _pair_contradiction(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Estimate logical opposition without treating mere diversity as conflict."""

    left_claim = str(left.get("claim") or "")
    right_claim = str(right.get("claim") or "")
    left_tokens = _proposition_tokens(left_claim)
    right_tokens = _proposition_tokens(right_claim)
    if not left_tokens or not right_tokens:
        return 0.0

    shared = left_tokens & right_tokens
    shared_ratio = len(shared) / max(1, min(len(left_tokens), len(right_tokens)))
    negation_mismatch = bool(_NEGATION_RE.search(left_claim)) != bool(
        _NEGATION_RE.search(right_claim)
    )
    explicit_opposition = _contains_opposition(left_tokens, right_tokens)
    valence_delta = min(
        1.0,
        abs(float(left.get("valence", 0.0)) - float(right.get("valence", 0.0))) / 2.0,
    )

    if negation_mismatch and shared_ratio >= 0.2:
        return min(1.0, 0.7 + (shared_ratio * 0.2) + (valence_delta * 0.1))
    if explicit_opposition and shared_ratio >= 0.1:
        return min(1.0, 0.65 + (shared_ratio * 0.2) + (valence_delta * 0.15))

    # Emotional coloration may create mild tension when drafts address the
    # same proposition, but it cannot alone make distinct perspectives a
    # logical contradiction.
    return min(0.2, shared_ratio * valence_delta * 0.2)


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

        pair_conflicts: dict[tuple[str, str], float] = {}
        contradiction_samples: list[float] = []
        for left, right in itertools.combinations(extracted, 2):
            conflict = _pair_contradiction(left, right)
            pair_conflicts[(str(left["draft_id"]), str(right["draft_id"]))] = conflict
            pair_conflicts[(str(right["draft_id"]), str(left["draft_id"]))] = conflict
            contradiction_samples.append(conflict)
        contradiction_score = sum(contradiction_samples) / max(1, len(contradiction_samples))
        consensus_score = max(0.0, min(1.0, 1.0 - contradiction_score))

        scored = []
        for item in extracted:
            local_conflict = sum(
                pair_conflicts.get(
                    (str(item["draft_id"]), str(other["draft_id"])),
                    0.0,
                )
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
