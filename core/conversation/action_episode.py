"""Typed conversation memory for actions Aura actually attempted.

An action receipt is authoritative for the turn that produced it. This module
reduces that receipt to a bounded, non-sensitive episode that can remain
attached to the same conversation exchange and ground later questions about
what happened.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from core.language.action_outcome import action_outcome_question

__all__ = [
    "ActionEpisode",
    "action_episode_from_execution",
    "action_episode_grounding",
    "is_action_episode_question",
    "select_action_episode",
]

_MAX_OBJECTIVE_CHARS = 500
_MAX_DETAIL_CHARS = 600
_MAX_SUMMARY_CHARS = 800
_MAX_EVIDENCE_REFS = 12
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RETRIEVAL_STOPWORDS = frozenset(
    {
        "a", "about", "an", "and", "can", "could", "did", "do", "does",
        "explain", "for", "how", "i", "it", "me", "of", "please", "that",
        "the", "there", "this", "to", "was", "what", "why", "you",
    }
)


def _text(value: Any, *, limit: int) -> str:
    try:
        text = " ".join(str(value or "").split())
    except (RuntimeError, TypeError, ValueError):
        return ""
    return text[:limit]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _first_detail(*containers: Mapping[str, Any]) -> tuple[str, str]:
    for key in ("error", "failure_reason", "blocked_reason", "reason"):
        for container in containers:
            value = _text(container.get(key), limit=_MAX_DETAIL_CHARS)
            if value:
                return key, value
    for container in containers:
        value = _text(container.get("status"), limit=120)
        if value and value.casefold() not in {"failed", "error", "false"}:
            return "status", value
    return "", ""


def _receipt_refs(result: Mapping[str, Any]) -> tuple[str, ...]:
    refs: list[str] = []
    receipts = result.get("receipts")
    if not isinstance(receipts, list):
        return ()
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        for key in (
            "receipt_id", "tool_receipt_id", "governance_receipt_id",
            "action_id", "execution_id",
        ):
            value = _text(receipt.get(key), limit=160)
            if value and value not in refs:
                refs.append(value)
                break
        if len(refs) >= _MAX_EVIDENCE_REFS:
            break
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class ActionEpisode:
    """A bounded causal account of one attempted action."""

    objective: str
    capability: str
    status: str
    succeeded: bool
    failure_kind: str = ""
    failure_detail: str = ""
    summary: str = ""
    steps_completed: int = 0
    steps_requested: int = 0
    evidence_refs: tuple[str, ...] = ()
    recorded_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> ActionEpisode | None:
        if not isinstance(value, Mapping):
            return None
        objective = _text(value.get("objective"), limit=_MAX_OBJECTIVE_CHARS)
        capability = _text(value.get("capability"), limit=120)
        status = _text(value.get("status"), limit=160)
        if not objective or not capability or not status:
            return None
        refs = value.get("evidence_refs")
        evidence_refs = tuple(
            text
            for item in (refs if isinstance(refs, (list, tuple)) else ())
            if (text := _text(item, limit=160))
        )[:_MAX_EVIDENCE_REFS]
        try:
            recorded_at = float(value.get("recorded_at") or 0.0)
        except (TypeError, ValueError):
            recorded_at = 0.0
        return cls(
            objective=objective,
            capability=capability,
            status=status,
            succeeded=bool(value.get("succeeded")),
            failure_kind=_text(value.get("failure_kind"), limit=80),
            failure_detail=_text(value.get("failure_detail"), limit=_MAX_DETAIL_CHARS),
            summary=_text(value.get("summary"), limit=_MAX_SUMMARY_CHARS),
            steps_completed=_nonnegative_int(value.get("steps_completed")),
            steps_requested=_nonnegative_int(value.get("steps_requested")),
            evidence_refs=evidence_refs,
            recorded_at=recorded_at,
        )


def action_episode_from_execution(
    objective: str,
    execution: Any,
    *,
    capability: str,
) -> ActionEpisode | None:
    """Build an episode from a governed capability result."""

    outer = _mapping(execution)
    if not outer:
        return None
    result = _mapping(outer.get("result"))
    succeeded = bool(outer.get("ok"))
    status = _text(
        outer.get("status") or result.get("status") or ("completed" if succeeded else "failed"),
        limit=160,
    )
    failure_kind, failure_detail = ("", "")
    if not succeeded:
        failure_kind, failure_detail = _first_detail(outer, result)
    return ActionEpisode(
        objective=_text(objective, limit=_MAX_OBJECTIVE_CHARS),
        capability=_text(capability, limit=120),
        status=status,
        succeeded=succeeded,
        failure_kind=failure_kind,
        failure_detail=failure_detail,
        summary=_text(
            outer.get("response") or result.get("summary"),
            limit=_MAX_SUMMARY_CHARS,
        ),
        steps_completed=_nonnegative_int(result.get("steps_completed")),
        steps_requested=_nonnegative_int(result.get("steps_requested")),
        evidence_refs=_receipt_refs(result),
        recorded_at=time.time(),
    )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text or "").casefold())
        if len(token) > 1 and token not in _RETRIEVAL_STOPWORDS
    }


def select_action_episode(
    question: str,
    episodes: Iterable[ActionEpisode],
) -> ActionEpisode | None:
    """Select the recent action episode referred to by an outcome question."""

    relation = action_outcome_question(question)
    if not relation.asks_about_outcome:
        return None
    candidates = list(episodes)
    if relation.asks_about_failure:
        failed = [episode for episode in candidates if not episode.succeeded]
        if failed:
            candidates = failed
    if not candidates:
        return None
    question_terms = _tokens(question)
    scored: list[tuple[int, int, ActionEpisode]] = []
    for recency, episode in enumerate(candidates):
        episode_terms = _tokens(f"{episode.objective} {episode.capability} {episode.status}")
        scored.append((len(question_terms & episode_terms), recency, episode))
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def is_action_episode_question(text: str) -> bool:
    """Whether ``text`` asks about the outcome of an earlier action."""

    return action_outcome_question(text).asks_about_outcome


def action_episode_grounding(episode: ActionEpisode) -> str:
    """Serialize verified episode facts for the conversational model."""

    lines = [
        "[PRIOR ACTION EPISODE]",
        f"objective: {episode.objective}",
        f"capability: {episode.capability}",
        f"status: {episode.status}",
        f"succeeded: {'true' if episode.succeeded else 'false'}",
    ]
    if episode.failure_kind:
        lines.append(f"failure_field: {episode.failure_kind}")
    if episode.failure_detail:
        lines.append(f"failure_detail: {episode.failure_detail}")
    if episode.steps_requested:
        lines.append(f"steps: {episode.steps_completed}/{episode.steps_requested} completed")
    if episode.evidence_refs:
        lines.append(f"evidence_refs: {', '.join(episode.evidence_refs)}")
    lines.append("[END PRIOR ACTION EPISODE]")
    return "\n".join(lines)
