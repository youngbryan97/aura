"""Shared, state-grounded topic selection for autonomous exploration.

This module deliberately does not invent a fixed menu of things Aura is
allowed to care about.  It ranks live signals from conversation, unresolved
tensions, goals, durable user context, the knowledge graph, and motivation.
The selected topic still has to pass the normal initiative and tool-governance
paths before it can cause an external action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from core.autonomy.research_goal_filter import is_unresearchable_goal


_LOW_INFORMATION = {
    "hello",
    "hey",
    "hi",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "continue",
    "yes",
    "no",
}


@dataclass(frozen=True)
class AutonomousTopic:
    text: str
    source: str
    reason: str
    score: float


def _clean_topic(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split()).strip(" -:;,.?!")
    if not text or text.lower() in _LOW_INFORMATION:
        return ""
    if is_unresearchable_goal(text):
        return ""
    text = re.sub(
        r"^(?:please\s+)?(?:research|explore|investigate|learn about|look into|find out about)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" -:;,.?!")
    if is_unresearchable_goal(text):
        return ""
    if len(text) < 8:
        return ""
    return text[:limit].rstrip()


def conversation_topic(text: Any) -> str:
    """Return a bounded substantive topic from ordinary conversation text."""

    cleaned = _clean_topic(text)
    if not cleaned:
        return ""
    # Prefer the first complete clause. Long task descriptions and transcripts
    # are poor autonomous search queries when treated as one giant string.
    clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
    substantive = [part for part in clauses if len(part.split()) >= 3]
    return _clean_topic(substantive[0] if substantive else cleaned, limit=140)


def _append(
    output: list[AutonomousTopic],
    seen: set[str],
    value: Any,
    *,
    source: str,
    reason: str,
    score: float,
) -> None:
    text = _clean_topic(value)
    fingerprint = text.casefold()
    if not text or fingerprint in seen:
        return
    seen.add(fingerprint)
    output.append(AutonomousTopic(text=text, source=source, reason=reason, score=score))


def _iter_values(items: Iterable[Any], *keys: str) -> Iterable[str]:
    for item in items:
        if isinstance(item, dict):
            for key in keys:
                value = item.get(key)
                if value:
                    yield str(value)
                    break
        elif item:
            yield str(item)


def collect_autonomous_topics(
    orchestrator: Any,
    state: Any | None = None,
    *,
    limit: int = 24,
) -> list[AutonomousTopic]:
    """Collect ranked exploration candidates from live canonical state.

    Failures are intentionally local: topic selection is advisory and must not
    destabilize the cognition loop. Callers expose an honest empty result when
    no grounded source is available instead of substituting a scripted topic.
    """

    from core.container import ServiceContainer

    output: list[AutonomousTopic] = []
    seen: set[str] = set()
    cognition = getattr(state, "cognition", None)

    for message in reversed(list(getattr(cognition, "working_memory", []) or [])[-16:]):
        if not isinstance(message, dict) or str(message.get("role", "")).lower() != "user":
            continue
        topic = conversation_topic(message.get("content"))
        _append(
            output,
            seen,
            topic,
            source="conversation",
            reason="substantive topic from recent shared context",
            score=0.96,
        )

    for initiative in list(getattr(cognition, "pending_initiatives", []) or []):
        _append(
            output,
            seen,
            initiative.get("goal") if isinstance(initiative, dict) else initiative,
            source="initiative",
            reason="unresolved canonical initiative",
            score=0.94,
        )

    synthesizer = ServiceContainer.get("initiative_synthesizer", default=None)
    if synthesizer is not None and hasattr(synthesizer, "get_tensions"):
        try:
            tensions = list(synthesizer.get_tensions() or [])
        except (AttributeError, RuntimeError, TypeError, ValueError):
            tensions = []
        for value in _iter_values(tensions, "content"):
            _append(
                output,
                seen,
                value,
                source="unresolved_tension",
                reason="unresolved question or stalled thread",
                score=0.91,
            )

    goal_engine = ServiceContainer.get("goal_engine", default=None)
    if goal_engine is not None and hasattr(goal_engine, "get_active_goals"):
        try:
            goals = list(goal_engine.get_active_goals(limit=8) or [])
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            goals = []
        for value in _iter_values(goals, "objective", "name", "summary"):
            _append(
                output,
                seen,
                value,
                source="goal",
                reason="active durable goal",
                score=0.87,
            )

    identity = ServiceContainer.get("identity", default=None)
    identity_state = getattr(identity, "state", None)
    for value in _iter_values(list(getattr(identity_state, "long_term_goals", []) or []), "objective", "name", "goal"):
        _append(
            output,
            seen,
            value,
            source="identity_goal",
            reason="identity-level long-term goal",
            score=0.84,
        )

    knowledge_graph = (
        ServiceContainer.get("knowledge_graph", default=None)
        or getattr(orchestrator, "knowledge_graph", None)
    )
    if knowledge_graph is not None:
        try:
            sparse = list(knowledge_graph.get_sparse_nodes(limit=8) or [])
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            sparse = []
        for value in _iter_values(sparse, "content", "label", "name"):
            _append(
                output,
                seen,
                value,
                source="knowledge_gap",
                reason="sparse region of persistent knowledge",
                score=0.82,
            )
        try:
            interests = list(knowledge_graph.get_recent_nodes(limit=8, type="interest") or [])
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            interests = []
        for value in _iter_values(interests, "content", "label", "name"):
            _append(
                output,
                seen,
                value,
                source="persistent_interest",
                reason="persisted interest with recent relevance",
                score=0.79,
            )

    motivation = getattr(state, "motivation", None)
    live_interests = list(getattr(motivation, "latent_interests", []) or [])
    if not live_interests:
        drive_engine = ServiceContainer.get("drive_engine", default=None)
        live_interests = list(getattr(drive_engine, "latent_interests", []) or [])
    for value in live_interests:
        _append(
            output,
            seen,
            value,
            source="latent_interest",
            reason="persisted motivational interest",
            score=0.68,
        )

    output.sort(key=lambda item: item.score, reverse=True)
    return output[: max(1, int(limit))]


def select_autonomous_topic(
    orchestrator: Any,
    state: Any | None = None,
    *,
    excluded: Iterable[str] = (),
) -> AutonomousTopic | None:
    excluded_set = {str(item).casefold() for item in excluded if str(item).strip()}
    for candidate in collect_autonomous_topics(orchestrator, state):
        if candidate.text.casefold() not in excluded_set:
            return candidate
    return None
