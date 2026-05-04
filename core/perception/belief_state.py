"""Persistent environment belief state for embodied cognition.

This module is deliberately domain-agnostic. It tracks what Aura believes
about any live environment: entities, spatial/topological context, events,
uncertain hypotheses, deferred intentions, and contradictions. NetHack can
stress it, but the same substrate also fits a browser session, a desktop UI,
a robot arena, or a long movie/game/social environment.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .environment_parser import EnvironmentState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class BeliefHypothesis:
    """One possible interpretation of an unknown fact."""

    label: str
    probability: float
    evidence: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


@dataclass
class EntityKnowledge:
    """What we know about a specific entity or object in the world."""

    entity_id: str
    entity_type: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_known_pos: Optional[Tuple[int, int]] = None
    confidence: float = 0.7
    salience: float = 0.4
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefNode:
    """A standing belief with confidence, TTL, and provenance."""

    key: str
    value: Any
    confidence: float = 0.7
    source: str = "observation"
    evidence: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    ttl: float = 3600.0
    contested: bool = False

    @property
    def expired(self) -> bool:
        return (time.time() - self.updated_at) > self.ttl


@dataclass
class EventRecord:
    """A significant event that occurred in the environment."""

    event_type: str
    description: str
    timestamp: float = field(default_factory=time.time)
    salience: float = 0.5
    related_entities: List[str] = field(default_factory=list)
    location: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeferredIntention:
    """A prospective memory: recall this intention when conditions fit."""

    intention: str
    trigger: str
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


class EnvironmentBeliefState:
    """Persistent, queryable, uncertainty-aware world model.

    The class is intentionally lightweight: exact Bayesian updates are used
    where the state space is explicit, and confidence-weighted belief revision
    is used for open-ended facts. This gives Aura the bookkeeping NetHack
    demands without making the substrate specific to NetHack.
    """

    def __init__(self, session_id: str = "default", *, max_events: int = 5000):
        self.session_id = session_id
        self.max_events = int(max_events)

        self.spatial_memory: Dict[str, Dict[str, Any]] = {}
        self.entities: Dict[str, EntityKnowledge] = {}
        self.event_log: List[EventRecord] = []
        self.beliefs: Dict[str, BeliefNode] = {}
        self.hypotheses: Dict[str, Dict[str, BeliefHypothesis]] = {}
        self.deferred_intentions: List[DeferredIntention] = []
        self.prediction_errors: List[Dict[str, Any]] = []
        self.action_outcomes: List[Dict[str, Any]] = []

        self.current_context: str = "unknown"
        self.self_history: List[Dict[str, Any]] = []
        self.last_observation_id: str = ""

    def update_from_observation(self, state: EnvironmentState, context_id: str = "default") -> None:
        """Integrate a new observation into the belief state."""
        self.current_context = context_id or state.context_id or "default"
        state.context_id = self.current_context
        self.last_observation_id = state.observation_id

        context = self.spatial_memory.setdefault(
            self.current_context,
            {
                "explored_tiles": set(),
                "visible_tiles": set(),
                "points_of_interest": [],
                "danger_zones": [],
                "last_seen_at": time.time(),
            },
        )
        context["last_seen_at"] = time.time()

        if state.self_state:
            hist_entry = dict(state.self_state)
            hist_entry["observation_id"] = state.observation_id
            self.self_history.append(hist_entry)
            if len(self.self_history) > 2000:
                self.self_history = self.self_history[-2000:]
            for key, value in state.self_state.items():
                self.set_belief(
                    f"self.{key}",
                    value,
                    confidence=0.95,
                    source="self_state",
                    ttl=7200.0,
                )

        for ent in state.entities:
            ent_id = self._entity_id(ent)
            if ent_id not in self.entities:
                self.entities[ent_id] = EntityKnowledge(
                    entity_id=ent_id,
                    entity_type=str(ent.get("type", "unknown")),
                    confidence=float(ent.get("confidence", 0.7) or 0.7),
                )
                self.record_event(
                    f"New entity observed: {ent_id}",
                    "entity_observed",
                    salience=float(ent.get("salience", 0.4) or 0.4),
                    related_entities=[ent_id],
                )

            knowledge = self.entities[ent_id]
            knowledge.last_seen = time.time()
            knowledge.entity_type = str(ent.get("type", knowledge.entity_type))
            knowledge.salience = max(knowledge.salience, float(ent.get("salience", 0.4) or 0.4))
            if "pos" in ent and isinstance(ent["pos"], tuple):
                knowledge.last_known_pos = ent["pos"]
            elif "pos" in ent and isinstance(ent["pos"], list) and len(ent["pos"]) == 2:
                knowledge.last_known_pos = (int(ent["pos"][0]), int(ent["pos"][1]))

            for tag in ent.get("tags", []) or []:
                if tag not in knowledge.tags:
                    knowledge.tags.append(tag)
            for key, value in ent.items():
                if key not in {"id", "type", "pos", "tags"}:
                    knowledge.properties[key] = value

            if ent.get("unknown") or ent.get("known") is False:
                self.ensure_hypotheses(ent_id, ["unknown"], evidence="observed as uncertain")

        for msg in state.messages:
            if self._should_log_message(msg):
                self.record_event(msg, "system_message", salience=self._message_salience(msg))

        player_pos = state.spatial_info.get("player_pos")
        if player_pos is not None:
            context["explored_tiles"].add(tuple(player_pos))

        for tile in state.spatial_info.get("visible_tiles", []) or []:
            try:
                context["visible_tiles"].add(tuple(tile))
            except TypeError:
                continue

        for ent in state.entities:
            ent_type = ent.get("type")
            if ent_type in ("stairs", "item_or_feature", "trap", "hazard", "resource", "goal"):
                poi = {
                    "type": ent_type,
                    "pos": ent.get("pos"),
                    "glyph": ent.get("glyph"),
                    "label": ent.get("label"),
                    "last_seen": time.time(),
                }
                if not self._poi_exists(context["points_of_interest"], poi):
                    context["points_of_interest"].append(poi)

        self._evict_expired()

    def set_belief(
        self,
        key: str,
        value: Any,
        *,
        confidence: float = 0.7,
        source: str = "observation",
        evidence: Optional[Iterable[str]] = None,
        ttl: float = 3600.0,
    ) -> BeliefNode:
        existing = self.beliefs.get(key)
        confidence = _clamp01(confidence)
        new_evidence = list(evidence or [])
        contested = False
        if existing is not None and existing.value != value:
            contested = existing.confidence > 0.75 and confidence > 0.75
            merged_confidence = max(0.05, min(0.95, (existing.confidence + confidence) / 2.0))
            evidence_list = existing.evidence[-8:] + new_evidence
        else:
            merged_confidence = confidence if existing is None else max(existing.confidence, confidence)
            evidence_list = (existing.evidence[-8:] if existing else []) + new_evidence

        node = BeliefNode(
            key=key,
            value=value,
            confidence=merged_confidence,
            source=source,
            evidence=evidence_list[-12:],
            ttl=ttl,
            contested=contested,
        )
        self.beliefs[key] = node
        if contested:
            self.record_event(
                f"Belief contradiction detected for {key}",
                "belief_revision",
                salience=0.75,
                metadata={"old": existing.value if existing else None, "new": value},
            )
        return node

    def get_belief(self, key: str, default: Any = None) -> Any:
        node = self.beliefs.get(key)
        if node is None or node.expired:
            return default
        return node.value

    def ensure_hypotheses(
        self,
        subject: str,
        labels: Iterable[str],
        *,
        evidence: str = "",
    ) -> Dict[str, BeliefHypothesis]:
        labels = [str(label) for label in labels if str(label)]
        if not labels:
            labels = ["unknown"]
        existing = self.hypotheses.setdefault(subject, {})
        if existing:
            return existing
        p = 1.0 / len(labels)
        for label in labels:
            existing[label] = BeliefHypothesis(
                label=label,
                probability=p,
                evidence=[evidence] if evidence else [],
            )
        return existing

    def update_hypotheses(
        self,
        subject: str,
        likelihoods: Dict[str, float],
        *,
        evidence: str = "",
    ) -> Dict[str, BeliefHypothesis]:
        """Bayesian-style update for an explicit finite hypothesis set."""
        current = self.ensure_hypotheses(subject, likelihoods.keys(), evidence=evidence)
        for label in likelihoods:
            current.setdefault(label, BeliefHypothesis(label=label, probability=0.01))

        total = 0.0
        updated: Dict[str, float] = {}
        for label, hypothesis in current.items():
            likelihood = max(0.001, float(likelihoods.get(label, 0.05)))
            value = hypothesis.probability * likelihood
            updated[label] = value
            total += value
        if total <= 0:
            total = 1.0
        for label, value in updated.items():
            current[label].probability = value / total
            current[label].updated_at = time.time()
            if evidence:
                current[label].evidence.append(evidence)
                current[label].evidence = current[label].evidence[-8:]
        return current

    def epistemic_uncertainty(self, subject: Optional[str] = None) -> float:
        """Return entropy-normalized uncertainty across hypotheses."""
        if subject is not None:
            groups = [self.hypotheses.get(subject, {})]
        else:
            groups = list(self.hypotheses.values())
        entropies: List[float] = []
        for group in groups:
            probs = [max(0.0, h.probability) for h in group.values()]
            if len(probs) <= 1:
                continue
            total = sum(probs)
            if total <= 0:
                continue
            normalized = [p / total for p in probs]
            entropy = -sum(p * math.log(p, 2) for p in normalized if p > 0)
            entropies.append(entropy / math.log(len(normalized), 2))
        if not entropies:
            return 0.0
        return _clamp01(sum(entropies) / len(entropies))

    def remember_intention(
        self,
        intention: str,
        trigger: str,
        *,
        priority: float = 0.5,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeferredIntention:
        item = DeferredIntention(
            intention=intention,
            trigger=trigger,
            priority=_clamp01(priority),
            expires_at=time.time() + ttl if ttl else None,
            metadata=dict(metadata or {}),
        )
        self.deferred_intentions.append(item)
        return item

    def due_intentions(self, state: EnvironmentState) -> List[DeferredIntention]:
        text = " ".join(
            [
                state.domain,
                state.context_id,
                " ".join(state.messages),
                " ".join(state.entity_labels()),
                repr(state.self_state),
            ]
        ).lower()
        due = [
            item
            for item in self.deferred_intentions
            if not item.expired() and item.trigger.lower() in text
        ]
        return sorted(due, key=lambda item: item.priority, reverse=True)

    def record_action_outcome(
        self,
        action: str,
        *,
        expected: str = "",
        observed: str = "",
        success: Optional[bool] = None,
        surprise: float = 0.0,
    ) -> None:
        outcome = {
            "action": action,
            "expected": expected,
            "observed": observed,
            "success": success,
            "surprise": _clamp01(surprise),
            "timestamp": time.time(),
            "context": self.current_context,
        }
        self.action_outcomes.append(outcome)
        self.action_outcomes = self.action_outcomes[-1000:]
        if surprise >= 0.5:
            self.prediction_errors.append(outcome)
            self.prediction_errors = self.prediction_errors[-500:]

    def record_event(
        self,
        description: str,
        event_type: str = "event",
        *,
        salience: float = 0.5,
        related_entities: Optional[List[str]] = None,
        location: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EventRecord:
        event = EventRecord(
            event_type=event_type,
            description=str(description),
            salience=_clamp01(salience),
            related_entities=list(related_entities or []),
            location=location,
            metadata=dict(metadata or {}),
        )
        self.event_log.append(event)
        if len(self.event_log) > self.max_events:
            self.event_log = self.event_log[-self.max_events :]
        return event

    def get_strategic_summary(self) -> str:
        """Produce a compact summary for deliberative cognition."""
        lines = [f"CURRENT CONTEXT: {self.current_context}"]

        if self.self_history:
            latest = self.self_history[-1]
            lines.append("SELF BELIEF:")
            for key, value in latest.items():
                lines.append(f"  {key}: {value}")

        ctx_mem = self.spatial_memory.get(self.current_context, {})
        pois = ctx_mem.get("points_of_interest", [])
        if pois:
            lines.append(f"KNOWN FEATURES IN {self.current_context}:")
            for poi in pois[-12:]:
                lines.append(f"  - {poi.get('type')} at {poi.get('pos')} [{poi.get('glyph')}]")

        uncertain = [
            (subject, self.epistemic_uncertainty(subject))
            for subject in self.hypotheses
            if self.epistemic_uncertainty(subject) > 0.35
        ]
        if uncertain:
            lines.append("HIGH UNCERTAINTY:")
            for subject, score in sorted(uncertain, key=lambda item: item[1], reverse=True)[:8]:
                lines.append(f"  - {subject}: {score:.2f}")

        due = [i for i in self.deferred_intentions if not i.expired()]
        if due:
            lines.append("DEFERRED INTENTIONS:")
            for item in sorted(due, key=lambda i: i.priority, reverse=True)[:5]:
                lines.append(f"  - {item.intention} when {item.trigger} (priority {item.priority:.2f})")

        recent_events = sorted(
            [e for e in self.event_log if e.event_type == "system_message"],
            key=lambda e: (e.timestamp, e.salience),
        )[-5:]
        if recent_events:
            lines.append("RECENT EVENTS:")
            for event in recent_events:
                lines.append(f"  - {event.description}")

        if self.prediction_errors:
            lines.append(f"PREDICTION ERRORS: {len(self.prediction_errors[-20:])} recent surprise signals")

        return "\n".join(lines)

    def _entity_id(self, ent: Dict[str, Any]) -> str:
        if ent.get("id"):
            return str(ent["id"])
        pos = ent.get("pos")
        glyph = ent.get("glyph", "")
        label = ent.get("label") or ent.get("name") or ent.get("type", "unknown")
        if pos is not None and ent.get("type") in {"stairs", "trap", "item_or_feature", "resource", "goal"}:
            return f"{self.current_context}:{label}:{glyph}:{tuple(pos)}"
        return f"{label}:{glyph}:{pos if pos is not None else 'unlocated'}"

    @staticmethod
    def _message_salience(message: str) -> float:
        lowered = message.lower()
        high = ("die", "killed", "critical", "danger", "weak", "faint", "poison", "error")
        medium = ("hit", "hurt", "hungry", "confused", "blind", "warning", "unknown")
        if any(token in lowered for token in high):
            return 0.9
        if any(token in lowered for token in medium):
            return 0.65
        return 0.35

    @staticmethod
    def _should_log_message(message: str) -> bool:
        if not message:
            return False
        lowered = message.lower()
        prompt_noise = ("what do you want to", "--more--", "press return")
        return not any(token in lowered for token in prompt_noise)

    @staticmethod
    def _poi_exists(items: List[Dict[str, Any]], candidate: Dict[str, Any]) -> bool:
        return any(
            item.get("type") == candidate.get("type")
            and item.get("pos") == candidate.get("pos")
            and item.get("glyph") == candidate.get("glyph")
            for item in items
        )

    def _evict_expired(self) -> None:
        expired = [key for key, node in self.beliefs.items() if node.expired]
        for key in expired:
            del self.beliefs[key]
        self.deferred_intentions = [
            item for item in self.deferred_intentions if not item.expired()
        ]
