"""Exact-agent observer context backed by calibrated social evidence.

The historical module fabricated three nested minds from one caller-supplied
trust value. This compatibility surface retains the useful, causal capability:
an ephemeral account of who is actively interacting and a privacy posture that
suppresses unrelated private/background activity while a person is present.
It does not claim recursive beliefs without recursive evidence.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_DEPTH = 0
BIAS_DIM = 16
OBSERVER_DECAY_S = 60.0

ACTION_CATEGORIES: tuple[str, ...] = (
    "rest",
    "explore",
    "engage_social",
    "consolidate",
    "tool_use",
    "pattern_match",
    "self_inspect",
    "approach_other",
    "withdraw",
    "attend_body",
    "dream",
    "persist_goal",
    "revise_goal",
    "rehearse_memory",
    "emit_narrative",
    "pause",
)
FOREGROUND_ACTIONS = {"engage_social", "emit_narrative", "pause"}
BACKGROUND_PRIVATE_ACTIONS = {
    "self_inspect",
    "dream",
    "revise_goal",
    "rehearse_memory",
}
# Compatibility names; tool use is intentionally absent from foreground boosts.
PUBLIC_ACTIONS = FOREGROUND_ACTIONS
PRIVATE_ACTIONS = BACKGROUND_PRIVATE_ACTIONS
_OBSERVATION_KINDS = {
    "conversation_turn",
    "paired_device_turn",
    "voice_session",
    "explicit_presence",
}
_AFFECT_SIGNALS = {"frustration", "urgency", "fatigue", "uncertainty"}


def _agent_id(value: Any) -> str:
    normalized = " ".join(str(value or "").strip().split())[:160]
    if not normalized:
        raise ValueError("observer context requires an exact non-empty agent_id")
    return normalized


def _digest(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError("observer evidence_digest must be a SHA-256 hex digest")
    return normalized


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: Any, default: float = 0.0) -> float:
    return min(1.0, max(0.0, _number(value, default)))


@dataclass(frozen=True)
class MindSnapshot:
    """Bounded depth-zero projection from the canonical social estimator."""

    agent_id: str
    depth: int
    confidence: float
    observations: int
    social_rupture_risk: float
    evidence_digest: str
    affect_hypotheses: dict[str, dict[str, float]]
    captured_at: float
    hypothesis: bool = True
    nested: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "depth": self.depth,
            "confidence": round(self.confidence, 3),
            "observations": self.observations,
            "social_rupture_risk": round(self.social_rupture_risk, 3),
            "evidence_digest": self.evidence_digest,
            "affect_hypotheses": self.affect_hypotheses,
            "captured_at": self.captured_at,
            "hypothesis": True,
            "recursive_claimed": False,
        }

    def depth_reached(self) -> int:
        return 0


@dataclass(frozen=True)
class ObservationEvent:
    agent_id: str
    ts: float
    kind: str
    strength: float
    evidence_digest: str


@dataclass(frozen=True)
class BiasProfile:
    """Privacy posture bias. Positive entries prioritize the active turn."""

    bias: np.ndarray
    total_observer_presence: float
    active_observers: list[str]
    privacy_posture: str
    ts: float = field(default_factory=lambda: time.time())


class ObserverContextModel:
    """Tracks digest-backed presence and canonical depth-zero hypotheses."""

    def __init__(self, max_depth: int = MAX_DEPTH) -> None:
        if max_depth != 0:
            raise ValueError("recursive depth requires explicit nested evidence support")
        self._lock = threading.RLock()
        self._snapshots: dict[str, MindSnapshot] = {}
        self._observations: deque[ObservationEvent] = deque(maxlen=512)
        self._last_presence: dict[str, float] = {}

    def observe_agent(
        self,
        agent_id: str,
        *,
        kind: str = "conversation_turn",
        strength: float = 0.6,
        evidence_digest: str,
        observed_at: float | None = None,
    ) -> bool:
        exact_id = _agent_id(agent_id)
        normalized_kind = " ".join(str(kind or "").strip().split())[:40]
        if normalized_kind not in _OBSERVATION_KINDS:
            raise ValueError("observer kind is not an accepted presence source")
        normalized_digest = _digest(evidence_digest)
        bounded_strength = _clamp(strength)
        if bounded_strength <= 0.0:
            return False
        now = time.time()
        timestamp = _number(now if observed_at is None else observed_at)
        if timestamp <= 0.0 or timestamp > now + 5.0:
            raise ValueError("observer timestamp must be finite, positive, and current")
        with self._lock:
            if any(
                event.agent_id == exact_id
                and event.evidence_digest == normalized_digest
                for event in self._observations
            ):
                return False
            self._observations.append(
                ObservationEvent(
                    agent_id=exact_id,
                    ts=timestamp,
                    kind=normalized_kind,
                    strength=bounded_strength,
                    evidence_digest=normalized_digest,
                )
            )
            self._refresh_presence_locked(exact_id, now=timestamp)
        return True

    def register_interaction(
        self,
        agent_id: str,
        snapshot: dict[str, Any],
    ) -> MindSnapshot:
        """Accept one calibrated estimator snapshot without inventing nesting."""
        exact_id = _agent_id(agent_id)
        if not isinstance(snapshot, dict) or snapshot.get("agent_id") != exact_id:
            raise ValueError("social snapshot must match the exact agent_id")
        evidence_digest = _digest(snapshot.get("evidence_digest"))
        affect = snapshot.get("affect_hypotheses")
        affect = affect if isinstance(affect, dict) else {}
        sanitized_affect: dict[str, dict[str, float]] = {}
        for name in sorted(_AFFECT_SIGNALS):
            raw = affect.get(name)
            if not isinstance(raw, dict):
                continue
            confidence = _clamp(raw.get("confidence"))
            if confidence <= 0.0:
                continue
            sanitized_affect[name] = {
                "value": round(_clamp(raw.get("value")), 4),
                "confidence": round(confidence, 4),
            }
        now = time.time()
        captured_at = _number(snapshot.get("at"), now)
        if captured_at <= 0.0 or captured_at > now + 5.0:
            captured_at = now
        projection = MindSnapshot(
            agent_id=exact_id,
            depth=0,
            confidence=_clamp(snapshot.get("confidence")),
            observations=min(1_000_000, max(0, int(_number(snapshot.get("observations"))))),
            social_rupture_risk=_clamp(snapshot.get("social_rupture_risk")),
            evidence_digest=evidence_digest,
            affect_hypotheses=sanitized_affect,
            captured_at=captured_at,
        )
        with self._lock:
            current = self._snapshots.get(exact_id)
            if current is None or current.evidence_digest != evidence_digest:
                self._snapshots[exact_id] = projection
                return projection
            return current

    def get_mind(self, agent_id: str) -> MindSnapshot | None:
        with self._lock:
            return self._snapshots.get(_agent_id(agent_id))

    def get_mind_at_depth(self, agent_id: str, depth: int) -> MindSnapshot | None:
        if depth != 0:
            return None
        return self.get_mind(agent_id)

    def depth_reached(self, agent_id: str) -> int:
        return 0 if self.get_mind(agent_id) is not None else -1

    def forget_agent(self, agent_id: str) -> None:
        exact_id = _agent_id(agent_id)
        with self._lock:
            self._snapshots.pop(exact_id, None)
            self._last_presence.pop(exact_id, None)
            self._observations = deque(
                (
                    event
                    for event in self._observations
                    if event.agent_id != exact_id
                ),
                maxlen=512,
            )

    def _refresh_presence_locked(self, agent_id: str, *, now: float | None = None) -> None:
        timestamp = time.time() if now is None else now
        score = 0.0
        for event in self._observations:
            if event.agent_id != agent_id:
                continue
            elapsed = max(0.0, timestamp - event.ts)
            score += event.strength * max(0.0, 1.0 - elapsed / OBSERVER_DECAY_S)
        self._last_presence[agent_id] = min(1.0, score)

    def active_observers(self, threshold: float = 0.15) -> list[tuple[str, float]]:
        bounded_threshold = _clamp(threshold)
        with self._lock:
            seen = {event.agent_id for event in self._observations}
            for agent_id in seen:
                self._refresh_presence_locked(agent_id)
            return sorted(
                (
                    (agent_id, presence)
                    for agent_id, presence in self._last_presence.items()
                    if presence >= bounded_threshold
                ),
                key=lambda item: (-item[1], item[0]),
            )

    def total_observer_presence(self) -> float:
        active = self.active_observers(threshold=0.05)
        return min(1.0, float(np.tanh(sum(value for _, value in active))))

    def get_observer_bias(self) -> BiasProfile:
        bias: np.ndarray = np.zeros(BIAS_DIM, dtype=np.float32)
        active = self.active_observers(threshold=0.15)
        if not active:
            return BiasProfile(
                bias=bias,
                total_observer_presence=0.0,
                active_observers=[],
                privacy_posture="background",
            )
        presence = self.total_observer_presence()
        for index, category in enumerate(ACTION_CATEGORIES):
            if category in FOREGROUND_ACTIONS:
                bias[index] += 0.45 * presence
            elif category in BACKGROUND_PRIVATE_ACTIONS:
                bias[index] -= 0.8 * presence
        return BiasProfile(
            bias=np.tanh(bias).astype(np.float32),
            total_observer_presence=presence,
            active_observers=[agent_id for agent_id, _ in active],
            privacy_posture="interactive",
        )

    def should_defer_background_action(self, action_category: str) -> bool:
        return (
            action_category in BACKGROUND_PRIVATE_ACTIONS
            and self.total_observer_presence() >= 0.15
        )

    def get_status(self) -> dict[str, Any]:
        active = self.active_observers(threshold=0.1)
        with self._lock:
            return {
                "model": "observer_context",
                "recursive_mind_claims": False,
                "max_supported_depth": 0,
                "snapshot_count": len(self._snapshots),
                "total_observer_presence": round(self.total_observer_presence(), 4),
                "privacy_posture": "interactive" if active else "background",
                "active_observers": [
                    {"id": agent_id, "presence": round(presence, 3)}
                    for agent_id, presence in active
                ],
                "snapshots": {
                    agent_id: snapshot.to_dict()
                    for agent_id, snapshot in self._snapshots.items()
                },
            }


# Compatibility class and service names while callers migrate terminology.
RecursiveTheoryOfMind = ObserverContextModel

_INSTANCE: ObserverContextModel | None = None


def get_recursive_tom() -> ObserverContextModel:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ObserverContextModel()
    return _INSTANCE


__all__ = [
    "ACTION_CATEGORIES",
    "BACKGROUND_PRIVATE_ACTIONS",
    "BIAS_DIM",
    "BiasProfile",
    "FOREGROUND_ACTIONS",
    "MAX_DEPTH",
    "MindSnapshot",
    "ObserverContextModel",
    "PRIVATE_ACTIONS",
    "PUBLIC_ACTIONS",
    "RecursiveTheoryOfMind",
    "get_recursive_tom",
]
