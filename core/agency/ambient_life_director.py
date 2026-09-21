"""Ambient life director for autonomy pacing and motive selection.

This module folds useful game-AI patterns into Aura's general agency path:

* utility/motive buckets from The Sims-style AI,
* pressure pacing from Alien: Isolation-style directors,
* persistent encounter memory inspired by Nemesis/companion systems,
* resource-aware LOD for KCD2-scale background simulation.

It does not add task-specific scripts. It annotates and mildly reorders already
eligible initiatives before the normal subjective-choice/governance path.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.AmbientLifeDirector")

BUCKETS = (
    "survival",
    "repair",
    "security",
    "social",
    "curiosity",
    "creativity",
    "maintenance",
    "play",
)
MAX_ENCOUNTERS = 300


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


@dataclass
class EncounterMemory:
    entity_id: str
    outcome: str
    valence: float
    traits: dict[str, Any] = field(default_factory=dict)
    seen_count: int = 1
    last_seen_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmbientDecisionContext:
    bucket_priorities: dict[str, float]
    pressure: float
    lod_mode: str
    foreground_sensitive: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_priorities": {key: round(value, 4) for key, value in self.bucket_priorities.items()},
            "pressure": round(self.pressure, 4),
            "lod_mode": self.lod_mode,
            "foreground_sensitive": self.foreground_sensitive,
        }


class AmbientLifeDirector:
    """General autonomy director that adds motive and pacing context."""

    SERVICE_NAME = "ambient_life_director"

    def __init__(self, state_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._last_goal: str = ""
        self._last_goal_at: float = 0.0
        self._encounters: dict[str, EncounterMemory] = {}
        if state_path is None:
            try:
                from core.config import config

                state_path = Path(config.paths.data_dir) / "cognitive" / "ambient_life_director.json"
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("ambient_life_director", exc, severity="debug")
                state_path = state_root() / "data" / "cognitive" / "ambient_life_director.json"
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def is_alive(self) -> bool:
        return True

    def classify_bucket(self, initiative: dict[str, Any]) -> str:
        metadata = dict(initiative.get("metadata", {}) or {})
        explicit = str(metadata.get("motive_bucket", "")).strip().lower()
        if explicit in BUCKETS:
            return explicit
        text = " ".join(
            str(initiative.get(key, ""))
            for key in ("goal", "description", "type", "source")
        ).lower()
        markers = (
            ("repair", ("repair", "fix", "heal", "patch", "regression", "degradation", "error")),
            ("security", ("security", "threat", "malware", "permission", "attack", "quarantine")),
            ("curiosity", ("research", "learn", "explore", "discover", "read", "question")),
            ("play", ("play", "game", "experiment", "sandbox", "whim")),
            ("creativity", ("imagine", "write", "draw", "compose", "story", "novel", "idea")),
            ("survival", ("oom", "thermal", "ram", "crash", "shutdown", "health", "boot", "alive")),
            ("social", ("conversation", "user", "bryan", "relationship", "friend", "check in")),
            ("maintenance", ("audit", "cleanup", "organize", "index", "consolidate", "routine")),
        )
        for bucket, words in markers:
            if any(word in text for word in words):
                return bucket
        return "maintenance"

    def build_context(self, state: Any | None = None) -> AmbientDecisionContext:
        pressure = self._runtime_pressure(state)
        priorities = {
            "survival": 0.50 + 0.50 * pressure,
            "repair": 0.45 + 0.45 * pressure,
            "security": 0.42 + 0.45 * pressure,
            "social": 0.62 - 0.18 * pressure,
            "curiosity": 0.66 - 0.28 * pressure,
            "creativity": 0.58 - 0.24 * pressure,
            "maintenance": 0.52 + 0.20 * pressure,
            "play": 0.36 - 0.28 * pressure,
        }
        priorities = {key: _clamp(value) for key, value in priorities.items()}
        return AmbientDecisionContext(
            bucket_priorities=priorities,
            pressure=pressure,
            lod_mode=self._lod_mode(pressure),
            foreground_sensitive=pressure >= 0.65,
        )

    def prioritize_scored(self, scored: list[Any], state: Any | None = None) -> list[Any]:
        if not scored:
            return scored
        context = self.build_context(state)
        now = time.time()
        for item in scored:
            initiative = getattr(item, "initiative", {}) or {}
            metadata = initiative.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                initiative["metadata"] = metadata
            bucket = self.classify_bucket(initiative)
            priority = context.bucket_priorities.get(bucket, 0.5)
            inertia = 1.0
            goal = str(initiative.get("goal") or initiative.get("description") or "")
            if goal and goal == self._last_goal and now - self._last_goal_at < 90.0:
                inertia = 1.06
            lod_penalty = 1.0
            if context.lod_mode == "reduced" and bucket in {"play", "creativity"}:
                lod_penalty = 0.82
            elif context.lod_mode == "deferred" and bucket in {"play", "creativity", "curiosity", "social"}:
                lod_penalty = 0.55
            adjusted = _clamp(float(getattr(item, "final_score", 0.0) or 0.0) * (0.85 + 0.30 * priority) * inertia * lod_penalty)
            item.final_score = adjusted
            metadata["ambient_bucket"] = bucket
            metadata["ambient_priority"] = round(priority, 4)
            metadata["ambient_lod_mode"] = context.lod_mode
            metadata["ambient_pressure"] = round(context.pressure, 4)
        scored.sort(key=lambda item: getattr(item, "final_score", 0.0), reverse=True)
        if scored:
            top = getattr(scored[0], "initiative", {}) or {}
            self._last_goal = str(top.get("goal") or top.get("description") or "")
            self._last_goal_at = time.time()
        return scored

    def record_encounter(
        self,
        entity_id: str,
        *,
        outcome: str,
        valence: float = 0.0,
        traits: dict[str, Any] | None = None,
    ) -> EncounterMemory:
        entity_id = str(entity_id or "unknown")
        with self._lock:
            existing = self._encounters.get(entity_id)
            if existing:
                existing.outcome = str(outcome or "")[:300]
                existing.valence = _clamp(valence, -1.0, 1.0)
                existing.traits.update(dict(traits or {}))
                existing.seen_count += 1
                existing.last_seen_at = time.time()
                memory = existing
            else:
                memory = EncounterMemory(
                    entity_id=entity_id,
                    outcome=str(outcome or "")[:300],
                    valence=_clamp(valence, -1.0, 1.0),
                    traits=dict(traits or {}),
                )
                self._encounters[entity_id] = memory
            if len(self._encounters) > MAX_ENCOUNTERS:
                oldest = sorted(self._encounters.values(), key=lambda item: item.last_seen_at)
                for item in oldest[: len(self._encounters) - MAX_ENCOUNTERS]:
                    self._encounters.pop(item.entity_id, None)
            self._save()
            return memory

    def recall_encounter(self, entity_id: str) -> EncounterMemory | None:
        with self._lock:
            return self._encounters.get(str(entity_id or "unknown"))

    def get_status(self) -> dict[str, Any]:
        context = self.build_context()
        with self._lock:
            return {
                "service": self.SERVICE_NAME,
                "running": True,
                "encounter_count": len(self._encounters),
                "context": context.to_dict(),
                "state_path": str(self._state_path),
            }

    status = get_status

    def _runtime_pressure(self, state: Any | None) -> float:
        candidates: list[float] = []
        for path in (
            ("runtime", "resource_pressure"),
            ("somatic", "resource_pressure"),
            ("health", "pressure"),
            ("motivation", "stress"),
        ):
            value = state
            for name in path:
                value = getattr(value, name, None) if value is not None else None
            if value is not None:
                candidates.append(_clamp(value))
        try:
            from core.runtime.pressure_stall import Resource, pressure

            candidates.append(_clamp(pressure(Resource.MEMORY)))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            # not a failure: one candidate of several, and the max below
            # takes whichever were readable.
            pass
        return max(candidates, default=0.0)

    @staticmethod
    def _lod_mode(pressure: float) -> str:
        if pressure >= 0.82:
            return "deferred"
        if pressure >= 0.58:
            return "reduced"
        return "full"

    def _save(self) -> None:
        payload = {
            "encounters": [memory.to_dict() for memory in self._encounters.values()],
            "last_goal": self._last_goal,
            "last_goal_at": self._last_goal_at,
            "saved_at": time.time(),
        }
        try:
            atomic_write_text(self._state_path, json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            record_degradation("ambient_life_director", exc, severity="debug")

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._last_goal = str(data.get("last_goal", ""))
            self._last_goal_at = float(data.get("last_goal_at", 0.0))
            self._encounters = {
                str(item["entity_id"]): EncounterMemory(**item)
                for item in data.get("encounters", [])
                if isinstance(item, dict) and item.get("entity_id")
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation("ambient_life_director", exc, severity="debug")


_director: AmbientLifeDirector | None = None
_director_lock = threading.Lock()


def get_ambient_life_director() -> AmbientLifeDirector:
    global _director
    if _director is None:
        with _director_lock:
            if _director is None:
                _director = AmbientLifeDirector()
                _register_in_container(_director)
    return _director


def _register_in_container(director: AmbientLifeDirector) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(AmbientLifeDirector.SERVICE_NAME):
            ServiceContainer.register_instance(
                AmbientLifeDirector.SERVICE_NAME,
                director,
                required=False,
                registered_by="ambient_life_director",
            )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("ambient_life_director_register", exc, severity="debug")


def reset_ambient_life_director_for_test() -> None:
    global _director
    _director = None
