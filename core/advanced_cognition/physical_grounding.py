"""Fast typed physical/digital grounding for reflex control."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import ActionCandidate, Observation, clamp, stable_hash


@dataclass
class TrackedObject:
    object_id: str
    kind: str
    position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    confidence: float = 0.5
    last_seen: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=32))

    def update(self, position: tuple[float, float, float] | None, confidence: float, attrs: Mapping[str, Any]) -> None:
        now = time.time()
        if position is not None and self.position is not None:
            dt = max(1e-3, now - self.last_seen)
            self.velocity = tuple((position[i] - self.position[i]) / dt for i in range(3))
        if self.position is not None:
            self.history.append((self.last_seen, self.position, self.confidence))
        self.position = position if position is not None else self.position
        self.confidence = clamp(0.15 * self.confidence + 0.85 * confidence)
        self.attributes.update(attrs)
        self.last_seen = now

    def predicted_position(self, t: float | None = None) -> tuple[float, float, float] | None:
        if self.position is None:
            return None
        dt = max(0.0, (t or time.time()) - self.last_seen)
        decay = math.exp(-dt / 5.0)
        return tuple(self.position[i] + self.velocity[i] * dt * decay for i in range(3))

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "position": self.position,
            "velocity": self.velocity,
            "confidence": self.confidence,
            "last_seen": self.last_seen,
            "attributes": self.attributes,
            "history": list(self.history),
        }


@dataclass(frozen=True)
class GroundedState:
    state_id: str
    objects: dict[str, dict[str, Any]]
    resources: dict[str, float]
    hazards: list[dict[str, Any]]
    affordances: list[dict[str, Any]]
    spatial_map: dict[str, Any]
    confidence: float
    created_at: float = field(default_factory=time.time)


class PhysicalGroundingEngine:
    """Converts observations into resources, hazards, affordances, and maps."""

    def __init__(self, *, state_path: str | Path | None = None, max_objects: int = 2048):
        self.state_path = Path(state_path) if state_path else None
        self.max_objects = max_objects
        self.objects: dict[str, TrackedObject] = {}
        self.resources: dict[str, float] = defaultdict(lambda: 0.5)
        self.last_grounded: GroundedState | None = None

    def ingest(self, observation: Observation | Mapping[str, Any]) -> GroundedState:
        obs = observation if isinstance(observation, Observation) else Observation(**dict(observation))
        self._resources(obs)
        self._objects(obs)
        hazards = self._hazards(obs)
        affordances = self._affordances(obs)
        spatial = self._spatial()
        confidence = clamp(
            0.25
            + obs.confidence * 0.45
            + min(0.25, len(self.objects) / 100)
            + (0.05 if hazards else 0)
            + (0.1 if affordances else 0)
        )
        state = GroundedState(
            stable_hash({"obs": obs.observation_id, "objects": sorted(self.objects), "resources": dict(self.resources)}, prefix="gr_"),
            {k: v.to_dict() for k, v in self.objects.items()},
            dict(self.resources),
            hazards,
            affordances,
            spatial,
            confidence,
        )
        self.last_grounded = state
        self._prune()
        return state

    def reflex_recommendation(
        self,
        observation: Observation | Mapping[str, Any],
        actions: Sequence[ActionCandidate | Mapping[str, Any]],
        *,
        max_risk: float = 0.45,
    ) -> dict[str, Any]:
        state = self.ingest(observation)
        acts = [a if isinstance(a, ActionCandidate) else ActionCandidate(**dict(a)) for a in actions]
        scored = []
        for action in acts:
            risk = 0.04 + 0.07 * action.authority_tier + (0.15 if not action.reversible else 0.0)
            if set(action.tags) & {"unknown_use", "delete", "deploy", "self_modify", "network_post"}:
                risk += 0.2
            if state.hazards and action.kind in {"move", "advance", "activate_affordance", "attack", "execute"}:
                risk += max(h["risk"] for h in state.hazards[:3]) * 0.45
            if action.kind in {"observe", "wait", "inspect", "probe", "observe_or_probe"}:
                risk *= 0.45
            if state.resources.get("health", 1) < 0.3 or state.resources.get("energy", 1) < 0.2:
                risk += 0.12
            risk = clamp(risk)
            score = (0.12 if "probe" in action.kind or "observe" in action.kind else 0.0) - risk - 0.05 * action.expected_cost
            for affordance in state.affordances:
                if affordance["action_kind"] == action.kind or affordance["action_kind"] in action.tags:
                    score += affordance.get("confidence", 0.2) * 0.25
            scored.append(
                {
                    "action": action.to_dict(),
                    "risk": risk,
                    "score": score,
                    "reason": "high" if risk > 0.7 else "moderate" if risk > 0.4 else "low",
                }
            )
        scored.sort(key=lambda x: (x["score"], -x["risk"]), reverse=True)
        selected = next((s for s in scored if s["risk"] <= max_risk), None)
        return {
            "selected": selected["action"] if selected else None,
            "ranking": scored,
            "grounded_state": state,
            "receipt_id": stable_hash({"state": state.state_id, "ranking": scored, "ts": round(time.time(), 3)}, prefix="phys_"),
        }

    def _resources(self, obs: Observation) -> None:
        mapping = {
            "hp": "health",
            "health": "health",
            "energy": "energy",
            "battery": "energy",
            "hunger": "nutrition",
            "food": "nutrition",
            "money": "capital",
            "time": "time",
            "trust": "social_trust",
        }
        for path, value in self._flat(obs.state):
            key = path.lower().split(".")[-1]
            if key in mapping and isinstance(value, (int, float, bool)):
                self.resources[mapping[key]] = clamp(float(value))
        self.resources["confidence"] = clamp(max(self.resources.get("confidence", 0.5), obs.confidence))

    def _objects(self, obs: Observation) -> None:
        for item in self._extract(obs):
            tracked = self.objects.get(item["object_id"]) or TrackedObject(item["object_id"], item.get("kind", "object"))
            self.objects[item["object_id"]] = tracked
            tracked.update(item.get("position"), item.get("confidence", obs.confidence), item.get("attributes", {}))

    def _extract(self, obs: Observation) -> list[dict[str, Any]]:
        out = []
        state = obs.state
        for key in ("objects", "entities", "items", "nodes", "elements"):
            vals = state.get(key) if isinstance(state, Mapping) else None
            if isinstance(vals, list):
                for i, item in enumerate(vals):
                    if isinstance(item, Mapping):
                        out.append(self._object(obs, key, i, item))
        grid = state.get("grid") if isinstance(state, Mapping) else None
        if isinstance(grid, list):
            for y, row in enumerate(grid[:200]):
                row_s = "".join(row) if isinstance(row, list) else str(row)
                for x, ch in enumerate(row_s[:300]):
                    if ch not in {" ", ".", "#"}:
                        kind = (
                            "self"
                            if ch == "@"
                            else "actor"
                            if ch.isalpha()
                            else "item"
                            if ch in "$%!?/=*"
                            else "transition"
                            if ch in "+<>"
                            else "glyph_entity"
                        )
                        out.append(
                            {
                                "object_id": stable_hash({"d": obs.domain, "g": ch, "x": x, "y": y}, prefix="obj_"),
                                "kind": kind,
                                "position": (float(x), float(y), 0.0),
                                "confidence": obs.confidence,
                                "attributes": {"glyph": ch, "source": "grid"},
                            }
                        )
        return out

    def _object(self, obs: Observation, key: str, i: int, item: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(item.get("type") or item.get("kind") or item.get("role") or key.rstrip("s") or "object").lower()
        name = str(item.get("id") or item.get("name") or item.get("label") or f"{key}_{i}")
        return {
            "object_id": stable_hash({"d": obs.domain, "k": kind, "n": name, "i": i}, prefix="obj_"),
            "kind": kind,
            "position": self._pos(item),
            "confidence": clamp(float(item.get("confidence", obs.confidence) or obs.confidence)),
            "attributes": dict(item),
        }

    def _hazards(self, obs: Observation) -> list[dict[str, Any]]:
        hazards = []
        self_objects = [o for o in self.objects.values() if o.kind in {"self", "player", "agent", "ego"} and o.position]
        words = {"hostile", "threat", "enemy", "danger", "trap", "fire", "error", "critical", "low", "collision"}
        for object_id, tracked in self.objects.items():
            attrs = json.dumps(tracked.attributes, sort_keys=True).lower()
            is_hazard = (
                tracked.kind in {"hazard", "enemy", "trap", "obstacle"}
                or any(w in attrs for w in words)
                or (tracked.kind == "actor" and tracked.attributes.get("glyph") not in {"@"})
            )
            if not is_hazard:
                continue
            risk = 0.35 * tracked.confidence
            nearest = None
            if self_objects and tracked.position:
                nearest = min(self._dist(tracked.position, s.position) for s in self_objects if s.position)
                risk += 0.45 if nearest < 2 else 0.25 if nearest < 6 else 0.0
            if self.resources.get("health", 1) < 0.35:
                risk += 0.18
            hazards.append(
                {
                    "object_id": object_id,
                    "kind": tracked.kind,
                    "risk": clamp(risk),
                    "distance_to_self": nearest,
                    "reason": "hazard/proximity/resource weighting",
                }
            )
        for path, value in self._flat(obs.state):
            text = f"{path}:{value}".lower()
            if any(w in text for w in ("critical", "timeout", "refused", "degraded", "unsafe")):
                hazards.append(
                    {
                        "object_id": stable_hash(text, prefix="haz_"),
                        "kind": "system_hazard",
                        "risk": 0.45 if "critical" in text or "unsafe" in text else 0.25,
                        "distance_to_self": None,
                        "reason": text[:160],
                    }
                )
        return sorted(hazards, key=lambda h: h["risk"], reverse=True)[:24]

    def _affordances(self, obs: Observation) -> list[dict[str, Any]]:
        affordances = []
        for object_id, tracked in self.objects.items():
            attrs = json.dumps(tracked.attributes, sort_keys=True).lower()
            action = None
            if tracked.kind in {"button", "link", "input", "transition"} or any(w in attrs for w in ("click", "open", "submit", "href", "door")):
                action = "activate_affordance"
            elif tracked.kind in {"item", "tool"} or any(w in attrs for w in ("pickup", "use", "apply")):
                action = "inspect_or_use"
            elif "prompt" in attrs or "modal" in attrs:
                action = "resolve_prompt"
            if action:
                affordances.append(
                    {
                        "object_id": object_id,
                        "action_kind": action,
                        "confidence": tracked.confidence,
                        "reversible_first": True,
                        "reason": f"{tracked.kind}->{action}",
                    }
                )
        return affordances or [
            {
                "object_id": "environment",
                "action_kind": "observe_or_probe",
                "confidence": 0.35,
                "reversible_first": True,
                "reason": "No explicit affordance; observe/probe.",
            }
        ]

    def _spatial(self) -> dict[str, Any]:
        positions = {oid: obj.predicted_position() for oid, obj in self.objects.items() if obj.predicted_position()}
        if not positions:
            return {"kind": "non_spatial", "object_count": len(self.objects)}
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        return {
            "kind": "metric_2d",
            "object_count": len(positions),
            "bounds": {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)},
            "self_objects": [oid for oid, obj in self.objects.items() if obj.kind in {"self", "player", "agent", "ego"}],
        }

    @staticmethod
    def _pos(item: Mapping[str, Any]) -> tuple[float, float, float] | None:
        if all(k in item for k in ("x", "y")):
            return (float(item.get("x", 0)), float(item.get("y", 0)), float(item.get("z", 0)))
        if "position" in item:
            pos = item["position"]
            if isinstance(pos, Mapping):
                return (float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0)))
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                return (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0)
        return None

    @staticmethod
    def _flat(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
        out = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                out.append((path, child))
                out += PhysicalGroundingEngine._flat(child, path)
        elif isinstance(value, list):
            out.append((prefix or "list", value))
            for i, child in enumerate(value[:32]):
                out.extend(PhysicalGroundingEngine._flat(child, f"{prefix}[{i}]"))
        else:
            out.append((prefix or "value", value))
        return out

    @staticmethod
    def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def _prune(self) -> None:
        if len(self.objects) > self.max_objects:
            overflow = len(self.objects) - self.max_objects
            for oid, _ in sorted(self.objects.items(), key=lambda kv: (kv[1].confidence, kv[1].last_seen))[:overflow]:
                self.objects.pop(oid, None)
