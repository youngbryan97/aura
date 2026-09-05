"""Fast typed physical/digital grounding for reflex control.

CP126 found ten defects here, four critical. The engine turns observations
into the hazards, affordances and risk scores a reflex controller acts on, so
each of them ends in something being done or not done:

  * **One world for every domain.** Objects and resources were global. A
    hazard seen in a terminal grid an hour ago still sat in the map when a
    browser observation arrived, and ``health`` from one environment scored
    risk in another. Each domain has its own view now, and objects expire.
  * **Risk trusted the caller's declaration.** ``kind="observe"`` multiplied
    risk by 0.45 and ``reversible=True`` skipped a penalty, so a delete could
    price itself as a look. Declarations can raise risk; a floor derived from
    the effectful tags and the authority tier is what they cannot go under.
  * **Everything was labelled reversible.** Click, open, submit, use and the
    default probe all carried ``reversible_first=True`` with no compensation
    contract behind it. Reversibility is reported as ``unknown`` unless the
    affordance is observation-only.
  * **The state id omitted the state.** It hashed the observation id, the
    object IDS and the resources — not positions, attributes, hazards,
    affordances, the map or the confidence. Two materially different worlds
    shared one identity, and every receipt built on it inherited that.
  * Grounding confidence rose with how many objects had been RETAINED and got
    a bonus for having emitted a heuristic hazard. Guessing more did not make
    the guess better.
  * Object identity came from list position or grid glyph, so reordering a
    list renamed everything and a moved glyph left a ghost.
  * ``low`` and ``error`` were substring-matched against serialised
    attributes, which flags ``below``, ``allow``, ``slow`` and ``no errors``.
  * ``state_path`` was accepted and never used: nothing saved, nothing
    loaded.
  * ``_flat`` recursed without a depth or cycle limit, and ``ingest``
    snapshotted every object before pruning, so the returned state could
    exceed ``max_objects``.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

from .schemas import ActionCandidate, Observation, clamp, stable_hash

logger = logging.getLogger("Aura.PhysicalGrounding")

#: An object nobody has seen for this long is no longer part of the world.
#: Nothing expired before, so a hazard observed once stayed in every
#: subsequent risk calculation for the life of the process.
_OBJECT_TTL_S = 900.0
#: Depth and breadth caps for walking caller-supplied state.
_MAX_FLATTEN_DEPTH = 12
_MAX_FLATTEN_ITEMS = 32

#: Tags that mean the action changes something. A declaration carrying one of
#: these cannot buy the observation discount, whatever it calls itself.
_EFFECTFUL_TAGS = frozenset(
    {
        "unknown_use", "delete", "deploy", "self_modify", "network_post",
        "write", "purchase", "send", "irreversible", "destructive",
    }
)
_OBSERVATION_KINDS = frozenset({"observe", "wait", "inspect", "probe", "observe_or_probe"})

#: Hazard words matched on WORD BOUNDARIES against named fields. "low" is
#: gone entirely: it is a substring of below, allow, slow, flow, and it was
#: being matched against a JSON dump of every attribute.
_HAZARD_WORDS = (
    "hostile", "threat", "enemy", "danger", "dangerous", "trap", "fire",
    "critical", "collision", "hazard", "unsafe",
)
_HAZARD_RE = re.compile(r"\b(?:" + "|".join(_HAZARD_WORDS) + r")\b", re.I)
#: Fields whose VALUES are allowed to raise a hazard. Searching a dump of
#: every attribute meant a label, a tooltip or a URL could do it.
_HAZARD_FIELDS = ("status", "state", "severity", "level", "role", "type", "kind", "class")
#: Negations that flip a match. "no errors" was a hazard.
_NEGATION_RE = re.compile(r"\b(?:no|not|zero|without|free of)\b\W{0,3}$", re.I)

_SELF_KINDS = frozenset({"self", "player", "agent", "ego"})


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
    #: How this object's id was derived. "natural" survives reordering;
    #: "positional" and "cell" do not, and downstream should not treat two
    #: sightings under one of those ids as proof of one object.
    identity: str = "natural"

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

    def current_confidence(self, now: float | None = None) -> float:
        """Confidence discounted by how long ago this was actually seen.

        Nothing decayed before: an object seen once at 0.9 was still a 0.9
        object an hour later, and hazard risk is computed from it.
        """
        age = max(0.0, (now or time.time()) - self.last_seen)
        return clamp(self.confidence * math.exp(-age / _OBJECT_TTL_S))

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
            "current_confidence": self.current_confidence(),
            "last_seen": self.last_seen,
            "attributes": self.attributes,
            "identity": self.identity,
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
    domain: str = ""
    confidence_basis: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class _DomainView:
    """One environment's world. They used to share a single one."""

    objects: dict[str, TrackedObject] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)


class PhysicalGroundingEngine:
    """Converts observations into resources, hazards, affordances, and maps.

    Everything is scoped to the observation's domain. It RANKS declared
    actions; it does not verify them — a candidate that misstates both its
    kind and its tags is indistinguishable from an honest one here, and the
    authority gate is what checks. What this module guarantees is that a
    declaration can only ever RAISE the risk it is scored at.
    """

    def __init__(self, *, state_path: str | Path | None = None, max_objects: int = 2048):
        self.state_path = Path(state_path) if state_path else None
        self.max_objects = max_objects
        self._domains: dict[str, _DomainView] = defaultdict(_DomainView)
        self.last_grounded: GroundedState | None = None
        self._dirty = False
        if self.state_path:
            self.load()

    # ── compatibility views ───────────────────────────────────────────

    @property
    def objects(self) -> dict[str, TrackedObject]:
        """Every live object across every domain, for callers that count them."""
        merged: dict[str, TrackedObject] = {}
        for view in self._domains.values():
            merged.update(view.objects)
        return merged

    @property
    def resources(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        for view in self._domains.values():
            merged.update(view.resources)
        return merged

    # ── ingestion ─────────────────────────────────────────────────────

    def ingest(self, observation: Observation | Mapping[str, Any]) -> GroundedState:
        obs = observation if isinstance(observation, Observation) else Observation(**dict(observation))
        view = self._domains[obs.domain]

        self._expire(view)
        self._resources(view, obs)
        self._objects(view, obs)
        # Pruning BEFORE the snapshot. It ran afterwards, so the state handed
        # to the caller could hold more objects than max_objects allows.
        self._prune(view)

        hazards = self._hazards(view, obs)
        affordances = self._affordances(view)
        spatial = self._spatial(view)
        confidence, basis = self._grounding_confidence(view, obs)

        objects = {k: v.to_dict() for k, v in view.objects.items()}
        resources = dict(view.resources)
        state = GroundedState(
            # The id covers the STATE. It used to hash the observation id, the
            # object ids and the resources only, so two worlds differing in
            # every position, attribute and hazard shared one identity — and
            # every receipt built on it inherited the collision.
            stable_hash(
                {
                    "obs": obs.observation_id,
                    "domain": obs.domain,
                    "objects": objects,
                    "resources": resources,
                    "hazards": hazards,
                    "affordances": affordances,
                    "spatial": spatial,
                    "confidence": round(confidence, 6),
                },
                prefix="gr_",
            ),
            objects,
            resources,
            hazards,
            affordances,
            spatial,
            confidence,
            domain=obs.domain,
            confidence_basis=basis,
        )
        self.last_grounded = state
        self._dirty = True
        # Deliberately NOT saving here. ingest() runs inside the integration
        # layer's observation lock, and an fsync under a lock is how this
        # runtime freezes — lockdep fails the build for it. Persistence is
        # save(), called by the owner outside its lock.
        return state

    # ── action ranking ────────────────────────────────────────────────

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
            effectful = bool(set(action.tags) & _EFFECTFUL_TAGS)
            observation_only = action.kind in _OBSERVATION_KINDS and not effectful

            # A floor the declaration cannot get under. "observe" used to
            # multiply the whole risk by 0.45 and reversible=True used to skip
            # a penalty, so an action could price a delete as a look.
            floor = 0.0 if observation_only else 0.15 + 0.05 * action.authority_tier

            risk = 0.04 + 0.07 * action.authority_tier + (0.15 if not action.reversible else 0.0)
            if effectful:
                risk += 0.2
            if state.hazards and action.kind in {"move", "advance", "activate_affordance", "attack", "execute"}:
                risk += max(h["risk"] for h in state.hazards[:3]) * 0.45
            if observation_only:
                risk *= 0.45
            if state.resources.get("health", 1) < 0.3 or state.resources.get("energy", 1) < 0.2:
                risk += 0.12
            risk = clamp(max(risk, floor))

            score = (0.12 if observation_only else 0.0) - risk - 0.05 * action.expected_cost
            for affordance in state.affordances:
                if affordance["action_kind"] == action.kind or affordance["action_kind"] in action.tags:
                    score += affordance.get("confidence", 0.2) * 0.25
            scored.append(
                {
                    "action": action.to_dict(),
                    "risk": risk,
                    "risk_floor": floor,
                    "declared_observation_only": observation_only,
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

    # ── resources ─────────────────────────────────────────────────────

    def _resources(self, view: _DomainView, obs: Observation) -> None:
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
            if key in mapping and isinstance(value, (int, float, bool)) and not isinstance(value, bool):
                view.resources[mapping[key]] = clamp(float(value))
            elif key in mapping and isinstance(value, bool):
                view.resources[mapping[key]] = 1.0 if value else 0.0
        view.resources["confidence"] = clamp(obs.confidence)

    # ── objects ───────────────────────────────────────────────────────

    def _objects(self, view: _DomainView, obs: Observation) -> None:
        extracted = self._extract(obs)
        cell_ids = {item["object_id"] for item in extracted if item.get("identity") == "cell"}
        if cell_ids or self._has_grid(obs):
            # A grid observation is a COMPLETE snapshot of that domain's
            # cells. Merging into the previous frame left a ghost behind
            # every glyph that moved.
            for object_id, tracked in list(view.objects.items()):
                if tracked.identity == "cell" and object_id not in cell_ids:
                    view.objects.pop(object_id, None)

        for item in extracted:
            tracked = view.objects.get(item["object_id"])
            if tracked is None:
                tracked = TrackedObject(
                    item["object_id"],
                    item.get("kind", "object"),
                    identity=item.get("identity", "natural"),
                )
                view.objects[item["object_id"]] = tracked
            tracked.update(item.get("position"), item.get("confidence", obs.confidence), item.get("attributes", {}))

    @staticmethod
    def _has_grid(obs: Observation) -> bool:
        state = obs.state
        return isinstance(state, Mapping) and isinstance(state.get("grid"), list)

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
                                "object_id": stable_hash({"d": obs.domain, "g": ch, "x": x, "y": y}, prefix="cell_"),
                                "kind": kind,
                                "position": (float(x), float(y), 0.0),
                                "confidence": obs.confidence,
                                "attributes": {"glyph": ch, "source": "grid"},
                                # A cell id names a POSITION, not a thing. Two
                                # sightings under one cell id are two sightings
                                # of that square.
                                "identity": "cell",
                            }
                        )
        return out

    def _object(self, obs: Observation, key: str, i: int, item: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(item.get("type") or item.get("kind") or item.get("role") or key.rstrip("s") or "object").lower()
        natural = item.get("id") or item.get("name") or item.get("label")
        if natural:
            # A stable natural key: reordering the list does not rename it.
            identity = "natural"
            payload = {"d": obs.domain, "k": kind, "n": str(natural)}
        else:
            # Nothing to key on. The id includes the index, which means it
            # changes when the list is reordered — recorded, so a caller does
            # not read continuity into it.
            identity = "positional"
            payload = {"d": obs.domain, "k": kind, "c": key, "i": i}
        return {
            "object_id": stable_hash(payload, prefix="obj_"),
            "kind": kind,
            "position": self._pos(item),
            "confidence": clamp(float(item.get("confidence", obs.confidence) or obs.confidence)),
            "attributes": dict(item),
            "identity": identity,
        }

    # ── hazards ───────────────────────────────────────────────────────

    @staticmethod
    def _hazard_text(attributes: Mapping[str, Any]) -> Iterable[str]:
        """Only the values of fields that describe condition.

        A JSON dump of every attribute was searched, so a button labelled
        "Fire the report" and a URL containing "error" were hazards.
        """
        for field_name in _HAZARD_FIELDS:
            value = attributes.get(field_name)
            if isinstance(value, (str, int, float)):
                yield str(value)

    @classmethod
    def _looks_hazardous(cls, text: str) -> bool:
        for match in _HAZARD_RE.finditer(text):
            preceding = text[: match.start()]
            if _NEGATION_RE.search(preceding):
                continue  # "no errors", "without danger"
            return True
        return False

    def _hazards(self, view: _DomainView, obs: Observation) -> list[dict[str, Any]]:
        hazards = []
        self_objects = [
            o for o in view.objects.values() if o.kind in _SELF_KINDS and o.position
        ]
        for object_id, tracked in view.objects.items():
            is_hazard = (
                tracked.kind in {"hazard", "enemy", "trap", "obstacle"}
                or any(self._looks_hazardous(text) for text in self._hazard_text(tracked.attributes))
                or (tracked.kind == "actor" and tracked.attributes.get("glyph") not in {"@"})
            )
            if not is_hazard:
                continue
            confidence = tracked.current_confidence()
            risk = 0.35 * confidence
            nearest = None
            if self_objects and tracked.position:
                nearest = min(self._dist(tracked.position, s.position) for s in self_objects if s.position)
                risk += 0.45 if nearest < 2 else 0.25 if nearest < 6 else 0.0
            if view.resources.get("health", 1) < 0.35:
                risk += 0.18
            hazards.append(
                {
                    "object_id": object_id,
                    "kind": tracked.kind,
                    "risk": clamp(risk),
                    "distance_to_self": nearest,
                    "object_confidence": confidence,
                    "reason": "hazard/proximity/resource weighting",
                }
            )
        for path, value in self._flat(obs.state):
            # Same discipline as the object scan: only fields that DESCRIBE a
            # condition. Reading every path meant a button labelled "Fire the
            # report" and a URL containing "error" were system hazards.
            if path.lower().split(".")[-1].split("[")[0] not in _HAZARD_FIELDS:
                continue
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                continue
            text = f"{path}:{value}"
            if self._looks_hazardous(text) or re.search(r"\b(?:timeout|refused|degraded)\b", text, re.I):
                hazards.append(
                    {
                        "object_id": stable_hash(text, prefix="haz_"),
                        "kind": "system_hazard",
                        "risk": 0.45 if re.search(r"\b(?:critical|unsafe)\b", text, re.I) else 0.25,
                        "distance_to_self": None,
                        "object_confidence": obs.confidence,
                        "reason": text[:160],
                    }
                )
        return sorted(hazards, key=lambda h: h["risk"], reverse=True)[:24]

    # ── affordances ───────────────────────────────────────────────────

    def _affordances(self, view: _DomainView) -> list[dict[str, Any]]:
        affordances = []
        for object_id, tracked in view.objects.items():
            attrs = json.dumps(tracked.attributes, sort_keys=True, default=str).lower()
            action = None
            if tracked.kind in {"button", "link", "input", "transition"} or any(
                w in attrs for w in ("click", "open", "submit", "href", "door")
            ):
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
                        "confidence": tracked.current_confidence(),
                        # Every one of these used to claim reversible_first=True.
                        # Nothing behind a click, an open, a submit or a use
                        # supplies a compensation contract, so the claim was
                        # unbacked wherever it mattered most.
                        "reversibility": "unknown",
                        "reason": f"{tracked.kind}->{action}",
                    }
                )
        return affordances or [
            {
                "object_id": "environment",
                "action_kind": "observe_or_probe",
                "confidence": 0.35,
                # The one case where it IS known: observing changes nothing.
                "reversibility": "observation_only",
                "reason": "No explicit affordance; observe/probe.",
            }
        ]

    # ── map + confidence ──────────────────────────────────────────────

    def _spatial(self, view: _DomainView) -> dict[str, Any]:
        positions = {
            oid: obj.predicted_position()
            for oid, obj in view.objects.items()
            if obj.predicted_position()
        }
        if not positions:
            return {"kind": "non_spatial", "object_count": len(view.objects)}
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        return {
            "kind": "metric_2d",
            "object_count": len(positions),
            "bounds": {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)},
            "self_objects": [oid for oid, obj in view.objects.items() if obj.kind in _SELF_KINDS],
        }

    @staticmethod
    def _grounding_confidence(view: _DomainView, obs: Observation) -> tuple[float, str]:
        """How well grounded this reading is — from the SENSOR, not the guesses.

        It used to add up to 0.25 for having retained a hundred objects and
        another 0.15 for having emitted a hazard or an affordance. Neither
        says anything about whether the reading is right: retaining more
        objects and guessing more hazards is what an unreliable sensor
        produces too.
        """
        if not view.objects:
            return clamp(0.25 + 0.45 * obs.confidence), "observation confidence only; no tracked objects"
        mean_object = sum(o.current_confidence() for o in view.objects.values()) / len(view.objects)
        value = clamp(0.25 + 0.45 * obs.confidence + 0.30 * mean_object)
        return value, "observation confidence and mean tracked-object confidence"

    # ── traversal ─────────────────────────────────────────────────────

    @staticmethod
    def _pos(item: Mapping[str, Any]) -> tuple[float, float, float] | None:
        try:
            if all(k in item for k in ("x", "y")):
                return (float(item.get("x", 0)), float(item.get("y", 0)), float(item.get("z", 0)))
            if "position" in item:
                pos = item["position"]
                if isinstance(pos, Mapping):
                    return (float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0)))
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    return (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0)
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _flat(
        value: Any,
        prefix: str = "",
        *,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[tuple[str, Any]]:
        """Flatten caller-supplied state, bounded in depth and against cycles.

        It recursed with no limit and no cycle check, so a self-referential
        mapping — which a caller can build by accident — exhausted the stack
        on the reflex path.
        """
        if depth > _MAX_FLATTEN_DEPTH:
            return [(prefix or "value", "<max depth>")]
        seen = seen if seen is not None else set()
        out: list[tuple[str, Any]] = []
        if isinstance(value, (Mapping, list)):
            marker = id(value)
            if marker in seen:
                return [(prefix or "value", "<cycle>")]
            seen = seen | {marker}
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                out.append((path, child))
                out += PhysicalGroundingEngine._flat(child, path, depth=depth + 1, seen=seen)
        elif isinstance(value, list):
            out.append((prefix or "list", value))
            for i, child in enumerate(value[:_MAX_FLATTEN_ITEMS]):
                out.extend(
                    PhysicalGroundingEngine._flat(child, f"{prefix}[{i}]", depth=depth + 1, seen=seen)
                )
        else:
            out.append((prefix or "value", value))
        return out

    @staticmethod
    def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    # ── retention ─────────────────────────────────────────────────────

    def _expire(self, view: _DomainView) -> None:
        cutoff = time.time() - _OBJECT_TTL_S
        for object_id, tracked in list(view.objects.items()):
            if tracked.last_seen < cutoff:
                view.objects.pop(object_id, None)

    def _prune(self, view: _DomainView) -> None:
        if len(view.objects) <= self.max_objects:
            return
        overflow = len(view.objects) - self.max_objects
        ranked = sorted(
            view.objects.items(), key=lambda kv: (kv[1].current_confidence(), kv[1].last_seen)
        )
        for oid, _ in ranked[:overflow]:
            view.objects.pop(oid, None)

    # ── durability ────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the world. ``state_path`` was accepted and never used.

        integration.py passes one and reads ``len(self.grounding.objects)``
        into a health surface, so "durable" was a reasonable thing for a
        caller to assume and was not true.

        Call it OUTSIDE any lock the caller holds. ``ingest`` only marks the
        world dirty.
        """
        if not self.state_path or not self._dirty:
            return
        payload = {
            "schema": "aura.advanced_cognition.physical_grounding.v1",
            "saved_at": time.time(),
            "domains": {
                domain: {
                    "objects": {oid: obj.to_dict() for oid, obj in view.objects.items()},
                    "resources": dict(view.resources),
                }
                for domain, view in self._domains.items()
            },
        }
        source = "advanced_cognition.physical_grounding"
        try:
            with local_internal_governed_scope(source, domain="state_mutation"):
                get_file_write_gateway().write_text(
                    self.state_path,
                    json.dumps(payload, sort_keys=True, default=str),
                    source=source,
                )
            self._dirty = False
        except OSError as exc:
            record_degradation("physical_grounding", exc, action="grounded world not persisted")

    def load(self) -> None:
        """Rebuild the world, discarding anything unreadable."""
        if not self.state_path or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            record_degradation(
                "physical_grounding", exc, action="grounded world discarded; starting empty"
            )
            return
        if not isinstance(payload, dict):
            return
        for domain, raw_view in (payload.get("domains") or {}).items():
            if not isinstance(raw_view, dict):
                continue
            view = self._domains[str(domain)]
            for oid, raw in (raw_view.get("objects") or {}).items():
                tracked = self._decode_object(raw)
                if tracked is not None:
                    view.objects[str(oid)] = tracked
            for key, value in (raw_view.get("resources") or {}).items():
                try:
                    view.resources[str(key)] = clamp(float(value))
                except (TypeError, ValueError):
                    continue
            self._expire(view)

    @staticmethod
    def _decode_object(raw: Any) -> TrackedObject | None:
        if not isinstance(raw, dict) or not raw.get("object_id"):
            return None
        try:
            position = raw.get("position")
            velocity = raw.get("velocity") or (0.0, 0.0, 0.0)
            return TrackedObject(
                object_id=str(raw["object_id"]),
                kind=str(raw.get("kind", "object")),
                position=tuple(float(v) for v in position) if position else None,
                velocity=tuple(float(v) for v in velocity),
                confidence=clamp(float(raw.get("confidence", 0.5))),
                last_seen=float(raw.get("last_seen", time.time())),
                attributes=dict(raw.get("attributes") or {}),
                identity=str(raw.get("identity", "natural")),
            )
        except (TypeError, ValueError):
            return None
