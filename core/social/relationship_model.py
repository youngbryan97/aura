"""core/social/relationship_model.py

Persistent Relationship Models
================================
Aura keeps a per-relationship dossier rather than treating the user as
prompt context. Each relationship records:

  * preferences (style, humor, sensitivities)
  * unresolved shared projects
  * prior emotional context (recent affect of the *interaction*, not Aura's
    own affect)
  * boundaries the user has set
  * trust history (count of moments that earned vs eroded trust)
  * "times she was wrong" (apology log, with what was acknowledged)
  * commitments she made (and whether they were kept)
  * topics that matter to the user
  * topics the user has asked Aura to drop

The dossier is used by the conversation lane to inform tone, by the
project ledger to wire commitments into long-horizon work, and by the
self-object to surface unfinished business.

Storage: durable JSON per-relationship in
``~/.aura/data/relationships/{relationship_id}.json``. The file is updated
atomically (write-tmp + rename) so a crash mid-write never produces a
half-written dossier.
"""
from __future__ import annotations

from core.runtime.atomic_writer import atomic_write_text

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Relationships")

_REL_DIR = Path.home() / ".aura" / "data" / "relationships"
_REL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrustEvent:
    when: float
    delta: float  # -1.0 to +1.0
    reason: str
    receipt_id: Optional[str] = None


@dataclass
class Apology:
    when: float
    acknowledged: str
    repair_offered: Optional[str] = None
    accepted_by_user: Optional[bool] = None


@dataclass
class Commitment:
    commitment_id: str
    when_made: float
    description: str
    deadline_hint: Optional[str] = None
    fulfilled_at: Optional[float] = None
    receipt_id: Optional[str] = None
    waived_at: Optional[float] = None
    waiver_reason: Optional[str] = None


@dataclass
class Boundary:
    when: float
    description: str  # e.g. "do not initiate after midnight"
    active: bool = True
    last_acknowledged_at: Optional[float] = None


@dataclass
class TopicNote:
    topic: str
    weight: float  # 0..1; matters-to-user score
    last_seen_at: float
    drop_requested: bool = False


@dataclass
class RelationshipDossier:
    relationship_id: str
    name: str
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    humor_style: str = ""
    sensitivities: List[str] = field(default_factory=list)
    boundaries: List[Boundary] = field(default_factory=list)
    trust_history: List[TrustEvent] = field(default_factory=list)
    apologies: List[Apology] = field(default_factory=list)
    commitments: List[Commitment] = field(default_factory=list)
    topics: List[TopicNote] = field(default_factory=list)
    open_threads: List[str] = field(default_factory=list)  # unresolved projects/questions
    interaction_affect_history: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    # ---- derived metrics ----

    def trust_score(self) -> float:
        if not self.trust_history:
            return 0.5
        # Smooth running average bounded into [0, 1].
        s = 0.5
        for e in self.trust_history[-32:]:
            s = max(0.0, min(1.0, s + e.delta * 0.05))
        return s

    def open_commitments(self) -> List[Commitment]:
        return [c for c in self.commitments if c.fulfilled_at is None and c.waived_at is None]

    def fulfilled_rate(self) -> float:
        if not self.commitments:
            return 1.0
        completed = [c for c in self.commitments if c.fulfilled_at is not None]
        return len(completed) / max(1, len(self.commitments))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class RelationshipStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._cache: Dict[str, RelationshipDossier] = {}

    def _path(self, relationship_id: str) -> Path:
        return _REL_DIR / f"{relationship_id}.json"

    # ---- read ----

    def get(self, relationship_id: str) -> Optional[RelationshipDossier]:
        with self._lock:
            if relationship_id in self._cache:
                return self._cache[relationship_id]
            p = self._path(relationship_id)
            if not p.exists():
                return None
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                dossier = self._from_dict(data)
                self._cache[relationship_id] = dossier
                return dossier
            except Exception as exc:
                logger.warning("relationship load failed for %s: %s", relationship_id, exc)
                return None

    def get_or_create(self, relationship_id: str, *, name: str) -> RelationshipDossier:
        existing = self.get(relationship_id)
        if existing:
            return existing
        dossier = RelationshipDossier(relationship_id=relationship_id, name=name)
        self._save(dossier)
        return dossier

    def list_all(self) -> List[RelationshipDossier]:
        with self._lock:
            out = list(self._cache.values())
            for path in _REL_DIR.glob("*.json"):
                rid = path.stem
                if rid not in self._cache:
                    d = self.get(rid)
                    if d is not None:
                        out.append(d)
            return out

    # ---- write ----

    def add_trust_event(self, relationship_id: str, *, delta: float, reason: str, receipt_id: Optional[str] = None) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        d.trust_history.append(TrustEvent(when=time.time(), delta=float(delta), reason=reason, receipt_id=receipt_id))
        d.last_seen_at = time.time()
        self._save(d)

    def add_apology(self, relationship_id: str, *, acknowledged: str, repair_offered: Optional[str] = None) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        d.apologies.append(Apology(when=time.time(), acknowledged=acknowledged, repair_offered=repair_offered))
        d.last_seen_at = time.time()
        self._save(d)

    def make_commitment(self, relationship_id: str, *, description: str, deadline_hint: Optional[str] = None) -> Commitment:
        d = self.get(relationship_id)
        if d is None:
            raise ValueError(f"unknown relationship_id={relationship_id}")
        c = Commitment(
            commitment_id=f"COMM-{uuid.uuid4().hex[:10]}",
            when_made=time.time(),
            description=description,
            deadline_hint=deadline_hint,
        )
        d.commitments.append(c)
        d.last_seen_at = time.time()
        self._save(d)
        return c

    def fulfill_commitment(self, relationship_id: str, commitment_id: str, *, receipt_id: Optional[str] = None) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        for c in d.commitments:
            if c.commitment_id == commitment_id and c.fulfilled_at is None and c.waived_at is None:
                c.fulfilled_at = time.time()
                c.receipt_id = receipt_id
                break
        self._save(d)

    def waive_commitment(self, relationship_id: str, commitment_id: str, *, reason: str) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        for c in d.commitments:
            if c.commitment_id == commitment_id and c.fulfilled_at is None and c.waived_at is None:
                c.waived_at = time.time()
                c.waiver_reason = reason
                break
        self._save(d)

    def add_boundary(self, relationship_id: str, description: str) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        d.boundaries.append(Boundary(when=time.time(), description=description))
        self._save(d)

    def acknowledge_boundary(self, relationship_id: str, description: str) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        for b in d.boundaries:
            if b.description == description and b.active:
                b.last_acknowledged_at = time.time()
        self._save(d)

    def touch_topic(self, relationship_id: str, topic: str, *, weight_delta: float = 0.05) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        for t in d.topics:
            if t.topic == topic:
                t.weight = max(0.0, min(1.0, t.weight + weight_delta))
                t.last_seen_at = time.time()
                self._save(d)
                return
        d.topics.append(TopicNote(topic=topic, weight=max(0.0, weight_delta), last_seen_at=time.time()))
        self._save(d)

    def request_topic_drop(self, relationship_id: str, topic: str) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        for t in d.topics:
            if t.topic == topic:
                t.drop_requested = True
        self._save(d)

    def add_open_thread(self, relationship_id: str, thread: str) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        if thread not in d.open_threads:
            d.open_threads.append(thread)
            self._save(d)

    def close_open_thread(self, relationship_id: str, thread: str) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        d.open_threads = [t for t in d.open_threads if t != thread]
        self._save(d)

    def record_interaction_affect(self, relationship_id: str, affect: Dict[str, Any]) -> None:
        d = self.get(relationship_id)
        if d is None:
            return
        affect = dict(affect)
        affect.setdefault("when", time.time())
        d.interaction_affect_history.append(affect)
        # cap history to 256 entries to keep the file bounded
        if len(d.interaction_affect_history) > 256:
            d.interaction_affect_history = d.interaction_affect_history[-256:]
        d.last_seen_at = time.time()
        self._save(d)

    # ---- io ----

    def _save(self, d: RelationshipDossier) -> None:
        with self._lock:
            self._cache[d.relationship_id] = d
            path = self._path(d.relationship_id)
            tmp = path.with_suffix(".json.tmp")
            atomic_write_text(tmp, json.dumps(d.to_dict(), indent=2, default=str), encoding="utf-8")
            os.replace(tmp, path)

    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> RelationshipDossier:
        return RelationshipDossier(
            relationship_id=data.get("relationship_id", "unknown"),
            name=data.get("name", "unknown"),
            created_at=float(data.get("created_at", time.time())),
            last_seen_at=float(data.get("last_seen_at", time.time())),
            style_preferences=dict(data.get("style_preferences", {}) or {}),
            humor_style=data.get("humor_style", ""),
            sensitivities=list(data.get("sensitivities", []) or []),
            boundaries=[Boundary(**b) for b in data.get("boundaries", []) or [] if isinstance(b, dict)],
            trust_history=[TrustEvent(**t) for t in data.get("trust_history", []) or [] if isinstance(t, dict)],
            apologies=[Apology(**a) for a in data.get("apologies", []) or [] if isinstance(a, dict)],
            commitments=[Commitment(**c) for c in data.get("commitments", []) or [] if isinstance(c, dict)],
            topics=[TopicNote(**t) for t in data.get("topics", []) or [] if isinstance(t, dict)],
            open_threads=list(data.get("open_threads", []) or []),
            interaction_affect_history=list(data.get("interaction_affect_history", []) or []),
            notes=data.get("notes", ""),
        )


_STORE: Optional[RelationshipStore] = None


def get_store() -> RelationshipStore:
    global _STORE
    if _STORE is None:
        _STORE = RelationshipStore()
    return _STORE


__all__ = [
    "RelationshipDossier",
    "RelationshipStore",
    "TrustEvent",
    "Apology",
    "Commitment",
    "Boundary",
    "TopicNote",
    "get_store",
]
