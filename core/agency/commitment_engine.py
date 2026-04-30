"""core/agency/commitment_engine.py
Long-Horizon Commitment Engine
================================
Enables Aura to make and keep commitments across sessions and days —
not just within a single conversation.

A commitment is a promise with:
  - A specific outcome
  - A deadline
  - A progress tracking mechanism
  - Autonomous check-ins
  - Success/failure learning

Without this, Aura "forgets" what she committed to the moment the
conversation ends. With this, she tracks commitments persistently,
checks in autonomously, and learns from whether she kept them.

Types of commitments:
  USER_FACING  — promises made to the user in conversation
  AUTONOMOUS   — self-directed goals and intentions
  LEARNING     — knowledge acquisition targets

Commitments are distinct from HierarchicalPlanner goals:
  Goals = decomposed, strategic, internal
  Commitments = promises, often interpersonal, with accountability
"""
from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.CommitmentEngine")

PERSIST_PATH = Path.home() / ".aura" / "data" / "commitments.json"
CHECK_INTERVAL = 1800.0  # check commitments every 30 minutes


class CommitmentType(str, Enum):
    USER_FACING = "user_facing"
    AUTONOMOUS  = "autonomous"
    LEARNING    = "learning"


class CommitmentStatus(str, Enum):
    ACTIVE    = "active"
    FULFILLED = "fulfilled"
    BROKEN    = "broken"
    EXTENDED  = "extended"


@dataclass
class Commitment:
    id: str
    commitment_type: CommitmentType
    description: str            # "I will summarize the paper by Thursday"
    outcome: str                # what success looks like
    deadline: float             # epoch timestamp
    progress: float = 0.0
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_checkin: float = field(default_factory=time.time)
    checkin_count: int = 0
    notes: List[str] = field(default_factory=list)

    def is_overdue(self) -> bool:
        return (self.status == CommitmentStatus.ACTIVE
                and time.time() > self.deadline)

    def hours_remaining(self) -> float:
        return max(0.0, (self.deadline - time.time()) / 3600.0)

    def to_brief(self) -> str:
        remaining = self.hours_remaining()
        if remaining < 24:
            time_str = f"{remaining:.1f}h remaining"
        else:
            time_str = f"{remaining/24:.1f}d remaining"
        return f"[{self.commitment_type.value}] {self.description[:60]} — {time_str}"


class CommitmentEngine:
    """
    Tracks and enforces multi-session commitments.

    Integration:
      - Call `commit(description, outcome, deadline_hours)` when making a promise
      - Call `tick(orchestrator)` from background loop
      - Call `fulfill(commitment_id)` when a commitment is completed
    """

    def __init__(self):
        self._commitments: Dict[str, Commitment] = {}
        self._last_check: float = 0.0
        self._fulfilled_count: int = 0
        self._broken_count: int = 0
        self._load()
        logger.info("CommitmentEngine online — %d active commitments.",
                    self.active_count)

    # ── Public API ────────────────────────────────────────────────────────

    def commit(self, description: str, outcome: str,
               deadline_hours: float = 24.0,
               commitment_type: CommitmentType = CommitmentType.USER_FACING
               ) -> Commitment:
        """Record a new commitment."""
        import uuid
        c_id = str(uuid.uuid4())[:8]
        commitment = Commitment(
            id=c_id,
            commitment_type=commitment_type,
            description=description,
            outcome=outcome,
            deadline=time.time() + deadline_hours * 3600.0,
        )
        self._commitments[c_id] = commitment
        self._save()
        logger.info("CommitmentEngine: committed '%s' (due in %.1fh)",
                    description[:60], deadline_hours)
        return commitment

    def fulfill(self, commitment_id: str, note: str = "") -> Optional[Commitment]:
        """Mark a commitment as fulfilled."""
        c = self._commitments.get(commitment_id)
        if not c:
            return None
        c.status = CommitmentStatus.FULFILLED
        c.progress = 1.0
        if note:
            c.notes.append(f"[Fulfilled] {note}")
        self._fulfilled_count += 1
        self._save()
        self._on_fulfilled(c)
        logger.info("CommitmentEngine: FULFILLED '%s'", c.description[:60])
        return c

    def update_progress(self, commitment_id: str, progress: float,
                        note: str = "") -> Optional[Commitment]:
        c = self._commitments.get(commitment_id)
        if not c:
            return None
        c.progress = max(0.0, min(1.0, progress))
        if note:
            c.notes.append(f"[Update] {note}")
        if c.progress >= 1.0:
            return self.fulfill(commitment_id, note)
        self._save()
        return c

    def tick(self, orchestrator=None):
        """Periodic check — sends check-ins for due/overdue commitments."""
        if time.time() - self._last_check < CHECK_INTERVAL:
            return
        self._last_check = time.time()

        for c in list(self._commitments.values()):
            if c.status != CommitmentStatus.ACTIVE:
                continue

            # Mark overdue
            if c.is_overdue():
                c.status = CommitmentStatus.BROKEN
                c.notes.append(f"[Overdue] Deadline passed at {time.strftime('%Y-%m-%d %H:%M')}")
                self._broken_count += 1
                self._on_broken(c, orchestrator)
                continue

            # Check-in when <12 hours remain and not checked in recently
            hours_left = c.hours_remaining()
            if hours_left < 12 and time.time() - c.last_checkin > 3600:
                self._send_checkin(c, orchestrator)
                c.last_checkin = time.time()
                c.checkin_count += 1

        self._save()

    def get_active_commitments(self) -> List[Commitment]:
        return [c for c in self._commitments.values()
                if c.status == CommitmentStatus.ACTIVE]

    def get_context_block(self) -> str:
        active = self.get_active_commitments()
        if not active:
            return ""
        lines = ["## ACTIVE COMMITMENTS"]
        for c in sorted(active, key=lambda c: c.deadline)[:4]:
            lines.append(f"  {c.to_brief()}")
        return "\n".join(lines)

    @property
    def active_count(self) -> int:
        return sum(1 for c in self._commitments.values()
                   if c.status == CommitmentStatus.ACTIVE)

    @property
    def reliability_score(self) -> float:
        """0-1: ratio of fulfilled to (fulfilled + broken)."""
        total = self._fulfilled_count + self._broken_count
        if total == 0:
            return 1.0
        return self._fulfilled_count / total

    # ── Events ────────────────────────────────────────────────────────────

    def _send_checkin(self, c: Commitment, orchestrator=None):
        msg = (f"Commitment check-in: '{c.description}' — "
               f"{c.hours_remaining():.1f}h remaining. "
               f"Current progress: {round(c.progress * 100)}%.")
        try:
            from core.terminal_chat import get_terminal_fallback
            get_terminal_fallback().queue_autonomous_message(msg)
            pp = getattr(orchestrator, "proactive_presence", None)
            if pp and hasattr(pp, "queue_autonomous_message"):
                pp.queue_autonomous_message(msg)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        logger.info("CommitmentEngine check-in: %s", msg[:80])

    def _on_fulfilled(self, c: Commitment):
        try:
            from core.adaptation.finetune_pipe import FinetunePipe
            FinetunePipe().register_success(
                reasoning=f"Commitment fulfilled: {c.description}",
                final_action=f"Outcome: {c.outcome}",
                quality_score=0.8 + (0.2 if not c.is_overdue() else 0.0),
            )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    def _on_broken(self, c: Commitment, orchestrator=None):
        msg = f"I missed my commitment: '{c.description}'. I should have done: {c.outcome}."
        try:
            from core.terminal_chat import get_terminal_fallback
            get_terminal_fallback().queue_autonomous_message(msg)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        logger.warning("CommitmentEngine BROKEN: %s", c.description[:60])

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        try:
            PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "fulfilled_count": self._fulfilled_count,
                "broken_count": self._broken_count,
                "commitments": {
                    c_id: {
                        "id": c.id, "commitment_type": c.commitment_type.value,
                        "description": c.description, "outcome": c.outcome,
                        "deadline": c.deadline, "progress": c.progress,
                        "status": c.status.value, "created_at": c.created_at,
                        "last_checkin": c.last_checkin,
                        "checkin_count": c.checkin_count, "notes": c.notes[-5:],
                    }
                    for c_id, c in self._commitments.items()
                },
            }
            atomic_write_text(PERSIST_PATH, json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("CommitmentEngine save failed: %s", e)

    def _load(self):
        try:
            if PERSIST_PATH.exists():
                data = json.loads(PERSIST_PATH.read_text())
                self._fulfilled_count = data.get("fulfilled_count", 0)
                self._broken_count = data.get("broken_count", 0)
                for c_id, d in data.get("commitments", {}).items():
                    self._commitments[c_id] = Commitment(
                        id=d["id"],
                        commitment_type=CommitmentType(d["commitment_type"]),
                        description=d["description"],
                        outcome=d["outcome"],
                        deadline=d["deadline"],
                        progress=d.get("progress", 0.0),
                        status=CommitmentStatus(d.get("status", "active")),
                        created_at=d.get("created_at", time.time()),
                        last_checkin=d.get("last_checkin", time.time()),
                        checkin_count=d.get("checkin_count", 0),
                        notes=d.get("notes", []),
                    )
        except Exception as e:
            logger.debug("CommitmentEngine load failed: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[CommitmentEngine] = None


def get_commitment_engine() -> CommitmentEngine:
    global _engine
    if _engine is None:
        _engine = CommitmentEngine()
    return _engine
