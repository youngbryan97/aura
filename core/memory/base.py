"""core/memory/base.py
────────────────────
Shared memory primitives imported by all memory subsystems.

BEFORE: MemoryEvent was defined separately in both:
  - core/memory/atomic_storage.py
  - core/memory/sqlite_storage.py

  These two definitions drifted: atomic_storage made `event_type` optional
  with a default of "undefined"; sqlite_storage made it required.
  Any code that created a MemoryEvent from one and passed it to the other
  could silently corrupt or fail.

AFTER: One definition here. Both modules import from here.
  Update atomic_storage.py line ~35: remove MemoryEvent definition, add:
      from core.memory.base import MemoryEvent, MemoryType
  Update sqlite_storage.py line ~24: same.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# ── MemoryType ───────────────────────────────────────────────

class MemoryType(Enum):
    """Semantic category of a memory store."""

    EPISODIC  = "episodic"
    SEMANTIC  = "semantic"
    GOAL      = "goals"
    KNOWLEDGE = "knowledge"
    SKILL     = "skills"


# ── MemoryEvent ──────────────────────────────────────────────

@dataclass
class MemoryEvent:
    """Structured unit of episodic memory.

    Field contract (resolves the atomic_storage / sqlite_storage divergence):
      - event_type: always required; represents *what happened* ("chat", "skill_exec", …)
      - timestamp:  defaults to now if omitted — never None after __post_init__
      - goal:       the agent's stated intention, if any
      - outcome:    result dict or summary string
      - cost:       computational or resource cost (≥ 0)
      - metadata:   arbitrary extra data (never None after __post_init__)
    """

    event_type: str
    timestamp:  float                              = field(default_factory=time.time)
    goal:       str | None                      = None
    outcome:    str | dict[str, Any] | None = None
    cost:       float                              = 0.0
    metadata:   dict[str, Any]                     = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sanitise timestamp
        if not self.timestamp or self.timestamp <= 0:
            self.timestamp = time.time()
        # Sanitise cost
        if self.cost < 0:
            raise ValueError(f"MemoryEvent.cost must be ≥ 0, got {self.cost!r}")
        # Sanitise metadata
        if self.metadata is None:
            self.metadata = {}

    # ── Serialisation ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (timestamp aliased to 't' for compactness)."""
        d = asdict(self)
        d["t"] = d.pop("timestamp")
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEvent:
        """Reconstruct from a dict.  Handles both the legacy 't' key
        and the canonical 'timestamp' key.
        """
        d = dict(data)
        if "t" in d and "timestamp" not in d:
            d["timestamp"] = d.pop("t")
        # Drop unknown fields gracefully (future-proofing)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    # ── Helpers ──────────────────────────────────────────────

    @property
    def is_failure(self) -> bool:
        """Quick check: did this event represent a failure?"""
        if isinstance(self.outcome, dict):
            return not self.outcome.get("ok", True)
        if isinstance(self.outcome, str):
            low = self.outcome.lower()
            return "fail" in low or "error" in low
        return False

    def is_high_latency(self, threshold_s: float = 10.0) -> bool:
        """Returns True if cost exceeds the latency threshold."""
        return self.cost > threshold_s

    def __repr__(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"MemoryEvent(type={self.event_type!r}, ts={ts}, goal={self.goal!r})"
