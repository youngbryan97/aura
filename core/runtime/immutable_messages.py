"""core/runtime/immutable_messages.py

Immutable message protocol for inter-subsystem communication.

All messages flowing through the event bus, actor bus, or shared memory bus
should use these frozen dataclasses.  An actor should never be able to mutate
a message that another actor is currently reading.

Usage:
    msg = CognitiveMessage(
        source="curiosity_engine",
        content="I want to explore this topic",
        domain="exploration",
        priority=0.6,
    )
    # msg.content = "mutated"  # raises FrozenInstanceError
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class CognitiveMessage:
    """Immutable message for inter-subsystem communication."""
    source: str
    content: str
    domain: str = "general"
    priority: float = 0.5
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    metadata: Tuple[Tuple[str, Any], ...] = ()  # frozen-compatible dict alternative

    def with_metadata(self, **kwargs: Any) -> "CognitiveMessage":
        """Return a new message with additional metadata (immutable update)."""
        existing = dict(self.metadata)
        existing.update(kwargs)
        return CognitiveMessage(
            source=self.source,
            content=self.content,
            domain=self.domain,
            priority=self.priority,
            trace_id=self.trace_id,
            timestamp=self.timestamp,
            metadata=tuple(existing.items()),
        )


@dataclass(frozen=True)
class ActionReceipt:
    """Immutable forensic receipt for any consequential action.

    Every action that passes through the authority system produces one of
    these.  The receipt links the Will decision, subsystem source, action
    domain, and outcome into a single auditable record.
    """
    receipt_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    will_receipt_id: str = ""         # links to WillDecision.receipt_id
    source: str = ""                  # which subsystem requested this
    domain: str = ""                  # action domain (tool, memory, initiative, etc.)
    action_summary: str = ""          # what was requested (truncated)
    outcome: str = ""                 # approved, refused, deferred, constrained
    reason: str = ""                  # why this outcome
    constraints: Tuple[str, ...] = () # any constraints applied
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    # Provenance chain
    substrate_receipt_id: str = ""
    executive_intent_id: str = ""
    capability_token_id: str = ""


@dataclass(frozen=True)
class SubsystemEvent:
    """Immutable event for the event bus / mycelium network."""
    event_type: str
    source: str
    data: Tuple[Tuple[str, Any], ...] = ()
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    @property
    def data_dict(self) -> Dict[str, Any]:
        """Read-only dict view of the data."""
        return dict(self.data)
