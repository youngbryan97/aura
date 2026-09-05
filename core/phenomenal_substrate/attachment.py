from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any

from .maths import clamp
from .types import AttachmentEvent

@dataclass
class AttachmentState:
    person_key: str
    trust: float = 0.50
    care: float = 0.00
    familiarity: float = 0.00
    rupture: float = 0.00
    repair_history: float = 0.00
    attachment: float = 0.00
    last_evidence_id: str = ""

    def recalc(self) -> None:
        self.attachment = clamp(
            0.30 * self.trust +
            0.30 * self.care +
            0.25 * self.familiarity +
            0.15 * self.repair_history -
            0.25 * self.rupture
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AttachmentSystem:
    """
    Evidence-locked bond machinery.

    It turns relationship events into action-relevant internal state.
    """
    def __init__(self) -> None:
        self.people: Dict[str, AttachmentState] = {}
        self.events: List[AttachmentEvent] = []

    def state_for(self, person_key: str) -> AttachmentState:
        return self.people.setdefault(person_key, AttachmentState(person_key=person_key))

    def record(self, event: AttachmentEvent) -> AttachmentState:
        if not event.evidence_id:
            raise ValueError("AttachmentEvent requires evidence_id")
        state = self.state_for(event.person_key)
        state.trust = clamp(state.trust + event.trust_delta - 0.35 * event.rupture_delta + 0.20 * event.repair_delta)
        state.care = clamp(state.care + event.care_delta + 0.10 * event.repair_delta)
        state.familiarity = clamp(state.familiarity + event.familiarity_delta)
        state.rupture = clamp(state.rupture + event.rupture_delta - event.repair_delta)
        state.repair_history = clamp(state.repair_history + event.repair_delta)
        state.last_evidence_id = event.evidence_id
        state.recalc()
        self.events.append(event)
        return state

    def recent_evidence(self, person_key: str, limit: int = 5) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self.events if e.person_key == person_key][-limit:]
