"""core/memory/commitment_ledger.py
Durable register of commitments, agreements, and preferences made by Aura.
"""
from dataclasses import dataclass, field
import time
from typing import List, Dict, Any


@dataclass
class Commitment:
    id: str
    description: str
    target_person: str
    created_at: float = field(default_factory=time.time)
    fulfilled: bool = False
    fulfilled_at: float = 0.0


class CommitmentLedger:
    """Ledger tracking active social commitments and operational tasks."""

    def __init__(self):
        self._commitments: List[Commitment] = []

    def record_commitment(self, id: str, description: str, person: str = "Bryan") -> None:
        self._commitments.append(Commitment(id=id, description=description, target_person=person))

    def fulfill_commitment(self, id: str) -> None:
        for c in self._commitments:
            if c.id == id and not c.fulfilled:
                c.fulfilled = True
                c.fulfilled_at = time.time()

    def get_commitments(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": c.id,
                "description": c.description,
                "person": c.target_person,
                "fulfilled": c.fulfilled
            }
            for c in self._commitments
        ]
