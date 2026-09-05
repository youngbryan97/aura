"""core/memory/memory_court.py
Vets candidate memory updates through truth validation and conflict resolution.
"""
import logging
from typing import Any

from core.memory.conflict_resolution import MemoryConflictResolver
from core.memory.source_provenance import SourceProvenanceResolver

logger = logging.getLogger("Memory.MemoryCourt")


class MemoryCourt:
    """Canonical memory validator checking assertions before database injection."""

    def __init__(self):
        self.provenance = SourceProvenanceResolver()
        self.resolver = MemoryConflictResolver()

    async def vet_fact(self, key: str, value: Any, origin: str, existing_facts: dict[str, Any]) -> dict[str, Any] | None:
        """Validates a new fact statement. Returns resolved fact if passed, or None to reject."""
        confidence = self.provenance.resolve_confidence(origin)
        
        # Guard: Reject facts with extremely low confidence
        if confidence < 0.25:
            logger.warning("Rejected low-confidence candidate memory statement: %s (%s)", key, origin)
            return None

        new_fact = {
            "key": key,
            "value": value,
            "origin": origin,
            "confidence": confidence
        }

        if key in existing_facts:
            existing = existing_facts[key]
            resolved = self.resolver.resolve_conflict(existing, new_fact)
            logger.info("Memory court resolved fact conflict for key: %s", key)
            return resolved

        return new_fact
