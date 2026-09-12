"""Letting a memory go, and saying how much is left.

Evaporation, deletion by id and in bulk, and the counts that make the vault
legible from outside. Every one of them is the locked half of a public method
that takes the lock — the split is deliberate, so a caller that already holds
it cannot deadlock against itself.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.BlackHoleVault")


class _LetsMemoriesGo:
    """Lifted whole from BlackHoleVault; see black_hole_vault.py."""

    def count(self) -> int:
        """Return the authoritative number of records in the event horizon."""
        with self._mutation_guard():
            return len(self.memories)

    @property
    def total_mass_kb(self) -> float:
        """Returns the current mass of the vault in KB."""
        total_bytes = len(json.dumps(self.memories).encode()) if self.memories else 0
        return round(total_bytes / 1024, 2)

    def _evaporate(self):
        if not self.memories: return
        
        # Notify Mycelium of qualitative shift (Evolution 8)
        try:
            from core.container import ServiceContainer
            mycelium = ServiceContainer.get("mycelium", default=None)
            if mycelium:
                mycelium.log_hypha(
                    "memory", "vault", "EVAPORATION: Qualitative shift in history."
                )
                mycelium.pulse_hypha("memory", "vault", success=True)
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('black_hole_vault', _e)
            logger.debug('Ignored Exception in black_hole_vault.py: %s', _e)

        keep_count = self._policy().keep_count(len(self.memories))
        self.memories = self._select_semantically_important(self.memories, keep_count)
        self._save_vault()

    def clear(self):
        """Standard interface: Reset the vault."""
        with self._mutation_guard():
            return self._clear_locked()

    def _clear_locked(self):
        self.memories = []
        self._dirty = True
        self._save_vault()
        logger.info("BlackHoleVault: Event horizon cleared.")

    def delete(self, ids: List[str]):
        """Standard interface: Delete memories by ID."""
        with self._mutation_guard():
            return self._delete_locked(ids)

    def _delete_locked(self, ids: List[str]):
        id_set = {str(memory_id) for memory_id in ids}
        self.memories = [
            m
            for m in self.memories
            if self._memory_id(m) not in id_set and str(m.get("created")) not in id_set
        ]
        self._dirty = True
        self._save_vault()
        logger.info("BlackHoleVault: Deleted %d memories.", len(ids))

    def delete_memories(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> int:
        """VectorMemory-compatible deletion shim used by episodic pruning."""
        with self._mutation_guard():
            return self._delete_memories_locked(
                ids=ids,
                filter_metadata=filter_metadata,
                **kwargs,
            )

    def _delete_memories_locked(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> int:
        id_set = {str(memory_id) for memory_id in (ids or []) if str(memory_id)}
        metadata_filters = dict(filter_metadata or {})
        if not id_set and not metadata_filters:
            return 0

        normalized_filters: Dict[str, set[str]] = {}
        for key, raw_value in metadata_filters.items():
            if isinstance(raw_value, (list, tuple, set)):
                values = {str(value) for value in raw_value}
            else:
                values = {str(raw_value)}
            normalized_filters[str(key)] = values

        def _matches(memory: Dict[str, Any]) -> bool:
            if self._memory_id(memory) in id_set or str(memory.get("created")) in id_set:
                return True
            metadata = memory.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                return False
            return any(
                str(metadata.get(key)) in accepted
                for key, accepted in normalized_filters.items()
            )

        before = len(self.memories)
        self.memories = [memory for memory in self.memories if not _matches(memory)]
        deleted = before - len(self.memories)
        if deleted:
            self._dirty = True
            self._save_vault()
            logger.info("BlackHoleVault: Deleted %d memories via compatibility filter.", deleted)
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Standard interface: Return collection statistics."""
        return {
            "total_vectors": self.count(),
            "total_mass_kb": self.total_mass_kb,
            "max_vectors": self._policy().max_items,
            "retention_policy": self._policy().to_dict(),
            "engine": "black_hole_vault",
            "status": "active"
        }
