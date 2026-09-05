"""core/memory_compaction_patch.py -- Compatibility Facade for MemoryCompactionPatch

All actual implementation has been consolidated under the memory subsystem:
core/memory/memory_compaction_patch.py

This module re-exports all elements to ensure complete backward-compatibility.
"""
from __future__ import annotations

from core.memory.memory_compaction_patch import (
    COMPACTION_THRESHOLD,
    MAX_RAW_TURNS,
    compact_if_needed,
    patch_memory_compaction,
    _patched_memory_consolidation_execute,
)

__all__ = [
    "COMPACTION_THRESHOLD",
    "MAX_RAW_TURNS",
    "compact_if_needed",
    "patch_memory_compaction",
    "_patched_memory_consolidation_execute",
]
