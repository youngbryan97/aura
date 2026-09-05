"""core/dual_memory.py -- Compatibility Facade for DualMemorySystem

All actual implementation has been consolidated under the memory subsystem:
core/memory/dual_memory.py

This module re-exports all elements to ensure complete backward-compatibility.
"""
from __future__ import annotations

from core.memory.dual_memory import (
    Episode,
    EpisodicMemoryStore,
    SemanticFact,
    SemanticMemoryStore,
    DualMemorySystem,
    retrieve_memories_sync,
)

__all__ = [
    "Episode",
    "EpisodicMemoryStore",
    "SemanticFact",
    "SemanticMemoryStore",
    "DualMemorySystem",
    "retrieve_memories_sync",
]
