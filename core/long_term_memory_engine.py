"""core/long_term_memory_engine.py -- Compatibility Facade for LongTermMemoryEngine

All actual implementation has been consolidated under the memory subsystem:
core/memory/long_term_memory_engine.py

This module re-exports all elements to ensure complete backward-compatibility.
"""
from __future__ import annotations

from core.memory.long_term_memory_engine import (
    TaggedMemory,
    LongTermMemoryEngine,
    get_long_term_memory_engine,
)

__all__ = [
    "TaggedMemory",
    "LongTermMemoryEngine",
    "get_long_term_memory_engine",
]
