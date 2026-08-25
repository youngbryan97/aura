"""Process-wide serialization for stateful third-party import frontiers.

Python serializes initialization of one module, but it permits unrelated
submodules to initialize concurrently.  Some ML packages expose attributes
through package-level lazy loaders which mutate shared package state while
their submodules import.  Aura starts voice and semantic-memory workers in
parallel, so those independent entry points can otherwise race inside the
same lazy package (observed with ``sentence_transformers`` and
``faster_whisper`` both entering ``huggingface_hub`` during boot).

Only import initialization is serialized.  Model construction and inference
remain independently scheduled under their existing model-lane owners.
"""

from __future__ import annotations

import importlib
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock

_THIRD_PARTY_IMPORT_LOCK = checked_lock(
    "runtime.third_party_imports",
    rank=LockRank.REGISTRY,
    reentrant=True,
)


def import_module_serialized(module_name: str) -> Any:
    """Import one third-party module under the process-wide init fence."""

    name = str(module_name or "").strip()
    if not name:
        raise ValueError("module_name must be non-empty")
    with _THIRD_PARTY_IMPORT_LOCK:
        return importlib.import_module(name)


def import_attribute_serialized(module_name: str, attribute: str) -> Any:
    """Import a module and resolve one required attribute atomically."""

    attr = str(attribute or "").strip()
    if not attr:
        raise ValueError("attribute must be non-empty")
    module = import_module_serialized(module_name)
    return getattr(module, attr)
