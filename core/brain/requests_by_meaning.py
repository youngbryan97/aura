"""Her embedding model reading requests by meaning, for the tool hand-off (core/intent/capability_by_meaning.py).

The index of meanings is in core.intent; holding the embedding model is the
runtime's, so the two meet here.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from core.intent.capability_by_meaning import TASK, Meanings

logger = logging.getLogger("Aura.CapabilityByMeaning")

__all__ = ["meant_capability", "served"]


_SHARED: dict[str, Meanings] = {}


def _the_meanings() -> Meanings | None:
    if "it" in _SHARED:
        return _SHARED["it"]
    try:
        from core.memory.embedding_runtime import (
            acquire_shared_embedding_engine,
            shared_embedding_runtime_snapshot,
        )
        from core.runtime.state_ownership import state_root

        # Only an embedding model the runtime already holds: a turn does not wait on loading one.
        if not shared_embedding_runtime_snapshot().get("engine_live"):
            return None

        engine = acquire_shared_embedding_engine("capability_by_meaning")

        def embed(texts: list[str]) -> np.ndarray:
            return np.asarray(engine.embed_batch(texts), dtype=np.float32)

        def embed_query(text: str) -> np.ndarray:
            return np.asarray(engine.embed_query(text, task=TASK), dtype=np.float32)

        _SHARED["it"] = Meanings(embed, state_root() / "capability_meanings.json", embed_query)
    except Exception as why:  # noqa: BLE001 - no embedding model here: no reading by meaning, the words still route
        logger.info("no embedding model to read requests by meaning: %s", why)
        return None
    return _SHARED["it"]


def meant_capability(text: str, skills: Mapping[str, Any]) -> str | None:
    """The capability ``text`` means among ``skills``, by her embedding of it; None when none is clearly meant or there is no model."""
    meanings = _the_meanings()
    if meanings is None:
        return None
    try:
        return meanings.which(text, skills)
    except Exception as why:  # noqa: BLE001 - a reading that fails is no reading; the words still route
        logger.info("a request could not be read by meaning: %s", why)
        return None


def served(text: str, name: str) -> None:
    """A capability came back done for this request: remember it as one of its meanings."""
    meanings = _the_meanings()
    if meanings is not None:
        meanings.learned(text, name)
