"""One shape for a memory result, whatever store it came from.

Lifted whole out of `memory_facade`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import Any


class _NormalisesWhatItReturns:
    """Lifted whole out of MemoryFacade; see memory_facade.py."""

    def _normalize_memory_result(
        self,
        *,
        content: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str = "",
        score: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": memory_id,
            "text": content,
            "content": content,
            "metadata": self._safe_metadata(metadata),
        }
        if score is not None:
            payload["score"] = score
        return payload

    @classmethod
    def _normalize_source_label(cls, raw: Any) -> str:
        return str(raw or "").strip().lower().replace("-", "_")

    @staticmethod
    def _normalize_search_limit(
        limit: int | None = None,
        *,
        top_k: int | None = None,
        default: int = 5,
    ) -> int:
        raw_limit = top_k if top_k is not None else limit
        try:
            return max(1, min(100, int(raw_limit if raw_limit is not None else default)))
        except (TypeError, ValueError, OverflowError):
            return default

