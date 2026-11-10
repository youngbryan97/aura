"""The subject runtime's retriever, and the longest budget any of its phases is given.

Lifted whole out of `driver`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .driver import (
        SubjectRuntime,
    )


def _longest_phase_budget(runtime: SubjectRuntime) -> float:
    """The longest a phase may run before the harness abandons it."""
    from .driver import (
        PHASE_TIMEOUT,
    )

    budget = PHASE_TIMEOUT
    if getattr(runtime, "whole", False):
        kernel = getattr(runtime, "kernel", None)
        for phase in getattr(kernel, "_phases", []) or []:
            for priority in (False, True):
                try:
                    budget = max(budget, float(kernel._phase_timeout_seconds(type(phase).__name__, priority=priority)))
                except (AttributeError, TypeError, ValueError):
                    continue
    return budget


def _build_retriever(runtime: SubjectRuntime) -> Any:
    """The real retriever, over adapters that read her own state.

    The stores are her working memory, her retained long-term list and her
    world facts, which is where the live stores get their contents from. What
    is real here is the planning, the ontogenetic breadth decision and the
    merge; what is scaffolding is the adapters.
    """

    try:
        from core.memory.intentional_retrieval import IntentionalRetriever, MemoryStoreType
    except ImportError:  # pragma: no cover - the retriever is optional to the rest
        # not a failure: the comment says the retriever is optional, and
        # the caller plans without one.
        return None

    retriever = IntentionalRetriever()

    def working(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = runtime.state.cognition.working_memory[-limit:]
        return [{"content": str(row)[:240], "score": 0.5} for row in rows]

    def semantic(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = runtime.state.cold.long_term_memory[-limit:]
        return [{"content": str(row)[:240], "score": 0.4} for row in rows]

    def facts(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = list(runtime.state.world.facts.items())[-limit:]
        return [{"content": f"{k}={v}"[:240], "score": 0.6} for k, v in rows]

    retriever.register_store(MemoryStoreType.EPISODIC, working)
    retriever.register_store(MemoryStoreType.SEMANTIC, semantic)
    retriever.register_store(MemoryStoreType.RECEIPT, facts)
    return retriever
