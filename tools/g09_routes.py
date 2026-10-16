"""How a G09 kept arm decides which kind's procedures answer a request: by words, by meaning, or both.

* ``words``: attempt 4's route (ProcedureBook.answer: distinct signatures; several kinds near enough answer
  only when all agree).
* ``meaning``: the kind her embedding finds (ProcedureBook.route_by_meaning, the rule of
  core/learning/kinds_by_meaning.nearest_kind), whose procedures read the request as it stands.
* ``words_then_meaning``: words, and meaning where words decline.

The embedding is her runtime's own (core/memory/vector_memory_engine.EmbeddingEngine), held for the run.
"""

from __future__ import annotations

from typing import Any

ROUTES = ("words", "meaning", "words_then_meaning")
_ENGINE: dict[str, Any] = {}


def her_query(text: str) -> Any:
    """Her embedding of a request as a query for the kind of problem it is; refuses if the model did not load."""
    from core.learning.kinds_by_meaning import TASK

    engine = _ENGINE.get("engine")
    if engine is None:
        from core.memory.vector_memory_engine import EmbeddingEngine

        engine = EmbeddingEngine()
        engine.embed_query("a probe", task=TASK)
        if getattr(engine, "_model", None) is None:
            raise RuntimeError("her embedding model did not load")
        _ENGINE["engine"] = engine
    return engine.embed_query(text, task=TASK)


def by_meaning(book: Any, request: str, query: Any = her_query) -> dict[str, Any] | None:
    """The receipt of the kind meant by ``request``, run on it as it stands; None when no kept kind is meant."""
    kind, similarity, _why = book.route_by_meaning(query(request))
    if kind is None or not book.procedures.get(kind):
        return None
    receipt = book._run_kind(kind, request, {"family": kind, "similarity": round(similarity, 4), "form": "meaning"})
    return receipt if receipt.get("answer") is not None else None


def kept_receipt(book: Any, request: str, route: str, query: Any = her_query) -> dict[str, Any] | None:
    """The receipt of the kept procedure that answers ``request`` by ``route``; None when none does."""
    if route not in ROUTES:
        raise ValueError(f"no route {route!r}")
    if route in ("words", "words_then_meaning"):
        receipt = book.answer(request)
        if receipt.get("answer") is not None:
            return receipt
        if route == "words":
            return None
    return by_meaning(book, request, query)
