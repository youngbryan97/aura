"""Real retrieval into the reasoning workspace (CP239).

The integrated eval (CP238) proved the load-bearing half of the thesis:
facts placed in the workspace took the model from 0% to 56% on questions it
could not answer from memory. But those facts were PLANTED -- a fixture
handed the model perfectly-retrieved passages. That isolates "can the model
use retrieved knowledge" (yes) from "can the organ find it" (untested), and
the honest next step is to close that gap.

This adapter backs the ``RetrievalSource`` protocol with Aura's real memory
(``memory_facade.search_sync``), so the same factorial that scored 0->56%
on planted facts can be run on ACTUAL recall. It is the seam that makes the
integration live: the workspace stops being fed a fixture and starts being
fed what Aura actually knows and can find.

Two disciplines carry over from everything that bit us this session:

* **Retrieval quality is measured, not assumed.** A real organ returns
  noisy, partially-relevant passages, not gold facts. ``recall_at`` reports
  whether the answer-bearing fact was actually surfaced, so a disappointing
  integrated-eval score can be attributed to retrieval MISS vs reasoning
  FAILURE -- two very different problems with two different fixes.
* **Degrades honestly.** If memory is unavailable or empty the adapter
  returns nothing and says so; it never fabricates a passage to keep a
  score up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Learning.FacadeRetrieval")

FACADE_RETRIEVAL_SCHEMA = "aura.facade_retrieval.v1"


@dataclass
class FacadeRetrieval:
    """RetrievalSource backed by the live memory facade.

    ``facade`` is any object exposing ``search_sync(query, limit=...) ->
    list[dict|str]``. Kept duck-typed so a test can pass a stub and the live
    path can pass the real ``MemoryFacade`` without either importing the
    other.
    """

    facade: Any
    max_chars: int = 300
    calls: int = 0
    empty_results: int = 0
    passages_returned: int = 0

    def _record(self, exc: BaseException, action: str) -> None:
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "facade_retrieval", exc, severity="warning", action=action,
                enforce_failure_policy=False,
            )
        except Exception as recorder_exc:  # noqa: BLE001 - reporting must not raise
            # The degradation recorder is itself unavailable. Nothing further
            # can be done here, but a blind reporting path must not look
            # identical to a quiet one.
            logger.debug(
                "facade_retrieval could not record a degradation (%s: %s); the "
                "original was: %s",
                type(recorder_exc).__name__,
                recorder_exc,
                exc,
            )

    def _passage_text(self, item: Any) -> str:
        if isinstance(item, dict):
            text = str(item.get("content") or item.get("text") or "").strip()
        else:
            text = str(item or "").strip()
        # Bound passage length: an unbounded memory blob can crowd the
        # workspace and push the actual question out of the attention window,
        # which would look like a reasoning failure and be a plumbing one.
        return text[: self.max_chars]

    def retrieve(self, query: str, *, limit: int) -> list[str]:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        self.calls += 1
        search = getattr(self.facade, "search_sync", None)
        if search is None:
            # No fabrication: an unavailable organ returns nothing and the
            # ablation will correctly show retrieval as non-contributing.
            self.empty_results += 1
            return []
        try:
            raw = search(query, limit=limit)
        except Exception as exc:
            # A retrieval failure is a degradation, not a reason to invent
            # context. Record it (info: expected backpressure) and let the
            # score reflect reality rather than a fabricated passage.
            self._record(exc, "memory retrieval failed")
            self.empty_results += 1
            return []
        passages = [self._passage_text(item) for item in (raw or [])]
        passages = [p for p in passages if p]
        if not passages:
            self.empty_results += 1
        self.passages_returned += len(passages)
        return passages[:limit]

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": FACADE_RETRIEVAL_SCHEMA,
            "calls": self.calls,
            "empty_results": self.empty_results,
            "passages_returned": self.passages_returned,
            "hit_rate": round(
                1.0 - self.empty_results / self.calls, 4
            ) if self.calls else 0.0,
        }


def recall_at(
    passages: list[str], answer: str, *, k: int | None = None
) -> dict[str, Any]:
    """Did retrieval actually surface the answer-bearing fact?

    This separates the two failure modes an integrated-eval miss can have:
    the organ did not FIND the fact (retrieval problem -> better index /
    query), or it found it and the model did not USE it (reasoning problem
    -> the RLC workspace / training). Conflating them is how an integration
    project spins its wheels blaming the wrong component.
    """
    needle = str(answer).strip().lower()
    if not needle:
        raise ValueError("answer must be non-empty to score recall")
    window = passages if k is None else passages[:k]
    rank = None
    for index, passage in enumerate(window):
        if needle in str(passage).lower():
            rank = index
            break
    return {
        "schema": FACADE_RETRIEVAL_SCHEMA,
        "recalled": rank is not None,
        "rank": rank,
        "considered": len(window),
    }


@dataclass
class RetrievalAttribution:
    """Aggregate: of the questions missed, how many were retrieval's fault?

    Run alongside the integrated factorial to answer the question that
    decides what to fix next -- index/query, or reasoning/training.
    """

    total: int = 0
    retrieval_hits: int = 0
    solved_when_recalled: int = 0
    recalled_count: int = 0
    _misses_with_recall: int = 0

    def observe(self, *, recalled: bool, solved: bool) -> None:
        self.total += 1
        if recalled:
            self.recalled_count += 1
            self.retrieval_hits += 1
            if solved:
                self.solved_when_recalled += 1
            else:
                self._misses_with_recall += 1

    def report(self) -> dict[str, Any]:
        return {
            "schema": FACADE_RETRIEVAL_SCHEMA,
            "total": self.total,
            "retrieval_recall_rate": round(
                self.recalled_count / self.total, 4
            ) if self.total else 0.0,
            # Of the questions where the fact WAS retrieved, how often the
            # model then used it -- the pure reasoning number, retrieval
            # noise removed.
            "use_rate_when_recalled": round(
                self.solved_when_recalled / self.recalled_count, 4
            ) if self.recalled_count else 0.0,
            "reasoning_failures_despite_recall": self._misses_with_recall,
        }


__all__ = [
    "FACADE_RETRIEVAL_SCHEMA",
    "FacadeRetrieval",
    "RetrievalAttribution",
    "recall_at",
]
