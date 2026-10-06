"""Intentional retrieval: task-driven store selection + weighted merge over the taxonomy."""
from __future__ import annotations

import pytest

from core.memory.intentional_retrieval import (
    IntentionalRetriever,
    MemoryStoreType,
    RetrievalIntent,
    get_intentional_retriever,
)

_T = MemoryStoreType


@pytest.fixture
def retriever():
    return IntentionalRetriever()


# ── planning: the right stores for the kind of task ──────────────────────────

def test_debug_plan_prioritizes_failure_and_tools(retriever):
    plan = retriever.plan(RetrievalIntent("login crashes", kind="debug"))
    top = list(plan.weights)[:3]
    assert _T.FAILURE.value in top
    assert _T.PROCEDURAL.value in plan.weights or _T.TOOL.value in plan.weights


def test_converse_plan_prioritizes_social(retriever):
    plan = retriever.plan(RetrievalIntent("how's it going", kind="converse"))
    assert list(plan.weights)[0] == _T.SOCIAL.value


def test_recall_fact_plan_prioritizes_semantic(retriever):
    plan = retriever.plan(RetrievalIntent("what's my timezone", kind="recall_fact"))
    assert list(plan.weights)[0] == _T.SEMANTIC.value


def test_risk_sensitive_boosts_failure_value_receipt(retriever):
    safe = retriever.plan(RetrievalIntent("delete config", kind="general"))
    risky = retriever.plan(RetrievalIntent("delete config", kind="general", risk_sensitive=True))
    for store in (_T.FAILURE.value, _T.VALUE.value, _T.RECEIPT.value):
        assert risky.weights.get(store, 0) > safe.weights.get(store, 0)


def test_time_horizon_now_boosts_world_state(retriever):
    plan = retriever.plan(RetrievalIntent("what's happening", kind="general", time_horizon="now"))
    assert _T.WORLD_STATE.value in plan.weights


def test_long_horizon_boosts_semantic_and_project(retriever):
    plan = retriever.plan(RetrievalIntent("where is this project going", kind="general",
                                          time_horizon="long"))
    assert _T.PROJECT.value in plan.weights
    assert _T.AUTOBIOGRAPHY.value in plan.weights


def test_plan_has_human_readable_rationale(retriever):
    plan = retriever.plan(RetrievalIntent("ship it", kind="act_irreversible", risk_sensitive=True))
    assert any("risk-sensitive" in r for r in plan.rationale)
    assert plan.to_dict()["kind"] == "act_irreversible"


def test_allocations_are_assigned_to_selected_stores(retriever):
    plan = retriever.plan(RetrievalIntent("plan the sprint", kind="plan", limit=8))
    assert set(plan.allocations) == set(plan.weights)
    assert all(v >= 2 for v in plan.allocations.values())


# ── retrieval: routing, weighting, merge, fault isolation ───────────────────

def test_retrieve_only_queries_registered_stores(retriever):
    retriever.register_store(_T.SOCIAL, lambda q, n: ["bryan likes concise answers"])
    res = retriever.retrieve(RetrievalIntent("chat", kind="converse"))
    assert res.stores_queried == [_T.SOCIAL.value]
    assert _T.EPISODIC.value in res.stores_missing  # selected but not registered
    assert res.hits and "concise" in res.hits[0].content


def test_higher_weight_store_outranks_lower_for_equal_raw_score(retriever):
    # SOCIAL weight (1.0 for converse) beats AUTOBIOGRAPHY (0.4) at equal raw rank.
    retriever.register_store(_T.SOCIAL, lambda q, n: [{"content": "social fact", "score": 0.8}])
    retriever.register_store(_T.AUTOBIOGRAPHY, lambda q, n: [{"content": "auto fact", "score": 0.8}])
    res = retriever.retrieve(RetrievalIntent("chat", kind="converse"))
    assert res.hits[0].content == "social fact"
    assert res.hits[0].store_type == _T.SOCIAL.value


def test_a_broken_store_does_not_sink_retrieval(retriever):
    def _boom(q, n):
        error = RuntimeError("store on fire")
        raise error

    retriever.register_store(_T.SOCIAL, _boom)
    retriever.register_store(_T.EPISODIC, lambda q, n: ["still here"])
    res = retriever.retrieve(RetrievalIntent("chat", kind="converse"))
    assert _T.SOCIAL.value in res.stores_missing
    assert any(h.content == "still here" for h in res.hits)


def test_results_are_deduped_keeping_strongest(retriever):
    retriever.register_store(_T.SEMANTIC, lambda q, n: [{"content": "dup", "score": 0.9}])
    retriever.register_store(_T.EPISODIC, lambda q, n: [{"content": "dup", "score": 0.5}])
    res = retriever.retrieve(RetrievalIntent("x", kind="recall_fact"))
    dups = [h for h in res.hits if h.content == "dup"]
    assert len(dups) == 1


def test_normalizes_strings_dicts_and_objects(retriever):
    class _Obj:
        content = "from object"
        score = 0.7

    retriever.register_store(_T.SEMANTIC, lambda q, n: ["a string"])
    retriever.register_store(_T.EPISODIC, lambda q, n: [{"text": "a dict"}])
    retriever.register_store(_T.AUTOBIOGRAPHY, lambda q, n: [_Obj()])
    res = retriever.retrieve(RetrievalIntent("x", kind="recall_fact", limit=10))
    contents = {h.content for h in res.hits}
    assert {"a string", "a dict", "from object"} <= contents


def test_limit_is_respected(retriever):
    retriever.register_store(_T.SEMANTIC, lambda q, n: [f"fact {i}" for i in range(50)])
    res = retriever.retrieve(RetrievalIntent("x", kind="recall_fact", limit=5))
    assert len(res.hits) <= 5


def test_query_defaults_to_task(retriever):
    seen = {}
    retriever.register_store(_T.SEMANTIC, lambda q, n: seen.setdefault("q", q) and [])
    retriever.retrieve(RetrievalIntent("remember the milk", kind="recall_fact"))
    assert seen["q"] == "remember the milk"


# ── default wiring + singleton ──────────────────────────────────────────────

def test_wire_default_stores_is_best_effort(retriever):
    wired = retriever.wire_default_stores()  # may wire some, must not raise
    assert isinstance(wired, list)


def test_singleton_is_stable():
    assert get_intentional_retriever() is get_intentional_retriever()


def test_the_reference_corpus_searches_within_the_callers_deadline(monkeypatch):
    """LIVE 2026-10-06: every background recall took 5.0 s.

    The reference store searched the 7M-page corpus with its 5 s backstop, and
    the any-term fallback on a phrase like "Running a self-integrity scan"
    ran to it and found nothing. The ceiling belongs to the caller's lane.
    """
    from core.knowledge import local_corpus
    from core.memory.intentional_retrieval import IntentionalRetriever, RetrievalIntent

    asked = []

    class _Corpus:
        def has_documents(self):
            return True

        def search(self, query, limit=5, **kwargs):
            asked.append(kwargs.get("deadline_s"))
            return []

    monkeypatch.setattr(local_corpus, "get_local_corpus_store", lambda: _Corpus())
    retriever = IntentionalRetriever()
    retriever.wire_default_stores()
    retriever.retrieve(
        RetrievalIntent(task="what the paper says", kind="recall_fact", reference_deadline_s=0.25)
    )
    retriever.retrieve(RetrievalIntent(task="what the paper says", kind="recall_fact"))
    assert asked[:1] == [0.25]
    assert asked[1:] == [None]
