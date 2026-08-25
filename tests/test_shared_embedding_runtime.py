from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from core.memory.embedding_runtime import SharedEmbeddingRuntime


@dataclass
class _Engine:
    close_count: int = 0
    marker: str = "shared"
    embed_count: int = 0

    def close(self) -> None:
        self.close_count += 1

    def embed(self, _text: str):
        self.embed_count += 1
        return [0.0, 1.0, 0.0]


def test_all_owners_share_one_engine_until_the_last_release() -> None:
    created: list[_Engine] = []

    def factory() -> _Engine:
        engine = _Engine()
        created.append(engine)
        return engine

    runtime = SharedEmbeddingRuntime(factory)
    memory = runtime.acquire("vector-memory")
    rag = runtime.acquire("rag")
    relevance = runtime.acquire("evidence-relevance")

    assert len(created) == 1
    assert memory.marker == rag.marker == relevance.marker == "shared"
    assert runtime.snapshot() == {
        "engine_live": True,
        "lease_count": 3,
        "owners": ("evidence-relevance", "rag", "vector-memory"),
    }

    memory.close()
    memory.close()
    rag.close()
    assert created[0].close_count == 0
    assert relevance.marker == "shared"

    relevance.close()
    assert created[0].close_count == 1
    assert runtime.snapshot()["engine_live"] is False
    with pytest.raises(RuntimeError, match="shared_embedding_lease_released"):
        _ = relevance.marker


def test_new_owner_after_full_release_gets_a_fresh_engine() -> None:
    created: list[_Engine] = []

    def factory() -> _Engine:
        engine = _Engine(marker=f"engine-{len(created) + 1}")
        created.append(engine)
        return engine

    runtime = SharedEmbeddingRuntime(factory)
    first = runtime.acquire("first")
    assert first.marker == "engine-1"
    first.close()

    second = runtime.acquire("second")
    assert second.marker == "engine-2"
    assert [engine.close_count for engine in created] == [1, 0]
    second.close()
    assert [engine.close_count for engine in created] == [1, 1]


def test_runtime_close_invalidates_all_leases_and_closes_once() -> None:
    engine = _Engine()
    runtime = SharedEmbeddingRuntime(lambda: engine)
    left = runtime.acquire("left")
    right = runtime.acquire("right")

    runtime.close()
    runtime.close()

    assert engine.close_count == 1
    assert runtime.snapshot() == {
        "engine_live": False,
        "lease_count": 0,
        "owners": (),
    }
    with pytest.raises(RuntimeError, match="shared_embedding_lease_released"):
        _ = left.marker
    with pytest.raises(RuntimeError, match="shared_embedding_lease_released"):
        _ = right.marker


def test_process_prewarm_is_idempotent_and_retained_until_close(monkeypatch) -> None:
    import core.memory.embedding_runtime as embedding_runtime

    engine = _Engine()
    runtime = SharedEmbeddingRuntime(lambda: engine)
    monkeypatch.setattr(embedding_runtime, "_RUNTIME", runtime)
    monkeypatch.setattr(embedding_runtime, "_PREWARM_LEASE", None)

    first = embedding_runtime.prewarm_shared_embedding_runtime()
    second = embedding_runtime.prewarm_shared_embedding_runtime()

    assert first["vector_dimensions"] == 3
    assert second["lease_count"] == 1
    assert runtime.snapshot()["owners"] == ("runtime-prewarm",)
    assert engine.embed_count == 2

    embedding_runtime.close_shared_embedding_runtime()
    assert engine.close_count == 1
    assert runtime.snapshot()["engine_live"] is False


def test_evidence_prewarm_batches_the_fixed_document_cohort_once(monkeypatch) -> None:
    import core.cognition.evidence_relevance as evidence_relevance

    class _BatchEngine:
        def __init__(self) -> None:
            self._model = object()
            self.batch_calls: list[tuple[str, ...]] = []
            self.document_calls: list[str] = []
            self.query_calls: list[str] = []

        def _checkout_model(self):
            return object()

        def _return_model(self) -> None:
            return None

        def embed_batch(self, texts):
            cohort = tuple(texts)
            self.batch_calls.append(cohort)
            return [[float(index + 1), 1.0] for index, _text in enumerate(cohort)]

        def embed_query(self, text, task=None):
            self.query_calls.append(f"{task}:{text}")
            return [1.0, 1.0]

        def embed(self, text):
            self.document_calls.append(text)
            return [1.0, 1.0]

    engine = _BatchEngine()
    monkeypatch.setattr(evidence_relevance, "_embedder", lambda: engine)
    monkeypatch.setattr(evidence_relevance, "_ANCHOR_CACHE", {})
    monkeypatch.setattr(evidence_relevance, "_REQUEST_CACHE", {})

    receipt = evidence_relevance.prewarm_evidence_relevance()

    assert len(engine.batch_calls) == 1
    assert len(engine.batch_calls[0]) == receipt["encoded_documents"]
    assert receipt["encoded_documents"] < sum(
        row["concept_vectors"] + row["baseline_vectors"]
        for row in receipt["families"].values()
    )
    assert engine.document_calls == ["How are you feeling today?"]
    assert engine.query_calls == []


def test_evidence_anchor_batch_cardinality_failure_never_misbinds_vectors(monkeypatch) -> None:
    import core.cognition.evidence_relevance as evidence_relevance

    class _ShortBatchEngine:
        def embed_batch(self, _texts):
            return [[1.0, 0.0]]

        def embed(self, text):
            return [float(len(text)), 1.0]

    monkeypatch.setattr(evidence_relevance, "_embedder", lambda: _ShortBatchEngine())

    vectors = evidence_relevance._embed_documents(("first", "second"))

    assert vectors == [[5.0, 1.0], [6.0, 1.0]]


def test_server_prewarm_waits_for_cortex_readiness(monkeypatch) -> None:
    import core.cognition.evidence_relevance as evidence_relevance_module
    import core.consciousness.unified_self as unified_self_module
    import core.memory.embedding_runtime as embedding_runtime
    import core.memory.profile_manager as profile_manager_module
    import core.self.self_condition as self_condition_module
    import interface.chat_dependencies as chat_dependencies_module
    from interface import server

    class _Gate:
        ready_events: list[tuple[bool, str]] = []

        def get_conversation_status(self):
            return {"conversation_ready": True}

        def set_chat_dependencies_ready(self, ready, *, blocker=""):
            self.ready_events.append((bool(ready), str(blocker)))

    gate = _Gate()

    calls: list[str] = []
    monkeypatch.setattr(
        server.ServiceContainer,
        "get",
        classmethod(
            lambda _cls, key, default=None: gate
            if key == "inference_gate"
            else default
        ),
    )
    monkeypatch.setattr(server, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        embedding_runtime,
        "prewarm_shared_embedding_runtime",
        lambda: calls.append("prewarmed")
        or {"vector_dimensions": 1024, "lease_count": 1},
    )
    monkeypatch.setattr(
        evidence_relevance_module,
        "prewarm_evidence_relevance",
        lambda: calls.append("evidence_routing") or {"elapsed_ms": 4.0},
    )
    async def _profile():
        calls.append("profile")
        return object()

    async def _self():
        calls.append("unified_self")
        return object()

    monkeypatch.setattr(
        profile_manager_module.ProfileManager,
        "get_instance",
        _profile,
    )
    monkeypatch.setattr(unified_self_module, "get_unified_self", _self)
    monkeypatch.setattr(
        self_condition_module,
        "build_self_condition_projection",
        lambda: type("Projection", (), {"evidence_id": "condition-proof"})(),
    )
    monkeypatch.setattr(
        chat_dependencies_module,
        "materialize_foreground_chat_dependencies",
        lambda: calls.append("foreground_services") or {"skill_count": 87},
    )

    asyncio.run(
        server._prewarm_chat_dependencies_after_cortex_ready(
            readiness_timeout_s=0.1,
            poll_interval_s=0.01,
        )
    )

    assert set(calls) == {
        "prewarmed",
        "profile",
        "unified_self",
        "foreground_services",
        "evidence_routing",
    }
    assert gate.ready_events == [
        (False, "chat_dependencies_warming"),
        (True, ""),
    ]


def test_server_prewarm_binds_gate_that_registers_after_lifespan(monkeypatch) -> None:
    import core.cognition.evidence_relevance as evidence_relevance_module
    import core.consciousness.unified_self as unified_self_module
    import core.memory.embedding_runtime as embedding_runtime
    import core.memory.profile_manager as profile_manager_module
    import core.self.self_condition as self_condition_module
    import interface.chat_dependencies as chat_dependencies_module
    from interface import server

    class _LateGate:
        def __init__(self):
            self.ready_events: list[tuple[bool, str]] = []

        def get_cortex_readiness_status(self):
            return {"conversation_ready": True}

        def set_chat_dependencies_ready(self, ready, *, blocker=""):
            self.ready_events.append((bool(ready), str(blocker)))

    gate = _LateGate()
    lookups = 0

    def _lookup(_cls, key, default=None):
        nonlocal lookups
        if key != "inference_gate":
            return default
        lookups += 1
        return None if lookups == 1 else gate

    async def _ready_object():
        return object()

    monkeypatch.setattr(server.ServiceContainer, "get", classmethod(_lookup))
    monkeypatch.setattr(server, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        embedding_runtime,
        "prewarm_shared_embedding_runtime",
        lambda: {"vector_dimensions": 384, "lease_count": 1},
    )
    monkeypatch.setattr(
        evidence_relevance_module,
        "prewarm_evidence_relevance",
        lambda: {"elapsed_ms": 4.0},
    )
    monkeypatch.setattr(
        profile_manager_module.ProfileManager,
        "get_instance",
        _ready_object,
    )
    monkeypatch.setattr(unified_self_module, "get_unified_self", _ready_object)
    monkeypatch.setattr(
        self_condition_module,
        "build_self_condition_projection",
        lambda: type("Projection", (), {"evidence_id": "condition-proof"})(),
    )
    monkeypatch.setattr(
        chat_dependencies_module,
        "materialize_foreground_chat_dependencies",
        lambda: {"skill_count": 87},
    )

    asyncio.run(
        server._prewarm_chat_dependencies_after_cortex_ready(
            readiness_timeout_s=0.1,
            poll_interval_s=0.01,
        )
    )

    assert lookups >= 2
    assert gate.ready_events == [
        (False, "chat_dependencies_warming"),
        (True, ""),
    ]


def test_server_prewarm_retries_a_completed_dependency_transaction(monkeypatch) -> None:
    import core.cognition.evidence_relevance as evidence_relevance_module
    import core.consciousness.unified_self as unified_self_module
    import core.memory.embedding_runtime as embedding_runtime
    import core.memory.profile_manager as profile_manager_module
    import core.self.self_condition as self_condition_module
    import interface.chat_dependencies as chat_dependencies_module
    from interface import server

    class _Gate:
        def __init__(self) -> None:
            self.ready_events: list[tuple[bool, str]] = []

        def get_cortex_readiness_status(self):
            return {"conversation_ready": True}

        def set_chat_dependencies_ready(self, ready, *, blocker=""):
            self.ready_events.append((bool(ready), str(blocker)))

    gate = _Gate()
    attempts = 0
    foreground_calls = 0

    def _embedding():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise KeyError("huggingface_hub.utils")
        return {"vector_dimensions": 384, "lease_count": 1}

    def _foreground():
        nonlocal foreground_calls
        foreground_calls += 1
        return {"skill_count": 87, "expression_path": {"elapsed_ms": 1.0}}

    async def _ready_object():
        return object()

    monkeypatch.setattr(
        server.ServiceContainer,
        "get",
        classmethod(
            lambda _cls, key, default=None: gate
            if key == "inference_gate"
            else default
        ),
    )
    monkeypatch.setattr(server, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(embedding_runtime, "prewarm_shared_embedding_runtime", _embedding)
    monkeypatch.setattr(
        evidence_relevance_module,
        "prewarm_evidence_relevance",
        lambda: {"elapsed_ms": 1.0},
    )
    monkeypatch.setattr(profile_manager_module.ProfileManager, "get_instance", _ready_object)
    monkeypatch.setattr(unified_self_module, "get_unified_self", _ready_object)
    monkeypatch.setattr(
        self_condition_module,
        "build_self_condition_projection",
        lambda: type("Projection", (), {"evidence_id": "condition-proof"})(),
    )
    monkeypatch.setattr(
        chat_dependencies_module,
        "materialize_foreground_chat_dependencies",
        _foreground,
    )
    monkeypatch.setattr(
        server,
        "record_degradation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a recovered attempt is not terminal degradation")
        ),
    )

    asyncio.run(
        server._prewarm_chat_dependencies_after_cortex_ready(
            readiness_timeout_s=0.1,
            poll_interval_s=0.01,
            dependency_attempts=2,
            dependency_retry_delay_s=0.0,
        )
    )

    assert attempts == 2
    assert foreground_calls == 2
    assert gate.ready_events == [
        (False, "chat_dependencies_warming"),
        (True, ""),
    ]


def test_server_prewarm_names_terminal_dependency_failure(monkeypatch) -> None:
    import core.cognition.evidence_relevance as evidence_relevance_module
    import core.consciousness.unified_self as unified_self_module
    import core.memory.embedding_runtime as embedding_runtime
    import core.memory.profile_manager as profile_manager_module
    import core.self.self_condition as self_condition_module
    import interface.chat_dependencies as chat_dependencies_module
    from interface import server

    class _Gate:
        def __init__(self) -> None:
            self.ready_events: list[tuple[bool, str]] = []

        def get_cortex_readiness_status(self):
            return {"conversation_ready": True}

        def set_chat_dependencies_ready(self, ready, *, blocker=""):
            self.ready_events.append((bool(ready), str(blocker)))

    gate = _Gate()
    degradations: list[tuple[str, str, str]] = []

    async def _ready_object():
        return object()

    monkeypatch.setattr(
        server.ServiceContainer,
        "get",
        classmethod(
            lambda _cls, key, default=None: gate
            if key == "inference_gate"
            else default
        ),
    )
    monkeypatch.setattr(server, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        embedding_runtime,
        "prewarm_shared_embedding_runtime",
        lambda: (_ for _ in ()).throw(KeyError("huggingface_hub.utils")),
    )
    monkeypatch.setattr(
        evidence_relevance_module,
        "prewarm_evidence_relevance",
        lambda: {"elapsed_ms": 1.0},
    )
    monkeypatch.setattr(profile_manager_module.ProfileManager, "get_instance", _ready_object)
    monkeypatch.setattr(unified_self_module, "get_unified_self", _ready_object)
    monkeypatch.setattr(
        self_condition_module,
        "build_self_condition_projection",
        lambda: type("Projection", (), {"evidence_id": "condition-proof"})(),
    )
    monkeypatch.setattr(
        chat_dependencies_module,
        "materialize_foreground_chat_dependencies",
        lambda: {"skill_count": 87, "expression_path": {"elapsed_ms": 1.0}},
    )
    monkeypatch.setattr(
        server,
        "record_degradation",
        lambda subsystem, error, **kwargs: degradations.append(
            (str(subsystem), str(error), str(kwargs.get("severity")))
        ),
    )

    asyncio.run(
        server._prewarm_chat_dependencies_after_cortex_ready(
            readiness_timeout_s=0.1,
            poll_interval_s=0.01,
            dependency_attempts=2,
            dependency_retry_delay_s=0.0,
        )
    )

    assert gate.ready_events == [
        (False, "chat_dependencies_warming"),
        (False, "chat_dependencies_failed"),
    ]
    assert degradations == [
        (
            "server.chat_dependency_warmup",
            "chat_dependency_stage_failed:readers:embedding:KeyError:"
            "'huggingface_hub.utils'",
            "degraded",
        )
    ]
