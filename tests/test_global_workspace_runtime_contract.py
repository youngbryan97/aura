from pathlib import Path

import pytest

from core.runtime.receipts import ReceiptStore

_DEFAULT_INHIBITION = object()


def test_global_workspace_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/consciousness/global_workspace.py")) == []


def _install_services(
    monkeypatch,
    *,
    mycelium=None,
    inhibition=_DEFAULT_INHIBITION,
):
    import core.consciousness.global_workspace as workspace_module

    if inhibition is _DEFAULT_INHIBITION:
        class HealthyInhibition:
            instance_id = "inhibition-test"

            async def is_inhibited(self, _source):
                return False

        inhibition = HealthyInhibition()

    def _get(name, default=None):
        if name == "mycelial_network":
            return mycelium
        if name == "inhibition_manager":
            return inhibition
        return default

    monkeypatch.setattr(workspace_module.ServiceContainer, "get", staticmethod(_get))


@pytest.fixture
def receipt_store_factory():
    stores = []

    def create(path):
        store = ReceiptStore(path)
        stores.append(store)
        return store

    yield create
    for store in reversed(stores):
        store.close()


def _install_gate_receipt_store(monkeypatch, tmp_path, receipt_store_factory):
    import core.consciousness.global_workspace as workspace_module

    store = receipt_store_factory(tmp_path / "receipts")
    monkeypatch.setattr(workspace_module, "get_receipt_store", lambda: store)
    return store


def _install_auxiliary_feeds(monkeypatch, *, broken=False):
    import core.consciousness.peripheral_awareness as peripheral_awareness
    import core.consciousness.theory_arbitration as theory_arbitration
    import core.thought_stream as thought_stream
    import core.unity as unity

    class Peripheral:
        def process_workspace_results(self, *, winner_source, all_candidates):
            if broken:
                reason = f"{winner_source}:{len(all_candidates)}:peripheral offline"
                raise RuntimeError(reason)
            return {"winner_source": winner_source, "count": len(all_candidates)}

    class UnityRuntime:
        def record_workspace_competition(self, winner, losers):
            if broken:
                reason = f"{winner.source}:{len(losers)}:unity offline"
                raise RuntimeError(reason)
            return True

    class TheoryArbitration:
        def log_prediction(self, *, theory, event_id, prediction, confidence):
            if broken:
                reason = f"{theory}:{event_id}:theory feed offline"
                raise RuntimeError(reason)
            return prediction

    class Emitter:
        def emit(self, *, title, content, level, metadata):
            if broken:
                reason = f"{title}:{level}:thought stream offline"
                raise RuntimeError(reason)
            return metadata

    monkeypatch.setattr(peripheral_awareness, "get_peripheral_awareness_engine", lambda: Peripheral())
    monkeypatch.setattr(unity, "get_unity_runtime", lambda: UnityRuntime())
    monkeypatch.setattr(theory_arbitration, "get_theory_arbitration", lambda: TheoryArbitration())
    monkeypatch.setattr(thought_stream, "get_emitter", lambda: Emitter())


@pytest.mark.asyncio
async def test_structured_publish_uses_governed_candidate_ingress(monkeypatch):
    from core.consciousness.global_workspace import ContentType, GlobalWorkspace

    checked_sources: list[str] = []

    class HealthyInhibition:
        instance_id = "inhibition-publish"

        async def is_inhibited(self, source):
            checked_sources.append(source)
            return False

    _install_services(monkeypatch, inhibition=HealthyInhibition())
    workspace = GlobalWorkspace()

    accepted = await workspace.publish(
        priority=0.8,
        source="Swarm::ag-123",
        payload={"status": "completed", "result_length": 42},
        reason="Swarm internal monologue step completed",
        content_type=ContentType.META,
    )

    assert accepted is True
    assert checked_sources == ["Swarm::ag-123"]
    candidate = workspace._candidates[0]
    assert candidate.source == "Swarm::ag-123"
    assert candidate.content_type is ContentType.META
    assert candidate.metadata == {
        "schema": "aura.workspace.signal.v1",
        "reason": "Swarm internal monologue step completed",
        "payload": {"status": "completed", "result_length": 42},
    }


@pytest.mark.asyncio
async def test_structured_publish_rejects_unattributed_signal(monkeypatch):
    from core.consciousness.global_workspace import GlobalWorkspace

    _install_services(monkeypatch)
    with pytest.raises(ValueError, match="source must be non-empty"):
        await GlobalWorkspace().publish(priority=0.5, source=" ", payload={})


@pytest.mark.asyncio
async def test_structured_publish_bounds_large_payload_with_reconstructable_digest(monkeypatch):
    import hashlib
    import json

    from core.consciousness.global_workspace import GlobalWorkspace

    _install_services(monkeypatch)
    payload = {"result": "x" * 70_000}
    payload_json = json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)
    workspace = GlobalWorkspace()

    assert await workspace.publish(priority=0.5, source="bounded-producer", payload=payload)

    preserved = workspace._candidates[0].metadata["payload"]
    assert preserved["truncated"] is True
    assert preserved["original_bytes"] == len(payload_json.encode("utf-8"))
    assert preserved["sha256"] == hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    assert len(preserved["preview"]) == 4096


@pytest.mark.asyncio
async def test_workspace_records_auxiliary_failures_without_losing_winner(monkeypatch):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class BrokenMycelium:
        def pulse_hypha(self, source, target, *, success=True):
            reason = f"{source}:{target}:mycelium offline"
            raise RuntimeError(reason)

    class BrokenAttention:
        async def set_focus(self, *, content, source, priority):
            reason = f"{source}:{priority}:focus unavailable"
            raise RuntimeError(reason)

    _install_services(monkeypatch, mycelium=BrokenMycelium())
    _install_auxiliary_feeds(monkeypatch, broken=True)
    workspace = GlobalWorkspace(attention_schema=BrokenAttention())

    assert await workspace.submit(CognitiveCandidate("urgent content", "drive", 1.0))
    winner = await workspace.run_competition()

    degraded = workspace.get_snapshot()["degraded_channels"]
    assert winner is not None
    assert winner.source == "drive"
    assert set(degraded) >= {
        "workspace_pulse",
        "peripheral_awareness",
        "unity_runtime",
        "theory_arbitration",
        "thought_stream",
        "attention_schema",
    }


@pytest.mark.asyncio
async def test_workspace_isolates_processor_failure_and_continues_broadcast(monkeypatch):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    _install_services(monkeypatch)
    _install_auxiliary_feeds(monkeypatch, broken=False)
    workspace = GlobalWorkspace()
    received = []

    async def broken_processor(event):
        reason = f"{len(event.winners)}:processor unavailable"
        raise RuntimeError(reason)

    async def working_processor(event):
        received.append(event.winners[0].source)

    workspace.register_processor(broken_processor)
    workspace.register_processor(working_processor)

    assert await workspace.submit(CognitiveCandidate("processor test", "memory", 0.9))
    winner = await workspace.run_competition()

    snapshot = workspace.get_snapshot()
    assert winner is not None
    assert received == ["memory"]
    assert "processor_broadcast" in snapshot["degraded_channels"]
    assert any(name.endswith("broken_processor") for name in snapshot["processor_failures"])


@pytest.mark.asyncio
async def test_workspace_flood_guard_drops_bid_and_records_reflex_failure(monkeypatch):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class BrokenMycelium:
        def set_hypha_strength(self, source, target, strength):
            reason = f"{source}:{target}:flood reflex offline"
            raise RuntimeError(reason)

    _install_services(monkeypatch, mycelium=BrokenMycelium())
    workspace = GlobalWorkspace()
    # Flood with STRONG bids so an incoming weak bid is genuinely least important.
    workspace._candidates = [
        CognitiveCandidate(f"content-{index}", f"source-{index}", 0.8)
        for index in range(workspace._MAX_CANDIDATES)
    ]

    # A weak bid into a strong, full field is dropped — and the (broken) flood reflex
    # fires, recording the degradation.
    accepted = await workspace.submit(CognitiveCandidate("extra", "overflow", 0.1))

    snapshot = workspace.get_snapshot()
    assert accepted is False
    assert "seizure_guard_reflex" in snapshot["degraded_channels"]
    assert len(workspace._candidates) == workspace._MAX_CANDIDATES
    assert "overflow" not in {c.source for c in workspace._candidates}

    # Backpressure (not blanket-drop): a strong bid into the same full field is
    # admitted by evicting the weakest queued candidate, still respecting the cap.
    admitted = await workspace.submit(CognitiveCandidate("urgent", "affect_distress", 0.99))
    assert admitted is True
    assert len(workspace._candidates) == workspace._MAX_CANDIDATES
    assert "affect_distress" in {c.source for c in workspace._candidates}


@pytest.mark.asyncio
async def test_workspace_inhibition_check_failure_rejects_and_receipts(
    monkeypatch,
    tmp_path,
    receipt_store_factory,
):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class BrokenInhibition:
        instance_id = "inhibition-broken"

        async def is_inhibited(self, source):
            raise RuntimeError(f"gate offline for {source}")

    _install_services(monkeypatch, inhibition=BrokenInhibition())
    store = _install_gate_receipt_store(monkeypatch, tmp_path, receipt_store_factory)
    workspace = GlobalWorkspace()

    accepted = await workspace.submit(CognitiveCandidate("unsafe", "drive", 1.0))

    assert accepted is False
    assert workspace._candidates == []
    snapshot = workspace.get_snapshot()["inhibition_gate"]
    assert snapshot["ready"] is False
    assert snapshot["reason"] == "gate_check_failed:RuntimeError"
    receipts = store.query_recent_persisted("workspace_gate", limit=10)
    assert len(receipts) == 1
    assert receipts[0].candidate_source == "drive"
    assert receipts[0].decision == "rejected"
    assert receipts[0].gate_instance_id == "inhibition-broken"


@pytest.mark.asyncio
async def test_workspace_inhibition_timeout_rejects_without_late_admission(
    monkeypatch,
    tmp_path,
    receipt_store_factory,
):

    import asyncio

    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class WedgedInhibition:
        instance_id = "inhibition-wedged"

        async def is_inhibited(self, _source):
            await asyncio.sleep(0.2)
            return False

    monkeypatch.setenv("AURA_WORKSPACE_INHIBITION_GATE_TIMEOUT_S", "0.01")
    _install_services(monkeypatch, inhibition=WedgedInhibition())
    store = _install_gate_receipt_store(monkeypatch, tmp_path, receipt_store_factory)
    workspace = GlobalWorkspace()

    assert await workspace.submit(CognitiveCandidate("late", "memory", 0.8)) is False
    assert workspace._candidates == []
    receipt = store.query_recent_persisted("workspace_gate", limit=1)[0]
    assert receipt.reason == "gate_check_failed:TimeoutError"


@pytest.mark.asyncio
async def test_workspace_gate_recovers_and_uses_canonical_fallback_instance(
    monkeypatch,
    tmp_path,
    receipt_store_factory,
):
    import core.consciousness.global_workspace as workspace_module
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace
    from core.resilience.inhibition_manager import get_inhibition_manager

    manager = get_inhibition_manager()
    published = []
    monkeypatch.setattr(
        workspace_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: None if name == "inhibition_manager" else default),
    )
    monkeypatch.setattr(
        workspace_module.ServiceContainer,
        "register_instance",
        classmethod(lambda _cls, name, instance, **_kwargs: published.append((name, instance))),
    )
    _install_gate_receipt_store(monkeypatch, tmp_path, receipt_store_factory)
    workspace = GlobalWorkspace()

    assert await workspace.submit(CognitiveCandidate("safe", "memory", 0.8)) is True
    assert workspace.get_snapshot()["inhibition_gate"] == {
        "ready": True,
        "reason": "healthy",
        "instance_id": manager.instance_id,
        "rejection_count": 0,
        "recent_rejections": [],
    }
    assert published == [("inhibition_manager", manager)]


@pytest.mark.asyncio
async def test_workspace_policy_inhibition_is_a_receipted_rejection(
    monkeypatch, tmp_path, receipt_store_factory
):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class ActiveInhibition:
        instance_id = "inhibition-active"

        async def is_inhibited(self, source):
            return source == "looping_source"

    _install_services(monkeypatch, inhibition=ActiveInhibition())
    store = _install_gate_receipt_store(monkeypatch, tmp_path, receipt_store_factory)
    workspace = GlobalWorkspace()

    accepted = await workspace.submit(
        CognitiveCandidate("repeat", "looping_source", 0.9)
    )

    assert accepted is False
    receipt = store.query_recent_persisted("workspace_gate", limit=1)[0]
    assert receipt.reason == "source_inhibited"
    assert receipt.retryable is True


@pytest.mark.asyncio
async def test_the_inhibition_gate_closes_the_checks_it_never_ran(monkeypatch):
    """A check the gate builds and never runs is a coroutine holding another one.

    `gather` wraps them the moment it is called, so the only way past it is
    that it was never called: a loop closing under the gate during shutdown.
    The live log carried seventeen "coroutine is_inhibited was never awaited"
    warnings from that, each one a real inhibition question nothing answered.
    """
    import inspect

    import core.consciousness.global_workspace as workspace_module
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    built: list[object] = []

    class ClosingLoop:
        instance_id = "inhibition-shutdown"

        async def is_inhibited(self, _source):  # pragma: no cover - never awaited
            return False

        def __getattribute__(self, name):
            attribute = object.__getattribute__(self, name)
            if name != "is_inhibited":
                return attribute

            def remember(source):
                coroutine = attribute(source)
                built.append(coroutine)
                return coroutine

            return remember

    _install_services(monkeypatch, inhibition=ClosingLoop())

    def _closed_loop(*_args, **_kwargs):
        raise RuntimeError("Event loop is closed")

    monkeypatch.setattr(workspace_module.asyncio, "gather", _closed_loop)

    workspace = GlobalWorkspace()
    workspace._global_inhibition = None
    candidates = [CognitiveCandidate(f"content-{index}", f"source-{index}", 0.5) for index in range(3)]
    with pytest.raises(RuntimeError):
        await workspace._check_candidates_for_competition(candidates)

    assert len(built) == len(candidates)
    for coroutine in built:
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
