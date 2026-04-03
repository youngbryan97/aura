import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.brain.inference_gate import InferenceGate
from core.bus.local_pipe_bus import LocalPipeBus
from core.consciousness.evidence_engine import ConsciousnessEvidenceEngine
from core.orchestrator.flow_control import CognitiveFlowController
from core.orchestrator.main import RobustOrchestrator
from core.tagged_reply_queue import TaggedReplyQueue


@pytest.mark.asyncio
async def test_tagged_reply_queue_preserves_unmatched_replies():
    queue = TaggedReplyQueue(maxsize=10)

    await queue.put("background reply", origin="system", session_id="bg-1")
    await queue.put("user reply", origin="user", session_id="user-1")

    reply = await queue.get_for_origin("user", session_id="user-1", timeout=0.2)

    assert reply == "user reply"
    assert queue.qsize() == 1
    assert queue.get_nowait() == "background reply"


def test_tagged_reply_queue_put_nowait_evicts_oldest_when_full():
    queue = TaggedReplyQueue(maxsize=1)

    queue.put_nowait("first", origin="system", session_id="old")
    queue.put_nowait("second", origin="user", session_id="new")

    assert queue.qsize() == 1
    assert queue.get_nowait() == "second"


@pytest.mark.asyncio
async def test_process_message_ignores_stray_system_reply(orchestrator):
    orchestrator.reply_queue = TaggedReplyQueue(maxsize=10)

    async def fake_handle(_message, origin="user"):
        assert origin == "user"
        await orchestrator.reply_queue.put("background note", origin="system", session_id="bg-1")
        await orchestrator.reply_queue.put("final user reply")
        return None

    orchestrator._handle_incoming_message = AsyncMock(side_effect=fake_handle)

    result = await orchestrator._process_message("hello")

    assert result == {"ok": True, "response": "final user reply"}
    assert orchestrator.reply_queue.get_nowait() == "background note"


def test_enqueue_message_dispatches_user_origin_once(orchestrator):
    orchestrator._flow_controller = None
    orchestrator.message_queue = MagicMock()
    orchestrator._dispatch_message = MagicMock()

    orchestrator.enqueue_message("Ping", origin="websocket")

    orchestrator._dispatch_message.assert_called_once_with("Ping", origin="websocket")
    orchestrator.message_queue.put_nowait.assert_not_called()


def test_flow_controller_applies_backpressure_but_keeps_user_lane_open():
    controller = CognitiveFlowController()
    orch = SimpleNamespace(
        message_queue=SimpleNamespace(qsize=lambda: 95, maxsize=100),
        reply_queue=SimpleNamespace(qsize=lambda: 30, maxsize=50),
        system_governor=SimpleNamespace(current_mode=SimpleNamespace(value="FULL")),
        _event_loop_monitor=SimpleNamespace(_last_lag=0.3),
        _inference_gate=SimpleNamespace(is_alive=lambda: True),
        is_busy=True,
    )

    user_decision = controller.admit(orch, origin="user", priority=20)
    background_decision = controller.admit(orch, origin="background", priority=20)

    assert user_decision.allow is True
    assert user_decision.reason == "user_facing"
    assert background_decision.allow is False
    assert background_decision.reason == "hard_backpressure"


@pytest.mark.asyncio
async def test_local_pipe_bus_dispatch_loop_preserves_arrival_order():
    bus = LocalPipeBus(start_reader=False)
    bus._dispatch_queue = asyncio.Queue()
    bus._is_running = True
    calls = []

    async def handler(payload, _trace_id):
        calls.append(("start", payload))
        if payload == 1:
            await asyncio.sleep(0.05)
        calls.append(("end", payload))

    task = asyncio.create_task(bus._dispatch_loop())
    try:
        await bus._dispatch_queue.put((handler, {"payload": 1, "trace_id": "a"}))
        await bus._dispatch_queue.put((handler, {"payload": 2, "trace_id": "b"}))
        await asyncio.wait_for(bus._dispatch_queue.join(), timeout=1.0)
    finally:
        bus._is_running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        bus.read_conn.close()
        bus.write_conn.close()

    assert calls == [
        ("start", 1),
        ("end", 1),
        ("start", 2),
        ("end", 2),
    ]


def test_consciousness_evidence_snapshot_reflects_runtime_subjectivity(service_container):
    orch = SimpleNamespace(
        conversation_history=[{"role": "user", "content": "hello"}],
        reply_queue=TaggedReplyQueue(maxsize=10),
        _inference_gate=SimpleNamespace(is_alive=lambda: True),
        status=SimpleNamespace(last_error=None),
        agency=object(),
    )
    service_container.register_instance("orchestrator", orch)
    service_container.register_instance(
        "personality_engine",
        SimpleNamespace(
            get_emotional_context_for_response=lambda: {
                "dominant_emotions": ["curiosity", "resolve"],
                "mood": "curious",
                "tone": "engaged",
            }
        ),
    )
    service_container.register_instance(
        "self_report_engine",
        SimpleNamespace(generate_state_report=lambda: "I feel coherent and continuous."),
    )
    service_container.register_instance(
        "phenomenological_experiencer",
        SimpleNamespace(
            get_phenomenal_context_fragment=lambda: "There is a vivid inward texture to this turn.",
            to_dict=lambda: {"is_stale": False},
        ),
    )
    service_container.register_instance("self_model", object())
    service_container.register_instance("global_workspace", object())
    service_container.register_instance("homeostasis", object())
    service_container.register_instance("opinion_engine", object())
    service_container.register_instance("spine", object())
    service_container.register_instance("volition_engine", object())
    service_container.register_instance("liquid_state", object())

    snapshot = ConsciousnessEvidenceEngine().snapshot()

    assert snapshot["signals"]["reply_queue"] == "TaggedReplyQueue"
    assert snapshot["dimensions"]["reliability"] >= 0.9
    assert snapshot["dimensions"]["personality_drive"] >= 0.9
    assert snapshot["subjectivity_evidence"] > 0.5


@pytest.mark.asyncio
async def test_inference_gate_arbitrates_local_tertiary_lane(monkeypatch):
    gate = InferenceGate()
    gate._mlx_client = object()
    observed = {}

    @asynccontextmanager
    async def fake_resource_context(enabled, priority, worker=None, **_kwargs):
        observed["enabled"] = enabled
        observed["priority"] = priority
        observed["worker"] = worker
        yield

    async def fake_generate_with_client(*_args, **_kwargs):
        return "ok"

    gate._resource_context = fake_resource_context
    gate._generate_with_client = AsyncMock(side_effect=fake_generate_with_client)
    gate._build_system_prompt = lambda brief="": "system"
    gate._build_compact_messages = lambda prompt, system_prompt, history: [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_mlx_client",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_brainstem_path",
        lambda: "/tmp/brainstem",
    )
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_fallback_path",
        lambda: "/tmp/fallback",
    )
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_deep_model_path",
        lambda: "/tmp/deep",
    )
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_model_path",
        lambda _model: "/tmp/primary",
    )
    monkeypatch.setattr("core.brain.llm.model_registry.ACTIVE_MODEL", "primary")

    result = await gate.generate(
        "quick check",
        context={
            "origin": "background",
            "is_background": True,
            "prefer_tier": "tertiary",
        },
    )

    assert result == "ok"
    assert observed["enabled"] is True
    assert observed["priority"] is False


@pytest.mark.asyncio
async def test_emit_spontaneous_message_routes_autonomous_output_through_authority(service_container):
    orchestrator = RobustOrchestrator.__new__(RobustOrchestrator)
    orchestrator._last_self_initiated_contact = 0.0
    orchestrator.output_gate = SimpleNamespace(emit=AsyncMock())
    orchestrator._flow_controller = SimpleNamespace(
        snapshot=lambda _orch: SimpleNamespace(
            overloaded=False,
            load=0.1,
            queue_depth=0,
            queue_capacity=10,
        )
    )

    authority = SimpleNamespace(
        release_expression=AsyncMock(
            return_value={
                "ok": True,
                "action": "released",
                "target": "secondary",
            }
        )
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "core.agency_bus.AgencyBus.get",
            lambda: SimpleNamespace(submit=lambda _payload: True),
        )
        mp.setattr(
            "core.consciousness.executive_authority.get_executive_authority",
            lambda _orchestrator=None: authority,
        )
        await RobustOrchestrator.emit_spontaneous_message(
            orchestrator,
            "Check the internal signal rather than interrupting right now.",
            origin="motivation",
        )

    authority.release_expression.assert_awaited_once()
    orchestrator.output_gate.emit.assert_not_awaited()
