import asyncio
from types import SimpleNamespace

import pytest

from core.coordinators.cognitive_coordinator import CognitiveCoordinator
from core.runtime.errors import get_degradation_tracker


class _Trace:
    def __init__(self):
        self.steps = []
        self.saved = False

    def record_step(self, name, payload):
        self.steps.append((name, payload))

    def save(self):
        self.saved = True


def _coordinator(orch):
    coord = CognitiveCoordinator.__new__(CognitiveCoordinator)
    coord.orch = orch

    async def _guard(response):
        return response

    async def _fallback(_message):
        return "fallback"

    coord.apply_constitutional_guard = _guard
    coord.generate_fallback = _fallback
    return coord


@pytest.mark.asyncio
async def test_finalize_response_survives_background_hook_failure():
    queue = asyncio.Queue()
    trace = _Trace()
    failures = []

    def _fail_reflection(_response):
        failures.append("reflection")
        raise RuntimeError("reflection unavailable")

    async def _satisfy(_drive, _amount):
        return None

    orch = SimpleNamespace(
        meta_learning=None,
        social=None,
        self_modifier=None,
        affect_engine=None,
        cognition=None,
        _epistemic_engine=SimpleNamespace(
            should_ask_for_clarification=lambda _message: (False, ""),
            apply_epistemic_humility=lambda _message, response: response,
        ),
        _trigger_background_reflection=_fail_reflection,
        _trigger_background_learning=lambda _message, _response: None,
        _last_thought_time=0.0,
        drives=SimpleNamespace(satisfy=_satisfy),
        reply_queue=queue,
        ears=None,
        _filter_output=lambda text: text,
    )

    result = await _coordinator(orch).finalize_response("hello", "reply", "user", trace, [])

    assert result == "reply"
    assert queue.get_nowait() == "reply"
    assert trace.saved is True
    assert failures == ["reflection"]


@pytest.mark.asyncio
async def test_gather_agentic_context_keeps_defaults_when_memory_retrieval_fails(monkeypatch):
    import core.container as container_module

    degradation_tracker = get_degradation_tracker()
    degradation_tracker.reset()

    async def _recall_strategy(_message):
        return {"strategy": "direct"}

    async def _retrieve_unified_context(_message, limit=5):
        assert limit == 5
        raise RuntimeError("memory unavailable")

    async def _environment():
        return {"screen": "ready"}

    orch = SimpleNamespace(
        meta_learning=SimpleNamespace(recall_strategy=_recall_strategy),
        memory=SimpleNamespace(retrieve_unified_context=_retrieve_unified_context),
        _get_personality_data=lambda: {"mood": "steady"},
        _stringify_personality=lambda data: f"mood={data['mood']}",
        _get_world_context=lambda: {"place": "runtime"},
        _get_environmental_context=_environment,
        _get_cleaned_history_context=lambda _limit: {"history": [{"role": "user", "content": "hi"}]},
        affect_engine=None,
        mind_model=None,
        social=None,
        strategic_planner=None,
        project_store=None,
        drive_engine=None,
        liquid_state=None,
    )

    monkeypatch.setattr(
        container_module.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    try:
        ctx = await _coordinator(orch).gather_agentic_context("hello")

        assert ctx["meta_learning"] == {"strategy": "direct"}
        assert ctx["unified_memory"] == ""
        assert ctx["environment"] == {"screen": "ready"}
        assert degradation_tracker.count("cognitive_coordinator") >= 1
    finally:
        degradation_tracker.reset()
