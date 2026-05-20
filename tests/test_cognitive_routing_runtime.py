import asyncio
import sys
from types import ModuleType

from core.phases import cognitive_routing as routing_module
from core.phases.cognitive_routing import CognitiveRoutingPhase
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState, CognitiveMode


class RouteContainer:
    def __init__(self, capability_engine=None):
        self.capability_engine = capability_engine

    def get(self, name, default=None):
        if name == "capability_engine":
            return self.capability_engine if self.capability_engine is not None else default
        return default


class CapabilityEngineFails:
    def __init__(self):
        self.calls = 0

    def detect_intent(self, _text):
        self.calls += 1
        raise RuntimeError("capability engine unavailable")


class CapabilityEngineSecondReadFails:
    def __init__(self):
        self.calls = 0

    def detect_intent(self, _text):
        self.calls += 1
        if self.calls == 1:
            return []
        raise RuntimeError("capability cache unavailable")


class ExecutiveReceiptFails:
    def __init__(self):
        self.calls = 0

    def record_user_objective(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError("receipt sink unavailable")


class ParallelStreamFails:
    def __init__(self):
        self.calls = 0

    def branch(self, _objective, _context):
        self.calls += 1
        raise RuntimeError("branch scheduler unavailable")


def _state_with_user_text(text: str) -> AuraState:
    state = AuraState()
    state.cognition.working_memory.append({"role": "user", "origin": "user", "content": text})
    return state


def test_capability_detection_failure_keeps_url_route_alive():
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        capability_engine = CapabilityEngineFails()
        phase = CognitiveRoutingPhase(RouteContainer(capability_engine))

        result = await phase.execute(
            _state_with_user_text("Please read https://example.com and summarize it.")
        )

        assert capability_engine.calls == 1
        assert result.cognition.current_mode == CognitiveMode.REACTIVE
        assert result.response_modifiers["intent_type"] == "SKILL"
        assert result.response_modifiers["matched_skills"] == ["sovereign_browser"]
        assert result.response_modifiers["auto_browse_urls"] == ["https://example.com"]
        assert any(
            "without skill fast-path detection" in record.action
            for record in tracker.recent(subsystem="cognitive_routing")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_matched_skill_cache_failure_does_not_break_chat_route():
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        capability_engine = CapabilityEngineSecondReadFails()
        phase = CognitiveRoutingPhase(RouteContainer(capability_engine))

        result = await phase.execute(
            _state_with_user_text("Compare two calm ways to organize a release checklist.")
        )

        assert capability_engine.calls == 2
        assert result.response_modifiers["intent_type"] == "CHAT"
        assert "matched_skills" not in result.response_modifiers
        assert any(
            "without matched skill cache" in record.action
            for record in tracker.recent(subsystem="cognitive_routing")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_executive_receipt_failure_preserves_user_response_lane(monkeypatch):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        authority = ExecutiveReceiptFails()
        monkeypatch.setattr(routing_module, "get_executive_authority", lambda: authority)
        phase = CognitiveRoutingPhase(RouteContainer())

        result = await phase.execute(_state_with_user_text("what is the capital of france"))

        assert authority.calls == 1
        assert result.cognition.current_mode == CognitiveMode.REACTIVE
        assert result.response_modifiers["intent_type"] == "CHAT"
        assert any(
            "objective receipt failed" in record.action
            for record in tracker.recent(subsystem="cognitive_routing")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_parallel_branch_failure_keeps_deliberate_route_foreground():
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        phase = CognitiveRoutingPhase(RouteContainer())
        phase.parallel_stream = ParallelStreamFails()

        result = await phase.execute(
            _state_with_user_text(
                "Please debug this architecture failure in core/runtime/errors.py, "
                "trace the root cause, and propose a durable fix across the routing path."
            )
        )

        assert phase.parallel_stream.calls == 1
        assert result.cognition.current_mode == CognitiveMode.DELIBERATE
        assert any(
            "without spawning parallel thought branch" in record.action
            for record in tracker.recent(subsystem="cognitive_routing")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_substrate_handoff_failure_records_keyword_fallback(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    substrate = ModuleType("core.voice.substrate_voice_engine")

    def extract_unified_field():
        extract_unified_field.calls += 1
        raise RuntimeError("substrate unavailable")

    def extract_neurochemicals():
        return {}

    extract_unified_field.calls = 0
    substrate._extract_unified_field = extract_unified_field
    substrate._extract_neurochemicals = extract_neurochemicals
    monkeypatch.setitem(sys.modules, "core.voice.substrate_voice_engine", substrate)
    phase = CognitiveRoutingPhase(RouteContainer())

    allowed = phase._should_allow_deep_handoff(
        "Do a flagship architecture deep dive and root cause analysis of this runtime.",
        CognitiveMode.DELIBERATE,
        False,
    )

    assert extract_unified_field.calls == 1
    assert allowed is True
    assert any(
        "keyword deep-handoff fallback" in record.action
        for record in tracker.recent(subsystem="cognitive_routing")
    )
    tracker.reset()
