import asyncio
import builtins
from pathlib import Path
from types import SimpleNamespace

from core.container import ServiceContainer
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState
from tools.audit_degradation import analyze_file


def _dynamics(**overrides):
    base = {
        "current_topic": "launch_readiness",
        "partner_frame": "focused",
        "partner_intensity": 0.8,
        "floor_state": "user_floor",
        "humor_frame_active": False,
        "humor_type": "",
        "escalation_invited": False,
        "last_speech_act": "directive",
        "illocutionary_intent": "request",
        "conditional_relevance_open": False,
        "register": "technical",
        "in_group_active": False,
        "accommodation_cues": [],
        "hedge_level": 0,
        "challenge_is_face_threat": False,
        "open_threads": [],
        "association_chain": [],
        "topic_anchors": [],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeDynamicsEngine:
    def __init__(self, dynamics=None, update_error: BaseException | None = None):
        self._dynamics = dynamics or _dynamics()
        self._update_error = update_error
        self.calls = []

    def update(self, *, message, role, working_memory):
        self.calls.append((message, role, list(working_memory or [])))
        if self._update_error:
            raise self._update_error
        return self._dynamics

    def get_prompt_injection(self):
        return "CONVERSATIONAL DYNAMICS BLOCK"


def test_conversational_dynamics_degradation_audit_is_clean():
    assert analyze_file(Path("core/phases/conversational_dynamics_phase.py")) == []


def test_phase_returns_original_state_when_core_engine_update_fails():
    tracker = get_degradation_tracker()
    tracker.reset()
    phase = ConversationalDynamicsPhase(kernel=None)
    phase._engine = FakeDynamicsEngine(update_error=RuntimeError("engine update failed"))
    state = AuraState()

    result = asyncio.run(phase.execute(state, "hello", origin="user"))

    assert result is state
    recent = tracker.recent(subsystem="conversational_dynamics_phase", limit=1)
    assert recent
    assert recent[0].severity == "error"
    assert recent[0].action == "returned original state after conversational dynamics phase failed"


def test_phase_preserves_core_dynamics_when_optional_context_modules_fail(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    phase = ConversationalDynamicsPhase(kernel=None)
    phase._engine = FakeDynamicsEngine()
    state = AuraState()
    state.cognition.working_memory = [{"role": "assistant", "content": "Earlier answer."}]

    blocked_imports = {
        "core.consciousness.agency_comparator",
        "core.consciousness.hot_engine",
        "core.consciousness.intersubjectivity",
        "core.consciousness.multiple_drafts",
        "core.consciousness.narrative_gravity",
    }
    real_import = builtins.__import__

    def controlled_import(name, *args, **kwargs):
        if name in blocked_imports:
            raise ImportError(f"blocked optional module: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: None),
    )

    result = asyncio.run(phase.execute(state, "ship it carefully", origin="user"))

    assert result is not state
    assert result.response_modifiers["conversational_dynamics"] == "CONVERSATIONAL DYNAMICS BLOCK"
    dynamics_state = result.response_modifiers["conv_dynamics_state"]
    assert dynamics_state["partner_frame"] == "focused"
    assert result.cognition.discourse_topic == "launch_readiness"
    assert result.cognition.conversation_energy == 0.8
    recent_actions = {
        event.action
        for event in tracker.recent(subsystem="conversational_dynamics_phase", limit=10)
    }
    assert "continued without multiple-drafts context for this turn" in recent_actions
    assert "continued without higher-order thought context for this turn" in recent_actions
    assert "continued without intersubjectivity context for this turn" in recent_actions
