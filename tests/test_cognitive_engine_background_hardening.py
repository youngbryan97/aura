import asyncio
import copy
from types import SimpleNamespace

import pytest

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.foreground_latent_runtime import ForegroundLatentOutcome
from core.brain.types import ThinkingMode, Thought
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState
from core.utils.injected_blocks import stamp_runtime_payload


class StateRepositoryFixture:
    def __init__(self, state):
        self._current = state
        self.get_current_calls = 0
        self.commits = []
        self.commit_snapshots = []

    async def get_current(self):
        self.get_current_calls += 1
        return self._current

    async def commit(self, state, *args, **kwargs):
        self.commits.append((state, args, kwargs))
        self.commit_snapshots.append(copy.deepcopy(state))
        self._current = state


@pytest.mark.asyncio
async def test_qualified_recurrent_answer_owns_turn_before_general_cognition(monkeypatch):
    engine = CognitiveEngine()
    repo = StateRepositoryFixture(AuraState.default())
    engine.state_repository = repo
    engine._phases = [
        SimpleNamespace(
            execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("qualified recurrence must bypass general phases")
            )
        )
    ]
    visible = (
        "Fresh calibration task. Before evidence E, hypothesis H has probability "
        "1/2. The likelihood "
        "of E is 4/5 if H is true and 1/10 if H is false. Using exact Bayes "
        "updating, return the more probable choice (H wins ties), the reduced "
        "posterior probability of H, and its band: below_50, 50_to_69, "
        "70_to_89, or 90_to_100. You may reason before the answer. End with "
        "exactly one line beginning "
        "FINAL_ANSWER:, followed by one JSON object and no trailing text. "
        "Required JSON keys and value types: choice (string), posterior "
        "(reduced-fraction string), confidence_band (string)."
    )
    expected = (
        'FINAL_ANSWER: {"choice":"H","confidence_band":"70_to_89",'
        '"posterior":"8/9"}'
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("qualified recurrence must run before cognitive advisors")

    monkeypatch.setattr(engine, "_apply_spiking_active_inference", _unexpected)
    monkeypatch.setattr(engine, "_apply_imagination_workspace", _unexpected)
    monkeypatch.setattr(engine, "_apply_entity_memory", _unexpected)
    monkeypatch.setattr(engine, "_apply_bicameral_advisory", _unexpected)
    monkeypatch.setattr(engine, "_apply_cognitive_situation_frame", _unexpected)

    async def _qualified_episode(**kwargs):
        assert kwargs["visible_objective"] == visible
        return ForegroundLatentOutcome(
            text=expected,
            trace={
                "qualified_recurrent_eligible": True,
                "qualified_recurrent_attempted": True,
                "qualified_recurrent_succeeded": True,
                "qualified_recurrent_reason": "qualified_semantic_neural_completed",
                "qualified_recurrent_receipt": {"receipt_sha256": "signed"},
                "latent_cortex_succeeded": True,
            },
            fallback_allowed=False,
            evidence=("qualified_semantic_neural_execution",),
        )

    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.run_foreground_latent_episode",
        _qualified_episode,
    )

    thought = await engine.think(
        visible,
        mode=ThinkingMode.FAST,
        origin="desktop_ui",
        context={
            "visible_user_message": visible,
            "desktop_cognitive_engine_required": True,
            "session_id": "live-test",
        },
        foreground_request=True,
        timeout_s=30.0,
    )

    assert thought.content == expected
    assert thought.metadata["response_path"] == "cognitive_engine_qualified_recurrent"
    assert thought.metadata["model_generation_used"] is False
    assert repo.commits
    committed = repo.commits[-1][0]
    assert committed.cognition.working_memory[-2]["content"] == visible
    assert committed.cognition.working_memory[-1]["content"] == expected
    assert committed.response_modifiers["qualified_recurrent_succeeded"] is True
    assert committed.response_modifiers["response_path"] == (
        "cognitive_engine_qualified_recurrent"
    )


@pytest.mark.asyncio
async def test_unsupported_language_does_not_touch_qualified_recurrent_service(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)

    async def _unexpected_episode(**_kwargs):
        raise AssertionError("unsupported language must not touch recurrent service")

    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.run_foreground_latent_episode",
        _unexpected_episode,
    )

    thought = await engine._qualified_recurrent_direct_reply(
        state,
        "How are you feeling right now?",
        ThinkingMode.FAST,
        "desktop_ui",
        {"visible_user_message": "How are you feeling right now?"},
        is_background=False,
        timeout_s=30.0,
    )

    assert thought is None


def test_cognitive_engine_treats_prefixed_user_origin_as_foreground():
    assert CognitiveEngine._is_background_request("routing_user", False) is False
    assert CognitiveEngine._is_background_request("routing_voice_command", False) is False
    assert CognitiveEngine._is_background_request("autonomous_thought", False) is True


def test_cognitive_engine_treats_live_desktop_origins_as_user_facing():
    assert CognitiveEngine._is_user_facing_origin("chat_api") is True
    assert CognitiveEngine._is_user_facing_origin("desktop_ui") is True
    assert CognitiveEngine._is_user_facing_origin("voice_bridge") is True
    assert CognitiveEngine._is_user_facing_origin("native-shell") is True
    assert CognitiveEngine._is_user_facing_origin("background_ui") is False
    assert CognitiveEngine._is_user_facing_origin("agency_core") is False


def test_cognitive_engine_live_desktop_origin_updates_working_memory(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default().derive("cognitive_intent: desktop_ui", origin="desktop_ui")
    engine.state_repository = None
    engine._phases = []

    monkeypatch.setenv("AURA_TESTING", "1")

    thought = asyncio.run(
        engine._run_thinking_loop(
            state,
            "Desktop live path should stay foreground.",
            ThinkingMode.FAST,
            "desktop_ui",
        )
    )

    assert state.transition_origin == "desktop_ui"
    assert state.cognition.working_memory[-1]["role"] == "user"
    assert state.cognition.working_memory[-1]["origin"] == "desktop_ui"
    assert thought.reasoning


def test_cognitive_engine_preserves_desktop_origin_after_phase_derives(monkeypatch):
    class _ResettingPhase:
        async def execute(self, state, objective=None, **_kwargs):
            derived = state.derive("phase_default_origin_reset")
            derived.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": "I will keep the live desktop turn on the foreground path.",
                }
            )
            return derived

    engine = CognitiveEngine()
    repo = StateRepositoryFixture(AuraState.default())
    engine.state_repository = repo
    engine._phases = [_ResettingPhase()]

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    thought = asyncio.run(
        engine.think(
            "Keep this live desktop turn foreground.",
            mode=ThinkingMode.FAST,
            origin="desktop_ui",
        )
    )

    assert thought.content == "I will keep the live desktop turn on the foreground path."
    assert repo.commits
    committed_state = repo.commits[-1][0]
    assert committed_state.transition_origin == "desktop_ui"
    assert committed_state.cognition.current_origin is None
    committed_snapshot = repo.commit_snapshots[-1]
    assert committed_snapshot.cognition.current_origin is None
    assert committed_snapshot.cognition.current_objective is None


@pytest.mark.parametrize(
    "background_objective",
    ["Investigate thermal pressure", "You with me?"],
)
def test_cognitive_engine_preserves_phase_selected_background_objective(
    monkeypatch,
    background_objective,
):
    class _BackgroundSelectionPhase:
        async def execute(self, state, objective=None, **_kwargs):
            derived = state.derive("phase_selected_background_objective")
            derived.cognition.current_objective = background_objective
            derived.cognition.current_origin = "curiosity"
            derived.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": "I am with you and will inspect that pressure separately.",
                }
            )
            return derived

    engine = CognitiveEngine()
    repo = StateRepositoryFixture(AuraState.default())
    engine.state_repository = repo
    engine._phases = [_BackgroundSelectionPhase()]

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: repo if name == "state_repository" else default
        ),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    asyncio.run(
        engine.think(
            "You with me?",
            mode=ThinkingMode.FAST,
            origin="native-shell",
        )
    )

    committed = repo.commit_snapshots[-1]
    assert committed.transition_origin == "native_shell"
    assert committed.cognition.current_objective == background_objective
    assert committed.cognition.current_origin == "curiosity"
    assert committed.response_modifiers["foreground_turn_completion"][
        "preserved_background"
    ] is True


def test_cognitive_engine_uses_canonical_context_assembler():
    from core.brain.llm.context_assembler import ContextAssembler

    CognitiveEngine()

    assert ContextAssembler.build_system_prompt.__module__ == "core.brain.llm.context_assembler"
    assert ContextAssembler.build_messages.__module__ == "core.brain.llm.context_assembler"
    assert not getattr(ContextAssembler, "_patched_v1", False)


@pytest.mark.asyncio
async def test_cognitive_engine_skips_identity_refresh_for_background_origin(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["objective"] = objective
        return Thought(id="bg-thought", content="ok", mode=mode)

    monitor = SimpleNamespace(needs_context_refresh=lambda *_args, **_kwargs: True)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: monitor if name == "drift_monitor" else default,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_engine.ContextAssembler.build_system_prompt",
        staticmethod(lambda _state: "context" * 500),
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think(
        "Summarize internal maintenance state.",
        mode=ThinkingMode.FAST,
        origin="autonomous",
        is_background=True,
    )

    assert thought.content == "ok"
    assert captured["objective"] == "Summarize internal maintenance state."


@pytest.mark.asyncio
async def test_cognitive_engine_suppresses_background_thoughts_when_background_policy_blocks(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else SimpleNamespace()),
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.20",
    )

    thought = await engine.think(
        "Distill this memory to its essential insight.",
        mode=ThinkingMode.FAST,
        origin="sovereign_pruner",
        is_background=True,
    )

    assert thought.metadata["suppressed"] is True
    assert "background_thought_suppressed" in thought.reasoning[0]


@pytest.mark.asyncio
async def test_cognitive_engine_background_no_response_is_quiet_noop(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    engine.state_repository = repo
    engine._phases = []

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    thought = await engine.think(
        "I was curious about the host environment, so I initiated a system scan.",
        mode=ThinkingMode.FAST,
        origin="agency_core_environmental_explorer",
        is_background=True,
    )

    assert thought.content == ""
    assert thought.metadata["suppressed"] is True
    assert "background_cycle_no_response" in thought.reasoning[0]


@pytest.mark.asyncio
async def test_cognitive_engine_resolves_missing_origin_from_orchestrator(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    orchestrator = SimpleNamespace(_current_origin="terminal_monitor")
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["origin"] = origin
        return Thought(id="origin-from-orch", content="ok", mode=mode)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: (
                orchestrator if name == "orchestrator" else repo if name == "state_repository" else default
            )
        ),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think("Investigate the timeout.", mode=ThinkingMode.FAST)

    assert thought.content == "ok"
    assert captured["origin"] == "terminal_monitor"


@pytest.mark.asyncio
async def test_cognitive_engine_defaults_missing_origin_to_system(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["origin"] = origin
        return Thought(id="origin-default", content="ok", mode=mode)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think("Perform internal maintenance.", mode=ThinkingMode.FAST)

    assert thought.content == "ok"
    assert captured["origin"] == "system"


@pytest.mark.asyncio
async def test_cognitive_engine_user_recovery_uses_bounded_primary_router(monkeypatch):
    engine = CognitiveEngine()
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am still on the live desktop thread and answering the user directly."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: _Router() if name == "llm_router" else default),
    )

    thought = await engine._reactive_recovery(
        "Answer directly: are you still on the live desktop thread?",
        ThinkingMode.FAST,
        "desktop_ui",
        "timeout",
    )

    assert thought.content.startswith("I am still on the live desktop thread")
    assert captured["prefer_tier"] == "primary"
    assert captured["foreground_request"] is True
    assert captured["skip_runtime_payload"] is False
    assert captured["allow_deep_handoff"] is False
    assert captured["max_tokens"] <= 384


@pytest.mark.asyncio
async def test_cognitive_engine_reactive_recovery_delegates_rollback_governance_to_repository(
    monkeypatch,
):
    from core.governance_context import get_active_governance

    engine = CognitiveEngine()
    captured = {}

    class _Repo:
        async def rollback(self, reason, *, expected_version=None):
            token = get_active_governance()
            captured["reason"] = reason
            captured["token"] = token
            captured["expected_version"] = expected_version

    class _Router:
        async def think(self, **_kwargs):
            return "I recovered the live user-facing turn through the governed primary router."

    engine.state_repository = _Repo()
    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: _Router() if name == "llm_router" else default),
    )

    thought = await engine._reactive_recovery(
        "Answer directly after a cognitive timeout.",
        ThinkingMode.FAST,
        "desktop_ui",
        "timeout",
    )

    assert thought.content.startswith("I recovered the live user-facing turn")
    assert captured["reason"] == "recovery: timeout"
    assert captured["token"] is None


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_reply_uses_governed_primary_router(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    engine.state_repository = repo
    engine._phases = [
        SimpleNamespace(
            execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("compact desktop quick reply should not enter phase loop")
            )
        )
    ]
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am on the live desktop path and answering this turn directly."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "You ok?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "You ok?",
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert thought.content.startswith("I am on the live desktop path")
    assert captured["prefer_tier"] == "primary"
    assert captured["protected_foreground_lane"] is True
    assert captured["desktop_cognitive_engine_required"] is True
    assert captured["allow_cloud_fallback"] is False
    assert captured["allow_deep_handoff"] is False
    assert captured["skip_runtime_payload"] is True
    assert repo.commits
    committed = repo.commits[-1][0]
    assert committed.cognition.working_memory[-2]["role"] == "user"
    assert committed.cognition.working_memory[-1]["role"] == "assistant"
    committed_snapshot = repo.commit_snapshots[-1]
    assert committed_snapshot.cognition.current_objective is None
    assert committed_snapshot.cognition.current_origin is None


@pytest.mark.asyncio
async def test_cognitive_engine_sizes_dense_technical_answer_from_visible_work(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}
    visible = (
        "Explain Dijkstra in one response. Include: (1) the invariant, "
        "(2) numbered pseudocode, (3) a worked example with at least five edges, "
        "(4) complexity with both a heap and an array, and (5) a negative-weight "
        "failure and the correct alternative."
    )

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "A complete answer generated inside the reserved surface."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        visible,
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": visible,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 896,
        },
        is_background=False,
        timeout_s=432.0,
    )

    assert thought is not None
    assert captured["max_tokens"] == 2560
    assert captured["user_surface_completion_floor"] == 2560
    assert captured["reply_needs_room"] is True
    assert captured["timeout"] == pytest.approx(427.0)


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_reply_carries_fresh_turn_sight(monkeypatch):
    from core.senses.turn_evidence import build_camera_turn_evidence

    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return (
                "I looked just now. I do not see another person in the current "
                "camera view, though that view cannot establish the whole room."
            )

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router()
            if name == "llm_router"
            else default
        ),
    )
    evidence = build_camera_turn_evidence(
        "Is anyone else physically here with me?",
        ok=True,
        observation="No other person is visible in the current camera view.",
        observed_at=1_786_857_523.0,
    )

    thought = await engine._run_thinking_loop(
        state,
        "Is anyone else physically here with me?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "Is anyone else physically here with me?",
            "turn_sensory_evidence": evidence,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert thought.content.startswith("I looked just now")
    assert captured["messages"][-1]["content"] == (
        "Is anyone else physically here with me?"
    )
    grounding_prompt = captured["messages"][-2]["content"]
    assert "[FRESH TURN SENSORY EVIDENCE]" in grounding_prompt
    assert "status: observed" in grounding_prompt
    assert "No other person is visible" in grounding_prompt
    assert captured["user_surface_sensory_evidence"] == evidence
    from core.utils.injected_blocks import is_stamped_grounding

    assert is_stamped_grounding(captured["messages"][0])
    assert is_stamped_grounding(captured["messages"][-2])


@pytest.mark.asyncio
async def test_cognitive_engine_runtime_status_contract_propagates_to_worker_boundary(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "Cortex (32B) is the active foreground lane."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "What model lane is speaking right now?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "What model lane is speaking right now?",
            "runtime_fact_status_contract": True,
            "grounded_runtime_status_contract": True,
            "grounded_runtime_status_context": (
                "Cortex (32B) is the active foreground lane, "
                "CognitiveEngine handled this turn: yes."
            ),
            "recent_completed_exchanges": [
                {"user": "Old topic", "aura": "Old answer"}
            ],
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 896,
        },
        is_background=False,
        timeout_s=60.0,
    )

    assert thought.content.startswith("Cortex (32B)")
    assert captured["runtime_fact_status_contract"] is True
    assert captured["grounded_runtime_status_contract"] is True
    assert captured["max_tokens"] == 256
    assert len(captured["messages"]) == 3
    assert "VERIFIED LIVE RUNTIME STATUS" in captured["messages"][-2]["content"]
    assert captured["messages"][-1]["content"] == (
        "What model lane is speaking right now?"
    )


@pytest.mark.asyncio
async def test_cognitive_engine_full_mind_planning_keeps_extended_live_budget(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return (
                "I would authorize the workflow, research the sources, draft the document, "
                "verify the visible result, and preserve effect receipts before reporting completion."
            )

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Give a practical multi-step desktop task you could attempt after authorization.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": (
                "Give a practical multi-step desktop task you could attempt after authorization."
            ),
            "bounded_planning_contract": True,
            "bounded_planning_reply": (
                "Authorize the requested desktop workflow, perform each step in order, "
                "verify the visible effects, and preserve receipts before reporting completion."
            ),
            "require_full_foreground_mind_reply": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "prompt_shape": {
                "question_parts": 1,
                "prefers_extended_answer": False,
                "requires_single_reply_coverage": False,
            },
            "max_tokens": 1536,
        },
        is_background=False,
        timeout_s=90.0,
    )

    assert thought.content.startswith("I would authorize")
    assert captured["max_tokens"] == 1536
    assert captured["messages"][-1]["content"] == (
        "Give a practical multi-step desktop task you could attempt after authorization."
    )
    grounding_prompt = captured["messages"][-2]["content"]
    assert "[GOVERNED PLANNING OUTLINE]" in grounding_prompt
    assert "Do not use a numbered list" in grounding_prompt


@pytest.mark.asyncio
async def test_cognitive_engine_capability_inventory_contract_uses_catalog_without_worker(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return (
                "I can coordinate desktop apps, browser/web research, files, documents, "
                "terminal/code, memory, and repair tools. Consequential use is governed "
                "by Will and Authority permissions, receipts, and effect verification. "
                "Hypothetically, I could research, write, export, and file artifacts, "
                "but I am not executing tools in this turn."
            )

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "What external tools can you use from the desktop?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "What external tools can you use from the desktop?",
            "capability_inventory_contract": True,
            "grounded_capability_inventory_context": (
                "desktop/app control; browser/web research; file/document/PDF operations; "
                "terminal/code; memory; repair. Governed by Will and Authority with "
                "receipts and effect verification."
            ),
            "response_style_contract": "This duplicate style contract should not bloat capability turns.",
            "live_speech_grounding_frame": {
                "mood": "curiosity",
                "dominant_emotions": ["curiosity"],
                "requires_explicit_live_grounding": True,
            },
            "live_mind_context": {
                "required_for_live_desktop": True,
                "must_answer_from_full_mind_path": True,
                "required_subsystems": {"kernel": True, "memory": True},
                # Part of the binding contract now: the snapshot carries this
                # runtime's stamp and declares its required subsystems healthy.
                # Neither was checked before, so a caller-supplied dictionary
                # could steer generation on its own say-so.
                **stamp_runtime_payload({}),
                "required_subsystems_ok": True,
                "lane": {"state": "ready"},
                "voice": {"mode": "normal"},
                "substrate": {"curiosity": 0.8, "verbose_blob": "x" * 5000},
                "mind_snapshot": {"verbose_blob": "y" * 5000},
                "mind_snapshot_quality": {"ready": True},
                "governance": {"will": "ok"},
            },
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 384,
        },
        is_background=False,
        timeout_s=60.0,
    )

    assert "browser/web research" in thought.content
    assert "Will and Authority" in thought.content
    assert captured == {}
    assert thought.metadata["response_path"] == "cognitive_engine_capability_catalog_grounding"
    assert thought.metadata["live_mind_controls_bound"] is True
    # A structured floor never runs the model worker; the honest receipt says
    # so (applied=False) while generation_required=False keeps the contract
    # satisfied. Pinning worker_applied=True here would demand a false claim.
    assert thought.metadata["live_mind_controls_worker_applied"] is False
    assert thought.metadata["live_mind_generation_required"] is False


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_reply_includes_recent_context(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am carrying the recent context forward instead of losing the thread."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Continue from there.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "Continue from there.",
            "recent_conversation_context": (
                "User: The live desktop lane lost context.\n"
                "Aura: I should preserve bounded recent exchanges through CognitiveEngine."
            ),
            # Stamped as the chat route produces them. These become user and
            # ASSISTANT messages, so an unattested entry would be an assistant
            # turn Aura never took, quoted back to her as her own prior words.
            "recent_completed_exchanges": [
                stamp_runtime_payload(
                    {
                        "user": "The live desktop lane lost context.",
                        "aura": (
                            "I should preserve bounded recent exchanges "
                            "through CognitiveEngine."
                        ),
                    }
                )
            ],
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert thought.content.startswith("I am carrying")
    assert captured["messages"][1]["role"] == "system"
    assert "RECENT COMPLETED LIVE DESKTOP CONVERSATION" in captured["messages"][1]["content"]
    assert captured["messages"][2]["role"] == "user"
    assert captured["messages"][2]["content"] == "The live desktop lane lost context."
    assert captured["messages"][3]["role"] == "assistant"
    assert (
        captured["messages"][3]["content"]
        == "I should preserve bounded recent exchanges through CognitiveEngine."
    )
    user_message = captured["messages"][-1]["content"]
    assert "[CURRENT USER MESSAGE]" not in user_message
    assert "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]" not in user_message
    assert "Continue from there." in user_message


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_memory_state_contract_uses_canonical_evidence_not_history(
    monkeypatch,
):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return (
                'I pinned "silver lantern" and I am keeping attention on the live '
                "desktop thread right now."
            )

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": (
                "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now."
            ),
            "memory_state_contract": True,
            "canonical_memory_state_evidence": (
                "status=session_memory_pin\n"
                'I\'ve pinned "silver lantern" in durable session memory.'
            ),
            "recent_completed_exchanges": [
                {
                    "user": "What pitch?",
                    "aura": "A stale pitch reply that must not leak into memory recall.",
                }
            ],
            "live_mind_context_required": True,
            "live_runtime_payload_required": True,
            "live_mind_context": {
                "required_for_live_desktop": True,
                "must_answer_from_full_mind_path": True,
                "required_subsystems_ok": True,
                "lane": {"state": "ready", "conversation_ready": True},
                "voice": {"attention": "current user message"},
            },
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert "silver lantern" in thought.content
    assert captured["memory_state_contract"] is True
    assert captured["clean_user_surface_contract"] is True
    assert captured["clean_user_surface_recurrent_loops"] == 1
    assert captured["clean_user_surface_steering_alpha"] <= 0.35
    assert len(captured["messages"]) == 3
    assert "RECENT COMPLETED LIVE DESKTOP CONVERSATION" not in captured["messages"][0]["content"]
    assert "CANONICAL MEMORY STATE EVIDENCE" in captured["messages"][-2]["content"]
    assert "silver lantern" in captured["messages"][-2]["content"]
    assert "stale pitch" not in captured["messages"][-2]["content"].lower()
    assert captured["messages"][-1]["content"].startswith("Remember this phrase")


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_includes_live_mind_context_without_payload_duplication(
    monkeypatch,
):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am answering from the live mind path instead of a detached assistant prompt."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "You with me?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "You with me?",
            "live_mind_context_required": True,
            "live_runtime_payload_required": True,
            "live_mind_context": {
                "required_for_live_desktop": True,
                "must_answer_from_full_mind_path": True,
                "required_subsystems_ok": True,
                "lane": {"state": "ready", "conversation_ready": True},
                "voice": {"mood": "steady"},
                "governance": {"legacy_fallback_allowed": False},
            },
            "mind_context_contract": "Answer through the live mind context.",
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert thought.content.startswith("I am answering from the live mind path")
    assert captured["skip_runtime_payload"] is True
    grounding_prompt = captured["messages"][-2]["content"]
    assert "LIVE MIND CONTEXT" in grounding_prompt
    assert "must_answer_from_full_mind_path" in grounding_prompt
    assert "Answer through the live mind context." in grounding_prompt
    assert captured["messages"][-1]["content"] == "You with me?"


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_uses_compact_grounding_when_required(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am answering from the live path with the current thread in view."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Hey Aura, are you there?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "live_runtime_payload_required": True,
            "visible_user_message": "Hey Aura, are you there?",
            "live_speech_grounding_frame": {
                "attention_focus": "Bryan's live desktop check",
                "dominant_action": "answer",
                "mood": "steady",
            },
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert thought.content.startswith("I am answering")
    assert captured["skip_runtime_payload"] is True
    assert captured["allow_cloud_fallback"] is False
    grounding_prompt = captured["messages"][-2]["content"]
    assert "LIVE SPEECH GROUNDING" in grounding_prompt
    assert "not prose to repeat" in grounding_prompt
    assert captured["messages"][-1]["content"] == "Hey Aura, are you there?"


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_failure_does_not_enter_second_model_path(
    monkeypatch,
):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    phase_calls = 0
    router_calls = 0
    degradation_calls = []

    class _Phase:
        async def execute(self, *_args, **_kwargs):
            nonlocal phase_calls
            phase_calls += 1
            raise AssertionError("failed compact desktop turn must not enter the full phase loop")

    class _Router:
        async def think(self, **_kwargs):
            nonlocal router_calls
            router_calls += 1
            raise RuntimeError("empty_generation_exhausted")

    engine._phases = [_Phase()]
    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )
    monkeypatch.setattr(
        "core.brain.cognitive_engine.record_degradation",
        lambda *args, **kwargs: degradation_calls.append((args, kwargs)),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Tell me about distributed systems.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
        },
        is_background=False,
        timeout_s=90.0,
    )

    assert router_calls == 1
    assert phase_calls == 0
    assert thought.metadata["desktop_cognitive_engine_failure"] is True
    assert thought.metadata["model_retry_suppressed"] is True
    assert "won't fabricate" in thought.content
    assert len(degradation_calls) == 1
    assert degradation_calls[0][0][0] == "cognitive_engine"
    assert degradation_calls[0][1]["enforce_failure_policy"] is False


def test_desktop_failure_thought_preserves_latent_attempt_receipt():
    engine = CognitiveEngine()
    receipt = {
        "episode_id": "live-timeout",
        "params_unchanged": True,
        "last_stage": "decode",
    }

    thought = engine._desktop_cognitive_failure_thought(
        ThinkingMode.DEEP,
        "latent_timeout:cooperative_cancelled",
        generation_metadata={
            "generation_failure_class": "latent_timeout:cooperative_cancelled",
            "latent_cortex_selected": True,
            "latent_cortex_prompt_shape": {
                "question_parts": 4,
                "requires_single_reply_coverage": True,
            },
            "latent_cortex_attempted": True,
            "latent_cortex_succeeded": False,
            "latent_cortex_fallback_used": True,
            "latent_cortex_failure_reason": "latent_timeout:cooperative_cancelled",
            "latent_cortex_receipt": receipt,
            "latent_cortex_ingress": {"schema": "aura.cognitive_ingress.v1"},
            "latent_cortex_progress": {"stage": "decode", "elapsed_s": 118.0},
            "live_mind_controls_bound": True,
            "response_path": "cognitive_engine_generation_timeout",
        },
    )

    assert thought.metadata["model_retry_suppressed"] is True
    assert thought.metadata["latent_cortex_selected"] is True
    assert thought.metadata["latent_cortex_attempted"] is True
    assert thought.metadata["latent_cortex_succeeded"] is False
    assert thought.metadata["latent_cortex_receipt"] == receipt
    assert thought.metadata["latent_cortex_prompt_shape"]["question_parts"] == 4
    assert (
        thought.metadata["latent_cortex_ingress"]["schema"]
        == "aura.cognitive_ingress.v1"
    )
    assert thought.metadata["latent_cortex_progress"]["stage"] == "decode"
    assert (
        thought.metadata["generation_failure_class"]
        == "latent_timeout:cooperative_cancelled"
    )


@pytest.mark.asyncio
async def test_cognitive_engine_preserves_worker_quality_rejection_metadata(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)

    class _Router:
        async def think(self, **_kwargs):
            return None

        @staticmethod
        def get_last_generation_metadata():
            return {
                "error": "surface_quality_rejected",
                "surface_control_receipt": {
                    "surface_quality_gate_enabled": True,
                    "surface_quality_gate_passed": False,
                    "surface_quality_gate_attempts": 3,
                    "surface_quality_gate_reasons": [
                        "missing_requested_word_count"
                    ],
                },
            }

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router()
            if name == "llm_router"
            else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "In exactly five words, state why checksums matter.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
        },
        is_background=False,
        timeout_s=30.0,
    )

    assert thought.metadata["generation_failure_class"] == "surface_quality_rejected"
    receipt = thought.metadata["live_mind_surface_control_receipt"]
    assert receipt["surface_quality_gate_attempts"] == 3
    assert receipt["surface_quality_gate_passed"] is False


@pytest.mark.asyncio
async def test_cognitive_engine_contains_empty_generation_exhaustion(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    degradation_calls = []

    class _Router:
        async def think(self, **_kwargs):
            return None

        @staticmethod
        def get_last_generation_metadata():
            return {"error": "empty_generation_exhausted"}

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router()
            if name == "llm_router"
            else default
        ),
    )
    monkeypatch.setattr(
        "core.brain.cognitive_engine.record_degradation",
        lambda *args, **kwargs: degradation_calls.append((args, kwargs)),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Give me a complete live answer.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
        },
        is_background=False,
        timeout_s=30.0,
    )

    assert thought.metadata["generation_failure_class"] == "empty_generation_exhausted"
    assert len(degradation_calls) == 1
    assert degradation_calls[0][1]["enforce_failure_policy"] is False


@pytest.mark.asyncio
async def test_cognitive_engine_strict_answer_recovery_propagates_cancellation(monkeypatch):
    import core.brain.llm_health_router as router_module

    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = None
    engine._phases = []

    class _CancellingRouter:
        async def think(self, *args, **kwargs):
            await asyncio.sleep(0)
            raise asyncio.CancelledError()

    monkeypatch.setattr(router_module, "get_llm_router", lambda: _CancellingRouter())

    with pytest.raises(asyncio.CancelledError):
        await engine._run_thinking_loop(
            state,
            "Solve exactly. <answer>required</answer>",
            ThinkingMode.FAST,
            "user",
            # The contract comes from the CALLER now. A "<answer>" substring
            # in the objective is text a person can type, and this recovery
            # sends the whole objective to a cloud provider.
            {"strict_answer_contract": True},
        )


@pytest.mark.asyncio
async def test_cognitive_engine_strict_answer_recovery_records_typed_failure(monkeypatch):
    import core.brain.llm_health_router as router_module

    tracker = get_degradation_tracker()
    tracker.reset()

    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = None
    engine._phases = []

    class _FailingRouter:
        async def think(self, *args, **kwargs):
            await asyncio.sleep(0)
            raise RuntimeError("router offline")

    monkeypatch.setattr(router_module, "get_llm_router", lambda: _FailingRouter())

    thought = await engine._run_thinking_loop(
        state,
        "Solve exactly. <answer>required</answer>",
        ThinkingMode.FAST,
        "user",
        {"strict_answer_contract": True},
    )

    assert thought.content == ""
    assert "strict_answer_recovery_failed" in thought.reasoning[0]
    records = tracker.recent(subsystem="cognitive_engine", limit=8)
    assert records
    # Searched, not indexed: this turn legitimately records more than one
    # degradation (an engine with no repository also withholds closure), and
    # asserting a position makes the test fail on an unrelated true record.
    assert any(
        record.action
        == "returned strict answer recovery failure after direct recovery failed"
        for record in records
    )
    tracker.reset()
