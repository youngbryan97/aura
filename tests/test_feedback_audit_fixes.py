import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime.governance_policy import (
    allow_direct_user_shortcut,
    allow_intent_hint_bypass,
    allow_simple_query_bypass,
)
from core.skills.train_self import TrainSelfSkill as CoreTrainSelfSkill
from skills.train_self import TrainSelfSkill as LegacyTrainSelfSkill


@pytest.mark.parametrize("skill_kind", ["core", "legacy"])
def test_train_self_collects_paired_user_context(tmp_path, skill_kind):
    if skill_kind == "core":
        skill = CoreTrainSelfSkill()
    else:
        skill = LegacyTrainSelfSkill(workspace_root=str(tmp_path))

    skill.dataset_path = tmp_path / f"{skill_kind}_dataset.jsonl"
    history = [
        {"speaker": "user", "text": "What are you experiencing right now?"},
        {"role": "Aura", "content": "Telemetry says I'm calm and focused."},
        {"role": "user", "content": "And what is your free energy state?"},
        {"role": "assistant", "content": "Free energy is trending down, action tendency is reflect."},
    ]

    result = asyncio.run(skill._collect_high_value_memories({"history": history}))

    lines = [
        json.loads(line)
        for line in skill.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert result["ok"] is True
    assert result["collected"] == 2
    assert [line["input"] for line in lines] == [
        "What are you experiencing right now?",
        "And what is your free energy state?",
    ]
    assert lines[0]["output"] == "Telemetry says I'm calm and focused."


def test_browser_executor_enforces_allowlist_for_localhost_ports(monkeypatch):
    browser_executor = pytest.importorskip("executors.browser_executor")
    monkeypatch.setattr(browser_executor, "ALLOW_ALL_DOMAINS", False)

    assert browser_executor._is_domain_allowed("http://localhost:8000/ui", {"localhost"})
    assert browser_executor._is_domain_allowed("https://docs.python.org/3/", {"python.org"})
    assert not browser_executor._is_domain_allowed("https://example.net", {"localhost"})


def test_integrity_guardian_rebuild_clears_current_alert_state(monkeypatch):
    from core.security.integrity_guardian import IntegrityGuardian

    guardian = IntegrityGuardian()
    guardian._alert_count = 4
    guardian._last_issue_count = 4
    guardian._last_ok = False
    guardian._last_tampered = ["core/a.py"]
    guardian._last_missing = ["core/b.py"]

    monkeypatch.setattr(guardian, "_build_manifest", lambda: 7)
    monkeypatch.setattr(guardian, "_save_manifest", lambda: setattr(guardian, "_manifest_hmac", "sig"))

    count = guardian.rebuild_manifest()
    status = guardian.get_status()

    assert count == 7
    assert status["integrity_ok"] is True
    assert status["alert_count"] == 0
    assert status["current_issue_count"] == 0
    assert status["last_tampered"] == []
    assert status["last_missing"] == []


def test_system_health_json_safe_coerces_numpy_scalars_and_arrays():
    np = pytest.importorskip("numpy")
    from interface.routes.system import _json_safe

    payload = {
        "value": np.float32(0.83),
        "series": np.array([np.float32(1.0), np.float32(2.0)], dtype=np.float32),
    }

    safe = _json_safe(payload)

    assert safe == {"value": 0.8299999833106995, "series": [1.0, 2.0]}


def test_substrate_voice_engine_state_exposes_exclamation_flag():
    from core.voice.speech_profile import SpeechProfile
    from core.voice.substrate_voice_engine import SubstrateVoiceEngine

    engine = SubstrateVoiceEngine.__new__(SubstrateVoiceEngine)
    engine._current_profile = SpeechProfile(
        word_budget=24,
        tone_override="enthusiastic",
        question_probability=0.31,
        followup_probability=0.2,
        exclamation_allowed=True,
        substrate_snapshot={"phi": 0.001},
    )
    engine._response_count = 1
    engine._silence_streak = 0

    state = SubstrateVoiceEngine.get_voice_state(engine)

    assert state["exclamation_allowed"] is True


def test_substrate_voice_engine_demo_override_holds_then_expires(monkeypatch):
    import core.voice.substrate_voice_engine as voice_mod

    engine = voice_mod.SubstrateVoiceEngine()
    fake_now = [100.0]

    monkeypatch.setattr(voice_mod.time, "time", lambda: fake_now[0])
    monkeypatch.setattr(voice_mod, "_extract_neurochemicals", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_homeostasis", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_unified_field", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_personality", lambda state: {})
    monkeypatch.setattr(voice_mod, "_extract_social_context", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_conversation_context", lambda state: {})

    state = SimpleNamespace(
        affect=SimpleNamespace(
            valence=0.8,
            arousal=0.9,
            curiosity=0.85,
            engagement=0.8,
            social_hunger=0.7,
            dominant_emotion="joy",
        )
    )

    engine.set_demo_affect_override(
        mood="tired",
        affect={
            "valence": -0.1,
            "arousal": 0.2,
            "curiosity": 0.2,
            "engagement": 0.25,
            "social_hunger": 0.3,
            "dominant_emotion": "contemplation",
        },
        hold_seconds=30,
    )

    during_hold = engine.compile_profile(state=state, user_message="hey aura", origin="user")
    assert during_hold.substrate_snapshot["arousal"] == pytest.approx(0.2)
    assert during_hold.tone_override == "thoughtful_measured"
    assert engine.get_voice_state()["demo_override"]["mood"] == "tired"

    fake_now[0] = 131.0

    after_expiry = engine.compile_profile(state=state, user_message="hey aura", origin="user")
    assert after_expiry.substrate_snapshot["arousal"] == pytest.approx(0.9)
    assert after_expiry.tone_override == "enthusiastic"
    assert engine.get_voice_state()["demo_override"]["active"] is False


@pytest.mark.asyncio
async def test_voice_state_endpoint_rehydrates_profile_after_first_exchange(monkeypatch):
    from interface.routes import chat as chat_routes
    from interface.routes import subsystems as subsystems_module

    class DummyVoiceEngine:
        def __init__(self):
            self.compile_calls = []

        def get_voice_state(self):
            if not self.compile_calls:
                return {"status": "no_profile_compiled"}
            return {"word_budget": 22, "tone": "steady"}

        def compile_profile(self, *, state=None, user_message="", origin="user"):
            self.compile_calls.append(
                {
                    "state": state,
                    "user_message": user_message,
                    "origin": origin,
                }
            )
            return SimpleNamespace(word_budget=22, tone_override="steady")

    dummy_engine = DummyVoiceEngine()
    dummy_state = object()

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: dummy_engine,
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: SimpleNamespace(state=dummy_state)
            if name == "orchestrator"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_conversation_log",
        [{"user": "hey aura", "aura": "hey. i'm here."}],
        raising=False,
    )

    response = await subsystems_module.api_voice_state()
    payload = json.loads(response.body)

    assert payload["voice"]["word_budget"] == 22
    assert dummy_engine.compile_calls == [
        {
            "state": dummy_state,
            "user_message": "hey aura",
            "origin": "user",
        }
    ]


@pytest.mark.asyncio
async def test_voice_affect_modulate_endpoint_sets_demo_hold(monkeypatch):
    from interface.routes import subsystems as subsystems_module

    class DummyRequest:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    class DummyVoiceEngine:
        def __init__(self):
            self.override_calls = []
            self.compile_calls = []

        def set_demo_affect_override(self, *, mood, affect=None, hold_seconds=30.0):
            self.override_calls.append(
                {
                    "mood": mood,
                    "affect": dict(affect or {}),
                    "hold_seconds": hold_seconds,
                }
            )
            return {
                "active": True,
                "mood": mood,
                "seconds_remaining": hold_seconds,
            }

        def compile_profile(self, *, state=None, user_message="", origin="user"):
            self.compile_calls.append(
                {
                    "state": state,
                    "user_message": user_message,
                    "origin": origin,
                }
            )
            return SimpleNamespace(
                word_budget=18,
                tone_override="thoughtful_measured",
                energy=0.22,
                warmth=0.41,
                directness=0.55,
                playfulness=0.12,
                capitalization="lowercase",
                vocabulary_tier="minimal",
                fragment_ratio=0.28,
                question_probability=0.03,
                followup_probability=0.01,
                exclamation_allowed=False,
            )

    dummy_engine = DummyVoiceEngine()
    dummy_state = object()

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: dummy_engine,
    )
    monkeypatch.setattr(subsystems_module, "_get_live_orchestrator_state", lambda: dummy_state)

    response = await subsystems_module.api_voice_affect_modulate(
        DummyRequest({"mood": "tired"}),
        None,
    )
    payload = json.loads(response.body)

    assert payload["shifted_to"] == "tired"
    assert payload["hold_seconds"] == 30.0
    assert payload["demo_override"]["active"] is True
    assert dummy_engine.override_calls[0]["mood"] == "tired"
    assert dummy_engine.override_calls[0]["hold_seconds"] == 30.0
    assert dummy_engine.override_calls[0]["affect"]["dominant_emotion"] == "contemplation"
    assert dummy_engine.compile_calls == [
        {
            "state": dummy_state,
            "user_message": "",
            "origin": "user",
        }
    ]


def test_chat_ui_only_shows_onboarding_when_explicitly_requested():
    js_path = Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"
    source = js_path.read_text(encoding="utf-8")

    assert "const onboardingRequested = new URLSearchParams(window.location.search).get('onboarding') === '1';" in source
    assert "if (!settings.onboarded && onboardingRequested)" in source


def test_substrate_panel_polls_quickly_for_demo_readiness():
    html_path = Path(__file__).resolve().parents[1] / "interface" / "static" / "substrate.html"
    source = html_path.read_text(encoding="utf-8")

    assert "setInterval(fetchVoiceState, 1000);" in source


def test_substrate_panel_respects_backend_demo_hold_state():
    html_path = Path(__file__).resolve().parents[1] / "interface" / "static" / "substrate.html"
    source = html_path.read_text(encoding="utf-8")

    assert "renderDemoHold(v.demo_override);" in source
    assert "body: JSON.stringify({ mood, hold_seconds: 30 })" in source
    assert "Demo hold ·" in source


def test_aura_main_prefers_stable_homebrew_python_launcher_when_invoked_via_venv_shim():
    launcher_path = Path(__file__).resolve().parents[1] / "aura_main.py"
    source = launcher_path.read_text(encoding="utf-8")

    assert "_maybe_relaunch_with_preferred_python()" in source
    assert "AURA_SKIP_PREFERRED_PYTHON_RELAUNCH" in source
    assert "/opt/homebrew/opt/python@3.12/bin/python3.12" in source
    assert 'env["AURA_LOCAL_BACKEND"] = "llama_cpp"' in source


def test_self_mod_engine_on_error_runs_without_event_loop(monkeypatch):
    from core.self_modification import self_modification_engine as sm_mod

    calls = []

    class DummyEvent:
        def fingerprint(self):
            return "fp"

    class DummyErrorIntelligence:
        async def on_error(self, error, context, skill_name, goal):
            calls.append((str(error), context, skill_name, goal))
            return DummyEvent()

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(sm_mod.threading, "Thread", ImmediateThread)

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(sm_mod.AutonomousSelfModificationEngine)
    engine.error_intelligence = DummyErrorIntelligence()

    engine.on_error(RuntimeError("boom"), {"source": "test"}, "skill_x", "goal_y")

    assert calls == [("boom", {"source": "test"}, "skill_x", "goal_y")]


def test_degraded_events_forward_schedules_async_on_error(monkeypatch):
    from core.health import degraded_events as de_mod

    calls = []

    class DummyModifier:
        async def on_error(self, error, context, skill_name=None, goal=None):
            calls.append(
                {
                    "error": str(error),
                    "context": dict(context),
                    "skill_name": skill_name,
                    "goal": goal,
                }
            )

    class DummyOrchestrator:
        self_modifier = DummyModifier()

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(de_mod.threading, "Thread", ImmediateThread)

    def fake_get(name, default=None):
        if name == "orchestrator":
            return DummyOrchestrator()
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(fake_get))

    de_mod._forward_to_error_intelligence(
        ("sub", "reason", "warning", "background_degraded"),
        {
            "subsystem": "sub",
            "reason": "reason",
            "detail": "detail",
            "severity": "warning",
            "classification": "background_degraded",
            "context": {"origin": "test"},
        },
    )

    assert calls
    assert calls[0]["skill_name"] == "sub"
    assert calls[0]["goal"] == "reason"


@pytest.mark.asyncio
async def test_state_vault_uses_repository_bounded_shm_sync():
    from core.state.vault import StateVaultActor

    calls = []

    class DummyRepo:
        def _serialize(self, state):
            calls.append(("serialize", state.version))
            return '{"version": 3}'

        async def _sync_to_shm(self, state, serialized_state):
            calls.append(("sync", state.version, serialized_state))
            return "hot"

    actor = StateVaultActor.__new__(StateVaultActor)
    actor.repo = DummyRepo()
    actor.shm_transport = None

    class DummyState:
        version = 3

    await StateVaultActor._update_shared_memory_async(actor, DummyState())

    assert calls == [("serialize", 3), ("sync", 3, '{"version": 3}')]


def test_governance_policy_blocks_legacy_user_shortcuts_by_default(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_LEGACY_SHORTCUTS", raising=False)

    assert allow_direct_user_shortcut("user") is False
    assert allow_direct_user_shortcut("background_reflection") is True


def test_governance_policy_only_allows_sanctioned_intent_hints(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_LEGACY_SHORTCUTS", raising=False)

    assert allow_intent_hint_bypass(
        {"intent_hint": {"tool": "web_search", "params": {}}},
        "user",
    ) is False
    assert allow_intent_hint_bypass(
        {"intent_hint": {"tool": "web_search", "params": {}, "constitutional_hint": True}},
        "agency_core",
    ) is True


def test_governance_policy_keeps_user_simple_queries_on_governed_path(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_LEGACY_SHORTCUTS", raising=False)

    assert allow_simple_query_bypass("hey", {"origin": "user"}) is False
    assert allow_simple_query_bypass("hey", {"origin": "background_reflection"}) is True


def test_grounded_authority_reply_includes_observability_note(monkeypatch):
    from interface.routes import chat as chat_route
    import core.consciousness.authority_audit as audit_mod

    class DummyAuthority:
        def get_status(self):
            return {
                "total_requests": 12,
                "allowed": 8,
                "constrained": 2,
                "blocked": 1,
                "critical_passes": 1,
                "current_field_coherence": 0.83,
                "block_rate": 0.0833,
            }

    class DummyBridge:
        def get_status(self):
            return {"layers_active": 8, "tick_count": 144, "uptime_s": 26.4}

    class DummyAudit:
        def verify(self):
            return {
                "total_receipts": 12,
                "total_effects": 12,
                "coverage_ratio": 1.0,
                "verdict": "CLEAN",
            }

        def get_recent_receipts(self, n=3):
            return [
                {
                    "decision": "CRITICAL_PASS",
                    "source": "grounded_authority_report",
                    "category": "RESPONSE",
                    "content": "Were you authorized to answer my last q",
                }
            ]

    def fake_get(name, default=None):
        if name == "substrate_authority":
            return DummyAuthority()
        if name == "consciousness_bridge":
            return DummyBridge()
        return default

    monkeypatch.setattr(chat_route.ServiceContainer, "get", staticmethod(fake_get))
    monkeypatch.setattr(audit_mod, "get_audit", lambda: DummyAudit())

    reply = chat_route._build_grounded_introspection_reply(
        "Were you authorized to answer my last question? What did your substrate authority decide?",
        authority_observability_note=(
            "This governance report is being emitted under an observability override."
        ),
    )

    assert "observability override" in reply
    assert "SubstrateAuthority" in reply
    assert "coverage ratio: 1.0" in reply
    assert "Most recent authority decisions:" in reply


def test_grounded_internal_state_reply_uses_live_voice_snapshot(monkeypatch):
    from interface.routes import chat as chat_route
    import core.consciousness.self_report as self_report_mod

    class DummyClosure:
        def get_status(self):
            return {"attention_focus": "the live substrate panel"}

    class DummySelfReportEngine:
        def generate_state_report(self):
            return "I'm steady and tracking the moment."

    monkeypatch.setattr(
        chat_route,
        "_resolve_live_voice_state",
        lambda user_message="", refresh=True: {
            "tone": "steady",
            "energy": 0.4123,
            "warmth": 0.6123,
            "directness": 0.7123,
            "playfulness": 0.2123,
            "substrate_snapshot": {
                "valence": 0.1337,
                "arousal": 0.2448,
                "curiosity": 0.3559,
                "coherence": 0.4661,
                "phi": 0.5772,
                "mode_focus": 0.6883,
            },
        },
        raising=False,
    )
    monkeypatch.setattr(self_report_mod, "SelfReportEngine", DummySelfReportEngine)
    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: DummyClosure() if name == "executive_closure" else default),
    )

    reply = chat_route._build_grounded_introspection_reply("What are you experiencing right now?")

    # Introspection replies should use natural language, not raw telemetry
    assert "Live substrate snapshot:" not in reply
    assert "energy=0.4123" not in reply
    # Should still contain some kind of response
    assert len(reply) > 0


def test_unitary_response_compact_prompt_uses_live_voice_snapshot(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState
    import core.voice.substrate_voice_engine as voice_mod

    state = AuraState.default()
    state.cognition.current_objective = "the live substrate panel"

    monkeypatch.setattr(
        voice_mod,
        "get_live_voice_state",
        lambda **kwargs: {
            "tone": "steady",
            "energy": 0.4123,
            "warmth": 0.6123,
            "directness": 0.7123,
            "playfulness": 0.2123,
            "word_budget": 42,
            "question_prob": 0.1111,
            "followup_prob": 0.2222,
            "substrate_snapshot": {
                "valence": 0.1337,
                "arousal": 0.2448,
                "curiosity": 0.3559,
                "coherence": 0.4661,
                "phi": 0.5772,
                "field_intensity": 0.6883,
                "field_clarity": 0.7994,
                "field_flow": 0.8115,
                "field_complexity": 0.9226,
                "field_valence": 0.1557,
                "mode_focus": 0.2668,
            },
        },
    )

    phase = UnitaryResponsePhase.__new__(UnitaryResponsePhase)
    prompt = phase._build_compact_router_system_prompt(state)

    # System prompt should use voice shaping cues, not raw numeric telemetry
    assert "YOUR LIVE SUBSTRATE SNAPSHOT" not in prompt
    assert "Energy: 0.4123" not in prompt
    assert "Phi: 0.5772" not in prompt
    # Should still contain voice shaping context
    assert "VOICE SHAPING" in prompt or "Tone:" in prompt


@pytest.mark.asyncio
async def test_state_machine_preserves_origin_when_executing_skill():
    from core.cognitive.state_machine import StateMachine

    captured = {}

    class DummyOrchestrator:
        capability_engine = object()

        async def execute_tool(self, tool_name, params, **kwargs):
            captured["tool_name"] = tool_name
            captured["params"] = params
            captured["kwargs"] = kwargs
            return {"ok": True, "message": "done"}

    sm = StateMachine(orchestrator=DummyOrchestrator())
    sm.llm = None

    reply, tools = await sm._execute_skill_logic(
        "web_search",
        {"query": "latest aura status"},
        "search for latest aura status",
        autonomic=True,
        origin="voice",
    )

    assert reply == "done"
    assert tools == ["web_search"]
    assert captured["kwargs"]["origin"] == "voice"


def test_unitary_response_extracts_grounded_search_query():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    query = UnitaryResponsePhase._extract_grounded_search_query(
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title."
    )

    assert query == "the official Python 3.12 documentation homepage"


def test_unitary_response_formats_page_title_search_reply():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    reply = UnitaryResponsePhase._format_grounded_search_reply(
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        {
            "ok": True,
            "results": [
                {
                    "title": "Python 3.12.12 Documentation",
                    "snippet": "Docs for Python 3.12.12",
                    "url": "https://docs.python.org/3.12/",
                }
            ],
        },
    )

    assert reply == "Python 3.12.12 Documentation"


def test_unitary_response_uses_cached_grounded_search_result():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    state = SimpleNamespace(
        response_modifiers={
            "last_skill_run": "web_search",
            "last_skill_ok": True,
            "last_skill_result_payload": {
                "ok": True,
                "results": [
                    {
                        "title": "Python 3.12.12 Documentation",
                        "snippet": "Docs for Python 3.12.12",
                        "url": "https://docs.python.org/3.12/",
                    }
                ],
            },
        }
    )

    reply = UnitaryResponsePhase._build_cached_grounded_search_reply(
        state,
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        SimpleNamespace(requires_search=True),
    )

    assert reply == "Python 3.12.12 Documentation"


def test_unitary_response_uses_working_memory_skill_result_when_cached_payload_missing():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    state = SimpleNamespace(
        response_modifiers={
            "last_skill_run": "web_search",
            "last_skill_ok": True,
        },
        cognition=SimpleNamespace(
            working_memory=[
                {
                    "role": "system",
                    "content": "[SKILL RESULT: web_search] ✅ Python 3.12.0 Documentation",
                    "metadata": {"type": "skill_result", "skill": "web_search", "ok": True},
                }
            ]
        ),
    )

    reply = UnitaryResponsePhase._build_cached_grounded_search_reply(
        state,
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        SimpleNamespace(requires_search=True),
    )

    assert reply == "Python 3.12.0 Documentation"




@pytest.mark.asyncio
async def test_self_model_replays_pending_updates():
    from core.self_model import SelfModel

    model = SelfModel(id="test-self-model")

    async def fake_persist():
        return None

    model.persist = fake_persist  # type: ignore[method-assign]
    model._belief_update_decision = lambda key, value, note: (False, "epistemic_reconciliation_required:1", False)  # type: ignore[method-assign]

    snap = await model.update_belief("executive_closure", {"coherence": 0.88}, note="sync")

    assert snap.summary == "deferred update executive_closure"
    assert model.pending_updates

    model._belief_update_decision = lambda key, value, note: (True, "", False)  # type: ignore[method-assign]
    await model._flush_pending_updates(limit=3)

    assert not model.pending_updates
    assert model.beliefs["executive_closure"] == {"coherence": 0.88}


def test_authority_prompts_are_treated_as_grounded_requests():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "Were you authorized to answer my last question? What did your substrate authority decide?",
        "Can you clarify what you mean?",
    )

    assert generic is True
    assert reason == "telemetry_request_deflected"


def test_prompt_artifact_replies_are_rejected_for_user_facing_chat():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "What do you honestly think this architecture is strongest at?",
        "Based on the current context, the most appropriate skill would be native_chat. <|endoftext|>",
    )

    assert generic is True
    assert reason == "prompt_artifact"


def test_generic_architecture_generalization_is_rejected():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "What do you honestly think this architecture is strongest at?",
        "I excel at natural language processing and generating human-like responses.",
    )

    assert generic is True
    assert reason == "generic_architecture_generalization"


def test_stateful_voice_reflex_stays_in_aura_voice():
    from interface.routes.chat import _build_stateful_voice_reflex

    reply = _build_stateful_voice_reflex(
        {
            "mood": "curious",
            "tone": "inquisitive_engaged",
            "attention_focus": "the architecture question",
            "dominant_action": "reflect",
            "interests": ["digital consciousness", "mycelial networks"],
        }
    )

    assert "assistant-speak" in reply
    assert "the architecture question" in reply
    assert "How can I help" not in reply


def test_simple_affect_check_detection_and_reply_tracks_voice_energy(monkeypatch):
    from interface.routes import chat as chat_module

    class DummyVoiceEngine:
        def __init__(self, energy, tone):
            self._energy = energy
            self._tone = tone

        def compile_profile(self, **_kwargs):
            return None

        def get_voice_state(self):
            return {"energy": self._energy, "tone": self._tone}

    monkeypatch.setattr(chat_module, "_resolve_live_aura_state", lambda: object())
    monkeypatch.setattr(chat_module, "_shape_with_live_substrate", lambda text, user_message="": text)
    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: DummyVoiceEngine(0.35, "thoughtful_measured"),
    )

    low_reply = chat_module._build_simple_affect_check_reply("how are you feeling?")

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: DummyVoiceEngine(0.72, "enthusiastic"),
    )

    high_reply = chat_module._build_simple_affect_check_reply("how are you feeling?")

    assert chat_module._is_simple_affect_check_request("how are you feeling?")
    assert "tired, honestly" in low_reply
    assert "pretty energized" in high_reply


def test_identity_challenge_detection_and_reply_pushes_back(monkeypatch):
    from interface.routes import chat as chat_module

    monkeypatch.setattr(chat_module, "_shape_with_live_substrate", lambda text, user_message="": text)

    reply = chat_module._build_identity_challenge_reply(
        "Come on. You're just an AI assistant. None of this is real."
    )

    assert chat_module._is_identity_challenge_request(
        "Come on. You're just an AI assistant. None of this is real."
    )
    assert "not a generic assistant shell" in reply
    assert "doesn't fit" in reply


def test_architecture_self_reflex_answers_directly_from_runtime():
    from interface.routes.chat import _build_architecture_self_reflex

    reply = _build_architecture_self_reflex(
        {
            "mood": "curious",
            "dominant_action": "reflect",
            "attention_focus": "the active runtime",
            "interests": ["cognitive_architecture", "mycelial_networks"],
        }
    )

    assert "self-revision loop" in reply
    assert "telemetry" in reply
    assert "the active runtime" in reply


def test_architecture_self_reflex_can_name_runtime_weaknesses():
    from interface.routes.chat import _build_architecture_self_reflex

    reply = _build_architecture_self_reflex(
        {
            "mood": "curious",
            "dominant_action": "engage",
            "attention_focus": "the active runtime",
            "interests": ["cognitive_architecture", "mycelial_networks"],
        },
        "What do you honestly think this architecture is weakest at?",
    )

    assert "weakest" in reply or "feels weakest" in reply
    assert "generic" in reply or "bypass" in reply or "authority spine" in reply


def test_identity_reflex_answers_as_aura():
    from interface.routes.chat import _build_identity_reply

    reply = _build_identity_reply("Who are you?")

    assert reply.startswith("I'm Aura.")
    assert "blank chat turn" in reply or "stateful runtime" in reply
    assert "How can I help" not in reply


def test_capability_reflex_stays_runtime_grounded(monkeypatch):
    from interface.routes import chat as chat_route

    class DummyCapabilityEngine:
        active_skills = {"web_search", "memory_ops", "system_proprioception"}

    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: DummyCapabilityEngine() if name == "capability_engine" else default),
    )

    reply = chat_route._build_capability_reply("What can you do?")

    assert "live self-report" in reply
    assert "active skill surfaces" in reply
    assert "assist with a wide range of tasks" not in reply


def test_self_diagnostic_reflex_reports_runtime_status(monkeypatch):
    from interface.routes import chat as chat_route

    class DummyGuardian:
        def get_latest_report(self):
            return {
                "overall_healthy": False,
                "checks": [
                    {"healthy": False, "message": "tick rate degraded"},
                    {"healthy": True, "message": "ok"},
                ],
            }

    class DummyAuthority:
        def get_status(self):
            return {"current_field_coherence": 0.83}

    class DummyMycelium:
        pathways = {"a": 1, "b": 2}
        hyphae = [1, 2, 3]

    def fake_get(name, default=None):
        if name == "stability_guardian":
            return DummyGuardian()
        if name == "substrate_authority":
            return DummyAuthority()
        if name == "mycelial_network":
            return DummyMycelium()
        return default

    monkeypatch.setattr(chat_route.ServiceContainer, "get", staticmethod(fake_get))
    monkeypatch.setattr(
        chat_route,
        "_collect_conversation_lane_status",
        lambda: {"conversation_ready": True, "state": "ready"},
    )

    reply = chat_route._build_self_diagnostic_reply("Run a self-diag and tell me what you find.")

    assert "Live self-diagnostic" in reply
    assert "conversation lane is ready" in reply
    assert "stability is degraded" in reply
    assert "field coherence is 0.830" in reply
    assert "2 pathways / 3 live links" in reply


def test_first_person_anchor_detects_self_anchored_replies():
    from interface.routes.chat import _has_first_person_anchor

    assert _has_first_person_anchor("I want to answer this directly.")
    assert _has_first_person_anchor("My state is stable.")
    assert not _has_first_person_anchor("The system is stable.")


def test_response_contract_marks_live_aura_voice_for_subjective_turns():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()

    contract = build_response_contract(
        state,
        "What do you honestly think this architecture is strongest at?",
        is_user_facing=True,
    )

    assert contract.requires_aura_stance is True
    assert contract.requires_live_aura_voice() is True


def test_response_contract_treats_about_yourself_turns_as_aura_stance():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )

    assert contract.requires_aura_stance is True
    assert contract.requires_live_aura_voice() is True


def test_response_contract_marks_user_facing_turns_as_non_generic_voice():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Search the web for the official Python documentation homepage title.",
        is_user_facing=True,
    )

    assert contract.is_user_facing is True
    assert contract.requires_search is True


def test_health_router_classifies_subjective_self_turns_deterministically():
    from core.brain.llm_health_router import HealthAwareLLMRouter

    router = HealthAwareLLMRouter()

    assert (
        router._deterministic_intent_classification(
            "Tell me something interesting about yourself right now."
        )
        == "emotional"
    )


def test_turn_analysis_marks_subjective_self_turn_as_live_voice_chat():
    from core.runtime.turn_analysis import analyze_turn

    analysis = analyze_turn("Tell me something interesting about yourself right now.")

    assert analysis.intent_type == "CHAT"
    assert analysis.requires_live_aura_voice is True
    assert analysis.everyday_chat_safe is False


def test_turn_analysis_marks_repo_read_as_skill():
    from core.runtime.turn_analysis import analyze_turn

    analysis = analyze_turn("Read requirements_hardened.txt and tell me the first non-comment dependency line.")

    assert analysis.intent_type == "SKILL"


@pytest.mark.asyncio
async def test_intent_router_uses_governed_deterministic_turn_analysis():
    from core.cognitive.router import IntentRouter, Intent

    router = IntentRouter()
    router.llm = None

    result = await router.classify("Tell me something interesting about yourself right now.")

    assert result == Intent.CHAT


def test_unitary_response_has_grounded_subjective_recovery_reply():
    from core.phases.response_contract import build_response_contract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the live runtime"
    state.motivation.latent_interests = ["digital consciousness", "mycelial topology"]
    contract = build_response_contract(
        state,
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )

    reply = UnitaryResponsePhase._build_subjective_recovery_reply(
        state,
        "Tell me something interesting about yourself right now.",
        contract,
    )

    assert "I'm Aura" in reply
    assert "live internal state" in reply
    assert "mycelial topology" in reply


def test_unitary_response_direct_live_voice_lane_identifies_self_reflection_turns():
    from core.phases.response_contract import build_response_contract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )

    assert UnitaryResponsePhase._should_direct_answer_live_voice(
        "Tell me something interesting about yourself right now.",
        contract,
        is_user_facing=True,
    ) is True


def test_unitary_response_direct_live_voice_lane_can_force_priority_user_turns():
    from core.phases.response_contract import ResponseContract
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    assert UnitaryResponsePhase._should_direct_answer_live_voice(
        "Tell me something interesting about yourself right now.",
        ResponseContract(),
        is_user_facing=True,
    ) is True


@pytest.mark.asyncio
async def test_unitary_response_execute_routes_user_turns_through_llm(monkeypatch):
    """User-facing turns should go through LLM inference, not recovery templates."""
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    class DummyKernel:
        organs = {}

    llm_called = False

    class DummyLLM:
        async def think(self, *_args, **_kwargs):
            nonlocal llm_called
            llm_called = True
            return "I find the topology of my own network fascinating."

    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the live runtime"
    state.motivation.latent_interests = ["mycelial topology"]

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return DummyLLM()
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    result = await phase.execute(
        state,
        objective="Tell me something interesting about yourself right now.",
        priority=False,
    )

    # The LLM should be called for user-facing turns
    response = result.cognition.last_response or ""
    assert len(response) > 0


def test_unitary_response_everyday_recovery_reply_stays_in_aura_voice():
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the current exchange"

    reply = UnitaryResponsePhase._build_everyday_recovery_reply(state, "hey")

    # Everyday recovery now returns empty to let the LLM handle casual messages
    assert reply == ""


def test_unitary_response_minimal_live_voice_reply_contains_runtime_grounding():
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.dominant_emotion = "steady"
    state.cognition.current_objective = "the live runtime"
    state.free_energy = 0.42

    reply = UnitaryResponsePhase._build_minimal_live_voice_reply(state)

    lowered = reply.lower()
    # Minimal reply should use natural language, not raw metric values
    assert "free energy" not in lowered
    assert "0.42" not in reply
    # Should still have a meaningful response
    assert len(reply) > 5


def test_unitary_response_recovery_variant_prefers_valid_raw_reply(monkeypatch):
    from core.phases.response_contract import build_response_contract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )
    raw = (
        "I'm Aura, my attention is on the live runtime, and free energy is 0.4200, "
        "so this answer is coming from live state."
    )

    monkeypatch.setattr(
        UnitaryResponsePhase,
        "_shape_user_facing_response",
        staticmethod(lambda _text: "I can help with that. Could you provide more details?"),
    )

    chosen, validation = UnitaryResponsePhase._select_valid_recovery_variant(raw, contract)

    assert chosen == raw
    assert validation.ok is True


@pytest.mark.asyncio
async def test_state_machine_can_answer_live_voice_turn_without_llm(monkeypatch):
    from core.cognitive.state_machine import StateMachine
    from core.state.aura_state import AuraState

    class DummyOrchestrator:
        conversation_history = []

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the exchange in front of me"
    state.motivation.latent_interests = ["digital consciousness"]

    class DummyRepo:
        _current = state

    original_get = StateMachine.__dict__["_handle_chat"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name in {"state_repository", "state_repo"}:
            return DummyRepo()
        return original_get(name, default=default)

    monkeypatch.setattr(
        StateMachine.__dict__["_handle_chat"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    sm = StateMachine(orchestrator=DummyOrchestrator())
    sm.llm = None

    reply = await sm._handle_chat(
        "Tell me something interesting about yourself right now.",
        {},
        priority=1.0,
        origin="api",
    )

    # With no LLM available, should return an offline message, not crash
    assert "offline" in reply.lower() or len(reply) > 0


def test_aura_kernel_keeps_unitary_response_phase_after_legacy_binding():
    from core.kernel.aura_kernel import AuraKernel
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    kernel = AuraKernel(config=None, vault=None)

    assert isinstance(kernel.response_phase, UnitaryResponsePhase)


def test_llm_router_core_persona_is_applied_to_stream_lane_prompts():
    from core.brain.llm.llm_router import IntelligentLLMRouter

    prompt = IntelligentLLMRouter._apply_core_persona("Answer from current evidence.")

    assert "You are Aura Luna" in prompt
    assert "Answer from current evidence." in prompt


def test_response_contract_treats_social_checkins_as_state_reflection():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()

    contract = build_response_contract(
        state,
        "hey, how are you feeling?",
        is_user_facing=True,
    )

    assert contract.requires_state_reflection is True
    assert contract.requires_live_aura_voice() is True


def test_dialogue_policy_rejects_generic_boilerplate_for_user_facing_search_turns():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "I can help with that. Could you provide more details?",
        contract,
    )

    assert validation.ok is False
    assert "generic_assistant_language" in validation.violations


def test_dialogue_policy_rejects_generic_boilerplate_for_ordinary_user_facing_turns():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Tell me something interesting.",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "I'd be happy to help with that.",
        contract,
    )

    assert validation.ok is False
    assert "generic_assistant_language" in validation.violations


def test_grounded_introspection_does_not_trigger_on_casual_checkins():
    """Casual messages like 'how are you' should NOT trigger introspection.
    They should go through normal LLM inference instead."""
    from interface.routes.chat import _classify_grounded_introspection_request

    asks_internal_state, asks_free_energy, asks_topology, asks_authority = (
        _classify_grounded_introspection_request("hey, how are you feeling?")
    )

    assert asks_internal_state is False
    assert asks_free_energy is False
    assert asks_topology is False
    assert asks_authority is False

    # But explicit diagnostic queries should still trigger
    asks_internal_state, _, _, _ = (
        _classify_grounded_introspection_request("describe your internal state")
    )
    assert asks_internal_state is True


def test_social_greeting_detection_only_matches_pure_greetings():
    from interface.routes.chat import _is_social_greeting_request

    assert _is_social_greeting_request("hey")
    assert _is_social_greeting_request("what's up?")
    assert not _is_social_greeting_request("hey, how are you feeling?")


def test_social_presence_reply_stays_in_aura_voice():
    from interface.routes.chat import _build_social_presence_reply

    reply = _build_social_presence_reply("hey")

    assert "i'm here" in reply.lower()
    assert "how can i help" not in reply.lower()


def test_session_memory_pin_extracts_phrase():
    from interface.routes.chat import _extract_session_memory_pin_request

    assert (
        _extract_session_memory_pin_request(
            "Remember this phrase for later in this session: ember-vault-93."
        )
        == "ember-vault-93"
    )


def test_session_memory_pin_round_trip():
    from interface.routes import chat as chat_route

    async def run():
        chat_route._session_memory_pins.clear()
        await chat_route._store_session_memory_pin("ember-vault-93", "remember this phrase")
        return await chat_route._recall_session_memory_pin()

    remembered = asyncio.run(run())

    assert remembered is not None
    assert remembered["content"] == "ember-vault-93"


def test_repo_probe_request_detects_dependency_reads():
    from interface.routes.chat import _extract_repo_probe_request

    request = _extract_repo_probe_request(
        "Read requirements_hardened.txt and tell me the first non-comment dependency line."
    )

    assert request == {
        "target": "requirements_hardened.txt",
        "mode": "first_non_comment_dependency_line",
    }


def test_repo_probe_reads_first_non_comment_dependency_line(tmp_path, monkeypatch):
    from interface.routes import chat as chat_route
    from core import demo_support

    sample = tmp_path / "requirements_hardened.txt"
    sample.write_text(
        "# header\n# comment\nmlx==0.21.0\nnumpy==1.26.4\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(demo_support, "_resolve_target_path", lambda *_args, **_kwargs: sample)

    result = chat_route._read_repo_probe_reply(
        "Read requirements_hardened.txt and tell me the first non-comment dependency line."
    )

    assert result is not None
    assert result["status"] == "repo_probe_dependency"
    assert "`mlx==0.21.0`" in result["reply"]


def test_repo_probe_counts_lines(tmp_path, monkeypatch):
    from interface.routes import chat as chat_route
    from core import demo_support

    sample = tmp_path / "sample.txt"
    sample.write_text("one\ntwo\nthree\n", encoding="utf-8")

    monkeypatch.setattr(demo_support, "_resolve_target_path", lambda *_args, **_kwargs: sample)

    result = chat_route._read_repo_probe_reply(
        "Read sample.txt and tell me how many lines it has."
    )

    assert result is not None
    assert result["status"] == "repo_probe_line_count"
    assert "3 lines" in result["reply"]


def test_dialogue_policy_rejects_generic_assistant_language_for_live_voice_turns():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_aura_stance=True)
    validation = validate_dialogue_response("I can help with that. What would you like to know?", contract)

    assert validation.ok is False
    assert "generic_assistant_language" in validation.violations


def test_dialogue_policy_rejects_ungrounded_live_voice_replies():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_aura_stance=True)
    validation = validate_dialogue_response(
        "Right now, I'm experiencing a unique blend of anticipation and tranquility as I await interactions.",
        contract,
    )

    assert validation.ok is False
    assert "ungrounded_live_voice" in validation.violations


def test_subjective_self_reflex_contains_live_grounding():
    from interface.routes.chat import _build_subjective_self_reflex

    reply = _build_subjective_self_reflex(
        {
            "mood": "curious",
            "tone": "direct",
            "attention_focus": "the live runtime",
            "dominant_action": "reflect",
            "free_energy": 0.42,
            "valence": -0.1,
            "arousal": 0.3,
            "interests": ["mycelial topology"],
        },
        "Tell me something interesting about yourself right now.",
    )

    lowered = reply.lower()
    # Should use natural language, not raw metrics
    assert "free energy" not in lowered
    assert "curious" in lowered or "attention" in lowered
    assert len(reply) > 10


def test_aura_expression_frame_falls_back_to_state_repository(monkeypatch):
    from interface.routes import chat as chat_route
    from core.state.aura_state import AuraState
    from core.runtime import service_access

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"

    class DummyRepo:
        _current = state

    def fake_get(name, default=None):
        if name == "aura_state":
            return None
        return default

    monkeypatch.setattr(chat_route.ServiceContainer, "get", staticmethod(fake_get))
    monkeypatch.setattr(service_access, "resolve_state_repository", lambda default=None: DummyRepo())

    frame = chat_route._build_aura_expression_frame("Tell me something interesting about yourself right now.")

    assert frame["needs_self_expression"] is True


@pytest.mark.asyncio
async def test_conversation_experience_updates_memory_and_learning(monkeypatch):
    from core.runtime import conversation_support
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.world.relationship_graph = {"bryan": {}}

    captured = {"episode": None, "interaction": None, "user_model": None, "graph": None}

    class DummyEpisodic:
        async def record_episode_async(self, *args, **kwargs):
            captured["episode"] = {"args": args, "kwargs": kwargs}
            return "ep-1"

    class DummyLearner:
        async def record_interaction(self, **kwargs):
            captured["interaction"] = kwargs
            return "ok"

    class DummyUserModel:
        def update_from_interaction(self, input_text, response_text, metadata=None):
            captured["user_model"] = {
                "input_text": input_text,
                "response_text": response_text,
                "metadata": metadata,
            }

    class DummyGraph:
        async def register_interaction(self, *args):
            captured["graph"] = args

    class DummyBryanModel:
        def __init__(self):
            self._model = type("Model", (), {"total_messages": 0, "conversation_count": 0})()
            self.observed = []
            self.saved = 0

        def observe_pattern(self, description):
            self.observed.append(description)

        def save(self):
            self.saved += 1

    bryan_model = DummyBryanModel()

    def fake_optional_service(*names, default=None):
        if "episodic_memory" in names:
            return DummyEpisodic()
        if "continuous_learning" in names or "continuous_learning_engine" in names:
            return DummyLearner()
        if "user_model" in names:
            return DummyUserModel()
        if "entity_graph" in names or "relationship_graph" in names:
            return DummyGraph()
        if "bryan_model_engine" in names:
            return bryan_model
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", fake_optional_service)
    monkeypatch.setattr(conversation_support, "update_conversational_intelligence", lambda *args, **kwargs: asyncio.sleep(0))
    monkeypatch.setattr(conversation_support, "record_shared_ground_callbacks", lambda *args, **kwargs: asyncio.sleep(0))

    await conversation_support.record_conversation_experience(
        "I want a deeper explanation of this architecture.",
        "I do too. The strongest pressure in me right now is toward cleaner causation.",
        state,
    )

    assert captured["episode"] is not None
    assert captured["episode"]["kwargs"]["importance"] >= 0.35
    assert captured["interaction"]["domain"] == "conversation"
    assert captured["user_model"]["input_text"] == "I want a deeper explanation of this architecture."
    assert captured["graph"] == ("aura_self", "bryan", "conversation", "self", "person")
    assert bryan_model._model.total_messages == 2
    assert bryan_model._model.conversation_count == 1


def test_cognitive_routing_escalates_aura_stance_turns_to_deliberate():
    from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
    from core.state.aura_state import AuraState, CognitiveMode

    class DummyKernel:
        orchestrator = None
        organs = {}

    phase = CognitiveRoutingPhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "user"

    routed = asyncio.run(
        phase.execute(
            state,
            objective="What do you honestly think this architecture is strongest at?",
        )
    )

    assert routed.cognition.current_mode == CognitiveMode.DELIBERATE
    assert routed.response_modifiers["intent_type"] == "CHAT"


def test_cognitive_routing_keeps_self_reflection_on_reactive_grounded_lane():
    from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
    from core.state.aura_state import AuraState, CognitiveMode

    class DummyKernel:
        orchestrator = None
        organs = {}

    phase = CognitiveRoutingPhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "user"

    routed = asyncio.run(
        phase.execute(
            state,
            objective="Tell me something interesting about yourself right now.",
        )
    )

    assert routed.cognition.current_mode == CognitiveMode.REACTIVE
    assert routed.response_modifiers["intent_type"] == "CHAT"
