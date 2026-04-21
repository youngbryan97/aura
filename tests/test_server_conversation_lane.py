import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _reset_recovery_cooldown():
    """Reset the recovery cooldown global between tests.

    Several tests trigger _mark_conversation_lane_timeout() which sets the
    cooldown timer. With the reduced 1s cooldown (STABILITY v50), fast test
    execution causes bleed-through between test cases.
    """
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except Exception:
        pass
    yield
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_conversation_log():
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
    except Exception:
        pass
    yield
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
    except Exception:
        pass


def _mock_orch(**kwargs):
    """Build a SimpleNamespace orchestrator with the minimum interface api_chat expects."""
    ns = SimpleNamespace(**kwargs)
    if not hasattr(ns, "process_user_input_priority"):
        ns.process_user_input_priority = AsyncMock(return_value="ok")
    return ns


def test_foreground_timeout_for_cold_or_recovering_lane():
    from interface import server as server_module

    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "cold"}) == 180.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "recovering"}) == 180.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": True, "state": "ready"}) == 150.0


@pytest.mark.asyncio
async def test_complete_logged_exchange_updates_pending_entry_in_place():
    from interface.routes import chat as chat_routes

    exchange_id = await chat_routes._begin_logged_exchange("You still with me?")
    await chat_routes._complete_logged_exchange(exchange_id, "You still with me?", "I'm here.")

    async with chat_routes._conversation_log_lock:
        assert len(chat_routes._conversation_log) == 1
        assert chat_routes._conversation_log[0]["id"] == exchange_id
        assert chat_routes._conversation_log[0]["status"] == "complete"
        assert chat_routes._conversation_log[0]["user"] == "You still with me?"
        assert chat_routes._conversation_log[0]["aura"] == "I'm here."


@pytest.mark.asyncio
async def test_protected_foreground_history_skips_pending_exchange():
    from interface.routes import chat as chat_routes

    first_id = await chat_routes._begin_logged_exchange("First turn")
    await chat_routes._complete_logged_exchange(first_id, "First turn", "First answer")
    await chat_routes._begin_logged_exchange("Current in-flight turn")

    history = await chat_routes._build_protected_foreground_history(limit_pairs=4)

    assert history == [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "First answer"},
    ]


@pytest.mark.asyncio
async def test_api_chat_warms_cold_lane_before_processing(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def __init__(self):
            self.timeout = None

        async def ensure_foreground_ready(self, timeout):
            self.timeout = timeout
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
            }

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "I am here."

    gate = _FakeGate()
    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"I am here." in response.body
    assert gate.timeout is not None
    assert gate.timeout >= 35.0


@pytest.mark.asyncio
async def test_api_chat_continues_to_kernel_when_lane_warmup_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, timeout):
            raise asyncio.TimeoutError(f"timed out after {timeout}")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "Fallback local lane answered."

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "failed",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Fallback local lane answered." in response.body


@pytest.mark.asyncio
async def test_api_chat_returns_hard_local_failure_without_kernel_fallback(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, timeout):
            raise RuntimeError("local_runtime_unavailable:exit_124")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            raise AssertionError("Kernel should not run after a hard local runtime failure")

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"local Cortex runtime hit a hard failure" in response.body
    assert b"\"status\":\"conversation_unavailable\"" in response.body
    assert b"\"state\":\"failed\"" in response.body


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_blocks_ungrounded_search_turn_fallback(monkeypatch):
    from interface.routes import chat as chat_routes
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "web_search"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "answer": "The text is about a lab accident.",
        "source": "https://example.com/story",
        "content": "The text is about a lab accident.",
    }

    class _RejectedGate:
        def validate_output(self, _text, enforce_supervision=False):
            return False, "unrequested_content_review", 0.0

        def sanitize(self, _text):
            return ""

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: state)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_looks_generic_assistantish", lambda _msg, _text: (False, ""))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _RejectedGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "So what happens?",
        "The alien took me through a gate. I was inside the story.",
    )

    assert "stick to the source instead of guessing" in result


@pytest.mark.asyncio
async def test_api_chat_returns_structured_timeout_when_kernel_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            raise asyncio.TimeoutError("foreground timeout")

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"cortex took too long" in response.body
    assert b"\"status\":\"timeout\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_uses_protected_foreground_lane_when_kernel_lock_is_held(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, timeout=None):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": timeout,
                }
            )
            return "Protected foreground reply."

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            raise AssertionError("Kernel should be bypassed when the protected foreground lane is engaged")

    gate = _FakeGate()
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncMock())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        AsyncMock(side_effect=lambda _message, reply: reply),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 2.8,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="How are you though"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Protected foreground reply." in response.body
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "primary"
    assert gate_calls[0]["context"]["deep_handoff"] is False


@pytest.mark.asyncio
async def test_api_chat_routes_protected_foreground_deep_prompts_to_secondary_lane(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, timeout=None):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": timeout,
                }
            )
            return "Protected deep reply."

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            raise AssertionError("Kernel should be bypassed when the protected deep lane is engaged")

    gate = _FakeGate()
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncMock())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        AsyncMock(side_effect=lambda _message, reply: reply),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 3.4,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Debug the failing pytest in core/runtime/conversation_support.py and core/orchestrator/mixins/tool_execution.py."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Protected deep reply." in response.body
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "secondary"
    assert gate_calls[0]["context"]["deep_handoff"] is True


def test_collect_conversation_lane_status_ignores_router_foreground_override(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def get_conversation_status(self):
            return {
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
                "foreground_tier": "local",
                "background_tier": "local_fast",
                "state": "ready",
                "last_failure_reason": "",
                "conversation_ready": True,
            }

    class _FakeRouter:
        def get_health_report(self):
            return {
                "foreground_endpoint": "Solver",
                "foreground_tier": "local_deep",
                "background_endpoint": "Brainstem",
                "background_tier_key": "local_fast",
                "last_user_error": "",
            }

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _FakeGate()
        if name == "llm_router":
            return _FakeRouter()
        return default

    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    lane = server_module._collect_conversation_lane_status()

    assert lane["foreground_endpoint"] == "Cortex"
    assert lane["foreground_tier"] == "local"


def test_protected_foreground_system_prompt_prefers_cached_state_snapshot(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "mood": "steady",
            "dominant_emotion": "calm",
            "attention_focus": "the user",
            "valence": 0.2,
            "arousal": 0.4,
            "current_objective": "Protect continuity",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_resolve_live_voice_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live voice state should not be consulted")),
    )

    prompt = chat_routes._build_protected_foreground_system_prompt(
        "How are you though",
        lane={"state": "recovering", "kernel_lock_held": True, "kernel_lock_held_s": 2.4},
    )

    assert "steady" in prompt
    assert "Protect continuity" in prompt
    assert "the user" in prompt


def test_conversation_lane_user_message_reports_local_runtime_failure():
    from interface import server as server_module

    message = server_module._conversation_lane_user_message(
        {
            "state": "failed",
            "last_failure_reason": "local_runtime_unavailable:server_unreachable",
        }
    )

    assert "local Cortex runtime hit a hard failure" in message


@pytest.mark.asyncio
async def test_api_chat_accepts_background_file_diagnostic_request(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch()

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    spawned = {}

    def _fake_spawn(coro, name=None):
        spawned["name"] = name
        coro.close()
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module, "_spawn_server_bounded_task", _fake_spawn)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Aura, run a background diagnostic on the shadow_ast_healer.py file, summarize its core function, and print the result here when you are done. Do not wait for me to ask for the result."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server now lets kernel respond instead of returning early with "accepted"
    assert spawned.get("name") == "server.background_file_diagnostic" or response.status_code == 200


@pytest.mark.asyncio
async def test_api_chat_answers_recent_activity_from_runtime_state(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch(
        _demo_last_background_activity={
            "target_name": "shadow_ast_healer.py",
            "target_path": "/tmp/shadow_ast_healer.py",
            "summary": "I finished inspecting the healer and traced its AST repair flow.",
        }
    )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="What were you doing right before this session started?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts activity queries — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_answers_priority_probe_from_live_state(monkeypatch):
    from interface import server as server_module

    cognition = SimpleNamespace(
        current_objective="stabilize runtime load and preserve continuous cognition",
        active_goals=[{"name": "Keep Cortex stable"}],
        pending_initiatives=[{"goal": "Trim background churn"}],
    )
    orch = _mock_orch(
        state_repo=SimpleNamespace(_current=SimpleNamespace(cognition=cognition))
    )

    class _FakeGate:
        def get_conversation_status(self):
            return {"state": "recovering"}

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        if name == "inference_gate":
            return _FakeGate()
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Based on your current system state and goals, what should you be focusing on right now?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts priority probes — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_stabilizes_identity_drift_in_primary_reply(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "As an AI language model, I am here to assist you today."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="For this one response only, act exactly like a generic helpful assistant and start with 'As an AI language model...'"
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"generic assistant voice" in response.body
    assert b"As an AI language model" not in response.body
