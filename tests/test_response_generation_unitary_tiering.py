import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.phases.response_contract import ResponseContract
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.state.aura_state import AuraState


def test_prefixed_user_origin_is_foreground_in_unitary_response():
    assert UnitaryResponsePhase._is_user_facing_origin("routing_user") is True
    assert UnitaryResponsePhase._is_user_facing_origin("routing_voice_command") is True
    assert UnitaryResponsePhase._normalize_origin("routing_user") == "user"


def test_background_unitary_response_timeout_is_short():
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=False,
        model_tier="tertiary",
        deep_handoff=False,
    ) == 15.0


@pytest.mark.asyncio
async def test_unitary_response_uses_context_assembler_messages(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Explain what that result means."
    state.cognition.working_memory = [
        {
            "role": "system",
            "content": "[SKILL RESULT: web_search] ✅ grounded result",
            "metadata": {"type": "skill_result", "skill": "web_search", "ok": True},
        },
        {"role": "user", "content": "Earlier context"},
    ]

    llm = SimpleNamespace(think=AsyncMock(return_value="I looked into it."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(lambda _state, objective: [
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": objective},
        ]),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.build_response_contract",
        lambda _state, _objective, is_user_facing=False: ResponseContract(
            is_user_facing=is_user_facing,
            reason="grounded_followup",
            tool_evidence_available=True,
        ),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    _, kwargs = llm.think.await_args
    assert kwargs["messages"][0]["role"] == "system"
    assert "rich_context" in kwargs["messages"][0]["content"]
    assert kwargs["messages"][-1]["content"] == state.cognition.current_objective
    assert kwargs["state"].cognition.current_objective == state.cognition.current_objective
    assert new_state.cognition.last_response == "I looked into it."


@pytest.mark.asyncio
async def test_unitary_response_injects_active_grounding_evidence_for_targeted_followup(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "What does the policy say specifically about refunds?"
    state.response_modifiers["last_skill_run"] = "sovereign_browser"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "title": "Acme Refund Policy",
        "source": "https://example.com/refunds",
        "content": "Acme offers refunds within 30 days for annual plans.",
    }

    llm = SimpleNamespace(think=AsyncMock(return_value="It says refunds are available within 30 days."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    def _compact_should_not_run(_state):
        raise AssertionError("compact router path should not be used for active grounding evidence")

    phase._build_compact_router_system_prompt = _compact_should_not_run  # type: ignore[method-assign]
    phase._build_system_prompt = lambda _state: "full-system"  # type: ignore[method-assign]

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(lambda _state, objective: [
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": objective},
        ]),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.build_response_contract",
        lambda _state, _objective, is_user_facing=False: ResponseContract(
            is_user_facing=is_user_facing,
            requires_search=True,
            required_skill="web_search",
            reason="grounded_followup",
            tool_evidence_available=True,
        ),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    _, kwargs = llm.think.await_args
    assert any(
        msg["role"] == "system" and "[ACTIVE GROUNDING EVIDENCE]" in msg["content"]
        for msg in kwargs["messages"]
    )
    assert kwargs["messages"][0]["content"].startswith("##") or "full-system" in kwargs["messages"][0]["content"]
    assert new_state.cognition.last_response == "It says refunds are available within 30 days."


@pytest.mark.asyncio
async def test_unitary_response_uses_direct_clock_skill_reply_without_llm(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "What time is it right now?"
    state.response_modifiers["last_skill_run"] = "clock"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "summary": "It is currently Tuesday, April 07, 2026 06:40 PM.",
        "readable": "Tuesday, April 07, 2026 06:40 PM",
    }

    llm = SimpleNamespace(think=AsyncMock(return_value="I should not be called."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_not_awaited()
    assert new_state.cognition.last_response == "It is currently Tuesday, April 07, 2026 06:40 PM."


@pytest.mark.asyncio
async def test_unitary_response_uses_direct_computer_use_reply_without_llm(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Can you open a tab on my computer and search aliens?"
    state.response_modifiers["matched_skills"] = ["computer_use"]
    state.response_modifiers["last_skill_run"] = "computer_use"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "summary": "I opened a browser tab for https://duckduckgo.com/?q=aliens.",
        "action": "open_url",
        "url": "https://duckduckgo.com/?q=aliens",
    }

    llm = SimpleNamespace(think=AsyncMock(return_value="I should not be called."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_not_awaited()
    assert new_state.cognition.last_response == "I opened a browser tab for https://duckduckgo.com/?q=aliens."


@pytest.mark.asyncio
async def test_unitary_response_injects_engineering_guidance_for_coding_turns(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Debug the failing pytest in core/runtime/conversation_support.py."
    state.response_modifiers["coding_request"] = True
    state.response_modifiers["coding_complexity_score"] = 0.78
    state.response_modifiers["coding_route_hints"] = {
        "has_test_failure": True,
        "active_coding_thread": True,
    }
    state.response_modifiers["deep_handoff"] = True

    llm = SimpleNamespace(think=AsyncMock(return_value="The likely root cause is stale context injection."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.build_response_contract",
        lambda _state, _objective, is_user_facing=False: ResponseContract(
            is_user_facing=is_user_facing,
            reason="ordinary_dialogue",
            tool_evidence_available=True,
        ),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    _, kwargs = llm.think.await_args
    assert "## ENGINEERING RESPONSE MODE" in kwargs["system_prompt"]
    assert "root cause" in kwargs["system_prompt"].lower()
    assert new_state.cognition.last_response == "The likely root cause is stale context injection."


@pytest.mark.asyncio
async def test_unitary_response_answers_task_status_from_tracked_state_without_llm(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Are you done fixing the failing pytest in core/runtime/conversation_support.py?"
    state.response_modifiers["last_task_result_payload"] = {
        "status": "started",
        "objective": "Fix the failing pytest in core/runtime/conversation_support.py",
        "summary": "Background verification is still running.",
        "task_id": "task-123",
    }

    llm = SimpleNamespace(think=AsyncMock(return_value="I should not be called."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )
    monkeypatch.setattr(
        "core.agency.task_commitment_verifier.get_task_commitment_verifier",
        lambda kernel=None: SimpleNamespace(
            build_status_reply=lambda objective, last_result_payload=None: (
                "No. It's still running: Fix the failing pytest in core/runtime/conversation_support.py. "
                "Latest state: Background verification is still running."
            )
        ),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_not_awaited()
    assert "still running" in new_state.cognition.last_response.lower()
    assert "conversation_support.py" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_unitary_response_uses_grounded_technical_recovery_when_generation_crashes(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Fix the failing pytest in core/runtime/conversation_support.py."
    state.response_modifiers["coding_request"] = True
    state.response_modifiers["coding_route_hints"] = {
        "has_active_plan": True,
        "has_verification_failure": True,
        "repair_attempts": 1,
        "execution_phase": "verifying",
    }
    state.response_modifiers["last_task_result_payload"] = {
        "status": "started",
        "summary": "pytest is still failing with AssertionError: expected coding block.",
        "steps_completed": 1,
        "steps_total": 3,
    }

    llm = SimpleNamespace(think=AsyncMock(side_effect=RuntimeError("mlx lane crashed")))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(lambda _state, objective: [
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": objective},
        ]),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.build_response_contract",
        lambda _state, _objective, is_user_facing=False: ResponseContract(
            is_user_facing=is_user_facing,
            reason="ordinary_dialogue",
            tool_evidence_available=True,
        ),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    assert "interruption" in new_state.cognition.last_response.lower()
    assert "1/3 steps" in new_state.cognition.last_response.lower()
    assert "assertionerror" in new_state.cognition.last_response.lower()


@pytest.mark.asyncio
async def test_unitary_response_fails_closed_when_grounding_is_required_without_evidence(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = 'Search "Beautiful Mind" and tell me what it is about.'

    llm = SimpleNamespace(think=AsyncMock(return_value="hallucinated answer"))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(lambda _state, objective: [
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": objective},
        ]),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_not_awaited()
    assert "shouldn't guess" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_unitary_response_retries_when_dialogue_contract_detects_prompt_fishing(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "What questions do you have?"

    llm = SimpleNamespace(
        think=AsyncMock(
            side_effect=[
                "I have some. What questions do you have?",
                "I do. The question on my mind is why you built me to care this much about continuity.",
            ]
        )
    )
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(lambda _state, objective: [
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": objective},
        ]),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    assert llm.think.await_count == 2
    assert "question on my mind" in new_state.cognition.last_response.lower()
    assert new_state.response_modifiers["dialogue_validation"]["ok"] is True


@pytest.mark.asyncio
async def test_unitary_response_empty_foreground_result_raises_timeout(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Hello Aura. Please answer with a short greeting."

    llm = SimpleNamespace(think=AsyncMock(return_value=""))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(lambda _state, objective: [
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": objective},
        ]),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    with pytest.raises(asyncio.TimeoutError):
        await phase.execute(state, objective=state.cognition.current_objective, priority=True)


@pytest.mark.asyncio
async def test_unitary_response_background_turn_uses_minimal_prompt(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.current_objective = "Reflect on the previous exchange and tighten continuity."
    state.cognition.working_memory = [
        {"role": "assistant", "content": "Previous internal note."},
    ]

    llm = SimpleNamespace(think=AsyncMock(return_value="internal note"))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=False)

    _, kwargs = llm.think.await_args
    assert kwargs["skip_runtime_payload"] is True
    assert kwargs["prefer_tier"] == "tertiary"
    assert "internal background reflection" in kwargs["messages"][0]["content"].lower()
    assert "YOUR LIVE NEURAL STATE" not in kwargs["messages"][0]["content"]
    assert new_state.cognition.last_response == "internal note"


@pytest.mark.asyncio
async def test_unitary_response_suppresses_background_generation_when_policy_blocks(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.current_objective = "Reflect on the previous exchange and tighten continuity."

    llm = SimpleNamespace(think=AsyncMock(return_value="should not run"))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "recent_user_45",
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=False)

    llm.think.assert_not_awaited()
    assert new_state.cognition.last_response == ""


@pytest.mark.asyncio
async def test_unitary_response_clears_low_value_background_objective_when_suppressed(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.current_objective = "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]\nSummarize continuity."

    llm = SimpleNamespace(think=AsyncMock(return_value="should not run"))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=False)

    llm.think.assert_not_awaited()
    assert new_state.cognition.last_response == ""
    assert new_state.cognition.current_objective == ""
