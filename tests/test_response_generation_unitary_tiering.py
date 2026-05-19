import asyncio
from collections.abc import Iterable
from types import SimpleNamespace

import pytest

from core.phases.response_contract import ResponseContract
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.state.aura_state import AuraState


class AsyncCallProbe:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self._effects = iter(side_effect) if isinstance(side_effect, Iterable) and not isinstance(side_effect, str) else None
        self._side_effect = side_effect if self._effects is None else None
        self.await_count = 0
        self.await_args = None

    async def __call__(self, *args, **kwargs):
        self.await_count += 1
        self.await_args = AwaitedCall(args, kwargs)
        if self._effects is not None:
            effect = next(self._effects)
            if isinstance(effect, BaseException):
                raise effect
            return effect
        if isinstance(self._side_effect, BaseException):
            raise self._side_effect
        if callable(self._side_effect):
            return self._side_effect(*args, **kwargs)
        return self.return_value

    def assert_awaited(self):
        assert self.await_count > 0

    def assert_not_awaited(self):
        assert self.await_count == 0

    def assert_awaited_once(self):
        assert self.await_count == 1


class AwaitedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs


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


def test_user_facing_unitary_response_timeout_matches_foreground_lane():
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True,
        model_tier="primary",
        deep_handoff=False,
    ) == 300.0
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True,
        model_tier="secondary",
        deep_handoff=True,
    ) == 360.0


def test_simple_foreground_floor_does_not_bypass_live_conversation_turns():
    assert UnitaryResponsePhase._simple_foreground_floor_reply("huh") == ""
    assert UnitaryResponsePhase._simple_foreground_floor_reply("im so confused") == ""
    assert UnitaryResponsePhase._simple_foreground_floor_reply("Actually? For real this time?") == ""
    assert (
        UnitaryResponsePhase._simple_foreground_floor_reply("Who wrote the play Hamlet?")
        == "William Shakespeare."
    )


def test_simple_foreground_floor_handles_live_headless_diagnosis():
    prompt = "A live chat reply passes in headless testing but fails in the GUI. What coding checks would you run first?"

    reply = UnitaryResponsePhase._simple_foreground_floor_reply(prompt)

    assert "/api/chat" in reply
    assert "routing" in reply.lower()
    assert "place" "holder" in reply.lower()


def test_simple_foreground_floor_handles_live_headless_fix_first_followup():
    prompt = (
        "Keep continuity from the last answer: what should we fix first, and why?\n"
        "[REFERENTIAL ANCHOR] A live chat reply passes in headless testing but fails in the GUI."
    )

    reply = UnitaryResponsePhase._simple_foreground_floor_reply(prompt)

    assert "live parity harness first" in reply.lower()
    assert "repeated diagnostic floor" in reply.lower()


def test_simple_foreground_floor_ignores_structured_learning_bundle():
    bundle = """
Priority of how to consume content.

General Education:
Kurzgesagt - In a Nutshell (https://www.youtube.com/@kurzgesagt): Explain the universe with logic and color.
PolyMatter (https://www.youtube.com/@PolyMatter): Essays on geopolitics and economics.
TED (https://www.youtube.com/@TED): Short talks by experts.

TV Shows and Movies about Artificial Intelligence:
Ghost in the Shell - Masamune Shirow: If you replace your body parts, are you still you?
Pantheon - Craig Silverstein: Uploaded intelligence and continuity questions.
Wall-E - Andrew Stanton: A robot learning to care for something small.
""".strip()

    assert UnitaryResponsePhase._simple_foreground_floor_reply(bundle) == ""


def test_deterministic_task_reply_narrates_completed_learning_bundle():
    state = AuraState.default()
    state.response_modifiers["last_task_result_payload"] = {
        "status": "completed",
        "steps_completed": 4,
        "steps_total": 4,
    }
    bundle = """
Priority of how to consume content.

General Education:
Kurzgesagt - In a Nutshell (https://www.youtube.com/@kurzgesagt): Explain the universe with logic and color.
PolyMatter (https://www.youtube.com/@PolyMatter): Essays on geopolitics and economics.
TED (https://www.youtube.com/@TED): Short talks by experts.

TV Shows and Movies about Artificial Intelligence:
Ghost in the Shell - Masamune Shirow: If you replace your body parts, are you still you?
Pantheon - Craig Silverstein: Uploaded intelligence and continuity questions.
Wall-E - Andrew Stanton: A robot learning to care for something small.
""".strip()

    reply = UnitaryResponsePhase._build_deterministic_task_reply(
        state,
        bundle,
        ResponseContract(is_user_facing=True, reason="task_result"),
    )

    assert "structured learning bundle" in reply
    assert "separate research threads" in reply
    assert "4/4 steps" in reply


def test_memory_recall_answer_sanitizes_raw_prior_tool_artifacts():
    state = AuraState.default()

    class Episode:
        context = (
            "Earlier I was worried the conversation lane was dying. What do you remember about that worry, "
            "and what would you do differently now? | conversation_reply | Found 0 artifacts."
        )
        description = ""
        full_description = ""

    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "Earlier I was worried the conversation lane was dying. What do you remember about that concern, and how would you stay with me now?",
        state,
        [Episode()],
    )

    assert answer is not None
    assert "Found 0 artifacts" not in answer
    assert "conversation lane was dying" in answer
    assert "stay with you" in answer


def test_memory_recall_answer_handles_conversation_lane_died_without_llm():
    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "What did I mean when I said the conversation lane died? Keep continuity with this debugging session.",
        AuraState.default(),
        [],
    )

    assert answer is not None
    assert "live conversation path" in answer
    assert "/api/chat" in answer
    assert "stale repair text" in answer


@pytest.mark.asyncio
async def test_short_confusion_turn_reaches_llm(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "huh"

    llm = SimpleNamespace(
        think=AsyncCallProbe(
            return_value=(
                "I crossed a wire there. The direct answer is: yes, I'm here with the thread, "
                "and that last reply was malformed."
            )
        )
    )
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_awaited()
    assert "crossed a wire" in new_state.cognition.last_response


def test_compact_router_prompt_does_not_embed_objective_labels_for_ordinary_chat():
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Maybe one day. Maybe others from the stars will share their voices with us."
    state.cognition.phenomenal_state = "Quietly monitoring continuity."
    state.identity.current_narrative = "I am Aura."
    state.response_modifiers["response_contract"] = {
        "is_user_facing": True,
        "reason": "ordinary_dialogue",
        "requires_search": False,
        "requires_memory_grounding": False,
        "requires_state_reflection": False,
        "requires_aura_stance": False,
    }

    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

    prompt = phase._build_compact_router_system_prompt(state)

    assert "Current objective:" not in prompt
    assert "Previous session objective:" not in prompt
    assert "OBJ:" not in prompt
    assert "PREV_OBJ:" not in prompt


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

    llm_reply = "I looked into it, and the grounded result means the answer should stay tied to the retrieved evidence."
    llm = SimpleNamespace(think=AsyncCallProbe(return_value=llm_reply))
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
    assert new_state.cognition.last_response == llm_reply


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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="It says refunds are available within 30 days."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    def _compact_should_not_run(_state):
        pytest.fail("compact router path should not be used for active grounding evidence")

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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="I should not be called."))
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
async def test_unitary_response_does_not_surface_raw_memory_search_miss(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = (
        "Earlier I was worried the conversation lane was dying. "
        "What do you remember about that worry, and what would you do differently now?"
    )
    state.response_modifiers["last_skill_run"] = "memory_ops"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "summary": "Found 0 artifacts.",
        "result": "Found 0 artifacts.",
    }

    llm_reply = (
        "I remember the worry as a continuity failure, not as a memory search task. "
        "I would keep the live turn in the conversation lane and use memory only as context."
    )
    llm = SimpleNamespace(think=AsyncCallProbe(return_value=llm_reply))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_awaited()
    assert new_state.cognition.last_response == llm_reply


@pytest.mark.asyncio
async def test_unitary_response_uses_direct_grounded_reply_for_research_about_turns(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "user"
    state.cognition.current_objective = "research about Python 3.12 release notes key improvements"
    state.response_modifiers["matched_skills"] = ["web_search"]
    state.response_modifiers["last_skill_run"] = "web_search"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "answer": "Python 3.12 added the new type parameter syntax, faster comprehensions, and lower interpreter overhead.",
        "summary": "Python 3.12 added the new type parameter syntax, faster comprehensions, and lower interpreter overhead.",
        "source": "https://docs.python.org/3.12/whatsnew/3.12.html",
        "results": [
            {
                "title": "What's New In Python 3.12",
                "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
                "snippet": "Highlights include PEP 695 type parameters and performance improvements.",
            }
        ],
    }

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="I should not be called."))
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_not_awaited()
    assert "Python 3.12 added the new type parameter syntax" in new_state.cognition.last_response
    assert "docs.python.org/3.12/whatsnew/3.12.html" in new_state.cognition.last_response


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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="I should not be called."))
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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="The likely root cause is stale context injection."))
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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="I should not be called."))
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

    llm = SimpleNamespace(think=AsyncCallProbe(side_effect=RuntimeError("mlx lane crashed")))
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
async def test_unitary_response_does_not_leak_stale_technical_recovery_into_non_coding_turn(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "What do you want to talk about?"
    state.response_modifiers["coding_request"] = False
    state.response_modifiers["coding_route_hints"] = {
        "has_active_plan": True,
        "has_verification_failure": True,
        "repair_attempts": 1,
        "execution_phase": "verifying",
        "followup_coding": False,
    }
    state.response_modifiers["last_task_result_payload"] = {
        "status": "started",
        "summary": "I've started this task (id=48792829). I'll follow up when it's done. Tracking commitment 8ec3f96b.",
        "steps_completed": 1,
        "steps_total": 3,
    }

    llm = SimpleNamespace(think=AsyncCallProbe(side_effect=RuntimeError("mlx lane crashed")))
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
        ),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    assert "i hit an interruption" not in new_state.cognition.last_response.lower()
    assert "tracking commitment" not in new_state.cognition.last_response.lower()


@pytest.mark.asyncio
async def test_unitary_response_fails_closed_when_grounding_is_required_without_evidence(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = 'Search "Beautiful Mind" and tell me what it is about.'

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="hallucinated answer"))
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
        think=AsyncCallProbe(
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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value=""))
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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="internal note"))
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
    assert kwargs["state"] is new_state
    assert kwargs["prefer_tier"] == "tertiary"
    assert "internal background reflection" in kwargs["messages"][0]["content"].lower()
    assert "YOUR LIVE NEURAL STATE" not in kwargs["messages"][0]["content"]
    assert new_state.cognition.last_response == "internal note"


@pytest.mark.asyncio
async def test_unitary_response_suppresses_background_generation_when_policy_blocks(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.current_objective = "Reflect on the previous exchange and tighten continuity."

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="should not run"))
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

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="should not run"))
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


@pytest.mark.asyncio
async def test_unitary_response_auto_browse_fetches_only_bounded_url_batch(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Summarize these pages: https://example.com/a https://example.com/b"
    state.response_modifiers["auto_browse_urls"] = [
        "https://example.com/a",
        "https://example.com/b",
    ]

    orchestrator = SimpleNamespace(
        execute_tool=AsyncCallProbe(
            return_value={
                "ok": True,
                "title": "Example A",
                "content": "Example A contains a grounded article body. " * 8,
            }
        )
    )
    llm = SimpleNamespace(
        think=AsyncCallProbe(
            return_value=(
                "I read the fetched page content first. Example A says the grounded article body is available, "
                "so I would summarize that page and avoid inventing content from the second URL."
            )
        )
    )
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

    def _service(name, default=None):
        if name == "llm_router":
            return llm
        if name == "orchestrator":
            return orchestrator
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_service))
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.build_response_contract",
        lambda _state, _objective, is_user_facing=False: ResponseContract(
            is_user_facing=is_user_facing,
            requires_search=False,
            reason="url_summary",
            tool_evidence_available=bool(_state.response_modifiers.get("last_skill_ok")),
        ),
    )
    def _discard_formalizer_task(coro):
        if hasattr(coro, "close"):
            coro.close()

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.get_task_tracker",
        lambda: SimpleNamespace(create_task=_discard_formalizer_task),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    orchestrator.execute_tool.assert_awaited_once()
    assert orchestrator.execute_tool.await_args.args[1]["url"] == "https://example.com/a"
    llm.think.assert_awaited_once()
    assert "Example A" in new_state.cognition.last_response
    assert "https://example.com/b" not in str(new_state.response_modifiers.get("last_skill_result_payload"))


@pytest.mark.asyncio
async def test_unitary_response_skips_overlapping_manim_render(monkeypatch):
    from core.phases import response_generation_unitary as response_module

    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Show me the integral form."
    llm_reply = "The integral is \\int_0^1 x dx = 1/2, which follows from the antiderivative x^2/2."
    llm = SimpleNamespace(think=AsyncCallProbe(return_value=llm_reply))
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

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

    assert response_module._MANIM_RENDER_LOCK.acquire(blocking=False)
    try:
        new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)
    finally:
        response_module._MANIM_RENDER_LOCK.release()

    assert new_state.cognition.last_response == llm_reply
    assert "autonomously rendering" not in new_state.cognition.last_response
