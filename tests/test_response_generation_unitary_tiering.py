import asyncio
from collections.abc import Iterable
from types import SimpleNamespace

import pytest

from core.phases.response_contract import ResponseContract
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.state.aura_state import AuraState
from tests.support.amplifier_doubles import amplified_answer


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


def test_strict_proof_task_reply_uses_fresh_run_code_stdout():
    objective = (
        "Analyze the exact reference status and compute the printed output.\n"
        "```python\n"
        "x = [1, 2, 3, 4, 5]\n"
        "y = x[1:4]\n"
        "y[0] = 99\n"
        "print(x[1], y[0])\n"
        "```\n"
        "Output your final answer inside <answer>...</answer> tags."
    )
    state = AuraState.default()
    state.cognition.current_origin = "proof"
    state.response_modifiers.update(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": UnitaryResponsePhase._objective_fingerprint(
                objective
            ),
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "2 99\n",
                "exit_code": 0,
            },
        }
    )

    reply = UnitaryResponsePhase._build_structured_proof_task_reply(
        state,
        objective,
        SimpleNamespace(requires_search=False),
    )

    assert reply == "<answer>2 99</answer>"


def test_strict_proof_task_reply_rejects_stale_run_code_stdout():
    objective = (
        "Analyze the behavior of this function and determine the length result.\n"
        "```python\n"
        "def f(a, b=[]):\n"
        "    b.append(a)\n"
        "    return b\n"
        "print(len(f(1)), len(f(2)), len(f(3)))\n"
        "```\n"
        "Output your final answer inside <answer>...</answer> tags."
    )
    state = AuraState.default()
    state.cognition.current_origin = "proof"
    state.response_modifiers.update(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": UnitaryResponsePhase._objective_fingerprint(
                "different objective"
            ),
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "2 99\n",
                "exit_code": 0,
            },
        }
    )

    reply = UnitaryResponsePhase._build_structured_proof_task_reply(
        state,
        objective,
        SimpleNamespace(requires_search=False),
    )

    assert reply == ""


def test_strict_proof_run_code_detector_handles_result_wording():
    from core.kernel.upgrades_10x import GodModeToolPhase

    objective = (
        "Analyze the behavior of the following simplified registry tracking function:\n"
        "```python\n"
        "def f(a, b=[]):\n"
        "    b.append(a)\n"
        "    return b\n"
        "print(len(f(1)), len(f(2)), len(f(3)))\n"
        "```\n"
        "Determine the length of the list returned by each invocation. "
        "Output your final answer inside <answer>...</answer> tags."
    )

    assert GodModeToolPhase._looks_like_direct_run_code_request(objective) is True


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
    ) == 180.0
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True,
        model_tier="secondary",
        deep_handoff=True,
    ) == 210.0


def test_simple_foreground_floor_does_not_bypass_live_conversation_turns():
    assert UnitaryResponsePhase._simple_foreground_floor_reply("huh") == ""
    assert UnitaryResponsePhase._simple_foreground_floor_reply("im so confused") == ""
    assert UnitaryResponsePhase._simple_foreground_floor_reply("Actually? For real this time?") == ""
    # 455577dde deleted the stored answer bank ("William Shakespeare." for
    # Hamlet, the capital of France, and so on). A knowledge question has no
    # deterministic floor — answering it from a branch measures the branch,
    # not the model. Real computation still has one.
    assert UnitaryResponsePhase._simple_foreground_floor_reply("Who wrote the play Hamlet?") == ""
    assert UnitaryResponsePhase._simple_foreground_floor_reply("what is 17 + 8?") == "25"


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


def test_substantive_truncated_foreground_reply_is_completed_without_new_model_call():
    from core.conversation.response_reliability import assess_user_facing_reply

    prompt = "Explain how you would keep a live desktop conversation coherent under load."
    draft = (
        "I would keep the foreground lane bounded by preserving the current user objective, "
        "checking the response contract before surfacing text, holding tool work behind governance, "
        "and treating memory or screen sensors as supporting evidence rather than blockers with"
    )

    initial = assess_user_facing_reply(prompt, draft)
    assert set(initial.reasons) == {"truncated_tail"}

    repaired = UnitaryResponsePhase._complete_substantive_truncated_foreground_reply(draft)
    repaired_quality = assess_user_facing_reply(prompt, repaired)

    assert repaired
    assert repaired.endswith(".")
    assert " with." not in repaired.lower()
    assert not repaired_quality.retryable


def test_substantive_truncated_foreground_reply_trims_partial_stem():
    from core.conversation.response_reliability import assess_user_facing_reply

    prompt = "Describe your private cognitive architecture in one grounded paragraph."
    draft = (
        "I model myself as a foreground attention loop over memory, affect, planning, and governed "
        "action gateways. That model changes my answer by making me check the live request, keep "
        "the plan bounded, and verify external claims before acting through cogn"
    )

    repaired = UnitaryResponsePhase._complete_substantive_truncated_foreground_reply(draft)

    assert repaired
    assert "cogn." not in repaired.lower()
    assert not assess_user_facing_reply(prompt, repaired).retryable


def test_user_facing_shape_removes_orphan_leading_period():
    shaped = UnitaryResponsePhase._shape_user_facing_response(
        ". Glass arithmetic keeps a running clarity score and connects each operation to a visible example.",
        "Stay with glass arithmetic.",
    )

    assert shaped.startswith("Glass arithmetic")


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


def test_memory_recall_answer_uses_recent_user_working_memory_without_llm():
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "Remember this phrase: blue lantern at 3:14."},
        {"role": "assistant", "content": "I have it anchored for this conversation."},
        {"role": "user", "content": "What did I tell you to remember?"},
    ]

    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "What did I tell you to remember?",
        state,
        [],
    )

    assert answer is not None
    assert "blue lantern at 3:14" in answer
    assert "What did I tell you to remember" not in answer
    assert answer.startswith("I remember you saying")


def test_memory_recall_answer_uses_recent_assistant_working_memory_when_asked():
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "What tools can you use externally?"},
        {
            "role": "assistant",
            "content": "I can use governed desktop, browser, file, search, and terminal tools when authorized.",
        },
        {"role": "user", "content": "What did you say earlier about tools?"},
    ]

    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "What did you say earlier about tools?",
        state,
        [],
    )

    assert answer is not None
    assert answer.startswith("I remember saying")
    assert "governed desktop" in answer
    assert "What did you say earlier" not in answer


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
async def test_desktop_descriptive_context_bounds_foreground_generation(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_objective = "What tools can you use externally?"
    state.response_modifiers["model_tier"] = "secondary"
    state.response_modifiers["deep_handoff"] = True

    llm = SimpleNamespace(
        think=AsyncCallProbe(return_value="I can describe governed tool surfaces without loading the solver lane.")
    )
    kernel = SimpleNamespace(organs={})
    phase = UnitaryResponsePhase(kernel)

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(lambda _state, objective: [
            {"role": "system", "content": "desktop_context"},
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

    new_state = await phase.execute(
        state,
        objective=state.cognition.current_objective,
        priority=True,
        context={
            "desktop_descriptive_turn": True,
            "capability_inventory_contract": True,
            "prefer_tier": "primary",
            "allow_deep_handoff": False,
            "max_tokens": 768,
            "skip_runtime_payload": True,
            "disable_prompt_cache": True,
            "clear_prompt_cache": True,
        },
    )

    _, kwargs = llm.think.await_args
    assert kwargs["prefer_tier"] == "primary"
    assert kwargs["deep_handoff"] is False
    assert kwargs["max_tokens"] == 768
    assert kwargs["num_predict"] == 768
    assert kwargs["user_surface_completion_floor"] == 768
    assert kwargs["skip_runtime_payload"] is True
    assert kwargs["disable_prompt_cache"] is True
    assert kwargs["clear_prompt_cache"] is True
    assert new_state.response_modifiers["model_tier"] == "primary"
    assert new_state.response_modifiers["deep_handoff"] is False


@pytest.mark.asyncio
async def test_desktop_cognitive_contract_forwards_flags_and_does_not_model_retry(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_objective = "Tell me what you think about distributed systems."

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="How can I assist?"))
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(
            lambda _state, objective: [
                {"role": "system", "content": "desktop_context"},
                {"role": "user", "content": objective},
            ]
        ),
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

    new_state = await phase.execute(
        state,
        objective=state.cognition.current_objective,
        priority=True,
        context={
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
        },
    )

    llm.think.assert_awaited_once()
    _, call_kwargs = llm.think.await_args
    assert call_kwargs["cognitive_engine_required"] is True
    assert call_kwargs["desktop_cognitive_engine_required"] is True
    assert call_kwargs["protected_foreground_lane"] is True
    assert "won't fabricate" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_unitary_healthy_chat_commits_full_latent_answer_before_direct_decode(
    monkeypatch,
):
    from core.brain.foreground_latent_runtime import ForegroundLatentOutcome

    objective = (
        "Compare Raft and PBFT, then explain why changing from a crash-only "
        "failure model to a Byzantine failure model changes the recommendation."
    )
    state = AuraState()
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_objective = objective
    llm = SimpleNamespace(
        think=AsyncCallProbe(return_value="The ordinary decoder must not run.")
    )
    orchestrator = SimpleNamespace(name="test-orchestrator")
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))
    captured = {}
    amplification = {}

    async def _run_latent(**kwargs):
        captured.update(kwargs)
        return ForegroundLatentOutcome(
            text=(
                "Raft is appropriate under the requested crash-only model, while PBFT "
                "addresses the requested Byzantine failure model. The recommendation "
                "changes because crash-only assumptions no longer protect the committed log."
            ),
            trace={
                "latent_cortex_selected": True,
                "latent_cortex_attempted": True,
                "latent_cortex_succeeded": True,
                "latent_cortex_fallback_used": False,
                "latent_cortex_failure_reason": "",
                "latent_cortex_identity_bound": True,
                "latent_cortex_receipt": {
                    "episode_id": "unitary-episode",
                    "last_stage": "complete",
                },
                "latent_cortex_progress": {"stage": "complete"},
                "response_path": "cognitive_engine_latent_cortex",
            },
            fallback_allowed=False,
            evidence=("Raft evidence from the admitted reference corpus.",),
        )

    async def _amplify(_objective, _generate, **kwargs):
        amplification.update(kwargs)
        # The REAL AmplifiedAnswer/ReasoningReceipt. The hand-rolled version
        # omitted promotion_authority, which became an adoption precondition
        # after this fake was written — so the phase correctly kept the latent
        # draft and this test asserted a composed answer on a path it never
        # took.
        return amplified_answer(
            "Verified composition: Raft is appropriate for a crash-only failure "
            "model, while PBFT handles Byzantine faults. The recommendation changes "
            "because Byzantine nodes may equivocate instead of merely stopping, so "
            "Raft's majority-log guarantees no longer cover the requested threat.",
            promotion_authority="checked_verifier",
            confidence=0.96,
            strategy_used="rlc_seed+verifier",
        )

    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.run_foreground_latent_episode",
        _run_latent,
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.reasoning_amplifier_v2_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "core.brain.reasoning_amplifier_v2.is_amplifiable",
        lambda _objective: "architecture",
    )
    monkeypatch.setattr(
        "core.brain.reasoning_amplifier_v2.amplify_turn",
        _amplify,
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(
            lambda _state, current: [
                {"role": "system", "content": "bounded context"},
                {"role": "user", "content": current},
            ]
        ),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: (
                llm
                if name == "llm_router"
                else orchestrator
                if name == "orchestrator"
                else default
            )
        ),
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.build_response_contract",
        lambda _state, _objective, is_user_facing=False: ResponseContract(
            is_user_facing=is_user_facing,
            reason="ordinary_dialogue",
        ),
    )

    new_state = await phase.execute(
        state,
        objective=objective,
        priority=True,
        context={
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "visible_user_message": objective,
        },
    )

    llm.think.assert_not_awaited()
    assert captured["desktop_required"] is True
    assert captured["messages"][-1]["content"] == objective
    assert new_state.response_modifiers["latent_cortex_succeeded"] is True
    assert new_state.response_modifiers["response_path"] == "cognitive_engine_latent_cortex"
    assert new_state.response_modifiers["latent_cortex_amplifier_composed"] is True
    assert amplification["extra_context"]["seed_candidates"][0].startswith("Raft is")
    assert amplification["evidence"] == [
        "Raft evidence from the admitted reference corpus."
    ]
    assert "Verified composition" in new_state.cognition.last_response


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_allowed", [True, False])
async def test_unitary_latent_failure_respects_resident_owner_disposition(
    monkeypatch,
    fallback_allowed,
):
    from core.brain.foreground_latent_runtime import ForegroundLatentOutcome

    objective = "Explain Raft's failure model and quorum behavior in detail."
    state = AuraState()
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_objective = objective
    llm = SimpleNamespace(
        think=AsyncCallProbe(
            return_value=(
                "Raft assumes a failed node stops participating rather than sending conflicting "
                "messages, and it uses a leader plus majority quorums to replicate its log. "
                "A committed entry remains durable because any later leader must win an election "
                "from a majority that intersects the committing quorum. This crash-fault model "
                "does not protect against a node that lies or sends conflicting messages."
            )
        )
    )
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

    async def _run_latent(**_kwargs):
        return ForegroundLatentOutcome(
            text="",
            trace={
                "latent_cortex_selected": True,
                "latent_cortex_attempted": True,
                "latent_cortex_succeeded": False,
                "latent_cortex_fallback_used": True,
                "latent_cortex_failure_reason": "worker_failure",
                "latent_cortex_identity_bound": False,
                "latent_cortex_receipt": {
                    "episode_id": "failed-unitary-episode",
                    "last_stage": "branch_select",
                },
                "latent_cortex_progress": {"stage": "branch_select"},
            },
            fallback_allowed=fallback_allowed,
        )

    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.run_foreground_latent_episode",
        _run_latent,
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(
            lambda _state, current: [
                {"role": "system", "content": "bounded context"},
                {"role": "user", "content": current},
            ]
        ),
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

    new_state = await phase.execute(
        state,
        objective=objective,
        priority=True,
        context={
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "visible_user_message": objective,
        },
    )

    if fallback_allowed:
        llm.think.assert_awaited_once()
        assert "failed node stops participating" in new_state.cognition.last_response
        assert new_state.response_modifiers.get("model_retry_suppressed") is not True
    else:
        llm.think.assert_not_awaited()
        assert new_state.response_modifiers["model_retry_suppressed"] is True
        assert (
            new_state.response_modifiers["response_path"]
            == "cognitive_engine_latent_owner_exhausted"
        )


@pytest.mark.asyncio
async def test_unitary_semantic_shadow_keeps_ordinary_reply_and_records_comparison(
    monkeypatch,
):
    from core.brain.foreground_latent_runtime import ForegroundLatentOutcome

    objective = "Compute the bounded posterior and return the required JSON."
    qualified = 'FINAL_ANSWER: {"posterior_denominator":7,"posterior_numerator":3}'
    ordinary = (
        "I independently computed the update.\n"
        'FINAL_ANSWER: {"posterior_denominator":7,"posterior_numerator":3}'
    )
    state = AuraState()
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_objective = objective
    llm = SimpleNamespace(think=AsyncCallProbe(return_value=ordinary))
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))
    observed = {}

    async def _run_latent(**_kwargs):
        return ForegroundLatentOutcome(
            text="",
            trace={
                "latent_cortex_selected": True,
                "latent_cortex_attempted": True,
                "latent_cortex_succeeded": False,
                "qualified_recurrent_shadowed": True,
                "qualified_recurrent_receipt": {
                    "admission": {
                        "family": "frontier_calibration",
                        "parser_id": "semantic_calibration_canonical.v1",
                        "receipt_sha256": "a" * 64,
                    },
                    "activation_receipt": {
                        "package_id": "cp568-resident-semantic-neural-shadow",
                        "promotion_mode": "shadow",
                        "activation_sha256": "b" * 64,
                    },
                },
            },
            fallback_allowed=True,
            evidence=("qualified_semantic_neural_shadow",),
            shadow_text=qualified,
        )

    async def _record(**kwargs):
        observed.update(kwargs)
        return {
            "schema": "aura.semantic_neural_shadow.v1",
            "receipt_sha256": "c" * 64,
            "answer_match": True,
            "persisted": True,
        }

    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.run_foreground_latent_episode",
        _run_latent,
    )
    monkeypatch.setattr(
        "core.brain.llm.semantic_neural_shadow.record_semantic_shadow_comparison",
        _record,
    )
    monkeypatch.setattr(
        "core.phases.response_generation_unitary.ContextAssembler.build_messages",
        staticmethod(
            lambda _state, current: [
                {"role": "system", "content": "bounded context"},
                {"role": "user", "content": current},
            ]
        ),
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

    new_state = await phase.execute(
        state,
        objective=objective,
        priority=True,
        context={
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "visible_user_message": objective,
        },
    )

    llm.think.assert_awaited()
    assert observed["qualified_text"] == qualified
    assert observed["ordinary_text"] == new_state.cognition.last_response
    assert new_state.cognition.last_response.startswith("I independently computed")
    assert new_state.response_modifiers["qualified_recurrent_shadow_recorded"] is True
    assert (
        new_state.response_modifiers["qualified_recurrent_shadow_comparison"][
            "receipt_sha256"
        ]
        == "c" * 64
    )


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
        "summary": (
            "Task accepted into governed background execution (id=48792829). "
            "The task ledger is tracking completion status. No completion is claimed yet. "
            "Tracking commitment 8ec3f96b."
        ),
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
async def test_unitary_response_started_task_ack_is_evidence_bounded_without_llm(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Open Notes, write a timestamped summary, and export it as a PDF."
    state.response_modifiers["last_task_outcome"] = "started"
    state.response_modifiers["last_task_result_payload"] = {
        "status": "started",
        "summary": "Queued desktop_task through governed task execution.",
        "task_id": "task-123",
        "commitment_id": "commit-456",
    }

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="I should not be called."))
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_not_awaited()
    reply = new_state.cognition.last_response
    # The acknowledgement stopped being a ticket on 2026-08-26 — "a task
    # somebody is watching stays in front of them" — and this asserted the
    # ticket's wording rather than what the wording was for. The properties
    # are the same: it names the governed path it went down, it claims no
    # completion, and it does not say work is under way when it is queued.
    lowered = reply.lower()
    assert "governed" in lowered, reply
    assert "desktop_task" in lowered, reply
    assert "started working" not in lowered
    assert "keep you updated" not in lowered
    # No completion claimed: it says it will report when the work is done.
    assert "done" in lowered or "no completion" in lowered, reply
    assert "i'll" not in reply.lower()


@pytest.mark.asyncio
async def test_unitary_response_started_task_missing_payload_fails_closed(monkeypatch):
    state = AuraState()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Keep working on the desktop task."
    state.response_modifiers["last_task_outcome"] = "started"
    state.response_modifiers["last_task_result_payload"] = {}

    llm = SimpleNamespace(think=AsyncCallProbe(return_value="I should not be called."))
    phase = UnitaryResponsePhase(SimpleNamespace(organs={}))

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: llm if name == "llm_router" else default),
    )

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    llm.think.assert_not_awaited()
    reply = new_state.cognition.last_response.lower()
    assert "no task id" in reply
    assert "will not claim progress" in reply
    assert "keep you updated" not in reply
    assert "started working" not in reply


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
