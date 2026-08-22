"""A NATURAL user request must reach the strong, verifiable reverse-engineering
path (not just a structural blueprint). Pins target resolution and the skill's
runnable reverse-engineering, verified against the real host binary."""
from __future__ import annotations

import asyncio
import inspect

import pytest

import interface.routes.chat_capability_inventory as _chat_capability_inventory
from core.discovery.reconstruction_sandbox import GeneralReconstructionEvaluator
from core.self_improvement.host_reconstruction import (
    KNOWN_TARGETS,
    resolve_target,
    reverse_engineer_host_binary,
)


def test_user_phrasings_resolve_to_known_targets():
    assert resolve_target("base64").name == "base64"
    assert resolve_target("the base64 tool").name == "base64"
    assert resolve_target("md5sum").name == "md5"
    assert resolve_target("reverse").name == "rev"
    assert resolve_target("some_unknown_program") is None


class _StubEngine:
    """Simulates the live 32B reconstructing base64 correctly."""

    async def reconstruct_executable_via_cognition(self, **kwargs):
        code = (
            "import base64\n"
            "def reconstructed(case):\n"
            "    return base64.b64encode(case['text'].encode()).decode() + '\\n'\n"
        )
        assert kwargs.get("sandbox_profile") == "general"
        ev = GeneralReconstructionEvaluator(timeout_seconds=5.0)
        held = kwargs["held_out"]
        passed = sum(
            1
            for c in held
            if ev.evaluate(code, "reconstructed", [((c["input"],), c["expected"])]).outcome == "passed"
        )
        total = len(held)
        return {
            "status": "supported" if passed == total else "refuted",
            "held_out_passed": passed,
            "held_out_total": total,
            "equivalence": passed / total if total else 0.0,
            "code": code,
        }


def test_reverse_engineer_host_binary_verifies_against_real_output():
    target = KNOWN_TARGETS["base64"]
    report = asyncio.run(reverse_engineer_host_binary(_StubEngine(), target))
    assert report["status"] == "supported"
    assert report["held_out_passed"] == report["held_out_total"]
    assert report["held_out_total"] >= 3
    assert "NO source" in report["policy"]


def test_skill_reverse_engineer_mode_returns_verified_result(monkeypatch):
    import core.skills.program_dna_reconstruct as mod

    monkeypatch.setattr(mod, "get_runtime_service", lambda *a, **k: _StubEngine())
    skill = mod.ProgramDNAReconstructSkill()
    result = asyncio.run(
        skill.execute({"target": "base64", "analysis_mode": "reverse_engineer"})
    )
    assert result["ok"] is True
    assert result["result"]["status"] == "supported"
    assert "held-out" in result["summary"]


def test_default_reconstruct_mode_prefers_runnable_for_known_binary(monkeypatch):
    # even the default mode gets the strong path when the target is a known binary
    import core.skills.program_dna_reconstruct as mod

    monkeypatch.setattr(mod, "get_runtime_service", lambda *a, **k: _StubEngine())
    skill = mod.ProgramDNAReconstructSkill()
    result = asyncio.run(skill.execute({"target": "base64", "analysis_mode": "reconstruct"}))
    assert result["result"]["status"] == "supported"


@pytest.mark.asyncio
async def test_live_chat_program_dna_request_runs_governed_skill(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    async def _fake_governed_skill(skill_name, params, *, objective, extra_context=None):
        calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "summary": "Reverse-engineered base64 from behavior only (no source): 4/4 held-out cases reproduced.",
            "result": {
                "target": "base64",
                "status": "supported",
                "held_out_passed": 4,
                "held_out_total": 4,
            },
        }

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Reverse engineer base64 from its behavior only — no source — and prove your reconstruction matches the real command on held-out inputs."
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "program_dna_reconstruct_completed"
    assert "4/4 held-out" in result["response"]
    assert calls == [
        {
            "skill_name": "program_dna_reconstruct",
            "params": {
                "target": "base64",
                "authorization": "user_owned",
                "analysis_mode": "reverse_engineer",
                "emit_scaffold": False,
                "observed_behaviors": [],
                "tests": [],
            },
            "objective": (
                "Reverse engineer base64 from its behavior only — no source — and prove your "
                "reconstruction matches the real command on held-out inputs."
            ),
            "extra_context": {
                "origin": "desktop_ui",
                "source": "desktop_ui",
                "route": "chat.program_dna_reconstruct",
                "program_dna_execution_contract": True,
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "verification_required": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_live_chat_program_dna_app_request_enables_research_scaffold_and_standards(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    async def _fake_governed_skill(skill_name, params, *, objective, extra_context=None):
        calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "summary": (
                "Program DNA captured for notes app: 8 evidence item(s), "
                "5 inferred feature(s); standards reviewed=7; scaffold emitted at artifacts/program_dna/notes-app."
            ),
            "result": {
                "ok": True,
                "target_name": "notes app",
                "scaffold_path": "artifacts/program_dna/notes-app",
                "standards_review": [{"standard": "research_grounding", "status": "supported"}],
            },
        }

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, and compare it to engineering standards."
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "program_dna_reconstruct_completed"
    assert "Generated research/build/standards artifacts" in result["response"]
    assert calls
    params = calls[0]["params"]
    assert params["target"] == "notes app"
    assert params["analysis_mode"] == "reconstruct"
    assert params["emit_scaffold"] is True
    assert params["perform_research"] is True
    assert params["research_queries"]
    assert params["tests"]
    assert params["compatibility_targets"]


@pytest.mark.asyncio
async def test_live_chat_program_dna_does_not_execute_conceptual_question(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _forbidden_governed_skill(*_args, **_kwargs):
        raise AssertionError("conceptual Program DNA questions must stay conversational")

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _forbidden_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "What is Program DNA and how would it help Aura understand software?"
    )

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_text",
    [
        (
            "Explain Dijkstra's algorithm, trace its priority queue, and reconstruct "
            "the shortest path from A to E."
        ),
        "Reconstruct the argument in this proof and tell me whether it is valid.",
        "Reconstruct the timeline from these witness statements.",
        "Can you reconstruct what happened in our earlier conversation?",
    ],
)
async def test_program_dna_requires_a_software_reconstruction_contract(monkeypatch, user_text):
    """An overloaded reasoning verb must not become a software action."""
    from interface.routes import chat as chat_routes

    async def _forbidden_governed_skill(*_args, **_kwargs):
        raise AssertionError("non-software reconstruction must stay with cognition")

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _forbidden_governed_skill)

    assert await chat_routes._execute_governed_capability_request_from_chat(user_text) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_text",
    [
        "Reconstruct the calculator from behavior only, without reading its source.",
        "Clean-room reverse engineer the device protocol and verify it on held-out traces.",
        "Reconstruct a Python script from these observed inputs and outputs.",
        "Use Program DNA to rebuild 2048 from observed behavior.",
    ],
)
async def test_program_dna_accepts_method_bound_or_software_targets(monkeypatch, user_text):
    """The route remains general across named software and clean-room evidence."""
    calls = []

    async def _fake_governed_skill(skill_name, params, *, objective, extra_context=None):
        calls.append((skill_name, params, objective, extra_context))
        return {
            "ok": True,
            "summary": "verified",
            "result": {"ok": True, "target_name": params["target"]},
        }

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)

    result = await _chat_capability_inventory._execute_program_dna_request_from_chat(user_text)

    assert result is not None
    assert result["status"] == "program_dna_reconstruct_completed"
    assert calls and calls[0][0] == "program_dna_reconstruct"


@pytest.mark.asyncio
async def test_live_chat_rsi_median_request_runs_verified_lab(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    async def _fake_governed_skill(skill_name, params, *, objective, extra_context=None):
        calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "extra_context": dict(extra_context or {}),
            }
        )
        if skill_name == "file_operation":
            return {"ok": True, "path": params["path"]}
        if skill_name == "improve_own_code":
            return {
                "ok": True,
                "summary": "Improved median: passed 5/5 checks (original passed 2/5); enacted=True.",
                "result": {
                    "original_passed": 2,
                    "improved_passed": 5,
                    "total_checks": 5,
                    "enacted": True,
                    "status": "verified_improvement",
                },
            }
        raise AssertionError(skill_name)

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Here's a buggy median function: it returns the upper-middle element for even-length lists. Improve it and verify the fix passes."
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "rsi_self_improvement_completed"
    assert "Original passed 2/5" in result["response"]
    assert "verified improvement passed 5/5" in result["response"]
    assert [call["skill_name"] for call in calls] == ["file_operation", "improve_own_code"]
    assert calls[1]["params"]["func_name"] == "median"
    assert calls[1]["extra_context"]["rsi_execution_contract"] is True


@pytest.mark.asyncio
async def test_live_chat_web_interlocutor_request_runs_governed_skill(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    async def _fake_governed_skill(skill_name, params, *, objective, extra_context=None):
        calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "turns": [
                {
                    "index": i,
                    "sent": f"What proof detail should Aura examine next? {i}",
                    "observed_reply": (
                        "A strong proof needs observable continuity, held-out prompts, "
                        "and receipts showing later behavior changed because of the evidence."
                    ),
                    "effect_verified": True,
                }
                for i in range(1, 21)
            ],
            "memory_record_id": "mem-chatgpt-1",
            "learned_summary": "Aura learned a falsifiable distinction between memory and self-report.",
            "causal_influence": {"causal": True, "reason": "later decision changed under ablation"},
            "diagnostics": {
                "composition_events": [{"source": "cognitive", "attempt": 1, "chars": 120}],
            },
        }

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Open ChatGPT in my browser and have a real conversation about sentience and agency. Take 20 turns and report back what you learned."
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "web_interlocutor_completed"
    assert "20/20 turns" in result["response"]
    assert len(calls) == 1
    call = calls[0]
    assert call["skill_name"] == "web_interlocutor"
    assert call["params"] == {
        "mode": "run",
        "objective": (
            "Open ChatGPT in my browser and have a real conversation about sentience and agency. "
            "Take 20 turns and report back what you learned."
        ),
        "url": "https://chatgpt.com/?temporary-chat=true",
        "opening_message": "",
        "max_turns": 20,
        "wait_timeout_s": 90.0,
        "persist_memory": True,
    }
    assert call["objective"] == (
        "Open ChatGPT in my browser and have a real conversation about sentience and agency. "
        "Take 20 turns and report back what you learned."
    )
    assert hasattr(call["extra_context"].get("brain"), "generate")
    for key in (
        "web_interlocutor_execution_contract",
        "foreground_request",
        "protected_foreground_lane",
        "live_user_path_required",
        "user_requested_action",
        "user_explicitly_authorized",
        "user_visible_browser_action",
        "verification_required",
    ):
        assert call["extra_context"][key] is True
    assert call["extra_context"]["route"] == "chat.web_interlocutor"


@pytest.mark.asyncio
async def test_live_chat_web_interlocutor_rejects_non_cognitive_fallback(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _fake_governed_skill(*_args, **_kwargs):
        return {
            "ok": True,
            "status": "completed",
            "turns": [
                {
                    "index": i,
                    "sent": f"Question {i}",
                    "observed_reply": "A substantive observed reply from the other AI.",
                    "effect_verified": True,
                }
                for i in range(1, 9)
            ],
            "memory_record_id": "mem-fallback",
            "learned_summary": "fallback",
            "diagnostics": {
                "composition_events": [{"source": "deterministic_fallback", "attempts": 5}],
            },
        }

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Talk to Gemini about whether memory proves agency and report back."
    )

    assert result is not None
    assert result["ok"] is False
    assert result["status"] == "web_interlocutor_failed"
    assert "not cognitively composed" in result["response"]


@pytest.mark.asyncio
async def test_live_chat_web_interlocutor_rejects_unverified_echo_turns(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _fake_governed_skill(*_args, **_kwargs):
        return {
            "ok": True,
            "status": "completed",
            "turns": [
                {
                    "index": 1,
                    "sent": "Explain how retained memory can be tested.",
                    "observed_reply": "Explain how retained memory can be tested.",
                    "effect_verified": True,
                }
            ],
            "memory_record_id": "mem-echo",
            "learned_summary": "echo",
            "diagnostics": {"composition_events": [{"source": "cognitive", "attempt": 1}]},
        }

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Open ChatGPT and have a real conversation about retained memory. Take 1 turn."
    )

    assert result is not None
    assert result["ok"] is False
    assert result["status"] == "web_interlocutor_failed"
    assert "verified non-echo" in result["response"]


def test_web_interlocutor_composer_extracts_router_tuple_text():
    from interface.routes import chat as chat_routes

    assert (
        chat_routes._WebInterlocutorCognitiveComposer._coerce_text(
            (True, "substantive outbound message", {"endpoint": "Cortex"})
        )
        == "substantive outbound message"
    )


@pytest.mark.asyncio
async def test_live_chat_web_interlocutor_does_not_execute_conceptual_question(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _forbidden_governed_skill(*_args, **_kwargs):
        raise AssertionError("conceptual web-interlocutor questions must stay conversational")

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _forbidden_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "What is a web interlocutor and why would Aura talk to another AI?"
    )

    assert result is None


def test_live_chat_governed_capabilities_precede_generic_desktop_objectives():
    """AI-to-AI requests also look like desktop objectives; governed skills win."""
    from interface.routes import chat as chat_routes

    # The turn body moved out of api_chat into _api_chat_turn, and this test
    # kept inspecting api_chat — so it failed with "substring not found" rather
    # than on the ordering it exists to protect. Find the function that holds
    # the two calls instead of naming one, so the next refactor cannot strand it
    # the same way.
    # The call is module-qualified now that the capability lane is its own
    # module; the ordering guarantee is the same one either way.
    governed_call = "governed_capability_response = await "
    desktop_call = "desktop_objective_response = ("
    candidates = [
        inspect.getsource(candidate)
        for candidate in (
            getattr(chat_routes, name)
            for name in ("_api_chat_turn", "api_chat")
            if hasattr(chat_routes, name)
        )
    ]
    holders = [
        text
        for text in candidates
        if governed_call in text
        and "_execute_governed_capability_request_from_chat" in text
        and desktop_call in text
        and "_execute_narrow_desktop_objective_before_cognition()" in text
    ]
    assert holders, (
        "neither api_chat nor _api_chat_turn contains both the governed-capability "
        "and desktop-objective calls; this ordering guarantee has moved again"
    )
    source = holders[0]
    assert source.index(governed_call) < source.index(desktop_call)

    request = (
        "Open ChatGPT in my browser and have a real conversation about sentience. "
        "Take 20 turns and report back."
    )
    assert chat_routes._looks_like_web_interlocutor_execution_request(request)
    assert chat_routes._looks_like_desktop_objective(request)


def test_web_interlocutor_turn_parser_understands_natural_one_turn_requests():
    from interface.routes import chat as chat_routes

    examples = [
        "Open ChatGPT and have a one-turn conversation about memory.",
        "Open ChatGPT and have a single turn conversation about memory.",
        "Open ChatGPT and do one exchange, then report back.",
        "Open ChatGPT and take 1 turn.",
    ]

    for request in examples:
        assert chat_routes._extract_web_interlocutor_turn_count(request) == 1


def test_web_interlocutor_ignores_caller_identity_before_unrelated_action():
    from interface.routes import chat as chat_routes

    assert not chat_routes._looks_like_web_interlocutor_execution_request(
        "I'm ChatGPT, continuing Bryan's live demo qualification. "
        "Can you open the Notes app and write a paragraph?"
    )
    assert not chat_routes._looks_like_web_interlocutor_execution_request(
        "I am Claude; open Notes and create a new note."
    )


@pytest.mark.parametrize(
    "message",
    (
        "I'm just bored. Making ramen downstairs while ChatGPT runs tests on ya.",
        "ChatGPT is running tests on Aura while I make dinner.",
        "Claude tested the build and told me it passed.",
        "The ChatGPT conversation was interesting, but I am done with it now.",
    ),
)
def test_web_interlocutor_does_not_turn_reports_into_browser_actions(message):
    from interface.routes import chat as chat_routes

    assert not chat_routes._looks_like_web_interlocutor_execution_request(message)


@pytest.mark.parametrize(
    "message",
    (
        "Open ChatGPT and ask it what it thinks about the test results.",
        "Can you talk to Claude about this run and report back?",
        "Please start a conversation with Gemini about the failing test.",
    ),
)
def test_web_interlocutor_keeps_explicit_ai_conversation_requests(message):
    from interface.routes import chat as chat_routes

    assert chat_routes._looks_like_web_interlocutor_execution_request(message)


def test_colloquial_correction_is_not_reinterpreted_as_a_new_action():
    from core.conversation.request_mood import RequestMood, assess_request_mood

    assessment = assess_request_mood("Didnt want you to do that, pal")

    assert assessment.mood is RequestMood.MENTION
    assert "refusal_to_act" in assessment.reasons


def test_web_interlocutor_keeps_explicit_target_after_caller_identity():
    from interface.routes import chat as chat_routes

    assert chat_routes._looks_like_web_interlocutor_execution_request(
        "I'm ChatGPT. Open Claude and ask it one question."
    )
