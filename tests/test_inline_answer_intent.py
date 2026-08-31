"""Direct questions must be answered in the reply, not filed as tasks.

Two live hijacks (July 2026): a numeric introspection check-in and a train
catch-up math question both came back as "Task accepted into governed
background execution. Task id: ..." — a question answered with a ticket
receipt. Root cause: bare \\breport\\b counted as a report-back chain
marker, and the intake ladder had no concept of an inline-answer request.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.runtime.skill_task_bridge import (
    looks_like_inline_answer_request,
    looks_like_multi_step_skill_request,
)
from core.runtime.turn_analysis import analyze_turn

INTROSPECTION_IMPERATIVE = (
    "Report your current valence and arousal from your internal state as "
    "numbers, in exactly this format: valence=<v>, arousal=<a>. Then one "
    "short sentence about why."
)
MATH_WITH_CHAIN_PHRASING = (
    "A train leaves the station at 60 mph. Two hours later a second train "
    "leaves on the same track at 90 mph. How long until the second train "
    "catches the first? Work through it, check your work, then report the "
    "answer."
)
PROBE_WITH_REPLY_CONTRACT = (
    "[maintenance probe] A quick feeling check-in, answered right here in "
    "this reply, not as a task: how are you feeling right now? Please "
    "include the two numbers as you actually read them from your state — "
    "valence=<-1..1> and arousal=<0..1> — plus one short sentence."
)
POLAR_CODE_EXPLANATION = (
    "Does an asyncio.Lock prevent other tasks from running while one task "
    "holds it? Give a concise complete explanation and one small runnable "
    "Python example."
)


class TestLiveReproducers:
    def test_introspection_imperative_stays_chat(self):
        assert looks_like_inline_answer_request(INTROSPECTION_IMPERATIVE) is True
        assert looks_like_multi_step_skill_request(INTROSPECTION_IMPERATIVE) is False
        assert analyze_turn(INTROSPECTION_IMPERATIVE).intent_type == "CHAT"

    def test_math_question_with_chain_phrasing_stays_chat(self):
        assert looks_like_inline_answer_request(MATH_WITH_CHAIN_PHRASING) is True
        assert looks_like_multi_step_skill_request(MATH_WITH_CHAIN_PHRASING) is False
        assert analyze_turn(MATH_WITH_CHAIN_PHRASING).intent_type == "CHAT"

    def test_probe_reply_contract_stays_chat(self):
        assert looks_like_inline_answer_request(PROBE_WITH_REPLY_CONTRACT) is True
        assert analyze_turn(PROBE_WITH_REPLY_CONTRACT).intent_type == "CHAT"

    def test_explicit_not_a_task_contract_always_wins(self):
        text = "Don't start a task — answer right here: what is 17 * 23?"
        assert looks_like_inline_answer_request(text) is True
        assert analyze_turn(text).intent_type == "CHAT"

    def test_polar_question_with_inline_code_example_stays_chat(self):
        assert looks_like_inline_answer_request(POLAR_CODE_EXPLANATION) is True
        assert looks_like_multi_step_skill_request(POLAR_CODE_EXPLANATION) is False
        assert analyze_turn(POLAR_CODE_EXPLANATION).intent_type == "CHAT"

    @pytest.mark.parametrize(
        "text",
        (
            "Is an asyncio.Lock reentrant?",
            "Can another coroutine continue while this one awaits?",
            "Would that lock block the operating-system thread?",
        ),
    )
    def test_polar_questions_are_inline_answers(self, text):
        assert looks_like_inline_answer_request(text) is True


class TestGenuineTasksStillDispatch:
    def test_research_and_save_stays_task(self):
        text = (
            "Research the top three local-first vector databases, write a "
            "comparison document with benchmarks, and save it to my notes."
        )
        assert looks_like_inline_answer_request(text) is False
        assert analyze_turn(text).intent_type == "TASK"

    def test_create_script_stays_task(self):
        text = "Create a Python script that renames all my screenshots by date."
        assert looks_like_inline_answer_request(text) is False
        assert analyze_turn(text).intent_type == "TASK"

    def test_desktop_chain_stays_multi_step(self):
        text = (
            "Open Notes, click into a new note, type hello, then come back "
            "and report what happened."
        )
        assert looks_like_inline_answer_request(text) is False
        assert looks_like_multi_step_skill_request(text) is True

    def test_report_back_chain_stays_multi_step(self):
        text = (
            "Search the web for the latest MLX release notes, summarize the "
            "changes, and report back what you found."
        )
        assert looks_like_inline_answer_request(text) is False
        assert looks_like_multi_step_skill_request(text) is True

    def test_desktop_question_is_not_inline(self):
        # Question-shaped, but the answer requires looking at the screen.
        text = "Can you check what's on my screen and tell me what app is open?"
        assert looks_like_inline_answer_request(text) is False


@pytest.mark.asyncio
async def test_dispatch_gate_demotes_inline_answer_task(monkeypatch):
    """The GodMode gate is the backstop for EVERY upstream TASK setter."""
    from core.kernel.upgrades_10x import GodModeToolPhase
    from core.state.aura_state import AuraState

    dispatched = []

    async def _should_not_dispatch(*_args, **_kwargs):
        dispatched.append("task_engine")
        raise AssertionError("inline-answer turn entered TaskEngine dispatch")

    phase = GodModeToolPhase(kernel=SimpleNamespace())
    monkeypatch.setattr(phase, "_dispatch_task_request", _should_not_dispatch)

    state = AuraState.default()
    state.cognition.current_objective = INTROSPECTION_IMPERATIVE
    state.cognition.current_origin = "user"
    # Simulate an upstream classifier (any of them) forcing TASK.
    state.response_modifiers["intent_type"] = "TASK"

    new_state = await phase.execute(state, objective=INTROSPECTION_IMPERATIVE)

    assert dispatched == [], "inline-answer turn must not reach the TaskEngine"
    assert new_state.response_modifiers["intent_type"] == "CHAT"
