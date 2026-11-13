"""The G09 harness samples across groups reproducibly and grades as each source does."""

from __future__ import annotations

from tools.run_g09_organ import _normalize, _parse_calendar, grade_knowledge, grade_planning, sample


def test_tasks_are_spread_across_groups_reproducibly_and_never_reuse_an_excluded_id() -> None:
    tasks = [{"id": f"{group}:{i}", "group": group} for group in "abc" for i in range(10)]
    first = sample(tasks, 6, 7, {"a:3"})
    assert [t["id"] for t in first] == [t["id"] for t in sample(tasks, 6, 7, {"a:3"})]
    assert sorted(t["group"] for t in first) == ["a", "a", "b", "b", "c", "c"]
    assert "a:3" not in {t["id"] for t in first}


def test_a_proposed_time_is_read_as_natural_plan_reads_it() -> None:
    golden = "Here is the proposed time: Monday, 14:30 - 15:30 "
    assert _parse_calendar(golden) == ("Monday", 14.5, 15.5)
    assert grade_planning("I propose Monday, 14:30 - 15:30 since all are free.", golden)[0]
    assert not grade_planning("Monday, 15:00 - 16:00", golden)[0]
    assert not grade_planning("No time works.", golden)[0]


def test_a_short_answer_is_matched_after_hotpotqas_normalisation() -> None:
    assert _normalize("The  Eiffel Tower!") == "eiffel tower"
    assert grade_knowledge("Both were born in the US.\n**Answer:** yes", "yes")[0]
    assert grade_knowledge("Answer: Dijon", "dijon")[0]
    assert not grade_knowledge("Answer: Paris", "Dijon")[0]


def test_bbeh_answers_are_read_as_bbehs_own_evaluator_reads_them() -> None:
    """The eight cases bbeh/evaluate.py checks itself with."""
    from tools.run_g09_organ import grade_bbeh

    cases = [("Ok The final answer is: \\boxed{4}.", "4", True), ("[Reasoning] The final answer is: \\boxed{4}.", "3", False),
             ("Alright! The final answer is: 2, 3, 4", "2,3,4", True), ("blah The final answer is: 2, 3, 4", "2,3,5", False),
             ("Ok The answer is: (A)", "a", True), ("Ok The answer is: (A)", "b", False),
             ("Ok The answer is: **25**\nHere's why.", "25.0", True), ("Ok The answer is: **25**\nHere's why.", "26.0", False)]
    for reply, reference, expected in cases:
        assert grade_bbeh(reply, reference)[0] is expected, (reply, reference)


def test_a_typed_generation_is_planned_by_the_code_that_plans_it_in_her_chat(monkeypatch) -> None:
    """Channel open only where her worker opens it, bounded where her decoder bounds it, the answer raised as the gate
    raises it and no further than the longest answer seen."""
    from pathlib import Path

    from core.brain.inference_gate import InferenceGate
    from core.brain.llm import a_bounded_private_channel, chat_format, thinking_reserve
    from tools.run_g09_organ import _how_her_chat_answers

    monkeypatch.setattr(thinking_reserve, "answer_tokens_seen", lambda model: 1194)
    monkeypatch.setattr(InferenceGate, "_tokens_the_turn_is_allowed_to_take", staticmethod(lambda **k: 9000))
    monkeypatch.setattr(InferenceGate, "_reasoning_reserve_for_generation", classmethod(lambda cls, **k: 0))
    monkeypatch.setattr(chat_format, "thinking_enabled_for_generation", lambda *a, **k: False)
    closed = _how_her_chat_answers("What is the sum of the squares of the first 30 positive integers?",
                                   Path("/m/model"), 175.0)
    assert closed["thinking"] is False and closed["channel"] == 0
    assert closed["answer_cap"] == closed["max_tokens"] == 1194

    monkeypatch.setattr(chat_format, "thinking_enabled_for_generation", lambda *a, **k: True)
    monkeypatch.setattr(InferenceGate, "_reasoning_reserve_for_generation", classmethod(lambda cls, **k: 1500))
    monkeypatch.setattr(a_bounded_private_channel, "the_channel_budget_for", lambda **k: 1200)
    derived = _how_her_chat_answers("Prove that the square root of 2 is irrational, then find all n with n^2 < 50.",
                                    Path("/m/model"), 475.0)
    assert derived["thinking"] is True and derived["channel"] == 1200
    assert derived["max_tokens"] == derived["answer_cap"] + 1500


def test_the_search_gets_the_room_her_thinking_loop_grants_the_turn(monkeypatch) -> None:
    """Read from the loop's own code: three answers' worth for a checkable request, inside the user-facing ceiling."""
    from core.brain import cognitive_engine, reasoning_amplifier_v2
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S
    from tools.run_g09_organ import _room_the_turn_grants

    monkeypatch.setattr(reasoning_amplifier_v2, "is_amplifiable", lambda q: "math")
    monkeypatch.setattr(cognitive_engine, "_time_the_answer_needs", lambda q: 100.0)
    assert _room_the_turn_grants("What is 7919 multiplied by 104729?") == 300.0
    monkeypatch.setattr(cognitive_engine, "_time_the_answer_needs", lambda q: 400.0)
    assert _room_the_turn_grants("What is 7919 multiplied by 104729?") == USER_FACING_COMPLETION_DEADLINE_MAX_S
