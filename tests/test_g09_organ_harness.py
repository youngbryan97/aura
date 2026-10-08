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
