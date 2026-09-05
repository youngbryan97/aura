"""The comparison that would be evidence, and the five objections to it.

Everything else in this work is an architectural improvement. A matched
independent experiment is the one thing that could be evidence, and only if it
answers the obvious objections before anyone raises them: the arms were not
matched, the tasks were written by the system under test, the grader knew
which arm it was reading, the win came from somewhere uninteresting, and there
was no null.

One test per objection, plus the two that decide whether the instrument works
at all: can it detect a real effect, and does it stay quiet on a fake one.
"""

from __future__ import annotations

import random
import re

import pytest

from core.evals.matched_experiment import (
    Arm,
    Budget,
    Outcome,
    Spend,
    Task,
    run_matched,
)


def _tasks(n: int, seed: int = 2, author: str = "published_benchmark", seen: bool = False):
    rng = random.Random(seed)
    out = []
    for index in range(n):
        left, right = rng.randint(11, 99), rng.randint(11, 99)
        out.append(
            Task(
                task_id=f"t{index}",
                prompt=f"What is {left} times {right}?",
                grade=(lambda answer, want=left * right: str(want) in answer),
                author=author,
                seen_before=seen,
            )
        )
    return out


def _arm_at(rate: float, seed: int):
    rng = random.Random(seed)

    def run(prompt: str, budget: Budget):
        left, right = map(int, re.findall(r"\d+", prompt)[:2])
        product = left * right
        answer = str(product) if rng.random() < rate else str(product + 1)
        return answer, Spend(tokens=40, seconds=0.001)

    return run


BUDGET = Budget(tokens=100, seconds=5.0, tool_calls=2)


# ── does the instrument work ─────────────────────────────────────────────


def test_it_detects_a_real_difference():
    report = run_matched(
        [Arm("base", _arm_at(0.55, 1)), Arm("full", _arm_at(0.85, 3), adds=("interiority",))],
        _tasks(300),
        budget=BUDGET,
    )
    rung = report.ladder[0]
    assert rung.attributable is True
    assert rung.delta > 0.15 and rung.p_value < 0.01
    assert "interiority" in report.verdict


def test_it_stays_quiet_on_two_arms_that_are_the_same():
    report = run_matched(
        [Arm("base", _arm_at(0.70, 7)), Arm("sham", _arm_at(0.70, 8), sham=True)],
        _tasks(300),
        budget=BUDGET,
    )
    assert report.ladder[0].attributable is False
    assert "no rung" in report.verdict


def test_a_real_gap_on_too_few_tasks_is_not_called_a_win():
    """The honest answer when the effect is real and the sample is small."""
    report = run_matched(
        [Arm("base", _arm_at(0.55, 1)), Arm("full", _arm_at(0.85, 3))],
        _tasks(20),
        budget=BUDGET,
    )
    assert report.ladder[0].attributable is False


# ── the five objections ──────────────────────────────────────────────────


def test_an_arm_cannot_win_by_spending_more():
    """Aura with unlimited time against a base model measures the allowance."""

    def greedy(prompt, budget):
        left, right = map(int, re.findall(r"\d+", prompt)[:2])
        return str(left * right), Spend(tokens=99999, seconds=0.001)

    report = run_matched(
        [Arm("base", _arm_at(0.5, 9)), Arm("greedy", greedy, adds=("unlimited tokens",))],
        _tasks(40),
        budget=BUDGET,
    )
    greedy_result = next(a for a in report.arms if a.arm == "greedy")
    assert greedy_result.correct == 0
    assert greedy_result.void == 40
    assert greedy_result.accuracy is None
    assert all(
        t.outcome is Outcome.VOID_BUDGET for t in report.trials if t.arm == "greedy"
    )


def test_a_task_the_system_wrote_is_refused():
    tasks = _tasks(10) + _tasks(1, author="aura")
    report = run_matched([Arm("base", _arm_at(0.5, 1))], tasks, budget=BUDGET)
    assert report.tasks_refused == 1
    assert report.tasks_admissible == 10


def test_a_task_the_system_has_already_seen_is_refused():
    report = run_matched(
        [Arm("base", _arm_at(0.5, 1))], _tasks(10, seen=True), budget=BUDGET
    )
    assert report.tasks_admissible == 0 and report.tasks_refused == 10


def test_the_grader_never_receives_the_arm_label():
    """It is the task's own grader, handed the answer alone."""
    seen = []

    def grade(answer):
        seen.append(answer)
        return True

    task = Task("t", "What is 2 times 2?", grade=grade, author="external_human")
    run_matched(
        [Arm("a", lambda p, b: ("4", Spend(tokens=1))), Arm("b", lambda p, b: ("4", Spend(tokens=1)))],
        [task],
        budget=BUDGET,
    )
    assert seen == ["4", "4"], "the grader was handed something other than the answer"


def test_the_ladder_attributes_a_gain_to_the_rung_that_bought_it():
    report = run_matched(
        [
            Arm("base", _arm_at(0.50, 1)),
            Arm("base+memory", _arm_at(0.52, 2), adds=("memory",)),
            Arm("base+memory+interiority", _arm_at(0.85, 3), adds=("interiority",)),
        ],
        _tasks(300),
        budget=BUDGET,
    )
    by_arm = {r.arm: r for r in report.ladder}
    assert by_arm["base+memory"].attributable is False
    assert by_arm["base+memory+interiority"].attributable is True
    assert by_arm["base+memory+interiority"].adds == ("interiority",)


def test_the_null_is_over_the_pairing_not_against_zero():
    """The arms saw the same tasks; ignoring that throws away the power."""
    import inspect

    from core.evals import matched_experiment

    source = inspect.getsource(matched_experiment._paired_null)
    assert "discordant" in source
    report = run_matched(
        [Arm("a", _arm_at(0.6, 1)), Arm("b", _arm_at(0.6, 1))], _tasks(50), budget=BUDGET
    )
    # Identical seeds mean identical answers: no discordant pairs at all.
    assert report.ladder[0].discordant == 0
    assert report.ladder[0].attributable is False


# ── the record ───────────────────────────────────────────────────────────


def test_the_design_is_sealed_before_the_run():
    report = run_matched([Arm("base", _arm_at(0.5, 1))], _tasks(10), budget=BUDGET)
    assert len(report.seal) == 16
    again = run_matched([Arm("base", _arm_at(0.5, 1))], _tasks(10), budget=BUDGET)
    assert again.seal == report.seal
    different = run_matched(
        [Arm("base", _arm_at(0.5, 1))], _tasks(10), budget=Budget(tokens=99)
    )
    assert different.seal != report.seal


def test_the_alpha_is_a_sealed_criterion():
    from core.verify.epistemic_independence import registry

    registry().clear()
    run_matched([Arm("base", _arm_at(0.5, 1))], _tasks(10), budget=BUDGET)
    criterion = registry().get("matched_experiment.alpha")
    assert criterion is not None and criterion.direction == "below"
    registry().clear()


def test_an_arm_that_raises_is_void_rather_than_wrong():
    def broken(prompt, budget):
        raise RuntimeError("the model was not loaded")

    report = run_matched(
        [Arm("base", _arm_at(0.5, 1)), Arm("broken", broken)], _tasks(5), budget=BUDGET
    )
    broken_result = next(a for a in report.arms if a.arm == "broken")
    assert broken_result.void == 5 and broken_result.wrong == 0


def test_too_few_admissible_tasks_is_inconclusive_not_a_verdict():
    report = run_matched([Arm("base", _arm_at(0.5, 1))], _tasks(3), budget=BUDGET)
    assert "inconclusive" in report.verdict
