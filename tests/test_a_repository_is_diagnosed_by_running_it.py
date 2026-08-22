"""A question about code goes to the code, not to the desktop.

LIVE, 2026-08-22, typed into the window: "there's a small python project at
<path> — one of its tests fails and I can't see why." The turn routed to
`os_automation`, timed out, and completed nothing. A question about a
directory of Python had gone to the mouse-and-keyboard lane.

There is a lot of code machinery in this tree — repair, refactor, health, the
AST analyser, the error intelligence — and every piece of it is pointed at her
own source. None could be aimed at a directory somebody names.

Everything reported here is observed by running the project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.diagnosis.repository import describe_diagnosis, diagnose_repository


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "ledger").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "ledger" / "__init__.py").write_text("")
    (tmp_path / "ledger" / "accounts.py").write_text(
        "class Ledger:\n"
        "    def __init__(self):\n"
        "        self.total = 0\n"
        "    def post(self, amount):\n"
        "        self.total += amount\n"
        "    def trial_balance(self):\n"
        "        return self.total\n"
    )
    (tmp_path / "tests" / "test_ledger.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from ledger.accounts import Ledger\n"
        "\n"
        "def test_passes():\n"
        "    assert Ledger().trial_balance() == 0\n"
        "\n"
        "def test_trial_balance_is_zero():\n"
        "    book = Ledger()\n"
        "    book.post(100.0)\n"
        "    assert book.trial_balance() == 0.0\n"
    )
    (tmp_path / "README.md").write_text(
        "# ledger\n\nA tiny double-entry ledger. `trial_balance()` must always be zero\n"
        "because every transfer posts an equal and opposite pair.\n"
    )
    return tmp_path


def test_a_directory_that_is_not_one_says_so(tmp_path: Path):
    diagnosis = diagnose_repository(tmp_path / "nowhere")
    assert diagnosis.error
    assert "not a directory" in diagnosis.error


def test_a_project_with_no_runner_says_so(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("nothing to run here")
    diagnosis = diagnose_repository(tmp_path)
    assert "no test runner" in diagnosis.error


def test_it_runs_the_project_and_reports_the_real_failure(project: Path):
    diagnosis = diagnose_repository(project)
    assert not diagnosis.error, diagnosis.error
    assert diagnosis.ok is False
    assert diagnosis.passed == 1
    assert diagnosis.failed == 1
    assert diagnosis.failures
    first = diagnosis.failures[0]
    assert "test_trial_balance_is_zero" in first.test
    # The assertion is the runner's own words, not a guess at them.
    assert "assert" in first.assertion and "100.0" in first.assertion


def test_it_quotes_the_source_and_marks_the_failing_line(project: Path):
    diagnosis = diagnose_repository(project)
    assert "trial_balance()" in diagnosis.source
    assert ">" in diagnosis.source
    assert "trial_balance" in diagnosis.called_functions


def test_it_surfaces_what_the_project_says_it_should_do(project: Path):
    """The stated invariant beside the failure is what makes the fault
    visible: a single-sided posting against "an equal and opposite pair"."""
    diagnosis = diagnose_repository(project)
    assert "equal and opposite" in diagnosis.stated_intent


def test_a_passing_project_reports_that_plainly(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    diagnosis = diagnose_repository(tmp_path)
    assert diagnosis.ok is True
    assert diagnosis.failed == 0
    assert "nothing failed" in describe_diagnosis(diagnosis)


def test_the_skill_is_not_a_desktop_action():
    """It was routed to os_automation, which is the whole defect."""
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    assert SKILL_EFFECT_SCOPES["diagnose_repo"] == "sandboxed_compute"


def test_it_is_rated_for_what_it_actually_runs():
    """LIVE, 2026-08-22: rated high by effect scope alone, it was refused with
    "Requires user confirmation" on the very turn that asked for it, after
    routing had finally found it.

    sandboxed_compute is rated high because it usually means executing code
    the model just wrote. This runs the project's own tests: the code was on
    disk before the turn began, the person named the directory, and nothing
    the model produces is executed. Medium rather than low, because a test
    suite is still somebody else's code running on this machine.
    """
    from core.executive.execution_policy import classify_execution_risk

    assert (
        classify_execution_risk("diagnose_repo", {}, effect_scope="sandboxed_compute")
        == "medium"
    )
    # The tools that DO run model-written code keep their rating.
    assert classify_execution_risk("code_repl", {}, effect_scope="sandboxed_compute") == "high"
    assert (
        classify_execution_risk("run_code", {}, effect_scope="sandboxed_compute") == "critical"
    )


def test_the_finding_becomes_the_turns_answer(tmp_path):
    """LIVE, 2026-08-22: the skill ran in 428ms and returned the failing test,
    the assertion, the line and the project's stated invariant — and the turn
    served "I couldn't get to an answer I'd stand behind", because the model's
    draft of that finding was rejected for missing the very numbers the
    finding contains. The runtime had the answer and was asking the model to
    reproduce it."""
    import asyncio

    from core.conversation.session_scope import set_user_question, solved_answers
    from core.skills.diagnose_repo import DiagnoseRepoSkill

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_broken():\n    assert 100.0 == 0.0\n"
    )

    async def run() -> str:
        set_user_question("why does it fail?")
        await DiagnoseRepoSkill().execute({"path": str(tmp_path)})
        return solved_answers().get("repo_diagnosis", "")

    recorded = asyncio.run(run())
    assert "test_broken" in recorded
    assert "100.0" in recorded


def test_a_skill_running_beneath_the_turn_can_still_record(tmp_path):
    """A ContextVar written in a child task is invisible to the parent, and a
    skill runs beneath the turn that reads its answer. That trap cost three
    separate mechanisms in one day."""
    import asyncio

    from core.conversation.session_scope import (
        record_solved_answer,
        set_user_question,
        solved_answers,
    )

    async def run() -> dict:
        set_user_question("anything")

        async def beneath() -> None:
            record_solved_answer("repo_diagnosis", "found it")

        await asyncio.wait_for(asyncio.create_task(beneath()), timeout=5)
        return solved_answers()

    assert asyncio.run(run())["repo_diagnosis"] == "found it"
