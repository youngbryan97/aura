"""A capability that answers one question should get better at the next.

The repository diagnosis ran, reported the failure, and forgot it. Meanwhile
ErrorPatternAnalyzer sits in this tree looking for failures that recur — and
could not see any of these, because the store it reads only accepted
exceptions. An exception is right for her own faults and cannot describe
somebody else's test run: `log_error` resolves the deepest traceback frame
inside this checkout, and a foreign project has none.

A failure found by running a project already has a type, a message and a
location, printed by that project's own runner. So the store takes one
directly now, and a fault seen in one project and again in another is
something the analyser can finally notice.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.self_modification.error_intelligence import (
    ErrorPatternAnalyzer,
    StructuredErrorLogger,
)


@pytest.fixture()
def store(tmp_path: Path) -> StructuredErrorLogger:
    return StructuredErrorLogger(str(tmp_path))


def failing_project(root: Path, name: str) -> Path:
    (root / "tests").mkdir(parents=True)
    (root / "tests" / f"test_{name}.py").write_text(
        "def test_balance_is_zero():\n    total = 100.0\n    assert total == 0.0\n"
    )
    (root / "README.md").write_text("# x\n\nEvery entry posts an equal and opposite pair.\n")
    return root


def test_the_store_takes_a_failure_with_no_exception(store: StructuredErrorLogger):
    async def run():
        return await store.log_observed_failure(
            error_type="AssertionError",
            error_message="assert 100.0 == 0.0",
            context={"project": "/somewhere/else"},
            skill_name="diagnose_repo",
            file_path="tests/test_x.py",
            line_number=3,
        )

    event = asyncio.run(run())
    assert event.error_type == "AssertionError"
    assert event.file_path == "tests/test_x.py"
    assert event.line_number == 3
    assert event.context["project"] == "/somewhere/else"


def test_a_diagnosis_records_what_it_found(tmp_path: Path):
    from core.skills.diagnose_repo import DiagnoseRepoSkill

    project = failing_project(tmp_path / "one", "one")
    asyncio.run(DiagnoseRepoSkill().execute({"path": str(project)}))

    # The same store the skill wrote to: the one the analyser reads.
    recorded = [
        event
        for event in StructuredErrorLogger().load_all_errors()
        if event.skill_name == "diagnose_repo"
        and str(project) in str(event.context.get("project", ""))
    ]
    assert recorded, "the diagnosis recorded nothing"
    assert recorded[-1].error_message == "assert 100.0 == 0.0"
    assert recorded[-1].context["classification"] == "observed_project_failure"


def test_failures_from_different_projects_all_reach_the_store(tmp_path: Path):
    """This is the point: one project's finding is still there when the next
    arrives, so the analyser has something to compare."""
    from core.skills.diagnose_repo import DiagnoseRepoSkill

    for name in ("one", "two"):
        asyncio.run(
            DiagnoseRepoSkill().execute({"path": str(failing_project(tmp_path / name, name))})
        )

    mine = [
        event
        for event in StructuredErrorLogger().load_all_errors()
        if event.skill_name == "diagnose_repo"
        and str(tmp_path) in str(event.context.get("project", ""))
    ]
    assert len({event.context.get("project") for event in mine}) == 2


def test_the_same_fault_seen_twice_becomes_a_pattern(tmp_path: Path):
    """The analyser fingerprints a fault by where it happens, so two runs of
    one project group and two different projects stay distinct. That is its
    behaviour, not a shortcoming of the feed — the feed's job is that both are
    in the store to be compared at all."""
    from core.skills.diagnose_repo import DiagnoseRepoSkill

    project = failing_project(tmp_path / "repeat", "repeat")
    for _ in range(2):
        asyncio.run(DiagnoseRepoSkill().execute({"path": str(project)}))

    store = StructuredErrorLogger()
    mine = [
        event
        for event in store.load_all_errors()
        if event.skill_name == "diagnose_repo"
        and str(project) in str(event.context.get("project", ""))
    ]
    assert len(mine) >= 2
    store.recent_errors = mine
    patterns = ErrorPatternAnalyzer(store).analyze_recent(window=len(mine))
    assert patterns, "the same fault twice and the analyser found no pattern"
    assert patterns[0].to_dict()["occurrences"] >= 2


def test_a_project_that_passes_records_nothing(tmp_path: Path):
    """Only failures are failures."""
    from core.skills.diagnose_repo import DiagnoseRepoSkill

    good = tmp_path / "good"
    (good / "tests").mkdir(parents=True)
    (good / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    asyncio.run(DiagnoseRepoSkill().execute({"path": str(good)}))

    recorded = [
        event
        for event in StructuredErrorLogger().load_all_errors()
        if event.skill_name == "diagnose_repo" and str(good) in str(event.context.get("project", ""))
    ]
    assert not recorded
