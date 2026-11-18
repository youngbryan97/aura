"""Before she takes a draft of hers into the source she runs on, her own yes asks the gates (Master Todo F05).

Her check_draft runs the test files a draft changed or named, so a draft that changed only a module and named no test
passed on compile and lint alone and went in. Her yes now asks the tests that name each module she changed, the smoke
suite, lint, layering and the writing gate, in her workspace, and keeps what each said with the merge.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency import her_goal_at_work as work
from tests.test_her_goal_is_worked_by_her import GOAL, NEW_TEST, _git, _HerModelStandIn, her_source  # noqa: F401


def _a_draft_of_hers():
    return _HerModelStandIn([[
        ("edit_draft", {"path": "core/more_notes.py", "old_text": "", "new_text": "MORE = []\n"}),
        ("edit_draft", {"path": "tests/test_notes.py", "old_text": "", "new_text": NEW_TEST}),
        ("check_draft", {}),
        ("offer_draft", {"summary": "A first test."}),
    ]])


@pytest.fixture
def gates_asked(her_source, monkeypatch):
    asked: list[list[str]] = []
    failing: set[str] = set()

    def run(argv, *, cwd, env):
        asked.append(list(argv))
        name = "tests" if "pytest" in argv else argv[-1]
        return (name not in failing, f"{name} said this")

    monkeypatch.setattr(work, "_run_a_gate", run)
    return asked, failing


def test_her_yes_takes_a_draft_that_passes_every_gate_and_keeps_what_they_said(her_source, gates_asked):
    asked, _failing = gates_asked
    record = asyncio.run(work.work_a_while(GOAL, router=_a_draft_of_hers()))
    assert record.status == work.MERGED
    gates = work.load(record.slug).offer["gates"]
    assert list(gates) == ["tests", "smoke", "lint", "layering", "writing", "governance-lint", "deps-check"], \
        "in the order given"
    assert all(gate["passed"] for gate in gates.values())
    assert [argv[-1] for argv in asked[1:]] == ["smoke", "lint", "layering", "writing", "governance-lint",
                                                 "deps-check"]
    assert (her_source / "core" / "more_notes.py").exists()


def test_the_outside_check_is_asked_last_and_its_refusal_sends_the_draft_back(her_source, monkeypatch):
    """Replayed on 18 November 2026 it refused her overfit 052670e4a, which every other gate had passed."""
    (her_source / "outside_bar").mkdir()
    (her_source / "outside_bar" / "verify_change.py").write_text("")
    asked: list[list[str]] = []

    def run(argv, *, cwd, env):
        asked.append(list(argv))
        refused = any(part.endswith("verify_change.py") for part in argv)
        return (not refused, "did not hold: an input that worked gives another value" if refused else "fine")

    monkeypatch.setattr(work, "_run_a_gate", run)
    record = asyncio.run(work.work_a_while(GOAL, router=_a_draft_of_hers()))
    assert record.status == work.WORKING and not (her_source / "core" / "more_notes.py").exists()
    assert "outside-check failed" in record.said[-1]["words"] and "gives another value" in record.said[-1]["words"]
    assert any(part.endswith("verify_change.py") for part in asked[-1]), "after every other gate"


def test_a_gate_that_fails_sends_the_draft_back_to_her_with_its_words(her_source, gates_asked):
    asked, failing = gates_asked
    failing.add("smoke")
    record = asyncio.run(work.work_a_while(GOAL, router=_a_draft_of_hers()))
    assert record.status == work.WORKING
    assert not (her_source / "core" / "more_notes.py").exists(), "nothing went into the source she runs on"
    assert "smoke failed" in record.said[-1]["words"] and "smoke said this" in record.said[-1]["words"]
    assert list(work.load(record.slug).offer["gates"]) == ["tests", "smoke"], "the gates stop at the first that fails"
    assert all("lint" not in argv for argv in asked)


def test_the_tests_of_a_changed_module_are_found_where_they_name_it(her_source):
    (her_source / "tests").mkdir(exist_ok=True)
    (her_source / "tests" / "test_counting.py").write_text("from core.notes import note_count\n")
    (her_source / "tests" / "test_other.py").write_text("import json\n")
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", "tests", cwd=her_source)
    assert work.tests_that_touch(["core/notes.py"], her_source) == ["tests/test_counting.py"]
    assert work.tests_that_touch(["tests/test_new.py", "README.md"], her_source) == ["tests/test_new.py"]


def test_a_gate_runs_through_the_subprocess_gateway_in_a_governed_scope(tmp_path, monkeypatch):
    import core.runtime.subprocess_gateway as gateway
    from core.governance_context import get_active_governance

    seen = {}

    class Gateway:
        def run(self, argv, **kwargs):
            seen.update(kwargs, argv=list(argv), token=get_active_governance())
            return type("Done", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(gateway, "get_subprocess_gateway", lambda: Gateway())
    passed, tail = work._run_a_gate(["make", "lint"], cwd=tmp_path, env={})
    assert passed and tail == "ok"
    assert seen["source"] == "her_goal_at_work.gate" and seen["timeout"] == work.GATE_SECONDS
    assert seen["token"] is not None, "inside a governed scope, so a live runtime does not refuse it"
