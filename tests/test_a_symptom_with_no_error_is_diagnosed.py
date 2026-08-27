"""A project with no tests, no traceback, and a wrong answer.

LIVE, 2026-08-22. Handed a small project whose second invoice came out holding
the first one's lines, `diagnose_repository` said "no test runner was found in
this project" and stopped. It knew one experiment.

These hold the three independent kinds of evidence a diagnosis now gathers, each
computed rather than proposed: what running the project produced, what the
source says survives a call, and what the project's own README claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.diagnosis.carried_state import carried_state, describe_carried_state
from core.diagnosis.experiment import affordances, observe
from core.diagnosis.repository import describe_diagnosis, diagnose_repository


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text(
        "# invoice-tools\n\n"
        "`add_line(item, price, lines=[])` appends a line and returns the invoice so far.\n"
        "Every call starts a fresh invoice unless you pass one in.\n"
    )
    (tmp_path / "invoice.py").write_text(
        '"""Helpers for assembling an invoice."""\n\n\n'
        "def add_line(item, price, lines=[]):\n"
        '    """Append one line item and return the invoice so far."""\n'
        '    lines.append({"item": item, "price": price})\n'
        "    return lines\n\n\n"
        "def total(lines):\n"
        '    """Sum an invoice to the cent."""\n'
        '    return round(sum(line["price"] for line in lines), 2)\n'
    )
    (tmp_path / "run.py").write_text(
        "from invoice import add_line, total\n\n"
        'first = add_line("consulting", 100.0)\n'
        'first = add_line("hosting", 25.0, first)\n'
        'print("invoice one:", total(first))\n\n'
        'second = add_line("consulting", 100.0)\n'
        'print("invoice two:", total(second))\n'
    )
    return tmp_path


def test_a_script_is_a_way_into_a_project_with_no_tests(project: Path) -> None:
    """run.py is an entry point because nothing imports it and loading it works."""
    ways = affordances(project)
    assert [way.kind for way in ways] == ["entry point"]
    assert ways[0].described == "python run.py"
    # invoice.py is imported by run.py, so it is a library, not a front door.
    assert "invoice.py" not in ways[0].argv


def test_it_runs_the_project_and_keeps_what_came_back(project: Path) -> None:
    seen = observe(project, affordances(project)[0])
    assert seen.exit_code == 0
    assert "invoice one: 125.0" in seen.output
    assert "invoice two: 225.0" in seen.output


def test_the_mutable_default_is_found_by_reading_the_source(project: Path) -> None:
    found = carried_state(project)
    assert len(found) == 1
    assert found[0].function == "add_line"
    assert found[0].name == "lines"
    assert found[0].kind == "default argument"
    assert "changed by the body" in found[0].detail
    assert "invoice.py:4" in describe_carried_state(found)


def test_a_module_level_list_a_function_appends_to_is_found(tmp_path: Path) -> None:
    """The same defect wearing its other shape."""
    (tmp_path / "log.py").write_text(
        "SEEN = []\n\n\n"
        "def note(event):\n"
        "    SEEN.append(event)\n"
        "    return len(SEEN)\n"
    )
    found = carried_state(tmp_path)
    assert [item.name for item in found] == ["SEEN"]
    assert found[0].kind == "module-level value"


def test_a_function_that_carries_nothing_is_not_reported(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text(
        "def add_line(item, price, lines=None):\n"
        "    lines = list(lines or [])\n"
        "    lines.append((item, price))\n"
        "    return lines\n"
    )
    assert carried_state(tmp_path) == ()


def test_the_whole_diagnosis_names_the_cause_and_what_contradicts_it(project: Path) -> None:
    diagnosis = diagnose_repository(project)
    assert not diagnosis.error
    told = describe_diagnosis(diagnosis)
    # It ran the project.
    assert "invoice two: 225.0" in told
    # It read the cause out of the source.
    assert "defaults to a list built once" in told
    # It quoted the line, marked.
    assert "> def add_line(item, price, lines=[])" in told
    # And it found the project's own claim that this breaks.
    assert "starts a fresh invoice" in diagnosis.stated_intent
    # Three independent things agree, so this is not one reading.
    assert diagnosis.evidence_count() >= 2


def test_the_finding_is_filed_as_something_that_can_be_wrong(project: Path) -> None:
    """A diagnosis outlives the turn, and the belief moves with the outcome."""
    from core.cognition.scientific_engine import get_scientific_engine
    from core.diagnosis.repository import confirm_diagnosis

    diagnosis = diagnose_repository(project)
    assert diagnosis.hypothesis_id
    engine = get_scientific_engine()
    before = engine.belief(diagnosis.hypothesis_id)
    assert before is not None
    confirm_diagnosis(diagnosis.hypothesis_id, worked=True)
    after = engine.belief(diagnosis.hypothesis_id)
    assert after is not None and after >= before


def test_her_explanation_leads_the_evidence_that_supports_it() -> None:
    """The answer, then the working — not the working, then the answer.

    LIVE, 2026-08-27: the finding was complete and correct, and fifteen lines
    of it sat above "Found it — classic mutable default argument, and exactly
    why nothing ever raises". Somebody who asked what the cause is and what to
    change reads the answer first.
    """
    from interface.routes.chat import _explains_the_finding

    found = (
        "I ran python run.py and it printed: invoice one: 125.0. add_line was called "
        "twice with the same arguments and answered differently (invoice.py:4). "
        "add_line(lines=...) defaults to a list built once."
    )
    explanation = (
        "Found it — a classic mutable default argument, and exactly why nothing ever "
        "raises. The cause is invoice.py:4: add_line(item, price, lines=[]) evaluates "
        "that default once, when the function is defined, so every call that omits "
        "lines appends into the same list object and invoice one's rows are still "
        "there when invoice two starts."
    )
    assert _explains_the_finding(explanation, found) is True

    # A remark is not an account of anything, and leading with it buries the
    # finding instead.
    assert _explains_the_finding("Yeah, that looks right to me.", found) is False
    assert (
        _explains_the_finding(
            "I had a look and something odd is going on with the totals here, though "
            "I would want to check a few more things before saying anything definite.",
            found,
        )
        is False
    )
