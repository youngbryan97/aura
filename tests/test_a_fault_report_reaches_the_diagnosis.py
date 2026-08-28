"""A skill is reachable by the ways its job is asked for, not only its name.

LIVE, 2026-08-28: the request that ends "Go through the code and tell me what's
actually happening, with the file and line" was offered code_repl and
file_operation. diagnose_repo — which runs the project, finds the mutable
default at invoice.py:4 and computes the remedy — was never offered, because
its declaration began "Use for any question about why code in a directory is
FAILING" and that request opens by saying nothing fails.
"""

from __future__ import annotations

import pytest

from core.intent.declared_capability import verb_class_of

_HANDING_OVER_A_FAULT = (
    "debug",
    "investigate",
    "troubleshoot",
    "examine",
    "review",
    "trace",
    "assess",
    "explain",
)


@pytest.mark.parametrize("word", _HANDING_OVER_A_FAULT)
def test_the_words_a_person_uses_name_the_act(word: str) -> None:
    """The class had only the words an engineer writes in a description."""

    assert verb_class_of(word), word


@pytest.mark.parametrize("word", _HANDING_OVER_A_FAULT)
def test_they_land_in_the_class_that_already_holds_diagnose(word: str) -> None:
    assert "diagnose" in verb_class_of(word), word


def test_the_declaration_covers_a_project_that_is_not_failing() -> None:
    from core.skills.diagnose_repo import DiagnoseRepoSkill

    said = DiagnoseRepoSkill.description.lower()
    # The engine runs the project when the suite passes and reports what
    # outlives a call; the description has to say so or nothing routes to it.
    assert "no error" in said
    assert "tests pass" in said
    assert "survives a call" in said
    for symptom in ("misbehaves", "bug", "wrong answer"):
        assert symptom in said, symptom


def test_the_engine_finds_the_fault_with_no_failing_test(tmp_path) -> None:
    """The claim in the declaration, checked against the engine."""

    from core.diagnosis.repository import describe_diagnosis, diagnose_repository

    project = tmp_path / "invoice-tools"
    project.mkdir()
    (project / "invoice.py").write_text(
        '"""Helpers for assembling an invoice."""\n'
        "\n"
        "\n"
        "def add_line(item, price, lines=[]):\n"
        '    """Append one line item and return the invoice so far."""\n'
        '    lines.append({"item": item, "price": price})\n'
        "    return lines\n"
    )
    (project / "run.py").write_text(
        "from invoice import add_line\n"
        "\n"
        'print("one:", add_line("a", 1.0))\n'
        'print("two:", add_line("b", 2.0))\n'
    )
    described = describe_diagnosis(diagnose_repository(str(project)))
    assert "invoice.py:4" in described
    assert "lines=none" in described.lower() or "lines=None" in described
