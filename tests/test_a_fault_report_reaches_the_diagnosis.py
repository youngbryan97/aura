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


# ---------------------------- a cause is not in the request or in memory


def test_asking_why_something_behaves_is_a_shape_not_a_vocabulary() -> None:
    """A person describes trouble in whatever words the trouble suggests.

    LIVE, 2026-08-28: "Have a look through <path> — I keep getting a different
    total on the second run and I can't see why" was offered grounded_search,
    file_operation and code_repl. "A different total on the second run" is not
    vocabulary anybody can enumerate. The shape of the question is.
    """

    from core.intent.declared_capability import asks_why_something_behaves

    for asked in (
        "I can't see why",
        "why does it do that",
        "how come the second one differs",
        "what's going on here",
        "there's a bug and I can't find it",
        "I don't know why it changes",
        "what is it doing on the second run",
    ):
        assert asks_why_something_behaves(asked), asked

    for asked in (
        "read the file to me",
        "what is in that directory",
        "list everything under there",
        "tell me the total",
    ):
        assert not asks_why_something_behaves(asked), asked


def test_a_why_question_alone_reaches_nothing() -> None:
    """It has to be about something real, or it is a question for memory."""

    from core.capability_engine import CapabilityEngine
    from core.intent.capability_selection import select_capabilities
    from core.phases.response_contract import requested_effect_ceiling

    engine = CapabilityEngine()
    skills = getattr(engine, "skills", None) or {}
    for asked in ("why is the sky blue", "how come people yawn"):
        ceiling, scopes = requested_effect_ceiling(asked)
        chosen = select_capabilities(
            asked, skills, ceiling=ceiling, admissible_scopes=scopes, limit=5
        )
        assert "diagnose_repo" not in chosen, asked


def test_the_skill_joins_the_set_by_describing_itself() -> None:
    """No skill is named in the selection: it declares that it runs and reports."""

    from pathlib import Path

    body = Path("core/intent/declared_capability.py").read_text()
    start = body.index("def behaviour_capabilities(")
    window = body[start : start + 1400]
    assert "diagnose_repo" not in window
    assert "verb_class_of(\"run\")" in window


def test_a_fault_report_about_a_real_path_reaches_what_can_run_it() -> None:
    import os

    from core.capability_engine import CapabilityEngine
    from core.intent.capability_selection import select_capabilities
    from core.phases.response_contract import requested_effect_ceiling

    engine = CapabilityEngine()
    skills = getattr(engine, "skills", None) or {}
    here = os.getcwd()
    reached = 0
    phrasings = (
        f"Have a look through {here} — I keep getting a different total and I can't see why.",
        f"there's a bug somewhere in {here} and I can't find it",
        f"the second run from {here} comes out wrong and I don't know why",
        f"why does the code in {here} give the wrong answer the second time round",
    )
    for asked in phrasings:
        ceiling, scopes = requested_effect_ceiling(asked)
        chosen = select_capabilities(
            asked, skills, ceiling=ceiling, admissible_scopes=scopes, limit=5
        )
        reached += int("diagnose_repo" in chosen)
    assert reached >= 3, f"only {reached} of {len(phrasings)} phrasings reached it"
