""""How many test files do you have" is a fair question with one answer.

LIVE, 2026-08-18. Asked for two counts and told not to guess either, the
filesystem checker resolved "how many python files are in core/agency" exactly
and returned nothing at all for "how many test files do you have" — so half
the question fell back to the model, against a real answer of 2444.

The counting pattern needs a preposition and a path ("... in core/agency"),
and this shape has neither. It also names a KIND the suffix table does not
know, which is refused on purpose: an unrecognised qualifier would silently
become "all files" and answer a different question.

Both are right in general and wrong here, because the place is implied by the
kind. Her tests live in tests/ and her docs in docs/. Naming that is what
turns an ambiguous question into a determinate one, rather than widening the
qualifier rule and answering something else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.filesystem_check import requested_filesystem_count

ROOT = Path(__file__).resolve().parents[1]


def _actual(directory: str, suffix: str) -> int:
    return len([p for p in (ROOT / directory).iterdir() if p.is_file() and p.name.endswith(suffix)])


@pytest.mark.parametrize(
    "question",
    [
        "how many test files do you have",
        "how many tests do you have?",
        "count the test files",
        "number of test files",
    ],
)
def test_her_tests_are_countable_however_it_is_asked(question):
    counted = requested_filesystem_count(question)
    assert counted is not None, question
    assert counted.exists is True
    assert counted.suffix == ".py"
    assert counted.count == _actual("tests", ".py")


@pytest.mark.parametrize(
    "kind,directory",
    [
        ("docs", "docs"),
        ("benchmarks", "benchmarks"),
        ("demos", "demos"),
        ("specs", "specs"),
    ],
)
def test_any_kind_the_repository_actually_has_is_countable(kind, directory):
    """Derived from the tree, so it is not one word behind every question.

    The first version of this carried a table mapping "test" to tests/ and
    "doc" to docs/. It answered exactly the two questions someone had thought
    of, and it was WRONG about a third: it mapped "spec" to tests/.py, while
    the repository has a specs/ directory full of .md. Looking the directory up
    instead of asserting it fixed the answer and removed the table.
    """
    counted = requested_filesystem_count(f"how many {kind} do you have")
    assert counted is not None, kind
    assert counted.exists is True
    assert Path(counted.path).name == directory


def test_her_docs_are_countable_too():
    counted = requested_filesystem_count("how many docs do you have")
    assert counted is not None
    assert counted.suffix == ".md"
    assert counted.count == _actual("docs", ".md")


def test_a_named_path_still_wins():
    """The explicit form must be untouched by the implied-place shortcut."""
    counted = requested_filesystem_count("how many python files are in core/agency")
    assert counted is not None
    assert counted.path.endswith("core/agency")
    assert counted.count == _actual("core/agency", ".py")


def test_a_word_that_merely_starts_with_test_is_not_a_test_file():
    """"testimonials" must not be read as "test"."""
    assert requested_filesystem_count("how many testimonials do you have") is None


def test_an_unrecognised_qualifier_is_still_refused():
    """The deliberate refusal that protects against answering a different
    question has to survive: only kinds with a known home are answered."""
    assert requested_filesystem_count("how many config files are in core") is None


def test_the_counts_match_the_directory():
    """Counted again outside this module's own logic.

    This used `ls | wc -l` through a shell. The independence is in not reusing
    the module's counting code; the shell added nothing but a `shell=True` the
    repository's own security gate refuses.
    """
    counted = requested_filesystem_count("how many test files do you have")
    on_disk = len(list((Path(ROOT) / "tests").glob("*.py")))
    assert counted is not None
    assert counted.count == on_disk


# ── a question can ask for more than one count ───────────────────────────


def test_two_counts_in_one_question_both_come_back():
    """Asking for two is the natural way to ask for two.

    LIVE, 2026-08-18: "how many test files do you have, and how many python
    files are in core/agency?" was answered "54 .py files" — the second
    number, exactly right, with the first silently dropped. Half an answer
    reads as a whole one, which is worse than saying a part is unavailable.
    """
    from core.conversation.filesystem_check import requested_filesystem_counts

    counts = requested_filesystem_counts(
        "two numbers, no guessing: how many test files do you have, and how "
        "many python files are in core/agency?"
    )
    assert len(counts) == 2, [c.path for c in counts]
    by_name = {Path(c.path).name: c for c in counts}
    assert by_name["tests"].count == _actual("tests", ".py")
    assert by_name["agency"].count == _actual("core/agency", ".py")


def test_the_reply_carries_both_numbers():
    import interface.routes.chat as chat

    served = chat._serve_measured_filesystem_count(
        "how many test files do you have, and how many python files are in core/agency?",
        "I think it's about 40 and 900.",
    )
    assert str(_actual("tests", ".py")) in served
    assert str(_actual("core/agency", ".py")) in served


def test_a_single_count_still_lists_the_directory():
    """The one-count answer showed its work; that must not be lost."""
    import interface.routes.chat as chat

    served = chat._serve_measured_filesystem_count(
        "how many python files are in core/agency", "roughly 40"
    )
    assert "listed the directory" in served
    assert "agency_core.py" in served


def test_a_reply_that_is_already_right_is_left_alone():
    import interface.routes.chat as chat

    mine = f"There are {_actual('core/agency', '.py')} python files in core/agency."
    assert chat._serve_measured_filesystem_count(
        "how many python files are in core/agency", mine
    ) == mine


def test_a_non_count_question_is_untouched():
    import interface.routes.chat as chat

    assert chat._serve_measured_filesystem_count("what's the weather", "Sunny.") == "Sunny."


def test_a_missing_directory_is_still_reported_when_asked_alongside_others():
    """`all()` over an empty sequence is True.

    Testing only the EXISTING counts meant a question whose only target was
    missing short-circuited to "she already has it right" and the report was
    never reached.
    """
    import interface.routes.chat as chat

    served = chat._serve_measured_filesystem_count(
        "how many files are in core/definitely_not_here", "There are 4 files."
    )
    assert "no directory" in str(served).lower()


def test_an_explicit_place_beats_an_implied_one():
    """"how many config files are in core" must not become a count of config/.

    The implied-place shortcut is only for a clause that named no place. A
    named place this cannot answer for is a deliberate refusal, and the
    shortcut must not talk over it.
    """
    from core.conversation.filesystem_check import requested_filesystem_counts

    assert requested_filesystem_counts("how many config files are in core") == []
    assert requested_filesystem_counts("how many files are in /etc") == []


def test_two_implied_places_in_one_question_both_come_back():
    """"how many test files do you have, and how many docs?" is two counts.

    Neither clause names a path, and the implied-place lookup ran once, so the
    docs count was silently dropped — measured live 2026-08-18.
    """
    from core.conversation.filesystem_check import requested_filesystem_counts

    counts = requested_filesystem_counts(
        "how many test files do you have, and how many docs? just the numbers."
    )
    names = {Path(c.path).name for c in counts}
    assert {"tests", "docs"} <= names, names


def test_just_the_numbers_suppresses_the_file_listing():
    """An instruction about the answer's shape is part of the question.

    Asked for "just the numbers", the reply opened with a dozen test filenames.
    """
    import interface.routes.chat as chat

    served = str(
        chat._serve_measured_filesystem_count(
            "how many tests do you have? just the number.", "about 40"
        )
    )
    assert "listed the directory" not in served
    assert "conftest.py" not in served
    assert str(_actual("tests", ".py")) in served


def test_an_ordinary_count_still_shows_its_working():
    import interface.routes.chat as chat

    served = str(
        chat._serve_measured_filesystem_count(
            "how many python files are in core/agency", "about 40"
        )
    )
    assert "listed the directory" in served
