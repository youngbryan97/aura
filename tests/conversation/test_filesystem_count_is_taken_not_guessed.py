"""A count of files on this disk is not a thing to estimate.

LIVE 2026-08-17: "count the .py files in core/introspection and tell me the
number" was answered "There are 3.py files in the core/introspection
directory." There are ten. Asked again with "use your tools and give me the
exact number", the answer was still 3 — and the log for that turn shows no tool
ran at all. The number came out of the model both times.

The runtime can take the count exactly, in microseconds, so a generated guess
was competing with a fact that was available the whole time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.filesystem_check import requested_filesystem_count

REPO = Path(__file__).resolve().parents[2]


def test_the_live_question_returns_the_real_count() -> None:
    truth = len(list((REPO / "core" / "introspection").glob("*.py")))
    result = requested_filesystem_count(
        "count the .py files in core/introspection and tell me the number"
    )

    assert result is not None
    assert result.count == truth
    assert result.count != 3, "the fabricated answer must not be reproducible"


@pytest.mark.parametrize(
    "question",
    [
        "how many python files are in the core/introspection folder?",
        "how many .py files are in core/introspection",
        "number of python files in core/introspection",
    ],
)
def test_phrasings_all_reach_the_same_count(question: str) -> None:
    truth = len(list((REPO / "core" / "introspection").glob("*.py")))

    result = requested_filesystem_count(question)

    assert result is not None and result.count == truth


def test_counting_every_file_is_not_the_same_as_counting_python_files() -> None:
    everything = requested_filesystem_count("how many files are in core/introspection")
    python_only = requested_filesystem_count(
        "how many python files are in core/introspection"
    )

    assert everything is not None and python_only is not None
    assert everything.count >= python_only.count


# ── it must not answer questions about the rest of the machine ──────────────

@pytest.mark.parametrize(
    "question",
    [
        "how many files are in /etc",
        "how many files are in ../../../etc",
        "how many files are in ~/Documents",
    ],
)
def test_paths_outside_her_roots_are_declined(question: str) -> None:
    """pathlib treats `root / '/etc'` as `/etc`, which silently escapes."""
    assert requested_filesystem_count(question) is None


def test_a_missing_directory_is_named_not_counted_as_zero() -> None:
    """'There are 0 files' and 'that is not there' are different answers."""
    result = requested_filesystem_count("how many files are in core/nope_not_real")

    assert result is not None
    assert result.exists is False


def test_an_unrecognised_qualifier_is_declined() -> None:
    """'how many test files' must not silently become 'how many files'."""
    assert requested_filesystem_count("how many test files are in core") is None


@pytest.mark.parametrize(
    "question",
    ["what is the weather", "hey", "", "how are you doing"],
)
def test_unrelated_questions_are_ignored(question: str) -> None:
    assert requested_filesystem_count(question) is None
