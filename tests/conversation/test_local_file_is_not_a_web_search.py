"""A file on this disk is not a live-search question.

LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me the first rule it
states" resolved to a contract with requires_search=True and was dispatched to
web_search, which failed with an empty error string:

    Required desktop search evidence failed before CognitiveEngine reply:
    query=read the file CONTRIBUTING.md result={'status': 'failed', 'error': ''}

The turn ended "I attempted to read the file and it failed... Would you like me
to check if the file exists?" The file was in the repo root the whole time, and
no search result could ever have answered the question.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _should_collect_desktop_required_search_evidence as gate


def test_reading_a_real_local_file_does_not_require_web_search() -> None:
    requires, _query, _contract = gate(
        "read the file CONTRIBUTING.md and tell me the first rule it states"
    )

    assert requires is False


@pytest.mark.parametrize(
    "question",
    [
        "what does CONTRIBUTING.md say about tests?",
        "open core/config.py",
        "check ARCHITECTURE.md for the layering rule",
    ],
)
def test_other_phrasings_of_a_local_read_also_skip_search(question: str) -> None:
    requires, _query, _contract = gate(question)

    assert requires is False


@pytest.mark.parametrize(
    "question",
    [
        "search the web for the latest MLX release notes",
        "look up who won the 2024 Nobel prize in physics",
    ],
)
def test_a_genuine_live_question_still_requires_search(question: str) -> None:
    """The exemption must not cost her live search."""
    requires, _query, _contract = gate(question)

    assert requires is True


def test_a_filename_that_does_not_resolve_does_not_suppress_search() -> None:
    """Only a file that actually exists settles it."""
    requires, _query, _contract = gate(
        "search the web for release notes about nonexistent_thing_xyz.md"
    )

    assert requires is True


def test_an_empty_message_is_handled() -> None:
    requires, _query, _contract = gate("")

    assert requires is False
