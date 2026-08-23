"""A requirement is read from the person, not from the model's restatement.

LIVE, 2026-08-22. Asked for six slides, the document builder received
request="present system funders" — the model's own paraphrase of the turn — so
the count reader found nothing, the check had nothing to enforce, and a
three-section deck was built and reported as finished.

The distinction is not about documents. A skill's arguments are filled in by
the model, so anything read from them is read from a restatement. Values the
model chose are properly its own: which path to run, which URL to fetch. A
REQUIREMENT — how many, what shape, what must be covered — belongs to whoever
asked, and is read from what they said.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.session_scope import set_user_question, the_persons_own_words


@pytest.fixture(autouse=True)
def clear_turn():
    set_user_question("")
    yield
    set_user_question("")


def test_the_persons_words_win_over_the_paraphrase():
    set_user_question("Six slides, no fluff: what you are.")
    assert the_persons_own_words("present system funders").startswith("Six slides")


def test_the_paraphrase_stands_when_there_is_no_turn():
    """Outside a turn — a background build, a test — the argument is all there
    is, and it is used rather than nothing."""
    set_user_question("")
    assert the_persons_own_words("present system funders") == "present system funders"


def test_nothing_at_all_is_empty():
    set_user_question("")
    assert the_persons_own_words("") == ""


@pytest.mark.parametrize(
    "module",
    ["core/skills/build_document.py", "core/skills/build_app.py"],
)
def test_the_builders_read_the_requirement_from_the_person(module: str):
    source = Path(module).read_text(encoding="utf-8")
    assert "the_persons_own_words(" in source, module


def test_a_count_survives_the_paraphrase():
    """The whole point, end to end: the count is found because the person's
    words are read."""
    from core.skills.build_document import _sections_asked_for

    set_user_question("Six slides, no fluff: what you are.")
    assert _sections_asked_for(the_persons_own_words("present system funders")) == 6
