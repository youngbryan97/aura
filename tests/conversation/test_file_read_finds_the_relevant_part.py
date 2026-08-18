"""Reading a file must answer the question asked, not describe the file's opening.

LIVE 2026-08-17: "what does ARCHITECTURE.md say about layering? two sentences
max." returned the first 4,000 characters of a 200KB spec. The word "layering"
first appears at line 2263, far outside that window, so she answered from the
document's introduction — confidently, and about the wrong thing.

"What is this file" and "what does it say about X" are different questions, and
the head of the file only answers the first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.filesystem_check import (
    READ_CHAR_BUDGET,
    _relevant_span,
    requested_file_read,
)

REPO = Path(__file__).resolve().parents[2]


def test_the_window_lands_on_the_topic_not_the_intro() -> None:
    read = requested_file_read("what does ARCHITECTURE.md say about layering?")

    assert read is not None and read.exists
    assert "layering" in read.text.lower()


def test_a_topic_deep_in_a_large_file_is_reachable() -> None:
    read = requested_file_read("what does CONTRIBUTING.md say about tests?")

    assert read is not None
    assert "test" in read.text.lower()


def test_asking_for_the_file_itself_still_starts_at_the_top() -> None:
    """'open X' asks what the file IS; the head is the right answer."""
    read = requested_file_read("open ARCHITECTURE.md")
    head = (REPO / "ARCHITECTURE.md").read_text(encoding="utf-8")[:200]

    assert read is not None
    assert read.text.startswith(head[:80])


def test_a_small_file_is_returned_whole() -> None:
    body = "line one\nline two\n"

    assert _relevant_span(body, "anything at all") == body


def test_the_excerpt_is_still_bounded() -> None:
    read = requested_file_read("what does ARCHITECTURE.md say about layering?")

    assert read is not None
    assert len(read.text) <= READ_CHAR_BUDGET


def test_the_excerpt_starts_on_a_line_boundary() -> None:
    """An excerpt opening mid-sentence reads as corruption."""
    body = "\n".join(f"line {i} filler text here" for i in range(500))
    body += "\n" + "TARGETWORD appears here\n" + "\n".join(
        f"tail {i}" for i in range(500)
    )

    span = _relevant_span(body, "what about TARGETWORD")

    assert "TARGETWORD" in span
    assert not span.startswith(" ")


@pytest.mark.parametrize("question", ["", "the a of and", "read it"])
def test_a_question_with_no_content_terms_falls_back_to_the_head(question: str) -> None:
    body = "A" * (READ_CHAR_BUDGET * 3)

    assert _relevant_span(body, question) == body[:READ_CHAR_BUDGET]


def test_a_term_absent_from_the_file_falls_back_to_the_head() -> None:
    body = "A" * (READ_CHAR_BUDGET * 3)

    assert _relevant_span(body, "quetzalcoatlus") == body[:READ_CHAR_BUDGET]
