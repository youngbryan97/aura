"""The reading carried the sizes; the gate never learned the question.

Every file's size, largest first, has been in the reading since
2026-09-16 — added for one reason, written in the field's own comment:
"a listing of names answers 'how many'; 'which is the largest' was
answered by a guess (2026-09-16: conscious_core.py, against phi_core.py at
125,814 bytes) because nothing in the reading carried a size."

The gate that decides whether to read at all was keyed to counting words.
Measured live 2026-09-18 against the running instance:

    "How many Python files are under core/consciousness ... and which one
     is the largest by bytes?"   -> read the directory, answered 171
                                    correctly, and never named the largest

    "Which single file under core/consciousness is the largest by bytes?"
                                 -> matched nothing, no reading attached,
                                    three drafts discarded, canned refusal

Half the fix had landed: the column was added for a question the matcher
could not see. The refusal was right — she declined to invent a filename.
The silence was the defect.
"""

from __future__ import annotations

import asyncio

import pytest

from core.conversation.filesystem_check import requested_filesystem_count

ASKED_LIVE = (
    "Which single file under core/consciousness is the largest by bytes? "
    "Name it and give the byte count."
)


@pytest.mark.parametrize(
    "question",
    [
        ASKED_LIVE,
        "which python file in core/consciousness is the biggest",
        "what is the largest file in core/consciousness",
        "the biggest module under core/brain",
        "what's the longest script in core/runtime",
        "which single file under core/brain is the largest",
    ],
)
def test_a_superlative_question_gets_the_reading(question):
    counted = requested_filesystem_count(question)
    assert counted is not None, f"no reading attached to {question!r}"
    assert counted.sizes, "the reading carried no sizes"


def test_the_reading_is_ordered_largest_first():
    counted = requested_filesystem_count(ASKED_LIVE)
    sizes = [size for _name, size in counted.sizes]
    assert sizes == sorted(sizes, reverse=True)


def test_the_counting_question_still_reads():
    """The order this already handled must keep working."""

    counted = requested_filesystem_count(
        "How many Python files are under core/consciousness, and which one "
        "is the largest by bytes?"
    )
    assert counted is not None
    assert counted.count > 0
    assert counted.sizes


@pytest.mark.parametrize(
    "question",
    [
        "what is the weather in london",
        "which file should I read in core/consciousness",
        "tell me about the largest planet",
    ],
)
def test_ordinary_questions_get_no_reading(question):
    assert requested_filesystem_count(question) is None


def test_the_observable_names_the_largest():
    from core.brain.observable_registry import _matches_count, _read_count

    assert _matches_count(ASKED_LIVE)
    block = asyncio.run(_read_count(ASKED_LIVE))
    assert "largest first" in block
    assert "phi_core.py" in block
    # The byte count she was asked for, formatted as a reader would want it.
    assert "125,814 bytes" in block
