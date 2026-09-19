"""She counted, and did not say which one was largest.

LIVE 2026-09-18, asked "How many Python files are under core/consciousness
in your source tree, and which one is the largest by bytes?", the reply was

    171 .py files. I listed the directory rather than estimating:
    __init__.py, absorbed_voices.py, ... and 159 more.

The count is right and measured from disk. The second half is missing, and
a list of names is not an answer to it.

The sizes had been in the reading since 2026-09-16, added precisely because
that half had been answered by a guess — conscious_core.py, against
phi_core.py at 125,814 bytes. Nothing ever read them. A reading that
carries the answer and a sentence that does not use it is the same defect
as never having taken the reading.
"""

from __future__ import annotations

import pytest

from core.conversation.filesystem_check import (
    FilesystemCount,
    superlative_asked,
    the_extreme_file,
)


def _counted(**over) -> FilesystemCount:
    base = dict(
        path="/repo/core/consciousness",
        suffix=".py",
        count=3,
        exists=True,
        names=("small.py", "middle.py", "huge.py"),
        sizes=(("huge.py", 125814), ("middle.py", 4096), ("small.py", 12)),
    )
    base.update(over)
    return FilesystemCount(**base)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("which one is the largest by bytes?", "largest"),
        ("which is biggest", "largest"),
        ("what is the longest file", "largest"),
        ("which one is smallest", "smallest"),
        ("the tiniest one", "smallest"),
        ("how many python files are there", ""),
        ("list the files", ""),
    ],
)
def test_the_question_word_is_the_question(question, expected):
    assert superlative_asked(question) == expected


def test_the_extreme_comes_from_the_reading_not_a_guess():
    counted = _counted()
    assert the_extreme_file(counted, "largest") == ("huge.py", 125814)
    assert the_extreme_file(counted, "smallest") == ("small.py", 12)


def test_a_reading_with_no_sizes_offers_no_extreme():
    counted = _counted(sizes=())
    assert the_extreme_file(counted, "largest") is None


def test_an_unasked_superlative_is_not_answered():
    counted = _counted()
    assert the_extreme_file(counted, "") is None


def test_the_served_answer_carries_both_halves():
    from interface.routes.chat_served_answers import (
        _serve_measured_filesystem_count,
    )

    question = (
        "How many Python files are under core/consciousness in your source "
        "tree, and which one is the largest by bytes?"
    )
    served = str(_serve_measured_filesystem_count(question, "about 40 files, I think."))

    assert "files" in served
    assert "largest is" in served
    assert "bytes" in served


def test_a_right_count_beside_a_wrong_largest_is_still_corrected():
    """Leaving her wording alone was applied to the whole question."""

    from interface.routes.chat_served_answers import (
        _serve_measured_filesystem_count,
    )

    question = (
        "How many Python files are under core/consciousness, and which is "
        "the largest by bytes?"
    )
    # The count is right; the largest named is not the largest on disk.
    already = "171 .py files, and the largest is conscious_core.py."
    served = str(_serve_measured_filesystem_count(question, already))

    assert served != already, "a right count kept a wrong superlative"
    assert "phi_core.py" in served


def test_a_plain_count_question_is_left_alone_when_it_is_right():
    from interface.routes.chat_served_answers import (
        _serve_measured_filesystem_count,
    )

    from core.conversation.filesystem_check import requested_filesystem_counts

    question = "How many Python files are under core/consciousness?"
    # Read the count rather than writing one down: this runs in whatever
    # checkout it is run in, and a worktree is not the primary tree.
    counted = requested_filesystem_counts(question)
    assert counted and counted[0].exists
    already = f"There are {counted[0].count} .py files under core/consciousness."
    assert str(_serve_measured_filesystem_count(question, already)) == already
