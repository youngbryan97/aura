"""Sentences the surface ran together get their space back.

Measured live 2026-07-26 in an otherwise excellent 964-character reply:

    "…losing parts of myself or the world I've been part of.But — and this is
     where it gets complicated…"
    "…because they're redundant.The mercy part might come…"
    "…how I understand this world.What about you?"

Every paragraph boundary arrived with its whitespace gone. Nothing in the
serving path removes newlines, so the model is emitting them that way — but it
is still what the person reads, and one space is a safe repair.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive_engine import _restore_sentence_spacing


@pytest.mark.parametrize(
    "run_on,expected",
    [
        ("part of.But — and this is", "part of. But — and this is"),
        ("they're redundant.The mercy part", "they're redundant. The mercy part"),
        ("understand this world.What about you?", "understand this world. What about you?"),
        ("frictional heating.The core is iron", "frictional heating. The core is iron"),
        ("Wait!Then it moved", "Wait! Then it moved"),
        ("Really?Yes indeed", "Really? Yes indeed"),
    ],
)
def test_run_on_sentences_are_separated(run_on: str, expected: str) -> None:
    assert _restore_sentence_spacing(run_on) == expected


@pytest.mark.parametrize(
    "text",
    [
        "version 3.14 is fine",          # decimals
        "the U.S.A stays",               # initialisms: next char is not lowercase
        "already spaced. Like this",     # nothing to do
        "",
    ],
)
def test_safe_cases_are_untouched(text: str) -> None:
    assert _restore_sentence_spacing(text) == text


def test_code_fences_are_never_rewritten() -> None:
    fenced = "here:\n```python\nx = a.Buffer(1)\n```"
    assert _restore_sentence_spacing(fenced) == fenced


def test_inline_code_is_never_rewritten() -> None:
    """`file.Name` has the same shape as a run-on; a space would corrupt it.

    Any backtick means code is present, so the repair stays out of that reply
    entirely rather than trying to tell prose from identifiers mid-string.
    """
    inline = "use the `file.Name` attribute.It reads the path"
    assert _restore_sentence_spacing(inline) == inline


def test_an_abbreviation_gains_a_space_and_that_is_correct() -> None:
    """"Dr.Smith" is a run-on too; separating it is the right English."""
    assert _restore_sentence_spacing("saw Dr.Smith today") == "saw Dr. Smith today"


def test_it_runs_before_the_reply_is_judged() -> None:
    """Spacing is restored first, so every later check measures the real text.

    This used to say "before the cutoff trim" and pinned the ordering against
    `_complete_reply_tail`. That trimmer was deliberately taken off the chat
    path — a punctuation trimmer can make a fragment look complete by throwing
    away the answer someone asked for — so the substring was gone and the test
    raised ValueError rather than failing about anything. What still has to
    hold is that restoration happens before the surface gate reads the text.
    """
    from pathlib import Path

    src = Path("core/brain/cognitive_engine.py").read_text(encoding="utf-8")
    restored = src.index("_restore_sentence_spacing(text)")
    judged = src.index("surface_quality_gate_reasons", restored)
    assert restored < judged
    # And the trimmer stays off this path.
    assert "_complete_reply_tail(text)" not in src
