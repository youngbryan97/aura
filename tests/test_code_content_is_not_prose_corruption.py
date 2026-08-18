"""Prose validators must not judge the inside of a code fence.

Three detectors, each written for a real prose defect, were rejecting correct
code answers. Live 2026-08-18 a request for a function plus an explanation
took 112 seconds and returned "I couldn't get to an answer I'd stand behind on
that one".

  * consonant runs mean corrupted language — but a base64 blob or a hex digest
    is a legitimate run of consonants, and inside a fence there is no language
    to corrupt;
  * CJK characters in a reply mean the model drifted into another script —
    unless the question was ABOUT that script. The exemption looked for CJK in
    the user's message, so "write a function to detect chinese characters",
    which carries none, never qualified, and the sample string the answer had
    to show was read as the drift.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import _has_unexpected_cjk_intrusion
from core.synthesis import _locally_corrupted_language


@pytest.mark.parametrize(
    "reply",
    [
        "Use:\n\n```python\nKEY = 'bXlzdHJpbmdzdHJpbmc='\n```\n",
        "```python\nDIGEST = 'a3f9bcdd8e7f4b2c9d'\n```",
        "```python\ndef rev(s):\n    return s[::-1]\n```",
    ],
)
def test_code_is_not_read_as_corrupted_language(reply: str) -> None:
    assert not _locally_corrupted_language(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "The answerrrrr is bcdfghjklmnp.",
        # Corruption in the prose still counts when code is present.
        "The answerrrrr is here:\n\n```python\nx = 1\n```",
    ],
)
def test_corrupted_prose_is_still_caught(reply: str) -> None:
    assert _locally_corrupted_language(reply)


def test_a_sample_string_answers_a_question_about_that_script() -> None:
    assert not _has_unexpected_cjk_intrusion(
        "write a python function to detect chinese characters",
        "Here:\n\n```python\nSAMPLE = '你好世界'\n```\n\nThat matches CJK.",
    )


def test_naming_the_script_in_english_counts_as_asking() -> None:
    assert not _has_unexpected_cjk_intrusion(
        "how do I say hello in chinese?", "You say 你好."
    )


def test_a_code_sample_alone_is_not_drift() -> None:
    assert not _has_unexpected_cjk_intrusion(
        "write a function to reverse a string", "Here:\n\n```python\nx = '你好'\n```\n"
    )


def test_real_script_drift_in_prose_is_still_caught() -> None:
    assert _has_unexpected_cjk_intrusion(
        "write a function to reverse a string",
        "The function reverses it 你好世界 completely.",
    )
