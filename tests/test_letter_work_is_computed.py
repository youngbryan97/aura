"""Letter-level work is computed, because that is where models are worst.

LIVE 2026-08-19: "spell 'necessary' backwards" returned the canned refusal.
The model had produced a nine-character answer — the right length for
yrassecen — and the turn was killed downstream before it reached anyone.

Reversing a word, counting the r's in "strawberry", checking a palindrome: a
language model is unreliable at all of them and `[::-1]` is exact. Asking the
model and then policing the result is the wrong shape.

Every expected value below is recomputed in the test rather than copied from
the module, so agreement means both agree with Python.
"""

from __future__ import annotations

import pytest

from core.conversation.computable_text import (
    computed_text_answer,
    text_form_failures,
)


def test_every_form_answers_the_questions_it_claims() -> None:
    assert text_form_failures() == []


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("spell 'necessary' backwards", "necessary"[::-1]),
        ("reverse the word stressed", "stressed"[::-1]),
        ("how many r's in strawberry", str("strawberry".count("r"))),
        ("how many s in mississippi", str("mississippi".count("s"))),
        ("how many letters in necessary", str(len("necessary"))),
        ("sort the letters of aura", "".join(sorted("aura"))),
        ("uppercase the word aura", "aura".upper()),
        ("is racecar a palindrome?", "yes"),
        ("is necessary a palindrome?", "no"),
    ],
)
def test_the_answer_matches_python(question: str, expected: str) -> None:
    assert computed_text_answer(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        # An idiom, not a string operation.
        "reverse the polarity of the flow",
        "what is 2 + 2",
        "tell me a joke",
        "how many people are in the room",
    ],
)
def test_an_ordinary_turn_is_not_claimed(question: str) -> None:
    assert computed_text_answer(question) is None


def test_the_reading_reaches_the_grounding_channel() -> None:
    import asyncio

    import core.brain.observable_registry  # noqa: F401
    from core.brain.observable_grounding import observable_blocks

    blocks = asyncio.run(observable_blocks("spell 'necessary' backwards"))
    text = "\n".join(blocks) if isinstance(blocks, list) else str(blocks)

    assert "yrassecen" in text
