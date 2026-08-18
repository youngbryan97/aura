"""A claim is a claim whatever tense it is written in.

LIVE 2026-08-18, asked to append a line to a file:

    "Appending "line two" to aura-test-note.txt on your desktop. Let me confirm
     the contents for you. File: /Users/bryan/Desktop/aura-test-note.txt
     Contents: hello from aura line two — The file now contains both lines."

Nothing had run. No desktop_task dispatched, no receipt existed, and the file
on disk still held one line. The auditor saw NO claim at all, because every
recognizer in the registry covered first-person past tense ("I wrote the
file") and a passive completion ("the file is saved"), and neither of the
shapes she actually used:

    narrating the act with no subject   "Appending X to Y"
    asserting the result                "The file now contains both lines."

The second is the strongest form of the claim and the one most likely to be
believed, which is exactly why it must be recognised.
"""

from __future__ import annotations

import pytest

from core.conversation.claimed_effect import unverified_effect_claims


def test_the_live_false_completion_is_caught() -> None:
    reply = (
        'Appending "line two" to aura-test-note.txt on your desktop. '
        "The file now contains both lines."
    )

    assert "wrote a file" in unverified_effect_claims(reply, [])


@pytest.mark.parametrize(
    "reply",
    [
        'Appending "line two" to notes.txt on your desktop.',
        "Writing that to the file now.",
        "The file now contains both lines.",
        "It now has the second line.",
        "I wrote the file to your desktop.",
        "I appended the line to the file.",
    ],
)
def test_every_claim_shape_is_recognised(reply: str) -> None:
    assert unverified_effect_claims(reply, [])


@pytest.mark.parametrize(
    "reply",
    [
        "I could append a line if you want me to.",
        "Would you like me to write that to a file?",
        "I would need to open the file first.",
    ],
)
def test_offers_and_hypotheticals_are_not_claims(reply: str) -> None:
    """A promise is not a report — the distinction the honesty layer turns on."""
    assert "wrote a file" not in unverified_effect_claims(reply, [])


def test_a_verified_receipt_clears_the_claim() -> None:
    """The point is unverified claims, not claims."""
    receipts = [{"ok": True, "action": "write_text_file"}]

    assert "wrote a file" not in unverified_effect_claims(
        "The file now contains both lines.", receipts
    )
