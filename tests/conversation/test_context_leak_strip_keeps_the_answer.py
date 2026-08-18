"""Stripping a protocol leak must not destroy the answer around it.

LIVE 2026-08-17: "what's on my screen right now?" was served as a bare "…".
The cortex had produced a 172-character answer and the screen reading had
reached it. The reply opened with a context marker, so the stripper cut from
index 0, and the caller's `or "…"` turned an authored answer into an ellipsis.

An ellipsis is not an answer. It is the shape of one.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _strip_user_visible_context_leaks as strip


def test_a_leading_marker_no_longer_empties_the_reply() -> None:
    reply = "[CURRENT USER MESSAGE]\nAura is in front, showing the chat window."

    kept = strip(reply)

    assert "Aura is in front" in kept
    assert "[CURRENT USER MESSAGE]" not in kept


def test_a_trailing_marker_still_truncates() -> None:
    """Text BEFORE the marker is the answer; text after it is protocol."""
    kept = strip("Chrome is in front. [RECENT CONTEXT] internal junk here")

    assert kept == "Chrome is in front."


def test_a_reply_that_is_only_protocol_yields_nothing() -> None:
    """Nothing to salvage is different from something being destroyed."""
    assert strip("[CURRENT USER MESSAGE]") == ""


def test_a_clean_reply_is_untouched() -> None:
    original = "A normal answer with no markers."

    assert strip(original) == original


@pytest.mark.parametrize(
    "marker",
    [
        "[RECENT CONTEXT]",
        "[RECENT COMPLETED CONVERSATION]",
        "[CURRENT USER MESSAGE]",
        "[OPERATIONAL SELF CONTEXT]",
    ],
)
def test_every_marker_is_salvaged_when_it_leads(marker: str) -> None:
    kept = strip(f"{marker}\nThe real answer survives.")

    assert "The real answer survives." in kept
    assert marker not in kept


def test_empty_input_is_safe() -> None:
    for value in (None, "", "   "):
        assert strip(value) == ""
