"""The fallback disclosure names the reason it was given, not a usual one.

When the small model answers, the person is told so. The sentence explaining
why asserted "the main one is still loading" whatever had actually happened —
and the reason is an argument to the function that writes it.

LIVE, 2026-09-07: said while the 27B had been resident for seven minutes and
the real cause was a latent-cortex receipt contract failing
(`kv_state_tree_unproven, decode_incumbent_unproven, ...`). The person was
told to wait for something waiting would not change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: The foreground lane, where the disclosure is composed now. It was in
#: `chat.py` until the route was split for size, and `body.index(...)` on the
#: old file raised rather than reporting a missing sentence.
CHAT = ROOT / "interface/routes/chat_foreground_lane.py"


def _disclosure_source() -> str:
    body = CHAT.read_text("utf-8")
    marker = body.index("That came from my smaller model")
    return body[marker - 1400 : marker + 400]


def test_the_reason_decides_the_sentence() -> None:
    body = _disclosure_source()
    assert "still_coming" in body
    assert "the main one could not finish this turn" in body
    assert "the main one is still loading" in body


@pytest.mark.parametrize(
    ("reason", "loading"),
    [
        ("the cortex is still loading", True),
        ("model warm-up in progress", True),
        ("the runtime is still booting", True),
        ("the cognitive engine could not serve this turn", False),
        ("receipt_contract_failed:kv_state_tree_unproven", False),
        ("", False),
    ],
)
def test_the_markers_separate_the_two_cases(reason: str, loading: bool) -> None:
    """The classification, as the code performs it."""

    lowered = str(reason or "").lower()
    still_coming = any(
        marker in lowered
        for marker in ("load", "warm", "booting", "starting", "not ready", "spawning")
    )
    assert still_coming is loading


def test_the_file_still_parses() -> None:
    ast.parse(CHAT.read_text("utf-8"))
