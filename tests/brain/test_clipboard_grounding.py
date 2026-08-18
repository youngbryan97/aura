"""A turn about the clipboard needs the clipboard read, not a capability claim.

LIVE 2026-08-17: "what's on my clipboard right now?" was answered "I can use
the clipboard — computer_use, desktop_task, os_automation are registered and
enabled right now... I can only work with the information you provide me during
our conversation." The clipboard held BUILD-7741-verify.

The capability was never missing. Nothing read it, so she had nothing to say,
and half of that reply was false in the same breath as the other half was true.
"""

from __future__ import annotations

import asyncio

import pytest

from core.brain.clipboard_grounding import (
    CLIPBOARD_HEADER,
    asks_about_clipboard,
    clipboard_block,
)


@pytest.mark.parametrize(
    "prompt",
    [
        "what's on my clipboard right now?",
        "read my clipboard",
        "what did I just copy?",
        "check the pasteboard",
    ],
)
def test_clipboard_questions_are_recognised(prompt: str) -> None:
    assert asks_about_clipboard(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    ["how are you", "read CONTRIBUTING.md", "what is 2 + 2", ""],
)
def test_other_turns_are_not(prompt: str) -> None:
    """Reading the clipboard is privacy-relevant; it is never ambient."""
    assert asks_about_clipboard(prompt) is False


def test_an_unrelated_turn_never_reads_the_clipboard(monkeypatch) -> None:
    read = {"called": False}

    class _Manager:
        async def get(self):
            read["called"] = True
            return "secret"

    monkeypatch.setattr(
        "core.capabilities.clipboard_manager.get_clipboard_manager", lambda: _Manager()
    )

    assert asyncio.run(clipboard_block("how are you")) == ""
    assert read["called"] is False


def test_the_contents_reach_the_block(monkeypatch) -> None:
    class _Manager:
        async def get(self):
            return "BUILD-7741-verify"

    monkeypatch.setattr(
        "core.capabilities.clipboard_manager.get_clipboard_manager", lambda: _Manager()
    )

    block = asyncio.run(clipboard_block("what's on my clipboard right now?"))

    assert CLIPBOARD_HEADER in block
    assert "BUILD-7741-verify" in block


def test_an_empty_clipboard_says_empty_rather_than_nothing(monkeypatch) -> None:
    """'It's empty' and 'I couldn't look' are different answers."""

    class _Manager:
        async def get(self):
            return ""

    monkeypatch.setattr(
        "core.capabilities.clipboard_manager.get_clipboard_manager", lambda: _Manager()
    )

    block = asyncio.run(clipboard_block("what's on my clipboard?"))

    assert "empty" in block.lower()


def test_a_large_clipboard_is_bounded(monkeypatch) -> None:
    class _Manager:
        async def get(self):
            return "x" * 50_000

    monkeypatch.setattr(
        "core.capabilities.clipboard_manager.get_clipboard_manager", lambda: _Manager()
    )

    block = asyncio.run(clipboard_block("what's on my clipboard?"))

    assert len(block) < 3000
    assert "50000 characters total" in block


def test_a_failing_read_yields_no_block(monkeypatch) -> None:
    class _Manager:
        async def get(self):
            raise OSError("pbpaste unavailable")

    monkeypatch.setattr(
        "core.capabilities.clipboard_manager.get_clipboard_manager", lambda: _Manager()
    )

    assert asyncio.run(clipboard_block("what's on my clipboard?")) == ""
