"""A keystroke had no address, and reported success for landing anywhere.

LIVE DEFECT, 2026-08-18. Her own browser controller opened play2048.co, her own
screen perception read the board, and hotkey("left") returned success=True
while nothing moved. The frontmost application was Claude. The arrow key went
there.

AppleScript delivers a keystroke to whatever is in front at that instant, so
every keyboard-driven task is aimed at whichever window the person last
touched — and the receipt says success, because the key WAS delivered. The loop
then observes no change and concludes the task failed, which is the wrong
lesson drawn from the wrong evidence.

Perception already knew what was in front. Actuation never asked. This is
general: an arrow key sent to the wrong window is noise, but a sentence typed
into the wrong window is content in someone's document, chat or terminal.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.capabilities.host_automation import HostAutomationProvider


class _Provider(HostAutomationProvider):
    """Real guard, fake frontmost reading."""

    def __init__(self, frontmost: str, *, readable: bool = True) -> None:
        self._frontmost = frontmost
        self._readable = readable
        self.logged: list = []

    async def get_frontmost_window_context(self):  # type: ignore[override]
        from types import SimpleNamespace

        return SimpleNamespace(
            success=self._readable, result=self._frontmost, error=""
        )

    def _log_receipt(self, receipt):  # type: ignore[override]
        self.logged.append(receipt)


def _guard(frontmost: str, expect: str, *, readable: bool = True):
    provider = _Provider(frontmost, readable=readable)
    return asyncio.run(provider._refuse_if_not_frontmost(expect, "hotkey"))


def test_a_keystroke_aimed_at_the_frontmost_app_is_allowed():
    assert _guard("Google Chrome|2048 - Play", "Google Chrome") is None


def test_the_exact_live_case_is_refused():
    """Chrome was the target; Claude was in front."""
    refusal = _guard("Claude|Claude", "Google Chrome")

    assert refusal is not None
    assert refusal.success is False
    assert "not frontmost" in refusal.error
    assert "Claude" in refusal.error, "the refusal must name what IS in front"


def test_an_unreadable_frontmost_refuses_rather_than_firing_blind():
    """Unable to tell is not permission to guess."""
    refusal = _guard("", "Google Chrome", readable=False)

    assert refusal is not None
    assert refusal.success is False


def test_no_target_means_no_guard():
    """Existing callers that aim at nothing keep working."""
    assert _guard("Claude|Claude", "") is None


@pytest.mark.parametrize(
    ("frontmost", "expect"),
    [
        ("Google Chrome|title", "chrome"),
        ("Google Chrome|title", "GOOGLE CHROME"),
        ("Preview|Document.pdf", "Preview"),
    ],
)
def test_matching_is_case_insensitive_and_ignores_the_window_title(frontmost, expect):
    """The reading is "App|Title" and a caller means the application."""
    assert _guard(frontmost, expect) is None


def test_a_different_app_with_a_similar_title_is_still_refused():
    """The title must never satisfy an application match."""
    refusal = _guard("Notes|Google Chrome notes", "Google Chrome")

    assert refusal is not None


@pytest.mark.parametrize("method", ["hotkey", "type_text"])
def test_both_input_paths_take_a_target(method):
    """Typing into the wrong window is worse than a stray arrow key."""
    signature = inspect.signature(getattr(HostAutomationProvider, method))

    assert "expect_app" in signature.parameters


@pytest.mark.parametrize("method", ["hotkey", "type_text"])
def test_the_guard_runs_before_the_keystroke(method):
    """A check after delivery is not a check.

    Measured against the CODE, not the docstring. A first version searched the
    whole source for "keystroke" and matched the prose above the guard, so it
    failed while the ordering was correct — the same class of mistake as
    asserting on a log line instead of a value.
    """
    source = inspect.getsource(getattr(HostAutomationProvider, method))
    body = source[source.index('"""', source.index('"""') + 3) + 3 :]
    guard = body.index("_refuse_if_not_frontmost")

    for sender in ("run_applescript", "_run_script", "osascript"):
        if sender in body:
            assert guard < body.index(sender), f"{method}: guard after {sender}"
