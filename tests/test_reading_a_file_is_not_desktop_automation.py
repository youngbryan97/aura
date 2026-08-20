"""Reading a file is an observation. Writing one is an actuation.

LIVE, 2026-08-19. Asked to debug an unfamiliar repository:

    os_automation failed: Skill error: TimeoutError (expected: OS automation
    returns a verifiable effect contract...). Completed 0/1 steps.

os_automation drives the screen. It cannot read a file, so it spent the turn
failing to verify an effect that was never going to happen — while
file_operation sat READY with a read action that is pure observation.

This module already draws exactly this line for the SCREEN. `looks_like_
screen_observation` was written after a screen read was sent to the actuation
lane and refused, with the note "Observation and actuation need different
lanes." The same mistake, one surface over.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import (
    looks_like_desktop_objective,
    looks_like_filesystem_observation,
)

REPO_TASK = (
    "there's a python project at /private/tmp/claude-501/x/ledger - one of its "
    "tests is failing. read the code, work out why, and tell me exactly which "
    "line is wrong and what it should be."
)


@pytest.mark.parametrize(
    "message",
    [
        REPO_TASK,
        "read /tmp/notes.txt and tell me what it says",
        "list the files in /tmp/proj",
        "show me what's in ~/Documents/report.md",
        "check /etc/hosts and tell me if the entry is there",
    ],
)
def test_a_read_is_an_observation(message: str):
    assert looks_like_filesystem_observation(message)
    assert not looks_like_desktop_objective(message)


@pytest.mark.parametrize(
    "message",
    [
        # The 2026-08-10 case: a file WRITE must still reach the body.
        "write a haiku to a file on my Desktop called aura_haiku.txt",
        "read /tmp/a.py and fix the bug in it",
        "delete /tmp/old.log",
        "rename /tmp/a.txt to /tmp/b.txt",
    ],
)
def test_anything_that_changes_disk_is_still_an_actuation(message: str):
    assert not looks_like_filesystem_observation(message)
    assert looks_like_desktop_objective(message)


def test_reading_and_then_changing_is_not_an_observation():
    """Only the lane that can write can finish it."""
    assert not looks_like_filesystem_observation("read /tmp/a.py and fix the bug in it")


@pytest.mark.parametrize(
    "message",
    ["open chrome and go to example.com", "click the save button in the window"],
)
def test_real_desktop_control_is_untouched(message: str):
    assert not looks_like_filesystem_observation(message)


@pytest.mark.parametrize(
    "message",
    ["how are you feeling today", "what is 2 + 2", "tell me a story"],
)
def test_conversation_is_neither(message: str):
    assert not looks_like_filesystem_observation(message)
    assert not looks_like_desktop_objective(message)


def test_a_path_is_required_so_ordinary_reading_words_do_not_claim_a_turn():
    """"read the room" is not a filesystem operation."""
    assert not looks_like_filesystem_observation("read the room before you answer")
    assert not looks_like_filesystem_observation("show me what you mean")
