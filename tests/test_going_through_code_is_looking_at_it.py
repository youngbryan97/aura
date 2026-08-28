"""Asking her to go through code and report back is a read, not a desktop job.

LIVE, 2026-08-28: "Something's off in <path> ... Go through the code and tell
me what's actually happening, with the file and line" was routed to the desktop
lane, which planned it as writing a document and came back "I could not write
the words you asked for, so I have not made the file." Nothing had asked for a
file. The diagnosis engine that owns this had the answer — invoice.py:4, a
mutable default, with the remedy — and was never reached.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import (
    looks_like_desktop_objective,
    looks_like_filesystem_observation,
)

_THE_LIVE_TURN = (
    "Something's off in /private/tmp/x/invoice-tools and I can't put my finger "
    "on it. No error, nothing crashes, the tests such as they are pass. But "
    "when I build two invoices in a row the second one comes out wrong. Go "
    "through the code and tell me what's actually happening, with the file "
    "and line."
)

_READS = (
    _THE_LIVE_TURN,
    "what's in /etc/hosts",
    "read the code at /tmp/x and work out why the test fails",
    "walk me through /tmp/proj/main.py",
    "trace what /tmp/proj/run.py does",
    "explain /tmp/proj/invoice.py to me",
    "step through /tmp/proj and tell me where it goes wrong",
    "review /tmp/proj/main.py",
    "diagnose /tmp/proj",
    "show me what /tmp/proj/a.py is doing",
)

_NOT_READS = (
    "read /tmp/x/a.py and fix the bug",
    "write hello into ~/Documents/x.txt",
    "open Chrome and search for otters",
    "delete /tmp/x/a.py",
)


@pytest.mark.parametrize("asked", _READS)
def test_a_request_to_look_and_report_is_an_observation(asked: str) -> None:
    assert looks_like_filesystem_observation(asked)


@pytest.mark.parametrize("asked", _READS)
def test_an_observation_does_not_go_to_the_desktop_lane(asked: str) -> None:
    assert not looks_like_desktop_objective(asked)


@pytest.mark.parametrize("asked", _NOT_READS)
def test_a_change_is_still_not_an_observation(asked: str) -> None:
    assert not looks_like_filesystem_observation(asked)


def test_telling_her_about_a_path_without_asking_is_not_a_read() -> None:
    # No question, no verb of looking: this is a remark, not a request.
    assert not looks_like_filesystem_observation(
        "I keep my invoices in /tmp/proj these days."
    )


def test_the_reporting_form_still_needs_a_path() -> None:
    assert not looks_like_filesystem_observation("tell me what you think")
    assert not looks_like_filesystem_observation("walk me through your reasoning")


def test_a_goal_needs_a_screen_to_be_watched_on() -> None:
    """Every field of a watched goal is about a screen.

    LIVE, 2026-08-28: "step through /tmp/proj and tell me where it goes wrong"
    matched the "step through" continuation cue, written for stepping through a
    game, and became a watched goal with no finishing condition, no app and the
    four arrow keys — routing a request to read code into the lane that drives
    the screen.
    """

    from core.runtime.watched_goal import read_watched_goal

    assert read_watched_goal("step through /tmp/proj and tell me where it goes wrong") is None
    assert read_watched_goal("keep refreshing /tmp/proj/out.log until it says done") is None


def test_naming_an_app_or_a_url_puts_the_screen_back() -> None:
    from core.runtime.watched_goal import read_watched_goal

    assert read_watched_goal("step through the code in /tmp/proj in Chrome") is not None
    assert read_watched_goal("play 2048 at https://play2048.co until you get a 256 tile") is not None


def test_a_pursuit_with_no_path_is_untouched() -> None:
    from core.runtime.watched_goal import read_watched_goal

    for asked in (
        "Go find a 2048 game online and play it until you get a 128 tile.",
        "find a sliding puzzle and work out how it moves by playing it",
    ):
        assert read_watched_goal(asked) is not None, asked


# ------------------------- the operation is inferred from its causal object


def _here() -> str:
    import os

    return os.getcwd()


def test_a_change_mentioned_is_not_a_change_asked_for() -> None:
    """Found by the formed concept, not by somebody noticing four failures.

    "a token is not a decision" named _FILESYSTEM_MUTATION_RE as deciding from
    the bare words copy, move, write and make. Probing the site it named turned
    up four ordinary read requests classified as changes by one word each, in
    every case a word belonging to something other than the thing named.
    """

    here = _here()
    for asked in (
        f"read {here}/README.md and tell me what approach they copy from elsewhere",
        f"go through {here}/Makefile and explain what the write targets do",
        f"what's in {here}/CLAUDE.md that I should copy into my own project",
        f"look at {here}/README.md - does it make sense?",
    ):
        assert looks_like_filesystem_observation(asked), asked


def test_a_change_asked_for_is_still_a_change() -> None:
    here = _here()
    for asked in (
        f"write hello into {here}/x.txt",
        f"delete {here}/x.txt",
        f"move {here}/a.py to {here}/b.py",
        f"copy {here}/a.py to my desktop",
    ):
        assert not looks_like_filesystem_observation(asked), asked


def test_a_pronoun_carries_the_thing_that_was_named() -> None:
    """"Read the log, then delete it" changes the log, and says so with "it"."""

    here = _here()
    assert not looks_like_filesystem_observation(
        f"read {here}/README.md and then delete it"
    )
    # And a pronoun with nothing named before it reaches nothing.
    assert not looks_like_filesystem_observation("delete it")


def test_the_verb_does_not_reach_past_a_clause_boundary() -> None:
    here = _here()
    # The copying is something the READER would do, in another clause.
    assert looks_like_filesystem_observation(
        f"tell me what {here}/README.md says, but I might copy their layout"
    )
