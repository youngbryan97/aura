"""A file holding prose about the request is worse than no file.

LIVE 2026-08-26, three times: "make a file on my Desktop with one sentence in
it about what you did tonight" created the file and filled it with

    Notes on the requested subject: The requested subject is the focus of
    this note. The important part is to describe the subject clearly...

and reported success. The last time, the cause was finally readable — the
resident worker was not alive yet, and the router had answered
"ROUTER_ERROR: worker_not_alive (at all_failed)".

This module calls a true receipt for the wrong artifact the worse failure
everywhere else it appears. Saying she could not write it costs the person
nothing they had; writing the template costs them a file they have to open to
discover is empty.
"""
from __future__ import annotations

import inspect

from core.skills.desktop_task import DesktopTaskSkill


def test_a_failed_authorship_returns_nothing_written():
    source = inspect.getsource(DesktopTaskSkill)
    where = source.index("desktop_task_content_unavailable")
    block = source[max(0, where - 1600) : where + 700]
    # It is the else of the authoring attempt, not a separate guess.
    assert "_synthesize_requested_writing" in block
    assert '"ok": False' in block
    assert '"steps_completed": 0' in block


def test_what_she_says_names_the_file_that_was_not_made():
    source = inspect.getsource(DesktopTaskSkill)
    where = source.index("desktop_task_content_unavailable")
    said = source[where : where + 600]
    assert "have not" in said and "made the file" in said
    assert "Nothing was created" in said
    # And it invites the retry, because the cause is usually momentary.
    assert "Ask me again" in said


def test_the_template_is_still_available_for_artifacts_that_are_not_authored():
    """A receipt document for a task that authored nothing is a different
    thing and keeps its composer."""
    body = DesktopTaskSkill._compose_requested_writing_body("write a note about whales")
    assert body, "the composer itself is unchanged"


def test_every_way_the_writer_can_come_back_empty_is_recorded():
    """Four returns of "" and only the exception left a trace, so a template
    could be written and nobody could tell which guard had fired."""
    source = inspect.getsource(DesktopTaskSkill._synthesize_requested_writing)
    assert source.count("_note_unauthored") >= 3


import pytest  # noqa: E402

from core.skills.desktop_task import _is_still_coming_up  # noqa: E402


@pytest.mark.parametrize(
    "answered",
    [
        "ROUTER_ERROR: worker_not_alive (at all_failed)",
        "ROUTER_ERROR: init_not_complete (at Cortex)",
        "ROUTER_ERROR: lane_handshaking (at Cortex)",
    ],
)
def test_a_lane_that_is_coming_up_is_not_a_lane_that_refused(answered):
    """LIVE 2026-08-26: the writing task got "worker_not_alive" seconds before
    the same runtime answered an ordinary question, because the Cortex lane
    was mid-warmup and this caller asked once and gave up."""
    assert _is_still_coming_up(answered)


@pytest.mark.parametrize(
    "answered",
    [
        "ROUTER_ERROR: bad_request (at Cortex)",
        "ROUTER_ERROR: context_too_long (at Cortex)",
        "A real sentence about the evening.",
        "",
    ],
)
def test_a_real_failure_or_real_text_is_not_treated_as_warming(answered):
    assert not _is_still_coming_up(answered)


def test_the_writer_waits_once_for_a_warming_lane():
    source = inspect.getsource(DesktopTaskSkill._synthesize_requested_writing)
    assert "_is_still_coming_up(text)" in source
    assert "_WARMING_RETRY_SECONDS" in source
    # One wait, not a loop.
    assert source.count("await _ask()") == 2
