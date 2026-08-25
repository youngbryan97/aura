"""A step count is not an answer, and an unread screen is not a blank one.

Both found in the live desktop on 2026-08-10.

1. "can you read text that is only pixels — words inside a video frame? answer
   yes or no first, then tell me how you know" came back as:

       Desktop task completed 1/1 governed computer-use steps through
       heuristic_compat planning. Completed 1/1 governed desktop steps.

   The step count, twice, in the branch whose own comment reads "What was
   PRODUCED, not how many steps produced it" — because the lane's `summary`
   is itself a step-count sentence, so it got pasted in front of the step
   sentence. This is the third recorded appearance of this defect shape
   (2026-07-27, 2026-07-30, 2026-08-04 all sit in comments at that call site).

2. ScreenSnapshot.screen_text is populated only from OCR of a screenshot. For
   the entire life of the governance defect that blocked take_screenshot, OCR
   never ran, screen_text was always "", and every consumer read that as "the
   screen has no text on it". Absence of a reading is not a reading of
   absence.
"""

from __future__ import annotations

import pytest

#: The owner's desktop on whatever machine runs this, rather than one
#: developer's account name baked into the fixture.
from pathlib import Path

_DESKTOP_NOTE = str(Path.home() / "Desktop" / "aura_haiku.txt")


# ── 1. Bookkeeping must not be served as an answer ─────────────────────────

@pytest.mark.parametrize(
    "summary",
    [
        "Desktop task completed 1/1 governed computer-use steps through heuristic_compat planning.",
        "Desktop task completed 2/2 governed computer-use steps through heuristic_compat planning",
        "Completed 2/2 governed desktop steps.",
        "completed 10/10 governed desktop steps",
        "",
    ],
)
def test_pure_step_reports_are_recognised_as_bookkeeping(summary: str) -> None:
    from interface.routes.chat import _is_step_bookkeeping_only

    assert _is_step_bookkeeping_only(summary) is True


@pytest.mark.parametrize(
    "summary",
    [
        "Chrome is in front, showing the YouTube documentary.",
        "I wrote the note in Notes. Completed 2/2 governed desktop steps.",
        "Completed 1/1 governed desktop steps. The file is at /tmp/x.txt",
        "Opened Notes and typed the paragraph.",
        "Completed the research and found three sources.",
    ],
)
def test_summaries_about_the_world_are_never_suppressed(summary: str) -> None:
    """The dangerous direction: silencing a real answer for mentioning steps."""
    from interface.routes.chat import _is_step_bookkeeping_only

    assert _is_step_bookkeeping_only(summary) is False


def test_bookkeeping_only_result_defers_only_when_nothing_happened() -> None:
    """Deferral is for an empty-handed lane, not for a completed action.

    The first version of this asserted a bare ``response = ""`` whenever the
    summary was step bookkeeping, and that regressed a real case the same day:
    a desktop task that COMPLETED and verified its effects, with no text
    deliverable to quote ("open Notes and write a note saying Hello"), produced
    an empty reply — and an empty reply is falsy at the caller, so a
    receipt-verified action fell through to cognition as though nothing had
    happened. It cost a foreground model pass and put the turn back into the
    lane whose failures this branch exists to avoid.

    So the contract is conditional: verified effects get a plain confirmation
    carrying the receipt; only an unverified, empty-handed lane defers.
    """
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._execute_desktop_objective_from_chat)
    marker = "elif _is_step_bookkeeping_only(summary):"
    assert marker in source
    branch = source[source.find(marker) : source.find(marker) + 3600]

    assert "verified_effects" in branch
    # Still defers when the effects were not proven.
    assert 'response = ""' in branch
    # And confirms plainly when they were, rather than saying nothing.
    assert "Done" in branch


# ── 2. An unread screen must not read as a blank screen ────────────────────

def test_screen_text_status_defaults_to_not_attempted() -> None:
    from core.perception.screen_perception import ScreenSnapshot

    assert ScreenSnapshot().screen_text_status == "not_attempted"
    assert ScreenSnapshot().screen_text == ""


def test_blank_screen_and_unread_screen_are_distinguishable() -> None:
    """The two states that were identical for the life of the capture defect."""
    from core.perception.screen_perception import ScreenSnapshot

    genuinely_blank = ScreenSnapshot()
    genuinely_blank.screen_text = ""
    genuinely_blank.screen_text_status = "read_empty"

    never_read = ScreenSnapshot()
    never_read.screen_text = ""
    never_read.screen_text_status = "unreadable:capture_failed"

    assert genuinely_blank.screen_text == never_read.screen_text
    assert genuinely_blank.screen_text_status != never_read.screen_text_status


def test_capture_path_sets_a_status_in_every_branch() -> None:
    """A silent "" is the whole defect; every path must say which it is."""
    import inspect

    from core.perception.screen_perception import ScreenPerception

    source = inspect.getsource(ScreenPerception.capture)

    assert 'screen_text_status = "read"' in source
    assert "unreadable:" in source
    assert 'screen_text_status = "not_attempted"' in source


def test_a_step_count_never_leads_a_real_deliverable() -> None:
    """LIVE, 2026-08-10, after the directory read finally worked:

        "Desktop task completed 2/2 governed computer-use steps through
         heuristic_compat planning. Here is what I wrote:

         9 file(s) matching *.py in ..."

    The answer was correct and complete, and it was introduced by a step count
    and the planner's internal identifier. "heuristic_compat" is a name for the
    engineering log, not for a person, and it arrived in front of the thing
    they asked for.

    A summary that says something about the world still leads — it is context.
    A summary that only counts steps is machinery and leads with nothing.
    """
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._execute_desktop_objective_from_chat)
    marker = "Here is what I wrote:"
    assert marker in source
    window = source[max(0, source.find(marker) - 900) : source.find(marker) + 200]

    assert "_is_step_bookkeeping_only(summary)" in window
    assert "response = produced" in window


def test_done_reports_the_effect_not_the_step_count() -> None:
    """LIVE, 2026-08-10, after the haiku task finally worked:

        "Done — Desktop task completed 1/1 governed computer-use steps through
         heuristic_compat planning."

    The receipt line exists to make "it is done" checkable. A step count with a
    planner identifier in it checks nothing a person can use. What makes the
    claim checkable is WHAT was done — the file that now exists.
    """
    from interface.routes.chat import _desktop_effect_summary

    assert _desktop_effect_summary({
        "receipts": [{
            "action": "write_text_file",
            "ok": True,
            "result": {"path": _DESKTOP_NOTE},
        }],
    }) == f"wrote {_DESKTOP_NOTE}."


def test_several_effects_are_listed_in_order() -> None:
    from interface.routes.chat import _desktop_effect_summary

    assert _desktop_effect_summary({
        "receipts": [
            {"action": "list_directory", "ok": True, "result": {"path": "/x"}},
            {"action": "write_text_file", "ok": True, "result": {"path": "/y.txt"}},
        ],
    }) == "read /x, and wrote /y.txt."


def test_an_unverified_step_is_not_reported_as_an_effect() -> None:
    """Naming a file that was not written is the false claim again."""
    from interface.routes.chat import _desktop_effect_summary

    assert _desktop_effect_summary({
        "receipts": [{"action": "write_text_file", "ok": False, "result": {"path": "/z.txt"}}],
    }) == ""


@pytest.mark.parametrize(
    ("domain", "applied", "expected"),
    [
        ("wallpaper", "/tmp/whale.jpg", "set wallpaper to /tmp/whale.jpg."),
        ("volume", "30", "set volume to 30."),
        ("focus_mode", "writing", "set focus mode to writing."),
    ],
)
def test_system_control_names_the_verified_goal_state(
    domain: str,
    applied: str,
    expected: str,
) -> None:
    """Any registered setting reads from the same domain/readback contract."""

    from interface.routes.chat import _desktop_effect_summary

    assert _desktop_effect_summary(
        {
            "receipts": [
                {
                    "action": "system_control",
                    "ok": True,
                    "effect_verified": True,
                    "result": {
                        "domain": domain,
                        "applied": applied,
                        "effect_verified": True,
                    },
                }
            ]
        }
    ) == expected


def test_the_done_branch_uses_the_effect_summary() -> None:
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._execute_desktop_objective_from_chat)
    assert '_desktop_effect_summary(result)' in source
    assert 'f"Done — {effect_line}"' in source


def test_a_question_is_never_answered_with_an_effect_receipt() -> None:
    """LIVE, 2026-08-10, with a bioRxiv preprint open in Chrome:

        "Look at what's on my screen right now and tell me what the paper is
         about. What is the actual mechanism they use?"
        → "Done — the desktop steps completed and their effects verified."

    The lane took a reading, verified the reading had happened, and reported
    the verification. "Did you do it" and "what is it" are different questions,
    and a receipt only ever answers the first.
    """
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._execute_desktop_objective_from_chat)
    marker = "effect_line = _desktop_effect_summary(result)"
    assert marker in source
    window = source[source.find(marker) : source.find(marker) + 1400]

    assert "_asks_for_information(user_message)" in window
    assert 'response = ""' in window


@pytest.mark.parametrize(
    "message",
    [
        "Look at what's on my screen and tell me what the paper is about",
        "what is on my clipboard?",
        "what's your opinion on this paper?",
        "explain the mechanism to me",
    ],
)
def test_information_requests_are_recognised(message: str) -> None:
    from interface.routes.chat import _asks_for_information

    assert _asks_for_information(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Put the text ORION-7 on my clipboard",
        "Make me a file on my Desktop called x.txt",
        "open Notes and write a note",
    ],
)
def test_effect_requests_still_get_their_receipt(message: str) -> None:
    """The direction that matters: "it is done" IS the answer to a do-request."""
    from interface.routes.chat import _asks_for_information

    assert _asks_for_information(message) is False
