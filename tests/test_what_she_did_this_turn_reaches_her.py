"""What she did in this turn, and what each call came to, reaches every generation of the turn as a reading.

LIVE 16 November 2026, 23:30 to 23:35 local: asked for a desktop notification, four notify_user calls in her tool loop
each failed ("argv entries must not be empty"), the loop ran out of time, and the pass that then wrote her reply was
given her capabilities and not this record. She said "I have a notify_user capability available, so let me push it to
your desktop now."
"""
from __future__ import annotations

import asyncio

from core.conversation.surface_disposition import WHAT_SHE_DID_THIS_TURN, record_tool_receipt, what_she_did_this_turn
from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

FAILED = "Skill error: ValueError: argv entries must not be empty"


def _the_night_of_the_notification():
    # Her four attempts that night, as the loop logged their arguments: with a sound and without, twice each. One call
    # recorded twice is one call (core/conversation/turn_evidence_custody.py); calls with other arguments are not.
    for attempt, args in enumerate(("message, sound, title", "message, title") * 2):
        record_tool_receipt("notify_user", ok=False, action="execute", object_ref=f"{args} #{attempt}",
                            observed_content=FAILED)
    record_tool_receipt("notify_user", ok=True, action="execute",
                        observed_content="Notification delivered to the user's desktop: 'Time to stretch'")


def test_each_call_and_what_it_came_to_is_told_once_with_how_often():
    with bind_turn_evidence_custody(session_id="s", turn_id="t-told"):
        _the_night_of_the_notification()
        told = what_she_did_this_turn()

    assert told.splitlines() == [
        WHAT_SHE_DID_THIS_TURN,
        f"- notify_user: failed (4 times): {FAILED}",
        "- notify_user: done: Notification delivered to the user's desktop: 'Time to stretch'",
    ]


def test_a_turn_with_no_calls_has_no_such_reading():
    with bind_turn_evidence_custody(session_id="s", turn_id="t-none"):
        assert what_she_did_this_turn() == ""
    assert what_she_did_this_turn() == "", "outside a turn there is no turn to tell of"


def test_her_gate_gives_her_the_calls_as_her_own_after_the_persons_message():
    """As calls she made, in her template's form for them, after the person's message: LIVE 17 November 2026, 01:58,
    as a reading under a heading the record was among the grounding a character budget drops whole, and her reply
    offered to point at a button the ring was already drawn around."""
    import json

    from core.brain.inference_gate_present_moment import _attach_the_present_moment
    from core.brain.inference_gate_turn_serving import _her_calls_after_the_persons_message

    asked = [{"role": "system", "content": "s"}, {"role": "user", "content": "Could you put up a notification?"}]
    ambient: list[str] = []
    with bind_turn_evidence_custody(session_id="s", turn_id="t-gate"):
        _the_night_of_the_notification()
        asyncio.run(_attach_the_present_moment(
            ambient_grounding_blocks=ambient, isolated_generation_contract=False,
            recent_actions_already_grounded=True, task_grounding_blocks=[],
            visible_user_prompt="Could you put up a desktop notification that says time to stretch?"))
        given = _her_calls_after_the_persons_message(list(asked))
        in_her_loop = _her_calls_after_the_persons_message([*asked, *given[2:4]])

    assert not any(block.startswith(WHAT_SHE_DID_THIS_TURN) for block in ambient), "not a reading among her grounding"
    # Four calls with other arguments are four calls, and the one that went is a fifth.
    assert given[:2] == asked and [m["role"] for m in given[2:]] == ["assistant", "tool"] * 5
    assert given[2]["tool_calls"][0]["function"] == {"name": "notify_user", "arguments": {"action": "execute",
                                                     "object": "message, sound, title #0"}}
    assert json.loads(given[3]["content"]) == {"ok": False, "came_to": FAILED}
    assert json.loads(given[-1]["content"]) == {"ok": True, "came_to": "Notification delivered to the user's desktop: "
                                                                     "'Time to stretch'"}
    assert in_her_loop == [*asked, *given[2:4]], "her tool loop's own calls are already there"


def test_no_call_this_turn_adds_nothing():
    from core.brain.inference_gate_turn_serving import _her_calls_after_the_persons_message

    asked = [{"role": "user", "content": "hello"}]
    with bind_turn_evidence_custody(session_id="s", turn_id="t-quiet"):
        assert _her_calls_after_the_persons_message(list(asked)) == asked
    assert _her_calls_after_the_persons_message(list(asked)) == asked
