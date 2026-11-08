"""The conversation in front of her is this sitting's; earlier sittings reach her as memory.

LIVE 7 November 2026, after a day of rehearsals: "So what did you think of all three?" bore by meaning on seven earlier
exchanges asking it, each answered of three other videos in a sitting of its own, and the next sitting's first answer
named the presenter of an earlier sitting's video.
"""

from core.conversation.delivered_history import reached_exchange_messages
from core.conversation.history_reach import measure_reach
from core.conversation.sitting import A_SITTING_BREAKS_AFTER_S, sitting_begins, when
from core.utils.injected_blocks import stamp_runtime_payload

ASKED = "So what did you think of all three?"
NOW = 1_800_000_000.0
HOUR = 3600.0


def _exchange(user, aura, at):
    return {"user": user, "aura": aura, "at": at}


def test_a_quiet_longer_than_the_break_begins_a_new_sitting():
    times = [NOW - 3 * HOUR, NOW - 2 * HOUR - 50 * 60, NOW - 20 * 60, NOW - 5 * 60]
    assert sitting_begins(times, NOW) == 2
    assert sitting_begins(times, NOW + 2 * HOUR) == 4
    assert sitting_begins([NOW - 60, None], NOW) == 0
    assert A_SITTING_BREAKS_AFTER_S == 30 * 60


def test_times_are_read_from_epoch_numbers_and_iso_times():
    assert when("1800000000.5") == 1_800_000_000.5
    assert when("2026-11-08T07:24:32+00:00") == when("2026-11-08T07:24:32Z")
    assert when("") is None and when("yesterday") is None


def test_an_earlier_sitting_is_not_read_with_a_request_however_near_its_meaning():
    exchanges = [
        _exchange(ASKED, "Traffic was the weirdest of the three.", NOW - 5 * HOUR),
        _exchange("Let's watch this together.", "Bicycles, I think.", NOW - 20 * 60),
        _exchange("Now Don't Hug Me I'm Scared.", "Episode one. Said after it: the song turned.", NOW - 30),
    ]
    reach = measure_reach(exchanges, request=ASKED, now=NOW)
    assert 0 not in reach.kept
    assert reach.kept[-1] == 2


def test_after_a_break_the_last_exchange_is_still_what_a_request_answers_into():
    exchanges = [
        _exchange("What should we call the cat?", "Pickle.", NOW - 9 * HOUR),
        _exchange("Tell me about herons.", "They wait.", NOW - 8 * HOUR),
    ]
    reach = measure_reach(exchanges, request="ok, go on about them", now=NOW)
    assert reach.kept == (1,)


def test_asked_again_within_a_sitting_it_is_still_open():
    exchanges = [
        _exchange(ASKED, "I couldn't verify a reply that answers that request.", NOW - 6 * 60),
        _exchange("Are you there?", "Yes.", NOW - 60),
    ]
    reach = measure_reach(exchanges, request=ASKED, now=NOW)
    assert "sittings before this one" not in reach.reason


def test_exchanges_with_no_time_are_divided_by_nothing():
    exchanges = [{"user": ASKED, "aura": "Of three other videos."}, {"user": "Pick one.", "aura": "This one."}]
    reach = measure_reach(exchanges, request=ASKED, now=NOW)
    assert "sittings before this one" not in reach.reason


def test_the_exchanges_handed_to_her_model_carry_when_she_last_spoke_in_them():
    exchanges = [
        stamp_runtime_payload({"user": ASKED, "aura": "Of three other videos.", "timestamp": str(NOW - 5 * HOUR)}),
        stamp_runtime_payload({"user": "Pick one.", "aura": "This one.", "timestamp": "2026-11-08T08:00:00+00:00",
                               "said_after": ["It's over."], "last_said_at": NOW - 10}),
    ]
    reached = reached_exchange_messages(exchanges, request=ASKED)
    assert reached.messages == [
        {"role": "user", "content": "Pick one."},
        {"role": "assistant", "content": "This one.\n\nIt's over."},
    ]
