"""An approach is not judged before she has acted under it.

Measured live on 2026-08-26: ten approaches decided for nine moves made, each
one a full pass at her reasoning, and the run spent its whole budget deciding
how to play rather than playing. The condition on a fresh approach was
checked on the very next cycle, and an anchor bound to a tile that merged
away broke at once.
"""

from __future__ import annotations

from screen_pursuit_support import pursuit_source

import re
from pathlib import Path

SOURCE = pursuit_source()
BODY = SOURCE[SOURCE.index("async def pursue_on_screen") :]


def test_a_fresh_approach_is_not_re_decided_before_a_move_is_made():
    window = BODY[BODY.index("time_to_ask = (") - 900 : BODY.index("time_to_ask = (") + 400]
    assert 'tried_it = len(moves) > plan["asked_at"]' in window
    assert re.search(r"holding is False\s*\n\s*and plan\[\"held\"\] is not None\s*\n\s*and tried_it", window)


def test_the_count_still_brings_the_question_back_on_its_own():
    window = BODY[BODY.index("time_to_ask = (") : BODY.index("time_to_ask = (") + 400]
    assert "_ask_again_after" in window


def test_a_line_that_stops_holding_says_so():
    assert "the line she was taking stopped holding" in BODY


def test_a_voice_that_came_back_with_nothing_is_asked_again_later_each_time():
    """Asked again five moves on, it was put the same question every ten seconds
    while it could not answer any of them (live, 2026-09-23)."""
    from core.skills.screen_pursuit_decision_branches import _decide_the_next_move_part_14

    def due(unanswered: int, since: int) -> bool:
        plan = {"held": None, "asked_at": 10, "unanswered": unanswered}
        return bool(_decide_the_next_move_part_14("", False, True, [None] * (10 + since), plan, {}))

    assert due(0, 6)
    assert not due(3, 6)
    assert due(3, 6 * 8)
    assert 'plan["unanswered"] = 0 if fresh is not None else' in BODY
