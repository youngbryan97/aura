"""An approach is not judged before she has acted under it.

Measured live on 2026-08-26: ten approaches decided for nine moves made, each
one a full pass at her reasoning, and the run spent its whole budget deciding
how to play rather than playing. The condition on a fresh approach was
checked on the very next cycle, and an anchor bound to a tile that merged
away broke at once.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path("core/skills/screen_pursuit.py").read_text()
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
