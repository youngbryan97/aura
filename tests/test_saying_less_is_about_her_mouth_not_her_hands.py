"""Choosing a quieter pace does not stop her committing to a plan.

Measured live on 2026-08-26: forty-eight cycles that had each committed to
two to four moves produced fifty-three moves between them, one screen reading
apiece, because one decision about commentary had collapsed every sequence to
a single key for the rest of the run.
"""

from __future__ import annotations

from pathlib import Path

SOURCE = Path("core/skills/screen_pursuit.py").read_text()
BODY = SOURCE[SOURCE.index("async def pursue_on_screen") :]


def test_a_quiet_pace_no_longer_shortens_the_sequence():
    assert "sequence = [key, *follow_on] if follow_on else [key]" in BODY
    assert 'if follow_on and not pacing["brief"]' not in BODY


def test_a_quiet_pace_still_says_less():
    assert 'aloud = narrate and (position == 0 or not pacing["brief"])' in BODY
    assert "_say_intent(step, reason, out_loud=aloud" in BODY


def test_a_pace_chosen_for_a_backlog_ends_with_the_backlog():
    where = BODY.index('if pacing["choice"] and not behind.get("waiting"):')
    window = BODY[where : where + 160]
    assert 'pacing["choice"] = ""' in window
    assert 'pacing["brief"] = False' in window


def test_the_first_of_a_sequence_still_carries_its_reason():
    assert 'reason = (None if pacing["brief"] else made) if position == 0 else None' in BODY
