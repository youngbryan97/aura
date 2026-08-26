"""What makes a cycle worth extra thought is the situation, not the buttons.

Measured live on 2026-08-26: seventeen language passes for fourteen moves,
twenty seconds a move. A way out is offered whenever the screen has one, a
game page carries a New Game button permanently, and counting the options
made every ordinary move a moment worth weighing.
"""

from __future__ import annotations

from pathlib import Path

SOURCE = Path("core/skills/screen_pursuit.py").read_text()
BODY = SOURCE[SOURCE.index("async def pursue_on_screen") :]


def test_the_option_count_no_longer_decides_it():
    assert "unusual = stuck(history) or len(available) > len(move_keys)" not in BODY


def test_being_stuck_still_earns_a_pass():
    assert "unusual = stuck(history) or ended or offered_pacing" in BODY


def test_a_world_that_stopped_answering_earns_one():
    where = BODY.index("unusual = stuck(history)")
    assert "ended" in BODY[where : where + 60]


def test_deciding_her_own_pacing_earns_one():
    assert "offered_pacing = bool(behind.get(\"waiting\")) and not pacing[\"choice\"]" in BODY
