"""The clock a caller started reaches the loop that has to obey it.

Measured live on 2026-08-26: a run that had built a 64 into the corner was
cancelled from outside at exactly the outer budget, 660s to the millisecond,
because the deadline stamped when the step was planned was dropped at the
hand-off and the pursuit began counting when it began running.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPUTER_USE = Path("core/skills/computer_use.py").read_text()
DESKTOP_TASK = Path("core/skills/desktop_task.py").read_text()
PURSUIT = Path("core/skills/screen_pursuit.py").read_text()


def test_the_task_stamps_a_deadline_when_it_plans_the_step():
    assert 'target["deadline_at"] = time.monotonic()' in DESKTOP_TASK


def test_the_hand_off_carries_it():
    body = COMPUTER_USE[COMPUTER_USE.index("async def _pursue_on_screen") :]
    start = body.index("result = await pursue_on_screen(")
    call = body[start : start + 1600]
    assert 'deadline_at=float(payload.get("deadline_at") or 0.0)' in call


def test_the_skill_that_takes_params_carries_it_too():
    assert "deadline_at=params.deadline_at," in PURSUIT


def test_the_loop_uses_whichever_clock_started_first():
    body = PURSUIT[PURSUIT.index("async def pursue_on_screen") :]
    window = body[body.index("began = time.monotonic()") : body.index("began = time.monotonic()") + 260]
    assert "ends_at = min(ends_at, float(deadline_at))" in window


def test_a_watched_goal_target_is_json_a_hand_off_can_read():
    from core.runtime.watched_goal import read_watched_goal

    watched = read_watched_goal("play 2048 until you get a 256 tile")
    assert watched is not None
    assert "max_seconds" in json.loads(json.dumps(watched.as_target()))
