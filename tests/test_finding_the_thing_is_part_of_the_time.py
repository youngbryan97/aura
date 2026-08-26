"""The clock on a watched goal covers finding the thing, not only playing it.

Measured live on 2026-08-26: sixty-five narrated moves and nine approaches
held in a game of 2048, cancelled by the outer deadline — the one that only
has room to report — and reported to the person as "Completed 0/0 steps".
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path("core/skills/screen_pursuit.py").read_text()


def test_the_clock_starts_before_the_page_is_found():
    body = SOURCE[SOURCE.index("async def pursue_on_screen") :]
    began = body.index("began = time.monotonic()")
    executor = body.index("executor.pursue(")
    assert began < executor


def test_what_the_setup_spent_comes_off_the_budget():
    body = SOURCE[SOURCE.index("async def pursue_on_screen") :]
    call = body[body.index("executor.pursue(") : body.index("executor.pursue(") + 700]
    assert re.search(r"max_seconds=max\(1\.0, ends_at - time\.monotonic\(\)\)", call)


def test_the_outer_grace_is_still_larger_than_the_pursuit():
    from core.runtime.watched_goal import PURSUIT_SECONDS
    from core.skills.desktop_task import DesktopTaskSkill

    asked = DesktopTaskSkill.timeout_for({"objective": "play 2048 until you get a 256 tile"})
    assert asked > PURSUIT_SECONDS
