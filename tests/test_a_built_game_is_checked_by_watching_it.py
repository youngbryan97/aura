"""A game is checked by watching it run: that it moves, and that it behaves the way games do.

The checks a person makes of a document program (select, click, see bold) say
nothing about a game. A game is seen running, and played: the same watch her
repair uses (core/self_modification/watching_a_program_run.py) tells whether
the keys move what is the player's as named, whether things turn back off it,
and whether a miss counts for the right side.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.rebuilding.checks_a_person_makes import Check, run_checks
from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt

pytestmark = pytest.mark.asyncio

_RUNS = Part("work area", '''const c = app.make("canvas", {width: 400, height: 300}); app.work.append(c); const g = c.getContext("2d"); let x = 0;
setInterval(() => { x = (x + 4) % 400; g.fillStyle = "#000"; g.fillRect(0, 0, 400, 300); g.fillStyle = "#fff"; g.fillRect(x, 140, 10, 10); }, 30);''')
_STILL = Part("work area", 'const c = app.make("canvas", {width: 400, height: 300}); app.work.append(c); c.getContext("2d").fillRect(0, 0, 40, 40);')


async def test_a_game_that_runs_is_seen_moving(tmp_path):
    moving = Check.model_validate({"feature": "runs", "steps": [], "expect": [{"see": "moving"}]})
    runs = await run_checks(ProgramAsBuilt("G", parts=[_RUNS]).write(tmp_path / "a.html"), [moving])
    still = await run_checks(ProgramAsBuilt("G", parts=[_STILL]).write(tmp_path / "b.html"), [moving])
    assert runs[0].held and not still[0].held


async def test_a_game_s_behaviour_is_watched(tmp_path):
    broken = Path.home() / "aura-demos" / "pong-spare" / "pong.html"
    if not broken.exists():
        pytest.skip("the broken Pong demo is not on this machine")
    controls = Check.model_validate({"feature": "paddle", "steps": [], "expect": [{"see": "watched", "target": "controls"}]})
    run = (await run_checks(broken, [controls]))[0]
    assert not run.held and "the other way" in run.why
