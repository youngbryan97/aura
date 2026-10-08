"""Renderer observations preserve paint and supply geometry without application state."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.the_drawing_as_objects import described, words_in


def _scene():
    return {"source": "canvas-paint-v1", "complete": True, "width": 200, "height": 120,
            "objects": [{"x": 10.25, "y": 30.5, "width": 8, "height": 16, "colour": "#ffffff"}],
            "texts": [{"text": "Score 7", "x": 70, "y": 4, "width": 60, "height": 12}]}


@pytest.mark.parametrize("change", [
    {"complete": False}, {"width": 201}, {"source": "application-private-state"},
    {"objects": [{"x": float("nan"), "y": 0, "width": 1, "height": 1}]},
    {"texts": [{"x": 0, "y": 0, "width": 1, "height": 1}]},
])
def test_unverified_or_incomplete_data_leaves_the_pixel_observation(change):
    picture = np.zeros((120, 200, 3), dtype=np.uint8)
    scene = {**_scene(), **change}
    assert described(picture, scene) is picture
    assert words_in(picture) is None


def test_rendered_words_have_the_screen_readers_coordinates():
    picture = described(np.zeros((120, 200, 3), dtype=np.uint8), _scene())
    [word] = words_in(picture)
    assert word["text"] == "Score 7"
    assert word["center_x"] == 0.5 and word["center_y"] == pytest.approx(1 / 12)


def test_structured_frame_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.the_drawing_as_objects import _drawing_scene_invariant

    assert _drawing_scene_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "drawing_scene_requires_matching_frame" for c in suite.claims())


def test_geometry_tracks_fractional_motion_without_turning_text_into_a_body():
    from core.perception.what_moves_in_the_picture import WhatMoves

    moves = WhatMoves()
    for n in range(50):
        scene = deepcopy(_scene())
        scene["objects"][0]["x"] += n * 0.37
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        x = int(scene["objects"][0]["x"])
        image[30:47, x:x + 9] = 255
        moves.see(described(image, scene), n * 0.03)
    [body] = list(moves.things.values())
    assert body.x == pytest.approx(10.25 + 49 * 0.37 + 4)
    assert body.y == 38.5 and body.w == 8 and body.h == 16
    assert body.vx == pytest.approx(0.37 / 0.03, rel=0.01)


@pytest.mark.parametrize("reflects", [True, False])
def test_runtime_boundary_check_uses_observed_paint_instead_of_a_missing_tracks_prediction(reflects):
    from core.agency.when_motion_breaks_a_rule import MotionChecks
    from core.perception.what_moves_in_the_picture import WhatMoves

    moves = WhatMoves()
    checks = MotionChecks(frozenset({"top"}), "test drawing contract")
    for n in range(40):
        y = abs(80 - 3 * n) if reflects else 80 - 3 * n
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        if y + 8 > 0:
            image[max(0, y):y + 8, 90:98] = 255
        scene = {"source": "canvas-paint-v1", "complete": True, "width": 200, "height": 120,
                 "objects": [{"x": 90, "y": y, "width": 8, "height": 8, "colour": "#ffffff"}], "texts": []}
        at = n * 0.03
        happened = moves.see(described(image, scene), at)
        checks.see(moves, happened, at, None)
    assert bool(checks.violations) is not reflects
    if reflects:
        assert not checks.unconfirmed


@pytest.mark.asyncio
async def test_renderer_text_reaches_the_counters_without_ocr():
    from core.agency.playing_as_it_happens import _keep_reading, _Run
    from core.agency.what_meeting_things_does import WhatMeetingDoes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.what_moves_in_the_picture import WhatMoves

    run, meeting, moves = _Run(keys=[], began=0), WhatMeetingDoes(), WhatMoves()
    moves.shape = (120, 200)
    picture = described(np.zeros((120, 200, 3), dtype=np.uint8), _scene())
    await _keep_reading(run, meeting, WhichIsHers(), moves, picture, 1, None)
    assert meeting.readouts.current == {"score": 7}
    assert run.reading is None


@pytest.fixture
async def canvas():
    from playwright.async_api import async_playwright

    from core.perception.frames_as_they_are_drawn import CanvasFrames

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 320, "height": 200})
        await page.set_content('<style>body{margin:0}</style><canvas width="320" height="200"></canvas>')
        frames = CanvasFrames(page)
        clip = {"x": 0, "y": 0, "width": 320, "height": 200}
        await frames.look(clip)
        yield SimpleNamespace(page=page, frames=frames, clip=clip)
        await frames.close()
        await browser.close()


@pytest.mark.asyncio
async def test_canvas_sensor_reports_transformed_paint_and_preserves_errors(canvas):
    await canvas.page.evaluate("""() => {
      const c=document.querySelector('canvas').getContext('2d');
      c.fillStyle='#000';c.fillRect(0,0,320,200);
      c.save();c.translate(20.25,30.5);c.scale(2,1);
      c.fillStyle='#fff';c.fillRect(1,2,10,30);c.restore();
      c.font='18px sans-serif';c.fillText('Score 7',100,20);
    }""")
    before = await canvas.page.screenshot()
    picture, _ = await canvas.frames.look(canvas.clip)
    [body] = picture.drawing_scene["objects"]
    assert (body["x"], body["y"], body["width"], body["height"]) == (22.25, 32.5, 20, 30)
    assert words_in(picture)[0]["text"] == "Score 7"
    assert before == await canvas.page.screenshot()
    assert await canvas.page.evaluate("""() => {
      try{document.querySelector('canvas').getContext('2d').arc(0,0,-1,0,7);return false;}
      catch(e){return e.name==='IndexSizeError';}
    }""")


@pytest.mark.asyncio
async def test_partial_clear_clipping_and_unsupported_paths_use_pixels(canvas):
    for unsupported in ["c.clearRect(0,0,5,5)", "c.save();c.rect(0,0,5,5);c.clip();c.fillRect(0,0,20,20);c.restore()",
                        "c.beginPath();c.moveTo(1,1);c.lineTo(20,20);c.stroke()"]:
        await canvas.page.evaluate("""(extra) => {
          const c=document.querySelector('canvas').getContext('2d');
          c.fillStyle='#000';c.fillRect(0,0,320,200);c.fillStyle='#fff';c.fillRect(20,30,10,10);
          // Test-only drawing snippets, unrelated to an application or game.
          Function('c',extra)(c);
        }""", unsupported)
        picture, _ = await canvas.frames.look(canvas.clip)
        assert getattr(picture, "drawing_scene", None) is None


@pytest.mark.asyncio
async def test_opaque_paint_cannot_leave_hidden_text_as_a_terminal_message(canvas):
    await canvas.page.evaluate("""() => {
      const c=document.querySelector('canvas').getContext('2d');
      c.fillStyle='#000';c.fillRect(0,0,320,200);
      c.fillStyle='#fff';c.font='18px sans-serif';c.fillText('You win',20,40);
      c.fillStyle='#000';c.fillRect(0,0,180,60);
    }""")
    picture, _ = await canvas.frames.look(canvas.clip)
    assert words_in(picture) == []


@pytest.mark.asyncio
async def test_the_last_frame_owner_restores_native_drawing_methods(canvas):
    from core.perception.frames_as_they_are_drawn import CanvasFrames

    second = CanvasFrames(canvas.page)
    await second.look(canvas.clip)
    await canvas.frames.close()
    assert await canvas.page.evaluate("!!window.__auraDrawingObservation")
    await second.close()
    assert not await canvas.page.evaluate("!!window.__auraDrawingObservation")
    assert await canvas.page.evaluate("CanvasRenderingContext2D.prototype.fillRect.toString().includes('[native code]')")


@pytest.mark.asyncio
async def test_a_preloaded_sensor_observes_cached_methods_across_attempts(canvas):
    from core.perception.frames_as_they_are_drawn import CanvasFrames
    from core.perception.the_drawing_as_objects import BOOTSTRAP

    await canvas.frames.close()
    await canvas.page.evaluate(BOOTSTRAP)
    await canvas.page.evaluate("""() => {
      const c=document.querySelector('canvas').getContext('2d'),paint=c.fillRect.bind(c);
      window.drawObservedExample=()=>{c.fillStyle='#000';paint(0,0,320,200);c.fillStyle='#fff';paint(31.25,40,8,16);};
      window.drawObservedExample();
    }""")
    for _ in range(2):
        reader = CanvasFrames(canvas.page)
        first, _ = await reader.look(canvas.clip)
        assert getattr(first, "drawing_scene", None) is None
        await canvas.page.evaluate("window.drawObservedExample()")
        picture, _ = await reader.look(canvas.clip)
        assert picture.drawing_scene["objects"][0]["x"] == 31.25
        await reader.close()


def test_rate_limited_score_changes_are_said_when_the_window_opens():
    from core.agency.playing_as_it_happens import _Run, _what_she_says
    from core.agency.what_meeting_things_does import WhatMeetingDoes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.what_moves_in_the_picture import WhatMoves

    run = _Run(keys=[], began=0, said_at=2)
    # How it stood when she came in is not news (LIVE 2026-10-07 "I have 0."); a change is, once the window opens.
    run.contest.mine, run.contest.theirs = 2, 1
    _what_she_says(run, None, WhatMoves(), WhichIsHers(), WhatMeetingDoes(), 2.5)
    assert not run.contest_said
    run.contest.mine = 3
    _what_she_says(run, None, WhatMoves(), WhichIsHers(), WhatMeetingDoes(), 3)
    assert not run.contest_said
    _what_she_says(run, None, WhatMoves(), WhichIsHers(), WhatMeetingDoes(), 8)
    assert run.contest_said.startswith('3–1') and run.lines
