"""Actual browser keys and captured RGB exercise the independent search path.

These synthetic scenes test acquisition and search; they cannot establish
original-game success or ordinary Aura prompt performance.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from playwright.async_api import async_playwright

from core.agency.collecting_independent_pixel_trials import (
    PixelExperimentInputs,
    collect_independent_pixel_trials,
)
from core.agency.discovering_control_in_current_pixels import PixelResponseDiscovery
from core.agency.pixels_inside_control_trials import RandomizedPixelExperiment
from core.agency.what_a_press_does import WhatAPressDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.frames_as_they_are_drawn import CanvasFrames
from core.perception.pixel_motion_before_object_identity import pixel_motion_signature
from core.runtime.executors import run_interactive_cpu

_SCENE = r"""({background,body,responds,bodySize,center}) => {
 document.body.style.margin='0';
 const canvas=document.createElement('canvas');canvas.width=192;canvas.height=144;document.body.append(canvas);
 const ctx=canvas.getContext('2d');
 const image=(wide,tall,data)=>{const c=document.createElement('canvas');c.width=wide;c.height=tall;
   c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(data),wide,tall),0,0);return c;};
 const floor=image(192,144,background),actor=image(48,48,body);
 let angle=0;window.delivered=[];
 const draw=()=>{ctx.drawImage(floor,0,0);ctx.save();ctx.translate(center[0],center[1]);ctx.rotate(angle);
   ctx.drawImage(actor,-bodySize/2,-bodySize/2,bodySize,bodySize);ctx.restore();};
 addEventListener('keydown',event=>{if(event.code!=='ArrowLeft'&&event.code!=='ArrowRight')return;
   window.delivered.push(['down',event.code]);if(responds)angle+=event.code==='ArrowLeft'?-.12:.12;draw();});
 addEventListener('keyup',event=>{window.delivered.push(['up',event.code]);});
 draw();
}"""


def _texture(height, width, seed):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    values = gaussian_filter(rng.uniform(20, 230, (height, width)), .5).astype(np.uint8)
    rgba = np.repeat(values[..., None], 4, axis=2)
    rgba[..., 3] = 255
    return rgba.reshape(-1).tolist()


@pytest.mark.asyncio
@pytest.mark.parametrize("responds,body_size", [(False, 48), (True, 48), (True, 20)])
async def test_physical_browser_inputs_reach_bound_pixel_discovery(responds, body_size):
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 192, "height": 144})
            center = (84, 60) if body_size == 20 else (96, 72)
            await page.evaluate(_SCENE, {"background": _texture(144, 192, 12),
                                        "body": _texture(48, 48, 31), "responds": responds,
                                        "bodySize": body_size, "center": center})
            frames = CanvasFrames(page)
            run = SimpleNamespace(held="", held_since=0., held_trying=False, presses=WhatAPressDoes(),
                                  things={}, input_key_downs=Counter(), responses=[], letting_go=[],
                                  charging=SimpleNamespace(key=""))
            hers = WhichIsHers()
            inputs = PixelExperimentInputs(SimpleNamespace(down=page.keyboard.down, up=page.keyboard.up),
                                           run, hers, owns=lambda: True)
            experiment = RandomizedPixelExperiment(("ArrowLeft", "ArrowRight"), seed=41)
            discovery = PixelResponseDiscovery(experiment)
            latest = None
            async def look():
                nonlocal latest
                latest = await frames.look({"x": 0, "y": 0, "width": 192, "height": 144})
                return latest
            async def measure(press):
                signature = await run_interactive_cpu(pixel_motion_signature, press.before.rgb(), press.after.rgb(),
                                                      tile=24, point_limit=256)
                return discovery.accept_last_press(press, signature)
            result = await collect_independent_pixel_trials(experiment, look=look, inputs=inputs, after_press=measure,
                                                           capture_s=.4, ack_s=.4, total_s=20.)
            assert result.status == "complete" and result.release_confirmed, (result, experiment.refused)
            assert experiment.complete and run.held == ""
            delivered = await page.evaluate("window.delivered")
            assert len(delivered) == 48 and len(experiment.trials) == 24
            assert delivered == [list(event) for press in experiment.schedule
                                  for event in (("down", press.key), ("up", press.key))]
            region = await run_interactive_cpu(discovery.region)
            if responds:
                stable_angles = np.all(np.asarray([s.angular_support for s in discovery._signatures]) >= 4, axis=0)
                assert region is not None, {
                    "stable_angular_cells": np.flatnonzero(stable_angles).tolist(),
                    "angle_extents": [(min(s.features, key=lambda f: f[2])[2], max(s.features, key=lambda f: f[2])[2])
                                      for s in discovery._signatures],
                }
                left, top, right, bottom = region.sampling_bounds
                assert left <= center[0] <= right and top <= center[1] <= bottom
                assert region.status == "search_hint_only" and hers.number is None
                from core.perception.a_search_hint_for_these_pixels import make_search_hint

                image, at = latest
                hint = make_search_hint(image, at, normalized_region=(left / 192, top / 144, right / 192, bottom / 144))
                assert hint is not None
            else:
                assert region is None and hers.number is None
            await frames.close()
        finally:
            await browser.close()
