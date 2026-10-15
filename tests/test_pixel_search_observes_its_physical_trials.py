"""Physical browser trials keep the ordinary tracker alive until control is measured."""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from types import SimpleNamespace

import pytest
from playwright.async_api import async_playwright

from core.agency.an_observed_scene_transition import _observed_transition_invariant
from core.agency.how_a_game_works import STEERED, Answer, HowItWorks
from core.agency.searching_before_a_body_is_known import (
    eligible_pixel_search,
    search_after_unresolved_body,
    search_inputs,
)
from core.agency.what_a_press_does import WhatAPressDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.frames_as_they_are_drawn import CanvasFrames
from core.perception.play_observation_without_holding_the_loop import open_play_observation
from core.perception.what_moves_in_the_picture import WhatMoves

_WORLD = r"""(responds) => {
 document.body.style.margin='0';
 const canvas=document.createElement('canvas');canvas.width=240;canvas.height=160;document.body.append(canvas);
 const c=canvas.getContext('2d');let x=120;let held='';let last=performance.now();window.delivered=[];
 addEventListener('keydown',e=>{held=e.code;window.delivered.push(['down',e.code]);});
 addEventListener('keyup',e=>{if(held===e.code)held='';window.delivered.push(['up',e.code]);});
 const draw=(now)=>{let dt=Math.min(.04,(now-last)/1000);last=now;
  if(responds)x=Math.max(25,Math.min(215,x+(held==='KeyJ'?-70:held==='KeyK'?70:0)*dt));
  c.fillStyle='#303030';c.fillRect(0,0,240,160);
  c.fillStyle='#dedede';for(let i=8;i<240;i+=24){for(const y of [8,40,116,148])c.fillRect(i,y,3,5);}
  c.fillStyle='#df2535';c.fillRect(x-11,65,22,28);
  c.fillStyle='#f8ba55';c.fillRect(x-8,69,5,6);c.fillStyle='#dedede';c.fillRect(x+2,78,6,9);
  requestAnimationFrame(draw);
 };requestAnimationFrame(draw);
}"""


def test_experiment_scene_transition_preserves_phase_bookkeeping():
    assert _observed_transition_invariant() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['responsive', 'no-effect', 'phase', 'slow', 'slow-phase'])
async def test_browser_search_captures_teach_existing_control_attribution_without_a_region(mode):
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        observation = None
        frames = None
        try:
            page = await browser.new_page(viewport={'width': 240, 'height': 160})
            world = _WORLD.replace("c.fillStyle='#303030'", "c.fillStyle=window.changed?'#f5f5f5':'#303030'")
            if mode == 'slow-phase':
                world = world.replace("let dt=", "if(held&&now-window.pressedAt>650)window.changed=true;let dt=")
            await page.evaluate(world, mode != 'no-effect')
            if mode == 'phase':
                await page.evaluate("addEventListener('keydown',()=>{window.changed=true;})")
            elif mode == 'slow-phase':
                await page.evaluate("addEventListener('keydown',()=>{window.pressedAt=performance.now();})")
            frames = CanvasFrames(page)
            delayed = SimpleNamespace(active=False)

            class Eyes(WhatMoves):
                def see(self, *args, **kwargs):
                    if delayed.active:
                        time.sleep(.32)
                    return super().see(*args, **kwargs)

            observation = await open_play_observation(factory=lambda **options: Eyes(**options),
                                                       frame_wait_s=.4)
            assert observation is not None
            how = HowItWorks(me=Answer(STEERED, 1.), body_keys_from_code=('j', 'k'))
            run = SimpleNamespace(how=how, keys=['j', 'k'], trying=8, held='', held_since=0.,
                held_trying=False, presses=WhatAPressDoes(), things={}, input_key_downs=Counter(),
                responses=[], letting_go=[], charging=SimpleNamespace(key=''), intervals=[.1] * 6,
                picture_at=None, pictures=0)
            hers = WhichIsHers()
            inputs = SimpleNamespace(down=lambda key: page.keyboard.down('Key' + key.upper()),
                                     up=lambda key: page.keyboard.up('Key' + key.upper()))
            hands = search_inputs(inputs, run, hers)

            async def look():
                return await frames.look({'x': 0, 'y': 0, 'width': 240, 'height': 160})

            origin = None
            for _ in range(7):
                await asyncio.sleep(.1)
                origin = await look()
                assert origin is not None
                await observation.see(*origin)
            assert hers.number is None
            if mode in ('slow', 'slow-phase'):
                delayed.active = True
                run.intervals = [.18] * 6
            started = time.monotonic()
            result = await search_after_unresolved_body(hands, run, hers, observation, look=look,
                context=SimpleNamespace(check=lambda: None), picture=origin[0], at=origin[1], deadline=started + 45.)
            assert result, (observation.seeing, eligible_pixel_search(run, hers, observation, origin[1], 45.),
                            vars(run._pixel_search), vars(origin[1]))
            state = run._pixel_search
            if mode in ('phase', 'slow', 'slow-phase'):
                assert state.release_confirmed and run.held == ''
                assert not state.hint_applied and state.region is None and not state.body_identified
                assert state.status == ('capture_or_delivery_timeout' if mode == 'slow' else 'observed phase change'), vars(state)
                assert state.phase_changed == (mode != 'slow')
                assert (run.picture_at.surface, run.picture_at.epoch) == (origin[1].surface, origin[1].epoch)
                assert observation.status != 'closed'
                delayed.active = False
                assert await observation.see(*await look()) is not None
                delivered = await page.evaluate('window.delivered')
                assert delivered and len(delivered) % 2 == 0
                assert all(delivered[i][0] == 'down' and delivered[i+1] == ['up', delivered[i][1]]
                           for i in range(0, len(delivered), 2))
                return
            if mode == 'no-effect':
                assert state.status == 'complete', vars(state)
                assert state.trials == 24 and not state.body_identified and hers.number is None
                assert not state.hint_applied and state.region is None
                assert state.release_confirmed and run.held == ''
                delivered = await page.evaluate('window.delivered')
                assert len(delivered) == 48
                return
            assert state.status == 'observation_resolved', vars(state)
            assert state.body_identified and hers.number is not None
            assert state.observed_frames >= 10 and not state.hint_applied and state.region is None
            assert state.release_confirmed and run.held == ''
            delivered = await page.evaluate('window.delivered')
            assert len(delivered) < 48 and len(delivered) % 2 == 0
            assert all(delivered[i][0] == 'down' and delivered[i+1] == ['up', delivered[i][1]]
                       for i in range(0, len(delivered), 2))
            receipt = hers.identification.receipts[-1]
            assert receipt['reason'] == 'independent repeated trials'
            assert sum(count >= 2 for count in receipt['independent_trials'].values()) >= 2
        finally:
            if observation is not None:
                await observation.close()
            if frames is not None:
                await frames.close()
            await browser.close()
