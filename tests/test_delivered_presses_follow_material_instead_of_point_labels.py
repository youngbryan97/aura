"""A changed coordinate label does not make a delivered material press new."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception import presses_on_the_same_material as memory
from core.perception.a_cursor_target_in_its_pixels import material_at
from core.perception.observed_transfer import ScreenReading, make_snapshot


def _target(seed=41, point=(.5, .5)):
    page = SimpleNamespace(viewport_size={'width': 240, 'height': 160}, url='test surface')
    image = np.random.default_rng(seed).integers(0, 256, (160, 240, 3), np.uint8)
    return material_at(image, point, page=page, bounds=(0., 0., 240., 160.))


def test_a_delivered_point_is_recognized_after_its_coordinate_and_label_change():
    target = _target()
    held = memory.MaterialPressHistory()
    assert held.count(target) == 0
    held.delivered(target)
    current = replace(target, sample_point=(170., 110.), captured=time.monotonic())
    assert held.count(current) == 1
    held.delivered(current)
    assert held.count(current) == 2 and len(held.entries) == 1


@pytest.mark.parametrize('change', ['address', 'viewport', 'bounds', 'page', 'pixels', 'point', 'old', 'future', 'flat'])
def test_different_or_unmeasured_material_does_not_inherit_a_delivered_press(change):
    target = _target()
    held = memory.MaterialPressHistory()
    held.delivered(target)
    if change == 'address':
        current = replace(target, address='different')
    elif change == 'viewport':
        current = replace(target, viewport=(480, 320))
    elif change == 'bounds':
        current = replace(target, bounds=(0., 0., 120., 80.))
    elif change == 'page':
        current = replace(target, page_id=target.page_id+1)
    elif change == 'pixels':
        current = replace(target, patch=_target(42).patch)
    elif change == 'point':
        current = replace(target, anchor=(target.anchor[0]+8., target.anchor[1]))
    elif change == 'old':
        current = replace(target, captured=time.monotonic()-memory.MAX_AGE_S-1)
    elif change == 'future':
        current = replace(target, captured=time.monotonic()+1)
    else:
        current = replace(target, patch=bytes(len(target.patch)))
    assert not held.count(current)


def test_repeated_local_witnesses_are_ambiguous_and_memory_is_bounded():
    target = _target()
    now = time.monotonic()
    held = memory.MaterialPressHistory(((target, 1, now), (target, 2, now)))
    assert held.count(target) is None
    held = memory.MaterialPressHistory()
    for i in range(12):
        held.delivered(replace(target, anchor=(20.+i*4, 48.)))
    assert len(held.entries) == memory.MOST_REFERENCES
    held = memory.MaterialPressHistory(((target, memory.MOST_PRESSES, now),))
    held.delivered(target)
    assert held.count(target) == memory.MOST_PRESSES


def test_historical_pixels_can_inform_exploration_without_becoming_a_fresh_action_reference():
    current = _target()
    now = time.monotonic()
    old = replace(current, captured=now-90.)
    held = memory.MaterialPressHistory(((old, 2, now-89.),))
    assert held.count(old) is None
    assert held.count(current) == 2
    held = memory.MaterialPressHistory(((old, 2, now-memory.MAX_HISTORY_AGE_S-1.),))
    assert held.count(current) == 0


def _reading(label):
    return ScreenReading(dict(ok=True, text='', shapes=[dict(text=label, center_x=.4, center_y=.6,
        answers_the_pointer=True)], layout=[], surface_id='one surface', bounds=[0., 0., 240., 160.],
        capture_at=time.time(), _capture_epoch='one capture'),
        picture_snapshot=make_snapshot(np.random.default_rng(4).integers(0, 256, (160, 240, 3), np.uint8)))


@pytest.mark.parametrize('change', ['none', 'dictionary', 'capture', 'surface', 'bounds', 'pixels', 'epoch', 'label', 'point', 'named'])
def test_a_material_cost_requires_the_same_private_reading_and_an_anonymous_current_point(change):
    label = 'the one that stands out at 40% across, 60% down'
    seen = _reading(label)
    memory.bind_material_presses(seen, {label: 2})
    if change == 'dictionary':
        seen = dict(seen, material_presses={label: 2})
    elif change == 'capture':
        seen['capture_at'] += 1.
    elif change == 'surface':
        seen['surface_id'] = 'other'
    elif change == 'bounds':
        seen['bounds'] = [0., 0., 120., 80.]
    elif change == 'pixels':
        seen.picture_snapshot = make_snapshot(np.zeros((160, 240, 3), np.uint8))
    elif change == 'epoch':
        seen['_capture_epoch'] = 'other'
    elif change == 'label':
        seen['shapes'][0]['text'] = 'the one that stands out at 41% across, 60% down'
    elif change == 'point':
        seen['shapes'][0]['center_x'] = .8
    elif change == 'named':
        seen['shapes'][0]['text'] = 'Continue'
    assert memory.current_material_presses(seen) == ({f'click "{label}"': 2} if change == 'none' else {})


def test_a_material_cost_changes_exploration_without_removing_a_move_or_creating_scene_identity():
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.skills.screen_pursuit_decision_lesson import _first_what_goes_on, _never_tried_here

    old, new = 'click "the one that stands out at 40% across, 60% down"', 'click "a fresh shape"'
    can_do = WhatWorksHere(pointer_said=True)
    can_do.material_presses = {old: 2}
    options = [SimpleNamespace(name=old), SimpleNamespace(name=new)]
    assert _never_tried_here(options, SimpleNamespace(tried=set()), can_do) == [new]
    valued = _first_what_goes_on(can_do, {old: 1., new: 1.})
    assert 0 < valued[old] < valued[new] and set(valued) == {old, new}
    assert can_do.leads.screens == [] and not can_do.leads.selector_signatures
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'delivered_material_costs_require_the_current_capture')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.asyncio
async def test_normal_page_click_memory_reaches_exploration_after_a_visible_layout_change(monkeypatch, tmp_path):
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.perception.a_cursor_target_in_its_pixels import target_picture
    from core.perception.recording_the_frames_and_inputs import record_frames_and_inputs
    from core.skills.screen_pursuit_decision_lesson import _first_what_goes_on, _never_tried_here
    from core.skills.screen_pursuit_looking import _take_in_the_screen
    from core.skills.screen_pursuit_on_a_page import OnAPage

    api = pytest.importorskip('playwright.async_api')
    tile = np.random.default_rng(61).integers(0, 256, (64, 128, 4), np.uint8)
    tile[..., 3] = 255
    encoded = base64.b64encode(tile.tobytes()).decode()
    html = '''<style>body{margin:0}canvas{display:block}</style><canvas width=320 height=240></canvas>
    <script>
    const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');
    const tile=document.createElement('canvas');tile.width=128;tile.height=64;
    tile.getContext('2d').putImageData(new ImageData(Uint8ClampedArray.from(atob('TEXTURE'),c=>c.charCodeAt(0)),128,64),0,0);
    let dx=0,dy=0;window.hits=[0,0];
    function draw(){ctx.fillStyle='rgb(45,45,45)';ctx.fillRect(0,0,320,240);
      ctx.drawImage(tile,0,0,64,64,48+dx,48+dy,64,64);
      ctx.drawImage(tile,64,0,64,64,198+dx,108+dy,64,64);}
    function hit(x,y){return Math.hypot(x-80-dx,y-80-dy)<12?0:Math.hypot(x-230-dx,y-140-dy)<12?1:-1;}
    canvas.onpointermove=e=>canvas.style.cursor=hit(e.offsetX,e.offsetY)>=0?'pointer':'auto';
    canvas.onpointerdown=e=>{const k=hit(e.offsetX,e.offsetY);if(k>=0)window.hits[k]++;};
    window.onkeydown=()=>{dx=15;dy=12;draw();};draw();
    </script>'''.replace('TEXTURE', encoded)
    async with api.async_playwright() as running:
        browser = await running.chromium.launch(headless=True, args=['--disable-gpu'])
        try:
            page = await browser.new_page(viewport={'width': 320, 'height': 240})
            await page.set_content(html)
            surface = OnAPage(page, 'pixel affordance fixture')
            supplied = [(80., 80.), (230., 140.)]

            async def reading(*_args, **_kwargs):
                rgb = await target_picture(page, (0., 0., 320., 240.))
                return ScreenReading(dict(ok=True, text='', layout=[], shapes=[], bounds=[0., 0., 320., 240.],
                    surface_id='fixture', capture_at=time.time(), _capture_epoch=str(time.monotonic_ns()),
                    settled=True), picture_snapshot=make_snapshot(rgb[..., ::-1]))

            async def sweep(*args, **kwargs):
                # This fixture supplies two initial hand affordances. It tests
                # material/action memory rather than discovery of those points.
                for x, y in supplied:
                    await page.mouse.move(x, y)
                    await kwargs['on_sample'](x, y, 'pointer')
                return [(x/320, y/240) for x, y in supplied]

            monkeypatch.setattr('core.perception.where_the_pointer_turns.where_the_pointer_turns', sweep)
            monkeypatch.setattr('core.perception.cursor_materials_after_inspection.look_at_a_page', reading)
            seen = await reading()
            await surface._where_the_pointer_turns(seen, None)
            first_label = seen['shapes'][0]['text']
            first = seen['shapes'][0]
            assert await surface.click(first['center_x'], first['center_y'], seen['bounds'])
            await page.keyboard.press('ArrowRight')  # Independent visible layout change.
            supplied[:] = [(95., 92.), (245., 152.)]
            seen = await reading()
            await surface._where_the_pointer_turns(seen, None)
            assert seen['shapes'][0]['text'] != first_label
            can_do = WhatWorksHere(pointer_said=True)
            directory = tmp_path/'material-action-memory'
            async with record_frames_and_inputs(directory):
                _take_in_the_screen(can_do, seen, True, False)
                options = [SimpleNamespace(name=f'click "{region["text"]}"') for region in seen['shapes']]
                costs = memory.current_material_presses(seen)
                assert list(costs.values()) == [1]
                assert _never_tried_here(options, SimpleNamespace(tried=set()), can_do) == [options[1].name]
                weights = _first_what_goes_on(can_do, {option.name: 1. for option in options})
                chosen = max(weights, key=weights.__getitem__)
                assert chosen == options[1].name
                second = seen['shapes'][1]
                def cannot_keep(_target):
                    raise ValueError('optional history unavailable after physical delivery')

                monkeypatch.setattr(surface._material_presses, 'delivered', cannot_keep)
                assert await surface.click(second['center_x'], second['center_y'], seen['bounds'])
            assert await page.evaluate('hits') == [1, 1]
            events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
            offered = next(event for event in events if event['kind'] == 'screen_options')
            assert offered['material_press_costs'] == costs
            completed = next(event for event in events if event['kind'] == 'navigation_input' and event['phase'] == 'completed')
            assert completed['delivered'] is True
        finally:
            await browser.close()
