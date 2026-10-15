"""Pointer-induced view movement is corrected from pictures, without game state."""
from __future__ import annotations

import asyncio
import base64
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from core.perception import a_cursor_target_in_its_pixels as pointing


class _MovingView:
    viewport_size = {'width': 320, 'height': 240}
    url = 'https://example.invalid/view'

    def __init__(self, response=((.4, .1), (-.1, .3))):
        self.response = np.asarray(response)
        self.at = np.array((80., 80.))
        self.mouse = SimpleNamespace(move=self.move)
        self.moves = []
        self.material = np.random.default_rng(61).integers(0, 256, (64, 64, 3), np.uint8)
        self.origin = np.array((160., 120.))
        self.lost = self.duplicate = False
        self.pictures = 0

    def position(self):
        return self.origin+self.response @ (self.at-(80., 80.))

    def picture(self):
        self.pictures += 1
        image = np.full((240, 320, 3), 45, np.uint8)
        if not self.lost:
            x, y = self.position()
            seed = np.zeros_like(image)
            seed[88:152, 128:192] = self.material
            shifted = ndimage.shift(seed.astype(float), (y-120, x-160, 0), order=1)
            mask = shifted.max(axis=2) > 0
            image[mask] = shifted[mask].astype(np.uint8)
            if self.duplicate:
                image[20:84, 20:84] = self.material
        return image

    async def move(self, x, y):
        self.at = np.array((x, y))
        self.moves.append((x, y))

    async def evaluate(self, script, at):
        return 'pointer' if np.linalg.norm(np.asarray(at)-self.position()) < 12 else 'auto'


def _target(page):
    return pointing.material_at(page.picture(), (.5, .5), page=page, bounds=(0., 0., 320., 240.))


def test_feedback_invariant():
    assert pointing._pointer_feedback_invariant() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [((0., 0.), (0., 0.)), ((.4, .1), (-.1, .3)),
                                     ((-.6, .2), (.1, -.4))])
async def test_repeated_pictures_close_pointer_to_material_even_when_hover_moves_the_view(monkeypatch, response):
    page = _MovingView(response)
    target = _target(page)

    async def picture(page, bounds):
        return page.picture()

    monkeypatch.setattr(pointing, 'target_picture', picture)
    monkeypatch.setattr(pointing, 'TARGET_WITHIN_S', 8.)
    landed = await pointing.arrive_at_cursor_target(page, target, (.5, .5), check=lambda: None)
    assert landed is not None
    assert np.linalg.norm(page.at-page.position()) <= 1.5
    assert page.pictures >= 2 and 1 <= len(page.moves) <= pointing.MAX_CORRECTIONS


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['lost', 'duplicate', 'address', 'viewport', 'unreachable'])
async def test_lost_ambiguous_changed_or_unreachable_material_does_not_authorize_a_press(monkeypatch, failure):
    page = _MovingView()
    target = _target(page)
    if failure in ('lost', 'duplicate'):
        setattr(page, failure, True)
    elif failure == 'address':
        page.url += '/other'
    elif failure == 'viewport':
        page.viewport_size = {'width': 640, 'height': 480}
    else:
        page.response = np.eye(2)

    async def picture(page, bounds):
        return page.picture()

    monkeypatch.setattr(pointing, 'target_picture', picture)
    assert await pointing.arrive_at_cursor_target(page, target, (.5, .5), check=lambda: None) is None


@pytest.mark.asyncio
async def test_external_cancellation_still_stops_feedback(monkeypatch):
    page = _MovingView()
    target = _target(page)
    entered = asyncio.Event()

    async def stalled(page, bounds):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(pointing, 'target_picture', stalled)
    task = asyncio.create_task(pointing.arrive_at_cursor_target(page, target, (.5, .5), check=lambda: None))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_nearby_hand_cannot_replace_lost_material_in_the_normal_click_port(monkeypatch):
    from core.skills.screen_pursuit_on_a_page import OnAPage, _CursorSweep

    page = _MovingView()
    target = _target(page)
    page.lost = True
    pressed = []

    async def down():
        pressed.append(1)

    async def other_hand(*args):
        return 'pointer'

    async def picture(page, bounds):
        return page.picture()

    page.mouse.down = down
    page.evaluate = other_hand
    surface = OnAPage(page, 'fixture')
    surface._cursor_sweeps[(0., 0., 320., 240.)] = _CursorSweep((), ((.5, .5),), (target,))
    monkeypatch.setattr(pointing, 'target_picture', picture)
    assert not await surface.click(.5, .5, [0, 0, 320, 240])
    assert not pressed and not surface._cursor_sweeps


def test_a_reduced_page_reading_preserves_css_coordinates_of_the_sampled_target():
    from PIL import Image

    page = _MovingView(((0., 0.), (0., 0.)))
    target = _target(page)
    reduced = np.asarray(Image.fromarray(page.picture()).resize((240, 180), Image.Resampling.BOX))
    assert pointing.target_in_picture(target, reduced) == pytest.approx((160., 120.), abs=1.)


@pytest.mark.asyncio
async def test_a_feedback_deadline_returns_no_press_authority(monkeypatch):
    page = _MovingView()
    target = _target(page)

    async def stalled(page, bounds):
        await asyncio.Event().wait()

    monkeypatch.setattr(pointing, 'target_picture', stalled)
    monkeypatch.setattr(pointing, 'TARGET_WITHIN_S', .02)
    assert await pointing.arrive_at_cursor_target(page, target, (.5, .5), check=lambda: None) is None


@pytest.mark.asyncio
async def test_chromium_normal_page_port_corrects_the_same_material_and_presses_without_another_move(monkeypatch, tmp_path):
    import json

    from core.perception.observed_transfer import ScreenReading, make_snapshot
    from core.perception.recording_the_frames_and_inputs import record_frames_and_inputs
    from core.skills.screen_pursuit_on_a_page import OnAPage

    api = pytest.importorskip('playwright.async_api')
    texture = np.random.default_rng(32).integers(0, 256, (64, 64, 4), np.uint8)
    texture[..., 3] = 255
    encoded = base64.b64encode(texture.tobytes()).decode()
    html = '''<style>body{margin:0}canvas{display:block}</style><canvas width=320 height=240></canvas>
    <script>
    const canvas=document.querySelector('canvas'), ctx=canvas.getContext('2d');
    const tile=document.createElement('canvas');tile.width=tile.height=64;
    tile.getContext('2d').putImageData(new ImageData(Uint8ClampedArray.from(atob('TEXTURE'),c=>c.charCodeAt(0)),64,64),0,0);
    let x=80,y=80,cx=160,cy=120;window.receipts=[];
    function draw(){cx=160+.4*(x-80)+.1*(y-80);cy=120-.1*(x-80)+.3*(y-80);
      ctx.fillStyle='rgb(45,45,45)';ctx.fillRect(0,0,320,240);ctx.drawImage(tile,cx-32,cy-32);
      canvas.style.cursor=Math.hypot(x-cx,y-cy)<12?'pointer':'auto';}
    canvas.addEventListener('pointermove',e=>{x=e.offsetX;y=e.offsetY;draw();
      window.receipts.push({kind:'move',x,y});});
    canvas.addEventListener('pointerdown',()=>window.receipts.push({kind:'down',x,y,hit:Math.hypot(x-cx,y-cy)<12}));
    canvas.addEventListener('pointerup',()=>window.receipts.push({kind:'up',x,y}));draw();
    </script>'''.replace('TEXTURE', encoded)
    async with api.async_playwright() as running:
        browser = await running.chromium.launch(headless=True, args=['--disable-gpu'])
        try:
            page = await browser.new_page(viewport={'width': 320, 'height': 240})
            await page.set_content(html)
            surface = OnAPage(page, 'test surface')

            async def reading(*_args, **_kwargs):
                rgb = await pointing.target_picture(page, (0., 0., 320., 240.))
                return ScreenReading(dict(ok=True, text='', layout=[], shapes=[], bounds=[0., 0., 320., 240.],
                                          capture_at=time.time(), _capture_epoch=str(time.monotonic_ns()), settled=True,
                                          surface_id='fixture'), picture_snapshot=make_snapshot(rgb[..., ::-1]))

            # A fixture starts with one independently acquired hand affordance.
            # The actor receives neither the view response matrix nor the hit test.
            fixed = np.linalg.solve(np.eye(2)-np.array(((.4, .1), (-.1, .3))),
                                    np.array((160., 120.))-np.array(((.4, .1), (-.1, .3))) @ (80., 80.))

            async def sweep(*args, **kwargs):
                return [(fixed[0]/320, fixed[1]/240)]

            monkeypatch.setattr('core.perception.where_the_pointer_turns.where_the_pointer_turns', sweep)
            monkeypatch.setattr('core.perception.cursor_materials_after_inspection.look_at_a_page', reading)
            seen = await reading()
            await surface._where_the_pointer_turns(seen, None)
            assert len(seen['shapes']) == 1
            selected = seen['shapes'][0]
            await page.mouse.move(80, 80)  # Later inspection leaves the pointer elsewhere.
            directory = tmp_path/'pointing'
            async with record_frames_and_inputs(directory):
                assert await surface.click(selected['center_x'], selected['center_y'], seen['bounds'])
            receipts = await page.evaluate('window.receipts')  # Assertions only, after the actor stopped.
            down = next(i for i, receipt in enumerate(receipts) if receipt['kind'] == 'down')
            assert receipts[down]['hit']
            assert receipts[down+1]['kind'] == 'up'
            assert receipts[down-1]['kind'] == 'move'
            assert (receipts[down-1]['x'], receipts[down-1]['y']) == (receipts[down]['x'], receipts[down]['y'])
            events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
            assert events[-1]['delivered'] is True
            assert events[0]['args'] == pytest.approx((receipts[down]['x'], receipts[down]['y']), abs=.6)
        finally:
            await browser.close()
