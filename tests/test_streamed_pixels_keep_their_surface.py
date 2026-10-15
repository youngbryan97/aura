"""Stream capture order survives crops, navigation and private workers."""
import asyncio
import copy
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from core.agency.pixels_inside_control_trials import freeze_frame
from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.perception.frames_as_they_are_drawn import PageFrames
from core.perception.streamed_pixels_keep_their_surface import StreamedSurface, captured_crop

META = dict(deviceWidth=80, deviceHeight=60, scrollOffsetX=0, scrollOffsetY=0, offsetTop=0)
CLIP = dict(x=0., y=0., width=80., height=60.)


def clock(at=1., *, lower=1.):
    return MeasuredFrameTime(at, lower=lower, upper=at + .01, arrived=at + .02,
                             source="browser-wall", source_at=1000. + at, epoch=2)


def test_each_actual_frame_keeps_distinct_sequence_and_original_source_bounds():
    source = StreamedSurface()
    first = captured_crop(source.frame(clock()), META, CLIP, (60, 80, 3))
    second = captured_crop(source.frame(clock(1.1, lower=1.1)), META, CLIP, (60, 80, 3))
    assert first.sequence == 1 and second.sequence == 2 and first.surface == second.surface
    assert (first.lower, first.upper, first.arrived, first.source_at) == (1., 1.01, 1.02, 1001.)
    frozen = freeze_frame(np.zeros((60, 80, 3), np.uint8), first)
    assert frozen is not None and frozen.surface == first.surface
    for restored in (copy.deepcopy(first), pickle.loads(pickle.dumps(first))):
        assert (restored.surface, restored.sequence, restored.epoch, restored.lower) == (first.surface, 1, 2, 1.)


def test_new_sessions_documents_scroll_or_crops_do_not_continue_old_material():
    first_source, second_source = StreamedSurface(), StreamedSurface()
    original = captured_crop(first_source.frame(clock()), META, CLIP, (60, 80, 3))
    assert second_source.frame(clock()).surface not in original.surface
    first_source.navigated({"frame": {"id": "child", "parentId": "main"}})
    same = captured_crop(first_source.frame(clock()), META, CLIP, (60, 80, 3))
    assert same.surface == original.surface
    first_source.navigated({"frame": {"id": "main"}})
    assert captured_crop(first_source.frame(clock()), META, CLIP, (60, 80, 3)).surface != original.surface
    frame = second_source.frame(clock())
    same = captured_crop(frame, META, CLIP, (60, 80, 3))
    assert captured_crop(frame, {**META, "scrollOffsetY": 4}, CLIP, (60, 80, 3)).surface != same.surface
    assert captured_crop(frame, META, {**CLIP, "x": 4}, (60, 80, 3)).surface != same.surface


def test_unknown_clock_or_crop_metadata_cannot_supply_causal_capture_custody():
    source = StreamedSurface()
    frame = source.frame(clock(lower=None))
    assert frame.lower is None and not hasattr(frame, "sequence")
    frame = source.frame(clock())
    assert not hasattr(captured_crop(frame, {}, CLIP, (60, 80, 3)), 'surface')
    assert not hasattr(captured_crop(frame, META, {**CLIP, "width": float("nan")}, (60, 80, 3)), 'surface')
    assert captured_crop(1., META, CLIP, (60, 80, 3)) == 1.


def test_queued_frame_from_before_document_change_keeps_no_new_owner():
    source = StreamedSurface()
    source.navigated({'frame': {'id': 'main'}}, at=2.)
    old = source.frame(clock())
    assert not hasattr(old, 'surface') and old.lower == 1.
    current = source.frame(clock(2.1, lower=2.1))
    assert current.surface == source.owner


@pytest.mark.asyncio
async def test_actual_screencast_event_is_consumed_once_with_its_cropped_owner(monkeypatch):
    from core.perception import frames_as_they_are_drawn as frames

    stream = PageFrames(SimpleNamespace())
    session = stream._session = SimpleNamespace(send=AsyncMock(), detach=AsyncMock())
    stream._generation = 2
    stream._clock.epoch = 2
    stream._clock.correlate("wall", 1000., 10., 10.01)
    stream._surface = StreamedSurface()
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: 10.3))
    monkeypatch.setattr(frames, "STILL_AFTER_S", .001)
    async def decode(*_args):
        return np.zeros((60, 80, 3), np.uint8)
    monkeypatch.setattr(frames, "run_interactive_cpu", decode)
    stream._frame({"data": "captured-data", "metadata": {**META, "timestamp": 1000.2}, "sessionId": 7},
                  source_session=session, generation=2)
    image, at = await stream.look(CLIP)
    assert freeze_frame(image, at) is not None
    assert at.lower == pytest.approx(10.2) and at.sequence == 1 and '/crop/' in at.surface
    assert await stream.look(CLIP) is None
    stream._frame({"data": "old-source", "metadata": {**META, "timestamp": 1000.25}, "sessionId": 8},
                  source_session=object(), generation=1)
    assert stream._surface.sequence == 1
    await stream.close()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_real_chromium_stream_keeps_physical_frame_order_and_document_generation():
    from playwright.async_api import async_playwright

    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 80, "height": 60})
            await page.set_content('<canvas width="80" height="60"></canvas>')
            await page.evaluate("""() => {
                const ctx=document.querySelector('canvas').getContext('2d');let n=0;
                const draw=()=>{ctx.fillStyle=`rgb(${n++%200},80,120)`;ctx.fillRect(0,0,80,60);requestAnimationFrame(draw);};draw();
            }""")
            stream = PageFrames(page)
            async def fresh():
                for _ in range(12):
                    result = await stream.look(CLIP)
                    if result is not None:
                        return result
                raise AssertionError("actual streamed capture remained unavailable")
            first = await asyncio.wait_for(fresh(), 5.)
            second = await asyncio.wait_for(fresh(), 5.)
            a, b = first[1], second[1]
            assert freeze_frame(first[0], a) is not None and freeze_frame(second[0], b) is not None
            assert a.surface == b.surface and b.sequence > a.sequence and b.lower > a.lower
            assert a.source in {'browser-monotonic', 'browser-wall'}
            await page.goto('data:text/html,<body style="background:rgb(80,40,20)">next document</body>')
            after = await asyncio.wait_for(fresh(), 5.)
            assert after[1].surface != b.surface
            await stream.close()
        finally:
            await browser.close()
