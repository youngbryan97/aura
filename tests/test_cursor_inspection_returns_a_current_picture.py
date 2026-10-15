"""Active inspection cannot attach old words or unobserved points to a new view."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception import cursor_materials_after_inspection as inspecting
from core.perception.a_cursor_target_in_its_pixels import material_at
from core.perception.observed_transfer import ScreenReading, make_snapshot


def _reading(image, *, captured=1., address='page', text='before inspection'):
    return ScreenReading(dict(ok=True, text=text, layout=[], shapes=[], surface_id=address,
                              bounds=[0., 0., 240., 160.], capture_at=captured, _capture_epoch=str(captured),
                              settled=True), picture_snapshot=make_snapshot(image[..., ::-1]))


@pytest.mark.asyncio
async def test_a_hover_material_is_relocated_in_the_fresh_reading_and_old_text_is_retired(monkeypatch):
    before = np.full((160, 240, 3), 60, np.uint8)
    before[40:104, 60:124] = np.random.default_rng(51).integers(0, 256, (64, 64, 3), np.uint8)
    after = np.roll(before, (10, 24), axis=(0, 1))
    page = SimpleNamespace(viewport_size={'width': 240, 'height': 160}, url='page')
    seen = _reading(before)

    async def sampled(*args):
        return material_at(before, (.4, .45), page=page, bounds=(0., 0., 240., 160.))

    async def latest(*args, **kwargs):
        return _reading(after, captured=time.time(), text='after inspection')

    monkeypatch.setattr(inspecting, 'sample_cursor_target', sampled)
    monkeypatch.setattr(inspecting, 'look_at_a_page', latest)
    points, materials = await inspecting.cursor_materials_after_inspection(
        page, seen, ((.4, .45),), (0., 0., 240., 160.), over=None, name='fixture')
    assert len(points) == len(materials) == 1
    assert points[0] == pytest.approx((.5, .5125), abs=.005)
    assert seen['text'] == 'after inspection'
    assert seen.picture_snapshot == make_snapshot(after[..., ::-1])
    assert 'materials' not in seen


@pytest.mark.asyncio
async def test_finished_material_samples_are_not_reacquired_by_later_pointer_movements(monkeypatch):
    image = np.random.default_rng(44).integers(0, 256, (160, 240, 3), np.uint8)
    page = SimpleNamespace(viewport_size={'width': 240, 'height': 160}, url='page')
    seen = _reading(image)
    captured = material_at(image, (.5, .5), page=page, bounds=(0., 0., 240., 160.))

    async def must_not_resample(*args):
        pytest.fail('the physical sample was replaced with a later hover')

    async def latest(*args, **kwargs):
        return _reading(image, captured=2.)

    monkeypatch.setattr(inspecting, 'sample_cursor_target', must_not_resample)
    monkeypatch.setattr(inspecting, 'look_at_a_page', latest)
    points, materials = await inspecting.cursor_materials_after_inspection(
        page, seen, ((.5, .5),), (0., 0., 240., 160.), over=None, name='fixture', sampled_targets=(captured,))
    assert len(points) == 1 and materials == (captured,)


@pytest.mark.asyncio
async def test_the_collector_reads_under_the_current_pointer_without_moving_it(monkeypatch):
    from core.perception import a_cursor_target_in_its_pixels as material

    page = SimpleNamespace(viewport_size={'width': 240, 'height': 160}, url='page')
    at, events = (120., 80.), []

    async def evaluate(script, point):
        assert tuple(point) == at
        events.append('cursor')
        return 'pointer'

    async def picture(*args):
        events.append('picture')
        return np.random.default_rng(11).integers(0, 256, (160, 240, 3), np.uint8)

    page.evaluate = evaluate
    monkeypatch.setattr(material, 'target_picture', picture)
    collector = inspecting.CursorMaterialCollector(page, (0., 0., 240., 160.))
    await collector.sampled(*at, 'pointer')
    assert len(collector.targets) == 1
    assert collector.targets[0].sample_point == at
    assert events == ['cursor', 'picture', 'cursor']


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['address', 'viewport', 'capture', 'unavailable', 'surface'])
async def test_failed_inspection_reconciliation_cannot_return_an_old_usable_reading(monkeypatch, change):
    image = np.full((160, 240, 3), 30, np.uint8)
    seen = _reading(image)
    page = SimpleNamespace(viewport_size={'width': 240, 'height': 160}, url='page')

    async def latest(*args, **kwargs):
        if change == 'address':
            page.url = 'elsewhere'
        elif change == 'viewport':
            page.viewport_size['width'] = 480
        if change == 'unavailable':
            return None
        return _reading(image, captured=1. if change == 'capture' else 2.,
                        address='other' if change == 'surface' else 'page')

    monkeypatch.setattr(inspecting, 'look_at_a_page', latest)
    assert await inspecting.cursor_materials_after_inspection(
        page, seen, (), (0., 0., 240., 160.), over=None, name='fixture') == ((), ())
    assert seen['ok'] is False and seen['text'] == '' and seen.picture_snapshot is None
