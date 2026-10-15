"""A remembered grid cannot erase a current captured pointer affordance."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.perception.observed_transfer import PixelSnapshot, ScreenReading
from core.perception.where_am_i import _pointer_target_frame_invariant, where_am_i


def _scene():
    region = dict(text='the one that stands out at 40% across, 30% down', x=.35, y=.25,
                  width=.1, height=.1, center_x=.4, center_y=.3, answers_the_pointer=True)
    return ScreenReading(dict(ok=True, shapes=[region], layout=[], surface_id='page:current',
                              bounds=[0, 0, 1000, 700], capture_at=3., _capture_epoch='current', settled=True),
                         picture_snapshot=PixelSnapshot(bytes(12*16*3), 12, 16))


def _glance():
    return SimpleNamespace(rows=32, columns=31, occupied=lambda: 71)


def _lattice():
    return SimpleNamespace(rows=2, columns=2, held=True, would_not_fit=0, not_here_for=0)


def test_current_target_is_independent_of_a_different_inferred_scene_grid():
    seen = _scene()
    label = seen['shapes'][0]['text']
    assert where_am_i(_glance(), lattice=_lattice(), observation=seen, aimed_at=label).the_thing_is_here
    assert not where_am_i(_glance(), lattice=_lattice(), observation=seen).the_thing_is_here
    assert _pointer_target_frame_invariant() == ()


@pytest.mark.parametrize('problem', ['no_pixels', 'no_scope', 'no_epoch', 'unsettled', 'occluded', 'refused',
                                  'no_time', 'invalid_bounds', 'hidden', 'not_a_hit', 'ambiguous', 'duplicate',
                                  'invalid_anchor', 'old_label'])
def test_only_a_unique_current_captured_hit_can_override_the_grid(problem):
    seen = _scene()
    label, region = seen['shapes'][0]['text'], seen['shapes'][0]
    if problem == 'no_pixels':
        seen = dict(seen)
    elif problem == 'no_scope':
        seen.pop('surface_id')
    elif problem == 'no_epoch':
        seen.pop('_capture_epoch')
    elif problem == 'unsettled':
        seen['settled'] = False
    elif problem == 'occluded':
        seen['occluded'] = True
    elif problem == 'refused':
        seen['ok'] = False
    elif problem == 'no_time':
        seen.pop('capture_at')
    elif problem == 'invalid_bounds':
        seen['bounds'][2] = 0
    elif problem == 'hidden':
        seen['her_window_showing'] = False
    elif problem == 'not_a_hit':
        region.pop('answers_the_pointer')
    elif problem == 'ambiguous':
        region['pointer_ambiguous'] = True
    elif problem == 'duplicate':
        seen['shapes'].append(dict(region))
    elif problem == 'invalid_anchor':
        region.update(pointer_x=float('nan'), pointer_y=.3)
    else:
        label = 'the old room'
    assert not where_am_i(_glance(), lattice=_lattice(), observation=seen, aimed_at=label).the_thing_is_here


@pytest.mark.asyncio
async def test_the_normal_actor_reaches_current_click_delivery_with_the_grid_still_held(monkeypatch):
    from core.skills import screen_pursuit_acting as acting

    seen = _scene()
    key = 'click "'+seen['shapes'][0]['text']+'"'
    run = SimpleNamespace(key=key, observation=seen, about_to={'key': key}, anchor={'app': 'test'}, at_rest={},
        busy=Mock(), expected={'took': 1, 'after': None}, follow_on=[], goal='explore', in_flight=Mock(),
        laid_out=_glance(), lattice=_lattice(), made=None, moves=[], narrate=False, pacing={'brief': True},
        pending={'watched': seen}, responds={'state': Mock(), 'lattice': _lattice()}, target_app='test', world=Mock())

    class _ReachedPortError(Exception):
        pass

    click = AsyncMock(side_effect=_ReachedPortError())
    monkeypatch.setattr(acting, '_click_what_she_named', click)
    monkeypatch.setattr(acting, '_say_intent', Mock())
    with pytest.raises(_ReachedPortError):
        await acting.carry_out_the_move(run)
    assert click.await_count == 1 and run.lattice.held
    assert not run.moves  # Reaching the port is no delivery or success receipt.
