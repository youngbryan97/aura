"""A qualified input surface can change its scene without losing capture custody."""
import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.a_capture_owned_by_the_input_surface import (
    InputSurfaceCaptures, _owned_input_surface_invariant,
    current_input_surface_is_qualified, qualify_current_input_surface,
)
from core.perception.observed_transfer import ScreenReading, make_snapshot
from core.perception.where_am_i import where_am_i

pytestmark = pytest.mark.unit


def observed(at=10., *, surface='surface', epoch='a', pixels=0, **changes):
    values = dict(ok=True, surface_id=surface, bounds=[0, 0, 100, 100], capture_at=at,
                  _capture_epoch=epoch, _work_epoch='work', settled=False)
    values.update(changes)
    return ScreenReading(values, picture_snapshot=make_snapshot(np.full((32, 32, 3), pixels, np.uint8)))


def test_a_new_scene_on_the_qualified_surface_is_not_an_unrecognized_environment(monkeypatch):
    owner = InputSurfaceCaptures(lambda: 'surface')
    first, new_scene = observed(), observed(11., epoch='b', pixels=180)
    assert owner.bind(first, now=10.1)
    assert not current_input_surface_is_qualified(first, now=10.1)
    assert qualify_current_input_surface(first, now=10.1)
    owner.delivered(True, now=10.1)
    assert owner.bind(new_scene, now=11.1)
    monkeypatch.setattr(time, 'time', lambda: 11.1)
    assert where_am_i(None, observation=new_scene).the_thing_is_here
    assert not current_input_surface_is_qualified(first, now=11.1)


@pytest.mark.parametrize('copying', [dict, copy.copy, lambda seen: ScreenReading(seen, picture_snapshot=seen.picture_snapshot)])
def test_public_flags_or_copied_readings_cannot_inherit_the_adapters_authority(copying):
    owner, seen = InputSurfaceCaptures(lambda: 'surface'), observed()
    assert owner.bind(seen, now=10.1) and qualify_current_input_surface(seen, now=10.1)
    owner.delivered(True, now=10.1)
    assert not current_input_surface_is_qualified(copying(seen), now=10.1)


@pytest.mark.parametrize('changes', [dict(ok=False), dict(occluded=True), dict(her_window_showing=False),
                                   dict(refused_because='changed'), dict(_capture_epoch=''),
                                   dict(bounds=None), dict(capture_at=20.), dict(capture_at=-10.)])
def test_unavailable_occluded_stale_and_future_captures_supply_no_environment_authority(changes):
    owner, seen = InputSurfaceCaptures(lambda: 'surface'), observed(**changes)
    assert not owner.bind(seen, now=10.1)
    assert not qualify_current_input_surface(seen, now=10.1)


def test_navigation_work_changes_and_closed_handovers_cannot_reuse_qualification():
    context = ['surface']
    owner = InputSurfaceCaptures(lambda: context[0])
    first = observed()
    assert owner.bind(first, now=10.1) and qualify_current_input_surface(first, now=10.1)
    owner.delivered(True, now=10.1)
    context[0] = 'elsewhere'
    assert not current_input_surface_is_qualified(first, now=10.1)
    next_view = observed(11., surface='elsewhere', epoch='b')
    assert owner.bind(next_view, now=11.1)
    assert not current_input_surface_is_qualified(next_view, now=11.1)
    assert qualify_current_input_surface(next_view, now=11.1)
    owner.delivered(True, now=11.1)
    changed_work = observed(12., surface='elsewhere', epoch='c', _work_epoch='different')
    assert owner.bind(changed_work, now=12.1) and not current_input_surface_is_qualified(changed_work, now=12.1)
    context[0] = None
    assert not qualify_current_input_surface(changed_work, now=12.1)


def test_changed_capture_fields_and_replaced_pixels_cannot_borrow_the_old_receipt():
    owner, seen = InputSurfaceCaptures(lambda: 'surface'), observed()
    assert owner.bind(seen, now=10.1) and qualify_current_input_surface(seen, now=10.1)
    owner.delivered(True, now=10.1)
    seen['capture_at'] = 10.05
    assert not current_input_surface_is_qualified(seen, now=10.1)
    seen['capture_at'] = 10.
    seen.picture_snapshot = make_snapshot(np.full((32, 32, 3), 10, np.uint8))
    assert not current_input_surface_is_qualified(seen, now=10.1)


def test_an_input_or_failed_new_capture_retires_the_old_capture():
    owner, seen = InputSurfaceCaptures(lambda: 'surface'), observed()
    assert owner.bind(seen, now=10.1) and qualify_current_input_surface(seen, now=10.1)
    owner.delivered(True, now=10.1)
    owner.invalidate()
    assert not current_input_surface_is_qualified(seen, now=10.1)
    next_view = observed(11., epoch='b')
    assert owner.bind(next_view, now=11.1) and current_input_surface_is_qualified(next_view, now=11.1)
    assert not owner.bind({'ok': False}, now=11.2)
    assert not current_input_surface_is_qualified(next_view, now=11.2)


@pytest.mark.parametrize('failure', ['missing', 'refused', 'late', 'foreign'])
def test_a_place_check_requires_its_own_timely_port_acknowledgement(failure):
    context = ['surface']
    owner, seen = InputSurfaceCaptures(lambda: context[0]), observed()
    assert owner.bind(seen, now=10.1)
    assert qualify_current_input_surface(seen, now=10.1)
    assert not current_input_surface_is_qualified(seen, now=10.1)
    if failure == 'foreign':
        context[0] = 'foreign'
    if failure != 'missing':
        owner.delivered(failure != 'refused', now=30. if failure == 'late' else 10.2)
    context[0] = 'surface'
    assert owner.bind(observed(11., epoch='b'), now=11.1)
    assert not current_input_surface_is_qualified(owner.current.observation, now=11.1)


def test_a_current_surface_does_not_override_a_measured_board_mismatch(monkeypatch):
    owner, seen = InputSurfaceCaptures(lambda: 'surface'), observed()
    assert owner.bind(seen, now=10.1) and qualify_current_input_surface(seen, now=10.1)
    owner.delivered(True, now=10.1)
    monkeypatch.setattr(time, 'time', lambda: 10.1)
    reading = SimpleNamespace(occupied=lambda: 2, rows=3, columns=3)
    assert not where_am_i(reading, lattice=SimpleNamespace(held=True, rows=4, columns=4),
                          observation=seen).the_thing_is_here


@pytest.mark.asyncio
async def test_the_normal_page_reader_binds_and_invalidates_its_actual_capture(monkeypatch):
    from core.skills import screen_pursuit_on_a_page as pages
    from core.perception import what_her_page_shows

    page = SimpleNamespace(url='https://example.test/drawing', is_closed=lambda: False)
    adapter = pages.OnAPage(page=page, name='her page')
    at = time.time()
    seen = observed(at, surface=f'page:{id(page)}:her page:{page.url}')
    async def look(*_args, **_kwargs):
        return seen
    async def nothing(*_args):
        return None
    monkeypatch.setattr(what_her_page_shows, 'look_at_a_page', look)
    monkeypatch.setattr(adapter, '_watch', nothing)
    monkeypatch.setattr(adapter, '_where_the_pointer_turns', nothing)
    token = pages.HER_OWN_PAGE.set(adapter)
    try:
        assert await adapter.read() is seen
        assert qualify_current_input_surface(seen)
        adapter._capture_authority.delivered(True)
        assert current_input_surface_is_qualified(seen)
        adapter._invalidate_cursor_samples()
        assert not current_input_surface_is_qualified(seen)
        assert await adapter.read() is seen and current_input_surface_is_qualified(seen)
        page.url = 'https://elsewhere.test/'
        assert not current_input_surface_is_qualified(seen)
    finally:
        pages.HER_OWN_PAGE.reset(token)
    assert not current_input_surface_is_qualified(seen)


def test_owned_capture_qualification_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _owned_input_surface_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'qualified_input_surfaces_keep_current_capture_custody')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
