"""A hover caption names the hit; its centre does not replace the hit point."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.perception.recording_the_frames_and_inputs import (
    _navigation_port_recording_invariant,
    record_frames_and_inputs,
)
from core.perception.where_the_pointer_turns import the_thing_there, where_the_pointer_turns
from core.skills.screen_pursuit_bearings import (
    _cursor_anchor_invariant,
    things_to_click,
    where_to_click,
)
from core.skills.screen_pursuit_looking import _what_answers_the_pointer
from core.skills.screen_pursuit_on_a_page import OnAPage


class _Page:
    viewport_size = {'width': 1000, 'height': 700}

    def __init__(self, point=(.875, 1-.25/9)):
        self.point, self.clicked, self.at = point, [], (0., 0.)
        self.mouse = SimpleNamespace(move=self.move, click=self.click)

    async def move(self, x, y, **_kwargs):
        self.at = x, y

    async def click(self, x, y, **_kwargs):
        self.clicked.append((x, y))

    def cursor(self, x, y):
        if self.point is None:
            return 'auto'
        return 'pointer' if abs(x/1000-self.point[0]) < .012 and abs(y/700-self.point[1]) < .012 else 'auto'

    async def evaluate(self, script, point):
        if len(point) == 1 and isinstance(point[0], list):
            return [self.cursor(x, y) for x, y in point[0]]
        return self.cursor(*point)


@pytest.mark.asyncio
async def test_boundary_sampling_can_find_a_small_hit_outside_the_last_grid_centre():
    page = _Page()
    places = await where_the_pointer_turns(page)
    assert any(page.cursor(x*1000, y*700) == 'pointer' for x, y in places)
    assert all(y > .95 for _x, y in places) and len(places) <= 24


@pytest.mark.asyncio
async def test_an_associated_caption_keeps_the_hit_through_the_normal_click_and_recorder(tmp_path, monkeypatch):
    page = _Page()
    surface = OnAPage(page, 'editor')
    label = dict(text='Continue', x=.74, y=.945, width=.14, height=.035, center_x=.81, center_y=.961)
    seen = dict(text='Continue', layout=[label])

    async def points(*_args):
        return [page.point]

    monkeypatch.setattr('core.perception.where_the_pointer_turns.where_the_pointer_turns', points)
    await surface._where_the_pointer_turns(seen, (0, 0, 1, 1))
    target = where_to_click(seen, 'Continue')
    assert target == page.point and target != (label['center_x'], label['center_y'])
    directory = tmp_path/'normal-input'
    async with record_frames_and_inputs(directory):
        assert await surface.click(*target, [0, 0, 1000, 700])
    actual = page.clicked[-1]
    assert actual == pytest.approx((page.point[0]*1000, page.point[1]*700))
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    request, receipt = events
    assert request['kind'] == receipt['kind'] == 'navigation_input'
    assert request['args'] == pytest.approx(actual) and request['bounds'] == [0., 0., 1000., 700.]
    assert receipt['delivered'] is True and receipt['request_sequence'] == request['sequence']
    assert 'task_passed' not in receipt
    assert _navigation_port_recording_invariant() == ()
    assert _cursor_anchor_invariant() == ()


@pytest.mark.asyncio
async def test_a_disappeared_cursor_hit_cannot_fall_back_to_the_old_position():
    page = _Page(point=None)
    surface = OnAPage(page, 'editor')
    surface._swept = (frozenset({'continue'}), [(.8, .9)])
    assert not await surface.click(.8, .9, [0, 0, 1000, 700])
    assert not page.clicked and surface._swept is None


@pytest.mark.asyncio
async def test_two_hits_attached_to_one_caption_remain_unlabelled_choices(monkeypatch):
    surface = OnAPage(_Page(), 'editor')
    label = dict(text='Continue', x=.1, y=.1, width=.8, height=.8, center_x=.5, center_y=.5)
    seen = dict(text='Continue', layout=[label])

    async def points(*_args):
        return [(.2, .2), (.8, .8)]

    monkeypatch.setattr('core.perception.where_the_pointer_turns.where_the_pointer_turns', points)
    await surface._where_the_pointer_turns(seen, None)
    assert label['pointer_ambiguous'] and where_to_click(seen, 'Continue') is None
    assert 'click "Continue"' not in things_to_click(seen, True)
    assert 'click "Continue"' not in _what_answers_the_pointer(seen, ())
    assert len(seen['shapes']) == 2


def test_equal_nearby_labels_cannot_claim_the_same_hit():
    boxes = [dict(text=name, x=.4, y=.4, width=.2, height=.2) for name in ('Back', 'Next')]
    assert the_thing_there((.5, .5), boxes) is None


@pytest.mark.parametrize('point', [(float('nan'), .5), (2., .5), (.5, True)])
def test_invalid_hit_geometry_cannot_fall_back_to_a_caption_centre(point):
    label = dict(text='Continue', center_x=.5, center_y=.5, answers_the_pointer=True,
                 pointer_x=point[0], pointer_y=point[1])
    assert where_to_click(dict(layout=[label]), 'Continue') is None


@pytest.mark.asyncio
async def test_a_transport_failure_is_recorded_without_claiming_no_partial_input(tmp_path):
    class Refused(_Page):
        async def click(self, *args, **kwargs):
            raise RuntimeError('transport refused')

    surface = OnAPage(Refused(), 'editor')
    directory = tmp_path/'refused-input'
    async with record_frames_and_inputs(directory):
        assert not await surface.click(.3, .4, [10, 20, 1000, 700])
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    assert events[0]['args'] == [310., 300.]
    assert events[-1]['phase'] == 'failed' and events[-1]['delivered'] is False
    assert events[-1]['partial_delivery_possible']


@pytest.mark.asyncio
@pytest.mark.parametrize('everywhere', [True, False])
async def test_a_uniform_cursor_supplies_no_local_affordance(everywhere):
    page = _Page()
    page.cursor = lambda x, y: 'pointer' if everywhere else 'auto'
    assert await where_the_pointer_turns(page) == []
