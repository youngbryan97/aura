"""A refused destination can request owned perception work without granting it."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.a_requested_relation_in_play import SpatialRequest, read_destination, requested_target
from core.perception.a_source_extent_requested_in_play import (
    SourceExtentRequest, _extent_demand_invariant, current_extent_requests,
    extent_reference_to_measure, measured_extent_receipts, request_source_extent,
)
from core.perception.play_observation_without_holding_the_loop import _seen
from core.perception.source_geometry_in_current_pixels import _reference_to_measure
from core.perception.source_material_groups_for_visual_attention import SourceMaterialGroupsInView
from core.perception.visual_assets_in_a_flash_file import FlashVisualAssets
from core.perception.whole_referents_from_source_material import WholeReferentsInView
from tests.test_source_material_reaches_visual_attention import match, mine
from tests.test_source_geometry_precedes_object_grouping import attach, moved, scene  # noqa: F401
from tests.test_whole_referents_require_current_source_material import registered

pytestmark = pytest.mark.unit


def request(scene):
    observer, registry, binding = scene[-1], SourceMaterialGroupsInView(), attach(scene).binding
    observer.source_material_groups = registry.observe(observer, scene[2], 4., (match(scene, binding),))
    group, = observer.source_material_groups.values()
    need = read_destination(SpatialRequest('on', 'platform', ''), mine(), [group], {group.kind: 'platform'}, 4.).need
    assert request_source_extent(observer, need, 4.)
    return registry, binding, group, observer.source_extent_requests


def test_the_actual_goal_selection_posts_the_current_extent_need(scene, monkeypatch):
    registry, binding, group, requests = request(scene)
    observer = scene[-1]
    observer.source_extent_requests = ()
    choosing = SimpleNamespace(mine=mine(), moves=observer, at=4., meeting=SimpleNamespace(
        spatial_request=SpatialRequest('on', 'platform', '')), stance=lambda _thing: 'meet')
    monkeypatch.setattr('core.agency.a_requested_relation_in_play.observed_names', lambda _moves: {group.kind: 'platform'})
    assert requested_target(choosing)[1] == 'wait'
    assert choosing.goal_reading.status == 'referent_extent_unavailable'
    assert observer.source_extent_requests == requests and choosing.goal_binding is None


@pytest.mark.parametrize('damage', ['owner', 'future', 'expired', 'source', 'capacity'])
def test_the_worker_rejects_requests_without_current_owned_scope(scene, damage):
    registry, _binding, group, requests = request(scene)
    table = FlashVisualAssets('source', (registered(scene).declaration,))
    at, tables, records = 4.1, (table,), registry.records
    if damage == 'owner':
        requests = (SourceExtentRequest(replace(group, number=group.number+10)),)
    elif damage == 'future':
        at = 3.9
    elif damage == 'expired':
        at = 4.751
    elif damage == 'source':
        tables = ()
    else:
        requests = requests*5
    current, refused = current_extent_requests(requests, records, at, tables)
    assert not current and refused


def test_refusal_type_and_prior_capture_are_preserved_after_a_real_measurement(scene):
    registry, binding, group, requests = request(scene)
    table = FlashVisualAssets('source', (registered(scene).declaration,))
    current, refused = current_extent_requests(requests, registry.records, 4.1, (table,))
    assert current == requests and not refused
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    picture[32:96, 106] = 0
    from core.perception.pixels_consumed_by_the_observer import consumed_pixels

    observer = scene[-1]
    observer.pixels_consumed = consumed_pixels(picture, 4.1)
    observer.material_marks = replace(observer.material_marks, pixels=observer.pixels_consumed)
    wholes = WholeReferentsInView()
    found = wholes.observe(observer, picture, 4.1, (match(scene, binding, 4.1),))
    receipt, = measured_extent_receipts(current, wholes.readings, found, 4.1, observer.pixels_consumed)
    assert receipt['status'] == 'current_extent_refused' and receipt['measurements'] == 1
    assert receipt['reasons'] == ('whole_edge_support_unresolved',)
    assert receipt['requested_at'] == 4. and receipt['measured_at'] == 4.1
    assert receipt['geometry_authority'] is False and receipt['observed_owner'] is None


def test_a_current_success_remains_an_observation_receipt_with_no_motor_authority(scene):
    registry, binding, group, requests = request(scene)
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    wholes = WholeReferentsInView()
    found = wholes.observe(scene[-1], picture, 4.1, (match(scene, binding, 4.1),))
    receipt, = measured_extent_receipts(requests, wholes.readings, found, 4.1, scene[-1].pixels_consumed)
    assert receipt['status'] == 'current_extent_measured' and receipt['observed_owner'] is not None
    assert receipt['geometry_authority'] is False and not scene[-1].things
    unrelated, = measured_extent_receipts(requests, (), {}, 4.1, scene[-1].pixels_consumed)
    assert unrelated['status'] == 'current_material_unavailable' and unrelated['measurements'] == 0


def test_same_asset_from_a_new_mark_epoch_cannot_satisfy_the_request(scene):
    _registry, binding, group, requests = request(scene)
    reading = dict(source=group.source, character_id=group.character_id, mark_epoch='2'*32,
                   material_anchors=tuple(group.anchors), reason='qualified')
    receipt, = measured_extent_receipts(requests, (reading,), {}, 4.1, replace(group.pixels, at=4.1))
    assert receipt['measurements'] == 0 and receipt['status'] == 'current_material_unavailable'


@pytest.mark.parametrize('damage', ['stale', 'different-pixels', 'no-capture'])
def test_prior_or_foreign_measurements_cannot_acknowledge_a_new_frame(scene, damage):
    _registry, binding, _group, requests = request(scene)
    wholes = WholeReferentsInView()
    found = wholes.observe(scene[-1], scene[2], 4., (match(scene, binding),))
    pixels = replace(scene[-1].pixels_consumed, at=4.1)
    if damage == 'different-pixels':
        pixels = replace(pixels, sha256='f'*64)
    elif damage == 'no-capture':
        pixels = None
    receipt, = measured_extent_receipts(requests, wholes.readings, found, 4.1, pixels)
    assert receipt['measurements'] == 0 and receipt['observed_owner'] is None
    assert receipt['status'] in ('current_material_unavailable', 'current_capture_unavailable')


@pytest.mark.parametrize('covered', [False, True])
def test_requested_family_gets_one_slot_and_every_other_reference_keeps_its_turn(covered):
    keys = tuple(('source', n//2, str(n), n%2+1) for n in range(10))
    coverage = {k[:3] for k in keys[:4]} if covered else set()
    choices = [extent_reference_to_measure(keys, coverage, n, (('source', 3),), _reference_to_measure) for n in range(200)]
    assert all(choices[n][:2] == ('source', 3) for n in range(0, 200, 4))
    assert set(choices) == set(keys) and _extent_demand_invariant() == ()


def test_an_owned_request_reaches_prefit_priority_and_current_measurement_in_the_worker(scene):
    registry, binding, group, requests = request(scene)
    reference, table = registered(scene), FlashVisualAssets('source', (registered(scene).declaration,))
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    snapshot = copy.deepcopy(scene[-1].__dict__)

    class Eyes:
        def see(self, _picture, _at):
            self.__dict__.update(copy.deepcopy(snapshot))
            return []

    class Geometry:
        material_frames = material_status = ()
        called = []

        def sampling_before_observation(self, _picture, _at, *_args, priority=()):
            self.called.append(priority)

        def measure_if_budget(self, _picture, _at, _state, *_args, priority=(), **_kwargs):
            self.called.append(priority)
            return (), ()

    geometry = Geometry()
    _, state = _seen(Eyes(), picture, 4.1, fresh_references=geometry,
        material_groups=registry, whole_referents=WholeReferentsInView(),
        reference_frames=((reference, binding),), reference_tables=(table,), extent_requests=requests)
    assert geometry.called == [(('source', 17),), (('source', 17),)]
    receipt, = state['source_extent_receipts']
    assert receipt['status'] == 'current_extent_measured' and receipt['measured_at'] == 4.1


def test_the_recorder_does_not_count_a_refused_request_as_qualified_geometry(scene):
    from core.perception.recording_the_frames_and_inputs import _source_reference_snapshot

    _registry, _binding, group, requests = request(scene)
    observer = scene[-1]
    observer.source_extent_receipts = measured_extent_receipts(requests, (), {}, 4.1, replace(group.pixels, at=4.1))
    snapshot = _source_reference_snapshot(observer)
    assert snapshot['extent_requests'][0]['status'] == 'current_material_unavailable'
    assert snapshot['whole_referents']['count'] is None


@pytest.mark.asyncio
async def test_real_observation_port_consumes_the_queued_need_once_and_clears_receipts(scene, monkeypatch):
    import asyncio

    from core.perception import source_references_while_play_advances as work
    from core.perception.pixels_consumed_by_the_observer import consumed_pixels
    from core.perception.play_observation_without_holding_the_loop import open_play_observation

    reference, binding = registered(scene), attach(scene).binding
    table = FlashVisualAssets('source', (reference.declaration,))
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    picture[32:96, 106] = 0
    observer = scene[-1]
    observer.pixels_consumed = consumed_pixels(picture, 4.1)
    observer.material_marks = replace(observer.material_marks, pixels=observer.pixels_consumed)
    snapshot = copy.deepcopy(observer.__dict__)

    class Eyes:
        def __init__(self, **_kwargs):
            self.things, self.pending, self.kinds = {}, {}, []
            self.scale = 1.

        def see(self, _picture, _at):
            self.__dict__.update(copy.deepcopy(snapshot))
            return []

    class References:
        tables, frames = (table,), ((reference, binding),)
        proposals = source_frames = points = ()
        status, retention = 'ready', None

        def advance(self, *_args):
            return ()

        def retire(self):
            return None

    monkeypatch.setattr(work, 'source_tables_in_play', lambda: (table,))
    view = await open_play_observation(factory=Eyes)
    view._references = References()
    try:
        assert await view.see(picture, 4.1) == []
        group, = view.source_material_groups.values()
        need = read_destination(SpatialRequest('on', 'platform', ''), mine(4.1), [group], {group.kind: 'platform'}, 4.1).need
        assert request_source_extent(view, need, 4.1)
        observer.pixels_consumed = consumed_pixels(picture, 4.2)
        observer.material_marks = replace(observer.material_marks, sequence=22, pixels=observer.pixels_consumed,
            points=tuple(replace(p, old_x=p.x, old_y=p.y, before=4.1, at=4.2, pictures=p.pictures+1)
                         for p in observer.material_marks.points))
        snapshot = copy.deepcopy(observer.__dict__)
        assert await view.see(picture, 4.2) == []
        receipt, = view.source_extent_receipts
        assert receipt['status'] == 'current_extent_refused'
        assert receipt['requested_at'] == 4.1 and receipt['measured_at'] == 4.2
        assert not view.source_extent_requests
        assert await view.see(picture, 4.2) is None and not view.source_extent_receipts
    finally:
        await view.close()
        await asyncio.sleep(0)
    assert not view.source_extent_requests and not view.source_extent_receipts
