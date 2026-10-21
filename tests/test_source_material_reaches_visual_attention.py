"""Current material gets an ordinary naming query without a fabricated outline."""
import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.a_requested_relation_in_play import SpatialRequest, position_for_relation, read_destination
from core.agency.evidence_for_a_requested_relation import observed_relation_region
from core.agency.looking_for_an_unbound_referent import referents_to_observe
from core.perception.names_for_observed_instances import InstanceNames, query_for
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.source_material_groups_for_visual_attention import (
    SourceMaterialGroupsInView, _material_attention_invariant, _material_questions,
)
from core.perception.whole_referents_from_source_material import WholeReferentsInView, scene_referents
from core.perception.what_things_look_like import LookingAtThings
from tests.test_source_geometry_precedes_object_grouping import attach, moved, scene  # noqa: F401
from tests.test_whole_referents_require_current_source_material import registered

pytestmark = pytest.mark.unit


def match(scene, binding, at=4., reference=None):
    reference = registered(scene) if reference is None else reference
    return reference, binding, binding.current(scene[-1], at, source=reference.source)


def mine(at=4.):
    return SimpleNamespace(number=1, born=3., kind=0, seen=at, x=10., y=10., w=8., h=8.)


def test_failed_whole_reconciliation_keeps_current_material_available_for_naming(scene):
    observer, binding = scene[-1], attach(scene).binding
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    picture[32:96, 106] = 0  # Contradicted outline, away from admitted patches.
    pixels = consumed_pixels(picture, 4.1)
    observer.pixels_consumed = pixels
    observer.material_marks = replace(observer.material_marks, pixels=pixels)
    matches = (match(scene, binding, 4.1),)
    assert not WholeReferentsInView().observe(observer, picture, 4.1, matches)
    observer.source_material_groups = SourceMaterialGroupsInView().observe(observer, picture, 4.1, matches)
    group, = observer.source_material_groups.values()
    assert group.material_bounds == (51., 40., 97., 86.)
    assert group.box() == pytest.approx((43., 32., 107., 96.), abs=.1)
    assert group.w > 46. and not group.geometry_qualified and not observer.things
    assert group.role is None and group.contact_verified is None
    assert 2**51 <= group.number < 2**52
    observer.shape = (128, 160)
    wanted = referents_to_observe(SpatialRequest('on', 'platform', ''), observer, mine(4.1), {}, 4.1)
    assert wanted == (query_for(group, observer, 4.1).key,)
    assert LookingAtThings()._to_look_at(observer, None, set(), at=4.1, sought=wanted)[0].number == group.number


@pytest.mark.parametrize('relation', ['on', 'above', 'below', 'inside', 'beside', 'at'])
def test_a_visual_name_cannot_promote_material_span_to_movement_geometry(scene, relation):
    observer, binding = scene[-1], attach(scene).binding
    observer.source_material_groups = SourceMaterialGroupsInView().observe(observer, scene[2], 4., (match(scene, binding),))
    group, = observer.source_material_groups.values()
    names = InstanceNames()
    names.begin(observer)
    assert names.record_for(observer, query_for(group, observer, 4.).with_picture(scene[2]), 'platform')
    reading = read_destination(SpatialRequest(relation, 'platform', ''), mine(), scene_referents(observer), names.scene(observer, {}), 4.)
    assert reading.status == 'referent_extent_unavailable' and reading.need.operation == 'measure_whole_extent'
    assert reading.binding is None and position_for_relation(relation, mine(), group) is None
    assert observed_relation_region(relation, mine(), group, 4.) is None


@pytest.mark.parametrize('damage', ['foreign', 'stale', 'epoch', 'marks', 'reference', 'pixels', 'point'])
def test_lost_or_foreign_evidence_cannot_keep_a_material_question(scene, damage):
    observer, registry, binding = scene[-1], SourceMaterialGroupsInView(), attach(scene).binding
    matches = (match(scene, binding),)
    assert registry.observe(observer, scene[2], 4., matches)
    reference, binding, point = matches[0]
    picture = scene[2]
    if damage == 'foreign':
        reference = replace(reference, source='different')
    elif damage == 'stale':
        observer.pixels_consumed = consumed_pixels(picture, 3.9)
    elif damage == 'epoch':
        observer.material_marks = replace(observer.material_marks, epoch='2'*32)
    elif damage == 'marks':
        observer.material_marks = replace(observer.material_marks, points=())
    elif damage == 'reference':
        reference = replace(reference, pixels=np.zeros_like(reference.pixels))
    elif damage == 'pixels':
        picture = np.zeros_like(picture)
    else:
        point = replace(point, position=(0., 0.))
    assert not registry.observe(observer, picture, 4., ((reference, binding, point),))


def test_names_continue_across_source_variants_only_on_continuing_owned_marks(scene):
    observer, registry, binding = scene[-1], SourceMaterialGroupsInView(), attach(scene).binding
    observer.source_material_groups = registry.observe(observer, scene[2], 4., (match(scene, binding),))
    first, = observer.source_material_groups.values()
    names = InstanceNames()
    names.begin(observer)
    assert names.record_for(observer, query_for(first, observer, 4.).with_picture(scene[2]), 'platform')
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    variant = replace(registered(scene), frame_number=2, declared_frame_count=2)
    observer.source_material_groups = registry.observe(observer, picture, 4.1, (match(scene, binding, 4.1, variant),))
    current, = observer.source_material_groups.values()
    assert (current.number, current.born) == (first.number, first.born)
    assert names.scene(observer, {}).for_thing(current) == 'platform'
    assert registry.observe(observer, picture, 4.2, ()) == {}
    # Merely reacquiring old locations cannot bridge a missing current sample.
    fresh, = registry.observe(observer, picture, 4.1, (match(scene, binding, 4.1, variant),)).values()
    assert fresh.number != first.number


def test_duplicate_static_hypotheses_do_not_duplicate_or_enlarge_a_question(scene):
    registry, binding = SourceMaterialGroupsInView(), attach(scene).binding
    original = match(scene, binding)
    variant = replace(original[0], frame_number=2, declared_frame_count=2)
    group, = registry.observe(scene[-1], scene[2], 4., (original, (variant, *original[1:]))).values()
    assert group.material_bounds == (48., 38., 94., 84.) and len(group.anchors) == 8
    assert group.box() == pytest.approx((40., 30., 104., 94.), abs=.1)
    assert group.source == 'source' and not hasattr(group, 'exported_name')


def test_independent_current_bindings_combine_only_when_their_material_directly_agrees(scene):
    binding = attach(scene).binding
    subset = replace(binding, anchors=binding.anchors[:-1])
    groups = SourceMaterialGroupsInView().observe(scene[-1], scene[2], 4.,
        (match(scene, binding), match(scene, subset)))
    group, = groups.values()
    assert len(group.anchors) == 8 and group.material_bounds == (48., 38., 94., 84.)


def test_material_overlap_chains_cannot_join_different_instances():
    def key(*names):
        return 'source', 17, frozenset((n, 1.) for n in names)
    first, bridge, last = key(1, 2, 3, 4), key(1, 2, 3, 5, 6, 7), key(5, 6, 7, 8)
    value = ((0., 0., 10., 10.), (1., 1., 9., 9.), ())
    questions, refused = _material_questions({first: value, bridge: value, last: value})
    assert not questions and refused[0]['reason'] == 'ambiguous_material_owner'
    questions, refused = _material_questions({first: value, last: value})
    assert len(questions) == 2 and not refused


def test_a_wider_inspection_window_still_cannot_supply_a_wider_destination(scene):
    observer, binding = scene[-1], attach(scene).binding
    # Inspection includes the projected source artwork, not just its points.
    group, = SourceMaterialGroupsInView().observe(observer, scene[2], 4., (match(scene, binding),)).values()
    assert group.w > group.material_bounds[2]-group.material_bounds[0]
    body = mine()
    body.w = 50.
    reading = read_destination(SpatialRequest('on', 'platform', ''), body, [group], {group.kind: 'platform'}, 4.)
    assert reading.binding is None and reading.status == 'referent_extent_unavailable'


def test_a_qualified_whole_or_complete_physical_instance_supersedes_attention_only_span(scene):
    observer, binding = scene[-1], attach(scene).binding
    matches = (match(scene, binding),)
    wholes = WholeReferentsInView().observe(observer, scene[2], 4., matches)
    assert wholes and not SourceMaterialGroupsInView().observe(observer, scene[2], 4., matches, wholes=wholes)
    observer.things[3] = SimpleNamespace(number=3, born=3., kind=2, seen=4., x=71., y=61., w=60., h=60.,
        point_pose=SimpleNamespace(born=3., at=4., visible_marks={p.number: (p.x, p.y) for p in observer.material_marks.points}))
    assert not SourceMaterialGroupsInView().observe(observer, scene[2], 4., matches)


def test_material_containing_her_body_cannot_be_her_destination(scene):
    observer, binding = scene[-1], attach(scene).binding
    body = mine()
    body.point_pose = SimpleNamespace(born=body.born, at=4.,
        visible_marks={p.number: (p.x, p.y) for p in observer.material_marks.points})
    observer.things[body.number] = body
    groups = SourceMaterialGroupsInView().observe(observer, scene[2], 4., (match(scene, binding),))
    group, = groups.values()
    reading = read_destination(SpatialRequest('at', 'platform', ''), body, [group], {group.kind: 'platform'}, 4.)
    assert reading.status == 'no_named_referent'


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [False, True])
async def test_actual_worker_publishes_and_retires_material_questions(scene, monkeypatch, changed):
    from core.perception import source_references_while_play_advances as work
    from core.perception.play_observation_without_holding_the_loop import open_play_observation
    from core.perception.visual_assets_in_a_flash_file import FlashVisualAssets

    observer, binding, reference = scene[-1], attach(scene).binding, registered(scene)
    table = FlashVisualAssets('source', (reference.declaration,))
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    picture[32:96, 106] = 0
    observer.pixels_consumed = consumed_pixels(picture, 4.1)
    observer.material_marks = replace(observer.material_marks, pixels=observer.pixels_consumed)
    state = copy.deepcopy(observer.__dict__)

    class Eyes:
        def __init__(self, **_kwargs):
            self.things, self.pending, self.kinds = {}, {}, []
            self.scale = 1.

        def see(self, _picture, _at):
            self.__dict__.update(copy.deepcopy(state))
            return []

    class References:
        tables = (table,)
        frames = ((reference, binding),)
        proposals = source_frames = points = ()
        status, retention = 'ready', None

        def advance(self, *_args):
            return ()

        def retire(self):
            return None

    calls = []

    def tables_now():
        calls.append(None)
        return () if changed and len(calls) > 1 else (table,)

    monkeypatch.setattr(work, 'source_tables_in_play', tables_now)
    view = await open_play_observation(factory=Eyes)
    view._references = References()
    try:
        assert await view.see(picture, 4.1) == []
        assert bool(view.source_material_groups) is not changed
        assert not view.source_referents and not view.things
        assert await view.see(picture, 4.1) is None
        assert not view.source_material_groups
    finally:
        await view.close()
        await asyncio.sleep(0)
    assert not view.source_material_groups


def test_recorder_keeps_material_attention_separate_from_qualified_geometry(scene):
    from core.perception.recording_the_frames_and_inputs import _source_reference_snapshot

    observer, binding = scene[-1], attach(scene).binding
    observer.source_material_groups = SourceMaterialGroupsInView().observe(observer, scene[2], 4., (match(scene, binding),))
    snapshot = _source_reference_snapshot(observer)
    record, = snapshot['material_attention']['records']
    assert snapshot['material_attention']['count'] == 1 and snapshot['whole_referents']['count'] is None
    assert record['geometry_qualified'] is False and record['role'] is None
    assert record['pixels']['sha256'] == observer.pixels_consumed.sha256
    assert record['material_bounds'] == [48., 38., 94., 84.]
    assert record['w'] > 46. and record['contact_verified'] is None
    assert _material_attention_invariant() == ()
