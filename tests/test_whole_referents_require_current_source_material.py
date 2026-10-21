"""A whole reference becomes an observed referent without inventing a collider."""
import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.a_requested_relation_in_play import SpatialRequest, read_destination, requested_target
from core.agency.looking_for_an_unbound_referent import referents_to_observe
from core.perception.names_for_observed_instances import InstanceNames, query_for
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset, FlashVisualAssets
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from core.perception.whole_referents_from_source_material import (
    WholeReferentsInView, _whole_source_referent_invariant, scene_referents,
)
from tests.test_source_geometry_precedes_object_grouping import attach, moved, scene  # noqa: F401

pytestmark = pytest.mark.unit


def registered(scene):
    coordinates, _pose, _picture, rgba, _observer = scene
    asset = DeclaredVisualAsset('platform', 17, 'sprite', 0, 39, (10,))
    return RegisteredVisualReference('source', asset, coordinates, rgba)


def observe(scene, registry, binding, picture=None, at=4., reference=None):
    observer = scene[-1]
    reference = registered(scene) if reference is None else reference
    picture = scene[2] if picture is None else picture
    point = binding.current(observer, at, source='source')
    return registry.observe(observer, picture, at, ((reference, binding, point),))


def test_full_material_extent_has_its_own_identity_without_a_fragment_or_role(scene):
    registry, binding = WholeReferentsInView(), attach(scene).binding
    referents = observe(scene, registry, binding)
    whole, = referents.values()
    assert whole.box() == pytest.approx((39.5, 29.5, 103.5, 93.5), abs=.08)
    assert whole.pixels == scene[-1].pixels_consumed and whole.role is None and whole.contact_verified is None
    assert whole.vx is None and whole.vy is None and whole.velocity_error is None
    assert not scene[-1].things
    assert 2**52 <= whole.number < 2**53
    scene[-1].source_referents = referents
    assert scene_referents(scene[-1]) == (whole,)
    # An exported noun never supplies the current semantic observation.
    mine = SimpleNamespace(number=1, born=3., kind=0, seen=4., x=10., y=10., w=8., h=8.)
    reading = read_destination(SpatialRequest('on', 'platform', ''), mine, [whole], {}, 4.)
    assert reading.status == 'no_named_referent'
    reading = read_destination(SpatialRequest('on', 'platform', ''), mine, [whole], {whole.kind: 'platform'}, 4.)
    assert reading.status == 'referent_motion_unavailable' and reading.need.operation == 'measure_reference_motion'
    with pytest.raises(ValueError, match='unmeasured'):
        whole.where_at(.2)


def test_an_existing_source_origin_binding_cannot_hide_whole_reference_material(scene, monkeypatch):
    from core.perception.a_material_frame_kept_for_its_owner import keep_material_frame
    from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
    from core.perception.source_references_while_play_advances import _matched

    # A canonical fragment has already accepted the origin. Whole material
    # must still be measured independently; this supplies no body identity.
    monkeypatch.setattr(SourceImageCoordinates, 'on_track', lambda *_args, **_kwargs:
                        SimpleNamespace(binding=object(), reason='qualified'))
    frame = keep_material_frame(scene[-1], scene[2], 4.)
    result = _matched(registered(scene), frame)
    assert result.bindings and result.frames


def test_current_whole_identity_and_measured_origin_survive_motion_before_grouping(scene):
    registry, binding = WholeReferentsInView(), attach(scene).binding
    before, = observe(scene, registry, binding).values()
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    current, = observe(scene, registry, binding, picture, 4.1).values()
    assert (current.number, current.born, current.kind) == (before.number, 4., before.kind)
    assert current.x-before.x == pytest.approx(3., abs=.01)
    assert (current.vx, current.vy) == pytest.approx((30., 20.), abs=.01)
    assert current.where_at(.1) == pytest.approx((current.x+3., current.y+2.))
    assert len(current.origin_path) == 2 and current.velocity_error >= 0.


@pytest.mark.parametrize('damage', ['pixels', 'stale', 'lost-marks', 'foreign-source', 'changed-image', 'forged-point'])
def test_broken_custody_or_whole_pixel_support_cannot_publish_a_referent(scene, damage):
    registry, binding = WholeReferentsInView(), attach(scene).binding
    picture, observer, reference = scene[2].copy(), scene[-1], registered(scene)
    point = binding.current(observer, 4., source='source')
    if damage == 'pixels':
        picture[50:85, 40:104] = 255
        receipt = consumed_pixels(picture, 4.)
        observer.pixels_consumed = receipt
        observer.material_marks = replace(observer.material_marks, pixels=receipt)
    elif damage == 'stale':
        observer.pixels_consumed = consumed_pixels(picture, 3.9)
    elif damage == 'lost-marks':
        observer.material_marks = replace(observer.material_marks, points=())
    elif damage == 'foreign-source':
        reference = replace(reference, source='another-source')
    elif damage == 'changed-image':
        reference = replace(reference, pixels=np.full_like(reference.pixels, 255))
    else:
        point = replace(point, sampling_corners=tuple((x+20., y) for x, y in point.sampling_corners))
    assert not registry.observe(observer, picture, 4., ((reference, binding, point),))


def test_two_current_measurements_cannot_borrow_one_previous_identity(scene):
    registry, binding = WholeReferentsInView(), attach(scene).binding
    observe(scene, registry, binding)
    moved(scene)
    picture, observer = np.roll(scene[2], (2, 3), axis=(0, 1)), scene[-1]
    match = registered(scene), binding, binding.current(observer, 4.1, source='source')
    assert not registry.observe(observer, picture, 4.1, (match, match))


def test_a_hidden_thin_outer_edge_cannot_supply_extra_width(scene):
    registry, binding, observer = WholeReferentsInView(), attach(scene).binding, scene[-1]
    picture = scene[2].copy()
    # Less than two percent is hidden; aggregate colour loss alone would pass.
    picture[30:94, 103] = 0
    receipt = consumed_pixels(picture, 4.1)
    observer.pixels_consumed = receipt
    observer.material_marks = replace(observer.material_marks, sequence=21, pixels=receipt,
        points=tuple(replace(p, at=4.1, before=4., pictures=p.pictures+1) for p in observer.material_marks.points))
    assert not observe(scene, registry, binding, picture, 4.1)
    assert registry.readings[0]['reason'] == 'whole_edge_support_unresolved'
    assert registry.readings[0]['maximum_edge_error'] > .12


@pytest.mark.parametrize('complete', [False, True])
def test_whole_material_keeps_fragment_ownership_without_duplicating_a_complete_instance(scene, complete):
    registry, binding, observer = WholeReferentsInView(), attach(scene).binding, scene[-1]
    marks = observer.material_marks.points
    part = SimpleNamespace(number=3, born=3., kind=2, seen=4., x=71.5, y=61.5,
        w=64. if complete else 8., h=64. if complete else 8.,
        point_pose=SimpleNamespace(born=3., at=4., visible_marks={p.number: (p.x, p.y) for p in marks}))
    observer.things[3] = part
    found = observe(scene, registry, binding)
    if complete:
        assert not found and registry.readings[-1]['reason'] == 'whole_already_observed_as_one_instance'
    else:
        whole, = found.values()
        assert whole.number != part.number and whole.observed_parts == ((3, 3., 2),)
        # A whole containing her observed material cannot become her destination.
        reading = read_destination(SpatialRequest('at', 'platform', ''), part, [whole], {whole.kind: 'platform'}, 4.)
        assert reading.status == 'no_named_referent'


def test_disconnected_visible_parts_join_only_through_the_verified_whole_reference(scene):
    from core.perception.a_visual_reference_in_the_picture import _digest, reference_appearances

    coordinates, _pose, picture, rgba, observer = scene
    rgba = rgba.copy()
    rgba[28:36, :, 3] = 0
    picture = picture.copy()
    picture[58:66, 40:104] = 0
    receipt = consumed_pixels(picture, 4.)
    observer.pixels_consumed = receipt
    observer.material_marks = replace(observer.material_marks, pixels=receipt)
    coordinates = replace(coordinates, reference_sha256=_digest(rgba))
    pose, = reference_appearances(picture, rgba, scales=(1.,))
    separated = coordinates, pose, picture, rgba, observer
    binding = attach(separated).binding
    assert binding is not None and len(binding.anchors) == 6
    whole, = observe(separated, WholeReferentsInView(), binding).values()
    assert (whole.w, whole.h) == pytest.approx((64., 64.), abs=.1)
    assert whole.contact_verified is None and whole.role is None and not observer.things


def test_retired_material_stream_cannot_keep_the_same_whole_identity(scene):
    registry, binding = WholeReferentsInView(), attach(scene).binding
    old, = observe(scene, registry, binding).values()
    observer = scene[-1]
    observer.material_marks = replace(observer.material_marks, epoch='2'*32)
    assert not observe(scene, registry, binding)
    fresh = attach(scene).binding
    new, = observe(scene, registry, fresh).values()
    assert new.number != old.number


def test_two_actually_visible_copies_stay_separate_and_compete_for_the_same_name(scene):
    from core.perception.a_source_point_on_visible_marks import attach_source_point_to_visible_marks
    from core.perception.a_visual_reference_in_the_picture import reference_appearances
    from core.perception.the_pixel_grid_a_mark_uses import SampleGrid

    reference, observer = registered(scene), scene[-1]
    picture = np.zeros((128, 320, 3), np.uint8)
    picture[:, :160] = scene[2]
    picture[:, 160:] = scene[2]
    receipt = consumed_pixels(picture, 4.)
    original = observer.material_marks.points
    copied = tuple(replace(p, number=p.number+100, x=p.x+160., old_x=p.old_x+160.) for p in original)
    observer.pixels_consumed = receipt
    observer.material_marks = replace(observer.material_marks, pixels=receipt, points=(*original, *copied),
        grid=SampleGrid((128, 320), (128, 320), (128, 320)))
    poses = reference_appearances(picture, reference.pixels, scales=(1.,))
    assert len(poses) == 2
    matches = []
    for pose in poses:
        binding = attach_source_point_to_visible_marks(reference.coordinates, (0., 0.), pose,
            picture, reference.pixels, observer, 4., source='source').binding
        assert binding is not None
        matches.append((reference, binding, binding.current(observer, 4., source='source')))
    wholes = WholeReferentsInView().observe(observer, picture, 4., tuple(matches))
    assert len(wholes) == 2 and len({w.born for w in wholes.values()}) == 1
    assert all(w.w == pytest.approx(64., abs=.1) for w in wholes.values())
    mine = SimpleNamespace(number=1, born=3., kind=0, seen=4., x=10., y=10., w=8., h=8.)
    reading = read_destination(SpatialRequest('on', 'platform', ''), mine, list(wholes.values()),
                               {w.kind: 'platform' for w in wholes.values()}, 4.)
    assert reading.status == 'ambiguous_named_referent' and reading.candidate_count == 2


def test_named_whole_reaches_actual_goal_binding_and_never_body_tracking(scene, monkeypatch):
    from core.agency.a_requested_relation_in_play import _scene_referents
    from core.perception.what_things_look_like import LookingAtThings

    registry, binding, observer = WholeReferentsInView(), attach(scene).binding, scene[-1]
    observer.shape = (128, 160)
    observer.source_referents = observe(scene, registry, binding)
    initial, = observer.source_referents.values()
    names = InstanceNames()
    names.begin(observer)
    query = query_for(initial, observer, 4.).with_picture(scene[2])
    assert names.record_for(observer, query, 'blue platform')
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    observer.source_referents = observe(scene, registry, binding, picture, 4.1)
    whole, = observer.source_referents.values()
    mine = SimpleNamespace(number=1, born=3., kind=0, seen=4.1, x=10., y=10., w=8., h=8.)
    request = SpatialRequest('on', 'blue platform', '')
    reading = read_destination(request, mine, _scene_referents(observer), names.scene(observer, {}), 4.1)
    assert reading.status == 'bound' and reading.binding.thing.number == initial.number
    assert reading.binding.region.contact_verified is None and reading.binding.region.maximum_speed is None
    chooser = SimpleNamespace(mine=mine, moves=observer, at=4.1, meeting=SimpleNamespace(spatial_request=request),
                              stance=lambda _thing: 'meet')
    monkeypatch.setattr('core.agency.a_requested_relation_in_play.observed_names', lambda _moves: names.scene(observer, {}))
    assert requested_target(chooser)[1:] == ('meet', whole)
    from core.agency.a_requested_region_in_a_plan import current_requested_region

    assert current_requested_region(chooser).target_owner == (whole.number, whole.born, whole.kind)
    assert not observer.things and whole.number not in observer.things
    assert not referents_to_observe(request, observer, mine, names.scene(observer, {}), 4.1)
    # Unnamed whole artwork gets a normal boxed observation, with unchanged prompts.
    wanted = referents_to_observe(request, observer, mine, {}, 4.1)
    assert wanted == (query_for(whole, observer, 4.1).key,)
    assert LookingAtThings()._to_look_at(observer, None, set(), at=4.1, sought=wanted)[0].number == whole.number
    mine.w = 100.
    assert read_destination(request, mine, [whole], names.scene(observer, {}), 4.1).status == 'relation_geometry_unavailable'


def test_the_actual_chooser_can_plan_toward_a_named_whole_without_tracking_it_as_a_body(scene):
    from core.agency.choosing_where_to_be import _Choosing
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from tests.test_requested_relations_reach_the_actual_target import _scene

    registry, binding, observer = WholeReferentsInView(), attach(scene).binding, scene[-1]
    observe(scene, registry, binding)
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    wholes = observe(scene, registry, binding, picture, 4.1)
    whole, = wholes.values()
    body, _old_target, _nearby, moves, hers, meeting, guide = _scene({})
    moves.source_referents, moves.pixels_consumed, moves.scale = wholes, observer.pixels_consumed, 1.
    moves.shape = (128, 160)
    body.x, body.y = 10., 10.
    for thing in moves.things.values():
        thing.seen = 4.1
    guide.seen.instances.begin(moves)
    query = query_for(whole, moves, 4.1).with_picture(picture)
    assert guide.seen.saw_instance(moves, query, 'platform') is not None
    hers.way_of = {'a': (-60., 0.), 'b': (60., 0.), 'c': (0., 60.)}.get
    token = THE_GUIDE.set(guide)
    try:
        choosing = _Choosing(moves, hers, meeting, ['a', 'b', 'c'])
        choosing.at = 4.1
        key, why, aim = choosing.key('')
        assert key in ('b', 'c') and why == 'meet' and aim is whole
        assert choosing.goal_binding.thing is whole and whole.number not in moves.things
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('source_changed', [False, True])
async def test_whole_material_measurement_runs_in_ordinary_observation_worker(scene, monkeypatch, source_changed):
    from core.perception import source_references_while_play_advances as work
    from core.perception.play_observation_without_holding_the_loop import open_play_observation

    binding, reference = attach(scene).binding, registered(scene)
    table = FlashVisualAssets('source', (reference.declaration,))
    moved(scene)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    state = copy.deepcopy(scene[-1].__dict__)

    class Eyes:
        def __init__(self, **_kwargs):
            self.things, self.pending, self.kinds = {}, {}, []
            self.scale = 1.

        def see(self, picture, at):
            self.__dict__.update(copy.deepcopy(state))
            return []

    class References:
        tables = (table,)
        frames = ((reference, binding),)
        proposals = source_frames = points = ()
        status = 'ready'
        retention = None

        def advance(self, observer, picture, at, tables):
            return ()

        def retire(self):
            return None

    calls = []

    def tables_now():
        calls.append(None)
        return () if source_changed and len(calls) > 1 else (table,)

    monkeypatch.setattr(work, 'source_tables_in_play', tables_now)
    observer = await open_play_observation(factory=Eyes)
    observer._references = References()
    try:
        assert await observer.see(picture, 4.1) == []
        if source_changed:
            assert not observer.source_referents
        else:
            whole, = observer.source_referents.values()
            assert whole.seen == 4.1 and whole.pixels == observer.pixels_consumed
        assert not observer.things
        assert await observer.see(picture, 4.1) is None
        assert not observer.source_referents
    finally:
        await observer.close()
        await asyncio.sleep(0)
    assert not observer.source_referents


def test_recorder_keeps_whole_identity_geometry_and_unknown_contact(scene):
    from core.perception.recording_the_frames_and_inputs import _source_reference_snapshot

    observer = scene[-1]
    observer.source_referents = observe(scene, WholeReferentsInView(), attach(scene).binding)
    whole, = observer.source_referents.values()
    retained = _source_reference_snapshot(observer)['whole_referents']
    assert retained['count'] == 1
    record, = retained['records']
    assert (record['number'], record['born'], record['kind']) == (whole.number, whole.born, whole.kind)
    assert record['pixels']['sha256'] == whole.pixels.sha256 and record['w'] == whole.w
    assert record['character_id'] == 17 and len(record['anchors']) == 8 and len(record['origin_path']) == 1
    assert record['role'] is None and record['contact_verified'] is None and record['vx'] is None


def test_whole_observation_boundary_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _whole_source_referent_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'whole_source_observations_keep_separate_identity_and_unknown_contact')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
