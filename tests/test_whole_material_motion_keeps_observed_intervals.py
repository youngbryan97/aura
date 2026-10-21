"""A whole visual referent needs observed motion, rather than the direct-state default."""
from dataclasses import replace

import numpy as np
import pytest

from core.perception.translation_from_visible_support import motion_status, motion_was_measured, told_directly
from core.perception.whole_referents_from_source_material import WholeReferentsInView
from tests.test_source_geometry_precedes_object_grouping import scene, attach, moved  # noqa: F401
from tests.test_source_material_renews_only_through_shared_pixels import reference

pytestmark = pytest.mark.unit


def observe(producer, ref, binding, observer, picture, at):
    point = binding.current(observer, at, source=ref.source)
    item, = producer.observe(observer, picture, at, ((ref, binding, point),)).values()
    return item


def test_a_new_whole_is_not_a_direct_state_or_a_measured_motion(scene):
    binding, ref = attach(scene).binding, reference(scene)
    whole = observe(WholeReferentsInView(), ref, binding, scene[-1], scene[2], 4.)
    assert not told_directly(whole)
    assert not motion_was_measured(whole, 4.) and motion_status(whole, 4.) == 'unavailable'
    assert whole.material_displacement is None
    assert whole.role is None and whole.contact_verified is None


def test_continuing_whole_exposes_its_actual_named_material_interval(scene):
    binding, ref, producer = attach(scene).binding, reference(scene), WholeReferentsInView()
    first = observe(producer, ref, binding, scene[-1], scene[2], 4.)
    observer = moved(scene)
    current = observe(producer, ref, binding, observer, np.roll(scene[2], (2, 3), axis=(0, 1)), 4.1)
    step = current.material_displacement
    assert step is not None and (step.number, step.born, step.reference_at) == (first.number, first.born, first.born)
    assert (step.before, step.at) == (4., 4.1)
    assert step.start == pytest.approx((52., 38.)) and step.end == pytest.approx((55., 40.))
    assert step.support >= 3 and 0 < step.error < 1.
    assert current.motion_position == step.end and current.motion_measured_at == 4.1
    assert motion_status(current, 4.1) == 'measured' and not told_directly(current)
    assert not observer.things and current.role is None and current.contact_verified is None


def continued(scene):
    binding, ref, producer = attach(scene).binding, reference(scene), WholeReferentsInView()
    first = observe(producer, ref, binding, scene[-1], scene[2], 4.)
    observer = moved(scene)
    current = observe(producer, ref, binding, observer, np.roll(scene[2], (2, 3), axis=(0, 1)), 4.1)
    return first, current, binding, ref, producer


@pytest.mark.parametrize('damage', ['pixels', 'epoch', 'missing', 'new_birth', 'interrupted',
                                   'source', 'reference', 'identity', 'gap', 'collinear'])
def test_uncertain_or_foreign_material_cannot_supply_a_whole_interval(scene, damage):
    from core.perception.a_whole_origin_carried_by_measured_material import whole_material_displacement

    old, current, _binding, _ref, _producer = continued(scene)
    current = replace(current, material_displacement=None)
    marks = scene[-1].material_marks
    if damage == 'pixels':
        marks = replace(marks, pixels=replace(marks.pixels, sha256='7'*64))
    elif damage == 'epoch':
        marks = replace(marks, epoch='2'*32)
    elif damage == 'missing':
        marks = replace(marks, points=marks.points[:2])
    elif damage == 'new_birth':
        marks = replace(marks, points=tuple(replace(p, born=3.8) for p in marks.points))
    elif damage == 'interrupted':
        marks = replace(marks, points=tuple(replace(p, before=4.05) for p in marks.points))
    elif damage == 'source':
        current = replace(current, source='different')
    elif damage == 'reference':
        current = replace(current, reference_sha256='8'*64)
    elif damage == 'identity':
        current = replace(current, number=current.number+1)
    elif damage == 'gap':
        old = replace(old, seen=3.7, born=3.7, pixels=replace(old.pixels, at=3.7),
                      origin_path=((3.7, *old.origin_path[-1][1:]),))
    else:
        marks = replace(marks, points=tuple(replace(p, old_y=50., y=52.) for p in marks.points))
    assert whole_material_displacement(old, current, marks) is None


def frame(at, sequence, *, surface='canvas'):
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime

    return CanvasFrameTime(at, lower=at-.001, upper=at+.001, arrived=at+.002,
                           source='canvas-bracket', source_at=at, epoch=1, surface=surface, sequence=sequence)


def test_actual_capture_and_input_intervals_consume_material_motion_without_retiming_averaged_velocity(scene):
    from core.agency.responses_from_measured_intervals import CapturedIntervals
    from core.perception.pixels_consumed_by_the_observer import consumed_pixels

    _old, previous, binding, ref, producer = continued(scene)
    observer = scene[-1]
    picture = np.roll(scene[2], (2, 10), axis=(0, 1))
    receipt = consumed_pixels(picture, 4.2)
    marks = observer.material_marks
    observer.material_marks = replace(marks, sequence=marks.sequence+1, pixels=receipt,
        points=tuple(replace(p, x=p.x+7., old_x=p.x, old_y=p.y, before=p.at, at=4.2,
                             pictures=p.pictures+1) for p in marks.points))
    observer.pixels_consumed = receipt
    current = observe(producer, ref, binding, observer, picture, 4.2)
    assert (current.vx, current.vy) == pytest.approx((50., 10.))
    assert current.material_displacement.before == previous.seen
    captures = CapturedIntervals()
    captures.remember(frame(4.1, 1))
    now = frame(4.2, 2)
    captures.remember(now)
    interval = captures.measured(current, now, lambda _at: (3.5, 'act', True), response_s=.05)
    assert interval is not None and interval.velocity == pytest.approx((70., 0.))
    assert interval.owner == (current.number, current.born, current.born)
    assert interval.before == 4.1 and interval.at == 4.2 and interval.error > 0
    captures = CapturedIntervals()
    captures.remember(frame(4.1, 1, surface='another'))
    captures.remember(now)
    assert captures.measured(current, now, lambda _at: (3.5, 'act', True), response_s=.05) is None


def test_a_material_interval_cannot_be_assigned_to_another_origin_or_identity(scene):
    _old, current, _binding, _ref, _producer = continued(scene)
    with pytest.raises(ValueError, match='own dated material interval'):
        replace(current, number=current.number+1)
    with pytest.raises(ValueError, match='own dated material interval'):
        replace(current, origin_path=(*current.origin_path[:-1], (4.1, 99., 99., .1)))
    assert not motion_was_measured(current, 4.2)


def test_motion_origin_does_not_follow_a_changed_artwork_box(scene):
    from core.perception.a_whole_origin_carried_by_measured_material import whole_material_displacement

    old, current, _binding, _ref, _producer = continued(scene)
    changed_bounds = replace(current, material_displacement=None, x=current.x+20., w=current.w+40.)
    step = whole_material_displacement(old, changed_bounds, scene[-1].material_marks)
    assert step == current.material_displacement


@pytest.mark.parametrize('damage', ['no_origin', 'wrong_origin_time', 'negative_origin_error'])
def test_unresolved_source_origin_cannot_supply_a_displacement(scene, damage):
    from core.perception.a_whole_origin_carried_by_measured_material import whole_material_displacement

    old, current, _binding, _ref, _producer = continued(scene)
    if damage == 'no_origin':
        old = replace(old, origin_path=())
    elif damage == 'wrong_origin_time':
        old = replace(old, origin_path=((3.9, 52., 38., .1),))
    else:
        old = replace(old, origin_path=((4., 52., 38., -.1),))
    assert whole_material_displacement(old, current, scene[-1].material_marks) is None


def test_recorded_whole_material_motion_keeps_its_actual_dates_and_values(scene):
    from core.perception.recording_the_frames_and_inputs import _source_reference_snapshot

    _old, current, _binding, _ref, _producer = continued(scene)
    observer = scene[-1]
    observer.source_referents = {current.number: current}
    snapshot = _source_reference_snapshot(observer)['whole_referents']['records'][0]
    observer.source_referents.clear()
    assert snapshot['motion_evidence'] == 'source material'
    assert snapshot['material_displacement']['before'] == 4.
    assert snapshot['material_displacement']['at'] == 4.1
    assert snapshot['material_displacement']['end'] == pytest.approx([55., 40.])


def test_whole_motion_has_a_registered_measured_claim():
    from core.perception.a_whole_origin_carried_by_measured_material import _whole_motion_origin_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _whole_motion_origin_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'whole_motion_cannot_default_to_direct_state')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
