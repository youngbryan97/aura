"""An active measurement can retain visible anchors without creating or restoring points."""
from dataclasses import replace

import pytest

from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.perception.points_a_consumer_still_needs import MaterialRegion, MaterialRetention, PointsStillNeeded
from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction
from core.perception.points_that_keep_their_motion import SeenPoint

pytestmark = pytest.mark.unit


def material():
    points = tuple(SeenPoint(n, 8.+n*2, 12., 8.+n*2, 12., 1., 4., 4.1, .001,
                             2 if n < 3 else 20) for n in range(12))
    marks = MaterialMarks('1'*32, 20, PixelsConsumed(4.1, (64, 64, 3), '0'*64), 1., 1., points)
    return points, marks


def test_a_visually_valid_anchor_is_kept_when_the_unrequested_allocator_would_drop_it():
    points, marks = material()
    proposed = PointsBeforeReduction._spatial_share(list(points), (1., 1.), 6)
    assert 0 not in {p.number for p in proposed}
    ledger = PointsStillNeeded()
    request = MaterialRetention(marks.epoch, 4.1, 4.6, ((0, 1.), (1, 1.)))
    assert ledger.request(request, marks) == 2
    kept = ledger.keep(list(points), proposed, 4.2, 6)
    assert {0, 1} <= {p.number for p in kept} and len(kept) == 6
    assert sum(p.pictures == 20 for p in kept) == 4
    assert all(any(p is original for original in points) for p in kept)


def test_a_retention_request_cannot_restore_a_point_that_failed_visual_tracking():
    points, marks = material()
    ledger = PointsStillNeeded()
    ledger.request(MaterialRetention(marks.epoch, 4.1, 4.6, ((0, 1.),)), marks)
    visible = list(points[1:])
    kept = ledger.keep(visible, PointsBeforeReduction._spatial_share(visible, (1., 1.), 6), 4.2, 6)
    assert 0 not in {p.number for p in kept}


def test_expiry_and_birth_prevent_permanent_or_recycled_point_priority():
    points, marks = material()
    ledger = PointsStillNeeded()
    ledger.request(MaterialRetention(marks.epoch, 4.1, 4.6, ((0, 1.),)), marks)
    changed = [replace(points[0], born=4.2), *points[1:]]
    proposed = PointsBeforeReduction._spatial_share(changed, (1., 1.), 6)
    assert 0 not in {p.number for p in ledger.keep(changed, proposed, 4.2, 6)}
    ledger.keep(list(points), proposed, 4.6, 6)
    assert not ledger.requests


def test_other_streams_and_unmeasured_first_sightings_cannot_acquire_priority():
    _, marks = material()
    ledger = PointsStillNeeded()
    assert ledger.request(MaterialRetention('2'*32, 4.1, 4.6, ((0, 1.),)), marks) == 0
    first = replace(marks, points=(replace(marks.points[0], pictures=1),))
    assert ledger.request(MaterialRetention(marks.epoch, 4.1, 4.6, ((0, 1.),)), first) == 0


def test_point_retention_has_a_hard_fraction_of_the_existing_budget():
    points, marks = material()
    ledger = PointsStillNeeded()
    ledger.request(MaterialRetention(marks.epoch, 4.1, 4.6, tuple((p.number, p.born) for p in points)), marks)
    assert len(ledger.keep(list(points), (), 4.2, 6)) == 2
    with pytest.raises(ValueError):
        MaterialRetention(marks.epoch, 4.1, 5.2, ((0, 1.),))
    with pytest.raises(ValueError):
        MaterialRetention(marks.epoch, 4.1, 4.6, tuple((n, 1.) for n in range(65)))


def region_request():
    points, marks = material()
    points = (points[0], replace(points[1], x=20., old_x=20.), replace(points[2], x=8., y=24., old_x=8., old_y=24.))
    marks = replace(marks, points=points)
    names = tuple((p.number, p.born) for p in points)
    request = MaterialRetention(marks.epoch, 4.1, 4.6, names, (MaterialRegion((4., 8., 24., 28.), names),))
    return points, marks, request


def test_sampling_attention_moves_with_current_material_instead_of_a_stale_box():
    points, marks, request = region_request()
    ledger = PointsStillNeeded()
    ledger.request(request, marks)
    current = [replace(p, x=p.x+3., y=p.y+2., pictures=p.pictures+1) for p in points]
    mask = ledger.priority(current, (1., 1.), (64, 64), 4.2)
    assert mask is not None and mask[20, 15] and not mask[5, 5]
    assert mask.sum() <= 64*64//4
    assert not any(hasattr(request, name) for name in ('controlled', 'role', 'body'))


def test_missing_deformed_expired_or_excessive_regions_supply_no_sampling_mask():
    points, marks, request = region_request()
    ledger = PointsStillNeeded()
    ledger.request(request, marks)
    assert ledger.priority(list(points[:2]), (1., 1.), (64, 64), 4.2) is None
    deformed = [points[0], points[1], replace(points[2], x=40.)]
    assert ledger.priority(deformed, (1., 1.), (64, 64), 4.2) is None
    assert ledger.priority(list(points), (1., 1.), (64, 64), 4.6) is None
    oversized = replace(request, regions=(MaterialRegion((0., 0., 63., 63.), request.names),))
    ledger.request(oversized, marks)
    assert ledger.priority(list(points), (1., 1.), (64, 64), 4.2) is None
