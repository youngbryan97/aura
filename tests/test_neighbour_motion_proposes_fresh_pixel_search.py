"""Independent neighbouring motion can propose a search, while pixels decide continuity."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception import points_before_the_picture_is_reduced as source
from tests.test_source_points_keep_their_visible_detail import render

pytestmark = pytest.mark.unit


def distracted_coarse_search(monkeypatch, *, all_guesses=True, tracker=None, owned=True):
    if tracker is None:
        tracker = source.PointsBeforeReduction(limit=64)
        first = render()
        tracker.see_source(first, first, .9)
        tracker.see_source(first, first, 1.)
    if owned:
        from core.perception.named_material_in_a_consumed_frame import MaterialMarks
        from core.perception.pixels_consumed_by_the_observer import consumed_pixels
        from core.perception.points_a_consumer_still_needs import MaterialRegion, MaterialRetention

        names = tuple((p.number, p.born) for p in tracker.points)
        marks = MaterialMarks('1'*32, 2, consumed_pixels(render(), 1.), 1., 1., tracker.points)
        request = MaterialRetention(marks.epoch, 1., 1.5, names, (MaterialRegion((120., 70., 180., 130.), names),))
        tracker.still_needed.request(request, marks)
    target = min(tracker._native_points, key=lambda p: np.hypot(p.x-150., p.y-100.))
    before = tracker._image
    original = source._follow

    def follow(old, new, positions):
        found, valid = original(old, new, positions)
        if old is before:
            affected = np.linalg.norm(positions-(target.x, target.y), axis=1) < 1e-8
            found[affected] = positions[affected]+(20., 12.)
        return found, valid

    monkeypatch.setattr(source, '_follow', follow)
    if all_guesses:
        from core.perception import patches_compared_on_a_common_grid as common

        monkeypatch.setattr(common, '_follow', follow)
    return tracker, target


@pytest.mark.parametrize('all_guesses', [False, True])
def test_an_independently_observed_neighbourhood_can_recover_a_bad_coarse_guess(monkeypatch, all_guesses):
    tracker, target = distracted_coarse_search(monkeypatch, all_guesses=all_guesses)
    after = render((151.25, 99.25))
    current = tracker.see_source(after, after, 1.1)
    recovered = next((p for p in current if p.number == target.number), None)
    assert recovered is not None
    assert (recovered.x, recovered.y) == pytest.approx((target.x+1.25, target.y-.75), abs=.25)
    assert recovered.born == target.born and recovered.pictures == target.pictures+1


def test_neighbour_motion_cannot_restore_a_patch_whose_current_pixels_disagree(monkeypatch):
    tracker, target = distracted_coarse_search(monkeypatch)
    after = render((151.25, 99.25))
    x, y = round(target.x+1.25), round(target.y-.75)
    after[y-6:y+7, x-6:x+7] = 15
    current = tracker.see_source(after, after, 1.1)
    assert target.number not in {p.number for p in current}


def test_nearby_motion_without_a_material_owner_cannot_continue_a_missing_name(monkeypatch):
    tracker, target = distracted_coarse_search(monkeypatch, owned=False)
    after = render((151.25, 99.25))
    current = tracker.see_source(after, after, 1.1)
    assert target.number not in {p.number for p in current}


def neighbourhood():
    from core.perception.points_that_keep_their_motion import SeenPoint

    old = tuple(SeenPoint(n, x, y, x, y, 1., 1., 1., .001, 1)
                for n, (x, y) in enumerate(((20., 20.), (32., 20.), (20., 32.), (32., 32.), (26., 26.))))
    measured = [replace(p, x=p.x+3., y=p.y-2., old_x=p.x, old_y=p.y, before=1., at=1.1, pictures=2) for p in old[:4]]
    return old, measured


@pytest.mark.parametrize('angle', [0., .12, -.2])
def test_rigid_neighbours_supply_only_a_bounded_search_proposal(angle):
    from core.perception.patch_searches_from_neighbours_that_were_seen import neighbour_patch_searches
    from core.perception.points_that_keep_their_motion import SeenPoint

    old, measured = neighbourhood()
    rotation = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    for i, point in enumerate(measured):
        x, y = (np.array((old[i].x, old[i].y))-(26., 26.)) @ rotation+(29., 24.)
        measured[i] = replace(point, x=float(x), y=float(y))
    hints = neighbour_patch_searches(old, measured, old[-1:], 1., 1.1)
    assert 1 <= len(hints) <= 3
    assert hints[0].position == pytest.approx((29., 24.)) and hints[0].angle == pytest.approx(angle)
    assert hints[0].number == old[-1].number and hints[0].before == 1. and hints[0].at == 1.1
    assert hints[0].error <= 1. and len(hints[0].neighbours) >= 3
    assert not isinstance(hints[0], SeenPoint)
    assert not any(hasattr(hints[0], name) for name in ('pixels', 'body', 'controlled', 'role', 'contact_verified'))


@pytest.mark.parametrize('damage', ['two', 'collinear', 'birth', 'stale', 'old_coordinates', 'bad_error',
                                   'duplicate', 'duplicate_target', 'foreign_target', 'gap', 'repeat', 'nan', 'huge'])
def test_unobserved_recycled_or_unresolved_neighbourhoods_supply_no_search(damage):
    from core.perception.patch_searches_from_neighbours_that_were_seen import neighbour_patch_searches

    old, measured = neighbourhood()
    target, at = old[-1:], 1.1
    if damage == 'two':
        measured = measured[:2]
    elif damage == 'collinear':
        old = tuple(replace(p, y=20., old_y=20.) for p in old)
        measured = [replace(p, old_y=20., y=18.) for p in measured]
        target = old[-1:]
    elif damage == 'birth':
        measured = [replace(p, born=.9) for p in measured]
    elif damage == 'stale':
        measured = [replace(p, at=1.05) for p in measured]
    elif damage == 'old_coordinates':
        measured = [replace(p, old_x=p.old_x+1.) for p in measured]
    elif damage == 'bad_error':
        measured = [replace(p, error=.03) for p in measured]
    elif damage == 'duplicate':
        measured = [measured[0]]*4
    elif damage == 'duplicate_target':
        target = target*2
    elif damage == 'foreign_target':
        target = (replace(target[0], born=.9),)
    elif damage == 'gap':
        at = 1.5
    elif damage == 'repeat':
        at = 1.
    else:
        old = tuple(replace(p, x=float('nan') if damage == 'nan' else 1e100) for p in old)
        target = old[-1:]
    assert neighbour_patch_searches(old, measured, target, 1., at) == ()


def test_recovered_marks_cannot_supply_their_own_search_witnesses(monkeypatch):
    from core.perception import patch_searches_from_neighbours_that_were_seen as module

    old, measured = neighbourhood()
    tracker = SimpleNamespace(_native_points=old, _at=1., still_needed=SimpleNamespace(requests={}))
    monkeypatch.setattr(module, 'verified_patch_journeys', lambda *_args: pytest.fail('self-validated search'))
    assert module.recover_from_neighbour_searches(tracker, [], np.empty(0), 1.1, [], measured) is measured


def test_distinct_accepted_hypotheses_cannot_silently_pick_one_identity(monkeypatch):
    from core.perception import patch_searches_from_neighbours_that_were_seen as module

    old, measured = neighbourhood()
    first = module.NeighbourPatchSearch(4, 1., 1., 1.1, (29., 24.), 0., .2, ((0, 1.), (1, 1.), (2, 1.)))
    second = replace(first, position=(40., 24.))
    monkeypatch.setattr(module, '_owned_patch_searches', lambda *_args: (first, second))
    monkeypatch.setattr(module, '_follow', lambda _a, _b, points: (points, np.ones(len(points), dtype=bool)))
    monkeypatch.setattr(module, 'verified_patch_journeys', lambda _a, _b, _c, _d, _starts, guesses, _forward:
                        (guesses, np.full(2, .001), np.zeros(2), np.ones(2, dtype=bool)))
    tracker = SimpleNamespace(_native_points=old, _at=1., _image=[], _colours=np.empty(0),
                              still_needed=SimpleNamespace(requests={}))
    current = module.recover_from_neighbour_searches(tracker, [], np.empty(0), 1.1, measured, measured)
    assert {p.number for p in current} == {0, 1, 2, 3}


def test_other_material_cannot_supply_a_missing_owners_motion_witnesses():
    from core.perception import patch_searches_from_neighbours_that_were_seen as module

    old, measured = neighbourhood()
    owner = tuple(((n, 1.), (old[n].x, old[n].y)) for n in (0, 1, 4))
    other = tuple(((n, 1.), (old[n].x, old[n].y)) for n in (1, 2, 3))
    tracker = SimpleNamespace(_native_points=old, _at=1., still_needed=SimpleNamespace(requests={(4, 1.): 1.5},
        regions=(((10., 10., 40., 40.), owner, 1.5), ((10., 10., 40., 40.), other, 1.5))))
    assert module.neighbour_patch_searches(old, measured, old[-1:], 1., 1.1)
    assert module._owned_patch_searches(tracker, measured, old[-1:], 1.1) == ()


@pytest.mark.parametrize('damage', ['expired', 'future', 'recycled', 'missing_request', 'target_not_owned',
                                  'duplicate', 'malformed', 'list_key', 'nan_birth', 'too_many_regions'])
def test_missing_or_invalid_material_membership_cannot_authorize_recovery(damage):
    from core.perception import patch_searches_from_neighbours_that_were_seen as module

    old, measured = neighbourhood()
    anchors = tuple(((p.number, p.born), (p.x, p.y)) for p in old)
    requests, until = {(4, 1.): 1.5}, 1.5
    if damage == 'expired':
        until = 1.1
    elif damage == 'future':
        until = 2.1
    elif damage == 'recycled':
        anchors = tuple(((p.number, .9), (p.x, p.y)) for p in old)
    elif damage == 'missing_request':
        requests = {}
    elif damage == 'target_not_owned':
        anchors = anchors[:-1]
    elif damage == 'duplicate':
        anchors = (anchors[0], *anchors)
    elif damage == 'malformed':
        anchors = (None, *anchors)
    elif damage == 'list_key':
        anchors = (([0, 1.], (20., 20.)), *anchors[1:])
    elif damage == 'nan_birth':
        anchors = (((0, float('nan')), (20., 20.)), *anchors[1:])
    regions = (((10., 10., 40., 40.), anchors, until),)
    if damage == 'too_many_regions':
        regions *= 5
    tracker = SimpleNamespace(_native_points=old, _at=1.,
        still_needed=SimpleNamespace(requests=requests, regions=regions))
    assert module._owned_patch_searches(tracker, measured, old[-1:], 1.1) == ()


def test_search_capacity_is_bounded_before_numerical_work(monkeypatch):
    from core.perception import patch_searches_from_neighbours_that_were_seen as module

    old, measured = neighbourhood()
    targets = tuple(replace(old[-1], number=n+4, x=26.+n*.1) for n in range(33))
    monkeypatch.setattr(module, 'rigid_motion', lambda *_args, **_kwargs: pytest.fail('unbounded request'))
    assert module.neighbour_patch_searches((*old[:4], *targets), measured, targets, 1., 1.1) == ()


def test_shared_current_cohorts_reuse_their_fit_with_each_points_own_projection(monkeypatch):
    from core.perception import patch_searches_from_neighbours_that_were_seen as module

    old, measured = neighbourhood()
    old = (*old, replace(old[-1], number=5, x=27., y=27.))
    original, calls = module.rigid_motion, []

    def fit(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'rigid_motion', fit)
    hints = module.neighbour_patch_searches(old, measured, old[-2:], 1., 1.1)
    assert len(calls) == 1
    assert len(hints) == 2 and hints[0].position == (29., 24.) and hints[1].position == (30., 25.)


def test_ordinary_observation_publishes_only_the_freshly_recovered_material(monkeypatch):
    from core.perception.play_observation_without_holding_the_loop import _seen
    from core.perception.what_moves_in_the_picture import WhatMoves

    eyes = WhatMoves(width=320, points=source.PointsBeforeReduction(limit=64))
    _seen(eyes, render(), .9)
    _events, old = _seen(eyes, render(), 1.)
    _tracker, target = distracted_coarse_search(monkeypatch, tracker=eyes.points)
    _events, current = _seen(eyes, render((151.25, 99.25)), 1.1)
    recovered = next((p for p in current['material_marks'].points if p.number == target.number), None)
    assert recovered is not None
    assert (recovered.x, recovered.y) == pytest.approx((target.x+1.25, target.y-.75), abs=.25)
    assert current['material_marks'].pixels == current['pixels_consumed'] and recovered.at == 1.1
    assert old['material_marks'].pixels.at == 1. and target.number in {p.number for p in old['material_marks'].points}


def test_neighbour_search_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.patch_searches_from_neighbours_that_were_seen import _neighbour_patch_search_invariant

    assert _neighbour_patch_search_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'neighbour_motion_can_only_propose_a_patch_search')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
