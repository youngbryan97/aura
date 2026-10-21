"""Current matched artwork requests sampling before it has a tracked owner."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage

from core.perception import play_observation_without_holding_the_loop as observation
from core.perception import source_geometry_in_current_pixels as geometry
from core.perception.a_search_hint_for_these_pixels import frame_priority_mask, sampling_hint_matches
from core.perception.source_references_while_play_advances import _matched
from core.perception.a_material_frame_kept_for_its_owner import keep_material_frame
from core.perception.what_moves_in_the_picture import WhatMoves
from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance, _digest
from core.perception.reference_material_near_a_measured_pose import near_reference_appearances
from tests.test_fresh_source_geometry_reaches_ordinary_observation import scene, _state
from tests.test_pixel_search_hints_reach_owned_sampling import captured

pytestmark = pytest.mark.unit


def test_current_appearance_reaches_actual_sampler_without_an_existing_object(scene):
    proposal, table, _, current = scene
    current[70:110, 100:160] = np.random.default_rng(187).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    worker, eyes, at = geometry.SourceGeometryInCurrentPixels(), WhatMoves(), captured(2.)
    worker._essential_seconds = .3
    hint = worker.sampling_before_observation(current, at, (proposal,), (table,))
    assert hint is not None and frame_priority_mask(hint, current, at) is not None
    assert not eyes.things
    eyes.see(current, at, search_hint=hint)
    receipt = eyes.sampling_hint_receipt
    assert sampling_hint_matches(receipt, hint)
    assert receipt.fresh_priority_points >= 6 and receipt.background_points > 0
    assert len(eyes.material_marks.points) <= eyes.points.limit
    frame = keep_material_frame(eyes, current, float(at))
    matched = _matched(proposal.reference, frame)
    assert matched.frames, matched.refused
    assert not hasattr(hint, 'number') and not hasattr(hint, 'role')


def test_prepared_fit_is_reused_after_observation_instead_of_fitting_twice(scene, monkeypatch):
    proposal, table, _, current = scene
    worker, at, calls = geometry.SourceGeometryInCurrentPixels(), captured(2.), []
    worker._essential_seconds = .3
    original = geometry.near_reference_appearances

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry, 'near_reference_appearances', counted)
    assert worker.sampling_before_observation(current, at, (proposal,), (table,)) is not None
    frames, _ = worker.measure_if_budget(current, 2., _state(current, 2.), (proposal,), (table,),
                                          essential_seconds=.09)
    assert len(frames) == 1 and len(calls) == 1
    assert calls == [dict(position=0)] and worker._essential_seconds == .09


@pytest.mark.parametrize('damage', ['hidden', 'source', 'future', 'plain_clock', 'no_budget'])
def test_absent_current_evidence_cannot_turn_a_proposal_into_priority(scene, damage):
    proposal, table, _, current = scene
    worker, at = geometry.SourceGeometryInCurrentPixels(), captured(2.)
    worker._essential_seconds = .3
    if damage == 'hidden':
        current = np.zeros_like(current)
    elif damage == 'source':
        table = replace(table, source='swf:'+'4'*64)
    elif damage == 'future':
        proposal = replace(proposal, capture=replace(proposal.capture, at=3.))
    elif damage == 'plain_clock':
        at = 2.
    else:
        worker._essential_seconds = 0.
    assert worker.sampling_before_observation(current, at, (proposal,), (table,)) is None


def test_generated_hint_and_prepared_geometry_cannot_cross_pixel_captures(scene):
    proposal, table, _, current = scene
    worker, at = geometry.SourceGeometryInCurrentPixels(), captured(2.)
    worker._essential_seconds = .3
    hint = worker.sampling_before_observation(current, at, (proposal,), (table,))
    assert hint is not None
    changed = current.copy()
    changed[0, 0, 0] ^= 1
    assert frame_priority_mask(hint, changed, at) is None
    assert frame_priority_mask(hint, current, captured(2., sequence=5)) is None
    frames, status = worker.measure_if_budget(changed, 2., _state(changed, 2.), (proposal,), (table,),
                                              essential_seconds=.09)
    assert not frames and status == ('fresh_reference_capture_unavailable',)


def test_ordinary_worker_earns_sampling_work_and_keeps_unsearched_copies_explicit(scene, monkeypatch):
    proposal, table, _, current = scene
    proposal = replace(proposal, appearances=proposal.appearances*2)
    clock = [0.]
    monkeypatch.setattr(observation.time, 'perf_counter', lambda: clock[0])

    class Eyes(WhatMoves):
        def see(self, *args, **kwargs):
            result = super().see(*args, **kwargs)
            clock[0] += .09
            return result

    eyes, worker = Eyes(), geometry.SourceGeometryInCurrentPixels()
    _, first = observation._seen(eyes, current, captured(2.), fresh_references=worker,
        reference_proposals=(proposal,), reference_tables=(table,))
    assert first['sampling_hint_receipt'] is None
    assert first['fresh_source_reference_status'] == ('fresh_reference_work_deferred',)
    _, second = observation._seen(eyes, current, captured(2.1, sequence=5), fresh_references=worker,
        reference_proposals=(proposal,), reference_tables=(table,))
    assert second['sampling_hint_receipt'].consumed
    assert 'fresh_reference_alternatives_unsearched' in second['fresh_source_reference_status']
    assert 'fresh_reference_positions_unsearched' in second['fresh_source_reference_status']
    assert second['fresh_source_reference_frames']
    key, binding = second['fresh_source_material_frames'][0]
    assert key == proposal.reference.key and binding.at == 2.1
    assert len(binding.anchors) >= 3


def test_source_departure_retires_sampling_hints_without_any_new_catalog(scene):
    proposal, table, _, current = scene
    worker = geometry.SourceGeometryInCurrentPixels()
    worker._essential_seconds = .3
    assert worker.sampling_before_observation(current, captured(2.), (proposal,), (table,))
    assert worker.records
    assert worker.sampling_before_observation(current, captured(3.), (), ()) is None
    assert not worker.records


def test_current_material_handoff_requires_source_and_pixel_custody(scene):
    from core.perception.source_references_while_play_advances import _with_current_material_frames

    proposal, table, _, current = scene
    eyes, worker, at = WhatMoves(), geometry.SourceGeometryInCurrentPixels(), captured(2.)
    worker._essential_seconds = .3
    observation._seen(eyes, current, at, fresh_references=worker,
        reference_proposals=(proposal,), reference_tables=(table,))
    assert len(worker.material_frames) == 1
    kept, refused = _with_current_material_frames((), worker.material_frames, eyes, 2., (table,))
    assert len(kept) == 1 and not refused
    renewed, refused = _with_current_material_frames(kept, worker.material_frames, eyes, 2., (table,))
    assert renewed == kept and not refused
    assert not _with_current_material_frames((), worker.material_frames, eyes, 2.1, (table,))[0]
    assert not _with_current_material_frames((), worker.material_frames, eyes, 2., ())[0]
    assert not _with_current_material_frames((), ('unreadable',), eyes, 2., (table,))[0]


def test_fresh_material_reaches_the_ordinary_reference_owner_and_retention(scene):
    from core.perception.source_references_while_play_advances import SourceReferencesInPlay

    proposal, table, _, current = scene
    eyes, geometry_worker = WhatMoves(), geometry.SourceGeometryInCurrentPixels()
    geometry_worker._essential_seconds = .3
    _, state = observation._seen(eyes, current, captured(2.), fresh_references=geometry_worker,
        reference_proposals=(proposal,), reference_tables=(table,))
    eyes.fresh_source_material_frames = state['fresh_source_material_frames']
    owner = SourceReferencesInPlay()
    owner.tables, owner.catalog, owner._loaded = (table,), (proposal.reference,), True
    owner._next = float('inf')
    try:
        owner.advance(eyes, current, 2., (table,), now=2.)
        assert len(owner.source_frames) == 1 and owner.source_frames[0].point.at == 2.
        assert owner.retention is not None and len(owner.retention.names) >= 3
        assert not eyes.things
    finally:
        owner.retire()


def test_refit_and_material_attachment_both_repay_their_optional_cost(scene, monkeypatch):
    proposal, table, _, current = scene
    worker, clock = geometry.SourceGeometryInCurrentPixels(), [0.]
    worker._essential_seconds = .09
    monkeypatch.setattr(geometry.time, 'perf_counter', lambda: clock[0])
    refit, measure = geometry.near_reference_appearances, worker._measured

    def timed_refit(*args, **kwargs):
        clock[0] += .03
        return refit(*args, **kwargs)

    def timed_measure(*args, **kwargs):
        clock[0] += .06
        return measure(*args, **kwargs)

    monkeypatch.setattr(geometry, 'near_reference_appearances', timed_refit)
    monkeypatch.setattr(worker, '_measured', timed_measure)
    assert worker.sampling_before_observation(current, captured(2.), (proposal,), (table,))
    assert worker.work_share.active
    worker.measure_if_budget(current, 2., _state(current, 2.), (proposal,), (table,), essential_seconds=.09)
    assert not worker.work_share.active and worker.work_share.credit == pytest.approx(-.06)
    assert worker.sampling_before_observation(current, captured(2.1), (proposal,), (table,)) is None


def test_current_reference_sampling_has_a_measured_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert geometry._reference_sampling_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    test = next(item for item in suite.tests() if item.name == 'source_appearances_reach_current_owned_sampling')
    assert test.predict(None) is True and any(claim.test == test.name for claim in suite.claims())


@pytest.mark.parametrize('offset', [(45., 90.), (-32., 30.), (105., 25.)])
def test_refit_preserves_affine_shape_when_material_crosses_the_capture_edge(offset):
    rgba = np.random.default_rng(349).integers(0, 255, (80, 96, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    linear = np.array(((1.1, .2), (.07, .9)))
    inverse, shift = np.linalg.inv(linear), np.asarray(offset)
    # The scene renderer uses output-to-input coordinates independently of
    # the refitter's cropped-template producer.
    picture = np.stack([ndimage.affine_transform(rgba[..., c], inverse[::-1, ::-1],
        offset=(-inverse @ shift)[::-1], output_shape=(128, 192), order=1,
        mode='constant', cval=0.) for c in range(3)], axis=-1)
    proposal = ReferenceAppearance(_digest(rgba), '0'*64, tuple(map(tuple, linear)),
        tuple(shift), (0., 0., 1., 1.), (), (), 0.)
    fits = near_reference_appearances(picture, rgba, (proposal,), position=0)
    assert fits
    actual = fits[0]
    assert np.asarray(actual.linear) == pytest.approx(linear, abs=.015)
    assert actual.offset == pytest.approx(offset, abs=.5)
    assert len(actual.source_points) >= 6
    assert not near_reference_appearances(picture*0, rgba, (proposal,), position=0)
