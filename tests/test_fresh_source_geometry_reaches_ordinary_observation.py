"""Fresh reference pixels can supply geometry while object ownership is unresolved."""
import asyncio
import threading
import time
import hashlib
from dataclasses import asdict, replace

import numpy as np
import pytest

from core.perception.a_visual_reference_in_the_picture import _digest, reference_appearances
from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.source_geometry_in_current_pixels import (
    CurrentSourceReference, SourceGeometryInCurrentPixels, _enclosed_measurements, _fresh_source_capture_invariant,
)
from core.perception.source_reference_proposals import SourceReferenceProposal
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset, FlashVisualAssets
from core.perception.visual_references_for_current_sources import RegisteredVisualReference


@pytest.fixture
def scene():
    rgba = np.random.default_rng(113).integers(0, 255, (40, 48, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    picture = np.zeros((120, 180, 3), np.uint8)
    picture[20:60, 30:78] = rgba[..., :3]
    poses = reference_appearances(picture, rgba, scales=(1.,))
    assert len(poses) == 1
    source = 'swf:'+'3'*64
    asset = DeclaredVisualAsset('unfamiliar', 17, 'sprite', 0, 39, (10,))
    coords = SourceImageCoordinates(source, _digest(rgba), '0'*64, (48, 40), ((1., 0.), (0., 1.)), (12., 8.), .01)
    reference = RegisteredVisualReference(source, asset, coords, rgba)
    proposal = SourceReferenceProposal(reference, consumed_pixels(picture, 1.), poses, 1)
    current = np.zeros_like(picture)
    current[22:62, 33:81] = rgba[..., :3]
    return proposal, FlashVisualAssets(source, (asset,)), picture, current


def _state(picture, at):
    receipt = consumed_pixels(picture, at)
    return dict(pixels_consumed=receipt, material_marks=MaterialMarks('0'*32, 1, receipt, 1., 1., ()))


def test_geometry_is_fresh_without_any_canonical_object_or_named_global_point(scene):
    proposal, table, _, current = scene
    frames, unavailable = SourceGeometryInCurrentPixels().measure(current, 2., _state(current, 2.), (proposal,), (table,))
    frame, = frames
    assert not unavailable and frame.point.position == pytest.approx((45., 30.), abs=.3)
    assert frame.point.support >= 6 and frame.point.error <= 1.
    assert frame.pixels == consumed_pixels(current, 2.) and frame.point.at == 2.
    assert not any(hasattr(frame.point, name) for name in ('number', 'kind', 'controlled', 'role'))
    assert _fresh_source_capture_invariant() == ()


@pytest.mark.parametrize('damage', ['receipt', 'source', 'future', 'hidden'])
def test_unconsumed_pixels_wrong_source_and_future_evidence_cannot_supply_geometry(scene, damage):
    proposal, table, old, current = scene
    state = _state(current, 2.)
    if damage == 'receipt':
        state = _state(old, 1.)
    elif damage == 'source':
        table = replace(table, source='swf:'+'4'*64)
    elif damage == 'future':
        proposal = replace(proposal, capture=replace(proposal.capture, at=3.))
    else:
        current = np.zeros_like(current)
        state = _state(current, 2.)
    frames, _ = SourceGeometryInCurrentPixels().measure(current, 2., state, (proposal,), (table,))
    assert not frames


def test_source_replacement_erases_the_old_search_proposal(scene):
    proposal, table, _, current = scene
    fresh = SourceGeometryInCurrentPixels()
    assert fresh.measure(current, 2., _state(current, 2.), (proposal,), (table,))[0]
    assert fresh.measure(current, 3., _state(current, 3.), (), ())[0] == ()
    assert not fresh.records
    assert fresh.measure(current, 4., _state(current, 4.), (), (table,))[0] == ()


def test_too_many_proposed_copies_cannot_be_silently_pruned_to_a_unique_one(scene):
    proposal, table, _, current = scene
    proposal = replace(proposal, appearances=proposal.appearances*5)
    frames, unavailable = SourceGeometryInCurrentPixels().measure(current, 2., _state(current, 2.), (proposal,), (table,))
    assert not frames and 'fresh_reference_pose_capacity_unavailable' in unavailable


def test_far_source_origins_need_the_same_precision_at_their_lever_arm(scene):
    proposal, table, _, current = scene
    coords = replace(proposal.reference.coordinates, offset=(10000., 10000.))
    proposal = replace(proposal, reference=replace(proposal.reference, coordinates=coords))
    frames, unavailable = SourceGeometryInCurrentPixels().measure(current, 2., _state(current, 2.), (proposal,), (table,))
    assert not frames and 'fresh_reference_precision_unavailable' in unavailable


def test_fresh_pose_consensus_encloses_errors_without_multiplying_support(scene):
    from core.perception.a_source_point_on_visible_marks import VisibleSourcePoint

    proposal, table, _, current = scene
    capture = consumed_pixels(current, 2.)
    a = CurrentSourceReference(table.source, 17, 'unfamiliar', proposal.reference.coordinates.reference_sha256,
                              capture, VisibleSourcePoint(2., (10., 10.), .3, 8))
    b = replace(a, point=VisibleSourcePoint(2., (10.4, 10.), .2, 6))
    point, = _enclosed_measurements((a, b))
    assert point.point.position == pytest.approx((10.2, 10.)) and point.point.error == pytest.approx(.5)
    assert point.point.support == 6 and point.hypotheses == 2
    far = replace(a, point=VisibleSourcePoint(2., (20., 10.), .2, 6))
    assert len(_enclosed_measurements((a, far))) == 2


def test_a_fresh_bounded_search_can_recover_after_old_search_capacity_is_exceeded(scene):
    proposal, table, _, current = scene
    fresh = SourceGeometryInCurrentPixels()
    too_many = replace(proposal, appearances=proposal.appearances*5)
    assert not fresh.measure(current, 2., _state(current, 2.), (too_many,), (table,))[0]
    replacement = replace(proposal, revision=2)
    frames, unavailable = fresh.measure(current, 3., _state(current, 3.), (replacement,), (table,))
    assert frames and 'fresh_reference_prior_search_retired' in unavailable


@pytest.mark.parametrize('clock', ['plain', 'canvas', 'page'])
def test_ordinary_observation_returns_the_current_measurement_and_clears_it_on_close(scene, monkeypatch, clock):
    from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime
    from core.perception import source_references_while_play_advances as work
    from core.perception.play_observation_without_holding_the_loop import open_play_observation

    proposal, table, _, current = scene
    at = 2.
    if clock != 'plain':
        cls = CanvasFrameTime if clock == 'canvas' else MeasuredFrameTime
        at = cls(2., lower=2., upper=2.01, arrived=2.02, source='capture-test', source_at=None, epoch=7,
                 **(dict(surface='test-surface', sequence=12) if clock == 'canvas' else {}))
    received = []

    class Eyes:
        def __init__(self, **kwargs):
            self.things, self.pending, self.kinds = {}, {}, []
            self.scale = 1.

        def see(self, picture, at):
            received.append(at)
            self.__dict__.update(_state(picture, float(at)))
            return []

    class References:
        tables = (table,)
        proposals = (proposal,)
        retention = None
        source_frames = ()
        status = 'ready'

        def advance(self, observer, picture, at, tables):
            assert type(at) is float and at == 2.
            return ()

        def retire(self):
            return None

    async def run():
        monkeypatch.setattr(work, 'source_tables_in_play', lambda: (table,))
        observer = await open_play_observation(factory=Eyes)
        observer._references = References()
        assert await observer.see(current, at) == []
        frame, = observer.fresh_source_reference_frames
        assert frame.pixels == observer.pixels_consumed and frame.point.at == 2.
        assert not observer.things and not observer.source_reference_frames
        assert isinstance(received[0], type(at))
        if clock != 'plain':
            assert received[0].lower == at.lower and received[0].source == at.source and received[0].epoch == 7
        if clock == 'canvas':
            assert received[0].surface == 'test-surface' and received[0].sequence == 12
        await observer.close()
        assert observer.fresh_source_reference_frames == ()

    asyncio.run(run())


def test_input_lease_releases_while_fresh_pixel_remeasurement_is_blocked():
    from core.agency.a_trial_has_a_physical_end import lease_control_trial
    from core.agency.giving_a_trial_time import ControlTrial
    from core.agency.inputs_while_measuring_control import hold_control
    from core.agency.playing_as_it_happens import _Run
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.play_observation_without_holding_the_loop import open_play_observation
    from tests.test_play_observation_keeps_input_deadlines_alive import Eyes

    async def run():
        entered, release = threading.Event(), threading.Event()

        class Delayed:
            def measure(self, *args):
                entered.set()
                assert release.wait(2.), 'test measurement was not released'
                return (), ()

        observer = await open_play_observation(factory=Eyes)
        observer._fresh_references = Delayed()
        task = asyncio.create_task(observer.see(np.zeros((32, 32, 3), np.uint8), time.monotonic()))
        try:
            assert await asyncio.to_thread(entered.wait, 1.)
            run, hers, actions = _Run(keys=['opaque'], began=time.monotonic()), WhichIsHers(), []

            class Hands:
                async def down(self, key):
                    actions.append(('down', key))

                async def up(self, key):
                    actions.append(('up', key))

            hands = Hands()
            await hold_control(hands, run, hers, 'opaque', time.monotonic(), trying=True)
            run.things = observer.things
            run.body_trial = ControlTrial('opaque', run.held_since, run.held_since+.01, 0)
            assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
            await asyncio.wait_for(run._trial_input_deadline, 1.)
            assert actions[-1] == ('up', 'opaque') and not run.held and not release.is_set()
        finally:
            release.set()
            await asyncio.wait_for(task, 1.)
            await observer.close()
        assert observer._fresh_references is None

    asyncio.run(run())


@pytest.mark.parametrize('damage', [False, True])
def test_recorded_checker_verifies_shape_prefixed_receipt_against_actual_rgb(scene, monkeypatch, damage):
    from core.perception import a_visual_reference_in_the_picture as matching
    from tools import measure_ordinary_reference_worker as replay
    from tools import measure_source_points_on_recorded_material as recorded

    proposal, table, _, current = scene
    state = _state(current, 2.)
    frames, _ = SourceGeometryInCurrentPixels().measure(current, 2., state, (proposal,), (table,))
    frame = asdict(frames[0])
    if damage:
        frame['pixels']['sha256'] = '0'*64
    digest = hashlib.sha256(current.tobytes()).hexdigest()
    event = dict(sequence=1, capture=dict(at=2.), raw_sha256=digest)
    result = dict(recording='unused', reference_bundle='unused', frames=[dict(sequence=1, at=2.,
        rgb_sha256=digest, points=[], source_frames=[], fresh_source_frames=[frame],
        sample_grid=asdict(state['material_marks'].grid))])
    from core.perception import visual_references_for_current_sources as catalog
    monkeypatch.setattr(catalog, 'registered_reference', lambda *_a, **_k: proposal.reference)
    monkeypatch.setattr(recorded, '_frames', lambda *_a: ([event], []))
    monkeypatch.setattr(recorded, '_pixels', lambda *_a: current)
    actual_matcher = matching.reference_appearances

    def fresh(*_args, **_kwargs):
        assert not damage, 'receipt validation must precede matching'
        return actual_matcher(current, proposal.reference.pixels, scales=(1.,))

    monkeypatch.setattr(matching, 'reference_appearances', fresh)
    if damage:
        with pytest.raises(ValueError, match='consumed RGB capture'):
            replay.check_published_points(result)
    else:
        checked = replay.check_published_points(result)
        assert checked['measured'] == 1 and checked['checks'][0]['capture_receipt_verified'] is True


def test_other_observation_channels_redirect_optional_work_without_starving_audits():
    from core.perception.source_geometry_in_current_pixels import _reference_attention_invariant, _reference_to_measure

    keys = ('first', 'second')
    first_missing = [_reference_to_measure(keys, {'second'}, i) for i in range(8)]
    second_missing = [_reference_to_measure(keys, {'first'}, i) for i in range(8)]
    assert first_missing.count('first') == second_missing.count('second') == 7
    assert first_missing.count('second') == second_missing.count('first') == 1
    assert [_reference_to_measure(keys, set(), i) for i in range(4)] == ['first', 'second']*2
    assert _reference_attention_invariant() == ()
