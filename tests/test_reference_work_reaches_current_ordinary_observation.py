"""A late reference result must reach current material through the normal observer."""
import asyncio
import copy
import threading
from dataclasses import replace
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception import source_references_while_play_advances as work
from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.position_from_the_same_points import PointPose
from core.perception.a_source_point_on_current_material import SourcePointSeen
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset, FlashVisualAssets
from core.perception.visual_references_for_current_sources import RegisteredVisualReference, ReferencesAcquired
from core.perception.what_moves_in_the_picture import Thing
from core.runtime.lockdep import checked_thread_semaphore

pytestmark = pytest.mark.unit


@pytest.fixture
def scene():
    pixels = np.random.default_rng(719).integers(10, 230, (64, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    picture = np.zeros((128, 160, 3), np.uint8)
    picture[30:94, 40:104] = pixels[..., :3]
    source = 'swf:'+'1'*64
    asset = DeclaredVisualAsset('arbitrary', 7, 'sprite', 0, 39, (10,))
    coordinates = SourceImageCoordinates(source, _digest(pixels), '0'*64, (64, 64),
        ((1., 0.), (0., 1.)), (12., 8.), .01)
    reference = RegisteredVisualReference(source, asset, coordinates, pixels)
    body = Thing(1, 72., 62., 64., 64., np.zeros(64), (120, 120, 120), pixels[..., :3].copy(),
                 1., 4., kind=0, support=np.ones((64, 64), bool))
    marks = dict(enumerate(((48., 38.), (94., 38.), (48., 84.), (94., 84.))))
    body.point_pose = PointPose(1., 4., (72., 62.), marks.copy(), (72., 62.), reference_at=4.,
                               visible_marks=marks.copy())
    observer = SimpleNamespace(things={1: body}, scale=1., pixels_consumed=consumed_pixels(picture, 4.))
    return reference, FlashVisualAssets(source, (asset,)), picture, observer


@pytest.fixture
def executor(monkeypatch):
    submitted = []

    def submit(fn, *args, **kwargs):
        future = Future()
        submitted.append((future, lambda: fn(*args)))
        return future

    monkeypatch.setattr(work, 'submit_blocking_io', submit)
    monkeypatch.setattr(work, '_SLOTS', checked_thread_semaphore('test.reference_work_slots', 1))
    return submitted


def complete(submitted):
    future, call = submitted[-1]
    future.set_running_or_notify_cancel()
    future.set_result(call())


def loaded(scene, monkeypatch, executor):
    reference, table, picture, observer = scene
    monkeypatch.setattr(work, 'acquire_visual_references', lambda *_args, **_kw: ReferencesAcquired((reference,)))
    worker = work.SourceReferencesInPlay(interval_s=0.)
    assert worker.advance(observer, picture, 4., (table,), now=0.) == ()
    complete(executor)
    worker.advance(observer, picture, 4., (table,), now=1.)
    assert worker.status == 'match_pending' and len(executor) == 2
    return worker


def moved(scene, at=4.1):
    _, _, picture, observer = scene
    body = observer.things[1]
    body.x += 3.
    body.y += 2.
    body.seen = body.point_pose.at = at
    body.point_pose.visible_marks = {n: (x+3., y+2.) for n, (x, y) in body.point_pose.visible_marks.items()}
    picture = np.roll(picture, (2, 3), axis=(0, 1))
    observer.pixels_consumed = consumed_pixels(picture, at)
    return picture


def test_a_late_measurement_is_transported_instead_of_applied_at_its_old_position(scene, monkeypatch, executor):
    worker = loaded(scene, monkeypatch, executor)
    reference, table, _, observer = scene
    picture = moved(scene)
    complete(executor)  # The matcher only sees its copied earlier capture.
    points = worker.advance(observer, picture, 4.1, (table,), now=2.)
    point, = points
    assert point.point.position == pytest.approx((55., 40.), abs=.05)
    assert point.point.at == 4.1 and point.source == table.source and point.exported_name == 'arbitrary'
    assert observer.things[1].point_pose.origin == (72., 62.)
    assert not hasattr(point, 'controlled') and not hasattr(point, 'role')
    worker.retire()


@pytest.mark.parametrize('damage', ['basis', 'birth', 'removed', 'pixels', 'source'])
def test_broken_delivery_cannot_publish_old_geometry(scene, monkeypatch, executor, damage):
    worker = loaded(scene, monkeypatch, executor)
    _, table, _, observer = scene
    picture = moved(scene)
    if damage == 'basis':
        observer.things[1].point_pose.reference_at = 4.1
    elif damage == 'birth':
        observer.things[1].born = 4.1
    elif damage == 'removed':
        observer.things.clear()
    elif damage == 'pixels':
        observer.pixels_consumed = None
    else:
        table = FlashVisualAssets('swf:'+'2'*64, table.assets)
    complete(executor)
    assert worker.advance(observer, picture, 4.1, (table,), now=2.) == ()
    worker.retire()


def test_a_running_job_cannot_queue_more_captures_or_return_its_permit_when_retired(scene, monkeypatch, executor):
    worker = loaded(scene, monkeypatch, executor)
    _, table, picture, observer = scene
    future = executor[-1][0]
    future.set_running_or_notify_cancel()
    for at in (5., 6., 7.):
        worker.advance(observer, picture, at, (table,), now=at)
    assert len(executor) == 2 and not work._SLOTS.acquire(blocking=False)
    worker.retire()
    assert not future.cancelled() and not work._SLOTS.acquire(blocking=False)
    future.set_result(executor[-1][1]())
    assert worker.advance(observer, picture, 8., (table,), now=8.) == ()
    assert work._SLOTS.acquire(blocking=False)
    work._SLOTS.release()


def test_capacity_is_checked_before_copying_a_capture(scene, monkeypatch, executor):
    worker = loaded(scene, monkeypatch, executor)
    _, table, picture, observer = scene
    complete(executor)
    worker.advance(observer, picture, 4., (table,), now=2.)
    complete(executor)
    worker.interval_s = 100.
    worker.advance(observer, picture, 4., (table,), now=3.)
    # The preceding step starts one final operation before the new cadence.
    complete(executor)
    worker._consume()
    assert work._SLOTS.acquire(blocking=False)
    monkeypatch.setattr(work, 'keep_material_frame', lambda *_args: pytest.fail('no permit, no copy'))
    worker._next = 0.
    worker.advance(observer, picture, 4., (table,), now=4.)
    assert worker.status == 'reference_capacity_unavailable'
    work._SLOTS.release()
    worker.retire()


def test_an_empty_appearance_search_does_not_discard_continuing_measured_geometry(scene, monkeypatch, executor):
    worker = loaded(scene, monkeypatch, executor)
    reference, table, picture, observer = scene
    complete(executor)
    assert worker.advance(observer, picture, 4., (table,), now=2.)
    worker._work.raw.set_result(work._Matched(reference, (), ('no_qualified_appearance',)))
    points = worker.advance(observer, picture, 4., (table,), now=3.)
    assert points and points[0].point.position == pytest.approx((52., 38.), abs=.05)
    worker.retire()


def test_a_result_valid_at_capture_cannot_replace_a_still_valid_binding_after_its_precision_expires(scene, monkeypatch, executor):
    from core.perception.names_for_observed_instances import query_for

    worker = loaded(scene, monkeypatch, executor)
    reference, table, picture, observer = scene
    complete(executor)
    assert worker.advance(observer, picture, 4., (table,), now=2.)
    good = worker.bindings[0][1]
    picture = moved(scene, at=4.1)
    fresh = replace(good, query=query_for(observer.things[1], observer, 4.1).with_picture(picture), registration_error=.75)
    assert fresh.current(observer, 4.1, source=reference.source) is not None
    picture = moved(scene, at=4.2)
    observer.things[1].point_pose.reference_error = .7
    assert fresh.current(observer, 4.2, source=reference.source) is None
    assert good.current(observer, 4.2, source=reference.source) is not None
    worker._work.raw.set_result(work._Matched(reference, (fresh,), ()))
    assert worker.advance(observer, picture, 4.2, (table,), now=3.)
    assert any(binding is good for _, binding in worker.bindings)
    assert 'reference_point_precision' in worker.refused
    worker.retire()


def test_measured_views_help_reacquisition_without_eliminating_broad_discovery(scene, monkeypatch, executor):
    worker = loaded(scene, monkeypatch, executor)
    reference, table, picture, observer = scene
    complete(executor)
    calls = []

    def no_match(ref, frame, **kwargs):
        calls.append(kwargs)
        return work._Matched(ref, (), ('no_qualified_appearance',))

    monkeypatch.setattr(work, '_matched', no_match)
    for now in (2., 3., 4., 5.):
        worker.advance(observer, picture, 4., (table,), now=now)
        complete(executor)
    assert calls[0]['view'] is not None and calls[1]['view'] is None and calls[2]['view'] is not None
    assert calls[0]['view'][0] == pytest.approx(1., abs=.01)
    assert worker.bindings  # No-match proposals cannot overwrite measured material.
    worker.advance(observer, picture, 4., (), now=6.)
    assert worker.views == {} and worker.points == ()
    worker.retire()


def test_a_running_match_cannot_cross_a_source_generation(scene, monkeypatch, executor):
    worker = loaded(scene, monkeypatch, executor)
    _, table, picture, observer = scene
    future, call = executor[-1]
    future.set_running_or_notify_cancel()
    other = FlashVisualAssets('swf:'+'2'*64, table.assets)
    assert worker.advance(observer, picture, 4., (other,), now=2.) == ()
    assert len(executor) == 2 and not work._SLOTS.acquire(blocking=False)
    future.set_result(call())
    assert worker.advance(observer, picture, 4., (other,), now=3.) == ()
    assert worker.bindings == () and not worker.views and len(executor) == 3
    worker.retire()


def test_pose_consensus_encloses_all_estimates_without_multiplying_confidence():
    a = work.SourceReferencePoint('source', 7, 'arbitrary', '0'*64, SourcePointSeen(1, 1., 4., (10., 10.), .2))
    b = work.SourceReferencePoint('source', 7, 'arbitrary', '0'*64, SourcePointSeen(1, 1., 4., (10.4, 10.), .3))
    points, refused = work._coalesced([a, b])
    point, = points
    assert not refused and point.point.position == pytest.approx((10.2, 10.))
    assert point.point.error == pytest.approx(.5) and point.hypotheses == 2
    far = work.SourceReferencePoint('source', 7, 'arbitrary', '0'*64, SourcePointSeen(1, 1., 4., (14., 10.), .3))
    assert work._coalesced([a, far]) == ((), ('reference_pose_disagreement',))


@pytest.mark.asyncio
async def test_ordinary_observation_starts_the_worker_and_publishes_only_current_points(scene, monkeypatch):
    from core.cognition.a_guide_to_a_place import Guide, THE_GUIDE
    from core.perception.play_observation_without_holding_the_loop import open_play_observation

    reference, table, picture, original = scene
    monkeypatch.setattr(work, 'acquire_visual_references', lambda *_args, **_kw: ReferencesAcquired((reference,)))
    token = THE_GUIDE.set(Guide(place='arbitrary', program_visual_assets=(table,)))
    main_thread, measured_threads = threading.get_ident(), []
    matcher = work._matched

    def matched(*args, **kwargs):
        measured_threads.append(threading.get_ident())
        return matcher(*args, **kwargs)

    monkeypatch.setattr(work, '_matched', matched)

    class Eyes:
        def __init__(self, kinds=None):
            self.__dict__.update(copy.deepcopy(vars(original)))
            self.kinds, self.pending = [], {}

        def see(self, rgb, at):
            body = self.things[1]
            body.seen = body.point_pose.at = at
            self.pixels_consumed = consumed_pixels(np.asarray(rgb), at)
            return []

    observation = await open_play_observation(factory=Eyes)
    try:
        assert await observation.see(picture, 4.) == []
        worker = observation._references
        assert worker is not None
        worker.interval_s, worker._next = 0., 0.
        await asyncio.wait_for(asyncio.wrap_future(worker._work.raw), 5.)
        assert await observation.see(picture, 4.1) == []
        await asyncio.wait_for(asyncio.wrap_future(worker._work.raw), 5.)
        assert await observation.see(picture, 4.2) == []
        point, = observation.source_reference_points
        assert point.point.at == 4.2 and point.point.position == pytest.approx((52., 38.), abs=.05)
        assert measured_threads and all(t != main_thread for t in measured_threads)
        THE_GUIDE.get().program_visual_assets = ()
        assert await observation.see(picture, 4.3) == []
        assert observation.source_reference_points == ()
        assert observation.source_reference_status == 'no_current_source'
    finally:
        pending = observation._references._work if observation._references is not None else None
        await observation.close()
        if pending is not None and not pending.raw.cancelled():
            await asyncio.wait_for(asyncio.wrap_future(pending.raw), 5.)
        THE_GUIDE.reset(token)
    assert observation.source_reference_points == ()


@pytest.mark.asyncio
async def test_geometry_before_object_grouping_requests_real_sampler_retention(scene, monkeypatch, executor):
    from core.cognition.a_guide_to_a_place import Guide, THE_GUIDE
    from core.perception.play_observation_without_holding_the_loop import open_play_observation
    from core.perception.what_moves_in_the_picture import WhatMoves

    reference, table, picture, _ = scene
    monkeypatch.setattr(work, 'acquire_visual_references', lambda *_args, **_kw: ReferencesAcquired((reference,)))
    token = THE_GUIDE.set(Guide(place='arbitrary', program_visual_assets=(table,)))
    observation = await open_play_observation(factory=WhatMoves)
    try:
        await observation.see(picture, 4.)
        worker = observation._references
        worker.interval_s, worker._next = 0., 0.
        complete(executor)
        await observation.see(picture, 4.1)
        complete(executor)
        await observation.see(picture, 4.2)
        assert observation.source_reference_frames and worker.retention is not None
        assert all(not hasattr(frame.point, 'number') for frame in observation.source_reference_frames)
        await observation.see(picture, 4.3)
        assert observation.material_retention_admitted >= 3
        assert observation.source_reference_frames[0].point.position == pytest.approx((52., 38.), abs=.1)
        assert not hasattr(observation.source_reference_frames[0], 'controlled')
    finally:
        await observation.close()
        THE_GUIDE.reset(token)


def test_retirement_canary():
    assert work._reference_worker_retirement_invariant() == ()


def test_two_current_copies_keep_two_instances_without_picking_the_player(scene, monkeypatch, executor):
    reference, table, picture, observer = scene
    wider = np.zeros((128, 300, 3), np.uint8)
    wider[:, :160] = picture
    wider[30:94, 160:224] = reference.pixels[..., :3]
    other = copy.deepcopy(observer.things[1])
    other.number, other.x = 2, 192.
    other.point_pose.origin = (192., 62.)
    other.point_pose.anchors = {n: (x+120., y) for n, (x, y) in other.point_pose.anchors.items()}
    other.point_pose.visible_marks = other.point_pose.anchors.copy()
    observer.things[2] = other
    observer.pixels_consumed = consumed_pixels(wider, 4.)
    worker = loaded((reference, table, wider, observer), monkeypatch, executor)
    complete(executor)
    points = worker.advance(observer, wider, 4., (table,), now=2.)
    assert {p.point.number for p in points} == {1, 2}
    assert sorted(p.point.position[0] for p in points) == pytest.approx([52., 172.], abs=.05)
    worker.retire()


@pytest.mark.asyncio
async def test_input_release_and_new_observations_continue_during_a_blocked_reference_match(scene, monkeypatch):
    from core.agency.a_trial_has_a_physical_end import lease_control_trial
    from core.agency.giving_a_trial_time import ControlTrial
    from core.agency.inputs_while_measuring_control import hold_control
    from core.agency.playing_as_it_happens import _Run
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import Guide, THE_GUIDE
    from core.perception.play_observation_without_holding_the_loop import open_play_observation
    import time

    reference, table, picture, original = scene
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(work, 'acquire_visual_references', lambda *_args, **_kw: ReferencesAcquired((reference,)))

    def blocked(ref, frame, **_kwargs):
        entered.set()
        assert release.wait(2.), 'test worker was not released'
        return work._Matched(ref, (), ('test_matcher_unmeasured',))

    monkeypatch.setattr(work, '_matched', blocked)

    class Eyes:
        def __init__(self, kinds=None):
            self.__dict__.update(copy.deepcopy(vars(original)))
            self.kinds, self.pending = [], {}

        def see(self, rgb, at):
            body = self.things[1]
            body.seen = body.point_pose.at = at
            self.pixels_consumed = consumed_pixels(np.asarray(rgb), at)
            return []

    actions = []

    class Hands:
        async def down(self, key):
            actions.append(('down', key))

        async def up(self, key):
            actions.append(('up', key))

    token = THE_GUIDE.set(Guide(place='arbitrary', program_visual_assets=(table,)))
    observation = await open_play_observation(factory=Eyes)
    pending = None
    try:
        await observation.see(picture, 4.)
        worker = observation._references
        worker.interval_s, worker._next = 0., 0.
        await asyncio.wait_for(asyncio.wrap_future(worker._work.raw), 2.)
        await observation.see(picture, 4.1)
        pending = worker._work.raw
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.001)
        assert entered.is_set() and not pending.done()
        run, hers, hands = _Run(keys=['opaque'], began=time.monotonic()), WhichIsHers(), Hands()
        await hold_control(hands, run, hers, 'opaque', time.monotonic(), trying=True)
        run.things = observation.things
        run.body_trial = ControlTrial('opaque', run.held_since, run.held_since+.01, 0)
        assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
        await asyncio.wait_for(run._trial_input_deadline, 1.)
        assert actions[-1] == ('up', 'opaque') and not pending.done()
        assert await observation.see(picture, 4.2) == []
        assert observation.things[1].seen == 4.2 and worker._work.raw is pending
    finally:
        await observation.close()
        release.set()
        if pending is not None:
            await asyncio.wait_for(asyncio.wrap_future(pending), 2.)
        THE_GUIDE.reset(token)
