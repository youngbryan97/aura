"""Independent image trials retain actual capture and acknowledged input phases."""
from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import collecting_independent_pixel_trials as collecting
from core.agency import inputs_while_measuring_control as input_lane
from core.agency.pixels_inside_control_trials import RandomizedPixelExperiment
from core.agency.what_a_press_does import WhatAPressDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.frames_as_they_are_drawn import CanvasFrameTime

pytestmark = pytest.mark.unit


def test_selected_settling_has_a_capture_contract():
    assert collecting._settled_pixel_capture_invariant() == ()


def test_selected_within_press_captures_are_bounded_and_keep_the_endpoint():
    assert collecting._pixel_sampling_time_invariant() == ()


class _Clock:
    def __init__(self):
        self.at = 10.

    def now(self):
        return self.at

    async def wait(self, seconds):
        self.at += seconds
        await asyncio.sleep(0)


@pytest.fixture
def lane(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(collecting, 'time', SimpleNamespace(monotonic=clock.now))
    monkeypatch.setattr(input_lane, 'time', SimpleNamespace(monotonic=clock.now))
    monkeypatch.setattr(collecting, 'asyncio', SimpleNamespace(
        sleep=clock.wait, create_task=asyncio.create_task, wait_for=asyncio.wait_for,
        shield=asyncio.shield, timeout=asyncio.timeout, CancelledError=asyncio.CancelledError))
    events = []

    class Hands:
        async def down(self, key):
            events.append(('down', key, clock.at))
            clock.at += .01

        async def up(self, key):
            events.append(('up', key, clock.at))
            clock.at += .01

        async def click(self):
            events.append(('click', '', clock.at))
            clock.at += .01

    run = SimpleNamespace(held='', held_since=clock.at, held_trying=False,
                          presses=WhatAPressDoes(), things={}, input_key_downs=Counter(),
                          responses=[], letting_go=[], charging=SimpleNamespace(key=''),
                          _input_clock_receipts=[])
    hers = WhichIsHers()
    owns = SimpleNamespace(current=True)
    inputs = collecting.PixelExperimentInputs(Hands(), run, hers, owns=lambda: owns.current)
    source = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)
    captures = []

    async def look():
        began = clock.at
        clock.at += .01
        at = CanvasFrameTime(began, lower=began, upper=clock.at, arrived=clock.at,
                             source='canvas-bracket', source_at=None, epoch=2,
                             surface='actual-document/canvas', sequence=len(captures) + 1)
        captures.append(at)
        return source, at

    return SimpleNamespace(clock=clock, events=events, inputs=inputs, run=run, hers=hers,
                           owns=owns, source=source, captures=captures, look=look)


@pytest.mark.asyncio
async def test_prespecified_schedule_uses_distinct_released_inputs_and_frozen_measured_pixels(lane):
    experiment = RandomizedPixelExperiment(('opaque-a', 'opaque-b'), seed=73)
    schedule = experiment.schedule
    result = await collecting.collect_independent_pixel_trials(experiment, look=lane.look, inputs=lane.inputs)
    assert result == collecting.PixelCollectionResult('complete', 24, True)
    assert [trial.schedule for trial in experiment.trials] == list(schedule)
    assert len(lane.captures) == 48 and len(lane.events) == 48
    assert [(kind, key) for kind, key, _at in lane.events] == [
        item for press in schedule for item in (('down', press.key), ('up', press.key))]
    assert all(trial.refusal() is None for trial in experiment.trials)
    for trial, before, after in zip(experiment.trials, lane.captures[::2], lane.captures[1::2], strict=True):
        assert (trial.before.lower, trial.before.upper) == (before.lower, before.upper)
        assert (trial.after.lower, trial.after.upper) == (after.lower, after.upper)
        assert trial.before.surface == before.surface and trial.after.sequence == after.sequence
    retained = experiment.trials[0].before.pixels
    lane.source.fill(0)
    assert retained == experiment.trials[0].before.pixels and any(retained)
    assert lane.run.held == ''


@pytest.mark.asyncio
async def test_captures_continue_inside_a_single_unchanged_physical_press(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    result = await collecting.collect_independent_pixel_trials(experiment, look=lane.look, inputs=lane.inputs,
                                                               settle_s=.5, sample_s=.1)
    assert result.status == 'complete' and result.trials == 24 and result.release_confirmed
    assert len(lane.captures) == 24 * 6 and len(lane.events) == 48
    for trial in experiment.trials:
        captured = [at for at in lane.captures if trial.down_ack < at.lower < trial.up_dispatch]
        assert len(captured) == 5 and trial.after.lower == captured[-1].lower
        assert trial.after.lower >= trial.down_ack + .5


@pytest.mark.asyncio
async def test_endpoint_capture_releases_input_before_processing_its_pixels(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    processed = []

    async def observe(frame):
        processed.append((frame.lower, lane.run.held))
        # Processing may be slow without lengthening the physical endpoint.
        lane.clock.at += .03

    result = await collecting.collect_independent_pixel_trials(experiment, look=lane.look,
        inputs=lane.inputs, settle_s=.5, sample_s=.1, after_capture=observe)
    assert result.status == 'complete' and result.release_confirmed
    for trial in experiment.trials:
        assert (trial.after.lower, '') in processed
        assert trial.up_dispatch == trial.after.upper
        assert any(trial.down_ack < at < trial.after.lower and held == trial.schedule.key
                   for at, held in processed)


@pytest.mark.asyncio
async def test_neutral_observer_latency_does_not_change_the_paired_physical_interval(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    lane.hers.holding('', lane.clock.at)
    observed_before = []
    observed = []

    async def observe(frame):
        observed.append((frame.lower, lane.hers._held_at(frame.lower), lane.run.held))
        if not lane.run.held:
            observed_before.append(frame.lower)
            lane.clock.at += .2 if len(observed_before) % 2 else .01

    result = await collecting.collect_independent_pixel_trials(experiment, look=lane.look,
        inputs=lane.inputs, settle_s=.5, sample_s=.1, after_capture=observe)
    assert result.status == 'complete' and result.trials == 24 and result.release_confirmed
    spans = [trial.after.lower - trial.before.upper for trial in experiment.trials]
    assert max(spans) - min(spans) < .001
    assert all(trial.before.lower not in observed_before for trial in experiment.trials)
    assert all(trial.down_dispatch == trial.before.upper for trial in experiment.trials)
    for trial in experiment.trials:
        baseline = next(row for row in observed if row[0] == trial.before.lower)
        assert baseline[1][1] == '' and baseline[2] == trial.schedule.key


@pytest.mark.asyncio
async def test_fresh_pre_dispatch_baseline_bridges_material_history(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    observed = []

    async def observe(frame):
        observed.append(frame.lower)
        if not lane.run.held:
            lane.clock.at += .18

    result = await collecting.collect_independent_pixel_trials(experiment, look=lane.look,
        inputs=lane.inputs, settle_s=.6, sample_s=.2, after_capture=observe)
    assert result.status == 'complete' and result.release_confirmed
    for trial in experiment.trials:
        baseline_index = observed.index(trial.before.lower)
        previous, baseline, held = observed[baseline_index-1:baseline_index+2]
        assert held - previous > .35
        assert baseline - previous <= .35 and held - baseline <= .35


@pytest.mark.asyncio
async def test_resolved_observation_releases_input_without_qualifying_a_partial_cohort(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    settled = []

    async def look():
        result = await lane.look()
        if lane.run.held and result[1].lower >= lane.run.held_since + .2:
            settled.append(result[1])
        return result

    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs,
        settle_s=.5, sample_s=.1, stop_when=lambda: len(settled) >= 2)
    assert result.status == 'observation_resolved' and result.release_confirmed
    assert not experiment.complete and experiment.refused == 'observation_resolved'
    assert not experiment.trials and lane.run.held == ''
    assert [event[0] for event in lane.events] == ['down', 'up']


@pytest.mark.asyncio
async def test_slow_observation_has_one_physical_press_deadline(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)

    async def look():
        if lane.run.held:
            await asyncio.sleep(.3)
        return await lane.look()

    began = asyncio.get_running_loop().time()
    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs,
        settle_s=.5, sample_s=.1, capture_s=.4)
    assert asyncio.get_running_loop().time() - began < 1.8
    assert result.status == 'capture_or_delivery_timeout' and result.release_confirmed
    assert not experiment.trials and lane.run.held == ''
    assert [event[0] for event in lane.events] == ['down', 'up']


@pytest.mark.asyncio
@pytest.mark.parametrize('missing', [None, (np.zeros((40, 60, 3), np.uint8), 10.)])
async def test_missing_capture_custody_retires_before_any_press(lane, missing):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)

    async def look():
        return missing

    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs)
    assert result.status in {'missing_capture', 'unmeasured_capture'}
    assert experiment.refused == result.status and not experiment.trials and not lane.events


@pytest.mark.asyncio
async def test_missing_after_picture_retires_and_releases_the_acknowledged_press(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)

    async def look():
        return await lane.look() if not lane.captures else None

    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs)
    assert result == collecting.PixelCollectionResult('missing_capture', 0, True)
    assert [kind for kind, _key, _at in lane.events] == ['down', 'up']
    assert lane.run.held == ''


@pytest.mark.asyncio
async def test_unrelated_click_invalidates_the_trial_instead_of_teaching_control(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)

    async def look():
        if lane.captures:
            await lane.inputs.click()
        return await lane.look()

    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs)
    assert result == collecting.PixelCollectionResult('input_interrupted', 0, True)
    assert [kind for kind, _key, _at in lane.events] == ['down', 'click', 'up']


@pytest.mark.asyncio
async def test_another_acknowledged_owner_is_never_released_by_old_trial_cleanup(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)

    async def look():
        if lane.captures:
            await input_lane.hold_control(lane.inputs, lane.run, lane.hers, 'other-owner', lane.clock.at)
        return await lane.look()

    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs)
    assert result.status == 'input_interrupted' and not result.release_confirmed
    assert not experiment.trials and lane.run.held == 'other-owner'
    assert not any(kind == 'up' and key == 'other-owner' for kind, key, _at in lane.events)


@pytest.mark.asyncio
async def test_cancellation_during_capture_releases_only_its_acknowledged_key(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    waiting = asyncio.Event()

    async def look():
        if lane.captures:
            waiting.set()
            await asyncio.Event().wait()
        return await lane.look()

    task = asyncio.create_task(collecting.collect_independent_pixel_trials(
        experiment, look=look, inputs=lane.inputs))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert experiment.refused == 'cancelled' and not experiment.trials
    assert [kind for kind, _key, _at in lane.events] == ['down', 'up']
    assert lane.inputs._collector is None and lane.run.held == ''


@pytest.mark.asyncio
async def test_cancellation_joins_actual_down_acknowledgement_before_releasing(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    dispatched, acknowledged = asyncio.Event(), asyncio.Event()
    original = lane.inputs.hands.down

    async def down(key):
        dispatched.set()
        await acknowledged.wait()
        await original(key)

    lane.inputs.hands.down = down
    task = asyncio.create_task(collecting.collect_independent_pixel_trials(
        experiment, look=lane.look, inputs=lane.inputs))
    await dispatched.wait()
    task.cancel()
    acknowledged.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert experiment.refused == 'cancelled' and not experiment.trials
    assert [kind for kind, _key, _at in lane.events] == ['down', 'up']
    assert lane.run.held == ''


@pytest.mark.asyncio
async def test_capture_timeout_has_a_finite_wait_and_releases_the_current_key(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)

    async def look():
        if lane.captures:
            await asyncio.Event().wait()
        return await lane.look()

    result = await collecting.collect_independent_pixel_trials(
        experiment, look=look, inputs=lane.inputs, capture_s=.01)
    assert result == collecting.PixelCollectionResult('capture_or_delivery_timeout', 0, True)
    assert lane.run.held == '' and [kind for kind, _key, _at in lane.events] == ['down', 'up']


@pytest.mark.asyncio
async def test_a_prior_user_action_and_an_oversize_schedule_are_refused_without_input(lane):
    lane.run.held, lane.run.held_since = 'user-key', lane.clock.at
    first = RandomizedPixelExperiment(('a', 'b'), seed=41)
    result = await collecting.collect_independent_pixel_trials(first, look=lane.look, inputs=lane.inputs)
    assert result.status == 'experiment_not_fresh_or_neutral' and lane.run.held == 'user-key'
    lane.run.held = ''
    oversize = RandomizedPixelExperiment(('a', 'b'), seed=41, per_key=14)
    result = await collecting.collect_independent_pixel_trials(oversize, look=lane.look, inputs=lane.inputs)
    assert result.status == 'experiment_not_fresh_or_neutral'
    assert not lane.events and not lane.captures


@pytest.mark.asyncio
async def test_after_press_work_cannot_reset_the_interference_baseline(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    async def after_press(_press):
        await lane.inputs.click()
        return True
    result = await collecting.collect_independent_pixel_trials(
        experiment, look=lane.look, inputs=lane.inputs, after_press=after_press)
    assert result.status == 'input_interrupted' and result.trials == 1
    assert not experiment.complete and lane.run.held == ''


@pytest.mark.asyncio
async def test_stopping_collection_still_releases_its_retained_physical_press(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    async def look():
        if lane.captures:
            lane.owns.current = False
        return await lane.look()
    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs)
    assert result.status == 'input_interrupted' and result.release_confirmed
    assert not experiment.complete and lane.run.held == ''
    assert [kind for kind, _key, _at in lane.events] == ['down', 'up']


@pytest.mark.asyncio
async def test_queued_pre_action_frame_is_skipped_within_the_capture_budget(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    queued = False
    async def look():
        nonlocal queued
        if lane.run.held and not queued:
            queued = True
            old = lane.captures[-1]
            return lane.source, old
        return await lane.look()
    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs)
    assert result.status == 'complete' and result.trials == 24 and result.release_confirmed
    assert queued and len(lane.captures) == 48


@pytest.mark.asyncio
async def test_a_post_action_frame_before_the_selected_settle_is_still_queued(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    queued = False

    async def look():
        nonlocal queued
        if lane.run.held and not queued:
            queued = True
            early = lane.run.held_since + .03
            return lane.source, CanvasFrameTime(early, lower=early, upper=early + .001,
                arrived=lane.clock.at, source='canvas-bracket', source_at=None, epoch=2,
                surface='actual-document/canvas', sequence=len(lane.captures) + 1)
        return await lane.look()

    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs,
                                                               settle_s=.2)
    assert result.status == 'complete' and result.trials == 24 and result.release_confirmed
    assert queued and all(press.after.lower >= press.down_ack + .2 for press in experiment.trials)
    assert all(press.before.lower >= press.neutral_began + .04 for press in experiment.trials)


@pytest.mark.asyncio
async def test_unchanging_queued_capture_cannot_extend_the_owned_hold(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    async def look():
        if lane.run.held:
            return lane.source, lane.captures[-1]
        return await lane.look()
    result = await collecting.collect_independent_pixel_trials(experiment, look=look, inputs=lane.inputs)
    assert result.status == 'queued_capture_budget' and result.trials == 0 and result.release_confirmed
    assert lane.run.held == ''


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_interrupt_the_owned_key_release(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    captured, releasing, acknowledge = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = lane.inputs.hands.up

    async def look():
        if lane.captures:
            captured.set()
            await asyncio.Event().wait()
        return await lane.look()

    async def up(key):
        releasing.set()
        await acknowledge.wait()
        await original(key)

    lane.inputs.hands.up = up
    task = asyncio.create_task(collecting.collect_independent_pixel_trials(
        experiment, look=look, inputs=lane.inputs))
    await captured.wait()
    task.cancel()
    await releasing.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    acknowledge.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert experiment.refused == 'cancelled' and not experiment.trials
    assert lane.inputs._cleanup.done() and lane.inputs._collector is None
    assert lane.inputs._pending_release is None and lane.inputs.release_problem is None
    assert lane.run.held == '' and [kind for kind, _key, _at in lane.events] == ['down', 'up']


@pytest.mark.asyncio
async def test_missing_down_acknowledgement_is_bounded_and_retains_dispatch_responsibility(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    acknowledge, cancelled = asyncio.Event(), asyncio.Event()
    original = lane.inputs.hands.down

    async def down(key):
        lane.events.append(('down-dispatched', key, lane.clock.at))
        try:
            await acknowledge.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await acknowledge.wait()
        await original(key)

    lane.inputs.hands.down = down
    result = await asyncio.wait_for(collecting.collect_independent_pixel_trials(
        experiment, look=lane.look, inputs=lane.inputs, ack_s=.01), 1.)
    assert result.status == 'capture_or_delivery_timeout' and not result.release_confirmed
    assert result.release_problem == lane.inputs.release_problem == 'dispatch_acknowledgement_unknown'
    assert lane.inputs.pending_dispatch == (experiment.schedule[0].key, lane.events[0][2])
    pending = lane.inputs._pending
    assert pending is not None and not pending.done() and not cancelled.is_set()
    assert lane.inputs._collector is None and not lane.inputs._idle()
    assert not experiment.trials and not any(kind == 'up' for kind, _key, _at in lane.events)
    acknowledge.set()
    await asyncio.wait_for(pending, 1.)
    assert lane.inputs._pending is pending and lane.inputs.pending_dispatch is not None
    assert await lane.inputs._finish(.05)
    assert lane.inputs.pending_dispatch is None and lane.inputs.release_problem is None
    assert lane.run.held == '' and lane.events[-1][0] == 'up'


@pytest.mark.asyncio
async def test_missing_up_acknowledgement_keeps_the_release_task_without_another_up(lane):
    experiment = RandomizedPixelExperiment(('a', 'b'), seed=41)
    acknowledge = asyncio.Event()
    original = lane.inputs.hands.up

    async def up(key):
        lane.events.append(('up-dispatched', key, lane.clock.at))
        await acknowledge.wait()
        await original(key)

    lane.inputs.hands.up = up
    result = await asyncio.wait_for(collecting.collect_independent_pixel_trials(
        experiment, look=lane.look, inputs=lane.inputs, ack_s=.01), 1.)
    assert not result.release_confirmed and result.release_problem == 'release_acknowledgement_unknown'
    pending = lane.inputs._pending_release
    assert pending is not None and not pending.done() and lane.inputs._collector is None
    assert sum(kind == 'up-dispatched' for kind, _key, _at in lane.events) == 1
    acknowledge.set()
    await asyncio.wait_for(pending, 1.)
    assert await lane.inputs._finish(.05)
    assert lane.inputs._pending_release is None and lane.inputs._owned_press is None
    assert lane.inputs.release_problem is None and lane.run.held == ''
