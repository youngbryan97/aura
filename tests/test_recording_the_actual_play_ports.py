"""Raw frames and input boundaries survive independently of a game's own claims."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core.perception.frames_as_they_are_drawn import CanvasFrameTime
from core.perception.recording_the_frames_and_inputs import (
    FrameAndInputRecording,
    _recorded_pixel_invariant,
    record_frames_and_inputs,
    recorded_play_ports,
)

pytestmark = pytest.mark.unit


def _events(directory):
    return [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]


def test_no_opt_in_preserves_the_original_ports_and_registered_byte_contract():
    look, hands = object(), object()
    assert recorded_play_ports(look, hands) == (look, hands)
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _recorded_pixel_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'recorded_pixels_keep_their_bytes')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.asyncio
async def test_exact_consumed_picture_and_clock_custody_survive_mutation(tmp_path):
    directory = tmp_path / 'take'
    picture = np.arange(240, dtype=np.uint8).reshape(8, 10, 3)
    expected = picture.copy()
    at = CanvasFrameTime(10., lower=9.9, upper=10.1, arrived=10.2, source='canvas', source_at=2.,
                         epoch=3, surface='surface-1', sequence=18)
    calls = []

    async def look():
        return picture, at

    async def down(key):
        calls.append(key)
        return 'delivered'

    async with record_frames_and_inputs(directory) as recorder:
        assert json.loads((directory / 'manifest.json').read_text())['state'] == 'running'
        looking, hands = recorded_play_ports(look, SimpleNamespace(down=down, page='original'))
        assert hands.page == 'original'
        result = await looking()
        assert result[0] is picture and result[1] is at
        picture[:] = 0
        assert await hands.down('up') == 'delivered'
        await asyncio.wait_for(recorder.queue.join(), 5.)
        events = [e for e in _events(directory) if e['kind'] != 'stream_started']
        pixels = np.asarray(Image.open(directory / events[0]['file']))
        assert np.array_equal(pixels, expected)
        assert events[0]['raw_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
        assert events[0]['capture'] == dict(at=10., lower=9.9, upper=10.1, arrived=10.2, source='canvas',
                                          source_at=2., epoch=3, surface='surface-1', sequence=18)
        assert [e.get('phase') for e in events[1:]] == ['requested', 'completed']
        assert events[2]['request_sequence'] == events[1]['sequence'] and calls == ['up']
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['frames'] == 1 and not manifest['presentation_qualified']
    with pytest.raises(ValueError, match='fresh'):
        async with record_frames_and_inputs(directory):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeError('unavailable'), asyncio.CancelledError()])
async def test_failed_and_cancelled_inputs_are_never_recorded_as_completed(tmp_path, error):
    async def down(key):
        raise error

    directory = tmp_path / 'failure'
    with pytest.raises(type(error)):
        async with record_frames_and_inputs(directory):
            _, hands = recorded_play_ports(None, SimpleNamespace(down=down))
            await hands.down('left')
    assert [e['phase'] for e in _events(directory) if e['kind'] == 'input'] == ['requested', 'failed']
    assert not json.loads((directory / 'manifest.json').read_text())['complete']


def test_queue_and_byte_bounds_mark_the_recording_incomplete():
    recorder = FrameAndInputRecording('unused', queued_bytes=24, total_bytes=24, queued_events=1)
    recorder.frame((np.ones((2, 2, 3), dtype=np.uint8), 1.))
    recorder.frame((np.ones((2, 2, 3), dtype=np.uint8), 2.))
    recorder.frame((np.ones((4, 4, 3), dtype=np.uint8), 3.))
    assert recorder.queue.qsize() == 1 and recorder.total_bytes == 12 and recorder.dropped == 2
    assert not recorder.manifest('closed')['complete']


@pytest.mark.asyncio
async def test_storage_failure_is_retained_and_cannot_be_certified(tmp_path, monkeypatch):
    from core.perception import recording_the_frames_and_inputs as module

    def fail(picture):
        raise OSError('disk refused')

    monkeypatch.setattr(module, '_encoded', fail)
    directory = tmp_path / 'storage'
    async with record_frames_and_inputs(directory) as recorder:
        recorder.frame((np.ones((3, 3, 4), dtype=np.uint8), 4.))
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert not manifest['complete'] and 'disk refused' in manifest['error'] and manifest['frames'] == 0


def test_recording_requires_an_explicit_trace_directory():
    from tools.measure_on_archived_games import _arguments

    with pytest.raises(SystemExit):
        _arguments(['--out', 'unused.json', '--record-interactions'])


@pytest.mark.asyncio
async def test_recorded_pixels_are_hash_checked_before_replay(tmp_path):
    from tools.replay_recorded_play_pixels import recorded_picture

    directory = tmp_path / 'hashed'
    async with record_frames_and_inputs(directory) as recorder:
        recorder.frame((np.ones((3, 3, 3), dtype=np.uint8), 1.))
    event = _events(directory)[0]
    assert recorded_picture(directory, event).shape == (3, 3, 3)
    event['raw_sha256'] = 'changed'
    with pytest.raises(ValueError, match='no longer match'):
        recorded_picture(directory, event)
    event['file'] = '../outside.png'
    with pytest.raises(ValueError, match='inside'):
        recorded_picture(directory, event)


@pytest.mark.asyncio
async def test_actual_control_sample_counts_remain_separate_from_input_delivery(tmp_path):
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.recording_the_frames_and_inputs import record_control_observation

    hers = WhichIsHers()
    hers._by_thing[8].add('a', 20., 0., press=1., when=1.3)
    directory = tmp_path / 'counts'
    async with record_frames_and_inputs(directory):
        record_control_observation(SimpleNamespace(held_trying=True, held='a', held_since=1., keys=['a']),
            SimpleNamespace(things={}), hers, 1.3)
    event = _events(directory)[0]
    assert event['kind'] == 'control_observation' and event['selected'] is None
    assert event['trying'] and event['evidence'][0]['presses'][0]['samples'] == 1
    assert event['evidence'][0]['ratio'] == [0., 0.]
    assert event['source_references']['fresh']['count'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('initial_count', [2, 6])
async def test_the_owned_interval_record_identifies_the_interval_actually_selected(tmp_path, initial_count):
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.recording_the_frames_and_inputs import record_control_observation

    hers = WhichIsHers()
    record = hers._hers
    original_times = [1.25+i*.1 for i in range(initial_count)]
    for at in original_times:
        record.add('q', at*30., 0., press=1., when=at)
    record.unseen('q', 1.)
    later_times = [2.1, 2.2, 2.3, 2.4]
    for at in later_times:
        record.add('q', at*40., 0., press=1., when=at)
    directory = tmp_path / 'owned-interval'
    async with record_frames_and_inputs(directory):
        record_control_observation(SimpleNamespace(held_trying=True, held='q', held_since=1., keys=['q']),
                                   SimpleNamespace(things={}), hers, 3.)
        record.timed.clear()
        record.resumed_pushes.clear()
    interval, = _events(directory)[0]['owned_control_intervals']
    assert interval['key'] == 'q' and interval['began'] == 1.
    assert interval['times'] == (later_times if initial_count == 2 else original_times)
    assert interval['later'] is (initial_count == 2)


@pytest.mark.asyncio
async def test_actual_source_geometry_is_copied_with_its_original_capture_not_a_control_role(tmp_path):
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.recording_the_frames_and_inputs import record_control_observation

    point = SimpleNamespace(at=1., position=[12., 19.], error=.4, support=9)
    receipt = SimpleNamespace(at=1., shape=(32, 40, 3), sha256='a'*64)
    reference = SimpleNamespace(source='source', character_id=7, exported_name='decoration',
                                reference_sha256='b'*64, point=point, pixels=receipt, hypotheses=2)
    moves = SimpleNamespace(things={}, pixels_consumed=receipt, source_reference_status='catalog_ready',
                            source_reference_points=(), source_reference_frames=(),
                            fresh_source_reference_frames=[reference], fresh_source_reference_status=())
    directory = tmp_path / 'reference-record'
    async with record_frames_and_inputs(directory):
        record_control_observation(SimpleNamespace(held_trying=True, held='a', held_since=1., keys=['a']),
                                   moves, WhichIsHers(), 2.)
        point.position[0] = 99.
        moves.fresh_source_reference_frames.clear()
    event = _events(directory)[0]
    snapshot = event['source_references']
    assert snapshot['fresh']['count'] == 1 and event['selected'] is None
    recorded = snapshot['fresh']['records'][0]
    assert recorded['point']['position'] == [12., 19.]
    assert recorded['point']['at'] == recorded['pixels']['at'] == 1. and event['at'] == 2.
    assert recorded['hypotheses'] == 2 and recorded['pixels']['sha256'] == 'a'*64
    assert not {'controlled', 'qualified', 'task_passed'} & recorded.keys()


def test_source_recording_bounds_remain_explicit_and_missing_is_not_empty():
    from core.perception.recording_the_frames_and_inputs import _source_reference_record_invariant, _source_reference_snapshot

    moves = SimpleNamespace(source_reference_frames=[SimpleNamespace()] * 200,
                            fresh_source_reference_frames=())
    snapshot = _source_reference_snapshot(moves)
    assert snapshot['object_bound']['count'] is None and not snapshot['object_bound']['available']
    assert snapshot['fresh']['available'] and snapshot['fresh']['count'] == 0
    assert snapshot['continuing']['count'] == 200 and snapshot['continuing']['truncated']
    assert len(snapshot['continuing']['records']) == 128
    assert _source_reference_record_invariant() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('bound', [False, True])
async def test_the_actual_choice_hook_records_a_made_target_without_selecting_again(tmp_path, bound):
    from core.agency.a_choice_and_its_reasons import chose_in_play
    from core.agency.a_requested_relation_in_play import SpatialRequest
    from core.agency.imagining_ahead import Imagined, Plan

    def no_selection():
        raise AssertionError('recording reran the target selector')

    mine = SimpleNamespace(number=1, born=1., kind=2, x=20., y=30., vx=4., vy=5.,
                           w=10., h=12., seen=10., moved=True)
    aim = SimpleNamespace(number=3, born=2., kind=4, x=80., y=90., vx=0., vy=0.,
                          w=30., h=20., seen=10., moved=False)
    choosing = SimpleNamespace(mine=mine, last_target=((80., 70.), 'meet', aim), target=no_selection,
        ways={'q': (4., 5.)}, rest=(0., 1.), response_s=.12, next_picture_s=.1,
        meeting=SimpleNamespace(told_gently=True, spatial_request=SpatialRequest('on', 'platform', 'Land on the platform')),
        goal_binding=SimpleNamespace(thing=aim) if bound else None,
        planned=Imagined(Plan(((0., 'q'),)), 2., gets=1, decides=True, ends_at=(50., 60.)))
    at = CanvasFrameTime(10., lower=9.99, upper=10.01, arrived=10.1, source='canvas', source_at=8.,
                         epoch=2, surface='one', sequence=37)
    directory = tmp_path / 'choices'
    async with record_frames_and_inputs(directory):
        chose_in_play(None, choosing, 'q', 'meet', aim, at, lambda *_: None)
        mine.x, aim.x = 999., 999.
    event = _events(directory)[0]
    assert event['kind'] == 'play_choice' and event['target'] == [80., 70.]
    assert event['mine']['position'] == [20., 30.] and event['aim']['position'] == [80., 90.]
    assert event['capture']['sequence'] == 37 and event['capture']['surface'] == 'one'
    assert event['key'] == 'q' and event['told_gently']
    assert event['requested'] == dict(relation='on', destination='platform', source='Land on the platform',
                                      bound_number=3 if bound else None)
    assert event['planned'] == dict(first='q', holds=[[0., 'q']], worth=2., decides=True,
                                    gets=1, costs=0, ends_at=[50., 60.])


def test_a_target_lookup_retains_the_made_result_for_its_action_record():
    from core.agency.choosing_where_to_be import _Choosing

    calls = []
    def selected():
        calls.append(1)
        return ((20., 30.), 'meet', None)
    choosing = SimpleNamespace(line=lambda: None, _target_anywhere=selected)
    result = _Choosing.target(choosing)
    assert len(calls) == 1 and choosing.last_target is result


@pytest.mark.asyncio
async def test_normal_pursuit_options_keep_their_original_reading_without_becoming_a_motor_frame(tmp_path, monkeypatch):
    from core.cognition import her_bearings
    from core.perception.observed_transfer import ScreenReading, make_snapshot
    from core.skills.screen_pursuit_looking import _her_bearings
    from tools.replay_recorded_play_pixels import recorded_picture

    picture = np.zeros((30, 60, 3), np.uint8)
    picture[5:15, 10:30] = (3, 27, 219)
    layout = [dict(text='Next', x=.2, y=.2, width=.3, height=.2, confidence=.9, of_its_own=True)]
    observation = ScreenReading(dict(text='Instructions Next', layout=layout, capture_at=2.,
                                    surface_id='one-view', bounds=[0, 0, 60, 30]),
                                picture_snapshot=make_snapshot(picture))
    called = []
    def works():
        called.append(1)
        return ['up', 'left']
    monkeypatch.setattr(her_bearings, 'take_bearings', lambda place: place)
    directory = tmp_path / 'normal-options'
    async with record_frames_and_inputs(directory):
        place = _her_bearings(observation, ['click "Next"'], observation['text'], SimpleNamespace(works=works))
        layout[0]['text'] = 'replaced'
        picture[:] = 0
    assert called == [1] and place.keys == ('up', 'left')
    event = _events(directory)[0]
    assert event['kind'] == 'screen_options' and event['kind'] != 'frame'
    assert event['layout'][0]['text'] == 'Next'
    assert event['layout'][0]['of_its_own'] is True
    assert event['offered_clicks'] == ['click "Next"'] and event['offered_keys'] == ['up', 'left']
    assert event['capture']['at'] == 2. and event['capture']['bounds'] == [0., 0., 60., 30.]
    assert tuple(recorded_picture(directory, event)[8, 15]) == (3, 27, 219)


@pytest.mark.asyncio
async def test_a_normal_reading_without_pixels_stays_without_pixels(tmp_path):
    from core.perception.recording_the_frames_and_inputs import record_screen_options

    directory = tmp_path / 'unpictured-options'
    async with record_frames_and_inputs(directory):
        record_screen_options({'text': 'Next'}, [], [])
    event = _events(directory)[0]
    assert event['kind'] == 'screen_options' and 'file' not in event
    assert event['capture']['at'] is None


def test_normal_option_recording_is_inactive_without_an_opted_in_recorder():
    from core.perception.recording_the_frames_and_inputs import (
        _screen_options_invariant,
        record_screen_options,
    )

    assert _screen_options_invariant() == ()
    record_screen_options(None, None, None)


def test_control_replay_uses_observed_independence_and_refuses_missing_or_conflicting_ownership():
    from tools.replay_recorded_play_pixels import control_histories

    events = [dict(kind='stream_started'),
              dict(kind='control_observation', held_since=1., held='a', trying=False),
              dict(kind='control_observation', held_since=2., held='', trying=True)]
    assert control_histories(events) == {1: [(1., 'a', False), (2., '', True)]}
    with pytest.raises(ValueError, match='conflicting'):
        control_histories(events+[dict(kind='control_observation', held_since=1., held='a', trying=True)])
    with pytest.raises(ValueError, match='every stream'):
        control_histories([dict(kind='stream_started')])
    with pytest.raises(ValueError, match='combined or pointer'):
        control_histories(events+[dict(kind='input', method='tap')])
    with pytest.raises(ValueError, match='unmeasured'):
        control_histories([dict(kind='stream_started'),
                           dict(kind='control_observation', held_since=float('nan'), held='a', trying=True)])


def test_later_overflow_cannot_certify_a_gapped_prefix_or_destroy_an_intact_one():
    from tools.replay_recorded_play_pixels import recorded_prefix

    events = [dict(sequence=1, kind='stream_started'), dict(sequence=2, kind='frame'),
              dict(sequence=3, kind='control_observation'), dict(sequence=4, kind='frame'),
              dict(sequence=9, kind='frame')]
    prefix, complete = recorded_prefix(events, 1)
    assert prefix == events[:3] and complete
    assert not recorded_prefix(events, 3)[1]
    assert not recorded_prefix([events[0], events[2], events[3], events[4]], 1)[1]
