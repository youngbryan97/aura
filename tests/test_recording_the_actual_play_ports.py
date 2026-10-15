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
        record_control_observation(SimpleNamespace(held_trying=True, held='a', held_since=1.),
            SimpleNamespace(things={}), hers, 1.3)
    event = _events(directory)[0]
    assert event['kind'] == 'control_observation' and event['selected'] is None
    assert event['trying'] and event['evidence'][0]['presses'][0]['samples'] == 1
    assert event['evidence'][0]['ratio'] == [0., 0.]
