"""Bounded, opt-in evidence from the actual fast play ports.

Transport completion is recorded separately from visible effect. Pixels retain
their original size, channel order and capture receipt. Drawing commands and
sounds are not serialized; this is a pixel replay, not a whole runtime replay.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.verify import invariant

_ACTIVE: ContextVar[Any] = ContextVar('play_pixel_recording', default=None)
_INPUTS = frozenset({'down', 'up', 'tap', 'click', 'point', 'press', 'release', 'type_text'})
_TIME_FIELDS = ('lower', 'upper', 'arrived', 'source', 'source_at', 'epoch', 'surface', 'sequence')


def _pixels(picture: Any) -> np.ndarray:
    value = np.asarray(picture)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] not in (3, 4):
        raise ValueError('a recorded picture must be uint8 RGB or RGBA')
    return np.array(value, copy=True, order='C', subok=False)


def _encoded(picture: np.ndarray) -> tuple[bytes, str]:
    stream = io.BytesIO()
    Image.fromarray(picture).save(stream, format='PNG', compress_level=1)
    return stream.getvalue(), hashlib.sha256(picture.tobytes()).hexdigest()


class FrameAndInputRecording:
    """A single writer owns a fresh directory; overflow stays explicitly incomplete."""

    def __init__(self, directory: Path, *, queued_bytes: int = 64 * 1024**2,
                 total_bytes: int = 2 * 1024**3, queued_events: int = 128) -> None:
        if min(queued_bytes, total_bytes, queued_events) <= 0:
            raise ValueError('recording bounds must be positive')
        self.directory = Path(directory)
        self.byte_limit, self.total_limit = queued_bytes, total_bytes
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queued_events)
        self.pending_bytes = self.total_bytes = 0
        self.sequence = self.frames = self.inputs = self.dropped = 0
        self.error = ''
        self.closed = False

    def event(self, kind: str, **fields: Any) -> dict:
        self.sequence += 1
        return dict(sequence=self.sequence, kind=kind, received_at=time.monotonic(), **fields)

    def offer(self, event: dict, picture: np.ndarray | None = None) -> None:
        size = picture.nbytes if picture is not None else 0
        if (self.closed or self.error or self.queue.full() or self.pending_bytes + size > self.byte_limit
                or self.total_bytes + size > self.total_limit):
            self.dropped += 1
            return
        self.pending_bytes += size
        self.total_bytes += size
        self.queue.put_nowait((event, picture))

    def frame(self, result: Any) -> None:
        if result is None:
            self.offer(self.event('missing_frame'))
            return
        picture, at = result
        size = getattr(picture, 'nbytes', self.byte_limit + 1)
        if size > self.byte_limit or self.pending_bytes + size > self.byte_limit:
            self.sequence += 1
            self.dropped += 1
            return
        receipt = {'at': float(at), **{key: getattr(at, key) for key in _TIME_FIELDS if hasattr(at, key)}}
        self.offer(self.event('frame', capture=receipt,
                             drawing_scene_present=getattr(picture, 'drawing_scene', None) is not None), _pixels(picture))

    async def write(self) -> None:
        gateway = get_file_write_gateway()
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            event, picture = item
            try:
                if not self.error:
                    with local_internal_governed_scope('play_pixel_recording'):
                        if picture is not None:
                            data, digest = await asyncio.to_thread(_encoded, picture)
                            name = f"frame-{event['sequence']:07d}.png"
                            await gateway.write_bytes_async(self.directory / name, data, source='play_pixel_recording')
                            event.update(file=name, shape=list(picture.shape), dtype='uint8', raw_sha256=digest)
                        await gateway.append_text_async(self.directory / 'events.jsonl', json.dumps(event) + '\n',
                                                        source='play_pixel_recording')
                    self.frames += int(picture is not None)
                    self.inputs += int(event['kind'] in ('input', 'navigation_input'))
            except Exception as error:  # noqa: BLE001 - the final manifest retains failed storage
                self.error = f'{type(error).__name__}: {error}'
            finally:
                if picture is not None:
                    self.pending_bytes -= picture.nbytes
                self.queue.task_done()

    def manifest(self, state: str) -> dict:
        return dict(schema='aura.pixel_input_recording.v1', state=state, frames=self.frames,
                    input_events=self.inputs, last_sequence=self.sequence, dropped_events=self.dropped,
                    error=self.error, complete=state == 'closed' and not self.error and not self.dropped,
                    presentation_qualified=False, omitted=['drawing_scene', 'sounds', 'unrecorded decision paths'],
                    input_receipt='port call completion; does not prove game acceptance or visible effect')


@asynccontextmanager
async def record_frames_and_inputs(directory: Path, **bounds: Any):
    recorder = FrameAndInputRecording(directory, **bounds)
    if recorder.directory.exists():
        raise ValueError('use a fresh pixel recording directory')
    gateway = get_file_write_gateway()
    with local_internal_governed_scope('play_pixel_recording'):
        await gateway.ensure_directory_async(recorder.directory, source='play_pixel_recording')
        if not await gateway.write_bytes_if_absent_async(recorder.directory / 'manifest.json',
                json.dumps(recorder.manifest('running')).encode(), source='play_pixel_recording'):
            raise ValueError('another recorder owns this directory')
    writer = asyncio.create_task(recorder.write())
    token = _ACTIVE.set(recorder)
    state = 'closed'
    try:
        yield recorder
    except BaseException:
        state = 'interrupted'
        raise
    finally:
        _ACTIVE.reset(token)
        recorder.closed = True
        # Drain only queued evidence, never wait for a new capture or input.
        async def drain():
            await recorder.queue.put(None)
            await writer

        try:
            await asyncio.wait_for(drain(), timeout=20.)
        except TimeoutError:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            recorder.error = 'recording writer did not finish in twenty seconds'
        with local_internal_governed_scope('play_pixel_recording'):
            await gateway.write_text_async(recorder.directory / 'manifest.json',
                json.dumps(recorder.manifest(state), indent=2), source='play_pixel_recording')


class _RecordedHands:
    def __init__(self, hands: Any, recorder: FrameAndInputRecording) -> None:
        self.hands, self.recorder = hands, recorder

    def __getattr__(self, name: str) -> Any:
        original = getattr(self.hands, name)
        if name not in _INPUTS or not callable(original):
            return original

        async def delivered(*args: Any, **kwargs: Any) -> Any:
            request = self.recorder.event('input', phase='requested', method=name, args=list(args), kwargs=kwargs)
            self.recorder.offer(request)
            try:
                result = await original(*args, **kwargs)
            except BaseException as error:
                self.recorder.offer(self.recorder.event('input', phase='failed', method=name,
                    request_sequence=request['sequence'], error=type(error).__name__))
                raise
            self.recorder.offer(self.recorder.event('input', phase='completed', method=name,
                request_sequence=request['sequence']))
            return result

        return delivered


def recorded_play_ports(look: Any, hands: Any) -> tuple[Any, Any]:
    recorder = _ACTIVE.get()
    if recorder is None:
        return look, hands
    recorder.offer(recorder.event('stream_started'))

    async def looked():
        result = await look()
        recorder.frame(result)
        return result

    return looked, _RecordedHands(hands, recorder)


def begin_navigation_input(page: Any, method: str, args: tuple, *, bounds: tuple = ()) -> Any:
    """Record a normal page-port request separately from fast-loop input."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return None
    request = recorder.event('navigation_input', phase='requested', method=method, args=list(args),
                             bounds=list(bounds), surface=f'page:{id(page)}')
    recorder.offer(request)
    return recorder, request['sequence']


def finish_navigation_input(request: Any, delivered: bool) -> None:
    """Port completion cannot establish acceptance, an effect or task completion."""
    if request is None:
        return
    recorder, sequence = request
    recorder.offer(recorder.event('navigation_input', phase='completed' if delivered else 'failed',
                                  request_sequence=sequence, delivered=delivered,
                                  partial_delivery_possible=not delivered))


@invariant('perception.normal_navigation_records_its_physical_port', scope='perception',
           owner='core/perception/recording_the_frames_and_inputs.py', observational=False)
def _navigation_port_recording_invariant() -> tuple:
    recorder = FrameAndInputRecording(Path('unused-canary-path'))
    token = _ACTIVE.set(recorder)
    try:
        finish_navigation_input(begin_navigation_input(object(), 'click', (210., 80.)), False)
        first, _ = recorder.queue.get_nowait()
        last, _ = recorder.queue.get_nowait()
        assert first['kind'] == last['kind'] == 'navigation_input'
        assert first['args'] == [210., 80.] and last['request_sequence'] == first['sequence']
        assert last['phase'] == 'failed' and last['delivered'] is False
        assert last['partial_delivery_possible'] and 'task_passed' not in last
    finally:
        _ACTIVE.reset(token)
    return ()


def record_control_observation(run: Any, moves: Any, hers: Any, at: Any) -> None:
    """Keep the actor's own sample counts, separate from later pixel replay."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    from core.perception.translation_from_visible_support import motion_status

    evidence = []
    for number, speeds in hers._by_thing.items():
        evidence.append(dict(number=int(number), ratio=list(speeds.ratio()),
            presses=[dict(key=key, began=began, samples=len(samples),
                times=[t for t, _vx, _vy in speeds.timed.get((key, began), ())],
                ended=(key, began) in speeds.ended, interrupted=(key, began) in speeds.interrupted)
                for (key, began), samples in speeds.by_press.items()],
            pushes={key: len(values) for key, values in speeds.pushes().items()}))
    recorder.offer(recorder.event('control_observation', at=float(at), selected=hers.number,
        epoch=hers.identification.epoch, trying=run.held_trying, held=run.held, held_since=run.held_since,
        carried=hers.carried, known_controls=list(hers.keys_that_move_her(run.keys)),
        measured_pushes={key: hers.push_of(key) for key in hers._hers.pushes()},
        observations=[dict(number=int(t.number), born=float(t.born), seen=float(t.seen), status=motion_status(t, at),
                           position=[float(t.x), float(t.y)]) for t in moves.things.values()], evidence=evidence))


def record_play_choice(choosing: Any, key: str, why: str, aim: Any, at: Any) -> None:
    """Retain a made choice without asking the target selector to make it again."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return

    def body(thing):
        if thing is None:
            return None
        return dict(number=int(thing.number), born=float(thing.born), kind=int(thing.kind),
                    position=[float(thing.x), float(thing.y)], velocity=[float(thing.vx), float(thing.vy)],
                    size=[float(thing.w), float(thing.h)], seen=float(thing.seen), moved=bool(thing.moved))

    target = getattr(choosing, 'last_target', None)
    request = getattr(choosing.meeting, 'spatial_request', None)
    binding = getattr(choosing, 'goal_binding', None)
    imagined = getattr(choosing, 'planned', None)
    planned = (dict(first=imagined.plan.first, holds=list(imagined.plan.held),
                    worth=float(imagined.worth), decides=bool(imagined.decides),
                    gets=int(imagined.gets), costs=int(imagined.costs), ends_at=list(imagined.ends_at))
               if imagined is not None else None)
    recorder.offer(recorder.event('play_choice', at=float(at), key=key, why=why,
        capture={'at': float(at), **{field: getattr(at, field) for field in _TIME_FIELDS if hasattr(at, field)}},
        mine=body(choosing.mine), aim=body(aim), target=list(target[0]) if target else None,
        ways={control: list(way) for control, way in choosing.ways.items()}, rest=list(choosing.rest),
        planned=planned, response_s=float(choosing.response_s), next_picture_s=float(choosing.next_picture_s),
        told_gently=getattr(choosing.meeting, 'told_gently', None),
        requested=(dict(relation=request.relation, destination=request.destination, source=request.source,
                        bound_number=binding.thing.number if binding is not None else None)
                   if request is not None else None)))


def record_screen_options(observation: Any, clickable: Any, keys: Any, *, leaving_alone: Any = ()) -> None:
    """Keep the normal pursuit's offered controls with its original reading.

    A reader snapshot has no declared RGB contract. Preserve its bytes and
    mark that channel order as unspecified. This event is not a motor frame
    and must not enter pixel-control replay as one.
    """
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    from core.perception.observed_transfer import _pixels as snapshot_pixels
    from core.perception.observed_transfer import freeze_screen

    screen = freeze_screen(observation)
    snapshot = snapshot_pixels(screen.picture) if screen.picture is not None else None
    layout = []
    for region in (observation.get('layout') or [])[:128]:
        if not isinstance(region, dict):
            continue
        measured = {key: value[:500] if isinstance(value, str) else value
                    for key in ('text', 'x', 'y', 'width', 'height', 'center_x', 'center_y', 'confidence', 'of_its_own',
                                'answers_the_pointer', 'pointer_x', 'pointer_y', 'pointer_ambiguous')
                    if isinstance(value := region.get(key), (str, int, float, bool))}
        layout.append(measured)
    recorder.offer(recorder.event('screen_options', text=str(observation.get('text') or '')[:8000],
        layout=layout, offered_clicks=[str(name)[:300] for name in list(clickable)[:256]],
        offered_keys=[str(key)[:80] for key in list(keys)[:128]],
        leaving_alone=[str(label)[:300] for label in list(leaving_alone)[:256]],
        capture=dict(at=screen.capture_at, surface=screen.surface, bounds=screen.bounds, epoch=screen.epoch),
        snapshot_channels='unspecified; original reader bytes',
        truncated=len(observation.get('layout') or []) > 128 or len(clickable) > 256),
        _pixels(snapshot) if snapshot is not None else None)


@invariant('perception.normal_screen_options_keep_their_reading', scope='perception',
           owner='core/perception/recording_the_frames_and_inputs.py', observational=False)
def _screen_options_invariant() -> tuple:
    from core.perception.observed_transfer import ScreenReading, make_snapshot

    pixels = np.zeros((8, 10, 3), np.uint8)
    pixels[2:5, 2:8] = (17, 49, 203)
    region = dict(text='(BLAY)', x=.2, y=.25, width=.6, height=.4)
    observation = ScreenReading(dict(text='Read the instructions (BLAY)', layout=[region], capture_at=2.),
                                picture_snapshot=make_snapshot(pixels))
    recorder = FrameAndInputRecording(Path('.'))
    token = _ACTIVE.set(recorder)
    try:
        record_screen_options(observation, ['click "(BLAY)"'], ['up'])
        event, picture = recorder.queue.get_nowait()
        region['text'], pixels[:] = 'changed', 0
        assert event['kind'] == 'screen_options' and event['layout'][0]['text'] == '(BLAY)'
        assert event['offered_keys'] == ['up'] and event['capture']['at'] == 2.
        assert tuple(picture[3, 3]) == (17, 49, 203)
        assert event['snapshot_channels'].startswith('unspecified')
    finally:
        _ACTIVE.reset(token)
    return ()


@invariant('perception.recorded_pixels_keep_their_bytes', scope='perception',
           owner='core/perception/recording_the_frames_and_inputs.py', observational=False)
def _recorded_pixel_invariant() -> tuple:
    source = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    retained = _pixels(source)
    source[:] = 0
    encoded, digest = _encoded(retained)
    decoded = np.asarray(Image.open(io.BytesIO(encoded)))
    assert np.array_equal(retained, decoded)
    assert digest == hashlib.sha256(decoded.tobytes()).hexdigest()
    assert retained.any() and not source.any()
    return ()
