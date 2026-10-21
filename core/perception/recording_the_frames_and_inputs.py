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
import math
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from itertools import islice
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


def record_keyboard_target(page: Any, *, status: str, began: float, ended: float, error: str = '') -> None:
    """Keep current focus evidence without claiming input acceptance or a game effect."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    if (status not in ('focused', 'not_focused', 'unmeasured', 'unavailable')
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in (began, ended))
            or began > ended or not isinstance(error, str) or len(error) > 80):
        return
    recorder.offer(recorder.event('keyboard_target', surface=f'page:{id(page)}', status=status,
                                 began=began, ended=ended, error=error))


@invariant('agency.keyboard_target_receipts_do_not_claim_acceptance', scope='agency',
           owner='core/perception/recording_the_frames_and_inputs.py', observational=False)
def _keyboard_target_receipt_invariant() -> tuple:
    from types import SimpleNamespace

    events = []
    recorder = SimpleNamespace(offer=events.append, event=lambda kind, **values: dict(kind=kind, **values))
    token = _ACTIVE.set(recorder)
    page = object()
    try:
        record_keyboard_target(page, status='focused', began=1., ended=1.01)
        record_keyboard_target(page, status='unavailable', began=2., ended=2.01, error='RuntimeError')
        record_keyboard_target(page, status='focused', began=float('nan'), ended=3.)
    finally:
        _ACTIVE.reset(token)
    assert len(events) == 2 and events[0]['surface'] == f'page:{id(page)}'
    assert events[1]['status'] == 'unavailable'
    assert not any({'delivered', 'accepted_by_game', 'controlled', 'task_passed'} & event.keys() for event in events)
    return ()


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


def record_pointer_reference(target: Any) -> int | None:
    """Retain the small RGB reference separately from the later matching pictures."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return None
    width, height = target.size
    patch = np.frombuffer(target.patch, np.uint8).reshape(height, width, 3)
    event = recorder.event('pointer_reference', channels='RGB', surface=f'page:{target.page_id}',
                           viewport=list(target.viewport), bounds=list(target.bounds),
                           anchor=list(target.anchor), sampled_at=target.captured,
                           sample_point=list(target.sample_point), authority='visible material only')
    recorder.offer(event, _pixels(patch))
    return event['sequence']


def record_pointer_correspondence(reference: int | None, picture: Any, *, phase: str,
                                  pointer: Any = None, matched: Any = None,
                                  captured: tuple | None = None, moved_at: float | None = None) -> None:
    """Keep a current RGB comparison, including a failed match; no task result."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    recorder.offer(recorder.event('pointer_correspondence', reference_sequence=reference, phase=phase,
        channels='RGB', pointer_in_bounds=[float(v) for v in pointer] if pointer is not None else None,
        matched_in_bounds=[float(v) for v in matched] if matched is not None else None,
        capture_interval=list(captured) if captured is not None else None, moved_at=moved_at,
        qualified=matched is not None), _pixels(picture))


@invariant('perception.pointer_comparisons_keep_their_reference_pixels', scope='perception',
           owner='core/perception/recording_the_frames_and_inputs.py', observational=False)
def _pointer_recording_invariant() -> tuple:
    from types import SimpleNamespace

    recorder = FrameAndInputRecording(Path('unused-pointer-canary'))
    token = _ACTIVE.set(recorder)
    try:
        picture = np.zeros((16, 16, 3), np.uint8)
        target = SimpleNamespace(size=(16, 16), patch=picture.tobytes(), page_id=1, viewport=(100, 100),
                                 bounds=(0., 0., 100., 100.), anchor=(8., 8.), captured=1., sample_point=(50., 50.))
        reference = record_pointer_reference(target)
        record_pointer_correspondence(reference, picture, phase='correct', pointer=(10., 10.))
        first, source = recorder.queue.get_nowait()
        last, comparison = recorder.queue.get_nowait()
        assert first['kind'] == 'pointer_reference' and first['sampled_at'] == 1.
        assert last['reference_sequence'] == first['sequence'] and last['qualified'] is False
        assert last['kind'] != 'frame' and 'task_passed' not in last
        assert first['channels'] == last['channels'] == 'RGB'
        assert np.array_equal(source, comparison)
    finally:
        _ACTIVE.reset(token)
    return ()


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


def _source_reference_snapshot(moves: Any) -> dict:
    """Copy published geometry as reported; no field grants control authority."""
    def scalar(value):
        return value if type(value) in (int, float, bool, type(None)) else value[:512] if isinstance(value, str) else None

    def point(value):
        position = getattr(value, 'position', None)
        return {**{name: scalar(getattr(value, name, None)) for name in ('at', 'error', 'support', 'number', 'born')},
                'position': [scalar(n) for n in position] if isinstance(position, (tuple, list)) and len(position) == 2 else None}

    def pixels(value):
        shape = getattr(value, 'shape', None)
        return {'at': scalar(getattr(value, 'at', None)), 'sha256': scalar(getattr(value, 'sha256', None)),
                'shape': list(shape) if isinstance(shape, tuple) and len(shape) == 3 and all(type(n) is int for n in shape) else None}

    def channel(name):
        values = getattr(moves, name, None)
        if not isinstance(values, (tuple, list)):
            return {'available': False, 'count': None, 'records': [], 'truncated': False}
        records = []
        for item in values[:128]:
            records.append({**{name: scalar(getattr(item, name, None)) for name in
                ('source', 'character_id', 'exported_name', 'reference_sha256', 'hypotheses')},
                'point': point(getattr(item, 'point', None)), 'pixels': pixels(getattr(item, 'pixels', None))})
        return {'available': True, 'count': len(values), 'records': records, 'truncated': len(values) > 128}

    statuses = getattr(moves, 'fresh_source_reference_status', None)
    wholes = getattr(moves, 'source_referents', None)
    groups = getattr(moves, 'source_material_groups', None)
    group_records = [{**{name: scalar(getattr(item, name, None)) for name in
        ('number', 'born', 'kind', 'seen', 'x', 'y', 'w', 'h', 'source', 'character_id',
         'geometry_qualified', 'role', 'contact_verified')},
        'material_bounds': list(getattr(item, 'material_bounds', ())),
        'anchors': [list(a) for a in sorted(getattr(item, 'anchors', ()))[:128]],
        'observed_parts': [list(p) for p in getattr(item, 'observed_parts', ())[:32]],
        'pixels': pixels(getattr(item, 'pixels', None))} for item in islice(groups.values(), 64)] if isinstance(groups, dict) else None
    readings = getattr(moves, 'source_referent_readings', ())
    coverage = getattr(moves, 'source_reference_variant_coverage', ())
    variants = [] if isinstance(coverage, tuple) else None
    for c in coverage[:32] if isinstance(coverage, tuple) else ():
        available, missing = getattr(c, 'available_frames', None), getattr(c, 'missing_frames', None)
        variants.append({**{name: scalar(getattr(c, name, None)) for name in ('source', 'character_id', 'declared_frames')},
            'available_frames': list(available[:32]) if isinstance(available, tuple) else None,
            'missing_frames': list(missing[:128]) if isinstance(missing, tuple) else None,
            'missing_count': len(missing) if isinstance(missing, tuple) else None,
            'missing_truncated': len(missing) > 128 if isinstance(missing, tuple) else None})
    whole_readings = [{name: scalar(item.get(name)) for name in
        ('source', 'character_id', 'reference_sha256', 'composition_identity', 'reason', 'mean_colour_error',
         'maximum_edge_error', 'colour_outlier_fraction', 'samples', 'visible_fraction')}
        for item in readings[:128] if isinstance(item, dict)] if isinstance(readings, (tuple, list)) else None
    whole_records = [{**{name: scalar(getattr(item, name, None)) for name in
        ('number', 'born', 'kind', 'seen', 'x', 'y', 'w', 'h', 'source', 'reference_sha256',
         'mark_epoch', 'character_id', 'vx', 'vy', 'velocity_error', 'contact_verified', 'role', 'visible_fraction')},
        'clipped_edges': list(getattr(item, 'clipped_edges', ())),
        'anchors': [list(a) for a in sorted(getattr(item, 'anchors', ()))[:128]],
        'origin_path': [list(p) for p in getattr(item, 'origin_path', ())[:16]],
        'pixels': pixels(getattr(item, 'pixels', None))} for item in islice(wholes.values(), 128)] if isinstance(wholes, dict) else None
    return {'pixels': pixels(getattr(moves, 'pixels_consumed', None)),
            'reference_variants': variants,
            'composed_search_references': [dict(record) for record in getattr(moves, 'source_reference_compositions', ())[:32]
                                          if isinstance(record, dict)],
            'status': scalar(getattr(moves, 'source_reference_status', None)),
            'fresh_status': [scalar(v) for v in statuses[:64]] if isinstance(statuses, (tuple, list)) else None,
            'object_bound': channel('source_reference_points'), 'continuing': channel('source_reference_frames'),
            'fresh': channel('fresh_source_reference_frames'),
            'material_attention': {'records': group_records, 'count': len(groups) if isinstance(groups, dict) else None},
            'extent_requests': [dict(item) for item in getattr(moves, 'source_extent_receipts', ())[:4]
                                if isinstance(item, dict)],
            'whole_referents': {'records': whole_records, 'count': len(wholes) if isinstance(wholes, dict) else None,
                               'readings': whole_readings}}


@invariant('perception.source_reference_records_keep_the_reported_capture', scope='perception',
           owner='core/perception/recording_the_frames_and_inputs.py', observational=False)
def _source_reference_record_invariant() -> tuple:
    from types import SimpleNamespace

    point = SimpleNamespace(at=1., position=[2., 3.])
    moves = SimpleNamespace(fresh_source_reference_frames=[SimpleNamespace(point=point)])
    snapshot = _source_reference_snapshot(moves)
    point.position[0] = 9.
    moves.fresh_source_reference_frames.clear()
    assert snapshot['fresh']['count'] == 1 and snapshot['fresh']['records'][0]['point']['position'] == [2., 3.]
    assert snapshot['fresh']['records'][0]['point']['at'] == 1.
    assert snapshot['object_bound']['count'] is None
    return ()


def record_control_observation(run: Any, moves: Any, hers: Any, at: Any) -> None:
    """Keep the actor's own sample counts, separate from later pixel replay."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    from core.perception.translation_from_visible_support import motion_status
    from dataclasses import asdict

    evidence = []
    family = getattr(hers, 'motion_family_reading', None)
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
        trial_decision=(asdict(run.trial_decision) if getattr(run, 'trial_decision', None) is not None else None),
        motion_family=(dict(at=family[0], number=family[1], **asdict(family[2])) if family is not None else None),
        measured_pushes={key: hers.push_of(key) for key in hers._hers.pushes()},
        owned_control_intervals=[dict(key=key, began=began, samples=len(samples),
            times=[sample[0] for sample in samples], later=samples is hers._hers.resumed_pushes.get((key, began)))
            for key, began in list(hers._hers.timed)[-32:] if (samples := hers._hers.push_samples((key, began)))],
        observations=[dict(number=int(t.number), born=float(t.born), seen=float(t.seen), status=motion_status(t, at),
                           position=[float(t.x), float(t.y)], velocity=[float(t.vx), float(t.vy)],
                           size=[float(t.w), float(t.h)]) for t in moves.things.values()], evidence=evidence,
        source_references=_source_reference_snapshot(moves)))


def record_play_choice(choosing: Any, key: str, why: str, aim: Any, at: Any) -> None:
    """Retain a made choice without asking the target selector to make it again."""
    from dataclasses import asdict
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
    reading = getattr(choosing, 'goal_reading', None)
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
                        **(dict(binding_status=reading.status, named_referents=list(reading.named),
                                named_count=reading.named_count, candidates=list(reading.candidates),
                                candidate_count=reading.candidate_count,
                                need=asdict(reading.need) if getattr(reading, 'need', None) is not None else None,
                                region=asdict(binding.region) if binding is not None and getattr(binding, 'region', None) is not None else None,
                                target_stance=getattr(choosing, 'goal_stance', None)) if reading is not None else {}),
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
    from core.perception.presses_on_the_same_material import current_material_presses

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
        material_press_costs=current_material_presses(observation),
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
