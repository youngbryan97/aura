"""Observe play off the event loop, keeping each worker's mutable eyes private.

A timed-out waiter does not stop a running thread. Its eyes keep that one
picture, but its result cannot reach the public view. Fresh pictures wait
for the operation to finish; they never queue behind it. Physical input
deadlines can run while observation waits for its process or its interpreter.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import math
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from core.perception.seeing_play_elsewhere import (
    SeenElsewhere,
    _brought_up_to_date,
    _state_of,
    eyes_for_play,
)
from core.perception.what_moves_in_the_picture import WhatMoves, prepare_play_vision
from core.runtime.executors import submit_blocking_io
from core.runtime.lockdep import checked_thread_semaphore
from core.verify import invariant

__all__ = ["PlayObservation", "open_play_observation"]

FRAME_WAIT_S = .5
START_WAIT_S = 2.
_SLOTS = checked_thread_semaphore("perception.play_observation_operations", 2)


@dataclass
class _Operation:
    raw: Future
    wrapped: asyncio.Future
    kind: str
    generation: int
    discard: bool = False


def _detached_state(eyes: Any) -> dict[str, Any]:
    """Copy the delegate's readable state while only its worker may change it."""
    holder = eyes
    if isinstance(eyes, SeenElsewhere):
        holder = eyes._here if eyes._here is not None else eyes._likeness
    readable = _state_of(holder)
    # The actor reads visible change and covered quiet, while the preceding
    # pixels stay with the worker that will compare the next picture.
    activity = getattr(holder, "activity", None)
    if activity is not None:
        activity = copy.copy(activity)
        if hasattr(activity, "_picture"):
            activity._picture = None
    readable["activity"] = activity
    # A delegate's alias cache belongs to that delegate. The public mirror
    # retains its own cache so a later snapshot updates the same public body.
    readable.pop("_known", None)
    state = copy.deepcopy(readable)
    state["elsewhere"] = bool(getattr(eyes, "elsewhere", False))
    return state


def _started(factory: Callable[..., Any], kinds: list[Any]) -> tuple[Any, dict[str, Any]]:
    prepare_play_vision()
    eyes = factory(kinds=kinds)
    prepare = getattr(eyes, 'prepare', None)
    if callable(prepare):
        prepare()
    return eyes, _detached_state(eyes)


def _seen(eyes: Any, picture: Any, at: float, *, search_hint: Any = None,
          material_retention: Any = None, fresh_references: Any = None,
          reference_proposals: tuple = (), reference_tables: tuple = (),
          reference_coverage: tuple = (), whole_referents: Any = None,
          reference_frames: tuple = ()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepare_sampling = getattr(fresh_references, 'sampling_before_observation', None)
    if callable(prepare_sampling):
        proposed_hint = prepare_sampling(picture, at, reference_proposals, reference_tables, reference_coverage)
        if search_hint is None:
            search_hint = proposed_hint
    began = time.perf_counter()
    kwargs = {'search_hint': search_hint} if search_hint is not None else {}
    if material_retention is not None and getattr(eyes, 'supports_material_retention', False):
        kwargs['material_retention'] = material_retention
    try:
        happened = eyes.see(picture, at, **kwargs)
        state = _detached_state(eyes)
    except BaseException:
        finish_sampling = getattr(fresh_references, 'finish_sampling', None)
        if callable(finish_sampling):
            finish_sampling()
        raise
    if fresh_references is not None:
        # The eyes receive the full capture interval. Geometry uses the same
        # scalar capture value as their consumed-pixel and material receipts.
        frames, unavailable = fresh_references.measure_if_budget(picture, float(at), state, reference_proposals,
            reference_tables, reference_coverage, essential_seconds=time.perf_counter()-began)
        state['fresh_source_reference_frames'], state['fresh_source_reference_status'] = frames, unavailable
        measured_frames = getattr(fresh_references, 'material_frames', ())
        state['fresh_source_material_frames'] = tuple((r.key, b) for r, b in measured_frames)
        state['fresh_source_material_status'] = getattr(fresh_references, 'material_status', ())
        if measured_frames:
            from types import SimpleNamespace
            from core.perception.source_references_while_play_advances import _with_current_material_frames

            reference_frames, refused = _with_current_material_frames(reference_frames, measured_frames,
                SimpleNamespace(**state), float(at), reference_tables)
            state['fresh_source_material_status'] += refused
    if whole_referents is not None:
        from types import SimpleNamespace

        current = SimpleNamespace(**state)
        matches = []
        for reference, binding in reference_frames:
            sample = binding.inspect(current, float(at), source=reference.source).current
            if sample is not None:
                matches.append((reference, binding, sample))
        state['source_referents'] = whole_referents.observe(current, picture, float(at), tuple(matches))
        state['source_referent_readings'] = whole_referents.readings
    return copy.deepcopy(happened), state


def _close_owned_eyes(eyes: Any) -> None:
    close = getattr(eyes, 'close', None)
    if not callable(close):
        close = getattr(getattr(eyes, 'points', None), 'close', None)
    if callable(close):
        close()


def _retired_operation(raw: Future, eyes: Any, kind: str) -> None:
    """Dispose on the executor after the owner finishes, including late startup."""
    if kind == 'start' and not raw.cancelled() and raw.exception() is None:
        eyes = raw.result()[0]
    if eyes is not None:
        cleanup = submit_blocking_io(_close_owned_eyes, eyes, label='play_observation_cleanup')
        cleanup.add_done_callback(_cleanup_received)


def _cleanup_received(raw: Future) -> None:
    if not raw.cancelled() and raw.exception() is not None:
        from core.runtime.errors import record_degradation

        record_degradation('play_observation_cleanup', raw.exception(), action='kept unreleased model memory counted')


def _completed(raw: Future, slots: Any) -> None:
    """The actual operation, including a cancelled queued call, returns its permit."""
    try:
        if not raw.cancelled():
            raw.exception()
    finally:
        slots.release()


def _received(wrapped: asyncio.Future) -> None:
    """A waiter may have left before the worker's eventual error arrives."""
    if not wrapped.cancelled():
        wrapped.exception()


class PlayObservation:
    """One worker-owned observer and one event-loop-owned likeness of its state."""

    def __init__(self, *, kinds: list[Any] | None = None, factory: Callable[..., Any] = eyes_for_play,
                 frame_wait_s: float = FRAME_WAIT_S, startup_wait_s: float = START_WAIT_S) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (frame_wait_s, startup_wait_s)):
            raise ValueError("observation waits must be finite and positive")
        self._factory = factory
        self._kinds = copy.deepcopy(list(kinds or []))
        self._mirror = WhatMoves(kinds=copy.deepcopy(self._kinds))
        self._mirror.elsewhere = False
        self._eyes: Any = None
        self._operation: _Operation | None = None
        self._generation = 0
        self._closed = False
        self._last_submitted_at = -math.inf
        self._status = "not started"
        self._frame_wait_s = frame_wait_s
        self._startup_wait_s = startup_wait_s
        self._references: Any = None
        from core.perception.source_geometry_in_current_pixels import SourceGeometryInCurrentPixels

        self._fresh_references = SourceGeometryInCurrentPixels()
        from core.perception.whole_referents_from_source_material import WholeReferentsInView

        self._whole_referents = WholeReferentsInView()
        self._mirror.source_referents = {}
        self._mirror.source_referent_readings = ()
        self._mirror.source_reference_points = ()
        self._mirror.source_reference_frames = ()
        self._mirror.fresh_source_reference_frames = ()
        self._mirror.fresh_source_material_frames = ()
        self._mirror.fresh_source_reference_status = ()
        self._mirror.source_reference_status = 'no_current_source'

    @property
    def status(self) -> str:
        return self._status

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mirror, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._mirror, name, value)

    def _submit(self, kind: str, call: Callable[[], Any]) -> _Operation | None:
        if self._closed or self._operation is not None:
            return None
        slots = _SLOTS
        if not slots.acquire(blocking=False):
            self._status = "observation capacity unavailable"
            return None
        context = contextvars.copy_context()
        try:
            raw = submit_blocking_io(context.run, call, label="play_observation")
        except BaseException:
            slots.release()
            raise
        raw.add_done_callback(lambda done: _completed(done, slots))
        wrapped = asyncio.wrap_future(raw)
        wrapped.add_done_callback(_received)
        operation = _Operation(raw, wrapped, kind, self._generation)
        self._operation = operation
        return operation

    def _consume(self, operation: _Operation, *, publish: bool) -> Any:
        if (self._closed or self._operation is not operation
                or operation.generation != self._generation or not operation.raw.done()):
            return None
        self._operation = None
        value, state = operation.raw.result()
        if operation.kind == "start":
            self._eyes = value
        if publish and not operation.discard:
            _brought_up_to_date(self._mirror, state)
            self._status = "ready" if operation.kind == "start" else "observed"
            return value
        self._status = "late observation discarded"
        return None

    async def _wait(self, operation: _Operation, seconds: float) -> Any:
        try:
            await asyncio.wait_for(asyncio.shield(operation.wrapped), seconds)
        except TimeoutError:
            operation.discard = operation.kind != "start"
            self._status = "startup pending" if operation.kind == "start" else "observation unobserved"
            return None
        except asyncio.CancelledError:
            self._retire()
            raise
        except BaseException:
            self._retire()
            raise
        return self._consume(operation, publish=True)

    async def start(self) -> bool:
        """Construct the actual eyes on their worker, with a finite caller wait."""
        if self._closed:
            return False
        operation = self._operation
        if operation is None and self._eyes is None:
            operation = self._submit("start", lambda: _started(self._factory, self._kinds))
        if operation is not None and operation.kind == "start":
            await self._wait(operation, self._startup_wait_s)
        return self._eyes is not None

    async def see(self, picture: Any, at: float, *, search_hint: Any = None) -> list[dict[str, Any]] | None:
        """Return this frame's events, or None while its measurement is unavailable."""
        self._mirror.source_reference_points = ()
        self._mirror.source_reference_frames = ()
        self._mirror.fresh_source_reference_frames = ()
        self._mirror.fresh_source_material_frames = ()
        self._mirror.source_referents = {}
        self._mirror.source_referent_readings = ()
        if self._closed or isinstance(at, bool) or not isinstance(at, (float, int)) or not math.isfinite(at):
            return None
        operation = self._operation
        if operation is not None:
            if not operation.raw.done():
                self._status = "observation pending"
                return None
            self._consume(operation, publish=operation.kind == "start")
        if self._eyes is None:
            self._submit("start", lambda: _started(self._factory, self._kinds))
            return None
        if at <= self._last_submitted_at:
            self._status = "repeated observation"
            return None
        owned_picture, owned_at = copy.deepcopy(picture), copy.deepcopy(at)
        if search_hint is not None:
            from core.perception.a_search_hint_for_these_pixels import PixelSearchHint

            search_hint = copy.deepcopy(search_hint) if isinstance(search_hint, PixelSearchHint) else None
        eyes = self._eyes
        retention = self._references.retention if self._references is not None else None
        from core.perception.source_references_while_play_advances import source_tables_in_play

        tables = source_tables_in_play()
        proposals = (self._references.proposals if self._references is not None
                     and self._references.tables == tables else ())
        coverage = ()
        reference_frames = ()
        if self._references is not None and self._references.tables == tables:
            reference_frames = getattr(self._references, 'frames', ())
            supplied = (*getattr(self._references, 'points', ()), *self._references.source_frames)
            coverage = tuple({(item.source, item.character_id, item.reference_sha256) for item in supplied
                              if item.point.at == self._last_submitted_at})
        operation = self._submit("frame", lambda: _seen(eyes, owned_picture, owned_at, search_hint=search_hint,
            material_retention=retention, fresh_references=self._fresh_references,
            reference_proposals=proposals, reference_tables=tables, reference_coverage=coverage,
            whole_referents=self._whole_referents, reference_frames=reference_frames))
        if operation is None:
            return None
        self._last_submitted_at = float(at)
        events = await self._wait(operation, self._frame_wait_s)
        if events is not None:
            self._reference_points(owned_picture, float(at))
        return events

    def _reference_points(self, picture: Any, at: float) -> None:
        from core.perception.source_references_while_play_advances import SourceReferencesInPlay, source_tables_in_play

        tables = source_tables_in_play()
        active = {table.source for table in tables}
        self._mirror.source_referents = {number: item for number, item in self._mirror.source_referents.items()
            if item.source in active and item.pixels == self._mirror.pixels_consumed and item.seen == at}
        self._mirror.source_referent_readings = tuple(item for item in self._mirror.source_referent_readings
            if item.get('source') is None or item['source'] in active)
        self._mirror.fresh_source_reference_frames = tuple(frame for frame in self._mirror.fresh_source_reference_frames
            if frame.source in active and frame.pixels == self._mirror.pixels_consumed and frame.point.at == at)
        if self._references is None and tables:
            self._references = SourceReferencesInPlay()
        if self._references is not None:
            self._mirror.source_reference_points = self._references.advance(self, picture, at, tables)
            self._mirror.source_reference_frames = self._references.source_frames
            self._mirror.source_reference_status = self._references.status

    def _retire(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._fresh_references = None
        self._whole_referents = None
        self._mirror.source_referents = {}
        self._mirror.source_referent_readings = ()
        if self._references is not None:
            self._references.retire()
        self._mirror.source_reference_points = ()
        self._mirror.source_reference_frames = ()
        self._mirror.fresh_source_reference_frames = ()
        self._mirror.fresh_source_material_frames = ()
        self._mirror.source_reference_status = 'closed'
        self._generation += 1
        eyes, self._eyes = self._eyes, None
        operation, self._operation = self._operation, None
        if operation is not None:
            operation.discard = True
            operation.raw.add_done_callback(lambda done: _retired_operation(done, eyes, operation.kind))
            operation.raw.cancel()
        elif eyes is not None:
            cleanup = submit_blocking_io(_close_owned_eyes, eyes, label='play_observation_cleanup')
            cleanup.add_done_callback(_cleanup_received)
        self._status = "closed"

    async def close(self) -> None:
        """Forbid late publication without waiting for an uncancellable running thread."""
        self._retire()


async def open_play_observation(**kwargs: Any) -> PlayObservation:
    observation = PlayObservation(**kwargs)
    await observation.start()
    return observation


def _private_snapshot_cannot_change_the_public_view() -> bool:
    from types import SimpleNamespace

    body = SimpleNamespace(number=1, x=10.)
    eyes = SimpleNamespace(things={1: body}, pending={}, kinds=[])
    state = _detached_state(eyes)
    mirror = WhatMoves.__new__(WhatMoves)
    _brought_up_to_date(mirror, state)
    first = mirror.things[1]
    body.x = 20.
    private = mirror.things[1].x == 10.
    _brought_up_to_date(mirror, _detached_state(eyes))
    return private and mirror.things[1] is first and first.x == 20.


@invariant("perception.worker_observations_publish_detached_owned_state", scope="perception",
           owner="core/perception/play_observation_without_holding_the_loop.py", observational=False)
def _owned_play_observation_invariant() -> tuple:
    assert _private_snapshot_cannot_change_the_public_view(), "private observation mutated the public view"
    return ()
